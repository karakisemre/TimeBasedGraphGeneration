from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from models import PsnrQualityTable, CollectLinesTable


# ============================================================
# Config: SVC reference levels (cumulative Kbps + PSNR)
# ============================================================
@dataclass(frozen=True)
class SVCRefLevels:
    l0_kbps: float  # base only
    l1_kbps: float  # L0+L1 (cumulative)
    l2_kbps: float  # L0+L1+L2 (cumulative)
    y0: float
    y1: float
    y2: float
    missing_y: float = 2.0  # old code used 2 for missing


# ============================================================
# Helpers
# ============================================================
def _contiguous_max_layer(received_layers: set[int], max_layer: int = 2) -> int:
    m = -1
    for l in range(0, max_layer + 1):
        if l in received_layers:
            m = l
        else:
            break
    return m


def _i_frames_from_psnr(psnr: PsnrQualityTable) -> List[int]:
    out: List[int] = []
    for f in psnr.frames:
        r0 = psnr.get(f, 0)
        if r0 and str(r0.frame_type).upper() == "I":
            out.append(f)
    out = sorted(set(out))
    if not out:
        raise ValueError("No I-frames found in PSNR reference table.")
    if out[0] != 0:
        raise ValueError(f"First I-frame is not at 0 (found {out[0]}).")
    return out


def _robust_max(vals: List[float], q: float = 0.995) -> float:
    v = [x for x in vals if x is not None]
    if not v:
        return 1.0
    v.sort()
    idx = int((len(v) - 1) * q)
    return v[idx]


def _subsample_markers(x: List[float], y: List[float], target_points: int = 40) -> Tuple[List[float], List[float]]:
    n = len(x)
    if n <= target_points:
        return x, y
    step = max(1, n // target_points)
    xm = x[::step]
    ym = y[::step]
    if xm[-1] != x[-1]:
        xm.append(x[-1])
        ym.append(y[-1])
    return xm, ym


def _parse_bw(name: str) -> Optional[float]:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*MB", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _dynamic_kind(folder: str) -> str:
    f = folder.lower()
    if "azalan" in f or "decreasing" in f:
        return "dec"
    if "degisken" in f or "değisken" in f or "değişken" in f or "variable" in f:
        return "var"
    if "artan" in f or "increasing" in f:
        return "inc"
    return "dyn"

def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"

def _mix_with_white(hex_color: str, alpha: float) -> str:
    """alpha=0 original, alpha=1 white."""
    alpha = max(0.0, min(1.0, alpha))
    r, g, b = _hex_to_rgb(hex_color)
    r2 = int(r * (1 - alpha) + 255 * alpha)
    g2 = int(g * (1 - alpha) + 255 * alpha)
    b2 = int(b * (1 - alpha) + 255 * alpha)
    return _rgb_to_hex(r2, g2, b2)

# ============================================================
# Experiments discovery
# ============================================================
@dataclass(frozen=True)
class Experiment:
    protocol: str              # UDP/BPP
    kind: str                  # fixed/dynamic
    bandwidth: Optional[float] # fixed => float
    name: str                  # folder name
    collect_dir: Path


def discover_fixed_bw_experiments(root: Path) -> List[Experiment]:
    out: List[Experiment] = []
    if not root.exists():
        return out

    for proto_dir in root.iterdir():
        if not proto_dir.is_dir():
            continue
        proto = proto_dir.name.upper()
        if proto not in ("UDP", "BPP"):
            continue

        for bw_dir in proto_dir.iterdir():
            if not bw_dir.is_dir():
                continue
            bw = _parse_bw(bw_dir.name)
            if bw is None:
                continue
            out.append(Experiment(proto, "fixed", bw, bw_dir.name, bw_dir))

    out.sort(key=lambda e: (e.protocol, e.bandwidth or 0.0))
    return out


def discover_dynamic_experiments(root: Path) -> List[Experiment]:
    out: List[Experiment] = []
    if not root.exists():
        return out

    for proto_dir in root.iterdir():
        if not proto_dir.is_dir():
            continue
        proto = proto_dir.name.upper()
        if proto not in ("UDP", "BPP"):
            continue

        for sc_dir in proto_dir.iterdir():
            if not sc_dir.is_dir():
                continue
            if (sc_dir / "CollectLines_by_frame.json").exists() or (sc_dir / "CollectLines.jsonl").exists() or (sc_dir / "CollectLines.txt").exists():
                out.append(Experiment(proto, "dynamic", None, sc_dir.name, sc_dir))

    out.sort(key=lambda e: (e.protocol, e.name))
    return out


def load_collect_table(d: Path) -> CollectLinesTable:
    p_byf = d / "CollectLines_by_frame.json"
    p_jsonl = d / "CollectLines.jsonl"
    p_txt = d / "CollectLines.txt"
    if p_byf.exists():
        return CollectLinesTable.from_json_by_frame(p_byf)
    if p_jsonl.exists():
        return CollectLinesTable.from_jsonl(p_jsonl)
    if p_txt.exists():
        return CollectLinesTable.from_txt(p_txt)
    raise FileNotFoundError(f"No CollectLines found in {d}")


# ============================================================
# New pipeline -> per-frame "last quality layer" extraction
# ============================================================
def _frame_received_prefix(
    collect: CollectLinesTable,
    *,
    max_layer: int = 2,
) -> Dict[int, int]:
    """
    From CollectLines: frame -> contiguous prefix max layer.
    -1 = nothing, 0 = L0, 1 = L0+L1, 2 = L0+L1+L2
    """
    received_layers_by_frame: Dict[int, set[int]] = {}
    for r in collect.collect_rows:
        if r.frame_no is None or r.quality_layer is None:
            continue
        fn = int(r.frame_no)
        ly = int(r.quality_layer)
        received_layers_by_frame.setdefault(fn, set()).add(ly)

    eff: Dict[int, int] = {}
    for fn, layers in received_layers_by_frame.items():
        eff[fn] = _contiguous_max_layer(layers, max_layer=max_layer)
    return eff


def compute_last_quality_layer_map(
    *,
    psnr: PsnrQualityTable,
    collect: CollectLinesTable,
    mode: str = "forward",   # "forward" or "checkpoint"
    max_layer: int = 2,
) -> Dict[int, int]:
    """
    Returns frame_no -> last_quality_layer (-1..2)
    "forward": GOP-start I limits P frames (classic decoding dependency)
    "checkpoint": NEXT I validates previous GOP (your earlier checkpoint logic)
    """
    if mode not in ("forward", "checkpoint"):
        raise ValueError("mode must be 'forward' or 'checkpoint'")

    frames = psnr.frames
    i_frames = _i_frames_from_psnr(psnr)

    eff = _frame_received_prefix(collect, max_layer=max_layer)
    eff_default = -1

    # helper: effective prefix for any frame
    def eff_f(f: int) -> int:
        return eff.get(f, eff_default)

    last: Dict[int, int] = {f: -1 for f in frames}

    if mode == "forward":
        # current GOP start I
        current_i = i_frames[0]
        eff_i = eff_f(current_i)
        for f in frames:
            if f in i_frames:
                current_i = f
                eff_i = eff_f(f)
                last[f] = eff_f(f)
            else:
                ef = eff_f(f)
                if ef < 0 or eff_i < 0:
                    last[f] = -1
                else:
                    last[f] = min(ef, eff_i)

        return last

    # checkpoint: [I_k .. I_next) validity depends on eff(I_next)
    for idx in range(len(i_frames) - 1):
        k = i_frames[idx]
        k_next = i_frames[idx + 1]
        eff_next = eff_f(k_next)

        for f in range(k, k_next):
            ef = eff_f(f)
            if ef < 0 or eff_next < 0:
                last[f] = -1
            else:
                last[f] = min(ef, eff_next)

    # tail after last I: fallback to forward within tail GOP
    last_i = i_frames[-1]
    eff_i = eff_f(last_i)
    for f in frames:
        if f < last_i:
            continue
        if f == last_i:
            last[f] = eff_f(f)
        else:
            ef = eff_f(f)
            if ef < 0 or eff_i < 0:
                last[f] = -1
            else:
                last[f] = min(ef, eff_i)

    return last


# ============================================================
# Average bitrate series (frame-time)
# ============================================================
@dataclass(frozen=True)
class QoESeries:
    time_s: List[float]
    avg_bitrate_kbps: List[float]
    avg_psnr: List[float]
    counters: Dict[str, int]  # final counters


def compute_qoe_avg_series(
    *,
    psnr: PsnrQualityTable,
    collect: CollectLinesTable,
    ref: SVCRefLevels,
    fps: float = 30.0,
    mode: str = "forward",
    max_layer: int = 2,
) -> QoESeries:
    """
    Frame-time based cumulative average QoE bitrate:
      avg(t) = (c100*L0 + c110*L1 + c111*L2) / N
    where L0/L1/L2 are cumulative Kbps.
    """

    last_layer = compute_last_quality_layer_map(psnr=psnr, collect=collect, mode=mode, max_layer=max_layer)
    frames = psnr.frames

    c000 = c100 = c110 = c111 = 0
    n = 0

    time_s: List[float] = []
    avg_bitrate: List[float] = []
    avg_psnr: List[float] = []

    sum_bitrate = 0.0
    sum_psnr = 0.0

    for f in frames:
        n += 1
        t = f / fps
        ll = last_layer.get(f, -1)

        if ll < 0:
            c000 += 1
            br = 0.0
            yp = ref.missing_y
        elif ll == 0:
            c100 += 1
            br = ref.l0_kbps
            yp = ref.y0
        elif ll == 1:
            c110 += 1
            br = ref.l1_kbps  # cumulative L0+L1
            yp = ref.y1
        else:
            c111 += 1
            br = ref.l2_kbps  # cumulative L0+L1+L2
            yp = ref.y2

        sum_bitrate += br
        sum_psnr += yp

        time_s.append(t)
        avg_bitrate.append(sum_bitrate / n)
        avg_psnr.append(sum_psnr / n)

    return QoESeries(
        time_s=time_s,
        avg_bitrate_kbps=avg_bitrate,
        avg_psnr=avg_psnr,
        counters={"000": c000, "100": c100, "110": c110, "111": c111},
    )


# ============================================================
# Plotting: 2 figures (Fixed vs Dynamic)
# ============================================================
_BASE = {"UDP": "#1f77b4", "BPP": "#ff7f0e"}  # saturated blue/orange

_FIXED_MARKER = {1.0: "triangle-up", 2.0: "square", 3.0: "cross"}
_DYNAMIC_MARKER = {"dec": "diamond", "var": "circle"}  # Scenario1/Scenario2


def _fixed_color(proto: str, bw: float) -> str:
    # 1MB darker, 3MB lighter but still visible
    if abs(bw - 1.0) < 1e-9:
        a = 0.05
    elif abs(bw - 2.0) < 1e-9:
        a = 0.20
    else:  # 3.0
        a = 0.35
    return _mix_with_white(_BASE[proto], a)


def _dynamic_color(proto: str, kind: str) -> str:
    if proto == "UDP":
        return {"dec": "#17becf", "var": "#9467bd"}.get(kind, _BASE[proto])
    return {"dec": "#d62728", "var": "#8c564b"}.get(kind, _BASE[proto])


def _apply_layout(fig: go.Figure, *, title: str, y_max: float) -> None:
    fig.update_layout(
        template="plotly_white",
        height=720,
        margin=dict(l=90, r=40, t=90, b=250),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center"),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.28, yanchor="top",
            font=dict(size=13),
            title_font=dict(size=14),
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor="rgba(0,0,0,0.08)",
            borderwidth=1,
            groupclick="toggleitem",
        ),
    )
    fig.update_xaxes(
        title_text="<b>Time (s)</b>",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        ticks="outside",
        showline=True,
        linecolor="rgba(0,0,0,0.25)",
    )
    fig.update_yaxes(
        title_text="<b>Average Bitrate (Kbps)</b>",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        ticks="outside",
        showline=True,
        linecolor="rgba(0,0,0,0.25)",
        range=[0, y_max],
        rangemode="tozero",
    )


def _add_line_with_sparse_markers(
    fig: go.Figure,
    *,
    x: List[float],
    y: List[float],
    color: str,
    name: str,
    legendgroup: str,
    legendgrouptitle: str,
    marker_symbol: str,
    line_width: float,
    marker_size: int,
) -> None:
    # Full line (no legend entry)
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=color, width=line_width),
            name=name,
            legendgroup=legendgroup,
            showlegend=False,
        )
    )
    # Sparse markers (legend entry)
    xm, ym = _subsample_markers(x, y, target_points=40)
    fig.add_trace(
        go.Scatter(
            x=xm, y=ym,
            mode="markers",
            marker=dict(symbol=marker_symbol, size=marker_size, color=color),
            name=name,
            legendgroup=legendgroup,
            legendgrouptitle_text=legendgrouptitle,
            showlegend=True,
        )
    )


def draw_fixed_plot_qoe(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    ref: SVCRefLevels,
    out_html: Path,
    fps: float = 30.0,
    mode: str = "forward",
    max_layer: int = 2,
    logger,  # <-- senin Logger sınıfından instance (main'den ver)
) -> None:
    """
    Fixed plot: only 1MB, 2MB, 3MB (UDP+BPP)
    Solid lines + sparse markers (no dashed).
    Logs counters and final average bitrate for each experiment.
    """
    fig = go.Figure()
    all_y: List[float] = []

    sanity_results: Dict[float, Dict[str, Dict[str, object]]] = {}

    def _rates(c: Dict[str, int]) -> Dict[str, float]:
        total = sum(c.values()) if c else 0
        if total <= 0:
            return {"p000": 0.0, "p100": 0.0, "p110": 0.0, "p111": 0.0, "total": 0.0}
        return {
            "p000": 100.0 * c["000"] / total,
            "p100": 100.0 * c["100"] / total,
            "p110": 100.0 * c["110"] / total,
            "p111": 100.0 * c["111"] / total,
            "total": float(total),
        }

    # thresholds
    SANITY_ERR_ABS_KBPS = 200.0  # UDP > BPP by >= 200 Kbps => error
    SANITY_ERR_REL = 0.10  # UDP > BPP by >= 10% of UDP => error

    # filter only 1/2/3MB
    fixed_123 = [
        e for e in fixed
        if e.bandwidth is not None and (
            abs(e.bandwidth - 1.0) < 1e-9 or abs(e.bandwidth - 2.0) < 1e-9 or abs(e.bandwidth - 3.0) < 1e-9
        )
    ]

    if not fixed_123:
        logger.warning("[FIXED] No fixed experiments found for 1/2/3MB.")
        return

    by_proto: Dict[str, List[Experiment]] = {"UDP": [], "BPP": []}
    for e in fixed_123:
        by_proto.setdefault(e.protocol, []).append(e)
    for p in by_proto:
        by_proto[p].sort(key=lambda z: float(z.bandwidth))

    logger.info(f"[FIXED] Drawing Average bitrate | mode={mode} | fps={fps:g} | max_layer={max_layer}")
    logger.info(f"[FIXED] Experiments count: UDP={len(by_proto.get('UDP', []))}, BPP={len(by_proto.get('BPP', []))}")

    for proto, lst in by_proto.items():
        if not lst:
            continue

        lg_group = f"{proto}_FIX"
        lg_title = f"{proto} · Fixed"

        for e in lst:
            bw = float(e.bandwidth)
            name = f"{int(bw)}MB" if bw.is_integer() else f"{bw:g}MB"
            exp_tag = f"[FIXED] {proto} {name} ({e.collect_dir})"

            try:
                collect = load_collect_table(e.collect_dir)

                # sanity: frame0 must have full prefix
                prefix = _frame_received_prefix(collect, max_layer=max_layer)
                if prefix.get(0, -1) < max_layer:
                    logger.critical(
                        f"{exp_tag} start-rule violated: frame0 prefix={prefix.get(0, -1)} expected={max_layer}. "
                        "This experiment may be invalid."
                    )

                s = compute_qoe_avg_series(
                    psnr=psnr,
                    collect=collect,
                    ref=ref,
                    fps=fps,
                    mode=mode,
                    max_layer=max_layer,
                )

                # log counters
                c = s.counters
                total = sum(c.values()) if c else 0
                if total == 0:
                    logger.warning(f"{exp_tag} counters empty (no frames).")
                else:
                    p111 = 100.0 * c["111"] / total
                    p110 = 100.0 * c["110"] / total
                    p100 = 100.0 * c["100"] / total
                    p000 = 100.0 * c["000"] / total
                    final_avg = s.avg_bitrate_kbps[-1] if s.avg_bitrate_kbps else 0.0
                    logger.info(
                        f"{exp_tag} counters: 000={c['000']} 100={c['100']} 110={c['110']} 111={c['111']} "
                        f"(rates: 000={p000:.1f}%, 100={p100:.1f}%, 110={p110:.1f}%, 111={p111:.1f}%) "
                        f"| final_avg={final_avg:.1f} Kbps"
                    )

                # ---- store for post-compare ----
                final_avg = float(s.avg_bitrate_kbps[-1]) if s.avg_bitrate_kbps else 0.0
                c = s.counters
                r = _rates(c)

                # normalize bw key (1.0/2.0/3.0)
                bw_key = float(round(bw, 3))
                sanity_results.setdefault(bw_key, {})[proto] = {
                    "avg": final_avg,
                    "c": c,
                    "rates": r,
                    "path": str(e.collect_dir),
                }

                # plotting
                all_y.extend(s.avg_bitrate_kbps)

                color = _fixed_color(proto, bw)
                symbol = _FIXED_MARKER.get(bw, "circle")

                _add_line_with_sparse_markers(
                    fig,
                    x=s.time_s,
                    y=s.avg_bitrate_kbps,
                    color=color,
                    name=name,
                    legendgroup=lg_group,
                    legendgrouptitle=lg_title,
                    marker_symbol=symbol,
                    line_width=3.2,
                    marker_size=7,
                )

            except FileNotFoundError as ex:
                logger.error(f"{exp_tag} missing CollectLines: {ex}")
                continue
            except Exception as ex:
                logger.error(f"{exp_tag} failed: {type(ex).__name__}: {ex}")
                continue

    # =========================================================
    # SANITY CHECK: compare UDP vs BPP per bandwidth
    # =========================================================
    for bw_key in sorted(sanity_results.keys()):
        row = sanity_results[bw_key]
        udp = row.get("UDP")
        bpp = row.get("BPP")

        if udp is None or bpp is None:
            logger.warning(f"[SANITY] Missing protocol for {bw_key:g}MB: have={list(row.keys())}")
            continue

        udp_avg = float(udp["avg"])
        bpp_avg = float(bpp["avg"])
        diff = bpp_avg - udp_avg  # expected positive

        udp_r = udp["rates"]
        bpp_r = bpp["rates"]

        # detailed one-line summary
        logger.info(
            f"[SANITY] {bw_key:g}MB | UDP={udp_avg:.1f} Kbps (111={udp_r['p111']:.1f}%,000={udp_r['p000']:.1f}%) "
            f"| BPP={bpp_avg:.1f} Kbps (111={bpp_r['p111']:.1f}%,000={bpp_r['p000']:.1f}%) "
            f"| diff(BPP-UDP)={diff:.1f} Kbps"
        )

        if diff <= 0:
            # BPP should be higher
            logger.warning(
                f"[SANITY] BPP <= UDP at {bw_key:g}MB: BPP-UDP={diff:.1f} Kbps "
                f"| paths: UDP={udp['path']} | BPP={bpp['path']}"
            )

            # escalate to error if far off
            abs_gap = udp_avg - bpp_avg
            rel_gap = abs_gap / udp_avg if udp_avg > 1e-9 else 0.0
            if abs_gap >= SANITY_ERR_ABS_KBPS or rel_gap >= SANITY_ERR_REL:
                logger.error(
                    f"[SANITY] LARGE GAP at {bw_key:g}MB: UDP-BPP={abs_gap:.1f} Kbps ({rel_gap * 100:.1f}%) "
                    f"| UDP 111={udp_r['p111']:.1f}% 000={udp_r['p000']:.1f}% "
                    f"| BPP 111={bpp_r['p111']:.1f}% 000={bpp_r['p000']:.1f}%"
                )

    if not all_y:
        logger.warning("[FIXED] No series produced (all experiments failed or empty).")
        return

    y_max = _robust_max(all_y, q=0.995) * 1.10
    _apply_layout(
        fig,
        title=f"Average Bitrate vs Time · Fixed (1/2/3MB) | Mode={mode} | FPS={fps:g}",
        y_max=y_max,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    logger.info(f"[FIXED] Saved HTML: {out_html}")

    try:
        import kaleido  # noqa: F401
        fig.write_image(str(out_html.with_suffix(".png")), scale=2)
        fig.write_image(str(out_html.with_suffix(".pdf")))
        logger.info(f"[FIXED] Saved PNG/PDF: {out_html.with_suffix('.png')}, {out_html.with_suffix('.pdf')}")
    except Exception as ex:
        logger.warning(f"[FIXED] kaleido export skipped: {type(ex).__name__}: {ex}")


def draw_dynamic_plot_qoe(
    *,
    psnr: PsnrQualityTable,
    dynamic: List[Experiment],
    ref: SVCRefLevels,
    out_html: Path,
    fps: float = 30.0,
    mode: str = "forward",
    max_layer: int = 2,
    logger,  # <-- senin Logger instance
) -> None:
    """
    Dynamic plot: ONLY Decreasing + Variable (Increasing excluded)
    Legend items:
      Scenario 1 = Decreasing
      Scenario 2 = Variable
    Logs counters and final average bitrate for each experiment.
    """
    fig = go.Figure()
    all_y: List[float] = []

    # filter only dec + var
    filtered: List[Tuple[Experiment, str]] = []
    for e in dynamic:
        k = _dynamic_kind(e.name)  # dec/var/inc/...
        if k in ("dec", "var"):
            filtered.append((e, k))

    if not filtered:
        logger.warning("[DYNAMIC] No dynamic experiments found for Decreasing/Variable.")
        return

    scenario_name = {"dec": "Scenario 1", "var": "Scenario 2"}
    dup: Dict[Tuple[str, str], int] = {}

    logger.info(f"[DYNAMIC] Drawing Average bitrate | mode={mode} | fps={fps:g} | max_layer={max_layer}")
    logger.info(f"[DYNAMIC] Experiments count (dec/var only): {len(filtered)}")

    for e, k in filtered:
        proto = e.protocol
        base = scenario_name[k]
        dup_key = (proto, k)
        dup[dup_key] = dup.get(dup_key, 0) + 1
        label = base if dup[dup_key] == 1 else f"{base} #{dup[dup_key]}"

        exp_tag = f"[DYNAMIC] {proto} {label} ({e.collect_dir})"

        try:
            collect = load_collect_table(e.collect_dir)

            # sanity: frame0 must have full prefix
            prefix = _frame_received_prefix(collect, max_layer=max_layer)
            if prefix.get(0, -1) < max_layer:
                logger.critical(
                    f"{exp_tag} start-rule violated: frame0 prefix={prefix.get(0, -1)} expected={max_layer}. "
                    "This experiment may be invalid."
                )

            s = compute_qoe_avg_series(
                psnr=psnr,
                collect=collect,
                ref=ref,
                fps=fps,
                mode=mode,
                max_layer=max_layer,
            )

            # log counters
            c = s.counters
            total = sum(c.values()) if c else 0
            if total == 0:
                logger.warning(f"{exp_tag} counters empty (no frames).")
            else:
                p111 = 100.0 * c["111"] / total
                p110 = 100.0 * c["110"] / total
                p100 = 100.0 * c["100"] / total
                p000 = 100.0 * c["000"] / total
                final_avg = s.avg_bitrate_kbps[-1] if s.avg_bitrate_kbps else 0.0
                logger.info(
                    f"{exp_tag} counters: 000={c['000']} 100={c['100']} 110={c['110']} 111={c['111']} "
                    f"(rates: 000={p000:.1f}%, 100={p100:.1f}%, 110={p110:.1f}%, 111={p111:.1f}%) "
                    f"| final_avg={final_avg:.1f} Kbps"
                )

            # plotting
            all_y.extend(s.avg_bitrate_kbps)

            lg_group = f"{proto}_DYN"
            lg_title = f"{proto} · Dynamic"

            color = _dynamic_color(proto, k)
            symbol = _DYNAMIC_MARKER.get(k, "circle")

            _add_line_with_sparse_markers(
                fig,
                x=s.time_s,
                y=s.avg_bitrate_kbps,
                color=color,
                name=label,
                legendgroup=lg_group,
                legendgrouptitle=lg_title,
                marker_symbol=symbol,
                line_width=3.6,
                marker_size=7,
            )

        except FileNotFoundError as ex:
            logger.error(f"{exp_tag} missing CollectLines: {ex}")
            continue
        except Exception as ex:
            logger.error(f"{exp_tag} failed: {type(ex).__name__}: {ex}")
            continue

    if not all_y:
        logger.warning("[DYNAMIC] No series produced (all experiments failed or empty).")
        return

    y_max = _robust_max(all_y, q=0.995) * 1.10
    _apply_layout(
        fig,
        title=f"Average Bitrate vs Time · Dynamic (Scenario1/2) | Mode={mode} | FPS={fps:g}",
        y_max=y_max,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    logger.info(f"[DYNAMIC] Saved HTML: {out_html}")

    try:
        import kaleido  # noqa: F401
        fig.write_image(str(out_html.with_suffix(".png")), scale=2)
        fig.write_image(str(out_html.with_suffix(".pdf")))
        logger.info(f"[DYNAMIC] Saved PNG/PDF: {out_html.with_suffix('.png')}, {out_html.with_suffix('.pdf')}")
    except Exception as ex:
        logger.warning(f"[DYNAMIC] kaleido export skipped: {type(ex).__name__}: {ex}")


def draw_fixed_and_dynamic_qoe(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    dynamic: List[Experiment],
    ref: SVCRefLevels,
    out_dir: Path,
    fps: float = 30.0,
    mode: str = "forward",
    max_layer: int = 2,
    logger
) -> None:
    """
    Produces exactly 2 plots:
      1) Fixed (1/2/3MB)
      2) Dynamic (Scenario1/2 = Dec/Var)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_fixed_plot_qoe(
        psnr=psnr,
        fixed=fixed,
        ref=ref,
        out_html=out_dir / f"qoe_fixed_1_2_3_{mode}_fps{int(fps)}.html",
        fps=fps,
        mode=mode,
        max_layer=max_layer,
        logger=logger
    )

    draw_dynamic_plot_qoe(
        psnr=psnr,
        dynamic=dynamic,
        ref=ref,
        out_html=out_dir / f"qoe_dynamic_s1_s2_{mode}_fps{int(fps)}.html",
        fps=fps,
        mode=mode,
        max_layer=max_layer,
        logger=logger
    )