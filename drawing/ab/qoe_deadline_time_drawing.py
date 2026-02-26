from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal

import plotly.graph_objects as go

from models import PsnrQualityTable, CollectLinesTable


# ============================================================
# Config: SVC reference levels (cumulative Kbps + PSNR)
# ============================================================
@dataclass(frozen=True)
class SVCRefLevels:
    l0_kbps: float
    l1_kbps: float  # cumulative (L0+L1)
    l2_kbps: float  # cumulative (L0+L1+L2)
    y0: float
    y1: float
    y2: float
    missing_y: float = 2.0


# ============================================================
# Experiment model + discovery
# ============================================================
@dataclass(frozen=True)
class Experiment:
    protocol: str              # UDP/BPP
    kind: str                  # fixed/dynamic
    bandwidth: Optional[float] # fixed => float
    name: str                  # folder name
    collect_dir: Path


def _parse_bw(name: str) -> Optional[float]:
    import re
    m = re.search(r"(\d+(?:\.\d+)?)\s*MB", name, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


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
# Low-level helpers
# ============================================================
def _dynamic_kind(folder: str) -> str:
    f = folder.lower()
    if "azalan" in f or "decreasing" in f:
        return "dec"
    if "degisken" in f or "değisken" in f or "değişken" in f or "variable" in f:
        return "var"
    if "artan" in f or "increasing" in f:
        return "inc"
    return "dyn"


def _auto_time_scale(dt: int) -> float:
    if dt >= 10_000_000_000:  # ns
        return 1e9
    if dt >= 10_000_000:      # us
        return 1e6
    if dt >= 10_000:          # ms
        return 1e3
    return 1.0


def _time_map_seconds_min(collect: CollectLinesTable) -> Dict[Tuple[int, int], float]:
    """
    FIX-2: (frame,layer) için EN ERKEN arrival (min) alınır.
    Times are normalized to seconds since t0.
    """
    raw_times: List[int] = []
    rows: List[Tuple[int, int, int]] = []

    for r in collect.collect_rows:
        if r.frame_no is None or r.quality_layer is None or r.time is None:
            continue
        fn = int(r.frame_no)
        ly = int(r.quality_layer)
        t = int(r.time)
        raw_times.append(t)
        rows.append((fn, ly, t))

    if not raw_times:
        return {}

    t0 = min(raw_times)
    tN = max(raw_times)
    scale = _auto_time_scale(tN - t0)

    out: Dict[Tuple[int, int], float] = {}
    for fn, ly, t in rows:
        key = (fn, ly)
        val = (t - t0) / scale
        if key in out:
            out[key] = min(out[key], val)
        else:
            out[key] = val
    return out


def _robust_max(vals: List[float], q: float = 0.995) -> float:
    v = [x for x in vals if x is not None]
    if not v:
        return 1.0
    v.sort()
    idx = int((len(v) - 1) * q)
    return v[idx]


def _subsample_markers(x: List[float], y: List[float], target_points: int = 36) -> Tuple[List[float], List[float]]:
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


# ============================================================
# Deadline-aware per-frame quality layer
# ============================================================
@dataclass(frozen=True)
class DeadlineResult:
    time_s: List[float]                # frame-time (i/fps)
    avg_bitrate_kbps: List[float]      # cumulative average QoE bitrate
    counters: Dict[str, int]           # 000/100/110/111
    interruption_count: int
    buffer_refill_count: int


def compute_qoe_avg_deadline_series(
    *,
    psnr: PsnrQualityTable,
    collect: CollectLinesTable,
    ref: SVCRefLevels,
    fps: float,
    initial_offset_s: float = 0.6,       # 600ms
    buffer_s: float = 1.5,               # seconds
    max_layer: int = 2,
    stall_policy: Literal["base", "max_prefix"] = "base",
    logger=None,
) -> DeadlineResult:
    """
    Deadline-aware QoE average bitrate (frame-time based).

    FIX-1: playout_start = initial_offset_s  (t0 zaten normalize edildi)
           Böylece "ilk frame daha geç geldi => daha geç playout başlar => avantaj" artefaktı kalkar.

    stall_policy:
      - "base": stall sadece L0 geç kalırsa (gerçekçi; enhancement late => stall yok)
      - "max_prefix": L1/L2 geç kalırsa da buffer tüketip stall edebilir (agresif kalite hedefi)
    """

    arrival_s = _time_map_seconds_min(collect)
    frames = psnr.frames
    n_frames = len(frames)

    if not arrival_s:
        return DeadlineResult(
            time_s=[f / fps for f in frames],
            avg_bitrate_kbps=[0.0 for _ in frames],
            counters={"000": n_frames, "100": 0, "110": 0, "111": 0},
            interruption_count=0,
            buffer_refill_count=0,
        )

    # FIX-1: playout schedule anchored only to offset (not to first-arrival)
    playout_start = float(initial_offset_s)

    buffer_remaining = float(buffer_s)
    interruption_count = 0
    buffer_refill_count = 1  # "initial buffer filled"

    c000 = c100 = c110 = c111 = 0
    sum_bitrate = 0.0

    out_t: List[float] = []
    out_avg: List[float] = []

    def contiguous_exists_level(frame_no: int) -> int:
        """
        Only checks existence (arrival recorded), not deadline.
        """
        if (frame_no, 0) not in arrival_s:
            return -1
        if (frame_no, 1) not in arrival_s:
            return 0
        if (frame_no, 2) not in arrival_s:
            return 1
        return 2

    def level_by_deadline(frame_no: int, deadline: float) -> int:
        """
        Highest contiguous prefix that arrives <= deadline.
        """
        t0 = arrival_s.get((frame_no, 0))
        if t0 is None or t0 > deadline:
            return -1
        t1 = arrival_s.get((frame_no, 1))
        if t1 is None or t1 > deadline:
            return 0
        t2 = arrival_s.get((frame_no, 2))
        if t2 is None or t2 > deadline:
            return 1
        return 2

    def stall_arrival_time(frame_no: int) -> Optional[float]:
        """
        Which arrival time is used to trigger buffer stall?
        - base: t(L0)
        - max_prefix: t(arrival of the highest contiguous EXISTING layer)
        """
        if stall_policy == "base":
            return arrival_s.get((frame_no, 0))

        # "max_prefix": use arrival of the highest contiguous layer that exists at all.
        ex = contiguous_exists_level(frame_no)
        if ex < 0:
            return None
        return arrival_s.get((frame_no, ex))

    # iterate frames
    frame_period = 1.0 / float(fps)

    for idx, f in enumerate(frames):
        frame_time = f / fps
        deadline = playout_start + idx * frame_period

        # ----------------------------
        # Buffer/stall logic (policy-driven)
        # ----------------------------
        t_stall = stall_arrival_time(f)
        if t_stall is not None and t_stall > deadline:
            lateness = t_stall - deadline
            if lateness <= buffer_remaining:
                buffer_remaining -= lateness
            else:
                # stall / interrupt
                shift = lateness - buffer_remaining
                playout_start += shift
                buffer_remaining = buffer_s
                interruption_count += 1
                buffer_refill_count += 1
                deadline = playout_start + idx * frame_period

        # ----------------------------
        # Determine quality level by deadline
        # ----------------------------
        ll = level_by_deadline(f, deadline)

        if ll < 0:
            c000 += 1
            br = 0.0
        elif ll == 0:
            c100 += 1
            br = ref.l0_kbps
        elif ll == 1:
            c110 += 1
            br = ref.l1_kbps
        else:
            c111 += 1
            br = ref.l2_kbps

        sum_bitrate += br
        out_t.append(frame_time)
        out_avg.append(sum_bitrate / (idx + 1))

    counters = {"000": c000, "100": c100, "110": c110, "111": c111}

    if logger is not None:
        r = _rates(counters)
        logger.info(
            f"[DEADLINE] policy={stall_policy} | counters 000={c000} 100={c100} 110={c110} 111={c111} "
            f"(111={r['p111']:.1f}%, 000={r['p000']:.1f}%) | final_avg={out_avg[-1]:.1f} Kbps "
            f"| interrupts={interruption_count} refills={buffer_refill_count}"
        )

    return DeadlineResult(
        time_s=out_t,
        avg_bitrate_kbps=out_avg,
        counters=counters,
        interruption_count=interruption_count,
        buffer_refill_count=buffer_refill_count,
    )


# ============================================================
# Plot styling (solid lines + markers)
# ============================================================
_BASE = {"UDP": "#1f77b4", "BPP": "#ff7f0e"}
_FIXED_MARKER = {1.0: "triangle-up", 2.0: "square", 3.0: "cross"}
_DYNAMIC_MARKER = {"dec": "diamond", "var": "circle"}  # Scenario1/Scenario2

def _fixed_color(proto: str, bw: float) -> str:
    # mild whitening by bw
    if abs(bw - 1.0) < 1e-9:
        a = 0.05
    elif abs(bw - 2.0) < 1e-9:
        a = 0.18
    else:
        a = 0.32

    def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    def _rgb_to_hex(r: int, g: int, b: int) -> str:
        return f"#{r:02x}{g:02x}{b:02x}"

    r, g, b = _hex_to_rgb(_BASE[proto])
    r2 = int(r * (1 - a) + 255 * a)
    g2 = int(g * (1 - a) + 255 * a)
    b2 = int(b * (1 - a) + 255 * a)
    return _rgb_to_hex(r2, g2, b2)

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
        title_text="<b>Deadline-aware Average Bitrate (Kbps)</b>",
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
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=color, width=line_width),
        name=name,
        legendgroup=legendgroup,
        showlegend=False,
    ))
    xm, ym = _subsample_markers(x, y, target_points=36)
    fig.add_trace(go.Scatter(
        x=xm, y=ym,
        mode="markers",
        marker=dict(symbol=marker_symbol, size=marker_size, color=color),
        name=name,
        legendgroup=legendgroup,
        legendgrouptitle_text=legendgrouptitle,
        showlegend=True,
    ))


# ============================================================
# Draw: Fixed & Dynamic (deadline-aware) + sanity checks
# ============================================================
def draw_fixed_plot_qoe_deadline(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    ref: SVCRefLevels,
    out_html: Path,
    fps: float,
    initial_offset_s: float,
    buffer_s: float,
    max_layer: int,
    stall_policy: Literal["base", "max_prefix"],
    logger,
) -> None:
    fig = go.Figure()
    all_y: List[float] = []

    # only 1/2/3MB
    fixed_123 = [e for e in fixed if e.bandwidth is not None and (
        abs(e.bandwidth-1.0)<1e-9 or abs(e.bandwidth-2.0)<1e-9 or abs(e.bandwidth-3.0)<1e-9
    )]
    by_proto: Dict[str, List[Experiment]] = {"UDP": [], "BPP": []}
    for e in fixed_123:
        by_proto.setdefault(e.protocol, []).append(e)
    for p in by_proto:
        by_proto[p].sort(key=lambda z: float(z.bandwidth))

    logger.info(f"[DEADLINE/FIXED] policy={stall_policy} fps={fps:g} offset={initial_offset_s:g}s buffer={buffer_s:g}s")

    # per-bw sanity
    sanity: Dict[float, Dict[str, Dict[str, object]]] = {}

    for proto, lst in by_proto.items():
        if not lst:
            continue

        lg_group = f"{proto}_FIX"
        lg_title = f"{proto} · Fixed"

        for e in lst:
            bw = float(e.bandwidth)
            name = f"{int(bw)}MB" if bw.is_integer() else f"{bw:g}MB"
            exp_tag = f"[DEADLINE/FIXED] {proto} {name} ({e.collect_dir})"

            try:
                collect = load_collect_table(e.collect_dir)
                s = compute_qoe_avg_deadline_series(
                    psnr=psnr,
                    collect=collect,
                    ref=ref,
                    fps=fps,
                    initial_offset_s=initial_offset_s,
                    buffer_s=buffer_s,
                    max_layer=max_layer,
                    stall_policy=stall_policy,
                    logger=None,  # per-experiment log below
                )

                all_y.extend(s.avg_bitrate_kbps)
                final_avg = float(s.avg_bitrate_kbps[-1]) if s.avg_bitrate_kbps else 0.0
                r = _rates(s.counters)

                logger.info(
                    f"{exp_tag} counters 000={s.counters['000']} 100={s.counters['100']} 110={s.counters['110']} 111={s.counters['111']} "
                    f"(111={r['p111']:.1f}%,000={r['p000']:.1f}%) | final_avg={final_avg:.1f} Kbps "
                    f"| interrupts={s.interruption_count} refills={s.buffer_refill_count}"
                )

                sanity.setdefault(float(round(bw, 3)), {})[proto] = {
                    "avg": final_avg,
                    "rates": r,
                    "interrupts": s.interruption_count,
                    "path": str(e.collect_dir),
                }

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
                    marker_size=6,
                )

            except Exception as ex:
                logger.error(f"{exp_tag} failed: {type(ex).__name__}: {ex}")
                continue

    # sanity compare
    SANITY_ERR_ABS_KBPS = 200.0
    SANITY_ERR_REL = 0.10

    for bw, row in sorted(sanity.items()):
        udp = row.get("UDP")
        bpp = row.get("BPP")
        if udp is None or bpp is None:
            logger.warning(f"[DEADLINE/SANITY] {bw:g}MB missing proto: have={list(row.keys())}")
            continue

        udp_avg = float(udp["avg"])
        bpp_avg = float(bpp["avg"])
        diff = bpp_avg - udp_avg

        logger.info(
            f"[DEADLINE/SANITY] {bw:g}MB | UDP={udp_avg:.1f} (111={udp['rates']['p111']:.1f}%,000={udp['rates']['p000']:.1f}%) "
            f"| BPP={bpp_avg:.1f} (111={bpp['rates']['p111']:.1f}%,000={bpp['rates']['p000']:.1f}%) "
            f"| diff(BPP-UDP)={diff:.1f} Kbps"
        )

        if diff <= 0:
            logger.warning(
                f"[DEADLINE/SANITY] BPP<=UDP at {bw:g}MB diff={diff:.1f} | UDP={udp['path']} | BPP={bpp['path']}"
            )
            abs_gap = udp_avg - bpp_avg
            rel_gap = abs_gap / udp_avg if udp_avg > 1e-9 else 0.0
            if abs_gap >= SANITY_ERR_ABS_KBPS or rel_gap >= SANITY_ERR_REL:
                logger.error(
                    f"[DEADLINE/SANITY] LARGE GAP at {bw:g}MB: UDP-BPP={abs_gap:.1f} ({rel_gap*100:.1f}%)"
                )

    if not all_y:
        logger.warning("[DEADLINE/FIXED] no series produced")
        return

    y_max = _robust_max(all_y, 0.995) * 1.10
    _apply_layout(fig, title=f"Deadline-aware Average Bitrate vs Time · Fixed (1/2/3MB) | FPS={fps:g} | policy={stall_policy}", y_max=y_max)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    logger.info(f"[DEADLINE/FIXED] Saved HTML: {out_html}")


def draw_dynamic_plot_qoe_deadline(
    *,
    psnr: PsnrQualityTable,
    dynamic: List[Experiment],
    ref: SVCRefLevels,
    out_html: Path,
    fps: float,
    initial_offset_s: float,
    buffer_s: float,
    max_layer: int,
    stall_policy: Literal["base", "max_prefix"],
    logger,
) -> None:
    fig = go.Figure()
    all_y: List[float] = []

    # only dec + var
    filtered: List[Tuple[Experiment, str]] = []
    for e in dynamic:
        k = _dynamic_kind(e.name)
        if k in ("dec", "var"):
            filtered.append((e, k))

    scenario_name = {"dec": "Scenario 1", "var": "Scenario 2"}

    logger.info(f"[DEADLINE/DYN] policy={stall_policy} fps={fps:g} offset={initial_offset_s:g}s buffer={buffer_s:g}s")

    # per-scenario sanity: scenario -> proto -> summary
    sanity: Dict[str, Dict[str, Dict[str, object]]] = {}

    dup: Dict[Tuple[str, str], int] = {}
    for e, k in filtered:
        proto = e.protocol
        base = scenario_name[k]
        dup_key = (proto, k)
        dup[dup_key] = dup.get(dup_key, 0) + 1
        label = base if dup[dup_key] == 1 else f"{base} #{dup[dup_key]}"

        exp_tag = f"[DEADLINE/DYN] {proto} {label} ({e.collect_dir})"
        try:
            collect = load_collect_table(e.collect_dir)
            s = compute_qoe_avg_deadline_series(
                psnr=psnr,
                collect=collect,
                ref=ref,
                fps=fps,
                initial_offset_s=initial_offset_s,
                buffer_s=buffer_s,
                max_layer=max_layer,
                stall_policy=stall_policy,
                logger=None,
            )

            all_y.extend(s.avg_bitrate_kbps)
            final_avg = float(s.avg_bitrate_kbps[-1]) if s.avg_bitrate_kbps else 0.0
            r = _rates(s.counters)

            logger.info(
                f"{exp_tag} counters 000={s.counters['000']} 100={s.counters['100']} 110={s.counters['110']} 111={s.counters['111']} "
                f"(111={r['p111']:.1f}%,000={r['p000']:.1f}%) | final_avg={final_avg:.1f} Kbps "
                f"| interrupts={s.interruption_count} refills={s.buffer_refill_count}"
            )

            # store scenario-level sanity only for non-duplicated base labels
            if "#" not in label:
                sanity.setdefault(base, {})[proto] = {
                    "avg": final_avg,
                    "rates": r,
                    "path": str(e.collect_dir),
                }

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
                marker_size=6,
            )

        except Exception as ex:
            logger.error(f"{exp_tag} failed: {type(ex).__name__}: {ex}")
            continue

    # sanity compare per scenario
    SANITY_ERR_ABS_KBPS = 200.0
    SANITY_ERR_REL = 0.10

    for scen, row in sanity.items():
        udp = row.get("UDP")
        bpp = row.get("BPP")
        if udp is None or bpp is None:
            logger.warning(f"[DEADLINE/SANITY] {scen} missing proto: have={list(row.keys())}")
            continue

        udp_avg = float(udp["avg"])
        bpp_avg = float(bpp["avg"])
        diff = bpp_avg - udp_avg

        logger.info(
            f"[DEADLINE/SANITY] {scen} | UDP={udp_avg:.1f} (111={udp['rates']['p111']:.1f}%,000={udp['rates']['p000']:.1f}%) "
            f"| BPP={bpp_avg:.1f} (111={bpp['rates']['p111']:.1f}%,000={bpp['rates']['p000']:.1f}%) "
            f"| diff(BPP-UDP)={diff:.1f} Kbps"
        )

        if diff <= 0:
            logger.warning(
                f"[DEADLINE/SANITY] BPP<=UDP in {scen} diff={diff:.1f} | UDP={udp['path']} | BPP={bpp['path']}"
            )
            abs_gap = udp_avg - bpp_avg
            rel_gap = abs_gap / udp_avg if udp_avg > 1e-9 else 0.0
            if abs_gap >= SANITY_ERR_ABS_KBPS or rel_gap >= SANITY_ERR_REL:
                logger.error(
                    f"[DEADLINE/SANITY] LARGE GAP in {scen}: UDP-BPP={abs_gap:.1f} ({rel_gap*100:.1f}%)"
                )

    if not all_y:
        logger.warning("[DEADLINE/DYN] no series produced")
        return

    y_max = _robust_max(all_y, 0.995) * 1.10
    _apply_layout(fig, title=f"Deadline-aware Average Bitrate vs Time · Dynamic (Scenario1/2) | FPS={fps:g} | policy={stall_policy}", y_max=y_max)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    logger.info(f"[DEADLINE/DYN] Saved HTML: {out_html}")


def draw_fixed_and_dynamic_qoe_deadline(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    dynamic: List[Experiment],
    ref: SVCRefLevels,
    out_dir: Path,
    fps: float = 24.0,
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    stall_policy: Literal["base", "max_prefix"] = "base",
    logger=None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_fixed_plot_qoe_deadline(
        psnr=psnr,
        fixed=fixed,
        ref=ref,
        out_html=out_dir / f"deadline_qoe_fixed_1_2_3_fps{int(fps)}_{stall_policy}.html",
        fps=fps,
        initial_offset_s=initial_offset_s,
        buffer_s=buffer_s,
        max_layer=max_layer,
        stall_policy=stall_policy,
        logger=logger,
    )

    draw_dynamic_plot_qoe_deadline(
        psnr=psnr,
        dynamic=dynamic,
        ref=ref,
        out_html=out_dir / f"deadline_qoe_dynamic_s1_s2_fps{int(fps)}_{stall_policy}.html",
        fps=fps,
        initial_offset_s=initial_offset_s,
        buffer_s=buffer_s,
        max_layer=max_layer,
        stall_policy=stall_policy,
        logger=logger,
    )