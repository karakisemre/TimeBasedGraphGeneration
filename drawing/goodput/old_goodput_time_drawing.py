from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from models import PsnrQualityTable, CollectLinesTable


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


def _auto_time_scale(dt: int) -> float:
    if dt >= 10_000_000_000:  # ns
        return 1e9
    if dt >= 10_000_000:      # us
        return 1e6
    if dt >= 10_000:          # ms
        return 1e3
    return 1.0


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


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix_with_white(hex_color: str, alpha: float) -> str:
    alpha = max(0.0, min(1.0, alpha))
    r, g, b = _hex_to_rgb(hex_color)
    r2 = int(r * (1 - alpha) + 255 * alpha)
    g2 = int(g * (1 - alpha) + 255 * alpha)
    b2 = int(b * (1 - alpha) + 255 * alpha)
    return _rgb_to_hex(r2, g2, b2)


def _robust_max(vals: List[float], q: float = 0.995) -> float:
    v = [x for x in vals if x is not None]
    if not v:
        return 1.0
    v.sort()
    idx = int((len(v) - 1) * q)
    return v[idx]


def _subsample_points(x: List[float], y: List[float], target_points: int = 40) -> Tuple[List[float], List[float]]:
    """Marker’ları seyrek göstermek için (line full kalsın)."""
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


# ============================================================
# Core compute
# ============================================================
def compute_avg_bitrate_series(
    *,
    psnr: PsnrQualityTable,
    collect: CollectLinesTable,
    max_layer: int = 2,
    mode: str = "forward",         # checkpoint | forward | merge
    require_full_start_iframe: bool = True,
    y_unit: str = "Kbps",          # Kbps | Mbps
    min_elapsed_s: float = 1.0,    # ilk spike’ı kes
) -> Tuple[List[float], List[float]]:

    if mode not in ("checkpoint", "forward", "merge"):
        raise ValueError("mode must be checkpoint|forward|merge")

    received_layers_by_frame: Dict[int, set[int]] = {}
    time_by_frame_layer: Dict[Tuple[int, int], int] = {}

    for r in collect.collect_rows:
        if r.frame_no is None or r.quality_layer is None or r.time is None:
            continue
        fn = int(r.frame_no)
        ly = int(r.quality_layer)
        received_layers_by_frame.setdefault(fn, set()).add(ly)
        time_by_frame_layer[(fn, ly)] = int(r.time)

    if require_full_start_iframe:
        needed = set(range(0, max_layer + 1))
        start_layers = received_layers_by_frame.get(0, set())
        if not needed.issubset(start_layers):
            raise ValueError(
                f"Start condition failed: frame 0 must include layers {sorted(needed)}. "
                f"Found layers={sorted(start_layers)}"
            )

    i_frames = _i_frames_from_psnr(psnr)
    events: List[Tuple[int, int]] = []

    def add_event(t_raw: int, bits: int) -> None:
        if bits:
            events.append((t_raw, bits))

    # I frames: immediate (prefix)
    for k in i_frames:
        eff = _contiguous_max_layer(received_layers_by_frame.get(k, set()), max_layer=max_layer)
        if eff < 0:
            continue
        for l in range(0, eff + 1):
            t = time_by_frame_layer.get((k, l))
            row = psnr.get(k, l)
            if t is not None and row is not None:
                add_event(t, int(row.bits))

    def eff_iframe(fn: int) -> int:
        return _contiguous_max_layer(received_layers_by_frame.get(fn, set()), max_layer=max_layer)

    for idx in range(len(i_frames) - 1):
        k = i_frames[idx]
        k_next = i_frames[idx + 1]

        eff_k = eff_iframe(k)
        eff_next = eff_iframe(k_next)

        pending_bits = [0] * (max_layer + 1)
        pending_events: List[List[Tuple[int, int]]] = [[] for _ in range(max_layer + 1)]

        for f in range(k + 1, k_next):
            eff_f = _contiguous_max_layer(received_layers_by_frame.get(f, set()), max_layer=max_layer)
            if eff_f < 0:
                continue
            for l in range(0, eff_f + 1):
                t = time_by_frame_layer.get((f, l))
                row = psnr.get(f, l)
                if t is None or row is None:
                    continue
                b = int(row.bits)

                if mode == "forward":
                    if l <= eff_k:
                        add_event(t, b)
                else:
                    pending_bits[l] += b
                    pending_events[l].append((t, b))

        if mode == "forward":
            continue

        # checkpoint/merge: NEXT I decides prefix validity
        for l in range(0, max_layer + 1):
            if l > eff_next:
                break

            if mode == "checkpoint":
                t_commit = time_by_frame_layer.get((k_next, l))
                if t_commit is not None and pending_bits[l] > 0:
                    add_event(t_commit, pending_bits[l])
            else:  # merge
                for t_p, b in pending_events[l]:
                    add_event(t_p, b)

    if not events:
        return [0.0], [0.0]

    events.sort(key=lambda x: x[0])

    # merge same-time
    merged: List[Tuple[int, int]] = []
    cur_t, cur_b = events[0]
    for t, b in events[1:]:
        if t == cur_t:
            cur_b += b
        else:
            merged.append((cur_t, cur_b))
            cur_t, cur_b = t, b
    merged.append((cur_t, cur_b))

    t0 = merged[0][0]
    tN = merged[-1][0]
    scale = _auto_time_scale(tN - t0)

    mult = 1e3 if y_unit.lower() == "kbps" else 1e6

    times: List[float] = []
    avg: List[float] = []
    cum_bits = 0

    for t_raw, dbits in merged:
        cum_bits += dbits
        elapsed_s = (t_raw - t0) / scale
        if elapsed_s <= 0:
            continue
        if elapsed_s < min_elapsed_s:
            continue

        bps = cum_bits / elapsed_s
        times.append(elapsed_s)
        avg.append(bps / mult)

    if len(times) == 1:
        times = [times[0], times[0] + 1e-9]
        avg = [avg[0], avg[0]]

    return times, avg


# ============================================================
# Experiments discovery
# ============================================================
@dataclass(frozen=True)
class Experiment:
    protocol: str              # UDP/BPP
    kind: str                  # fixed/dynamic
    bandwidth: Optional[float] # fixed => float, dynamic => None
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
# Filtering + labeling
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


def _keep_fixed_1_2_3(e: Experiment) -> bool:
    if e.bandwidth is None:
        return False
    # 1,2,3 MB filter (tolerant)
    return abs(e.bandwidth - 1.0) < 1e-9 or abs(e.bandwidth - 2.0) < 1e-9 or abs(e.bandwidth - 3.0) < 1e-9


# ============================================================
# Plot styling
# ============================================================
_BASE = {"UDP": "#1f77b4", "BPP": "#ff7f0e"}  # saturated blue / orange

# fixed bandwidth marker symbols
_FIXED_MARKER = {
    1.0: "triangle-up",
    2.0: "square",
    3.0: "cross",
}

# dynamic scenario marker symbols (Scenario 1/2)
_DYNAMIC_MARKER = {
    "dec": "diamond",
    "var": "circle",
}

def _fixed_color(proto: str, bw: float) -> str:
    # 1MB darker, 3MB lighter but still visible
    if abs(bw - 1.0) < 1e-9:
        a = 0.05
    elif abs(bw - 2.0) < 1e-9:
        a = 0.22
    else:  # 3.0
        a = 0.38
    return _mix_with_white(_BASE[proto], a)

def _dynamic_color(proto: str, kind: str) -> str:
    # keep dynamic very distinguishable
    if proto == "UDP":
        return {"dec": "#17becf", "var": "#9467bd"}.get(kind, _BASE[proto])
    else:
        return {"dec": "#d62728", "var": "#8c564b"}.get(kind, _BASE[proto])


# ============================================================
# Figure builders (2 plots)
# ============================================================
def _apply_layout(fig: go.Figure, *, title: str, y_unit: str, y_max: float) -> None:
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
        title_text=f"<b>Goodput ({y_unit})</b>",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        ticks="outside",
        showline=True,
        linecolor="rgba(0,0,0,0.25)",
        range=[0, y_max],
        rangemode="tozero",
    )

def _add_line_plus_markers(
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
    showlegend: bool = True,
) -> None:
    # full line (no legend)
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
    # sparse markers (legend shown)
    xm, ym = _subsample_points(x, y, target_points=36)
    fig.add_trace(
        go.Scatter(
            x=xm, y=ym,
            mode="markers",
            marker=dict(symbol=marker_symbol, size=marker_size, color=color, line=dict(width=0)),
            name=name,
            legendgroup=legendgroup,
            legendgrouptitle_text=legendgrouptitle,
            showlegend=showlegend,
        )
    )


def draw_fixed_plot(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    out_html: Path,
    mode: str = "forward",
    y_unit: str = "Kbps",
    max_layer: int = 2,
    min_elapsed_s: float = 1.0,
) -> None:
    """
    Fixed plot: only 1MB, 2MB, 3MB (UDP+BPP)
    Lines solid; markers distinguish bandwidth.
    """
    fig = go.Figure()
    all_y: List[float] = []

    fixed_123 = [e for e in fixed if _keep_fixed_1_2_3(e)]
    fixed_by_proto: Dict[str, List[Experiment]] = {"UDP": [], "BPP": []}
    for e in fixed_123:
        fixed_by_proto[e.protocol].append(e)
    for proto in fixed_by_proto:
        fixed_by_proto[proto].sort(key=lambda z: float(z.bandwidth))

    for proto, lst in fixed_by_proto.items():
        if not lst:
            continue
        lg_group = f"{proto}_FIX"
        lg_title = f"{proto} · Fixed"

        for e in lst:
            bw = float(e.bandwidth)
            collect = load_collect_table(e.collect_dir)
            x, y = compute_avg_bitrate_series(
                psnr=psnr,
                collect=collect,
                max_layer=max_layer,
                mode=mode,
                y_unit=y_unit,
                min_elapsed_s=min_elapsed_s,
                require_full_start_iframe=True,
            )
            all_y.extend(y)

            color = _fixed_color(proto, bw)
            symbol = _FIXED_MARKER.get(bw, "circle")
            name = f"{int(bw)}MB" if bw.is_integer() else f"{bw:g}MB"

            _add_line_plus_markers(
                fig,
                x=x, y=y,
                color=color,
                name=name,
                legendgroup=lg_group,
                legendgrouptitle=lg_title,
                marker_symbol=symbol,
                line_width=3.2,
                marker_size=4,
                showlegend=True,
            )

    y_max = _robust_max(all_y, q=0.995) * 1.10
    _apply_layout(fig, title=f"Goodput vs Time · Fixed (1/2/3MB) | Mode={mode} | Y={y_unit}", y_unit=y_unit, y_max=y_max)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    try:
        import kaleido  # noqa: F401
        fig.write_image(str(out_html.with_suffix(".png")), scale=2)
        fig.write_image(str(out_html.with_suffix(".pdf")))
    except Exception:
        pass


def draw_dynamic_plot(
    *,
    psnr: PsnrQualityTable,
    dynamic: List[Experiment],
    out_html: Path,
    mode: str = "forward",
    y_unit: str = "Kbps",
    max_layer: int = 2,
    min_elapsed_s: float = 1.0,
) -> None:
    """
    Dynamic plot: only Decreasing + Variable (Increasing excluded).
    Legend shows Scenario 1/2 (Scenario1=Decreasing, Scenario2=Variable).
    Lines solid; markers distinguish scenarios.
    """
    fig = go.Figure()
    all_y: List[float] = []

    # Filter only dec+var
    dyn_filtered: List[Tuple[Experiment, str]] = []
    for e in dynamic:
        k = _dynamic_kind(e.name)
        if k in ("dec", "var"):
            dyn_filtered.append((e, k))

    # For each protocol, map:
    # dec -> Scenario 1, var -> Scenario 2
    scenario_name = {"dec": "Scenario 1", "var": "Scenario 2"}

    # If multiple dec or var experiments exist per protocol, suffix #2, #3
    dup: Dict[Tuple[str, str], int] = {}

    for e, k in dyn_filtered:
        collect = load_collect_table(e.collect_dir)
        x, y = compute_avg_bitrate_series(
            psnr=psnr,
            collect=collect,
            max_layer=max_layer,
            mode=mode,
            y_unit=y_unit,
            min_elapsed_s=min_elapsed_s,
            require_full_start_iframe=True,
        )
        all_y.extend(y)

        proto = e.protocol
        lg_group = f"{proto}_DYN"
        lg_title = f"{proto} · Dynamic"

        base = scenario_name[k]
        dk = (proto, k)
        dup[dk] = dup.get(dk, 0) + 1
        name = base if dup[dk] == 1 else f"{base} #{dup[dk]}"

        color = _dynamic_color(proto, k)
        symbol = _DYNAMIC_MARKER.get(k, "circle")

        _add_line_plus_markers(
            fig,
            x=x, y=y,
            color=color,
            name=name,
            legendgroup=lg_group,
            legendgrouptitle=lg_title,
            marker_symbol=symbol,
            line_width=3.8,
            marker_size=5,
            showlegend=True,
        )

    y_max = _robust_max(all_y, q=0.995) * 1.10
    _apply_layout(fig, title=f"Goodput vs Time · Dynamic | Mode={mode} | Y={y_unit}", y_unit=y_unit, y_max=y_max)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    try:
        import kaleido  # noqa: F401
        fig.write_image(str(out_html.with_suffix(".png")), scale=2)
        fig.write_image(str(out_html.with_suffix(".pdf")))
    except Exception:
        pass


def draw_fixed_and_dynamic(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    dynamic: List[Experiment],
    out_dir: Path,
    mode: str = "forward",
    y_unit: str = "Kbps",
    max_layer: int = 2,
    min_elapsed_s: float = 1.0,
) -> None:
    """
    Convenience wrapper: produces exactly 2 plots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_fixed_plot(
        psnr=psnr,
        fixed=fixed,
        out_html=out_dir / f"fixed_1_2_3_{mode}_{y_unit.lower()}.html",
        mode=mode,
        y_unit=y_unit,
        max_layer=max_layer,
        min_elapsed_s=min_elapsed_s,
    )

    draw_dynamic_plot(
        psnr=psnr,
        dynamic=dynamic,
        out_html=out_dir / f"dynamic_dec_var_{mode}_{y_unit.lower()}.html",
        mode=mode,
        y_unit=y_unit,
        max_layer=max_layer,
        min_elapsed_s=min_elapsed_s,
    )