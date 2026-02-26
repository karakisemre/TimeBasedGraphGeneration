from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as go

from models import PsnrQualityTable, CollectLinesTable


@dataclass(frozen=True)
class SVCRefLevels:
    l0_kbps: float
    l1_kbps: float  # cumulative
    l2_kbps: float  # cumulative
    y0: float
    y1: float
    y2: float
    missing_y: float = 2.0


@dataclass(frozen=True)
class Experiment:
    protocol: str
    kind: str
    bandwidth: Optional[float]
    name: str
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
    if dt >= 10_000_000_000:
        return 1e9
    if dt >= 10_000_000:
        return 1e6
    if dt >= 10_000:
        return 1e3
    return 1.0


def _time_map_seconds(collect: CollectLinesTable) -> Dict[Tuple[int, int], float]:
    times_raw: List[int] = []
    for r in collect.collect_rows:
        if r.frame_no is None or r.quality_layer is None or r.time is None:
            continue
        times_raw.append(int(r.time))
    if not times_raw:
        return {}
    t0 = min(times_raw)
    tN = max(times_raw)
    scale = _auto_time_scale(tN - t0)

    out: Dict[Tuple[int, int], float] = {}
    for r in collect.collect_rows:
        if r.frame_no is None or r.quality_layer is None or r.time is None:
            continue
        out[(int(r.frame_no), int(r.quality_layer))] = (int(r.time) - t0) / scale
    return out


def _robust_max(vals: List[float], q: float = 0.995) -> float:
    v = [x for x in vals if x is not None]
    if not v:
        return 1.0
    v.sort()
    idx = int((len(v) - 1) * q)
    return v[idx]


def _subsample_markers(x: List[float], y: List[float], target_points: int = 34) -> Tuple[List[float], List[float]]:
    n = len(x)
    if n <= target_points:
        return x, y
    step = max(1, n // target_points)
    xm = x[::step]
    ym = y[::step]
    if xm[-1] != x[-1]:
        xm.append(x[-1]); ym.append(y[-1])
    return xm, ym


# ------------------------------------------------------------
# Arrival-upgrade global average series
# ------------------------------------------------------------
def compute_arrival_upgrade_avg_series(
    *,
    psnr: PsnrQualityTable,
    collect: CollectLinesTable,
    ref: SVCRefLevels,
    max_layer: int = 2,
) -> Tuple[List[float], List[float]]:
    """
    Event-time based:
      - each frame starts at bitrate 0
      - when L0 arrives -> bitrate becomes L0_kbps
      - when L1 arrives (and L0 already) -> becomes L1_kbps
      - when L2 arrives (and L0+L1 already) -> becomes L2_kbps
    We maintain global sum_kbps over ALL frames and compute avg = sum_kbps / Nframes.
    """
    arrival = _time_map_seconds(collect)
    frames = psnr.frames
    n_frames = max(1, len(frames))

    # per-frame arrivals
    per_frame: Dict[int, Dict[int, float]] = {}
    for (f, l), t in arrival.items():
        if 0 <= l <= max_layer:
            per_frame.setdefault(f, {})[l] = t

    # events: (time_s, delta_kbps)
    events: List[Tuple[float, float]] = []
    current_level: Dict[int, int] = {f: -1 for f in frames}

    # helper to push delta based on upgrade
    def add_upgrade_event(f: int, new_level: int, t: float) -> None:
        old = current_level.get(f, -1)
        if new_level <= old:
            return
        # compute kbps for old/new level (cumulative)
        def level_kbps(lv: int) -> float:
            if lv < 0:
                return 0.0
            if lv == 0:
                return ref.l0_kbps
            if lv == 1:
                return ref.l1_kbps
            return ref.l2_kbps
        delta = level_kbps(new_level) - level_kbps(old)
        current_level[f] = new_level
        if delta != 0.0:
            events.append((t, delta))

    # build events per frame in time order
    for f in frames:
        a = per_frame.get(f, {})
        # require contiguous: L1 only after L0, L2 only after L1
        if 0 in a:
            add_upgrade_event(f, 0, a[0])
        if 0 in a and 1 in a and a[1] >= a[0]:
            add_upgrade_event(f, 1, a[1])
        if 0 in a and 1 in a and 2 in a and a[2] >= a[1] >= a[0]:
            add_upgrade_event(f, 2, a[2])

    if not events:
        return [0.0], [0.0]

    events.sort(key=lambda x: x[0])

    # merge same-time
    merged: List[Tuple[float, float]] = []
    cur_t, cur_d = events[0]
    for t, d in events[1:]:
        if t == cur_t:
            cur_d += d
        else:
            merged.append((cur_t, cur_d))
            cur_t, cur_d = t, d
    merged.append((cur_t, cur_d))

    sum_kbps = 0.0
    xs: List[float] = []
    ys: List[float] = []
    for t, d in merged:
        sum_kbps += d
        xs.append(t)
        ys.append(sum_kbps / n_frames)

    return xs, ys


# ============================================================
# Plot styling
# ============================================================
_BASE = {"UDP": "#1f77b4", "BPP": "#ff7f0e"}
_FIXED_MARKER = {1.0: "triangle-up", 2.0: "square", 3.0: "cross"}
_DYNAMIC_MARKER = {"dec": "diamond", "var": "circle"}

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
        title_text="<b>Arrival-upgrade Avg Bitrate (Kbps)</b>",
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
    xm, ym = _subsample_markers(x, y, target_points=34)
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
# Two plots: fixed and dynamic (arrival upgrade)
# ============================================================
def draw_fixed_plot_arrival_upgrade(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    ref: SVCRefLevels,
    out_html: Path,
    max_layer: int = 2,
    logger=None,
) -> None:
    fig = go.Figure()
    all_y: List[float] = []

    fixed_123 = [e for e in fixed if e.bandwidth is not None and (abs(e.bandwidth-1.0)<1e-9 or abs(e.bandwidth-2.0)<1e-9 or abs(e.bandwidth-3.0)<1e-9)]
    by_proto: Dict[str, List[Experiment]] = {"UDP": [], "BPP": []}
    for e in fixed_123:
        by_proto.setdefault(e.protocol, []).append(e)
    for p in by_proto:
        by_proto[p].sort(key=lambda z: float(z.bandwidth))

    for proto, lst in by_proto.items():
        if not lst:
            continue
        lg_group = f"{proto}_FIX"
        lg_title = f"{proto} · Fixed"

        for e in lst:
            bw = float(e.bandwidth)
            name = f"{int(bw)}MB" if bw.is_integer() else f"{bw:g}MB"
            tag = f"[ARRIVAL/FIXED] {proto} {name} ({e.collect_dir})"

            try:
                collect = load_collect_table(e.collect_dir)
                x, y = compute_arrival_upgrade_avg_series(psnr=psnr, collect=collect, ref=ref, max_layer=max_layer)
                all_y.extend(y)

                color = _BASE[proto]
                symbol = _FIXED_MARKER.get(bw, "circle")

                _add_line_with_sparse_markers(
                    fig,
                    x=x, y=y,
                    color=color,
                    name=name,
                    legendgroup=lg_group,
                    legendgrouptitle=lg_title,
                    marker_symbol=symbol,
                    line_width=3.2,
                    marker_size=6,
                )

                if logger is not None:
                    logger.info(f"{tag} final_avg_at_end={y[-1]:.1f} Kbps")

            except Exception as ex:
                if logger is not None:
                    logger.error(f"{tag} failed: {type(ex).__name__}: {ex}")
                continue

    if not all_y:
        if logger is not None:
            logger.warning("[ARRIVAL/FIXED] no series produced")
        return

    y_max = _robust_max(all_y, 0.995) * 1.10
    _apply_layout(fig, title="Arrival-upgrade Avg Bitrate vs Time · Fixed (1/2/3MB)", y_max=y_max)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    if logger is not None:
        logger.info(f"[ARRIVAL/FIXED] Saved HTML: {out_html}")


def draw_dynamic_plot_arrival_upgrade(
    *,
    psnr: PsnrQualityTable,
    dynamic: List[Experiment],
    ref: SVCRefLevels,
    out_html: Path,
    max_layer: int = 2,
    logger=None,
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
    dup: Dict[Tuple[str, str], int] = {}

    for e, k in filtered:
        proto = e.protocol
        base = scenario_name[k]
        dup_key = (proto, k)
        dup[dup_key] = dup.get(dup_key, 0) + 1
        label = base if dup[dup_key] == 1 else f"{base} #{dup[dup_key]}"

        tag = f"[ARRIVAL/DYN] {proto} {label} ({e.collect_dir})"
        try:
            collect = load_collect_table(e.collect_dir)
            x, y = compute_arrival_upgrade_avg_series(psnr=psnr, collect=collect, ref=ref, max_layer=max_layer)
            all_y.extend(y)

            lg_group = f"{proto}_DYN"
            lg_title = f"{proto} · Dynamic"
            color = _BASE[proto]
            symbol = _DYNAMIC_MARKER.get(k, "circle")

            _add_line_with_sparse_markers(
                fig,
                x=x, y=y,
                color=color,
                name=label,
                legendgroup=lg_group,
                legendgrouptitle=lg_title,
                marker_symbol=symbol,
                line_width=3.6,
                marker_size=6,
            )

            if logger is not None:
                logger.info(f"{tag} final_avg_at_end={y[-1]:.1f} Kbps")

        except Exception as ex:
            if logger is not None:
                logger.error(f"{tag} failed: {type(ex).__name__}: {ex}")
            continue

    if not all_y:
        if logger is not None:
            logger.warning("[ARRIVAL/DYN] no series produced")
        return

    y_max = _robust_max(all_y, 0.995) * 1.10
    _apply_layout(fig, title="Arrival-upgrade Avg Bitrate vs Time · Dynamic (Scenario1/2)", y_max=y_max)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    if logger is not None:
        logger.info(f"[ARRIVAL/DYN] Saved HTML: {out_html}")


def draw_fixed_and_dynamic_arrival_upgrade(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    dynamic: List[Experiment],
    ref: SVCRefLevels,
    out_dir: Path,
    max_layer: int = 2,
    logger=None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_fixed_plot_arrival_upgrade(
        psnr=psnr,
        fixed=fixed,
        ref=ref,
        out_html=out_dir / "arrival_upgrade_fixed_1_2_3.html",
        max_layer=max_layer,
        logger=logger,
    )

    draw_dynamic_plot_arrival_upgrade(
        psnr=psnr,
        dynamic=dynamic,
        ref=ref,
        out_html=out_dir / "arrival_upgrade_dynamic_s1_s2.html",
        max_layer=max_layer,
        logger=logger,
    )