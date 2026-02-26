from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Sequence, Union

import plotly.graph_objects as go

from models import PsnrQualityTable, CollectLinesTable


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
# Helpers (layers / I-frame)
# ============================================================
def _contiguous_max_layer(layers: set[int], max_layer: int = 2) -> int:
    """Layers must be contiguous from 0: {0,1}->1, {0,2}->0, {1}->-1"""
    m = -1
    for l in range(0, max_layer + 1):
        if l in layers:
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


def _dynamic_kind(folder: str) -> str:
    f = folder.lower()
    if "azalan" in f or "decreasing" in f:
        return "dec"
    if "degisken" in f or "değisken" in f or "değişken" in f or "variable" in f:
        return "var"
    if "artan" in f or "increasing" in f:
        return "inc"
    return "dyn"


# ============================================================
# Deadline time normalization (min timestamp per key)
# ============================================================
def _auto_time_scale(dt: int) -> float:
    if dt >= 10_000_000_000:  # ns
        return 1e9
    if dt >= 10_000_000:      # us
        return 1e6
    if dt >= 10_000:          # ms
        return 1e3
    return 1.0


def _arrival_map_seconds_min(collect: CollectLinesTable) -> Dict[Tuple[int, int], float]:
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


# ============================================================
# FIX: subsample markers (MISSING BEFORE)
# ============================================================
def _subsample_markers(
    x: Sequence[float],
    y: Sequence[Union[int, float]],
    target_points: int = 36,
) -> Tuple[List[float], List[float]]:
    """
    Markerları seyrek göstermek için.
    y int olabilir -> float'a çeviriyoruz (plotly sorun çıkarmasın).
    """
    n = len(x)
    if n == 0:
        return [], []
    if n <= target_points:
        return list(x), [float(v) for v in y]

    step = max(1, n // target_points)
    xm = list(x[::step])
    ym = [float(v) for v in y[::step]]

    if xm[-1] != x[-1]:
        xm.append(float(x[-1]))
        ym.append(float(y[-1]))

    return xm, ym


# ============================================================
# Per-frame quality level maps  last_quality_layer ∈ {-1,0,1,2}
# ============================================================
def _frame_received_prefix_map(collect: CollectLinesTable, *, max_layer: int = 2) -> Dict[int, int]:
    received: Dict[int, set[int]] = {}
    for r in collect.collect_rows:
        if r.frame_no is None or r.quality_layer is None:
            continue
        fn = int(r.frame_no)
        ly = int(r.quality_layer)
        if 0 <= ly <= max_layer:
            received.setdefault(fn, set()).add(ly)

    eff: Dict[int, int] = {}
    for fn, layers in received.items():
        eff[fn] = _contiguous_max_layer(layers, max_layer=max_layer)
    return eff


def compute_last_quality_layer_map_forward_or_checkpoint(
    *,
    psnr: PsnrQualityTable,
    collect: CollectLinesTable,
    mode: Literal["forward", "checkpoint"],
    max_layer: int = 2,
) -> Dict[int, int]:
    if mode not in ("forward", "checkpoint"):
        raise ValueError("mode must be forward|checkpoint")

    frames = psnr.frames
    i_frames = _i_frames_from_psnr(psnr)
    eff = _frame_received_prefix_map(collect, max_layer=max_layer)

    def eff_f(f: int) -> int:
        return eff.get(f, -1)

    last: Dict[int, int] = {f: -1 for f in frames}

    if mode == "forward":
        eff_i = eff_f(i_frames[0])
        for f in frames:
            if f in i_frames:
                eff_i = eff_f(f)
                last[f] = eff_f(f)
            else:
                ef = eff_f(f)
                last[f] = -1 if (ef < 0 or eff_i < 0) else min(ef, eff_i)
        return last

    # checkpoint
    for idx in range(len(i_frames) - 1):
        k = i_frames[idx]
        k_next = i_frames[idx + 1]
        eff_next = eff_f(k_next)
        for f in range(k, k_next):
            ef = eff_f(f)
            last[f] = -1 if (ef < 0 or eff_next < 0) else min(ef, eff_next)

    # tail: forward fallback
    last_i = i_frames[-1]
    eff_i = eff_f(last_i)
    for f in frames:
        if f < last_i:
            continue
        if f == last_i:
            last[f] = eff_f(f)
        else:
            ef = eff_f(f)
            last[f] = -1 if (ef < 0 or eff_i < 0) else min(ef, eff_i)

    return last


@dataclass(frozen=True)
class DeadlineQualityResult:
    last_layer: Dict[int, int]  # frame -> -1..2
    interruptions: int
    refills: int


def compute_last_quality_layer_map_deadline(
    *,
    psnr: PsnrQualityTable,
    collect: CollectLinesTable,
    fps: float,
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    stall_policy: Literal["base", "max_prefix"] = "base",
) -> DeadlineQualityResult:
    arrival = _arrival_map_seconds_min(collect)
    frames = psnr.frames
    if not arrival:
        return DeadlineQualityResult(last_layer={f: -1 for f in frames}, interruptions=0, refills=0)

    playout_start = float(initial_offset_s)
    buffer_remaining = float(buffer_s)
    interrupts = 0
    refills = 1
    period = 1.0 / float(fps)

    def contiguous_exists_level(frame_no: int) -> int:
        if (frame_no, 0) not in arrival:
            return -1
        if (frame_no, 1) not in arrival:
            return 0
        if (frame_no, 2) not in arrival:
            return 1
        return 2

    def stall_arrival_time(frame_no: int) -> Optional[float]:
        if stall_policy == "base":
            return arrival.get((frame_no, 0))
        ex = contiguous_exists_level(frame_no)
        if ex < 0:
            return None
        return arrival.get((frame_no, ex))

    def level_by_deadline(frame_no: int, deadline: float) -> int:
        t0 = arrival.get((frame_no, 0))
        if t0 is None or t0 > deadline:
            return -1
        t1 = arrival.get((frame_no, 1))
        if t1 is None or t1 > deadline:
            return 0
        t2 = arrival.get((frame_no, 2))
        if t2 is None or t2 > deadline:
            return 1
        return 2

    last: Dict[int, int] = {f: -1 for f in frames}

    for idx, f in enumerate(frames):
        deadline = playout_start + idx * period

        t_stall = stall_arrival_time(f)
        if t_stall is not None and t_stall > deadline:
            lateness = t_stall - deadline
            if lateness <= buffer_remaining:
                buffer_remaining -= lateness
            else:
                shift = lateness - buffer_remaining
                playout_start += shift
                buffer_remaining = buffer_s
                interrupts += 1
                refills += 1
                deadline = playout_start + idx * period

        last[f] = level_by_deadline(f, deadline)

    return DeadlineQualityResult(last_layer=last, interruptions=interrupts, refills=refills)


# ============================================================
# Quality change series
# ============================================================
@dataclass(frozen=True)
class QualityChangeSeries:
    time_s: List[float]
    cum_changes: List[int]
    final_changes: int
    changes_per_sec: float


def compute_quality_change_series(
    *,
    frames: List[int],
    last_layer_map: Dict[int, int],
    fps: float,
    change_mode: Literal["any", "upgrade", "downgrade"] = "any",
) -> QualityChangeSeries:
    t: List[float] = []
    cum: List[int] = []
    c = 0

    if not frames:
        return QualityChangeSeries(time_s=[], cum_changes=[], final_changes=0, changes_per_sec=0.0)

    prev = last_layer_map.get(frames[0], -1)

    for i, f in enumerate(frames):
        cur = last_layer_map.get(f, -1)
        if i > 0:
            if change_mode == "any":
                changed = (cur != prev)
            elif change_mode == "upgrade":
                changed = (cur > prev)
            else:
                changed = (cur < prev)
            if changed:
                c += 1

        t.append(f / fps)
        cum.append(c)
        prev = cur

    total_time = (frames[-1] / fps)
    cps = (c / total_time) if total_time > 1e-9 else 0.0

    return QualityChangeSeries(time_s=t, cum_changes=cum, final_changes=c, changes_per_sec=cps)


# ============================================================
# Plot styling
# ============================================================
_BASE = {"UDP": "#1f77b4", "BPP": "#ff7f0e"}
_FIXED_MARKER = {1.0: "triangle-up", 2.0: "square", 3.0: "cross"}
_DYNAMIC_MARKER = {"dec": "diamond", "var": "circle"}

def _fixed_color(proto: str, bw: float) -> str:
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

def _robust_max(vals: List[float], q: float = 0.995) -> float:
    v = [x for x in vals if x is not None]
    if not v:
        return 1.0
    v.sort()
    idx = int((len(v) - 1) * q)
    return v[idx]

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
        title_text="<b>Cumulative Quality Changes (#)</b>",
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
    y: List[int],
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
# Draw: Fixed and Dynamic (2 plots)
# ============================================================
def draw_fixed_plot_quality_changes(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    out_html: Path,
    fps: float = 24.0,
    mode: Literal["forward", "checkpoint", "deadline_base", "deadline_max_prefix"] = "checkpoint",
    change_mode: Literal["any", "upgrade", "downgrade"] = "any",
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    logger=None,
) -> None:
    fig = go.Figure()
    all_y: List[float] = []

    fixed_123 = [e for e in fixed if e.bandwidth is not None and (
        abs(e.bandwidth-1.0)<1e-9 or abs(e.bandwidth-2.0)<1e-9 or abs(e.bandwidth-3.0)<1e-9
    )]
    by_proto: Dict[str, List[Experiment]] = {"UDP": [], "BPP": []}
    for e in fixed_123:
        by_proto.setdefault(e.protocol, []).append(e)
    for p in by_proto:
        by_proto[p].sort(key=lambda z: float(z.bandwidth))

    if logger:
        logger.info(f"[QCHANGE/FIXED] mode={mode} change_mode={change_mode} fps={fps:g}")

    for proto, lst in by_proto.items():
        if not lst:
            continue

        lg_group = f"{proto}_FIX"
        lg_title = f"{proto} · Fixed"

        for e in lst:
            bw = float(e.bandwidth)
            name = f"{int(bw)}MB" if bw.is_integer() else f"{bw:g}MB"
            exp_tag = f"[QCHANGE/FIXED] {proto} {name} ({e.collect_dir})"

            try:
                collect = load_collect_table(e.collect_dir)

                if mode in ("forward", "checkpoint"):
                    last_map = compute_last_quality_layer_map_forward_or_checkpoint(
                        psnr=psnr, collect=collect, mode=mode, max_layer=max_layer
                    )
                    extra = ""
                else:
                    stall = "base" if mode == "deadline_base" else "max_prefix"
                    dq = compute_last_quality_layer_map_deadline(
                        psnr=psnr, collect=collect, fps=fps,
                        initial_offset_s=initial_offset_s, buffer_s=buffer_s,
                        max_layer=max_layer, stall_policy=stall
                    )
                    last_map = dq.last_layer
                    extra = f" | interrupts={dq.interruptions} refills={dq.refills}"

                s = compute_quality_change_series(
                    frames=psnr.frames, last_layer_map=last_map, fps=fps, change_mode=change_mode
                )

                if logger:
                    logger.info(f"{exp_tag} final_changes={s.final_changes} changes_per_sec={s.changes_per_sec:.3f}{extra}")

                # plot
                all_y.extend([float(v) for v in s.cum_changes])
                color = _fixed_color(proto, bw)
                symbol = _FIXED_MARKER.get(bw, "circle")

                _add_line_with_sparse_markers(
                    fig,
                    x=s.time_s, y=s.cum_changes,
                    color=color,
                    name=name,
                    legendgroup=lg_group,
                    legendgrouptitle=lg_title,
                    marker_symbol=symbol,
                    line_width=3.0,
                    marker_size=6,
                )

            except Exception as ex:
                if logger:
                    logger.error(f"{exp_tag} failed: {type(ex).__name__}: {ex}")
                continue

    if not fig.data:
        if logger:
            logger.warning("[QCHANGE/FIXED] No traces added -> skip writing.")
        return

    y_max = _robust_max(all_y, 0.995) * 1.10
    _apply_layout(fig, title=f"Quality Change Count vs Time · Fixed (1/2/3MB) | mode={mode} | fps={fps:g}", y_max=y_max)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    if logger:
        logger.info(f"[QCHANGE/FIXED] Saved HTML: {out_html}")


def draw_dynamic_plot_quality_changes(
    *,
    psnr: PsnrQualityTable,
    dynamic: List[Experiment],
    out_html: Path,
    fps: float = 24.0,
    mode: Literal["forward", "checkpoint", "deadline_base", "deadline_max_prefix"] = "checkpoint",
    change_mode: Literal["any", "upgrade", "downgrade"] = "any",
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    logger=None,
) -> None:
    fig = go.Figure()
    all_y: List[float] = []

    filtered: List[Tuple[Experiment, str]] = []
    for e in dynamic:
        k = _dynamic_kind(e.name)
        if k in ("dec", "var"):
            filtered.append((e, k))

    scenario_name = {"dec": "Scenario 1", "var": "Scenario 2"}
    dup: Dict[Tuple[str, str], int] = {}

    if logger:
        logger.info(f"[QCHANGE/DYN] mode={mode} change_mode={change_mode} fps={fps:g}")

    for e, k in filtered:
        proto = e.protocol
        base = scenario_name[k]
        dk = (proto, k)
        dup[dk] = dup.get(dk, 0) + 1
        label = base if dup[dk] == 1 else f"{base} #{dup[dk]}"

        exp_tag = f"[QCHANGE/DYN] {proto} {label} ({e.collect_dir})"

        try:
            collect = load_collect_table(e.collect_dir)

            if mode in ("forward", "checkpoint"):
                last_map = compute_last_quality_layer_map_forward_or_checkpoint(
                    psnr=psnr, collect=collect, mode=mode, max_layer=max_layer
                )
                extra = ""
            else:
                stall = "base" if mode == "deadline_base" else "max_prefix"
                dq = compute_last_quality_layer_map_deadline(
                    psnr=psnr, collect=collect, fps=fps,
                    initial_offset_s=initial_offset_s, buffer_s=buffer_s,
                    max_layer=max_layer, stall_policy=stall
                )
                last_map = dq.last_layer
                extra = f" | interrupts={dq.interruptions} refills={dq.refills}"

            s = compute_quality_change_series(
                frames=psnr.frames, last_layer_map=last_map, fps=fps, change_mode=change_mode
            )

            if logger:
                logger.info(f"{exp_tag} final_changes={s.final_changes} changes_per_sec={s.changes_per_sec:.3f}{extra}")

            all_y.extend([float(v) for v in s.cum_changes])

            lg_group = f"{proto}_DYN"
            lg_title = f"{proto} · Dynamic"
            color = _dynamic_color(proto, k)
            symbol = _DYNAMIC_MARKER.get(k, "circle")

            _add_line_with_sparse_markers(
                fig,
                x=s.time_s, y=s.cum_changes,
                color=color,
                name=label,
                legendgroup=lg_group,
                legendgrouptitle=lg_title,
                marker_symbol=symbol,
                line_width=3.2,
                marker_size=6,
            )

        except Exception as ex:
            if logger:
                logger.error(f"{exp_tag} failed: {type(ex).__name__}: {ex}")
            continue

    if not fig.data:
        if logger:
            logger.warning("[QCHANGE/DYN] No traces added -> skip writing.")
        return

    y_max = _robust_max(all_y, 0.995) * 1.10
    _apply_layout(fig, title=f"Quality Change Count vs Time · Dynamic (Scenario1/2) | mode={mode} | fps={fps:g}", y_max=y_max)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    if logger:
        logger.info(f"[QCHANGE/DYN] Saved HTML: {out_html}")


def draw_fixed_and_dynamic_quality_changes(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    dynamic: List[Experiment],
    out_dir: Path,
    fps: float = 24.0,
    mode: Literal["forward", "checkpoint", "deadline_base", "deadline_max_prefix"] = "checkpoint",
    change_mode: Literal["any", "upgrade", "downgrade"] = "any",
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    logger=None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    draw_fixed_plot_quality_changes(
        psnr=psnr,
        fixed=fixed,
        out_html=out_dir / f"qchange_fixed_1_2_3_{mode}_{change_mode}_fps{int(fps)}.html",
        fps=fps,
        mode=mode,
        change_mode=change_mode,
        initial_offset_s=initial_offset_s,
        buffer_s=buffer_s,
        max_layer=max_layer,
        logger=logger,
    )

    draw_dynamic_plot_quality_changes(
        psnr=psnr,
        dynamic=dynamic,
        out_html=out_dir / f"qchange_dynamic_s1_s2_{mode}_{change_mode}_fps{int(fps)}.html",
        fps=fps,
        mode=mode,
        change_mode=change_mode,
        initial_offset_s=initial_offset_s,
        buffer_s=buffer_s,
        max_layer=max_layer,
        logger=logger,
    )


from collections import deque

# ============================================================
# Rolling change-rate series (changes/sec)
# ============================================================
@dataclass(frozen=True)
class QualityChangeRateSeries:
    time_s: List[float]
    rate_per_s: List[float]
    final_rate_per_s: float


def _change_event(prev: int, cur: int, change_mode: Literal["any", "upgrade", "downgrade"]) -> int:
    if change_mode == "any":
        return 1 if cur != prev else 0
    if change_mode == "upgrade":
        return 1 if cur > prev else 0
    return 1 if cur < prev else 0


def compute_quality_change_rate_series(
    *,
    frames: List[int],
    last_layer_map: Dict[int, int],
    fps: float,
    window_s: float = 2.0,
    change_mode: Literal["any", "upgrade", "downgrade"] = "upgrade",
) -> QualityChangeRateSeries:
    """
    Rolling change rate: changes/sec in a sliding window of length window_s.

    For each frame i:
      rate(i) = (# of change-events in last window) / (window_duration_seconds)

    window_duration_seconds uses actual frames in window (for early prefix).
    """
    if not frames:
        return QualityChangeRateSeries(time_s=[], rate_per_s=[], final_rate_per_s=0.0)

    # change events per frame index (event at i means change from i-1 -> i)
    events: List[int] = [0] * len(frames)
    prev = last_layer_map.get(frames[0], -1)
    for i in range(1, len(frames)):
        cur = last_layer_map.get(frames[i], -1)
        events[i] = _change_event(prev, cur, change_mode)
        prev = cur

    w_frames = max(1, int(round(window_s * fps)))

    # sliding sum over events
    q = deque()  # store events
    s = 0

    t_out: List[float] = []
    r_out: List[float] = []

    for i, f in enumerate(frames):
        ev = events[i]
        q.append(ev)
        s += ev

        if len(q) > w_frames:
            s -= q.popleft()

        # actual window duration (early points have shorter window)
        eff_frames = len(q)
        eff_dt = eff_frames / fps
        rate = (s / eff_dt) if eff_dt > 1e-9 else 0.0

        t_out.append(f / fps)
        r_out.append(rate)

    return QualityChangeRateSeries(
        time_s=t_out,
        rate_per_s=r_out,
        final_rate_per_s=float(r_out[-1]) if r_out else 0.0,
    )


def _apply_layout_rate(fig: go.Figure, *, title: str, y_max: float) -> None:
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
        title_text="<b>Quality Change Rate (changes/s)</b>",
        title_font=dict(size=18),
        tickfont=dict(size=14),
        ticks="outside",
        showline=True,
        linecolor="rgba(0,0,0,0.25)",
        range=[0, y_max],
        rangemode="tozero",
    )


def _add_line_with_sparse_markers_floaty(
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
# Draw: Rolling change-rate vs time (Fixed + Dynamic)
# ============================================================
def draw_fixed_plot_quality_change_rate(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    out_html: Path,
    fps: float = 24.0,
    mode: Literal["forward", "checkpoint", "deadline_base", "deadline_max_prefix"] = "checkpoint",
    change_mode: Literal["any", "upgrade", "downgrade"] = "upgrade",
    window_s: float = 2.0,
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    logger=None,
) -> None:
    fig = go.Figure()
    all_y: List[float] = []

    fixed_123 = [e for e in fixed if e.bandwidth is not None and (
        abs(e.bandwidth-1.0)<1e-9 or abs(e.bandwidth-2.0)<1e-9 or abs(e.bandwidth-3.0)<1e-9
    )]
    by_proto: Dict[str, List[Experiment]] = {"UDP": [], "BPP": []}
    for e in fixed_123:
        by_proto.setdefault(e.protocol, []).append(e)
    for p in by_proto:
        by_proto[p].sort(key=lambda z: float(z.bandwidth))

    if logger:
        logger.info(f"[QRATE/FIXED] mode={mode} change_mode={change_mode} window={window_s:g}s fps={fps:g}")

    for proto, lst in by_proto.items():
        if not lst:
            continue

        lg_group = f"{proto}_FIX"
        lg_title = f"{proto} · Fixed"

        for e in lst:
            bw = float(e.bandwidth)
            name = f"{int(bw)}MB" if bw.is_integer() else f"{bw:g}MB"
            exp_tag = f"[QRATE/FIXED] {proto} {name} ({e.collect_dir})"

            try:
                collect = load_collect_table(e.collect_dir)

                if mode in ("forward", "checkpoint"):
                    last_map = compute_last_quality_layer_map_forward_or_checkpoint(
                        psnr=psnr, collect=collect, mode=mode, max_layer=max_layer
                    )
                    extra = ""
                else:
                    stall = "base" if mode == "deadline_base" else "max_prefix"
                    dq = compute_last_quality_layer_map_deadline(
                        psnr=psnr, collect=collect, fps=fps,
                        initial_offset_s=initial_offset_s, buffer_s=buffer_s,
                        max_layer=max_layer, stall_policy=stall
                    )
                    last_map = dq.last_layer
                    extra = f" | interrupts={dq.interruptions} refills={dq.refills}"

                s = compute_quality_change_rate_series(
                    frames=psnr.frames,
                    last_layer_map=last_map,
                    fps=fps,
                    window_s=window_s,
                    change_mode=change_mode,
                )

                all_y.extend(s.rate_per_s)

                if logger:
                    logger.info(f"{exp_tag} final_rate={s.final_rate_per_s:.3f} changes/s{extra}")

                color = _fixed_color(proto, bw)
                symbol = _FIXED_MARKER.get(bw, "circle")

                _add_line_with_sparse_markers_floaty(
                    fig,
                    x=s.time_s,
                    y=s.rate_per_s,
                    color=color,
                    name=name,
                    legendgroup=lg_group,
                    legendgrouptitle=lg_title,
                    marker_symbol=symbol,
                    line_width=3.0,
                    marker_size=6,
                )

            except Exception as ex:
                if logger:
                    logger.error(f"{exp_tag} failed: {type(ex).__name__}: {ex}")
                continue

    if not fig.data:
        if logger:
            logger.warning("[QRATE/FIXED] No traces added -> skip writing.")
        return

    y_max = _robust_max(all_y, 0.995) * 1.10
    _apply_layout_rate(
        fig,
        title=f"Quality Change Rate vs Time · Fixed (1/2/3MB) | mode={mode} | window={window_s:g}s | fps={fps:g}",
        y_max=y_max,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    if logger:
        logger.info(f"[QRATE/FIXED] Saved HTML: {out_html}")


def draw_dynamic_plot_quality_change_rate(
    *,
    psnr: PsnrQualityTable,
    dynamic: List[Experiment],
    out_html: Path,
    fps: float = 24.0,
    mode: Literal["forward", "checkpoint", "deadline_base", "deadline_max_prefix"] = "checkpoint",
    change_mode: Literal["any", "upgrade", "downgrade"] = "upgrade",
    window_s: float = 2.0,
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    logger=None,
) -> None:
    fig = go.Figure()
    all_y: List[float] = []

    # only dec+var
    filtered: List[Tuple[Experiment, str]] = []
    for e in dynamic:
        k = _dynamic_kind(e.name)
        if k in ("dec", "var"):
            filtered.append((e, k))

    scenario_name = {"dec": "Scenario 1", "var": "Scenario 2"}
    dup: Dict[Tuple[str, str], int] = {}

    if logger:
        logger.info(f"[QRATE/DYN] mode={mode} change_mode={change_mode} window={window_s:g}s fps={fps:g}")

    for e, k in filtered:
        proto = e.protocol
        base = scenario_name[k]
        dk = (proto, k)
        dup[dk] = dup.get(dk, 0) + 1
        label = base if dup[dk] == 1 else f"{base} #{dup[dk]}"

        exp_tag = f"[QRATE/DYN] {proto} {label} ({e.collect_dir})"

        try:
            collect = load_collect_table(e.collect_dir)

            if mode in ("forward", "checkpoint"):
                last_map = compute_last_quality_layer_map_forward_or_checkpoint(
                    psnr=psnr, collect=collect, mode=mode, max_layer=max_layer
                )
                extra = ""
            else:
                stall = "base" if mode == "deadline_base" else "max_prefix"
                dq = compute_last_quality_layer_map_deadline(
                    psnr=psnr, collect=collect, fps=fps,
                    initial_offset_s=initial_offset_s, buffer_s=buffer_s,
                    max_layer=max_layer, stall_policy=stall
                )
                last_map = dq.last_layer
                extra = f" | interrupts={dq.interruptions} refills={dq.refills}"

            s = compute_quality_change_rate_series(
                frames=psnr.frames,
                last_layer_map=last_map,
                fps=fps,
                window_s=window_s,
                change_mode=change_mode,
            )

            all_y.extend(s.rate_per_s)

            if logger:
                logger.info(f"{exp_tag} final_rate={s.final_rate_per_s:.3f} changes/s{extra}")

            lg_group = f"{proto}_DYN"
            lg_title = f"{proto} · Dynamic"
            color = _dynamic_color(proto, k)
            symbol = _DYNAMIC_MARKER.get(k, "circle")

            _add_line_with_sparse_markers_floaty(
                fig,
                x=s.time_s,
                y=s.rate_per_s,
                color=color,
                name=label,
                legendgroup=lg_group,
                legendgrouptitle=lg_title,
                marker_symbol=symbol,
                line_width=3.2,
                marker_size=6,
            )

        except Exception as ex:
            if logger:
                logger.error(f"{exp_tag} failed: {type(ex).__name__}: {ex}")
            continue

    if not fig.data:
        if logger:
            logger.warning("[QRATE/DYN] No traces added -> skip writing.")
        return

    y_max = _robust_max(all_y, 0.995) * 1.10
    _apply_layout_rate(
        fig,
        title=f"Quality Change Rate vs Time · Dynamic (Scenario1/2) | mode={mode} | window={window_s:g}s | fps={fps:g}",
        y_max=y_max,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    if logger:
        logger.info(f"[QRATE/DYN] Saved HTML: {out_html}")


# ============================================================
# Overall slope bar chart (avg changes/sec)
# ============================================================
def _bar_layout(fig: go.Figure, *, title: str) -> None:
    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=80, r=30, t=90, b=120),
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor="center"),
        barmode="group",
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.25, yanchor="top"),
    )
    fig.update_xaxes(title_text="<b>Case</b>", title_font=dict(size=16), tickfont=dict(size=13))
    fig.update_yaxes(title_text="<b>Avg Change Rate (changes/s)</b>", title_font=dict(size=16), tickfont=dict(size=13), rangemode="tozero")


def draw_fixed_change_rate_barchart(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    out_html: Path,
    fps: float = 24.0,
    mode: Literal["forward", "checkpoint", "deadline_base", "deadline_max_prefix"] = "checkpoint",
    change_mode: Literal["any", "upgrade", "downgrade"] = "upgrade",
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    logger=None,
) -> None:
    # collect slopes per bw per proto
    cases = ["1MB", "2MB", "3MB"]
    vals: Dict[str, Dict[str, float]] = {"UDP": {}, "BPP": {}}

    fixed_123 = [e for e in fixed if e.bandwidth is not None and (
        abs(e.bandwidth-1.0)<1e-9 or abs(e.bandwidth-2.0)<1e-9 or abs(e.bandwidth-3.0)<1e-9
    )]

    for e in fixed_123:
        bw = float(e.bandwidth)
        case = "1MB" if abs(bw-1.0)<1e-9 else ("2MB" if abs(bw-2.0)<1e-9 else "3MB")
        try:
            collect = load_collect_table(e.collect_dir)

            if mode in ("forward", "checkpoint"):
                last_map = compute_last_quality_layer_map_forward_or_checkpoint(psnr=psnr, collect=collect, mode=mode, max_layer=max_layer)
            else:
                stall = "base" if mode == "deadline_base" else "max_prefix"
                dq = compute_last_quality_layer_map_deadline(
                    psnr=psnr, collect=collect, fps=fps,
                    initial_offset_s=initial_offset_s, buffer_s=buffer_s,
                    max_layer=max_layer, stall_policy=stall
                )
                last_map = dq.last_layer

            s = compute_quality_change_series(frames=psnr.frames, last_layer_map=last_map, fps=fps, change_mode=change_mode)
            vals[e.protocol][case] = s.changes_per_sec
        except Exception as ex:
            if logger:
                logger.error(f"[BAR/FIXED] {e.protocol} {case} failed: {type(ex).__name__}: {ex}")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="UDP", x=cases, y=[vals["UDP"].get(c, 0.0) for c in cases], marker_color=_BASE["UDP"]))
    fig.add_trace(go.Bar(name="BPP", x=cases, y=[vals["BPP"].get(c, 0.0) for c in cases], marker_color=_BASE["BPP"]))
    _bar_layout(fig, title=f"Overall Quality-Upgrade Rate (avg) · Fixed | mode={mode} | fps={fps:g}")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    if logger:
        logger.info(f"[BAR/FIXED] Saved HTML: {out_html}")


def draw_dynamic_change_rate_barchart(
    *,
    psnr: PsnrQualityTable,
    dynamic: List[Experiment],
    out_html: Path,
    fps: float = 24.0,
    mode: Literal["forward", "checkpoint", "deadline_base", "deadline_max_prefix"] = "checkpoint",
    change_mode: Literal["any", "upgrade", "downgrade"] = "upgrade",
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    logger=None,
) -> None:
    cases = ["Scenario 1", "Scenario 2"]
    vals: Dict[str, Dict[str, float]] = {"UDP": {}, "BPP": {}}

    scenario_name = {"dec": "Scenario 1", "var": "Scenario 2"}

    for e in dynamic:
        k = _dynamic_kind(e.name)
        if k not in ("dec", "var"):
            continue
        case = scenario_name[k]

        try:
            collect = load_collect_table(e.collect_dir)

            if mode in ("forward", "checkpoint"):
                last_map = compute_last_quality_layer_map_forward_or_checkpoint(psnr=psnr, collect=collect, mode=mode, max_layer=max_layer)
            else:
                stall = "base" if mode == "deadline_base" else "max_prefix"
                dq = compute_last_quality_layer_map_deadline(
                    psnr=psnr, collect=collect, fps=fps,
                    initial_offset_s=initial_offset_s, buffer_s=buffer_s,
                    max_layer=max_layer, stall_policy=stall
                )
                last_map = dq.last_layer

            s = compute_quality_change_series(frames=psnr.frames, last_layer_map=last_map, fps=fps, change_mode=change_mode)
            vals[e.protocol][case] = s.changes_per_sec

        except Exception as ex:
            if logger:
                logger.error(f"[BAR/DYN] {e.protocol} {case} failed: {type(ex).__name__}: {ex}")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="UDP", x=cases, y=[vals["UDP"].get(c, 0.0) for c in cases], marker_color=_BASE["UDP"]))
    fig.add_trace(go.Bar(name="BPP", x=cases, y=[vals["BPP"].get(c, 0.0) for c in cases], marker_color=_BASE["BPP"]))
    _bar_layout(fig, title=f"Overall Quality-Upgrade Rate (avg) · Dynamic | mode={mode} | fps={fps:g}")

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    if logger:
        logger.info(f"[BAR/DYN] Saved HTML: {out_html}")


# ============================================================
# One-shot wrapper (rate plots + bar charts)
# ============================================================
def draw_fixed_and_dynamic_quality_change_rate_suite(
    *,
    psnr: PsnrQualityTable,
    fixed: List[Experiment],
    dynamic: List[Experiment],
    out_dir: Path,
    fps: float = 24.0,
    mode: Literal["forward", "checkpoint", "deadline_base", "deadline_max_prefix"] = "checkpoint",
    change_mode: Literal["any", "upgrade", "downgrade"] = "upgrade",
    window_s: float = 2.0,
    initial_offset_s: float = 0.6,
    buffer_s: float = 1.5,
    max_layer: int = 2,
    logger=None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # rolling rate plots
    draw_fixed_plot_quality_change_rate(
        psnr=psnr, fixed=fixed,
        out_html=out_dir / f"qrate_fixed_1_2_3_{mode}_{change_mode}_w{window_s:g}_fps{int(fps)}.html",
        fps=fps, mode=mode, change_mode=change_mode, window_s=window_s,
        initial_offset_s=initial_offset_s, buffer_s=buffer_s, max_layer=max_layer,
        logger=logger,
    )
    draw_dynamic_plot_quality_change_rate(
        psnr=psnr, dynamic=dynamic,
        out_html=out_dir / f"qrate_dynamic_s1_s2_{mode}_{change_mode}_w{window_s:g}_fps{int(fps)}.html",
        fps=fps, mode=mode, change_mode=change_mode, window_s=window_s,
        initial_offset_s=initial_offset_s, buffer_s=buffer_s, max_layer=max_layer,
        logger=logger,
    )

    # overall slope bar charts
    draw_fixed_change_rate_barchart(
        psnr=psnr, fixed=fixed,
        out_html=out_dir / f"qrate_bar_fixed_{mode}_{change_mode}_fps{int(fps)}.html",
        fps=fps, mode=mode, change_mode=change_mode,
        initial_offset_s=initial_offset_s, buffer_s=buffer_s, max_layer=max_layer,
        logger=logger,
    )
    draw_dynamic_change_rate_barchart(
        psnr=psnr, dynamic=dynamic,
        out_html=out_dir / f"qrate_bar_dynamic_{mode}_{change_mode}_fps{int(fps)}.html",
        fps=fps, mode=mode, change_mode=change_mode,
        initial_offset_s=initial_offset_s, buffer_s=buffer_s, max_layer=max_layer,
        logger=logger,
    )