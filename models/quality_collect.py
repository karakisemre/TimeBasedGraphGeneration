from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import re

# Örnek satır:
# QUALITY: COLLECT VCLNo: 1 FrameNo: 0 Frame: I Temporal: T0  Qualitylayer: 0 Time: 17494882265172

_EVENT_RE = re.compile(r"^QUALITY:\s*(?P<event>\w+)\s*(?P<rest>.*)$", re.IGNORECASE)
_KV_RE = re.compile(r"(?P<k>[A-Za-z_]+):\s*(?P<v>[^\s]+)")

def _to_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None

def _to_temporal(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    v = v.strip()
    # T0, T1 ...
    if len(v) >= 2 and (v[0] in ("T", "t")):
        return _to_int(v[1:])
    return _to_int(v)

@dataclass(frozen=True, slots=True)
class CollectLineRow:
    """
    One event row parsed from CollectLines.txt

    event: COLLECT / LOST / DROP ... (ne gelirse)
    """
    event: str                  # "COLLECT" vb.
    vcl_no: Optional[int]
    frame_no: Optional[int]
    frame_type: Optional[str]   # I/P/B (yoksa None)
    temporal: Optional[int]     # T0 -> 0
    quality_layer: Optional[int]# 0/1/2
    time: Optional[int]         # trace time (int)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "vcl_no": self.vcl_no,
            "frame_no": self.frame_no,
            "frame_type": self.frame_type,
            "temporal": self.temporal,
            "quality_layer": self.quality_layer,
            "time": self.time,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CollectLineRow":
        return CollectLineRow(
            event=str(d.get("event", "COLLECT")).upper(),
            vcl_no=_to_int(d.get("vcl_no", d.get("VCLNo"))),
            frame_no=_to_int(d.get("frame_no", d.get("FrameNo"))),
            frame_type=(None if d.get("frame_type") is None else str(d.get("frame_type"))),
            temporal=_to_int(d.get("temporal")),
            quality_layer=_to_int(d.get("quality_layer", d.get("Qualitylayer"))),
            time=_to_int(d.get("time", d.get("Time"))),
        )

    @staticmethod
    def parse_line(line: str) -> "CollectLineRow":
        line = line.strip()
        if not line:
            raise ValueError("Empty line")

        m = _EVENT_RE.match(line)
        if not m:
            raise ValueError(f"Unrecognized CollectLines line: {line!r}")

        event = m.group("event").upper()
        rest = m.group("rest")

        kv = {mm.group("k").lower(): mm.group("v") for mm in _KV_RE.finditer(rest)}

        vcl_no = _to_int(kv.get("vclno"))
        frame_no = _to_int(kv.get("frameno"))
        frame_type = kv.get("frame")
        temporal = _to_temporal(kv.get("temporal"))
        ql = _to_int(kv.get("qualitylayer"))
        t = _to_int(kv.get("time"))

        # bazı loglarda "QualityLayer" gibi gelebilir; normalize:
        if ql is None:
            ql = _to_int(kv.get("qualitylayer".lower()))

        return CollectLineRow(
            event=event,
            vcl_no=vcl_no,
            frame_no=frame_no,
            frame_type=frame_type,
            temporal=temporal,
            quality_layer=ql,
            time=t,
        )

class CollectLinesTable:
    """
    - rows: tüm event satırları
    - collect_rows: event == COLLECT olanlar
    - index: (frame_no, quality_layer) -> CollectLineRow  (COLLECT için)
    """
    def __init__(self, rows: List[CollectLineRow], *, source_path: Optional[Path] = None):
        self.rows = rows
        self.source_path = source_path

        self.collect_rows = [r for r in rows if r.event == "COLLECT" and r.frame_no is not None and r.quality_layer is not None]

        self._by_frame_layer: Dict[Tuple[int, int], CollectLineRow] = {}
        for r in self.collect_rows:
            # aynı frame/layer birden çok kez gelirse son geleni yaz (istersen burada listeye de çevirebiliriz)
            self._by_frame_layer[(int(r.frame_no), int(r.quality_layer))] = r

        self.frames = sorted({int(r.frame_no) for r in self.collect_rows})
        self.layers = sorted({int(r.quality_layer) for r in self.collect_rows if r.quality_layer is not None})

    def __repr__(self) -> str:
        return f"CollectLinesTable(frames={len(self.frames)}, collect_rows={len(self.collect_rows)}, layers={self.layers})"

    # --------- load ---------
    @classmethod
    def from_txt(cls, path: str | Path) -> "CollectLinesTable":
        path = Path(path)
        rows: List[CollectLineRow] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for ln_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(CollectLineRow.parse_line(line))
                except Exception as e:
                    raise ValueError(f"{path}:{ln_no}: {e}") from e
        return cls(rows, source_path=path)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CollectLinesTable":
        path = Path(path)
        rows: List[CollectLineRow] = []
        with path.open("r", encoding="utf-8") as f:
            for ln_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(CollectLineRow.from_dict(json.loads(line)))
                except Exception as e:
                    raise ValueError(f"{path}:{ln_no}: JSONL parse error: {e}") from e
        return cls(rows, source_path=path)

    @classmethod
    def from_json_by_frame(cls, path: str | Path) -> "CollectLinesTable":
        """
        Reads: { "frame_no": { "layer": row_dict, ... }, ... }
        Note: Bu dosyada sadece COLLECT index'i var.
        """
        path = Path(path)
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows: List[CollectLineRow] = []
        for fn in sorted(obj.keys(), key=lambda x: int(x)):
            for ly in sorted(obj[fn].keys(), key=lambda x: int(x)):
                rows.append(CollectLineRow.from_dict(obj[fn][ly]))
        return cls(rows, source_path=path)

    # --------- query ---------
    def get_collect(self, frame_no: int, layer: int) -> Optional[CollectLineRow]:
        return self._by_frame_layer.get((frame_no, layer))

    def frame_collect_rows(self, frame_no: int) -> List[CollectLineRow]:
        out = []
        for l in self.layers:
            r = self.get_collect(frame_no, l)
            if r is not None:
                out.append(r)
        return out

    # --------- export ---------
    def to_jsonl(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
        return out_path

    def to_json_by_frame(self, out_path: str | Path) -> Path:
        """
        Writes { frame_no: { layer: row_dict, ... }, ... } only for COLLECT rows.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        obj: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for r in self.collect_rows:
            fn = str(int(r.frame_no))
            ly = str(int(r.quality_layer))
            obj.setdefault(fn, {})[ly] = r.as_dict()

        out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def meta(self) -> Dict[str, Any]:
        times = [r.time for r in self.collect_rows if r.time is not None]
        frames = [r.frame_no for r in self.collect_rows if r.frame_no is not None]
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "rows_total": len(self.rows),
            "rows_collect": len(self.collect_rows),
            "frames_min": min(frames) if frames else None,
            "frames_max": max(frames) if frames else None,
            "time_min": min(times) if times else None,
            "time_max": max(times) if times else None,
            "layers": self.layers,
        }

    def write_meta(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.meta(), ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path