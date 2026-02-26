from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import csv
import json
import re

# Matches lines like:
# 0 I T0 L0 Q0 QP 30 Y 61.2376 U 66.4653 V 99.9900 4936 bit
_PSNr_RE = re.compile(
    r"^(?P<frame_no>\d+)\s+"
    r"(?P<frame_type>[IPB])\s+"
    r"(?P<temporal>T\d+)\s+"
    r"(?P<layer>L\d+)\s+"
    r"(?P<qid>Q\d+)\s+"
    r"QP\s+(?P<qp>\d+)\s+"
    r"Y\s+(?P<y>\d+\.\d+)\s+"
    r"U\s+(?P<u>\d+\.\d+)\s+"
    r"V\s+(?P<v>\d+\.\d+)\s+"
    r"(?P<bits>\d+)\s+bit\s*$"
)


@dataclass(frozen=True, slots=True)
class PsnrQualityRow:
    """One row (one layer of one frame)."""
    frame_no: int
    frame_type: str            # 'I' or 'P' (or 'B' if present)
    temporal: int              # parsed from 'T0' -> 0
    layer: int                 # parsed from 'L0' -> 0  (SNR/quality layer in your trace)
    qid: int                   # parsed from 'Q0' -> 0 (kept for future-proofing)
    qp: int
    y_psnr: float
    u_psnr: float
    v_psnr: float
    bits: int                  # bits for this layer for this frame

    @property
    def bitrate_bits(self) -> int:
        return self.bits

    def as_dict(self) -> Dict[str, object]:
        return {
            "frame_no": self.frame_no,
            "frame_type": self.frame_type,
            "temporal": self.temporal,
            "layer": self.layer,
            "qid": self.qid,
            "qp": self.qp,
            "y_psnr": self.y_psnr,
            "u_psnr": self.u_psnr,
            "v_psnr": self.v_psnr,
            "bits": self.bits,
        }


    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PsnrQualityRow":
        frame_no = int(d.get("frame_no", d.get("frameNo")))
        frame_type = str(d.get("frame_type", d.get("frame", "P")))
        temporal = int(d.get("temporal", d.get("temporalId", 0)))
        layer = int(d.get("layer", d.get("qualityLayer", 0)))
        qid = int(d.get("qid", d.get("qId", 0)))
        qp = int(d.get("qp", d.get("QP", 0)))

        y = float(d.get("y_psnr", d.get("ypsnr", d.get("Y", 0.0))))
        u = float(d.get("u_psnr", d.get("upsnr", d.get("U", 0.0))))
        v = float(d.get("v_psnr", d.get("vpsnr", d.get("V", 0.0))))

        bits = int(d.get("bits", d.get("bitrate", 0)))

        return PsnrQualityRow(
            frame_no=frame_no,
            frame_type=frame_type,
            temporal=temporal,
            layer=layer,
            qid=qid,
            qp=qp,
            y_psnr=y,
            u_psnr=u,
            v_psnr=v,
            bits=bits,
        )

    @staticmethod
    def parse_line(line: str) -> "PsnrQualityRow":
        m = _PSNr_RE.match(line.strip())
        if not m:
            raise ValueError(f"Unrecognized PSNR line: {line!r}")

        return PsnrQualityRow(
            frame_no=int(m.group("frame_no")),
            frame_type=m.group("frame_type"),
            temporal=int(m.group("temporal")[1:]),
            layer=int(m.group("layer")[1:]),
            qid=int(m.group("qid")[1:]),
            qp=int(m.group("qp")),
            y_psnr=float(m.group("y")),
            u_psnr=float(m.group("u")),
            v_psnr=float(m.group("v")),
            bits=int(m.group("bits")),
        )


class PsnrQualityTable:
    """
    Fast lookup table:
      - get(frame_no, layer) -> PsnrQualityRow
      - bitrate_sum(frame_no, max_layer=2) -> int bits
    """

    def __init__(self, rows: List[PsnrQualityRow]):
        self.rows: List[PsnrQualityRow] = rows

        # (frame_no, layer) -> row
        self._by_frame_layer: Dict[Tuple[int, int], PsnrQualityRow] = {
            (r.frame_no, r.layer): r for r in rows
        }

        self.frames = sorted({r.frame_no for r in rows})
        self.layers = sorted({r.layer for r in rows})
        self.temporals = sorted({r.temporal for r in rows})

    def __repr__(self) -> str:
        return (
            f"PsnrQualityTable(frames={len(self.frames)}, "
            f"rows={len(self.rows)}, layers={self.layers}, temporals={self.temporals})"
        )

    @classmethod
    def from_txt(cls, path: str | Path) -> "PsnrQualityTable":
        path = Path(path)
        rows: List[PsnrQualityRow] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for ln_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(PsnrQualityRow.parse_line(line))
                except Exception as e:
                    raise ValueError(f"{path}:{ln_no}: {e}") from e
        return cls(rows)

    # ------------------------
    # Query helpers
    # ------------------------
    def get(self, frame_no: int, layer: int) -> Optional[PsnrQualityRow]:
        return self._by_frame_layer.get((frame_no, layer))

    def bitrate_sum(self, frame_no: int, max_layer: int = 2) -> int:
        """Sum bits for layers 0..max_layer inclusive."""
        total = 0
        for l in range(0, max_layer + 1):
            r = self.get(frame_no, l)
            if r is not None:
                total += r.bits
        return total

    def frame_rows(self, frame_no: int) -> List[PsnrQualityRow]:
        """All rows for a given frame, sorted by layer."""
        out = []
        for l in self.layers:
            r = self.get(frame_no, l)
            if r is not None:
                out.append(r)
        return out

    def head(self, n: int = 5) -> List[Dict[str, object]]:
        return [r.as_dict() for r in self.rows[:n]]

    # ------------------------
    # Export helpers
    # ------------------------
    def to_jsonl(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in self.rows:
                f.write(json.dumps(r.as_dict(), ensure_ascii=False) + "\n")
        return out_path

    def to_csv(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "frame_no", "frame_type", "temporal", "layer", "qid",
            "qp", "y_psnr", "u_psnr", "v_psnr", "bits"
        ]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in self.rows:
                w.writerow(r.as_dict())
        return out_path

    def to_json(self, out_path: str | Path) -> str:
        """Writes a single JSON array file."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump([r.as_dict() for r in self.rows], f, ensure_ascii=False, indent=2)
        return str(out_path)

    def to_json_by_frame(self, out_path: str | Path) -> str:
        """
        Writes { frame_no: { layer: row_dict, ... }, ... } for super-fast lookup after load.
        Keys are strings in JSON.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        obj: dict[str, dict[str, dict]] = {}
        for r in self.rows:
            fn = str(r.frame_no)
            ly = str(r.layer)
            obj.setdefault(fn, {})[ly] = r.as_dict()
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return str(out_path)

    # ------------------------
    # LOADERS (cache'den okuma)
    # ------------------------
    @classmethod
    def from_json_array(cls, path: str | Path) -> "PsnrQualityTable":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            arr = json.load(f)
        rows = [PsnrQualityRow.from_dict(x) for x in arr]
        return cls(rows)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PsnrQualityTable":
        path = Path(path)
        rows: List[PsnrQualityRow] = []
        with path.open("r", encoding="utf-8") as f:
            for ln_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rows.append(PsnrQualityRow.from_dict(d))
                except Exception as e:
                    raise ValueError(f"{path}:{ln_no}: JSONL parse error: {e}") from e
        return cls(rows)

    @classmethod
    def from_csv(cls, path: str | Path) -> "PsnrQualityTable":
        path = Path(path)
        rows: List[PsnrQualityRow] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for ln_no, d in enumerate(r, start=2):  # header=1
                try:
                    rows.append(
                        PsnrQualityRow.from_dict(d))  # DictReader zaten string döndürür; from_dict cast eder
                except Exception as e:
                    raise ValueError(f"{path}:{ln_no}: CSV parse error: {e}") from e
        return cls(rows)

    @classmethod
    def from_json_by_frame(cls, path: str | Path) -> "PsnrQualityTable":
        """
        Reads { "frame_no": { "layer": row_dict, ... }, ... }
        and rebuilds rows.
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        rows: List[PsnrQualityRow] = []
        # Stabil sıralama: frame_no asc, layer asc
        for fn_str in sorted(obj.keys(), key=lambda x: int(x)):
            layers_obj = obj[fn_str]
            for ly_str in sorted(layers_obj.keys(), key=lambda x: int(x)):
                rows.append(PsnrQualityRow.from_dict(layers_obj[ly_str]))
        return cls(rows)

    @classmethod
    def load_cached(cls, data_dir: str | Path, stem: str = "psnr_bbb_1000") -> "PsnrQualityTable":
        """
        Prefer: by_frame.json -> .json -> .jsonl -> .csv -> .txt
        Accepts both *_by_frame.json and *_byframe.json names.
        """
        d = Path(data_dir)

        candidates = [
            d / f"{stem}_by_frame.json",
            d / f"{stem}_byframe.json",
            d / f"{stem}.json",
            d / f"{stem}.jsonl",
            d / f"{stem}.csv",
            d / f"{stem}.txt",
        ]

        for p in candidates:
            if p.exists():
                if p.name.endswith(("_by_frame.json", "_byframe.json")):
                    return cls.from_json_by_frame(p)
                if p.suffix == ".json":
                    return cls.from_json_array(p)
                if p.suffix == ".jsonl":
                    return cls.from_jsonl(p)
                if p.suffix == ".csv":
                    return cls.from_csv(p)
                if p.suffix == ".txt":
                    return cls.from_txt(p)

        raise FileNotFoundError(
            "No PSNR cache found. Expected one of: "
            + ", ".join(str(x) for x in candidates)
        )