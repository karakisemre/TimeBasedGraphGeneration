from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
from models import CollectLinesTable

def iter_collectlines_files(roots: List[Path], filename: str = "CollectLines.txt") -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        yield from root.rglob(filename)

def convert_one(path_txt: Path, overwrite: bool = True) -> None:
    out_jsonl = path_txt.with_suffix(".jsonl")  # CollectLines.jsonl
    out_byf = path_txt.with_name("CollectLines_by_frame.json")
    out_meta = path_txt.with_name("CollectLines_meta.json")

    if (not overwrite) and (out_jsonl.exists() or out_byf.exists()):
        return

    table = CollectLinesTable.from_txt(path_txt)
    table.to_jsonl(out_jsonl)
    table.to_json_by_frame(out_byf)
    table.write_meta(out_meta)

    print(f"[OK] {path_txt} -> {out_jsonl.name}, {out_byf.name}, {out_meta.name} | {table}")

def main() -> None:
    roots = [Path("tests"), Path("dynamictests")]
    files = sorted(iter_collectlines_files(roots), key=lambda p: str(p))

    if not files:
        print("No CollectLines.txt found under tests/ or dynamictests/")
        return

    for p in files:
        convert_one(p, overwrite=True)

if __name__ == "__main__":
    main()