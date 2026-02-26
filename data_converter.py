from pathlib import Path
from models import PsnrQualityTable

def main() -> None:
    psnr_txt = Path("data/psnr_bbb_1000.txt")
    table = PsnrQualityTable.from_txt(psnr_txt)

    # "Single-line listing": both the summary and the dictionary representation of the first 5 lines.
    print(table, "| head5 =", table.head(5))

    # Example: Bit sum by layers for frame 0
    f = 0
    print("frame", f, "L0 bits =", table.get(f, 0).bits, "L0+L1 bits =", table.bitrate_sum(f, 1), "L0+L1+L2 bits =", table.bitrate_sum(f, 2))

    # cache export (you can remove it from the comments if you want)
    table.to_jsonl("data/cache/psnr_bbb_1000.jsonl")
    table.to_json("data/cache/psnr_bbb_1000.json")
    table.to_json_by_frame("data/cache/psnr_bbb_1000_by_frame.json")
    table.to_csv("data/cache/psnr_bbb_1000.csv")

if __name__ == "__main__":
    main()