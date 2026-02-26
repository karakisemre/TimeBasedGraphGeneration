from pathlib import Path
from drawing.ab.average_bitrate_time_drawing import SVCRefLevels
from drawing.goodput.old_goodput_time_drawing import discover_dynamic_experiments, discover_fixed_bw_experiments
from drawing.qc.quality_change_time_drawing import draw_fixed_and_dynamic_quality_changes, \
    draw_fixed_and_dynamic_quality_change_rate_suite
from logger.logger import Logger
from models import PsnrQualityTable


def main() -> None:
    logger = Logger()
    psnr = PsnrQualityTable.load_cached(Path("data"), stem="psnr_bbb_1000")
    fixed = discover_fixed_bw_experiments(Path("tests"))
    dynamic = discover_dynamic_experiments(Path("dynamictests"))
    outputDir = Path("overallresults")

    ref = SVCRefLevels(
        l0_kbps=897.7,
        l1_kbps=1927.3,
        l2_kbps=4384.2,
        y0=35.44,
        y1=35.64,
        y2=37.55,
        missing_y=2.0,
    )
    ##################################### GOODPUT ############################################
    mode = "checkpoint"  # "checkpoint" | "forward" "
    y_unit = "Kbps"  # "Kbps" | "Mbps"

    out = outputDir / f"goodput_time_with_forward_checkpoint_{mode.lower()}_{y_unit.lower()}.html"
    # draw_fixed_and_dynamic(
    #     psnr=psnr,
    #     fixed=fixed,
    #     dynamic=dynamic,
    #     out_dir=out,
    #     mode=mode,
    #     y_unit=y_unit,
    #     min_elapsed_s=1.0,
    # )
    #
    # out = outputDir / f"goodput_time_actual_{mode.lower()}_{y_unit.lower()}.html"
    # draw_fixed_and_dynamic_goodput(
    #     psnr=psnr,
    #     fixed=fixed,
    #     dynamic=dynamic,
    #     out_dir=out,
    #     mode="raw",  # GOODPUT
    #     y_unit="Kbps",
    #     min_elapsed_s=1.0,
    # )
    ##################################### AVERAGE BITRATE ############################################
    # out = outputDir / f"average_bitrate_time{mode.lower()}_{y_unit.lower()}.html"
    # draw_fixed_and_dynamic_qoe(
    #     psnr=psnr,
    #     fixed=fixed,
    #     dynamic=dynamic,
    #     ref=ref,
    #     out_dir=out,
    #     fps=24.0,        # <- kendi frame rate'in neyse burada ver
    #     mode="checkpoint",  # veya "checkpoint"
    #     max_layer=2,
    #     logger=logger
    # )

    # 1) Deadline-aware QoE (frame-time)
    # out = outputDir / f"deadline_aware_average_bitrate_time{mode.lower()}_{y_unit.lower()}.html"
    # draw_fixed_and_dynamic_qoe_deadline(
    #     psnr=psnr,
    #     fixed=fixed,
    #     dynamic=dynamic,
    #     ref=ref,
    #     out_dir=out,
    #     fps=24.0,
    #     initial_offset_s=0.6,
    #     buffer_s=1.5,
    #     max_layer=2,
    #     logger=logger,
    # )

    # # gerçekçi (enhancement late => stall yok)
    # draw_fixed_and_dynamic_qoe_deadline(
    #     psnr=psnr, fixed=fixed, dynamic=dynamic, ref=ref,
    #     out_dir=out,
    #     fps=24.0, initial_offset_s=0.6, buffer_s=1.5,
    #     stall_policy="base",
    #     logger=logger,
    # )

    # # agresif kalite (enhancement için de bekler) -> BPP avantajı daha görünür olabilir
    # draw_fixed_and_dynamic_qoe_deadline(
    #     psnr=psnr, fixed=fixed, dynamic=dynamic, ref=ref,
    #     out_dir=out,
    #     fps=24.0, initial_offset_s=0.6, buffer_s=1.5,
    #     stall_policy="max_prefix",
    #     logger=logger,
    # )

    # # # 2) Arrival-upgrade (event-time)
    # out = outputDir / f"arrival_upgrade_average_bitrate_time{mode.lower()}_{y_unit.lower()}.html"
    # draw_fixed_and_dynamic_arrival_upgrade(
    #     psnr=psnr,
    #     fixed=fixed,
    #     dynamic=dynamic,
    #     ref=ref,
    #     out_dir=out,
    #     max_layer=2,
    #     logger=logger,
    # )
    out = outputDir / "draw_fixed_and_dynamic_quality_changes"
    draw_fixed_and_dynamic_quality_changes(
        psnr=psnr,
        fixed=fixed,
        dynamic=dynamic,
        out_dir=out,
        fps=24.0,
        mode="checkpoint",
        change_mode="any",
        logger=logger,
    )
    out = outputDir / "draw_fixed_and_dynamic_quality_changes"
    draw_fixed_and_dynamic_quality_changes(
        psnr=psnr,
        fixed=fixed,
        dynamic=dynamic,
        out_dir=out,
        fps=24.0,
        mode="deadline_base",
        change_mode="any",
        initial_offset_s=0.6,
        buffer_s=1.5,
        logger=logger,
    )

    draw_fixed_and_dynamic_quality_change_rate_suite(
        psnr=psnr,
        fixed=fixed,
        dynamic=dynamic,
        out_dir=Path("overallresults/qrate_suite"),
        fps=24.0,
        mode="deadline_base",  # veya "checkpoint"
        change_mode="upgrade",  # önerdiğim
        window_s=2.0,  # önerdiğim
        initial_offset_s=0.6,
        buffer_s=1.5,
        logger=logger,
    )

    print("Saved: in out directory")

if __name__ == "__main__":
    main()