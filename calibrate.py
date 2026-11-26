"""
《部落冲突》跨平台视觉模板一键采集与实机校准命令行工具 (CLI Calibrator)
用法示例：
  python calibrate.py --init-templates
  python calibrate.py --check-live
  python calibrate.py --harvest village
  python calibrate.py --interactive bubble_gold
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path

from core.device import AdbDeviceController
from tools.calibrator import TemplateHarvester, LiveCalibrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("coc.calibrate_cli")


def main():
    parser = argparse.ArgumentParser(description="COC-AutoApp 视觉模板一键采集与真机/模拟器实机校准工具")
    parser.add_argument("--serial", type=str, default="auto", help="指定 ADB 设备序列号 (默认 auto 自动扫描)")
    parser.add_argument("--init-templates", action="store_true", help="初始化生成全套 12 张基础模板到 assets/templates/")
    parser.add_argument("--check-live", action="store_true", help="抓取当前在线模拟器/真机实时画面并生成实机校准诊断报告图")
    parser.add_argument(
        "--harvest",
        type=str,
        choices=["village", "battle", "chat", "builder", "army", "attack_menu", "all"],
        default=None,
        help="从当前实机画面一键按标准 ROI 切取指定场景的模板",
    )
    parser.add_argument("--interactive", type=str, default=None, help="弹出鼠标框选窗口，截取指定 key 的模板 (如 bubble_gold)")
    parser.add_argument(
        "--output",
        type=str,
        default="tests/calibration_live_report.png",
        help="实机校准诊断报告图输出路径",
    )
    args = parser.parse_args()

    harvester = TemplateHarvester()

    if args.init_templates:
        saved = harvester.generate_bootstrap_templates(overwrite=False)
        print(f"已初始化/确认 {len(saved)} 张核心视觉模板文件于: {harvester.assets_dir}")

    if args.harvest or args.interactive or args.check_live:
        device = AdbDeviceController(serial=args.serial)
        calibrator = LiveCalibrator(device=device)

        if args.harvest:
            if not device.connect():
                raise RuntimeError("未能连接在线模拟器或手机设备")
            screen = device.screencap()
            crops = harvester.harvest_from_screen_by_scene(screen, args.harvest)
            print(f"已从实机画面 [{args.harvest}] 场景一键切取 {len(crops)} 张模板:")
            for p in crops:
                print(f"  -> {p}")

        if args.interactive:
            if not device.connect():
                raise RuntimeError("未能连接在线模拟器或手机设备")
            screen = device.screencap()
            saved_path = harvester.interactive_crop(screen, args.interactive)
            if saved_path:
                print(f"已通过鼠标框选保存模板: {saved_path}")
            else:
                print("已取消鼠标框选。")

        if args.check_live:
            report = calibrator.run_live_calibration(output_report_path=args.output)
            print("================ 实机视觉校准诊断报告 ================")
            print(f"连接设备     : {report['device']}")
            print(f"OCR 识别引擎 : {report['ocr_backend']} (整帧耗时 {report['elapsed_ms']} ms)")
            print(f"敌方资源读数 : 金={report['enemy_loot'].gold:,} | 水={report['enemy_loot'].elixir:,} | 黑油={report['enemy_loot'].dark_elixir:,}")
            print(f"主村状态读数 : 空闲工人={report['village_status'].free_builders}/{report['village_status'].total_builders} | 金={report['village_status'].gold:,} | 水={report['village_status'].elixir:,}")
            print(f"死鱼采集器数 : {report['full_collectors']} 个满液位 (死鱼判定={report['is_dead_base']})")
            print(f"当前匹配模板 : {len(report['matched_templates'])} 项 -> {list(report['matched_templates'].keys())}")
            print(f"诊断大图保存 : {Path(report['report_path']).resolve()}")


if __name__ == "__main__":
    main()
