"""
全模块自动化单元测试与合成视觉闭环验证套件 (Unit & Integration Test Suite)
无需挂载真实手机或安卓模拟器，通过合成 1280x720 游戏画面（含敌方红线、圣水满采集器紫红液柱、英雄受损血条、资源气泡）
以及 Mock 设备控制器，对所有 6 大核心模块进行端到端验证。
"""
from __future__ import annotations
import unittest
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np

from config import load_config, AppConfig
from core.humanizer import HumanJitter, SessionScheduler
from core.device import BaseDeviceController, MultiTouchProtocol
from core.vision import TemplateMatcher, OcrScanner, RedZoneDetector, DeadBaseChecker, EnemyLoot, VillageStatus
from core.tasks import LootSearcherTask, AutoUpgraderTask
from tools.calibrator import TemplateHarvester, LiveCalibrator
from main import CocAutoBotOrchestrator


class MockAndroidDevice(BaseDeviceController):
    """用于无头测试的虚拟 Android 设备，自动生成包含红线、满采集器和英雄血条的 1280x720 画面"""

    def __init__(self):
        self.taps: List[Tuple[int, int]] = []
        self.swipes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.zoom_outs: int = 0
        self.current_screen: np.ndarray = self._create_synthetic_battle_screen()

    @staticmethod
    def _create_synthetic_battle_screen() -> np.ndarray:
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        # 填充草地底色
        img[:, :] = (45, 115, 55)
        # 绘制敌方阵型中心红色部署禁区框 (BGR 亮红: 20, 20, 235)
        cv2.rectangle(img, (440, 190), (840, 440), (20, 20, 235), thickness=6)
        # 绘制 4 个满液位圣水收集器紫品红液柱斑块 (HSV H≈152 -> BGR: 210, 30, 230)
        for cx, cy in [(350, 250), (920, 250), (350, 460), (920, 460)]:
            cv2.circle(img, (cx, cy), 10, (210, 30, 230), -1)
        # 在底部英雄 #1 槽位 (680, 595, 760, 695) 顶部绘制仅剩 30% 的绿色生命条 (触发自动开大)
        cv2.rectangle(img, (684, 596), (704, 604), (40, 235, 40), -1)
        return img

    def connect(self) -> bool:
        return True

    def screencap(self) -> np.ndarray:
        return self.current_screen.copy()

    def tap(self, x: int, y: int, humanize: bool = True) -> Tuple[int, int]:
        self.taps.append((x, y))
        return (x, y)

    def swipe(self, start: Tuple[int, int], end: Tuple[int, int], duration_ms: int = 350, use_bezier: bool = True) -> None:
        self.swipes.append((start, end))

    def pinch_zoom_out(self) -> None:
        self.zoom_outs += 1


class TestAllModules(unittest.TestCase):

    def test_01_config_loader(self):
        cfg = load_config()
        self.assertIsInstance(cfg, AppConfig)
        self.assertEqual(cfg.device.standard_width, 1280)
        self.assertEqual(cfg.device.standard_height, 720)
        self.assertGreaterEqual(cfg.upgrade.reserve_builders_for_wall, 1)

    def test_02_humanizer_jitter_and_bezier(self):
        jitter = HumanJitter(sigma_px=6.0, max_offset_px=15)
        for _ in range(30):
            jx, jy = jitter.jitter_point(640, 360)
            self.assertTrue(625 <= jx <= 655)
            self.assertTrue(345 <= jy <= 375)

        curve = jitter.generate_bezier_curve((200, 200), (900, 500), num_steps=12)
        self.assertEqual(len(curve), 12)
        self.assertEqual(curve[0], (200, 200))
        self.assertEqual(curve[-1], (900, 500))

    def test_03_multitouch_protocol(self):
        pinch_cmds = MultiTouchProtocol.build_pinch_zoom_out_commands()
        self.assertTrue(any(cmd.startswith("d 0") for cmd in pinch_cmds))
        self.assertTrue(any(cmd.startswith("d 1") for cmd in pinch_cmds))
        self.assertTrue(any(cmd.startswith("u 0") for cmd in pinch_cmds))

    def test_04_vision_red_zone_and_dead_base(self):
        mock_dev = MockAndroidDevice()
        screen = mock_dev.screencap()

        # 1. 验证动态红线四边检测
        detector = RedZoneDetector()
        sides = detector.detect_deployable_sides(screen, points_per_side=8)
        for key in ("NW", "NE", "SW", "SE"):
            self.assertIn(key, sides)
            self.assertGreaterEqual(len(sides[key]), 3)

        # 2. 验证死鱼满采集器紫红斑块检测 & 英雄血条占比检测
        checker = DeadBaseChecker()
        full_count = checker.count_full_elixir_collectors(screen)
        self.assertGreaterEqual(full_count, 4)
        self.assertTrue(checker.is_dead_base(screen, min_full_collectors=3))

        hp_ratio = checker.check_hero_hp_ratio(screen, (680, 595, 760, 695))
        self.assertLess(hp_ratio, 0.45)

    def test_05_tasks_loot_and_upgrade_logic(self):
        cfg = load_config()
        mock_dev = MockAndroidDevice()
        matcher = TemplateMatcher()
        ocr = OcrScanner(custom_ocr_fn=lambda _: "650000")
        checker = DeadBaseChecker(matcher)

        # 验证搜鱼过滤
        searcher = LootSearcherTask(mock_dev, matcher, ocr, checker, cfg.loot_filter)
        matched, reason = searcher.evaluate_base(EnemyLoot(gold=600_000, elixir=600_000, dark_elixir=5000), is_dead=True)
        self.assertTrue(matched)

        # 验证“留 1 工专门刷墙”策略：当只有 1 名空闲工人且金币达标时，只刷墙不升普通建筑
        upgrader = AutoUpgraderTask(mock_dev, matcher, ocr, cfg.upgrade)
        can_wall, can_bldg, _ = upgrader.plan_upgrade_actions(
            VillageStatus(free_builders=1, total_builders=5, gold=5_000_000, elixir=2_000_000)
        )
        self.assertTrue(can_wall)
        self.assertFalse(can_bldg)

        # 当有 2 名空闲工人时，既可刷墙也可升级推荐建筑
        can_wall2, can_bldg2, _ = upgrader.plan_upgrade_actions(
            VillageStatus(free_builders=2, total_builders=5, gold=5_000_000, elixir=2_000_000)
        )
        self.assertTrue(can_wall2)
        self.assertTrue(can_bldg2)

    def test_06_full_orchestrator_single_cycle(self):
        cfg = load_config()
        cfg.combat.early_exit_no_loot_sec = 0  # 测试中立即触发退场
        cfg.humanizer.night_sleep_window = "03:00-03:01"
        mock_dev = MockAndroidDevice()

        bot = CocAutoBotOrchestrator(cfg, device=mock_dev)
        bot.village_task.travel.detector.scene = lambda _: "home"
        bot.ocr = OcrScanner(custom_ocr_fn=lambda _: "600000")
        bot.search_task.ocr = bot.ocr
        bot.combat_task.ocr = bot.ocr
        bot.upgrade_task.ocr = OcrScanner(custom_ocr_fn=lambda _: "1/5")

        res = bot.run_single_cycle()
        self.assertEqual(res["status"], "completed")
        self.assertGreater(len(mock_dev.taps), 0)
        self.assertEqual(res["combat"]["deployment"][0]["status"], "unconfirmed")
        self.assertGreaterEqual(mock_dev.zoom_outs, 1)

    def test_07_calibrator_and_builtin_ocr(self):
        # 1. 验证基础模板初始化与场景切图 (包含渡船模板共 14 张)
        harvester = TemplateHarvester()
        saved = harvester.generate_bootstrap_templates(overwrite=False)
        self.assertGreaterEqual(len(saved), 12)
        for p in saved:
            self.assertTrue(p.exists())

        # 2. 验证内置 OpenCV 数字识别器 (即使不依赖外部 OCR 库也能识别 Supercell 高亮字体)
        mock_dev = MockAndroidDevice()
        screen = mock_dev.screencap()
        # 在 EnemyGold ROI (66, 98, 75, 220) 绘制高亮白字 "580000"
        cv2.putText(screen, "580000", (80, 92), cv2.FONT_HERSHEY_DUPLEX, 0.75, (250, 250, 250), 2)
        ocr = OcrScanner()
        loot = ocr.scan_enemy_loot(screen)
        self.assertGreater(loot.gold, 0)

        # 3. 验证实机校准报告生成器 (1700x720 分屏仪表盘输出)
        calibrator = LiveCalibrator(device=mock_dev)
        out_report = harvester.assets_dir.parent.parent / "tests" / "test_calib_out.png"
        report = calibrator.run_live_calibration(out_report, screen_override=screen)
        self.assertTrue(out_report.exists())
        img_out = cv2.imread(str(out_report))
        self.assertEqual(img_out.shape, (720, 1700, 3))

    def test_08_pyqt5_gui_main_window_and_snapshot(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "windows")
        from PyQt5.QtWidgets import QApplication
        from platforms.pc_desktop.main_window import CocDesktopMainWindow

        app = QApplication.instance() or QApplication([])
        mock_dev = MockAndroidDevice()
        win = CocDesktopMainWindow(mock_device=mock_dev)
        win.show()
        app.processEvents()

        # 触发一次视觉画面刷新并同步渲染到左侧画布
        win.refresh_live_frame()
        if win._vision_worker:
            win._vision_worker.wait(4000)
        app.processEvents()

        # 验证配置保存与画布图像加载
        win.spin_min_gold.setValue(520000)
        win.save_gui_config()
        self.assertEqual(win.cfg.loot_filter.min_gold, 520000)
        self.assertIsNotNone(win.canvas._raw_bgr)

        # 截取整个 PyQt5 主窗口界面保存为预览图
        pixmap = win.grab()
        gui_snap_path = Path(__file__).resolve().parent / "gui_window_snapshot.png"
        pixmap.save(str(gui_snap_path), "PNG")
        self.assertTrue(gui_snap_path.exists())
        win.close()

    def test_09_deployment_geometry_and_single_instance_lock(self):
        import gui_app

        # 1. 验证主世界移镜后四侧双层外圈安全点 100% 位于 PLAYABLE_SAFE_POLY 安全区内
        detector = RedZoneDetector()
        mock_dev = MockAndroidDevice()
        screen = mock_dev.screencap()
        playable_mask = np.zeros((720, 1280), dtype=np.uint8)
        cv2.fillPoly(playable_mask, [detector.PLAYABLE_SAFE_POLY], 255)

        for side in ("NW", "NE", "SW", "SE"):
            main_pts, guard_pts = detector.detect_shifted_side_points(screen, side, points_per_side=12)
            self.assertEqual(len(main_pts), 12)
            self.assertEqual(len(guard_pts), 12)
            for px, py in main_pts + guard_pts:
                self.assertGreater(
                    playable_mask[py, px],
                    0,
                    f"Side {side} point ({px}, {py}) must be strictly inside PLAYABLE_SAFE_POLY",
                )

        # 验证 EXE / GUI 面板全局单实例互斥锁 (同一互斥名只能获取一次)
        import os
        test_mutex = "Local\\COC_AutoApp_UnitTest_SingleInstance_Mutex"
        gui_app.release_single_instance_lock()
        try:
            first_ok = gui_app.acquire_single_instance_lock(test_mutex)
            self.assertTrue(first_ok)
            if os.name == "nt":
                import ctypes
                h2 = ctypes.windll.kernel32.CreateMutexW(None, False, test_mutex)
                err2 = ctypes.windll.kernel32.GetLastError()
                if h2:
                    ctypes.windll.kernel32.CloseHandle(h2)
                self.assertEqual(err2, 183)  # ERROR_ALREADY_EXISTS
        finally:
            gui_app.release_single_instance_lock()


if __name__ == "__main__":
    unittest.main()
