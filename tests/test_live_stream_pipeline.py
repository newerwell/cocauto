"""
实时预览解耦架构、画布 33ms 刷新、图层/会话清理、无损冻结框选与自动回退验证套件
"""
from __future__ import annotations

import os
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "windows")

from PyQt5.QtWidgets import QApplication

from core.device import BaseDeviceController
from platforms.pc_desktop.interactive_canvas import InteractiveGameCanvas
from platforms.pc_desktop.main_window import CocDesktopMainWindow
from platforms.pc_desktop.worker_threads import LiveVisionWorker, VisionOverlayData


class FastCounterMockDevice(BaseDeviceController):
    """每调用一次 screencap 产出带递增帧标记的 1280x720 BGR 图，便于验证无损与实时更新"""

    def __init__(self):
        self.count = 0

    def connect(self) -> bool:
        return True

    def screencap(self) -> np.ndarray:
        self.count += 1
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        img[:, :] = (30, 90, 40)
        # 在左上角写入独一无二的颜色标记供颜色探针和无损裁切校验
        img[100:160, 200:260] = (12, 34, 210)
        cv2.putText(img, f"FRAME {self.count}", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        return img

    def tap(self, x: int, y: int, humanize: bool = True):
        return (x, y)

    def swipe(self, start, end, duration_ms: int = 350, use_bezier: bool = True):
        pass

    def pinch_zoom_out(self):
        pass


class TestLiveStreamPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _pump_qt(self, duration_sec: float):
        deadline = time.perf_counter() + duration_sec
        while time.perf_counter() < deadline:
            self.app.processEvents()
            time.sleep(0.01)

    def test_01_decoupled_slow_analysis_does_not_block_capture(self):
        """即使人为让识别耗时达到 2 秒，采集端仍持续产出新帧且后台无积压"""
        dev = FastCounterMockDevice()

        def slow_hook(_img):
            time.sleep(2.0)

        worker = LiveVisionWorker(
            serial="mock",
            continuous=True,
            interval_ms=25,
            mock_device=dev,
            session_id=1,
            prefer_h264=False,
            analysis_interval_sec=0.5,
            analysis_hook=slow_hook,
        )
        worker.start()
        time.sleep(2.3)
        worker.stop_stream()
        worker.wait(3000)

        # 2.3 秒内，25ms 间隔的采集端应稳定产出数十帧，而 2 秒识别最多跑完 1 轮且发生跳过
        self.assertGreaterEqual(worker.total_captured_frames, 30)
        self.assertLessEqual(worker.total_analysis_rounds, 2)
        latest = worker.get_latest_frame()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.seq, worker.total_captured_frames)

    def test_02_canvas_ignores_duplicate_frame_seq(self):
        """重复帧序号不应重复渲染，也不应虚抬 FPS"""
        canvas = InteractiveGameCanvas()
        img = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.assertTrue(canvas.set_video_frame(img, seq=10))
        self.assertFalse(canvas.set_video_frame(img, seq=10))
        self.assertTrue(canvas.set_video_frame(img, seq=11))

    def test_03_window_fallback_layer_clear_and_lossless_crop(self):
        """验证回退提示、图层即时清除、旧会话丢弃与无损冻结框选"""
        dev = FastCounterMockDevice()
        win = CocDesktopMainWindow(mock_device=dev)
        win.show()
        self._pump_qt(0.1)

        # 1. 开启连续流（触发模拟 H.264 失败回退模式），确认状态栏显示回退且正常出帧
        win._start_vision_worker(continuous=True, force_h264_fail=True)
        self._pump_qt(1.2)
        self.assertEqual(win._capture_mode_code, "png_fallback")
        self.assertIn("回退", win.lbl_status_badge.text())
        self.assertGreater(win._rendered_new_frame_count, 2)

        # 2. 注入一条旧会话的过期标注回调，确认会被彻底丢弃
        stale_overlay = VisionOverlayData(
            session_id=win._vision_session_id - 1,
            device="stale-dev",
            scene="旧场景",
            is_home_village=False,
            ocr_engine="stale",
            analysis_ms=999.0,
            frame_seq=1,
            analyzed_at=time.perf_counter(),
        )
        win._on_vision_analysis_ready(stale_overlay.session_id, stale_overlay)
        self.assertNotEqual(win._last_scene, "旧场景")

        # 3. 关闭图层开关，确认画布状态立即更新
        win.chk_show_ocr.setChecked(False)
        win.chk_show_tpl.setChecked(False)
        self.assertFalse(win.canvas._show_ocr)
        self.assertFalse(win.canvas._show_templates)

        # 4. 进入模板框选：暂停流并冻结抓取一张无损 PNG，验证框选态激活且像素精确一致
        win.trigger_canvas_crop_for_selected()
        self._pump_qt(1.5)
        self.assertIsNotNone(win.canvas._cropping_key)
        self.assertIsNotNone(win.canvas._raw_bgr)
        self.assertEqual(tuple(int(v) for v in win.canvas._raw_bgr[120, 220]), (12, 34, 210))

        # 5. 停止并重开预览，确认旧标注被清除
        win.canvas.cancel_interactive_crop()
        win._stop_vision_worker(clear_overlay=True)
        self.assertIsNone(win.canvas._overlay_data)

        win.close()


if __name__ == "__main__":
    unittest.main()
