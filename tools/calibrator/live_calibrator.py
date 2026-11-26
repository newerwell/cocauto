"""
真机/模拟器实机全链路视觉校准与透视诊断引擎 (Live Emulator & Device Calibrator)
负责：
1. 连接当前在线的安卓模拟器 (如 127.0.0.1:16384 MuMu12 / 雷电9) 或手机无线 ADB
2. 抓取实时 1280x720 画面，执行 OCR 资源/工人读取、动态红线四边检测、死鱼满采集器检测与全部模板匹配检验
3. 将原图、所有 ROI 检测框、二值化预处理缩略图、OCR 识别数值以及红线采样点合成为一张高分辨率可视化诊断报告图
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np

from core.device import BaseDeviceController, AdbDeviceController
from core.vision import TemplateMatcher, OcrScanner, RedZoneDetector, DeadBaseChecker
from .template_harvester import TemplateHarvester

logger = logging.getLogger("coc.calibrator")


class LiveCalibrator:
    """实机/模拟器画面全链路视觉校准与可视化透视报告生成器"""

    def __init__(
        self,
        device: Optional[BaseDeviceController] = None,
        assets_dir: Optional[str | Path] = None,
    ):
        self.device = device or AdbDeviceController(serial="auto")
        self.matcher = TemplateMatcher(assets_dir=assets_dir)
        self.ocr = OcrScanner()
        self.red_detector = RedZoneDetector()
        self.dead_checker = DeadBaseChecker(self.matcher)
        self.harvester = TemplateHarvester(assets_dir=assets_dir)

    def run_live_calibration(
        self,
        output_report_path: str | Path,
        screen_override: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        抓取设备实时画面（或使用传入的画面），执行全套视觉校准并输出可视化透视大图。
        """
        t0 = time.perf_counter()
        if screen_override is not None:
            screen = screen_override.copy()
            device_info = "Mock/Provided Frame (1280x720)"
        else:
            if not self.device.connect():
                raise RuntimeError("未检测到在线的模拟器或手机 ADB 设备")
            screen = self.device.screencap()
            device_info = getattr(self.device, "_resolved_serial", "Connected Android Device")

        # 1. 执行双场景 OCR 读取 (同时测试敌方资源 ROI 与主村庄工人/资源 ROI)
        enemy_loot = self.ocr.scan_enemy_loot(screen)
        village_status = self.ocr.scan_village_status(screen)

        # 2. 执行动态红线四边检测与死鱼满采集器检测
        sides = self.red_detector.detect_deployable_sides(screen, points_per_side=10)
        full_collectors = self.dead_checker.count_full_elixir_collectors(screen)
        is_dead = self.dead_checker.is_dead_base(screen, min_full_collectors=3)

        # 3. 扫描当前画面上能匹配到的所有模板
        matched_templates: Dict[str, Tuple[int, int, float]] = {}
        for key, spec in self.harvester.TEMPLATE_SPECS.items():
            res = self.matcher.find_one(screen, spec.rel_path, threshold=0.72)
            if res is not None:
                cx, cy = res.center
                matched_templates[key] = (cx, cy, round(res.score, 3))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 4. 绘制左侧 1280x720 实机透视标注图 + 右侧 420x720 诊断仪表盘面板
        report_img = self._compose_diagnostic_dashboard(
            screen=screen,
            device_info=str(device_info),
            enemy_loot=enemy_loot,
            village_status=village_status,
            sides=sides,
            full_collectors=full_collectors,
            is_dead=is_dead,
            matched_templates=matched_templates,
            elapsed_ms=elapsed_ms,
        )

        out_path = Path(output_report_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), report_img)

        return {
            "device": device_info,
            "ocr_backend": self.ocr.backend_name,
            "enemy_loot": enemy_loot,
            "village_status": village_status,
            "full_collectors": full_collectors,
            "is_dead_base": is_dead,
            "matched_templates": matched_templates,
            "elapsed_ms": round(elapsed_ms, 2),
            "report_path": str(out_path),
        }

    def _compose_diagnostic_dashboard(
        self,
        screen: np.ndarray,
        device_info: str,
        enemy_loot,
        village_status,
        sides: dict,
        full_collectors: int,
        is_dead: bool,
        matched_templates: Dict[str, Tuple[int, int, float]],
        elapsed_ms: float,
    ) -> np.ndarray:
        """拼接 1700x720 实机校准诊断大图（左：1280x720 标注画面，右：420x720 ROI 二值化预览与数据面板）"""
        vis = screen.copy()

        # A. 绘制 OCR ROI 框
        rois_to_draw = [
            ("EnemyGold", self.ocr.ENEMY_GOLD_ROI, (0, 215, 255)),
            ("EnemyElixir", self.ocr.ENEMY_ELIXIR_ROI, (255, 100, 255)),
            ("EnemyDark", self.ocr.ENEMY_DARK_ROI, (200, 200, 200)),
            ("Builders", self.ocr.BUILDER_TEXT_ROI, (0, 255, 120)),
            ("OwnGold", self.ocr.OWN_GOLD_ROI, (0, 215, 255)),
            ("OwnElixir", self.ocr.OWN_ELIXIR_ROI, (255, 100, 255)),
        ]
        for label, (y1, y2, x1, x2), color in rois_to_draw:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # B. 绘制红线四边下兵采样点
        side_colors = {
            "NW": (255, 220, 0),
            "NE": (0, 255, 255),
            "SW": (255, 100, 255),
            "SE": (0, 255, 120),
        }
        for s_name, pts in sides.items():
            c = side_colors.get(s_name, (0, 255, 0))
            for px, py in pts:
                cv2.circle(vis, (px, py), 5, c, -1)
                cv2.circle(vis, (px, py), 7, (0, 0, 0), 1)

        # C. 绘制匹配成功的模板位置
        for key, (cx, cy, sc) in matched_templates.items():
            cv2.circle(vis, (cx, cy), 14, (0, 255, 0), 2)
            cv2.putText(vis, f"{key}({sc:.2f})", (cx - 30, cy - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # D. 构建右侧 420x720 诊断侧边栏
        panel = np.full((720, 420, 3), 24, dtype=np.uint8)
        cv2.line(panel, (0, 0), (0, 720), (80, 200, 120), 2)

        def _put(text: str, y: int, color=(230, 230, 230), scale=0.48, thick=1):
            cv2.putText(panel, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

        _put("COC-AutoApp Live Calibrator", 32, (0, 255, 180), 0.62, 2)
        _put(f"Device : {device_info}", 60, (180, 220, 255))
        _put(f"OCR Eng: {self.ocr.backend_name} ({elapsed_ms:.1f}ms)", 84, (180, 220, 255))

        _put("--- 1. Battle Loot OCR ---", 120, (0, 215, 255), 0.52, 2)
        _put(f"Enemy Gold  : {enemy_loot.gold:,}", 145)
        _put(f"Enemy Elixir: {enemy_loot.elixir:,}", 168)
        _put(f"Enemy Dark  : {enemy_loot.dark_elixir:,}", 191)

        _put("--- 2. Village Status OCR ---", 228, (0, 255, 120), 0.52, 2)
        _put(f"Free Builders: {village_status.free_builders} / {village_status.total_builders}", 253)
        _put(f"Own Gold     : {village_status.gold:,}", 276)
        _put(f"Own Elixir   : {village_status.elixir:,}", 299)

        _put("--- 3. Vision & Templates ---", 336, (255, 180, 80), 0.52, 2)
        _put(f"Full Collectors: {full_collectors} (DeadBase={is_dead})", 361)
        _put(f"RedZone Points : NW={len(sides['NW'])}, NE={len(sides['NE'])}, SW={len(sides['SW'])}, SE={len(sides['SE'])}", 384)
        _put(f"Matched Tpls   : {len(matched_templates)} / {len(self.harvester.TEMPLATE_SPECS)}", 407)

        # E. 在侧边栏下方贴出 3 个关键 ROI 的二值化预处理实时缩略图
        _put("--- 4. Binarized OCR Previews ---", 446, (0, 255, 255), 0.52, 2)
        preview_specs = [
            ("EnemyGold ROI", self.ocr.ENEMY_GOLD_ROI, 465),
            ("EnemyElixir ROI", self.ocr.ENEMY_ELIXIR_ROI, 545),
            ("Builder ROI", self.ocr.BUILDER_TEXT_ROI, 625),
        ]
        for title, (y1, y2, x1, x2), top_y in preview_specs:
            _put(title, top_y, (180, 180, 180), 0.42)
            crop = screen[y1:y2, x1:x2]
            bin_img = self.ocr.preprocess_coc_text(crop)
            bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
            thumb = cv2.resize(bin_bgr, (260, 48), interpolation=cv2.INTER_NEAREST)
            panel[top_y + 8:top_y + 56, 16:276] = thumb
            cv2.rectangle(panel, (16, top_y + 8), (276, top_y + 56), (100, 255, 100), 1)

        return np.hstack([vis, panel])
