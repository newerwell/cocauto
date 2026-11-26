"""
后台多线程工作引擎、解耦式实时画面直播流 (Live Vision Worker) 与日志信号桥接器

核心架构：
1. 采集与识别彻底解耦：
   - 采集端（主循环）：实时流模式优先通过 ADB screenrecord (H.264, 1280x720, 4Mbps) + PyAV 持续解码，
     只保留最新一帧（单槽覆盖，附带递增序号 seq 与采集时间戳），无待处理队列积压；
     达到 180 秒上限自动续接，启动或续接失败自动回退至 PNG 连续采集。
   - 识别端（独立后台线程）：每秒最多启动一轮，上一轮未完成自动跳过本次调度；
     统一完成场景判断、OCR、模板匹配和红线检测，并将结构化标注结果一次发布。
2. 单帧诊断模式 (`continuous=False`)：始终使用无损 ADB PNG 截图并执行完整视觉识别，
   确保单帧诊断、颜色探针与鼠标框选模板裁切 100% 无损一致。
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from config import AppConfig
from config.paths import get_app_root
from core.device import AdbDeviceController, BaseDeviceController
from core.tasks.village_travel import TRAVEL_REASONS
from core.vision import (
    DeadBaseChecker,
    EnemyLoot,
    OcrScanner,
    RedZoneDetector,
    TemplateMatcher,
    VillageStatus,
)
from tools.calibrator import TemplateHarvester
from .combat_run import CombatRunLog, LOOP_TITLES, round_report
from .screen_stream import AdbH264StreamClient, CapturedFrame, is_pyav_available

logger = logging.getLogger("coc.worker")


def travel_failure_text(result: dict) -> str:
    return f"渡船切换失败（{TRAVEL_REASONS.get(result.get('reason'), '无法确认抵达')}）"


@dataclass
class TemplateBoxItem:
    label: str
    x: int
    y: int
    w: int
    h: int
    bgr_color: Tuple[int, int, int]
    is_bubble: bool = False


@dataclass
class OcrRoiItem:
    label: str
    roi: Tuple[int, int, int, int]  # (y1, y2, x1, x2)
    bgr_color: Tuple[int, int, int]


@dataclass
class VisionOverlayData:
    """识别线程一次发布的完整结构化视觉结果（不含整帧像素，零大图拷贝）"""

    session_id: int
    device: str
    scene: str
    is_home_village: bool
    ocr_engine: str
    analysis_ms: float
    frame_seq: int
    analyzed_at: float
    enemy_loot: EnemyLoot = field(default_factory=EnemyLoot)
    village_status: VillageStatus = field(default_factory=VillageStatus)
    full_collectors: int = 0
    is_dead: bool = False
    matched_templates: List[str] = field(default_factory=list)
    template_boxes: List[TemplateBoxItem] = field(default_factory=list)
    ocr_rois: List[OcrRoiItem] = field(default_factory=list)
    red_zone_sides: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)
    map_diamond: Optional[np.ndarray] = None


def render_overlay_on_bgr(
    raw_bgr: np.ndarray,
    overlay: Optional[VisionOverlayData],
    show_ocr: bool = True,
    show_templates: bool = True,
    show_red_zone: bool = False,
) -> np.ndarray:
    """将结构化标注绘制到 BGR 副本（用于兼容旧接口/静态导出）"""
    vis = raw_bgr.copy()
    if overlay is None:
        return vis

    if show_templates:
        for box in overlay.template_boxes:
            cv2.rectangle(vis, (box.x, box.y), (box.x + box.w, box.y + box.h), box.bgr_color, 2)
            cv2.putText(
                vis,
                box.label,
                (box.x, max(14, box.y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42 if box.is_bubble else 0.45,
                box.bgr_color,
                1 if box.is_bubble else 2,
            )

    if show_ocr:
        for item in overlay.ocr_rois:
            y1, y2, x1, x2 = item.roi
            cv2.rectangle(vis, (x1, y1), (x2, y2), item.bgr_color, 2)
            cv2.putText(
                vis,
                item.label,
                (x1, max(14, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                item.bgr_color,
                1,
            )

    if show_red_zone and overlay.red_zone_sides:
        if overlay.map_diamond is not None:
            cv2.polylines(vis, [overlay.map_diamond], True, (130, 130, 130), 1)
        side_colors = {
            "NW": (255, 220, 0),
            "NE": (0, 255, 255),
            "SW": (255, 100, 255),
            "SE": (0, 255, 120),
        }
        for s_name, pts in overlay.red_zone_sides.items():
            c = side_colors.get(s_name, (0, 255, 0))
            for px, py in pts:
                cv2.circle(vis, (px, py), 5, c, -1)
                cv2.circle(vis, (px, py), 7, (0, 0, 0), 1)
    return vis


class QtLogSignalBridge(QObject):
    logEmitted = pyqtSignal(str, str)


class QtSignalLogHandler(logging.Handler):
    """将 Python 标准 logging 实时转发到 PyQt5 日志窗口的 Handler"""

    def __init__(self):
        super().__init__()
        self.bridge = QtLogSignalBridge()
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            self.bridge.logEmitted.emit(record.levelname, msg)
        except Exception:
            pass


class LiveVisionWorker(QThread):
    """
    解耦式实时画面采集与视觉分析后台工作线程：
    - 采集侧只维护最新一帧 `_latest_frame`（单调递增 seq），主线程按约 33ms 周期读取渲染；
    - 识别侧由独立子线程每秒最多触发一轮，上一轮未完成直接跳过，无待分析帧积压；
    - 支持 `continuous=False` 无损 PNG 单帧诊断。
    """

    # 兼容原有单帧/旧测试信号: (raw_bgr, overlay_bgr, metrics_dict)
    frameReady = pyqtSignal(object, object, dict)
    # 新增解耦信号（全部携带 session_id，便于切换设备/重开时丢弃旧会话回调）
    singleFrameReady = pyqtSignal(int, object, object, dict)  # (session_id, raw_bgr, VisionOverlayData, metrics)
    analysisReady = pyqtSignal(int, object)                   # (session_id, VisionOverlayData)
    captureModeChanged = pyqtSignal(int, str, str)            # (session_id, mode_code, mode_label)
    errorOccurred = pyqtSignal(int, str)                      # (session_id, err_msg)

    BUBBLE_KEYS = {"bubble_gold", "bubble_elixir", "bubble_dark", "night_bubble_gold", "night_bubble_elixir"}

    def __init__(
        self,
        serial: str = "auto",
        show_ocr_rois: bool = True,
        show_red_zone: bool = False,
        show_templates: bool = True,
        continuous: bool = False,
        interval_ms: int = 160,
        mock_device: Optional[BaseDeviceController] = None,
        session_id: int = 0,
        prefer_h264: bool = True,
        segment_limit_sec: int = 180,
        analysis_interval_sec: float = 1.0,
        force_h264_fail: bool = False,
        analysis_hook: Optional[Callable[[np.ndarray], None]] = None,
        skip_analysis: bool = False,
    ):
        super().__init__()
        self.serial = serial
        self.show_ocr_rois = show_ocr_rois
        self.show_red_zone = show_red_zone
        self.show_templates = show_templates
        self.continuous = continuous
        self.interval_ms = interval_ms
        self.mock_device = mock_device
        self.session_id = session_id
        self.prefer_h264 = prefer_h264
        self.segment_limit_sec = segment_limit_sec
        self.analysis_interval_sec = max(0.2, float(analysis_interval_sec))
        self.force_h264_fail = force_h264_fail
        self.analysis_hook = analysis_hook
        self.skip_analysis = skip_analysis

        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[CapturedFrame] = None
        self._frame_seq: int = 0
        self._stream_client: Optional[AdbH264StreamClient] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._resolved_device_serial: str = serial
        self._capture_mode: str = "png_single"
        self._template_cache_dirty = threading.Event()
        self._template_cache_dirty.set()
        self._analysis_trigger_event = threading.Event()

        # 运行期统计（便于验收与监控）
        self.total_captured_frames: int = 0
        self.total_analysis_rounds: int = 0
        self.skipped_analysis_ticks: int = 0
        self.first_frame_ms: Optional[float] = None
        self.max_reconnect_gap_ms: float = 0.0
        self.reconnect_count: int = 0

    def set_layer_flags(self, show_ocr: bool, show_templates: bool, show_red_zone: bool) -> None:
        changed = (
            self.show_ocr_rois != show_ocr
            or self.show_templates != show_templates
            or self.show_red_zone != show_red_zone
        )
        self.show_ocr_rois = show_ocr
        self.show_templates = show_templates
        self.show_red_zone = show_red_zone
        if changed:
            self._analysis_trigger_event.set()

    def invalidate_template_cache(self) -> None:
        self._template_cache_dirty.set()
        self._analysis_trigger_event.set()

    def get_latest_frame(self) -> Optional[CapturedFrame]:
        """供 Qt 主线程 33ms 定时器无阻塞读取最新一帧"""
        with self._frame_lock:
            return self._latest_frame

    def get_capture_mode(self) -> str:
        return self._capture_mode

    def _publish_latest_frame(
        self,
        bgr: np.ndarray,
        captured_at: float,
        source: str,
    ) -> CapturedFrame:
        with self._frame_lock:
            self._frame_seq += 1
            frame = CapturedFrame(
                seq=self._frame_seq,
                bgr=bgr,
                captured_at=captured_at,
                source=source,
                session_id=self.session_id,
            )
            self._latest_frame = frame
            self.total_captured_frames += 1
            return frame

    def stop_stream(self) -> None:
        self._stop_event.set()
        self._analysis_trigger_event.set()
        client = self._stream_client
        if client is not None:
            client.stop()

    def _run_one_analysis(
        self,
        raw: np.ndarray,
        frame_seq: int,
        ocr: OcrScanner,
        matcher: TemplateMatcher,
        red_detector: RedZoneDetector,
        dead_checker: DeadBaseChecker,
        harvester: TemplateHarvester,
        force_all_layers: bool = False,
    ) -> VisionOverlayData:
        t0 = time.perf_counter()

        if self._template_cache_dirty.is_set():
            matcher._cache.clear()
            self._template_cache_dirty.clear()

        if self.analysis_hook is not None:
            self.analysis_hook(raw)

        # 1. 场景判断：主村庄 vs 夜世界 vs 搜鱼/战场
        from core.tasks.village_routine import VillageRoutineTask
        atk_btn = matcher.find_one(raw, "village/btn_attack.png", threshold=0.76, scales=(1.0,))
        night_atk_btn = matcher.find_one(raw, "night/btn_night_attack.png", threshold=0.76, scales=(1.0,))
        is_night_village = (night_atk_btn is not None) or (atk_btn is None and VillageRoutineTask.is_in_builder_base(raw))
        is_home_village = (atk_btn is not None) or is_night_village
        if is_night_village:
            scene_name = "夜世界 (Builder Base)"
        elif atk_btn is not None:
            scene_name = "主村庄 (Home Village)"
        else:
            scene_name = "搜鱼/战场 (Battle)"

        # 2. OCR 数字识别
        want_ocr = force_all_layers or self.show_ocr_rois
        if want_ocr:
            if is_home_village:
                village_status = ocr.scan_village_status(raw)
                enemy_loot = EnemyLoot()
            else:
                enemy_loot = ocr.scan_enemy_loot(raw)
                village_status = ocr.scan_village_status(raw)
        else:
            enemy_loot = EnemyLoot()
            village_status = VillageStatus()

        ocr_rois: List[OcrRoiItem] = []
        if is_home_village:
            ocr_rois = [
                OcrRoiItem(
                    f"Builders:{village_status.free_builders}/{village_status.total_builders}",
                    ocr.BUILDER_TEXT_ROI,
                    (0, 255, 120),
                ),
                OcrRoiItem(f"Gold:{village_status.gold:,}", ocr.OWN_GOLD_ROI, (0, 215, 255)),
                OcrRoiItem(f"Elixir:{village_status.elixir:,}", ocr.OWN_ELIXIR_ROI, (255, 100, 255)),
                OcrRoiItem(f"Dark:{village_status.dark_elixir:,}", ocr.OWN_DARK_ROI, (200, 200, 200)),
            ]
        else:
            ocr_rois = [
                OcrRoiItem(f"EnemyGold:{enemy_loot.gold:,}", ocr.ENEMY_GOLD_ROI, (0, 215, 255)),
                OcrRoiItem(f"EnemyElixir:{enemy_loot.elixir:,}", ocr.ENEMY_ELIXIR_ROI, (255, 100, 255)),
                OcrRoiItem(f"EnemyDark:{enemy_loot.dark_elixir:,}", ocr.ENEMY_DARK_ROI, (200, 200, 200)),
                OcrRoiItem("Builders", ocr.BUILDER_TEXT_ROI, (0, 255, 120)),
            ]

        # 3. 气泡与按钮模板匹配 (金币/圣水/黑油三类物资各仅高亮置信度最高的 1 个首选点击气泡)
        matched_names: List[str] = []
        template_boxes: List[TemplateBoxItem] = []
        want_templates = force_all_layers or self.show_templates
        if want_templates:
            if is_home_village:
                village_roi = (55, 560, 140, 1050)
                if is_night_village:
                    bubble_groups = [
                        ("NightGold#1", (0, 215, 255), ["night/bubble_gold.png", "night/bubble_gold_full.png", "village/bubble_gold_full.png"]),
                        ("NightElixir#1", (255, 100, 255), ["night/bubble_elixir.png", "night/bubble_elixir_full.png", "village/bubble_elixir.png"]),
                    ]
                else:
                    bubble_groups = [
                        ("Gold#1", (0, 215, 255), ["village/bubble_gold.png", "village/bubble_gold_full.png"]),
                        ("Elixir#1", (255, 100, 255), ["village/bubble_elixir.png", "village/bubble_elixir_full.png"]),
                        ("Dark#1", (0, 255, 180), ["village/bubble_dark.png", "village/bubble_dark_full.png"]),
                    ]
                for b_label, b_color, tpl_candidates in bubble_groups:
                    best_m = None
                    for tp in tpl_candidates:
                        cand = matcher.find_one(raw, tp, threshold=0.72 if is_night_village else 0.78, roi=village_roi, scales=(1.0,))
                        if cand is not None and (best_m is None or cand.score > best_m.score):
                            best_m = cand
                    if best_m is not None:
                        matched_names.append(f"{b_label}({best_m.score:.2f})")
                        template_boxes.append(
                            TemplateBoxItem(
                                label=f"{b_label}:{best_m.score:.2f}",
                                x=best_m.x,
                                y=best_m.y,
                                w=best_m.w,
                                h=best_m.h,
                                bgr_color=b_color,
                                is_bubble=True,
                            )
                        )

            for key, spec in harvester.TEMPLATE_SPECS.items():
                if key in self.BUBBLE_KEYS:
                    continue
                if is_home_village and spec.scene == "battle":
                    continue
                m = atk_btn if spec.rel_path == "village/btn_attack.png" else matcher.find_one(
                    raw, spec.rel_path, threshold=0.76, scales=(1.0,)
                )
                if m:
                    matched_names.append(f"{key}({m.score:.2f})")
                    template_boxes.append(
                        TemplateBoxItem(
                            label=f"{key}:{m.score:.2f}",
                            x=m.x,
                            y=m.y,
                            w=m.w,
                            h=m.h,
                            bgr_color=(0, 255, 0),
                            is_bubble=False,
                        )
                    )

        # 4. 动态红线与死鱼采集器检测（复用 full_collectors 结果避免二次全图扫描）
        sides: Dict[str, List[Tuple[int, int]]] = {}
        if self.show_red_zone:
            sides = red_detector.detect_deployable_sides(raw, points_per_side=10)

        full_collectors = 0 if is_home_village else dead_checker.count_full_elixir_collectors(raw)
        is_dead = False if is_home_village else (full_collectors >= 3)

        analysis_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return VisionOverlayData(
            session_id=self.session_id,
            device=self._resolved_device_serial,
            scene=scene_name,
            is_home_village=is_home_village,
            ocr_engine=ocr.backend_name,
            analysis_ms=analysis_ms,
            frame_seq=frame_seq,
            analyzed_at=time.perf_counter(),
            enemy_loot=enemy_loot,
            village_status=village_status,
            full_collectors=full_collectors,
            is_dead=is_dead,
            matched_templates=matched_names,
            template_boxes=template_boxes,
            ocr_rois=ocr_rois,
            red_zone_sides=sides,
            map_diamond=red_detector.MAP_DIAMOND.copy() if sides else None,
        )

    def _analysis_loop(self) -> None:
        """
        独立识别线程：每秒最多启动一轮；上一轮未完成自然跳过本次调度，后台永远不积压待分析帧。
        """
        try:
            # 限制 OpenCV 后台模板匹配并发线程，避免抢占 Qt 主线程和 H.264 解码 CPU
            cv2.setNumThreads(2)
        except Exception:
            pass
        try:
            ocr = OcrScanner()
            matcher = TemplateMatcher()
            red_detector = RedZoneDetector()
            dead_checker = DeadBaseChecker(matcher)
            harvester = TemplateHarvester()
        except Exception as e:
            logger.error(f"[视觉识别线程] 初始化视觉组件失败: {e}")
            return

        last_analyzed_seq = -1
        next_due = time.perf_counter()

        while not self._stop_event.is_set():
            now = time.perf_counter()
            wait_sec = next_due - now
            if wait_sec > 0:
                triggered = self._analysis_trigger_event.wait(timeout=min(wait_sec, 0.25))
                if self._stop_event.is_set():
                    break
                if triggered:
                    self._analysis_trigger_event.clear()
                    # 图层变更立即触发一次分析，但仍然受单线程串行保护
                else:
                    continue

            frame = self.get_latest_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            # 若画面未更新且不是图层显式刷新，略过本轮等待下一个周期
            if frame.seq == last_analyzed_seq and not self._template_cache_dirty.is_set():
                next_due = time.perf_counter() + self.analysis_interval_sec
                continue

            round_start = time.perf_counter()
            raw_bgr = frame.get_bgr()
            if raw_bgr is None:
                time.sleep(0.05)
                continue
            try:
                overlay = self._run_one_analysis(
                    raw_bgr,
                    frame.seq,
                    ocr,
                    matcher,
                    red_detector,
                    dead_checker,
                    harvester,
                )
            except Exception as e:
                logger.warning(f"[视觉识别线程] 单轮分析异常（不影响实时预览）: {e}")
                next_due = time.perf_counter() + self.analysis_interval_sec
                continue

            if self._stop_event.is_set():
                break

            last_analyzed_seq = frame.seq
            self.total_analysis_rounds += 1
            self.analysisReady.emit(self.session_id, overlay)

            # 若本轮耗时超过 1 个调度周期，统计跳过的周期数并直接对齐到下一个未来节拍（永不追赶积压）
            finished_at = time.perf_counter()
            elapsed = finished_at - round_start
            if elapsed > self.analysis_interval_sec:
                self.skipped_analysis_ticks += int(elapsed // self.analysis_interval_sec)
            next_due = finished_at + self.analysis_interval_sec

    def _run_png_fallback_loop(self, dev: BaseDeviceController, reason: str = "") -> None:
        self._capture_mode = "png_fallback"
        label = f"PNG 回退模式 ({reason})" if reason else "PNG 回退模式"
        self.captureModeChanged.emit(self.session_id, "png_fallback", label)
        t_start = time.perf_counter()

        while not self._stop_event.is_set():
            t0 = time.perf_counter()
            try:
                raw = dev.screencap()
            except Exception as e:
                if self._stop_event.is_set():
                    break
                logger.warning(f"[PNG 回退采集] 截屏失败: {e}")
                self._stop_event.wait(0.3)
                continue

            now = time.perf_counter()
            if self.first_frame_ms is None:
                self.first_frame_ms = (now - t_start) * 1000.0
            self._publish_latest_frame(raw, now, "png_fallback")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            sleep_sec = max(0.015, (self.interval_ms - elapsed_ms) / 1000.0)
            self._stop_event.wait(sleep_sec)

    def run(self) -> None:
        try:
            dev = self.mock_device or AdbDeviceController(serial=self.serial)
            if not dev.connect():
                self.errorOccurred.emit(self.session_id, f"未能连接设备 [{self.serial}]，请确认模拟器已开启。")
                return

            self._resolved_device_serial = getattr(dev, "_resolved_serial", None) or self.serial

            # ==================== 1. 单帧无损 PNG 诊断模式 ====================
            if not self.continuous:
                self._capture_mode = "png_single"
                self.captureModeChanged.emit(self.session_id, "png_single", "单帧无损 PNG")
                t0 = time.perf_counter()
                raw = dev.screencap()
                now = time.perf_counter()
                self.first_frame_ms = (now - t0) * 1000.0
                frame = self._publish_latest_frame(raw, now, "png_single")

                if self.skip_analysis:
                    if self._stop_event.is_set():
                        return
                    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                    empty_overlay = VisionOverlayData(
                        session_id=self.session_id,
                        device=self._resolved_device_serial,
                        scene="冻结无损 PNG",
                        is_home_village=True,
                        ocr_engine="-",
                        analysis_ms=0.0,
                        frame_seq=frame.seq,
                        analyzed_at=now,
                    )
                    metrics = {
                        "session_id": self.session_id,
                        "device": self._resolved_device_serial,
                        "scene": empty_overlay.scene,
                        "ocr_engine": "-",
                        "elapsed_ms": elapsed_ms,
                        "analysis_ms": 0.0,
                        "fps": round(1000.0 / max(1.0, elapsed_ms), 1),
                        "capture_mode": "png_single",
                        "enemy_loot": empty_overlay.enemy_loot,
                        "village_status": empty_overlay.village_status,
                        "full_collectors": 0,
                        "is_dead": False,
                        "matched_templates": [],
                    }
                    self.singleFrameReady.emit(self.session_id, raw, empty_overlay, metrics)
                    self.frameReady.emit(raw, raw.copy(), metrics)
                    return

                ocr = OcrScanner()
                matcher = TemplateMatcher()
                red_detector = RedZoneDetector()
                dead_checker = DeadBaseChecker(matcher)
                harvester = TemplateHarvester()
                overlay = self._run_one_analysis(
                    raw,
                    frame.seq,
                    ocr,
                    matcher,
                    red_detector,
                    dead_checker,
                    harvester,
                )
                if self._stop_event.is_set():
                    return

                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
                vis = render_overlay_on_bgr(
                    raw,
                    overlay,
                    show_ocr=self.show_ocr_rois,
                    show_templates=self.show_templates,
                    show_red_zone=self.show_red_zone,
                )
                metrics = {
                    "session_id": self.session_id,
                    "device": self._resolved_device_serial,
                    "scene": overlay.scene,
                    "ocr_engine": overlay.ocr_engine,
                    "elapsed_ms": elapsed_ms,
                    "analysis_ms": overlay.analysis_ms,
                    "fps": round(1000.0 / max(1.0, elapsed_ms), 1),
                    "capture_mode": "png_single",
                    "enemy_loot": overlay.enemy_loot,
                    "village_status": overlay.village_status,
                    "full_collectors": overlay.full_collectors,
                    "is_dead": overlay.is_dead,
                    "matched_templates": overlay.matched_templates,
                }
                self.singleFrameReady.emit(self.session_id, raw, overlay, metrics)
                self.analysisReady.emit(self.session_id, overlay)
                self.frameReady.emit(raw, vis, metrics)
                return

            # ==================== 2. 连续实时画面流模式 ====================
            # 启动独立识别线程（采集端只负责把最新帧写入单槽，完全不被识别耗时拖慢）
            self._analysis_thread = threading.Thread(
                target=self._analysis_loop,
                daemon=True,
                name=f"coc-vision-analyzer-{self.session_id}",
            )
            self._analysis_thread.start()

            use_h264 = (
                self.prefer_h264
                and self.mock_device is None
                and isinstance(dev, AdbDeviceController)
                and not self.force_h264_fail
                and is_pyav_available()
            )

            if use_h264:
                self._capture_mode = "h264"
                self.captureModeChanged.emit(self.session_id, "h264", "H.264 实时流")
                client = AdbH264StreamClient(
                    serial=self._resolved_device_serial,
                    segment_limit_sec=self.segment_limit_sec,
                )
                self._stream_client = client

                def _on_h264_frame(bgr: np.ndarray, ts: float):
                    if self.first_frame_ms is None and client.first_frame_ms is not None:
                        self.first_frame_ms = client.first_frame_ms
                    self._publish_latest_frame(bgr, ts, "h264")

                def _on_reconnect(count: int, gap_ms: float):
                    self.reconnect_count = count
                    if gap_ms > self.max_reconnect_gap_ms:
                        self.max_reconnect_gap_ms = gap_ms

                try:
                    client.run_stream(on_frame=_on_h264_frame, on_reconnect=_on_reconnect)
                    if self.first_frame_ms is None:
                        self.first_frame_ms = client.first_frame_ms
                    self.reconnect_count = client.reconnect_count
                    self.max_reconnect_gap_ms = max(self.max_reconnect_gap_ms, client.max_reconnect_gap_ms)
                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.warning(f"[实时流降级] H.264 采集失败，自动回退到 PNG 预览: {e}")
                        self._run_png_fallback_loop(dev, reason="H.264 失败自动回退")
                finally:
                    self._stream_client = None
            else:
                reason = ""
                if self.force_h264_fail:
                    reason = "模拟视频流失败回退"
                elif self.mock_device is not None:
                    reason = "模拟设备"
                elif not is_pyav_available():
                    reason = "缺少 PyAV"
                self._run_png_fallback_loop(dev, reason=reason)

        except Exception as e:
            if not self._stop_event.is_set():
                self.errorOccurred.emit(self.session_id, f"实时画面流异常: {e}")
        finally:
            self._stop_event.set()
            self._analysis_trigger_event.set()
            if self._analysis_thread is not None and self._analysis_thread.is_alive():
                self._analysis_thread.join(timeout=2.0)


class BotActionWorker(QThread):
    """后台挂机主循环与单步功能验证工作线程"""

    statusChanged = pyqtSignal(str)
    cycleCompleted = pyqtSignal(dict)
    finishedTask = pyqtSignal(str)
    runLogOpened = pyqtSignal(str)

    def __init__(
        self,
        cfg: AppConfig,
        mode: str = "full_auto",
        align_mode: str = "center",
        mock_device: Optional[BaseDeviceController] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.align_mode = align_mode
        self.mock_device = mock_device
        self._stop_requested = False
        self._active_device: Optional[BaseDeviceController] = None
        self._run_log: Optional[CombatRunLog] = None
        self._run_lock = threading.RLock()
        self._active_round = 0

    def request_stop(self):
        with self._run_lock:
            self._stop_requested = True
            if self._run_log:
                self._run_log.write("收到正常停止请求 round=%d", self._active_round)

    def force_kill_now(self):
        """🚨 0.1 秒强行终止：立即杀掉当前 ADB 子进程、释放多指触控点并终止 QThread"""
        with self._run_lock:
            self._stop_requested = True
            if self._run_log:
                self._run_log.write("收到强行停止请求 round=%d", self._active_round)
        dev = self._active_device
        if dev is not None and hasattr(dev, "force_abort"):
            try:
                dev.force_abort()
            except Exception:
                pass
        if self.isRunning():
            with self._run_lock:
                if self._active_round:
                    self._publish_round({"status": "interrupted", "reason": "force_stopped"})
                self._close_run_log("force_stopped")
                self.terminate()
            self.wait(300)

    def _close_run_log(self, reason: str):
        with self._run_lock:
            if self._run_log:
                self._run_log.write("运行结束 reason=%s", reason)
                self._run_log.close()
                self._run_log = None

    def _publish_round(self, result: dict) -> dict:
        with self._run_lock:
            index = self._active_round
            report = round_report(index, self.mode, result)
            self._run_log.record_round(index, result, report)
            self._active_round = 0
            self.cycleCompleted.emit({"combat_loop_round": index, "result": result, "round_report": report})
            return report

    def _run_combat_loop(self, bot):
        title = LOOP_TITLES[self.mode]
        execute = getattr(bot, {
            "loop_dual_combat": "run_dual_world_combat_cycle", "loop_night_combat": "run_night_battle_once",
            "loop_home_combat": "run_home_battle_once",
        }[self.mode])
        index = 0
        while not self._stop_requested:
            with self._run_lock:
                index += 1
                self._active_round = index
                self._run_log.write("轮次开始 round=%d mode=%s", index, self.mode)
            self.statusChanged.emit(f"正在运行：{title} (第 {index} 轮)")
            result = execute()
            if self._stop_requested:
                result = {**result, "status": "interrupted"}
            report = self._publish_round(result)
            self.statusChanged.emit(report["headline"])
            if report["status"] == "travel_failed":
                self._close_run_log("travel_failed")
                self.finishedTask.emit(travel_failure_text(result["travel"]))
                return
            if result.get("status") == "interrupted":
                break
            for _ in range(5):
                if self._stop_requested:
                    break
                time.sleep(0.5)
        self._close_run_log("interrupted")
        self.finishedTask.emit(f"已停止 {title}（运行 {index} 轮）")

    def run(self):
        bot_logger = logging.getLogger("coc.gui_worker")
        try:
            if self.mode in LOOP_TITLES:
                with self._run_lock:
                    self._run_log = CombatRunLog(get_app_root(), self.mode, self.cfg.device.serial)
                    self.runLogOpened.emit(str(self._run_log.path))
            from main import CocAutoBotOrchestrator
            bot_logger.info(f"▶ [启动测试/任务] 当前模式={self.mode} | 目标设备={self.cfg.device.serial}")
            bot = CocAutoBotOrchestrator(self.cfg, device=self.mock_device, stopped=lambda: self._stop_requested)
            self._active_device = bot.device
            if not bot.device.connect():
                bot_logger.error("❌ 设备连接失败，请先确认模拟器 (如 127.0.0.1:5561) 在线！")
                self._close_run_log("connection_failed")
                self.finishedTask.emit("连接失败")
                return
            if self._run_log:
                serial = getattr(bot.device, "_resolved_serial", None) or getattr(bot.device, "serial", None)
                self._run_log.write("设备已连接 resolved_serial=%s controller=%s", serial, type(bot.device).__name__)
                self._run_combat_loop(bot)
                return

            if self.mode == "step_zoom":
                self.statusChanged.emit(f"正在执行：视角缩放归一化 ({self.align_mode})")
                try:
                    bot.device.pinch_zoom_out(align_mode=self.align_mode)
                except TypeError:
                    bot.device.pinch_zoom_out()
                self.finishedTask.emit("视角归一化完成")
                return

            if self.mode == "step_village":
                self.statusChanged.emit("正在执行：独立测试自动收菜 (金/水/黑油各仅点1次)")
                bot_logger.info("🏡 [单步调试-仅收菜] 开始执行自动收菜 (金币、圣水、黑油每种物资仅点 1 次，不做捐兵与补兵)...")
                res = bot.village_task.run_collect_only()
                if res.get("status") == "travel_failed":
                    self.cycleCompleted.emit({"village": res})
                    self.finishedTask.emit(travel_failure_text(res["travel"]))
                    return
                bot_logger.info(
                    f"✅ [单步调试完成] 自动收菜结果: 共点击收取 {res.get('collected_bubbles', 0)}/3 类物资！"
                )
                self.cycleCompleted.emit({"village": res})
                self.finishedTask.emit(f"自动收菜完成 ({res.get('collected_bubbles', 0)}/3类)")
                return

            if self.mode == "step_upgrade":
                self.statusChanged.emit("正在执行：工人建议列表刷墙与建筑升级")
                bot_logger.info("🧱 [单步调试] 开始扫描空闲工人与建议升级列表...")
                res = bot.upgrade_task.run_check_and_upgrade()
                bot_logger.info(
                    f"✅ [单步调试完成] 刷墙={res.get('walls_upgraded', 0)} 块, 升级建筑={res.get('buildings_upgraded', 0)} 个"
                )
                self.cycleCompleted.emit({"upgrade": res})
                self.finishedTask.emit("刷墙与升级检查完成")
                return

            if self.mode == "step_search":
                mode_cn = "排位战 (搜到谁直接打谁)" if self.cfg.loot_filter.match_mode == "ranked" else "常规战 (按阈值挑鱼)"
                self.statusChanged.emit(f"正在执行：联机搜鱼 ({mode_cn})")
                bot_logger.info(f"🔍 [单步调试-联机搜鱼] 启动四步进场与搜鱼 (模式={mode_cn})...")
                loot = bot.search_task.start_matchmaking_and_find_target()
                if loot is None and self.cfg.loot_filter.match_mode == "ranked":
                    bot_logger.info("🏆 [排位赛已全部完成] 排位赛场次已打满或未进入战场，已安全返回主世界！")
                    self.finishedTask.emit("排位赛已完成")
                    return
                bot_logger.info(f"✅ [搜鱼阶段完成] 锁定对手资源: {loot}")
                combat_res = {}
                if loot is not None and (
                    self.cfg.loot_filter.match_mode == "ranked" or self.cfg.loot_filter.auto_attack_on_found
                ):
                    self.statusChanged.emit("已锁定对手 -> 正在执行动态红线智能下兵战斗")
                    bot_logger.info("⚔️ [自动衔接下兵] 已锁定目标对手，立即启动动态红线检测与智能下兵战斗！")
                    combat_res = bot.combat_task.execute_attack()
                self.cycleCompleted.emit({"target_loot": loot, "combat": combat_res})
                self.finishedTask.emit("联机搜鱼与战斗完成")
                return

            if self.mode == "step_night_boat":
                self.statusChanged.emit("正在识别木船并切换世界")
                res = bot.village_task.travel.toggle()
                self.cycleCompleted.emit({"night_world": res})
                if res["arrived_home"]:
                    self.finishedTask.emit("已乘船返回主世界")
                elif res["arrived_night_world"]:
                    self.finishedTask.emit("已乘船抵达夜世界")
                else:
                    self.finishedTask.emit(travel_failure_text(res))
                return

            if self.mode == "step_night_collect":
                self.statusChanged.emit("正在执行：夜世界自动收菜 (金币/圣水各仅点1次)")
                bot_logger.info("🌙 [单步调试-夜世界收菜] 开始执行夜世界建筑大师金币与圣水收集 (各仅点 1 次)...")
                res = bot.night_task.collect_night_resources()
                self.cycleCompleted.emit({"night_collect": res})
                if res.get("status") == "travel_failed":
                    self.finishedTask.emit(travel_failure_text(res["travel"]))
                    return
                self.finishedTask.emit(f"夜世界收菜完成 ({res.get('night_collected_bubbles', 0)}/2类)")
                return

            if self.mode == "step_night_battle":
                self.statusChanged.emit("正在执行：夜世界 2.0 智能战斗 (先清边后中路齐下+技能+自动回营)")
                bot_logger.info("⚔️ [单步调试-夜世界战斗] 启动夜世界进场匹配、先两翼清边后中路齐下、技能触发与自动回营...")
                res = bot.night_task.execute_night_battle()
                self.cycleCompleted.emit({"night_battle": res})
                travel = bot.night_task.last_travel
                if travel is not None and not travel["arrived_night_world"]:
                    self.finishedTask.emit(travel_failure_text(travel))
                    return
                self.finishedTask.emit("夜世界自动战斗完成")
                return

            if self.mode == "step_combat":
                self.statusChanged.emit("正在执行：自动进场/红线检测与智能下兵战斗")
                screen_now = bot.device.screencap()
                if not bot.search_task.is_on_battlefield_screen(screen_now):
                    bot_logger.info("ℹ️ [智能战斗检测] 当前尚未处于敌方战场，先自动执行进场与搜鱼锁定对手...")
                    loot = bot.search_task.start_matchmaking_and_find_target()
                    if loot is None:
                        bot_logger.warning("⚠️ 未能搜到符合条件的对手，取消本次下兵。")
                        self.finishedTask.emit("未搜到目标对手")
                        return
                bot_logger.info("⚔️ [单步调试-智能下兵] 开始在敌方战场提取动态红线并投放全部出战卡牌...")
                res = bot.combat_task.execute_attack()
                bot_logger.info(f"✅ [单步调试完成] 战斗结束，触发英雄技能={res.get('hero_abilities_used', 0)} 次")
                self.cycleCompleted.emit({"combat": res})
                self.finishedTask.emit("智能战斗执行完成")
                return

            cycle = 0
            while not self._stop_requested:
                cycle += 1
                self.statusChanged.emit(f"全自动挂机运行中 (第 {cycle} 轮)")
                bot_logger.info(f"========== [GUI全自动挂机] 开始第 {cycle} 轮循环 ==========")
                summary = bot.run_single_cycle()
                self.cycleCompleted.emit(summary)

                if self._stop_requested:
                    break
                if summary.get("status") == "resting":
                    rest_sec = int(summary.get("rest_sec", 60))
                    for _ in range(min(rest_sec, 1800)):
                        if self._stop_requested:
                            break
                        time.sleep(1)
                    bot.scheduler.reset_session()

            self.finishedTask.emit("已安全停止挂机")
        except Exception as e:
            bot_logger.error(f"执行任务异常: {e}\n{traceback.format_exc()}")
            reason = "interrupted" if self._stop_requested else "error"
            if self._active_round:
                self._publish_round({"status": reason, "error": str(e)})
            self._close_run_log(reason)
            self.finishedTask.emit("已停止" if self._stop_requested else f"异常终止: {e}")
        finally:
            self._close_run_log("worker_exit")
