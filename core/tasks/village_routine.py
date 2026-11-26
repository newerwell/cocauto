"""
村庄日常管家任务 (Village Routine Task v2.3)
模块化重构升级：
1. 自动收菜 (`collect_resource_bubbles`)：
   - 遵循《部落冲突》“点击 1 个同类气泡即可收取全村所有同类采集器”机制，对【金币 Gold】、【圣水 Elixir】、【黑油 Dark Elixir】三类物资分别独立检测；
   - 每类物资同时匹配“正常白底气泡”与“满仓红底气泡”双模板，在主村庄安全区 `VILLAGE_MAP_ROI = (80, 560, 150, 1050)` 内仅选取置信度最高的 1 个气泡点击 1 次（单轮最多点击 3 下），彻底杜绝重复点击打开底部建筑详情栏！
2. 独立收菜测试入口 (`run_collect_only`)：
   - 仅测试自动收菜，彻底剥离自动捐兵、请求援军与快速补兵，支持分模块独立验收。
"""
from __future__ import annotations
import logging
from pathlib import Path
import time
from typing import Callable, List, Tuple, Optional
import cv2
import numpy as np

from config import VillageConfig
from core.device import BaseDeviceController
from core.vision import TemplateMatcher, OcrScanner, MatchResult
from core.vision.wooden_boat import WoodenBoatDetector
from .village_travel import VillageTravel

logger = logging.getLogger("coc.tasks.village")


class VillageRoutineTask:
    """主村庄内政收菜、捐兵、训练补兵与跨世界渡船管理器（支持分模块独立调用）"""

    # 主村庄内部建筑与采集器有效地图 ROI (y1, y2, x1, x2)，完全避开顶栏、底栏与左右两侧 UI 按钮
    VILLAGE_MAP_ROI: Tuple[int, int, int, int] = (80, 560, 150, 1050)

    # 1280x720 标准分辨率下的关键 UI 基准坐标 (已针对最新国服实机界面校准)
    BTN_CLAN_CHAT_OPEN = (55, 485)     # 左侧部落城堡聊天图标按钮
    BTN_CLAN_CHAT_CLOSE = (515, 355)   # 部落聊天收起按钮
    BTN_ARMY_OVERVIEW = (55, 615)      # 左侧双剑(部队训练总览)按钮
    BTN_QUICK_TRAIN_TAB = (925, 105)   # 一键快速训练标签页
    BTN_QUICK_TRAIN_APPLY = (1035, 295)# 快速训练“训练”绿色按钮
    BTN_REQUEST_CC = (145, 635)        # 训练总览内“请求增援”按钮
    BTN_CLOSE_MODAL = (1145, 75)       # 通用弹窗右上角关闭 X 按钮

    # 三类独立物资气泡模板组 (每类包含：正常未满仓白底模板 + 满仓/半满红底模板)
    RESOURCE_BUBBLE_GROUPS: List[Tuple[str, str, List[str]]] = [
        ("金币 (Gold)", "🪙", ["village/bubble_gold.png", "village/bubble_gold_full.png"]),
        ("圣水 (Elixir)", "💧", ["village/bubble_elixir.png", "village/bubble_elixir_full.png"]),
        ("黑油 (Dark Elixir)", "🛢️", ["village/bubble_dark.png", "village/bubble_dark_full.png"]),
    ]

    # 保持向后兼容的扁平模板列表
    BUBBLE_TEMPLATES = [
        "village/bubble_gold.png",
        "village/bubble_gold_full.png",
        "village/bubble_elixir.png",
        "village/bubble_elixir_full.png",
        "village/bubble_dark.png",
        "village/bubble_dark_full.png",
    ]

    def __init__(
        self,
        device: BaseDeviceController,
        matcher: TemplateMatcher,
        ocr: OcrScanner,
        cfg: VillageConfig,
        *,
        stopped: Callable[[], bool] = lambda: False,
        wait: Callable[[float], None] = time.sleep,
        travel_evidence_dir: Path | None = None,
    ):
        self.device = device
        self.matcher = matcher
        self.ocr = ocr
        self.cfg = cfg
        self.travel = VillageTravel(device, WoodenBoatDetector(matcher), stopped=stopped,
                                    wait=wait, evidence_dir=travel_evidence_dir)

    def find_best_bubble_per_resource(
        self,
        screen: np.ndarray,
        threshold: float = 0.78,
    ) -> List[Tuple[str, str, str, Optional[MatchResult]]]:
        """
        在主村庄安全区 VILLAGE_MAP_ROI 内，为【金币】、【圣水】、【黑油】三类物资各找出置信度最高的 1 个气泡。
        返回: [(res_label, emoji, matched_tpl_path, best_match_or_None), ...]
        """
        results: List[Tuple[str, str, str, Optional[MatchResult]]] = []
        for res_label, emoji, tpl_list in self.RESOURCE_BUBBLE_GROUPS:
            best_match: Optional[MatchResult] = None
            best_tpl: str = ""
            for tpl_path in tpl_list:
                m = self.matcher.find_one(
                    screen,
                    tpl_path,
                    threshold=threshold,
                    roi=self.VILLAGE_MAP_ROI,
                    scales=(1.0,),
                )
                if m is not None and (best_match is None or m.score > best_match.score):
                    best_match = m
                    best_tpl = tpl_path
            results.append((res_label, emoji, best_tpl, best_match))
        return results

    def collect_resource_bubbles(self, screen: np.ndarray | None = None) -> int:
        """
        识别并点击村庄内的金币、圣水、黑油采集器气泡：
        每种物资严格最多只点击 1 次（取置信度最高的气泡），单轮最多点击 3 下！
        """
        if screen is None:
            screen = self.travel.collection_frame("home")
        if screen is None:
            logger.warning("主世界收菜前未确认全景")
            return 0
        best_bubbles = self.find_best_bubble_per_resource(screen, threshold=0.78)
        total_clicked = 0

        logger.info("========== [自动收菜模块] 开始扫描主村庄三类物资气泡 (金/水/黑油各仅点 1 次) ==========")
        for idx, (res_label, emoji, best_tpl, match) in enumerate(best_bubbles, start=1):
            if self.travel.stopped():
                break
            if match is None:
                logger.info(f"ℹ️ [自动收菜 {idx}/3 - {res_label}] 当前村庄未检测到可收集气泡，跳过。")
                continue

            cx, cy = match.center
            state_tag = "满仓红底" if "_full" in best_tpl else "正常白底"
            logger.info(
                f"{emoji} [自动收菜 {idx}/3 - {res_label}] 识别到最佳气泡 -> 坐标=({cx}, {cy}), "
                f"置信度={match.score:.2f} ({state_tag} {best_tpl})，执行点击 1 次！"
            )
            self.device.tap(cx, cy)
            total_clicked += 1

        logger.info(f"✅ [自动收菜完成] 本轮共点击收取 {total_clicked}/3 类物资 (金币/圣水/黑油各最多 1 次)。")
        return total_clicked

    def run_collect_only(self) -> dict[str, object]:
        """
        独立测试【自动收菜】模块专用入口：
        1. 检查当前画面是否已处于最大全景视野，若处于放大特写则先缩至全景居中；
        2. 仅执行金币、圣水、黑油各点 1 次的自动收菜，绝不执行自动捐兵、请求援军与快速补兵！
        """
        screen = self.device.screencap()
        travel = self.travel.go("home", screen)
        if not travel["arrived_home"]:
            return {"status": "travel_failed", "travel": travel, "collected_bubbles": 0, "donated_times": 0}
        screen = self.travel.collection_frame("home")
        if screen is None:
            reason = "stopped" if self.travel.stopped() else "view_unconfirmed"
            return {"status": "travel_failed", "travel": {"reason": reason},
                    "collected_bubbles": 0, "donated_times": 0}
        collected = self.collect_resource_bubbles(screen)
        return {"collected_bubbles": collected, "donated_times": 0}

    def run_all(self) -> dict[str, int]:
        """执行完整的一轮村庄日常巡检（全自动挂机主循环调用）"""
        stats = {"collected_bubbles": 0, "donated_times": 0}
        if self.cfg.collect_resources:
            stats["collected_bubbles"] = self.collect_resource_bubbles()
        elif not self.travel.stopped():
            self.device.pinch_zoom_out()

        if self.cfg.auto_donate:
            stats["donated_times"] = self.check_and_donate_clan_troops()

        self.quick_train_and_request_cc()
        return stats

    def should_donate_for_message(self, request_text: str) -> bool:
        """根据队友请求援军的留言文本判断是否符合捐兵条件"""
        text = request_text.strip()
        for bad_word in self.cfg.donate_blacklist:
            if bad_word and bad_word in text:
                return False
        if not text:
            return True
        for good_word in self.cfg.donate_keywords:
            if good_word and good_word in text:
                return True
        return False

    def check_and_donate_clan_troops(self) -> int:
        """展开左侧部落聊天栏，寻找 [增援] 按钮并按关键词规则捐兵"""
        self.device.tap(*self.BTN_CLAN_CHAT_OPEN)
        screen = self.device.screencap()
        donate_btns = self.matcher.find_all(screen, "village/btn_donate.png", threshold=0.80, max_results=4)
        donated = 0

        for btn in donate_btns:
            msg_roi = (max(0, btn.y - 68), btn.y, 25, 480)
            req_text = self.ocr.read_text_from_roi(screen, msg_roi)
            if not self.should_donate_for_message(req_text):
                logger.info(f"跳过含有限制词的援军请求: '{req_text}'")
                continue

            self.device.tap(*btn.center)
            panel_screen = self.device.screencap()
            slot = self.matcher.find_one(panel_screen, "village/donate_troop_slot.png", threshold=0.76)
            if slot:
                for _ in range(3):
                    self.device.tap(*slot.center)
                donated += 1

        self.device.tap(*self.BTN_CLAN_CHAT_CLOSE)
        return donated

    def quick_train_and_request_cc(self) -> None:
        """打开左下角部队训练总览，一键补满队列并请求部落援军"""
        self.device.tap(*self.BTN_ARMY_OVERVIEW)
        if self.cfg.auto_request_cc:
            screen = self.device.screencap()
            req_btn = self.matcher.find_one(screen, "village/btn_request_cc.png", threshold=0.80)
            if req_btn:
                self.device.tap(*req_btn.center)
                confirm = self.matcher.find_one(self.device.screencap(), "common/btn_send_request.png", threshold=0.80)
                if confirm:
                    self.device.tap(*confirm.center)

        self.device.tap(*self.BTN_QUICK_TRAIN_TAB)
        self.device.tap(*self.BTN_QUICK_TRAIN_APPLY)
        self.device.tap(*self.BTN_CLOSE_MODAL)

    def find_builder_boat(
        self,
        screen_bgr: np.ndarray,
        threshold: float = 0.73,
    ) -> Optional[Tuple[int, int, float, str]]:
        candidate = self.travel.detector.find(screen_bgr, "home", threshold)
        if candidate is None:
            return None
        x, y = candidate.point
        return x, y, candidate.score, "红白船帆与木船船身"

    @classmethod
    def is_in_builder_base(cls, screen_bgr: np.ndarray) -> bool:
        """
        视觉判断是否已进入夜世界 (建筑大师基地 Builder Base)：
        - 实测 127.0.0.1:5561 实机夜世界地图中心 `[y:160..500, x:300..950]` 的夜间青蓝绿草地色调 (`HSV H ∈ [58, 105]`) 占比高达 ~58.1%；
        - 而主世界白天黄绿草地该色调占比仅为 ~3.6%（相差 16 倍）。
        """
        if screen_bgr is None or screen_bgr.size == 0:
            return False
        roi = screen_bgr[160:500, 300:950]
        if roi.size == 0:
            return False
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        teal_mask = (hsv[:, :, 0] >= 58) & (hsv[:, :, 0] <= 108) & (hsv[:, :, 1] >= 35)
        teal_ratio = float(np.mean(teal_mask))
        mean_v = float(np.mean(hsv[:, :, 2]))
        return (teal_ratio >= 0.25) and (mean_v < 138.0)

    def travel_to_builder_base(self) -> dict[str, object]:
        return self.travel.go("night")
