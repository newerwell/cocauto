"""Home Village battle entry point, tactics and monitoring."""
from __future__ import annotations
import logging
from pathlib import Path
import random
import time
from typing import Callable, List, Tuple, Optional
import cv2
import numpy as np

from config import CombatConfig
from core.device import BaseDeviceController
from core.vision import RedZoneDetector, DeadBaseChecker, OcrScanner

from core.vision.deployment_bar import DeploymentBarReader, card_at
from core.vision.red_zone_detector import is_battle_frame
from .deployment import BattleDeployment, tactical_hints

logger = logging.getLogger("coc.tasks.combat")


class CombatAiTask:
    """基于真人感开局侦察、下兵侧视角转移、先清边后中路总攻、摧毁率停滞放弃与结算自动回营的智能战斗引擎"""

    # 基于用户实机战斗截图 (1280x720) 像素级精确标定的核心按钮中心坐标：
    BTN_END_BATTLE = (90, 538)         # 战场左下角红色 [放弃] / [结束战斗] (X=18..162, Y=508..568)
    BTN_CONFIRM_SURRENDER = (778, 462) # "放弃？"确认弹窗右侧绿色 [确定] (X=660..895, Y=415..505)
    BTN_RETURN_HOME = (640, 618)       # 战斗结束结算页面底部中央绿色 [回营] (X=560..720, Y=582..656)
    BTN_STAR_BONUS_OK = (640, 562)     # 回营后弹出的“胜利之星奖励！”弹窗底部中央绿色 [确定] (X=545..735, Y=520..605)
    BTN_DEFENSE_REPLAY_HOME = (85, 650)# 村庄被进攻实时回放结束页左下角村民头像 [回营]
    BTN_RELOAD_GAME = (330, 460)       # “还在吗？因为太久没有进行操作”断线弹窗左下 [重新载入游戏]

    def __init__(
        self,
        device: BaseDeviceController,
        red_detector: RedZoneDetector,
        dead_checker: DeadBaseChecker,
        ocr: OcrScanner,
        cfg: CombatConfig,
        *,
        stopped: Callable[[], bool] = lambda: False,
        now: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] = time.sleep,
        evidence_dir: Path | None = None,
    ):
        self.device = device
        self.red_detector = red_detector
        self.dead_checker = dead_checker
        self.ocr = ocr
        self.cfg = cfg

        self.stopped, self.now, self.wait = stopped, now, wait
        self.evidence_dir = evidence_dir

    @classmethod
    def has_battle_controls(cls, frame: np.ndarray) -> bool:
        if not is_battle_frame(frame):
            return False
        hsv = cv2.cvtColor(frame[519:555, 35:149], cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 110, 145), (12, 255, 255))
        red |= cv2.inRange(hsv, (165, 110, 145), (180, 255, 255))
        return float(np.mean(red > 0)) >= 0.40

    @classmethod
    def detect_active_battle_slots(cls, frame: np.ndarray) -> list[int]:
        bar = DeploymentBarReader(OcrScanner(), "home").scan(frame)
        return list(bar.cards) if bar.supported else []

    @classmethod
    def is_slot_depleted(cls, frame: np.ndarray, slot_x: int) -> bool | None:
        card = card_at(DeploymentBarReader(OcrScanner(), "home").scan(frame), slot_x)
        return None if card is None or card.remaining is None else card.remaining == 0

    @classmethod
    def is_hero_or_siege_deployed(
        cls, before: np.ndarray | None, after: np.ndarray | None, slot_x: int,
    ) -> bool | None:
        reader = DeploymentBarReader(OcrScanner(), "home")
        initial, observed = card_at(reader.scan(before), slot_x), card_at(reader.scan(after), slot_x)
        if initial is None or observed is None or observed.remaining is None:
            return None
        return initial.kind in {"hero", "siege"} and observed.remaining == 0

    @classmethod
    def is_battle_result_screen(cls, screen_bgr: np.ndarray) -> bool:
        """
        检测当前画面是否已进入战斗结束结算页 (胜利/战败，底部中央出现高亮草绿色 [回营] 按钮)：
        - 结算页底部中央 `[y:585..650, x:560..720]` 为草绿色 [回营] 按钮 (`H∈[32..68], S>100, V>140` 占比 >= 50%)；
        - 同时左下角 `[y:512..562, x:25..155]` 的红色 [放弃] 按钮已消失 (`red_ratio < 15%`)。
        """
        return cls.result_return_target(screen_bgr) is not None

    @classmethod
    def result_return_target(cls, frame: np.ndarray | None) -> tuple[int, int] | None:
        if not is_battle_frame(frame) or cls.has_battle_controls(frame):
            return None
        lower = cv2.cvtColor(frame[495:650, 355:740], cv2.COLOR_BGR2HSV)
        green = cv2.inRange(lower, (32, 100, 140), (68, 255, 255)) > 0
        if np.mean(green[90:, 205:365]) >= 0.50:
            return cls.BTN_RETURN_HOME
        retry = cv2.inRange(lower[6:75, 14:173], (113, 65, 135), (151, 255, 255)) > 0
        if np.mean(green[6:75, 214:371]) >= 0.48 and np.mean(retry) >= 0.42:
            return (646, 537)
        return None

    @classmethod
    def is_surrender_confirm_popup_open(cls, screen_bgr: np.ndarray) -> bool:
        """
        检测当前画面是否弹出了“放弃？”二次确认弹窗：
        - 弹窗右侧 `[y:415..505, x:660..895]` 为亮绿色 [确定] 按钮 (`green_ratio >= 50%`)；
        - 弹窗左侧 `[y:415..505, x:380..615]` 为亮橙色 [取消] 按钮 (`orange_ratio >= 45%`)。
        """
        if screen_bgr is None or screen_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
        btn_ok = hsv[415:505, 660:895]
        btn_cancel = hsv[415:505, 380:615]
        if btn_ok.size == 0 or btn_cancel.size == 0:
            return False
        green_mask = cv2.inRange(btn_ok, np.array([32, 100, 140]), np.array([68, 255, 255]))
        orange_mask = cv2.inRange(btn_cancel, np.array([6, 100, 140]), np.array([25, 255, 255]))
        return float(np.mean(green_mask > 0)) >= 0.50 and float(np.mean(orange_mask > 0)) >= 0.45

    @classmethod
    def is_star_bonus_popup_open(cls, screen_bgr: np.ndarray) -> bool:
        """
        检测当前画面是否弹出了【胜利之星奖励！】弹窗 (见 night_star_bonus_1280x720_grid.png 实机校准)：
        1. 弹窗底部中央 `[y:525..600, x:555..725]` 为草绿色 [确定] 按钮 (`H∈[32, 68], S>100, V>140` 占比 >= 48%，实测 69.7%)；
        2. 弹窗主体 `[y:140..500, x:280..1000]` 为星空皇家蓝背景 (`H∈[96, 126], S>105, V>115` 占比 >= 48%，实测 86.6%)。
        """
        if screen_bgr is None or screen_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
        btn_ok = hsv[525:600, 555:725]
        dialog_bg = hsv[140:500, 280:1000]
        if btn_ok.size == 0 or dialog_bg.size == 0:
            return False
        green_mask = cv2.inRange(btn_ok, np.array([32, 100, 140]), np.array([68, 255, 255]))
        blue_mask = cv2.inRange(dialog_bg, np.array([96, 105, 115]), np.array([126, 255, 255]))
        return float(np.mean(green_mask > 0)) >= 0.48 and float(np.mean(blue_mask > 0)) >= 0.48

    @classmethod
    def is_reload_game_popup_open(cls, screen_bgr: np.ndarray) -> bool:
        """
        检测当前画面是否弹出了“还在吗？因为太久没有进行操作，您已断开连接 -> 重新载入游戏”深灰黑弹窗：
        - 弹窗主体 `[y:200..510, x:260..1010]` 为深炭灰纯色背景 (`mean_S < 18.0`, `mean_V ∈ [20.0, 48.0]`, 标准差 `std_V < 38.0`)；
        - 同时左下 `[y:435..485, x:235..430]` 存在“重新载入游戏”浅蓝白文字 (`white_px >= 90`)。
        """
        if screen_bgr is None or screen_bgr.size == 0:
            return False
        hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
        box = hsv[200:510, 260:1010]
        txt_roi = hsv[435:485, 235:430]
        if box.size == 0 or txt_roi.size == 0:
            return False
        mean_s = float(np.mean(box[:, :, 1]))
        mean_v = float(np.mean(box[:, :, 2]))
        std_v = float(np.std(box[:, :, 2]))
        white_px = int(np.sum((txt_roi[:, :, 1] < 55) & (txt_roi[:, :, 2] > 180)))
        return (mean_s < 18.0) and (20.0 <= mean_v <= 48.0) and (std_v < 38.0) and (white_px >= 90)

    @classmethod
    def check_and_dismiss_star_bonus_popup(
        cls,
        device: BaseDeviceController,
        screen_bgr: Optional[np.ndarray] = None,
        is_fast_test: bool = False,
    ) -> bool:
        """若当前画面存在【胜利之星奖励！】弹窗或【还在吗？重新载入游戏】弹窗，自动点击按钮恢复"""
        if is_fast_test and screen_bgr is None:
            return False
        scr = screen_bgr if screen_bgr is not None else device.screencap()
        if cls.is_star_bonus_popup_open(scr):
            logger.info(
                f"🌟 [胜利之星奖励!] 视觉识别到『胜利之星奖励！』弹窗，立即点击绿色 [确定] 按钮 {cls.BTN_STAR_BONUS_OK} 领取奖励！"
            )
            device.tap(*cls.BTN_STAR_BONUS_OK)
            time.sleep(0.1 if is_fast_test else 1.1)
            return True
        if cls.is_reload_game_popup_open(scr):
            logger.warning(
                f"🔄 [断线自动重连] 检测到『还在吗？重新载入游戏』弹窗，立即点击 {cls.BTN_RELOAD_GAME} 重新载入游戏！"
            )
            device.tap(*cls.BTN_RELOAD_GAME)
            time.sleep(0.1 if is_fast_test else 5.5)
            return True
        return False

    def focus_camera_on_deploy_side(self, side_name: str) -> None:
        if not self.stopped():
            start, end = self.red_detector.get_camera_focus_swipe(side_name)
            self.device.swipe(start, end, duration_ms=460)
            self.wait(0.20)

    def _deployment_hints(self, side: str, pattern: str) -> list[tuple[int, int]]:
        main = self.red_detector._sample_polyline(self.red_detector.SHIFTED_OUTER_PERIMETER_POLYLINES[side], 12)
        guard = self.red_detector._sample_polyline(self.red_detector.SHIFTED_OUTER_GUARD_POLYLINES[side], 12)
        return tactical_hints([main, guard], pattern)

    def _deploy_funnel_then_center_push(self, side_order: list[str], deadline: float) -> BattleDeployment:
        phase = BattleDeployment(
            world="home", phase=1, device=self.device, reader=DeploymentBarReader(self.ocr, "home"),
            detector=self.red_detector, side_order=side_order, focus=self.focus_camera_on_deploy_side,
            hints=self._deployment_hints, result_screen=self.is_battle_result_screen,
            battle_screen=self.has_battle_controls, stopped=self.stopped, deadline=deadline,
            now=self.now, wait=self.wait, evidence_dir=self.evidence_dir,
        )
        try:
            phase.run()
        except InterruptedError:
            phase.halt_reason = "stopped"
        return phase

    def _deploy_line_air_spam(self, deadline: float) -> BattleDeployment:
        sides = ["NW", "NE", "SE", "SW"]
        chosen = random.choice(sides)
        return self._deploy_funnel_then_center_push([chosen, *[side for side in sides if side != chosen]], deadline)

    def _deploy_surround_barch(self, deadline: float) -> BattleDeployment:
        sides = ["NW", "NE", "SE", "SW"]
        random.shuffle(sides)
        return self._deploy_funnel_then_center_push(sides, deadline)

    def execute_attack(self) -> dict:
        deadline = self.now() + self.cfg.max_battle_duration_sec
        if self.stopped():
            return {"hero_abilities_used": 0, "deployment": [interrupted_deployment()]}
        if self.cfg.strategy == "line_air_spam":
            phase = self._deploy_line_air_spam(deadline)
        else:
            phase = self._deploy_surround_barch(deadline)
        abilities, reason = self._monitor_battle_and_hero_abilities(phase)
        return {"hero_abilities_used": abilities, "deployment": [phase.summary(reason)]}

    def _monitor_battle_and_hero_abilities(self, phase: BattleDeployment) -> tuple[int, str]:
        last_progress = self.now()
        last_observed = None
        triggered: set[int] = set()
        recovered = False
        lost_at = None
        stall = battle_stall_seconds(self.cfg)
        try:
            while phase.running():
                frame = phase.capture()
                if frame is None:
                    if phase.halt_reason:
                        break
                    lost_at = self.now() if lost_at is None else lost_at
                    if self.now() - lost_at >= min(6.0, stall):
                        phase.halt_reason = "battle_unrecognized"
                        break
                    self.wait(0.25)
                    continue
                lost_at = None
                for card in phase.skills(frame):
                    if card.x in triggered or not phase.running():
                        continue
                    hp = self.dead_checker.check_hero_hp_ratio(frame, (card.x - 43, 571, card.x + 43, 653))
                    if hp < self.cfg.hero_ability_hp_ratio:
                        self.device.tap(card.x, card.y, humanize=False)
                        triggered.add(card.x)
                rate = self.ocr.scan_destruction_rate(frame)
                loot = self.ocr.scan_enemy_loot(frame)
                observed = (rate if rate is not None else self.ocr.get_destruction_glyph_signature(frame),
                            loot.gold, loot.elixir, loot.dark_elixir)
                if observed != last_observed:
                    last_progress, last_observed = self.now(), observed
                elif self.now() - last_progress >= stall:
                    if not recovered and (not phase.bar_supported or phase.summary()["remaining_slots"]):
                        recovered = True
                        before = sum(slot.deployed for slot in phase.slots.values())
                        phase.run()
                        if not phase.running():
                            break
                        if sum(slot.deployed for slot in phase.slots.values()) > before:
                            last_progress = self.now()
                            continue
                    phase.halt_reason = "stagnation"
                    break
                self.wait(0.65)
            reason = phase.halt_reason or "timeout"
            if reason != "stopped" and not self.stopped():
                self.finish_battle(self.device, phase.last_capture, self.wait, self.stopped)
            return len(triggered), reason
        except InterruptedError:
            return len(triggered), "stopped"

    @classmethod
    def finish_battle(
        cls, device: BaseDeviceController, screen: np.ndarray | None,
        wait: Callable[[float], None], stopped: Callable[[], bool],
        *, end_target: Callable[[np.ndarray], tuple[int, int] | None] | None = None,
    ) -> None:
        if stopped():
            return
        if not is_battle_frame(screen):
            screen = device.screencap()
        target = end_target(screen) if end_target is not None else (
            cls.BTN_END_BATTLE if cls.has_battle_controls(screen) else None)
        if target and not cls.is_surrender_confirm_popup_open(screen):
            device.tap(*target, humanize=False)
            wait(0.55)
            if stopped():
                return
            screen = device.screencap()
        if cls.is_surrender_confirm_popup_open(screen) and not stopped():
            device.tap(*cls.BTN_CONFIRM_SURRENDER, humanize=False)
            for _ in range(5):
                wait(0.30)
                if stopped():
                    return
                screen = device.screencap()
                if cls.is_battle_result_screen(screen):
                    break
        return_target = cls.result_return_target(screen)
        if return_target and not stopped():
            device.tap(*return_target, humanize=False)
            wait(1.45)
            if not stopped():
                cls.check_and_dismiss_star_bonus_popup(device)


def battle_stall_seconds(cfg: CombatConfig) -> float:
    if cfg.early_exit_no_loot_sec <= 0:
        return 0.0
    base = float(np.clip(cfg.early_exit_no_loot_sec, 10.0, 15.0))
    return random.uniform(max(10.0, base - 2.0), min(15.0, base + 0.5))


def interrupted_deployment() -> dict:
    return {"phase": 1, "status": "interrupted", "reason": "stopped",
            "touch_count": 0, "remaining_slots": [], "slots": [], "evidence": None}
