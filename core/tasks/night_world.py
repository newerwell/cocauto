"""Builder Base village routines and two-stage battle handling."""
from __future__ import annotations
import logging
from pathlib import Path
import re
import time
from typing import Callable, Dict, List, Optional, Tuple
import cv2
import numpy as np

from config import CombatConfig
from core.device import BaseDeviceController
from core.vision import TemplateMatcher, OcrScanner, MatchResult, RedZoneDetector
from core.vision.deployment_bar import DeploymentBar, DeploymentBarReader, card_at
from core.vision.red_zone_detector import is_battle_frame
from .village_routine import VillageRoutineTask
from .combat_ai import CombatAiTask, battle_stall_seconds, interrupted_deployment
from .deployment import BattleDeployment, DeploymentEvidence, battlefield_view, tactical_hints, view_difference

logger = logging.getLogger("coc.tasks.night")
EntryScene = tuple[str, np.ndarray | None]


class NightWorldTask:
    """夜世界（建筑大师基地）自动收菜、双向渡船与 2.0 智能战斗引擎"""

    # 夜世界主地图气泡搜索安全区 ROI (y1, y2, x1, x2)
    NIGHT_MAP_ROI: Tuple[int, int, int, int] = (55, 560, 140, 1040)

    # 127.0.0.1:5561 实机 1280x720 校准坐标：
    BTN_NIGHT_ATTACK = (70, 642)       # 夜世界左下角双斧 [战斗] 按钮 (X=18..120, Y=585..695)
    BTN_FIND_NOW = (950, 460)          # 夜世界“开始进攻”弹窗右侧亮绿色 [立即寻找！] 按钮 (X=824..1076, Y=415..505)

    # 四侧折线用于决定当前地形上的搜索范围。
    NIGHT_OUTER_POLYLINES: Dict[str, List[Tuple[int, int]]] = {
        "NW": [(82, 328), (180, 252), (278, 176), (376, 100), (475, 28)],
        "NE": [(725, 42), (825, 118), (925, 195), (1025, 272), (1125, 348)],
        "SE": [(1120, 375), (1025, 438), (930, 502), (845, 558), (785, 572)],
        "SW": [(82, 348), (115, 375), (150, 405), (185, 435)],
    }

    # 夜世界下兵前将视角转移到下兵一侧的定向滑动坐标对: side -> (swipe_start, swipe_end)
    NIGHT_CAMERA_FOCUS_SWIPES: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {
        "NW": ((410, 225), (760, 415)),  # 将夜世界视角移向左上侧 (NW 悬崖外圈)
        "NE": ((870, 225), (520, 415)),  # 将夜世界视角移向右上侧 (NE 松林外圈)
        "SE": ((870, 435), (520, 235)),  # 将夜世界视角移向右下侧 (SE 松林外圈)
        "SW": ((410, 435), (760, 235)),  # 将夜世界视角移向左下侧 (SW 森林外圈)
    }

    # 夜世界两类核心物资气泡模板组 (每种物资严格仅点 1 次)
    NIGHT_BUBBLE_GROUPS: List[Tuple[str, str, List[str]]] = [
        (
            "建筑大师金币 (Builder Gold)",
            "🪙",
            ["night/bubble_gold.png", "night/bubble_gold_full.png", "village/bubble_gold_full.png"],
        ),
        (
            "建筑大师圣水 (Builder Elixir)",
            "💧",
            ["night/bubble_elixir.png", "night/bubble_elixir_full.png", "village/bubble_elixir.png", "village/bubble_elixir_full.png"],
        ),
    ]

    def __init__(
        self,
        device: BaseDeviceController,
        matcher: TemplateMatcher,
        ocr: OcrScanner,
        village_task: VillageRoutineTask,
        combat_cfg: CombatConfig,
        *,
        stopped: Callable[[], bool] = lambda: False,
        now: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] = time.sleep,
        evidence_dir: Path | None = None,
    ):
        self.device = device
        self.matcher = matcher
        self.ocr = ocr
        self.village_task = village_task
        self.cfg = combat_cfg
        self.stopped, self.now, self.wait = stopped, now, wait
        self.red_detector = RedZoneDetector()
        self.evidence_dir = evidence_dir
        self.last_travel: dict[str, object] | None = None
        self.last_entry_frame: np.ndarray | None = None
        self.entry_reason: str | None = None

    def _find_now(self, frame: np.ndarray) -> MatchResult | None:
        return self.matcher.find_one(frame, "night/btn_find_now.png", threshold=0.75,
                                     roi=(350, 550, 760, 1150), scales=(1.0,))

    def observe_entry_scene(self, frame: np.ndarray | None = None, *,
                            home_battle: Callable[[np.ndarray], bool] | None = None) -> EntryScene:
        """Wait through a short return/loading animation, dispatching only confirmed scenes."""
        if frame is None and not self.stopped():
            frame = self.device.screencap()
        scene = "unknown"
        return_tapped = False
        reward_skipped = False
        reward_confirmed = False
        star_bonus_claimed = False
        reward_continued = False
        previous_seals = None
        for attempt in range(18):
            if self.stopped():
                return "stopped", frame
            if attempt:
                self.wait(0.45)
                if self.stopped():
                    return "stopped", frame
                frame = self.device.screencap()
            if not is_battle_frame(frame):
                scene = "unknown"
                continue
            if self.is_on_night_battlefield(frame):
                return "night_battle", frame
            if self._find_now(frame):
                return "night_match", frame
            if CombatAiTask.check_and_dismiss_star_bonus_popup(self.device, screen_bgr=frame):
                continue
            if self.matcher.find_one(frame, "battle/return_star_bonus.png", threshold=0.89,
                                     roi=(110, 205, 425, 860), scales=(1.0,)):
                scene = "loading"
                if not star_bonus_claimed and not self.stopped():
                    logger.info("领取主世界胜利之星奖励，等待返回村庄")
                    self.device.tap(640, 565, humanize=False)
                    star_bonus_claimed = True
                continue
            scene = self.village_task.travel.detector.scene(frame)
            if scene in {"home", "night", "capital"}:
                return scene, frame
            if CombatAiTask.has_battle_controls(frame) or (home_battle and home_battle(frame)):
                return "home_battle", frame
            if self.matcher.find_one(frame, "battle/return_reward_continue.png", threshold=0.90,
                                     roi=(502, 680, 455, 800), scales=(1.0,)):
                scene = "loading"
                if not reward_continued and not self.stopped():
                    logger.info("战后开箱奖励已到账，点击继续")
                    self.device.tap(628, 618, humanize=False)
                    reward_continued = True
                continue
            target = CombatAiTask.result_return_target(frame)
            if target:
                scene = "battle_result"
                if not return_tapped and not self.stopped():
                    logger.info("战斗结算页已确认，点击回营并等候村庄 HUD")
                    self.device.tap(*target, humanize=False)
                    return_tapped = True
                continue
            if self.matcher.find_one(frame, "battle/return_reward_confirm.png", threshold=0.88,
                                     roi=(175, 265, 510, 775), scales=(1.0,)):
                scene = "loading"
                if not reward_confirmed and not self.stopped():
                    logger.info("确认跳过战后开箱动画，等待返回村庄")
                    self.device.tap(778, 462, humanize=False)
                    reward_confirmed = True
                continue
            if self.matcher.find_one(frame, "battle/return_reward_chest.png", threshold=0.90,
                                     roi=(247, 411, 885, 1270), scales=(1.0,)):
                scene = "loading"
                skip = self.matcher.find_one(frame, "battle/return_reward_skip.png", threshold=0.88,
                                             roi=(620, 686, 1145, 1245), scales=(1.0,))
                if skip and not reward_skipped and not self.stopped():
                    logger.info("识别到战后奖励画面的跳过按钮，等待返回村庄")
                    self.device.tap(*skip.center, humanize=False)
                    reward_skipped = True
                elif not reward_skipped and not self.stopped():
                    seals = cv2.cvtColor(frame[575:632, 515:750], cv2.COLOR_BGR2HSV)
                    gold = cv2.countNonZero(cv2.inRange(seals, (15, 95, 142), (35, 255, 255)))
                    if gold >= 150 and (previous_seals is None or previous_seals - gold >= 200):
                        logger.info("战后奖励还显示待开启标记，点击开箱")
                        self.device.tap(625, 320, humanize=False)
                        previous_seals = gold
        return scene, frame

    def _travel_to(self, target: str, entry: EntryScene | None = None) -> dict[str, object]:
        scene, frame = self.observe_entry_scene() if entry is None else entry
        if not self.stopped() and scene in {"home", "night"}:
            self.last_travel = self.village_task.travel.go(target, frame)
            if (not self.stopped() and self.last_travel.get("reason") in {"loading", "unknown_scene"}
                    and not self.last_travel.get("touches")):
                logger.info("渡船前村庄画面发生变化，重新确认场景并处理战后奖励")
                scene, frame = self.observe_entry_scene()
                if not self.stopped() and scene in {"home", "night"}:
                    self.last_travel = self.village_task.travel.go(target, frame)
        else:
            reason = {"capital": "wrong_world", "loading": "loading", "stopped": "stopped"}.get(
                scene, "unknown_scene")
            self.last_travel = {"source": scene, "target": target, "scene": scene, "reason": reason,
                                "arrived_home": False, "arrived_night_world": False, "touches": [],
                                "boat_pt": None, "score": 0.0, "evidence": None}
        return self.last_travel

    def ensure_in_night_world(self) -> bool:
        return bool(self._travel_to("night")["arrived_night_world"])

    def travel_to_home_village(self, entry: EntryScene | None = None) -> dict[str, object]:
        return self._travel_to("home", entry)

    def find_best_night_bubbles(
        self,
        screen: np.ndarray,
        threshold: float = 0.72,
    ) -> List[Tuple[str, str, str, Optional[MatchResult]]]:
        """在夜世界安全区 NIGHT_MAP_ROI 内，为【建筑大师金币】与【建筑大师圣水】各找出置信度最高的 1 个气泡"""
        results: List[Tuple[str, str, str, Optional[MatchResult]]] = []
        for res_label, emoji, tpl_list in self.NIGHT_BUBBLE_GROUPS:
            best_match: Optional[MatchResult] = None
            best_tpl: str = ""
            for tpl_path in tpl_list:
                m = self.matcher.find_one(
                    screen,
                    tpl_path,
                    threshold=threshold,
                    roi=self.NIGHT_MAP_ROI,
                    scales=(1.0,),
                )
                if m is not None and (best_match is None or m.score > best_match.score):
                    best_match = m
                    best_tpl = tpl_path
            results.append((res_label, emoji, best_tpl, best_match))
        return results

    def collect_night_resources(self, entry: EntryScene | None = None) -> dict[str, object]:
        """
        夜世界自动收菜（建筑大师金币、建筑大师圣水每种物资严格仅点 1 次）：
        1. 若当前在主世界，先自动乘船前往夜世界；
        2. 确认夜世界处于全景视野；
        3. 对夜世界方形金币气泡与圣水气泡各取置信度最高的 1 个点击 1 次！
        """
        if not self._travel_to("night", entry)["arrived_night_world"]:
            return {"status": "travel_failed", "night_collected_bubbles": 0, "travel": self.last_travel}
        screen = self.village_task.travel.collection_frame("night")
        if screen is None:
            reason = "stopped" if self.stopped() else "view_unconfirmed"
            return {"status": "travel_failed", "night_collected_bubbles": 0,
                    "travel": {"reason": reason}}
        best_bubbles = self.find_best_night_bubbles(screen, threshold=0.72)
        total_clicked = 0

        logger.info("========== [夜世界自动收菜] 开始扫描夜世界金币与圣水气泡 (每种物资各仅点 1 次) ==========")
        for idx, (res_label, emoji, best_tpl, match) in enumerate(best_bubbles, start=1):
            if match is None:
                logger.info(f"ℹ️ [夜世界收菜 {idx}/2 - {res_label}] 当前未检测到可收集气泡，跳过。")
                continue
            if self.stopped():
                break
            cx, cy = match.center
            state_tag = "满仓红底" if "_full" in best_tpl else "正常白底"
            logger.info(
                f"{emoji} [夜世界收菜 {idx}/2 - {res_label}] 识别到最佳气泡 -> 坐标=({cx}, {cy}), "
                f"置信度={match.score:.2f} ({state_tag} {best_tpl})，执行点击 1 次！"
            )
            self.device.tap(cx, cy)
            total_clicked += 1

        logger.info(f"✅ [夜世界自动收菜完成] 本轮共点击收取 {total_clicked}/2 类夜世界物资！")
        return {"night_collected_bubbles": total_clicked, "travel": self.last_travel}

    @classmethod
    def detect_active_night_slots(cls, frame: np.ndarray) -> tuple[int | None, list[int]]:
        bar = DeploymentBarReader(OcrScanner(), "night").scan(frame)
        if not bar.supported:
            return None, []
        hero = next((card.x for card in bar.cards.values() if card.kind == "hero"), None)
        return hero, [card.x for card in bar.cards.values() if card.kind == "troop"]

    @classmethod
    def has_undeployed_night_troops(cls, frame: np.ndarray, slot_x: int) -> bool | None:
        bar = DeploymentBarReader(OcrScanner(), "night").scan(frame)
        card = card_at(bar, slot_x) if bar.supported else None
        return None if card is None or card.remaining is None else card.remaining > 0

    def _choose_best_night_side(self, frame: np.ndarray) -> list[str]:
        order = ["NE", "NW", "SE", "SW"]
        if not is_battle_frame(frame):
            return order
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, (56, 28, 65), (102, 194, 180))
        scores = {}
        for side, polyline in self.NIGHT_OUTER_POLYLINES.items():
            points = self.red_detector._sample_polyline(polyline, 12)
            scores[side] = sum(cv2.mean(grass[y-8:y+9, x-8:x+9])[0] for x, y in points)
        return sorted(order, key=scores.get, reverse=True)

    @staticmethod
    def _night_emblem(frame: np.ndarray) -> bool:
        trophy = cv2.cvtColor(frame[25:63, 20:125], cv2.COLOR_BGR2HSV)
        badge = cv2.inRange(trophy, (34, 50, 96), (105, 255, 255))
        return bool(np.mean(badge > 0) >= 0.26)

    @staticmethod
    def _night_scout_hud(frame: np.ndarray) -> bool:
        report = cv2.cvtColor(frame[6:59, 323:486], cv2.COLOR_BGR2HSV)
        orange = cv2.inRange(report, (3, 100, 99), (30, 255, 255))
        _, _, boxes, _ = cv2.connectedComponentsWithStats(orange)
        if not any(w >= 58 and h >= 26 and area >= 1500 for _, _, w, h, area in boxes[1:]):
            return False
        badges = cv2.cvtColor(frame[592:619, 198:795], cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(badges, (95, 95, 128), (128, 255, 255))
        return bool(np.mean(blue > 0) >= 0.24)

    @classmethod
    def is_night_opening_countdown(cls, frame: np.ndarray) -> bool:
        if not (is_battle_frame(frame) and cls._night_emblem(frame) and cls._night_scout_hud(frame)):
            return False
        timer = cv2.cvtColor(frame[30:80, 555:720], cv2.COLOR_BGR2HSV)
        white = cv2.inRange(timer, (0, 0, 195), (180, 65, 255))
        _, _, boxes, _ = cv2.connectedComponentsWithStats(white)
        return sum(8 <= w <= 31 and 19 <= h <= 31 and area >= 55 for _, _, w, h, area in boxes[1:]) >= 2

    def is_on_night_battlefield(self, frame: np.ndarray) -> bool:
        return (is_battle_frame(frame) and self._night_emblem(frame)
                and (self._night_end_button(frame) or self._night_scout_hud(frame)))

    @staticmethod
    def _night_end_button(frame: np.ndarray) -> bool:
        if not is_battle_frame(frame):
            return False
        end_button = cv2.cvtColor(frame[484:517, 35:148], cv2.COLOR_BGR2HSV)
        red = cv2.inRange(end_button, (0, 100, 135), (13, 255, 255))
        red |= cv2.inRange(end_button, (166, 100, 135), (180, 255, 255))
        return bool(np.mean(red > 0) >= 0.48)

    def _battle_end_target(self, frame: np.ndarray) -> tuple[int, int] | None:
        if self.is_on_night_battlefield(frame) and self._night_end_button(frame):
            return 90, 500
        return None

    def enter_night_matchmaking(self, entry: EntryScene | None = None) -> bool:
        """
        从主世界/夜世界大本营/开始进攻弹窗智能进入夜世界战场：
        1. 若当前已在夜世界战场，直接返回 True；
        2. 若尚未在夜世界，先乘船前往夜世界；
        3. 点击夜世界左下角双斧 [战斗] (70, 642)；
        4. 在“开始进攻”弹窗点击右侧亮绿色 [立即寻找！] (950, 460)；
        5. 等待搜索云层散去进入夜世界战场。
        """
        self.last_travel = None
        self.entry_reason = None
        scene, screen = self.observe_entry_scene() if entry is None else entry
        self.last_entry_frame = screen
        if scene == "night_battle" and not self.stopped():
            logger.info("👁️ [夜世界进场] 检测到当前已处于夜世界敌方战场，直接开始战术部署！")
            return True

        if self.stopped():
            self.entry_reason = "stopped"
            return False
        if scene in {"home", "night"}:
            if not self._travel_to("night", (scene, screen))["arrived_night_world"] or self.stopped():
                self.entry_reason = "travel_failed"
                return False
            logger.info(f"⚔️ [夜世界进场 Step 1/3] 点击夜世界左下角双斧 [战斗] 按钮 {self.BTN_NIGHT_ATTACK}...")
            self.device.tap(*self.BTN_NIGHT_ATTACK)
            scene = "night"
            for _ in range(8):
                self.wait(0.40)
                if self.stopped():
                    return False
                screen = self.device.screencap()
                self.last_entry_frame = screen
                if self.is_on_night_battlefield(screen):
                    return True
                if is_battle_frame(screen) and self._find_now(screen):
                    scene = "night_match"
                    break
        if scene != "night_match":
            self.entry_reason = "match_dialog_missing" if scene == "night" else f"entry_scene_{scene}"
            return False
        find_now_m = self._find_now(screen)
        if find_now_m is None:
            self.entry_reason = "match_dialog_missing"
            return False
        find_pt = find_now_m.center
        logger.info(f"🚀 [夜世界进场 Step 2/3] 在开始进攻弹窗点击亮绿色 [立即寻找！] 按钮 {find_pt}...")
        if self.stopped():
            return False
        self.device.tap(*find_pt)

        logger.info("☁️ [夜世界进场 Step 3/3] 正在等待夜世界匹配白云散去...")
        deadline = self.now() + 60.0
        confirmed_frames = 0
        while not self.stopped() and self.now() < deadline:
            scr = self.device.screencap()
            self.last_entry_frame = scr
            confirmed_frames = confirmed_frames + 1 if self.is_night_opening_countdown(scr) else 0
            if confirmed_frames >= 2 and not self.stopped():
                logger.info("夜世界开战倒计时已连续两帧确认，开始部署")
                return True
            self.wait(min(0.40, max(0.0, deadline - self.now())))

        logger.warning("未检测到夜世界开战页面")
        self.entry_reason = "stopped" if self.stopped() else "matchmaking_timeout"
        return False

    def focus_camera_on_night_side(self, side_name: str) -> None:
        if not self.stopped():
            start, end = self.NIGHT_CAMERA_FOCUS_SWIPES[side_name]
            self.device.swipe(start, end, duration_ms=420)
            self.wait(0.20)

    def _deployment_hints(self, side: str, pattern: str) -> list[tuple[int, int]]:
        outer = self.red_detector._sample_polyline(self.NIGHT_OUTER_POLYLINES[side], 12)
        return tactical_hints([outer], pattern)

    def _deploy_night_wave_funnel_and_center(
        self, frame: np.ndarray, stage: int, deadline: float,
    ) -> BattleDeployment:
        phase = BattleDeployment(
            world="night", phase=stage, device=self.device, reader=DeploymentBarReader(self.ocr, "night"),
            detector=self.red_detector, side_order=self._choose_best_night_side(frame),
            focus=self.focus_camera_on_night_side, hints=self._deployment_hints,
            result_screen=CombatAiTask.is_battle_result_screen, battle_screen=self.is_on_night_battlefield,
            stopped=self.stopped, deadline=deadline, now=self.now, wait=self.wait, evidence_dir=self.evidence_dir,
        )
        if stage == 1 and self.is_night_opening_countdown(frame):
            phase.evidence.record({"world": "night", "phase": stage, "reason": "opening_countdown", "diagnostic": True},
                                  frame, None, [])
        try:
            phase.run()
        except InterruptedError:
            phase.halt_reason = "stopped"
        return phase

    def execute_night_battle(self, entry: EntryScene | None = None) -> dict:
        self.last_entry_frame = None
        self.entry_reason = None
        try:
            if not self.enter_night_matchmaking(entry):
                return self.entry_failure(self.entry_reason, self.last_entry_frame)
            deadline = self.now() + self.cfg.max_battle_duration_sec
            frame = self.last_entry_frame
            phase = self._deploy_night_wave_funnel_and_center(frame, 1, deadline)
            activations, phases = self._monitor_night_battle(phase)
            return {"skill_activations": activations, "deployment": phases}
        except InterruptedError:
            return self.entry_failure("stopped", self.last_entry_frame)

    def entry_failure(self, reason: str | None, frame: np.ndarray | None) -> dict:
        stopped = self.stopped() or reason == "stopped"
        phase = {**interrupted_deployment(), "status": "interrupted" if stopped else "unconfirmed",
                 "reason": "stopped" if stopped else "battle_not_entered",
                 "entry_reason": "stopped" if stopped else reason, "entered_battle": False, "touch_count": 0}
        evidence = DeploymentEvidence(self.evidence_dir, "night", 1)
        phase["evidence"] = str(evidence.folder) if evidence.folder else None
        evidence.record({**phase, "world": "night", "diagnostic": True}, frame, None, [])
        logger.warning("夜世界进场未下兵 reason=%s evidence=%s", phase["entry_reason"], phase["evidence"])
        return {"skill_activations": 0, "deployment": [phase]}

    def _monitor_night_battle(self, phase: BattleDeployment) -> tuple[int, list[dict]]:
        completed_phases = []
        transition = NightStageTransition(self.ocr, phase.last_frame, phase.reader)
        if phase.halt_reason == "battlefield_changed":
            transition.gap = 2
        last_progress = last_skill = self.now()
        last_observed = None
        skill_count = 0
        stall = battle_stall_seconds(self.cfg)
        frame = phase.last_capture
        reason = phase.halt_reason
        try:
            while reason not in {"natural_end", "stopped", "timeout"} and self.now() < phase.deadline:
                if self.stopped():
                    reason = "stopped"
                    break
                frame = self.device.screencap()
                if CombatAiTask.is_battle_result_screen(frame):
                    reason = "natural_end"
                    break
                in_battle = self.is_on_night_battlefield(frame)
                rate = self.ocr.scan_destruction_rate(frame) if in_battle else None
                if phase.phase == 1:
                    phase_state = transition.observe(frame, in_battle, rate)
                    if phase_state == "second_stage":
                        completed_phases.append(phase.summary("stage_advanced"))
                        phase = self._deploy_night_wave_funnel_and_center(
                            frame, 2, self.now() + self.cfg.max_battle_duration_sec
                        )
                        frame = phase.last_capture
                        reason = phase.halt_reason
                        last_progress = last_skill = self.now()
                        last_observed = None
                        continue
                    if phase_state == "transition":
                        last_progress = self.now()
                        self.wait(0.35)
                        continue
                if not in_battle:
                    self.wait(0.35)
                    continue

                observed = rate if rate is not None else self.ocr.get_destruction_glyph_signature(frame)
                if observed != last_observed:
                    last_progress, last_observed = self.now(), observed
                elif self.now() - last_progress >= stall:
                    reason = "stagnation"
                    break
                if self.now() - last_skill >= 4.5:
                    for card in phase.skills(frame):
                        if self.stopped() or self.now() >= phase.deadline:
                            break
                        self.device.tap(card.x, card.y, humanize=False)
                        skill_count += 1
                    last_skill = self.now()
                self.wait(0.65)
            reason = reason if reason in {"natural_end", "stopped", "stagnation"} else "timeout"
            if reason != "stopped" and not self.stopped() and not (phase.phase == 1 and transition.pending):
                CombatAiTask.finish_battle(self.device, frame, self.wait, self.stopped, end_target=self._battle_end_target)
            elif reason == "natural_end":
                CombatAiTask.finish_battle(self.device, frame, self.wait, self.stopped, end_target=self._battle_end_target)
        except InterruptedError:
            reason = "stopped"
        completed_phases.append(phase.summary(reason))
        return skill_count, completed_phases


class NightStageTransition:
    """Require a new aligned battlefield and a newly playable deployment bar."""

    def __init__(self, ocr: OcrScanner, frame: np.ndarray | None, reader: DeploymentBarReader):
        self.ocr, self.reader = ocr, reader
        self.reference = battlefield_view(frame) if is_battle_frame(frame) else None
        self.lineup = self._lineup(reader.scan(frame))
        self.high_rate = 0
        self.gap = 0
        self.pending = False
        self.candidate = None

    @staticmethod
    def _lineup(bar: DeploymentBar) -> tuple:
        return tuple((x, card.kind, card.remaining) for x, card in bar.cards.items())

    def observe(self, frame: np.ndarray | None, in_battle: bool, rate: int | None) -> str:
        bar = self.reader.scan(frame) if in_battle else DeploymentBar({}, False)
        if not bar.supported:
            self.gap += 1
            self.candidate = None
            return "transition"
        view = battlefield_view(frame)
        lineup = self._lineup(bar)
        ready = any(card.remaining and card.remaining > 0 for card in bar.cards.values())
        if rate == 100:
            self.pending = True
        if self.reference is None:
            self.reference, self.lineup = view, lineup
        changed = view_difference(self.reference, view) > 16
        reset = rate is not None and rate <= 25 and self.high_rate >= 80
        if changed and ready:
            text = self.ocr.read_text_from_roi(frame, (20, 90, 485, 795)).casefold().replace(" ", "")
            marker = bool(re.search(r"(?:stage(?:2|ii)(?:/2)?|第[2二](?:阶段|关))", text))
            evidence = marker or (self.pending and (self.gap or reset)) or (self.gap >= 2 and lineup != self.lineup) or reset
            if evidence:
                self.pending = True
                if self.candidate is not None and view_difference(self.candidate, view) <= 8:
                    self.pending = False
                    return "second_stage"
                self.candidate = view
                return "transition"
        self.candidate = None
        if self.pending or (changed and self.gap >= 2):
            return "transition"
        self.gap = 0
        self.reference, self.lineup = view, lineup
        self.high_rate = max(self.high_rate, rate or 0)
        return "battle"
