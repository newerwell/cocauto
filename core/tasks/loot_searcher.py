"""
自动搜鱼与新版联机模式进场状态机 (Loot Searcher Task v2.4)
已针对最新国服《部落冲突》1280x720 实机界面完成全链路校准：
1. 支持新版四步进场状态机：
   - Step 1 (主世界): 点击左下角 [战斗] (66, 650) 进入联机模式界面
   - Step 2 (联机模式界面):
     * 常规战 (regular): 点击左侧 [常规战 -> 搜索对手] (195, 540)
     * 排位战 (ranked):  点击右侧 [排位战] (535, 540)
   - Step 3 (编辑军队页面): 识别并点击右下角绿色 [进攻！] 按钮 (1129, 636)
   - Step 4 (排位战确认弹窗): 若为排位战，点击弹出的 [确认] 按钮 (640, 445) 开始搜索
2. 严格区分【常规战】与【排位战】搜鱼规则：
   - 常规战 (regular): 白云散去后通过 RapidOCR 读取左上角金/水/黑油，达标则锁定目标，不达标则点击右下角 [下一个] (1162, 505) 循环挑鱼。
   - 排位战 (ranked):  排位战不能再次搜索（搜到谁只能打谁）！一旦白云散去进入敌方战场，跳过资源阈值过滤与[下一个]，直接无条件开打！
"""
from __future__ import annotations
import logging
import re
import time
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4
import cv2
import numpy as np

from config import LootFilterConfig
from core.device import BaseDeviceController
from core.vision import TemplateMatcher, OcrScanner, DeadBaseChecker, EnemyLoot

logger = logging.getLogger("coc.tasks.search")


def home_match_failure(reason: str, frame: np.ndarray | None, evidence_dir: Path | None) -> dict:
    path = None
    if evidence_dir is not None and frame is not None and frame.size:
        try:
            folder = evidence_dir / f"home-match-{uuid4().hex[:12]}"
            folder.mkdir(parents=True, exist_ok=True)
            ok, png = cv2.imencode(".png", frame)
            if ok:
                image = folder / "last-match-frame.png"
                image.write_bytes(png.tobytes())
                path = str(image)
        except (cv2.error, OSError) as exc:
            logger.warning("主世界匹配失败画面保存失败 %s", exc)
    return {"phase": 1, "status": "interrupted" if reason == "stopped" else "unconfirmed",
            "reason": reason if reason in {"stopped", "target_not_found"} else "battle_not_entered",
            "match_reason": reason, "touch_count": 0, "remaining_slots": [], "slots": [], "evidence": path}


class LootSearcherTask:
    """新版联机模式（常规战/排位战）进场、排位赛完成状态检测、自动搜鱼与目标过滤状态机"""

    # 1280x720 实机像素级校准按钮坐标 (127.0.0.1:5561 实测标定)
    BTN_ATTACK_MAIN = (66, 650)        # Step 1: 主世界左下角 [战斗] 大按钮中心
    BTN_MODE_REGULAR = (195, 540)      # Step 2A: 联机模式界面左侧 [常规战 -> 搜索对手 1500] 橙色按钮
    BTN_MODE_RANKED = (535, 540)       # Step 2B: 联机模式界面右侧 [排位战] 按钮
    BTN_CLOSE_MODE_PANEL = (900, 360)  # 联机模式界面右侧空白区 (点击关闭联机面板回主世界)
    BTN_ARMY_ATTACK = (1129, 636)      # Step 3: 编辑军队页面右下角绿色 [⚔️ 进攻！] 按钮
    BTN_RANKED_CONFIRM = (640, 445)    # Step 4: 排位战点击进攻后的 [确认] 弹窗按钮
    BTN_NEXT_BASE = (1162, 505)        # 常规战侦察界面右下角 [🔍 下一个 1500] 按钮
    BTN_END_BATTLE = (90, 538)         # 侦察界面左下角红色 [结束战斗] 按钮

    # 联机模式面板中【常规战】与【排位战】底部按钮与完成次数 ROI (y1, y2, x1, x2)
    REGULAR_BTN_ROI = (485, 585, 55, 325)
    RANKED_BTN_ROI = (490, 575, 410, 660)
    RANKED_STATUS_ROI = (490, 585, 430, 635)
    RANKED_COUNT_ROI = (535, 585, 480, 605)

    def __init__(
        self,
        device: BaseDeviceController,
        matcher: TemplateMatcher,
        ocr: OcrScanner,
        dead_checker: DeadBaseChecker,
        cfg: LootFilterConfig,
    ):
        self.device = device
        self.matcher = matcher
        self.ocr = ocr
        self.dead_checker = dead_checker
        self.cfg = cfg
        self.last_match_reason: str | None = None
        self.last_match_frame: np.ndarray | None = None

    def evaluate_base(self, loot: EnemyLoot, is_dead: bool) -> Tuple[bool, str]:
        """
        判断当前搜到的村庄是否满足开打条件。
        注意：若当前为【排位赛 (ranked)】，排位赛不能再次搜索，搜到谁必须直接打谁！
        """
        if getattr(self.cfg, "match_mode", "regular") == "ranked":
            return True, f"⚔️ [排位赛模式] 排位赛不能再次搜索(搜到谁只能打谁)，直接锁定对手 (金={loot.gold:,}, 水={loot.elixir:,}, 黑油={loot.dark_elixir:,})"

        if self.cfg.only_dead_base and not is_dead:
            return False, f"非死鱼跳过 (金={loot.gold:,}, 水={loot.elixir:,})"

        if loot.gold >= self.cfg.min_gold and loot.elixir >= self.cfg.min_elixir:
            return True, f"金水双达标 (金={loot.gold:,}, 水={loot.elixir:,}, 死鱼={is_dead})"

        if loot.gold_plus_elixir >= self.cfg.min_gold_plus_elixir:
            return True, f"金水总和达标 (总和={loot.gold_plus_elixir:,}, 死鱼={is_dead})"

        if loot.dark_elixir >= self.cfg.min_dark_elixir:
            return True, f"暗黑重油达标 (黑油={loot.dark_elixir:,}, 死鱼={is_dead})"

        return False, f"资源未达标 (金={loot.gold:,}, 水={loot.elixir:,}, 黑油={loot.dark_elixir:,})"

    def is_on_battlefield_screen(self, screen: np.ndarray) -> bool:
        """检查当前画面是否已经处于敌方村庄侦察/战斗界面"""
        if self.matcher.find_one(screen, "battle/btn_end_battle.png", threshold=0.76, scales=(1.0,)):
            return True
        if self.matcher.find_one(screen, "battle/btn_next.png", threshold=0.76, scales=(1.0,)):
            return True
        return False

    def check_ranked_battle_status(self, screen_bgr: np.ndarray) -> Tuple[bool, str]:
        """
        检测联机模式界面右侧【排位战】是否已完成 (如“进攻已完成！ ⚔️ 12/12”)：
        1. RapidOCR 读取 `RANKED_COUNT_ROI = (535, 585, 480, 605)` 与 `RANKED_STATUS_ROI`：
           - 若识别出 `X/Y` 比例且 `X >= Y > 0` (如 `12/12`) 或包含“完成”字样，判定为已完成；
        2. 视觉按钮状态对比：
           - 当位于联机模式界面时，左侧常规战 `REGULAR_BTN_ROI` 存在高亮橙黄色 [搜索对手] 按钮 (`reg_orange >= 0.35`)；
           - 若右侧排位战底部 `RANKED_BTN_ROI = (490, 575, 410, 660)` 无橙色按钮 (`rank_orange < 0.08` 且 `mean_V < 105`)，
             说明排位战按钮已变为深灰“进攻已完成！”状态。
        返回: (is_completed: bool, reason_detail: str)
        """
        if screen_bgr is None or screen_bgr.size == 0:
            return False, "空画面"

        cnt_txt = self.ocr.read_text_from_roi(screen_bgr, self.RANKED_COUNT_ROI).strip()
        status_txt = self.ocr.read_text_from_roi(screen_bgr, self.RANKED_STATUS_ROI).strip()

        # 1. 解析 X/Y 场次比例 (例如 "#12/12" 或 "12/12")
        for raw_txt in (cnt_txt, status_txt):
            m = re.search(r"(\d+)\s*/\s*(\d+)", raw_txt)
            if m:
                done_n, total_n = int(m.group(1)), int(m.group(2))
                if total_n > 0 and done_n >= total_n:
                    return True, f"排位赛进攻已完成 ({done_n}/{total_n})"
            if "完成" in raw_txt:
                return True, f"排位赛显示『进攻已完成！』({cnt_txt or raw_txt})"

        # 2. HSV 橙色进攻按钮消失对比检测
        hsv = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2HSV)
        y1_r, y2_r, x1_r, x2_r = self.REGULAR_BTN_ROI
        y1_k, y2_k, x1_k, x2_k = self.RANKED_BTN_ROI
        reg_patch = hsv[y1_r:y2_r, x1_r:x2_r]
        rank_patch = hsv[y1_k:y2_k, x1_k:x2_k]

        if reg_patch.size > 0 and rank_patch.size > 0:
            reg_orange = float(
                np.mean(cv2.inRange(reg_patch, np.array([10, 110, 160]), np.array([28, 255, 255])) > 0)
            )
            rank_orange = float(
                np.mean(cv2.inRange(rank_patch, np.array([10, 110, 160]), np.array([28, 255, 255])) > 0)
            )
            rank_v = float(np.mean(rank_patch[:, :, 2]))
            if reg_orange >= 0.35 and rank_orange < 0.08 and rank_v < 105.0:
                return True, f"排位赛橙色进攻按钮已消失且呈完成暗灰态 ({cnt_txt or '12/12'})"

        return False, f"排位赛未完成，可继续进攻 ({cnt_txt or '可进攻'})"

    def enter_multiplayer_search(self) -> bool:
        """
        从任意起始界面（主世界 / 联机模式面板 / 编辑军队页面 / 已在战场）智能推进到敌方搜索云层：
        - Step 1: 主世界点击左下角 [战斗] (66, 650)
        - Step 2: 联机模式界面：
          * 若为排位战 (ranked)，先执行 `check_ranked_battle_status` 检测排位赛是否已完成 (如 12/12)；
            若已完成则立即打印日志并点击空白区关闭面板返回主世界！
          * 否则点击 [常规战] (195, 540) 或 [排位战] (535, 540)
        - Step 3: 编辑军队页面点击右下角绿色 [进攻！] (1129, 636)
        - Step 4: 若为排位战 (ranked)，点击弹出的 [确认] 按钮开始搜索
        """
        mode = getattr(self.cfg, "match_mode", "regular")
        mode_cn = "排位赛 (搜到谁直接打谁)" if mode == "ranked" else "常规赛 (按阈值挑鱼)"
        screen = self.device.screencap()
        from .combat_ai import CombatAiTask
        if CombatAiTask.check_and_dismiss_star_bonus_popup(self.device, screen_bgr=screen):
            screen = self.device.screencap()

        # 情况 A：当前已经在敌方战场侦察界面
        if self.is_on_battlefield_screen(screen):
            logger.info("👁️ [进场状态机] 检测到当前已处于敌方战场侦察界面，直接开始评估对手！")
            return True

        # 情况 B：当前在主世界 -> 先点左下角 [战斗] (66, 650) 打开联机模式界面
        atk_green_match = self.matcher.find_one(screen, "village/btn_army_attack.png", threshold=0.76, scales=(1.0,))
        reg_mode_match = self.matcher.find_one(screen, "village/btn_find_match.png", threshold=0.75, scales=(1.0,))

        if atk_green_match is None and reg_mode_match is None:
            logger.info(f"⚔️ [进场 Step 1/4] 在主世界点击左下角 [战斗] 按钮 {self.BTN_ATTACK_MAIN}...")
            self.device.tap(*self.BTN_ATTACK_MAIN)
            time.sleep(1.2)
            screen = self.device.screencap()
            reg_mode_match = self.matcher.find_one(screen, "village/btn_find_match.png", threshold=0.75, scales=(1.0,))
            atk_green_match = self.matcher.find_one(screen, "village/btn_army_attack.png", threshold=0.76, scales=(1.0,))

        # 情况 C：当前在联机模式选择界面 -> 检测排位赛是否完成，并点击 [常规战] 或 [排位战] 进入编辑军队页面
        if atk_green_match is None:
            if mode == "ranked":
                is_ranked_done, ranked_detail = self.check_ranked_battle_status(screen)
                if is_ranked_done:
                    self.last_match_frame = screen
                    self.last_match_reason = "ranked_finished"
                    logger.warning(
                        f"🏆 [排位赛完成检测] 检测到 {ranked_detail}！本周排位赛场次已全部打完，自动关闭联机面板返回主世界！"
                    )
                    self.device.tap(*self.BTN_CLOSE_MODE_PANEL)
                    time.sleep(0.8)
                    return False
                logger.info(f"🏆 [进场 Step 2/4] 排位赛状态检查: {ranked_detail} -> 点击右侧 [排位战] {self.BTN_MODE_RANKED} 进入编辑军队页面...")
                self.device.tap(*self.BTN_MODE_RANKED)
            else:
                target_pt = reg_mode_match.center if reg_mode_match else self.BTN_MODE_REGULAR
                logger.info(f"🎯 [进场 Step 2/4] 当前模式={mode_cn} -> 点击左侧 [常规战 搜索对手] {target_pt} 进入编辑军队页面...")
                self.device.tap(*target_pt)
            time.sleep(1.2)
            screen = self.device.screencap()
            atk_green_match = self.matcher.find_one(screen, "village/btn_army_attack.png", threshold=0.76, scales=(1.0,))

        # 情况 D：当前在编辑军队页面 -> 点击右下角绿色 [进攻！] (1129, 636)
        atk_pt = atk_green_match.center if atk_green_match else self.BTN_ARMY_ATTACK
        logger.info(f"🚀 [进场 Step 3/4] 在编辑军队页面点击右下角绿色 [进攻！] 按钮 {atk_pt}...")
        self.device.tap(*atk_pt)
        time.sleep(1.0)

        # 情况 E：若为排位赛模式 (ranked)，点击弹出的 [确认] 按钮开始搜索
        if mode == "ranked":
            screen = self.device.screencap()
            confirm_m = self.matcher.find_one(screen, "common/btn_ranked_confirm.png", threshold=0.75, scales=(1.0,))
            confirm_pt = confirm_m.center if confirm_m else self.BTN_RANKED_CONFIRM
            logger.info(f"✅ [进场 Step 4/4] 排位赛模式：点击确认弹窗 [确认] 按钮 {confirm_pt}，正式开始匹配对手！")
            self.device.tap(*confirm_pt)
            time.sleep(0.8)
        else:
            logger.info("☁️ [进场 Step 4/4] 常规赛模式：已直接发起匹配，等待搜索白云散去...")

        return True

    def start_matchmaking_and_find_target(self) -> Optional[EnemyLoot]:
        """从主村庄/联机菜单发起进攻，并在常规赛中循环点击[下一个]挑鱼（排位赛直接锁定首名对手）"""
        self.last_match_frame = None
        self.last_match_reason = None
        mode = getattr(self.cfg, "match_mode", "regular")
        if not self.enter_multiplayer_search():
            return None

        max_clicks = 1 if mode == "ranked" else max(1, self.cfg.max_search_clicks)

        for search_idx in range(1, max_clicks + 1):
            screen = self._wait_for_clouds_to_clear()
            if screen is None:
                self.last_match_reason = "cloud_timeout"
                logger.warning("⚠️ 等待搜鱼白云散去超时，尝试返回主村庄")
                self.device.tap(*self.BTN_END_BATTLE)
                return None

            self.last_match_frame = screen
            loot = self.ocr.scan_enemy_loot(screen)
            is_dead = True
            if mode != "ranked" and self.cfg.only_dead_base:
                is_dead = self.dead_checker.is_dead_base(screen, self.cfg.dead_base_min_collectors)

            matched, reason = self.evaluate_base(loot, is_dead)
            logger.info(f"🔎 [第 {search_idx} 次搜索] {reason}")

            if matched:
                self.last_match_reason = "target_found"
                self.last_match_frame = None
                logger.info(
                    f"🎯 [锁定目标对手!] 可掠夺金币={loot.gold:,} | 圣水={loot.elixir:,} | 暗黑重油={loot.dark_elixir:,}"
                )
                return loot

            # 常规赛资源未达标 -> 点击右下角 [下一个 1500] (1162, 505)
            next_m = self.matcher.find_one(screen, "battle/btn_next.png", threshold=0.75, scales=(1.0,))
            next_pt = next_m.center if next_m else self.BTN_NEXT_BASE
            logger.info(f"⏭️ [继续挑鱼] 当前对手资源未达标，点击右下角 [下一个] {next_pt}...")
            self.device.tap(*next_pt)
            time.sleep(1.2)

        self.last_match_reason = "target_not_found"
        logger.warning(f"已连续搜索 {max_clicks} 次未遇到目标鱼，主动点击左下角 [结束战斗] 回村")
        self.device.tap(*self.BTN_END_BATTLE)
        return None

    def _wait_for_clouds_to_clear(self, timeout_sec: float = 25.0) -> Optional[np.ndarray]:
        """等待搜索白云散去，直到侦察界面的 [结束战斗] / [下一个] 按钮或左上角资源数字出现"""
        deadline = time.time() + timeout_sec
        time.sleep(1.2)
        while time.time() < deadline:
            screen = self.device.screencap()
            self.last_match_frame = screen
            if self.is_on_battlefield_screen(screen):
                loot = self.ocr.scan_enemy_loot(screen)
                if loot.gold > 0 or loot.elixir > 0:
                    return screen
                # 即使 OCR 首次刚好在动画帧，只要 [结束战斗] 按钮已稳定出现也再等 0.6s 确认
                time.sleep(0.6)
                return self.device.screencap()
            loot = self.ocr.scan_enemy_loot(screen)
            if loot.gold >= 1000 or loot.elixir >= 1000:
                return screen
            time.sleep(0.7)
        return None
