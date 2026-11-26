"""
《部落冲突》跨平台全自动辅助软件 (COC-AutoApp) 总控入口
同时支持：
- Windows 电脑端独立运行 (自动连接 MuMu/雷电/蓝叠等安卓模拟器)
- Android 手机端脱离电脑独立运行 (Termux 本机无线 ADB / Shizuku)
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from typing import Callable

from config import load_config, AppConfig
from config.paths import get_app_root
from core.humanizer import HumanJitter, SessionScheduler
from core.device import AdbDeviceController, BaseDeviceController
from core.vision import TemplateMatcher, OcrScanner, RedZoneDetector, DeadBaseChecker
from core.tasks import VillageRoutineTask, LootSearcherTask, CombatAiTask, AutoUpgraderTask, NightWorldTask
from core.tasks.loot_searcher import home_match_failure
from platforms.pc_desktop.emulator_host import PcEmulatorHost

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("coc.main")


class CocAutoBotOrchestrator:
    """跨平台全自动挂机总控状态机 (Orchestrator)"""

    def __init__(self, cfg: AppConfig, device: BaseDeviceController | None = None,
                 stopped: Callable[[], bool] = lambda: False):
        self.cfg = cfg
        self.stopped = stopped
        self.jitter = HumanJitter(
            sigma_px=cfg.humanizer.click_sigma_px,
            max_offset_px=cfg.humanizer.click_max_offset_px,
            min_delay_sec=cfg.humanizer.min_action_delay_sec,
            max_delay_sec=cfg.humanizer.max_action_delay_sec,
            screen_width=cfg.device.standard_width,
            screen_height=cfg.device.standard_height,
        )
        self.scheduler = SessionScheduler(
            max_session_minutes=cfg.humanizer.max_session_minutes,
            rest_duration_minutes=cfg.humanizer.rest_duration_minutes,
            night_sleep_window=cfg.humanizer.night_sleep_window,
        )
        self.device = device or AdbDeviceController(
            serial=cfg.device.serial,
            jitter=self.jitter,
            screencap_timeout=cfg.device.screencap_timeout,
        )

        # 初始化视觉感知组件
        self.matcher = TemplateMatcher()
        self.ocr = OcrScanner()
        self.red_detector = RedZoneDetector()
        self.dead_checker = DeadBaseChecker(self.matcher)

        # 初始化四大核心业务状态机
        travel_evidence = get_app_root() / "logs" / "travel" if device is None else None
        self.village_task = VillageRoutineTask(self.device, self.matcher, self.ocr, cfg.village,
                                               stopped=stopped, travel_evidence_dir=travel_evidence)
        self.upgrade_task = AutoUpgraderTask(self.device, self.matcher, self.ocr, cfg.upgrade)
        self.search_task = LootSearcherTask(
            self.device, self.matcher, self.ocr, self.dead_checker, cfg.loot_filter
        )
        evidence_dir = get_app_root() / "logs" / "deployment" if device is None else None
        self.combat_task = CombatAiTask(
            self.device, self.red_detector, self.dead_checker, self.ocr, cfg.combat,
            stopped=stopped, evidence_dir=evidence_dir,
        )
        self.night_task = NightWorldTask(
            self.device, self.matcher, self.ocr, self.village_task, cfg.combat,
            stopped=stopped, evidence_dir=evidence_dir,
        )

    def run_single_cycle(self) -> dict:
        """
        执行完整的一轮挂机闭环：
        1. 作息与防封检查 -> 2. 村庄收菜/捐兵/补兵 -> 3. 工人建议列表自动刷墙/升建筑 -> 4. 自动搜鱼 -> 5. 动态红线智能战斗
        """
        need_break, reason, rest_sec = self.scheduler.should_take_break()
        if need_break:
            logger.info(f"[拟人化防封作息] {reason}，暂停休息 {rest_sec // 60} 分钟")
            return {"status": "resting", "reason": reason, "rest_sec": rest_sec}

        # 阶段 1：村庄日常管家 (Zoom-Out 视角归一化 + 收菜 + 部落关键词捐兵 + 快速训练补兵)
        village_stats = self.village_task.run_all()

        # 阶段 2：基于顶部工人建议列表的零误差自动刷墙与建筑升级
        upgrade_stats = self.upgrade_task.run_check_and_upgrade()

        # 阶段 3：联机自动搜鱼与资源/死鱼过滤
        target_loot = self.search_task.start_matchmaking_and_find_target()
        combat_stats = {"hero_abilities_used": 0}

        # 阶段 4：若搜到目标鱼，执行动态红线边缘检测与智能下兵战斗
        if target_loot is not None:
            combat_stats = self.combat_task.execute_attack()

        return {
            "status": "completed",
            "village": village_stats,
            "upgrade": upgrade_stats,
            "target_loot": target_loot,
            "combat": combat_stats,
        }

    def run_home_battle_once(self) -> dict:
        """
        执行单轮【主世界】收菜 + 搜鱼 + 智能下兵 + 回营胜利之星领取：
        - 若当前位于夜世界，先自动乘船返回主世界；
        - 若当前选择了排位赛 (ranked) 且检测到本周排位赛已完成 (12/12)，自动无缝降级为常规赛 (regular) 继续搜鱼战斗！
        """
        logger.info("世界开始 world=home")
        scene, scr = self.night_task.observe_entry_scene(home_battle=self.search_task.is_on_battlefield_screen)
        logger.info("主世界入口 scene=%s", scene)
        collected = 0
        if scene in {"night_battle", "night_match", "stopped"} or self.stopped():
            why = "stopped" if self.stopped() else scene
            phase = home_match_failure(why, scr, self.combat_task.evidence_dir)
            logger.info("世界结束 world=home 匹配结果=%s", phase)
            return {"status": phase["status"], "target_loot": None,
                    "combat": {"hero_abilities_used": 0, "deployment": [phase]}}
        if scene != "home_battle":
            travel = (self.night_task.travel_to_home_village((scene, scr))
                      if scene in {"home", "night"} else self.village_task.travel.go("home", scr))
            if not travel["arrived_home"]:
                logger.info("世界结束 world=home status=travel_failed travel=%s", travel)
                phase = home_match_failure(travel["reason"], None, None)
                phase["evidence"] = travel.get("evidence")
                return {"status": "travel_failed", "travel": travel,
                        "combat": {"hero_abilities_used": 0, "deployment": [phase]}}
            collected = self.village_task.collect_resource_bubbles()

        logger.info("匹配开始 world=home mode=%s", self.cfg.loot_filter.match_mode)
        target_loot = self.search_task.start_matchmaking_and_find_target()
        if target_loot is None and self.cfg.loot_filter.match_mode == "ranked" and not self.stopped():
            logger.info("主世界排位匹配结束 reason=%s，切换常规赛继续搜索", self.search_task.last_match_reason)
            orig_mode = self.cfg.loot_filter.match_mode
            self.cfg.loot_filter.match_mode = "regular"
            try:
                target_loot = self.search_task.start_matchmaking_and_find_target()
            finally:
                self.cfg.loot_filter.match_mode = orig_mode

        if target_loot is not None and not self.stopped():
            logger.info("匹配结束 world=home reason=target_found loot=%s", target_loot)
            combat_stats = self.combat_task.execute_attack()
        else:
            why = "stopped" if self.stopped() else self.search_task.last_match_reason or "battle_not_entered"
            phase = home_match_failure(why, self.search_task.last_match_frame, self.combat_task.evidence_dir)
            combat_stats = {"hero_abilities_used": 0, "deployment": [phase]}
            logger.info("匹配结束 world=home reason=%s evidence=%s", why, phase["evidence"])
        logger.info("世界结束 world=home deployment=%s", combat_stats.get("deployment"))

        return {
            "status": combat_stats["deployment"][0]["status"],
            "collected_bubbles": collected,
            "target_loot": target_loot,
            "combat": combat_stats,
        }

    def run_night_battle_once(self, entry=None) -> dict:
        """
        执行单轮【夜世界】跨海导航 + 收菜 + 夜世界 2.0 智能战斗 + 回营自动领取胜利之星奖励：
        """
        logger.info("夜世界战斗轮次开始 world=night")
        if entry is None:
            entry = self.night_task.observe_entry_scene(home_battle=self.search_task.is_on_battlefield_screen)
        scene, frame = entry
        logger.info("夜世界入口场景 world=night scene=%s", scene)
        collected = 0
        if scene in {"home", "night"}:
            collect_res = self.night_task.collect_night_resources(entry)
            if collect_res.get("status") == "travel_failed":
                last_scene = self.device.screencap() if not self.stopped() else frame
                battle = self.night_task.entry_failure("travel_failed", last_scene)
                result = {"status": "travel_failed", "travel": collect_res["travel"],
                          "night_collected": 0, "night_battle": battle}
                logger.warning("夜世界轮次结束 world=night status=travel_failed travel=%s", collect_res["travel"])
                return result
            collected = collect_res.get("night_collected_bubbles", 0)
            entry = None
        if self.stopped():
            logger.info("夜世界轮次结束 world=night status=interrupted")
            return {"status": "interrupted", "night_collected": collected,
                    "night_battle": self.night_task.entry_failure("stopped", frame)}
        battle_res = self.night_task.execute_night_battle(entry)
        travel = self.night_task.last_travel
        if travel is not None and not travel["arrived_night_world"]:
            logger.warning("夜世界轮次结束 world=night status=travel_failed travel=%s", travel)
            return {"status": "travel_failed", "travel": travel, "night_battle": battle_res}
        phases = battle_res["deployment"]
        statuses = {phase["status"] for phase in phases}
        status = next((value for value in ("interrupted", "unconfirmed", "partial") if value in statuses), "completed")
        logger.info("夜世界轮次结束 world=night status=%s phases=%s", status,
                    [(phase["phase"], phase["reason"], phase.get("evidence")) for phase in phases])
        return {"status": status, "night_collected": collected, "night_battle": battle_res}

    def run_dual_world_combat_cycle(self) -> dict:
        """
        执行完整的一轮【主世界 ↔ 夜世界 双世界交替循环自动战斗】：
        1. 主世界收菜 & 自动战斗 (含排位赛满 12/12 自动切常规赛 + 回营领胜利之星)
        2. 主世界点击左下角渡船前往夜世界
        3. 夜世界收菜 & 夜世界 2.0 自动战斗 (先清边后中路 0x 全放 + 主动技能 + 回营领胜利之星)
        4. 夜世界点击右上角码头渡船返回主世界
        """
        entry = self.night_task.observe_entry_scene(home_battle=self.search_task.is_on_battlefield_screen)
        night_first = entry[0] in {"night_battle", "night_match"}
        order = ("night", "home") if night_first else ("home", "night")
        result = {}
        statuses = set()
        for world in order:
            if self.stopped():
                return {**result, "status": "interrupted"}
            logger.info("双世界轮次 world=%s", world)
            report = (self.run_home_battle_once() if world == "home"
                      else self.run_night_battle_once(entry if night_first else None))
            result[world] = report
            status = report.get("status", "completed")
            statuses.add(status)
            if self.stopped() or status == "travel_failed":
                if "travel" in report:
                    result["travel"] = report["travel"]
                return {**result, "status": "interrupted" if self.stopped() else status}
            # A partially deployed army can still finish and return to camp.
            # Continue the other world's turn only after seeing a village.
            next_entry = None
            if status != "completed":
                battle = report.get("combat" if world == "home" else "night_battle", {})
                if any(p.get("reason") in {"battle_not_entered", "target_not_found", "stopped"}
                       for p in battle.get("deployment", [])):
                    return {**result, "status": status}
                next_entry = self.night_task.observe_entry_scene(home_battle=self.search_task.is_on_battlefield_screen)
                if next_entry[0] not in {"home", "night"}:
                    return {**result, "status": "interrupted" if self.stopped() else status}
            if world == "night":
                travel = self.night_task.travel_to_home_village(next_entry)
                result["travel"] = travel
                if not travel["arrived_home"]:
                    return {**result, "status": "travel_failed"}
        status = next((s for s in ("interrupted", "unconfirmed", "partial") if s in statuses), "completed")
        return {**result, "status": status}

    def run_forever(self, max_cycles: int | None = None) -> None:
        """启动持续挂机主循环"""
        if not self.device.connect():
            logger.error("未能连接到 Android 设备（请检查电脑模拟器是否启动，或手机无线调试是否配对）")
            sys.exit(1)

        cycle = 0
        while True:
            cycle += 1
            logger.info(f"========== 开始第 {cycle} 轮自动化循环 ==========")
            summary = self.run_single_cycle()
            if summary.get("status") == "resting":
                time.sleep(min(summary.get("rest_sec", 60), 3600))
                self.scheduler.reset_session()

            if max_cycles is not None and cycle >= max_cycles:
                logger.info(f"已完成设定的 {max_cycles} 轮循环，安全退出。")
                break


def main():
    parser = argparse.ArgumentParser(description="《部落冲突》跨平台全自动辅助软件 (手机端独立 + 电脑模拟器独立)")
    parser.add_argument("--config", type=str, default=None, help="自定义 YAML 配置文件路径")
    parser.add_argument("--serial", type=str, default=None, help="覆盖 ADB 设备地址 (如 127.0.0.1:16384 或 auto)")
    parser.add_argument("--cycles", type=int, default=None, help="运行指定轮数后退出")
    parser.add_argument("--scan-emulators", action="store_true", help="扫描电脑本地正在运行的安卓模拟器并退出")
    args = parser.parse_args()

    if args.scan_emulators:
        emulators = PcEmulatorHost.discover_emulators()
        if not emulators:
            print("未检测到正在运行的电脑安卓模拟器。")
        else:
            for emu in emulators:
                print(f"发现模拟器: {emu.name} -> {emu.serial}")
        return

    cfg = load_config(args.config)
    if args.serial:
        cfg.device.serial = args.serial

    bot = CocAutoBotOrchestrator(cfg)
    bot.run_forever(max_cycles=args.cycles)


if __name__ == "__main__":
    main()
