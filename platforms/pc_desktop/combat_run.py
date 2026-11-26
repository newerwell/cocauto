"""Per-run diagnostics and the deployment results shown in the desktop console."""
from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import threading
from uuid import uuid4

from core.tasks.village_travel import TRAVEL_REASONS

LOOP_TITLES = {
    "loop_dual_combat": "主世界 ↔ 夜世界 双世界交替循环自动战斗",
    "loop_night_combat": "仅夜世界 连续循环自动战斗",
    "loop_home_combat": "仅主世界 连续循环自动战斗",
}

OUTCOME_TEXT = {
    "completed": "全部确认投放", "partial": "部分完成", "unconfirmed": "投放未确认",
    "zero_touches": "零落兵触点", "battle_not_entered": "未进入战场",
    "target_not_found": "未匹配到符合条件的对手",
    "travel_failed": "渡船切换失败", "interrupted": "已中断", "error": "异常终止",
}


def phase_outcome(entry: dict) -> str:
    if entry.get("status") == "interrupted":
        return "interrupted"
    if entry.get("reason") in {"battle_not_entered", "target_not_found"}:
        return entry["reason"]
    if entry.get("entered_battle") is False and entry.get("touch_count") == 0:
        return "battle_not_entered"
    if entry.get("touch_count") == 0:
        return "zero_touches"
    return entry.get("status", "unconfirmed")


def deployment_lines(summary: dict) -> list[tuple[str, str]]:
    reason_text = {
        **TRAVEL_REASONS,
        "stopped": "手动停止", "timeout": "超时", "natural_end": "自然结算",
        "stagnation": "摧毁率停滞", "stage_advanced": "已进入二阶段",
        "battle_not_entered": "未进入战场", "bar_unrecognized": "兵栏未识别",
        "battle_unrecognized": "战场状态无法确认",
        "recovery_exhausted": "补投次数用尽", "no_safe_points": "未找到可信落点",
        "quantity_unknown": "余量读不清", "selection_unverified": "选中态未确认",
        "camera_unstable": "镜头未稳定", "red_zone": "游戏提示落点无效",
        "no_progress": "余量无变化", "not_deployed": "尚未投放",
        "night_battle": "当前仍在夜世界战场", "cloud_timeout": "等待匹配画面超时",
        "ranked_finished": "排位场次已用完", "target_not_found": "搜索次数已用完",
        "night_match": "当前是夜世界匹配弹窗", "match_dialog_missing": "未识别到匹配弹窗",
        "matchmaking_timeout": "未检测到夜世界开战页面",
        "match_timeout": "匹配等待超时", "confirmed": "卡牌消耗已确认",
    }
    push_text = {
        "both_flanks_engaged": "两翼受击已确认，开始中推",
        "flank_wait_timeout": "清边观察满六秒，开始中推",
        "flank_wait_interrupted": "清边观察中断",
        "insufficient_dragons": "可用龙不足两条，直接中推",
        "insufficient_troops": "可用普通兵不足两只，直接中推",
        "dragon_unconfirmed": "复读后龙卡或余量仍不清楚，已识别的兵力直接中推",
        "troop_count_unconfirmed": "复读后普通兵余量仍不清楚，已识别的兵力直接中推",
    }
    lines = []
    for field, world in (("combat", "主世界"), ("night_battle", "夜世界")):
        for entry in summary.get(field, {}).get("deployment", []):
            pending = entry.get("remaining_slots", [])
            outcome = phase_outcome(entry)
            short = f"{world}第{entry['phase']}阶段 {OUTCOME_TEXT[outcome]}"
            reason = entry.get("entry_reason") or entry.get("match_reason") or entry.get("reason")
            if reason and reason != outcome:
                short += f"（{reason_text.get(reason, reason)}）"
            if entry.get("touch_count") is not None:
                short += f" · 实际落兵触点 {entry['touch_count']}"
            if pending:
                short += f" · {len(pending)} 个未完成槽位"
            details = []
            deploy_reason = entry.get("deployment_reason")
            if deploy_reason and deploy_reason != reason:
                details.append(f"下兵原因 {reason_text.get(deploy_reason, deploy_reason)}")
            counts = entry.get("candidate_counts")
            if counts:
                details.append("候选落点 " + "、".join(
                    f"{name} {counts.get(key, 0)}" for name, key in
                    (("左翼", "left"), ("右翼", "right"), ("中路", "center"), ("法术", "spell"))))
            if entry.get("center_push") in push_text:
                details.append(push_text[entry["center_push"]])
            if entry.get("panorama_status") == "confirmed":
                details.append("下兵后已确认全景居中")
            elif entry.get("panorama_status") == "view_unconfirmed":
                details.append("下兵后全景未确认，暂缓施法")
            wave = entry.get("center_wave")
            if wave:
                details.append(f"中路总攻 {wave['touches']} 个触点，发送耗时 {wave['dispatch_seconds']:.2f} 秒"
                               f"，补投 {wave['recovery_touches']} 次")
                if wave["confirmed_seconds"] is not None:
                    details.append(f"中路部队确认下完耗时 {wave['confirmed_seconds']:.2f} 秒")
            for label, wing in (("左翼", "left"), ("右翼", "right")):
                flank = entry.get("flanks", {}).get(wing)
                if not flank or flank["status"] == "pending":
                    continue
                value = (f"已投放 1 个，{flank['before']} → {flank['after']}" if flank["status"] == "confirmed"
                         else reason_text.get(flank["reason"], "投放未确认"))
                details.append(f"{label} {value}")
            for slot in pending:
                remaining = slot["remaining"]
                if remaining is None:
                    remaining = "未知"
                    if slot["last_known_remaining"] is not None:
                        remaining += f"（上次确认 {slot['last_known_remaining']}）"
                details.append(f"槽位 X={slot['x']}，剩余 {remaining}，{reason_text.get(slot['reason'], slot['reason'])}")
            if entry.get("evidence"):
                details.append(f"证据 {entry['evidence']}")
            lines.append((short, "\n".join([short, *details])))
    for key in ("result", "home", "night"):
        if isinstance(summary.get(key), dict):
            lines.extend(deployment_lines(summary[key]))
    return lines


def round_report(index: int, mode: str, result: dict) -> dict:
    """Derive the round result from its actual phases, including both worlds."""
    statuses = []
    lines = deployment_lines(result)
    for key, field in (("home", "combat"), ("night", "night_battle")):
        world = result.get(key, {}) if mode == "loop_dual_combat" else result
        for phase in world.get(field, {}).get("deployment", []):
            statuses.append(phase_outcome(phase))
        if field == "combat" and "target_loot" in world and world["target_loot"] is None and not world.get(field, {}).get("deployment"):
            statuses.append("target_not_found")
            lines.append(("主世界未匹配到目标 · 实际落兵触点 0", "主世界未匹配到目标 · 实际落兵触点 0"))
    status = result.get("status")
    if status == "travel_failed":
        travel = result.get("travel", {})
        why = travel.get("reason")
        detail = f"渡船切换失败（{TRAVEL_REASONS.get(why, why or '无法确认抵达')}）"
        lines.append((detail, detail + (f"\n证据 {travel['evidence']}" if travel.get("evidence") else "")))
    elif status == "error":
        message = str(result.get("error", ""))
        lines.append((message, message))
    elif status != "interrupted":
        if "interrupted" in statuses:
            status = "interrupted"
        elif len(set(statuses)) == 1:
            status = statuses[0]
        elif "completed" in statuses or "partial" in statuses:
            status = "partial"
        else:
            status = "unconfirmed"
    headline = f"第 {index} 轮 · {OUTCOME_TEXT[status]}"
    return {"status": status, "headline": headline, "lines": lines}


class CombatRunLog:
    """Capture this worker's own log records, including lower-level device and vision events."""

    def __init__(self, root: Path, mode: str, configured_serial: str):
        folder = (root / "logs" / "runs").resolve()
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.path = folder / f"{stamp}_{uuid4().hex}.log"
        self.handler = logging.FileHandler(self.path, mode="x", encoding="utf-8")
        self.handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s %(message)s"))
        self.logger = logging.getLogger("coc")
        worker_thread = threading.get_ident()
        self.handler.addFilter(lambda record: record.thread == worker_thread or getattr(record, "run_file", None) == self.path)
        self.logger.addHandler(self.handler)
        self.write("运行开始 mode=%s configured_serial=%s 日志=%s", mode, configured_serial, self.path)

    def write(self, message: str, *args) -> None:
        # Also accepts stop requests from the GUI thread. Each FileHandler.emit flushes.
        record = self.logger.makeRecord("coc.run", logging.INFO, "", 0, message, args, None, extra={"run_file": self.path})
        self.handler.handle(record)

    def record_round(self, index: int, result: dict, report: dict) -> None:
        self.write("轮次结束 round=%d status=%s %s", index, report["status"], report["headline"])
        for _, detail in report["lines"]:
            self.write("%s", detail)
        self.write("轮次数据 %s", json.dumps(result, ensure_ascii=False, default=str))

    def close(self) -> None:
        self.logger.removeHandler(self.handler)
        self.handler.close()
