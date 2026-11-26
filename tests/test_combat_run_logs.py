"""Regression checks for the desktop combat-loop entrypoints, without ADB."""
import logging
from pathlib import Path
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from config import AppConfig
from core.tasks.loot_searcher import home_match_failure
import main
from platforms.pc_desktop import worker_threads
from platforms.pc_desktop.combat_run import CombatRunLog, round_report


def phase(status, reason, touches):
    return {"phase": 1, "status": status, "reason": reason, "touch_count": touches,
            "remaining_slots": [], "slots": [], "evidence": "logs/deployment/fixture"}


@pytest.mark.parametrize(("mode", "method", "result", "status"), [
    ("loop_home_combat", "run_home_battle_once", {
        "combat": {"deployment": [phase("unconfirmed", "bar_unrecognized", 0)]}}, "zero_touches"),
    ("loop_night_combat", "run_night_battle_once", {
        "night_battle": {"deployment": [phase("unconfirmed", "battle_not_entered", 0)]}}, "battle_not_entered"),
    ("loop_dual_combat", "run_dual_world_combat_cycle", {
        "status": "completed", "home": {"combat": {"deployment": [phase("completed", "confirmed", 2)]}},
        "night": {"night_battle": {"deployment": [phase("partial", "no_progress", 3)]}}}, "partial"),
])
def test_loop_emits_persisted_result_and_captures_only_its_own_logs(tmp_path, monkeypatch, caplog,
                                                                  mode, method, result, status):
    caplog.set_level(logging.INFO, logger="coc")
    cfg = AppConfig()
    cfg.device.serial = "auto"
    worker = worker_threads.BotActionWorker(cfg, mode)
    paths, rounds = [], []
    worker.runLogOpened.connect(lambda name: paths.append(Path(name)))
    worker.cycleCompleted.connect(rounds.append)
    worker.cycleCompleted.connect(lambda _: worker.request_stop())
    device = SimpleNamespace(connect=lambda: True, _resolved_serial="serial-from-device")

    def execute():
        logging.getLogger("coc.tasks.deployment").info("匹配记录和下兵证据")
        other = threading.Thread(target=lambda: logging.getLogger("coc.worker").info("另一线程的预览"))
        other.start()
        other.join()
        live = paths[0].read_text(encoding="utf-8")
        assert "匹配记录和下兵证据" in live
        assert "轮次开始 round=1" in live
        return result

    fake_bot = SimpleNamespace(device=device, **{method: execute})
    monkeypatch.setattr(worker_threads, "get_app_root", lambda: tmp_path)
    monkeypatch.setattr(main, "CocAutoBotOrchestrator", lambda *args, **kwargs: fake_bot)
    worker.run()
    [path] = paths
    [summary] = rounds
    assert path.parent == tmp_path / "logs" / "runs"
    text = path.read_text(encoding="utf-8")
    assert mode in text and "resolved_serial=serial-from-device" in text
    assert "另一线程的预览" not in text
    assert "收到正常停止请求" in text and "运行结束" in text
    assert summary["result"] == result
    report = summary["round_report"]
    assert report["status"] == status
    assert report["headline"] in text
    for _, detail in report["lines"]:
        assert detail in text
    assert "logs/deployment/fixture" in text


def test_round_status_preserves_travel_interruption_and_deployment():
    finished = {"combat": {"deployment": [phase("completed", "natural_end", 8)]}}
    mode = "loop_dual_combat"
    assert round_report(1, "loop_home_combat", finished)["status"] == "completed"
    travel = {"status": "travel_failed", "home": finished,
              "travel": {"reason": "boat_unrecognized", "evidence": "boat-frame"}}
    report = round_report(2, mode, travel)
    assert report["status"] == "travel_failed"
    assert any("boat-frame" in detail for _, detail in report["lines"])
    assert round_report(3, mode, {"status": "interrupted", "home": finished})["status"] == "interrupted"


@pytest.mark.parametrize(("connected", "expected"), [(False, "connection_failed"), (True, "error")])
def test_run_records_failure_before_first_deployment(tmp_path, monkeypatch, connected, expected):
    worker = worker_threads.BotActionWorker(AppConfig(), "loop_home_combat")
    paths, rounds = [], []
    worker.runLogOpened.connect(paths.append)
    worker.cycleCompleted.connect(rounds.append)

    def fail():
        raise ValueError("测试匹配失败")

    device = SimpleNamespace(connect=lambda: connected, serial="fake")
    monkeypatch.setattr(worker_threads, "get_app_root", lambda: tmp_path)
    monkeypatch.setattr(main, "CocAutoBotOrchestrator",
                        lambda *args, **kwargs: SimpleNamespace(device=device, run_home_battle_once=fail))
    worker.run()
    text = Path(paths[0]).read_text(encoding="utf-8")
    assert "运行结束 reason=" + expected in text
    if connected:
        assert rounds[0]["round_report"]["status"] == "error"
        assert "测试匹配失败" in text
    else:
        assert not rounds
    assert not any(isinstance(h, logging.FileHandler) and h.baseFilename == paths[0]
                   for h in logging.getLogger("coc").handlers)


def test_home_failure_uses_actual_frame_and_skips_night_battlefield(tmp_path):
    frame = np.full((32, 48, 3), 111, dtype=np.uint8)
    fake = SimpleNamespace(
        night_task=SimpleNamespace(observe_entry_scene=lambda **kwargs: ("night_battle", frame)),
        search_task=SimpleNamespace(is_on_battlefield_screen=lambda _: True),
        combat_task=SimpleNamespace(evidence_dir=tmp_path),
        stopped=lambda: False,
    )
    result = main.CocAutoBotOrchestrator.run_home_battle_once(fake)
    [entry] = result["combat"]["deployment"]
    assert entry["touch_count"] == 0 and entry["match_reason"] == "night_battle"
    assert entry["reason"] == "battle_not_entered"
    saved = cv2.imdecode(np.frombuffer(Path(entry["evidence"]).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(saved, frame)
    assert home_match_failure("target_not_found", frame, None)["reason"] == "target_not_found"


def test_consecutive_runs_get_distinct_files(tmp_path):
    records = []
    for _ in range(2):
        writer = CombatRunLog(tmp_path, "loop_dual_combat", "auto")
        records.append(writer.path)
        writer.close()
    assert len(set(records)) == 2
    assert all(path.read_text(encoding="utf-8") for path in records)


def test_force_stop_persists_active_round_before_thread_termination(tmp_path, monkeypatch):
    worker = worker_threads.BotActionWorker(AppConfig(), "loop_home_combat")
    worker._run_log = CombatRunLog(tmp_path, worker.mode, "auto")
    path = worker._run_log.path
    worker._active_round = 4
    calls, rounds = [], []
    worker.cycleCompleted.connect(rounds.append)
    worker._active_device = SimpleNamespace(force_abort=lambda: calls.append("abort"))
    monkeypatch.setattr(worker, "isRunning", lambda: True)

    def terminate():
        assert "运行结束 reason=force_stopped" in path.read_text(encoding="utf-8")
        calls.append("terminate")

    monkeypatch.setattr(worker, "terminate", terminate)
    monkeypatch.setattr(worker, "wait", lambda timeout: True)
    worker.force_kill_now()
    assert calls == ["abort", "terminate"]
    assert rounds[0]["round_report"]["status"] == "interrupted"
    assert rounds[0]["combat_loop_round"] == 4
