"""Replay the battle loop entry with a real night HUD and the production scene readers."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from config import load_config
from core.tasks.combat_ai import CombatAiTask
from main import CocAutoBotOrchestrator


class EntryReplay:
    def __init__(self, frames):
        self.frames = list(frames)
        self.taps = []
        self.seconds = 0.0

    def screencap(self):
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]

    def tap(self, x, y, humanize=True):
        self.taps.append((x, y))

    def wait(self, duration):
        self.seconds += duration


def fixture(name):
    path = Path(__file__).parent / "fixtures" / "deployment_cards" / f"{name}.png"
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def bot_with(frames):
    replay = EntryReplay(frames)
    bot = CocAutoBotOrchestrator(load_config(), device=replay)
    bot.night_task.wait = replay.wait
    bot.night_task.now = lambda: replay.seconds
    return bot, replay


def test_night_loop_resumes_after_clouds_without_collection_or_ferry():
    field = fixture("night_blue_rank_hud")
    bot, replay = bot_with([np.full_like(field, 248), field])
    bot.night_task.collect_night_resources = Mock(side_effect=AssertionError("battle already open"))
    bot.village_task.travel.go = Mock(side_effect=AssertionError("boat not available in battle"))
    bot.night_task.execute_night_battle = Mock(return_value={
        "deployment": [{"phase": 1, "status": "partial", "reason": "natural_end"}]})
    report = bot.run_night_battle_once()
    scene, observed = bot.night_task.execute_night_battle.call_args.args[0]
    assert scene == "night_battle" and observed is field
    assert report["night_collected"] == 0 and report["status"] == "partial"
    assert replay.taps == [] and replay.seconds > 0


def test_return_from_results_waits_through_clouds_before_routing_the_ferry():
    clouds = np.full((720, 1280, 3), 245, np.uint8)
    result = np.zeros_like(clouds)
    result[585:651, 560:721] = (20, 200, 50)
    village = np.zeros_like(clouds)
    bot, replay = bot_with([result, result, clouds, village])
    village[587:696, 28:142] = bot.matcher.load_template("night/travel_attack.png")
    village[17:64, 1221:1268] = bot.matcher.load_template("night/travel_gold_mark.png")
    bot.village_task.travel.go = Mock(return_value={"arrived_home": True})
    assert bot.night_task.travel_to_home_village()["arrived_home"]
    assert replay.taps == [CombatAiTask.BTN_RETURN_HOME]
    target, observed = bot.village_task.travel.go.call_args.args
    assert target == "home" and observed is village


def test_dismiss_late_star_bonus_after_a_ferry_pan_loses_the_village():
    village = np.zeros((720, 1280, 3), dtype=np.uint8)
    reward = village.copy()
    reward[140:500, 280:1000] = (210, 99, 30)
    reward[525:600, 555:725] = (25, 200, 65)
    bot, replay = bot_with([reward, village])
    village[587:696, 28:142] = bot.matcher.load_template("night/travel_attack.png")
    village[17:64, 1221:1268] = bot.matcher.load_template("night/travel_gold_mark.png")
    bot.village_task.travel.go = Mock(side_effect=[
        {"arrived_home": False, "reason": "unknown_scene", "touches": []},
        {"arrived_home": True},
    ])
    assert bot.night_task.travel_to_home_village(("night", village))["arrived_home"]
    assert replay.taps == [CombatAiTask.BTN_STAR_BONUS_OK]
    assert bot.village_task.travel.go.call_count == 2


def test_post_battle_reward_waits_for_skip_before_routing():
    animation = np.zeros((720, 1280, 3), dtype=np.uint8)
    village = animation.copy()
    bot, replay = bot_with([animation])
    chest = bot.matcher.load_template("battle/return_reward_chest.png")
    skip = bot.matcher.load_template("battle/return_reward_skip.png")
    confirm = bot.matcher.load_template("battle/return_reward_confirm.png")
    stars = bot.matcher.load_template("battle/return_star_bonus.png")
    animation[260:395, 910:1245] = chest
    skippable = animation.copy()
    skippable[635:666, 1165:1222] = skip
    dialog = animation.copy()
    dialog[194:248, 531:755] = confirm
    bonus = np.zeros_like(animation)
    bonus[130:181, 466:816] = stars
    village[587:696, 28:142] = bot.matcher.load_template("night/travel_attack.png")
    village[17:64, 1221:1268] = bot.matcher.load_template("night/travel_gold_mark.png")
    replay.frames[:] = [animation, skippable, skippable, dialog, dialog, bonus, bonus, village]
    bot.village_task.travel.go = Mock(return_value={"arrived_home": True})
    assert bot.night_task.travel_to_home_village()["arrived_home"]
    assert replay.taps == [(1193, 650), (778, 462), (640, 565)]
    assert bot.village_task.travel.go.call_args.args[1] is village


def test_post_battle_chest_progresses_only_after_a_seal_changes():
    chest = np.zeros((720, 1280, 3), dtype=np.uint8)
    village = chest.copy()
    bot, replay = bot_with([chest])
    chest[260:395, 910:1245] = bot.matcher.load_template("battle/return_reward_chest.png")
    chest[587:618, 522:545] = (28, 153, 216)
    chest[587:618, 582:605] = (28, 153, 216)
    one_seal = chest.copy()
    one_seal[587:618, 522:545] = 0
    reward = np.zeros_like(chest)
    reward[531:654, 480:774] = bot.matcher.load_template("battle/return_reward_continue.png")
    result = np.zeros_like(chest)
    result[585:651, 560:721] = (20, 200, 50)
    village[587:696, 28:142] = bot.matcher.load_template("night/travel_attack.png")
    village[17:64, 1221:1268] = bot.matcher.load_template("night/travel_gold_mark.png")
    replay.frames[:] = [result, chest, chest, one_seal, one_seal, reward, reward, village]
    bot.village_task.travel.go = Mock(return_value={"arrived_home": True})
    assert bot.night_task.travel_to_home_village()["arrived_home"]
    assert replay.taps == [CombatAiTask.BTN_RETURN_HOME, (625, 320), (625, 320), (628, 618)]


def test_matchmaking_timeout_saves_the_final_cloud_and_zero_touches():
    clouds = np.full((720, 1280, 3), 246, np.uint8)
    dialog = np.zeros_like(clouds)
    bot, replay = bot_with([dialog, clouds])
    dialog[415:505, 824:1076] = bot.matcher.load_template("night/btn_find_now.png")
    bot.village_task.travel.go = Mock(side_effect=AssertionError("already in matchmaking"))
    with TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        bot.night_task.evidence_dir = Path(tmp)
        result = bot.run_night_battle_once()
        phase, = result["night_battle"]["deployment"]
        assert result["status"] == phase["status"] == "unconfirmed"
        assert (phase["reason"], phase["entry_reason"], phase["touch_count"]) == (
            "battle_not_entered", "matchmaking_timeout", 0)
        assert replay.taps == [(950, 460)] and replay.seconds >= 60
        folder = Path(phase["evidence"])
        assert np.array_equal(cv2.imread(str(folder / "001-before.png")), clouds)
        record = json.loads((folder / "attempts.jsonl").read_text(encoding="utf-8"))
        assert record["world"] == "night" and record["evidence"] == str(folder) and record["touches"] == []


def test_matchmaking_waits_for_two_fresh_countdown_frames():
    ready = fixture("night_scout_countdown")
    active = fixture("night_battle_hud")
    clouds = np.full_like(ready, 248)
    dialog = np.zeros_like(ready)
    bot, replay = bot_with([active, clouds, ready, clouds, ready, ready.copy()])
    dialog[415:505, 824:1076] = bot.matcher.load_template("night/btn_find_now.png")
    assert bot.night_task.enter_night_matchmaking(("night_match", dialog))
    assert bot.night_task.last_entry_frame is not ready
    assert np.array_equal(bot.night_task.last_entry_frame, ready)
    assert replay.taps == [(950, 460)] and len(replay.frames) == 1


@pytest.mark.parametrize("opening,order", [("night_battle", ["night", "boat", "home"]),
                                          ("home", ["home", "night", "boat"])])
def test_dual_cycle_keeps_partial_outcome_after_a_confirmed_return(opening, order):
    bot, _ = bot_with([fixture("night_blue_rank_hud")])
    calls = []
    bot.night_task.observe_entry_scene = Mock(side_effect=[(opening, None), ("night", None)])

    def run_night(entry=None):
        calls.append("night")
        assert (entry is not None) == (opening == "night_battle")
        return {"status": "partial", "night_battle": {"deployment": [{"reason": "natural_end"}]}}

    def ferry(entry):
        assert entry[0] == "night"
        calls.append("boat")
        return {"arrived_home": True}

    bot.run_home_battle_once = lambda: calls.append("home") or {"status": "completed"}
    bot.run_night_battle_once = run_night
    bot.night_task.travel_to_home_village = ferry
    result = bot.run_dual_world_combat_cycle()
    assert calls == order
    assert result["status"] == "partial" and result["home"]["status"] == "completed"
