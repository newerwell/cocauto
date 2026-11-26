"""Cropped card rows from the user's 11/25 screenshots and adb battle replay."""
from pathlib import Path

import cv2
import numpy as np

from core.tasks.night_world import NightWorldTask
from core.vision.deployment_bar import DeploymentBarReader, card_at
from core.vision.ocr_scanner import OcrScanner


def card_crop(name):
    source = Path(__file__).parent / "fixtures" / "deployment_cards" / f"{name}.png"
    return cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_COLOR)


def card_frame(name):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[572:720, 72:823] = card_crop(name)
    return frame


def test_home_reference_frame_locates_seven_visible_cards():
    bar = DeploymentBarReader(OcrScanner(), "home").scan(card_frame("home_crop"))
    assert bar.supported
    cards = sorted(bar.cards.values(), key=lambda card: card.x)
    assert [card.kind for card in cards] == [
        "troop", "siege", "hero", "hero", "hero", "hero", "spell"
    ]
    assert [card.remaining for card in cards] == [16, 1, 1, 1, 1, 1, 11]
    assert cards[0].unit == "dragon"
    assert not any(card.selected or card.skill for card in cards)


def test_dragon_slot_stays_at_same_location_after_selection_and_exhaustion():
    reader = DeploymentBarReader(OcrScanner(), "home")
    initial = card_at(reader.scan(card_frame("home_crop")), 137)
    selected = card_at(reader.scan(card_frame("home_dragon_selected"), (initial.x,)), initial.x)
    exhausted = card_at(reader.scan(card_frame("home_dragon_spent"), (initial.x,)), initial.x)
    assert initial.remaining == 16
    assert selected is not None
    assert (selected.kind, selected.unit, selected.remaining, selected.selected) == ("troop", "dragon", 16, True)
    assert exhausted is not None
    assert (exhausted.kind, exhausted.unit, exhausted.remaining, exhausted.selected) == ("spent", "dragon", 0, False)


def test_animated_siege_badge_does_not_change_card_kind():
    frame = card_frame("home_crop")
    frame[593:713, 207:299] = card_crop("siege_alert")
    card = card_at(DeploymentBarReader(OcrScanner(), "home").scan(frame), 252)
    assert card is not None
    assert (card.kind, card.unit, card.remaining) == ("siege", "siege_machine", 1)


def test_siege_roof_and_freeze_are_read_from_the_live_card_row():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[572:720, 72:910] = card_crop("home_siege_and_freeze")
    bar = DeploymentBarReader(OcrScanner(), "home").scan(frame)
    siege = card_at(bar, 240)
    freeze = card_at(bar, 854)
    assert (siege.kind, siege.remaining) == ("siege", 1)
    assert (freeze.kind, freeze.remaining) == ("spell", 3)


def test_night_copter_card_moves_up_and_displays_hp_after_deployment():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[565:720, 72:805] = card_crop("night_copter_deployed")
    card = card_at(DeploymentBarReader(OcrScanner(), "night").scan(frame), 129)
    assert (card.kind, card.remaining, card.skill) == ("hero", 0, True)


def test_matched_dragon_with_hidden_count_keeps_identity_and_unknown_quantity():
    frame = card_frame("home_crop")
    frame[595:621, 138:181] = frame[598, 136]
    card = card_at(DeploymentBarReader(OcrScanner(), "home").scan(frame), 137)
    assert card is not None
    assert (card.kind, card.unit, card.remaining) == ("troop", "dragon", None)


def test_matched_hero_without_activation_evidence_remains_unknown_when_dim():
    frame = card_frame("home_crop")
    portrait = frame[594:712, 501:592]
    portrait[:] = cv2.cvtColor(cv2.cvtColor(portrait, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    card = card_at(DeploymentBarReader(OcrScanner(), "home").scan(frame), 548)
    assert card is not None
    assert (card.kind, card.remaining, card.skill) == ("hero", None, False)


def test_matched_hero_with_health_bar_is_deployed_and_skill_eligible():
    frame = card_frame("home_crop")
    frame[586:592, 520:570] = (22, 225, 15)
    card = card_at(DeploymentBarReader(OcrScanner(), "home").scan(frame), 548)
    assert card is not None
    assert (card.kind, card.remaining, card.skill) == ("hero", 0, True)


def test_night_preview_and_live_controls_from_device_frames():
    night = NightWorldTask(None, None, None, None, None)
    reader = DeploymentBarReader(OcrScanner(), "night")
    for name, end_target in (("night_preview_hud", None), ("night_battle_hud", (90, 500))):
        frame = card_crop(name)
        assert night.is_on_night_battlefield(frame)
        assert night._battle_end_target(frame) == end_target
        bar = reader.scan(frame)
        assert bar.supported
        copter, = [card for card in bar.cards.values() if card.unit == "night_battle_copter"]
        assert (copter.kind, copter.remaining, copter.y) == ("hero", 1, 622)
        assert [card.remaining for card in bar.cards.values() if card.kind == "troop"] == [4] * 6
    assert not night.is_on_night_battlefield(np.full((720, 1280, 3), 248, np.uint8))


def test_blue_rank_night_scouting_frame_keeps_each_camp_and_count():
    frame = card_crop("night_blue_rank_hud")
    night = NightWorldTask(None, None, None, None, None)
    assert night.is_on_night_battlefield(frame)
    assert night._battle_end_target(frame) is None
    bar = DeploymentBarReader(OcrScanner(), "night").scan(frame)
    assert bar.supported
    assert [(card.kind, card.remaining) for card in bar.cards.values()] == [("hero", 1)] + [("troop", 4)] * 6
    home = card_frame("home_crop")
    home[519:555, 35:149] = (20, 20, 225)
    assert not night.is_on_night_battlefield(home)
