from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from config import CombatConfig, VillageConfig
from core.device import BaseDeviceController
from core.tasks.night_world import NightWorldTask
from core.tasks.village_routine import VillageRoutineTask
from core.vision import OcrScanner, TemplateMatcher
from tools.calibrator.template_harvester import TemplateHarvester


FIXTURES = Path(__file__).parent / "fixtures" / "travel"


def home_coast() -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[397:627, 155:565] = cv2.imread(str(FIXTURES / "home-coast.png"))
    return frame


def test_wooden_boat_wins_over_airship_and_seawater():
    village = VillageRoutineTask(None, TemplateMatcher(), OcrScanner(), VillageConfig())
    frame = home_coast()
    x, y, _, _ = village.find_builder_boat(frame)
    assert 312 <= x <= 363 and 478 <= y <= 530


def test_bubble_without_visible_hull_cannot_be_used_as_input():
    village = VillageRoutineTask(None, TemplateMatcher(), OcrScanner(), VillageConfig())
    frame = home_coast()
    frame[463:541, 298:386] = frame[463:541, 200:288].copy()
    assert village.find_builder_boat(frame) is None


def test_fixed_scene_harvest_preserves_calibrated_boat_templates():
    sources = ["village/boat_to_night.png", "village/boat_bubble.png", "night/boat_to_home.png"]
    with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
        tmp_path = Path(temporary)
        harvester = TemplateHarvester(tmp_path)
        for relpath in sources:
            path = tmp_path / relpath
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"calibrated")
        frame = home_coast()
        harvester.generate_bootstrap_templates(overwrite=True)
        harvester.harvest_from_screen_by_scene(frame, "all")
        assert [(tmp_path / rel).read_bytes() for rel in sources] == [b"calibrated"] * 3


def village_frame(world, *, harbor_dx=0, boat=True):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    if world == "capital":
        frame[36:68, 615:733] = TemplateMatcher().load_template("common/travel_capital.png")
        return frame
    folder = "village" if world == "home" else "night"
    frame[587:696, 28:142] = TemplateMatcher().load_template(f"{folder}/travel_attack.png")
    frame[17:64, 1221:1268] = TemplateMatcher().load_template(f"{folder}/travel_gold_mark.png")
    if boat:
        coast = cv2.imread(str(FIXTURES / f"{world}-coast.png"))
        x, y = (155, 397) if world == "home" else (775, 58)
        x += harbor_dx
        frame[y:y+coast.shape[0], x:x+coast.shape[1]] = coast
    return frame


class FerryReplay(BaseDeviceController):
    def __init__(self, initial, after_taps=(), after_pans=(), after_zooms=()):
        self.frame = initial
        self.tap_frames = iter(after_taps)
        self.pan_frames = iter(after_pans)
        self.zoom_frames = iter(after_zooms)
        self.observations = iter(())
        self.taps = []
        self.pans = []
        self.zoom_count = 0

    def connect(self):
        return True

    def screencap(self):
        self.frame = next(self.observations, self.frame)
        return self.frame

    def tap(self, x, y, humanize=True):
        assert not humanize
        self.taps.append((x, y))
        self.observations = iter(next(self.tap_frames, ()))
        return x, y

    def swipe(self, start, end, duration_ms=350, use_bezier=True):
        self.pans.append((start, end))
        self.frame = next(self.pan_frames, self.frame)

    def pinch_zoom_out(self):
        self.zoom_count += 1
        self.frame = next(self.zoom_frames, self.frame)


def replay_village(replay):
    return VillageRoutineTask(replay, TemplateMatcher(), OcrScanner(), VillageConfig(), wait=lambda _: None)


def test_loading_transition_waits_for_two_positive_destination_frames():
    loading = np.full((720, 1280, 3), 255, dtype=np.uint8)
    arrival = village_frame("night", boat=False)
    replay = FerryReplay(village_frame("home"), after_taps=[(loading, arrival, loading, arrival, arrival)])
    report = replay_village(replay).travel.toggle()
    assert report["arrived_night_world"] and report["scene"] == "night"
    assert replay.taps == [(337, 515)]
    assert replay.zoom_count == 2


def test_boarding_uses_the_hull_at_its_post_zoom_position():
    initial = village_frame("home")
    panorama = village_frame("home", harbor_dx=77)
    night = village_frame("night")
    replay = FerryReplay(initial, after_taps=[(night,)], after_zooms=(panorama, night))
    result = replay_village(replay).travel.toggle()
    assert result["arrived_night_world"] and replay.zoom_count == 2
    assert replay.taps == [(414, 515)]


def test_unconfirmed_panorama_stops_before_boarding():
    replay = FerryReplay(village_frame("home"), after_zooms=[village_frame("capital")])
    result = replay_village(replay).travel.toggle()
    assert result["reason"] == "view_unconfirmed" and replay.taps == []


def test_return_trip_pans_until_current_hull_is_visible():
    replay = FerryReplay(village_frame("night", boat=False), after_pans=[village_frame("night", harbor_dx=-70)],
                         after_taps=[(village_frame("home"),)])
    report = replay_village(replay).travel.toggle()
    assert report["arrived_home"] and report["scene"] == "home"
    assert len(replay.pans) == 1 and replay.taps == [(882, 169)]


def test_retry_requires_the_wooden_boat_on_a_fresh_frame():
    moved = village_frame("home", harbor_dx=77)
    replay = FerryReplay(village_frame("home"), after_taps=[(moved,), (village_frame("night"),)])
    report = replay_village(replay).travel.toggle()
    assert report["arrived_night_world"]
    assert replay.taps == [(337, 515), (414, 515)]


def test_capital_or_unrecognized_arrival_must_not_retry_boat_input():
    for wrong_scene, reason in ((village_frame("capital"), "wrong_world"),
                                (np.zeros((720, 1280, 3), dtype=np.uint8), "unknown_scene"),
                                (np.full((720, 1280, 3), 255, dtype=np.uint8), "loading")):
        replay = FerryReplay(village_frame("home"), after_taps=[(wrong_scene,)])
        report = replay_village(replay).travel.toggle()
        assert not report["arrived_night_world"] and report["reason"] == reason
        assert replay.taps == [(337, 515)]


def test_night_collection_does_not_act_when_arrival_is_unconfirmed():
    replay = FerryReplay(village_frame("capital"))
    village = replay_village(replay)
    night = NightWorldTask(replay, village.matcher, village.ocr, village, CombatConfig(), wait=lambda _: None)
    report = night.collect_night_resources()
    assert report["status"] == "travel_failed" and report["travel"]["reason"] == "wrong_world"
    assert replay.taps == [] and replay.pans == []
