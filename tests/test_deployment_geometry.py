from pathlib import Path

import cv2
import numpy as np
import pytest

from core.device import AdbDeviceController
from core.tasks.deployment import BattleDeployment, battlefield_view, tactical_hints
from core.vision import RedZoneDetector


ROOT = Path(__file__).resolve().parents[1] / "logs"


def geometry_phase(frame, world, side):
    detector = RedZoneDetector()
    # Exercise both ends of the 1600x900 physical short swipe without sending it.
    device = AdbDeviceController()
    device._phys_width, device._phys_height = 1600, 900
    phase = BattleDeployment(
        world=world, phase=1, device=device, reader=None, detector=detector, side_order=[side],
        focus=lambda _: None,
        hints=lambda s, r: tactical_hints([detector._sample_polyline(detector.SHIFTED_OUTER_PERIMETER_POLYLINES[s], 12)], r),
        result_screen=lambda _: False, battle_screen=lambda _: True, stopped=lambda: False, deadline=0,
    )
    phase._new_side(side)
    points = []
    for region in ("left", "center", "right"):
        proposed, accepts = phase._tactical_points(frame, battlefield_view(frame), region)
        assert proposed, region
        for point in proposed:
            touch = device.deployment_touch(point)
            assert accepts(touch.logical_start) and accepts(touch.logical_end)
        points.append(proposed)
    assert max(x for x, _ in points[0]) < min(x for x, _ in points[1])
    assert max(x for x, _ in points[1]) < min(x for x, _ in points[2])
    return phase, points


@pytest.mark.parametrize(("path", "world", "side", "excluded"), [
    ("deployment/home-1-dd2df658f7/001-before.png", "home", "NW",
     [(135, 286), (690, 368), (730, 535)]),
    ("deployment/home-1-e4a62d3989/005-before.png", "home", "NE",
     [(690, 183), (823, 288), (623, 368)]),
    ("deployment/home-1-35adc89947/001-before.png", "home", "NW",
     [(114, 434), (447, 183), (590, 399), (698, 538)]),
    ("deployment/night-1-971689c48d/003-before.png", "night", "SW",
     [(349, 196), (541, 330), (215, 500), (1100, 118)]),
    ("travel/home-to-night-a13dec1895/final-scene.png", "night", "NW",
     [(120, 250), (640, 420), (460, 500), (1130, 175)]),
    ("deployment/home-1-1461a86357/003-before.png", "home", "NE",
     [(640, 324), (1015, 110), (692, 540), (79, 623)]),
    ("deployment/home-1-943238b3e6/001-before.png", "home", "NE",
     [(438, 249), (780, 395), (690, 540), (79, 623)]),
    ("deployment/night-2-7d93a1ae2a/005-before.png", "night", "SW",
     [(654, 270), (444, 498), (180, 510), (82, 630)]),
    ("deployment/night-2-2b3d3f77fd/001-before.png", "night", "NE",
     [(648, 232), (930, 310), (75, 500), (229, 620)]),
])
def test_recorded_battlefield_proposes_three_external_regions(path, world, side, excluded):
    path = ROOT / path
    if not path.exists():
        pytest.skip("local battle capture unavailable")
    frame = cv2.imread(str(path))
    phase, points = geometry_phase(frame, world, side)
    area = phase.detector.deployment_area(frame, world, phase.side)
    assert area.side_span == phase._side_span
    assert all(not area.accepts(point) for point in excluded)
    if "dd2df658f7" in str(path):
        # Hand-traced NW edge from (217, 504) to (770, 89).
        assert all(9 < 667 - 0.75 * x - y < 58 for group in points for x, y in group)


def test_unmarked_blue_paving_does_not_establish_a_deployment_side():
    hsv = np.full((720, 1280, 3), (109, 99, 109), dtype=np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detector = RedZoneDetector()
    for side in detector.CAMERA_FOCUS_SWIPES:
        assert not detector.deployment_area(frame, "night", side).clearance.any()
