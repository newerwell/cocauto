"""Wooden ferries and world-specific destination markers."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .template_matcher import MatchResult, TemplateMatcher


@dataclass(frozen=True)
class BoardingPoint:
    point: tuple[int, int]
    score: float
    hull: MatchResult


class WoodenBoatDetector:
    """Require a matching red-striped sail at the expected place on a wooden hull."""

    _SCALES = (1.0, 0.93, 1.07, 0.85, 1.15, 0.78, 1.24, 0.70, 1.35, 1.48, 1.60)
    _HARBORS = {
        "home": ("village", "boat_to_night.png", (265, 628, 145, 820)),
        "night": ("night", "boat_to_home.png", (0, 440, 510, 1120)),
    }

    def __init__(self, matcher: TemplateMatcher):
        self.matcher = matcher

    @staticmethod
    def _valid(frame: np.ndarray | None) -> bool:
        return frame is not None and frame.shape == (720, 1280, 3) and frame.dtype == np.uint8

    def scene(self, frame: np.ndarray | None) -> str:
        if not self._valid(frame):
            return "unknown"
        if self.matcher.find_one(frame, "common/travel_capital.png", 0.88,
                                 (24, 78, 600, 748), (1.0,)):
            return "capital"
        # A dimmed modal still has the same HUD contours. Test the actual
        # yellow button brightness as well as both world-specific shapes.
        button = cv2.cvtColor(frame[582:704, 24:145], cv2.COLOR_BGR2HSV)
        lit = (button[:, :, 0] >= 15) & (button[:, :, 0] <= 40) & (button[:, :, 2] >= 215)
        if float(np.mean(lit)) > 0.33:
            for world, folder in (("home", "village"), ("night", "night")):
                if (self.matcher.find_one(frame, f"{folder}/travel_attack.png", 0.83,
                                          (570, 714, 12, 157), (1.0, 0.95, 1.05))
                        and self.matcher.find_one(frame, f"{folder}/travel_gold_mark.png", 0.79,
                                                  (12, 70, 1213, 1278), (1.0, 0.95, 1.05))):
                    return world
        hsv = cv2.cvtColor(frame[85:635, 140:1140], cv2.COLOR_BGR2HSV)
        if np.mean((hsv[:, :, 1] <= 32) & (hsv[:, :, 2] >= 205)) >= 0.58:
            return "loading"
        return "unknown"

    def find(self, frame: np.ndarray, source: str, threshold: float = 0.73) -> BoardingPoint | None:
        if not self._valid(frame):
            return None
        folder, hull_file, harbor = self._HARBORS[source]
        hull_template = self.matcher.load_template(f"{folder}/{hull_file}")
        sail_template = self.matcher.load_template(f"{folder}/boat_sail.png")
        if hull_template is None or sail_template is None:
            return None

        y1, y2, x1, x2 = harbor
        bubble = self.matcher.find_one(frame, f"{folder}/boat_bubble.png", 0.82,
                                       (max(0, y1-76), y2, x1, x2), self._SCALES)
        regions = [harbor]
        if bubble is not None:
            bx, by = bubble.center
            regions.insert(0, (max(y1, by - 4), min(y2, by + 145),
                               max(x1, bx - 102), min(x2, bx + 65)))

        for region in regions:
            hull = self.matcher.find_one(frame, hull_template, max(0.73, threshold), region, self._SCALES)
            if hull is None:
                continue
            scale = hull.w / hull_template.shape[1]
            right = hull.x + hull.w
            top = hull.y
            sail = self.matcher.find_one(frame, sail_template, 0.79,
                                         (top, top + round(hull.h * 0.79), hull.x, right),
                                         (scale, scale * 0.96, scale * 1.04))
            if sail is None or not self._striped_sail(frame, sail):
                continue
            point = (hull.x + round(hull.w * 0.44), hull.y + round(hull.h * 0.72))
            if source == "night" and (point[0] >= 1048 or (point[0] >= 994 and point[1] <= 110)):
                continue
            return BoardingPoint(point, min(hull.score, sail.score), hull)
        return None

    @staticmethod
    def _striped_sail(frame: np.ndarray, sail: MatchResult) -> bool:
        hsv = cv2.cvtColor(frame[sail.y:sail.y+sail.h, sail.x:sail.x+sail.w], cv2.COLOR_BGR2HSV)
        hue, sat, val = cv2.split(hsv)
        red = ((hue <= 13) | (hue >= 168)) & (sat >= 70) & (val >= 95)
        stripes = (hue >= 12) & (hue <= 48) & (sat <= 134) & (val >= 135)
        return bool(np.mean(red) > 0.10 and np.mean(stripes) > 0.13)
