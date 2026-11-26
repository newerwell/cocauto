"""Observe damage on an outer building near a confirmed deployment point."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .red_zone_detector import Point, RedZoneDetector


@dataclass
class FlankWatch:
    point: Point
    world: str
    reference: np.ndarray
    reference_grass: float
    reference_edges: int
    gone_frames: int = 0

    @classmethod
    def find(cls, frame: np.ndarray, world: str, landing: Point) -> FlankWatch | None:
        # Look inward from the landing point. A structure must be inside
        # the observed red area and have texture beyond the surrounding turf.
        terrain = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass = RedZoneDetector.terrain_mask(terrain, world)
        forbidden, enclosed, _ = RedZoneDetector()._red_boundary(terrain)
        if not enclosed:
            return None
        rows, cols = np.where(forbidden[85:510, 175:1050] > 0)
        if not len(rows):
            return None
        direction = np.array((float(np.median(cols)) + 175, float(np.median(rows)) + 85)) - landing
        length = float(np.linalg.norm(direction))
        if length < 30:
            return None
        for distance in range(34, min(212, round(length)), 16):
            x, y = map(round, landing + direction * distance / length)
            if not (188 <= x <= 1040 and 92 <= y <= 507):
                continue
            patch = np.s_[y-18:y+19, x-21:x+22]
            if np.mean(forbidden[patch] > 0) < 0.72:
                continue
            turf = float(np.mean(grass[patch] > 0))
            view = frame[patch]
            edges = cls._edges(view)
            if turf < 0.55 and edges >= 130:
                return cls((x, y), world, view.copy(), turf, edges)
        return None

    @staticmethod
    def _edges(patch: np.ndarray) -> int:
        return int(np.count_nonzero(cv2.Canny(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 55, 125)))

    def damaged(self, frame: np.ndarray) -> bool:
        x, y = self.point
        patch = frame[y-18:y+19, x-21:x+22]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        change = np.max(cv2.absdiff(patch, self.reference), axis=2) > 54
        flash = ((hsv[:, :, 0] <= 22) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 145) & (hsv[:, :, 2] > 215)
        if np.count_nonzero(cv2.morphologyEx((change & flash).astype(np.uint8), cv2.MORPH_OPEN,
                                             np.ones((3, 3), np.uint8))) >= 22:
            return True
        grass = float(np.mean(RedZoneDetector.terrain_mask(hsv, self.world) > 0))
        vanished = grass > self.reference_grass + 0.24 and self._edges(patch) < self.reference_edges * 0.59
        self.gone_frames = self.gone_frames + 1 if vanished else 0
        return self.gone_frames >= 2
