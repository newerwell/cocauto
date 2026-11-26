"""Bounded ferry navigation shared by desktop and battle orchestration."""
from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path
import time
from typing import Callable
from uuid import uuid4

import cv2
import numpy as np

from core.device import BaseDeviceController
from core.device.base_controller import DeploymentTouch
from core.vision.wooden_boat import WoodenBoatDetector

logger = logging.getLogger("coc.tasks.travel")

TRAVEL_REASONS = {
    "loading": "地图仍在加载",
    "unknown_scene": "无法确认当前场景",
    "wrong_world": "当前画面为部落都城",
    "boat_unrecognized": "未看到完整的木船船身",
    "not_arrived": "点击后尚未抵达目标世界",
    "view_unconfirmed": "无法确认村庄已缩到全景并居中",
    "stopped": "已停止渡船切换",
}


class VillageTravel:
    _PANS = {
        "home": (((428, 491), (528, 323)), ((449, 492), (596, 347))),
        "night": (((890, 235), (768, 370)), ((861, 300), (686, 432))),
    }

    def __init__(self, device: BaseDeviceController, detector: WoodenBoatDetector, *,
                 stopped: Callable[[], bool] = lambda: False,
                 wait: Callable[[float], None] = time.sleep, evidence_dir: Path | None = None):
        self.device = device
        self.detector = detector
        self.stopped = stopped
        self.wait = wait
        self.evidence_dir = evidence_dir

    def toggle(self) -> dict[str, object]:
        frame = self.device.screencap()
        scene = self.detector.scene(frame)
        if scene not in self._PANS:
            return self._result(scene, None, scene, self._scene_reason(scene), [], None, frame)
        return self.go("night" if scene == "home" else "home", frame)

    def _panorama(self, world: str, frame: np.ndarray) -> np.ndarray | None:
        return self.device.normalize_map_view(frame, world=world,
                                              accepts=lambda image: self.detector.scene(image) == world,
                                              stopped=self.stopped)

    def collection_frame(self, world: str) -> np.ndarray | None:
        if self.stopped():
            return None
        return self._panorama(world, self.device.screencap())

    def go(self, target: str, frame: np.ndarray | None = None) -> dict[str, object]:
        source = "home" if target == "night" else "night"
        frame = self.device.screencap() if frame is None else frame
        scene = self.detector.scene(frame)
        touches: list[DeploymentTouch] = []
        evidence = (self.evidence_dir / f"{source}-to-{target}-{uuid4().hex[:10]}"
                    if self.evidence_dir is not None else None)
        if self.stopped():
            return self._result(source, target, scene, "stopped", touches, evidence, frame)
        if scene == target:
            return self._result(source, target, scene, "already_there", touches, None, frame)
        if scene != source:
            return self._result(source, target, scene, self._scene_reason(scene), touches, evidence, frame)

        self._save_frame(evidence, "source-before-zoom", frame)
        panorama = self._panorama(source, frame)
        if panorama is None:
            reason = "stopped" if self.stopped() else "view_unconfirmed"
            return self._result(source, target, scene, reason, touches, evidence, frame)
        frame = panorama
        self._save_frame(evidence, "source-panorama", frame)

        directions = iter(self._PANS[source])
        for _ in range(2):
            candidate = self.detector.find(frame, source)
            while candidate is None and not self.stopped():
                pan = next(directions, None)
                if pan is None:
                    return self._result(source, target, scene, "boat_unrecognized", touches, evidence, frame)
                self.device.swipe(*pan, duration_ms=440)
                self.wait(0.55)
                frame = self.device.screencap()
                scene = self.detector.scene(frame)
                if scene != source:
                    return self._result(source, target, scene, self._scene_reason(scene), touches, evidence, frame)
                candidate = self.detector.find(frame, source)

            if self.stopped():
                return self._result(source, target, scene, "stopped", touches, evidence, frame)
            # Validate both physical endpoints projected back to the matched hull.
            hull = candidate.hull
            sent = self.device.deployment_batch_tap(
                [candidate.point],
                lambda p: 0.28 <= (p[0] - hull.x) / hull.w <= 0.62
                and 0.57 <= (p[1] - hull.y) / hull.h <= 0.86,
                self.stopped,
            )
            if not sent:
                reason = "stopped" if self.stopped() else "boat_unrecognized"
                return self._result(source, target, scene, reason, touches, evidence, frame)
            touches.extend(sent)
            marked = frame.copy()
            cv2.drawMarker(marked, candidate.point, (245, 245, 245), cv2.MARKER_CROSS, 17, 2)
            self._save_frame(evidence, f"boarding-{len(touches)}", marked)
            logger.info("渡船登船点 %s -> %s，坐标 %s，分数 %.3f", source, target, candidate.point, candidate.score)
            scene, frame, reason = self._observe_arrival(source, target, frame)
            if reason == "arrived":
                self._save_frame(evidence, "arrival-before-zoom", frame)
                panorama = self._panorama(target, frame)
                if panorama is None:
                    reason = "stopped" if self.stopped() else "view_unconfirmed"
                else:
                    frame = panorama
            if reason != "not_arrived":
                return self._result(source, target, scene, reason, touches, evidence, frame, candidate.score)
            # Only the verified source village can reach this retry. The next
            # iteration must match a fresh hull on that frame before touching it.

        return self._result(source, target, scene, "not_arrived", touches, evidence, frame, candidate.score)

    def _observe_arrival(self, source: str, target: str, frame: np.ndarray) -> tuple[str, np.ndarray, str]:
        scene = source
        source_frames = 0
        target_frames = 0
        for _ in range(16):
            if self.stopped():
                return scene, frame, "stopped"
            self.wait(0.5)
            frame = self.device.screencap()
            scene = self.detector.scene(frame)
            if scene == "capital":
                return scene, frame, "wrong_world"
            target_frames = target_frames + 1 if scene == target else 0
            if target_frames >= 2:
                return scene, frame, "arrived"
            source_frames = source_frames + 1 if scene == source else 0
            if source_frames >= 5:
                return scene, frame, "not_arrived"
        return scene, frame, self._scene_reason(scene)

    @staticmethod
    def _scene_reason(scene: str) -> str:
        return {"capital": "wrong_world", "loading": "loading"}.get(scene, "unknown_scene")

    @staticmethod
    def _save_frame(folder: Path | None, label: str, frame: np.ndarray | None) -> None:
        if folder is None or frame is None or frame.size == 0:
            return
        try:
            folder.mkdir(parents=True, exist_ok=True)
            ok, data = cv2.imencode(".png", frame)
            if ok:
                (folder / f"{label}.png").write_bytes(data.tobytes())
        except (OSError, cv2.error) as exc:
            logger.warning("无法保存渡船画面 %s %s", folder, exc)

    def _result(self, source: str, target: str | None, scene: str, reason: str,
                touches: list[DeploymentTouch], folder: Path | None, frame: np.ndarray,
                score: float = 0.0) -> dict[str, object]:
        arrived = reason in {"arrived", "already_there"}
        result = {
            "arrived_home": target == "home" and arrived,
            "arrived_night_world": target == "night" and arrived,
            "source": source, "target": target, "scene": scene, "reason": reason,
            "boat_pt": touches[-1].logical if touches else None,
            "score": round(score, 3), "touches": [asdict(touch) for touch in touches],
            "evidence": str(folder) if folder else None,
        }
        logger.info("渡船结果 %s", result)
        if folder is not None:
            self._save_frame(folder, "arrival" if arrived else "final-scene", frame)
            try:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "travel.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as exc:
                logger.warning("无法保存渡船记录 %s %s", folder, exc)
        return result
