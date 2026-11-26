"""Execute one phase of visually confirmed troop deployment."""
from __future__ import annotations

from dataclasses import asdict, dataclass
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
from core.device.maatouch_client import VisionZoomAnalyzer
from core.vision.deployment_bar import Card, DeploymentBar, DeploymentBarReader, card_at
from core.vision.flank_damage import FlankWatch
from core.vision.red_zone_detector import DeploymentArea, Point, RedZoneDetector, is_battle_frame

logger = logging.getLogger("coc.tasks.deployment")
WINGS = ("left", "right")
SPELL_OFFSETS = ((0, 0), (-50, -30), (50, -30), (-50, 35), (50, 35), (0, -60),
                 (0, 60), (-90, 0), (90, 0), (-25, -15), (25, 20), (0, 5))


def tactical_hints(lanes: list[list[Point]], pattern: str) -> list[Point]:
    sections = sorted(zip(*lanes), key=lambda section: sum(x for x, _ in section))
    if pattern == "left":
        sections = sections[:2]
    elif pattern == "right":
        sections = sections[-2:][::-1]
    else:
        sections = sections[2:-2]
    return [point for section in sections for point in section]


def battlefield_view(frame: np.ndarray) -> np.ndarray:
    """Small grayscale map crop, excluding the controls and deployment bar."""
    gray = cv2.cvtColor(frame[130:465, 190:1040], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (96, 48), interpolation=cv2.INTER_AREA)


def view_difference(a: np.ndarray, b: np.ndarray) -> float:
    blocks = cv2.absdiff(a, b).reshape(3, 16, 4, 24)
    return float(np.median(blocks.mean(axis=(1, 3))))


def camera_aligned(a: np.ndarray, b: np.ndarray, max_difference: float) -> bool:
    if view_difference(a, b) > max_difference:
        return False
    (dx, dy), confidence = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    return confidence < 0.20 or max(abs(dx), abs(dy)) <= 0.8


@dataclass
class SlotDeployment:
    x: int
    kind: str
    initial: int | None = None
    remaining: int | None = None
    last_known_remaining: int | None = None
    deployed: int = 0
    skill: bool = False
    reason: str = "quantity_unknown"
    unit: str | None = None


@dataclass
class FlankDeployment:
    status: str = "pending"
    side: str | None = None
    slot: int | None = None
    point: Point | None = None
    before: int | None = None
    after: int | None = None
    reason: str | None = None


class DeploymentEvidence:
    def __init__(self, root: Path | None, world: str, phase: int):
        self.folder = root / f"{world}-{phase}-{uuid4().hex[:10]}" if root else None
        self.index = 0
        self._saved_confirmations: set[tuple[str | None, str | None]] = set()

    def record(self, fields: dict, before: np.ndarray | None, after: np.ndarray | None,
               touches: list[DeploymentTouch]) -> None:
        self.index += 1
        fields["touches"] = [asdict(touch) for touch in touches]
        logger.info("下兵确认 %s", fields)
        if self.folder is None:
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            with (self.folder / "attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"attempt": self.index, **fields}, ensure_ascii=False) + "\n")
            confirmation = (fields.get("region"), fields.get("kind"))
            first_confirmation = fields["reason"] == "confirmed" and confirmation not in self._saved_confirmations
            save_frames = first_confirmation or fields.get("diagnostic") or fields.get("region") in WINGS or fields["reason"] in {
                "no_progress", "quantity_unknown", "red_zone", "selection_unverified",
                "no_safe_points", "camera_unstable", "stopped", "natural_end",
            }
            if save_frames and is_battle_frame(before):
                self._write_frames(before, after, touches)
                if first_confirmation:
                    self._saved_confirmations.add(confirmation)
        except (OSError, cv2.error) as exc:
            logger.warning("无法保存下兵证据 %s: %s", self.folder, exc)

    def _write_frames(self, before: np.ndarray, after: np.ndarray | None,
                      touches: list[DeploymentTouch]) -> None:
        def save(label: str, frame: np.ndarray) -> None:
            ok, data = cv2.imencode(".png", frame)
            if ok:
                (self.folder / f"{self.index:03d}-{label}.png").write_bytes(data.tobytes())

        save("before", before)
        if is_battle_frame(after):
            save("after", after)
        marked = before.copy()
        for touch in touches:
            start = tuple(map(round, touch.logical_start))
            end = tuple(map(round, touch.logical_end))
            cv2.drawMarker(marked, start, (255, 255, 255), cv2.MARKER_CROSS, 13, 2)
            cv2.line(marked, start, end, (0, 255, 255), 2)
        save("touches", marked)


class BattleDeployment:
    def __init__(
        self, *, world: str, phase: int, device: BaseDeviceController,
        reader: DeploymentBarReader, detector: RedZoneDetector,
        side_order: list[str], focus: Callable[[str], None],
        hints: Callable[[str, str], list[Point]],
        result_screen: Callable[[np.ndarray], bool],
        battle_screen: Callable[[np.ndarray], bool],
        stopped: Callable[[], bool], deadline: float,
        now: Callable[[], float] = time.monotonic, wait: Callable[[float], None] = time.sleep,
        evidence_dir: Path | None = None,
    ):
        self.world, self.phase = world, phase
        self.device, self.reader, self.detector = device, reader, detector
        self._side_order = iter(side_order)
        self.focus, self.hints = focus, hints
        self.result_screen, self.battle_screen = result_screen, battle_screen
        self.stopped, self.deadline, self.now, self.wait = stopped, deadline, now, wait
        self.evidence = DeploymentEvidence(evidence_dir, world, phase)
        self.touch_count = 0
        self.candidate_counts = dict.fromkeys(("left", "right", "center", "spell"), 0)
        self._entry_diagnostic = False
        self.slots: dict[int, SlotDeployment] = {}
        self.side = None
        self.camera_id = 0
        self.last_frame: np.ndarray | None = None
        self.last_capture: np.ndarray | None = None
        self.halt_reason: str | None = None
        self.bar_supported = False
        self._blocked: list[Point] = []
        self._verified_slots: set[int] = set()
        self._cursor: dict[str, int] = {}
        self._previous_area: tuple[np.ndarray, DeploymentArea] | None = None
        self._pending: dict[int, tuple[int, int]] = {}
        self._camera_view: np.ndarray | None = None
        self._missing_battle_frames = 0
        self.flanks = {name: FlankDeployment() for name in WINGS}
        self.center_push: str | None = None
        self.center_wave: dict | None = None
        self.panorama_status = "pending"
        self._spell_view: np.ndarray | None = None
        self._flank_side: str | None = None
        self._flank_frames: dict[str, tuple[np.ndarray, Card]] = {}
        self._regions: dict[str, np.ndarray] = {}

    def running(self) -> bool:
        if self.stopped():
            self.halt_reason = "stopped"
        elif self.halt_reason is None and self.now() >= self.deadline:
            self.halt_reason = "timeout"
        return self.halt_reason is None

    def capture(self) -> np.ndarray | None:
        if not self.running():
            return None
        self.last_capture = self.device.screencap()
        if is_battle_frame(self.last_capture) and self.result_screen(self.last_capture):
            self.halt_reason = "natural_end"
            return None
        if not is_battle_frame(self.last_capture) or not self.battle_screen(self.last_capture):
            self._missing_battle_frames += 1
            if self.world == "night" and self._camera_view is not None and self._missing_battle_frames >= 2:
                self.halt_reason = "battlefield_changed"
            return None
        self._missing_battle_frames = 0
        if self.world == "night" and self._camera_view is not None:
            if view_difference(self._camera_view, battlefield_view(self.last_capture)) > 22:
                self.halt_reason = "battlefield_changed"
                return None
        self.last_frame = self.last_capture
        return self.last_frame

    def _inventory(self, frame: np.ndarray | None = None) -> DeploymentBar:
        if frame is None:
            frame = self.capture()
        bar = self.reader.scan(frame, self.slots)
        if not bar.supported and self.running():
            self.wait(0.12)
            retry = self.capture()
            bar = self.reader.scan(retry, self.slots)
            frame = retry if retry is not None else frame
        if frame is not None:
            self.bar_supported = bar.supported
        if not bar.supported:
            if not self._entry_diagnostic and self.evidence.index == 0:
                self._entry_diagnostic = True
                self.evidence.record({"world": self.world, "phase": self.phase, "reason":
                                      self.halt_reason or "bar_unrecognized", "diagnostic": True},
                                     self.last_capture, None, [])
            return bar
        for x, card in bar.cards.items():
            key = next((known for known in self.slots if abs(known - x) <= 6), x)
            slot = self.slots.setdefault(key, SlotDeployment(key, card.kind))
            if slot.kind == "unknown" and card.kind != "spent":
                slot.kind = card.kind
            slot.unit = getattr(card, "unit", None) or slot.unit
            if card.remaining is not None and card.remaining > 0 and slot.initial is None:
                slot.initial = slot.remaining = slot.last_known_remaining = card.remaining
                slot.reason = "not_deployed"
            elif card.remaining == 0 and slot.initial is None:
                slot.remaining = slot.last_known_remaining = 0
                slot.reason = "already_deployed"
                slot.skill = card.skill
        return bar

    def _needs_deployment(self, slot: SlotDeployment) -> bool:
        return slot.remaining != 0 and slot.kind in {"troop", "hero", "siege", "spell"}

    def _wait_for_camera(self, reference: np.ndarray, slot: SlotDeployment) -> tuple[np.ndarray | None, Card | None]:
        view = battlefield_view(reference)
        card = card_at(self.reader.scan(reference), slot.x)
        for _ in range(5):
            if not self.running():
                break
            self.wait(0.12)
            fresh = self.capture()
            if fresh is None:
                continue
            next_view = battlefield_view(fresh)
            bar = self.reader.scan(fresh)
            next_card = card_at(bar, slot.x) if bar.supported else None
            if (camera_aligned(view, next_view, 8) and card is not None and next_card is not None
                    and next_card.selected and card.selected and next_card.kind == slot.kind
                    and next_card.remaining == card.remaining and next_card.remaining is not None):
                self._camera_view = next_view
                return fresh, next_card
            view, card = next_view, next_card
        reason = "camera_unstable" if card and card.selected and card.remaining is not None else "selection_unverified"
        self._log(slot, reference, self.last_capture, [], None, None, reason)
        return None, None

    def _read_feedback(self, slot: SlotDeployment, before: Card, sent: int,
                       before_view: np.ndarray) -> tuple[int | None, bool, str]:
        until = self.now() + 2.5
        previous: tuple[int, bool] | None = None
        unchanged = False
        saw_decrement = False
        samples = 0
        # A single slow ADB screenshot can use the whole feedback interval.
        # Give the HUD a second read while the battle itself is still active.
        while self.running() and (samples < 2 or self.now() < until):
            samples += 1
            self.wait(0.11)
            frame = self.capture()
            bar = self.reader.scan(frame, self.slots)
            card = card_at(bar, slot.x) if bar.supported else None
            if (card is None or card.kind not in {slot.kind, "spent", "skill"}
                    or card.remaining is None or view_difference(before_view, battlefield_view(frame)) > 15
                    or (card.remaining > 0 and not card.selected)):
                previous = None
                unchanged = False
                continue
            observed = card.remaining, card.skill
            # A visual decrement can only be attributed to this attempt within
            # the number of touches dispatched for the same selected card.
            delta = before.remaining - card.remaining
            if not 0 <= delta <= sent:
                previous = None
                unchanged = False
                continue
            saw_decrement |= delta > 0
            if previous == observed:
                if delta > 0:
                    return card.remaining, card.skill, "confirmed"
                unchanged = True
            else:
                unchanged = False
            previous = observed
        if self.halt_reason:
            return None, False, self.halt_reason
        if unchanged and not saw_decrement:
            return before.remaining, False, "no_progress"
        return None, False, "quantity_unknown"

    def _log(self, slot: SlotDeployment, before: np.ndarray, after: np.ndarray | None,
             touches: list[DeploymentTouch], count: int | None, remaining: int | None,
             reason: str, tap_time: float | None = None, *, region: str | None = None,
             candidates: int | None = None) -> None:
        self.evidence.record({
            "world": self.world, "phase": self.phase, "camera": self.camera_id,
            "side": self.side, "slot": slot.x, "kind": slot.kind, "before": count,
            "after": remaining, "time": tap_time, "reason": reason, "region": region,
            "candidate_count": candidates, "touch_count": len(touches),
        }, before, after, touches)

    def _attempt(self, slot: SlotDeployment, pattern: str, limit: int) -> int:
        screen = self.capture()
        if pattern == "spell" and is_battle_frame(screen) and (
            self._spell_view is None or not camera_aligned(self._spell_view, battlefield_view(screen), 14)
        ):
            slot.reason = "camera_unstable"
            return 0
        bar = self.reader.scan(screen, self.slots)
        card = card_at(bar, slot.x) if bar.supported else None
        if card is None or card.remaining is None or card.kind not in {slot.kind, "spent", "skill"}:
            slot.remaining, slot.reason = None, "quantity_unknown"
            if is_battle_frame(screen):
                self._log(slot, screen, screen, [], slot.last_known_remaining, None, slot.reason)
            return 0
        if card.remaining <= 0:
            # A single background observation must not retire a known slot.
            self.wait(0.12)
            next_bar = self.reader.scan(self.capture(), self.slots)
            second = card_at(next_bar, slot.x) if next_bar.supported else None
            if second and second.remaining == 0 and second.kind == card.kind:
                self._reconcile(slot, 0)
                slot.remaining = slot.last_known_remaining = 0
                slot.reason = "confirmed"
                slot.skill = card.skill and second.skill
            return 0
        if slot.last_known_remaining is not None and card.remaining != slot.last_known_remaining:
            # Reconcile a delayed HUD change only after two fresh matching reads.
            slot.remaining, slot.reason = None, "quantity_unknown"
        if not card.selected or slot.reason == "no_safe_points":
            if not self.running():
                return 0
            self.device.tap(card.x, card.y, humanize=False)
        before, selected = self._wait_for_camera(screen, slot)
        if selected is None:
            slot.remaining = None
            slot.reason = self.halt_reason or "selection_unverified"
            return 0
        self._reconcile(slot, selected.remaining)
        slot.remaining = slot.last_known_remaining = selected.remaining
        if slot.initial is None:
            slot.initial = selected.remaining
        if selected.remaining == 0:
            slot.reason = "confirmed"
            return 0

        view = battlefield_view(before)
        if pattern == "spell":
            preview = self.detector.preview_toolbar_visible(cv2.cvtColor(before, cv2.COLOR_BGR2HSV))
            max_y = 507 if preview else 540
            accepts = lambda point: 175 < point[0] < 1030 and 80 < point[1] < max_y
            center = (VisionZoomAnalyzer.playable_center(before, self.world)
                      if self._spell_view is not None and camera_aligned(self._spell_view, view, 14) else None)
            points = [(center[0] + dx, center[1] + dy) for dx, dy in SPELL_OFFSETS] if center else []
        else:
            points, accepts = self._tactical_points(before, view, pattern)
        points = [point for point in points if accepts(point)]
        self.candidate_counts[pattern] += len(points)
        if not points:
            slot.reason = "no_safe_points"
            self._log(slot, before, before, [], selected.remaining, selected.remaining, slot.reason, region=pattern, candidates=0)
            return 0

        cursor = self._cursor.get(pattern, 0)
        size = min(limit, selected.remaining, 4 if slot.x in self._verified_slots else 1)
        batch = [points[(cursor + i) % len(points)] for i in range(size)]
        self._cursor[pattern] = cursor + size
        tap_time = time.time()
        sent = []
        for point in batch:
            dispatched = self.device.deployment_batch_tap([point], accepts, lambda: not self.running())
            self.touch_count += len(dispatched)
            sent.extend(dispatched)
            if dispatched:
                logger.info("实际触点 world=%s phase=%s region=%s slot=%s point=%s total=%d",
                            self.world, self.phase, pattern, slot.x, dispatched[0].logical, self.touch_count)
        if sent:
            remaining, skill, reason = self._read_feedback(slot, selected, len(sent), view)
        else:
            remaining, skill, reason = selected.remaining, False, self.halt_reason or "no_safe_points"
        after = self.last_capture
        if reason == "no_progress" and self._red_zone_message(after):
            reason = "red_zone"
            self._blocked.extend(touch.logical for touch in sent)
        slot.reason = reason
        slot.remaining = remaining
        if remaining is not None:
            slot.last_known_remaining = remaining
        elif sent and reason == "quantity_unknown" and pattern not in WINGS:
            self._pending[slot.x] = (selected.remaining, len(sent))
        confirmed = selected.remaining - remaining if remaining is not None else 0
        if confirmed:
            slot.deployed += confirmed
            slot.skill = skill
            self._verified_slots.add(slot.x)
        elif reason in {"no_progress", "red_zone", "quantity_unknown"}:
            self._verified_slots.discard(slot.x)
        if pattern in WINGS and sent:
            status = "confirmed" if confirmed else "unconfirmed" if remaining is None else "no_progress"
            self.flanks[pattern] = FlankDeployment(status, self.side, slot.x, sent[0].logical,
                                                    selected.remaining, remaining, reason)
            self._flank_frames[pattern] = before, selected
            if confirmed or remaining is None:
                self._flank_side = self.side
        self._log(slot, before, after, sent, selected.remaining, remaining, reason, tap_time,
                  region=pattern, candidates=len(points))
        return confirmed

    def _reconcile(self, slot: SlotDeployment, remaining: int) -> None:
        pending = self._pending.pop(slot.x, None)
        if pending:
            count, sent = pending
            if 0 < count - remaining <= sent:
                slot.deployed += count - remaining

    def _sample_region(self, area: DeploymentArea, region: str) -> tuple[list[Point], Callable]:
        eligible = self._regions[region]
        span = self._side_span or area.side_span
        if span:
            limits = {"left": (-0.03, 0.30), "center": (0.32, 0.68), "right": (0.70, 1.03)}
            lower, upper = limits[region]
            along = (np.arange(area.clearance.shape[1]) - span[0]) / (span[1] - span[0])
            eligible = np.broadcast_to((along >= lower) & (along <= upper), area.clearance.shape)
        if self._blocked:
            allowed = eligible.astype(np.uint8)
            for point in self._blocked:
                cv2.circle(allowed, point, 34, 0, cv2.FILLED)
            eligible = allowed != 0

        def accepts(point: tuple[float, float]) -> bool:
            return area.accepts(point) and bool(eligible[round(point[1]), round(point[0])])

        if span:
            fractions = {"left": (0.13, 0.23, 0.05), "center": (0.50, 0.40, 0.60, 0.35, 0.65),
                         "right": (0.87, 0.77, 0.95)}
            proposed = area.sample_side(span, eligible, fractions[region])
        else:
            proposed = area.sample(self.hints(self.side, region), eligible)
        points = []
        for point in proposed:
            touch = self.device.deployment_touch(point)
            if accepts(touch.logical_start) and accepts(touch.logical_end):
                points.append(point)
        return points, accepts

    def _tactical_points(self, frame: np.ndarray, view: np.ndarray, region: str) -> tuple[list[Point], Callable]:
        area = self.detector.deployment_area(frame, self.world, self.side)

        def candidates(area: DeploymentArea) -> tuple[list[Point], Callable]:
            points, accepts = self._sample_region(area, region)
            if region == "left" and self.flanks["left"].status != "confirmed":
                right = self._sample_region(area, "right")[0]
                center = self._sample_region(area, "center")[0]
                if self._previous_area is None:
                    logger.info("部署侧 %s %s，可见红线横向范围 %s，左/中/右候选 %d/%d/%d",
                                self.world, self.side, area.side_span, len(points), len(center), len(right))
                if not right or not center:
                    return [], accepts
            return points, accepts

        # After choosing both wings and the center, keep their screen-space
        # intervals fixed until the next camera focus.
        if self._side_span and area.side_span is None:
            points, accepts = [], lambda _: False
        else:
            points, accepts = candidates(area)
        if (not points and self._previous_area is None and self._flank_side is None
                and region in {"left", "center"}):
            # Camera drags can stop at the map limit before the requested
            # edge is centred. Try another visible outer face on this frame.
            visible = []
            for side in self.detector.CAMERA_FOCUS_SWIPES:
                if side == self.side:
                    continue
                alternative = self.detector.deployment_area(frame, self.world, side)
                if alternative.side_span is not None:
                    width = alternative.side_span[1] - alternative.side_span[0]
                    visible.append((width, side, alternative))
            for _, side, alternative in sorted(visible, key=lambda item: item[0], reverse=True):
                if all(self._sample_region(alternative, part)[0] for part in ("left", "center", "right")):
                    logger.info("移镜目标 %s，当前帧可部署侧 %s，横向范围 %s",
                                self.side, side, alternative.side_span)
                    self.side = side
                    area = alternative
                    points, accepts = candidates(area)
                    break
        if points:
            if self._side_span is None:
                self._side_span = area.side_span
            self._previous_area = view, area
        elif self._previous_area is not None:
            previous_view, previous_area = self._previous_area
            # Some games dim the selected red line after the first landing.
            # An aligned view can reuse that mask with fresh turf and HUD checks.
            aligned = camera_aligned(previous_view, view, 4)
            if not aligned and previous_area.side_span is not None:
                (dx, dy), confidence = cv2.phaseCorrelate(previous_view.astype(np.float32), view.astype(np.float32))
                aligned = (confidence >= 0.45 and max(abs(dx), abs(dy)) <= 0.50
                           and view_difference(previous_view, view) <= 13)
            if aligned:
                terrain = self.detector.current_terrain_accepts(frame, self.world, observed=bool(self._side_span))
                return candidates(DeploymentArea(np.minimum(previous_area.clearance, terrain.clearance),
                                                 previous_area.side_span))
        return points, accepts

    def _red_zone_message(self, frame: np.ndarray | None) -> bool:
        if not is_battle_frame(frame):
            return False
        text = self.reader.ocr.read_text_from_roi(frame, (195, 252, 350, 930)).casefold().replace(" ", "")
        return any(message in text for message in ("can'tdeployhere", "cannotdeployhere", "无法在此部署", "不能在这里部署", "无法在此处部署"))

    def _new_side(self, side: str) -> None:
        self.camera_id += 1
        self.side = side
        self._blocked.clear()
        self._cursor.clear()
        self._verified_slots.clear()
        self._previous_area = None
        self._side_span = None
        self._camera_view = None
        left, right = (np.mean(self.hints(side, name), axis=0) for name in WINGS)
        axis = right - left
        ys, xs = np.ogrid[:720, :1280]
        along = ((xs - left[0]) * axis[0] + (ys - left[1]) * axis[1]) / (axis @ axis)
        self._regions = {"left": along < 0.18, "right": along > 0.82,
                         "center": (along >= 0.18) & (along <= 0.82)}
        self.focus(side)

    def _flank_troops(self, bar: DeploymentBar) -> list[Card]:
        if not bar.supported:
            return []
        return sorted((card for card in bar.cards.values() if card.kind == "troop"
                       and (self.world == "night" or getattr(card, "unit", None) == "dragon")
                       and card.remaining is not None and card.remaining > 0), key=lambda card: card.x)

    def _resume_wing(self, name: str) -> bool:
        flank = self.flanks[name]
        before, card = self._flank_frames[name]
        slot = self.slots[flank.slot]
        remaining, skill, reason = self._read_feedback(slot, card, 1, battlefield_view(before))
        flank.after, flank.reason = remaining, reason
        slot.remaining, slot.reason = remaining, reason
        if remaining is not None:
            slot.last_known_remaining = remaining
        if reason == "confirmed":
            flank.status = "confirmed"
            self._flank_side = self.side
            slot.deployed += 1
            slot.skill = skill
            self._verified_slots.add(slot.x)
            self._log(slot, before, self.last_capture, [], card.remaining, remaining, reason, region=name)
            return True
        flank.status = "unconfirmed" if remaining is None else "no_progress"
        if remaining is None:
            self._flank_side = self.side
        return False

    def _deploy_wing(self, name: str, bar: DeploymentBar) -> bool:
        if self.flanks[name].status == "confirmed":
            return True
        for _ in range(2):
            if not self.running():
                return False
            flank = self.flanks[name]
            if flank.side == self.side and flank.status in {"unconfirmed", "no_progress"}:
                if self._resume_wing(name):
                    return True
                if self.flanks[name].status == "unconfirmed":
                    return False
                bar = self._inventory()
            troops = self._flank_troops(bar)
            if not troops:
                return False
            card = troops[0]
            slot = next(slot for slot in self.slots.values() if abs(slot.x - card.x) <= 6)
            if self._attempt(slot, name, 1):
                return True
        flank = self.flanks[name]
        return flank.side == self.side and flank.status == "unconfirmed" and self._resume_wing(name)

    def _wait_for_flank_damage(self) -> str:
        until = min(self.now() + 6, self.deadline)
        before, _ = self._flank_frames["left"]
        view = battlefield_view(before)
        watches = {name: FlankWatch.find(before, self.world, flank.point) for name, flank in self.flanks.items()}
        while self.running() and self.now() < until:
            frame = self.capture()
            if frame is not None and camera_aligned(view, battlefield_view(frame), 8):
                watches = {name: watch for name, watch in watches.items() if watch is None or not watch.damaged(frame)}
                if not watches:
                    return "both_flanks_engaged"
            self.wait(min(0.25, max(0.0, until - self.now())))
        return "flank_wait_timeout" if self.running() else "flank_wait_interrupted"

    def _deploy_center_wave(self, kinds: tuple[str, ...]) -> None:
        started = self.now()
        # Flank observation just captured this same view. A direct center
        # push still needs a new frame after the initial camera movement.
        fresh = self.last_frame if self._flank_side and self.last_frame is self.last_capture else None
        bar = self._inventory(fresh)
        before = self.last_frame
        if not bar.supported or before is None:
            return
        view = battlefield_view(before)
        points, accepts = self._tactical_points(before, view, "center")
        landings = [self.device.deployment_touch(point) for point in points]
        landings = [t for t in landings if accepts(t.logical_start) and accepts(t.logical_end)]
        self.candidate_counts["center"] += len(landings)
        if not landings:
            return

        # Keep each selection directly before that card's deployments. Slots in
        # skill state and cards with unreadable counts are excluded from the wave.
        cards: dict[int, Card] = {}
        actions: list[tuple[int, DeploymentTouch, bool]] = []
        cursor = self._cursor.get("center", 0)
        for kind in kinds:
            for slot in self.slots.values():
                if slot.kind != kind:
                    continue
                card = card_at(bar, slot.x)
                if card is None or card.kind != kind or not card.remaining or card.skill or slot.x in self._pending:
                    continue
                cards[slot.x] = card
                if actions or not card.selected:
                    actions.append((slot.x, self.device.deployment_touch((card.x, card.y)), False))
                for _ in range(card.remaining):
                    actions.append((slot.x, landings[cursor % len(landings)], True))
                    cursor += 1
        if not actions:
            return
        self._cursor["center"] = cursor

        sent: dict[int, list[DeploymentTouch]] = {x: [] for x in cards}
        attempted: set[int] = set()
        limit = self.device.DEPLOYMENT_SEQUENCE_LIMIT
        for offset in range(0, len(actions), limit):
            if not self.running():
                break
            if offset:
                fresh = self.capture()
                if fresh is None:
                    break
                if not camera_aligned(view, battlefield_view(fresh), 14):
                    self.halt_reason = "battlefield_changed" if self.world == "night" else "camera_unstable"
                    break
            chunk = actions[offset:offset + limit]
            attempted.update(x for x, _, deployment in chunk if deployment)
            dispatched = self.device.send_deployment_sequence([touch for _, touch, _ in chunk],
                                                               lambda: not self.running())
            for (x, _, deployment), touch in zip(chunk, dispatched):
                if deployment:
                    sent[x].append(touch)
                    self.touch_count += 1
            if len(dispatched) != len(chunk):
                break
        touch_count = sum(map(len, sent.values()))
        self.center_wave = {
            "touches": touch_count, "dispatch_seconds": round(self.now() - started, 3),
            "confirmed_seconds": None, "recovery_touches": 0,
        }
        self._confirm_center_wave(cards, sent, attempted)
        self.evidence.record({"world": self.world, "phase": self.phase, "region": "center",
                              "reason": "center_wave", "diagnostic": True, **self.center_wave},
                             before, self.last_capture, [touch for touches in sent.values() for touch in touches])
        self._recover_center(kinds, confirmed_only=True)
        if all(slot.remaining == 0 for slot in self.slots.values() if slot.kind in kinds):
            self.center_wave["confirmed_seconds"] = round(self.now() - started, 3)
        logger.info("中路总攻 world=%s phase=%d %s", self.world, self.phase, self.center_wave)

    def _confirm_center_wave(self, cards: dict[int, Card], sent: dict[int, list[DeploymentTouch]],
                             attempted: set[int]) -> None:
        previous = {}
        pending = set(attempted)
        for sample in range(4):
            if not pending or not self.running():
                break
            self.wait(0.10)
            frame = self.capture()
            bar = self.reader.scan(frame, self.slots)
            observed = {}
            for x in sorted(pending):
                original = cards[x]
                card = card_at(bar, x) if bar.supported else None
                if card is None or card.kind not in {original.kind, "spent", "skill"} or card.remaining is None:
                    continue
                if not 0 <= original.remaining - card.remaining <= len(sent[x]):
                    continue
                observed[x] = card.remaining, card.skill
                if previous.get(x) != observed[x] or card.remaining == original.remaining:
                    continue
                slot = self.slots[x]
                slot.remaining = slot.last_known_remaining = card.remaining
                slot.deployed += original.remaining - card.remaining
                slot.skill, slot.reason = card.skill, "confirmed"
                self._verified_slots.add(x)
                pending.remove(x)
            stable = {x for x in pending if x in previous and observed.get(x) == previous[x]}
            previous = observed
            if stable and sample == 3:
                red = self._red_zone_message(frame)
                for x in sorted(stable):
                    slot = self.slots[x]
                    slot.remaining = slot.last_known_remaining = cards[x].remaining
                    slot.reason = "red_zone" if red else "no_progress"
                    if red:
                        self._blocked.extend(touch.logical for touch in sent[x])
                pending.difference_update(stable)
        for x in pending:
            slot = self.slots[x]
            slot.remaining = None
            slot.reason = self.halt_reason or "quantity_unknown"
            self._pending[x] = cards[x].remaining, len(sent[x])

    def _recover_center(self, kinds: tuple[str, ...], *, confirmed_only: bool) -> None:
        before = self.touch_count
        for kind in kinds:
            self._deploy_slots((kind,), "center", confirmed_only=confirmed_only)
        self.center_wave["recovery_touches"] += self.touch_count - before

    def _deploy_slots(self, kinds: tuple[str, ...], pattern: str, *, confirmed_only: bool = False) -> None:
        failures = 0
        for slot in self.slots.values():
            if slot.kind not in kinds or (confirmed_only and slot.remaining is None):
                continue
            while self.running() and self._needs_deployment(slot) and failures < 2:
                progress = self._attempt(slot, pattern, 4)
                failures = 0 if progress else failures + 1
            if failures >= 2 and slot.reason in {"no_safe_points", "red_zone", "camera_unstable"}:
                break
            if failures >= 2:
                failures = 0

    def _army_deployed(self, bar: DeploymentBar, kinds: tuple[str, ...]) -> bool:
        army_kinds = (*kinds, "unknown")
        return bool(bar.supported and self.slots
                    and all(s.remaining == 0 for s in self.slots.values() if s.kind in army_kinds)
                    and all(c.remaining == 0 for c in bar.cards.values() if c.kind in army_kinds))

    def _prepare_spell_view(self, bar: DeploymentBar, kinds: tuple[str, ...]) -> bool:
        if not self.running() or not self._army_deployed(bar, kinds):
            return False
        if self.panorama_status != "pending":
            return self.panorama_status == "confirmed"
        before = self.last_frame
        self.panorama_status = "view_unconfirmed"

        def same_phase(frame: np.ndarray) -> bool:
            self.last_capture = frame
            if not is_battle_frame(frame):
                return False
            if self.result_screen(frame):
                self.halt_reason = "natural_end"
                return False
            return self.battle_screen(frame) and self._army_deployed(self.reader.scan(frame, self.slots), kinds)

        self._camera_view = None
        frame = self.device.normalize_map_view(before, world=self.world, accepts=same_phase,
                                               stopped=lambda: not self.running())
        if frame is not None and self.running():
            self.last_capture = self.last_frame = frame
            self._camera_view = self._spell_view = battlefield_view(frame)
            self._previous_area = None
            self._missing_battle_frames = 0
            self.camera_id += 1
            self.panorama_status = "confirmed"
            self._inventory(frame)
        self.evidence.record({"world": self.world, "phase": self.phase, "diagnostic": True,
                              "reason": "panorama_" + self.panorama_status}, before, frame, [])
        return self.panorama_status == "confirmed"

    def run(self) -> None:
        logger.info("开始部署 world=%s phase=%d evidence=%s", self.world, self.phase, self.evidence.folder)
        bar = self._inventory()
        logger.info("读取兵栏 world=%s phase=%d supported=%s slots=%s", self.world, self.phase,
                    bar.supported, [(s.x, s.kind, s.unit, s.remaining) for s in self.slots.values()])
        kinds = ("hero", "siege", "troop") if self.world == "night" else ("troop", "hero", "siege")
        if self.center_push is None and self._flank_side is None and bar.supported:
            if sum(card.remaining for card in self._flank_troops(bar)) < 2:
                self.wait(0.12)
                bar = self._inventory()
                if bar.supported and sum(card.remaining for card in self._flank_troops(bar)) < 2:
                    if self.world == "night":
                        unreadable = any(c.kind == "unknown" or (c.kind == "troop" and c.remaining is None)
                                         for c in bar.cards.values())
                        self.center_push = "troop_count_unconfirmed" if unreadable else "insufficient_troops"
                    elif any(c.kind == "unknown" or (c.kind == "troop" and
                             (getattr(c, "unit", None) is None or c.remaining is None)) for c in bar.cards.values()):
                        self.center_push = "dragon_unconfirmed"
                    else:
                        self.center_push = "insufficient_dragons"
        while self.running() and bar.supported:
            if self._army_deployed(bar, kinds):
                if self._prepare_spell_view(bar, kinds):
                    self._deploy_slots(("spell",), "spell")
                return
            if not any(s.kind in kinds and self._needs_deployment(s) for s in self.slots.values()):
                break
            side = self._flank_side or next(self._side_order, None)
            if side is None:
                break
            if self.capture() is None:
                break
            if self.side != side:
                self._new_side(side)
            if self.center_push is None:
                left = self._deploy_wing("left", bar)
                bar = self._inventory()
                if not left:
                    if self._flank_side:
                        break
                    continue
                if not self._deploy_wing("right", bar):
                    break
                if not self.running():
                    break
                if all(s.remaining == 0 for s in self.slots.values() if s.kind in (*kinds, "unknown")):
                    bar = self._inventory()
                    if self._army_deployed(bar, kinds):
                        continue
                self.center_push = self._wait_for_flank_damage()
            if self.center_wave is None:
                self._deploy_center_wave(kinds)
            elif self.running():
                self._recover_center(kinds, confirmed_only=False)
            bar = self._inventory()
            if self._army_deployed(bar, kinds):
                continue
            if self._flank_side:
                if self.center_wave is not None and self.running() and bar.supported:
                    self._recover_center(kinds, confirmed_only=False)
                    bar = self._inventory()
                    if self._army_deployed(bar, kinds):
                        continue
                break
        if self.running():
            self._inventory()

    def skills(self, frame: np.ndarray) -> list[Card]:
        bar = self.reader.scan(frame)
        if not bar.supported:
            return []
        return [card for slot in self.slots.values() if slot.skill
                for card in [card_at(bar, slot.x)] if card and card.skill]

    def summary(self, end_reason: str | None = None) -> dict:
        pending = [asdict(slot) for slot in self.slots.values() if slot.remaining != 0]
        reason = end_reason or self.halt_reason
        if not pending and self.slots and self.bar_supported:
            status = "completed"
        elif reason in {"stopped", "timeout"}:
            status = "interrupted"
        elif any(s.deployed for s in self.slots.values()):
            status = "partial"
        else:
            status = "unconfirmed"
        unsuccessful = next((s.reason for s in self.slots.values()
                             if s.reason not in {"confirmed", "already_deployed",
                                                 "not_deployed", "quantity_unknown"}), None)
        deployment_reason = (unsuccessful or ("bar_unrecognized" if not self.bar_supported else None)
                             or next((s.reason for s in self.slots.values() if s.remaining != 0), None))
        if not self.evidence.index and end_reason:
            self.evidence.record({"world": self.world, "phase": self.phase, "diagnostic": True,
                                  "reason": deployment_reason or reason or "no_deployment_attempt"},
                                 self.last_frame if self.last_frame is not None else self.last_capture, None, [])
        if end_reason:
            logger.info("部署结束 world=%s phase=%s status=%s reason=%s deployment_reason=%s "
                        "touches=%s candidates=%s evidence=%s", self.world, self.phase, status, reason,
                        deployment_reason, self.touch_count, self.candidate_counts, self.evidence.folder)
        return {
            "phase": self.phase, "status": status,
            "entered_battle": self.last_frame is not None,
            "touch_count": self.touch_count, "candidate_counts": self.candidate_counts.copy(),
            "deployment_reason": deployment_reason,
            "reason": "bar_unrecognized" if not self.bar_supported and reason in {None, "stagnation"}
            else reason or ("recovery_exhausted" if pending else "confirmed"),
            "remaining_slots": pending,
            "slots": [asdict(slot) for slot in self.slots.values()],
            "flanks": {name: asdict(flank) for name, flank in self.flanks.items()},
            "center_push": self.center_push,
            "center_wave": self.center_wave,
            "panorama_status": self.panorama_status,
            "evidence": str(self.evidence.folder) if self.evidence.folder and self.evidence.index else None,
        }
