"""Recognize the deployment bar and keep unreadable counts as None."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import cv2
import numpy as np

from config.paths import get_templates_dir

from .ocr_scanner import OcrScanner
from .red_zone_detector import is_battle_frame


@dataclass(frozen=True)
class Card:
    x: int
    y: int
    kind: str
    remaining: int | None
    selected: bool = False
    skill: bool = False
    unit: str | None = None


@dataclass
class DeploymentBar:
    cards: dict[int, Card]
    supported: bool


class DeploymentBarReader:
    _HOME_PORTRAITS = {
        "dragon": "troop",
        "siege_machine": "siege",
        "siege_machine_roof": "siege",
        "minion_prince": "hero",
        "barbarian_king": "hero",
        "archer_queen": "hero",
        "grand_warden": "hero",
        "earthquake_spell": "spell",
        "freeze_spell": "spell",
    }
    _NIGHT_PORTRAITS = {"night_battle_copter": "hero"}
    # The templates exclude counts, equipment badges and selection borders.
    # The siege crop also excludes the animated exclamation mark at its center.
    _SIEGE_PORTRAIT = "siege_machine"
    _PORTRAIT_TOP_OFFSET = 41
    _PORTRAIT_SCALES = (1.0, 1.04, 1.09)

    def __init__(self, ocr: OcrScanner, world: str):
        self.ocr = ocr
        self.world = world
        self._counts: dict[bytes, int | None] = {}
        self._portrait_kinds = self._HOME_PORTRAITS if world == "home" else self._NIGHT_PORTRAITS
        self._portraits = self._load_portraits(self._portrait_kinds)
        self._portrait_centers: dict[str, int] = {}
        self._health_slots: dict[int, tuple[str, str | None]] = {}

    @classmethod
    def _load_portraits(cls, kinds: dict[str, str]) -> dict[str, list[tuple[float, np.ndarray]]]:
        folder = get_templates_dir() / "battle" / "deployment_cards"
        portraits = {}
        for unit in kinds:
            path = folder / f"{unit}.png"
            try:
                data = path.read_bytes()
                portrait = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            except (OSError, cv2.error):
                continue
            width = 23 if unit == cls._SIEGE_PORTRAIT else 46
            if portrait is not None and portrait.shape == (43, width):
                portraits[unit] = [
                    (scale, cv2.resize(portrait, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR))
                    for scale in cls._PORTRAIT_SCALES
                ]
        return portraits

    def _locate_portraits(self, frame: np.ndarray) -> list[tuple[str, int, int, int]]:
        if not self._portraits:
            return []
        top = 595
        strip = cv2.cvtColor(frame[top:695], cv2.COLOR_BGR2GRAY)
        screen_width = strip.shape[1]
        locations: list[tuple[float, str, int, int, int]] = []
        for unit, variants in self._portraits.items():
            best = None
            ranges = [(0, screen_width)]
            if unit in self._portrait_centers:
                old_x = self._portrait_centers[unit]
                ranges.insert(0, (max(0, old_x - 75), min(screen_width, old_x + 75)))
            for left, right in ranges:
                for scale, portrait in variants:
                    result = cv2.matchTemplate(strip[:, left:right], portrait, cv2.TM_CCOEFF_NORMED)
                    _, score, _, (x, y) = cv2.minMaxLoc(result)
                    if score < 0.85 or (best is not None and score <= best[0]):
                        continue
                    x_offset = round(32 * scale) if unit == self._SIEGE_PORTRAIT else portrait.shape[1] // 2
                    center = left + x + x_offset
                    top_offset = 33 if unit == "night_battle_copter" else self._PORTRAIT_TOP_OFFSET
                    card_top = top + y - round(top_offset * scale)
                    width = round(89 * scale)
                    min_top = 567 if self._portrait_kinds[unit] in {"hero", "siege"} else 581
                    if min_top <= card_top <= 601 and width // 2 <= center < screen_width - width // 2:
                        best = (score, unit, center, card_top, width)
                if best is not None:
                    break
            if best is not None:
                locations.append(best)
        found = []
        for _, unit, x, y, width in sorted(locations, reverse=True):
            if not any(abs(x - entry[1]) < 35 for entry in found):
                found.append((unit, x, y, width))
        self._portrait_centers = {unit: x for unit, x, _, _ in found}
        return found

    def _read_portrait_card(self, frame: np.ndarray, hsv: np.ndarray, unit: str,
                            center: int, y: int, width: int) -> Card:
        left = center - width // 2
        right = left + width
        kind = self._portrait_kinds[unit]
        tap_y = 645 if self.world == "home" else 622
        selected = self._white_selection(hsv, left, y, width, 711 - y)
        art = hsv[y + 35 - 460:679 - 460, center - 16:center + 16]
        saturated = np.mean(art[:, :, 1] >= 65)
        if kind in {"troop", "spell", "siege"}:
            dim = saturated < 0.22
            count = self._read_quantity(frame, (y + 1, y + 25, center - 6, right - 2), dim=dim)
            if count == 0 or kind != "siege":
                return Card(center, tap_y, "spent" if count == 0 else kind, count, selected, unit=unit)

        # A narrow HP strip is evidence of deployment. A dim or shifted card
        # without that strip remains unknown until another frame confirms it.
        strip = hsv[y - 14 - 460:y + 14 - 460, left + 8:right - 8]
        if self._health_bar(strip):
            return Card(center, tap_y, kind, 0, selected, kind == "hero", unit)
        remaining = 1 if saturated >= 0.22 and (y >= 590 or selected) else None
        return Card(center, tap_y, kind, remaining, selected, unit=unit)

    @staticmethod
    def _health_bar(strip: np.ndarray) -> bool:
        green = cv2.inRange(strip, (38, 140, 145), (85, 255, 255))
        contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = strip.shape[1] // 2
        for contour in contours:
            x, _, w, h = cv2.boundingRect(contour)
            if (x <= center - 3 and x + w >= center + 3 and w >= 13 and 3 <= h <= 11
                    and cv2.contourArea(contour) >= 0.65 * w * h):
                return True
        return False

    def _read_quantity(self, frame: np.ndarray, roi: tuple[int, int, int, int], *, dim: bool = False) -> int | None:
        y, bottom, x, right = roi
        key = bytes((int(dim),)) + frame[y:bottom, x:right].tobytes()
        if key not in self._counts:
            text = self.ocr.read_text_from_roi(frame, roi).strip().replace(" ", "")
            if dim and re.fullmatch(r"[xX×][oO]|[oO][xX×]", text):
                text = "x0"
            match = re.fullmatch(r"[xX×](\d{1,3})|(\d{1,3})[xX×]", text)
            if len(self._counts) >= 128:
                self._counts.clear()
            self._counts[key] = int(match[1] or match[2]) if match else None
        return self._counts[key]

    @staticmethod
    def _boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame[460:719, 4:1275], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 115)
        candidates = []
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        joined, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in (*contours, *joined):
            x, y, w, h = cv2.boundingRect(contour)
            if not (52 <= w <= 132 and 65 <= h <= 155):
                continue
            left = np.mean(np.any(edges[y+9:y+h-9, x:x+5], axis=1))
            right = np.mean(np.any(edges[y+9:y+h-9, x+w-5:x+w], axis=1))
            bottom = np.mean(np.any(edges[y+h-5:y+h, x+8:x+w-8], axis=0))
            top = np.mean(np.any(edges[y:y+6, x+8:x+w-8], axis=0))
            closed_box = cv2.contourArea(contour) >= 0.73 * w * h
            if closed_box or (min(left, right) >= 0.60 and max(bottom, top) >= 0.60):
                candidates.append((x + 4, y + 460, w, h))

        boxes = []
        for box in sorted(candidates, key=lambda r: (r[1], abs(r[2] - 88))):
            x, y, w, h = box
            if any(abs(x + w / 2 - bx - bw / 2) < 19 and abs(y + h - by - bh) < 22
                   for bx, by, bw, bh in boxes):
                continue
            boxes.append(box)
        return sorted(boxes)

    def scan(self, frame: np.ndarray | None, known_slots: Iterable[int] = ()) -> DeploymentBar:
        if not is_battle_frame(frame):
            return DeploymentBar({}, False)
        hsv = cv2.cvtColor(frame[460:], cv2.COLOR_BGR2HSV)
        cards = {
            x: self._read_portrait_card(frame, hsv, unit, x, y, width)
            for unit, x, y, width in self._locate_portraits(frame)
        }
        portrait_centers = tuple(cards)
        unsupported = False
        boxes = self._boxes(frame)
        if self.world == "night":
            # Terrain sometimes joins a dark card outline through the transparent bar.
            # The blue count header still spans the full camp even if only the
            # interior portrait contour survives. Use one box per visible header.
            headers = cv2.inRange(hsv[570-460:624-460], (95, 120, 142), (126, 255, 255))
            contours, _ = cv2.findContours(headers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, top, w, h = cv2.boundingRect(contour)
                if not (80 <= w <= 101 and 19 <= h <= 48):
                    continue
                y = 570 + top - 2
                header = hsv[y+3-460:y+24-460, x+w//2:x+w]
                if np.count_nonzero((header[:, :, 1] < 45) & (header[:, :, 2] > 208)) < 22:
                    continue
                boxes = [box for box in boxes if box[1] < 560 or abs(box[0] + box[2] / 2 - x - w / 2) >= 18]
                boxes.append((x - 1, y, w + 1, 711 - y))
        for x, y, w, h in boxes:
            if y + h < 580:
                continue
            # Some dark card outlines merge into the background at the top.
            # A remaining portrait frame still anchors the top badge search.
            if 609 <= y <= 630 and y + h >= 697 and w >= 65:
                h += y - 594
                y = 594
            header = hsv[y+5-460:y+22-460, x+5:x+w-5]
            left_header = header[:, :int(w * 0.41)]
            bright = (header[:, :, 1] < 45) & (header[:, :, 2] > 208)
            badge_visible = np.count_nonzero(bright[:, int(w * 0.43):]) >= 22
            blue = np.mean(cv2.inRange(left_header, (95, 80, 110), (118, 255, 255)) > 0)
            purple_mask = cv2.inRange(header, (119, 95, 105), (146, 255, 255)) > 0
            purple = np.mean(purple_mask[:, :int(w * 0.41)])
            gray = np.mean((header[:, :, 1] < 30) & (header[:, :, 2] > 38))
            # A solid, horizontally aligned green strip is the deployed hero's HP.
            strip = hsv[y+3-460:y+15-460, x+6:x+w-6]
            green = cv2.inRange(strip, (38, 140, 145), (85, 255, 255)) > 0
            hp = np.count_nonzero(np.mean(green, axis=0) > 0.5) >= 13
            lvl = hsv[y+h-31-460:y+h-7-460, x+4:x+32]
            level_badge = (np.mean(lvl[:, :, 2] < 90) >= 0.25
                           and np.count_nonzero((lvl[:, :, 1] < 48) & (lvl[:, :, 2] > 180)) >= 12)

            lower_row = 560 <= y <= 635 and y + h >= 679
            if not lower_row:
                unsupported |= (blue >= 0.45 or purple >= 0.38) and badge_visible and level_badge
                continue
            selected = self._white_selection(hsv, x, y, w, h)
            center = x + w // 2
            if any(abs(center - portrait_center) < 35 for portrait_center in portrait_centers):
                continue
            roi = (max(y + 1, 0), y + 24, x + int(w * 0.40), x + w - 3)
            dim_card = gray >= 0.66
            count = self._read_quantity(frame, roi, dim=dim_card) if badge_visible or dim_card else None
            switch = hsv[y+h-37-460:y+h-5-460, x+w-38:x+w-7]
            switch_green = cv2.inRange(switch, (40, 145, 150), (79, 255, 255))
            switch_white = (switch[:, :, 1] < 45) & (switch[:, :, 2] > 215)
            has_switch = np.mean(switch_green > 0) >= 0.34 and np.count_nonzero(switch_white) >= 16
            kind, remaining, skill = "unknown", None, False
            if dim_card and count == 0:
                kind, remaining = "spent", 0
            elif hp and y <= 590 and level_badge and count is None:
                kind = "skill" if self.world == "night" and blue >= 0.35 else "hero"
                remaining, skill = 0, True
            elif self.world == "home" and has_switch and count == 1 and blue >= 0.42:
                kind, remaining = "siege", count
            elif blue >= 0.52:
                kind, remaining = "troop", count
            elif self.world == "home" and purple >= 0.48:
                kind, remaining = "spell", count
            elif purple >= 0.14 and not badge_visible and level_badge:
                kind, remaining = "hero", 1
            elif self.world == "night" and purple >= 0.32 and count is not None:
                kind, remaining = "troop", count
            elif not (badge_visible or count is not None or level_badge or blue > 0.30 or purple > 0.22):
                continue
            cards[center] = Card(center, 622 if self.world == "night" else 645, kind, remaining, selected, skill)
        bar = DeploymentBar(cards, False)
        for card in cards.values():
            if card.kind in {"hero", "siege"} and not any(abs(card.x - x) <= 12 for x in self._health_slots):
                self._health_slots[card.x] = card.kind, card.unit
        # Active abilities can overlay most of a previously recognized portrait.
        # Their fixed slot still contains a solid health bar above the picture.
        for x, (kind, unit) in self._health_slots.items():
            card = card_at(bar, x)
            if card is not None and card.remaining == 0 and card.kind in {kind, "spent"}:
                continue
            strip = hsv[568-460:595-460, max(0, x-34):x+34]
            if self._health_bar(strip):
                if card:
                    cards.pop(card.x)
                cards[x] = Card(x, 622 if self.world == "night" else 645, kind, 0, skill=kind == "hero", unit=unit)
        for x in known_slots:
            known = card_at(bar, x)
            if known is not None and known.remaining is not None:
                continue
            # The black top border can disappear on depletion. A previously
            # located slot can still be confirmed by an explicit dimmed x0.
            for y in (595, 587, 579):
                header = hsv[y+4-460:y+20-460, max(0, x-34):x+38]
                if np.mean((header[:, :, 1] < 30) & (header[:, :, 2] > 38)) < 0.74:
                    continue
                if self._read_quantity(frame, (y-1, y+23, max(0, x-8), x+42), dim=True) == 0:
                    if known is not None:
                        cards.pop(known.x)
                    cards[x] = Card(x, 622 if self.world == "night" else 645, "spent", 0,
                                    unit=known.unit if known else None)
                    break
        bar.cards = dict(sorted(cards.items()))
        bar.supported = bool(cards) and not unsupported
        return bar

    @staticmethod
    def _white_selection(hsv: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
        strip = hsv[y+16-460:y+h-10-460, x+w-6:x+w+2]
        white = (strip[:, :, 1] < 44) & (strip[:, :, 2] > 228)
        return bool(np.any(np.mean(white, axis=0) >= 0.65))


def card_at(bar: DeploymentBar, x: int) -> Card | None:
    return next((card for center, card in bar.cards.items() if abs(center - x) <= 6), None)
