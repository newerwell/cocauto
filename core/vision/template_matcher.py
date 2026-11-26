"""
多尺度模板匹配与非极大值抑制 (Multi-Scale Template Matcher & NMS)
用于识别 UI 按钮（进攻、搜索对手、下一个、返回家乡、增援、工人建议列表）以及村庄资源气泡。
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class MatchResult:
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class TemplateMatcher:
    """支持多尺度缩放与 NMS 去重的 OpenCV 模板匹配引擎"""

    def __init__(self, assets_dir: str | Path | None = None):
        from config.paths import get_templates_dir
        self.assets_dir = Path(assets_dir) if assets_dir else get_templates_dir()
        self._cache: dict[str, np.ndarray] = {}

    def load_template(self, relative_path: str) -> Optional[np.ndarray]:
        if relative_path in self._cache:
            return self._cache[relative_path]
        full_path = self.assets_dir / relative_path
        if not full_path.exists():
            return None
        img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
        if img is not None:
            self._cache[relative_path] = img
        return img

    def find_one(
        self,
        screen: np.ndarray,
        template: np.ndarray | str,
        threshold: float = 0.80,
        roi: Optional[Tuple[int, int, int, int]] = None,
        scales: Tuple[float, ...] = (1.0, 0.95, 1.05),
    ) -> Optional[MatchResult]:
        """
        在屏幕（或指定 ROI: y1, y2, x1, x2）中查找最佳匹配的单个目标。
        支持多尺度匹配以应对不同模拟器或手机微小的 UI DPI 差异。
        """
        tpl = self.load_template(template) if isinstance(template, str) else template
        if tpl is None:
            return None

        offset_x, offset_y = 0, 0
        search_img = screen
        if roi is not None:
            y1, y2, x1, x2 = roi
            search_img = screen[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1

        sh, sw = search_img.shape[:2]
        th0, tw0 = tpl.shape[:2]
        best_match: Optional[MatchResult] = None

        for scale in scales:
            tw = max(4, int(tw0 * scale))
            th = max(4, int(th0 * scale))
            if tw > sw or th > sh:
                continue
            resized_tpl = tpl if scale == 1.0 else cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(search_img, resized_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val >= threshold:
                if best_match is None or max_val > best_match.score:
                    best_match = MatchResult(
                        x=max_loc[0] + offset_x,
                        y=max_loc[1] + offset_y,
                        w=tw,
                        h=th,
                        score=float(max_val),
                    )
        return best_match

    def find_all(
        self,
        screen: np.ndarray,
        template: np.ndarray | str,
        threshold: float = 0.78,
        max_results: int = 15,
        iou_threshold: float = 0.35,
    ) -> List[MatchResult]:
        """查找屏幕上所有匹配目标（例如所有金币/圣水收菜气泡），并执行 NMS 非极大值抑制去重"""
        tpl = self.load_template(template) if isinstance(template, str) else template
        if tpl is None:
            return []

        th, tw = tpl.shape[:2]
        sh, sw = screen.shape[:2]
        if tw > sw or th > sh:
            return []

        res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        scores = res[ys, xs]

        if len(scores) == 0:
            return []

        # 按置信度从高到低排序并做 NMS
        order = np.argsort(scores)[::-1]
        results: List[MatchResult] = []

        for idx in order:
            cx, cy = int(xs[idx]), int(ys[idx])
            sc = float(scores[idx])
            candidate = MatchResult(x=cx, y=cy, w=tw, h=th, score=sc)

            overlap = False
            for kept in results:
                if self._compute_iou(candidate, kept) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                results.append(candidate)
                if len(results) >= max_results:
                    break

        return results

    @staticmethod
    def _compute_iou(a: MatchResult, b: MatchResult) -> float:
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w)
        y2 = min(a.y + a.h, b.y + b.h)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = a.w * a.h
        area_b = b.w * b.h
        return inter / float(area_a + area_b - inter)
