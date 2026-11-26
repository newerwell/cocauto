"""
死鱼（采集器满溢）与英雄血条视觉检测器 (Dead-Base & Hero Health Checker)
1. 死鱼判定：检测外围圣水收集器紫红色满液位特征斑块 / 模板匹配数量，区分“资源在采集器的死鱼”与“资源在罐子里的活鱼”。
2. 英雄血条监控：实时监测底部选兵栏英雄头像上方的绿色血条比例，低于阈值时触发自动开大招。
"""
from __future__ import annotations
from typing import Tuple, List
import cv2
import numpy as np

from .template_matcher import TemplateMatcher


class DeadBaseChecker:
    """死鱼采集器满度与战斗英雄生命值视觉分析器"""

    def __init__(self, matcher: TemplateMatcher | None = None):
        self.matcher = matcher or TemplateMatcher()

    def count_full_elixir_collectors(self, screen_bgr: np.ndarray) -> int:
        """
        统计战场上处于“满溢”状态的圣水收集器数量：
        优先使用 templates/battle/full_collector.png 模板匹配；
        若模板文件尚未截取，则使用圣水收集器特有的高饱和度紫品红色液柱 (HSV H∈[142, 165], S>=140, V>=140) 连通域斑块检测。
        """
        tpl_matches = self.matcher.find_all(screen_bgr, "battle/full_collector.png", threshold=0.76)
        if tpl_matches:
            return len(tpl_matches)

        # HSV 紫品红圣水满槽特征斑块分析 (限制在战场中部区域)
        roi = screen_bgr[80:560, 120:1160]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        elixir_mask = cv2.inRange(hsv, np.array([142, 145, 145]), np.array([166, 255, 255]))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(elixir_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 过滤出面积符合满采集器储液柱大小的斑块 (35 ~ 450 平方像素)
        valid_blobs = [c for c in contours if 35 <= cv2.contourArea(c) <= 450]
        return len(valid_blobs)

    def is_dead_base(self, screen_bgr: np.ndarray, min_full_collectors: int = 3) -> bool:
        """判定当前敌方村庄是否为死鱼"""
        return self.count_full_elixir_collectors(screen_bgr) >= min_full_collectors

    @staticmethod
    def check_hero_hp_ratio(screen_bgr: np.ndarray, slot_box: Tuple[int, int, int, int]) -> float:
        """
        检测底部英雄槽位 (x1, y1, x2, y2) 顶部生命值条的剩余绿色血量比例 [0.0 ~ 1.0]。
        当英雄受伤时，血条右侧会变暗/变空，左侧剩余亮绿色/黄绿色 (HSV H∈[32, 88], S>=110, V>=120)。
        """
        x1, y1, x2, y2 = slot_box
        # 英雄血条位于头像框顶部区域
        bar_y1 = max(0, y1)
        bar_y2 = min(screen_bgr.shape[0], y1 + max(6, (y2 - y1) // 6))
        bar_crop = screen_bgr[bar_y1:bar_y2, max(0, x1 + 4):min(screen_bgr.shape[1], x2 - 4)]
        if bar_crop.size == 0:
            return 1.0

        hsv = cv2.cvtColor(bar_crop, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, np.array([32, 110, 120]), np.array([88, 255, 255]))
        # 按列统计：只要某一列存在绿色像素即视为该刻度有血
        col_has_hp = np.any(green_mask > 0, axis=0)
        return float(np.mean(col_has_hp))
