"""
拟人化空间与时间抖动引擎 (Spatial & Temporal Jitter Engine)
提供二维高斯随机坐标偏移、对数正态分布反应延迟、以及三阶贝塞尔曲线滑动轨迹生成，
用于对抗 Supercell 的固定坐标点击与机械节拍器风控检测。
"""
from __future__ import annotations
import math
import random
import time
from typing import List, Tuple


class HumanJitter:
    """仿生学空间坐标抖动与曲线轨迹生成器"""

    def __init__(
        self,
        sigma_px: float = 6.0,
        max_offset_px: int = 15,
        min_delay_sec: float = 0.22,
        max_delay_sec: float = 0.65,
        screen_width: int = 1280,
        screen_height: int = 720,
    ):
        self.sigma_px = sigma_px
        self.max_offset_px = max_offset_px
        self.min_delay_sec = min_delay_sec
        self.max_delay_sec = max_delay_sec
        self.screen_width = screen_width
        self.screen_height = screen_height

    def jitter_point(
        self,
        x: int,
        y: int,
        custom_sigma: float | None = None,
        custom_max_offset: int | None = None,
    ) -> Tuple[int, int]:
        """
        对目标坐标 (x, y) 施加二维独立高斯分布偏移，并限制在最大偏移半径与屏幕安全边界内。
        """
        sigma = custom_sigma if custom_sigma is not None else self.sigma_px
        max_off = custom_max_offset if custom_max_offset is not None else self.max_offset_px

        dx = int(max(-max_off, min(max_off, random.gauss(0, sigma))))
        dy = int(max(-max_off, min(max_off, random.gauss(0, sigma))))

        safe_x = max(8, min(self.screen_width - 8, x + dx))
        safe_y = max(8, min(self.screen_height - 8, y + dy))
        return safe_x, safe_y

    def random_press_duration_ms(self, base_ms: int = 70) -> int:
        """生成模拟人类手指真实接触屏幕的按压时长 (45ms ~ 125ms)"""
        val = int(random.gauss(base_ms, 18))
        return max(42, min(135, val))

    def sleep_human_delay(self, multiplier: float = 1.0) -> float:
        """
        按对数正态分布 (Log-Normal) 休眠一段仿生反应时间，返回实际休眠秒数。
        相比均匀分布，对数正态分布具有人类反应特有的“偏态长尾”特征（偶尔会有略微停顿）。
        """
        mid = (self.min_delay_sec + self.max_delay_sec) / 2.0
        raw = random.lognormvariate(math.log(max(0.05, mid)), 0.35)
        delay = max(self.min_delay_sec, min(self.max_delay_sec * 1.6, raw)) * multiplier
        time.sleep(delay)
        return delay

    def generate_bezier_curve(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        num_steps: int = 16,
    ) -> List[Tuple[int, int]]:
        """
        生成连接起点和终点的三阶贝塞尔曲线 (Cubic Bézier Curve) 采样点序列，
        并加入 Ease-In-Out (先加速后减速) 的非线性时间步长，彻底消除直线匀速滑动特征。
        """
        x0, y0 = start
        x3, y3 = end
        dist = math.hypot(x3 - x0, y3 - y0)
        arc_offset = max(15.0, dist * 0.18)

        # 随机生成两个偏离直线的控制点 P1, P2
        angle = math.atan2(y3 - y0, x3 - x0) + random.choice([-1, 1]) * random.uniform(0.25, 0.65)
        x1 = x0 + (x3 - x0) * 0.33 + math.cos(angle) * random.uniform(5, arc_offset)
        y1 = y0 + (y3 - y0) * 0.33 + math.sin(angle) * random.uniform(5, arc_offset)
        x2 = x0 + (x3 - x0) * 0.66 + math.cos(angle) * random.uniform(5, arc_offset)
        y2 = y0 + (y3 - y0) * 0.66 + math.sin(angle) * random.uniform(5, arc_offset)

        points: List[Tuple[int, int]] = []
        for i in range(num_steps):
            linear_t = i / max(1, num_steps - 1)
            # Cosine Ease-In-Out 变换：两端点密集(速度慢)，中间点稀疏(速度快)
            t = 0.5 * (1.0 - math.cos(math.pi * linear_t))
            u = 1.0 - t

            bx = (u**3) * x0 + 3 * (u**2) * t * x1 + 3 * u * (t**2) * x2 + (t**3) * x3
            by = (u**3) * y0 + 3 * (u**2) * t * y1 + 3 * u * (t**2) * y2 + (t**3) * y3

            px = max(5, min(self.screen_width - 5, int(round(bx))))
            py = max(5, min(self.screen_height - 5, int(round(by))))
            points.append((px, py))

        return points
