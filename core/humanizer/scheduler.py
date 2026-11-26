"""
人类作息与会话调度器 (Human Session & Sleep Scheduler)
防止 24 小时连续在线或固定周期卡点上线触发 Supercell 反作弊封禁。
"""
from __future__ import annotations
import random
import time
from datetime import datetime, time as dt_time
from typing import Tuple


class SessionScheduler:
    """管理单次连续打鱼会话时长上限与夜间仿生睡眠窗口"""

    def __init__(
        self,
        max_session_minutes: int = 75,
        rest_duration_minutes: int = 30,
        night_sleep_window: str = "01:30-07:30",
    ):
        self.base_max_session_min = max_session_minutes
        self.base_rest_min = rest_duration_minutes
        self.sleep_start, self.sleep_end = self._parse_window(night_sleep_window)
        self.session_start_ts: float = time.time()
        self.current_limit_sec: float = self._roll_next_session_limit()

    @staticmethod
    def _parse_window(window_str: str) -> Tuple[dt_time, dt_time]:
        try:
            start_s, end_s = window_str.strip().split("-")
            sh, sm = map(int, start_s.split(":"))
            eh, em = map(int, end_s.split(":"))
            return dt_time(sh, sm), dt_time(eh, em)
        except Exception:
            return dt_time(1, 30), dt_time(7, 30)

    def _roll_next_session_limit(self) -> float:
        """每次开启新会话时，在设定上限 ±20% 范围内随机生成本次会话的最大持续秒数"""
        jitter_factor = random.uniform(0.80, 1.20)
        return self.base_max_session_min * 60.0 * jitter_factor

    def reset_session(self):
        self.session_start_ts = time.time()
        self.current_limit_sec = self._roll_next_session_limit()

    def is_in_night_sleep_window(self, now: datetime | None = None) -> bool:
        """判断当前本地时间是否处于夜间睡眠区间"""
        now_t = (now or datetime.now()).time()
        if self.sleep_start <= self.sleep_end:
            return self.sleep_start <= now_t <= self.sleep_end
        # 跨午夜区间 (例如 23:30-07:00)
        return now_t >= self.sleep_start or now_t <= self.sleep_end

    def should_take_break(self) -> Tuple[bool, str, int]:
        """
        检查当前是否应当暂停挂机进入休息。
        返回: (是否需要休息, 原因说明, 建议休息秒数)
        """
        if self.is_in_night_sleep_window():
            rest_sec = int(random.uniform(1800, 3600))
            return True, "处于夜间仿生睡眠时间窗", rest_sec

        elapsed = time.time() - self.session_start_ts
        if elapsed >= self.current_limit_sec:
            rest_sec = int(self.base_rest_min * 60 * random.uniform(0.85, 1.25))
            return True, f"已连续运行 {int(elapsed // 60)} 分钟达到单次会话上限", rest_sec

        return False, "正常活跃期内", 0
