"""
设备控制抽象基类 (Base Device Controller)
定义手机端（本机无线调试 / Shizuku / 无障碍）与电脑端（安卓模拟器 ADB）统一遵循的屏幕抓取与触控协议。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import time
from typing import Callable, Tuple, List
import numpy as np


@dataclass(frozen=True)
class DeploymentTouch:
    logical: tuple[int, int]
    physical_start: tuple[int, int]
    physical_end: tuple[int, int]
    logical_start: tuple[float, float]
    logical_end: tuple[float, float]
    dispatched_at: float | None = None


class BaseDeviceController(ABC):
    """跨平台 Android 设备抽象接口 (统一工作于 1280x720 逻辑坐标系)"""

    STANDARD_WIDTH: int = 1280
    STANDARD_HEIGHT: int = 720
    DEPLOYMENT_SEQUENCE_LIMIT: int = 1

    @abstractmethod
    def connect(self) -> bool:
        """连接或初始化目标设备"""
        pass

    @abstractmethod
    def screencap(self) -> np.ndarray:
        """获取当前画面并归一化为 1280x720 BGR 格式的 numpy.ndarray"""
        pass

    @abstractmethod
    def tap(self, x: int, y: int, humanize: bool = True) -> Tuple[int, int]:
        """点击屏幕逻辑坐标 (x, y)，返回实际点击的物理/抖动后坐标"""
        pass

    @abstractmethod
    def swipe(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        duration_ms: int = 350,
        use_bezier: bool = True,
    ) -> None:
        """从 start 滑动到 end"""
        pass

    @abstractmethod
    def pinch_zoom_out(self) -> None:
        """双指捏合将游戏镜头缩放到最远视野 (Zoom Out)，并对齐基准视角"""
        pass

    def normalize_map_view(
        self, frame: np.ndarray, *, world: str,
        accepts: Callable[[np.ndarray], bool], stopped: Callable[[], bool],
    ) -> np.ndarray | None:
        """Return a newly observed map after normalizing this device's camera."""
        if stopped() or not accepts(frame):
            return None
        self.pinch_zoom_out()
        if stopped():
            return None
        observed = self.screencap()
        return observed if accepts(observed) else None

    def fast_batch_tap(
        self,
        points: List[Tuple[int, int]],
        interval_ms: int = 38,
        humanize: bool = True,
    ) -> int:
        """
        高速批量连续点击一组逻辑坐标点 (用于战场一字划快速连下多兵/多法术)。
        基类默认回退为逐点调用 tap(x, y, humanize=False)；
        AdbDeviceController 会将其重写为单次 `adb shell sh` 内核级批量点击脚本，耗时降低 90%。
        """
        count = 0
        for x, y in points:
            self.tap(x, y, humanize=False)
            count += 1
        return count

    def deployment_touch(self, point: tuple[int, int]) -> DeploymentTouch:
        return DeploymentTouch(point, point, point, point, point)

    def send_deployment_touch(self, touch: DeploymentTouch) -> None:
        self.tap(*touch.logical, humanize=False)

    def send_deployment_sequence(
        self, touches: list[DeploymentTouch], stopped: Callable[[], bool],
    ) -> list[DeploymentTouch]:
        sent = []
        for touch in touches:
            if stopped():
                break
            self.send_deployment_touch(touch)
            sent.append(replace(touch, dispatched_at=time.time()))
        return sent

    def deployment_batch_tap(
        self,
        points: List[Tuple[int, int]],
        accepts: Callable[[tuple[float, float]], bool],
        stopped: Callable[[], bool],
    ) -> list[DeploymentTouch]:
        sent = []
        for point in points:
            if stopped():
                break
            touch = self.deployment_touch(point)
            if accepts(touch.logical_start) and accepts(touch.logical_end):
                touch = replace(touch, dispatched_at=time.time())
                self.send_deployment_touch(touch)
                sent.append(touch)
        return sent
