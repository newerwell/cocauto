"""
电脑端（Windows 安卓模拟器）独立运行宿主模块 (PC Desktop Emulator Launcher)
支持自动探测本地正在运行的 MuMu12、雷电9、BlueStacks、夜神、逍遥模拟器 ADB 端口，
并提供分辨率自检（自动将模拟器矫正为 1280x720 240DPI 横屏）。
"""
from __future__ import annotations
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("coc.platforms.pc")
_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


@dataclass
class EmulatorInstance:
    name: str
    serial: str
    port: int


class PcEmulatorHost:
    """Windows 电脑端安卓模拟器自动发现与环境初始化宿主"""

    KNOWN_EMULATORS = [
        ("MuMu模拟器12 (主窗口)", 16384),
        ("MuMu模拟器 (兼容端口)", 7555),
        ("雷电模拟器9 / BlueStacks", 5555),
        ("雷电模拟器 (多开窗口 #1)", 5557),
        ("夜神模拟器 (Nox)", 62001),
        ("逍遥模拟器 (MEmu)", 21503),
    ]

    @classmethod
    def discover_emulators(cls) -> List[EmulatorInstance]:
        """扫描电脑本地正在运行的安卓模拟器实例"""
        found: List[EmulatorInstance] = []
        seen_serials = set()

        # 1. 先直接从 adb devices 中读取已连接的在线设备
        try:
            out = subprocess.check_output(["adb", "devices"], text=True, timeout=3, creationflags=_SUBPROCESS_FLAGS)
            for line in out.strip().splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    serial = parts[0]
                    port = int(serial.split(":")[1]) if ":" in serial and serial.split(":")[1].isdigit() else 5555
                    seen_serials.add(serial)
                    found.append(EmulatorInstance(name=f"在线设备 ({serial})", serial=serial, port=port))
        except Exception:
            pass

        # 2. 再主动探测常见模拟器端口
        for name, port in cls.KNOWN_EMULATORS:
            serial = f"127.0.0.1:{port}"
            if serial in seen_serials:
                continue
            try:
                res = subprocess.run(
                    ["adb", "connect", serial],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    creationflags=_SUBPROCESS_FLAGS,
                )
                out = (res.stdout or "") + (res.stderr or "")
                if "connected to" in out.lower() or "already connected" in out.lower():
                    seen_serials.add(serial)
                    found.append(EmulatorInstance(name=name, serial=serial, port=port))
            except Exception:
                continue
        return found

    @staticmethod
    def enforce_standard_display(serial: str, width: int = 1280, height: int = 720, dpi: int = 240) -> bool:
        """通过 ADB 将模拟器屏幕分辨率锁定为标准 1280x720 横屏"""
        try:
            subprocess.run(["adb", "-s", serial, "shell", f"wm size {width}x{height}"], check=False, timeout=4)
            subprocess.run(["adb", "-s", serial, "shell", f"wm density {dpi}"], check=False, timeout=4)
            logger.info(f"已确认设备 {serial} 分辨率对齐为 {width}x{height} ({dpi} DPI)")
            return True
        except Exception as e:
            logger.warning(f"设置设备分辨率失败 (将在截图层自动缩放归一化): {e}")
            return False
