"""
统一 ADB 设备控制器 (ADB Device Controller v2.2)
重大升级：
1. 原生 Linux Multi-Touch Protocol B (/dev/input/event*) 硬件双指并发捏合缩小 (Pinch-to-Zoom Out)，
   彻底解决《部落冲突》不响应 keyevent 169 导致视角无法缩小的问题！
2. 自动感知设备真实物理横屏分辨率 (如 1280x720 或 1600x900)，自动对 tap/swipe 坐标做物理像素映射。
3. 三段式相机归位算法：连续硬件双指捏合缩至最小 -> 左上边界阻尼归零 -> 无惯性刹车定量回拉居中。
4. 全链路细粒度实时调试日志输出 + `force_abort()` 0.1 秒瞬间强杀 ADB 子进程与释放触控状态。
"""
from __future__ import annotations
import logging
import os
import subprocess
import time
from typing import Callable, List, Tuple, Optional
import numpy as np
import cv2

from .base_controller import BaseDeviceController, DeploymentTouch
from .maatouch_client import MultiTouchProtocol, TouchDeviceInfo, VisionZoomAnalyzer
from core.humanizer import HumanJitter

logger = logging.getLogger("coc.device.adb")
_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class AdbDeviceController(BaseDeviceController):
    """基于标准 ADB 与 Linux Multi-Touch 硬件事件注入的跨平台控制器"""

    # 优先将 5561 (用户的部落冲突模拟器实例) 排在首位，并覆盖主流多开端口
    COMMON_EMULATOR_PORTS = [5561, 16384, 7555, 5555, 5557, 5559, 5563, 16416, 62001, 21503]
    DEPLOYMENT_SEQUENCE_LIMIT = 16

    def __init__(
        self,
        serial: str = "127.0.0.1:5561",
        jitter: Optional[HumanJitter] = None,
        screencap_timeout: float = 5.0,
    ):
        self.serial = serial
        self.jitter = jitter or HumanJitter()
        self.screencap_timeout = screencap_timeout
        self._resolved_serial: Optional[str] = None if serial == "auto" else serial

        # 物理横屏宽高 (默认 1280x720，在首次截图或探测时自动更新为真实物理像素)
        self._phys_width: int = 1280
        self._phys_height: int = 720
        self._touch_info: Optional[TouchDeviceInfo] = None
        self._normalized_map: tuple[str, np.ndarray] | None = None

        # 用于【🚨 强行停止】的子进程追踪与中断标志
        self._active_proc: Optional[subprocess.Popen] = None
        self._abort_requested: bool = False

    def _adb_prefix(self) -> List[str]:
        if self._resolved_serial:
            return ["adb", "-s", self._resolved_serial]
        return ["adb"]

    def reset_abort(self) -> None:
        self._abort_requested = False

    def force_abort(self) -> None:
        """
        🚨 0.1 秒强行停止：
        1. 置位 _abort_requested 中断标志
        2. 立即 kill 当前正在运行的 ADB 子进程
        3. 发送多点触控抬起事件，防止手指残留在按下状态
        """
        self._abort_requested = True
        proc = self._active_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
                logger.warning("🚨 [强行停止] 已立即强杀当前正在执行的 ADB 子进程！")
            except Exception:
                pass
        if self._touch_info is not None:
            try:
                rel_script = MultiTouchProtocol.build_emergency_release_script(self._touch_info).encode("utf-8")
                p = subprocess.Popen(
                    self._adb_prefix() + ["shell", "sh"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=_SUBPROCESS_FLAGS,
                )
                p.communicate(input=rel_script, timeout=1.0)
            except Exception:
                pass

    def _check_aborted(self) -> None:
        if self._abort_requested:
            raise InterruptedError("操作已被用户强行停止 (Force Aborted)")

    def _run_adb_interruptible(self, args: List[str], input_bytes: Optional[bytes] = None,
                               timeout: float = 5.0, *, check: bool = False) -> bytes:
        """可被 force_abort() 随时在 0.1 秒内强行中止的 ADB 命令执行器"""
        self._check_aborted()
        cmd = self._adb_prefix() + args
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_SUBPROCESS_FLAGS,
        )
        self._active_proc = proc
        try:
            try:
                out, err = proc.communicate(input=input_bytes, timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise
            if check and proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, cmd, output=out, stderr=err)
            return out or b""
        finally:
            self._active_proc = None
            self._check_aborted()

    def connect(self) -> bool:
        """自动扫描或连接指定设备序列号"""
        self.reset_abort()
        if self.serial != "auto":
            if self.serial not in self.list_online_devices():
                if ":" not in self.serial:
                    return False
                try:
                    subprocess.run(["adb", "connect", self.serial], capture_output=True,
                                   timeout=4, creationflags=_SUBPROCESS_FLAGS)
                except (OSError, subprocess.TimeoutExpired):
                    return False
                if self.serial not in self.list_online_devices():
                    return False
            self._resolved_serial = self.serial
            return True

        online = self.list_online_devices()
        if online:
            # 优先匹配 5561
            preferred = next((d for d in online if "5561" in d), online[0])
            self._resolved_serial = preferred
            return True

        for port in self.COMMON_EMULATOR_PORTS:
            candidate = f"127.0.0.1:{port}"
            try:
                subprocess.run(
                    ["adb", "connect", candidate],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    creationflags=_SUBPROCESS_FLAGS,
                )
            except Exception:
                continue

        online = self.list_online_devices()
        if online:
            preferred = next((d for d in online if "5561" in d), online[0])
            self._resolved_serial = preferred
            return True
        return False

    @staticmethod
    def list_online_devices() -> List[str]:
        try:
            out = subprocess.check_output(["adb", "devices"], text=True, timeout=4, creationflags=_SUBPROCESS_FLAGS)
            devices = []
            for line in out.strip().splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    devices.append(parts[0])
            return devices
        except Exception:
            return []

    def _ensure_touch_device_probed(self) -> TouchDeviceInfo:
        """自动探测目标设备的多点触控 event 节点、64/32位架构与坐标范围"""
        if self._touch_info is not None:
            return self._touch_info
        try:
            arch_out = self._run_adb_interruptible(["shell", "uname -m"], timeout=3.0).decode("utf-8", errors="ignore").strip()
            is_64 = ("64" in arch_out) or ("aarch64" in arch_out) or ("x86_64" in arch_out)
            ge_out = self._run_adb_interruptible(["shell", "getevent -pl"], timeout=4.0).decode("utf-8", errors="ignore")
            self._touch_info = MultiTouchProtocol.parse_touch_device_from_getevent(ge_out, is_64bit=is_64)
            logger.info(
                f"🔍 [硬件触控探针] 设备={self._resolved_serial} | 架构={arch_out or 'aarch64'}(64bit={is_64}) | "
                f"多点触控节点={self._touch_info.event_path} (HW_X=0..{self._touch_info.max_x}, HW_Y=0..{self._touch_info.max_y})"
            )
        except Exception as e:
            logger.warning(f"探测 getevent 失败，使用默认 /dev/input/event4 参数: {e}")
            self._touch_info = TouchDeviceInfo()
        return self._touch_info

    def _to_physical_xy(self, logic_x: int, logic_y: int) -> Tuple[int, int]:
        """将 1280x720 逻辑坐标转换为设备真实物理横屏坐标"""
        px = int(round(logic_x * self._phys_width / float(self.STANDARD_WIDTH)))
        py = int(round(logic_y * self._phys_height / float(self.STANDARD_HEIGHT)))
        return px, py

    def screencap(self) -> np.ndarray:
        """通过 adb exec-out screencap -p 高速获取内存截图并归一化至 1280x720 BGR"""
        raw_bytes = self._run_adb_interruptible(["exec-out", "screencap", "-p"], timeout=self.screencap_timeout)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("无法解码 ADB 屏幕截图数据")

        h, w = img.shape[:2]
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h, w = img.shape[:2]

        # 记录真实物理横屏宽高，供 tap / swipe 自动比例换算
        self._phys_width, self._phys_height = w, h

        if (w, h) != (self.STANDARD_WIDTH, self.STANDARD_HEIGHT):
            img = cv2.resize(img, (self.STANDARD_WIDTH, self.STANDARD_HEIGHT), interpolation=cv2.INTER_AREA)
        return img

    def tap(self, x: int, y: int, humanize: bool = True) -> Tuple[int, int]:
        """执行带二维高斯抖动、物理分辨率自动换算与实时日志输出的点击"""
        self._check_aborted()
        tx, ty = self.jitter.jitter_point(x, y) if humanize else (x, y)
        px, py = self._to_physical_xy(tx, ty)
        press_ms = self.jitter.random_press_duration_ms() if humanize else 55
        end_x = px + (1 if (px % 2 == 0) else -1)
        end_y = py + (1 if (py % 2 == 0) else -1)

        logger.info(f"👆 [点击 Tap] 逻辑坐标=({tx}, {ty}) -> 物理坐标=({px}, {py}), 按压={press_ms}ms")
        self._run_adb_interruptible(
            ["shell", f"input swipe {px} {py} {end_x} {end_y} {press_ms}"],
            timeout=4.0,
        )
        if humanize:
            self.jitter.sleep_human_delay()
        return tx, ty

    def deployment_touch(self, point: tuple[int, int]) -> DeploymentTouch:
        width, height = self._phys_width, self._phys_height
        px = round(point[0] * width / self.STANDARD_WIDTH)
        py = round(point[1] * height / self.STANDARD_HEIGHT)
        start = px, py
        end = px + (1 if px % 2 == 0 else -1), py + (1 if py % 2 == 0 else -1)
        return DeploymentTouch(
            point, start, end,
            (px * self.STANDARD_WIDTH / width, py * self.STANDARD_HEIGHT / height),
            (end[0] * self.STANDARD_WIDTH / width, end[1] * self.STANDARD_HEIGHT / height),
        )

    def send_deployment_touch(self, touch: DeploymentTouch) -> None:
        px, py = touch.physical_start
        ex, ey = touch.physical_end
        self._run_adb_interruptible(["shell", f"input swipe {px} {py} {ex} {ey} 48"], timeout=4.0, check=True)

    def send_deployment_sequence(
        self, touches: list[DeploymentTouch], stopped: Callable[[], bool],
    ) -> list[DeploymentTouch]:
        if not touches or stopped():
            return []
        # Android's input service avoids starting a Java process for every touch.
        script = [
            'set -e',
            'if cmd -l | grep -q "^[[:space:]]*input$"; then INPUT="cmd input"; else INPUT=input; fi',
            'press_point() { $INPUT swipe "$1" "$2" "$3" "$4" 44; sleep 0.026; }',
        ]
        for index, touch in enumerate(touches, 1):
            px, py = touch.physical_start
            ex, ey = touch.physical_end
            script.extend((f"press_point {px} {py} {ex} {ey}", f"echo {index}"))
        try:
            output = self._run_adb_interruptible(
                ["shell", "sh"], input_bytes=("\n".join(script) + "\n").encode(), timeout=4.0, check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning("快速部署序列中断，等待兵栏确认: %s", exc)
            output = exc.output or b""
        completed = 0
        for line in output.splitlines():
            if line.strip() == str(completed + 1).encode():
                completed += 1
        return touches[:completed]

    def fast_batch_tap(
        self,
        points: List[Tuple[int, int]],
        interval_ms: int = 32,
        press_ms: int = 48,
        humanize: bool = True,
    ) -> int:
        """
        通过单次 `adb shell sh` 高速连续批量点击一组逻辑坐标点：
        - 关键升级：将 0ms 瞬时 `input tap` 升级为跨 3 个 60FPS 渲染帧的 `input swipe px py (px+1) (py+1) 48` (按住 48ms)，
          彻底解决《部落冲突》OpenGL 渲染引擎丢弃瞬时点击导致 16 条龙只放出 2~3 条的问题！
        - 保持微幅高斯抖动 (±3px) 防止落点机械完全重合，同时不会偏离外圈安全草地带；
        - 全程受 `_run_adb_interruptible` 托管，点击【🚨 强行停止】可在 0.1 秒内立即中断！
        """
        self._check_aborted()
        if not points:
            return 0

        lines: List[str] = []
        phys_pts: List[Tuple[int, int]] = []
        usleep_val = max(15, int(interval_ms)) * 1000
        hold_ms = max(36, int(press_ms))

        for x, y in points:
            if humanize:
                tx, ty = self.jitter.jitter_point(x, y, custom_sigma=1.5, custom_max_offset=3)
            else:
                tx, ty = x, y
            px, py = self._to_physical_xy(tx, ty)
            end_x = px + (1 if (px % 2 == 0) else -1)
            end_y = py + (1 if (py % 2 == 0) else -1)
            phys_pts.append((px, py))
            lines.append(f"input swipe {px} {py} {end_x} {end_y} {hold_ms}")
            lines.append(f"usleep {usleep_val}")

        script_str = "\n".join(lines) + "\n"
        logger.info(
            f"⚡ [高速跨帧放兵 BatchTap] 单次连点 {len(points)} 下 | 首点={phys_pts[0]} -> 末点={phys_pts[-1]} | 按压={hold_ms}ms, 间隔={interval_ms}ms"
        )
        timeout_sec = max(8.0, len(points) * 0.35)
        self._run_adb_interruptible(["shell", "sh"], input_bytes=script_str.encode("utf-8"), timeout=timeout_sec)
        return len(points)

    def swipe(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
        duration_ms: int = 350,
        use_bezier: bool = True,
    ) -> None:
        """滑动屏幕，自动换算物理分辨率并输出详细调试日志"""
        self._check_aborted()
        sx, sy = self.jitter.jitter_point(*start, custom_sigma=2.5, custom_max_offset=6)
        ex, ey = self.jitter.jitter_point(*end, custom_sigma=2.5, custom_max_offset=6)
        psx, psy = self._to_physical_xy(sx, sy)
        pex, pey = self._to_physical_xy(ex, ey)

        logger.info(
            f"👉 [滑动 Swipe] 逻辑({sx},{sy})->({ex},{ey}) | 物理({psx},{psy})->({pex},{pey}) | 时长={duration_ms}ms"
        )
        self._run_adb_interruptible(
            ["shell", f"input swipe {psx} {psy} {pex} {pey} {duration_ms}"],
            timeout=5.0,
        )
        self.jitter.sleep_human_delay(0.6)

    def _execute_sendevent_pinch(
        self,
        touch_info: TouchDeviceInfo,
        f0_start: Tuple[int, int] = (260, 170),
        f0_end: Tuple[int, int] = (575, 335),
        f1_start: Tuple[int, int] = (980, 520),
        f1_end: Tuple[int, int] = (705, 385),
        steps: int = 10,
    ) -> None:
        """
        通过 `adb shell sh` 一次性执行带 `usleep 14000` (14ms 60FPS 帧同步) 的原生 `sendevent` 双指捏合脚本。
        零二进制管道错误，且起止坐标严格位于纯地图区域 (完全避开四周 UI 按钮)！
        """
        script_str = MultiTouchProtocol.build_sendevent_pinch_script(
            touch_info=touch_info,
            steps=steps,
            f0_start=f0_start,
            f0_end=f0_end,
            f1_start=f1_start,
            f1_end=f1_end,
            step_usleep=14000,
        )
        self._run_adb_interruptible(["shell", "sh"], input_bytes=script_str.encode("utf-8"),
                                    timeout=6.0, check=True)

    def pinch_zoom_out(self, align_mode: str = "center") -> None:
        """
        基于图像识别闭环反馈 (Computer Vision Closed-Loop) 的视角缩小与四角居中归一化：
        1. 视觉初态分析 (`VisionZoomAnalyzer.analyze_village_viewport`)：
           识别当前画面的外圈森林+海水总占比 (`total_outer_ratio`, 放大特写时仅 ~5.8%, 最大全景时 >= 13.0%)
           以及顶部 12点钟方向森林占比 (`top_forest_ratio`, 理想四角居中为 28% ~ 50%)。
        2. 图像识别闭环双指缩小 (Stage 1)：
           若视觉检测到尚未处于最大全景视野 (或执行首次强制确认捏合)，则执行内核级 `sendevent` 帧同步双指捏合，
           并在每轮捏合后通过多尺度模板匹配 (`measure_scale_change`) 与外圈占比检测是否已缩至极限！
        3. 图像识别闭环四角居中对齐 (Stage 2)：
           不再盲目固定滑动，而是先用图像识别检测顶部森林占比与左右对称度：
           - 若刚缩完视野偏上/偏下/偏左/偏右，或用户要求强制居中，先执行左上基准归零 + 黄金回拉；
           - 随后启动最多 3 轮**图像识别闭环微调 (`compute_centering_swipe`)**，根据 `top_forest_ratio` 和左右森林对称度动态计算拖拽像素，直到 12/3/6/9 点钟四角完美居中！
        """
        self._check_aborted()
        t_start = time.perf_counter()
        curr_img = self.screencap()
        touch_info = self._ensure_touch_device_probed()
        v_state = VisionZoomAnalyzer.analyze_village_viewport(curr_img)

        logger.info(
            f"========== [图像识别闭环视角归一化] 启动 (模式={align_mode}, 屏幕分辨率={self._phys_width}x{self._phys_height}) =========="
        )
        logger.info(
            f"👁️ [初始图像识别] 外圈森林+海水占比={v_state.total_outer_ratio:.1%} (阈值>=13.0%) | "
            f"顶部森林={v_state.top_forest_ratio:.1%} | 右下森林={v_state.bot_forest_ratio:.1%} | 已达最大视野={v_state.is_max_panorama}"
        )

        # ---- 阶段 1：图像识别闭环双指捏合缩小 ----
        pinch_Safe_Coords = [
            ((260, 170), (575, 335), (980, 520), (705, 385)),
            ((275, 180), (580, 338), (960, 505), (700, 382)),
            ((250, 165), (575, 335), (990, 525), (705, 385)),
        ]
        for idx, (f0_s, f0_e, f1_s, f1_e) in enumerate(pinch_Safe_Coords, start=1):
            self._check_aborted()
            logger.info(
                f"🤏 [Step 1.{idx} 内核帧同步双指捏合] (sendevent + 14ms/步 -> {touch_info.event_path}): "
                f"手指0{f0_s}->{f0_e} | 手指1{f1_s}->{f1_e}"
            )
            try:
                self._execute_sendevent_pinch(
                    touch_info=touch_info,
                    f0_start=f0_s,
                    f0_end=f0_e,
                    f1_start=f1_s,
                    f1_end=f1_e,
                    steps=10,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                if align_mode != "zoom_only":
                    raise
                release = MultiTouchProtocol.build_emergency_release_script(touch_info).encode("utf-8")
                self._run_adb_interruptible(["shell", "sh"], input_bytes=release, timeout=3.0, check=True)
                logger.warning("缩放中断，已释放触控。使用当前画面继续识别 (%s)", exc)
                return
            time.sleep(0.25)
            next_img = self.screencap()
            scale_factor, score = VisionZoomAnalyzer.measure_scale_change(curr_img, next_img)
            v_state = VisionZoomAnalyzer.analyze_village_viewport(next_img)
            curr_img = next_img

            logger.info(
                f"👁️ [缩放视觉闭环 #{idx}] 本轮建筑缩放比={scale_factor:.2f}x | "
                f"外圈森林+海水占比={v_state.total_outer_ratio:.1%} (顶部={v_state.top_forest_ratio:.1%}) | "
                f"已达最大全景={v_state.is_max_panorama}"
            )
            if v_state.total_outer_ratio >= 0.142 or (v_state.is_max_panorama and scale_factor >= 0.95):
                logger.info(
                    f"✅ [图像识别确认最大视野] 外圈占比已达 {v_state.total_outer_ratio:.1%} (>=14.2%)，图像识别确认已缩至最大全景视野！"
                )
                break

        if align_mode == "zoom_only":
            elapsed = (time.perf_counter() - t_start) * 1000.0
            logger.info(f"✅ [视角归一化完成] 仅执行双指缩小至最大视野 (总耗时 {elapsed:.0f}ms)")
            return

        # ---- 阶段 2：先用图像识别判断是否已经四角居中；若未居中，执行边界归位 + 视觉闭环微调 ----
        self._check_aborted()
        need_initial_reset = (
            align_mode != "center"
            or v_state.top_forest_ratio < 0.25
            or v_state.top_forest_ratio > 0.52
            or abs((v_state.right_top_forest - v_state.left_top_forest) - 0.04) > 0.12
        )

        if need_initial_reset:
            logger.info("🧭 [Step 2 左上边界阻尼归零] 正在将镜头推至地图左上角基准位 (320,180)->(960,540)...")
            p_s1 = self._to_physical_xy(320, 180)
            p_e1 = self._to_physical_xy(960, 540)
            self._run_adb_interruptible(
                ["shell", f"input swipe {p_s1[0]} {p_s1[1]} {p_e1[0]} {p_e1[1]} 400"],
                timeout=4.0,
            )
            time.sleep(0.32)

            if align_mode == "top":
                pull_start, pull_end = (740, 410), (660, 390)
            elif align_mode == "bottom":
                pull_start, pull_end = (780, 460), (520, 290)
            else:
                pull_start, pull_end = (740, 410), (590, 355)

            p_pull_s = self._to_physical_xy(*pull_start)
            p_pull_e = self._to_physical_xy(*pull_end)
            logger.info(f"🎯 [Step 3 黄金比例回拉居中] 从 {p_pull_s} 慢速回拉至 {p_pull_e} (650ms)...")
            self._run_adb_interruptible(
                ["shell", f"input swipe {p_pull_s[0]} {p_pull_s[1]} {p_pull_e[0]} {p_pull_e[1]} 650"],
                timeout=4.0,
            )
            time.sleep(0.35)
            curr_img = self.screencap()
            v_state = VisionZoomAnalyzer.analyze_village_viewport(curr_img)

        # ---- 阶段 3：图像识别四角森林对称度闭环微调 (最多 2 轮) ----
        if align_mode == "center":
            for adj_i in range(1, 3):
                self._check_aborted()
                correction = VisionZoomAnalyzer.compute_centering_swipe(v_state)
                if correction is None:
                    logger.info(
                        f"👁️ [图像识别居中确认] 顶部12点钟森林占比={v_state.top_forest_ratio:.1%} (理想26%~52%)，"
                        f"左上={v_state.left_top_forest:.1%}/右上={v_state.right_top_forest:.1%}，阵型四角已完美居中！"
                    )
                    break
                c_start, c_end, reason = correction
                ps = self._to_physical_xy(*c_start)
                pe = self._to_physical_xy(*c_end)
                logger.info(f"🔧 [视觉闭环微调 #{adj_i}] {reason} -> 执行微调拖拽 {ps}->{pe}")
                self._run_adb_interruptible(
                    ["shell", f"input swipe {ps[0]} {ps[1]} {pe[0]} {pe[1]} 580"],
                    timeout=4.0,
                )
                time.sleep(0.32)
                curr_img = self.screencap()
                v_state = VisionZoomAnalyzer.analyze_village_viewport(curr_img)

        elapsed = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            f"✅ [视觉闭环视角归一化完成] 外圈占比={v_state.total_outer_ratio:.1%}, 顶部森林={v_state.top_forest_ratio:.1%} (总耗时 {elapsed:.0f}ms)"
        )

    def normalize_map_view(
        self, frame: np.ndarray, *, world: str,
        accepts: Callable[[np.ndarray], bool], stopped: Callable[[], bool],
    ) -> np.ndarray | None:
        if stopped() or not accepts(frame):
            return None
        cached = self._normalized_map
        if cached and cached[0] == world and VisionZoomAnalyzer.same_map_view(cached[1], frame):
            return frame
        self._normalized_map = None
        touch = self._ensure_touch_device_probed()
        for _ in range(3):
            if stopped():
                return None
            try:
                self._execute_sendevent_pinch(touch_info=touch, f0_start=(260, 170), f0_end=(575, 335),
                                              f1_start=(980, 520), f1_end=(705, 385), steps=10)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                release = MultiTouchProtocol.build_emergency_release_script(touch).encode()
                try:
                    self._run_adb_interruptible(["shell", "sh"], input_bytes=release, timeout=3.0)
                except subprocess.TimeoutExpired:
                    logger.warning("归一化后释放触控超时")
                logger.warning("地图缩小手势失败 world=%s error=%s", world, exc)
                return None
            time.sleep(0.20)
            if stopped():
                return None
            current = self.screencap()
            if not accepts(current):
                return None
            scale, score = VisionZoomAnalyzer.measure_scale_change(frame, current)
            frame = current
            logger.info("地图缩放检查 world=%s scale=%.2f confidence=%.2f", world, scale, score)
            if 0.96 <= scale <= 1.025 and score >= 0.58:
                break
        else:
            return None

        for adjustment in range(3):
            if stopped():
                return None
            center = VisionZoomAnalyzer.playable_center(frame, world)
            if center is None:
                return None
            dx, dy = 640 - center[0], 337 - center[1]
            if abs(dx) <= 66 and abs(dy) <= 56:
                observed = self.screencap()
                if not stopped() and accepts(observed) and VisionZoomAnalyzer.same_map_view(frame, observed):
                    self._normalized_map = world, observed
                    logger.info("已确认地图全景居中 world=%s center=%s", world, center)
                    return observed
                return None
            if adjustment == 2:
                break
            self.swipe((640, 337), (640 + int(np.clip(dx, -145, 145)), 337 + int(np.clip(dy, -120, 120))),
                       duration_ms=430, use_bezier=False)
            time.sleep(0.20)
            if stopped():
                return None
            frame = self.screencap()
            if not accepts(frame):
                return None
        return None
