"""
基于 ADB screenrecord (H.264) + PyAV 的低延迟连续视频采集与无缝续流引擎

核心能力：
1. 绑定指定 ADB 设备序列号，以 1280x720 / 4 Mbps 输出裸 H.264 视频流，使用 PyAV 直接解帧为 BGR ndarray；
2. 通过远端 PID 文件精确跟踪并回收 Android 设备侧的 screenrecord 进程，避免停止、切换设备或续流时残留僵尸进程；
3. 在接近 180 秒录制上限前提前后台预热下一路视频流与解码器，解出首帧后原子无缝切换（若设备不支持并发则自动退化为快速重连）；
4. 启动或续流失败时向上抛出明确异常，由上层实时预览线程平滑降级回退到 PNG 连续采集。
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("coc.stream")

_SUBPROCESS_FLAGS = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
STANDARD_W = 1280
STANDARD_H = 720


@dataclass
class CapturedFrame:
    """采集线程产出的单帧封装（只保留最新帧，附带递增序号和单调时间戳）"""

    seq: int
    bgr: np.ndarray
    captured_at: float
    source: str  # "h264" | "png_fallback" | "png_single"
    session_id: int = 0

    def get_bgr(self) -> Optional[np.ndarray]:
        return self.bgr


def normalize_frame_bgr(img: np.ndarray, width: int = STANDARD_W, height: int = STANDARD_H) -> np.ndarray:
    """保证帧方向为横屏且分辨率与标准 1280x720 完全对齐"""
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    if (w, h) != (width, height):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img


def is_pyav_available() -> bool:
    try:
        import av  # noqa: F401
        return True
    except Exception:
        return False


class _H264SegmentProcess:
    """单段 screenrecord 子进程 + 独立 PyAV 解码上下文 + 远端 PID 精确清理"""

    def __init__(
        self,
        serial: str,
        width: int = STANDARD_W,
        height: int = STANDARD_H,
        bitrate: int = 4_000_000,
        time_limit_sec: int = 180,
    ):
        self.serial = serial
        self.width = width
        self.height = height
        self.bitrate = bitrate
        self.time_limit_sec = max(5, min(180, int(time_limit_sec)))
        self.token = uuid.uuid4().hex[:10]
        self.pid_file = f"/data/local/tmp/coc_sr_{self.token}.pid"
        self.proc: Optional[subprocess.Popen] = None
        self.codec = None
        self.started_at: float = 0.0
        self._closed = False
        self._close_lock = threading.Lock()

    def _adb_prefix(self) -> list[str]:
        if self.serial and self.serial != "auto":
            return ["adb", "-s", self.serial]
        return ["adb"]

    def start(self) -> None:
        import av

        self.codec = av.CodecContext.create("h264", "r")
        try:
            self.codec.thread_type = "SLICE"
            self.codec.thread_count = 2
        except Exception:
            pass

        shell_cmd = (
            f"echo $$ > {self.pid_file}; "
            f"exec screenrecord --output-format=h264 "
            f"--size {self.width}x{self.height} "
            f"--bit-rate {self.bitrate} "
            f"--time-limit {self.time_limit_sec} -"
        )
        cmd = self._adb_prefix() + ["exec-out", "sh", "-c", shell_cmd]
        self.started_at = time.perf_counter()
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_SUBPROCESS_FLAGS,
        )

    def read_chunk(self, size: int = 32768) -> bytes:
        proc = self.proc
        if proc is None or proc.stdout is None or self._closed:
            return b""
        try:
            if hasattr(proc.stdout, "read1"):
                return proc.stdout.read1(size)
            return proc.stdout.read(size)
        except Exception:
            return b""

    def decode_chunk_av_frames(self, chunk: bytes) -> list:
        """仅执行 H.264 解码产出轻量 av.VideoFrame（零 swscale 色彩转换开销，不丢参考帧）"""
        if not chunk or self.codec is None or self._closed:
            return []
        latest_av_frame = None
        try:
            packets = self.codec.parse(chunk)
        except Exception:
            return []
        for pkt in packets:
            if self._closed:
                break
            try:
                decoded = self.codec.decode(pkt)
            except Exception:
                continue
            if decoded:
                latest_av_frame = decoded[-1]
        return [latest_av_frame] if (latest_av_frame is not None and not self._closed) else []

    def decode_chunk(self, chunk: bytes) -> list[np.ndarray]:
        av_frames = self.decode_chunk_av_frames(chunk)
        out: list[np.ndarray] = []
        for frame in av_frames:
            try:
                bgr = frame.to_ndarray(format="bgr24")
                out.append(normalize_frame_bgr(bgr, self.width, self.height))
            except Exception:
                continue
        return out

    def elapsed_sec(self) -> float:
        if self.started_at <= 0:
            return 0.0
        return time.perf_counter() - self.started_at

    def read_stderr_snippet(self) -> str:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return ""
        try:
            if proc.poll() is not None:
                data = proc.stderr.read()
                return data.decode("utf-8", errors="ignore").strip()[:240]
        except Exception:
            pass
        return ""

    def close(self, async_remote_cleanup: bool = False) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            proc = self.proc
            self.proc = None
            self.codec = None

        if proc is not None:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=0.5)
                except Exception:
                    pass
            try:
                if proc.stderr:
                    proc.stderr.close()
            except Exception:
                pass

        def _cleanup_remote():
            kill_cmd = (
                f"if [ -f {self.pid_file} ]; then "
                f"kill -INT $(cat {self.pid_file}) 2>/dev/null || "
                f"kill -KILL $(cat {self.pid_file}) 2>/dev/null; "
                f"rm -f {self.pid_file}; fi"
            )
            try:
                subprocess.run(
                    self._adb_prefix() + ["shell", kill_cmd],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                    creationflags=_SUBPROCESS_FLAGS,
                )
            except Exception:
                pass

        if async_remote_cleanup:
            threading.Thread(target=_cleanup_remote, daemon=True, name="coc-sr-cleanup").start()
        else:
            _cleanup_remote()


class AdbH264StreamClient:
    """
    面向指定 ADB 序列号的 H.264 连续采集客户端：
    - 负责持续解码并将最新 1280x720 BGR 帧推送给回调；
    - 在 180s 上限前自动预热下一段并无缝切换；
    - 支持随时从外部线程 stop()，立即中断阻塞 I/O 并回收远端 screenrecord。
    """

    def __init__(
        self,
        serial: str,
        width: int = STANDARD_W,
        height: int = STANDARD_H,
        bitrate: int = 4_000_000,
        segment_limit_sec: int = 180,
        prewarm_lead_sec: float = 8.0,
        first_frame_timeout_sec: float = 4.5,
        max_consecutive_failures: int = 2,
    ):
        self.serial = serial
        self.width = width
        self.height = height
        self.bitrate = bitrate
        self.segment_limit_sec = max(6, min(180, int(segment_limit_sec)))
        self.prewarm_lead_sec = min(float(prewarm_lead_sec), max(2.0, self.segment_limit_sec * 0.25))
        self.first_frame_timeout_sec = first_frame_timeout_sec
        self.max_consecutive_failures = max_consecutive_failures

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._active_seg: Optional[_H264SegmentProcess] = None
        self._standby_seg: Optional[_H264SegmentProcess] = None
        self._standby_ready_frames: list[np.ndarray] = []
        self._standby_thread: Optional[threading.Thread] = None
        self._prewarming = False

        # 统计指标（可用于验收日志）
        self.first_frame_ms: Optional[float] = None
        self.reconnect_count: int = 0
        self.max_reconnect_gap_ms: float = 0.0

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            active = self._active_seg
            standby = self._standby_seg
            self._active_seg = None
            self._standby_seg = None
            self._standby_ready_frames = []
        if standby is not None:
            standby.close(async_remote_cleanup=False)
        if active is not None:
            active.close(async_remote_cleanup=False)

    def _start_and_wait_first_frame(
        self,
        seg: _H264SegmentProcess,
        timeout_sec: float,
    ) -> list[np.ndarray]:
        """启动一个视频流段并等待其解出至少 1 帧 BGR（带独立看门狗超时，防止 pipe read 永久挂起）"""
        seg.start()
        timed_out = threading.Event()

        def _watchdog():
            if not timed_out.wait(timeout_sec):
                # 仅在仍未解出首帧时触发超时关闭
                seg.close(async_remote_cleanup=True)

        wd = threading.Thread(target=_watchdog, daemon=True, name="coc-sr-wd")
        wd.start()
        try:
            while not self._stop_event.is_set():
                chunk = seg.read_chunk(32768)
                if not chunk:
                    break
                frames = seg.decode_chunk(chunk)
                if frames:
                    timed_out.set()
                    return frames
            timed_out.set()
            return []
        finally:
            timed_out.set()

    def _maybe_start_prewarm(self) -> None:
        with self._lock:
            if self._prewarming or self._standby_seg is not None or self._stop_event.is_set():
                return
            self._prewarming = True

        def _worker():
            candidate = _H264SegmentProcess(
                serial=self.serial,
                width=self.width,
                height=self.height,
                bitrate=self.bitrate,
                time_limit_sec=self.segment_limit_sec,
            )
            try:
                frames = self._start_and_wait_first_frame(candidate, self.first_frame_timeout_sec)
                if frames and not self._stop_event.is_set():
                    with self._lock:
                        self._standby_seg = candidate
                        self._standby_ready_frames = frames
                        self._prewarming = False
                    logger.debug(f"[H.264 续流预热] 新流已就绪，首帧数={len(frames)}")
                    return
            except Exception as e:
                logger.debug(f"[H.264 续流预热] 预热未成功（将在段结束时常规续接）: {e}")

            candidate.close(async_remote_cleanup=True)
            with self._lock:
                self._prewarming = False

        t = threading.Thread(target=_worker, daemon=True, name="coc-sr-prewarm")
        self._standby_thread = t
        t.start()

    def _take_standby(self) -> tuple[Optional[_H264SegmentProcess], list[np.ndarray]]:
        with self._lock:
            seg = self._standby_seg
            frames = self._standby_ready_frames
            self._standby_seg = None
            self._standby_ready_frames = []
        return seg, frames

    def run_stream(
        self,
        on_frame: Callable[[np.ndarray, float], None],
        on_reconnect: Optional[Callable[[int, float], None]] = None,
    ) -> None:
        """
        阻塞运行连续 H.264 解码循环直到 stop() 被调用；
        若无法启动首段或连续续流失败，抛出 RuntimeError 供上层回退到 PNG 采集。
        """
        if not is_pyav_available():
            raise RuntimeError("当前环境缺少 PyAV (av) 解码库")

        t_stream_start = time.perf_counter()
        last_frame_ts = t_stream_start
        consecutive_failures = 0

        try:
            while not self._stop_event.is_set():
                # 1. 优先复用预热好的下一段流，否则新建一段
                seg, initial_frames = self._take_standby()
                is_handover = seg is not None

                if seg is None:
                    seg = _H264SegmentProcess(
                        serial=self.serial,
                        width=self.width,
                        height=self.height,
                        bitrate=self.bitrate,
                        time_limit_sec=self.segment_limit_sec,
                    )
                    with self._lock:
                        self._active_seg = seg
                    initial_frames = self._start_and_wait_first_frame(seg, self.first_frame_timeout_sec)
                else:
                    with self._lock:
                        self._active_seg = seg

                if self._stop_event.is_set():
                    seg.close(async_remote_cleanup=False)
                    break

                if not initial_frames:
                    err_info = seg.read_stderr_snippet()
                    seg.close(async_remote_cleanup=True)
                    with self._lock:
                        if self._active_seg is seg:
                            self._active_seg = None
                    consecutive_failures += 1
                    logger.warning(
                        f"[H.264 采集] 启动/续流未解出首帧 ({consecutive_failures}/{self.max_consecutive_failures})"
                        + (f": {err_info}" if err_info else "")
                    )
                    if consecutive_failures >= self.max_consecutive_failures:
                        raise RuntimeError(f"H.264 视频流启动或续接失败: {err_info or '首帧超时'}")
                    time.sleep(0.25)
                    continue

                consecutive_failures = 0
                now = time.perf_counter()
                if self.first_frame_ms is None:
                    self.first_frame_ms = (now - t_stream_start) * 1000.0
                    logger.info(f"[H.264 采集] 设备 {self.serial} 首帧解码完成: {self.first_frame_ms:.1f} ms")
                else:
                    gap_ms = (now - last_frame_ts) * 1000.0
                    self.reconnect_count += 1
                    if gap_ms > self.max_reconnect_gap_ms:
                        self.max_reconnect_gap_ms = gap_ms
                    mode_desc = "无缝预热切换" if is_handover else "断流重建"
                    logger.info(
                        f"[H.264 续流] 第 {self.reconnect_count} 次续接完成 ({mode_desc}, 间隔={gap_ms:.1f} ms)"
                    )
                    if on_reconnect:
                        on_reconnect(self.reconnect_count, gap_ms)

                for bgr in initial_frames:
                    if self._stop_event.is_set():
                        break
                    last_frame_ts = time.perf_counter()
                    on_frame(bgr, last_frame_ts)

                # 2. 持续读流解码，接近段上限时触发后台预热
                prewarm_trigger = max(2.0, self.segment_limit_sec - self.prewarm_lead_sec)
                while not self._stop_event.is_set():
                    elapsed = seg.elapsed_sec()
                    if elapsed >= prewarm_trigger:
                        # 若预热流已就绪，立即原子交接，无需等待旧段硬断
                        with self._lock:
                            standby_ready = self._standby_seg is not None
                        if standby_ready:
                            break
                        self._maybe_start_prewarm()

                    chunk = seg.read_chunk(32768)
                    if not chunk:
                        break
                    decoded_frames = seg.decode_chunk(chunk)
                    for bgr in decoded_frames:
                        if self._stop_event.is_set():
                            break
                        last_frame_ts = time.perf_counter()
                        on_frame(bgr, last_frame_ts)

                # 当前段结束或准备无缝交接：关闭当前段并异步回收远端 PID
                with self._lock:
                    if self._active_seg is seg:
                        self._active_seg = None
                seg.close(async_remote_cleanup=not self._stop_event.is_set())
        finally:
            self.stop()
