"""
《部落冲突》跨平台全自动辅助软件 - Windows 桌面可视化客户端一键启动入口 (GUI App Entry)
双击 EXE 或运行 `python gui_app.py` 即可直接打开 PyQt5 深色主题桌面控制台。
支持可选验证参数：
- `--self-test`: 打开窗口立即自检退出
- `--verify-live-sec <秒数> [--verify-out <json路径>] [--segment-limit <秒>]`:
  在真实窗口中开启实时画面流并采集指定秒数，输出实际 FPS、P95 显示间隔、采集方式与续流指标，便于打包后 EXE 验收。
- `--verify-boat-device <设备序列号> --verify-out <json路径>`:
  从主世界开始连续运行渡船按钮三次往返，输出每段渡船的结果和证据路径。
- `--verify-combat-device <设备序列号> --verify-rounds <轮数> --verify-out <json路径> [--verify-combat-world dual|home|night]`:
  从桌面连续作战入口运行指定轮数，记录每轮结果和实际日志路径。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QLockFile, Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox

from platforms.pc_desktop.main_window import CocDesktopMainWindow

# 全局保持单实例内核互斥体与跨平台文件锁引用，防止被垃圾回收提前释放
_SINGLE_INSTANCE_MUTEX_HANDLE = None
_SINGLE_INSTANCE_LOCKFILE: QLockFile | None = None
SINGLE_INSTANCE_MUTEX_NAME = "Local\\COC_AutoApp_SingleInstance_Mutex_v3"


def _focus_existing_coc_window() -> bool:
    """在 Windows 下自动查找已打开的 COC-AutoApp 主窗口并将其恢复、置顶激活"""
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        found_hwnds: list[int] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def _enum_cb(hwnd, _lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    if "COC-AutoApp" in buf.value and "《部落冲突》" in buf.value:
                        found_hwnds.append(int(hwnd))
                        return False
            return True

        user32.EnumWindows(_enum_cb, 0)
        if found_hwnds:
            hwnd = found_hwnds[0]
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            else:
                user32.ShowWindow(hwnd, 5)  # SW_SHOW
            user32.SetForegroundWindow(hwnd)
            return True
    except Exception:
        pass
    return False


def acquire_single_instance_lock(mutex_name: str = SINGLE_INSTANCE_MUTEX_NAME) -> bool:
    """
    获取全局单实例互斥锁（限制 EXE 打开的面板最多只能打开一个）：
    - Windows 下优先使用 OS 内核命名互斥体 `CreateMutexW`（即使进程异常终止或被强杀也会由内核自动释放，无残留死锁）；
    - 同时兼容非 Windows 系统的 `QLockFile` 进程锁。
    返回 True 表示当前是唯一实例，返回 False 表示已有实例正在运行。
    """
    global _SINGLE_INSTANCE_MUTEX_HANDLE, _SINGLE_INSTANCE_LOCKFILE

    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.CreateMutexW(None, False, mutex_name)
            last_err = kernel32.GetLastError()
            if not handle:
                return True
            if last_err == 183:  # ERROR_ALREADY_EXISTS
                kernel32.CloseHandle(handle)
                return False
            _SINGLE_INSTANCE_MUTEX_HANDLE = handle
            return True
        except Exception:
            pass

    lock_path = str(Path(tempfile.gettempdir()) / "coc_autoapp_single_instance.lock")
    lock_file = QLockFile(lock_path)
    lock_file.setStaleLockTime(0)
    if not lock_file.tryLock(100):
        return False
    _SINGLE_INSTANCE_LOCKFILE = lock_file
    return True


def release_single_instance_lock() -> None:
    """主动释放当前进程持有的单实例互斥锁（供退出或单元测试清理使用）"""
    global _SINGLE_INSTANCE_MUTEX_HANDLE, _SINGLE_INSTANCE_LOCKFILE
    if os.name == "nt" and _SINGLE_INSTANCE_MUTEX_HANDLE is not None:
        try:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(_SINGLE_INSTANCE_MUTEX_HANDLE)
        except Exception:
            pass
        _SINGLE_INSTANCE_MUTEX_HANDLE = None
    if _SINGLE_INSTANCE_LOCKFILE is not None:
        try:
            _SINGLE_INSTANCE_LOCKFILE.unlock()
        except Exception:
            pass
        _SINGLE_INSTANCE_LOCKFILE = None


def _get_arg_value(flag: str, default: str = "") -> str:
    if flag in sys.argv:
        idx = sys.argv.index(flag)
        if idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
    return default


def _verify_boat_round_trips(app, win, serial: str, out_path: Path) -> None:
    win.combo_device.setCurrentText(serial)
    expected_scenes = ("night", "home") * 3
    legs: list[dict] = []
    current_result: dict = {}

    def record_result(summary: dict) -> None:
        nonlocal current_result
        current_result = summary.get("night_world", {})

    def finish_leg() -> None:
        target = expected_scenes[len(legs)]
        source = "home" if target == "night" else "night"
        result = current_result or {"reason": "worker_no_travel_result", "status": win.lbl_status_badge.text()}
        evidence = result.get("evidence")
        verified = bool(
            result.get("reason") == "arrived" and result.get("source") == source
            and result.get("scene") == target and result.get("touches") and evidence
            and (Path(evidence) / "arrival.png").is_file()
        )
        legs.append({"expected_scene": target, "verified": verified, **result})
        done = len(legs) == len(expected_scenes)
        out_path.write_text(json.dumps({
            "executable": sys.executable, "frozen": bool(getattr(sys, "frozen", False)),
            "device": serial, "verified": verified and done, "legs": legs,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        if not verified or done:
            app.exit(0 if verified else 1)
        else:
            QTimer.singleShot(0, dispatch_next)

    def dispatch_next() -> None:
        nonlocal current_result
        current_result = {}
        win.btn_night_boat.click()
        while_running.start()

    def check_finished() -> None:
        if win._bot_worker.isRunning():
            return
        while_running.stop()
        QTimer.singleShot(0, finish_leg)

    win.botCycleCompleted.connect(record_result)
    while_running = QTimer(win)
    while_running.setInterval(100)
    while_running.timeout.connect(check_finished)
    QTimer.singleShot(0, dispatch_next)


def _verify_combat_rounds(app, win, serial: str, limit: int, out_path: Path, mode: str) -> None:
    rounds = []
    files = []
    win.combo_device.setCurrentText(serial)

    def save():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "executable": sys.executable, "device": serial, "requested_rounds": limit, "mode": mode,
            "run_logs": files, "rounds": rounds,
        }, ensure_ascii=False, default=str, indent=2), encoding="utf-8")

    def on_log(path):
        files.append(path)
        save()
        if limit == 0:
            win.stop_bot_worker()

    def on_round(summary):
        rounds.append(summary)
        save()
        if len(rounds) >= limit:
            win.stop_bot_worker()

    def finish():
        save()
        app.exit(0 if files and len(rounds) >= limit else 1)

    win.botRunLogOpened.connect(on_log)
    win.botCycleCompleted.connect(on_round)
    win.start_bot_worker(mode)
    win._bot_worker.finished.connect(finish)


def launch_gui():
    # 开启高分屏 DPI 自适应缩放
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("COC-AutoApp")

    # 非自动化自检模式下，严格限制 EXE / GUI 控制台面板只能打开一个实例
    verify_boats = "--verify-boat-device" in sys.argv
    is_automated_check = "--self-test" in sys.argv or (
        not verify_boats and bool(_get_arg_value("--verify-live-sec"))
    )
    if not is_automated_check and not acquire_single_instance_lock():
        focused = _focus_existing_coc_window()
        if not focused:
            QMessageBox.information(
                None,
                "COC-AutoApp 已在运行中",
                "检测到《部落冲突》全自动辅助控制台 (COC-AutoApp) 已经打开了一个面板！\n为防止多开争抢模拟器 ADB 连接，请直接使用已打开的窗口。",
            )
        sys.exit(0)

    win = CocDesktopMainWindow()
    win.show()

    if "--self-test" in sys.argv:
        app.processEvents()
        win.close()
        sys.exit(0)

    if "--verify-combat-device" in sys.argv:
        serial = _get_arg_value("--verify-combat-device")
        dest = _get_arg_value("--verify-out")
        count = _get_arg_value("--verify-rounds", "2")
        mode = {"dual": "loop_dual_combat", "home": "loop_home_combat", "night": "loop_night_combat"}.get(
            _get_arg_value("--verify-combat-world", "dual")
        )
        if not serial or not dest or not count.isdecimal() or mode is None:
            sys.exit(2)
        QTimer.singleShot(0, lambda: _verify_combat_rounds(app, win, serial, int(count), Path(dest), mode))
        sys.exit(app.exec_())

    if verify_boats:
        serial = _get_arg_value("--verify-boat-device")
        boat_out = _get_arg_value("--verify-out")
        if not serial or not boat_out:
            sys.exit(2)
        _verify_boat_round_trips(app, win, serial, Path(boat_out))
        sys.exit(app.exec_())

    verify_sec_str = _get_arg_value("--verify-live-sec")
    if verify_sec_str:
        duration_sec = max(2.0, float(verify_sec_str))
        seg_limit = int(_get_arg_value("--segment-limit", "180") or "180")
        out_path = _get_arg_value("--verify-out")
        force_fail = "--force-h264-fail" in sys.argv

        win.btn_toggle_stream.setChecked(True)
        win.btn_toggle_stream.setText("⏹ 停止实时画面流 (直播中...)")
        win._start_vision_worker(
            continuous=True,
            force_h264_fail=force_fail,
            segment_limit_sec=seg_limit,
        )

        # 直接挂钩真实画布新帧显示事件，避免轮询引入额外时间量化误差
        all_gaps_ms: list[float] = []
        first_display_ts = {"t": None}
        last_display_ts = {"t": None}
        orig_set_video_frame = win.canvas.set_video_frame

        def _hooked_set_video_frame(bgr_img, seq=0, copy_raw=False):
            ok = orig_set_video_frame(bgr_img, seq=seq, copy_raw=copy_raw)
            if ok and seq > 0:
                now = time.perf_counter()
                if first_display_ts["t"] is None:
                    first_display_ts["t"] = now
                if last_display_ts["t"] is not None:
                    all_gaps_ms.append((now - last_display_ts["t"]) * 1000.0)
                last_display_ts["t"] = now
            return ok

        win.canvas.set_video_frame = _hooked_set_video_frame  # type: ignore[assignment]
        t_start = time.perf_counter()

        sample_timer = QTimer()
        sample_timer.setInterval(100)

        def _sample():
            if (time.perf_counter() - t_start) >= duration_sec:
                sample_timer.stop()
                worker = win._vision_worker
                elapsed = time.perf_counter() - t_start
                new_frames = len(all_gaps_ms) + (1 if first_display_ts["t"] is not None else 0)
                play_span = (
                    (last_display_ts["t"] - first_display_ts["t"])
                    if (first_display_ts["t"] is not None and last_display_ts["t"] is not None)
                    else elapsed
                )
                avg_fps = round(len(all_gaps_ms) / max(0.1, play_span), 2) if all_gaps_ms else 0.0
                avg_fps_total = round(new_frames / max(0.1, elapsed), 2)
                p95_ms = round(float(np.percentile(all_gaps_ms, 95)), 2) if all_gaps_ms else 0.0
                max_gap_ms = round(float(max(all_gaps_ms)), 2) if all_gaps_ms else 0.0
                result = {
                    "executable": sys.executable,
                    "frozen": bool(getattr(sys, "frozen", False)),
                    "duration_sec": round(elapsed, 2),
                    "capture_mode": win._capture_mode_code,
                    "capture_mode_label": win._capture_mode_label,
                    "status_badge": win.lbl_status_badge.text(),
                    "rendered_new_frames": new_frames,
                    "avg_fps": avg_fps,
                    "avg_fps_including_startup": avg_fps_total,
                    "live_window_fps": win._current_preview_fps,
                    "p95_gap_ms": p95_ms,
                    "max_gap_ms": max_gap_ms,
                    "first_frame_ms": round(worker.first_frame_ms, 1) if (worker and worker.first_frame_ms) else None,
                    "reconnect_count": worker.reconnect_count if worker else 0,
                    "max_reconnect_gap_ms": round(worker.max_reconnect_gap_ms, 1) if worker else 0.0,
                    "analysis_rounds": worker.total_analysis_rounds if worker else 0,
                    "last_analysis_ms": win._last_analysis_ms,
                    "scene": win._last_scene,
                }
                win.btn_toggle_stream.setChecked(False)
                win.toggle_live_stream(False)
                win.close()
                if out_path:
                    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                app.quit()

        sample_timer.timeout.connect(_sample)
        sample_timer.start()
        sys.exit(app.exec_())

    sys.exit(app.exec_())


if __name__ == "__main__":
    launch_gui()
