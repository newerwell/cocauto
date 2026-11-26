# 模块总结：跨平台设备控制层 (`core/device`)

## 1. 模块职责与设计目标
`core/device` 模块是屏蔽**电脑端（Windows 安卓模拟器，默认首选 `127.0.0.1:5561`）**与**手机端（Android 真机脱离电脑独立运行）**底层差异的核心桥梁。通过统一的 `BaseDeviceController` 抽象接口、`1280×720` 标准横屏坐标系归一化以及**底层 `/dev/input/event*` Linux Multi-Touch Protocol B 双指并发触控注入**，彻底解决了《部落冲突》不响应键盘 `KEYCODE_ZOOM_OUT (169)` 导致视角无法缩小的问题。

## 2. 包含的文件清单
- [base_controller.py](base_controller.py)：定义 `BaseDeviceController` 抽象基类（`connect`、`screencap`、`tap`、`swipe`、`pinch_zoom_out`）。
- [adb_controller.py](adb_controller.py)：实现 `AdbDeviceController`，支持：
  - 默认置顶连接 **`127.0.0.1:5561`** 部落冲突模拟器端口，并自动探测设备物理分辨率与 `/dev/input/event4` 多点触控节点；
  - 三段式硬件级双指捏合缩小与黄金比例四角居中对齐（`pinch_zoom_out`）；
  - 全链路细粒度实时调试日志输出；
  - `force_abort()` **0.1 秒瞬间强杀机制**（立即 `kill` 当前 ADB 子进程并释放多指触控点）。
- [maatouch_client.py](maatouch_client.py)：实现 `MultiTouchProtocol` 与 `TouchDeviceInfo`，使用 Python `struct.pack` 直接生成 64 位（`<QQHHi`）/ 32 位（`<IIHHi`）Linux `struct input_event` 原生双指并发捏合二进制字节流（`build_raw_mt_pinch_binary`）与紧急触控释放字节流（`build_emergency_release_binary`）。
- [__init__.py](__init__.py)：模块导出入口。

## 3. 核心特性与技术细节
1. **原生 Linux Multi-Touch Protocol B 硬件双指并发捏合 (`build_raw_mt_pinch_binary`)**：
   - 自动解析 `getevent -pl` 与 `uname -m`，定位支持 `ABS_MT_SLOT` 与 `ABS_MT_POSITION_X/Y` 的触控设备节点（如 `127.0.0.1:5561` 上的 `/dev/input/event4`，`HW_X=0..720, HW_Y=0..1280`）。
   - 同时激活 `Slot 0`（手指 0 从左上 `(200, 130)` 滑向中心 `(570, 325)`）与 `Slot 1`（手指 1 从右下 `(1080, 590)` 滑向中心 `(710, 395)`），打包为原生 `input_event` 二进制流一次性写入 `/dev/input/event4`，零延迟将任何放大状态的村庄缩至最远全景视野。
2. **三段式黄金比例四角居中归位 (`pinch_zoom_out`)**：
   - **Step 1**：连续 3 轮硬件级双指向心捏合，确保视野缩至最小极限；
   - **Step 2**：左上边界死角阻尼归零（`(320, 180) -> (960, 540)`），消除随机初始位置；
   - **Step 3**：黄金比例慢速低惯性回拉居中（`(740, 410) -> (590, 355)`，耗时 `650ms`），使 17 本菱形阵型的 12 点、3 点、6 点、9 点四个角同时完整居中。
3. **高速批量连点 (`fast_batch_tap`) 与 0.1 秒瞬间强行停止 (`force_abort`)**：
   - `fast_batch_tap(points, interval_ms=38)` 通过单次 `adb shell sh` 批量执行带 `usleep` 帧间隔与 `±3px` 微幅高斯抖动的连续点击脚本（平均 `~25ms/点`），可在 `< 0.9 秒` 内沿外圈红线安全边界一字划投放下 18~20 个单位，彻底解决逐兵种单独启动 ADB 进程导致兵种放不完的问题。
   - 所有 ADB 调用均通过可中断的 `subprocess.Popen` 执行，点击 GUI 上的 **`[🚨 强行停止]`** 后立即强杀正在运行的 ADB 子进程并向 `/dev/input/event4` 注入 `ABS_MT_TRACKING_ID = -1` 抬起双指。
