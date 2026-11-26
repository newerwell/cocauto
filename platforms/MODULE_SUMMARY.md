# 模块总结：双端独立运行宿主模块 (`platforms`)

## 1. 模块职责与设计目标
`platforms` 模块为同一套核心业务引擎（`core/`）提供**电脑端（Windows 安卓模拟器）**与**手机端（Android 真机脱离电脑独立运行）**的专属宿主环境适配层。

## 2. 包含的文件清单
- [pc_desktop/emulator_host.py](pc_desktop/emulator_host.py)：Windows 电脑端模拟器宿主 `PcEmulatorHost`，自动发现本地正在运行的 MuMu12（`16384/7555`）、雷电9（`5555/5557`）、BlueStacks、夜神（`62001`）、逍遥（`21503`）模拟器，并可一键对齐 `1280×720 (240 DPI)` 标准分辨率。
- [android_standalone/termux_bootstrap.sh](android_standalone/termux_bootstrap.sh)：Android 手机端脱离电脑独立运行引导脚本。利用 Android 11+ 原生自带的“无线调试（Wireless Debugging）”功能，在手机 Termux 内部直接 `adb connect 127.0.0.1:本机端口`，免 Root、免电脑运行整套 Python 自动化系统。
- [android_standalone/maafw_pipeline/coc_pipeline.json](android_standalone/maafw_pipeline/coc_pipeline.json)：适配 `MaaFramework` 与 `MaaFwApp`（Android 原生 APK + Shizuku 免 Root 驱动）的声明式任务流水线定义文件，支持直接挂载自定义识别器（`CocLootOcrRecognizer`）与红线智能下兵动作（`CocRedZoneSmartDeployAction`）。

## 3. 双端独立运行工作流对比
1. **在电脑上独立运行**：
   - 打开任意安卓模拟器（推荐 MuMu 模拟器 12 或雷电模拟器 9），运行 `python main.py`，`PcEmulatorHost` 自动扫描端口并连接模拟器开始挂机。
2. **在手机上完全脱离电脑独立运行**：
   - **方式 A（Termux + 本机无线调试）**：在手机上打开“开发者选项 -> 无线调试”，运行 `./termux_bootstrap.sh` 输入本机端口即可在手机后台独立运行。
   - **方式 B（MaaFwApp + Shizuku 原生 APK）**：加载 `maafw_pipeline/coc_pipeline.json` 资源包，通过手机上的 Shizuku 授权直接以原生 APK 悬浮窗或虚拟屏幕后台挂机。
