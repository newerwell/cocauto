# 模块总结：Windows 电脑端 PyQt5 桌面客户端模块 (`platforms/pc_desktop`)

## 1. 模块职责与设计目标
`platforms/pc_desktop` 模块为电脑端用户提供了一套**零命令行门槛、专业游戏辅助级的 `PyQt5` 现代化深色主题桌面客户端**。用户双击 `.exe` 启动后，即可在一个窗口内完成**模拟器自动发现与 `1280×720` 分辨率锁定、实时画面与红线/OCR 透视预览、直接在游戏画面上用鼠标拖拽框选截取新模板、单步独立功能调试、全套参数持久化修改与一键全自动挂机**。

## 2. 包含的文件清单
- [emulator_host.py](emulator_host.py)：`PcEmulatorHost` 本地模拟器扫描器，自动探测 MuMu12（`16384/7555`）、雷电9（`5555/5557`）、BlueStacks、夜神、逍遥模拟器并一键锁定 `1280×720 (240 DPI)`。
- [qt_styles.py](qt_styles.py)：现代深色电竞主题样式表（`DARK_GAMING_QSS`）。
- [interactive_canvas.py](interactive_canvas.py)：`InteractiveGameCanvas` 16:9 交互式游戏画面画布，支持实时显示视觉诊断图层以及**直接在画面上按住鼠标左键拖拽画框截取 UI 模板**。
- [worker_threads.py](worker_threads.py)：`LiveVisionWorker`（后台异步截图与视觉透视线程）、`BotActionWorker`（后台全自动挂机与 5 项单步调试工作线程）以及 `QtSignalLogHandler`（实时日志信号桥接器）。
- [main_window.py](main_window.py)：`CocDesktopMainWindow` 主界面窗口，整合左侧实时画布与右侧五大业务标签页。
- 根目录 GUI 启动入口：[gui_app.py](../../gui_app.py)。

## 3. 核心交互功能亮点
1. **画布内直接鼠标拖拽框选切图 (`InteractiveGameCanvas`)**：
   - 在“🎯 模板采集校准”标签页选中任意模板后点击 `[在左侧画面鼠标拖拽框选]`，直接在左侧游戏画面上拉出矩形框，自动换算回 `1280×720` 标准坐标并覆盖保存到外部 `assets/templates/` 目录。
2. **五大独立单步调试按钮 (`BotActionWorker`)**：
   - 在“📊 运行看板”标签页提供了 `1️⃣ 测试视角缩小归一化`、`2️⃣ 测试村庄收菜+捐兵+补兵`、`3️⃣ 测试顶部工人建议列表刷墙`、`4️⃣ 测试联机自动搜鱼`、`5️⃣ 测试动态红线检测与智能下兵` 五个独立按钮，无需跑完全流程即可随时单独测试任意环节。
3. **跨 EXE 运行态外部配置与模板热更新**：
   - 配合 [config/paths.py](../../config/paths.py)，所有在界面上修改的参数与框选的模板均保存在 `.exe` 同级外部 `config/user_profile.yaml` 与 `assets/templates/` 中，重启 EXE 永久保留。
