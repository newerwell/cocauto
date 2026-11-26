# 模块总结：视觉模板一键采集与实机校准模块 (`tools/calibrator`)

## 1. 模块职责与设计目标
`tools/calibrator` 模块是连接“代码算法框架”与“真机/模拟器实机稳定运行”的桥梁。它解决了视觉自动化中最繁琐的两大痛点：
1. **无需手动用画图软件抠图**：提供 `1280×720` 标准坐标系下的 **12 张核心 UI 模板一键预设切图**、**鼠标交互式框选切图** 与 **高对比度基准模板初始化**。
2. **实机全链路视觉透视诊断**：一键抓取当前在线安卓模拟器（或手机无线 ADB）的实时画面，在左侧画面标注所有 OCR 区域、红线四边下兵采样点与模板匹配得分，在右侧仪表盘实时贴出 **Supercell 字体二值化预处理缩略图** 与识别读数。

## 2. 包含的文件清单
- [template_harvester.py](template_harvester.py)：实现 `TemplateHarvester` 与 `TemplateSpec`，内置 12 项核心模板规范，支持 `generate_bootstrap_templates()`、`harvest_from_screen_by_scene()`、`crop_custom_roi()` 与 `interactive_crop()`。
- [live_calibrator.py](live_calibrator.py)：实现 `LiveCalibrator`，连接在线 Android 设备生成 `1700×720` 分屏式实机视觉校准诊断大图。
- [__init__.py](__init__.py)：模块导出入口。
- 根目录命令行入口：[calibrate.py](../../calibrate.py)。

## 3. 12 项标准模板规范清单 (`1280×720` 基准)

| 模板 Key | 相对路径 (`assets/templates/`) | 所属场景 (`scene`) | 默认 ROI (`y1, y2, x1, x2`) | 用途说明 |
| :--- | :--- | :--- | :--- | :--- |
| `bubble_gold` | `village/bubble_gold.png` | `village` | `(240, 286, 520, 566)` | 金矿上方漂浮的金币收集气泡 |
| `bubble_elixir` | `village/bubble_elixir.png` | `village` | `(240, 286, 580, 626)` | 圣水收集器上方漂浮的圣水气泡 |
| `bubble_dark` | `village/bubble_dark.png` | `village` | `(240, 286, 640, 686)` | 暗黑重油钻井上方漂浮的黑油气泡 |
| `btn_attack` | `village/btn_attack.png` | `village` | `(585, 665, 65, 165)` | 主村庄左下角 `[进攻!]` 按钮 |
| `btn_find_match` | `village/btn_find_match.png` | `attack_menu` | `(455, 535, 865, 1025)` | 联机模式 `[搜索对手]` 按钮 |
| `btn_donate` | `village/btn_donate.png` | `chat` | `(260, 305, 360, 455)` | 部落聊天栏绿色 `[增援]` 按钮 |
| `donate_troop_slot` | `village/donate_troop_slot.png` | `chat` | `(250, 320, 540, 610)` | 捐兵面板常驻兵种图标 |
| `btn_request_cc` | `village/btn_request_cc.png` | `army` | `(605, 660, 95, 195)` | 部队总览左下角 `[请求增援]` 按钮 |
| `suggest_wall_item` | `village/suggest_wall_item.png` | `builder` | `(260, 302, 465, 645)` | 顶部工人建议下拉菜单 `[城墙]` 项 |
| `btn_next` | `battle/btn_next.png` | `battle` | `(485, 555, 1075, 1205)` | 搜鱼侦察右下角 `[下一个]` 按钮 |
| `full_collector` | `battle/full_collector.png` | `battle` | `(225, 275, 325, 375)` | 满液位圣水收集器紫品红储液柱 |
| `btn_send_request` | `common/btn_send_request.png` | `army` | `(420, 475, 660, 790)` | 发送援军请求确认按钮 |

## 4. 常用校准命令行操作指南
```powershell
# 1. 初始化全套 12 张默认模板文件到 assets/templates/
python calibrate.py --init-templates

# 2. 连接当前在线模拟器/手机，抓取实时画面生成 1700x720 实机校准诊断报告图
python calibrate.py --check-live

# 3. 当模拟器处于主村庄或搜鱼界面时，一键从实机画面切取该场景所有按钮模板
python calibrate.py --harvest village
python calibrate.py --harvest battle

# 4. 弹出鼠标框选窗口，手动从实机画面框选任意模板 (如 bubble_gold)
python calibrate.py --interactive bubble_gold
```
