# 模块总结：视觉感知引擎模块 (`core/vision`)

## 1. 模块职责与设计目标
由于《部落冲突》基于 2.5D 渲染引擎，界面无任何 Android 原生控件节点，所有游戏状态判断、数字读取、阵型边界分析均由 `core/vision` 模块完成。本模块在统一的 `1280×720` 分辨率下提供毫秒级的图像识别能力，并内置了**零外部模型文件也可运行的 HSV 形态学降级算法**，确保在手机端和电脑端开箱即用。

## 2. 包含的文件清单
- [template_matcher.py](template_matcher.py)：实现 `TemplateMatcher` 与 `MatchResult`，提供多尺度模板匹配（`find_one`）及带 IoU 非极大值抑制（NMS）的多目标查找（`find_all`）。
- [ocr_scanner.py](ocr_scanner.py)：实现 `OcrScanner`、`EnemyLoot` 与 `VillageStatus`，针对 Supercell 高亮带黑边字体进行 HSV 掩码二值化预处理，精确读取敌方金/水/黑油及己方空闲建筑工人数量（如 `2/5`）。
- [red_zone_detector.py](red_zone_detector.py)：实现 `RedZoneDetector`，动态提取敌方阵型红色部署禁区线外侧的安全下兵环带，并划分为 `NW`（左上）、`NE`（右上）、`SW`（左下）、`SE`（右下）四个进攻象限。
- [deployment_bar.py](deployment_bar.py) 从当前卡框定位单排兵栏，读取数量、选中态和可证实的技能态。
- [dead_base_checker.py](dead_base_checker.py)：实现 `DeadBaseChecker`，通过模板匹配与圣水收集器紫品红满液柱 HSV 连通域斑块分析判定“死鱼”，并实时计算底部英雄槽位绿色生命条占比。
- [__init__.py](__init__.py)：统一导出所有视觉感知类与数据结构。

## 3. 核心算法与技术亮点
1. **多尺度模板匹配 + NMS 去重 (`TemplateMatcher`)**：
   - 支持 `(1.0, 0.95, 1.05)` 多尺度搜索，消除不同手机/模拟器渲染微差；在查找村庄收菜气泡或同类建筑时，通过 `_compute_iou` 非极大值抑制消除同一个图标周围的重叠检测框。
2. **Supercell 字体颜色掩码增强与右下角摧毁率 OCR (`OcrScanner.preprocess_coc_text` & `scan_destruction_rate`)**：
   - 搜鱼界面左上角的数字背景是动态飘动的云层或草地，直接 OCR 极易出错。本模块先提取 `V >= 185, S <= 95` 的高亮字体笔画掩码，放大 2 倍后反色为纯净的白底黑字，使 `RapidOCR (ONNX)` 的识别准确率接近 100%。
   - 新增战场右下角**“摧毁率 XX%” (`DESTRUCTION_RATE_ROI = (518, 572, 1140, 1265)`)** 实时读取 (`scan_destruction_rate`) 与白色笔画指纹签名 (`get_destruction_glyph_signature`)，供战斗状态机判断 10~15 秒内摧毁率是否停滞。
3. **可部署地块与兵栏识别**
   - `RedZoneDetector` 闭合红线短缺口并填充围线内部。截断的红线需要额外的地形证据。草地、地图边缘和 HUD 掩膜一起限制可用区域，按距离边界的余量选点。
   - `DeploymentBarReader` 在定位卡框后读取 `xN` 或 `Nx`，读数缺失保留为空。识别到双排时返回兵栏未识别。
4. **双模态死鱼判定与英雄血条列投影 (`DeadBaseChecker`)**：
   - 支持模板匹配与 HSV 紫红色液柱斑块面积过滤双重死鱼判定；通过按列投影统计英雄头像框顶部的绿色像素占比（`col_has_hp`），毫秒级感知英雄掉血并触发大招。
