# 模块总结：双端共享视觉资产库模块 (`assets`)

## 1. 模块职责与设计目标
`assets` 模块存放手机真机与电脑安卓模拟器 **100% 共享的 `1280×720` 标准分辨率视觉模板图与轻量级离线 AI 模型**。由于设备控制层已将所有设备画面自动归一化为 `1280×720`，因此任何在一端截取的按钮模板图在另一端无需任何修改即可直接匹配。

## 2. 目录结构规范
```text
assets/
├── MODULE_SUMMARY.md             # 本模块说明与模板截取规范文档
├── templates/                    # 1280x720 基准分辨率下的 PNG 裁剪模板
│   ├── village/                  # 主村庄界面模板
│   │   ├── bubble_gold.png       # 金矿上方漂浮的金币收集气泡
│   │   ├── bubble_elixir.png     # 圣水收集器上方漂浮的圣水气泡
│   │   ├── bubble_dark.png       # 暗黑重油钻井上方漂浮的黑油气泡
│   │   ├── btn_donate.png        # 部落聊天栏内的绿色 [增援] 按钮
│   │   ├── donate_troop_slot.png # 捐兵弹窗中常驻捐赠兵种图标
│   │   ├── btn_request_cc.png    # 部队总览内的 [请求增援] 按钮
│   │   └── suggest_wall_item.png # 顶部建筑工人建议下拉列表中的 [城墙] 条目
│   ├── battle/                   # 战斗与搜鱼界面模板
│   │   ├── btn_next.png          # 搜鱼界面右下角 [下一个] 按钮
│   │   └── full_collector.png    # (可选) 满液位圣水收集器模板 (未提供时自动启用 HSV 紫红液柱连通域检测)
│   └── common/                   # 通用弹窗模板
│       └── btn_send_request.png  # 发送部落援军请求确认按钮
└── models/                       # (可选) 离线推理 ONNX 模型存放目录
```

## 3. 零模板文件下的高可用降级设计
即使在初次克隆项目、`assets/templates/` 目录尚未放入自定义截图时，系统依然能够依靠内置的**标准相对坐标 + HSV 色彩形态学引擎**正常工作：
- **死鱼识别**：当 `battle/full_collector.png` 不存在时，`DeadBaseChecker` 自动切换为 HSV 紫品红圣水液柱连通域斑块分析。
- **红线检测**：`RedZoneDetector` 完全基于实时画面 HSV 红色禁区掩码与形态学膨胀差分计算，不依赖任何静态模板图片。
- **资源与工人数量读取**：`OcrScanner` 直接基于 `1280×720` 标准 ROI 与颜色掩码二值化读取。
