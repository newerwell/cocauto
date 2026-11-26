# 模块总结：配置管理模块 (`config`)

## 1. 模块职责与设计目标
`config` 模块负责管理《部落冲突》跨平台自动化软件的所有运行参数。由于本软件需同时在 **Windows 电脑端（控制安卓模拟器）** 和 **Android 手机端（Termux 本地无线 ADB / Shizuku 独立运行）** 运行，配置模块通过统一的 `1280×720` 标准坐标系与无平台耦合的参数结构，实现了一套配置文件在手机和电脑双端 100% 通用。

## 2. 包含的文件清单
- [default_config.yaml](default_config.yaml)：全局默认 YAML 配置文件，包含设备连接、搜鱼过滤、战斗流派、村庄收菜捐兵、工人建议列表自动刷墙、拟人化防封共 6 大配置域。
- [__init__.py](__init__.py)：基于 Python `dataclass` 的强类型配置加载器与校验器（`AppConfig` 及各子域配置类）。

## 3. 核心特性与技术细节
1. **六大强类型业务配置域**：
   - `DeviceConfig`：支持 `"auto"` 自动探测（自动区分 PC 上的 MuMu/雷电模拟器端口或手机本机无线调试端口）与 `1280×720` 分辨率基准。
   - `LootFilterConfig`：支持单科（金 AND 水）、总和（金 + 水 >= 阈值）、高黑油（Dark Elixir >= 阈值）多维逻辑，以及“仅打死鱼（满采集器数量阈值）”开关。
   - `CombatConfig`：支持流派切换（`surround_barch` 四边环绕平铺流 / `line_air_spam` 单侧一字划空军流）、英雄低血量自动开大阈值（默认 `45%`）及无收益提前退场时间。
   - `VillageConfig`：管理自动收菜、清障、部落请求援军以及基于**白名单/黑名单关键词**的智能捐兵策略。
   - `UpgradeConfig`：管理**“留 N 名空闲工人专门刷墙”**（`reserve_builders_for_wall: 1`）以及刷墙触发资源线。
   - `HumanizerConfig`：管理二维高斯点击偏移（$\sigma=6.0\text{px}$）、随机延迟区间、单次挂机会话上限与夜间仿生睡眠窗口。
2. **极简环境降级容错（Fallback Design）**：
   - 在手机端 Termux 初次运行若尚未安装 `PyYAML` 时，`load_config()` 会自动平滑降级为内置强类型默认值，确保零依赖也可直接启动。

## 4. 对外接口示例
```python
from config import load_config

cfg = load_config()
print(f"当前搜鱼金水总和阈值: {cfg.loot_filter.min_gold_plus_elixir}")
print(f"预留专门刷墙工人数: {cfg.upgrade.reserve_builders_for_wall}")
```
