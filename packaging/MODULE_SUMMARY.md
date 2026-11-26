# 模块总结：双阶段 Windows `.exe` 一键启动打包模块 (`packaging`)

## 1. 模块职责与设计目标
`packaging` 模块负责将整套《部落冲突》PyQt5 桌面客户端与视觉自动化引擎编译为无需安装 Python 环境的原生 Windows `.exe` 可执行程序。按照“先完成阶段一绿色便携文件夹版、在功能完全实现并验证后再构建阶段二单文件独立版”的策略，本模块同时交付了两种形态的 `.exe` 一键启动方案。

## 2. 包含的文件与构建产物清单
- 构建脚本：[packaging/build_exe.py](build_exe.py)
- **阶段一产物（绿色便携文件夹版 `--onedir`，双击秒开无需解压等待）**：
  - 主程序：`dist/COC_AutoApp/COC_AutoApp.exe`
  - 根目录快捷启动器：[一键启动_COC_AutoApp.bat](../一键启动_COC_AutoApp.bat)
- **阶段二产物（单文件独立可执行程序 `--onefile`，外挂同级 `config/` 与 `assets/`）**：
  - 根目录单文件 EXE：`COC_AutoApp.exe`
  - 构建输出备份：`dist/COC_AutoApp_Single.exe`

## 3. 核心工程特性
1. **外部可写配置与模板热替换 (`config/paths.py`)**：
   - 无论是阶段一的 `dist/COC_AutoApp/COC_AutoApp.exe` 还是阶段二根目录下的单文件 `COC_AutoApp.exe`，启动后都会自动定位 `.exe` 所在同级目录下的 `config/user_profile.yaml` 与 `assets/templates/`。
   - 用户在 GUI 界面上用鼠标拖拽框选截取的新按钮模板，或者修改的搜鱼金水阈值，都会直接保存在外部文件夹中，重启 `.exe` 永久生效。
2. **自动裁剪冗余科学计算包**：
   - 构建脚本自动通过 `--exclude-module` 剔除 `matplotlib`、`scipy`、`pandas`、`torch`、`tensorflow`、`streamlit` 等与游戏挂机无关的大型包，大幅加快启动速度并缩减体积。

## 4. 重新打包命令
```powershell
# 仅构建阶段一（绿色便携文件夹版 EXE）
python packaging/build_exe.py --mode onedir

# 仅构建阶段二（单文件独立版 EXE）
python packaging/build_exe.py --mode onefile

# 同时构建两阶段全部产物
python packaging/build_exe.py --mode both
```
