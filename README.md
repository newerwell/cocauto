# COC AutoApp

COC AutoApp 是通过 ADB 控制安卓模拟器的 Windows 桌面工具，支持实时预览、双世界渡船、自动收菜和战斗流程。默认参数在 [default_config.yaml](config/default_config.yaml)。桌面界面保存的配置写入 `config/user_profile.yaml`。

## 从源码启动

安装 Python 3.12，将 `adb` 加入 `PATH`，并确认 `adb devices` 能列出目标模拟器。然后在项目根目录运行以下命令。

```powershell
python -m pip install -r requirements.txt
python gui_app.py
```

## 测试和打包

```powershell
python -m pip install pytest pyinstaller
python -m pytest -q
python packaging/build_exe.py --mode both
```

打包后可以双击 `一键启动_COC_AutoApp.bat` 启动文件夹版。单文件版 `COC_AutoApp.exe` 从同级 `assets/` 和 `config/` 加载模板和配置。
