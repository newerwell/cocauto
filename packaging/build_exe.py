"""
双阶段 Windows EXE 一键启动自动化打包构建引擎 (Dual-Stage PyInstaller EXE Builder)
按照渐进式发布策略支持两种打包模式，并将 PyAV (av) 及其 FFmpeg 动态库 (av.libs) 完整打入两种发布包：
1. 阶段一 (--mode onedir)：构建绿色便携文件夹版 `dist/COC_AutoApp/COC_AutoApp.exe`，
   自动将 config/ 与 assets/ 复制到 EXE 同级目录，并在项目根目录生成 `一键启动_COC_AutoApp.bat`。
2. 阶段二 (--mode onefile)：构建单文件独立可执行程序 `dist/COC_AutoApp_Single.exe`，
   同步更新项目根目录的 `COC_AutoApp_v2.1_实时流版.exe` 与 `COC_AutoApp.exe`，
   直接读取同级外挂的 config/ 与 assets/。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _collect_pyav_args() -> list[str]:
    """收集 PyAV 隐藏模块与 delvewheel av.libs 下的全部 FFmpeg 动态库"""
    args: list[str] = ["--collect-all", "av"]
    try:
        import av

        site_packages = Path(av.__file__).resolve().parent.parent
        av_libs = site_packages / "av.libs"
        if av_libs.exists():
            for dll in av_libs.glob("*.dll"):
                args.extend(["--add-binary", f"{dll};av.libs"])
            for manifest in av_libs.glob(".load-order-*"):
                args.extend(["--add-data", f"{manifest};av.libs"])
    except Exception as e:
        print(f"[警告] 未能探测本地 av.libs 动态库目录: {e}")
    return args


def copy_external_assets(target_dir: Path) -> None:
    """将 config/ 与 assets/ 复制到目标 EXE 所在同级目录，支持用户在外部直接热替换模板与修改配置"""
    target_dir.mkdir(parents=True, exist_ok=True)
    for folder_name in ("config", "assets"):
        src = ROOT_DIR / folder_name
        dst = target_dir / folder_name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def _base_pyinstaller_cmd(mode_flag: str, name: str) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--windowed",
        mode_flag,
        "--name",
        name,
        "--collect-data",
        "rapidocr_onnxruntime",
    ]
    cmd.extend(_collect_pyav_args())
    cmd.extend(
        [
            "--exclude-module",
            "matplotlib",
            "--exclude-module",
            "scipy",
            "--exclude-module",
            "pandas",
            "--exclude-module",
            "torch",
            "--exclude-module",
            "tensorflow",
            "--exclude-module",
            "streamlit",
            str(ROOT_DIR / "gui_app.py"),
        ]
    )
    return cmd


def build_stage1_onedir() -> Path:
    """阶段一：构建绿色便携文件夹版 EXE (--onedir)，若旧目录被占用则自动输出至 COC_AutoApp_v2.2"""
    print("========== [阶段一] 正在构建绿色便携文件夹版 COC_AutoApp.exe (--onedir) ==========")
    folder_name = "COC_AutoApp"
    test_lock_dir = ROOT_DIR / "dist" / folder_name
    if test_lock_dir.exists():
        try:
            shutil.rmtree(test_lock_dir)
        except PermissionError:
            folder_name = "COC_AutoApp_v2.2"
            print(f"[提示] dist/COC_AutoApp 正在运行中，自动切换输出目录至 dist/{folder_name}")

    cmd = _base_pyinstaller_cmd("--onedir", folder_name)
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)

    out_dir = ROOT_DIR / "dist" / folder_name
    copy_external_assets(out_dir)

    launcher_bat = ROOT_DIR / "一键启动_COC_AutoApp.bat"
    launcher_bat.write_text(
        f'@echo off\r\ncd /d "%~dp0"\r\nstart "" "%~dp0dist\\{folder_name}\\{folder_name}.exe" %*\r\n',
        encoding="gbk",
    )
    exe_path = out_dir / f"{folder_name}.exe"
    print(f"[阶段一完成] 绿色便携版 EXE 已生成: {exe_path}")
    print(f"[阶段一完成] 根目录快捷启动器已更新: {launcher_bat}")
    return exe_path


def build_stage2_onefile() -> Path:
    """阶段二：构建单文件独立版 EXE (--onefile)，同步更新两个单文件发布入口"""
    print("========== [阶段二] 正在构建单文件独立版 COC_AutoApp.exe (--onefile) ==========")
    cmd = _base_pyinstaller_cmd("--onefile", "COC_AutoApp_Single")
    subprocess.run(cmd, cwd=str(ROOT_DIR), check=True)

    single_exe = ROOT_DIR / "dist" / "COC_AutoApp_Single.exe"
    root_exe = ROOT_DIR / "COC_AutoApp.exe"
    live_exe = ROOT_DIR / "COC_AutoApp_v2.1_实时流版.exe"
    v22_exe = ROOT_DIR / "COC_AutoApp_v2.2_全景缩放版.exe"
    v23_exe = ROOT_DIR / "COC_AutoApp_v2.3_视觉闭环缩放版.exe"
    if single_exe.exists():
        copy_external_assets(ROOT_DIR / "dist")
        for target_exe in (v23_exe, v22_exe, live_exe, root_exe):
            try:
                shutil.copy2(single_exe, target_exe)
                print(f"[阶段二完成] 已同步单文件 EXE 至: {target_exe}")
            except PermissionError:
                print(f"[提示] {target_exe.name} 当前正被打开运行中，已跳过覆盖 (可直接运行 {v23_exe.name})")
    return v23_exe


def main():
    parser = argparse.ArgumentParser(description="COC-AutoApp 双阶段 Windows EXE 构建工具")
    parser.add_argument(
        "--mode",
        choices=["onedir", "onefile", "both"],
        default="onedir",
        help="选择打包阶段：onedir (阶段一绿色便携文件夹版), onefile (阶段二单文件独立版), both (依次构建两阶段)",
    )
    args = parser.parse_args()

    if args.mode in ("onedir", "both"):
        build_stage1_onedir()
    if args.mode in ("onefile", "both"):
        build_stage2_onefile()


if __name__ == "__main__":
    main()
