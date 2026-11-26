"""
跨运行态（Python源码 / PyInstaller --onedir 文件夹版 EXE / --onefile 单文件版 EXE）路径解析器
确保无论以哪种方式打包 EXE，配置文件 (config/) 与视觉模板库 (assets/templates/) 均优先指向 EXE 所在同级外部可写目录，
从而保证用户在 GUI 前端修改的参数与鼠标框选截取的模板图能够永久保存。
"""
from __future__ import annotations
import sys
from pathlib import Path


def get_app_root() -> Path:
    """
    获取应用外部根目录：
    - 若处于 PyInstaller 冻结态 (sys.frozen == True)，返回 .exe 可执行文件所在的外部目录
      (如果 .exe 位于 dist/COC_AutoApp/ 下且同级无 assets，还会智能向上回溯查找项目根目录)
    - 若处于 Python 源码运行态，返回项目根目录
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if (exe_dir / "assets").exists() or (exe_dir / "config").exists():
            return exe_dir
        # 若在 dist/ 或 dist/COC_AutoApp/ 中运行，且外部根目录有 config/assets
        for parent in [exe_dir.parent, exe_dir.parent.parent]:
            if (parent / "config").exists() and (parent / "assets").exists():
                return parent
        return exe_dir
    return Path(__file__).resolve().parents[1]


def get_config_dir() -> Path:
    d = get_app_root() / "config"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_templates_dir() -> Path:
    d = get_app_root() / "assets" / "templates"
    d.mkdir(parents=True, exist_ok=True)
    return d
