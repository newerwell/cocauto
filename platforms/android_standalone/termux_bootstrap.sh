#!/data/data/com.termux/files/usr/bin/bash
# ==============================================================================
# 《部落冲突》手机端脱离电脑独立运行一键引导脚本 (Termux + Android 11+ 本机无线调试)
# 用途：在 Android 手机本地 Termux 终端中配对并连接本机无线 ADB (localhost)，
#       直接运行与电脑端 100% 相同的 Python 核心引擎，无需连接任何电脑！
# ==============================================================================

set -e

echo "========================================================"
echo "  COC-AutoApp 手机端独立运行引导程序 (Termux Standalone)"
echo "========================================================"

# 1. 检查并安装 Termux 必需的软件包
if ! command -v adb &> /dev/null || ! command -v python3 &> /dev/null; then
    echo "[1/3] 正在安装 android-tools (ADB), python, opencv..."
    pkg update -y
    pkg install -y android-tools python python-numpy opencv-python pyyaml
fi

# 2. 检查本机无线 ADB 是否已连接
ONLINE_COUNT=$(adb devices | grep -w "device" | wc -l)
if [ "$ONLINE_COUNT" -eq 0 ]; then
    echo "[2/3] 当前尚未连接本机无线调试。"
    echo "请在手机【开发者选项】->【无线调试】中开启无线调试："
    read -p "请输入本机无线调试 IP:端口 (例如 127.0.0.1:38945): " LOCAL_ADDR
    adb connect "$LOCAL_ADDR"
fi

echo "[3/3] 正在启动 COC-AutoApp 跨平台核心状态机..."
python3 ../../main.py --serial auto
