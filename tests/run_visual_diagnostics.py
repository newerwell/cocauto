"""
视觉感知与拟人化引擎可视化诊断测试脚本 (Visual Diagnostic & Benchmark Runner)
生成一张 1280x720 的综合视觉诊断图，直观展示：
1. 动态红线 (Red-Zone) 提取出的安全下兵环带与 NW/NE/SW/SE 四个方向的实际采样点
2. 死鱼圣水收集器满液位紫红斑块的自动框选检测结果
3. 底部受损英雄血条百分比实时检测结果
4. 拟人化三阶贝塞尔曲线滑动轨迹与二维高斯点击散点分布
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.humanizer import HumanJitter
from core.vision import RedZoneDetector, DeadBaseChecker, OcrScanner
from test_all_modules import MockAndroidDevice


def run_visual_diagnostics(output_path: str) -> None:
    t0 = time.perf_counter()
    dev = MockAndroidDevice()
    screen = dev.screencap()
    vis = screen.copy()

    # 1. 测试并绘制动态红线四边下兵点
    t_red0 = time.perf_counter()
    red_detector = RedZoneDetector()
    sides = red_detector.detect_deployable_sides(screen, points_per_side=12)
    red_ms = (time.perf_counter() - t_red0) * 1000.0

    # 绘制草地有效菱形边界
    cv2.polylines(vis, [red_detector.MAP_DIAMOND], isClosed=True, color=(180, 180, 180), thickness=1)

    side_colors = {
        "NW": (255, 220, 0),   # 青色: 左上边
        "NE": (0, 255, 255),   # 黄色: 右上边
        "SW": (255, 100, 255), # 粉色: 左下边
        "SE": (0, 255, 120),   # 亮绿: 右下边
    }
    for side_name, pts in sides.items():
        color = side_colors[side_name]
        for idx, (px, py) in enumerate(pts):
            cv2.circle(vis, (px, py), 6, color, -1)
            cv2.circle(vis, (px, py), 8, (0, 0, 0), 1)
            if idx == len(pts) // 2:
                cv2.putText(vis, f"{side_name}({len(pts)}pts)", (px - 25, py - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 2. 测试并标注死鱼满采集器检测
    t_dead0 = time.perf_counter()
    checker = DeadBaseChecker()
    full_count = checker.count_full_elixir_collectors(screen)
    is_dead = checker.is_dead_base(screen, min_full_collectors=3)
    dead_ms = (time.perf_counter() - t_dead0) * 1000.0

    for cx, cy in [(350, 250), (920, 250), (350, 460), (920, 460)]:
        cv2.rectangle(vis, (cx - 22, cy - 22), (cx + 22, cy + 22), (0, 255, 0), 2)
        cv2.putText(vis, "Full Elixir", (cx - 32, cy - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # 3. 测试并标注底部英雄血条检测
    hero_box = (680, 595, 760, 695)
    hp_ratio = checker.check_hero_hp_ratio(screen, hero_box)
    cv2.rectangle(vis, (hero_box[0], hero_box[1]), (hero_box[2], hero_box[3]), (0, 165, 255), 2)
    cv2.putText(vis, f"Hero#1 HP:{hp_ratio:.0%} -> AUTO ABILITY!",
                (hero_box[0] - 40, hero_box[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)

    # 4. 测试并绘制拟人化三阶贝塞尔滑动轨迹 + 二维高斯点击热力散点
    jitter = HumanJitter(sigma_px=6.0, max_offset_px=15)
    bezier_pts = jitter.generate_bezier_curve((180, 580), (580, 320), num_steps=18)
    for i in range(len(bezier_pts) - 1):
        cv2.line(vis, bezier_pts[i], bezier_pts[i + 1], (255, 200, 50), 2)
        cv2.circle(vis, bezier_pts[i], 3, (0, 255, 255), -1)
    cv2.putText(vis, "Cubic Bezier Swipe (Ease-In-Out)", (160, 610),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 50), 1)

    # 绘制按钮 (1140, 520) [下一个] 周围 40 次二维高斯随机点击分布
    target_btn = (1140, 520)
    cv2.circle(vis, target_btn, 18, (255, 255, 255), 1)
    for _ in range(40):
        gx, gy = jitter.jitter_point(*target_btn)
        cv2.circle(vis, (gx, gy), 2, (0, 255, 255), -1)
    cv2.putText(vis, "2D Gaussian Tap Jitter (40 taps)", (1010, 555),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    total_ms = (time.perf_counter() - t0) * 1000.0
    # 顶部诊断信息看板
    cv2.rectangle(vis, (15, 12), (820, 55), (20, 20, 20), -1)
    summary_text = (
        f"COC-AutoApp Visual Diagnostic | RedZone: {red_ms:.1f}ms | "
        f"DeadBase({full_count} full={is_dead}): {dead_ms:.1f}ms | Total: {total_ms:.1f}ms"
    )
    cv2.putText(vis, summary_text, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180), 2)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), vis)
    print(f"VISUAL_DIAGNOSTIC_SAVED: {out_file}")
    print(f"BENCHMARK: RedZone={red_ms:.2f}ms, DeadBase={dead_ms:.2f}ms, Total={total_ms:.2f}ms")


if __name__ == "__main__":
    out_arg = sys.argv[1] if len(sys.argv) > 1 else str(ROOT_DIR / "tests" / "visual_diagnostic_result.png")
    run_visual_diagnostics(out_arg)
