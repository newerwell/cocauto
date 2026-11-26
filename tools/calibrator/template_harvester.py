"""
视觉模板一键采集器与预设切图引擎 (Template Harvester & Bootstrap Generator)
负责：
1. 定义 1280x720 标准分辨率下 10 张核心 UI 模板的标准 ROI 坐标与元数据
2. 支持从真机/模拟器实时画面按场景一键切图 (harvest_scene_presets)
3. 支持弹出鼠标交互式框选窗口 (interactive_crop) 截取任意自定义按钮或气泡
4. 支持一键初始化全套高对比度基础模板 (generate_bootstrap_templates)，确保 assets/templates/ 完整无缺
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np


@dataclass
class TemplateSpec:
    rel_path: str
    scene: str                  # "village" | "battle" | "chat" | "builder" | "army"
    default_roi: Tuple[int, int, int, int]  # (y1, y2, x1, x2) in 1280x720
    description: str
    theme_bgr: Tuple[int, int, int]
    label_text: str
    manual_roi: bool = False


class TemplateHarvester:
    """1280x720 跨平台共享视觉模板采集器与生成器"""

    TEMPLATE_SPECS: Dict[str, TemplateSpec] = {
        "bubble_gold": TemplateSpec(
            rel_path="village/bubble_gold.png",
            scene="village",
            default_roi=(240, 286, 520, 566),
            description="金矿上方漂浮的金币收集气泡",
            theme_bgr=(30, 210, 255),
            label_text="GOLD",
        ),
        "bubble_elixir": TemplateSpec(
            rel_path="village/bubble_elixir.png",
            scene="village",
            default_roi=(240, 286, 580, 626),
            description="圣水收集器上方漂浮的圣水收集气泡",
            theme_bgr=(210, 45, 240),
            label_text="ELIX",
        ),
        "bubble_dark": TemplateSpec(
            rel_path="village/bubble_dark.png",
            scene="village",
            default_roi=(240, 286, 640, 686),
            description="暗黑重油钻井上方漂浮的黑油气泡",
            theme_bgr=(55, 45, 65),
            label_text="DARK",
        ),
        "btn_attack": TemplateSpec(
            rel_path="village/btn_attack.png",
            scene="village",
            default_roi=(585, 665, 65, 165),
            description="主村庄左下角 [进攻!] 按钮",
            theme_bgr=(35, 145, 245),
            label_text="ATK",
        ),
        "btn_find_match": TemplateSpec(
            rel_path="village/btn_find_match.png",
            scene="attack_menu",
            default_roi=(455, 535, 865, 1025),
            description="联机模式 [搜索对手] 橙色按钮",
            theme_bgr=(40, 165, 250),
            label_text="FIND",
        ),
        "btn_donate": TemplateSpec(
            rel_path="village/btn_donate.png",
            scene="chat",
            default_roi=(260, 305, 360, 455),
            description="左侧部落聊天栏内的绿色 [增援] 按钮",
            theme_bgr=(60, 205, 110),
            label_text="DONATE",
        ),
        "donate_troop_slot": TemplateSpec(
            rel_path="village/donate_troop_slot.png",
            scene="chat",
            default_roi=(250, 320, 540, 610),
            description="捐兵面板中的常驻捐赠兵种图标槽位",
            theme_bgr=(180, 120, 60),
            label_text="TROOP",
        ),
        "btn_request_cc": TemplateSpec(
            rel_path="village/btn_request_cc.png",
            scene="army",
            default_roi=(605, 660, 95, 195),
            description="军队训练总览左下角 [请求增援] 按钮",
            theme_bgr=(70, 195, 120),
            label_text="REQ_CC",
        ),
        "suggest_wall_item": TemplateSpec(
            rel_path="village/suggest_wall_item.png",
            scene="builder",
            default_roi=(260, 302, 465, 645),
            description="顶部建筑工人建议下拉列表中的 [城墙] 条目",
            theme_bgr=(110, 160, 190),
            label_text="WALL",
        ),
        "btn_next": TemplateSpec(
            rel_path="battle/btn_next.png",
            scene="battle",
            default_roi=(485, 555, 1075, 1205),
            description="搜鱼侦察界面右下角 [下一个] 按钮",
            theme_bgr=(40, 170, 245),
            label_text="NEXT",
        ),
        "full_collector": TemplateSpec(
            rel_path="battle/full_collector.png",
            scene="battle",
            default_roi=(225, 275, 325, 375),
            description="满液位圣水收集器紫品红储液罐特征图",
            theme_bgr=(210, 30, 230),
            label_text="FULL",
        ),
        "btn_send_request": TemplateSpec(
            rel_path="common/btn_send_request.png",
            scene="army",
            default_roi=(420, 475, 660, 790),
            description="发送部落援军请求弹窗中的绿色 [发送] 按钮",
            theme_bgr=(55, 210, 105),
            label_text="SEND",
        ),
        "boat_to_night": TemplateSpec(
            rel_path="village/boat_to_night.png",
            scene="village",
            default_roi=(505, 588, 228, 298),
            description="主世界左下角海岸前往夜世界的红白条纹渡船船体",
            theme_bgr=(65, 110, 195),
            label_text="BOAT",
            manual_roi=True,
        ),
        "boat_bubble": TemplateSpec(
            rel_path="village/boat_bubble.png",
            scene="village",
            default_roi=(460, 506, 250, 290),
            description="主世界左下角渡船桅杆上方的悬浮小船气泡图标",
            theme_bgr=(215, 210, 195),
            label_text="SHIP_B",
            manual_roi=True,
        ),
        "boat_sail_home": TemplateSpec(
            rel_path="village/boat_sail.png",
            scene="village",
            default_roi=(0, 0, 0, 0),
            description="主世界木船红白条纹帆，手动画框采集",
            theme_bgr=(65, 110, 195),
            label_text="SAIL",
            manual_roi=True,
        ),
        "night_bubble_gold": TemplateSpec(
            rel_path="night/bubble_gold.png",
            scene="night",
            default_roi=(398, 424, 725, 751),
            description="夜世界金矿上方漂浮的方形金币收集气泡",
            theme_bgr=(30, 215, 255),
            label_text="N_GOLD",
        ),
        "night_bubble_elixir": TemplateSpec(
            rel_path="night/bubble_elixir.png",
            scene="night",
            default_roi=(89, 115, 826, 852),
            description="夜世界圣水收集器上方漂浮的圣水收集气泡",
            theme_bgr=(220, 75, 210),
            label_text="N_ELIX",
        ),
        "btn_night_attack": TemplateSpec(
            rel_path="night/btn_night_attack.png",
            scene="night",
            default_roi=(585, 695, 18, 120),
            description="夜世界左下角双斧 [战斗] 按钮",
            theme_bgr=(40, 185, 245),
            label_text="N_ATK",
        ),
        "boat_to_home": TemplateSpec(
            rel_path="night/boat_to_home.png",
            scene="night",
            default_roi=(20, 140, 958, 1025),
            description="夜世界右上角码头返回主世界的渡船",
            theme_bgr=(65, 115, 190),
            label_text="N_BOAT",
            manual_roi=True,
        ),
        "boat_sail_night": TemplateSpec(
            rel_path="night/boat_sail.png",
            scene="night",
            default_roi=(0, 0, 0, 0),
            description="夜世界木船红白条纹帆，手动画框采集",
            theme_bgr=(65, 115, 190),
            label_text="N_SAIL",
            manual_roi=True,
        ),
        "boat_bubble_night": TemplateSpec(
            rel_path="night/boat_bubble.png",
            scene="night",
            default_roi=(0, 0, 0, 0),
            description="夜世界木船上方气泡，手动画框采集",
            theme_bgr=(215, 210, 195),
            label_text="N_SHIP_B",
            manual_roi=True,
        ),
        "btn_star_bonus_ok": TemplateSpec(
            rel_path="night/btn_star_bonus_ok.png",
            scene="night",
            default_roi=(520, 605, 545, 735),
            description="夜世界/主世界【胜利之星奖励！】弹窗底部绿色 [确定] 按钮",
            theme_bgr=(65, 205, 110),
            label_text="STAR_OK",
        ),
    }

    def __init__(self, assets_dir: str | Path | None = None):
        from config.paths import get_templates_dir
        self.assets_dir = Path(assets_dir) if assets_dir else get_templates_dir()

    def generate_bootstrap_templates(self, overwrite: bool = False) -> List[Path]:
        """
        生成全套 12 张核心基准模板文件到 assets/templates/ 目录（若尚不存在或指定 overwrite=True）。
        生成的模板带有与游戏 UI 一致的高对比度色彩特征与边框，可直接用于管线验证并在后续被实机截图无缝覆盖。
        """
        saved: List[Path] = []
        for key, spec in self.TEMPLATE_SPECS.items():
            out_path = self.assets_dir / spec.rel_path
            if spec.manual_roi:
                if out_path.exists():
                    saved.append(out_path)
                continue
            if out_path.exists() and not overwrite:
                saved.append(out_path)
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            y1, y2, x1, x2 = spec.default_roi
            h, w = max(24, y2 - y1), max(32, x2 - x1)
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:, :] = spec.theme_bgr
            cv2.rectangle(img, (2, 2), (w - 3, h - 3), (245, 245, 245), 2)
            cv2.putText(
                img,
                spec.label_text,
                (6, max(16, h // 2 + 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (20, 20, 20),
                2,
            )
            cv2.imwrite(str(out_path), img)
            saved.append(out_path)
        return saved

    def harvest_from_screen_by_scene(self, screen_bgr: np.ndarray, scene: str) -> List[Path]:
        """
        当真机/模拟器处于指定游戏场景 (如 'village' 主村庄 或 'battle' 搜鱼界面) 时，
        一键按照标准 ROI 坐标从当前 1280x720 画面切取所有对应按钮模板并保存。
        """
        saved: List[Path] = []
        for key, spec in self.TEMPLATE_SPECS.items():
            if spec.manual_roi or (spec.scene != scene and scene != "all"):
                continue
            y1, y2, x1, x2 = spec.default_roi
            crop = screen_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            out_path = self.assets_dir / spec.rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), crop)
            saved.append(out_path)
        return saved

    def crop_custom_roi(
        self,
        screen_bgr: np.ndarray,
        template_key: str,
        roi: Tuple[int, int, int, int],
    ) -> Path:
        """按指定 (y1, y2, x1, x2) 裁剪并保存指定 key 的模板"""
        if template_key in self.TEMPLATE_SPECS:
            rel_path = self.TEMPLATE_SPECS[template_key].rel_path
        else:
            rel_path = template_key if template_key.endswith(".png") else f"common/{template_key}.png"

        y1, y2, x1, x2 = roi
        crop = screen_bgr[y1:y2, x1:x2]
        out_path = self.assets_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), crop)
        return out_path

    def interactive_crop(self, screen_bgr: np.ndarray, template_key: str) -> Optional[Path]:
        """弹出 OpenCV 交互式鼠标框选窗口，框选后按回车或空格自动保存为目标模板"""
        win_name = f"Select ROI for [{template_key}] (Press ENTER to save, ESC to cancel)"
        rect = cv2.selectROI(win_name, screen_bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win_name)
        x, y, w, h = [int(v) for v in rect]
        if w <= 4 or h <= 4:
            return None
        return self.crop_custom_roi(screen_bgr, template_key, (y, y + h, x, x + w))
