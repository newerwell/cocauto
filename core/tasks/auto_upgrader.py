"""
自动升级建筑与刷墙状态机 (Auto Upgrader Task)
采用工业级“顶部建筑工人建议列表 (Builder Suggestions)”零误差定位方案：
1. OCR 读取顶部状态栏的空闲工人数 (如 '2/5') 与当前拥有的金币、圣水
2. 若空闲工人 >= 1 且金币/圣水超过刷墙阈值，点击工人头像打开建议下拉列表，匹配“城墙 (Wall)”自动聚焦并升级
3. 若空闲工人 > 预留专门刷墙工人数 (如 reserve_builders_for_wall = 1)，则从建议列表中选择首个非城墙建筑自动升级
"""
from __future__ import annotations
import logging
import time
from typing import Tuple

from config import UpgradeConfig
from core.device import BaseDeviceController
from core.vision import TemplateMatcher, OcrScanner, VillageStatus

logger = logging.getLogger("coc.tasks.upgrade")


class AutoUpgraderTask:
    """基于顶部建筑工人建议列表的零误差自动刷墙与建筑升级引擎"""

    # 1280x720 下的建筑工人建议面板坐标 (最新 UI 工人头像位于右下角 [建造] 上方)
    BTN_BUILDER_ICON = (1162, 555)       # 右下角建筑工人头像 ("0/6")
    SUGGEST_LIST_ROI = (250, 560, 780, 1220)# 工人建议列表区域 (y1, y2, x1, x2)
    FIRST_SUGGEST_ITEM = (980, 360)      # 建议列表第一项坐标
    BTN_UPGRADE_GOLD = (640, 615)        # 底部中央金币升级按钮
    BTN_UPGRADE_ELIXIR = (760, 615)      # 底部中央圣水升级按钮 (9级以上城墙支持圣水刷墙)
    BTN_CONFIRM_UPGRADE = (640, 485)     # 升级确认按钮

    def __init__(
        self,
        device: BaseDeviceController,
        matcher: TemplateMatcher,
        ocr: OcrScanner,
        cfg: UpgradeConfig,
    ):
        self.device = device
        self.matcher = matcher
        self.ocr = ocr
        self.cfg = cfg

    def plan_upgrade_actions(self, status: VillageStatus) -> Tuple[bool, bool, str]:
        """
        根据当前村庄空闲工人与资源储量，纯逻辑计算本次是否应当执行刷墙与建筑升级。
        返回: (can_upgrade_wall, can_upgrade_building, decision_summary)
        """
        if status.free_builders <= 0:
            return False, False, "当前无空闲建筑工人 (0 空闲)"

        can_wall = False
        if self.cfg.enable_auto_wall:
            if status.gold >= self.cfg.wall_min_gold or status.elixir >= self.cfg.wall_min_elixir:
                can_wall = True

        can_building = False
        if self.cfg.enable_auto_building:
            # 只有当空闲工人数大于“预留给刷墙的工人数”时，才允许派工人去升耗时长的普通建筑
            if status.free_builders > self.cfg.reserve_builders_for_wall:
                if status.gold >= self.cfg.keep_minimum_resource or status.elixir >= self.cfg.keep_minimum_resource:
                    can_building = True

        reason = (
            f"空闲工人={status.free_builders}/{status.total_builders}, "
            f"金币={status.gold:,}, 圣水={status.elixir:,} -> "
            f"刷墙={can_wall}, 升建筑={can_building}"
        )
        return can_wall, can_building, reason

    def run_check_and_upgrade(self) -> dict[str, int]:
        """扫描村庄顶部工人与资源状态，按策略自动刷墙与升级建筑"""
        screen = self.device.screencap()
        status = self.ocr.scan_village_status(screen)
        can_wall, can_building, summary = self.plan_upgrade_actions(status)
        logger.info(f"[内政升级检查] {summary}")

        result = {"walls_upgraded": 0, "buildings_upgraded": 0}
        if not can_wall and not can_building:
            return result

        if can_wall:
            use_elixir = status.elixir >= self.cfg.wall_min_elixir and status.elixir > status.gold
            if self._upgrade_wall_via_builder_menu(use_elixir=use_elixir):
                result["walls_upgraded"] += 1

        if can_building:
            if self._upgrade_recommended_building():
                result["buildings_upgraded"] += 1

        return result

    def _upgrade_wall_via_builder_menu(self, use_elixir: bool = False) -> bool:
        """点击顶部工人头像 -> 在建议列表中找到[城墙] -> 游戏自动定位选中 -> 点击底部升级按钮"""
        self.device.tap(*self.BTN_BUILDER_ICON)
        time.sleep(0.5)
        screen = self.device.screencap()

        # 优先在工人建议下拉菜单 ROI 中模板匹配“城墙”条目
        wall_item = self.matcher.find_one(
            screen,
            "village/suggest_wall_item.png",
            threshold=0.78,
            roi=self.SUGGEST_LIST_ROI,
        )
        if wall_item:
            self.device.tap(*wall_item.center)
        else:
            # 向下滑动建议列表到底部（城墙通常排在建议列表最下方）
            self.device.swipe((575, 320), (575, 120), duration_ms=300)
            time.sleep(0.4)
            screen = self.device.screencap()
            wall_item = self.matcher.find_one(
                screen,
                "village/suggest_wall_item.png",
                threshold=0.78,
                roi=self.SUGGEST_LIST_ROI,
            )
            if not wall_item:
                # 关闭建议下拉菜单
                self.device.tap(640, 15)
                return False
            self.device.tap(*wall_item.center)

        time.sleep(0.5)
        upgrade_btn = self.BTN_UPGRADE_ELIXIR if use_elixir else self.BTN_UPGRADE_GOLD
        self.device.tap(*upgrade_btn)
        time.sleep(0.4)
        self.device.tap(*self.BTN_CONFIRM_UPGRADE)
        logger.info(f"已通过工人建议列表成功自动刷墙 1 块 ({'圣水' if use_elixir else '金币'})")
        return True

    def _upgrade_recommended_building(self) -> bool:
        """点击顶部工人头像 -> 点击建议升级列表中的第 1 项推荐建筑 -> 确认升级"""
        self.device.tap(*self.BTN_BUILDER_ICON)
        time.sleep(0.5)
        self.device.tap(*self.FIRST_SUGGEST_ITEM)
        time.sleep(0.5)
        self.device.tap(*self.BTN_UPGRADE_GOLD)
        time.sleep(0.4)
        self.device.tap(*self.BTN_CONFIRM_UPGRADE)
        logger.info("已通过工人建议列表自动启动 1 项推荐建筑升级")
        return True
