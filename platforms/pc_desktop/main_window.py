"""
PyQt5 现代化深色主题桌面客户端主窗口 (COC-AutoApp Desktop Main Window v2.2)

高性能实时预览架构：
1. 🎥 30FPS H.264 实时画面流：采集与识别彻底解耦，Qt 主线程每 ~33ms 拉取最新帧，仅在帧序号变化时更新画布，
   不重复绘制同一帧、不虚抬 FPS；状态栏分别显示采集方式、实际预览 FPS 与识别耗时。
2. 🧩 矢量叠加与即时清理：识别结果以结构化数据发布，画布按图层开关即时绘制；切换设备、场景变化、关闭图层或重开流时立即清除旧标注并丢弃旧会话回调。
3. 📸 无损诊断与冻结框选：单帧诊断始终使用无损 PNG；进入模板框选时自动暂停预览并抓取冻结一张无损 PNG，鼠标探针与裁切共享同一套 1280x720 坐标。
4. 🖱️ 异步画布直控：点击画面直控模拟器在后台线程执行 ADB 点击，不阻塞主线程渲染。
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import AppConfig, load_config, save_user_config
from core.device import AdbDeviceController, BaseDeviceController
from tools.calibrator import TemplateHarvester
from .combat_run import deployment_lines
from .emulator_host import PcEmulatorHost
from .interactive_canvas import InteractiveGameCanvas
from .qt_styles import DARK_GAMING_QSS
from .worker_threads import BotActionWorker, LiveVisionWorker, QtSignalLogHandler, VisionOverlayData

logger = logging.getLogger("coc.gui")


class CocDesktopMainWindow(QMainWindow):
    """《部落冲突》跨平台自动化软件 PC 桌面实时调试与总控台"""

    botCycleCompleted = pyqtSignal(dict)
    botRunLogOpened = pyqtSignal(str)

    RENDER_POLL_MS = 16       # 16ms 轻量检查帧序号，降低帧到达后的等待抖动
    MIN_RENDER_INTERVAL_SEC = 0.030  # 目标刷新上限 ~33 FPS（实测稳在 ~30 FPS）

    def __init__(self, cfg: Optional[AppConfig] = None, mock_device: Optional[BaseDeviceController] = None):
        super().__init__()
        self.cfg = cfg or load_config()
        self.mock_device = mock_device
        self.harvester = TemplateHarvester()
        self.harvester.generate_bootstrap_templates(overwrite=False)

        self._vision_worker: Optional[LiveVisionWorker] = None
        self._bot_worker: Optional[BotActionWorker] = None
        self._vision_session_id: int = 0
        self._last_rendered_seq: int = 0
        self._last_scene: Optional[str] = None
        self._last_analysis_ms: Optional[float] = None
        self._last_overlay: Optional[VisionOverlayData] = None
        self._capture_mode_code: str = "idle"
        self._capture_mode_label: str = "待命"
        self._active_device_label: str = self.cfg.device.serial
        self._pending_crop_key: Optional[str] = None
        self._updating_device_combo: bool = False

        # 按画布真实渲染的新帧序号统计预览 FPS（滑动窗口 2 秒）
        self._frame_display_times: Deque[float] = deque(maxlen=180)
        self._recent_display_gaps_ms: Deque[float] = deque(maxlen=600)
        self._last_frame_display_ts: Optional[float] = None
        self._current_preview_fps: float = 0.0
        self._last_status_refresh_ts: float = 0.0
        self._rendered_new_frame_count: int = 0

        self._total_cycles = 0
        self._total_bubbles = 0
        self._total_walls = 0

        # Qt 主线程轻量拉帧器（16ms 检查新帧序号 + 30ms 节流，目标 ~30 FPS 且无新帧直接跳过）
        self._render_timer = QTimer(self)
        self._render_timer.setTimerType(Qt.PreciseTimer)
        self._render_timer.setInterval(self.RENDER_POLL_MS)
        self._render_timer.timeout.connect(self._on_render_timer_tick)

        self.setWindowTitle("COC-AutoApp 《部落冲突》全自动辅助控制台 v2.2 (H.264 低延迟实时流版)")
        self.resize(1380, 860)
        self.setStyleSheet(DARK_GAMING_QSS)

        self._init_ui()
        self._attach_log_handler()
        self._load_config_into_widgets()
        self._refresh_template_list()
        self.scan_emulators(silent=True)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # ================= 1. 顶部模拟器连接与实时直播控制栏 =================
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("📱 目标设备/模拟器:"))
        self.combo_device = QComboBox()
        self.combo_device.setEditable(True)
        self.combo_device.setMinimumWidth(220)
        self.combo_device.addItem(self.cfg.device.serial)
        self.combo_device.currentTextChanged.connect(self._on_device_changed)
        top_bar.addWidget(self.combo_device)

        btn_scan = QPushButton("🔍 扫描模拟器")
        btn_scan.clicked.connect(lambda: self.scan_emulators(silent=False))
        top_bar.addWidget(btn_scan)

        btn_fix_res = QPushButton("📐 一键锁定 1280x720")
        btn_fix_res.clicked.connect(self.enforce_emulator_resolution)
        top_bar.addWidget(btn_fix_res)

        self.btn_toggle_stream = QPushButton("🎥 开启实时画面流 (Live)")
        self.btn_toggle_stream.setCheckable(True)
        self.btn_toggle_stream.setObjectName("btnStartBot")
        self.btn_toggle_stream.clicked.connect(self.toggle_live_stream)
        top_bar.addWidget(self.btn_toggle_stream)

        top_bar.addStretch()
        self.lbl_mouse_probe = QLabel("🎯 鼠标探针: (X=---, Y=---) RGB=(---)")
        self.lbl_mouse_probe.setStyleSheet("color: #FBBF24; font-family: Consolas; font-weight: bold;")
        top_bar.addWidget(self.lbl_mouse_probe)

        self.lbl_status_badge = QLabel("状态：就绪 (待命)")
        self.lbl_status_badge.setObjectName("statusBadge")
        top_bar.addWidget(self.lbl_status_badge)
        root_layout.addLayout(top_bar)

        # ================= 2. 主分割区 (左: 60% 实时画布, 右: 40% 标签页+日志) =================
        splitter = QSplitter(Qt.Horizontal)

        # ---- 左侧面板 ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 4, 0)

        # 图层叠加与鼠标直控开关栏
        layer_bar = QHBoxLayout()
        self.chk_show_ocr = QCheckBox("显示 OCR 资源/工人框")
        self.chk_show_ocr.setChecked(True)
        self.chk_show_ocr.toggled.connect(self._update_stream_flags)

        self.chk_show_tpl = QCheckBox("显示气泡/按钮匹配框")
        self.chk_show_tpl.setChecked(True)
        self.chk_show_tpl.toggled.connect(self._update_stream_flags)

        self.chk_show_redzone = QCheckBox("显示战斗红线四边下兵点")
        self.chk_show_redzone.setChecked(False)  # 默认在主村庄关闭红线点，画面清爽
        self.chk_show_redzone.toggled.connect(self._update_stream_flags)

        self.chk_direct_tap = QCheckBox("🖱️ 点击画面直控模拟器")
        self.chk_direct_tap.setStyleSheet("color: #34D399; font-weight: bold;")
        self.chk_direct_tap.toggled.connect(self._on_toggle_direct_tap)

        layer_bar.addWidget(self.chk_show_ocr)
        layer_bar.addWidget(self.chk_show_tpl)
        layer_bar.addWidget(self.chk_show_redzone)
        layer_bar.addWidget(self.chk_direct_tap)
        layer_bar.addStretch()
        left_layout.addLayout(layer_bar)

        # 16:9 交互式游戏画面画布
        self.canvas = InteractiveGameCanvas()
        self.canvas.set_layer_visibility(
            self.chk_show_ocr.isChecked(),
            self.chk_show_tpl.isChecked(),
            self.chk_show_redzone.isChecked(),
        )
        self.canvas.roiSelected.connect(self._on_canvas_roi_cropped)
        self.canvas.mouseMovedOnGame.connect(self._on_mouse_moved_on_canvas)
        self.canvas.canvasTapped.connect(self._on_canvas_direct_tapped)
        left_layout.addWidget(self.canvas, stretch=1)

        # 底部主控按钮栏
        ctrl_bar = QHBoxLayout()
        self.btn_refresh_frame = QPushButton("📸 单帧抓取 & 视觉诊断")
        self.btn_refresh_frame.clicked.connect(self.refresh_live_frame)
        ctrl_bar.addWidget(self.btn_refresh_frame)

        self.btn_save_cfg = QPushButton("💾 保存全部配置")
        self.btn_save_cfg.clicked.connect(self.save_gui_config)
        ctrl_bar.addWidget(self.btn_save_cfg)

        self.btn_start_bot = QPushButton("▶ 一键启动全自动挂机")
        self.btn_start_bot.setObjectName("btnStartBot")
        self.btn_start_bot.clicked.connect(lambda: self.start_bot_worker("full_auto"))
        ctrl_bar.addWidget(self.btn_start_bot)

        self.btn_stop_bot = QPushButton("⏹ 正常停止")
        self.btn_stop_bot.setEnabled(False)
        self.btn_stop_bot.clicked.connect(self.stop_bot_worker)
        ctrl_bar.addWidget(self.btn_stop_bot)

        self.btn_force_kill = QPushButton("🚨 强行停止 (Force Kill)")
        self.btn_force_kill.setObjectName("btnStopBot")
        self.btn_force_kill.setEnabled(False)
        self.btn_force_kill.clicked.connect(self.force_kill_bot_worker)
        ctrl_bar.addWidget(self.btn_force_kill)

        left_layout.addLayout(ctrl_bar)
        splitter.addWidget(left_widget)

        # ---- 右侧面板 ----
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_dashboard(), "📊 实时调试与看板")
        self.tabs.addTab(self._build_tab_loot_combat(), "⚔️ 搜鱼与战斗")
        self.tabs.addTab(self._build_tab_village(), "🏡 村庄与捐兵")
        self.tabs.addTab(self._build_tab_upgrade(), "🧱 刷墙与升级")
        self.tabs.addTab(self._build_tab_calibrator(), "🎯 模板采集校准")
        right_layout.addWidget(self.tabs, stretch=3)

        # 底部彩色日志控制台
        grp_log = QGroupBox("📜 实时战报与调试日志")
        log_layout = QVBoxLayout(grp_log)
        self.lbl_run_log = QLabel()
        self.lbl_run_log.setWordWrap(True)
        self.lbl_run_log.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_run_log.hide()
        log_layout.addWidget(self.lbl_run_log)
        self.txt_log = QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumBlockCount(1500)
        log_layout.addWidget(self.txt_log)
        right_layout.addWidget(grp_log, stretch=2)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        root_layout.addWidget(splitter, stretch=1)

    # ================= 五大功能标签页构建 =================
    def _build_tab_dashboard(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        grp_metrics = QGroupBox("实时视觉识别与场景状态 (采集与识别解耦架构)")
        grid = QGridLayout(grp_metrics)
        self.lbl_metric_scene = QLabel("当前识别场景: 待检测 | 预览: 0.0 FPS | 识别耗时: --- ms")
        self.lbl_metric_scene.setStyleSheet("color: #38BDF8; font-weight: bold;")
        self.lbl_metric_village = QLabel("本村工人/资源: 空闲工人=0/6 | 金=0 | 水=0 | 黑油=0")
        self.lbl_metric_loot = QLabel("敌方资源读数: 金=0 | 水=0 | 黑油=0")
        self.lbl_metric_dead = QLabel("已识别气泡/模板: 0 项")
        self.lbl_metric_totals = QLabel("累计运行轮数: 0 轮 | 已收菜气泡: 0 个 | 已自动刷墙: 0 块")
        self.lbl_deployment = QLabel()
        self.lbl_deployment.setWordWrap(True)
        self.lbl_deployment.hide()
        grid.addWidget(self.lbl_metric_scene, 0, 0)
        grid.addWidget(self.lbl_metric_village, 1, 0)
        grid.addWidget(self.lbl_metric_loot, 2, 0)
        grid.addWidget(self.lbl_metric_dead, 3, 0)
        grid.addWidget(self.lbl_metric_totals, 4, 0)
        grid.addWidget(self.lbl_deployment, 5, 0)
        lay.addWidget(grp_metrics)

        grp_steps = QGroupBox("🛠️ 独立单步功能调试 (详细步骤实时打印到下方日志，测试期间可随时点[🚨强行停止])")
        step_grid = QGridLayout(grp_steps)

        zoom_row = QHBoxLayout()
        btn_s1 = QPushButton("1️⃣ 测试视角缩小归一化 (Zoom-Out)")
        btn_s1.clicked.connect(lambda: self.start_bot_worker("step_zoom"))
        self.combo_zoom_align = QComboBox()
        self.combo_zoom_align.addItem("四角全景居中 (center - 推荐)", "center")
        self.combo_zoom_align.addItem("仅双指缩小不平移 (zoom_only)", "zoom_only")
        self.combo_zoom_align.addItem("偏上方对齐 (top)", "top")
        self.combo_zoom_align.addItem("偏下方对齐 (bottom)", "bottom")
        zoom_row.addWidget(btn_s1, stretch=2)
        zoom_row.addWidget(self.combo_zoom_align, stretch=1)

        btn_s2 = QPushButton("2️⃣ 测试村庄自动收菜 (金币/圣水/黑油各仅点 1 次，不做捐兵补兵)")
        btn_s2.clicked.connect(lambda: self.start_bot_worker("step_village"))
        btn_s3 = QPushButton("3️⃣ 测试右下角[工人建议列表]自动刷墙/升级")
        btn_s3.clicked.connect(lambda: self.start_bot_worker("step_upgrade"))

        search_row = QHBoxLayout()
        btn_s4 = QPushButton("4️⃣ 测试联机搜鱼 ([战斗]->[常规/排位]->[进攻])")
        btn_s4.clicked.connect(lambda: self.start_bot_worker("step_search"))
        self.combo_match_mode = QComboBox()
        self.combo_match_mode.addItem("常规赛 (按阈值挑鱼，可点下一个)", "regular")
        self.combo_match_mode.addItem("排位赛 (不能再搜，搜到谁直接打谁)", "ranked")
        self.combo_match_mode.currentIndexChanged.connect(self._on_quick_match_mode_changed)
        search_row.addWidget(btn_s4, stretch=2)
        search_row.addWidget(self.combo_match_mode, stretch=1)

        btn_s5 = QPushButton("5️⃣ 测试主世界自动战斗与动态红线智能下兵 (若在主世界则自动先进场搜鱼再下兵)")
        btn_s5.setObjectName("btnWarning")
        btn_s5.clicked.connect(lambda: self.start_bot_worker("step_combat"))

        night_row = QHBoxLayout()
        btn_s6 = QPushButton("6️⃣ ⛵ 主世界↔夜世界渡船切换")
        self.btn_night_boat = btn_s6
        btn_s6.clicked.connect(lambda: self.start_bot_worker("step_night_boat"))
        btn_s7 = QPushButton("7️⃣ 🌙 夜世界自动收菜 (金/水各点1次)")
        btn_s7.clicked.connect(lambda: self.start_bot_worker("step_night_collect"))
        btn_s8 = QPushButton("8️⃣ ⚔️ 夜世界自动战斗 (清边中推+技能+回营)")
        btn_s8.setObjectName("btnWarning")
        btn_s8.clicked.connect(lambda: self.start_bot_worker("step_night_battle"))
        night_row.addWidget(btn_s6, stretch=1)
        night_row.addWidget(btn_s7, stretch=1)
        night_row.addWidget(btn_s8, stretch=1)

        loop_row = QHBoxLayout()
        btn_s9 = QPushButton("9️⃣ 🔄 启动循环自动战斗 (持续循环直至点停止)")
        btn_s9.setObjectName("btnStartBot")
        btn_s9.clicked.connect(lambda: self.start_bot_worker(self.combo_combat_loop_mode.currentData()))
        self.combo_combat_loop_mode = QComboBox()
        self.combo_combat_loop_mode.addItem("🔄 主世界 ↔ 夜世界 双世界交替循环战斗 (推荐)", "loop_dual_combat")
        self.combo_combat_loop_mode.addItem("🌙 仅夜世界 连续循环自动战斗 (自动领胜利之星)", "loop_night_combat")
        self.combo_combat_loop_mode.addItem("⚔️ 仅主世界 连续循环自动战斗", "loop_home_combat")
        loop_row.addWidget(btn_s9, stretch=3)
        loop_row.addWidget(self.combo_combat_loop_mode, stretch=2)

        self.btn_force_kill_step = QPushButton("🚨 强行停止当前测试 / 循环战斗 (0.1秒瞬间急停)")
        self.btn_force_kill_step.setObjectName("btnStopBot")
        self.btn_force_kill_step.setEnabled(False)
        self.btn_force_kill_step.clicked.connect(self.force_kill_bot_worker)

        step_grid.addLayout(zoom_row, 0, 0)
        step_grid.addWidget(btn_s2, 1, 0)
        step_grid.addWidget(btn_s3, 2, 0)
        step_grid.addLayout(search_row, 3, 0)
        step_grid.addWidget(btn_s5, 4, 0)
        step_grid.addLayout(night_row, 5, 0)
        step_grid.addLayout(loop_row, 6, 0)
        step_grid.addWidget(self.btn_force_kill_step, 7, 0)
        lay.addWidget(grp_steps)
        lay.addStretch()
        return w

    def _build_tab_loot_combat(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        grp_loot = QGroupBox("🔍 联机模式选择 & 自动搜鱼资源过滤条件")
        form_loot = QFormLayout(grp_loot)
        self.chk_auto_attack_found = QCheckBox("常规赛搜到满足条件的鱼后立即自动下兵开打 (排位赛始终搜到谁直接打谁)")
        self.chk_auto_attack_found.setChecked(True)
        self.spin_min_gold = QSpinBox()
        self.spin_min_gold.setRange(0, 5_000_000)
        self.spin_min_gold.setSingleStep(50_000)
        self.spin_min_elixir = QSpinBox()
        self.spin_min_elixir.setRange(0, 5_000_000)
        self.spin_min_elixir.setSingleStep(50_000)
        self.spin_min_sum = QSpinBox()
        self.spin_min_sum.setRange(0, 10_000_000)
        self.spin_min_sum.setSingleStep(100_000)
        self.spin_min_dark = QSpinBox()
        self.spin_min_dark.setRange(0, 100_000)
        self.spin_min_dark.setSingleStep(500)
        self.chk_only_dead = QCheckBox("仅攻击死鱼 (外围满液位圣水收集器 >= 3 个)")

        form_loot.addRow("", self.chk_auto_attack_found)
        form_loot.addRow("单科最低金币 (Gold):", self.spin_min_gold)
        form_loot.addRow("单科最低圣水 (Elixir):", self.spin_min_elixir)
        form_loot.addRow("或 金+水总和达到:", self.spin_min_sum)
        form_loot.addRow("或 暗黑重油达到:", self.spin_min_dark)
        form_loot.addRow("", self.chk_only_dead)
        lay.addWidget(grp_loot)

        grp_combat = QGroupBox("⚔️ 动态红线智能下兵与英雄技能")
        form_c = QFormLayout(grp_combat)
        self.combo_strategy = QComboBox()
        self.combo_strategy.addItem("四边红线环绕平铺流 (surround_barch - 超哥/打死鱼推荐)", "surround_barch")
        self.combo_strategy.addItem("单侧红线一字划空军流 (line_air_spam - 雷龙/火龙气球推荐)", "line_air_spam")
        self.spin_waves = QSpinBox()
        self.spin_waves.setRange(1, 8)
        self.spin_hero_hp = QDoubleSpinBox()
        self.spin_hero_hp.setRange(0.10, 0.90)
        self.spin_hero_hp.setSingleStep(0.05)

        form_c.addRow("战斗下兵流派:", self.combo_strategy)
        form_c.addRow("红线环绕下兵波数:", self.spin_waves)
        form_c.addRow("英雄自动开大血量阈值:", self.spin_hero_hp)
        lay.addWidget(grp_combat)
        lay.addStretch()
        return w

    def _build_tab_village(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("🏡 村庄日常收菜与智能部落捐兵")
        form = QFormLayout(grp)

        self.chk_collect_res = QCheckBox("每次回村自动点击金币/圣水/黑油气泡收菜")
        self.chk_req_cc = QCheckBox("自动请求部落城堡援军与一键快速训练补兵")
        self.chk_auto_donate = QCheckBox("启用部落聊天栏智能关键词识别捐兵")
        self.edit_donate_white = QLineEdit()
        self.edit_donate_black = QLineEdit()

        form.addRow("", self.chk_collect_res)
        form.addRow("", self.chk_req_cc)
        form.addRow("", self.chk_auto_donate)
        form.addRow("捐兵触发词 (逗号分隔):", self.edit_donate_white)
        form.addRow("禁捐黑名单 (逗号分隔):", self.edit_donate_black)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _build_tab_upgrade(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        grp = QGroupBox("🧱 基于右下角[建筑工人建议列表]的零误差刷墙与升级")
        form = QFormLayout(grp)

        self.chk_auto_wall = QCheckBox("启用自动刷墙 (从工人建议列表自动选中[城墙])")
        self.chk_auto_bldg = QCheckBox("启用自动升级推荐建筑")
        self.spin_reserve_builders = QSpinBox()
        self.spin_reserve_builders.setRange(0, 5)
        self.spin_wall_gold = QSpinBox()
        self.spin_wall_gold.setRange(500_000, 30_000_000)
        self.spin_wall_gold.setSingleStep(500_000)
        self.spin_wall_elixir = QSpinBox()
        self.spin_wall_elixir.setRange(500_000, 30_000_000)
        self.spin_wall_elixir.setSingleStep(500_000)

        form.addRow("", self.chk_auto_wall)
        form.addRow("", self.chk_auto_bldg)
        form.addRow("预留专门刷墙工人数:", self.spin_reserve_builders)
        form.addRow("触发金币刷墙储量线:", self.spin_wall_gold)
        form.addRow("触发圣水刷墙储量线:", self.spin_wall_elixir)
        lay.addWidget(grp)
        lay.addStretch()
        return w

    def _build_tab_calibrator(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        grp = QGroupBox("🎯 12 项核心视觉模板库 & 无损 PNG 鼠标交互框选采集")
        vlay = QVBoxLayout(grp)
        vlay.addWidget(QLabel("提示：点击框选将自动暂停直播流并冻结抓取一张 1280x720 无损 PNG，直接拖拽即可精准裁切！"))

        self.list_templates = QListWidget()
        vlay.addWidget(self.list_templates)

        btn_row1 = QHBoxLayout()
        btn_crop_mouse = QPushButton("✂️ 在左侧画面鼠标拖拽框选选中模板")
        btn_crop_mouse.setObjectName("btnWarning")
        btn_crop_mouse.clicked.connect(self.trigger_canvas_crop_for_selected)
        btn_row1.addWidget(btn_crop_mouse)

        btn_harvest_village = QPushButton("⚡ 一键切取[主村庄]模板")
        btn_harvest_village.clicked.connect(lambda: self.harvest_scene_templates("village"))
        btn_row1.addWidget(btn_harvest_village)

        btn_harvest_battle = QPushButton("⚡ 一键切取[搜鱼战斗]模板")
        btn_harvest_battle.clicked.connect(lambda: self.harvest_scene_templates("battle"))
        btn_row1.addWidget(btn_harvest_battle)

        vlay.addLayout(btn_row1)
        lay.addWidget(grp)
        return w

    # ================= 实时直播流控制、33ms 主线程刷新与会话管理 =================
    def _reset_stream_metrics(self, clear_overlay: bool = True) -> None:
        self._last_rendered_seq = 0
        self._frame_display_times.clear()
        self._recent_display_gaps_ms.clear()
        self._last_frame_display_ts = None
        self._current_preview_fps = 0.0
        self._rendered_new_frame_count = 0
        self._last_analysis_ms = None
        self._last_scene = None
        self._last_overlay = None
        if clear_overlay:
            self.canvas.clear_overlay()

    def _update_status_badge(self) -> None:
        analysis_part = f"识别 {self._last_analysis_ms:.0f}ms" if self._last_analysis_ms is not None else "识别中..."
        if self._vision_worker and self._vision_worker.isRunning() and self._vision_worker.continuous:
            self.lbl_status_badge.setText(
                f"状态：🎥 {self._capture_mode_label} | {self._active_device_label} | "
                f"{self._current_preview_fps:.1f} FPS | {analysis_part}"
            )
        elif self._capture_mode_code == "png_single":
            self.lbl_status_badge.setText(
                f"状态：📸 {self._capture_mode_label} | {self._active_device_label} | {analysis_part}"
            )

    def _update_scene_metric_label(self) -> None:
        ov = self._last_overlay
        scene = ov.scene if ov is not None else "待检测"
        engine = ov.ocr_engine if ov is not None else "---"
        analysis_str = f"{self._last_analysis_ms:.1f}ms" if self._last_analysis_ms is not None else "--- ms"
        if self._vision_worker and self._vision_worker.isRunning() and self._vision_worker.continuous:
            self.lbl_metric_scene.setText(
                f"当前识别场景: {scene} | 引擎: {engine} | "
                f"预览: {self._current_preview_fps:.1f} FPS | 识别耗时: {analysis_str}"
            )
        else:
            self.lbl_metric_scene.setText(
                f"当前识别场景: {scene} | 引擎: {engine} | 采集: {self._capture_mode_label} | 识别耗时: {analysis_str}"
            )

    def _on_device_changed(self, new_serial: str):
        if self._updating_device_combo:
            return
        new_serial = new_serial.strip()
        if not new_serial or new_serial == self._active_device_label:
            return
        self._active_device_label = new_serial
        self._pending_crop_key = None
        self.canvas.cancel_interactive_crop()
        was_streaming = self.btn_toggle_stream.isChecked()
        self._stop_vision_worker(clear_overlay=True)
        logger.info(f"📱 已切换目标设备为 [{new_serial}]，已清除旧设备标注")
        if was_streaming:
            self._start_vision_worker(continuous=True)

    def toggle_live_stream(self, checked: bool):
        if checked:
            self._pending_crop_key = None
            self.canvas.cancel_interactive_crop()
            self.btn_toggle_stream.setText("⏹ 停止实时画面流 (直播中...)")
            self.btn_toggle_stream.setObjectName("btnStopBot")
            self.btn_toggle_stream.setStyle(self.btn_toggle_stream.style())
            self._start_vision_worker(continuous=True)
        else:
            self._stop_vision_worker(clear_overlay=True)
            self.btn_toggle_stream.setText("🎥 开启实时画面流 (Live)")
            self.btn_toggle_stream.setObjectName("btnStartBot")
            self.btn_toggle_stream.setStyle(self.btn_toggle_stream.style())
            self._capture_mode_code = "idle"
            self._capture_mode_label = "已停止"
            self.lbl_status_badge.setText("状态：实时画面流已停止")
            self._update_scene_metric_label()

    def _update_stream_flags(self):
        show_ocr = self.chk_show_ocr.isChecked()
        show_tpl = self.chk_show_tpl.isChecked()
        show_red = self.chk_show_redzone.isChecked()

        # 立即生效：关闭图层时 0ms 清除画布上对应标注
        self.canvas.set_layer_visibility(show_ocr, show_tpl, show_red)

        if self._vision_worker and self._vision_worker.isRunning():
            self._vision_worker.set_layer_flags(show_ocr, show_tpl, show_red)
        else:
            # 静态单帧时，如果开启了之前未计算红线/模板的数据则抓一帧补齐，否则纯隐藏/恢复无需重抓
            need_refresh = (
                self._last_overlay is None
                or (show_red and not self._last_overlay.red_zone_sides)
                or (show_tpl and not self._last_overlay.template_boxes)
            )
            if need_refresh and self.canvas._raw_bgr is not None and not self.canvas._cropping_key:
                self.refresh_live_frame()

    def _on_toggle_direct_tap(self, checked: bool):
        self.canvas.direct_tap_enabled = checked
        if checked:
            logger.info("🖱️ 已开启[点击画面直控模拟器]：直接在左侧游戏画面点击，模拟器将异步同步点击该坐标！")

    def _on_mouse_moved_on_canvas(self, gx: int, gy: int, bgr: tuple):
        b, g, r = bgr
        self.lbl_mouse_probe.setText(f"🎯 鼠标探针: (X={gx:4d}, Y={gy:3d}) RGB=({r:3d},{g:3d},{b:3d})")

    def _on_canvas_direct_tapped(self, gx: int, gy: int):
        serial = self.combo_device.currentText().strip() or "auto"

        def _do_tap():
            try:
                dev = self.mock_device or AdbDeviceController(serial=serial)
                if dev.connect():
                    tx, ty = dev.tap(gx, gy, humanize=False)
                    logger.info(f"🖱️ [画布直控点击] 已向模拟器 ({serial}) 发送点击: X={tx}, Y={ty}")
            except Exception as e:
                logger.warning(f"🖱️ [画布直控点击] 发送失败: {e}")

        threading.Thread(target=_do_tap, daemon=True, name="coc-direct-tap").start()

    def _start_vision_worker(
        self,
        continuous: bool = False,
        force_h264_fail: Optional[bool] = None,
        segment_limit_sec: int = 180,
        skip_analysis: bool = False,
    ):
        self._stop_vision_worker(clear_overlay=True)
        self._vision_session_id += 1
        session_id = self._vision_session_id

        serial = self.combo_device.currentText().strip() or "auto"
        self._active_device_label = serial
        if force_h264_fail is None:
            force_h264_fail = os.environ.get("COC_FORCE_H264_FAIL", "").strip().lower() in ("1", "true", "yes")

        self._vision_worker = LiveVisionWorker(
            serial=serial,
            show_ocr_rois=self.chk_show_ocr.isChecked(),
            show_red_zone=self.chk_show_redzone.isChecked(),
            show_templates=self.chk_show_tpl.isChecked(),
            continuous=continuous,
            interval_ms=160,
            mock_device=self.mock_device,
            session_id=session_id,
            prefer_h264=continuous,
            segment_limit_sec=segment_limit_sec,
            analysis_interval_sec=1.0,
            force_h264_fail=force_h264_fail,
            skip_analysis=skip_analysis,
        )
        self._vision_worker.singleFrameReady.connect(self._on_vision_single_frame_ready)
        self._vision_worker.analysisReady.connect(self._on_vision_analysis_ready)
        self._vision_worker.captureModeChanged.connect(self._on_capture_mode_changed)
        self._vision_worker.errorOccurred.connect(self._on_vision_error)
        self._vision_worker.start()

        if continuous:
            self._capture_mode_code = "starting"
            self._capture_mode_label = "正在启动 H.264..."
            self.lbl_status_badge.setText(f"状态：🎥 正在连接 {serial} 启动实时画面流...")
            self._render_timer.start()
        else:
            self._render_timer.stop()

    def _stop_vision_worker(self, clear_overlay: bool = True):
        self._render_timer.stop()
        # 递增会话 ID，彻底丢弃任何尚未出队的旧 Qt 信号回调
        self._vision_session_id += 1
        worker = self._vision_worker
        self._vision_worker = None
        if worker is not None:
            try:
                worker.stop_stream()
                if worker.isRunning():
                    worker.wait(2500)
            except Exception:
                pass
        self._reset_stream_metrics(clear_overlay=clear_overlay)

    def _on_render_timer_tick(self):
        """Qt 主线程定时检查最新帧，仅在帧序号变化且达到刷新节拍时更新画布并统计真实 FPS"""
        worker = self._vision_worker
        if worker is None or not worker.continuous or worker.session_id != self._vision_session_id:
            return

        now = time.perf_counter()
        if (
            self._last_frame_display_ts is not None
            and (now - self._last_frame_display_ts) < self.MIN_RENDER_INTERVAL_SEC
        ):
            return

        frame = worker.get_latest_frame()
        if frame is None or frame.session_id != self._vision_session_id:
            return
        if frame.seq == self._last_rendered_seq:
            return

        bgr = frame.get_bgr()
        if bgr is None:
            return
        rendered = self.canvas.set_video_frame(bgr, seq=frame.seq, copy_raw=False)
        if not rendered:
            return

        now = time.perf_counter()
        self._last_rendered_seq = frame.seq
        self._rendered_new_frame_count += 1

        if self._last_frame_display_ts is not None:
            self._recent_display_gaps_ms.append((now - self._last_frame_display_ts) * 1000.0)
        self._last_frame_display_ts = now

        self._frame_display_times.append(now)
        # 保留最近 2.0 秒的新帧显示时间戳计算实时 FPS
        cutoff = now - 2.0
        while len(self._frame_display_times) > 2 and self._frame_display_times[0] < cutoff:
            self._frame_display_times.popleft()
        if len(self._frame_display_times) >= 2:
            span = self._frame_display_times[-1] - self._frame_display_times[0]
            if span > 0.05:
                self._current_preview_fps = round((len(self._frame_display_times) - 1) / span, 1)

        if (now - self._last_status_refresh_ts) >= 0.25:
            self._last_status_refresh_ts = now
            self._update_status_badge()
            self._update_scene_metric_label()

    def refresh_live_frame(self):
        if self._vision_worker and self._vision_worker.isRunning() and self._vision_worker.continuous:
            return
        self.lbl_status_badge.setText("状态：📸 正在抓取 1280x720 无损 PNG...")
        self._start_vision_worker(continuous=False)

    def _on_capture_mode_changed(self, session_id: int, mode_code: str, mode_label: str):
        if session_id != self._vision_session_id:
            return
        self._capture_mode_code = mode_code
        self._capture_mode_label = mode_label
        if mode_code == "png_fallback":
            logger.warning(f"⚠️ 实时预览采集已切换为: {mode_label}")
        elif mode_code == "h264":
            logger.info(f"🎥 实时预览采集已启动: {mode_label} ({self._active_device_label})")
        self._update_status_badge()
        self._update_scene_metric_label()

    def _on_vision_single_frame_ready(
        self,
        session_id: int,
        raw_bgr: np.ndarray,
        overlay: VisionOverlayData,
        metrics: dict,
    ):
        if session_id != self._vision_session_id:
            return
        self._active_device_label = metrics.get("device", self._active_device_label)
        self._capture_mode_code = "png_single"
        self._capture_mode_label = "单帧无损 PNG"
        self.canvas.set_video_frame(raw_bgr, seq=0, copy_raw=True)

        # 若正在等待冻结无损 PNG 以进入模板框选模式，则隐藏识别框并直接启动框选
        if self._pending_crop_key:
            crop_key = self._pending_crop_key
            self._pending_crop_key = None
            self.canvas.clear_overlay()
            self.canvas.start_interactive_crop(crop_key)
            self.lbl_status_badge.setText(f"状态：✂️ 已冻结无损 PNG，请框选 [{crop_key}]")
            logger.info(f"✂️ 已抓取并冻结 1280x720 无损 PNG：请在左侧画面按住鼠标左键拖拽框选 [{crop_key}]")
            return

        self._apply_overlay_to_ui(overlay)

    def _on_vision_analysis_ready(self, session_id: int, overlay: VisionOverlayData):
        if session_id != self._vision_session_id:
            return
        # 若刚切到无损框选模式，不覆盖冻结画布的清净视图
        if self._pending_crop_key or self.canvas._cropping_key:
            return
        self._apply_overlay_to_ui(overlay)

    def _apply_overlay_to_ui(self, overlay: VisionOverlayData) -> None:
        # 场景切换时先主动清除旧场景标注
        if self._last_scene is not None and overlay.scene != self._last_scene:
            self.canvas.clear_overlay()

        self._last_scene = overlay.scene
        self._last_analysis_ms = overlay.analysis_ms
        self._last_overlay = overlay
        self._active_device_label = overlay.device

        self.canvas.set_overlay_data(overlay)

        loot = overlay.enemy_loot
        vs = overlay.village_status
        self.lbl_metric_village.setText(
            f"本村工人/资源: 空闲工人={vs.free_builders}/{vs.total_builders} | "
            f"金={vs.gold:,} | 水={vs.elixir:,} | 黑油={vs.dark_elixir:,}"
        )
        self.lbl_metric_loot.setText(
            f"敌方资源读数: 金={loot.gold:,} | 水={loot.elixir:,} | 黑油={loot.dark_elixir:,}"
        )
        tpl_str = ", ".join(overlay.matched_templates[:6]) if overlay.matched_templates else "无"
        self.lbl_metric_dead.setText(
            f"已识别气泡/模板 ({len(overlay.matched_templates)}类): {tpl_str}"
        )
        self._update_scene_metric_label()
        self._update_status_badge()

    def _on_vision_error(self, session_id: int, err_msg: str):
        if session_id != self._vision_session_id:
            return
        self._pending_crop_key = None
        logger.warning(err_msg)
        self.lbl_status_badge.setText("状态：连接模拟器失败")
        if self.btn_toggle_stream.isChecked():
            self.btn_toggle_stream.setChecked(False)
            self.toggle_live_stream(False)

    # ================= 配置双向绑定与日志桥接 =================
    def _attach_log_handler(self):
        self._qt_log_handler = QtSignalLogHandler()
        self._qt_log_handler.setLevel(logging.INFO)
        self._qt_log_handler.bridge.logEmitted.connect(self._append_log_line)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(self._qt_log_handler)

    def _append_log_line(self, level: str, msg: str):
        self.txt_log.appendPlainText(msg)

    def _on_quick_match_mode_changed(self):
        self.cfg.loot_filter.match_mode = self.combo_match_mode.currentData() or "regular"

    def _load_config_into_widgets(self):
        c = self.cfg
        if hasattr(self, "combo_match_mode"):
            m_idx = self.combo_match_mode.findData(getattr(c.loot_filter, "match_mode", "regular"))
            if m_idx >= 0:
                self.combo_match_mode.setCurrentIndex(m_idx)
        if hasattr(self, "chk_auto_attack_found"):
            self.chk_auto_attack_found.setChecked(getattr(c.loot_filter, "auto_attack_on_found", True))
        self.spin_min_gold.setValue(c.loot_filter.min_gold)
        self.spin_min_elixir.setValue(c.loot_filter.min_elixir)
        self.spin_min_sum.setValue(c.loot_filter.min_gold_plus_elixir)
        self.spin_min_dark.setValue(c.loot_filter.min_dark_elixir)
        self.chk_only_dead.setChecked(c.loot_filter.only_dead_base)

        idx = self.combo_strategy.findData(c.combat.strategy)
        if idx >= 0:
            self.combo_strategy.setCurrentIndex(idx)
        self.spin_waves.setValue(c.combat.deploy_wave_count)
        self.spin_hero_hp.setValue(c.combat.hero_ability_hp_ratio)

        self.chk_collect_res.setChecked(c.village.collect_resources)
        self.chk_req_cc.setChecked(c.village.auto_request_cc)
        self.chk_auto_donate.setChecked(c.village.auto_donate)
        self.edit_donate_white.setText(",".join(c.village.donate_keywords))
        self.edit_donate_black.setText(",".join(c.village.donate_blacklist))

        self.chk_auto_wall.setChecked(c.upgrade.enable_auto_wall)
        self.chk_auto_bldg.setChecked(c.upgrade.enable_auto_building)
        self.spin_reserve_builders.setValue(c.upgrade.reserve_builders_for_wall)
        self.spin_wall_gold.setValue(c.upgrade.wall_min_gold)
        self.spin_wall_elixir.setValue(c.upgrade.wall_min_elixir)

    def _sync_widgets_to_config(self) -> AppConfig:
        c = self.cfg
        c.device.serial = self.combo_device.currentText().strip() or "auto"
        if hasattr(self, "combo_match_mode"):
            c.loot_filter.match_mode = self.combo_match_mode.currentData() or "regular"
        if hasattr(self, "chk_auto_attack_found"):
            c.loot_filter.auto_attack_on_found = self.chk_auto_attack_found.isChecked()
        c.loot_filter.min_gold = self.spin_min_gold.value()
        c.loot_filter.min_elixir = self.spin_min_elixir.value()
        c.loot_filter.min_gold_plus_elixir = self.spin_min_sum.value()
        c.loot_filter.min_dark_elixir = self.spin_min_dark.value()
        c.loot_filter.only_dead_base = self.chk_only_dead.isChecked()

        c.combat.strategy = self.combo_strategy.currentData() or "surround_barch"
        c.combat.deploy_wave_count = self.spin_waves.value()
        c.combat.hero_ability_hp_ratio = round(self.spin_hero_hp.value(), 2)

        c.village.collect_resources = self.chk_collect_res.isChecked()
        c.village.auto_request_cc = self.chk_req_cc.isChecked()
        c.village.auto_donate = self.chk_auto_donate.isChecked()
        c.village.donate_keywords = [s.strip() for s in self.edit_donate_white.text().split(",") if s.strip()]
        c.village.donate_blacklist = [s.strip() for s in self.edit_donate_black.text().split(",") if s.strip()]

        c.upgrade.enable_auto_wall = self.chk_auto_wall.isChecked()
        c.upgrade.enable_auto_building = self.chk_auto_bldg.isChecked()
        c.upgrade.reserve_builders_for_wall = self.spin_reserve_builders.value()
        c.upgrade.wall_min_gold = self.spin_wall_gold.value()
        c.upgrade.wall_min_elixir = self.spin_wall_elixir.value()
        return c

    def save_gui_config(self):
        self._sync_widgets_to_config()
        saved_path = save_user_config(self.cfg)
        logger.info(f"已保存当前配置至: {saved_path}")

    def scan_emulators(self, silent: bool = False):
        emulators = PcEmulatorHost.discover_emulators()
        current_text = self.combo_device.currentText()
        self._updating_device_combo = True
        try:
            self.combo_device.clear()
            for emu in emulators:
                self.combo_device.addItem(emu.serial)
            self.combo_device.addItem("auto")
            if emulators:
                preferred = next((e.serial for e in emulators if "5561" in e.serial), emulators[0].serial)
                self.combo_device.setCurrentText(preferred)
                self._active_device_label = preferred
                logger.info(f"扫描到 {len(emulators)} 个在线安卓模拟器，当前选中: {preferred}")
            else:
                fallback_serial = current_text or "127.0.0.1:5561"
                self.combo_device.setCurrentText(fallback_serial)
                self._active_device_label = fallback_serial
                if not silent:
                    logger.warning("未扫描到本地模拟器端口，请确认模拟器已开启。")
        finally:
            self._updating_device_combo = False

    def enforce_emulator_resolution(self):
        serial = self.combo_device.currentText().strip()
        if serial == "auto":
            emulators = PcEmulatorHost.discover_emulators()
            serial = emulators[0].serial if emulators else "127.0.0.1:7555"
        PcEmulatorHost.enforce_standard_display(serial, 1280, 720, 240)
        if not (self._vision_worker and self._vision_worker.isRunning() and self._vision_worker.continuous):
            self.refresh_live_frame()

    def _refresh_template_list(self):
        self.list_templates.clear()
        for key, spec in self.harvester.TEMPLATE_SPECS.items():
            full_path = self.harvester.assets_dir / spec.rel_path
            status = "✅ 已就绪" if full_path.exists() else "❌ 缺失"
            item = QListWidgetItem(f"[{status}] {key:<18} | 场景:{spec.scene:<11} | {spec.description}")
            item.setData(Qt.UserRole, key)
            self.list_templates.addItem(item)
        if self.list_templates.count() > 0:
            self.list_templates.setCurrentRow(0)

    def trigger_canvas_crop_for_selected(self):
        """
        进入无损模板框选模式：
        1. 暂停实时视频流，避免 H.264 有损帧和画面跳动影响框选；
        2. 显式抓取并冻结一张 1280x720 无损 PNG；
        3. 待无损 PNG 渲染到画布后自动开启鼠标框选，确保探针、显示与裁切严格使用同一张无损 PNG。
        """
        item = self.list_templates.currentItem()
        if not item:
            return
        key = item.data(Qt.UserRole)

        if self.btn_toggle_stream.isChecked():
            self.btn_toggle_stream.setChecked(False)
            self.toggle_live_stream(False)

        self._pending_crop_key = key
        self.canvas.clear_overlay()
        self.lbl_status_badge.setText(f"状态：📸 正在抓取无损 PNG 准备框选 [{key}]...")
        logger.info(f"✂️ 已暂停实时预览，正在抓取 1280x720 无损 PNG 用于框选 [{key}]...")
        self._start_vision_worker(continuous=False, skip_analysis=True)

    def _on_canvas_roi_cropped(self, template_key: str, roi: tuple, cropped_bgr: np.ndarray):
        if self.canvas._raw_bgr is None:
            return
        saved_path = self.harvester.crop_custom_roi(self.canvas._raw_bgr, template_key, roi)
        logger.info(f"🎯 成功通过无损 PNG 框选保存模板 [{template_key}] (ROI={roi}) -> {saved_path}")
        self._refresh_template_list()
        if self._vision_worker and self._vision_worker.isRunning():
            self._vision_worker.invalidate_template_cache()
        else:
            self.refresh_live_frame()

    def harvest_scene_templates(self, scene: str):
        if self.canvas._raw_bgr is None:
            logger.warning("请先获取模拟器画面后再执行一键切图。")
            return
        saved = self.harvester.harvest_from_screen_by_scene(self.canvas._raw_bgr, scene)
        logger.info(f"⚡ 已从当前画布一键按标准坐标切取 [{scene}] 场景共 {len(saved)} 张模板！")
        self._refresh_template_list()
        if self._vision_worker and self._vision_worker.isRunning():
            self._vision_worker.invalidate_template_cache()
        else:
            self.refresh_live_frame()

    def start_bot_worker(self, mode: str = "full_auto"):
        if self._bot_worker and self._bot_worker.isRunning():
            logger.warning("当前已有任务正在运行中，请先停止后再启动新任务。")
            return

        self.save_gui_config()
        self.btn_start_bot.setEnabled(False)
        self.btn_stop_bot.setEnabled(True)
        self.btn_force_kill.setEnabled(True)
        self.btn_force_kill_step.setEnabled(True)
        align_mode = self.combo_zoom_align.currentData() if hasattr(self, "combo_zoom_align") else "center"
        self._bot_worker = BotActionWorker(self.cfg, mode=mode, align_mode=align_mode, mock_device=self.mock_device)
        self._bot_worker.statusChanged.connect(lambda s: self.lbl_status_badge.setText(f"状态：{s}"))
        self._bot_worker.runLogOpened.connect(self._show_run_log)
        self._bot_worker.cycleCompleted.connect(self._on_bot_cycle_completed)
        self._bot_worker.cycleCompleted.connect(self.botCycleCompleted)
        self._bot_worker.finishedTask.connect(self._on_bot_worker_finished)
        self._bot_worker.start()

    def _show_run_log(self, path: str):
        self.lbl_run_log.setText(f"本次连续战斗日志 {path}")
        self.lbl_run_log.show()
        self.lbl_deployment.hide()
        logger.info("本次连续战斗日志 %s", path)
        self.botRunLogOpened.emit(path)

    def stop_bot_worker(self):
        if self._bot_worker and self._bot_worker.isRunning():
            logger.info("⏹ 正在向挂机线程发送正常停止信号 (将在当前动作完成后停下，若需瞬间急停请点[🚨强行停止])...")
            self._bot_worker.request_stop()
            self.lbl_status_badge.setText("状态：正在正常停止...")

    def force_kill_bot_worker(self):
        """🚨 0.1 秒瞬间强行停止当前正在运行的任何测试或挂机任务"""
        if self._bot_worker and self._bot_worker.isRunning():
            self._bot_worker.force_kill_now()
            logger.warning("🚨 [强行停止 Force Kill] 已立即强行终止当前测试线程、强杀 ADB 进程并释放所有触控点！")
        self._on_bot_worker_finished("🚨 已强行停止")

    def _on_bot_cycle_completed(self, summary: dict):
        self._total_cycles += 1
        if "village" in summary and isinstance(summary["village"], dict):
            self._total_bubbles += summary["village"].get("collected_bubbles", 0)
        if "upgrade" in summary and isinstance(summary["upgrade"], dict):
            self._total_walls += summary["upgrade"].get("walls_upgraded", 0)
        self.lbl_metric_totals.setText(
            f"累计运行轮数: {self._total_cycles} 轮 | 已收菜气泡: {self._total_bubbles} 个 | 已自动刷墙: {self._total_walls} 块"
        )
        report = summary.get("round_report")
        if report:
            deployment = [(report["headline"], report["headline"]), *report["lines"]]
        else:
            deployment = deployment_lines(summary)
        if deployment:
            self.lbl_deployment.setText("\n".join(short for short, _ in deployment))
            self.lbl_deployment.setToolTip("\n\n".join(detail for _, detail in deployment))
            self.lbl_deployment.show()
            for _, detail in deployment:
                logger.info("%s", detail)
        self.refresh_live_frame()

    def _on_bot_worker_finished(self, msg: str):
        self.btn_start_bot.setEnabled(True)
        self.btn_stop_bot.setEnabled(False)
        self.btn_force_kill.setEnabled(False)
        self.btn_force_kill_step.setEnabled(False)
        self.lbl_status_badge.setText(f"状态：{msg}")
        self.refresh_live_frame()

    def closeEvent(self, event):
        self._stop_vision_worker(clear_overlay=False)
        if self._bot_worker and self._bot_worker.isRunning():
            self._bot_worker.request_stop()
            self._bot_worker.wait(1000)
        super().closeEvent(event)
