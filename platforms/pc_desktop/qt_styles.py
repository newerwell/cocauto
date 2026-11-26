"""
PyQt5 现代化游戏辅助深色主题样式表 (Dark Gaming Theme QSS)
采用深灰蓝电竞底色 (#121622)、卡片面板 (#1B2030)、琥珀金主色 (#F59E0B) 与翡翠绿启动按钮 (#10B981)。
"""

DARK_GAMING_QSS = """
QMainWindow, QWidget {
    background-color: #121622;
    color: #E2E8F0;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 13px;
}

QGroupBox {
    background-color: #1A1F2E;
    border: 1px solid #2D3748;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: bold;
    color: #FBBF24;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #FBBF24;
}

QTabWidget::pane {
    border: 1px solid #2D3748;
    background-color: #171C28;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #1E2536;
    color: #94A3B8;
    padding: 8px 14px;
    margin-right: 4px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #2563EB;
    color: #FFFFFF;
}

QTabBar::tab:hover:!selected {
    background-color: #2D3748;
    color: #E2E8F0;
}

QPushButton {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #3B82F6;
}

QPushButton:pressed {
    background-color: #1D4ED8;
}

QPushButton:disabled {
    background-color: #334155;
    color: #64748B;
}

QPushButton#btnStartBot {
    background-color: #10B981;
    color: #06281E;
    font-size: 14px;
    padding: 9px 18px;
}

QPushButton#btnStartBot:hover {
    background-color: #34D399;
}

QPushButton#btnStopBot {
    background-color: #EF4444;
    color: #FFFFFF;
    font-size: 14px;
    padding: 9px 18px;
}

QPushButton#btnStopBot:hover {
    background-color: #F87171;
}

QPushButton#btnWarning {
    background-color: #D97706;
    color: #FFFFFF;
}

QPushButton#btnWarning:hover {
    background-color: #F59E0B;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0F131D;
    border: 1px solid #334155;
    border-radius: 5px;
    padding: 5px 8px;
    color: #F8FAFC;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #3B82F6;
}

QPlainTextEdit, QListWidget {
    background-color: #0B0E17;
    border: 1px solid #2D3748;
    border-radius: 6px;
    color: #A7F3D0;
    font-family: "Consolas", "Microsoft YaHei UI", monospace;
    font-size: 12px;
}

QCheckBox {
    spacing: 6px;
    color: #E2E8F0;
}

QLabel#statusBadge {
    background-color: #1E293B;
    color: #38BDF8;
    border: 1px solid #0284C7;
    border-radius: 5px;
    padding: 4px 10px;
    font-weight: bold;
}
"""
