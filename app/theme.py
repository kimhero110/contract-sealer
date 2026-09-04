"""现代扁平主题：Fusion 基座 + QSS。

设计基调：浅色、干净、留白充分；主题色取印章红（#C0392B），
与产品内容（红色印泥）呼应；画布背景用中性灰让白纸页面突出。
"""

# 主题色
ACCENT = "#C0392B"        # 印章红
ACCENT_HOVER = "#A93226"
ACCENT_PRESSED = "#922B21"
BG = "#F5F6F7"            # 窗口底色
PANEL_BG = "#FFFFFF"      # 面板/列表底
BORDER = "#DCDDE0"
TEXT = "#2C3E50"
TEXT_DIM = "#7F8C8D"
CANVAS_BG = "#E5E6E8"     # 画布底色（衬托白纸）

THEME_QSS = f"""
/* ── 全局 ── */
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 13px;
}}

/* ── 菜单栏 / 工具栏 ── */
QMenuBar {{
    background: {PANEL_BG};
    border-bottom: 1px solid {BORDER};
    padding: 2px;
}}
QMenuBar::item {{
    padding: 5px 12px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{ background: #F0E6E4; color: {ACCENT}; }}
QMenu {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item {{ padding: 6px 24px; border-radius: 4px; }}
QMenu::item:selected {{ background: #F0E6E4; color: {ACCENT}; }}

QToolBar {{
    background: {PANEL_BG};
    border-bottom: 1px solid {BORDER};
    padding: 4px 8px;
    spacing: 6px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px 10px;
}}
QToolButton:hover {{ background: #F0E6E4; color: {ACCENT}; }}
QToolButton:pressed {{ background: #E8D5D1; }}

/* ── 画布 ── */
QGraphicsView {{
    background: {CANVAS_BG};
    border: none;
}}

/* ── 页面列表（缩略图）── */
QListWidget {{
    background: {PANEL_BG};
    border: none;
    border-right: 1px solid {BORDER};
    padding: 6px;
    outline: none;
}}
QListWidget::item {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px;
    margin: 4px 2px;
    background: #FAFBFC;
}}
QListWidget::item:selected {{
    border: 2px solid {ACCENT};
    background: #FDF5F4;
}}
QListWidget::item:hover:!selected {{ border-color: {ACCENT}; }}

/* ── 右侧面板 ── */
QGroupBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT};
}}

/* ── 按钮 ── */
QPushButton {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: #F0E6E4; }}
QPushButton:disabled {{ color: #B0B6BB; border-color: #EBEDEF; }}

/* 主操作按钮（导出）——印章红填充 */
QPushButton#primary {{
    background: {ACCENT};
    border: none;
    color: white;
    font-weight: bold;
    padding: 10px 16px;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: {ACCENT_PRESSED}; }}

/* ── 输入控件 ── */
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 4px 8px;
    selection-background-color: {ACCENT};
}}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}

/* ── 滑杆 ── */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 16px; height: 16px;
    margin: -7px 0;
    border-radius: 8px;
    background: {ACCENT};
    border: 2px solid white;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HOVER}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}

/* ── 勾选框 ── */
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {PANEL_BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ── 滚动条 ── */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #C9CCD1;
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_DIM}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #C9CCD1;
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── 分割条 ── */
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}

/* ── 提示 ── */
QToolTip {{
    background: {TEXT};
    color: white;
    border: none;
    border-radius: 4px;
    padding: 5px 8px;
}}
"""
