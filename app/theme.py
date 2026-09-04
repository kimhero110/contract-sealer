"""现代扁平主题 v2：Fusion 基座 + QSS。

设计基调（对齐 2020s 桌面应用审美：Linear / Notion / 飞书）：
- 三区色彩分层：左栏米灰、画布中性灰、右栏纯白卡片；
- 大圆角（10px）、柔和边框、充分留白；
- 字层级：分组标题 13px 加粗，正文 13px，辅助说明 11px 灰；
- 主题色印章红（#C0392B），hover 用浅红底胶囊，不用突兀填充；
- 右侧面板卡片用 QGraphicsDropShadowEffect 柔和投影（代码侧添加）。
"""

ACCENT = "#C0392B"
ACCENT_HOVER = "#A93226"
ACCENT_PRESSED = "#922B21"
ACCENT_SOFT = "#FDF0EE"      # 浅红胶囊底
BG = "#F7F7F8"               # 窗口底
SIDEBAR_BG = "#F0F1F3"       # 左栏
PANEL_BG = "#FFFFFF"
BORDER = "#E4E5E8"
TEXT = "#1F2329"
TEXT_DIM = "#8A9199"
CANVAS_BG = "#E9EAED"

THEME_QSS = f"""
/* ── 全局 ── */
QWidget {{
    background: {BG};
    color: {TEXT};
    font-size: 13px;
}}

/* ── 菜单栏 / 工具栏：融为一条顶部区域 ── */
QMenuBar {{
    background: {PANEL_BG};
    border-bottom: 1px solid {BORDER};
    padding: 3px 6px;
    font-size: 13px;
}}
QMenuBar::item {{
    padding: 6px 12px;
    border-radius: 7px;
}}
QMenuBar::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
QMenu {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{ padding: 7px 26px; border-radius: 5px; }}
QMenu::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}

QToolBar {{
    background: {PANEL_BG};
    border-bottom: 1px solid {BORDER};
    padding: 6px 10px;
    spacing: 4px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 8px;
    padding: 6px 12px;
    color: {TEXT};
}}
QToolButton:hover {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}
QToolButton:pressed {{ background: #F5DEDA; }}

/* ── 画布 ── */
QGraphicsView {{
    background: {CANVAS_BG};
    border: none;
}}

/* ── 左侧页面栏 ── */
QListWidget {{
    background: {SIDEBAR_BG};
    border: none;
    padding: 10px 8px;
    outline: none;
    font-size: 12px;
    color: {TEXT_DIM};
}}
QListWidget::item {{
    border: 2px solid transparent;
    border-radius: 8px;
    padding: 8px 6px;
    margin: 3px 2px;
    background: {PANEL_BG};
}}
QListWidget::item:selected {{
    border: 2px solid {ACCENT};
    color: {TEXT};
}}
QListWidget::item:hover:!selected {{ background: #FAFAFB; border-color: #D8DADD; }}

/* ── 右侧卡片面板 ── */
QGroupBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin-top: 18px;
    padding: 14px 12px 12px 12px;
    font-weight: bold;
    font-size: 13px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 14px;
    top: 2px;
    padding: 2px 8px;
    background: {PANEL_BG};
    border-radius: 4px;
    color: {TEXT};
}}

/* ── 按钮：胶囊化 ── */
QPushButton {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ background: {ACCENT_SOFT}; border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: #F5DEDA; }}
QPushButton:disabled {{ color: #B9BEC4; border-color: {BORDER}; background: #F3F4F5; }}

QPushButton#primary {{
    background: {ACCENT};
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: bold;
    font-size: 14px;
    padding: 12px 18px;
}}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:pressed {{ background: {ACCENT_PRESSED}; }}

/* ── 输入控件 ── */
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 6px 9px;
    selection-background-color: {ACCENT};
}}
QDoubleSpinBox:hover, QSpinBox:hover, QComboBox:hover {{ border-color: #C9CCD1; }}
QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus, QComboBox:focus {{
    border: 2px solid {ACCENT};
    padding: 5px 8px;
}}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 4px;
    outline: none;
}}
QComboBox QAbstractItemView::item {{ padding: 6px 10px; border-radius: 5px; }}
QComboBox QAbstractItemView::item:selected {{ background: {ACCENT_SOFT}; color: {ACCENT}; }}

/* ── 滑杆 ── */
QSlider::groove:horizontal {{
    height: 6px;
    background: #E3E4E7;
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 18px; height: 18px;
    margin: -6px 0;
    border-radius: 9px;
    background: {ACCENT};
    border: 2px solid white;
}}
QSlider::handle:horizontal:hover {{ background: {ACCENT_HOVER}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}

/* ── 勾选框 ── */
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 17px; height: 17px;
    border: 2px solid #C9CCD1;
    border-radius: 5px;
    background: {PANEL_BG};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ── 滚动条：细、悬浮 ── */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 3px 3px 3px 0;
}}
QScrollBar::handle:vertical {{
    background: #CBCDD2;
    border-radius: 4px;
    min-height: 32px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_DIM}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0 3px 3px 3px;
}}
QScrollBar::handle:horizontal {{
    background: #CBCDD2;
    border-radius: 4px;
    min-width: 32px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── 分割条 ── */
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:horizontal {{ width: 1px; }}

/* ── 对话框 ── */
QDialog {{ background: {BG}; }}

/* ── 提示 ── */
QToolTip {{
    background: {TEXT};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 10px;
}}
"""
