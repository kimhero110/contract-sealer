"""浮在画布上的确认条：给纯鼠标操作补一条与 Enter / Esc 等价的出口。

四点校准原先只认回车确认、Esc 取消。两个问题叠在一起：快捷键只写在右侧
信息栏里，画布上没有任何可点的东西；而且回车要送到 `keyPressEvent`，前提是
画布持有键盘焦点——用户中途点过右侧面板、工具栏或分割条，焦点就走了，
回车按下去毫无反应，看上去像程序卡死。

确认条把这两个出口摆在画布底部。按钮设 `NoFocus`：点它不会把键盘焦点从
画布上抢走，点完仍然能继续用方向键微调把手。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QStyle,
    QStyleOption,
    QWidget,
)

from app.widgets import FitButton

HINT_MIN_WIDTH_PX = 240   # 钉住宽度，提示文案实时变化时确认条不会左右抖
BOTTOM_MARGIN_PX = 18


class ConfirmBar(QWidget):
    """画布底部居中的「确认 / 取消」浮条。"""

    accepted = Signal()
    cancelled = Signal()

    def __init__(self, parent: QWidget, hint: str, accept_text: str, cancel_text: str):
        super().__init__(parent)
        self.setObjectName("overlayBar")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(10)

        self.hint_label = QLabel(hint)
        self.hint_label.setObjectName("overlayHint")
        self.hint_label.setMinimumWidth(HINT_MIN_WIDTH_PX)
        lay.addWidget(self.hint_label)

        self.cancel_btn = FitButton(cancel_text)
        self.accept_btn = FitButton(accept_text)
        self.accept_btn.setObjectName("primary")
        for btn, sig in ((self.cancel_btn, self.cancelled), (self.accept_btn, self.accepted)):
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # 不抢画布焦点，点完还能方向键微调
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(sig)
            lay.addWidget(btn)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)
        self.hide()

    # ── 对外 ──

    def set_hint(self, text: str) -> None:
        """只改文字，不重新量尺寸——拖动中每帧都在调，量一次抖一次。"""
        self.hint_label.setText(text)

    def show_at_bottom(self) -> None:
        self.adjustSize()
        self.show()
        self.reposition()

    def reposition(self) -> None:
        """贴住父控件底边居中。父控件比确认条还窄时左对齐，宁可右边被裁也不让左边跑出去。"""
        parent = self.parentWidget()
        if parent is None:
            return
        x = max(0, (parent.width() - self.width()) // 2)
        y = max(0, parent.height() - self.height() - BOTTOM_MARGIN_PX)
        self.move(x, y)
        self.raise_()

    # ── 绘制 ──

    def paintEvent(self, event) -> None:
        """QWidget 子类不画 QSS 的背景/边框，得自己转发一次 PE_Widget。

        少了这一步，浮条就是一块透明的按钮堆，直接飘在合同页面上。
        """
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PrimitiveElement.PE_Widget, opt, painter, self)
