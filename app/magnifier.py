"""跟随鼠标的局部放大镜：四点校准 / 跟随落章等精确点选场景用。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from app.canvas import np_rgb_to_qpixmap

MAG_SIZE = 180          # 放大镜边长（px）
HALF_WINDOW_MM = 8.0    # 取景半径（mm）→ 180px / 16mm ≈ 11px/mm ≈ 6 倍放大


class Magnifier(QWidget):
    """浮在画布上、跟随光标的放大取景器。

    源图像由外部提供（回调），十字准线永远在正中（光标位置）。
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFixedSize(MAG_SIZE, MAG_SIZE)
        self._pixmap: QPixmap | None = None
        self.hide()

    def set_crop(self, pm: QPixmap) -> None:
        self._pixmap = pm
        self.update()

    def follow_cursor(self, global_pos: QPoint) -> None:
        """贴着光标放置，画布边缘自动翻转避让。"""
        vp = self.parentWidget()
        if vp is None:
            return
        pos = vp.mapFromGlobal(global_pos)
        x = pos.x() + 26
        y = pos.y() + 26
        if x + self.width() > vp.width():
            x = pos.x() - self.width() - 26
        if y + self.height() > vp.height():
            y = pos.y() - self.height() - 26
        self.move(max(0, x), max(0, y))
        self.raise_()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        # 背景 + 取景画面
        p.fillRect(self.rect(), QColor(245, 246, 247))
        if self._pixmap is not None:
            p.drawPixmap(self.rect(), self._pixmap)
        cx, cy = self.width() / 2, self.height() / 2
        # 三分线（辅助对齐纸边）
        grid = QPen(QColor(255, 255, 255, 110), 1)
        p.setPen(grid)
        for f in (1 / 3, 2 / 3):
            p.drawLine(int(self.width() * f), 0, int(self.width() * f), self.height())
            p.drawLine(0, int(self.height() * f), self.width(), int(self.height() * f))
        # 中心十字准线（光标位置）
        pen = QPen(QColor(192, 57, 43), 2)  # 印章红
        p.setPen(pen)
        arm = 18
        p.drawLine(int(cx - arm), int(cy), int(cx + arm), int(cy))
        p.drawLine(int(cx), int(cy - arm), int(cx), int(cy + arm))
        p.drawPoint(int(cx), int(cy))
        # 边框
        p.setPen(QPen(QColor(31, 35, 41, 200), 3))
        p.drawRect(1, 1, self.width() - 3, self.height() - 3)
