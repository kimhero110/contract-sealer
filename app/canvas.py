"""页面画布：渲染页面 + 印章拖拽/缩放/旋转交互。

场景坐标系直接使用物理 mm（原点在页面左上角）：
- 页面图像缩放铺满 (0, 0, phys_w_mm, phys_h_mm)；
- 印章位置/尺寸即物理值，导出时零换算误差（方案 §4.2 关键设计）。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QKeyEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


def np_rgb_to_qpixmap(img: np.ndarray) -> QPixmap:
    h, w = img.shape[:2]
    data = np.ascontiguousarray(img)
    qimg = QImage(data.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def np_rgba_to_qpixmap(img: np.ndarray) -> QPixmap:
    h, w = img.shape[:2]
    data = np.ascontiguousarray(img)
    qimg = QImage(data.data, w, h, w * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class StampItem(QGraphicsPixmapItem):
    """一枚已落章的印章/签名。pos() 为印章中心（mm），rotation 为手动旋转角。"""

    def __init__(self, rgba: np.ndarray, size_mm: float, center_x_mm: float, center_y_mm: float):
        super().__init__()
        self.size_mm = size_mm
        self.setPixmap(np_rgba_to_qpixmap(rgba))
        self._update_scale()
        self.set_center(center_x_mm, center_y_mm)
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable | QGraphicsPixmapItem.ItemIsSelectable
        )
        self.setTransformOriginPoint(self.boundingRect().center())

    def _update_scale(self) -> None:
        pm = self.pixmap()
        if pm.width() > 0:
            self.setScale(self.size_mm / pm.width())

    def set_pixmap_rgba(self, rgba: np.ndarray) -> None:
        """随机效果更新后刷新图像，保持中心与尺寸不变。"""
        center = self.center()
        self.setPixmap(np_rgba_to_qpixmap(rgba))
        self._update_scale()
        self.set_center(*center)

    def center(self) -> tuple[float, float]:
        r = self.boundingRect()
        return (self.pos().x() + r.width() * self.scale() / 2,
                self.pos().y() + r.height() * self.scale() / 2)

    def set_center(self, x_mm: float, y_mm: float) -> None:
        r = self.boundingRect()
        self.setPos(x_mm - r.width() * self.scale() / 2,
                    y_mm - r.height() * self.scale() / 2)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        # 中心点十字准星：让操作者看清落章中心（修改意见 #2/#3）
        r = self.boundingRect()
        cx, cy = r.center().x(), r.center().y()
        arm = min(r.width(), r.height()) * 0.15  # 局部坐标，随章大小自适应
        pen = QPen(Qt.red, 0)  # 0 = 1px 细线
        pen.setCosmetic(True)  # 线宽不随视图缩放变化
        painter.setPen(pen)
        painter.drawLine(cx - arm, cy, cx + arm, cy)
        painter.drawLine(cx, cy - arm, cx, cy + arm)


class PageCanvas(QGraphicsView):
    """页面视图：滚轮缩放、印章拖动、方向键 0.1mm 微调（Shift=1mm）。"""

    stamp_moved = Signal(object)  # StampItem

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self._page_item: QGraphicsPixmapItem | None = None

    def show_page(self, page_rgb: np.ndarray, phys_w_mm: float, phys_h_mm: float) -> None:
        self._scene.clear()
        pm = np_rgb_to_qpixmap(page_rgb)
        self._page_item = self._scene.addPixmap(pm)
        # 页图缩放到物理尺寸（mm）
        self._page_item.setScale(phys_w_mm / pm.width())
        self._page_item.setZValue(-1)
        self._scene.setSceneRect(0, 0, phys_w_mm, phys_h_mm)
        self.fit_page()

    def fit_page(self) -> None:
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def add_stamp(self, item: StampItem) -> None:
        self._scene.addItem(item)
        item.setSelected(True)

    def remove_selected_stamp(self) -> None:
        for it in self._scene.selectedItems():
            if isinstance(it, StampItem):
                self._scene.removeItem(it)

    def stamps(self) -> list[StampItem]:
        return [it for it in self._scene.items() if isinstance(it, StampItem)]

    def selected_stamp(self) -> StampItem | None:
        for it in self._scene.selectedItems():
            if isinstance(it, StampItem):
                return it
        return None

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        step = 1.0 if event.modifiers() & Qt.ShiftModifier else 0.1
        moves = {
            Qt.Key_Left: (-step, 0),
            Qt.Key_Right: (step, 0),
            Qt.Key_Up: (0, -step),
            Qt.Key_Down: (0, step),
        }
        stamp = self.selected_stamp()
        if stamp and event.key() in moves:
            dx, dy = moves[event.key()]
            cx, cy = stamp.center()
            stamp.set_center(cx + dx, cy + dy)
            self.stamp_moved.emit(stamp)
            return
        if stamp and event.key() == Qt.Key_Delete:
            self.remove_selected_stamp()
            return
        super().keyPressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        stamp = self.selected_stamp()
        if stamp:
            self.stamp_moved.emit(stamp)
