"""页面画布：渲染页面 + 印章拖拽/缩放/旋转交互。

场景坐标系直接使用物理 mm（原点在页面左上角）：
- 页面图像缩放铺满 (0, 0, phys_w_mm, phys_h_mm)；
- 印章位置/尺寸即物理值，导出时零换算误差（方案 §4.2 关键设计）。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QKeyEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
)


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
    """页面视图：滚轮缩放、印章拖动、方向键 0.1mm 微调（Shift=1mm）。

    额外交互模式（修改意见）：
    - 跟随落章：印章跟随鼠标，单击落位，Esc 取消；
    - 单击移位：空白处单击（位移 <5px，区别于拖拽平移）把选中章移过去；
    - 四点拾取：依次点击 4 个点（纸面角点），画标记①②③④+连线。
    """

    stamp_moved = Signal(object)      # StampItem
    stamp_placed = Signal(float, float)  # 跟随落章完成（mm）
    follow_cancelled = Signal()
    canvas_clicked = Signal(float, float)  # 空白处单击（mm）
    points_picked = Signal(list)      # [QPointF x 4]（mm 场景坐标）
    pick_cancelled = Signal()

    CLICK_THRESHOLD_PX = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self._page_item: QGraphicsPixmapItem | None = None
        self._press_view_pos = None
        # 跟随落章状态
        self._follow_item: StampItem | None = None
        # 四点拾取状态
        self._pick_mode = False
        self._pick_points: list = []
        self._pick_items: list = []

    # ── 页面显示 ──

    def show_page(self, page_rgb: np.ndarray, phys_w_mm: float, phys_h_mm: float) -> None:
        self.cancel_follow()
        self.cancel_pick()
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

    # ── 印章管理 ──

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

    # ── 跟随落章模式（修改意见：鼠标点哪盖哪）──

    def start_follow(self, rgba: np.ndarray, size_mm: float) -> None:
        """印章跟随鼠标，单击落位，Esc 取消。"""
        self.cancel_follow()
        self.cancel_pick()
        self._follow_item = StampItem(rgba, size_mm, 0, 0)
        self._follow_item.setOpacity(0.7)  # 跟随中半透明示意
        self._follow_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, False)
        self._follow_item.setFlag(QGraphicsPixmapItem.ItemIsSelectable, False)
        self._follow_item.setZValue(10)
        self._scene.addItem(self._follow_item)
        self.setDragMode(QGraphicsView.NoDrag)  # 跟随期间禁用平移，避免误拖
        self.setMouseTracking(True)

    def cancel_follow(self) -> None:
        if self._follow_item is not None:
            self._scene.removeItem(self._follow_item)
            self._follow_item = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setMouseTracking(False)

    @property
    def following(self) -> bool:
        return self._follow_item is not None

    # ── 四点拾取模式（修改意见：纸面四顶点）──

    def start_pick_points(self) -> None:
        self.cancel_follow()
        self.cancel_pick()
        self._pick_mode = True
        self._pick_points = []
        self._pick_items = []
        self.setDragMode(QGraphicsView.NoDrag)

    def cancel_pick(self) -> None:
        self._pick_mode = False
        self._pick_points = []
        for it in self._pick_items:
            self._scene.removeItem(it)
        self._pick_items = []
        if not self.following:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    @property
    def picking(self) -> bool:
        return self._pick_mode

    def _add_pick_marker(self, pos, index: int) -> None:
        r = 1.2  # mm
        dot = QGraphicsEllipseItem(pos.x() - r, pos.y() - r, 2 * r, 2 * r)
        dot.setBrush(QBrush(QColor(0, 120, 255)))
        dot.setPen(QPen(Qt.white, 0))
        dot.setZValue(20)
        self._scene.addItem(dot)
        self._pick_items.append(dot)
        label = QGraphicsSimpleTextItem(str(index))
        label.setBrush(QBrush(QColor(0, 120, 255)))
        label.setScale(0.6 / self.transform().m11() if self.transform().m11() > 0 else 3)
        label.setPos(pos.x() + r, pos.y() - 4 * r)
        label.setZValue(20)
        self._scene.addItem(label)
        self._pick_items.append(label)
        if len(self._pick_points) > 1:
            prev = self._pick_points[-2]
            line = QGraphicsLineItem(prev.x(), prev.y(), pos.x(), pos.y())
            line.setPen(QPen(QColor(0, 120, 255), 0))
            line.setZValue(19)
            self._scene.addItem(line)
            self._pick_items.append(line)
        if len(self._pick_points) == 4:  # 闭合
            first, last = self._pick_points[0], self._pick_points[3]
            line = QGraphicsLineItem(last.x(), last.y(), first.x(), first.y())
            line.setPen(QPen(QColor(0, 120, 255), 0))
            line.setZValue(19)
            self._scene.addItem(line)
            self._pick_items.append(line)

    # ── 事件 ──

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
        if self._pick_mode and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            self._pick_points.append(pos)
            self._add_pick_marker(pos, len(self._pick_points))
            if len(self._pick_points) == 4:
                pts = [(p.x(), p.y()) for p in self._pick_points]
                self._pick_mode = False
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                self.points_picked.emit(pts)
            return
        self._press_view_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._follow_item is not None:
            pos = self.mapToScene(event.position().toPoint())
            self._follow_item.set_center(pos.x(), pos.y())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._follow_item is not None and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            self._follow_item.set_center(pos.x(), pos.y())
            self.stamp_placed.emit(pos.x(), pos.y())
            return
        super().mouseReleaseEvent(event)
        # 单击（位移小于阈值）且未点在章上 → 发射空白单击
        if (
            self._press_view_pos is not None
            and event.button() == Qt.LeftButton
            and (event.position().toPoint() - self._press_view_pos).manhattanLength()
            < self.CLICK_THRESHOLD_PX
        ):
            item_at = self.itemAt(event.position().toPoint())
            if not isinstance(item_at, StampItem):
                pos = self.mapToScene(event.position().toPoint())
                self.canvas_clicked.emit(pos.x(), pos.y())
        self._press_view_pos = None
        stamp = self.selected_stamp()
        if stamp:
            self.stamp_moved.emit(stamp)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            if self.following:
                self.cancel_follow()
                self.follow_cancelled.emit()
                return
            if self.picking:
                self.cancel_pick()
                self.pick_cancelled.emit()
                return
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
