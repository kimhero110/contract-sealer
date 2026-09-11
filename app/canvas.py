"""页面画布：渲染页面 + 印章拖拽/缩放/旋转交互。

场景坐标系直接使用物理 mm（原点在页面左上角）：
- 页面图像缩放铺满 (0, 0, phys_w_mm, phys_h_mm)；
- 印章位置/尺寸即物理值，导出时零换算误差（方案 §4.2 关键设计）。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
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
    """一枚已落章的印章/签名。center() 为印章视觉中心（mm），rotation 为手动旋转角。

    注意：设置了 transformOriginPoint（旋转围绕章中心）后，Qt 的变换矩阵为
    pos + origin + S*(p - origin)，章的视觉中心 = pos + origin。
    因此 set_center/center 必须以 origin 为基准——曾经直接用 boundingRect*scale
    计算，导致视觉位置偏移 origin*(1-scale)（数百毫米），章被画到页面外，
    这就是"点击后毫无变化"的真正元凶。
    """

    def __init__(self, rgba: np.ndarray, size_mm: float, center_x_mm: float, center_y_mm: float):
        super().__init__()
        self.size_mm = size_mm
        self.setPixmap(np_rgba_to_qpixmap(rgba))
        self._update_scale()
        self._origin = self.boundingRect().center()
        self.setTransformOriginPoint(self._origin)
        self.set_center(center_x_mm, center_y_mm)
        self.setFlags(
            QGraphicsPixmapItem.ItemIsMovable | QGraphicsPixmapItem.ItemIsSelectable
        )

    def _update_scale(self) -> None:
        pm = self.pixmap()
        if pm.width() > 0:
            self.setScale(self.size_mm / pm.width())

    def set_pixmap_rgba(self, rgba: np.ndarray) -> None:
        """随机效果更新后刷新图像，保持中心与尺寸不变。"""
        center = self.center()
        self.setPixmap(np_rgba_to_qpixmap(rgba))
        self._update_scale()
        self._origin = self.boundingRect().center()
        self.setTransformOriginPoint(self._origin)
        self.set_center(*center)

    def center(self) -> tuple[float, float]:
        """视觉中心 = pos + origin（transformOriginPoint 即章图像中心）。"""
        return (self.pos().x() + self._origin.x(), self.pos().y() + self._origin.y())

    def set_center(self, x_mm: float, y_mm: float) -> None:
        self.setPos(x_mm - self._origin.x(), y_mm - self._origin.y())

    def paint(self, painter: QPainter, option, widget=None) -> None:
        # 三步序列（顺序不可调换）：
        # 1) multiply 绘制章体——印泥压进纸面，与导出 _multiply_composite 同语义
        painter.setCompositionMode(QPainter.CompositionMode_Multiply)
        painter.drawPixmap(0, 0, self.pixmap())
        # 2) 立即恢复 SourceOver，避免污染后续绘制
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        r = self.boundingRect()
        cx, cy = r.center().x(), r.center().y()
        # 3) 中心点十字准星（亮红，1px cosmetic）
        arm = min(r.width(), r.height()) * 0.15
        pen = QPen(Qt.red, 0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(cx - arm, cy, cx + arm, cy)
        painter.drawLine(cx, cy - arm, cx, cy + arm)
        # 选中框（手动画，避免 super().paint() 把它也 multiply 掉）
        if self.isSelected():
            dash = QPen(QColor(0, 120, 255), 0, Qt.DashLine)
            dash.setCosmetic(True)
            painter.setPen(dash)
            painter.drawRect(r)


class QuadHandle(QGraphicsEllipseItem):
    """四点校准的可拖角点。位置即场景 mm 坐标。

    半径按屏幕像素维持恒定：场景单位是毫米，不跟着缩放补偿的话，
    放大到 10 倍时把手会变成一个盖住半张纸的大圆饼。
    """

    RADIUS_PX = 9.0
    LABELS = ("左上", "右上", "右下", "左下")

    def __init__(self, index: int, x_mm: float, y_mm: float, px_per_mm: float):
        super().__init__()
        self.index = index
        self._notify = None  # 由画布注入：位置变化回调
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges  # itemChange 才会收到位置变化
        )
        self.setBrush(QBrush(QColor(0, 120, 255, 170)))
        self.setPen(QPen(Qt.white, 0))
        self.setZValue(30)
        self.setToolTip(f"{self.LABELS[index]}角：拖动调整，方向键微调 0.1mm（Shift 1mm）")
        self.set_screen_scale(px_per_mm)
        self.setPos(x_mm, y_mm)

    def set_screen_scale(self, px_per_mm: float) -> None:
        r = self.RADIUS_PX / max(px_per_mm, 1e-6)
        self.setRect(-r, -r, 2 * r, 2 * r)
        pen = QPen(Qt.white, 2.0 / max(px_per_mm, 1e-6))
        self.setPen(pen)

    def point(self) -> tuple[float, float]:
        return self.pos().x(), self.pos().y()

    def itemChange(self, change, value):
        # 钳在页面范围内用 ItemPositionChange（移动**之前**改写目标位置），
        # 而不是移动之后再 setPos 拉回来——后者会让 itemChange 递归自触发。
        if change == QGraphicsItem.ItemPositionChange and self.scene() is not None:
            rect = self.scene().sceneRect()
            if rect.isValid():
                return QPointF(
                    min(max(value.x(), rect.left()), rect.right()),
                    min(max(value.y(), rect.top()), rect.bottom()),
                )
        elif change == QGraphicsItem.ItemPositionHasChanged and self._notify:
            self._notify(self)
        return super().itemChange(change, value)


class PageCanvas(QGraphicsView):
    """页面视图：滚轮缩放、印章拖动、方向键 0.1mm 微调（Shift=1mm）。

    额外交互模式：
    - 跟随落章：印章跟随鼠标，单击落位，Esc 取消；
    - 单击移位：空白处单击（位移 <5px，区别于拖拽平移）把选中章移过去；
    - 四点校准：四个可拖把手 + 实时四边形，画布底部确认条或回车确认，Esc 取消。

    四点校准为什么是"拖把手"而不是"点四下"：点击模型下点了就定死，
    错一个角只能整个重来；点击自带一两像素抖动，而放大镜是六倍，
    用户在放大镜里看到没对准时点已经落下了；更要命的是取点期间
    dragMode 被设成 NoDrag，能缩放却不能平移——放大到看得清纸角后
    根本挪不到下一个角。把手模型天然是"按下→拖→松开"，随时可改，
    且沿用 ScrollHandDrag：拖把手是调整，拖空白处是平移。
    """

    stamp_moved = Signal(object)      # StampItem
    stamps_deleted = Signal(list)     # [StampItem] 用户在画布上按 Delete（显式删除）
    stamp_placed = Signal(float, float)  # 跟随落章完成（mm）
    place_rejected = Signal()         # 落位点击在页面外（护栏拦截）
    follow_cancelled = Signal()
    canvas_clicked = Signal(object, float, float)  # (候选 StampItem, x_mm, y_mm)
    quad_changed = Signal(list)       # [(x_mm, y_mm) x 4] 拖动中实时上报
    quad_accepted = Signal(list)      # 回车确认
    quad_cancelled = Signal()

    CLICK_THRESHOLD_PX = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHints(self.renderHints())
        self._page_item: QGraphicsPixmapItem | None = None
        self._press_view_pos = None
        self._press_stamp: StampItem | None = None  # press 时记录的候选章（selection 随后会被场景清空）
        self._just_placed = False                   # 消费式：落位后紧邻一次 click 不触发移位
        # 跟随落章状态
        self._follow_item: StampItem | None = None
        # 四点校准状态
        self._quad_mode = False
        self._quad_handles: list[QuadHandle] = []
        self._quad_outline: QGraphicsPolygonItem | None = None
        from app.confirmbar import ConfirmBar
        from app.magnifier import Magnifier

        # 放大镜（精确点选辅助）：取景回调由主窗口注入
        self.magnifier_source = None  # (x_mm, y_mm) -> QPixmap
        self._magnifier = Magnifier(self.viewport())
        # 四点校准的鼠标出口：不是所有人都知道要按回车，焦点也未必在画布上
        self.quad_bar = ConfirmBar(
            self.viewport(),
            "拖蓝色把手贴住纸的四个角",
            "✓ 完成校准",
            "✕ 取消",
        )
        self.quad_bar.accepted.connect(self.accept_quad_adjust)
        self.quad_bar.cancelled.connect(self.cancel_quad_from_user)

    # ── 页面显示 ──

    def show_page(self, page_rgb: np.ndarray, phys_w_mm: float, phys_h_mm: float) -> None:
        self.cancel_follow()
        was_adjusting = self._quad_mode
        self.cancel_quad_adjust()
        if was_adjusting:
            # 校准途中翻页/刷新会丢掉把手，得说一声，不能让它们无声消失
            self.quad_cancelled.emit()
        self._scene.clear()
        pm = np_rgb_to_qpixmap(page_rgb)
        self._page_item = self._scene.addPixmap(pm)
        # 页图缩放到物理尺寸（mm）
        self._page_item.setScale(phys_w_mm / pm.width())
        self._page_item.setZValue(-1)
        self._scene.setSceneRect(0, 0, phys_w_mm, phys_h_mm)
        self.fit_page()

    def scale(self, sx: float, sy: float) -> None:
        """重写以便任何缩放路径（菜单放大/缩小、代码调用）都同步把手尺寸。"""
        super().scale(sx, sy)
        self._rescale_quad_handles()

    def clear_page(self) -> None:
        """清空画布（关闭文档 / 删光页面时）。"""
        self.cancel_follow()
        self.cancel_quad_adjust()
        self._scene.clear()
        self._page_item = None
        self._scene.setSceneRect(0, 0, 0, 0)

    def fit_page(self) -> None:
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._rescale_quad_handles()

    # ── 印章管理 ──

    def add_stamp(self, item: StampItem) -> None:
        self._scene.addItem(item)
        item.setSelected(True)

    def remove_selected_stamp(self) -> None:
        removed = [it for it in self._scene.selectedItems() if isinstance(it, StampItem)]
        for it in removed:
            self._scene.removeItem(it)
        if removed:
            self.stamps_deleted.emit(removed)  # 显式删除：唯一的记录删除入口

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
        self.cancel_quad_adjust()
        self._reset_press_state()
        self._follow_item = StampItem(rgba, size_mm, 0, 0)
        self._follow_item.setOpacity(0.7)  # 跟随中半透明示意
        self._follow_item.setFlag(QGraphicsPixmapItem.ItemIsMovable, False)
        self._follow_item.setFlag(QGraphicsPixmapItem.ItemIsSelectable, False)
        self._follow_item.setZValue(10)
        self._scene.addItem(self._follow_item)
        self.setDragMode(QGraphicsView.NoDrag)  # 跟随期间禁用平移，避免误拖
        self.setMouseTracking(True)
        self.viewport().setCursor(Qt.CrossCursor)  # 十字光标：明确"正在选位置"

    def cancel_follow(self) -> None:
        if self._follow_item is not None:
            self._scene.removeItem(self._follow_item)
            self._follow_item = None
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.setMouseTracking(False)
            self.viewport().unsetCursor()
            self._magnifier.hide()
        self._reset_press_state()

    @property
    def following(self) -> bool:
        return self._follow_item is not None

    def _reset_press_state(self) -> None:
        """清理按下状态（模式切换时防状态泄漏）。"""
        self._press_view_pos = None
        self._press_stamp = None

    # ── 四点校准模式（可拖把手）──

    def start_quad_adjust(self, points_mm: list[tuple[float, float]]) -> None:
        """进入四点校准：按给定初值放四个把手，用户拖动调整。

        points_mm 已由调用方排好序（左上/右上/右下/左下），初值来自
        自动纸边检测，检测失败时是页面内缩框——用户永远不从零开始。
        """
        self.cancel_follow()
        self.cancel_quad_adjust()
        self._reset_press_state()
        self._quad_mode = True
        px_per_mm = self._px_per_mm()
        for i, (x, y) in enumerate(points_mm[:4]):
            handle = QuadHandle(i, x, y, px_per_mm)
            handle._notify = self._on_handle_moved
            self._scene.addItem(handle)
            self._quad_handles.append(handle)
        self._quad_outline = QGraphicsPolygonItem()
        self._quad_outline.setPen(QPen(QColor(0, 120, 255), 0, Qt.DashLine))
        self._quad_outline.setBrush(QBrush(QColor(0, 120, 255, 28)))
        self._quad_outline.setZValue(29)
        self._scene.addItem(self._quad_outline)
        self._refresh_quad_outline()
        # 保持 ScrollHandDrag：拖把手=调整，拖空白=平移。
        # 旧流程在这里设 NoDrag，于是能缩放却不能平移，是最劝退的一环。
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setMouseTracking(True)
        if self._quad_handles:
            self._quad_handles[0].setSelected(True)
        self.quad_bar.show_at_bottom()
        self.quad_changed.emit(self.quad_points())

    def cancel_quad_adjust(self) -> None:
        self._quad_mode = False
        self.quad_bar.hide()
        for handle in self._quad_handles:
            handle._notify = None
            self._scene.removeItem(handle)
        self._quad_handles = []
        if self._quad_outline is not None:
            self._scene.removeItem(self._quad_outline)
            self._quad_outline = None
        if not self.following:
            self.setMouseTracking(False)
            self._magnifier.hide()
        self._reset_press_state()

    @property
    def adjusting_quad(self) -> bool:
        return self._quad_mode

    def quad_points(self) -> list[tuple[float, float]]:
        return [h.point() for h in self._quad_handles]

    def accept_quad_adjust(self) -> None:
        """确认当前四边形。确认条按钮与回车共用这一条路径。"""
        if self._quad_mode:
            self.quad_accepted.emit(self.quad_points())

    def cancel_quad_from_user(self) -> None:
        """用户主动取消（确认条按钮 / Esc）：撤掉把手并通知主窗口。"""
        if self._quad_mode:
            self.cancel_quad_adjust()
            self.quad_cancelled.emit()

    def set_quad_hint(self, text: str) -> None:
        """把实时状态写到确认条上——用户的视线在画布，不在右侧信息栏。"""
        self.quad_bar.set_hint(text)

    def selected_handle(self) -> QuadHandle | None:
        for handle in self._quad_handles:
            if handle.isSelected():
                return handle
        return None

    def _px_per_mm(self) -> float:
        """视图缩放系数：场景单位是 mm，m11 即每毫米占多少屏幕像素。"""
        scale = self.transform().m11()
        return scale if scale > 1e-9 else 1.0

    def _rescale_quad_handles(self) -> None:
        """缩放后让把手保持恒定的屏幕尺寸。"""
        px_per_mm = self._px_per_mm()
        for handle in self._quad_handles:
            handle.set_screen_scale(px_per_mm)

    def _refresh_quad_outline(self) -> None:
        if self._quad_outline is None:
            return
        self._quad_outline.setPolygon(
            QPolygonF([QPointF(x, y) for x, y in self.quad_points()])
        )

    def _on_handle_moved(self, handle: QuadHandle) -> None:
        """把手位置变化：刷新轮廓、驱动放大镜、上报。范围钳制在把手内部完成。"""
        cx, cy = handle.point()
        self._refresh_quad_outline()
        if self.magnifier_source is not None:
            pm = self.magnifier_source(cx, cy)
            if pm is not None:
                self._magnifier.set_crop(pm)
                self._magnifier.follow_cursor(self._handle_global_pos(handle))
                self._magnifier.show()
        self.quad_changed.emit(self.quad_points())

    def _handle_global_pos(self, handle: QuadHandle):
        """把手在屏幕上的位置：拖动时放大镜贴着把手走，而不是贴着光标。"""
        view_pos = self.mapFromScene(handle.pos())
        return self.viewport().mapToGlobal(view_pos)

    def _nudge_handle(self, dx: float, dy: float) -> bool:
        handle = self.selected_handle()
        if handle is None:
            return False
        handle.setPos(handle.pos().x() + dx, handle.pos().y() + dy)
        return True

    # ── 事件 ──

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)  # 先让 super 更新 viewport 几何，再定位浮条
        if self._quad_mode:
            self.quad_bar.reposition()

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        # 缩放上下限：0.1x ~ 20x（防止缩到消失或糊成一团）
        sx = self.transform().m11()
        if sx * factor < 0.1 or sx * factor > 20.0:
            return
        self.scale(factor, factor)  # 重写过的 scale 会同步把手尺寸

    def mousePressEvent(self, event) -> None:
        if self._quad_mode:
            # 把手由场景自己处理拖动，空白处交给 ScrollHandDrag 平移；
            # 单击移位那套逻辑在校准模式下不参与，避免误移印章
            super().mousePressEvent(event)
            return
        # 先记录候选章，再调 super()——super 在空白处按下时会清空选中（Bug 1 修复）
        if event.button() == Qt.LeftButton:
            hit = self.itemAt(event.position().toPoint())
            self._press_stamp = hit if isinstance(hit, StampItem) else self.selected_stamp()
            self._press_view_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def _update_magnifier(self, scene_pos, global_pos) -> None:
        """跟随落章时驱动放大镜。四点校准的放大镜由把手移动回调驱动。"""
        active = self._follow_item is not None
        if active and self.magnifier_source is not None:
            pm = self.magnifier_source(scene_pos.x(), scene_pos.y())
            if pm is not None:
                self._magnifier.set_crop(pm)
                self._magnifier.follow_cursor(global_pos)
                self._magnifier.show()
                return
        self._magnifier.hide()

    def mouseMoveEvent(self, event) -> None:
        if self._follow_item is not None:
            pos = self.mapToScene(event.position().toPoint())
            self._follow_item.set_center(pos.x(), pos.y())
            self._update_magnifier(pos, event.globalPosition().toPoint())
            return
        if self._quad_mode:
            super().mouseMoveEvent(event)  # 把手拖动 / 空白处平移
            return
        self._magnifier.hide()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._quad_mode:
            super().mouseReleaseEvent(event)
            self._magnifier.hide()  # 松开才定稿，放大镜随即收起
            return
        if self._follow_item is not None and event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            # 页面外护栏：完全点在页面矩形外时忽略，防止盖出"隐形章"
            rect = self._scene.sceneRect()
            if rect.contains(pos):
                self._follow_item.set_center(pos.x(), pos.y())
                self._just_placed = True  # 消费式：下一次 click 判定吃掉
                self.stamp_placed.emit(pos.x(), pos.y())
            else:
                self.place_rejected.emit()  # 页面外点击，护栏拦截
            self._press_view_pos = None
            return
        super().mouseReleaseEvent(event)
        # 单击（位移小于阈值）且未点在章上 → 移动候选章
        if (
            self._press_view_pos is not None
            and event.button() == Qt.LeftButton
            and (event.position().toPoint() - self._press_view_pos).manhattanLength()
            < self.CLICK_THRESHOLD_PX
        ):
            item_at = self.itemAt(event.position().toPoint())
            if not isinstance(item_at, StampItem) and self._press_stamp is not None:
                if self._just_placed:
                    self._just_placed = False  # 吃掉落位后的紧邻点击
                else:
                    pos = self.mapToScene(event.position().toPoint())
                    self.canvas_clicked.emit(self._press_stamp, pos.x(), pos.y())
        self._press_view_pos = None
        self._press_stamp = None
        stamp = self.selected_stamp()
        if stamp:
            self.stamp_moved.emit(stamp)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            if self.following:
                self.cancel_follow()
                self.follow_cancelled.emit()
                return
            if self._quad_mode:
                self.cancel_quad_from_user()
                return
        step = 1.0 if event.modifiers() & Qt.ShiftModifier else 0.1
        if self._quad_mode:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.accept_quad_adjust()
                return
            nudges = {
                Qt.Key_Left: (-step, 0.0),
                Qt.Key_Right: (step, 0.0),
                Qt.Key_Up: (0.0, -step),
                Qt.Key_Down: (0.0, step),
            }
            if event.key() in nudges and self._nudge_handle(*nudges[event.key()]):
                return
            if event.key() == Qt.Key_Tab:  # 在四个角之间轮转，纯键盘也能走完
                handles = self._quad_handles
                current = self.selected_handle()
                nxt = handles[(handles.index(current) + 1) % len(handles)] if current else handles[0]
                self._scene.clearSelection()
                nxt.setSelected(True)
                return
            super().keyPressEvent(event)
            return
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
