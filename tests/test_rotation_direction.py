"""旋转方向一致性回归测试（"预览正向、输出倒立" bug）。

根因：画布 Qt setRotation 与导出 PIL rotate 正角度方向相反，
用户修正 90° 时导出反向转 90°，净误差 180°。
"""

import cv2
import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsScene

from app.canvas import StampItem
from core.document import Page
from core.stamp import Placement, stamp_page


def _up_arrow() -> np.ndarray:
    """不对称测试图：红色上箭头（尖端朝上，质心偏下）。"""
    w = h = 200
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    pts = np.array([[100, 20], [30, 170], [170, 170]], dtype=np.int32)
    cv2.fillPoly(rgb, [pts], (200, 30, 30))
    alpha = np.where(rgb.sum(axis=2) > 0, 255, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def _ink_centroid_offset(img_rgb: np.ndarray, center_xy: tuple[float, float]) -> tuple[float, float]:
    r = img_rgb.astype(int)
    mask = (r[:, :, 0] - r[:, :, 1] > 20) & (r[:, :, 0] > 80)
    ys, xs = np.nonzero(mask)
    assert len(xs) > 50, "未检测到墨迹"
    return float(xs.mean() - center_xy[0]), float(ys.mean() - center_xy[1])


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_rotation_direction_canvas_matches_export(qapp):
    """画布与导出的旋转方向必须一致（任意角度符号相同）。"""
    arrow = _up_arrow()
    rot = 90.0

    # 画布渲染
    scene = QGraphicsScene()
    scene.setSceneRect(0, 0, 210, 297)
    white = QPixmap(400, 565)
    white.fill(0xFFFFFFFF)
    pi = scene.addPixmap(white)
    pi.setScale(210.0 / 400)
    pi.setZValue(-1)
    item = StampItem(arrow, 40.0, 105.0, 148.5)
    item.setRotation(rot)
    scene.addItem(item)
    img = QImage(400, 565, QImage.Format_RGBA8888)
    img.fill(0xFFFFFFFF)
    p = QPainter(img)
    scene.render(p)
    p.end()
    canvas_arr = np.frombuffer(img.bits(), dtype=np.uint8).reshape(565, 400, 4)[:, :, :3]
    cx_px, cy_px = 105 / 210 * 400, 148.5 / 297 * 565
    cdx, _ = _ink_centroid_offset(canvas_arr, (cx_px, cy_px))

    # 导出渲染
    page = Page(image=np.full((565, 400, 3), 255, np.uint8), phys_w_mm=210.0, phys_h_mm=297.0)
    out = stamp_page(page, arrow, Placement(105.0, 148.5, 40.0, rotation_deg=rot))
    edx, _ = _ink_centroid_offset(out, (cx_px, cy_px))

    # 方向一致：质心偏移同号且幅度接近
    assert cdx * edx > 0, f"旋转方向仍相反：画布 {cdx:.1f} vs 导出 {edx:.1f}"
    assert abs(abs(cdx) - abs(edx)) < 3.0


def test_zero_and_180_rotation_unchanged(qapp):
    """0° 和 180° 是对称角，修复前后行为不得改变。"""
    arrow = _up_arrow()
    page = Page(image=np.full((565, 400, 3), 255, np.uint8), phys_w_mm=210.0, phys_h_mm=297.0)
    out0 = stamp_page(page, arrow, Placement(105.0, 148.5, 40.0, rotation_deg=0.0))
    out180 = stamp_page(page, arrow, Placement(105.0, 148.5, 40.0, rotation_deg=180.0))
    dx0, dy0 = _ink_centroid_offset(out0, (200, 282.5))
    dx180, dy180 = _ink_centroid_offset(out180, (200, 282.5))
    # 0°：上箭头质心偏下（dy > 0）；180°：质心偏上（dy < 0）
    assert dy0 > 1 and dy180 < -1
