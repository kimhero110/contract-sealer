"""四点纸边校准测试（修改意见）：用户点四顶点 → 透视拉伸为标准 A4。"""

import numpy as np
import cv2
import pytest

from core.autocal import map_points_through, warp_to_a4
from core.document import A4_H_MM, A4_W_MM, Page


def _photo_with_margin() -> tuple[np.ndarray, np.ndarray]:
    """合成带白边+微旋转的"扫描页"，返回 (图像, 纸面四顶点 4x2)。"""
    w, h = 1600, 2200
    img = np.full((h, w, 3), 70, dtype=np.uint8)  # 深灰背景
    # 纸面区域（带白边，模拟扫描件自带白边）
    quad = np.array(
        [[180, 120], [1420, 90], [1470, 1980], [150, 2010]], dtype=np.float32
    )
    paper = np.full((1900, 1240, 3), 250, dtype=np.uint8)
    cv2.putText(paper, "CONTRACT", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (60, 60, 60), 5)
    dst = np.array([[0, 0], [1240, 0], [1240, 1900], [0, 1900]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(dst, quad)
    cv2.warpPerspective(paper, m, (w, h), img, borderMode=cv2.BORDER_TRANSPARENT)
    return img, quad


def test_warp_to_a4_from_user_quad():
    img, quad = _photo_with_margin()
    page = Page(image=img, phys_w_mm=999.0, phys_h_mm=999.0, needs_calibration=True)
    H = warp_to_a4(page, quad)
    assert H is not None and H.shape == (3, 3)
    # 物理尺寸 = 标准 A4
    assert abs(page.phys_w_mm - A4_W_MM) < 1e-6
    assert abs(page.phys_h_mm - A4_H_MM) < 1e-6
    assert page.needs_calibration is False
    # 输出图像比例 = √2
    h, w = page.image.shape[:2]
    assert abs(h / w - A4_H_MM / A4_W_MM) < 0.02


def test_warp_rejects_degenerate_quad():
    img = np.full((800, 600, 3), 200, dtype=np.uint8)
    page = Page(image=img, phys_w_mm=210.0, phys_h_mm=297.0)
    tiny = np.array([[10, 10], [12, 10], [12, 12], [10, 12]], dtype=np.float32)
    assert warp_to_a4(page, tiny) is None


def test_stamp_position_follows_warp():
    """已盖章位置必须按同一变换映射：纸面中心点校准后应在 A4 页面中心。"""
    img, quad = _photo_with_margin()
    page = Page(image=img, phys_w_mm=160.0, phys_h_mm=220.0)  # 校准前的错误物理尺寸
    H = warp_to_a4(page, quad)
    assert H is not None
    # 纸面四点的质心（像素）→ 应映射到新 A4 页面中心
    center_px = quad.mean(axis=0, keepdims=True)
    mapped = map_points_through(H, center_px)
    new_h, new_w = page.image.shape[:2]
    assert abs(mapped[0][0] - new_w / 2) < new_w * 0.02
    assert abs(mapped[0][1] - new_h / 2) < new_h * 0.02


def test_quad_order_insensitive():
    """四点任意顺序点击都能得到相同校准结果（内部自动排序）。"""
    img, quad = _photo_with_margin()
    shuffled = quad[[2, 0, 3, 1]]  # 乱序
    page1 = Page(image=img.copy(), phys_w_mm=999.0, phys_h_mm=999.0)
    page2 = Page(image=img.copy(), phys_w_mm=999.0, phys_h_mm=999.0)
    H1 = warp_to_a4(page1, quad)
    H2 = warp_to_a4(page2, shuffled)
    assert H1 is not None and H2 is not None
    assert page1.image.shape == page2.image.shape
