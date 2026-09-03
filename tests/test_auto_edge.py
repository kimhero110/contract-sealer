"""骑缝章自适应纸边测试：真实纸边检测 + 贴合位置（消除扫描白边留空）。"""

import numpy as np

from core.document import A4_H_MM, A4_W_MM, Page
from core.perforation import (
    PerforationSpec,
    detect_paper_edge_x_px,
    plan_perforation,
)


def _page_with_margin(margin_px: int, bg: int = 60) -> Page:
    """合成带暗色边距的"扫描页"：白纸 + 右侧 margin_px 宽的暗背景。"""
    w, h = 1000, 1414
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    img[:, : w - margin_px] = 245  # 纸面
    # 纸面上有点内容（避免纯平）
    img[200:210, 100:800] = 120
    return Page(image=img, phys_w_mm=A4_W_MM, phys_h_mm=A4_H_MM)


def _red_block(w=472, h=472) -> np.ndarray:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 220
    rgba[:, :, 3] = 255
    return rgba


def test_detect_right_edge_with_margin():
    margin = 80
    page = _page_with_margin(margin)
    edge = detect_paper_edge_x_px(page.image, "right")
    expected = page.width_px - margin - 1
    assert abs(edge - expected) <= 5, f"检测纸边 {edge}，期望 {expected}±5"


def test_detect_left_edge_with_margin():
    w, h = 1000, 1414
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    margin = 60
    img[:, margin:] = 245
    edge = detect_paper_edge_x_px(img, "left")
    assert abs(edge - margin) <= 5


def test_full_bleed_fallback():
    """纸铺满画面时回退到图像边缘。"""
    img = np.full((800, 600, 3), 240, dtype=np.uint8)
    assert detect_paper_edge_x_px(img, "right") == 599
    assert detect_paper_edge_x_px(img, "left") == 0


def test_auto_edge_eliminates_margin_gap():
    """自适应贴合：切片右边缘应紧贴检测到的纸边，而不是图像边缘（消除留空）。"""
    margin = 80
    pages = [_page_with_margin(margin) for _ in range(3)]
    spec = PerforationSpec(seed=1, inset_mm=0.5, auto_edge=True,
                           width_jitter=0.0, offset_jitter_mm=0.0, rot_jitter_deg=0.0)
    placements = plan_perforation(_red_block(), pages, [0, 1, 2], spec)
    page = pages[0]
    paper_edge_mm = (page.width_px - margin) / page.dpi * 25.4
    for pl in placements:
        # 切片右边缘 = 纸边 - 0.5mm 内缩
        assert abs(pl.right_edge_mm - (paper_edge_mm - 0.5)) < 0.3
        # 且明显不同于旧的"图像边缘 - inset"行为
        assert pl.right_edge_mm < page.phys_w_mm - 1.0


def test_auto_edge_off_uses_image_edge():
    pages = [_page_with_margin(80) for _ in range(2)]
    spec = PerforationSpec(seed=1, inset_mm=1.5, auto_edge=False,
                           width_jitter=0.0, offset_jitter_mm=0.0, rot_jitter_deg=0.0)
    placements = plan_perforation(_red_block(), pages, [0, 1], spec)
    for pl in placements:
        assert abs(pl.right_edge_mm - (A4_W_MM - 1.5)) < 1e-6


def test_vignette_tolerance():
    """渐晕（纸边发暗但仍是纸）不应被误判为背景。"""
    w, h = 1000, 1414
    img = np.full((h, w, 3), 50, dtype=np.uint8)
    img[:, :950] = 245
    # 纸面右缘 30px 渐晕到 190（仍是纸）
    for i in range(30):
        img[:, 920 + i] = 245 - i * 2
    page = Page(image=img, phys_w_mm=A4_W_MM, phys_h_mm=A4_H_MM)
    edge = detect_paper_edge_x_px(page.image, "right")
    assert edge >= 940, f"渐晕被误判为背景，纸边检到 {edge}"
