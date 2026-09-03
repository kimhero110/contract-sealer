"""骑缝章测试：宽度守恒、页序映射、拼合还原、最小宽度约束、抖动硬上限。"""

import numpy as np
import pytest

from core.document import A4_H_MM, A4_W_MM, Page, mm_to_px
from core.extract import extract_red_seal
from core.perforation import (
    CAP_OFFSET_JITTER_MM,
    CAP_ROT_JITTER_DEG,
    MIN_SLICE_WIDTH_MM,
    PerforationSpec,
    apply_perforation,
    assemble_preview,
    min_slice_warning,
    plan_perforation,
    slice_seal,
    slice_widths_px,
)


def _a4_pages(n: int, dpi: int = 300) -> list[Page]:
    w = round(A4_W_MM / 25.4 * dpi)
    h = round(A4_H_MM / 25.4 * dpi)
    return [
        Page(image=np.full((h, w, 3), 255, dtype=np.uint8), phys_w_mm=A4_W_MM, phys_h_mm=A4_H_MM)
        for _ in range(n)
    ]


def _red_block(w=472, h=472) -> np.ndarray:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 220
    rgba[:, :, 3] = 255
    return rgba


# ── 宽度守恒属性测试（方案 §4.7：100 个随机种子）──

def test_width_conservation_property():
    for seed in range(100):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 30))
        total = int(rng.integers(max(n, 100), 2000))
        widths = slice_widths_px(total, n, jitter=0.4, rng=rng)
        assert sum(widths) == total, f"seed={seed} 宽度不守恒"
        assert all(w >= 1 for w in widths), f"seed={seed} 出现 0 宽切片"


def test_zero_jitter_is_uniform():
    rng = np.random.default_rng(0)
    widths = slice_widths_px(100, 4, jitter=0.0, rng=rng)
    assert widths == [25, 25, 25, 25]


# ── 最小宽度约束 ──

def test_min_slice_warning():
    assert min_slice_warning(40.0, 10) is None
    assert min_slice_warning(40.0, 40) is not None  # 1mm < 1.5mm
    assert min_slice_warning(40.0, 0) is not None
    with pytest.raises(ValueError):
        plan_perforation(_red_block(), _a4_pages(40), list(range(40)), PerforationSpec(seed=1))


# ── 页序映射 ──

def test_page_order_mapping():
    pages = _a4_pages(4)
    seal = _red_block()
    spec = PerforationSpec(seed=7, width_jitter=0.0, offset_jitter_mm=0.0, rot_jitter_deg=0.0)
    placements = plan_perforation(seal, pages, [0, 1, 2, 3], spec)
    assert [p.page_index for p in placements] == [0, 1, 2, 3]
    # 等宽切分时第 i 页切片应与手动切分一致
    total_px = round(mm_to_px(40.0, pages[0].dpi))
    expected = slice_seal(_red_block(w=total_px, h=total_px), [total_px // 4] * 3 + [total_px - 3 * (total_px // 4)])
    for i, p in enumerate(placements):
        assert p.slice_rgba.shape[1] == expected[i].shape[1]


def test_slice_right_edge_position():
    pages = _a4_pages(2)
    spec = PerforationSpec(seed=3, inset_mm=2.0, width_jitter=0.0,
                           offset_jitter_mm=0.0, rot_jitter_deg=0.0)
    placements = plan_perforation(_red_block(), pages, [0, 1], spec)
    for p in placements:
        assert abs(p.right_edge_mm - (A4_W_MM - 2.0)) < 1e-6


# ── 拼合还原（golden 断言：无抖动时拼合 ≈ 原图）──

def test_reassembly_without_jitter(seal_png):
    pages = _a4_pages(5)
    seal = extract_red_seal(seal_png)
    spec = PerforationSpec(seed=11, width_jitter=0.3, offset_jitter_mm=0.0, rot_jitter_deg=0.0)
    placements = plan_perforation(seal, pages, list(range(5)), spec)
    preview = assemble_preview(placements, pages[0].dpi)

    # 拼合图应包含印章的红色内容：红色覆盖率与原章同量级
    import cv2

    hsv = cv2.cvtColor(preview, cv2.COLOR_RGB2HSV)
    red = ((hsv[:, :, 0] <= 14) | (hsv[:, :, 0] >= 160)) & (hsv[:, :, 1] > 60)
    src_red_ratio = float(np.mean(seal[:, :, 3] > 128))
    assert red.mean() > src_red_ratio * 0.3  # 拼合图有大量红色


def test_reassembly_alpha_conservation(seal_png):
    """宽度守恒 ⇒ 拼合后墨迹总量 ≈ 原章墨迹总量（容差 5%）。"""
    pages = _a4_pages(6)
    seal = extract_red_seal(seal_png)
    spec = PerforationSpec(seed=5, width_jitter=0.25, offset_jitter_mm=0.0, rot_jitter_deg=0.0)
    placements = plan_perforation(seal, pages, list(range(6)), spec)
    total_alpha = sum(int(p.slice_rgba[:, :, 3].sum()) for p in placements)
    # 与切割前的缩放原图比较
    from core.perforation import _resize_width

    total_px = round(mm_to_px(40.0, pages[0].dpi))
    resized = _resize_width(seal, total_px)
    src_alpha = int(resized[:, :, 3].sum())
    assert abs(total_alpha - src_alpha) / src_alpha < 0.05


# ── 逐页抖动硬上限 ──

def test_jitter_hard_caps():
    pages = _a4_pages(10)
    spec = PerforationSpec(seed=9, offset_jitter_mm=99.0, rot_jitter_deg=99.0)
    placements = plan_perforation(_red_block(), pages, list(range(10)), spec)
    for p in placements:
        assert abs(p.y_offset_mm) <= CAP_OFFSET_JITTER_MM
    # clamp 后 spec 本身也被限制
    clamped = spec.clamped()
    assert clamped.offset_jitter_mm == CAP_OFFSET_JITTER_MM
    assert clamped.rot_jitter_deg == CAP_ROT_JITTER_DEG


# ── 落章到页面 ──

def test_apply_perforation_marks_pages():
    pages = _a4_pages(3)
    spec = PerforationSpec(seed=2, width_jitter=0.2, offset_jitter_mm=0.0, rot_jitter_deg=0.0)
    placements = plan_perforation(_red_block(), pages, [0, 1, 2], spec)
    out = apply_perforation(pages, placements)
    assert set(out.keys()) == {0, 1, 2}
    for idx, img in out.items():
        assert img.shape == pages[idx].image.shape
        assert not np.array_equal(img, pages[idx].image)
        # 墨迹应出现在右边缘附近（右 10% 区域内变暗/变红）
        right_strip = img[:, int(img.shape[1] * 0.9):]
        assert right_strip[:, :, 0].mean() < pages[idx].image[:, :, 0].mean() - 5 or \
               (right_strip[:, :, 0].astype(int) - right_strip[:, :, 1].astype(int)).mean() > 5


def test_partial_page_range():
    """页范围 = 文档子集：只有选中页被盖章（v1.3：N 为选中页数）。"""
    pages = _a4_pages(8)
    spec = PerforationSpec(seed=4)
    placements = plan_perforation(_red_block(), pages, [2, 3, 4, 5], spec)
    assert [p.page_index for p in placements] == [2, 3, 4, 5]
    out = apply_perforation(pages, placements)
    assert set(out.keys()) == {2, 3, 4, 5}


def test_left_side_marks_left_edge():
    pages = _a4_pages(2)
    spec = PerforationSpec(seed=6, side="left", width_jitter=0.0,
                           offset_jitter_mm=0.0, rot_jitter_deg=0.0)
    placements = plan_perforation(_red_block(), pages, [0, 1], spec)
    out = apply_perforation(pages, placements)
    for img in out.values():
        left_strip = img[:, : int(img.shape[1] * 0.1)]
        right_strip = img[:, int(img.shape[1] * 0.9):]
        left_ink = (255 - left_strip.mean(axis=2)).mean()
        right_ink = (255 - right_strip.mean(axis=2)).mean()
        assert left_ink > right_ink + 1
