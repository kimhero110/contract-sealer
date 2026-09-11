"""骑缝章测试：宽度守恒、页序映射、拼合还原、最小宽度约束、抖动硬上限。"""

import numpy as np
import pytest

from core.document import A4_H_MM, A4_W_MM, Page, mm_to_px
from core.extract import extract_red_seal
from core.perforation import (
    CAP_OFFSET_JITTER_MM,
    CAP_ROT_JITTER_DEG,
    SIDE_LEFT,
    SIDE_RIGHT,
    PerforationSpec,
    apply_perforation,
    assemble_preview,
    min_slice_warning,
    plan_perforation,
    slice_index_for_page,
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
    spec = PerforationSpec(seed=6, side=SIDE_LEFT, width_jitter=0.0,
                           offset_jitter_mm=0.0, rot_jitter_deg=0.0)
    placements = plan_perforation(_red_block(), pages, [0, 1], spec)
    out = apply_perforation(pages, placements)
    for img in out.values():
        left_strip = img[:, : int(img.shape[1] * 0.1)]
        right_strip = img[:, int(img.shape[1] * 0.9):]
        left_ink = (255 - left_strip.mean(axis=2)).mean()
        right_ink = (255 - right_strip.mean(axis=2)).mean()
        assert left_ink > right_ink + 1


# ── 旋转抖动下的宽度守恒（回归：PIL expand 把切片撑宽，守恒被悄悄破坏）──

def test_rotation_jitter_preserves_slice_widths():
    """开了逐页旋转抖动后，切片实际宽度之和仍须等于印章总宽。

    PIL rotate 的 expand 会把位图撑宽 |w·cosθ| + |h·sinθ|。不裁回原宽的话，
    40mm 章切 5 片、抖动 1° 时每片多出约 0.7mm（≈9%），拼合预览偏宽、
    导出时每片墨迹相对纸边内移半个扩边量。
    """
    pages = _a4_pages(5)
    total_px = max(5, round(mm_to_px(40.0, pages[0].dpi)))
    for seed in range(20):
        placements = plan_perforation(
            _red_block(),
            pages,
            [0, 1, 2, 3, 4],
            PerforationSpec(seed=seed, rot_jitter_deg=CAP_ROT_JITTER_DEG, auto_edge=False),
        )
        actual = sum(p.slice_rgba.shape[1] for p in placements)
        declared = sum(p.width_px for p in placements)
        assert actual == total_px, f"seed={seed} 旋转后宽度和 {actual} != {total_px}"
        assert declared == total_px, f"seed={seed} 声明宽度和 {declared} != {total_px}"


def test_assemble_preview_width_matches_seal_width():
    """拼合预览的墨迹总宽 = 印章宽（±padding），不能被旋转扩边撑胖。"""
    pages = _a4_pages(4)
    total_px = round(mm_to_px(40.0, pages[0].dpi))
    placements = plan_perforation(
        _red_block(),
        pages,
        [0, 1, 2, 3],
        PerforationSpec(seed=5, rot_jitter_deg=CAP_ROT_JITTER_DEG, auto_edge=False),
    )
    pad_mm = 2.0
    canvas = assemble_preview(placements, pages[0].dpi, pad_mm=pad_mm)
    pad_px = round(mm_to_px(pad_mm, pages[0].dpi))
    assert canvas.shape[1] == total_px + 2 * pad_px


def test_rotation_jitter_still_rotates():
    """裁回原宽不等于把旋转裁没了：抖动后的切片不应与未抖动的完全相同。"""
    pages = _a4_pages(3)
    seal = _red_block()
    seal[: seal.shape[0] // 2] = 0  # 上下不对称，旋转后必然有差异
    seal[: seal.shape[0] // 2, :, 3] = 0
    common = {"auto_edge": False, "offset_jitter_mm": 0.0, "width_jitter": 0.0}
    flat = plan_perforation(seal, pages, [0, 1, 2], PerforationSpec(seed=1, rot_jitter_deg=0.0, **common))
    tilted = plan_perforation(seal, pages, [0, 1, 2], PerforationSpec(seed=1, rot_jitter_deg=CAP_ROT_JITTER_DEG, **common))
    assert any(
        not np.array_equal(a.slice_rgba, b.slice_rgba) for a, b in zip(flat, tilted, strict=True)
    ), "旋转抖动没有生效"


# ── 左开口页序映射（回归：位置做了镜像，页序映射没做，拼出来印文左右颠倒）──

def _fan_order(placements, side):
    """把文件按 side 扇开后，从左到右看到的切片顺序。

    右开口：页 1 在最上，往右扇开露出的顺序是页 1..N；
    左开口：往左扇开时越靠后的页露得越靠左，从左到右是页 N..1。
    """
    by_page = sorted(placements, key=lambda p: p.page_index)
    return by_page if side == SIDE_RIGHT else by_page[::-1]


def _gradient_seal(w=400, h=400) -> np.ndarray:
    """左暗右亮的不对称印章：拼反了一眼就能看出来。"""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = np.linspace(10, 250, w).astype(np.uint8)[None, :]
    rgba[:, :, 3] = 255
    return rgba


def _plan(side, n=4, **kwargs):
    spec = {
        "side": side, "seed": 0, "width_jitter": 0.0, "rot_jitter_deg": 0.0,
        "offset_jitter_mm": 0.0, "auto_edge": False,
    }
    spec.update(kwargs)
    return plan_perforation(_gradient_seal(), _a4_pages(n), list(range(n)), PerforationSpec(**spec))


def test_slice_index_mapping_mirrors_with_side():
    assert [slice_index_for_page(i, 4, SIDE_RIGHT) for i in range(4)] == [0, 1, 2, 3]
    assert [slice_index_for_page(i, 4, SIDE_LEFT) for i in range(4)] == [3, 2, 1, 0]


def test_left_side_gives_page_one_the_rightmost_slice():
    """左开口时页 1 露在最右，必须拿印章最右那条。"""
    placements = _plan(SIDE_LEFT, n=4)
    by_page = sorted(placements, key=lambda p: p.page_index)
    assert [p.slice_order for p in by_page] == [3, 2, 1, 0]


def test_right_side_mapping_unchanged():
    by_page = sorted(_plan(SIDE_RIGHT, n=4), key=lambda p: p.page_index)
    assert [p.slice_order for p in by_page] == [0, 1, 2, 3]


@pytest.mark.parametrize("side", [SIDE_RIGHT, SIDE_LEFT])
def test_fanned_reassembly_reproduces_seal(side):
    """按实际扇开顺序拼回去，必须还原印章本身的左右朝向。"""
    placements = _plan(side, n=5)
    fanned = np.concatenate(
        [p.slice_rgba for p in _fan_order(placements, side)], axis=1
    )
    profile = fanned[:, :, 0].mean(axis=0)
    assert profile[0] < profile[-1], f"{side} 拼合结果左右颠倒"
    original = _gradient_seal()[:, :, 0].mean(axis=0)
    assert np.corrcoef(
        np.interp(np.linspace(0, 1, 64), np.linspace(0, 1, len(profile)), profile),
        np.interp(np.linspace(0, 1, 64), np.linspace(0, 1, len(original)), original),
    )[0, 1] > 0.99


@pytest.mark.parametrize("side", [SIDE_RIGHT, SIDE_LEFT])
def test_assemble_preview_matches_fanned_reality(side):
    """拼合预览就是用户拿来确认"拼得回去"的图，不能和实际扇开结果不一致。"""
    pages = _a4_pages(4)
    placements = _plan(side, n=4)
    preview = assemble_preview(placements, pages[0].dpi, pad_mm=0.0)
    fanned = np.concatenate(
        [p.slice_rgba for p in _fan_order(placements, side)], axis=1
    )
    assert preview.shape[1] == fanned.shape[1]
    # 预览是白底正片叠底的结果，比较横向亮度剖面的走向即可
    pv = preview[:, :, 0].mean(axis=0)
    assert (pv[0] < pv[-1]) == (fanned[:, :, 0].mean(axis=0)[0] < fanned[:, :, 0].mean(axis=0)[-1])


def test_left_side_slice_hugs_left_paper_edge():
    """页序反了不代表位置也该反：每片仍旧贴各自页面的左纸边。"""
    pages = _a4_pages(3)
    placements = _plan(SIDE_LEFT, n=3, inset_mm=2.0)
    for p in placements:
        width_mm = p.slice_rgba.shape[1] / pages[p.page_index].dpi * 25.4
        left_edge_mm = p.right_edge_mm - width_mm
        assert abs(left_edge_mm - 2.0) < 1e-6
