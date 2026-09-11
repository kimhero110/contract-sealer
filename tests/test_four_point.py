"""四点纸边校准测试：自动检测给初值 → 用户拖四角 → 透视拉伸为标准 A4。"""

import cv2
import numpy as np

from core.autocal import (
    initial_quad,
    map_points_through,
    quad_area_ratio,
    quad_aspect,
    quad_target_size,
    warp_to_a4,
)
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


# ── 初值：自动检测成功用检测结果，失败给默认框（用户永远不从零点起）──

def test_initial_quad_uses_detection_when_paper_is_visible():
    img, quad = _photo_with_margin()
    got, detected = initial_quad(img)
    assert detected is True
    assert got.shape == (4, 2)
    # 检测出的四角应贴近真实纸角（容差取图像长边的 3%）
    tolerance = max(img.shape[:2]) * 0.03
    expected = sorted(map(tuple, quad))
    actual = sorted(map(tuple, got))
    for (ex, ey), (ax, ay) in zip(expected, actual, strict=True):
        assert abs(ax - ex) < tolerance and abs(ay - ey) < tolerance


def _undetectable() -> np.ndarray:
    """检测必然失败的图：暗底上散落几块小亮斑，没有一块够得上纸面。"""
    img = np.full((2000, 1400, 3), 40, dtype=np.uint8)
    for x, y in ((100, 150), (900, 400), (300, 1500), (1000, 1700)):
        img[y : y + 220, x : x + 220] = 245
    return img


def test_full_bleed_scan_detects_whole_image():
    """纸铺满画面的无边扫描件：整幅就是纸面，检测应当成功并返回全图。

    这不是误报——此时把手落在图像四角、确认后近似恒等变换，正是想要的结果。
    """
    img = np.full((2000, 1400, 3), 252, dtype=np.uint8)
    got, detected = initial_quad(img)
    assert detected is True
    h, w = img.shape[:2]
    assert got[:, 0].max() > w * 0.9 and got[:, 1].max() > h * 0.9


def test_initial_quad_falls_back_to_inset_box():
    """检测失败时仍要给出一个可拖的框，绝不把用户丢回"从零点四下"。"""
    img = _undetectable()
    got, detected = initial_quad(img, inset_ratio=0.05)
    assert detected is False
    assert got.shape == (4, 2)
    h, w = img.shape[:2]
    xs, ys = got[:, 0], got[:, 1]
    assert abs(xs.min() - w * 0.05) < 1.0
    assert abs(xs.max() - (w - 1 - w * 0.05)) < 1.0
    assert abs(ys.min() - h * 0.05) < 1.0
    assert abs(ys.max() - (h - 1 - h * 0.05)) < 1.0


def test_initial_quad_is_ordered_top_left_first():
    """返回值必须是有序的，画布按顺序连线才不会连成交叉的蝴蝶形。"""
    got, _ = initial_quad(_undetectable())
    tl, tr, br, bl = got
    assert tl[0] < tr[0] and bl[0] < br[0]     # 左列在右列左边
    assert tl[1] < bl[1] and tr[1] < br[1]     # 上行在下行上面


def test_initial_quad_always_inside_image():
    img = _undetectable()
    got, _ = initial_quad(img)
    h, w = img.shape[:2]
    assert got[:, 0].min() >= 0 and got[:, 0].max() <= w - 1
    assert got[:, 1].min() >= 0 and got[:, 1].max() <= h - 1


# ── 实时比例提示 ──

def test_quad_aspect_reports_pre_warp_ratio():
    """比例取拉正前的边长。取拉正后的恒等于 √2，拿来提示毫无意义。"""
    square = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    assert abs(quad_aspect(square) - 1.0) < 1e-6
    a4ish = np.array([[0, 0], [100, 0], [100, 141], [0, 141]], dtype=np.float32)
    assert abs(quad_aspect(a4ish) - 1.41) < 0.01


def test_quad_aspect_is_order_insensitive():
    quad = np.array([[0, 0], [100, 0], [100, 141], [0, 141]], dtype=np.float32)
    assert abs(quad_aspect(quad) - quad_aspect(quad[[2, 0, 3, 1]])) < 1e-6


def test_quad_aspect_handles_degenerate_quad():
    collinear = np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32)
    assert quad_aspect(collinear) == 0.0


def test_quad_target_size_respects_orientation():
    portrait = np.array([[0, 0], [100, 0], [100, 300], [0, 300]], dtype=np.float32)
    w, h, landscape = quad_target_size(portrait)
    assert landscape is False and h > w
    landscape_quad = np.array([[0, 0], [300, 0], [300, 100], [0, 100]], dtype=np.float32)
    w2, h2, landscape2 = quad_target_size(landscape_quad)
    assert landscape2 is True and w2 > h2


def test_quad_area_ratio():
    quad = np.array([[0, 0], [50, 0], [50, 50], [0, 50]], dtype=np.float32)
    assert abs(quad_area_ratio(quad, (100, 100)) - 0.25) < 1e-3
