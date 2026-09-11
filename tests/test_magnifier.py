"""放大镜（精确点选取景器）测试。"""

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")  # 没装 Qt 的机器上只跑 core 层
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from core.document import Document, Page

FIX = "tests/fixtures"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_window(qapp, page: Page | None = None) -> MainWindow:
    win = MainWindow()
    if page is None:
        page = Page(image=np.full((1754, 1240, 3), 250, np.uint8), phys_w_mm=210.0, phys_h_mm=297.0)
    win.doc = Document(pages=[page])
    win._rebuild_page_list(select=0)
    win.show()
    return win


def test_magnifier_crop_centers_on_cursor(qapp):
    """取景画面以光标为中心：在红点位置取景，中心像素应为红。"""
    img = np.full((1754, 1240, 3), 250, np.uint8)
    page = Page(image=img, phys_w_mm=210.0, phys_h_mm=297.0)
    # 在 (100mm, 150mm) 画一个 2mm 红点
    dpi = page.dpi
    cx, cy = int(100 / 25.4 * dpi), int(150 / 25.4 * dpi)
    r = int(1 / 25.4 * dpi)
    img[cy - r : cy + r, cx - r : cx + r] = (180, 30, 30)

    win = _make_window(qapp, page)
    pm = win._magnifier_crop(100.0, 150.0)
    assert pm is not None and not pm.isNull()
    # 中心像素 ≈ 红（允 PNG 转换误差）
    img_q = pm.toImage()
    c = img_q.pixelColor(img_q.width() // 2, img_q.height() // 2)
    assert c.red() > 120 and c.red() - c.green() > 60
    win.close()


def test_magnifier_crop_pads_at_edge(qapp):
    """页面角落取景：补边后仍是正方形且不崩溃。"""
    page = Page(image=np.full((1754, 1240, 3), 250, np.uint8), phys_w_mm=210.0, phys_h_mm=297.0)
    win = _make_window(qapp, page)
    pm = win._magnifier_crop(0.5, 0.5)  # 左上角
    assert pm is not None and pm.width() == pm.height() and pm.width() > 50  # 补边后仍为正方形
    win.close()


def test_magnifier_follows_quad_handle(qapp):
    """四点校准：拖动把手时放大镜贴着把手显示，退出模式后隐藏。"""
    win = _make_window(qapp)
    win.canvas.start_quad_adjust([(20.0, 20.0), (190.0, 20.0), (190.0, 277.0), (20.0, 277.0)])
    qapp.processEvents()
    handle = win.canvas._quad_handles[0]
    handle.setPos(30.0, 35.0)          # 模拟拖动，itemChange 会驱动放大镜
    qapp.processEvents()
    assert win.canvas._magnifier.isVisible()
    win.canvas.cancel_quad_adjust()
    qapp.processEvents()
    assert not win.canvas._magnifier.isVisible()
    win.close()


def test_quad_handles_seeded_and_clamped(qapp):
    """把手按初值落位；拖出页面范围会被钳回纸内。"""
    win = _make_window(qapp)
    pts = [(20.0, 20.0), (190.0, 20.0), (190.0, 277.0), (20.0, 277.0)]
    win.canvas.start_quad_adjust(pts)
    assert win.canvas.adjusting_quad
    assert [(round(x, 1), round(y, 1)) for x, y in win.canvas.quad_points()] == pts
    win.canvas._quad_handles[0].setPos(-50.0, -50.0)
    qapp.processEvents()
    x, y = win.canvas.quad_points()[0]
    assert x >= 0.0 and y >= 0.0
    win.close()


def test_quad_handle_nudge_uses_physical_step(qapp):
    """方向键微调走物理毫米，与落章微调同一套步进。"""
    win = _make_window(qapp)
    win.canvas.start_quad_adjust([(20.0, 20.0), (190.0, 20.0), (190.0, 277.0), (20.0, 277.0)])
    win.canvas._quad_handles[0].setSelected(True)
    assert win.canvas._nudge_handle(0.1, 0.0)
    x, _y = win.canvas.quad_points()[0]
    assert abs(x - 20.1) < 1e-6
    win.close()
