"""放大镜（精确点选取景器）测试。"""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QListWidgetItem

from app.canvas import np_rgb_to_qpixmap
from app.main_window import MainWindow, _thumbnail
from core.document import Document, Page
from core.seal import Seal
from core.extract import extract_ink

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
    win.page_list.addItem(QListWidgetItem(np_rgb_to_qpixmap(_thumbnail(page.thumbnail(), 140)), "第 1 页"))
    win.page_list.setCurrentRow(0)
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


def test_magnifier_visibility_in_modes(qapp):
    """拾取模式悬停显示放大镜；退出模式隐藏。"""
    win = _make_window(qapp)
    win.canvas.start_pick_points()
    target = win.canvas.mapFromScene(win.canvas.mapToScene(QPoint(200, 200)))
    QTest.mouseMove(win.canvas.viewport(), target)
    qapp.processEvents()
    assert win.canvas._magnifier.isVisible()
    # 点满 4 点（拾取自动结束）
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, target)
    # 退出后隐藏
    win.canvas.cancel_pick()
    qapp.processEvents()
    assert not win.canvas._magnifier.isVisible()
    win.close()
