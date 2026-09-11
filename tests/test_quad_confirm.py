"""四点校准的鼠标出口：画布底部确认条。

回归的是"点完四个角只能按回车"。两个毛病：快捷键没写在画布上，
用户不知道往哪点；而且回车要走 canvas.keyPressEvent，前提是画布持有
键盘焦点——中途点过右侧面板焦点就走了，回车按下去毫无反应。
"""

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")  # 没装 Qt 的机器上只跑 core 层
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog

import app.main_window as mw
from app.main_window import MainWindow
from core.document import A4_W_MM, Document, Page


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _AutoAcceptPreview:
    """替掉模态的拉正预览对话框：测试里没人点确认。"""

    def __init__(self, parent, warped_img):
        pass

    def exec(self):
        return QDialog.Accepted


def _make_window(qapp) -> MainWindow:
    win = MainWindow()
    page = Page(image=np.full((1754, 1240, 3), 250, np.uint8), phys_w_mm=200.0, phys_h_mm=280.0)
    win.doc = Document(pages=[page])
    win._rebuild_page_list(select=0)
    win.show()
    qapp.processEvents()
    return win


def test_bar_shows_only_during_calibration(qapp):
    win = _make_window(qapp)
    bar = win.canvas.quad_bar
    assert not bar.isVisible()
    win._four_point_calibrate()
    qapp.processEvents()
    assert bar.isVisible() and win.canvas.adjusting_quad()
    win.canvas.cancel_quad_adjust()
    assert not bar.isVisible()


def test_mouse_cancel_exits_calibration(qapp):
    win = _make_window(qapp)
    win._four_point_calibrate()
    QTest.mouseClick(win.canvas.quad_bar.cancel_btn, Qt.LeftButton)
    qapp.processEvents()
    assert not win.canvas.adjusting_quad()
    assert not win.canvas.quad_bar.isVisible()
    assert "取消" in win.info_label.text()


def test_mouse_confirm_calibrates_even_without_canvas_focus(qapp, monkeypatch):
    """核心回归：焦点不在画布上时回车是送不到的，鼠标这条路必须照样能走完。"""
    monkeypatch.setattr(mw, "WarpPreviewDialog", _AutoAcceptPreview)
    win = _make_window(qapp)
    win._four_point_calibrate()
    win.panel.btn_export.setFocus()  # 焦点离开画布
    qapp.processEvents()
    assert win.focusWidget() is not win.canvas

    QTest.mouseClick(win.canvas.quad_bar.accept_btn, Qt.LeftButton)
    qapp.processEvents()
    assert abs(win.doc.pages[0].phys_w_mm - A4_W_MM) < 1e-6
    assert not win.canvas.adjusting_quad()
    assert not win.canvas.quad_bar.isVisible()


def test_bar_buttons_do_not_steal_keyboard_focus(qapp):
    """点了确认条还要能接着用方向键微调把手，所以按钮必须 NoFocus。"""
    win = _make_window(qapp)
    bar = win.canvas.quad_bar
    assert bar.accept_btn.focusPolicy() == Qt.NoFocus
    assert bar.cancel_btn.focusPolicy() == Qt.NoFocus

    win._four_point_calibrate()
    win.canvas.setFocus()
    qapp.processEvents()
    QTest.mouseClick(bar.cancel_btn, Qt.LeftButton)
    qapp.processEvents()
    assert win.focusWidget() is win.canvas


def test_bar_stays_inside_viewport_after_resize(qapp):
    win = _make_window(qapp)
    win._four_point_calibrate()
    win.resize(900, 620)
    qapp.processEvents()
    bar, vp = win.canvas.quad_bar, win.canvas.viewport()
    assert bar.x() >= 0
    assert bar.geometry().bottom() <= vp.height()
    assert bar.y() >= 0


def test_hint_updates_do_not_resize_the_bar(qapp):
    """拖动中每帧都在刷提示。确认条宽度一变就左右抖，比不显示还难受。"""
    win = _make_window(qapp)
    win._four_point_calibrate()
    qapp.processEvents()
    bar = win.canvas.quad_bar
    before = bar.width()
    for hint in ("纸面比例 1.414　✓ 接近 A 系纸", "纸面比例 1.732　⚠ 偏离 A 系纸"):
        win.canvas.set_quad_hint(hint)
        qapp.processEvents()
        assert bar.width() == before
