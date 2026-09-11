"""窄窗口下的控件尺寸：按钮文字不许被省略号吃掉。

回归的是"窗口没最大化时按钮显示成『↻ 重盖选…』"。根因不在文案长度，
而在 QPushButton 没有重写 minimumSizeHint——布局据此认为按钮可以缩到 0，
于是右侧面板一被挤窄，按钮先让位、文字先挨刀。
"""

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")  # 没装 Qt 的机器上只跑 core 层
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QSplitter, QWidget

from app.main_window import MainWindow
from app.widgets import ElidedLabel, FitButton, VScrollArea
from core.document import Document, Page


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_window(qapp) -> MainWindow:
    win = MainWindow()
    page = Page(image=np.full((1754, 1240, 3), 250, np.uint8), phys_w_mm=210.0, phys_h_mm=297.0)
    win.doc = Document(pages=[page])
    win._rebuild_page_list(select=0)
    win.show()
    qapp.processEvents()
    return win


def _too_narrow(win) -> list[str]:
    """返回"实际宽度不够放下完整文字"的按钮文案。"""
    return [
        b.text()
        for b in win.findChildren(FitButton)
        if b.isVisible() and b.width() < b.sizeHint().width()
    ]


def test_fit_button_reports_text_width_as_minimum(qapp):
    """根因测试：minimumSizeHint 必须等于 sizeHint，布局才不会把文字挤掉。"""
    btn = FitButton("导出已盖章 PDF…")
    assert btn.minimumSizeHint() == btn.sizeHint()
    assert btn.minimumSizeHint().width() > 0


def test_no_button_truncated_at_layout_minimum(qapp):
    """把窗口收到布局允许的最窄，所有可见按钮仍应放得下完整文字。"""
    win = _make_window(qapp)
    win.resize(win.minimumSizeHint())
    qapp.processEvents()
    assert not _too_narrow(win)


def test_splitter_cannot_squeeze_panel_below_its_content(qapp):
    """用户把分割条往右拽到底，右侧面板也不能窄过内容需要的宽度。"""
    win = _make_window(qapp)
    splitter = win.centralWidget()
    assert isinstance(splitter, QSplitter)
    splitter.setSizes([1, 10_000, 1])  # 极端：试图把两侧栏压没
    qapp.processEvents()
    right = splitter.widget(2)
    assert right.width() >= right.minimumSizeHint().width()
    assert not _too_narrow(win)


def test_scroll_area_propagates_content_width(qapp):
    """VScrollArea 存在的理由：普通 QScrollArea 会把内容的宽度需求吞掉。"""
    win = _make_window(qapp)
    scroll = win.findChild(VScrollArea)
    assert scroll is not None
    assert scroll.minimumSizeHint().width() >= scroll.widget().minimumSizeHint().width()
    # 横向滚动条是被禁掉的——靠它兜底就等于承认内容被挤窄了
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


# 侧栏宽度预算：它每胖一分，画布就瘦一分。1280 宽的窗口里超过这个数就该有人解释。
PANEL_WIDTH_BUDGET_PX = 480


def _widest(widget, n=6):
    """列出最宽的几个子控件，供预算超支时指认元凶。"""
    rows = []
    for w in widget.findChildren(QWidget):
        if not w.isVisible():
            continue
        text = getattr(w, "text", None)
        label = (text() if callable(text) else "")[:24]
        rows.append((w.minimumSizeHint().width(), type(w).__name__, label))
    rows.sort(reverse=True)
    return rows[:n]


def test_panel_width_stays_within_budget(qapp):
    """右侧面板不许无声变胖。

    历史教训：印章库路径是一条没有空格的 Windows 路径，`QLabel` 的 wordWrap
    断不开它，`minimumSizeHint` 直接等于整行宽度，把面板撑到 540px。
    这条断言在超支时报出最宽的几个控件，省得下一个人再猜一遍。
    """
    win = _make_window(qapp)
    right = win.centralWidget().widget(2)
    need = right.minimumSizeHint().width()
    assert need <= PANEL_WIDTH_BUDGET_PX, (
        f"右侧面板需要 {need}px，预算 {PANEL_WIDTH_BUDGET_PX}px；最宽的控件：{_widest(right)}"
    )


def test_library_path_label_does_not_stretch_the_panel(qapp):
    """路径标签必须是可省略的，且最小宽度与路径长度无关。"""
    win = _make_window(qapp)
    label = win.panel.lib_path_label
    assert isinstance(label, ElidedLabel)
    short = label.minimumSizeHint().width()
    label.setText("库位置：" + "C:\\Users\\someone\\AppData\\Roaming\\contract-sealer\\seals" * 3)
    qapp.processEvents()
    assert label.minimumSizeHint().width() == short
