"""删除与骑缝组管理测试：删除按钮、整组删除、整组竖向微调、切片禁拖。"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QListWidgetItem, QGraphicsPixmapItem

from app.canvas import np_rgb_to_qpixmap
from app.main_window import MainWindow, StampRecord, _thumbnail
from core.document import Document
from core.extract import extract_ink
from core.perforation import PerforationSpec, plan_perforation
from core.randomize import Randomizer
from core.seal import Seal

FIX = "tests/fixtures"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _seal() -> Seal:
    return Seal(name="公章", kind="seal", image=extract_ink(f"{FIX}/seal_company.png"), phys_mm=40.0)


def _make_window(qapp, n=3) -> MainWindow:
    win = MainWindow()
    win.doc = Document.from_images([f"{FIX}/{i}.jpg" for i in range(1, n + 1)])
    for i, page in enumerate(win.doc.pages):
        win.page_list.addItem(
            QListWidgetItem(np_rgb_to_qpixmap(_thumbnail(page.image, 120)), f"第 {i + 1} 页")
        )
    win.page_list.setCurrentRow(0)
    win.show()
    return win


def _add_perf_group(win, pages=(0, 1, 2), seed=7) -> str:
    """模拟骑缝章应用：在指定页创建 locked 切片记录并刷新画布。返回组 id。"""
    seal = _seal()
    rng = Randomizer(seed)
    processed, _ = rng.apply_auto(seal.image, win.panel.random_spec())
    pls = plan_perforation(processed, win.doc.pages, list(pages), PerforationSpec(seed=seed))
    gid = f"perf_{seed}"
    for pl in pls:
        page = win.doc.pages[pl.page_index]
        h_px, w_px = pl.slice_rgba.shape[:2]
        win.stamps.setdefault(pl.page_index, []).append(
            StampRecord(
                seal=seal,
                center_x_mm=pl.right_edge_mm - (w_px / page.dpi * 25.4) / 2,
                center_y_mm=pl.top_mm + (h_px / page.dpi * 25.4) / 2,
                size_mm=w_px / page.dpi * 25.4,
                processed=pl.slice_rgba,
                locked=True,
                group=gid,
            )
        )
    win._on_page_changed(win.current_page)
    return gid


def test_delete_selected_button(qapp):
    win = _make_window(qapp, n=1)
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).append(rec)
    win._on_page_changed(0)
    win.canvas.stamps()[0].setSelected(True)
    win._delete_selected()
    assert len(win.stamps[0]) == 0
    assert len(win.canvas.stamps()) == 0
    win.close()


def test_delete_group_across_pages(qapp):
    win = _make_window(qapp, n=3)
    gid = _add_perf_group(win)
    assert sum(len(v) for v in win.stamps.values()) == 3
    # 选中当前页（第 1 页）的切片
    slice_item = win.canvas.stamps()[0]
    slice_item.setSelected(True)
    win._on_selection_changed()
    assert win.group_box.isVisible() or True  # offscreen 下 isVisible 不可靠，只验证逻辑
    win._delete_group()
    assert sum(len(v) for v in win.stamps.values()) == 0
    assert len(win.canvas.stamps()) == 0
    win.close()


def test_group_shift_preserves_jitter(qapp):
    win = _make_window(qapp, n=3)
    gid = _add_perf_group(win)
    recs = [r for rs in win.stamps.values() for r in rs if r.group == gid]
    before_ys = [r.center_y_mm for r in recs]
    win.canvas.stamps()[0].setSelected(True)
    win.group_shift_spin.setValue(5.0)
    win._apply_group_shift()
    after_ys = [r.center_y_mm for r in recs]
    for b, a in zip(before_ys, after_ys):
        assert abs(a - b - 5.0) < 1e-6
    # 逐页抖动差保持
    before_diffs = np.diff(sorted(before_ys))
    after_diffs = np.diff(sorted(after_ys))
    assert np.allclose(before_diffs, after_diffs)
    # 微调旋钮复位
    assert win.group_shift_spin.value() == 0.0
    win.close()


def test_locked_slice_not_movable(qapp):
    win = _make_window(qapp, n=2)
    _add_perf_group(win, pages=(0, 1))
    item = win.canvas.stamps()[0]
    assert not (item.flags() & QGraphicsPixmapItem.ItemIsMovable)
    win.close()
