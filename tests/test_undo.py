"""撤销栈测试（P1）：落章/删除/组删除/组微调/旋转/校准/移动。"""

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QListWidgetItem

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


def _make_window(qapp, n=3) -> MainWindow:
    win = MainWindow()
    win.doc = Document.from_images([f"{FIX}/{i}.jpg" for i in range(1, n + 1)])
    for i, page in enumerate(win.doc.pages):
        win.page_list.addItem(
            QListWidgetItem(np_rgb_to_qpixmap(_thumbnail(page.thumbnail(), 140)), f"第 {i + 1} 页")
        )
    win.page_list.setCurrentRow(0)
    win.show()
    return win


def _seal() -> Seal:
    return Seal(name="公章", kind="seal", image=extract_ink(f"{FIX}/seal_company.png"), phys_mm=40.0)


def test_undo_place(qapp):
    win = _make_window(qapp, n=1)
    win._add_stamp(_seal())
    win._on_stamp_placed(100.0, 200.0)
    assert len(win.stamps[0]) == 1
    win._undo()
    assert len(win.stamps.get(0, [])) == 0
    win.close()


def test_undo_delete(qapp):
    win = _make_window(qapp, n=1)
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).append(rec)
    win._on_page_changed(0)
    win.canvas.stamps()[0].setSelected(True)
    win._delete_selected()
    assert len(win.stamps[0]) == 0
    win._undo()
    assert len(win.stamps[0]) == 1
    assert win.stamps[0][0] is rec
    win.close()


def test_undo_group_delete(qapp):
    win = _make_window(qapp, n=3)
    seal = _seal()
    rng = Randomizer(11)
    processed, _ = rng.apply_auto(seal.image, win.panel.random_spec())
    pls = plan_perforation(processed, win.doc.pages, [0, 1, 2], PerforationSpec(seed=11))
    for pl in pls:
        page = win.doc.pages[pl.page_index]
        h_px, w_px = pl.slice_rgba.shape[:2]
        win.stamps.setdefault(pl.page_index, []).append(
            StampRecord(
                seal=seal,
                center_x_mm=pl.right_edge_mm - (w_px / page.dpi * 25.4) / 2,
                center_y_mm=pl.top_mm + (h_px / page.dpi * 25.4) / 2,
                size_mm=w_px / page.dpi * 25.4,
                processed=pl.slice_rgba, locked=True, group="g1",
            )
        )
    win._on_page_changed(0)
    win.canvas.stamps()[0].setSelected(True)
    win._delete_group()
    assert sum(len(v) for v in win.stamps.values()) == 0
    win._undo()
    assert sum(len(v) for v in win.stamps.values()) == 3
    win.close()


def test_undo_group_shift(qapp):
    win = _make_window(qapp, n=2)
    for i in range(2):
        rec = StampRecord(seal=_seal(), center_x_mm=200.0, center_y_mm=148.0,
                          size_mm=10.0, locked=True, group="g2")
        rec.processed = np.zeros((10, 10, 4), dtype=np.uint8)
        win.stamps.setdefault(i, []).append(rec)
    win._on_page_changed(0)
    win.canvas.stamps()[0].setSelected(True)
    win.group_shift_spin.setValue(5.0)
    win._apply_group_shift()
    assert all(r.center_y_mm == 153.0 for rs in win.stamps.values() for r in rs)
    win._undo()
    assert all(r.center_y_mm == 148.0 for rs in win.stamps.values() for r in rs)
    win.close()


def test_undo_rotate(qapp):
    win = _make_window(qapp, n=1)
    before = win.doc.pages[0].image.copy()
    win._rotate_page(2)  # 180°
    assert not np.array_equal(win.doc.pages[0].image, before)
    win._undo()  # 逆操作=再转 180° → 回到原图
    assert np.array_equal(win.doc.pages[0].image, before)
    win.close()


def test_undo_move(qapp):
    win = _make_window(qapp, n=1)
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).append(rec)
    win._on_page_changed(0)
    item = win.canvas.stamps()[0]
    win._on_canvas_clicked(item, 60.0, 90.0)
    assert rec.center_x_mm == 60.0
    win._undo()
    assert rec.center_x_mm == 100.0 and rec.center_y_mm == 200.0
    win.close()


def test_undo_stack_cap(qapp):
    win = _make_window(qapp, n=1)
    for _ in range(60):
        win._push_undo("x", lambda: None)
    assert len(win._undo_stack) == 50
    win.close()
