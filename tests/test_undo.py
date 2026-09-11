"""撤销/重做命令栈测试：落章/删除/组删除/组微调/旋转/移动/删页/重盖。"""

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets")  # 没装 Qt 的机器上只跑 core 层
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow, StampRecord
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
    win._rebuild_page_list(select=0)
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
        with win._command("x"):
            pass
    assert len(win._undo_stack) == 50
    win.close()


# ── 重做 ──

def test_redo_replays_place(qapp):
    """撤销后重做：章回来，且位置一模一样。"""
    win = _make_window(qapp, n=1)
    win._add_stamp(_seal())
    win._on_stamp_placed(100.0, 200.0)
    win._undo()
    assert len(win.stamps.get(0, [])) == 0
    win._redo()
    assert len(win.stamps[0]) == 1
    assert win.stamps[0][0].center_x_mm == 100.0
    assert win.stamps[0][0].center_y_mm == 200.0
    win.close()


def test_redo_stack_cleared_by_new_command(qapp):
    """撤销后又做了新动作，重做栈必须作废（否则会重放出一个分叉的历史）。"""
    win = _make_window(qapp, n=1)
    win._add_stamp(_seal())
    win._on_stamp_placed(100.0, 200.0)
    win._undo()
    assert win._redo_stack
    win._add_stamp(_seal())
    win._on_stamp_placed(50.0, 60.0)
    assert win._redo_stack == []
    win.close()


def test_undo_redo_move_round_trip(qapp):
    win = _make_window(qapp, n=1)
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).append(rec)
    win._on_page_changed(0)
    win._on_canvas_clicked(win.canvas.stamps()[0], 60.0, 90.0)
    win._undo()
    assert (rec.center_x_mm, rec.center_y_mm) == (100.0, 200.0)
    win._redo()
    assert (rec.center_x_mm, rec.center_y_mm) == (60.0, 90.0)
    win.close()


def test_undo_page_delete_restores_pages_and_stamps(qapp, monkeypatch):
    """删页可撤销：页回来，页上的章也回来。"""
    from PySide6.QtWidgets import QMessageBox

    win = _make_window(qapp, n=3)
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(1, []).append(rec)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    win.page_list.setCurrentRow(1)
    win._delete_pages()
    assert len(win.doc.pages) == 2
    assert sum(len(v) for v in win.stamps.values()) == 0

    win._undo()
    assert len(win.doc.pages) == 3
    assert win.stamps[1][0] is rec
    win.close()


def test_delete_page_reindexes_later_stamps(qapp, monkeypatch):
    """删掉第 1 页后，原第 3 页的章要跟到新的第 2 页，不能留在旧下标上。"""
    from PySide6.QtWidgets import QMessageBox

    win = _make_window(qapp, n=3)
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=50.0, center_y_mm=50.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(2, []).append(rec)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)

    win.page_list.setCurrentRow(0)
    win._delete_pages()
    assert list(win.stamps.keys()) == [1]
    assert win.stamps[1][0] is rec
    win.close()


def test_restamp_removes_then_replaces(qapp):
    """重盖：旧的那枚先消失，落位后变成新的一枚；撤销能退回原来的位置。"""
    win = _make_window(qapp, n=1)
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).append(rec)
    win._on_page_changed(0)
    win.canvas.stamps()[0].setSelected(True)

    win._restamp_selected()
    assert win.stamps.get(0, []) == []      # 旧的已撤掉
    assert win.canvas.following             # 正在等用户点新位置
    win._on_stamp_placed(70.0, 80.0)
    assert len(win.stamps[0]) == 1
    assert win.stamps[0][0].center_x_mm == 70.0

    win._undo()   # 退回"刚删掉旧章"的状态
    win._undo()   # 再退回原来那一枚
    assert win.stamps[0][0] is rec
    assert rec.center_x_mm == 100.0
    win.close()


def test_command_rolls_back_on_exception(qapp):
    """动作中途抛异常：状态回滚，也不往撤销栈里塞半成品。"""
    win = _make_window(qapp, n=1)
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).append(rec)
    depth = len(win._undo_stack)

    with pytest.raises(RuntimeError):
        with win._command("会炸的动作"):
            rec.center_x_mm = 5.0
            raise RuntimeError("boom")

    assert rec.center_x_mm == 100.0
    assert len(win._undo_stack) == depth
    win.close()
