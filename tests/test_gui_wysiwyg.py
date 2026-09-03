"""GUI 所见即所得测试：真实事件路径（QTest 注入，不走槽函数捷径）。

覆盖：
- P0 回归：骑缝/模板新增记录在页面刷新时不被误删（"第 3 页骑缝消失" bug）；
- 显式删除是唯一记录删除入口；
- 跟随落章真实点击、单击移章、落位后紧邻点击不移动；
- multiply 渲染可视性（视口抓图像素断言）。
"""

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
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


def _make_window(qapp, pages=("1.jpg", "2.jpg", "3.jpg")) -> MainWindow:
    win = MainWindow()
    win.doc = Document.from_images([f"{FIX}/{p}" for p in pages])
    for i, page in enumerate(win.doc.pages):
        win.page_list.addItem(
            QListWidgetItem(np_rgb_to_qpixmap(_thumbnail(page.image, 120)), f"第 {i + 1} 页")
        )
    win.page_list.setCurrentRow(0)
    win.show()
    return win


def _seal() -> Seal:
    return Seal(name="公章", kind="seal", image=extract_ink(f"{FIX}/seal_company.png"), phys_mm=40.0)


def _view_pos(canvas, x_mm: float, y_mm: float) -> QPoint:
    return canvas.mapFromScene(QPointF(x_mm, y_mm))


# ── P0 回归：记录误删 bug ──

def test_perforation_records_survive_refresh_on_current_page(qapp):
    """"第 3 页骑缝消失"复现路径：用户停在第 3 页应用骑缝，切片记录必须存活。"""
    win = _make_window(qapp)
    win.page_list.setCurrentRow(2)  # 复现现场：用户停在第 3 页
    seal = _seal()
    rng = Randomizer(999)
    processed_seal, _ = rng.apply_auto(seal.image, win.panel.random_spec())
    pls = plan_perforation(processed_seal, win.doc.pages, [0, 1, 2], PerforationSpec(seed=999))
    touched = set()
    for pl in pls:
        page = win.doc.pages[pl.page_index]
        h_px, w_px = pl.slice_rgba.shape[:2]
        rec = StampRecord(
            seal=seal,
            center_x_mm=pl.right_edge_mm - (w_px / page.dpi * 25.4) / 2,
            center_y_mm=pl.top_mm + (h_px / page.dpi * 25.4) / 2,
            size_mm=w_px / page.dpi * 25.4,
            processed=pl.slice_rgba,
            locked=True,
            group="perf_999",
        )
        win.stamps.setdefault(pl.page_index, []).append(rec)
        touched.add(pl.page_index)
    win._on_page_changed(win.current_page)  # _add_perforation 的收尾刷新
    assert {k: len(v) for k, v in win.stamps.items()} == {0: 1, 1: 1, 2: 1}
    win.close()


def test_template_records_survive_refresh(qapp):
    """模板套用到当前页：新增记录同样不得被刷新剪掉。"""
    win = _make_window(qapp, pages=("1.jpg",))
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, rec.applied = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).append(rec)
    win._on_page_changed(0)  # _apply_template 的收尾刷新
    assert len(win.stamps[0]) == 1
    win.close()


def test_explicit_delete_is_the_only_removal(qapp):
    """只有画布 Delete 才删记录；翻页来回不丢记录。"""
    win = _make_window(qapp, pages=("1.jpg", "2.jpg"))
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).append(rec)
    win._on_page_changed(0)
    # 翻页来回
    win.page_list.setCurrentRow(1)
    win.page_list.setCurrentRow(0)
    assert len(win.stamps[0]) == 1
    # 显式删除
    win._on_stamps_deleted(win.canvas.stamps())
    assert len(win.stamps[0]) == 0
    win.close()


# ── 真实事件路径 ──

def test_follow_place_by_real_click(qapp):
    """QTest 真实点击：跟随落章，记录中心 = 点击坐标，视口可见墨迹。"""
    win = _make_window(qapp, pages=("1.jpg",))
    win._add_stamp(_seal())
    assert win.canvas.following
    target = _view_pos(win.canvas, 100.0, 200.0)
    QTest.mouseMove(win.canvas.viewport(), target)
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, target)
    assert len(win.stamps[0]) == 1
    rec = win.stamps[0][0]
    assert abs(rec.center_x_mm - 100.0) < 2.0  # fit 换算容差
    assert abs(rec.center_y_mm - 200.0) < 2.0
    # 视口抓图：章中心附近应有墨迹（multiply 渲染可视性）
    qapp.processEvents()
    grab = win.canvas.viewport().grab().toImage().convertToFormat(QImage.Format_RGB888)
    px = _view_pos(win.canvas, rec.center_x_mm, rec.center_y_mm)
    region = grab.copy(max(0, px.x() - 20), max(0, px.y() - 20), 40, 40)
    img = np.frombuffer(region.bits(), dtype=np.uint8).reshape(region.height(), region.width(), 3)
    red_pixels = int(((img[:, :, 0].astype(int) - img[:, :, 1].astype(int) > 20)).sum())
    assert red_pixels > 20, "画布上看不到章的墨迹"
    win.close()


def test_click_to_move_by_real_events(qapp):
    """QTest 真实事件：选中章后空白处单击，章移到点击处（Bug 1 时序修复）。"""
    win = _make_window(qapp, pages=("1.jpg",))
    seal = _seal()
    rec = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=200.0, size_mm=40.0)
    rec.processed, _ = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).append(rec)
    win._on_page_changed(0)
    # 先点中章（选中）
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, _view_pos(win.canvas, 100.0, 200.0))
    # 再点空白处（移章）
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, _view_pos(win.canvas, 60.0, 90.0))
    assert abs(rec.center_x_mm - 60.0) < 2.0
    assert abs(rec.center_y_mm - 90.0) < 2.0
    win.close()


def test_just_placed_click_is_consumed(qapp):
    """落位后紧邻的一次空白点击不得移动刚放下的章（防双击挪偏）。"""
    win = _make_window(qapp, pages=("1.jpg",))
    win._add_stamp(_seal())
    target = _view_pos(win.canvas, 100.0, 200.0)
    QTest.mouseMove(win.canvas.viewport(), target)
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, target)
    rec = win.stamps[0][0]
    cx, cy = rec.center_x_mm, rec.center_y_mm
    # 紧邻再点一次别处
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, _view_pos(win.canvas, 50.0, 60.0))
    assert abs(rec.center_x_mm - cx) < 1e-6
    assert abs(rec.center_y_mm - cy) < 1e-6
    win.close()


def test_place_outside_page_is_rejected(qapp):
    """页面外点击被护栏拦截，不产生记录。"""
    win = _make_window(qapp, pages=("1.jpg",))
    win._add_stamp(_seal())
    QTest.mouseClick(win.canvas.viewport(), Qt.LeftButton, Qt.NoModifier, _view_pos(win.canvas, -30.0, -30.0))
    assert len(win.stamps.get(0, [])) == 0
    assert win.canvas.following  # 跟随模式仍在，等用户点有效位置
    win.close()


# ── 换一批手感 ──

def test_reroll_changes_normal_keeps_locked(qapp):
    win = _make_window(qapp, pages=("1.jpg",))
    seal = _seal()
    normal = StampRecord(seal=seal, center_x_mm=100.0, center_y_mm=100.0, size_mm=40.0)
    locked = StampRecord(seal=seal, center_x_mm=200.0, center_y_mm=100.0, size_mm=10.0, locked=True, group="g1")
    for r in (normal, locked):
        r.processed, r.applied = win._session_rng.apply_auto(seal.image, win.panel.random_spec())
    win.stamps.setdefault(0, []).extend([normal, locked])
    before_applied = normal.applied
    before_normal = normal.processed.copy()
    before_locked = locked.processed.copy()
    win._reroll_random()
    assert normal.applied != before_applied
    assert not np.array_equal(normal.processed, before_normal)
    assert np.array_equal(locked.processed, before_locked)
    win.close()
