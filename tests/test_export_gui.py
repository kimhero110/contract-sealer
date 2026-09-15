"""导出走工作线程：主线程不冻结、能取消、结果与同步导出一致；导出设置对话框。"""

import json

import numpy as np
import pytest

pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)  # 没装 Qt / 缺 libEGL 的机器上只跑 core 层
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QProgressDialog, QPushButton

from app.dialogs import ExportOptionsDialog
from app.main_window import MainWindow
from app.worker import run_with_progress
from core.document import Document
from core.export import ExportOptions
from core.extract import extract_ink
from core.seal import Seal
from core.session import ExportCancelled

FIX = "tests/fixtures"


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_window(qapp, n=2) -> MainWindow:
    win = MainWindow()
    win.doc = Document.from_images([f"{FIX}/{i}.jpg" for i in range(1, n + 1)])
    win._rebuild_page_list(select=0)
    win.show()
    return win


def _seal() -> Seal:
    return Seal(name="公章", kind="seal", image=extract_ink(f"{FIX}/seal_company.png"), phys_mm=40.0)


def test_export_runs_off_main_thread_and_keeps_ui_alive(qapp, tmp_path):
    """导出期间主线程仍在跑事件循环：一个 QTimer 能在导出结束前多次触发。"""
    win = _make_window(qapp)
    win._add_stamp(_seal())
    win._on_stamp_placed(100.0, 200.0)
    ticks: list[int] = []
    timer = QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(5)

    out = win._export_document(tmp_path / "threaded.pdf")
    timer.stop()
    assert out.exists() and out.with_suffix(".sealog").exists()
    assert len(ticks) >= 2, "导出期间主线程没有处理事件——界面会冻结"
    log = json.loads(out.with_suffix(".sealog").read_text(encoding="utf-8"))
    assert log["stamps"][0]["seal"] == "公章"
    # 导出完成后主窗口状态原封不动
    assert len(win.stamps[0]) == 1 and win.current_page == 0
    win.close()


def test_export_cancel_leaves_no_file(qapp, tmp_path, monkeypatch):
    """进度框上点「取消」：导出中止，目标文件、sealog、临时文件一个都不留。"""
    import time

    from core.session import Session

    win = _make_window(qapp, n=3)
    real_render = Session.render_page

    def slow_render(self, page_index):
        time.sleep(0.2)  # 给主线程留出点"取消"的时间窗，否则 3 张空白页瞬间就导完了
        return real_render(self, page_index)

    monkeypatch.setattr(Session, "render_page", slow_render)

    def press_cancel() -> None:
        # 点真实的「取消」按钮：QProgressDialog.cancel() 只重置状态，不发 canceled 信号
        for w in qapp.topLevelWidgets():
            if isinstance(w, QProgressDialog) and w.isVisible():
                w.findChild(QPushButton).click()

    QTimer.singleShot(0, press_cancel)
    with pytest.raises(ExportCancelled):
        win._export_document(tmp_path / "cancelled.pdf")
    assert not (tmp_path / "cancelled.pdf").exists()
    assert not list(tmp_path.glob("*.tmp*"))
    win.close()


def test_run_with_progress_reraises_worker_exception(qapp):
    def fail(progress, cancelled):
        raise RuntimeError("工作线程里炸了")

    with pytest.raises(RuntimeError, match="工作线程里炸了"):
        run_with_progress(None, "测试", fail)


def test_export_uses_settings_options(qapp, tmp_path, monkeypatch):
    win = _make_window(qapp, n=1)
    win.settings.export_format = "png"
    out = win._export_document(tmp_path / "png.pdf")
    log = json.loads(out.with_suffix(".sealog").read_text(encoding="utf-8"))
    assert log["export_options"]["image_format"] == "png"
    win.close()


def test_export_options_dialog_roundtrip(qapp):
    dlg = ExportOptionsDialog(None, ExportOptions("png", 80))
    assert dlg.png_radio.isChecked() and not dlg.quality_spin.isEnabled()
    dlg.jpeg_radio.setChecked(True)
    dlg.quality_spin.setValue(70)
    assert dlg.values() == ExportOptions("jpeg", 70)
    dlg.png_radio.setChecked(True)
    assert dlg.values().image_format == "png"


def test_snapshot_uses_page_state_api(qapp):
    """撤销快照通过 Page.state()/restore() 走，不碰私有字段。"""
    win = _make_window(qapp, n=1)
    page = win.doc.pages[0]
    before = page.image.copy()
    win._rotate_page(2)
    assert not np.array_equal(page.image, before)
    snap = win._undo_stack[-1][1]
    assert snap.page_state[0].revision == 0 and snap.page_state[0].override is None
    win._undo()
    assert np.array_equal(page.image, before) and page.revision == 0
    win.close()
