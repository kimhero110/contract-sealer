"""主窗口：左侧页缩略图 / 中间画布 / 右侧印章与参数面板。

盖章会话状态模型：
- Document（core）提供页面图像与物理尺寸；
- 每页若干 StampRecord（印章 + 物理位置/尺寸/旋转/不透明度 + 已采样随机效果）；
- 预览与导出共用同一份 processed 图像——所见即所得（方案 §4.9）；
- 骑缝章切片是 locked 记录：随机在生成时已定，导出不再重采样（方案 §4.4）。

撤销/重做：快照式命令栈（见 _command）。每个会改状态的动作把"改之前"
与"改之后"两份快照压栈，撤销与重做就是把对应快照写回去。
（旧实现是逐个动作手写逆操作闭包：漏一个字段就是一个静默 bug，而且
天生做不了重做。四点校准那条逆操作还引用了尚未赋值的局部变量。）
"""

from __future__ import annotations

import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.canvas import PageCanvas, StampItem, np_rgb_to_qpixmap
from app.dialogs import AboutDialog, CalibrateDialog, DateStampDialog, WarpPreviewDialog
from app.imageutil import display_image, natural_key, thumbnail
from app.perforation_dialog import PerforationDialog
from app.seal_panel import SealPanel
from core.autocal import (
    auto_calibrate_page,
    initial_quad,
    map_points_through,
    quad_aspect,
    warp_to_a4,
)
from core.datestamp import KIND_DATE, ink_width_mm, render_date_ink
from core.document import (
    ASPECT_A_SERIES,
    ASPECT_TOLERANCE,
    Document,
    Page,
    calibrate_paper_edge,
)
from core.export import export_pdf, make_output_path
from core.perforation import PerforationSpec, SlicePlacement, plan_perforation
from core.randomize import AppliedRandom, Randomizer
from core.seal import KIND_SEAL, Seal, default_library_dir, list_library
from core.settings import Settings
from core.stamp import Placement, stamp_page
from core.template import (
    default_template_dir,
    list_templates,
    load_template,
    save_template,
)

MM_PER_INCH = 25.4
UNDO_LIMIT = 50

# 撤销快照要还原的记录字段。新增可变字段必须同步加进来，否则撤销会漏改。
_RECORD_FIELDS = (
    "center_x_mm",
    "center_y_mm",
    "size_mm",
    "rotation_deg",
    "opacity",
    "applied",
    "processed",
    "locked",
    "group",
)


@dataclass
class StampRecord:
    seal: Seal
    center_x_mm: float
    center_y_mm: float
    size_mm: float
    rotation_deg: float = 0.0
    opacity: float = 1.0
    applied: AppliedRandom | None = None
    processed: np.ndarray | None = None  # 随机效果后的 RGBA，预览/导出共用
    locked: bool = False                 # 骑缝章切片：导出不重采样
    group: str | None = None             # 骑缝章分组标识（sealog 用）


@dataclass
class _Snapshot:
    """一份可完整还原的会话状态。页与记录都存对象引用 + 字段值，不深拷像素。"""

    pages: list[Page]
    # (变异图像, 图像版本号, 物理宽, 物理高, 待校准标志)
    page_state: list[tuple[np.ndarray | None, int, float, float, bool]]
    stamps: dict[int, list[tuple[StampRecord, tuple]]]
    current: int


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("合同盖章工具")
        self.resize(1280, 860)

        self.doc: Document | None = None
        self.stamps: dict[int, list[StampRecord]] = {}
        self.current_page = -1
        self.settings = Settings.load()
        self._syncing = False
        self._pending_rec: StampRecord | None = None
        self._undo_stack: list[tuple[str, _Snapshot, _Snapshot]] = []
        self._redo_stack: list[tuple[str, _Snapshot, _Snapshot]] = []
        # 页缩略图缓存：键含"当前变异图像"的身份，旋转/校准后自动失效
        self._thumb_cache: dict[tuple[int, int], object] = {}
        # 会话随机器：落章即采样，预览立即带真实感
        self._session_rng = Randomizer(secrets.randbelow(2**31 - 1))

        self._build_ui()
        self._build_menu()
        self._update_info(None)
        self._update_history_actions()

    # ── UI 搭建 ──

    def _build_ui(self) -> None:
        splitter = QSplitter()

        self.page_list = QListWidget()
        self.page_list.setMaximumWidth(180)
        # 多选：一次删掉整个导入文件的所有页
        self.page_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.page_list.currentRowChanged.connect(self._on_page_changed)
        self.page_list.setContextMenuPolicy(Qt.ActionsContextMenu)
        splitter.addWidget(self.page_list)

        self.canvas = PageCanvas()
        self.canvas.stamp_moved.connect(self._on_stamp_moved)
        self.canvas.stamps_deleted.connect(self._on_stamps_deleted)
        self.canvas.stamp_placed.connect(self._on_stamp_placed)
        self.canvas.place_rejected.connect(self._on_place_rejected)
        self.canvas.follow_cancelled.connect(self._on_follow_cancelled)
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.quad_changed.connect(self._on_quad_changed)
        self.canvas.quad_accepted.connect(self._on_quad_accepted)
        self.canvas.quad_cancelled.connect(self._on_quad_cancelled)
        self.canvas.scene().selectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.canvas)

        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self._build_toolbar()

    def _build_right_panel(self) -> QWidget:
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QGraphicsDropShadowEffect

        right = QWidget()
        right_layout = QHBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)  # 卡片四周留白
        panel_container = QWidget()
        # 柔和投影：卡片浮起来的现代感
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 3)
        shadow.setColor(QColor(0, 0, 0, 28))
        panel_container.setGraphicsEffect(shadow)
        v = QVBoxLayout(panel_container)
        v.setContentsMargins(12, 10, 12, 12)
        v.setSpacing(8)

        self.panel = SealPanel()
        self.panel.stamp_requested.connect(self._add_stamp)
        self.panel.perforation_requested.connect(self._add_perforation)
        self.panel.reroll_requested.connect(self._reroll_random)
        self.panel.date_requested.connect(self._add_date_stamp)
        self.panel.export_requested.connect(self._export)
        v.addWidget(self.panel)

        # 选中印章的微调控件（位置用鼠标点，不用输入框——修改意见）
        adj = QFormLayout()
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(3.0, 300.0)
        self.size_spin.setSuffix(" mm")
        self.size_spin.valueChanged.connect(self._apply_adjustments)
        self.rot_spin = QDoubleSpinBox()
        self.rot_spin.setRange(-180.0, 180.0)
        self.rot_spin.setSuffix(" °")
        self.rot_spin.valueChanged.connect(self._apply_adjustments)
        self.opa_spin = QDoubleSpinBox()
        self.opa_spin.setRange(0.05, 1.0)
        self.opa_spin.setSingleStep(0.05)
        self.opa_spin.setValue(1.0)
        self.opa_spin.valueChanged.connect(self._apply_adjustments)
        adj.addRow("尺寸", self.size_spin)
        adj.addRow("旋转", self.rot_spin)
        adj.addRow("不透明度", self.opa_spin)
        v.addLayout(adj)

        row = QHBoxLayout()
        self.btn_restamp = QPushButton("↻ 重盖选中章")
        self.btn_restamp.setToolTip("撤掉选中的章，换一次手感重新点位置盖（Ctrl+R）")
        self.btn_restamp.clicked.connect(self._restamp_selected)
        row.addWidget(self.btn_restamp)
        self.btn_delete = QPushButton("🗑 删除选中章")
        self.btn_delete.setToolTip("删除选中的章/签名（快捷键 Delete）")
        self.btn_delete.clicked.connect(self._delete_selected)
        row.addWidget(self.btn_delete)
        v.addLayout(row)

        # 骑缝组管理（选中骑缝切片时可见）
        self.group_box = QWidget()
        gb = QVBoxLayout(self.group_box)
        gb.setContentsMargins(0, 0, 0, 0)
        gb.addWidget(QLabel("骑缝章整组操作："))
        self.group_shift_spin = QDoubleSpinBox()
        self.group_shift_spin.setRange(-100.0, 100.0)
        self.group_shift_spin.setSingleStep(0.5)
        self.group_shift_spin.setSuffix(" mm（正值向下）")
        self.group_shift_spin.setValue(0.0)
        gb.addWidget(self.group_shift_spin)
        btn_shift = QPushButton("竖向整体微调")
        btn_shift.clicked.connect(self._apply_group_shift)
        gb.addWidget(btn_shift)
        self.btn_delete_group = QPushButton("删除整组骑缝章")
        self.btn_delete_group.clicked.connect(self._delete_group)
        gb.addWidget(self.btn_delete_group)
        self.group_box.setVisible(False)
        v.addWidget(self.group_box)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)

        right_layout.addWidget(panel_container)
        right.setMaximumWidth(360)
        return right

    def _build_toolbar(self) -> None:
        """工具栏：页面旋转 + 页序调整——短文案+悬浮提示，防挤换行。"""
        from PySide6.QtWidgets import QToolBar

        tb = QToolBar("页面操作")
        tb.setMovable(False)
        for label, tip, fn in [
            ("↻ 90°", "当前页顺时针旋转 90°", lambda: self._rotate_page(-1)),
            ("↺ 90°", "当前页逆时针旋转 90°", lambda: self._rotate_page(1)),
            ("180°", "当前页旋转 180°", lambda: self._rotate_page(2)),
            ("↑ 上移", "当前页上移一位", lambda: self._move_page(-1)),
            ("↓ 下移", "当前页下移一位", lambda: self._move_page(1)),
            ("✖ 删除页", "删除页列表里选中的页（可撤销）", self._delete_pages),
            ("▣ 四点校准", "拖四个把手贴住纸角，透视拉正为标准 A4（自动检测给初值）", self._four_point_calibrate),
            ("⛶ 适应", "视图缩放至整页（快捷键 F）", self.canvas.fit_page),
        ]:
            act = tb.addAction(label, fn)
            act.setToolTip(tip)
        self.addToolBar(tb)

    def _build_menu(self) -> None:
        # ── 文件：打开/追加/关闭/导出 ──
        m_file = self.menuBar().addMenu("文件")
        self._add_action(m_file, "打开合同…", self._open_files, "Ctrl+O")
        self._add_action(m_file, "追加导入文件…", self._append_files, "Ctrl+Shift+O")
        self._add_action(m_file, "关闭当前文档", self._close_document, "Ctrl+W")
        m_file.addSeparator()
        self._add_action(m_file, "批量导出…", self._batch_export)
        self._add_action(m_file, "导出 PDF…", self._export, "Ctrl+E")
        m_file.addSeparator()
        self._add_action(m_file, "设置输出目录…", self._choose_output_dir)
        self.out_dir_act = self._add_action(
            m_file, "输出目录：与源文件同目录", self._reset_output_dir
        )
        self._refresh_output_dir_action()

        # ── 编辑：撤销/重做/重盖 ──
        m_edit = self.menuBar().addMenu("编辑")
        self.undo_act = self._add_action(m_edit, "撤销", self._undo, "Ctrl+Z")
        self.redo_act = self._add_action(m_edit, "重做", self._redo, "Ctrl+Y")
        self.redo_act.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z"])
        m_edit.addSeparator()
        self._add_action(m_edit, "重盖选中章", self._restamp_selected, "Ctrl+R")
        self._add_action(m_edit, "删除选中章", self._delete_selected)

        # ── 页面：旋转/页序/删除/校准 ──
        m_page = self.menuBar().addMenu("页面")
        m_page.addAction("顺时针旋转 90°", lambda: self._rotate_page(-1))
        m_page.addAction("逆时针旋转 90°", lambda: self._rotate_page(1))
        m_page.addAction("旋转 180°", lambda: self._rotate_page(2))
        m_page.addSeparator()
        m_page.addAction("页面上移", lambda: self._move_page(-1))
        m_page.addAction("页面下移", lambda: self._move_page(1))
        del_act = self._add_action(m_page, "删除选中页", self._delete_pages)
        # 页列表右键同款删除（ActionsContextMenu 直接复用 QAction）
        self.page_list.addAction(del_act)
        m_page.addSeparator()
        m_page.addAction("四点纸边校准（拖四个角）", self._four_point_calibrate)
        self._add_action(m_page, "手动校准（输入尺寸）…", self._calibrate)
        m_page.addAction("自动纸边检测（全部页）", self._auto_calibrate)

        # ── 插入 ──
        m_insert = self.menuBar().addMenu("插入")
        self._add_action(m_insert, "加盖日期…", self._add_date_stamp, "Ctrl+D")

        # ── 视图 ──
        m_view = self.menuBar().addMenu("视图")
        self._add_action(m_view, "适应页面", self.canvas.fit_page, "F")
        self._add_action(m_view, "放大", lambda: self.canvas.scale(1.25, 1.25), "Ctrl+=")
        self._add_action(m_view, "缩小", lambda: self.canvas.scale(0.8, 0.8), "Ctrl+-")

        # ── 模板（含动态列表）──
        self.tpl_menu = self.menuBar().addMenu("模板")
        self._add_action(self.tpl_menu, "把当前页存为模板…", self._save_template)
        self.tpl_menu.addSeparator()
        self.tpl_menu.aboutToShow.connect(self._refresh_template_menu)

        # ── 帮助（永远最后）──
        m_help = self.menuBar().addMenu("帮助")
        self._add_action(m_help, "关于…", self._show_about)

    def _add_action(self, menu, text: str, slot, shortcut: str | None = None) -> QAction:
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(shortcut)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    # ── 撤销 / 重做 ──

    def _snapshot(self) -> _Snapshot:
        pages = list(self.doc.pages) if self.doc is not None else []
        return _Snapshot(
            pages=pages,
            page_state=[
                (p._override, p.revision, p.phys_w_mm, p.phys_h_mm, p.needs_calibration)
                for p in pages
            ],
            stamps={
                i: [(r, tuple(getattr(r, f) for f in _RECORD_FIELDS)) for r in recs]
                for i, recs in self.stamps.items()
            },
            current=self.current_page,
        )

    def _restore(self, snap: _Snapshot) -> None:
        if self.doc is None:
            return
        self.doc.pages = list(snap.pages)
        for page, (override, revision, w, h, flag) in zip(
            snap.pages, snap.page_state, strict=True
        ):
            page._override = override
            page.revision = revision  # 缩略图缓存跟着回退，否则撤销后还显示旧图
            page.phys_w_mm, page.phys_h_mm = w, h
            page.needs_calibration = flag
        self.stamps = {}
        for index, entries in snap.stamps.items():
            recs = []
            for rec, values in entries:
                for field, value in zip(_RECORD_FIELDS, values, strict=True):
                    setattr(rec, field, value)
                recs.append(rec)
            if recs:
                self.stamps[index] = recs
        self._rebuild_page_list(select=snap.current)

    @contextmanager
    def _command(self, label: str):
        """把一次状态变更包成可撤销/可重做的命令。

        用法：with self._command("盖章"): ...改状态...
        没打开文档时退化为直通，不压栈。
        """
        if self.doc is None:
            yield
            return
        before = self._snapshot()
        try:
            yield
        except Exception:
            # 全有或全无：动作中途失败就回滚，绝不把半改完的状态留给用户
            self._restore(before)
            raise
        after = self._snapshot()
        self._undo_stack.append((label, before, after))
        del self._undo_stack[:-UNDO_LIMIT]
        self._redo_stack.clear()
        self._update_history_actions()

    def _undo(self) -> None:
        if not self._undo_stack:
            self.info_label.setText("没有可撤销的操作")
            return
        label, before, after = self._undo_stack.pop()
        self._restore(before)
        self._redo_stack.append((label, before, after))
        self._update_history_actions()
        self.info_label.setText(f"已撤销：{label}")

    def _redo(self) -> None:
        if not self._redo_stack:
            self.info_label.setText("没有可重做的操作")
            return
        label, before, after = self._redo_stack.pop()
        self._restore(after)
        self._undo_stack.append((label, before, after))
        self._update_history_actions()
        self.info_label.setText(f"已重做：{label}")

    def _update_history_actions(self) -> None:
        self.undo_act.setEnabled(bool(self._undo_stack))
        self.undo_act.setText(
            f"撤销：{self._undo_stack[-1][0]}" if self._undo_stack else "撤销"
        )
        self.redo_act.setEnabled(bool(self._redo_stack))
        self.redo_act.setText(
            f"重做：{self._redo_stack[-1][0]}" if self._redo_stack else "重做"
        )

    def _clear_history(self) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_history_actions()

    # ── 文件打开与文档管理 ──

    _FILE_FILTER = "合同文件 (*.pdf *.jpg *.jpeg *.png *.bmp *.tif *.tiff)"

    def _open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "打开合同（PDF 或图片，可多选图片）", "", self._FILE_FILTER
        )
        if not paths:
            return
        try:
            self._load_document(paths)
        except Exception as e:
            QMessageBox.warning(self, "打开失败", str(e))

    def _append_files(self) -> None:
        """追加导入：把更多文件的页接到当前文档末尾（可撤销）。"""
        if self.doc is None:
            self._open_files()
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "追加导入（PDF 或图片，可多选）", "", self._FILE_FILTER
        )
        if not paths:
            return
        try:
            extra = self._build_document(paths)
        except Exception as e:
            QMessageBox.warning(self, "导入失败", str(e))
            return
        added = len(extra.pages)
        with self._command(f"追加导入 {added} 页"):
            self.doc.extend(extra)
            self._rebuild_page_list(select=len(self.doc.pages) - added)
        self.info_label.setText(f"已追加 {added} 页，可用「页面 → 删除选中页」移除")

    def _close_document(self) -> None:
        if self.doc is None:
            return
        if QMessageBox.question(
            self, "关闭文档", "关闭当前文档？未导出的盖章会丢失。"
        ) != QMessageBox.Yes:
            return
        self.doc.close()
        self.doc = None
        self.stamps = {}
        self.current_page = -1
        self._pending_rec = None
        self._clear_history()
        self.page_list.clear()
        self.canvas.clear_page()
        self._update_info(None)

    @staticmethod
    def _build_document(paths: list[str]) -> Document:
        if len(paths) == 1 and paths[0].lower().endswith(".pdf"):
            return Document.open(paths[0])
        if any(p.lower().endswith(".pdf") for p in paths):
            raise ValueError("PDF 请一次只选一个（图片可以多选）")
        return Document.from_images(sorted(paths, key=natural_key))

    def _load_document(self, paths: list[str]) -> None:
        doc = self._build_document(paths)
        if self.doc is not None:
            self.doc.close()  # 释放上一个文档的 PDF 句柄与页面缓存
        self.doc = doc
        self.stamps = {}
        self.current_page = -1
        self._pending_rec = None
        self._clear_history()
        self._rebuild_page_list(select=0 if doc.pages else -1)

        # 比例检测：有页面偏离 A 系纸比例时提示校准（方案 v1.3 §4.3）
        flagged = [i + 1 for i, p in enumerate(doc.pages) if p.needs_calibration]
        if flagged:
            ret = QMessageBox.question(
                self,
                "需要纸边校准",
                f"第 {flagged} 页的长宽比偏离标准 A 系纸，按 A4 假定盖章会尺寸失真。\n"
                "是否现在校准（输入纸张真实尺寸）？\n"
                "（也可以稍后用「页面 → 自动纸边检测」）",
            )
            if ret == QMessageBox.Yes:
                self._calibrate()

    def _rebuild_page_list(self, select: int = -1) -> None:
        """按当前 doc.pages 重建页列表并选中指定页。页增删/撤销后统一走这里。"""
        if self.doc is None:
            self.page_list.clear()
            return
        multi_source = len({p.source_name for p in self.doc.pages if p.source_name}) > 1
        self.page_list.blockSignals(True)
        self.page_list.clear()
        for i, page in enumerate(self.doc.pages):
            icon = self._page_icon(page)
            label = f"第 {i + 1} 页"
            if multi_source and page.source_name:
                label += f"\n{page.source_name}"
            self.page_list.addItem(QListWidgetItem(icon, label))
        self.page_list.blockSignals(False)
        self.current_page = -1  # 强制重建画布
        if 0 <= select < len(self.doc.pages):
            self.page_list.setCurrentRow(select)
        elif self.doc.pages:
            self.page_list.setCurrentRow(min(max(select, 0), len(self.doc.pages) - 1))
        else:
            self.canvas.clear_page()

    def _selected_page_rows(self) -> list[int]:
        rows = sorted(idx.row() for idx in self.page_list.selectedIndexes())
        if rows:
            return rows
        return [self.current_page] if self.current_page >= 0 else []

    def _delete_pages(self) -> None:
        """删除页列表里选中的页（可撤销）。多选可一次删掉整个追加进来的文件。"""
        if self.doc is None:
            return
        rows = self._selected_page_rows()
        if not rows:
            return
        if len(rows) >= len(self.doc.pages):
            QMessageBox.information(
                self, "提示", "不能删除全部页面。要清空请用「文件 → 关闭当前文档」。"
            )
            return
        names = "、".join(f"第 {r + 1} 页" for r in rows[:5])
        more = f" 等 {len(rows)} 页" if len(rows) > 5 else ""
        if QMessageBox.question(self, "删除页面", f"删除 {names}{more}？可用 Ctrl+Z 撤销。") != (
            QMessageBox.Yes
        ):
            return
        self._sync_canvas_to_records()
        with self._command(f"删除 {len(rows)} 页"):
            self.doc.remove_pages(rows)
            self.stamps = self._reindex_stamps(rows)
            self._rebuild_page_list(select=min(rows[0], len(self.doc.pages) - 1))
        self.info_label.setText(f"已删除 {len(rows)} 页")

    def _reindex_stamps(self, removed_rows: list[int]) -> dict[int, list[StampRecord]]:
        """删页后重建 {页码: 记录} 映射：被删页的记录丢弃，其后各页下标前移。"""
        doomed = set(removed_rows)
        out: dict[int, list[StampRecord]] = {}
        for old_index, recs in self.stamps.items():
            if old_index in doomed or not recs:
                continue
            out[old_index - sum(1 for r in doomed if r < old_index)] = recs
        return out

    # ── 页面显示 ──

    def _magnifier_crop(self, x_mm: float, y_mm: float):
        """放大镜取景：光标 ±HALF_WINDOW_MM 的全分辨率画面，越界补边、准线恒在中心。"""
        from app.magnifier import HALF_WINDOW_MM

        if self.doc is None or self.current_page < 0:
            return None
        page = self.doc.pages[self.current_page]
        dpi = page.dpi
        cx = round(x_mm / MM_PER_INCH * dpi)
        cy = round(y_mm / MM_PER_INCH * dpi)
        r = round(HALF_WINDOW_MM / MM_PER_INCH * dpi)
        img = page.image
        h, w = img.shape[:2]
        x0, y0 = max(0, cx - r), max(0, cy - r)
        x1, y1 = min(w, cx + r), min(h, cy + r)
        crop = img[y0:y1, x0:x1]
        # 越界补中性灰，保证准线恒在中心（光标位置）
        pad_l, pad_t = r - (cx - x0), r - (cy - y0)
        pad_r, pad_b = 2 * r - crop.shape[1] - pad_l, 2 * r - crop.shape[0] - pad_t
        if pad_l or pad_t or pad_r or pad_b:
            crop = np.pad(
                crop,
                ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                mode="constant",
                constant_values=205,
            )
        return np_rgb_to_qpixmap(np.ascontiguousarray(crop))

    def _on_page_changed(self, row: int, resync: bool = True) -> None:
        if self.doc is None or row < 0 or row >= len(self.doc.pages):
            return
        if resync:
            self._sync_canvas_to_records()
        self.current_page = row
        page = self.doc.pages[row]
        # 放大镜取景源：全分辨率 + 变异感知（旋转/校准后的页面取景同样正确）
        self.canvas.magnifier_source = self._magnifier_crop
        self.canvas.show_page(display_image(page.image), page.phys_w_mm, page.phys_h_mm)
        for rec in self.stamps.get(row, []):
            self.canvas.add_stamp(self._make_stamp_item(rec))

    def _page_icon(self, page: Page):
        """页缩略图（低 DPI 独立渲染，不占用全尺寸 LRU）。带缓存：撤销/重做
        会整表重建页列表，50 页文档每次重渲染是实打实的卡顿。"""
        key = (page.uid, page.revision)
        icon = self._thumb_cache.get(key)
        if icon is None:
            icon = np_rgb_to_qpixmap(thumbnail(page.thumbnail(), 140))
            if len(self._thumb_cache) > 400:
                self._thumb_cache.clear()
            self._thumb_cache[key] = icon
        return icon

    def _refresh_page_thumbnail(self, row: int) -> None:
        if self.doc is None:
            return
        item = self.page_list.item(row)
        if item:
            item.setIcon(self._page_icon(self.doc.pages[row]))

    # ── 盖章交互 ──

    def _require_document(self) -> bool:
        if self.doc is None or self.current_page < 0:
            QMessageBox.information(self, "提示", "请先打开合同文件")
            return False
        return True

    def _add_stamp(self, seal: Seal) -> None:
        """盖章：印章跟随鼠标，在页面上点哪盖哪（修改意见）。"""
        if not self._require_document():
            return
        # 落章即采样随机效果，预览立即所见即所得
        processed, applied = self._session_rng.apply_auto(
            seal.image, self.panel.random_spec()
        )
        self._start_placement(
            StampRecord(
                seal=seal,
                center_x_mm=0.0,
                center_y_mm=0.0,
                size_mm=seal.phys_mm,
                applied=applied,
                processed=processed,
            ),
            f"「{seal.name}」跟随鼠标中——在页面上点击落位，Esc 取消",
        )

    def _add_date_stamp(self) -> None:
        """加盖日期：默认系统当天，可改日期/格式/字高，之后点哪盖哪。"""
        if not self._require_document():
            return
        dlg = DateStampDialog(
            self, date.today(), self.settings.date_format, self.settings.date_height_mm
        )
        if dlg.exec() != QDialog.Accepted:
            return
        text, fmt, height_mm = dlg.values()
        page = self.doc.pages[self.current_page]
        try:
            ink = render_date_ink(text, height_mm, page.dpi)
        except ValueError as e:
            QMessageBox.warning(self, "日期渲染失败", str(e))
            return
        self.settings.date_format = fmt
        self.settings.date_height_mm = height_mm
        self._save_settings()

        # 日期不加随机手感：打印/书写的日期没有印泥深浅，加蒙尘反而假
        seal = Seal(name=text, kind=KIND_DATE, image=ink, phys_mm=ink_width_mm(ink, height_mm))
        self._start_placement(
            StampRecord(
                seal=seal,
                center_x_mm=0.0,
                center_y_mm=0.0,
                size_mm=seal.phys_mm,
                processed=ink,
            ),
            f"日期「{text}」跟随鼠标中——在页面上点击落位，Esc 取消",
        )

    def _start_placement(self, rec: StampRecord, hint: str) -> None:
        self._pending_rec = rec
        self.canvas.start_follow(rec.processed, rec.size_mm)
        self.info_label.setText(hint)

    def _on_stamp_placed(self, x_mm: float, y_mm: float) -> None:
        rec = self._pending_rec
        if rec is None:
            self.canvas.cancel_follow()
            return
        rec.center_x_mm = x_mm
        rec.center_y_mm = y_mm
        self.canvas.cancel_follow()
        self._pending_rec = None
        page_idx = self.current_page
        with self._command(f"盖章「{rec.seal.name}」"):
            self.stamps.setdefault(page_idx, []).append(rec)
            item = self._make_stamp_item(rec)
            self.canvas.add_stamp(item)
            self.canvas.centerOn(item)
        self._update_info(rec)

    def _on_follow_cancelled(self) -> None:
        self._pending_rec = None
        self.info_label.setText("已取消盖章")

    def _on_canvas_clicked(self, item, x_mm: float, y_mm: float) -> None:
        """空白处单击：把候选章移过去（press 时记录的章，不依赖 release 时的选中态）。"""
        rec = self._find_record(item)
        if rec is None:
            return
        with self._command("移动章"):
            rec.center_x_mm = x_mm
            rec.center_y_mm = y_mm
            item.set_center(x_mm, y_mm)
        self._update_info(rec)
        self._sync_adjust_spins(rec)

    def _on_place_rejected(self) -> None:
        self.info_label.setText("请点击页面范围内落章（章整个飞出纸面会被忽略）")

    def _restamp_selected(self) -> None:
        """重盖：撤掉选中的章，换一次手感重新点位置盖（骑缝切片不适用）。"""
        rec = self._selected_record()
        if rec is None:
            self.info_label.setText("先点击选中一个章，再点重盖")
            return
        if rec.locked:
            QMessageBox.information(
                self, "提示", "骑缝切片不能单独重盖。请删除整组后重新开骑缝章对话框。"
            )
            return
        page_idx = self.current_page
        size_mm, rotation, opacity = rec.size_mm, rec.rotation_deg, rec.opacity
        with self._command(f"重盖「{rec.seal.name}」"):
            self.stamps[page_idx] = [r for r in self.stamps.get(page_idx, []) if r is not rec]
            self._on_page_changed(page_idx, resync=False)
        processed, applied = self._session_rng.apply_auto(
            rec.seal.image, self.panel.random_spec()
        )
        self._start_placement(
            StampRecord(
                seal=rec.seal,
                center_x_mm=0.0,
                center_y_mm=0.0,
                size_mm=size_mm,
                rotation_deg=rotation,
                opacity=opacity,
                applied=applied,
                processed=processed,
            ),
            f"重盖「{rec.seal.name}」：点击新位置落位，Esc 取消（Ctrl+Z 可退回原来那枚）",
        )

    # ── 骑缝章 ──

    def _add_perforation(self, seal: Seal) -> None:
        """骑缝章：对话框 → 拼合预览确认 → 切片作为 locked 记录落到各页。"""
        if self.doc is None:
            QMessageBox.information(self, "提示", "请先打开合同文件")
            return
        seed = secrets.randbelow(2**31 - 1)
        # 印章先过一遍全局随机效果（角度/色度/蒙尘），再切割
        processed_seal, _applied = Randomizer(seed).apply_auto(
            seal.image, self.panel.random_spec()
        )
        dlg = PerforationDialog(
            self, self.doc.pages, processed_seal, seal.name, seal.phys_mm, seed
        )
        if dlg.exec() != QDialog.Accepted or dlg.placements is None:
            return

        group = f"perf_{seed}"
        with self._command("应用骑缝章"):
            touched_pages = self._append_perforation_records(seal, dlg.placements, group)
            target = (
                self.current_page if self.current_page in touched_pages else min(touched_pages)
            )
            self.page_list.setCurrentRow(target)
            self._on_page_changed(self.current_page)
        # 定位到第一个切片并放大，让细条带直接可见（骑缝可见性）
        if min(touched_pages) == self.current_page:
            for item in self.canvas.stamps():
                rec = self._find_record(item)
                if rec is not None and rec.locked:
                    self.canvas.centerOn(item)
                    self.canvas.scale(2.0, 2.0)
                    item.setSelected(True)
                    break
        QMessageBox.information(
            self, "骑缝章已应用", f"已在 {len(touched_pages)} 页放置切片（组 {group}）。"
        )

    def _append_perforation_records(
        self, seal: Seal, placements: list[SlicePlacement], group: str
    ) -> set[int]:
        """把切片放置结果转成 locked 记录挂到各页。返回受影响的页码集合。"""
        touched: set[int] = set()
        for pl in placements:
            page = self.doc.pages[pl.page_index]
            h_px, w_px = pl.slice_rgba.shape[:2]
            w_mm = w_px / page.dpi * MM_PER_INCH
            h_mm = h_px / page.dpi * MM_PER_INCH
            self.stamps.setdefault(pl.page_index, []).append(
                StampRecord(
                    seal=seal,
                    center_x_mm=pl.right_edge_mm - w_mm / 2,
                    center_y_mm=pl.top_mm + h_mm / 2,
                    size_mm=w_mm,
                    processed=pl.slice_rgba,
                    locked=True,
                    group=group,
                )
            )
            touched.add(pl.page_index)
        return touched

    def _apply_perforation_records(
        self, seal: Seal, seed: int, page_indices: list[int] | None = None
    ) -> int:
        """不弹对话框的骑缝章应用（批量导出用）。返回切片数。"""
        processed_seal, _ = Randomizer(seed).apply_auto(seal.image, self.panel.random_spec())
        if page_indices is None:
            page_indices = list(range(len(self.doc.pages)))
        try:
            placements = plan_perforation(
                processed_seal, self.doc.pages, page_indices, PerforationSpec(seed=seed)
            )
        except ValueError as e:
            raise ValueError(f"骑缝章：{e}") from e
        self._append_perforation_records(seal, placements, f"perf_{seed}")
        return len(placements)

    # ── 记录与画布同步 ──

    def _make_stamp_item(self, rec: StampRecord) -> StampItem:
        rgba = rec.processed if rec.processed is not None else rec.seal.image
        item = StampItem(rgba, rec.size_mm, rec.center_x_mm, rec.center_y_mm)
        item.setRotation(rec.rotation_deg)
        item.setOpacity(rec.opacity)
        item.setData(0, id(rec))  # 画布回同步时定位记录
        if rec.locked:
            # 骑缝切片禁止拖拽：单片拖动会破坏跨页对齐，只能整组微调
            item.setFlag(QGraphicsPixmapItem.ItemIsMovable, False)
        return item

    def _find_record(self, item: StampItem) -> StampRecord | None:
        for rec in self.stamps.get(self.current_page, []):
            if id(rec) == item.data(0):
                return rec
        return None

    def _on_stamp_moved(self, item: StampItem) -> None:
        rec = self._find_record(item)
        if rec is None:
            return
        rec.center_x_mm, rec.center_y_mm = item.center()
        self._update_info(rec)
        self._sync_adjust_spins(rec)

    def _apply_adjustments(self) -> None:
        if self._syncing:
            return
        item = self.canvas.selected_stamp()
        if item is None:
            return
        rec = self._find_record(item)
        if rec is None:
            return
        rec.size_mm = self.size_spin.value()
        rec.rotation_deg = self.rot_spin.value()
        rec.opacity = self.opa_spin.value()
        center = item.center()
        item.size_mm = rec.size_mm
        item._update_scale()
        item.set_center(*center)
        item.setRotation(rec.rotation_deg)
        item.setOpacity(rec.opacity)
        self._update_info(rec)

    def _sync_adjust_spins(self, rec: StampRecord) -> None:
        self._syncing = True
        self.size_spin.setValue(rec.size_mm)
        self.rot_spin.setValue(rec.rotation_deg)
        self.opa_spin.setValue(rec.opacity)
        self._syncing = False

    def _update_info(self, rec: StampRecord | None) -> None:
        if rec is None:
            self.info_label.setText("选中印章后，在页面空白处单击即可移动章到点击处")
            return
        size_text = (
            f"⌀{rec.size_mm:.1f}mm" if rec.seal.kind == KIND_SEAL else f"宽{rec.size_mm:.1f}mm"
        )
        lock = "（骑缝切片）" if rec.locked else ""
        self.info_label.setText(
            f"{rec.seal.name}{lock}：中心 ({rec.center_x_mm:.1f}, {rec.center_y_mm:.1f}) mm，"
            f"{size_text}，旋转 {rec.rotation_deg:.1f}°"
        )

    def _sync_canvas_to_records(self) -> None:
        """翻页/导出前：把画布上的印章位置写回记录。

        注意：本函数只做位置同步，绝不删除记录——删除记录的唯一入口是
        用户在画布上按 Delete（stamps_deleted 显式信号）。
        （教训：曾经的差集剪除把"还没渲染上画布的新记录"误删，导致
        骑缝章/模板在当前页丢失。）
        """
        if self.current_page < 0:
            return
        for item in self.canvas.stamps():
            rec = self._find_record(item)
            if rec is None:
                continue
            rec.center_x_mm, rec.center_y_mm = item.center()

    def _on_stamps_deleted(self, items: list) -> None:
        """显式删除：只处理用户在画布上按 Delete 移除的章。"""
        page_idx = self.current_page
        doomed = {it.data(0) for it in items}
        current = self.stamps.get(page_idx, [])
        removed = [r for r in current if id(r) in doomed]
        if not removed:
            return
        with self._command(f"删除 {len(removed)} 枚章"):
            self.stamps[page_idx] = [r for r in current if id(r) not in doomed]
        self.info_label.setText("已删除")

    # ── 页面操作 ──

    def _move_page(self, delta: int) -> None:
        """页序调整：当前页上移/下移一位，盖章记录随页面走。"""
        if self.doc is None or self.current_page < 0:
            return
        old = self.current_page
        new = old + delta
        if new < 0 or new >= len(self.doc.pages):
            return
        self._sync_canvas_to_records()
        with self._command("调整页序"):
            self.doc.pages[old], self.doc.pages[new] = (
                self.doc.pages[new],
                self.doc.pages[old],
            )
            self.stamps[old], self.stamps[new] = (
                self.stamps.get(new, []),
                self.stamps.get(old, []),
            )
            self.stamps = {k: v for k, v in self.stamps.items() if v}
            self._rebuild_page_list(select=new)

    def _rotate_page(self, k: int) -> None:
        """旋转当前页图像与物理尺寸，印章坐标联动。

        k: np.rot90 的次数（1=逆时针90, -1=顺时针90, 2=180）。
        """
        if self.doc is None or self.current_page < 0:
            return
        with self._command("旋转页面"):
            page = self.doc.pages[self.current_page]
            old_w, old_h = page.phys_w_mm, page.phys_h_mm
            page.image = np.ascontiguousarray(np.rot90(page.image, k))
            k_mod = k % 4
            if k_mod in (1, 3):
                page.phys_w_mm, page.phys_h_mm = old_h, old_w
            # 印章坐标联动（mm 坐标系旋转）
            for rec in self.stamps.get(self.current_page, []):
                x, y = rec.center_x_mm, rec.center_y_mm
                if k_mod == 1:      # 逆时针 90
                    rec.center_x_mm, rec.center_y_mm = y, old_w - x
                    rec.rotation_deg -= 90
                elif k_mod == 3:    # 顺时针 90
                    rec.center_x_mm, rec.center_y_mm = old_h - y, x
                    rec.rotation_deg += 90
                else:               # 180
                    rec.center_x_mm, rec.center_y_mm = old_w - x, old_h - y
                    rec.rotation_deg += 180
                rec.rotation_deg = (rec.rotation_deg + 180) % 360 - 180
            self._on_page_changed(self.current_page, resync=False)
            self._refresh_page_thumbnail(self.current_page)

    def _four_point_calibrate(self) -> None:
        """四点纸边校准：自动检测纸面四角作为初值，用户拖把手微调。

        扫描件自带白边 → 页面边界 ≠ 纸张边界。确认后按透视变换把页面
        拉伸裁正为标准 A4，白边消失、物理尺寸精确。
        """
        if self.doc is None or self.current_page < 0:
            return
        page = self.doc.pages[self.current_page]
        quad_px, detected = initial_quad(page.image)
        dpi = page.dpi
        pts_mm = [
            (float(x) / dpi * MM_PER_INCH, float(y) / dpi * MM_PER_INCH) for x, y in quad_px
        ]
        self.canvas.start_quad_adjust(pts_mm)
        source = "已自动检测到纸面四角" if detected else "未检测到纸边，先给了个默认框"
        self.info_label.setText(
            f"四点校准：{source}。拖动蓝色把手贴住纸的四个角，"
            "拖空白处可平移、滚轮缩放，方向键微调 0.1mm，回车确认，Esc 取消。"
        )

    def _on_quad_changed(self, pts_mm: list) -> None:
        """拖动中实时提示长宽比：偏离 A 系纸的 √2 通常意味着角点没贴准。"""
        if self.doc is None or self.current_page < 0:
            return
        dpi = self.doc.pages[self.current_page].dpi
        quad_px = np.array(
            [[x / MM_PER_INCH * dpi, y / MM_PER_INCH * dpi] for x, y in pts_mm],
            dtype=np.float32,
        )
        ratio = quad_aspect(quad_px)
        deviation = abs(ratio - ASPECT_A_SERIES) / ASPECT_A_SERIES
        verdict = (
            "接近 A 系纸"
            if deviation <= ASPECT_TOLERANCE
            else "偏离 A 系纸，检查角点是否贴准"
        )
        self.info_label.setText(
            f"当前框选长宽比 {ratio:.3f}（A 系纸为 {ASPECT_A_SERIES:.3f}）——{verdict}。\n"
            "回车确认，Esc 取消。"
        )

    def _on_quad_cancelled(self) -> None:
        self.info_label.setText("已取消四点校准")

    def _on_quad_accepted(self, pts_mm: list) -> None:
        page = self.doc.pages[self.current_page]
        old_dpi = page.dpi
        # 场景 mm → 页面像素
        quad_px = np.array(
            [[x / MM_PER_INCH * old_dpi, y / MM_PER_INCH * old_dpi] for x, y in pts_mm],
            dtype=np.float32,
        )
        # 先在副本上试变换，用户确认后再应用
        trial = Page(image=page.image, phys_w_mm=page.phys_w_mm, phys_h_mm=page.phys_h_mm)
        homography = warp_to_a4(trial, quad_px)
        if homography is None:
            QMessageBox.warning(
                self, "四点无效",
                "四个把手围成的区域太小（可能挤在一起或几乎共线），请拉开后再确认。",
            )
            return  # 把手留在原处，用户接着调，不必从头再来
        if WarpPreviewDialog(self, trial.image).exec() != QDialog.Accepted:
            return  # 同上：预览里觉得不对就继续拖，把手不撤

        with self._command("四点纸边校准"):
            # 已盖章坐标映射：旧 mm → 旧像素 →(H)→ 新像素 → 新 mm
            recs = self.stamps.get(self.current_page, [])
            new_dpi = trial.dpi
            if recs:
                old_pts = np.array(
                    [
                        [r.center_x_mm / MM_PER_INCH * old_dpi,
                         r.center_y_mm / MM_PER_INCH * old_dpi]
                        for r in recs
                    ],
                    dtype=np.float32,
                )
                for r, (nx, ny) in zip(recs, map_points_through(homography, old_pts), strict=True):
                    r.center_x_mm = float(nx) / new_dpi * MM_PER_INCH
                    r.center_y_mm = float(ny) / new_dpi * MM_PER_INCH
            page.image = trial.image
            page.phys_w_mm, page.phys_h_mm = trial.phys_w_mm, trial.phys_h_mm
            page.needs_calibration = False
            self.canvas.cancel_quad_adjust()
            self._on_page_changed(self.current_page, resync=False)
            self._refresh_page_thumbnail(self.current_page)
        self.info_label.setText("四点校准完成：页面已拉伸为标准 A4（210×297mm）")

    def _calibrate(self) -> None:
        if self.doc is None:
            return
        page = self.doc.pages[max(self.current_page, 0)]
        dlg = CalibrateDialog(self, page.phys_w_mm, page.phys_h_mm)
        if dlg.exec() != QDialog.Accepted:
            return
        w_mm, h_mm, apply_all = dlg.values()
        with self._command("纸边校准"):
            for p in self.doc.pages if apply_all else [page]:
                calibrate_paper_edge(p, w_mm, h_mm)
            self._on_page_changed(self.current_page, resync=False)

    def _auto_calibrate(self) -> None:
        if self.doc is None:
            return
        ok, fail = 0, []
        with self._command("自动纸边检测"):
            for i, page in enumerate(self.doc.pages):
                if auto_calibrate_page(page):
                    ok += 1
                else:
                    fail.append(i + 1)
            self._on_page_changed(self.current_page, resync=False)
        msg = f"{ok} 页检测并校正成功。"
        if fail:
            msg += f"\n第 {fail} 页未检测到纸边，请改用手动校准。"
        QMessageBox.information(self, "自动纸边检测", msg)

    # ── 删除与骑缝组管理 ──

    def _show_about(self) -> None:
        from core import __version__

        AboutDialog(self, __version__).exec()

    def _on_selection_changed(self) -> None:
        item = self.canvas.selected_stamp()
        rec = self._find_record(item) if item else None
        self.group_box.setVisible(bool(rec and rec.locked))
        if rec:
            self._update_info(rec)
            self._sync_adjust_spins(rec)

    def _selected_record(self) -> StampRecord | None:
        item = self.canvas.selected_stamp()
        return self._find_record(item) if item else None

    def _delete_selected(self) -> None:
        """删除选中的章/签名（Delete 键的可视入口）。"""
        if self.canvas.selected_stamp() is None:
            self.info_label.setText("先点击选中一个章，再删除")
            return
        self.canvas.remove_selected_stamp()  # 内部发 stamps_deleted → 记录同步移除

    def _delete_group(self) -> None:
        """删除整组骑缝章（跨所有页）。"""
        rec = self._selected_record()
        if rec is None or not rec.locked or not rec.group:
            return
        gid = rec.group
        count = sum(1 for recs in self.stamps.values() for r in recs if r.group == gid)
        with self._command(f"删除整组骑缝章（{count} 切片）"):
            self.stamps = {
                page_idx: [r for r in recs if r.group != gid]
                for page_idx, recs in self.stamps.items()
            }
            self.stamps = {k: v for k, v in self.stamps.items() if v}
            self._on_page_changed(self.current_page, resync=False)
        self.info_label.setText(f"已删除整组骑缝章（{count} 个切片）")

    def _apply_group_shift(self) -> None:
        """骑缝组竖向整体微调：所有切片统一下移/上移，保留逐页抖动差。"""
        rec = self._selected_record()
        if rec is None or not rec.locked or not rec.group:
            return
        delta = self.group_shift_spin.value()
        if abs(delta) < 1e-9:
            return
        gid = rec.group
        with self._command(f"骑缝组竖向移动 {delta:+.1f}mm"):
            for records in self.stamps.values():
                for r in records:
                    if r.group == gid:
                        r.center_y_mm += delta
            # 先让当前页画布贴图跟上记录，再刷新——否则 _on_page_changed 的
            # 位置回同步会把当前页的记录改回贴图旧位置
            for item in self.canvas.stamps():
                r2 = self._find_record(item)
                if r2 is not None and r2.group == gid:
                    item.set_center(r2.center_x_mm, r2.center_y_mm)
            self._on_page_changed(self.current_page)
        self.group_shift_spin.setValue(0.0)
        self.info_label.setText(f"整组骑缝章已竖向移动 {delta:+.1f}mm（逐页抖动保持）")

    # ── 随机手感与模板 ──

    def _reroll_random(self) -> None:
        """「换一批手感」：重摇所有普通章/签名的随机效果（骑缝切片与日期除外）。

        骑缝切片是一整枚章切出来的，参数未保留，无法重摇——
        需要换手感请重新打开骑缝章对话框。日期戳本来就不加随机。
        """
        rng = Randomizer(secrets.randbelow(2**31 - 1))
        spec = self.panel.random_spec()
        count = 0
        with self._command("换一批手感"):
            for records in self.stamps.values():
                for rec in records:
                    if rec.locked or rec.seal.kind == KIND_DATE:
                        continue
                    rec.processed, rec.applied = rng.apply_auto(rec.seal.image, spec)
                    count += 1
            # 刷新当前页画布上的贴图
            for item in self.canvas.stamps():
                rec = self._find_record(item)
                if rec is not None and not rec.locked and rec.processed is not None:
                    item.set_pixmap_rgba(rec.processed)
        self.info_label.setText(f"已为 {count} 枚章/签名更换手感（骑缝切片与日期不受影响）")

    def _save_template(self) -> None:
        if self.doc is None or self.current_page < 0:
            return
        self._sync_canvas_to_records()
        page = self.doc.pages[self.current_page]
        # 日期戳不入模板：它的"印章"是临时合成的，不在库里，套用时必然落空
        records = [r for r in self.stamps.get(self.current_page, []) if r.seal.kind != KIND_DATE]
        entries = [
            {
                "seal_name": r.seal.name,
                "kind": r.seal.kind,
                "rel_x": r.center_x_mm / page.phys_w_mm,
                "rel_y": r.center_y_mm / page.phys_h_mm,
                "size_mm": r.size_mm,
                "rotation_deg": r.rotation_deg,
                "opacity": r.opacity,
            }
            for r in records
        ]
        name, ok = QInputDialog.getText(self, "保存模板", "模板名称：")
        if not ok or not name.strip():
            return
        try:
            save_template(default_template_dir(), name.strip(), entries)
        except ValueError as e:
            QMessageBox.warning(self, "保存失败", str(e))

    def _refresh_template_menu(self) -> None:
        # 保留前两个固定项，重建模板列表
        for act in self.tpl_menu.actions()[2:]:
            self.tpl_menu.removeAction(act)
        for path in list_templates(default_template_dir()):
            act = QAction(path.stem, self)
            act.triggered.connect(lambda checked=False, p=path: self._apply_template_action(p))
            self.tpl_menu.addAction(act)

    def _apply_template_action(self, path: Path) -> None:
        with self._command(f"套用模板「{path.stem}」"):
            count = self._apply_template(path)
            if count:
                self._on_page_changed(self.current_page)
        if count == 0:
            QMessageBox.information(self, "提示", "模板中的印章都不在库中，没有可套用的章。")

    def _apply_template(self, path: Path, target_page: int | None = None) -> int:
        """把模板套用到目标页（默认当前页）。返回套用的章数，不负责刷新画布。"""
        if self.doc is None:
            return 0
        page_idx = self.current_page if target_page is None else target_page
        if page_idx < 0:
            return 0
        page = self.doc.pages[page_idx]
        seals_by_name = self._library_seals_by_name()
        count = 0
        for e in load_template(path):
            seal = seals_by_name.get(e["seal_name"])
            if seal is None:
                continue
            processed, applied = self._session_rng.apply_auto(
                seal.image, self.panel.random_spec()
            )
            self.stamps.setdefault(page_idx, []).append(
                StampRecord(
                    seal=seal,
                    center_x_mm=e["rel_x"] * page.phys_w_mm,
                    center_y_mm=e["rel_y"] * page.phys_h_mm,
                    size_mm=e["size_mm"],
                    rotation_deg=e.get("rotation_deg", 0.0),
                    opacity=e.get("opacity", 1.0),
                    applied=applied,
                    processed=processed,
                )
            )
            count += 1
        return count

    def _library_seals_by_name(self) -> dict[str, Seal]:
        out: dict[str, Seal] = {}
        for png in list_library(default_library_dir()):
            try:
                seal = Seal.load(png)
            except Exception:
                continue
            out[seal.name] = seal
        return out

    # ── 输出目录设置 ──

    def _save_settings(self) -> None:
        try:
            self.settings.save()
        except OSError as e:
            self.info_label.setText(f"设置保存失败（不影响本次操作）：{e}")

    def _refresh_output_dir_action(self) -> None:
        target = self.settings.output_dir or "与源文件同目录"
        self.out_dir_act.setText(f"输出目录：{target}")
        self.out_dir_act.setToolTip("点击恢复为「与源文件同目录」")
        self.out_dir_act.setEnabled(bool(self.settings.output_dir))

    def _choose_output_dir(self) -> None:
        start = self.settings.output_dir or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(self, "选择盖章后文件的输出目录", start)
        if not chosen:
            return
        self.settings.output_dir = chosen
        self._save_settings()
        self._refresh_output_dir_action()
        self.info_label.setText(f"输出目录已设为：{chosen}")

    def _reset_output_dir(self) -> None:
        self.settings.output_dir = ""
        self._save_settings()
        self._refresh_output_dir_action()
        self.info_label.setText("输出目录已恢复为「与源文件同目录」")

    def _default_output_path(self) -> Path:
        out_dir = self.settings.resolved_output_dir(self.doc.source_path)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            out_dir = Path.cwd()  # 设置里的目录被删了/没权限，退回当前目录
        return make_output_path(self.doc.source_path, out_dir)

    # ── 导出 ──

    def _export_document(self, out_path: Path | None = None) -> Path:
        """把当前 doc + stamps 导出为 PDF（批量与单次导出共用）。返回输出路径。

        所见即所得：所有章直接使用落章时已采样的 processed 图像，
        导出不做任何重采样（v0.2.1 起废除"每次导出新种子"——画布显示
        什么就导出什么；想换手感用「换一批手感」按钮主动重摇）。
        """
        spec = self.panel.random_spec()
        stamp_logs = [
            {
                "page": page_idx + 1,
                "seal": rec.seal.name,
                "kind": "perforation_slice" if rec.locked else rec.seal.kind,
                "group": rec.group,
                "center_mm": [round(rec.center_x_mm, 2), round(rec.center_y_mm, 2)],
                "size_mm": rec.size_mm,
                "rotation_deg": rec.rotation_deg,
                "opacity": rec.opacity,
                "random": rec.applied.to_dict() if rec.applied else None,
            }
            for page_idx, records in self.stamps.items()
            for rec in records
        ]

        images: list[np.ndarray] = []
        for page_idx, page in enumerate(self.doc.pages):
            img = page.image
            for rec in self.stamps.get(page_idx, []):
                cur = Page(image=img, phys_w_mm=page.phys_w_mm, phys_h_mm=page.phys_h_mm)
                img = stamp_page(
                    cur,
                    rec.processed if rec.processed is not None else rec.seal.image,
                    Placement(
                        center_x_mm=rec.center_x_mm,
                        center_y_mm=rec.center_y_mm,
                        size_mm=rec.size_mm,
                        rotation_deg=rec.rotation_deg,
                        opacity=rec.opacity,
                    ),
                )
            images.append(img)

        out_path = Path(out_path) if out_path else self._default_output_path()
        sealog = {
            # 注意：顶层 seed 仅供参考；复现的权威数据是每枚章的 random.applied 值
            "seed": self._session_rng.seed,
            "seed_note": "per-stamp random.applied values are authoritative for replay",
            "random_spec": {
                "angle_deg": spec.angle_deg,
                "tone": spec.tone,
                "dust": spec.dust,
            },
            "source": str(self.doc.source_path) if self.doc.source_path else None,
            "page_count": len(self.doc.pages),
            "stamps": stamp_logs,
        }
        export_pdf(self.doc.pages, images, out_path, sealog)
        return out_path

    def _export(self) -> None:
        if self.doc is None:
            QMessageBox.information(self, "提示", "请先打开合同文件")
            return
        self._sync_canvas_to_records()
        total_stamps = sum(len(v) for v in self.stamps.values())
        if total_stamps == 0:
            if QMessageBox.question(
                self, "确认", "当前没有任何盖章，仍要导出吗？"
            ) != QMessageBox.Yes:
                return

        default_path = self._default_output_path()
        chosen, _ = QFileDialog.getSaveFileName(
            self, "导出已盖章 PDF", str(default_path), "PDF 文件 (*.pdf)"
        )
        if not chosen:
            return
        out_path = Path(chosen).with_suffix(".pdf")
        # 记住这次选的目录，下次默认还去那里
        if str(out_path.parent) != str(default_path.parent):
            self.settings.output_dir = str(out_path.parent)
            self._save_settings()
            self._refresh_output_dir_action()
        try:
            out_path = self._export_document(out_path)
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        # 预览刷新为导出同源结果（所见即所得）
        self._on_page_changed(self.current_page)
        QMessageBox.information(
            self, "导出完成", f"已导出：\n{out_path}\n\n随机种子与参数已写入同名 .sealog"
        )

    def _batch_export(self) -> None:
        templates = list_templates(default_template_dir())
        if not templates:
            QMessageBox.information(self, "提示", "还没有模板。请先在「模板」菜单保存一个。")
            return
        names = [t.stem for t in templates]
        name, ok = QInputDialog.getItem(self, "批量导出", "选择模板：", names, 0, False)
        if not ok:
            return
        tpl_path = templates[names.index(name)]
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要批量盖章的合同（可多选）", "", self._FILE_FILTER
        )
        if not paths:
            return
        out_dir = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录（直接取消 = 各自输出到源文件所在目录）",
            self.settings.output_dir or "",
        )
        if out_dir:
            self.settings.output_dir = out_dir
            self._save_settings()
            self._refresh_output_dir_action()
        target_last = QMessageBox.question(
            self, "盖章位置", "把模板盖在每个文件的【最后一页】吗？\n（选 No 则盖第一页）"
        )
        # 批量也支持骑缝章（用面板当前选中的印章，全部页，每文件独立手感）
        perf_seal = self.panel.current_seal()
        with_perf = perf_seal is not None and QMessageBox.question(
            self, "骑缝章", f"是否同时加盖骑缝章（全部页，用「{perf_seal.name}」）？"
        ) == QMessageBox.Yes

        progress = QProgressDialog("批量盖章中…", "取消", 0, len(paths), self)
        progress.setWindowModality(Qt.WindowModal)
        done: list[Path] = []
        failed: list[tuple[str, str]] = []
        for i, path in enumerate(paths):
            progress.setValue(i)
            progress.setLabelText(f"正在处理（{i + 1}/{len(paths)}）：{Path(path).name}")
            if progress.wasCanceled():
                break
            try:
                self._load_document([path])
                page_idx = len(self.doc.pages) - 1 if target_last == QMessageBox.Yes else 0
                if self._apply_template(tpl_path, target_page=page_idx) == 0:
                    failed.append((path, "模板中的印章不在库中"))
                    continue
                if with_perf:
                    self._apply_perforation_records(perf_seal, secrets.randbelow(2**31 - 1))
                done.append(self._export_document())
            except Exception as e:
                failed.append((path, str(e)))
        progress.setValue(len(paths))

        msg = f"成功 {len(done)} 个：\n" + "\n".join(str(p) for p in done[:10])
        if failed:
            msg += f"\n\n失败 {len(failed)} 个：\n" + "\n".join(
                f"{p}: {e}" for p, e in failed[:5]
            )
        QMessageBox.information(self, "批量导出完成", msg)
        # 批量结束后清空当前文档，避免误操作
        if self.doc is not None:
            self.doc.close()
        self.doc = None
        self.stamps = {}
        self.current_page = -1
        self._clear_history()
        self.page_list.clear()
        self.canvas.clear_page()
