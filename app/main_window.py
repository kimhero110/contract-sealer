"""主窗口：左侧页缩略图 / 中间画布 / 右侧印章与参数面板。

盖章会话状态模型：
- Document（core）提供页面图像与物理尺寸；
- 每页若干 StampRecord（印章 + 物理位置/尺寸/旋转/不透明度 + 已采样随机效果）；
- 预览与导出共用同一份 processed 图像——所见即所得（方案 §4.9）；
- 骑缝章切片是 locked 记录：随机在生成时已定，导出不再重采样（方案 §4.4）。
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
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
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.canvas import PageCanvas, StampItem, np_rgb_to_qpixmap
from app.perforation_dialog import PerforationDialog
from app.seal_panel import SealPanel
from core.autocal import auto_calibrate_page
from core.document import Document, Page, calibrate_paper_edge
from core.export import export_pdf, make_output_path
from core.perforation import apply_perforation
from core.randomize import AppliedRandom, Randomizer
from core.seal import KIND_SEAL, Seal, default_library_dir, list_library
from core.stamp import Placement, stamp_page
from core.template import (
    default_template_dir,
    list_templates,
    load_template,
    save_template,
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("合同盖章工具")
        self.resize(1280, 860)

        self.doc: Document | None = None
        self.stamps: dict[int, list[StampRecord]] = {}
        self.current_page = -1
        self._syncing = False
        # 会话随机器：落章即采样，预览立即带真实感
        self._session_rng = Randomizer(secrets.randbelow(2**31 - 1))

        self._build_ui()
        self._build_menu()
        self._update_info(None)

    # ── UI 搭建 ──

    def _build_ui(self) -> None:
        splitter = QSplitter()

        self.page_list = QListWidget()
        self.page_list.setMaximumWidth(160)
        self.page_list.currentRowChanged.connect(self._on_page_changed)
        splitter.addWidget(self.page_list)

        self.canvas = PageCanvas()
        self.canvas.stamp_moved.connect(self._on_stamp_moved)
        self.canvas.stamps_deleted.connect(self._on_stamps_deleted)
        self.canvas.stamp_placed.connect(self._on_stamp_placed)
        self.canvas.place_rejected.connect(self._on_place_rejected)
        self.canvas.follow_cancelled.connect(self._on_follow_cancelled)
        self.canvas.canvas_clicked.connect(self._on_canvas_clicked)
        self.canvas.points_picked.connect(self._on_points_picked)
        self.canvas.pick_cancelled.connect(self._on_pick_cancelled)
        self.canvas.scene().selectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self.canvas)

        right = QWidget()
        right_layout = QHBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        panel_container = QWidget()
        v = QVBoxLayout(panel_container)
        self.panel = SealPanel()
        self.panel.stamp_requested.connect(self._add_stamp)
        self.panel.perforation_requested.connect(self._add_perforation)
        self.panel.reroll_requested.connect(self._reroll_random)
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

        from PySide6.QtWidgets import QPushButton

        self.btn_delete = QPushButton("🗑 删除选中的章（Delete）")
        self.btn_delete.clicked.connect(self._delete_selected)
        v.addWidget(self.btn_delete)

        # 骑缝组管理（选中骑缝切片时可见）
        self.group_box = QWidget()
        gb = QVBoxLayout(self.group_box)
        gb.setContentsMargins(0, 0, 0, 0)
        gb.addWidget(QLabel("骑缝章整组操作："))
        shift_row = QHBoxLayout()
        self.group_shift_spin = QDoubleSpinBox()
        self.group_shift_spin.setRange(-100.0, 100.0)
        self.group_shift_spin.setSingleStep(0.5)
        self.group_shift_spin.setSuffix(" mm")
        self.group_shift_spin.setValue(0.0)
        btn_shift = QPushButton("竖向整体微调")
        btn_shift.clicked.connect(self._apply_group_shift)
        shift_row.addWidget(self.group_shift_spin)
        shift_row.addWidget(btn_shift)
        gb.addLayout(shift_row)
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
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # 工具栏：页面旋转 + 页序调整（修改意见 #1）
        from PySide6.QtWidgets import QToolBar

        tb = QToolBar("页面操作")
        tb.addAction("↻ 顺转 90°", lambda: self._rotate_page(-1))
        tb.addAction("↺ 逆转 90°", lambda: self._rotate_page(1))
        tb.addAction("180°", lambda: self._rotate_page(2))
        tb.addSeparator()
        tb.addAction("↑ 页面上移", lambda: self._move_page(-1))
        tb.addAction("↓ 页面下移", lambda: self._move_page(1))
        tb.addSeparator()
        tb.addAction("▣ 四点纸边校准", self._four_point_calibrate)
        self.addToolBar(tb)

    def _build_menu(self) -> None:
        open_act = QAction("打开合同…", self)
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self._open_files)
        export_act = QAction("导出…", self)
        export_act.setShortcut("Ctrl+E")
        export_act.triggered.connect(self._export)
        fit_act = QAction("适应页面", self)
        fit_act.triggered.connect(self.canvas.fit_page)

        m = self.menuBar().addMenu("文件")
        m.addAction(open_act)
        m.addAction(export_act)
        m.addAction(fit_act)

        rot_cw = QAction("当前页顺时针 90°", self)
        rot_cw.triggered.connect(lambda: self._rotate_page(-1))
        rot_ccw = QAction("当前页逆时针 90°", self)
        rot_ccw.triggered.connect(lambda: self._rotate_page(1))
        rot_180 = QAction("当前页 180°", self)
        rot_180.triggered.connect(lambda: self._rotate_page(2))
        cal_act = QAction("纸边校准（手动）…", self)
        cal_act.triggered.connect(self._calibrate)
        autocal_act = QAction("自动纸边检测（全部页）", self)
        autocal_act.triggered.connect(self._auto_calibrate)

        m2 = self.menuBar().addMenu("页面")
        m2.addAction(rot_cw)
        m2.addAction(rot_ccw)
        m2.addAction(rot_180)
        m2.addSeparator()
        m2.addAction(cal_act)
        m2.addAction(autocal_act)

        tpl_save = QAction("把当前页存为模板…", self)
        tpl_save.triggered.connect(self._save_template)
        self.tpl_menu = self.menuBar().addMenu("模板")
        self.tpl_menu.addAction(tpl_save)
        self.tpl_menu.addSeparator()
        self.tpl_menu.aboutToShow.connect(self._refresh_template_menu)

        batch_act = QAction("批量导出（套用模板）…", self)
        batch_act.triggered.connect(self._batch_export)
        m3 = self.menuBar().addMenu("批量")
        m3.addAction(batch_act)

        about_act = QAction("关于…", self)
        about_act.triggered.connect(self._show_about)
        m4 = self.menuBar().addMenu("帮助")
        m4.addAction(about_act)

    # ── 文件打开 ──

    def _open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "打开合同（PDF 或图片，可多选图片）",
            "",
            "合同文件 (*.pdf *.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not paths:
            return
        try:
            self._load_document(paths)
        except Exception as e:
            QMessageBox.warning(self, "打开失败", str(e))

    def _load_document(self, paths: list[str]) -> None:
        if len(paths) == 1 and paths[0].lower().endswith(".pdf"):
            doc = Document.open(paths[0])
        else:
            paths = sorted(paths, key=_natural_key)
            doc = Document.from_images(paths)

        self.doc = doc
        self.stamps = {}
        self.current_page = -1
        self.page_list.clear()
        for i, page in enumerate(doc.pages):
            thumb = np_rgb_to_qpixmap(_thumbnail(page.image, 120))
            item = QListWidgetItem(thumb, f"第 {i + 1} 页")
            self.page_list.addItem(item)
        if doc.pages:
            self.page_list.setCurrentRow(0)

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

    def _on_page_changed(self, row: int) -> None:
        if self.doc is None or row < 0 or row >= len(self.doc.pages):
            return
        self._sync_canvas_to_records()
        self.current_page = row
        page = self.doc.pages[row]
        self.canvas.show_page(page.image, page.phys_w_mm, page.phys_h_mm)
        for rec in self.stamps.get(row, []):
            self.canvas.add_stamp(self._make_stamp_item(rec))
        self._refresh_page_thumbnail(row)

    def _refresh_page_thumbnail(self, row: int) -> None:
        if self.doc is None:
            return
        item = self.page_list.item(row)
        if item:
            item.setIcon(np_rgb_to_qpixmap(_thumbnail(self.doc.pages[row].image, 120)))

    # ── 盖章交互 ──

    def _add_stamp(self, seal: Seal) -> None:
        """盖章：印章跟随鼠标，在页面上点哪盖哪（修改意见）。"""
        if self.doc is None or self.current_page < 0:
            QMessageBox.information(self, "提示", "请先打开合同文件")
            return
        # 落章即采样随机效果，预览立即所见即所得
        processed, applied = self._session_rng.apply_auto(
            seal.image, self.panel.random_spec()
        )
        self._pending_rec = StampRecord(
            seal=seal,
            center_x_mm=0.0,
            center_y_mm=0.0,
            size_mm=seal.phys_mm,
            applied=applied,
            processed=processed,
        )
        self.canvas.start_follow(processed, seal.phys_mm)
        self.info_label.setText(f"「{seal.name}」跟随鼠标中——在页面上点击落位，Esc 取消")

    def _on_stamp_placed(self, x_mm: float, y_mm: float) -> None:
        rec = getattr(self, "_pending_rec", None)
        if rec is None:
            self.canvas.cancel_follow()
            return
        rec.center_x_mm = x_mm
        rec.center_y_mm = y_mm
        self.canvas.cancel_follow()
        self.stamps.setdefault(self.current_page, []).append(rec)
        item = self._make_stamp_item(rec)
        self.canvas.add_stamp(item)
        self.canvas.centerOn(item)
        self._pending_rec = None
        self._update_info(rec)

    def _on_follow_cancelled(self) -> None:
        self._pending_rec = None
        self.info_label.setText("已取消盖章")

    def _on_canvas_clicked(self, item, x_mm: float, y_mm: float) -> None:
        """空白处单击：把候选章移过去（press 时记录的章，不依赖 release 时的选中态）。"""
        rec = self._find_record(item)
        if rec is None:
            return
        rec.center_x_mm = x_mm
        rec.center_y_mm = y_mm
        item.set_center(x_mm, y_mm)
        self._update_info(rec)
        self._sync_adjust_spins(rec)

    def _on_place_rejected(self) -> None:
        self.info_label.setText("请点击页面范围内落章（章整个飞出纸面会被忽略）")

    def _add_perforation(self, seal: Seal) -> None:
        """骑缝章：对话框 → 拼合预览确认 → 切片作为 locked 记录落到各页。"""
        if self.doc is None:
            QMessageBox.information(self, "提示", "请先打开合同文件")
            return
        seed = secrets.randbelow(2**31 - 1)
        # 印章先过一遍全局随机效果（角度/色度/蒙尘），再切割
        rng = Randomizer(seed)
        processed_seal, _applied = rng.apply_auto(seal.image, self.panel.random_spec())

        dlg = PerforationDialog(
            self, self.doc.pages, processed_seal, seal.name, seal.phys_mm, seed
        )
        if dlg.exec() != QDialog.Accepted or dlg.placements is None:
            return

        group = f"perf_{seed}"
        touched_pages = set()
        for pl in dlg.placements:
            page = self.doc.pages[pl.page_index]
            h_px, w_px = pl.slice_rgba.shape[:2]
            w_mm = w_px / page.dpi * 25.4
            h_mm = h_px / page.dpi * 25.4
            rec = StampRecord(
                seal=seal,
                center_x_mm=pl.right_edge_mm - w_mm / 2,
                center_y_mm=pl.top_mm + h_mm / 2,
                size_mm=w_mm,
                rotation_deg=0.0,
                opacity=1.0,
                applied=None,
                processed=pl.slice_rgba,
                locked=True,
                group=group,
            )
            self.stamps.setdefault(pl.page_index, []).append(rec)
            touched_pages.add(pl.page_index)
        # 当前页不在骑缝范围内时，跳到第一个受影响的页，否则用户看不到任何变化
        target_page = self.current_page if self.current_page in touched_pages else min(touched_pages)
        self.page_list.setCurrentRow(target_page)
        # 刷新当前页显示
        self._on_page_changed(self.current_page)
        # 定位到第一个切片并放大，让细条带直接可见（骑缝可见性）
        first_page = min(touched_pages)
        if first_page == self.current_page:
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
        kind = "⌀" if rec.seal.kind == KIND_SEAL else "宽"
        lock = "（骑缝切片）" if rec.locked else ""
        self.info_label.setText(
            f"{rec.seal.name}{lock}：中心 ({rec.center_x_mm:.1f}, {rec.center_y_mm:.1f}) mm，"
            f"{kind}{rec.size_mm:.1f}mm，旋转 {rec.rotation_deg:.1f}°"
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
        doomed = {it.data(0) for it in items}
        self.stamps[self.current_page] = [
            r for r in self.stamps.get(self.current_page, []) if id(r) not in doomed
        ]

    def _move_page(self, delta: int) -> None:
        """页序调整：当前页上移/下移一位，盖章记录随页面走（修改意见 #1）。"""
        if self.doc is None or self.current_page < 0:
            return
        old = self.current_page
        new = old + delta
        if new < 0 or new >= len(self.doc.pages):
            return
        self._sync_canvas_to_records()
        # 交换页面与各自的盖章记录
        self.doc.pages[old], self.doc.pages[new] = self.doc.pages[new], self.doc.pages[old]
        self.stamps[old], self.stamps[new] = self.stamps.get(old, []), self.stamps.get(new, [])
        self.stamps = {k: v for k, v in self.stamps.items() if v}
        self.current_page = -1  # 强制重建画布
        self.page_list.setCurrentRow(new)
        # 两个位置的缩略图都要刷新
        self._refresh_page_thumbnail(old)
        self._refresh_page_thumbnail(new)

    # ── 页面操作（M3）──

    def _rotate_page(self, k: int) -> None:
        """旋转当前页图像与物理尺寸，印章坐标联动。

        k: np.rot90 的次数（1=逆时针90, -1=顺时针90, 2=180）。
        """
        if self.doc is None or self.current_page < 0:
            return
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
        self._on_page_changed(self.current_page)

    def _four_point_calibrate(self) -> None:
        """四点纸边校准：用户在页面上点纸面四个角点（修改意见）。

        扫描件自带白边 → 页面边界 ≠ 纸张边界。点完四个顶点后按透视变换
        把页面拉伸裁正为标准 A4，白边消失、物理尺寸精确。
        """
        if self.doc is None or self.current_page < 0:
            return
        self.info_label.setText("四点校准：请依次点击纸面的 4 个角点（顺序随意），Esc 取消")
        self.canvas.start_pick_points()

    def _on_pick_cancelled(self) -> None:
        self.info_label.setText("已取消四点校准")

    def _on_points_picked(self, pts_mm: list) -> None:
        page = self.doc.pages[self.current_page]
        old_w_mm, old_h_mm = page.phys_w_mm, page.phys_h_mm
        old_dpi = page.dpi
        # 场景 mm → 页面像素
        quad_px = np.array(
            [[x / 25.4 * old_dpi, y / 25.4 * old_dpi] for x, y in pts_mm],
            dtype=np.float32,
        )
        # 先在副本上试变换，用户确认后再应用
        from core.autocal import map_points_through, warp_to_a4

        trial = Page(image=page.image, phys_w_mm=old_w_mm, phys_h_mm=old_h_mm)
        H = warp_to_a4(trial, quad_px)
        if H is None:
            QMessageBox.warning(
                self, "四点无效",
                "四个点围成的区域太小（可能点挤在一起或几乎共线），请重新点取纸面四角。",
            )
            self.canvas.cancel_pick()
            return
        # 预览确认
        dlg = _WarpPreviewDialog(self, trial.image)
        if dlg.exec() != QDialog.Accepted:
            self.canvas.cancel_pick()
            self._on_page_changed(self.current_page)
            return
        # 应用：页面图像与物理尺寸已被 trial 变换，转移回正式 page
        # 已盖章坐标映射：旧 mm → 旧像素 →(H)→ 新像素 → 新 mm
        new_dpi = trial.dpi
        recs = self.stamps.get(self.current_page, [])
        if recs:
            old_pts = np.array(
                [[r.center_x_mm / 25.4 * old_dpi, r.center_y_mm / 25.4 * old_dpi] for r in recs],
                dtype=np.float32,
            )
            new_pts = map_points_through(H, old_pts)
            for r, (nx, ny) in zip(recs, new_pts):
                r.center_x_mm = float(nx) / new_dpi * 25.4
                r.center_y_mm = float(ny) / new_dpi * 25.4
        page.image = trial.image
        page.phys_w_mm, page.phys_h_mm = trial.phys_w_mm, trial.phys_h_mm
        page.needs_calibration = False
        self.canvas.cancel_pick()
        self._on_page_changed(self.current_page)
        self.info_label.setText("四点校准完成：页面已拉伸为标准 A4（210×297mm）")

    def _calibrate(self) -> None:
        if self.doc is None:
            return
        page = self.doc.pages[max(self.current_page, 0)]
        dlg = _CalibrateDialog(self, page.phys_w_mm, page.phys_h_mm)
        if dlg.exec() != QDialog.Accepted:
            return
        w_mm, h_mm, apply_all = dlg.values()
        targets = self.doc.pages if apply_all else [page]
        for p in targets:
            calibrate_paper_edge(p, w_mm, h_mm)
        self._on_page_changed(self.current_page)

    def _auto_calibrate(self) -> None:
        if self.doc is None:
            return
        ok, fail = 0, []
        for i, page in enumerate(self.doc.pages):
            if auto_calibrate_page(page):
                ok += 1
            else:
                fail.append(i + 1)
        self._on_page_changed(self.current_page)
        msg = f"{ok} 页检测并校正成功。"
        if fail:
            msg += f"\n第 {fail} 页未检测到纸边，请改用手动校准。"
        QMessageBox.information(self, "自动纸边检测", msg)

    # ── 删除与骑缝组管理 ──

    def _show_about(self) -> None:
        from core import __version__

        QMessageBox.about(
            self,
            "关于 合同盖章工具",
            f"<h3>合同盖章工具</h3>"
            f"<p>版本 v{__version__}</p>"
            f"<p>扫描合同盖章：物理尺寸 1:1、骑缝章、手写签名。<br>"
            f"全程本地离线运行，不修改原始文件。</p>"
            f"<p>源码：github.com/kimhero110/contract-sealer<br>"
            f"gitee.com/xu512/contract-sealer</p>"
            f"<p style='color:gray'>仅限本单位已授权印章的内部流程使用。</p>",
        )

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
        self.info_label.setText("已删除")

    def _delete_group(self) -> None:
        """删除整组骑缝章（跨所有页）。"""
        rec = self._selected_record()
        if rec is None or not rec.locked or not rec.group:
            return
        gid = rec.group
        n = 0
        for page_idx in list(self.stamps.keys()):
            before = len(self.stamps[page_idx])
            self.stamps[page_idx] = [r for r in self.stamps[page_idx] if r.group != gid]
            n += before - len(self.stamps[page_idx])
        self.stamps = {k: v for k, v in self.stamps.items() if v}
        self._on_page_changed(self.current_page)  # 重建画布，清掉切片贴图
        self.info_label.setText(f"已删除整组骑缝章（{n} 个切片）")

    def _apply_group_shift(self) -> None:
        """骑缝组竖向整体微调：所有切片统一下移/上移，保留逐页抖动差。"""
        rec = self._selected_record()
        if rec is None or not rec.locked or not rec.group:
            return
        delta = self.group_shift_spin.value()
        if abs(delta) < 1e-9:
            return
        gid = rec.group
        for records in self.stamps.values():
            for r in records:
                if r.group == gid:
                    r.center_y_mm += delta
        # 先让当前页画布贴图跟上记录，再刷新——否则 _on_page_changed 的
        # 位置回同步会把当前页的记录改回贴图旧位置（记录在贴图之前更新会被覆盖）
        for item in self.canvas.stamps():
            r2 = self._find_record(item)
            if r2 is not None and r2.group == gid:
                item.set_center(r2.center_x_mm, r2.center_y_mm)
        self.group_shift_spin.setValue(0.0)
        self._on_page_changed(self.current_page)
        self.info_label.setText(f"整组骑缝章已竖向移动 {delta:+.1f}mm（逐页抖动保持）")

    # ── 模板（M3）──

    def _reroll_random(self) -> None:
        """「换一批手感」：重摇所有普通章/签名的随机效果（骑缝切片除外）。

        骑缝切片是一整枚章切出来的，参数未保留，无法重摇——
        需要换手感请重新打开骑缝章对话框。
        """
        count = 0
        rng = Randomizer(secrets.randbelow(2**31 - 1))
        spec = self.panel.random_spec()
        for page_idx, records in self.stamps.items():
            for rec in records:
                if rec.locked:
                    continue
                rec.processed, rec.applied = rng.apply_auto(rec.seal.image, spec)
                count += 1
        # 刷新当前页画布上的贴图
        for item in self.canvas.stamps():
            rec = self._find_record(item)
            if rec is not None and not rec.locked and rec.processed is not None:
                item.set_pixmap_rgba(rec.processed)
        self.info_label.setText(f"已为 {count} 枚章/签名更换手感（骑缝切片不受影响）")

    def _save_template(self) -> None:
        if self.doc is None or self.current_page < 0:
            return
        self._sync_canvas_to_records()
        records = self.stamps.get(self.current_page, [])
        page = self.doc.pages[self.current_page]
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
        actions = self.tpl_menu.actions()
        for a in actions[2:]:
            self.tpl_menu.removeAction(a)
        for path in list_templates(default_template_dir()):
            act = QAction(path.stem, self)
            act.triggered.connect(lambda checked=False, p=path: self._apply_template(p))
            self.tpl_menu.addAction(act)

    def _apply_template(self, path: Path, target_page: int | None = None) -> int:
        """把模板套用到目标页（默认当前页）。返回套用的章数。"""
        if self.doc is None:
            return 0
        page_idx = self.current_page if target_page is None else target_page
        if page_idx < 0:
            return 0
        page = self.doc.pages[page_idx]
        entries = load_template(path)
        seals_by_name = self._library_seals_by_name()
        count = 0
        for e in entries:
            seal = seals_by_name.get(e["seal_name"])
            if seal is None:
                continue
            rec = StampRecord(
                seal=seal,
                center_x_mm=e["rel_x"] * page.phys_w_mm,
                center_y_mm=e["rel_y"] * page.phys_h_mm,
                size_mm=e["size_mm"],
                rotation_deg=e.get("rotation_deg", 0.0),
                opacity=e.get("opacity", 1.0),
            )
            rec.processed, rec.applied = self._session_rng.apply_auto(
                seal.image, self.panel.random_spec()
            )
            self.stamps.setdefault(page_idx, []).append(rec)
            count += 1
        if target_page is None and count:
            self._on_page_changed(self.current_page)
        return count

    def _library_seals_by_name(self) -> dict[str, Seal]:
        out = {}
        for png in list_library(default_library_dir()):
            try:
                seal = Seal.load(png)
            except Exception:
                continue
            out[seal.name] = seal
        return out

    # ── 批量导出（M3）──

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
            self, "选择要批量盖章的合同（可多选）", "",
            "合同文件 (*.pdf *.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not paths:
            return
        target_last = QMessageBox.question(
            self, "盖章位置", "把模板盖在每个文件的【最后一页】吗？\n（选 No 则盖第一页）"
        )

        done, failed = [], []
        for path in paths:
            try:
                self._load_document([path])
                page_idx = len(self.doc.pages) - 1 if target_last == QMessageBox.Yes else 0
                count = self._apply_template(tpl_path, target_page=page_idx)
                if count == 0:
                    failed.append((path, "模板中的印章不在库中"))
                    continue
                out = self._export_document()
                done.append(out)
            except Exception as e:
                failed.append((path, str(e)))

        msg = f"成功 {len(done)} 个：\n" + "\n".join(str(p) for p in done[:10])
        if failed:
            msg += f"\n\n失败 {len(failed)} 个：\n" + "\n".join(
                f"{p}: {e}" for p, e in failed[:5]
            )
        QMessageBox.information(self, "批量导出完成", msg)
        # 批量结束后清空当前文档，避免误操作
        self.doc = None
        self.stamps = {}
        self.page_list.clear()

    # ── 导出 ──

    def _export_document(self) -> Path:
        """把当前 doc + stamps 导出为 PDF（批量与单次导出共用）。返回输出路径。

        所见即所得：所有章直接使用落章时已采样的 processed 图像，
        导出不做任何重采样（v0.2.1 起废除"每次导出新种子"——画布显示
        什么就导出什么；想换手感用「换一批手感」按钮主动重摇）。
        """
        spec = self.panel.random_spec()
        stamp_logs = []
        for page_idx, records in self.stamps.items():
            for rec in records:
                stamp_logs.append(
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
                )

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

        out_dir = Path(self.doc.source_path).parent if self.doc.source_path else Path.cwd()
        out_path = make_output_path(self.doc.source_path, out_dir)
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
            ret = QMessageBox.question(self, "确认", "当前没有任何盖章，仍要导出吗？")
            if ret != QMessageBox.Yes:
                return
        try:
            out_path = self._export_document()
        except Exception as e:
            QMessageBox.warning(self, "导出失败", str(e))
            return
        # 预览刷新为导出同源结果（所见即所得）
        self._on_page_changed(self.current_page)
        QMessageBox.information(
            self, "导出完成", f"已导出：\n{out_path}\n\n随机种子与参数已写入同名 .sealog"
        )


class _WarpPreviewDialog(QDialog):
    """四点校准预览：显示拉伸裁正后的 A4 页面，确认才应用。"""

    def __init__(self, parent, warped_img: np.ndarray):
        super().__init__(parent)
        self.setWindowTitle("四点校准预览")
        layout = QVBoxLayout(self)
        hint = QLabel("页面将按你点的四个角拉伸为标准 A4，效果如下。确认应用吗？")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        pm = np_rgb_to_qpixmap(warped_img)
        if pm.height() > 640:
            pm = pm.scaledToHeight(640, Qt.SmoothTransformation)
        label = QLabel()
        label.setPixmap(pm)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class _CalibrateDialog(QDialog):
    """纸边校准（M1 手动路径）：输入纸张真实物理尺寸。"""

    def __init__(self, parent, cur_w: float, cur_h: float):
        super().__init__(parent)
        self.setWindowTitle("纸边校准")
        form = QFormLayout(self)
        self.w_spin = QDoubleSpinBox()
        self.w_spin.setRange(20.0, 2000.0)
        self.w_spin.setSuffix(" mm")
        self.w_spin.setValue(cur_w)
        self.h_spin = QDoubleSpinBox()
        self.h_spin.setRange(20.0, 2000.0)
        self.h_spin.setSuffix(" mm")
        self.h_spin.setValue(cur_h)
        form.addRow("纸张宽", self.w_spin)
        form.addRow("纸张高", self.h_spin)
        hint = QLabel("用尺子量扫描件上的纸张实际尺寸（A4 = 210 × 297mm）。")
        hint.setStyleSheet("color: gray;")
        form.addRow(hint)
        from PySide6.QtWidgets import QCheckBox

        self.all_check = QCheckBox("应用到所有页面")
        self.all_check.setChecked(True)
        form.addRow(self.all_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[float, float, bool]:
        return self.w_spin.value(), self.h_spin.value(), self.all_check.isChecked()


def _thumbnail(img: np.ndarray, width: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def _natural_key(path: str):
    import re

    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]
