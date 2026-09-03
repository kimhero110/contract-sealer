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
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
        splitter.addWidget(self.canvas)

        right = QWidget()
        right_layout = QHBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        panel_container = QWidget()
        v = QVBoxLayout(panel_container)
        self.panel = SealPanel()
        self.panel.stamp_requested.connect(self._add_stamp)
        self.panel.perforation_requested.connect(self._add_perforation)
        self.panel.export_requested.connect(self._export)
        v.addWidget(self.panel)

        # 选中印章的微调控件
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
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        v.addWidget(self.info_label)

        right_layout.addWidget(panel_container)
        right.setMaximumWidth(360)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

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
        if self.doc is None or self.current_page < 0:
            QMessageBox.information(self, "提示", "请先打开合同文件")
            return
        page = self.doc.pages[self.current_page]
        rec = StampRecord(
            seal=seal,
            center_x_mm=page.phys_w_mm / 2,
            center_y_mm=page.phys_h_mm * 0.75,
            size_mm=seal.phys_mm,
        )
        # 落章即采样随机效果，预览立即所见即所得
        rec.processed, rec.applied = self._session_rng.apply_auto(
            seal.image, self.panel.random_spec()
        )
        self.stamps.setdefault(self.current_page, []).append(rec)
        self.canvas.add_stamp(self._make_stamp_item(rec))
        self._update_info(rec)

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
        # 刷新当前页显示
        self._on_page_changed(self.current_page)
        QMessageBox.information(
            self, "骑缝章已应用", f"已在 {len(touched_pages)} 页放置切片（组 {group}）。"
        )

    def _make_stamp_item(self, rec: StampRecord) -> StampItem:
        rgba = rec.processed if rec.processed is not None else rec.seal.image
        item = StampItem(rgba, rec.size_mm, rec.center_x_mm, rec.center_y_mm)
        item.setRotation(rec.rotation_deg)
        item.setOpacity(rec.opacity)
        item.setData(0, id(rec))  # 画布回同步时定位记录
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
            self.info_label.setText("选中页面上的印章后可微调尺寸/旋转/不透明度")
            return
        kind = "⌀" if rec.seal.kind == KIND_SEAL else "宽"
        lock = "（骑缝切片）" if rec.locked else ""
        self.info_label.setText(
            f"{rec.seal.name}{lock}：中心 ({rec.center_x_mm:.1f}, {rec.center_y_mm:.1f}) mm，"
            f"{kind}{rec.size_mm:.1f}mm，旋转 {rec.rotation_deg:.1f}°"
        )

    def _sync_canvas_to_records(self) -> None:
        """翻页/导出前：把画布上的印章状态写回记录。"""
        if self.current_page < 0:
            return
        for item in self.canvas.stamps():
            rec = self._find_record(item)
            if rec is None:
                continue
            rec.center_x_mm, rec.center_y_mm = item.center()
        # 删除在画布上已不存在的记录（用户在画布上按了 Delete）
        alive = {item.data(0) for item in self.canvas.stamps()}
        self.stamps[self.current_page] = [
            r for r in self.stamps.get(self.current_page, []) if id(r) in alive
        ]

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

    # ── 模板（M3）──

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
        """把当前 doc + stamps 导出为 PDF（批量与单次导出共用）。返回输出路径。"""
        # 每次导出一个新种子；重采样未锁定印章的随机效果（方案 §4.9）
        seed = secrets.randbelow(2**31 - 1)
        rng = Randomizer(seed)
        spec = self.panel.random_spec()
        stamp_logs = []
        for page_idx, records in self.stamps.items():
            for rec in records:
                if not rec.locked:
                    rec.processed, rec.applied = rng.apply_auto(rec.seal.image, spec)
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
            "seed": seed,
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
