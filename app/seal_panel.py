"""印章/签名面板：库管理、导入抠图、随机强度、落章与导出参数。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from core.extract import KIND_SEAL, KIND_SIGNATURE, detect_kind, extract_ink
from core.randomize import RandomSpec
from core.seal import (
    DEFAULT_SEAL_DIAMETER_MM,
    DEFAULT_SIGNATURE_WIDTH_MM,
    Seal,
    default_library_dir,
    list_library,
)


class SealPanel(QWidget):
    """右侧栏。信号驱动，不直接操作画布。"""

    stamp_requested = Signal(object)   # Seal
    perforation_requested = Signal(object)  # Seal（骑缝章）
    export_requested = Signal()
    random_changed = Signal(object)    # RandomSpec

    def __init__(self, parent=None):
        super().__init__(parent)
        self.library_dir = default_library_dir()
        self._seals: dict[int, Seal] = {}
        self._build_ui()
        self.reload_library()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── 印章库 ──
        lib_group = QGroupBox("印章 / 签名库")
        lib_layout = QVBoxLayout(lib_group)
        self.lib_list = QListWidget()
        self.lib_list.currentRowChanged.connect(self._on_select)
        lib_layout.addWidget(self.lib_list)
        btns = QHBoxLayout()
        self.btn_import = QPushButton("导入图片…")
        self.btn_import.clicked.connect(self._import_seal)
        self.btn_stamp = QPushButton("盖到当前页")
        self.btn_stamp.clicked.connect(self._request_stamp)
        self.btn_perforation = QPushButton("骑缝章…")
        self.btn_perforation.clicked.connect(self._request_perforation)
        self.btn_delete = QPushButton("删除")
        self.btn_delete.clicked.connect(self._delete_seal)
        btns.addWidget(self.btn_import)
        btns.addWidget(self.btn_stamp)
        btns.addWidget(self.btn_perforation)
        btns.addWidget(self.btn_delete)
        lib_layout.addLayout(btns)
        self.lib_path_label = QLabel(f"库位置：{self.library_dir}")
        self.lib_path_label.setWordWrap(True)
        self.lib_path_label.setStyleSheet("color: gray; font-size: 10px;")
        lib_layout.addWidget(self.lib_path_label)
        layout.addWidget(lib_group)

        # ── 随机效果 ──
        rnd_group = QGroupBox("随机效果（每次导出略有不同，更像真章）")
        rnd_layout = QFormLayout(rnd_group)
        self.slider_angle = self._make_slider(0, 50, 20)   # 0-5.0°
        self.slider_tone = self._make_slider(0, 25, 10)    # 0-25%
        self.slider_dust = self._make_slider(0, 50, 15)    # 0-50%
        rnd_layout.addRow("角度 ±°", self.slider_angle)
        rnd_layout.addRow("色度 ±%", self.slider_tone)
        rnd_layout.addRow("蒙尘", self.slider_dust)
        for s in (self.slider_angle, self.slider_tone, self.slider_dust):
            s.valueChanged.connect(self._emit_random)
        layout.addWidget(rnd_group)

        # ── 选中印章信息 ──
        self.info_label = QLabel("未选中印章")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        layout.addStretch(1)
        self.btn_export = QPushButton("导出已盖章 PDF…")
        self.btn_export.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_export.clicked.connect(self.export_requested)
        layout.addWidget(self.btn_export)

    def _make_slider(self, lo: int, hi: int, val: int) -> QSlider:
        s = QSlider()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    def random_spec(self) -> RandomSpec:
        return RandomSpec(
            angle_deg=self.slider_angle.value() / 10.0,
            tone=self.slider_tone.value() / 100.0,
            dust=self.slider_dust.value() / 100.0,
        )

    def _emit_random(self) -> None:
        self.random_changed.emit(self.random_spec())

    # ── 库管理 ──

    def reload_library(self) -> None:
        self.lib_list.clear()
        self._seals.clear()
        for i, png in enumerate(list_library(self.library_dir)):
            try:
                seal = Seal.load(png)
            except Exception as e:  # 单个坏文件不拖垮整个库
                print(f"跳过损坏的印章文件 {png}: {e}")
                continue
            kind_label = "印章" if seal.kind == KIND_SEAL else "签名"
            size_label = f"⌀{seal.phys_mm:g}mm" if seal.kind == KIND_SEAL else f"宽{seal.phys_mm:g}mm"
            item = QListWidgetItem(f"{seal.name}（{kind_label} {size_label}）")
            self.lib_list.addItem(item)
            self._seals[i] = seal

    def _on_select(self, row: int) -> None:
        seal = self._seals.get(row)
        if seal:
            kind_label = "印章" if seal.kind == KIND_SEAL else "签名"
            self.info_label.setText(f"已选：{seal.name}（{kind_label}，物理尺寸 {seal.phys_mm:g}mm）")

    def current_seal(self) -> Seal | None:
        return self._seals.get(self.lib_list.currentRow())

    def _request_stamp(self) -> None:
        seal = self.current_seal()
        if seal is None:
            QMessageBox.information(self, "提示", "请先在库中选择一枚印章或签名")
            return
        self.stamp_requested.emit(seal)

    def _request_perforation(self) -> None:
        seal = self.current_seal()
        if seal is None:
            QMessageBox.information(self, "提示", "请先在库中选择一枚印章")
            return
        self.perforation_requested.emit(seal)

    def _delete_seal(self) -> None:
        seal = self.current_seal()
        if seal is None:
            return
        if QMessageBox.question(self, "确认", f"从库中删除「{seal.name}」？") == QMessageBox.Yes:
            from core.seal import _safe_slug

            for ext in (".png", ".json"):
                p = self.library_dir / f"{_safe_slug(seal.name)}{ext}"
                if p.exists():
                    p.unlink()
            self.reload_library()

    def _import_seal(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择印章/签名图片", "", "图片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        for path in paths:
            self._import_one(Path(path))
        self.reload_library()

    def _import_one(self, path: Path) -> None:
        try:
            kind = detect_kind(path)
            rgba = extract_ink(path, kind=kind)
        except Exception as e:
            QMessageBox.warning(self, "导入失败", f"{path.name}：{e}")
            return

        dlg = _ImportDialog(self, path.stem, kind)
        if dlg.exec() != QDialog.Accepted:
            return
        name, kind, phys_mm = dlg.values()
        seal = Seal(name=name, kind=kind, image=rgba, phys_mm=phys_mm)
        try:
            seal.save(self.library_dir)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))


class _ImportDialog(QDialog):
    """导入确认：名称、类型、真实物理尺寸（比例换算的基准，方案 §4.3）。"""

    def __init__(self, parent, default_name: str, kind: str):
        super().__init__(parent)
        self.setWindowTitle("导入印章/签名")
        form = QFormLayout(self)
        self.name_edit = QLineEdit(default_name)
        self.kind_combo = QComboBox()
        self.kind_combo.addItem("印章（按直径）", KIND_SEAL)
        self.kind_combo.addItem("签名（按宽度）", KIND_SIGNATURE)
        self.kind_combo.setCurrentIndex(0 if kind == KIND_SEAL else 1)
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(5.0, 200.0)
        self.size_spin.setSuffix(" mm")
        self.size_spin.setValue(
            DEFAULT_SEAL_DIAMETER_MM if kind == KIND_SEAL else DEFAULT_SIGNATURE_WIDTH_MM
        )
        self.kind_combo.currentIndexChanged.connect(
            lambda i: self.size_spin.setValue(
                DEFAULT_SEAL_DIAMETER_MM if i == 0 else DEFAULT_SIGNATURE_WIDTH_MM
            )
        )
        form.addRow("名称", self.name_edit)
        form.addRow("类型", self.kind_combo)
        form.addRow("真实物理尺寸", self.size_spin)
        hint = QLabel("物理尺寸决定盖出来的章和真章一样大。\n印章量直径（公章常见 40/42mm），签名量宽度。")
        hint.setStyleSheet("color: gray;")
        hint.setWordWrap(True)
        form.addRow(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[str, str, float]:
        return (
            self.name_edit.text().strip() or "未命名",
            self.kind_combo.currentData(),
            self.size_spin.value(),
        )
