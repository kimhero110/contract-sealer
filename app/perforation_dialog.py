"""骑缝章对话框：参数设置 + 拼合预览（导出前强制确认，方案 §4.4）。"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from app.canvas import np_rgb_to_qpixmap
from core.document import Page
from core.perforation import (
    SIDE_LEFT,
    SIDE_RIGHT,
    PerforationSpec,
    SlicePlacement,
    assemble_preview,
    min_slice_warning,
    plan_perforation,
)


class PerforationDialog(QDialog):
    """返回 (placements, spec)。拼合预览按钮可反复查看。"""

    def __init__(self, parent, pages: list[Page], seal_rgba: np.ndarray, seal_name: str, seal_diameter: float, seed: int):
        super().__init__(parent)
        self.setWindowTitle(f"骑缝章 —— {seal_name}")
        self._pages = pages
        self._seal_rgba = seal_rgba
        self._seed = seed
        self.placements: list[SlicePlacement] | None = None
        self.spec: PerforationSpec | None = None

        form = QFormLayout(self)
        n = len(pages)
        self.from_spin = QSpinBox()
        self.from_spin.setRange(1, n)
        self.from_spin.setValue(1)
        self.to_spin = QSpinBox()
        self.to_spin.setRange(1, n)
        self.to_spin.setValue(n)
        form.addRow("起始页", self.from_spin)
        form.addRow("结束页", self.to_spin)

        self.side_combo = QComboBox()
        self.side_combo.addItem("右侧", SIDE_RIGHT)
        self.side_combo.addItem("左侧", SIDE_LEFT)
        form.addRow("侧边", self.side_combo)

        self.inset_spin = QDoubleSpinBox()
        self.inset_spin.setRange(0.0, 20.0)
        self.inset_spin.setValue(1.5)
        self.inset_spin.setSuffix(" mm")
        form.addRow("切片内缩", self.inset_spin)

        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(0.0, 999.0)
        self.y_spin.setValue(0.0)
        self.y_spin.setSuffix(" mm")
        self.y_spin.setSpecialValueText("垂直居中")
        form.addRow("距页顶", self.y_spin)

        self.diameter_spin = QDoubleSpinBox()
        self.diameter_spin.setRange(10.0, 100.0)
        self.diameter_spin.setValue(seal_diameter)
        self.diameter_spin.setSuffix(" mm")
        form.addRow("印章直径", self.diameter_spin)

        self.jitter_slider = QSlider(Qt.Horizontal)
        self.jitter_slider.setRange(0, 100)
        self.jitter_slider.setValue(50)
        form.addRow("随机强度", self.jitter_slider)

        self.warn_label = QLabel("")
        self.warn_label.setStyleSheet("color: #b00;")
        self.warn_label.setWordWrap(True)
        form.addRow(self.warn_label)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.preview_btn = QPushButton("拼合预览…")
        self.preview_btn.clicked.connect(self._preview)
        btns.addButton(self.preview_btn, QDialogButtonBox.ActionRole)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        for w in (self.from_spin, self.to_spin, self.diameter_spin):
            w.valueChanged.connect(self._update_warning)
        self._update_warning()

    def _page_indices(self) -> list[int]:
        a, b = self.from_spin.value(), self.to_spin.value()
        if a > b:
            a, b = b, a
        return list(range(a - 1, b))

    def _jitter_scale(self) -> float:
        return self.jitter_slider.value() / 100.0

    def _make_spec(self) -> PerforationSpec:
        j = self._jitter_scale()
        y = self.y_spin.value()
        return PerforationSpec(
            diameter_mm=self.diameter_spin.value(),
            side=self.side_combo.currentData(),
            inset_mm=self.inset_spin.value(),
            y_mm=None if y <= 0 else y,
            width_jitter=0.4 * j,
            offset_jitter_mm=1.0 * j,
            rot_jitter_deg=2.0 * j,
            seed=self._seed,
        )

    def _update_warning(self) -> None:
        n = len(self._page_indices())
        warn = min_slice_warning(self.diameter_spin.value(), n)
        self.warn_label.setText(warn or "")

    def _plan(self) -> list[SlicePlacement]:
        self.spec = self._make_spec()
        return plan_perforation(self._seal_rgba, self._pages, self._page_indices(), self.spec)

    def _preview(self) -> None:
        try:
            placements = self._plan()
        except ValueError as e:
            QMessageBox.warning(self, "无法生成", str(e))
            return
        ref_dpi = self._pages[self._page_indices()[0]].dpi
        img = assemble_preview(placements, ref_dpi)
        PreviewDialog(self, img, len(placements)).exec()

    def _on_accept(self) -> None:
        try:
            self.placements = self._plan()
        except ValueError as e:
            QMessageBox.warning(self, "无法应用", str(e))
            return
        # 方案 §4.4：拼合预览强制确认后才允许应用
        ret = QMessageBox.question(
            self,
            "确认骑缝章",
            f"将在 {len(self.placements)} 页上放置骑缝章切片。\n"
            "建议先点「拼合预览」确认多页拼合能还原完整印文。\n\n确定应用吗？",
        )
        if ret == QMessageBox.Yes:
            self.accept()


class PreviewDialog(QDialog):
    """拼合预览：按真实导出结果渲染全部切片的拼接图。"""

    def __init__(self, parent, img: np.ndarray, page_count: int):
        super().__init__(parent)
        self.setWindowTitle(f"拼合预览（{page_count} 页切片按页序拼接）")
        layout = QVBoxLayout(self)
        hint = QLabel("以下为导出后各页骑缝条带拼接的真实效果，请确认印文连续可辨：")
        layout.addWidget(hint)
        pm = np_rgb_to_qpixmap(img)
        if pm.width() > 900:
            pm = pm.scaledToWidth(900, Qt.SmoothTransformation)
        label = QLabel()
        label.setPixmap(pm)
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        btns.accepted.connect(self.accept)
        layout.addWidget(btns)
        self.resize(960, 420)
