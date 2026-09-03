"""骑缝章对话框：参数设置 + 拼合预览 + 页面效果预览（可点选位置）。

修改意见 #4：预览要渲染切片盖在真实页面上的实际效果，并允许操作者直接
在页面预览图上点选垂直位置。
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
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
    apply_perforation,
    assemble_preview,
    min_slice_warning,
    plan_perforation,
)

# 垂直位置预设：印章中心位于页面高度的比例
Y_PRESETS = {
    "垂直居中": 0.50,
    "偏上（1/4 处）": 0.25,
    "偏下（3/4 处）": 0.75,
    "自定义 mm": None,
}


class PerforationDialog(QDialog):
    """返回 (placements, spec)。拼合预览/页面效果预览可反复查看。"""

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

        # 垂直位置：预设 + 自定义 mm（修改意见 #4）
        y_row = QHBoxLayout()
        self.y_preset = QComboBox()
        for label in Y_PRESETS:
            self.y_preset.addItem(label)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(0.0, 999.0)
        self.y_spin.setValue(0.0)
        self.y_spin.setSuffix(" mm")
        self.y_spin.setSpecialValueText("自动")
        self.y_spin.setEnabled(False)
        self.y_preset.currentTextChanged.connect(self._on_preset_changed)
        y_row.addWidget(self.y_preset)
        y_row.addWidget(self.y_spin)
        form.addRow("垂直位置", y_row)

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
        self.effect_btn = QPushButton("页面效果预览…")
        self.effect_btn.clicked.connect(self._effect_preview)
        self.preview_btn = QPushButton("拼合预览…")
        self.preview_btn.clicked.connect(self._preview)
        btns.addButton(self.effect_btn, QDialogButtonBox.ActionRole)
        btns.addButton(self.preview_btn, QDialogButtonBox.ActionRole)
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        form.addRow(btns)

        for w in (self.from_spin, self.to_spin, self.diameter_spin):
            w.valueChanged.connect(self._update_warning)
        self._update_warning()

    def _on_preset_changed(self, label: str) -> None:
        self.y_spin.setEnabled(Y_PRESETS[label] is None)

    def _page_indices(self) -> list[int]:
        a, b = self.from_spin.value(), self.to_spin.value()
        if a > b:
            a, b = b, a
        return list(range(a - 1, b))

    def _resolve_y_mm(self) -> float | None:
        """把预设/自定义换算成切片顶部 y（mm）。"""
        label = self.y_preset.currentText()
        frac = Y_PRESETS[label]
        idx = self._page_indices()
        ref_page = self._pages[idx[0]]
        if frac is None:
            return self.y_spin.value() or None
        return ref_page.phys_h_mm * frac - self.diameter_spin.value() / 2

    def set_y_from_fraction(self, frac: float) -> None:
        """效果预览点选回调：按页面高度比例设置自定义 y（修改意见 #4）。"""
        idx = self._page_indices()
        ref_page = self._pages[idx[0]]
        y = ref_page.phys_h_mm * frac - self.diameter_spin.value() / 2
        self.y_preset.setCurrentText("自定义 mm")
        self.y_spin.setValue(max(0.0, y))

    def _jitter_scale(self) -> float:
        return self.jitter_slider.value() / 100.0

    def _make_spec(self) -> PerforationSpec:
        j = self._jitter_scale()
        return PerforationSpec(
            diameter_mm=self.diameter_spin.value(),
            side=self.side_combo.currentData(),
            inset_mm=self.inset_spin.value(),
            y_mm=self._resolve_y_mm(),
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

    def _effect_preview(self) -> None:
        """页面效果预览：渲染切片盖在真实页面上的样子（修改意见 #4）。"""
        try:
            placements = self._plan()
        except ValueError as e:
            QMessageBox.warning(self, "无法生成", str(e))
            return
        dlg = EffectPreviewDialog(self, self._pages, placements)
        dlg.position_clicked.connect(self.set_y_from_fraction)
        dlg.exec()

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


class _ClickablePageLabel(QLabel):
    """可点击的页面预览图：点击发射"页面高度比例"（用于选择骑缝章垂直位置）。"""

    position_clicked = Signal(float)

    def mousePressEvent(self, event) -> None:
        pm = self.pixmap()
        if pm and pm.height() > 0:
            frac = event.position().y() / pm.height()
            self.position_clicked.emit(min(max(frac, 0.0), 1.0))
        super().mousePressEvent(event)


class EffectPreviewDialog(QDialog):
    """页面效果预览：首/中/末三页，渲染切片盖在真实页面上的实际效果。

    点击任意页面即可把骑缝章垂直位置设到点击处（修改意见 #4）。
    """

    position_clicked = Signal(float)

    def __init__(self, parent, pages: list[Page], placements: list[SlicePlacement]):
        super().__init__(parent)
        self.setWindowTitle("页面效果预览（点击页面可设置垂直位置）")
        layout = QVBoxLayout(self)
        hint = QLabel("切片盖在真实页面上的效果如下。点击某页的预览图，可把骑缝章移到点击的高度：")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # 首/中/末三页
        idxs = sorted({pl.page_index for pl in placements})
        picks = sorted({idxs[0], idxs[len(idxs) // 2], idxs[-1]})
        row = QHBoxLayout()
        for page_idx in picks:
            pl = next(p for p in placements if p.page_index == page_idx)
            thumb = self._render_effect(pages[page_idx], pl)
            label = _ClickablePageLabel(f"第 {page_idx + 1} 页")
            label.setAlignment(Qt.AlignCenter)
            pm = np_rgb_to_qpixmap(thumb)
            label.setPixmap(pm)
            label.position_clicked.connect(self._on_click)
            row.addWidget(label)
        layout.addLayout(row)

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        self.resize(1000, 560)

    def _render_effect(self, page: Page, pl: SlicePlacement, thumb_w: int = 300) -> np.ndarray:
        """把单个切片按真实坐标合成到缩放后的页面上。

        关键：切片必须与页面缩放到同一 DPI——apply_perforation 按像素合成，
        全分辨率切片直接画到缩略图上会大出好几倍（"章比页面大" bug）。
        """
        h, w = page.image.shape[:2]
        thumb_h = round(h * thumb_w / w)
        thumb_img = cv2.resize(page.image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        thumb_page = Page(image=thumb_img, phys_w_mm=page.phys_w_mm, phys_h_mm=page.phys_h_mm)
        # 切片同步缩放到缩略图 DPI
        scale = thumb_page.dpi / page.dpi
        sl = pl.slice_rgba
        if abs(scale - 1.0) > 1e-6:
            sl = cv2.resize(
                sl,
                (max(1, round(sl.shape[1] * scale)), max(1, round(sl.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        out = apply_perforation([thumb_page], [
            SlicePlacement(
                page_index=0,
                slice_rgba=sl,
                right_edge_mm=pl.right_edge_mm,
                top_mm=pl.top_mm,
                y_offset_mm=0.0,
                width_px=pl.width_px,
            )
        ])
        return out.get(0, thumb_img)

    def _on_click(self, frac: float) -> None:
        self.position_clicked.emit(frac)
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
