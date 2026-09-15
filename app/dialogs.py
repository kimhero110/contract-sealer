"""主窗口用到的各类对话框（从 main_window 拆出，主窗口只留编排逻辑）。"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from app.canvas import np_rgb_to_qpixmap, np_rgba_to_qpixmap
from core.datestamp import (
    DATE_FORMATS,
    DEFAULT_HEIGHT_MM,
    format_date,
    render_date_ink,
)


def load_qr_pixmap(width: int) -> QPixmap | None:
    """加载赞赏码：冻结版从 _MEIPASS，源码版从项目根目录 docs/。"""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    path = base / "docs" / "coffee.png"
    if not path.exists():
        return None
    pm = QPixmap(str(path))
    if pm.isNull():
        return None
    return pm.scaledToWidth(width, Qt.TransformationMode.SmoothTransformation)


class AboutDialog(QDialog):
    """关于对话框：版本/链接/咖啡文案 + 内嵌赞赏码（打包进 exe，离线可见）。"""

    def __init__(self, parent, version: str):
        super().__init__(parent)
        self.setWindowTitle("关于 合同盖章工具")
        layout = QVBoxLayout(self)
        text = QLabel(
            f"<h3>合同盖章工具</h3>"
            f"<p>版本 v{version}</p>"
            f"<p>扫描合同盖章：物理尺寸 1:1、骑缝章、手写签名、日期戳。<br>"
            f"全程本地离线运行，不修改原始文件。</p>"
            f"<p>源码：github.com/kimhero110/contract-sealer<br>"
            f"gitee.com/xu512/contract-sealer</p>"
            f"<p>本程序以 GNU AGPL-3.0 授权发布，依赖 PyMuPDF（同为 AGPL）。</p>"
            f"<p>☕ 这个工具没收你一分钱。<br>"
            f"如果它帮你省过一个加班的晚上——<br>"
            f"<b>给码农买杯咖啡，是他的福报。</b></p>"
            f"<p style='color:gray'>仅限本单位已授权印章的内部流程使用。</p>"
        )
        text.setWordWrap(True)
        layout.addWidget(text)

        qr = load_qr_pixmap(220)
        if qr is not None:
            img_label = QLabel()
            img_label.setPixmap(qr)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(img_label)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(self.accept)
        layout.addWidget(btns)


class WarpPreviewDialog(QDialog):
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
            pm = pm.scaledToHeight(640, Qt.TransformationMode.SmoothTransformation)
        label = QLabel()
        label.setPixmap(pm)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class CalibrateDialog(QDialog):
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
        self.all_check = QCheckBox("应用到所有页面")
        self.all_check.setChecked(True)
        form.addRow(self.all_check)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def values(self) -> tuple[float, float, bool]:
        return self.w_spin.value(), self.h_spin.value(), self.all_check.isChecked()


class DateStampDialog(QDialog):
    """日期戳：默认系统当天，可改成任意日期与格式，实时预览墨迹。

    预览按屏幕 DPI 渲染一份小的，真正落章用的墨迹按页面 DPI 重新渲染，
    避免把预览分辨率的图放大到 300DPI 盖出来发虚。
    """

    PREVIEW_DPI = 220.0

    def __init__(self, parent, default_date: date, fmt: str, height_mm: float):
        super().__init__(parent)
        self.setWindowTitle("加盖日期")
        form = QFormLayout(self)

        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setDate(QDate(default_date.year, default_date.month, default_date.day))
        form.addRow("日期", self.date_edit)

        self.today_check = QCheckBox("使用系统当天日期")
        self.today_check.setChecked(True)
        self.today_check.toggled.connect(self._on_today_toggled)
        form.addRow("", self.today_check)

        self.fmt_combo = QComboBox()
        for sample, f in DATE_FORMATS:
            self.fmt_combo.addItem(sample, f)
        index = max(0, next((i for i, (_, f) in enumerate(DATE_FORMATS) if f == fmt), 0))
        self.fmt_combo.setCurrentIndex(index)
        form.addRow("格式", self.fmt_combo)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(2.0, 30.0)
        self.height_spin.setSingleStep(0.5)
        self.height_spin.setSuffix(" mm")
        self.height_spin.setValue(height_mm or DEFAULT_HEIGHT_MM)
        form.addRow("字高", self.height_spin)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumHeight(60)
        form.addRow("预览", self.preview)

        self.warn = QLabel("")
        self.warn.setWordWrap(True)
        self.warn.setStyleSheet("color: #C0392B;")
        form.addRow("", self.warn)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)

        self.date_edit.dateChanged.connect(self._refresh)
        self.fmt_combo.currentIndexChanged.connect(self._refresh)
        self.height_spin.valueChanged.connect(self._refresh)
        self._on_today_toggled(True)

    # ── 取值 ──

    def selected_date(self) -> date:
        if self.today_check.isChecked():
            return date.today()
        q = self.date_edit.date()
        return date(q.year(), q.month(), q.day())

    def values(self) -> tuple[str, str, float]:
        """返回 (日期文本, strftime 格式, 字高 mm)。"""
        fmt = self.fmt_combo.currentData()
        return format_date(self.selected_date(), fmt), fmt, self.height_spin.value()

    # ── 交互 ──

    def _on_today_toggled(self, checked: bool) -> None:
        self.date_edit.setEnabled(not checked)
        if checked:
            today = date.today()
            self.date_edit.blockSignals(True)
            self.date_edit.setDate(QDate(today.year, today.month, today.day))
            self.date_edit.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        text, _fmt, height_mm = self.values()
        try:
            ink = render_date_ink(text, height_mm, self.PREVIEW_DPI)
        except ValueError as e:
            self.preview.clear()
            self.warn.setText(str(e))
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return
        self.warn.setText("")
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        self.preview.setPixmap(np_rgba_to_qpixmap(ink))


class ExportOptionsDialog(QDialog):
    """导出编码设置：JPEG 质量 / 无损 PNG。

    默认 JPEG 92：300DPI 下肉眼看不出，体积小。原件是文字页、或要交给 OCR 时，
    JPEG 的压缩痕迹会咬字边——那就选 PNG，代价是文件大好几倍。
    """

    def __init__(self, parent, options):
        from PySide6.QtWidgets import QRadioButton, QSpinBox

        from core.export import FORMAT_PNG

        super().__init__(parent)
        self.setWindowTitle("导出设置")
        layout = QVBoxLayout(self)
        self.jpeg_radio = QRadioButton("JPEG（有损，文件小；默认）")
        self.png_radio = QRadioButton("PNG（无损，文件大；文字页/需要 OCR 时选它）")
        layout.addWidget(self.jpeg_radio)
        form = QFormLayout()
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(1, 100)
        self.quality_spin.setValue(options.jpeg_quality)
        form.addRow("JPEG 质量（1–100）", self.quality_spin)
        layout.addLayout(form)
        layout.addWidget(self.png_radio)
        if options.image_format == FORMAT_PNG:
            self.png_radio.setChecked(True)
        else:
            self.jpeg_radio.setChecked(True)
        self.jpeg_radio.toggled.connect(self.quality_spin.setEnabled)
        self.quality_spin.setEnabled(self.jpeg_radio.isChecked())
        note = QLabel("无论哪种编码，导出件都是整页位图：原件里可选中的文字不再是文字。")
        note.setWordWrap(True)
        layout.addWidget(note)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def values(self):
        from core.export import FORMAT_JPEG, FORMAT_PNG, ExportOptions

        fmt = FORMAT_PNG if self.png_radio.isChecked() else FORMAT_JPEG
        return ExportOptions(fmt, self.quality_spin.value()).normalized()
