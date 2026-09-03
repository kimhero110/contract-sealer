"""文档模型：把 PDF / 图片文件统一抽象为"页列表"。

每页携带：
- image：RGB 像素（numpy HxWx3 uint8）；
- phys_w_mm / phys_h_mm：页面物理尺寸（毫米）——盖章比例换算的唯一基准；
- dpi：image 像素与物理尺寸的换算关系（由二者推导，不单独存储）。

物理尺寸来源（优先级从高到低）：
1. 用户纸边校准（calibrate_paper_edge）；
2. PDF page box（pt → mm）；
3. A4 假定（按图像长宽比推断方向）。

打开文件时做长宽比检测：偏离 A 系纸 √2±5% 的页面被标记 needs_calibration=True，
UI 应提示用户校准，不允许静默按 A4 盖章（方案 v1.3 §4.3）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0
A4_W_MM = 210.0
A4_H_MM = 297.0
DEFAULT_DPI = 300.0
# A 系纸长宽比 √2，容差 ±5%（方案 v1.3）
ASPECT_A_SERIES = math.sqrt(2.0)
ASPECT_TOLERANCE = 0.05

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def mm_to_px(mm: float, dpi: float) -> float:
    """毫米 → 像素。例：40mm @300DPI ≈ 472.44px。"""
    return mm / MM_PER_INCH * dpi


def px_to_mm(px: float, dpi: float) -> float:
    """像素 → 毫米。"""
    return px / dpi * MM_PER_INCH


@dataclass
class Page:
    image: np.ndarray  # RGB uint8
    phys_w_mm: float
    phys_h_mm: float
    needs_calibration: bool = False

    @property
    def width_px(self) -> int:
        return int(self.image.shape[1])

    @property
    def height_px(self) -> int:
        return int(self.image.shape[0])

    @property
    def dpi(self) -> float:
        """由像素宽与物理宽推导的水平 DPI（假定横竖 DPI 一致）。"""
        return self.width_px / (self.phys_w_mm / MM_PER_INCH)

    def aspect_needs_calibration(self) -> bool:
        """长宽比偏离 A 系纸 √2±5% 时返回 True。"""
        long_side = max(self.width_px, self.height_px)
        short_side = min(self.width_px, self.height_px)
        ratio = long_side / short_side
        return abs(ratio - ASPECT_A_SERIES) / ASPECT_A_SERIES > ASPECT_TOLERANCE


@dataclass
class Document:
    pages: list[Page] = field(default_factory=list)
    source_path: Path | None = None

    @classmethod
    def open(cls, path: str | Path, dpi: float = DEFAULT_DPI) -> "Document":
        """打开 PDF 或单张图片。多图合成文档用 from_images。"""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return cls._open_pdf(path, dpi)
        if suffix in SUPPORTED_IMAGE_EXTS:
            return cls.from_images([path])
        raise ValueError(f"不支持的文件类型: {path.suffix}")

    @classmethod
    def _open_pdf(cls, path: Path, dpi: float) -> "Document":
        pages: list[Page] = []
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise ValueError("PDF 已加密，无法打开（方案 §6：检测到即提示）")
            for pdf_page in pdf:
                # 物理尺寸来自 page box：1pt = 1/72 inch
                rect = pdf_page.rect
                phys_w_mm = rect.width / PT_PER_INCH * MM_PER_INCH
                phys_h_mm = rect.height / PT_PER_INCH * MM_PER_INCH
                pix = pdf_page.get_pixmap(dpi=round(dpi), alpha=False)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                if pix.n == 4:  # 防御：alpha=False 时应为 3 通道
                    img = img[:, :, :3]
                page = Page(
                    image=np.ascontiguousarray(img),
                    phys_w_mm=phys_w_mm,
                    phys_h_mm=phys_h_mm,
                )
                page.needs_calibration = page.aspect_needs_calibration()
                pages.append(page)
        if not pages:
            raise ValueError("PDF 没有页面")
        return cls(pages=pages, source_path=path)

    @classmethod
    def from_images(cls, paths: list[str | Path]) -> "Document":
        """图片合成文档。物理尺寸按 A4 假定（按长宽比推断方向），并做比例检测。"""
        pages: list[Page] = []
        for p in paths:
            with Image.open(p) as im:
                img = np.array(im.convert("RGB"))
            h, w = img.shape[:2]
            if h >= w:
                phys_w, phys_h = A4_W_MM, A4_H_MM
            else:
                phys_w, phys_h = A4_H_MM, A4_W_MM
            page = Page(image=img, phys_w_mm=phys_w, phys_h_mm=phys_h)
            page.needs_calibration = page.aspect_needs_calibration()
            pages.append(page)
        if not pages:
            raise ValueError("没有可用的图片")
        return cls(pages=pages, source_path=Path(paths[0]) if len(paths) == 1 else None)


def calibrate_paper_edge(page: Page, paper_w_mm: float, paper_h_mm: float) -> None:
    """纸边校准（手动路径）：用户确认纸面实际物理尺寸后调用。

    M1 简化：用户框选纸面全幅后输入纸张真实尺寸（如 A4 210x297），
    即把页面物理尺寸设为该值。M3 自动纸边检测+透视纠偏。
    """
    if paper_w_mm <= 0 or paper_h_mm <= 0:
        raise ValueError("纸张尺寸必须为正数")
    page.phys_w_mm = float(paper_w_mm)
    page.phys_h_mm = float(paper_h_mm)
    page.needs_calibration = False
