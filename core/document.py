"""文档模型：把 PDF / 图片文件统一抽象为"页列表"。

v0.3.0 起页面图像**懒加载**（P0 内存修复）：
- Page 只持有元数据 + 渲染函数；首次访问 image 时才渲染；
- Document 维护 LRU 缓存（默认同时驻留 6 页 ≈ 150MB 上限），
  50 页 PDF 不再一次性吃掉 1.25GB；
- 任何对 page.image 的**写入**（旋转/透视校准）会使该页脱离懒加载，
  变异结果永久驻留（不参与淘汰）；
- 缩略图走 `Page.thumbnail()` 低 DPI 独立渲染，不触碰全尺寸缓存。

每页携带：
- image：RGB 像素（numpy HxWx3 uint8）——懒加载属性；
- phys_w_mm / phys_h_mm：页面物理尺寸（毫米）——盖章比例换算的唯一基准。

物理尺寸来源（优先级从高到低）：
1. 用户纸边校准（calibrate_paper_edge / 四点校准 warp）；
2. PDF page box（pt → mm）；
3. A4 假定（按图像长宽比推断方向）。
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageOps

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0
A4_W_MM = 210.0
A4_H_MM = 297.0
DEFAULT_DPI = 300.0
THUMB_DPI = 40.0
# A 系纸长宽比 √2，容差 ±5%（方案 v1.3）
ASPECT_A_SERIES = math.sqrt(2.0)
ASPECT_TOLERANCE = 0.05

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# LRU 容量：同时驻留的全尺寸页面数（A4@300DPI 每页约 25MB）
PAGE_CACHE_LIMIT = 6


def mm_to_px(mm: float, dpi: float) -> float:
    """毫米 → 像素。例：40mm @300DPI ≈ 472.44px。"""
    return mm / MM_PER_INCH * dpi


def px_to_mm(px: float, dpi: float) -> float:
    """像素 → 毫米。"""
    return px / dpi * MM_PER_INCH


class Page:
    """一页。image 属性懒加载；写入后固定为变异结果。"""

    def __init__(
        self,
        image: np.ndarray | None = None,
        phys_w_mm: float = 210.0,
        phys_h_mm: float = 297.0,
        needs_calibration: bool = False,
        render_fn=None,
        thumbnail_fn=None,
    ):
        if image is None and render_fn is None:
            raise ValueError("Page 需要 image 或 render_fn 之一")
        self._override: np.ndarray | None = image  # 非空 = 已变异或直接构造
        self._render_fn = render_fn
        self._thumbnail_fn = thumbnail_fn
        self.phys_w_mm = float(phys_w_mm)
        self.phys_h_mm = float(phys_h_mm)
        self.needs_calibration = needs_calibration
        self._owner: Document | None = None  # 由 Document 设置，LRU 用

    # ── 图像访问（懒加载核心）──

    @property
    def image(self) -> np.ndarray:
        if self._override is not None:
            return self._override
        assert self._owner is not None, "懒加载页必须挂在 Document 下"
        return self._owner._cache_get(self)

    @image.setter
    def image(self, arr: np.ndarray) -> None:
        """写入即变异：脱离懒加载，永久驻留（旋转/校准依赖此语义）。"""
        self._override = arr

    @property
    def mutated(self) -> bool:
        return self._override is not None

    def thumbnail(self, dpi: float = THUMB_DPI) -> np.ndarray:
        """低 DPI 缩略图，独立渲染，不占用全尺寸 LRU。

        变异页（旋转/四点校准后）必须从变异结果缩放——
        用 _thumbnail_fn 会从原始文件渲染出旧方向/未校准的画面。
        """
        import cv2

        if self._override is not None:
            h, w = self._override.shape[:2]
            tw = max(1, round(w * dpi / self.dpi))
            return cv2.resize(
                self._override, (tw, max(1, round(h * tw / w))),
                interpolation=cv2.INTER_AREA,
            )
        if self._thumbnail_fn is not None:
            return self._thumbnail_fn(dpi)
        h, w = self.image.shape[:2]
        tw = max(1, round(w * dpi / self.dpi))
        return cv2.resize(self.image, (tw, max(1, round(h * tw / w))), interpolation=cv2.INTER_AREA)

    # ── 尺寸 ──

    @property
    def width_px(self) -> int:
        return int(self.image.shape[1])

    @property
    def height_px(self) -> int:
        return int(self.image.shape[0])

    @property
    def dpi(self) -> float:
        """由像素宽与物理宽推导的水平 DPI（假定横竖 DPI 一致）。"""
        return self.image.shape[1] / (self.phys_w_mm / MM_PER_INCH)

    def aspect_needs_calibration(self) -> bool:
        """长宽比偏离 A 系纸 √2±5% 时返回 True。"""
        long_side = max(self.image.shape[1], self.image.shape[0])
        short_side = min(self.image.shape[1], self.image.shape[0])
        ratio = long_side / short_side
        return abs(ratio - ASPECT_A_SERIES) / ASPECT_A_SERIES > ASPECT_TOLERANCE


@dataclass
class Document:
    pages: list[Page] = field(default_factory=list)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_limit = PAGE_CACHE_LIMIT
        self._pdf = None  # 懒加载需要保持 PDF 句柄打开

    # ── LRU ──

    def _cache_get(self, page: Page) -> np.ndarray:
        key = id(page)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        img = page._render_fn()
        self._cache[key] = img
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return img

    def close(self) -> None:
        """释放 PDF 句柄与缓存。"""
        self._cache.clear()
        if self._pdf is not None:
            try:
                self._pdf.close()
            except Exception:
                pass
            self._pdf = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # ── 打开 ──

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
        doc = cls()
        pdf = pymupdf.open(path)
        if pdf.needs_pass:
            pdf.close()
            raise ValueError("PDF 已加密，无法打开（方案 §6：检测到即提示）")
        doc._pdf = pdf  # 保持打开：懒渲染需要

        pages: list[Page] = []
        for i in range(len(pdf)):
            pdf_page = pdf[i]
            rect = pdf_page.rect
            phys_w_mm = rect.width / PT_PER_INCH * MM_PER_INCH
            phys_h_mm = rect.height / PT_PER_INCH * MM_PER_INCH
            page = Page(
                phys_w_mm=phys_w_mm,
                phys_h_mm=phys_h_mm,
                render_fn=lambda idx=i: doc._render_pdf_page(idx, dpi),
                thumbnail_fn=lambda d, idx=i: doc._render_pdf_page(idx, d),
            )
            # 比例检测用 page box（pt 比例 = 像素比例），无需渲染
            long_s = max(rect.width, rect.height)
            short_s = min(rect.width, rect.height)
            ratio = long_s / short_s if short_s > 0 else 1.0
            page.needs_calibration = (
                abs(ratio - ASPECT_A_SERIES) / ASPECT_A_SERIES > ASPECT_TOLERANCE
            )
            pages.append(page)

        if not pages:
            pdf.close()
            raise ValueError("PDF 没有页面")
        doc.pages = pages
        doc.source_path = path
        for p in doc.pages:
            p._owner = doc
        return doc

    def _render_pdf_page(self, index: int, dpi: float) -> np.ndarray:
        assert self._pdf is not None, "PDF 句柄已关闭"
        pdf_page = self._pdf[index]
        pix = pdf_page.get_pixmap(dpi=round(dpi), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        return np.ascontiguousarray(img)

    @classmethod
    def from_images(cls, paths: list[str | Path], dpi: float = DEFAULT_DPI) -> "Document":
        """图片合成文档。物理尺寸按 A4 假定（按长宽比推断方向），并做比例检测。"""
        import cv2

        doc = cls()
        pages: list[Page] = []
        for p in paths:
            path = Path(p)
            with Image.open(path) as im:
                w, h = _exif_size(im)  # EXIF 修正后的显示尺寸（方向自愈）

            def render(pp=path):
                return _load_rgb_exif(pp)

            def thumb(d, pp=path, w=w, h=h):
                tw = max(1, round(w * d / dpi))
                with Image.open(pp) as im:
                    # JPEG draft 模式：DCT 域直接缩放解码，比全量解码快一个数量级
                    # （多图文档打开/翻页的主要卡顿源，网络共享盘尤甚）
                    im.draft("RGB", (tw * 2, max(1, round(h * tw * 2 / w))))
                    arr = np.array(ImageOps.exif_transpose(im).convert("RGB"))
                return cv2.resize(
                    arr, (tw, max(1, round(h * tw / w))), interpolation=cv2.INTER_AREA
                )

            if h >= w:
                phys_w, phys_h = A4_W_MM, A4_H_MM
            else:
                phys_w, phys_h = A4_H_MM, A4_W_MM
            page = Page(
                phys_w_mm=phys_w,
                phys_h_mm=phys_h,
                render_fn=render,
                thumbnail_fn=thumb,
            )
            ratio = max(w, h) / min(w, h)
            page.needs_calibration = (
                abs(ratio - ASPECT_A_SERIES) / ASPECT_A_SERIES > ASPECT_TOLERANCE
            )
            pages.append(page)

        if not pages:
            raise ValueError("没有可用的图片")
        doc.pages = pages
        doc.source_path = Path(paths[0]) if len(paths) == 1 else None
        for pg in doc.pages:
            pg._owner = doc
        return doc


def _exif_size(im: Image.Image) -> tuple[int, int]:
    """按 EXIF orientation 修正后的显示尺寸（手机拍照件方向自愈，只读头部不解码像素）。"""
    w, h = im.size
    try:
        orientation = (im.getexif() or {}).get(274, 1)
    except Exception:
        orientation = 1
    if orientation in (5, 6, 7, 8):  # 需要旋转 90° 的方向 → 宽高互换
        return h, w
    return w, h


def _load_rgb_exif(path: Path) -> np.ndarray:
    """加载图片并应用 EXIF 方向（横躺的照片自动转正）。"""
    with Image.open(path) as im:
        return np.array(ImageOps.exif_transpose(im).convert("RGB"))


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
