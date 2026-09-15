"""压平导出：页面图像（已含章）→ 新 PDF / 图片。

方案 v1.1 §3.5 / v1.3：
- 压平默认强制：PDF 中每页就是一张合成后的整页图像，不存在可分离的印章对象；
- 绝不覆盖任何已有文件：调用前由 make_output_path 保证目标路径唯一；
- 原子写：先写同目录临时文件，完成后 os.replace；
- 每次导出写同名 .sealog（JSON：种子、页范围、参数快照），P0 必备；
- 页面图像的编码由 ExportOptions 决定：默认 JPEG 质量 92（体积小、300DPI 下肉眼看不出），
  需要无损时选 PNG——原件是文字页时 JPEG 会留下压缩痕迹，这是用户该有的选择。
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

from .document import MM_PER_INCH, PT_PER_INCH, Page

FORMAT_JPEG = "jpeg"
FORMAT_PNG = "png"
DEFAULT_JPEG_QUALITY = 92


@dataclass(frozen=True)
class ExportOptions:
    """页面图像嵌入 PDF 时的编码方式。"""

    image_format: str = FORMAT_JPEG  # jpeg（有损，小）/ png（无损，大）
    jpeg_quality: int = DEFAULT_JPEG_QUALITY  # 仅 jpeg 有效，1–100

    def normalized(self) -> ExportOptions:
        fmt = self.image_format.lower()
        if fmt not in (FORMAT_JPEG, FORMAT_PNG):
            raise ValueError(f"不支持的导出图像格式: {self.image_format}（可选 jpeg / png）")
        return ExportOptions(fmt, int(min(100, max(1, self.jpeg_quality))))

    def to_dict(self) -> dict:
        return {"image_format": self.image_format, "jpeg_quality": self.jpeg_quality}


def make_output_path(source_path: Path | None, out_dir: Path, suffix: str = ".pdf") -> Path:
    """生成不覆盖任何已有文件的输出路径：{原名}_已盖章_{日期}_{时分秒}[_{序号}].pdf"""
    stem = source_path.stem if source_path else "document"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = out_dir / f"{stem}_已盖章_{stamp}{suffix}"
    n = 1
    while candidate.exists() or candidate.with_suffix(".sealog").exists():
        candidate = out_dir / f"{stem}_已盖章_{stamp}_{n}{suffix}"
        n += 1
    return candidate


def export_pdf(
    pages: list[Page],
    images: list[np.ndarray],
    out_path: Path,
    sealog: dict,
    options: ExportOptions | None = None,
) -> Path:
    """压平导出 PDF。images[i] 为 pages[i] 盖章后的 RGB 图像（尺寸可与原页不同）。

    页面物理尺寸取自 pages[i]，保证打印尺寸正确。编码方式见 ExportOptions。
    """
    options = (options or ExportOptions()).normalized()
    if len(pages) != len(images):
        raise ValueError("pages 与 images 数量不一致")
    if not pages:
        raise ValueError("没有可导出的页面")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{os.getpid()}{out_path.suffix}")

    try:
        doc = pymupdf.open()
        for page, img in zip(pages, images, strict=True):
            w_pt = page.phys_w_mm / MM_PER_INCH * PT_PER_INCH
            h_pt = page.phys_h_mm / MM_PER_INCH * PT_PER_INCH
            pdf_page = doc.new_page(width=w_pt, height=h_pt)
            pdf_page.insert_image(pdf_page.rect, stream=_encode_page(img, options))
        # 先写临时文件
        doc.save(tmp_path, deflate=True)
        doc.close()
        # 原子改名（同目录，Windows 下 os.replace 可覆盖空目标，但目标由 make_output_path 保证唯一）
        os.replace(tmp_path, out_path)
    except BaseException:
        # 中断/失败：清理临时文件，不留下损坏输出
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    log = dict(sealog)
    log.setdefault("export_options", options.to_dict())
    _write_sealog(out_path, log)
    return out_path


def export_page_image(image: np.ndarray, out_path: Path, sealog: dict | None = None) -> Path:
    """单页导出 PNG/JPG（P1）。同样走原子写。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f"{out_path.stem}.tmp{os.getpid()}{out_path.suffix}")
    pil_format = "PNG" if out_path.suffix.lower() == ".png" else "JPEG"
    try:
        Image.fromarray(image).save(tmp_path, format=pil_format)
        os.replace(tmp_path, out_path)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    if sealog is not None:
        _write_sealog(out_path, sealog)
    return out_path


def count_separable_ink_images(pdf_path: Path, page_coverage: float = 0.9) -> int:
    """统计 PDF 中"可分离的墨迹图片对象"数量（压平验收用）。

    压平正确时应为 0：每页唯一的图片对象就是整页图像本身。
    返回所有"明显小于所在页面"的图片对象数（这些才可能是可提取的印章）。
    """
    suspicious = 0
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            page_area = page.rect.width * page.rect.height
            for info in page.get_image_info():
                bbox = info["bbox"]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if page_area > 0 and (w * h) / page_area < page_coverage:
                    suspicious += 1
    return suspicious


def _encode_page(img: np.ndarray, options: ExportOptions) -> bytes:
    if options.image_format == FORMAT_PNG:
        return _to_png_bytes(img)
    return _to_jpeg_bytes(img, options.jpeg_quality)


def _to_jpeg_bytes(img: np.ndarray, quality: int = DEFAULT_JPEG_QUALITY) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _to_png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    # 压缩级别 6：再高只换来几个百分点体积，导出时间却成倍增长
    Image.fromarray(img).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def _write_sealog(out_path: Path, sealog: dict) -> Path:
    log = dict(sealog)
    log.setdefault("exported_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    log.setdefault("output", out_path.name)
    log_path = out_path.with_suffix(".sealog")
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log_path
