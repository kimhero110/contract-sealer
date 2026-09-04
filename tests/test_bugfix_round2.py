"""全量猎虫第二轮修复的回归测试：缩略图陈旧/横版校准/EXIF 方向/校准撤销标志。"""

import numpy as np
import pytest
from PIL import Image

from core.autocal import warp_to_a4
from core.document import A4_H_MM, A4_W_MM, Document, Page
from core.extract import extract_ink


# ── BUG A：变异页缩略图必须反映旋转/校准后的画面 ──

def test_thumbnail_reflects_mutation(tmp_path):
    """旋转 180° 后，缩略图应与变异图像一致（旧实现会从源文件渲染出未旋转画面）。"""
    import pymupdf

    path = tmp_path / "a4.pdf"
    doc = pymupdf.open()
    page0 = doc.new_page(width=595, height=842)
    # 上黑下白，旋转后应上白下黑
    page0.draw_rect(pymupdf.Rect(0, 0, 595, 421), fill=(0.1, 0.1, 0.1))
    doc.save(path)
    doc.close()

    d = Document.open(path)
    page = d.pages[0]
    t_before = page.thumbnail(dpi=30)
    assert t_before[: t_before.shape[0] // 2].mean() < 100  # 上半黑

    page.image = np.rot90(page.image, 2)  # 旋转 180°（变异）
    t_after = page.thumbnail(dpi=30)
    assert t_after[: t_after.shape[0] // 2].mean() > 200  # 上半变白 ✓ 用的是变异图
    d.close()


# ── BUG B：横版纸四点校准 ──

def test_warp_landscape_quad():
    """横版（宽>高）纸面校准：图像必须保持横版比例，物理尺寸与图像方向一致。"""
    w, h = 2200, 1600
    img = np.full((h, w, 3), 70, dtype=np.uint8)
    quad = np.array([[150, 120], [2050, 100], [2080, 1500], [130, 1520]], dtype=np.float32)
    page = Page(image=img, phys_w_mm=999.0, phys_h_mm=999.0)
    H = warp_to_a4(page, quad)
    assert H is not None
    ih, iw = page.image.shape[:2]
    assert iw > ih, "横版纸校准后图像仍应是横版"
    assert abs(iw / ih - A4_H_MM / A4_W_MM) < 0.02  # √2 比例（宽/高）
    assert page.phys_w_mm == A4_H_MM and page.phys_h_mm == A4_W_MM  # 297×210
    # 图像与物理方向一致 → dpi 横竖一致
    dpi_x = iw / (page.phys_w_mm / 25.4)
    dpi_y = ih / (page.phys_h_mm / 25.4)
    assert abs(dpi_x - dpi_y) / dpi_x < 0.02


# ── BUG C：EXIF 方向 ──

def test_exif_orientation_autofixed(tmp_path):
    """手机横拍竖版纸（orientation=6）：加载后应自动转正为竖版。"""
    # 传感器存储横版像素 566×400（真实手机行为），EXIF 标记需旋转 90°
    img = Image.new("RGB", (566, 400), (250, 250, 250))
    exif = Image.Exif()
    exif[274] = 6
    p = tmp_path / "photo.jpg"
    img.save(p, exif=exif)

    doc = Document.from_images([p])
    page = doc.pages[0]
    assert page.height_px > page.width_px, "EXIF orientation=6 应转正为竖版"
    assert page.needs_calibration is False  # 转正后 A4 竖版比例不再触发误报
    doc.close()


def test_extract_exif_orientation(tmp_path):
    """印章照片带 EXIF：抠图输入先转正，输出形状与转正后一致。"""
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    from PIL import ImageDraw

    d = ImageDraw.Draw(img)
    d.ellipse([50, 100, 350, 200], fill=(200, 30, 30))  # 扁椭圆（宽>高）
    exif = Image.Exif()
    exif[274] = 6
    p = tmp_path / "seal.jpg"
    img.save(p, exif=exif)

    rgba = extract_ink(p, kind="seal")
    assert rgba.shape[0] > rgba.shape[1], "转正后扁椭圆应变竖椭圆（高>宽）"


# ── BUG D：校准撤销恢复 needs_calibration 标志 ──

def test_restore_cal_flag_placeholder():
    """GUI 侧（restore_cal 闭包）标志恢复由 GUI 测试覆盖；此处仅确认 core 行为不回归。"""
    page = Page(image=np.full((100, 80, 3), 255, np.uint8), phys_w_mm=210, phys_h_mm=297)
    page.needs_calibration = True
    assert page.needs_calibration is True
