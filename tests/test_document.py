"""文档模型与物理尺寸换算测试。"""

import numpy as np
import pymupdf
import pytest

from core.document import (
    A4_H_MM,
    A4_W_MM,
    Document,
    calibrate_paper_edge,
    mm_to_px,
    px_to_mm,
)


def test_mm_px_roundtrip():
    # 方案 §4.3 数值断言：40mm @300DPI ≈ 472px
    assert abs(mm_to_px(40, 300) - 472.44) < 0.5
    assert abs(px_to_mm(mm_to_px(40, 300), 300) - 40) < 1e-9


def test_from_images_a4_assumption(scan_jpg):
    doc = Document.from_images([scan_jpg])
    assert len(doc.pages) == 1
    page = doc.pages[0]
    # 竖版扫描件 → A4 竖版假定
    assert page.phys_w_mm == A4_W_MM
    assert page.phys_h_mm == A4_H_MM
    # 真实 A4 扫描件比例正常，不应要求校准
    assert page.needs_calibration is False


def test_non_a4_aspect_flagged():
    # 正方形图片（如收据）必须触发校准提示，禁止静默按 A4 盖
    img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    from core.document import Page

    page = Page(image=img, phys_w_mm=A4_W_MM, phys_h_mm=A4_H_MM)
    assert page.aspect_needs_calibration() is True


def test_calibrate_paper_edge(scan_jpg):
    doc = Document.from_images([scan_jpg])
    page = doc.pages[0]
    page.needs_calibration = True
    calibrate_paper_edge(page, 210.0, 297.0)
    assert page.needs_calibration is False
    assert abs(page.dpi - page.width_px / (210.0 / 25.4)) < 1e-6
    with pytest.raises(ValueError):
        calibrate_paper_edge(page, 0, 297)


def test_open_pdf_physical_size(tmp_path):
    # 造一个 A4 单页 PDF，验证 page box → mm 换算
    src = tmp_path / "a4.pdf"
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)  # A4 in pt
    doc.save(src)
    doc.close()

    d = Document.open(src)
    assert len(d.pages) == 1
    assert abs(d.pages[0].phys_w_mm - 210.0) < 1.0
    assert abs(d.pages[0].phys_h_mm - 297.0) < 1.0


def test_unsupported_type(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError):
        Document.open(bad)
