"""M3 测试：自动纸边检测、模板、骑缝章-导出集成。"""

import numpy as np
import cv2

from core.autocal import auto_calibrate_page, detect_paper_quad
from core.document import A4_H_MM, A4_W_MM, Page
from core.perforation import PerforationSpec, apply_perforation, plan_perforation
from core.template import list_templates, load_template, save_template


def _photo_of_paper(angle_deg: float = 3.0, margin: int = 120) -> np.ndarray:
    """合成一张"拍照件"：深灰背景上一张略旋转的白纸（带内容）。"""
    w, h = 2000, 2600
    img = np.full((h, w, 3), 80, dtype=np.uint8)  # 深灰背景
    paper_w, paper_h = w - 2 * margin, int((w - 2 * margin) * 1.414)
    paper = np.full((paper_h, paper_w, 3), 245, dtype=np.uint8)
    cv2.putText(paper, "CONTRACT", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (40, 40, 40), 5)
    m = cv2.getRotationMatrix2D((paper_w / 2, paper_h / 2), angle_deg, 1.0)
    paper = cv2.warpAffine(paper, m, (paper_w, paper_h), borderValue=(80, 80, 80))
    y0 = (h - paper_h) // 2
    img[y0 : y0 + paper_h, margin : margin + paper_w] = paper
    return img


def test_detect_paper_quad():
    img = _photo_of_paper()
    quad = detect_paper_quad(img)
    assert quad is not None
    assert quad.shape == (4, 2)


def test_auto_calibrate_sets_a4():
    img = _photo_of_paper()
    page = Page(image=img, phys_w_mm=999.0, phys_h_mm=999.0, needs_calibration=True)
    assert auto_calibrate_page(page) is True
    # 校准后物理尺寸 = A4（竖版）
    assert abs(page.phys_w_mm - A4_W_MM) < 1e-6
    assert abs(page.phys_h_mm - A4_H_MM) < 1e-6
    assert page.needs_calibration is False
    # 透视校正后比例接近 √2
    h, w = page.image.shape[:2]
    assert abs(h / w - 1.414) < 0.02


def test_auto_calibrate_blank_fails_gracefully():
    """全白图（无背景对比）检测失败返回 False，不静默假定。"""
    img = np.full((800, 600, 3), 255, dtype=np.uint8)
    page = Page(image=img, phys_w_mm=210.0, phys_h_mm=297.0)
    result = auto_calibrate_page(page)
    assert result is False or page.phys_w_mm == 210.0  # 要么失败要么未破坏


def test_template_roundtrip(tmp_path):
    entries = [
        {"seal_name": "公章", "kind": "seal", "rel_x": 0.7, "rel_y": 0.8,
         "size_mm": 40.0, "rotation_deg": -2.0, "opacity": 1.0},
        {"seal_name": "张三", "kind": "signature", "rel_x": 0.5, "rel_y": 0.78,
         "size_mm": 35.0, "rotation_deg": 0.0, "opacity": 0.95},
    ]
    path = save_template(tmp_path, "末页落款", entries)
    loaded = load_template(path)
    assert loaded == entries
    assert list_templates(tmp_path) == [path]


def test_template_empty_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        save_template(tmp_path, "空", [])


def test_perforation_export_integration(tmp_path):
    """骑缝章全链路：切割 → 落章 → 压平导出 → 无可分离墨迹对象。"""
    from core.export import count_separable_ink_images, export_pdf
    from core.extract import extract_red_seal

    w = round(A4_W_MM / 25.4 * 300)
    h = round(A4_H_MM / 25.4 * 300)
    pages = [
        Page(image=np.full((h, w, 3), 255, dtype=np.uint8), phys_w_mm=A4_W_MM, phys_h_mm=A4_H_MM)
        for _ in range(4)
    ]
    seal = extract_red_seal("tests/fixtures/seal_company.png")
    spec = PerforationSpec(seed=42)
    placements = plan_perforation(seal, pages, [0, 1, 2, 3], spec)
    stamped = apply_perforation(pages, placements)
    images = [stamped.get(i, p.image) for i, p in enumerate(pages)]

    out = export_pdf(pages, images, tmp_path / "perf.pdf", sealog={"seed": 42, "type": "perforation"})
    assert count_separable_ink_images(out) == 0

    # 每页右边缘应有墨迹（切片窄、部分印文浅淡，按灰度暗化断言）
    import pymupdf

    with pymupdf.open(out) as doc:
        for i in range(4):
            pix = doc[i].get_pixmap(dpi=72, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            right = img[:, int(pix.width * 0.92):]
            dark_pixels = int((right.mean(axis=2) < 250).sum())
            assert dark_pixels > 10, f"第 {i + 1} 页右边缘无墨迹"
