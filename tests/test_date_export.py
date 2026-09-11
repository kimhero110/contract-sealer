"""日期戳端到端（不经 GUI）：落章 → 导出 PDF → 物理尺寸与输出目录都对。"""

import numpy as np
import pymupdf

from core.datestamp import ink_width_mm, render_date_ink
from core.document import A4_H_MM, A4_W_MM, Page
from core.export import export_pdf, make_output_path
from core.settings import Settings
from core.stamp import Placement, stamp_page


def _blank_page(dpi: int = 300) -> Page:
    w = round(A4_W_MM / 25.4 * dpi)
    h = round(A4_H_MM / 25.4 * dpi)
    return Page(
        image=np.full((h, w, 3), 255, dtype=np.uint8),
        phys_w_mm=A4_W_MM,
        phys_h_mm=A4_H_MM,
    )


def _ink_bbox_mm(img: np.ndarray, page: Page) -> tuple[float, float]:
    """页面上墨迹包围盒的物理宽高（mm）。"""
    dark = img.min(axis=2) < 200
    ys, xs = np.nonzero(dark)
    assert len(xs) > 0, "页面上没有墨迹"
    w_px = xs.max() - xs.min() + 1
    h_px = ys.max() - ys.min() + 1
    return w_px / page.dpi * 25.4, h_px / page.dpi * 25.4


def test_date_lands_at_requested_physical_height():
    """盖到页面上的日期，实测高度就是用户填的字高。"""
    page = _blank_page()
    height_mm = 5.0
    ink = render_date_ink("2026-09-11", height_mm, page.dpi)
    out = stamp_page(
        page,
        ink,
        Placement(
            center_x_mm=150.0,
            center_y_mm=250.0,
            size_mm=ink_width_mm(ink, height_mm),
        ),
    )
    w_mm, h_mm = _ink_bbox_mm(out, page)
    assert abs(h_mm - height_mm) < 0.3
    assert abs(w_mm - ink_width_mm(ink, height_mm)) < 0.3


def test_date_lands_where_clicked():
    page = _blank_page()
    ink = render_date_ink("2026-09-11", 4.5, page.dpi)
    cx, cy = 120.0, 260.0
    out = stamp_page(page, ink, Placement(cx, cy, ink_width_mm(ink, 4.5)))
    dark = out.min(axis=2) < 200
    ys, xs = np.nonzero(dark)
    got_cx = (xs.min() + xs.max()) / 2 / page.dpi * 25.4
    got_cy = (ys.min() + ys.max()) / 2 / page.dpi * 25.4
    assert abs(got_cx - cx) < 0.5
    assert abs(got_cy - cy) < 0.5


def test_export_writes_into_chosen_output_dir(tmp_path):
    """选定输出目录后，导出件与 .sealog 都落在那里，源目录一个文件不多。"""
    source_dir = tmp_path / "合同原件"
    out_dir = tmp_path / "已盖章"
    source_dir.mkdir()
    out_dir.mkdir()
    source = source_dir / "劳动合同.pdf"

    page = _blank_page()
    ink = render_date_ink("2026-09-11", 4.5, page.dpi)
    img = stamp_page(page, ink, Placement(150.0, 250.0, ink_width_mm(ink, 4.5)))

    settings = Settings(output_dir=str(out_dir))
    target = make_output_path(source, settings.resolved_output_dir(source))
    written = export_pdf([page], [img], target, {"stamps": []})

    assert written.parent == out_dir
    assert written.exists()
    assert written.with_suffix(".sealog").exists()
    assert list(source_dir.iterdir()) == []


def test_exported_pdf_keeps_a4_physical_size(tmp_path):
    page = _blank_page()
    ink = render_date_ink("2026-09-11", 4.5, page.dpi)
    img = stamp_page(page, ink, Placement(150.0, 250.0, ink_width_mm(ink, 4.5)))
    out = export_pdf([page], [img], tmp_path / "out.pdf", {"stamps": []})
    with pymupdf.open(out) as doc:
        rect = doc[0].rect
    assert abs(rect.width / 72 * 25.4 - A4_W_MM) < 0.5
    assert abs(rect.height / 72 * 25.4 - A4_H_MM) < 0.5


def test_default_output_dir_is_beside_source(tmp_path):
    source = tmp_path / "合同.pdf"
    assert Settings().resolved_output_dir(source) == tmp_path
