"""日期戳测试：物理高度精确、格式可选、缺字必须报错而不是画豆腐块。"""

from datetime import date

import numpy as np
import pytest

from core.datestamp import (
    DATE_FORMATS,
    format_date,
    ink_width_mm,
    missing_glyphs,
    render_date_ink,
    today_text,
)

ISO = "%Y-%m-%d"


def test_format_date_matches_strftime():
    assert format_date(date(2026, 9, 11), ISO) == "2026-09-11"
    assert format_date(date(2026, 9, 11), "%Y年%m月%d日") == "2026年09月11日"


def test_today_text_uses_system_date():
    assert today_text(ISO) == date.today().strftime(ISO)


def test_all_listed_formats_are_valid_strftime():
    for sample, fmt in DATE_FORMATS:
        rendered = format_date(date(2026, 9, 11), fmt)
        assert len(rendered) == len(sample), f"{fmt} 渲染长度与样例 {sample} 不符"


def test_ink_height_is_exactly_requested_mm():
    """用户填 4.5mm，打印出来量到的就得是 4.5mm（容差半个像素）。"""
    dpi = 300.0
    for height_mm in (3.0, 4.5, 8.0):
        ink = render_date_ink("2026-09-11", height_mm, dpi)
        actual_mm = ink.shape[0] / dpi * 25.4
        assert abs(actual_mm - height_mm) < 25.4 / dpi


def test_ink_is_transparent_outside_glyphs():
    ink = render_date_ink("2026-09-11", 4.5, 300.0)
    assert ink.shape[2] == 4
    assert ink[:, :, 3].max() == 255      # 笔画中心是实心的
    assert ink[:, :, 3].min() == 0        # 字间空白透明
    assert (ink[:, :, 3] > 8).mean() < 0.6  # 不是一块实心矩形


def test_ink_width_scales_with_text_length():
    short = render_date_ink("2026", 4.5, 300.0)
    long = render_date_ink("2026-09-11", 4.5, 300.0)
    assert long.shape[1] > short.shape[1]
    assert ink_width_mm(long, 4.5) > ink_width_mm(short, 4.5)


def test_higher_dpi_gives_more_pixels_same_physical_size():
    low = render_date_ink("2026-09-11", 4.5, 150.0)
    high = render_date_ink("2026-09-11", 4.5, 600.0)
    assert high.shape[0] > low.shape[0] * 3
    assert abs(ink_width_mm(low, 4.5) - ink_width_mm(high, 4.5)) < 0.5


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        render_date_ink("   ", 4.5, 300.0)
    with pytest.raises(ValueError):
        render_date_ink("2026-09-11", 0.0, 300.0)


def test_missing_glyphs_reported_not_rendered_as_boxes():
    """缺字必须抛错。静默画一排豆腐块盖到合同上才发现就晚了。"""
    missing = missing_glyphs("2026年09月11日")
    if not missing:
        pytest.skip("当前系统有中文字体，测不到缺字路径")
    with pytest.raises(ValueError) as excinfo:
        render_date_ink("2026年09月11日", 4.5, 300.0)
    assert "年" in str(excinfo.value)


def test_ascii_date_always_renderable():
    """纯数字格式在任何有字体的机器上都必须能渲染（缺字时的兜底格式）。"""
    assert missing_glyphs("2026-09-11") == ""
    assert render_date_ink("2026-09-11", 4.5, 300.0).size > 0


def test_ink_color_is_applied():
    ink = render_date_ink("2026-09-11", 4.5, 300.0, color=(200, 10, 10))
    opaque = ink[ink[:, :, 3] > 200]
    assert len(opaque) > 0
    assert np.all(opaque[:, 0] == 200) and np.all(opaque[:, 1] == 10)
