"""日期戳：把日期文字渲染成透明墨迹，走和印章完全一样的落章/导出管线。

设计取舍：
- 不做随机手感。章有印泥深浅，打印/书写的日期没有，加蒙尘反而假；
- 物理尺寸以"墨迹实际高度"为准：先大号渲染再裁到墨迹包围盒、缩放到目标
  高度，所以用户填 4.5mm 就是打印出来量得到的 4.5mm，与字体的行高无关；
- 字体缺字必须报错而不是画方框：中文格式在没有中文字体的机器上会静默
  渲染成一排豆腐块，盖到合同上才发现就晚了。
"""

from __future__ import annotations

import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .document import MM_PER_INCH, mm_to_px

KIND_DATE = "date"

# (下拉框里显示的样例, strftime 格式)
DATE_FORMATS: list[tuple[str, str]] = [
    ("2026年09月11日", "%Y年%m月%d日"),
    ("2026-09-11", "%Y-%m-%d"),
    ("2026/09/11", "%Y/%m/%d"),
    ("2026.09.11", "%Y.%m.%d"),
]

DEFAULT_DATE_FORMAT = DATE_FORMATS[0][1]
DEFAULT_HEIGHT_MM = 4.5          # 约等于合同正文字号
DEFAULT_INK = (32, 36, 44)       # 近黑，略偏冷，和签字笔一致

# 渲染基准字号：足够大以保证缩小到目标高度时边缘干净
_BASE_PX = 220

_FONT_CANDIDATES = {
    "win32": [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhl.ttc",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ],
    "darwin": [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ],
}
_FONT_FALLBACKS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/wqy-microhei/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def format_date(d: date, fmt: str = DEFAULT_DATE_FORMAT) -> str:
    """按 strftime 格式渲染日期文本。默认取系统当天由调用方传入。"""
    return d.strftime(fmt)


def today_text(fmt: str = DEFAULT_DATE_FORMAT) -> str:
    return format_date(date.today(), fmt)


@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """按平台优先级找一个可用字体。全都找不到时退到 Pillow 内置字体。"""
    for path in _FONT_CANDIDATES.get(sys.platform, []) + _FONT_FALLBACKS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def font_supports(font, text: str) -> bool:
    """字体是否真的有这些字的字形。

    缺字时 FreeType 会画 .notdef（豆腐块/空白），和真字形一样有 bitmap，
    所以用"和一个必然缺失的码位渲染结果相同"来判定。
    """
    try:
        missing = font.getmask("\U0010fffd", mode="L")
        missing_bytes = bytes(missing)
    except Exception:
        return True  # 判定不了就别拦着用户
    for ch in text:
        if ch.isspace():
            continue
        try:
            mask = font.getmask(ch, mode="L")
        except Exception:
            return False
        if bytes(mask) == missing_bytes:
            return False
    return True


def missing_glyphs(text: str) -> str:
    """返回当前字体渲染不出来的字符，空串表示都能渲染。"""
    font = _load_font(_BASE_PX)
    return "".join(
        ch for ch in dict.fromkeys(text) if not ch.isspace() and not font_supports(font, ch)
    )


def render_date_ink(
    text: str,
    height_mm: float = DEFAULT_HEIGHT_MM,
    dpi: float = 300.0,
    color: tuple[int, int, int] = DEFAULT_INK,
) -> np.ndarray:
    """把日期文字渲染成 RGBA 墨迹，墨迹高度精确等于 height_mm。

    缺字时抛 ValueError，由调用方提示换格式或装字体。
    """
    if not text.strip():
        raise ValueError("日期文本为空")
    if height_mm <= 0:
        raise ValueError("日期字高必须为正数")

    font = _load_font(_BASE_PX)
    missing = "".join(
        ch for ch in dict.fromkeys(text) if not ch.isspace() and not font_supports(font, ch)
    )
    if missing:
        raise ValueError(
            f"当前系统字体渲染不出「{missing}」。请换用纯数字日期格式，或安装中文字体。"
        )

    # 1) 大号渲染到白底，取灰度作为墨量
    pad = _BASE_PX // 2
    box = font.getbbox(text)
    canvas = Image.new("L", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), 255)
    ImageDraw.Draw(canvas).text((pad - box[0], pad - box[1]), text, font=font, fill=0)
    alpha = 255 - np.array(canvas)

    # 2) 裁到墨迹包围盒
    ys, xs = np.nonzero(alpha > 8)
    if len(ys) == 0:
        raise ValueError("日期文本渲染为空")
    alpha = alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    # 3) 缩放到目标物理高度（墨迹高度即用户填的 height_mm）
    target_h = max(1, round(mm_to_px(height_mm, dpi)))
    target_w = max(1, round(alpha.shape[1] * target_h / alpha.shape[0]))
    alpha_img = Image.fromarray(alpha).resize((target_w, target_h), Image.LANCZOS)

    rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2] = color
    rgba[:, :, 3] = np.array(alpha_img)
    return rgba


def ink_width_mm(ink: np.ndarray, height_mm: float) -> float:
    """墨迹的物理宽度（落章时 size_mm 按宽度计，与签名同语义）。"""
    h, w = ink.shape[:2]
    return w / h * height_mm


def px_to_mm_height(ink: np.ndarray, dpi: float) -> float:
    """墨迹的物理高度，供信息栏显示。"""
    return ink.shape[0] / dpi * MM_PER_INCH
