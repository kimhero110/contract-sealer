"""画布/列表用的图像降采样与文件名排序小工具（与 Qt 无关，纯 numpy）。"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

# 屏显上限：~150DPI 的 A4 长边。全尺寸纹理只在深放大时才需要，翻页时纯浪费。
DISPLAY_MAX_LONG_SIDE = 1754


def display_image(img: np.ndarray, max_long_side: int = DISPLAY_MAX_LONG_SIDE) -> np.ndarray:
    """画布显示用降采样。导出仍走 300DPI 全尺寸，不受此函数影响。"""
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= max_long_side:
        return img
    scale = max_long_side / long_side
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def thumbnail(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def natural_key(path: str):
    """自然排序：第 2 页排在第 10 页前面（多图合成文档的页序）。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]
