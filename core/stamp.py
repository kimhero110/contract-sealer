"""盖章合成：把印章/签名按物理尺寸盖到页面图像上。

核心约定（方案 §4.3/§4.2）：
- 位置、尺寸一律使用物理 mm；像素换算只在此处发生；
- 混合模式为正片叠底（multiply），模拟印泥/墨迹压在纸上；
- 函数纯图像进图像出，与任何 UI 无关。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .document import Page, mm_to_px


@dataclass(frozen=True)
class Placement:
    """一次落章的全部参数（物理单位）。"""

    center_x_mm: float   # 章中心相对页面左上角的物理位置
    center_y_mm: float
    size_mm: float       # 章=直径；签名=宽度
    scale: float = 1.0   # 物理 1:1 基础上的用户缩放（±50% 由 UI 限制）
    opacity: float = 1.0
    rotation_deg: float = 0.0  # 与画布 Qt setRotation 同语义（导出侧内部会取负适配 PIL）


def stamp_page(page: Page, seal_rgba: np.ndarray, placement: Placement) -> np.ndarray:
    """在页面上盖章，返回新的 RGB 页面图像（原图像不被修改）。

    seal_rgba：已完成抠图与随机效果的透明墨迹图。
    """
    dpi = page.dpi
    scale = placement.scale
    if not (0.1 <= scale <= 10.0):
        raise ValueError(f"缩放超出合理范围: {scale}")
    opacity = float(np.clip(placement.opacity, 0.0, 1.0))

    # 1) 印章缩放到物理尺寸对应的像素大小
    target_px = max(1, round(mm_to_px(placement.size_mm * scale, dpi)))
    resized = _resize_keep_alpha(seal_rgba, target_px)

    # 2) 手动旋转（expand 防裁切）
    # 符号约定：Placement.rotation_deg 与画布 Qt setRotation 同语义；
    # PIL rotate 方向与之相反——必须取负，否则用户在预览里拧正的角度
    # 会在导出时反向旋转（误差 = 修正量的两倍，90° 修正变 180° 倒立）。
    if abs(placement.rotation_deg) > 1e-6:
        resized = _rotate_rgba(resized, -placement.rotation_deg)

    # 3) 正片叠底合成
    cx = mm_to_px(placement.center_x_mm, dpi)
    cy = mm_to_px(placement.center_y_mm, dpi)
    return _multiply_composite(page.image, resized, cx, cy, opacity)


def measure_ink_diameter_mm(page_image: np.ndarray, page: Page) -> float:
    """测量页面图像中红色墨迹区域的外接圆直径（mm）。验收/测试用。

    阈值 s>45：兼顾公章淡红外圈（阈值过高会丢失外圈导致测量偏小）
    与扫描件底色干扰（阈值过低会误检）。
    """
    hsv = cv2.cvtColor(page_image, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (((h <= 14) | (h >= 160)) & (s > 45) & (v > 60)).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        raise ValueError("页面上未检测到红色墨迹")
    diameter_px = max(xs.max() - xs.min(), ys.max() - ys.min())
    return float(diameter_px / page.dpi * 25.4)


def _resize_keep_alpha(rgba: np.ndarray, target_width_px: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    if w == target_width_px:
        return rgba
    target_h = max(1, round(h * target_width_px / w))
    interp = cv2.INTER_AREA if target_width_px < w else cv2.INTER_CUBIC
    return cv2.resize(rgba, (target_width_px, target_h), interpolation=interp)


def _rotate_rgba(rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    from PIL import Image

    im = Image.fromarray(rgba, "RGBA")
    return np.array(im.rotate(angle_deg, resample=Image.BICUBIC, expand=True))


def _multiply_composite(
    base_rgb: np.ndarray, ink_rgba: np.ndarray, cx: float, cy: float, opacity: float
) -> np.ndarray:
    """正片叠底：out = base*(1-a) + (base*ink/255)*a。允许 ink 部分超出页面（自动裁切）。"""
    bh, bw = base_rgb.shape[:2]
    ih, iw = ink_rgba.shape[:2]

    x0 = round(cx - iw / 2)
    y0 = round(cy - ih / 2)
    # 页面内可见区域
    px0, py0 = max(0, x0), max(0, y0)
    px1, py1 = min(bw, x0 + iw), min(bh, y0 + ih)
    if px0 >= px1 or py0 >= py1:
        return base_rgb.copy()
    # ink 对应区域
    ix0, iy0 = px0 - x0, py0 - y0
    ix1, iy1 = ix0 + (px1 - px0), iy0 + (py1 - py0)

    region = base_rgb[py0:py1, px0:px1].astype(np.float32)
    ink_rgb = ink_rgba[iy0:iy1, ix0:ix1, :3].astype(np.float32)
    alpha = ink_rgba[iy0:iy1, ix0:ix1, 3:4].astype(np.float32) / 255.0 * opacity

    multiplied = region * (ink_rgb / 255.0)
    out_region = region * (1 - alpha) + multiplied * alpha

    out = base_rgb.copy()
    out[py0:py1, px0:px1] = np.clip(out_region, 0, 255).astype(np.uint8)
    return out
