"""墨迹提取：把白底照片上的印章（红）或签名（蓝/黑）抠成透明 PNG。

方案 v1.2 §4.5：算法从"红色分割"泛化为任意墨色提取。
- 红章：HSV 红色区间分割，红色纯度 → alpha（保留印泥浓淡）；
- 深色墨迹（签名）：相对纸面亮度的暗度/蓝色差分 → alpha；
- 连通域过滤去纸面噪点（阈值相对墨迹自身尺寸，小数字不会被当噪点）；
- 结果裁剪到墨迹包围盒。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

KIND_SEAL = "seal"
KIND_SIGNATURE = "signature"


def _load_rgb(image: str | Path | np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[2] == 4:
            # RGBA：先合成到白底（素材可能本身带透明通道）
            alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
            arr = (arr[:, :, :3].astype(np.float32) * alpha + 255.0 * (1 - alpha)).astype(
                np.uint8
            )
        return arr[:, :, :3]
    if isinstance(image, Image.Image):
        return _load_rgb(np.array(image.convert("RGBA")))
    with Image.open(image) as im:
        # EXIF 方向自愈：手机拍的印章/签名照片横躺时自动转正
        from PIL import ImageOps

        return _load_rgb(np.array(ImageOps.exif_transpose(im).convert("RGBA")))


# 噪点判定尺度：以墨迹自身的外接尺寸 D 为基准，边长 D×NOISE_SCALE 的方块以下算噪点。
# 不能按整张图片面积算——12MP 手机照片里一枚 800px 的章，"整图 0.05%" 是 6000px，
# 而印文里一个小数字只有一两百像素，全部被当噪点扔掉（"公章里偶发丢几个小数字"）。
NOISE_SCALE = 0.005
# 估计墨迹外接范围时只看"大块"：面积不到最大连通域 1% 的碎片不参与定界
_EXTENT_MIN_RATIO = 0.01
# 外接范围向外放宽的比例：范围之外的红/黑点是纸面污渍，不是印文
_EXTENT_MARGIN = 0.15


def _remove_noise(alpha: np.ndarray, min_alpha: float = 0.15) -> np.ndarray:
    """连通域过滤：去掉孤立噪点，保留墨迹——包括印文里最小的那个数字。

    判定规则全部相对于墨迹自身：
    - 先用大块连通域（≥ 最大块的 1%）估计墨迹外接范围 D（章≈直径，签名≈宽度）；
    - 面积 ≥ (D×0.5%)² 且落在外接范围（外扩 15%）内的连通域保留，其余视为噪点。
      800px 的章 → 阈值 16px；2000px 的章 → 100px；一个小数字有几百到上千像素，稳稳保住。

    不再做 3×3 形态学开运算：开运算会把宽度 ≤2px 的笔画整条抹掉，
    小尺寸素材里的数字和签名细笔画正是这么没的。单像素噪点由面积下限兜住。
    """
    mask = (alpha > min_alpha).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return alpha * mask.astype(np.float32)

    areas = stats[1:, cv2.CC_STAT_AREA]
    big = np.nonzero(areas >= max(1, areas.max() * _EXTENT_MIN_RATIO))[0] + 1
    x0 = int(stats[big, cv2.CC_STAT_LEFT].min())
    y0 = int(stats[big, cv2.CC_STAT_TOP].min())
    x1 = int((stats[big, cv2.CC_STAT_LEFT] + stats[big, cv2.CC_STAT_WIDTH]).max())
    y1 = int((stats[big, cv2.CC_STAT_TOP] + stats[big, cv2.CC_STAT_HEIGHT]).max())
    extent = max(x1 - x0, y1 - y0, 1)
    margin = extent * _EXTENT_MARGIN
    min_area = max(6, round((extent * NOISE_SCALE) ** 2))

    cx, cy = centroids[1:, 0], centroids[1:, 1]
    inside = (cx >= x0 - margin) & (cx <= x1 + margin) & (cy >= y0 - margin) & (cy <= y1 + margin)
    keep_ids = np.nonzero((areas >= min_area) & inside)[0] + 1
    keep = np.isin(labels, keep_ids)
    return alpha * keep.astype(np.float32)


def _crop_to_ink(rgba: np.ndarray, margin: int = 4) -> np.ndarray:
    alpha = rgba[:, :, 3]
    ys, xs = np.nonzero(alpha > 8)
    if len(ys) == 0:
        return rgba
    y0, y1 = max(0, ys.min() - margin), min(rgba.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(rgba.shape[1], xs.max() + margin + 1)
    return rgba[y0:y1, x0:x1]


def extract_red_seal(image: str | Path | np.ndarray | Image.Image, strength: float = 1.0) -> np.ndarray:
    """提取红色印文，返回 RGBA（红得越纯 alpha 越高）。"""
    rgb = _load_rgb(image)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s = hsv[:, :, 0], hsv[:, :, 1] / 255.0

    # 红色色相区间（OpenCV H: 0-179，红色跨 0 点）
    red_hue = ((h <= 14) | (h >= 160)).astype(np.float32)
    # 红色纯度：R 超出 G/B 的程度，叠加饱和度
    redness = np.clip((r - np.maximum(g, b)) / 255.0, 0.0, 1.0)
    alpha = red_hue * np.clip(redness * 1.6, 0.0, 1.0) * np.clip(s * 1.5, 0.0, 1.0)
    # 纸面白色自动变透明；暗红印泥（V 低但 hue 对）保留
    alpha = np.clip(alpha * strength, 0.0, 1.0)
    alpha = _remove_noise(alpha)

    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    return _crop_to_ink(rgba)


def extract_dark_ink(image: str | Path | np.ndarray | Image.Image, strength: float = 1.0) -> np.ndarray:
    """提取深色墨迹（蓝/黑签名），返回 RGBA（墨越浓 alpha 越高）。"""
    rgb = _load_rgb(image)
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    v = hsv[:, :, 2]

    # 估计纸面亮度：取高亮分位数作为"纸白"
    paper_v = float(np.percentile(v, 95))
    paper_v = max(paper_v, 128.0)
    # 相对纸面的暗度
    darkness = np.clip((paper_v - v) / max(paper_v * 0.6, 1.0), 0.0, 1.0)
    # 蓝色墨迹增强：B 明显高于 R/G
    blue_excess = np.clip((b - np.maximum(r, g)) / 255.0 * 2.0, 0.0, 1.0)

    alpha = np.clip(np.maximum(darkness, blue_excess * 0.9) * 1.4 * strength, 0.0, 1.0)
    alpha = _remove_noise(alpha)

    rgba = np.dstack([rgb, (alpha * 255).astype(np.uint8)])
    return _crop_to_ink(rgba)


def detect_kind(image: str | Path | np.ndarray | Image.Image) -> str:
    """自动判断素材类型：红色像素占比显著则为印章，否则按签名处理。"""
    rgb = _load_rgb(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    red_ratio = float(np.mean(((h <= 14) | (h >= 160)) & (s > 60)))
    return KIND_SEAL if red_ratio > 0.01 else KIND_SIGNATURE


def extract_ink(
    image: str | Path | np.ndarray | Image.Image,
    kind: str = "auto",
    strength: float = 1.0,
) -> np.ndarray:
    """统一入口：按类型提取墨迹为透明 RGBA。"""
    if kind == "auto":
        kind = detect_kind(image)
    if kind == KIND_SEAL:
        return extract_red_seal(image, strength)
    if kind == KIND_SIGNATURE:
        return extract_dark_ink(image, strength)
    raise ValueError(f"未知类型: {kind}")
