"""骑缝章：随机宽度切片平铺算法 + 逐页放置 + 拼合预览。

方案 v1.3 §4.4（唯一定义）：
- 几何模型 = 切片平铺：每条切片完整落在页面内，右侧贴齐纸边（内缩 inset）；
- 页序映射固定：页码升序 = 切片从左到右；
- 随机宽度切分：各切片宽度之和恒等于印章总宽（拼合数学上必然还原）；
- 逐页微抖动有硬上限（垂直 ≤1mm、旋转 ≤2°），拼合断言用带容差相似度；
- 最小切片宽度约束：W/N < 1.5mm 时拒绝（切片过窄骑缝失效）；
- y 按各页纸边独立度量（混尺寸页面安全）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .document import Page, mm_to_px
from .stamp import _multiply_composite, _rotate_rgba

SIDE_LEFT = "left"
SIDE_RIGHT = "right"

MIN_SLICE_WIDTH_MM = 1.5

# 逐页抖动硬上限（方案 §4.4）
CAP_OFFSET_JITTER_MM = 1.0
CAP_ROT_JITTER_DEG = 2.0
CAP_WIDTH_JITTER = 0.4


def detect_paper_edge_x_px(img: np.ndarray, side: str, search_ratio: float = 0.25) -> int:
    """检测真实纸边的 x 像素位置（"自适应完美位置"的核心）。

    扫描件的图像边缘 ≠ 纸张边缘（自带白边/灰边）。用列亮度剖面找
    "纸（亮）→ 背景（暗）"的过渡带，返回最靠外的纸面列：
    - 中部 60% 高度做剖面，避开角落污渍/装订孔；
    - 阈值取纸内参考亮度的 80%，兼顾渐晕（扫描仪边缘发暗）；
    - 全幅扫描（纸铺满画面、无过渡带）回退为图像边缘；
    - 检测结果是像素，调用方负责换算 mm（各页 DPI 可能不同）。
    """
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    band = gray[int(h * 0.2) : int(h * 0.8)]
    profile = band.mean(axis=0).astype(np.float32)
    # 平滑抑制噪点
    profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 15), 0).ravel()

    # 纸内参考亮度：中央区域中位数
    paper_ref = float(np.median(band[:, w // 3 : 2 * w // 3]))
    thr = paper_ref * 0.80

    limit = int(w * (1 - search_ratio))
    if side == SIDE_RIGHT:
        xs = range(w - 1, limit, -1)
        fallback = w - 1
    else:
        xs = range(0, w - limit)
        fallback = 0
    for x in xs:
        if profile[x] >= thr:
            return x  # 最靠外的"纸面"列
    return fallback


@dataclass(frozen=True)
class PerforationSpec:
    """一次骑缝章的全部参数。"""

    diameter_mm: float = 40.0
    side: str = SIDE_RIGHT
    inset_mm: float = 1.5          # 切片右边缘距纸边的内缩
    y_mm: float | None = None      # 切片顶部距页顶；None = 页面垂直居中
    rotation_deg: float = 0.0      # 印章整体旋转（切割前）
    width_jitter: float = 0.2      # 切片宽度 ±比例（0 = 等宽）
    offset_jitter_mm: float = 0.5  # 逐页垂直抖动幅度
    rot_jitter_deg: float = 1.0    # 逐页旋转抖动幅度
    seed: int = 0
    auto_edge: bool = True         # 逐页检测真实纸边并贴合（关闭则用图像边缘）

    def clamped(self) -> "PerforationSpec":
        return PerforationSpec(
            diameter_mm=self.diameter_mm,
            side=self.side,
            inset_mm=max(0.0, self.inset_mm),
            y_mm=self.y_mm,
            rotation_deg=self.rotation_deg,
            width_jitter=min(abs(self.width_jitter), CAP_WIDTH_JITTER),
            offset_jitter_mm=min(abs(self.offset_jitter_mm), CAP_OFFSET_JITTER_MM),
            rot_jitter_deg=min(abs(self.rot_jitter_deg), CAP_ROT_JITTER_DEG),
            seed=self.seed,
            auto_edge=self.auto_edge,
        )


@dataclass
class SlicePlacement:
    """一个切片放到一页上的全部信息（拼合预览与导出共用）。"""

    page_index: int            # 文档页码（0-based）
    slice_rgba: np.ndarray     # 已应用逐页抖动（旋转）的切片图
    right_edge_mm: float       # 切片右边缘 x（物理 mm，页坐标系）
    top_mm: float              # 切片顶部 y（物理 mm，页坐标系）
    y_offset_mm: float         # 实际垂直抖动量（拼合预览对齐用）
    width_px: int              # 切割宽度（原始，不含旋转扩边）


def min_slice_warning(diameter_mm: float, page_count: int) -> str | None:
    """最小切片宽度约束。返回 None 表示通过，否则返回提示文案。"""
    if page_count <= 0:
        return "页范围为空"
    w = diameter_mm / page_count
    if w < MIN_SLICE_WIDTH_MM:
        return (
            f"切片宽度 {w:.2f}mm 小于最小值 {MIN_SLICE_WIDTH_MM}mm"
            f"（{diameter_mm:g}mm 章 / {page_count} 页）。"
            "打印不可见、骑缝验证失效。请缩小页范围或加大印章直径。"
        )
    return None


def slice_widths_px(total_px: int, n: int, jitter: float, rng: np.random.Generator) -> list[int]:
    """随机宽度切分：返回 n 个整数宽度，之和恒等于 total_px，每个 ≥1px。"""
    if n <= 0:
        raise ValueError("切片数必须为正")
    if total_px < n:
        raise ValueError(f"印章像素宽 {total_px} 小于切片数 {n}")
    if n == 1:
        return [total_px]
    jitter = min(abs(jitter), CAP_WIDTH_JITTER)
    weights = rng.uniform(1.0 - jitter, 1.0 + jitter, size=n)
    weights /= weights.sum()
    raw = weights * total_px
    widths = np.floor(raw).astype(int)
    widths = np.maximum(widths, 1)
    # 余数按小数部分从大到小分配，保证总和恒等
    remainder = total_px - int(widths.sum())
    order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while remainder > 0:
        widths[order[i % n]] += 1
        remainder -= 1
        i += 1
    while remainder < 0:  # 防御：max(,1) 导致超出时从最大片回收
        j = int(np.argmax(widths))
        if widths[j] <= 1:
            break
        widths[j] -= 1
        remainder += 1
    return widths.tolist()


def slice_seal(rgba: np.ndarray, widths: list[int]) -> list[np.ndarray]:
    """竖向切分 RGBA 图为若干竖条。"""
    out = []
    x = 0
    for w in widths:
        out.append(rgba[:, x : x + w].copy())
        x += w
    return out


def plan_perforation(
    seal_rgba: np.ndarray,
    pages: list[Page],
    page_indices: list[int],
    spec: PerforationSpec,
) -> list[SlicePlacement]:
    """计算骑缝章全部切片的放置（不落章，纯计算——预览/导出/测试共用）。

    seal_rgba：已抠图、已应用全局随机效果（角度/色度/蒙尘）的印章图。
    """
    spec = spec.clamped()
    if not page_indices:
        raise ValueError("页范围为空")
    warn = min_slice_warning(spec.diameter_mm, len(page_indices))
    if warn:
        raise ValueError(warn)

    rng = np.random.default_rng(spec.seed)
    # 以第一页的 DPI 为切割分辨率基准（同一文档内 DPI 一致；混尺寸页面各自换算）
    ref_page = pages[page_indices[0]]
    total_px = max(len(page_indices), round(mm_to_px(spec.diameter_mm, ref_page.dpi)))

    # 整体旋转在切割前完成
    seal = seal_rgba
    if abs(spec.rotation_deg) > 1e-6:
        seal = _rotate_rgba(seal, spec.rotation_deg)
    seal = _resize_width(seal, total_px)

    widths = slice_widths_px(total_px, len(page_indices), spec.width_jitter, rng)
    slices = slice_seal(seal, widths)

    placements: list[SlicePlacement] = []
    for order, page_idx in enumerate(page_indices):
        page = pages[page_idx]
        sl = slices[order]

        # 逐页微抖动（有硬上限）
        dy_mm = float(rng.uniform(-spec.offset_jitter_mm, spec.offset_jitter_mm))
        drot = float(rng.uniform(-spec.rot_jitter_deg, spec.rot_jitter_deg))
        if abs(drot) > 1e-6:
            sl = _rotate_rgba(sl, drot)

        # 位置：右边缘贴 纸边 - inset；y 按各页独立度量
        if spec.auto_edge:
            # 逐页检测真实纸边（扫描件白边/灰边自适应），各页独立
            edge_px = detect_paper_edge_x_px(page.image, spec.side)
            paper_edge_mm = px_to_mm_at(edge_px + (1 if spec.side == SIDE_RIGHT else 0), page.dpi)
        else:
            paper_edge_mm = page.phys_w_mm if spec.side == SIDE_RIGHT else 0.0
        if spec.side == SIDE_RIGHT:
            right_edge_mm = paper_edge_mm - spec.inset_mm
        else:
            right_edge_mm = paper_edge_mm + spec.inset_mm + px_to_mm_at(sl.shape[1], page.dpi)
        if spec.y_mm is None:
            top_mm = (page.phys_h_mm - px_to_mm_at(sl.shape[0], page.dpi)) / 2
        else:
            top_mm = spec.y_mm
        top_mm += dy_mm

        placements.append(
            SlicePlacement(
                page_index=page_idx,
                slice_rgba=sl,
                right_edge_mm=right_edge_mm,
                top_mm=top_mm,
                y_offset_mm=dy_mm,
                width_px=widths[order],
            )
        )
    return placements


def apply_perforation(
    pages: list[Page], placements: list[SlicePlacement]
) -> dict[int, np.ndarray]:
    """把切片落章到各页，返回 {页码: 盖章后 RGB 图像}。"""
    out: dict[int, np.ndarray] = {}
    for pl in placements:
        page = pages[pl.page_index]
        img = out.get(pl.page_index, page.image)
        h_px, w_px = pl.slice_rgba.shape[:2]
        cx_mm = pl.right_edge_mm - px_to_mm_at(w_px, page.dpi) / 2
        cy_mm = pl.top_mm + px_to_mm_at(h_px, page.dpi) / 2
        cx_px = mm_to_px(cx_mm, page.dpi)
        cy_px = mm_to_px(cy_mm, page.dpi)
        out[pl.page_index] = _multiply_composite(img, pl.slice_rgba, cx_px, cy_px, 1.0)
    return out


def assemble_preview(
    placements: list[SlicePlacement], ref_dpi: float, pad_mm: float = 2.0
) -> np.ndarray:
    """拼合预览：按页序、实际宽度累积定位，渲染真实导出结果（所见即所得）。

    逐页垂直抖动也按实际值还原——用户看到的就是打印后拼起来的样子。
    """
    if not placements:
        raise ValueError("没有切片")
    ordered = sorted(placements, key=lambda p: p.page_index)
    total_w = sum(p.slice_rgba.shape[1] for p in ordered)
    max_h = max(p.slice_rgba.shape[0] for p in ordered)
    pad_px = round(mm_to_px(pad_mm, ref_dpi))
    jitter_range_px = max(
        (round(mm_to_px(abs(p.y_offset_mm), ref_dpi)) for p in ordered), default=0
    )
    canvas_h = max_h + 2 * (pad_px + jitter_range_px)
    canvas_w = total_w + 2 * pad_px
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    x = pad_px
    base_y = pad_px + jitter_range_px
    for p in ordered:
        sl = p.slice_rgba
        dy_px = round(mm_to_px(p.y_offset_mm, ref_dpi))
        region = canvas[base_y + dy_px : base_y + dy_px + sl.shape[0], x : x + sl.shape[1]]
        if region.shape[:2] != sl.shape[:2]:
            continue
        alpha = sl[:, :, 3:4].astype(np.float32) / 255.0
        ink = sl[:, :, :3].astype(np.float32)
        base = region.astype(np.float32)
        region[:] = np.clip(
            base * (1 - alpha) + (base * ink / 255.0) * alpha, 0, 255
        ).astype(np.uint8)
        x += sl.shape[1]
    return canvas


def px_to_mm_at(px: float, dpi: float) -> float:
    return px / dpi * 25.4


def _resize_width(rgba: np.ndarray, target_w: int) -> np.ndarray:
    import cv2

    h, w = rgba.shape[:2]
    if w == target_w:
        return rgba
    target_h = max(1, round(h * target_w / w))
    interp = cv2.INTER_AREA if target_w < w else cv2.INTER_CUBIC
    return cv2.resize(rgba, (target_w, target_h), interpolation=interp)
