"""自动纸边检测（M3）：OpenCV 找纸面四边形 → 透视校正 → 校准为 A4。

适用场景：拍照件/带黑边扫描件，纸面通常比背景亮。
检测失败（返回 False）时调用方应回退到手动校准，不得静默假定。
"""

from __future__ import annotations

import cv2
import numpy as np

from .document import A4_H_MM, A4_W_MM, Page


def detect_paper_quad(img: np.ndarray) -> np.ndarray | None:
    """检测纸面四边形顶点（4x2，顺时针从左上）。失败返回 None。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Otsu 阈值：纸（亮）与背景（暗）分离
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    img_area = img.shape[0] * img.shape[1]
    for c in contours[:5]:
        area = cv2.contourArea(c)
        if area < img_area * 0.25:  # 纸面至少占画面 1/4
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return _order_quad(approx.reshape(4, 2).astype(np.float32))
        # 四边形拟合失败时用最小外接矩形兜底
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(np.float32)
        if area > img_area * 0.4:
            return _order_quad(box)
    return None


def auto_calibrate_page(page: Page) -> bool:
    """检测纸边 → 透视校正页面图像 → 物理尺寸设为 A4。成功返回 True。"""
    quad = detect_paper_quad(page.image)
    if quad is None:
        return False
    return warp_to_a4(page, quad) is not None


def warp_to_a4(page: Page, quad: np.ndarray, min_area_ratio: float = 0.1) -> np.ndarray | None:
    """按给定四边形顶点透视校正页面为标准 A4（四点纸边校准的核心，修改意见）。

    quad：4x2 像素坐标（任意顺序，内部自动排序为左上/右上/右下/左下）。
    变换后页面图像被拉伸裁正为 A4 比例，物理尺寸设为 210×297mm。
    返回 3x3 变换矩阵 H（旧页像素 → 新页像素，用于映射已盖章坐标）；失败返回 None。
    """
    ordered = _order_quad(np.asarray(quad, dtype=np.float32))

    # 防御：四点围成的面积太小（共线/挤在一起）时拒绝，避免产出垃圾
    area = cv2.contourArea(ordered)
    img_area = page.image.shape[0] * page.image.shape[1]
    if area < img_area * min_area_ratio:
        return None

    # 目标尺寸：保持原图约等分辨率，A4 比例
    w_top = np.linalg.norm(ordered[1] - ordered[0])
    w_bot = np.linalg.norm(ordered[2] - ordered[3])
    h_left = np.linalg.norm(ordered[3] - ordered[0])
    h_right = np.linalg.norm(ordered[2] - ordered[1])
    out_w = int(max(w_top, w_bot))
    out_h = int(max(h_left, h_right))
    landscape = out_w > out_h
    if landscape:
        out_w, out_h = max(out_w, out_h), min(out_w, out_h)
    # 强制 A4 比例（纸是 A 系，比例即 √2）。
    # 注意横竖方向：竖版 H/W = 297/210，横版 H/W = 210/297——
    # 曾经不分方向一律乘 297/210，横版纸被拉成竖版图而物理尺寸是横版（图像与物理矛盾）。
    if landscape:
        out_h = round(out_w * (A4_W_MM / A4_H_MM))
    else:
        out_h = round(out_w * (A4_H_MM / A4_W_MM))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(page.image, m, (out_w, out_h))

    page.image = warped
    if landscape:
        page.phys_w_mm, page.phys_h_mm = A4_H_MM, A4_W_MM
    else:
        page.phys_w_mm, page.phys_h_mm = A4_W_MM, A4_H_MM
    page.needs_calibration = False
    return m


def map_points_through(homography: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """把像素点经透视矩阵 H 映射到新坐标（用于校准后已盖章位置跟随）。"""
    pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """四点排序为 [左上, 右上, 右下, 左下]。"""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # 左上
    ordered[2] = pts[np.argmax(s)]   # 右下
    ordered[1] = pts[np.argmin(d)]   # 右上
    ordered[3] = pts[np.argmax(d)]   # 左下
    return ordered
