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

    # 目标尺寸：保持原图约等分辨率，A4 比例
    w_top = np.linalg.norm(quad[1] - quad[0])
    w_bot = np.linalg.norm(quad[2] - quad[3])
    h_left = np.linalg.norm(quad[3] - quad[0])
    h_right = np.linalg.norm(quad[2] - quad[1])
    out_w = int(max(w_top, w_bot))
    out_h = int(max(h_left, h_right))
    landscape = out_w > out_h
    if landscape:
        out_w, out_h = max(out_w, out_h), min(out_w, out_h)
    # 强制 A4 比例（纸是 A 系，比例即 √2）
    out_h = round(out_w * (A4_H_MM / A4_W_MM))

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    m = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(page.image, m, (out_w, out_h))

    page.image = warped
    if landscape:
        page.phys_w_mm, page.phys_h_mm = A4_H_MM, A4_W_MM
    else:
        page.phys_w_mm, page.phys_h_mm = A4_W_MM, A4_H_MM
    page.needs_calibration = False
    return True


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
