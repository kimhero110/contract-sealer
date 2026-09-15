"""墨迹抠图测试：真实素材（红章 + 手写签名）。"""

import numpy as np

from core.extract import (
    KIND_SEAL,
    KIND_SIGNATURE,
    detect_kind,
    extract_dark_ink,
    extract_ink,
    extract_red_seal,
)


def _ink_stats(rgba: np.ndarray) -> dict:
    alpha = rgba[:, :, 3]
    return {
        "coverage": float(np.mean(alpha > 128)),
        "corner_alpha": int(max(alpha[0, 0], alpha[-1, -1], alpha[0, -1], alpha[-1, 0])),
        "shape": rgba.shape,
    }


def test_detect_kind(seal_png, signature_png):
    assert detect_kind(seal_png) == KIND_SEAL
    assert detect_kind(signature_png) == KIND_SIGNATURE


def test_extract_red_seal(seal_png):
    rgba = extract_red_seal(seal_png)
    stats = _ink_stats(rgba)
    # 公章文字+圆环+五角星应占画面相当比例
    assert 0.03 < stats["coverage"] < 0.6
    # 四角背景必须透明（背景残留断言）
    assert stats["corner_alpha"] == 0
    # 结果应裁剪到墨迹包围盒（小于原图）
    from PIL import Image

    with Image.open(seal_png) as im:
        assert rgba.shape[0] <= im.height and rgba.shape[1] <= im.width


def test_extract_signature(signature_png):
    rgba = extract_dark_ink(signature_png)
    stats = _ink_stats(rgba)
    # 签名笔画覆盖率较低但必须有墨迹
    assert stats["coverage"] > 0.005
    assert stats["corner_alpha"] == 0


def test_extract_ink_auto_dispatch(seal_png, signature_png):
    seal = extract_ink(seal_png, kind="auto")
    sig = extract_ink(signature_png, kind="auto")
    assert seal.shape[2] == 4 and sig.shape[2] == 4


def test_strength_increases_coverage(seal_png):
    weak = extract_red_seal(seal_png, strength=0.5)
    strong = extract_red_seal(seal_png, strength=2.0)
    # 强度越大 alpha 越高（裁剪尺寸可能不同，比较平均 alpha）
    assert strong[:, :, 3].astype(float).mean() >= weak[:, :, 3].astype(float).mean()


# ── 回归：印文里的小数字不能被当噪点扔掉 ──


def _seal_with_code(canvas_wh: tuple[int, int], seal_px: int, digit_px: int) -> np.ndarray:
    """合成：大白纸中央一枚只有外环 + 底部 13 位小数字编号的红章。数字之间留空不粘连。"""
    from PIL import Image, ImageDraw, ImageFont

    w, h = canvas_wh
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    red = (200, 30, 30)
    cx, cy, r = w // 2, h // 2, seal_px // 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=red, width=max(3, seal_px // 90))
    font = ImageFont.load_default(size=digit_px)  # Pillow 自带矢量字体，跨平台一致
    code = "1234567890123"
    x = cx - d.textlength(code, font=font) * 1.4 / 2
    for ch in code:
        d.text((x, cy + r * 0.6), ch, font=font, fill=red)
        x += d.textlength(ch, font=font) * 1.4
    return np.array(img)


def _component_count(mask: np.ndarray) -> int:
    import cv2

    n, _labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return n - 1


def test_small_digits_survive_in_large_photo():
    """12MP 手机照片里一枚 800px 的章：13 个小数字（每个一两百像素）必须全部保留。

    曾经的阈值按整图面积算（0.05% = 6000px），数字全军覆没——"公章里偶发丢几个小数字"。
    """
    arr = _seal_with_code((3000, 4000), 800, 40)
    rgba = extract_red_seal(arr)
    assert _component_count(rgba[:, :, 3] > 128) == 14  # 外环 + 13 个数字


def test_small_digits_survive_at_low_resolution():
    """小尺寸素材：14px 高的数字笔画只有 1–2px 宽，3×3 开运算会整条抹掉。"""
    arr = _seal_with_code((400, 400), 300, 14)
    rgba = extract_red_seal(arr)
    assert _component_count(rgba[:, :, 3] > 128) >= 14


def test_isolated_specks_far_from_seal_are_dropped():
    """章外的红色污点仍然是噪点：不在墨迹外接范围内就丢，大小无关。"""
    arr = _seal_with_code((2000, 2000), 600, 30)
    arr[100:106, 100:106] = (200, 30, 30)  # 角落一个 6×6 红点
    arr[1900:1903, 300:303] = (200, 30, 30)
    rgba = extract_red_seal(arr)
    # 裁剪后的结果尺寸应只包住印章（外接约 600px + 边距），而不是被角落污点撑到整图
    assert max(rgba.shape[:2]) < 700
    assert _component_count(rgba[:, :, 3] > 128) == 14
