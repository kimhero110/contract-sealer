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
