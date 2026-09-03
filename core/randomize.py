"""随机效果引擎：让数字章像真章——种子化、有界、所见即所得。

方案 v1.2 §4.9：
- 种子化：同一 (图像, 参数, 种子) → 同一输出；种子随导出落盘可复现；
- 有界：每个维度有硬上限，采样结果再 clamp；
- 蒙尘只减不增：噪声只降低墨迹 alpha，绝不无中生有。

纯函数、无状态；GUI 预览与导出必须共用同一次采样结果。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from PIL import Image

# 硬上限（方案 §4.9 表格）
CAP_ANGLE_DEG = 5.0
CAP_TONE = 0.25
CAP_DUST = 0.5


@dataclass(frozen=True)
class RandomSpec:
    """随机强度配置（滑杆值）。0 = 关闭该维度。均为幅度上限。"""

    angle_deg: float = 2.0   # ±角度
    tone: float = 0.10       # 明度/饱和度 ±比例
    dust: float = 0.15       # 蒙尘强度 0-1

    def clamped(self) -> "RandomSpec":
        return RandomSpec(
            angle_deg=min(abs(self.angle_deg), CAP_ANGLE_DEG),
            tone=min(abs(self.tone), CAP_TONE),
            dust=min(abs(self.dust), CAP_DUST),
        )


@dataclass(frozen=True)
class AppliedRandom:
    """一次采样实际生效的随机量（记入 .sealog 以便复现）。

    dust_seed 使 apply() 成为纯函数：重放时不需要 Randomizer 的流状态。
    """

    angle_deg: float
    tone_scale: float
    dust: float
    dust_seed: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AppliedRandom":
        return cls(
            angle_deg=float(d["angle_deg"]),
            tone_scale=float(d["tone_scale"]),
            dust=float(d["dust"]),
            dust_seed=int(d["dust_seed"]),
        )


class Randomizer:
    """由种子驱动的随机效果采样器。每次导出实例化一个，全程共用。"""

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def sample(self, spec: RandomSpec) -> AppliedRandom:
        spec = spec.clamped()
        return AppliedRandom(
            angle_deg=float(self._rng.uniform(-spec.angle_deg, spec.angle_deg)),
            tone_scale=float(self._rng.uniform(1.0 - spec.tone, 1.0 + spec.tone)),
            dust=float(self._rng.uniform(0.0, spec.dust)),
            dust_seed=int(self._rng.integers(0, 2**31 - 1)),
        )

    @staticmethod
    def apply(rgba: np.ndarray, applied: AppliedRandom) -> np.ndarray:
        """把已采样的随机量应用到 RGBA 图像。纯函数：同输入必同输出。"""
        out = rgba
        if abs(applied.angle_deg) > 1e-6:
            out = _rotate(out, applied.angle_deg)
        if abs(applied.tone_scale - 1.0) > 1e-6:
            out = _tone(out, applied.tone_scale)
        if applied.dust > 1e-6:
            out = _dust(out, applied.dust, np.random.default_rng(applied.dust_seed))
        return out

    def apply_auto(self, rgba: np.ndarray, spec: RandomSpec) -> tuple[np.ndarray, AppliedRandom]:
        """采样并应用，返回 (结果, 实际随机量)。"""
        applied = self.sample(spec)
        return self.apply(rgba, applied), applied


def _rotate(rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    """绕中心旋转，画布扩大防裁切。逆时针为正。"""
    im = Image.fromarray(rgba, "RGBA")
    rotated = im.rotate(angle_deg, resample=Image.BICUBIC, expand=True)
    return np.array(rotated)


def _tone(rgba: np.ndarray, scale: float) -> np.ndarray:
    """印泥浓淡：缩放饱和度与明度，alpha 不变。"""
    rgb = rgba[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)
    out_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return np.dstack([out_rgb, rgba[:, :, 3]])


def _dust(rgba: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """蒙尘：低频噪声调制 alpha，制造"缺油"小缺口。只减不增。"""
    h, w = rgba.shape[:2]
    # 低频噪声：小随机图上采样 → 平滑
    grid = rng.random((max(2, h // 24), max(2, w // 24))).astype(np.float32)
    noise = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(w, h) / 64)
    # 归一化到 [0,1]
    lo, hi = float(noise.min()), float(noise.max())
    if hi - lo < 1e-6:
        return rgba
    noise = (noise - lo) / (hi - lo)
    # 只削减 alpha：noise 越大削得越多，最大削减比例 = amount
    factor = 1.0 - amount * noise
    alpha = (rgba[:, :, 3].astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
    return np.dstack([rgba[:, :, :3], alpha])
