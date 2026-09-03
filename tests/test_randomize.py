"""随机效果引擎测试：种子可复现、幅度有界、蒙尘只减不增。"""

import numpy as np

from core.randomize import (
    CAP_ANGLE_DEG,
    AppliedRandom,
    Randomizer,
    RandomSpec,
)


def _sample_rgba(w=200, h=200) -> np.ndarray:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = 200  # 红色块
    rgba[:, :, 3] = 255
    return rgba


def test_same_seed_reproducible(seal_png):
    from core.extract import extract_red_seal

    ink = extract_red_seal(seal_png)
    spec = RandomSpec()
    out1, _ = Randomizer(seed=42).apply_auto(ink, spec)
    out2, _ = Randomizer(seed=42).apply_auto(ink, spec)
    assert out1.shape == out2.shape
    assert np.array_equal(out1, out2)


def test_different_seed_differs(seal_png):
    from core.extract import extract_red_seal

    ink = extract_red_seal(seal_png)
    spec = RandomSpec()
    out1, _ = Randomizer(seed=1).apply_auto(ink, spec)
    out2, _ = Randomizer(seed=2).apply_auto(ink, spec)
    assert out1.shape != out2.shape or not np.array_equal(out1, out2)


def test_angle_hard_cap():
    spec = RandomSpec(angle_deg=99.0).clamped()
    assert spec.angle_deg == CAP_ANGLE_DEG
    rng = Randomizer(seed=7)
    for _ in range(200):
        applied = rng.sample(spec)
        assert abs(applied.angle_deg) <= CAP_ANGLE_DEG


def test_tone_and_dust_caps():
    spec = RandomSpec(tone=9.9, dust=9.9).clamped()
    assert spec.tone == 0.25 and spec.dust == 0.5
    rng = Randomizer(seed=3)
    for _ in range(200):
        a = rng.sample(spec)
        assert 0.75 <= a.tone_scale <= 1.25
        assert 0.0 <= a.dust <= 0.5


def test_dust_only_reduces_alpha():
    ink = _sample_rgba()
    applied = AppliedRandom(angle_deg=0.0, tone_scale=1.0, dust=0.5, dust_seed=123)
    out = Randomizer.apply(ink, applied)
    assert np.all(out[:, :, 3] <= ink[:, :, 3])
    # 但蒙尘确实造成了削减
    assert out[:, :, 3].mean() < ink[:, :, 3].mean()


def test_zero_spec_is_identity():
    ink = _sample_rgba()
    spec = RandomSpec(angle_deg=0, tone=0, dust=0)
    out, applied = Randomizer(seed=5).apply_auto(ink, spec)
    assert applied.angle_deg == 0 and applied.tone_scale == 1.0 and applied.dust == 0
    assert np.array_equal(out, ink)


def test_replay_from_dict(seal_png):
    from core.extract import extract_red_seal

    ink = extract_red_seal(seal_png)
    out1, applied = Randomizer(seed=99).apply_auto(ink, RandomSpec())
    # 从 .sealog 字典重放，不依赖 Randomizer 流状态
    replayed = Randomizer.apply(ink, AppliedRandom.from_dict(applied.to_dict()))
    assert replayed.shape == out1.shape
    assert np.array_equal(replayed, out1)
