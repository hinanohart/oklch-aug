"""Color-space conversion tests for oklch_aug.color."""

from __future__ import annotations

import numpy as np
import pytest

from oklch_aug.color import (
    bgr_to_oklab,
    oklab_to_bgr,
    oklab_to_rgb,
    rgb_to_oklab,
)


def _seeded_rgb(seed: int, shape: tuple[int, int, int] = (32, 32, 3)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_rgb_oklab_roundtrip_within_uint8_lsb() -> None:
    rgb = _seeded_rgb(seed=0)
    out = oklab_to_rgb(rgb_to_oklab(rgb))
    # sRGB -> linear -> Oklab -> linear -> sRGB -> uint8 is reversible to
    # within the uint8 LSB plus the gamma cbrt floating-point error.
    assert np.max(np.abs(out.astype(np.int16) - rgb.astype(np.int16))) <= 2


def test_bgr_oklab_roundtrip_within_uint8_lsb() -> None:
    bgr = _seeded_rgb(seed=1)
    out = oklab_to_bgr(bgr_to_oklab(bgr))
    assert np.max(np.abs(out.astype(np.int16) - bgr.astype(np.int16))) <= 2


def test_bgr_equals_rgb_with_channel_flip() -> None:
    rgb = _seeded_rgb(seed=2)
    bgr = rgb[..., ::-1].copy()
    np.testing.assert_allclose(rgb_to_oklab(rgb), bgr_to_oklab(bgr), rtol=0, atol=0)


def test_oklab_L_range_for_white_black_grey() -> None:
    # Reference points from Ottosson 2020: pure black is L=0, pure white
    # is L=1, neutral grey is around L=0.7 (gamma curve, not linear).
    white = np.full((1, 1, 3), 255, dtype=np.uint8)
    black = np.zeros((1, 1, 3), dtype=np.uint8)
    grey = np.full((1, 1, 3), 128, dtype=np.uint8)
    assert rgb_to_oklab(white)[0, 0, 0] == pytest.approx(1.0, abs=1e-6)
    assert rgb_to_oklab(black)[0, 0, 0] == pytest.approx(0.0, abs=1e-6)
    grey_L = rgb_to_oklab(grey)[0, 0, 0]
    assert 0.55 < grey_L < 0.8
