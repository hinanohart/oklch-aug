"""Rotation tests — the L-preservation contract is the core claim."""

from __future__ import annotations

import numpy as np
import pytest

from oklch_aug.color import oklab_to_rgb, rgb_to_oklab
from oklch_aug.rotate import rotate_hue_oklch


def _moderate_chroma_image(seed: int, n: int = 32) -> np.ndarray:
    """Random Oklab image with chroma in [0, 0.1] so rotation stays in sRGB gamut.

    Rendering through ``oklab_to_rgb`` materialises the image as uint8
    sRGB; subsequent rotation operates on that uint8, mirroring real
    user input.
    """
    rng = np.random.default_rng(seed)
    L = rng.uniform(0.4, 0.7, size=(n, n))  # mid-tones avoid gamut walls
    theta = rng.uniform(0, 2 * np.pi, size=(n, n))
    c = rng.uniform(0.0, 0.05, size=(n, n))  # low chroma stays in gamut at any hue
    oklab = np.stack([L, c * np.cos(theta), c * np.sin(theta)], axis=-1)
    return oklab_to_rgb(oklab)


@pytest.mark.parametrize("angle", [0.0, 30.0, 72.0, 144.0, 180.0, 216.0, 288.0, 360.0, -45.0])
def test_L_is_preserved_within_uint8_quantisation(angle: float) -> None:
    rgb = _moderate_chroma_image(seed=0)
    L_in = rgb_to_oklab(rgb)[..., 0]
    rotated = rotate_hue_oklch(
        rgb, hue_shift_deg=angle, protect_highlights=False, protect_shadows=False
    )
    L_out = rgb_to_oklab(rotated)[..., 0]
    # L is exact in float64 by construction; the uint8 sRGB round-trip
    # introduces sub-LSB error, empirically < 0.01 across the entire L range.
    assert np.max(np.abs(L_in - L_out)) < 0.01


def test_zero_angle_is_near_identity() -> None:
    rgb = _moderate_chroma_image(seed=3)
    out = rotate_hue_oklch(rgb, hue_shift_deg=0.0, protect_highlights=False, protect_shadows=False)
    # uint8 round-trip introduces sub-LSB error, but no perceptible change.
    assert np.max(np.abs(out.astype(np.int16) - rgb.astype(np.int16))) <= 3


def test_chroma_scale_zero_collapses_to_grey() -> None:
    rgb = _moderate_chroma_image(seed=4)
    out = rotate_hue_oklch(rgb, hue_shift_deg=90.0, chroma_scale=0.0)
    oklab_out = rgb_to_oklab(out)
    # chroma_scale=0 -> a = b = 0 in float64; uint8 round-trip leaves tiny residue.
    assert np.max(np.abs(oklab_out[..., 1])) < 0.005
    assert np.max(np.abs(oklab_out[..., 2])) < 0.005


def test_bgr_channel_order_matches_rgb_flip() -> None:
    rgb = _moderate_chroma_image(seed=5)
    bgr = rgb[..., ::-1].copy()
    out_rgb = rotate_hue_oklch(rgb, hue_shift_deg=120.0, channel_order="rgb")
    out_bgr = rotate_hue_oklch(bgr, hue_shift_deg=120.0, channel_order="bgr")
    np.testing.assert_array_equal(out_rgb, out_bgr[..., ::-1])


def test_invalid_channel_order_rejected() -> None:
    rgb = _moderate_chroma_image(seed=6)
    with pytest.raises(ValueError, match="channel_order"):
        rotate_hue_oklch(rgb, hue_shift_deg=10.0, channel_order="xyz")  # type: ignore[arg-type]
