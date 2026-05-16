"""HueRotatePool expansion tests."""

from __future__ import annotations

import numpy as np
import pytest

from oklch_aug.pool import DEFAULT_HUE_SCHEDULE, HueRotatePool
from oklch_aug.rotate import rotate_hue_oklch


def _imgs(n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (16, 16, 3), dtype=np.uint8) for _ in range(n)]


def test_default_expands_5x_with_originals_first() -> None:
    src = _imgs(3)
    out = HueRotatePool()(src)
    assert len(out) == 3 * (1 + 4)
    for i, original in enumerate(src):
        np.testing.assert_array_equal(out[i], original)


def test_n_variants_zero_is_identity() -> None:
    src = _imgs(3)
    out = HueRotatePool(n_variants=0)(src)
    assert len(out) == 3
    for a, b in zip(out, src, strict=True):
        np.testing.assert_array_equal(a, b)


def test_n_variants_none_is_identity() -> None:
    src = _imgs(3)
    out = HueRotatePool(n_variants=None)(src)
    assert len(out) == 3


def test_explicit_schedule_overrides_n_variants() -> None:
    src = _imgs(2)
    out = HueRotatePool(n_variants=10, hue_schedule=(60.0, 180.0))(src)
    assert len(out) == 2 * (1 + 2)


def test_non_default_n_variants_uses_even_spacing() -> None:
    src = _imgs(1)
    # n=3 -> step = 360 / 4 = 90 -> (90, 180, 270)
    pool = HueRotatePool(n_variants=3)
    schedule = pool._resolve_schedule()
    assert schedule == (90.0, 180.0, 270.0)
    out = pool(src)
    assert len(out) == 1 * (1 + 3)


def test_rotated_variant_matches_direct_rotation() -> None:
    src = _imgs(2, seed=42)
    pool = HueRotatePool(n_variants=4)
    out = pool(src)
    # Layout: [src0, src1, rot0_src0, rot0_src1, rot1_src0, rot1_src1, ...]
    for chunk_idx, angle in enumerate(DEFAULT_HUE_SCHEDULE):
        for img_idx, img in enumerate(src):
            expected = rotate_hue_oklch(img, hue_shift_deg=angle)
            actual = out[2 + chunk_idx * 2 + img_idx]
            np.testing.assert_array_equal(actual, expected)


def test_frozen_dataclass() -> None:
    pool = HueRotatePool()
    with pytest.raises((AttributeError, Exception)):
        pool.n_variants = 8  # type: ignore[misc]


def test_bgr_channel_order_propagates() -> None:
    src = _imgs(1, seed=7)
    rgb_pool = HueRotatePool(n_variants=2, channel_order="rgb")
    bgr_pool = HueRotatePool(n_variants=2, channel_order="bgr")
    rgb_out = rgb_pool(src)
    bgr_out = bgr_pool([img[..., ::-1].copy() for img in src])
    for r, b in zip(rgb_out, bgr_out, strict=True):
        np.testing.assert_array_equal(r, b[..., ::-1])
