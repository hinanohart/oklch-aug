"""Smoke tests for the optional Albumentations / Kornia adapters.

Each test class guards its own host-library import via
:func:`pytest.importorskip`, so the file as a whole runs to completion
regardless of which extras are installed.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rng_uint8(rng: np.random.Generator, h: int = 24, w: int = 24) -> np.ndarray:
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Albumentations
# ---------------------------------------------------------------------------
class TestAlbumentationsAdapter:
    @pytest.fixture(autouse=True)
    def _imports(self) -> None:
        self.albumentations = pytest.importorskip("albumentations")
        from oklch_aug.adapters.albumentations import OklchHueRotation

        self.AlbOklch = OklchHueRotation

    def test_apply_preserves_shape_and_dtype(self) -> None:
        img = _rng_uint8(np.random.default_rng(0))
        out = self.AlbOklch(hue_shift_range=(72.0, 72.0), p=1.0)(image=img)["image"]
        assert out.shape == img.shape
        assert out.dtype == img.dtype

    def test_zero_shift_no_protect_is_near_identity(self) -> None:
        # 0° rotation with both protect_* off should be a pure Oklab
        # round-trip — only uint8 quantisation loss.
        img = _rng_uint8(np.random.default_rng(1))
        out = self.AlbOklch(
            hue_shift_range=(0.0, 0.0),
            protect_highlights=False,
            protect_shadows=False,
            p=1.0,
        )(image=img)["image"]
        diff = np.abs(out.astype(np.int16) - img.astype(np.int16))
        assert int(diff.max()) <= 2, f"round-trip should be ≤2 LSB; got {diff.max()}"

    def test_p_zero_skips_transform(self) -> None:
        img = _rng_uint8(np.random.default_rng(2))
        out = self.AlbOklch(hue_shift_range=(72.0, 72.0), p=0.0)(image=img)["image"]
        np.testing.assert_array_equal(out, img)

    def test_compose_chains_cleanly(self) -> None:
        img = _rng_uint8(np.random.default_rng(3))
        pipe = self.albumentations.Compose(
            [
                self.AlbOklch(hue_shift_range=(72.0, 72.0), p=1.0),
                self.albumentations.HorizontalFlip(p=1.0),
            ]
        )
        out = pipe(image=img)["image"]
        assert out.shape == img.shape


# ---------------------------------------------------------------------------
# Kornia / torch
# ---------------------------------------------------------------------------
class TestKorniaAdapter:
    @pytest.fixture(autouse=True)
    def _imports(self) -> None:
        self.torch = pytest.importorskip("torch")
        from oklch_aug.adapters.kornia import OklchHueRotation

        self.KorOklch = OklchHueRotation

    def test_batched_preserves_shape_and_dtype(self) -> None:
        x = self.torch.rand(2, 3, 16, 16, dtype=self.torch.float32)
        y = self.KorOklch(hue_shift_deg=72.0)(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert self.torch.isfinite(y).all()

    def test_single_image_preserves_3d_shape(self) -> None:
        x = self.torch.rand(3, 16, 16, dtype=self.torch.float32)
        y = self.KorOklch(hue_shift_deg=72.0)(x)
        assert y.shape == x.shape

    def test_zero_shift_no_protect_is_near_identity(self) -> None:
        x = self.torch.rand(1, 3, 16, 16, dtype=self.torch.float32)
        y = self.KorOklch(hue_shift_deg=0.0, protect_highlights=False, protect_shadows=False)(x)
        diff = (y - x).abs().max().item()
        # uint8 quantisation alone is ~1/255 ≈ 0.004.
        assert diff < 0.02, f"round-trip should be <2% of dynamic range; got {diff}"

    def test_runtime_override_takes_precedence(self) -> None:
        x = self.torch.rand(1, 3, 16, 16, dtype=self.torch.float32)
        aug = self.KorOklch(hue_shift_deg=0.0)
        y_default = aug(x)
        y_override = aug(x, hue_shift_deg=180.0)
        assert not self.torch.allclose(y_default, y_override)

    def test_rejects_wrong_channel_count(self) -> None:
        x = self.torch.rand(1, 4, 16, 16, dtype=self.torch.float32)
        with pytest.raises(ValueError, match="3 channels"):
            self.KorOklch()(x)

    def test_rejects_wrong_rank(self) -> None:
        x = self.torch.rand(16, 16, dtype=self.torch.float32)
        with pytest.raises(ValueError, match=r"\(C,H,W\)"):
            self.KorOklch()(x)
