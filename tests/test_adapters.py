"""Tests for the optional Albumentations / Torch adapters.

Each test class guards its own host-library import via
:func:`pytest.importorskip`, so the file as a whole runs to completion
regardless of which extras are installed.

These cases were extended after the initial PyPI-prep audit (2026-05-16)
to cover input-contract violations that previously failed silently:
grayscale/RGBA/float input, ``Compose(seed=...)`` reproducibility,
``requires_grad=True`` warning, out-of-range tensor values, ``B=0``
fast path, and AlbumentationsX ``always_apply`` deprecation.
"""

from __future__ import annotations

import warnings

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

    # ---- happy path -------------------------------------------------------
    def test_apply_preserves_shape_and_dtype(self) -> None:
        img = _rng_uint8(np.random.default_rng(0))
        out = self.AlbOklch(hue_shift_range=(72.0, 72.0), p=1.0)(image=img)["image"]
        assert out.shape == img.shape
        assert out.dtype == img.dtype

    def test_zero_shift_no_protect_is_near_identity(self) -> None:
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

    # ---- input contract gates --------------------------------------------
    def test_grayscale_rejected(self) -> None:
        img = np.zeros((24, 24), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(H, W, 3\) RGB"):
            self.AlbOklch(hue_shift_range=(72.0, 72.0), p=1.0)(image=img)

    def test_rgba_rejected(self) -> None:
        img = np.zeros((24, 24, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match=r"\(H, W, 3\) RGB"):
            self.AlbOklch(hue_shift_range=(72.0, 72.0), p=1.0)(image=img)

    def test_float_dtype_rejected(self) -> None:
        img = np.zeros((24, 24, 3), dtype=np.float32)
        with pytest.raises(TypeError, match="uint8"):
            self.AlbOklch(hue_shift_range=(72.0, 72.0), p=1.0)(image=img)

    # ---- reproducibility under Compose(seed=...) -------------------------
    def test_compose_seed_reproducible(self) -> None:
        # Albumentations 1.4+ accepts ``A.Compose(..., seed=...)``; older
        # versions accept ``random.seed(...)``. Both are honoured because
        # the adapter draws from ``self.py_random`` / stdlib ``random``,
        # not numpy's global RNG.
        img = _rng_uint8(np.random.default_rng(4))
        import random

        random.seed(7)
        try:
            pipe_a = self.albumentations.Compose(
                [self.AlbOklch(hue_shift_range=(-180.0, 180.0), p=1.0)],
                seed=7,
            )
        except TypeError:
            pipe_a = self.albumentations.Compose(
                [self.AlbOklch(hue_shift_range=(-180.0, 180.0), p=1.0)]
            )
        out_a = pipe_a(image=img)["image"]

        random.seed(7)
        try:
            pipe_b = self.albumentations.Compose(
                [self.AlbOklch(hue_shift_range=(-180.0, 180.0), p=1.0)],
                seed=7,
            )
        except TypeError:
            pipe_b = self.albumentations.Compose(
                [self.AlbOklch(hue_shift_range=(-180.0, 180.0), p=1.0)]
            )
        out_b = pipe_b(image=img)["image"]
        np.testing.assert_array_equal(out_a, out_b)

    # ---- AlbumentationsX deprecation warning -----------------------------
    def test_always_apply_true_deprecation_warning(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.AlbOklch(hue_shift_range=(0.0, 0.0), always_apply=True)
        assert any(
            issubclass(rec.category, DeprecationWarning) and "always_apply" in str(rec.message)
            for rec in w
        )


# ---------------------------------------------------------------------------
# Torch
# ---------------------------------------------------------------------------
class TestTorchAdapter:
    @pytest.fixture(autouse=True)
    def _imports(self) -> None:
        self.torch = pytest.importorskip("torch")
        from oklch_aug.adapters.torch import OklchHueRotation

        self.TorchOklch = OklchHueRotation

    # ---- happy path -------------------------------------------------------
    def test_batched_preserves_shape_and_dtype(self) -> None:
        x = self.torch.rand(2, 3, 16, 16, dtype=self.torch.float32)
        y = self.TorchOklch(hue_shift_deg=72.0)(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype
        assert self.torch.isfinite(y).all()

    def test_single_image_preserves_3d_shape(self) -> None:
        x = self.torch.rand(3, 16, 16, dtype=self.torch.float32)
        y = self.TorchOklch(hue_shift_deg=72.0)(x)
        assert y.shape == x.shape

    def test_zero_shift_no_protect_is_near_identity(self) -> None:
        x = self.torch.rand(1, 3, 16, 16, dtype=self.torch.float32)
        y = self.TorchOklch(hue_shift_deg=0.0, protect_highlights=False, protect_shadows=False)(x)
        diff = (y - x).abs().max().item()
        assert diff < 0.02, f"round-trip should be <2% of dynamic range; got {diff}"

    def test_runtime_override_takes_precedence(self) -> None:
        x = self.torch.rand(1, 3, 16, 16, dtype=self.torch.float32)
        aug = self.TorchOklch(hue_shift_deg=0.0)
        y_default = aug(x)
        y_override = aug(x, hue_shift_deg=180.0)
        assert not self.torch.allclose(y_default, y_override)

    # ---- input contract gates --------------------------------------------
    def test_rejects_wrong_channel_count(self) -> None:
        x = self.torch.rand(1, 4, 16, 16, dtype=self.torch.float32)
        with pytest.raises(ValueError, match="3 channels"):
            self.TorchOklch()(x)

    def test_rejects_wrong_rank(self) -> None:
        x = self.torch.rand(16, 16, dtype=self.torch.float32)
        with pytest.raises(ValueError, match=r"\(C,H,W\)"):
            self.TorchOklch()(x)

    def test_rejects_integer_dtype(self) -> None:
        x = self.torch.randint(0, 255, (1, 3, 16, 16), dtype=self.torch.uint8)
        with pytest.raises(TypeError, match="floating tensor"):
            self.TorchOklch()(x)

    def test_rejects_out_of_range_values(self) -> None:
        x = self.torch.full((1, 3, 8, 8), 1.5, dtype=self.torch.float32)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            self.TorchOklch()(x)
        x_neg = self.torch.full((1, 3, 8, 8), -0.1, dtype=self.torch.float32)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            self.TorchOklch()(x_neg)

    def test_zero_batch_fast_path(self) -> None:
        x = self.torch.empty(0, 3, 8, 8, dtype=self.torch.float32)
        y = self.TorchOklch(hue_shift_deg=72.0)(x)
        assert y.shape == x.shape
        assert y.dtype == x.dtype

    def test_requires_grad_warns(self) -> None:
        x = self.torch.rand(1, 3, 8, 8, dtype=self.torch.float32, requires_grad=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = self.TorchOklch(hue_shift_deg=72.0)(x)
        assert any(
            issubclass(rec.category, UserWarning) and "non-differentiable" in str(rec.message)
            for rec in w
        )
        assert not y.requires_grad
