"""Oklab metric tests."""

from __future__ import annotations

import numpy as np
import pytest

from oklch_aug.metric import oklab_distance


def test_shape_and_dtype() -> None:
    a = np.random.default_rng(0).standard_normal((5, 3))
    b = np.random.default_rng(1).standard_normal((7, 3))
    d = oklab_distance(a, b)
    assert d.shape == (5, 7)
    assert d.dtype == np.float64


def test_self_distance_is_zero_on_diagonal() -> None:
    a = np.random.default_rng(2).standard_normal((6, 3))
    d = oklab_distance(a, a)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)


def test_symmetric_under_transpose() -> None:
    a = np.random.default_rng(3).standard_normal((4, 3))
    b = np.random.default_rng(4).standard_normal((9, 3))
    d_ab = oklab_distance(a, b)
    d_ba = oklab_distance(b, a)
    np.testing.assert_allclose(d_ab, d_ba.T, atol=1e-12)


def test_matches_naive_loop() -> None:
    rng = np.random.default_rng(5)
    a = rng.standard_normal((3, 3))
    b = rng.standard_normal((4, 3))
    d = oklab_distance(a, b)
    naive = np.array([[np.linalg.norm(a[i] - b[j]) for j in range(4)] for i in range(3)])
    np.testing.assert_allclose(d, naive, atol=1e-12)


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="grid_means"):
        oklab_distance(np.zeros((4, 4)), np.zeros((4, 3)))
    with pytest.raises(ValueError, match="tile_means"):
        oklab_distance(np.zeros((4, 3)), np.zeros((4, 4)))


def test_oklab_distance_torch_optional_import() -> None:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("torch not installed")
    from oklch_aug.metric import oklab_distance_torch

    a = torch.randn(5, 3, requires_grad=True)
    b = torch.randn(7, 3)
    d = oklab_distance_torch(a, b)
    assert tuple(d.shape) == (5, 7)
    d.sum().backward()
    assert a.grad is not None
