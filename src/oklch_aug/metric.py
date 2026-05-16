"""Perceptual Oklab distance.

The numpy version is always available. A torch version is exposed as
:func:`oklab_distance_torch` when ``torch`` is installed; importing
this module never requires torch.

The numerical contract mirrors
``mosaicraft-active-vision/src/mosaicraft_active_vision/cost.py:129-152``
so the upstream Sinkhorn-OT cost matrix is bit-identical when this lib
is used in its place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["oklab_distance", "oklab_distance_torch"]


def oklab_distance(grid_means: NDArray, tile_means: NDArray) -> NDArray:
    """Pairwise Euclidean distance in Oklab between two sets of points.

    Parameters
    ----------
    grid_means : np.ndarray, shape ``(n, 3)``
        Oklab points, typically per-cell means.
    tile_means : np.ndarray, shape ``(m, 3)``
        Oklab points, typically per-tile means.

    Returns
    -------
    np.ndarray, shape ``(n, m)``, float64
        ``out[i, j] = || grid_means[i] - tile_means[j] ||_2`` in Oklab.
    """
    if grid_means.ndim != 2 or grid_means.shape[1] != 3:
        raise ValueError(f"grid_means must be (n, 3); got {grid_means.shape}")
    if tile_means.ndim != 2 or tile_means.shape[1] != 3:
        raise ValueError(f"tile_means must be (m, 3); got {tile_means.shape}")
    diff = grid_means[:, None, :] - tile_means[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=2)).astype(np.float64)


def oklab_distance_torch(grid_means: Any, tile_means: Any) -> Any:
    """Differentiable Oklab distance (torch).

    Same shape contract as :func:`oklab_distance`, but inputs and outputs
    are ``torch.Tensor`` and the result has a meaningful gradient w.r.t.
    both inputs. Raises ``ImportError`` if torch is not installed.

    Parameters
    ----------
    grid_means : torch.Tensor, shape ``(n, 3)``
    tile_means : torch.Tensor, shape ``(m, 3)``

    Returns
    -------
    torch.Tensor, shape ``(n, m)``
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "oklab_distance_torch requires torch. Install with: pip install 'oklch-aug[torch]'"
        ) from exc

    if grid_means.ndim != 2 or grid_means.shape[1] != 3:
        raise ValueError(f"grid_means must be (n, 3); got {tuple(grid_means.shape)}")
    if tile_means.ndim != 2 or tile_means.shape[1] != 3:
        raise ValueError(f"tile_means must be (m, 3); got {tuple(tile_means.shape)}")
    return torch.cdist(grid_means, tile_means, p=2.0)
