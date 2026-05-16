"""Albumentations adapter for Oklch hue rotation.

Use this when you want Oklch hue rotation as a drop-in inside an
Albumentations / AlbumentationsX ``Compose`` pipeline. Albumentations
expects numpy ``uint8`` HWC RGB images, which is exactly what
:func:`oklch_aug.rotate_hue_oklch` consumes, so the wrapper is a thin
shim around angle sampling.

Example
-------
>>> import albumentations as A
>>> from oklch_aug.adapters.albumentations import OklchHueRotation
>>> pipeline = A.Compose([
...     OklchHueRotation(hue_shift_range=(-180, 180), chroma_scale=1.0, p=0.5),
...     A.HorizontalFlip(p=0.5),
... ])
>>> out = pipeline(image=img)["image"]

Notes
-----
The adapter targets both legacy ``albumentations`` and the
actively-maintained ``AlbumentationsX`` fork; both expose
``ImageOnlyTransform`` under the same import path. To stay
reproducible under ``A.Compose(seed=...)`` the angle is drawn from
``self.py_random`` (AlbumentationsX) or the legacy ``random`` module
(older releases), not from numpy's global RNG.

Input contract
--------------
* dtype: ``np.uint8``
* layout: ``(H, W, 3)`` RGB
* anything else (grayscale, RGBA, ``float32``, ``float64``) raises
  ``TypeError`` / ``ValueError`` — we do **not** silently coerce.
"""

from __future__ import annotations

import random
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from albumentations.core.transforms_interface import ImageOnlyTransform
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "OklchHueRotation requires `albumentations` (or `AlbumentationsX`). "
        "Install via `pip install oklch-aug[albumentations]`."
    ) from exc

from ..rotate import rotate_hue_oklch

__all__ = ["OklchHueRotation"]


class OklchHueRotation(ImageOnlyTransform):
    """Albumentations transform: rotate hue in Oklch, preserving L.

    Parameters
    ----------
    hue_shift_range : tuple of float, default ``(-180.0, 180.0)``
        Closed interval ``[lo, hi]`` from which the per-call shift is
        sampled uniformly, in degrees. ``lo == hi`` acts as a fixed
        angle.
    chroma_scale : float, default ``1.0``
        Forwarded to :func:`rotate_hue_oklch`.
    protect_highlights : bool, default ``True``
        Forwarded to :func:`rotate_hue_oklch`.
    protect_shadows : bool, default ``True``
        Forwarded to :func:`rotate_hue_oklch`.
    p : float, default ``0.5``
        Standard Albumentations probability of applying the transform.

    The transform assumes the image is already in **RGB** channel order
    (the Albumentations convention). For BGR pipelines, call
    :func:`rotate_hue_oklch` directly with ``channel_order="bgr"``.
    """

    def __init__(
        self,
        hue_shift_range: tuple[float, float] = (-180.0, 180.0),
        chroma_scale: float = 1.0,
        protect_highlights: bool = True,
        protect_shadows: bool = True,
        always_apply: bool | None = None,
        p: float = 0.5,
    ) -> None:
        kwargs: dict[str, Any] = {"p": p}
        if always_apply is True:
            warnings.warn(
                "`always_apply=True` is deprecated in AlbumentationsX; pass p=1.0 instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["p"] = 1.0
        elif always_apply is False:
            warnings.warn(
                "`always_apply=False` is deprecated in AlbumentationsX; the value is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__init__(**kwargs)
        lo, hi = float(hue_shift_range[0]), float(hue_shift_range[1])
        if not lo <= hi:
            raise ValueError(f"hue_shift_range must be ordered; got {hue_shift_range!r}")
        self.hue_shift_range = (lo, hi)
        self.chroma_scale = float(chroma_scale)
        self.protect_highlights = bool(protect_highlights)
        self.protect_shadows = bool(protect_shadows)

    def get_params(self) -> dict[str, float]:
        lo, hi = self.hue_shift_range
        # Honour A.Compose(seed=...) by sourcing the angle from the
        # transform-owned RNG when available (AlbumentationsX). Fall
        # back to the stdlib `random` module otherwise; both observe
        # Albumentations' global seed propagation (`A.set_seed(...)` /
        # `random.seed(...)`), unlike `np.random.uniform`.
        py_rand = getattr(self, "py_random", None)
        if py_rand is not None:
            return {"hue_shift_deg": float(py_rand.uniform(lo, hi))}
        return {"hue_shift_deg": float(random.uniform(lo, hi))}

    def apply(self, img: NDArray, *, hue_shift_deg: float = 0.0, **_: Any) -> NDArray:
        if not isinstance(img, np.ndarray):
            raise TypeError(f"OklchHueRotation expects a numpy array; got {type(img).__name__}")
        if img.dtype != np.uint8:
            raise TypeError(
                f"OklchHueRotation expects dtype=uint8 RGB; got dtype={img.dtype}."
                " Convert to uint8 before this transform (e.g. `(img*255).astype(np.uint8)`)."
            )
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(
                f"OklchHueRotation expects a (H, W, 3) RGB image; got shape {img.shape}."
                " Convert grayscale to RGB and drop the alpha channel before this transform."
            )
        return rotate_hue_oklch(
            img,
            hue_shift_deg=hue_shift_deg,
            chroma_scale=self.chroma_scale,
            protect_highlights=self.protect_highlights,
            protect_shadows=self.protect_shadows,
            channel_order="rgb",
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return (
            "hue_shift_range",
            "chroma_scale",
            "protect_highlights",
            "protect_shadows",
        )
