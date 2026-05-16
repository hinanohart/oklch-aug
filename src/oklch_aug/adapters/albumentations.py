"""Albumentations adapter for Oklch hue rotation.

Use this when you want Oklch hue rotation as a drop-in inside an
Albumentations / AlbumentationsX ``Compose`` pipeline. Albumentations
expects numpy uint8 HWC RGB images, which is exactly what
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
The adapter targets both ``albumentations`` (unmaintained as of
2025-06) and the actively-maintained ``AlbumentationsX`` fork. Both
expose ``ImageOnlyTransform`` under the same import path.
"""

from __future__ import annotations

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
        Half-open range from which the per-call shift is sampled
        uniformly, in degrees.
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
        # Newer Albumentations dropped ``always_apply``; pass through only
        # when the user explicitly supplied it.
        kwargs: dict[str, Any] = {"p": p}
        if always_apply is not None:
            kwargs["always_apply"] = always_apply
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
        return {"hue_shift_deg": float(np.random.uniform(lo, hi))}

    def apply(self, img: NDArray, *, hue_shift_deg: float = 0.0, **_: Any) -> NDArray:
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
