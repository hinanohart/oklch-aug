"""Single-image Oklch hue rotation, L-preserving by construction.

Ported from `hinanohart/mosaicraft/src/mosaicraft/color_augment.py`
(commit 2918137, lines 76-131) with two generalisations:

* ``channel_order`` selects between BGR (mosaicraft default) and RGB
  (everywhere else); the math is identical, only the channel reorder
  differs.
* No TileSet / feature dependency. Pure image-in image-out.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .color import bgr_to_oklab, oklab_to_bgr, oklab_to_rgb, rgb_to_oklab

__all__ = ["rotate_hue_oklch"]


def rotate_hue_oklch(
    image: NDArray,
    hue_shift_deg: float,
    *,
    chroma_scale: float = 1.0,
    protect_highlights: bool = True,
    protect_shadows: bool = True,
    channel_order: Literal["bgr", "rgb"] = "rgb",
) -> NDArray:
    """Rotate an image's hue in Oklch, preserving Oklab L exactly.

    Parameters
    ----------
    image : np.ndarray
        uint8 image of shape ``(..., 3)``. Interpreted as sRGB-encoded
        either as RGB or BGR depending on ``channel_order``.
    hue_shift_deg : float
        Signed rotation in degrees. Positive shifts go counter-clockwise
        on the Oklab a/b plane; 360 is a no-op.
    chroma_scale : float, default 1.0
        Multiplier applied to chroma after rotation. 1.0 keeps
        saturation unchanged. Values < 1 desaturate; > 1 boost.
    protect_highlights : bool, default True
        If True, fade chroma toward zero as L approaches 1.0 so
        speculars stay clean instead of picking up a colour cast.
    protect_shadows : bool, default True
        If True, fade chroma toward zero as L approaches 0.0 so shadow
        depth is not accidentally colorised.
    channel_order : {"rgb", "bgr"}, default "rgb"
        Whether ``image`` is in RGB order (PIL / albumentations / kornia
        default) or BGR order (OpenCV default).

    Returns
    -------
    np.ndarray
        uint8 image with the same shape and channel order as the input.

    Notes
    -----
    L preservation is exact in the float64 Oklab representation. The
    final uint8 round-trip introduces sub-LSB quantisation only.
    """
    if channel_order == "rgb":
        oklab = rgb_to_oklab(image)
    elif channel_order == "bgr":
        oklab = bgr_to_oklab(image)
    else:
        raise ValueError(f"channel_order must be 'rgb' or 'bgr'; got {channel_order!r}")

    lightness = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]

    chroma = np.sqrt(a * a + b * b)
    hue = np.arctan2(b, a) + np.radians(float(hue_shift_deg))

    new_chroma = chroma * float(chroma_scale)

    if protect_highlights:
        highlight = np.clip((lightness - 0.85) / 0.15, 0.0, 1.0)
        new_chroma = new_chroma * (1.0 - highlight * 0.5)
    if protect_shadows:
        shadow = np.clip((0.25 - lightness) / 0.25, 0.0, 1.0)
        new_chroma = new_chroma * (1.0 - shadow * 0.3)

    new_chroma = np.clip(new_chroma, 0.0, 0.4)

    new_a = new_chroma * np.cos(hue)
    new_b = new_chroma * np.sin(hue)

    rotated = np.stack([lightness, new_a, new_b], axis=-1)
    return oklab_to_rgb(rotated) if channel_order == "rgb" else oklab_to_bgr(rotated)
