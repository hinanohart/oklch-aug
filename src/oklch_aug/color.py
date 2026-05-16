"""Oklab / sRGB conversions (Ottosson 2020), numpy-only.

References
----------
Björn Ottosson, "A perceptual color space for image processing", 2020.
    https://bottosson.github.io/posts/oklab/

Ported from `hinanohart/mosaicraft/src/mosaicraft/color.py` (commit
2918137, lines 48-132), with cv2 dependency removed and BGR-only
helpers generalised to also accept RGB.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "bgr_to_oklab",
    "oklab_to_bgr",
    "oklab_to_rgb",
    "rgb_to_oklab",
]


# ---------------------------------------------------------------------------
# sRGB <-> Linear RGB
# ---------------------------------------------------------------------------
def _srgb_to_linear(c: NDArray) -> NDArray:
    """Convert sRGB (0-1) to linear RGB using the standard gamma curve."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: NDArray) -> NDArray:
    """Convert linear RGB (0-1) to sRGB (0-1)."""
    return np.where(
        c <= 0.0031308,
        c * 12.92,
        1.055 * np.power(np.maximum(c, 0), 1.0 / 2.4) - 0.055,
    )


# ---------------------------------------------------------------------------
# Oklab conversions (Björn Ottosson)
# ---------------------------------------------------------------------------
def _linear_rgb_to_oklab(linear: NDArray) -> NDArray:
    """Linear RGB float64 in [0, 1], shape (..., 3) -> Oklab float64."""
    long = (
        0.4122214708 * linear[..., 0]
        + 0.5363325363 * linear[..., 1]
        + 0.0514459929 * linear[..., 2]
    )
    medium = (
        0.2119034982 * linear[..., 0]
        + 0.6806995451 * linear[..., 1]
        + 0.1073969566 * linear[..., 2]
    )
    short = (
        0.0883024619 * linear[..., 0]
        + 0.2817188376 * linear[..., 1]
        + 0.6299787005 * linear[..., 2]
    )
    long_ = np.cbrt(np.maximum(long, 0))
    medium_ = np.cbrt(np.maximum(medium, 0))
    short_ = np.cbrt(np.maximum(short, 0))
    lightness = 0.2104542553 * long_ + 0.7936177850 * medium_ - 0.0040720468 * short_
    a = 1.9779984951 * long_ - 2.4285922050 * medium_ + 0.4505937099 * short_
    b = 0.0259040371 * long_ + 0.7827717662 * medium_ - 0.8086757660 * short_
    return np.stack([lightness, a, b], axis=-1)


def _oklab_to_linear_rgb(oklab: NDArray) -> NDArray:
    """Oklab float64, shape (..., 3) -> linear RGB float64 (unclamped)."""
    lightness = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]
    long_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    medium_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    short_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    long = long_ * long_ * long_
    medium = medium_ * medium_ * medium_
    short = short_ * short_ * short_
    r = +4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short
    g = -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short
    b_ch = -0.0041960863 * long - 0.7034186147 * medium + 1.7076147010 * short
    return np.stack([r, g, b_ch], axis=-1)


def rgb_to_oklab(rgb_uint8: NDArray) -> NDArray:
    """Convert sRGB uint8 image to Oklab float64.

    Parameters
    ----------
    rgb_uint8 : np.ndarray
        sRGB image with dtype uint8 and shape ``(..., 3)``.

    Returns
    -------
    np.ndarray
        Oklab image with dtype float64 and shape ``(..., 3)``.
        L is in [0, 1]; a/b are roughly in [-0.4, 0.4].
    """
    rgb = np.asarray(rgb_uint8, dtype=np.float64) / 255.0
    return _linear_rgb_to_oklab(_srgb_to_linear(rgb))


def oklab_to_rgb(oklab: NDArray) -> NDArray:
    """Convert Oklab float64 to sRGB uint8 (clipped to display gamut).

    Parameters
    ----------
    oklab : np.ndarray
        Oklab image with shape ``(..., 3)``, float64.

    Returns
    -------
    np.ndarray
        sRGB uint8 image, shape unchanged, clipped to [0, 255].
    """
    rgb_linear = _oklab_to_linear_rgb(oklab)
    rgb = _linear_to_srgb(np.clip(rgb_linear, 0.0, 1.0))
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def bgr_to_oklab(bgr_uint8: NDArray) -> NDArray:
    """Convert OpenCV BGR uint8 image to Oklab float64.

    Channel-flipped wrapper around :func:`rgb_to_oklab` for users coming
    from OpenCV. Same numerical contract as the original
    ``mosaicraft.color.bgr_to_oklab``.
    """
    return rgb_to_oklab(np.asarray(bgr_uint8)[..., ::-1])


def oklab_to_bgr(oklab: NDArray) -> NDArray:
    """Convert Oklab float64 to OpenCV BGR uint8 (clipped to display gamut)."""
    return oklab_to_rgb(oklab)[..., ::-1]
