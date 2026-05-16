"""oklch-aug: perceptually-uniform Oklch hue-rotation pool augmentation.

L-preserving by construction (Ottosson 2020). numpy-core, torch optional.
"""

from __future__ import annotations

from .color import (
    bgr_to_oklab,
    oklab_to_bgr,
    oklab_to_rgb,
    rgb_to_oklab,
)
from .metric import oklab_distance
from .pool import DEFAULT_HUE_SCHEDULE, HueRotatePool
from .rotate import rotate_hue_oklch

__version__ = "0.1.0a1"

__all__ = [
    "DEFAULT_HUE_SCHEDULE",
    "HueRotatePool",
    "__version__",
    "bgr_to_oklab",
    "oklab_distance",
    "oklab_to_bgr",
    "oklab_to_rgb",
    "rgb_to_oklab",
    "rotate_hue_oklch",
]
