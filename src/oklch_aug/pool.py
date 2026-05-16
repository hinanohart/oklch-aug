"""Pool expansion via Oklch hue rotation.

Turns an ``N``-image pool into an ``N * (1 + k)``-image pool by
appending evenly-spaced hue-rotated variants of each input. Originals
come first; rotated variants follow in stable order (all images at the
first angle, then all images at the second, ...). The order matters
for bipartite-matching downstream that wants reproducible candidate
indices.

The technique was first used in
`hinanohart/mosaicraft/src/mosaicraft/color_augment.py:146-258`
(``expand_color_variants``) to enlarge the candidate set fed to a
Hungarian assignment in photomosaic construction, without the usual
fidelity-vs-diversity tradeoff of HSV jitter — Oklch hue rotation
keeps Oklab L exact, so every variant retains edge structure and
texture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from numpy.typing import NDArray

from .rotate import rotate_hue_oklch

__all__ = ["DEFAULT_HUE_SCHEDULE", "HueRotatePool"]


# Default hue-rotation schedule, in degrees. Picked so each angle lands
# in a different quadrant of the Oklab a/b plane (warm, green, cool,
# magenta) and the gaps are perceptually balanced.
DEFAULT_HUE_SCHEDULE: tuple[float, ...] = (72.0, 144.0, 216.0, 288.0)


@dataclass(frozen=True)
class HueRotatePool:
    """Stateless, picklable pool expander.

    Parameters
    ----------
    n_variants : int or None, default 4
        Number of hue-rotated copies to add per input image. If
        ``hue_schedule`` is also given, ``n_variants`` is ignored.
        Setting ``n_variants=0`` (or ``None``) returns the originals
        unchanged.
    hue_schedule : tuple of float or None, default None
        Explicit list of rotation angles in degrees. Takes precedence
        over ``n_variants``. When ``None`` and ``n_variants`` differs
        from ``len(DEFAULT_HUE_SCHEDULE)``, angles are spread evenly on
        ``(0, 360)`` skipping 0 so variants are always distinct from
        the originals.
    chroma_scale : float, default 1.0
        Chroma multiplier applied to the rotated variants only.
    protect_highlights : bool, default True
        Forwarded to :func:`rotate_hue_oklch`.
    protect_shadows : bool, default True
        Forwarded to :func:`rotate_hue_oklch`.
    channel_order : {"rgb", "bgr"}, default "rgb"
        Channel order of the input images.
    """

    n_variants: int | None = 4
    hue_schedule: tuple[float, ...] | None = None
    chroma_scale: float = 1.0
    protect_highlights: bool = True
    protect_shadows: bool = True
    channel_order: Literal["rgb", "bgr"] = "rgb"

    def _resolve_schedule(self) -> tuple[float, ...]:
        if self.hue_schedule is not None:
            return tuple(float(a) for a in self.hue_schedule)
        if self.n_variants is None or self.n_variants <= 0:
            return ()
        n = int(self.n_variants)
        if n == len(DEFAULT_HUE_SCHEDULE):
            return DEFAULT_HUE_SCHEDULE
        step = 360.0 / (n + 1)
        return tuple(step * (i + 1) for i in range(n))

    def __call__(self, images: Sequence[NDArray]) -> list[NDArray]:
        """Expand a pool of ``N`` images into ``N * (1 + len(schedule))`` images.

        Order: all originals, then all images rotated by the first angle,
        then by the second, and so on. The relative order within each
        chunk matches the input order.
        """
        schedule = self._resolve_schedule()
        out: list[NDArray] = list(images)
        if not schedule:
            return out
        for angle in schedule:
            for img in images:
                out.append(
                    rotate_hue_oklch(
                        img,
                        hue_shift_deg=angle,
                        chroma_scale=self.chroma_scale,
                        protect_highlights=self.protect_highlights,
                        protect_shadows=self.protect_shadows,
                        channel_order=self.channel_order,
                    )
                )
        return out
