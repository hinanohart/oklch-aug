"""Torch adapter for Oklch hue rotation.

A thin :class:`torch.nn.Module` that round-trips a tensor batch
through CPU numpy so the canonical numpy implementation of
:func:`oklch_aug.rotate_hue_oklch` can be reused. Useful as a
drop-in inside torch / kornia / torchvision pipelines, but **note**:

* the operation is **non-differentiable** (we ``.detach()`` and round
  through ``uint8``);
* it expects a floating tensor in ``[0, 1]`` with channel-first layout
  ``(B, 3, H, W)`` or ``(3, H, W)``;
* anything else (integer dtypes, ``requires_grad``, values outside
  ``[0, 1]``, alpha channels) raises or warns rather than silently
  corrupting.

This is a plain ``nn.Module``; it does **not** subclass
``kornia.augmentation.AugmentationBase2D``. If you need true kornia
pipeline compatibility (``AugmentationSequential``,
``same_on_batch``, parameter generation), wrap this with your own
kornia adapter.

Example
-------
>>> import torch
>>> from oklch_aug.adapters.torch import OklchHueRotation
>>> aug = OklchHueRotation(hue_shift_deg=72.0)
>>> x = torch.rand(4, 3, 64, 64)         # B, C, H, W in [0, 1]
>>> y = aug(x)                            # same shape, same dtype
"""

from __future__ import annotations

import warnings

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "OklchHueRotation requires `torch`. Install via `pip install oklch-aug[torch]`."
    ) from exc

from ..rotate import rotate_hue_oklch

__all__ = ["OklchHueRotation"]


def _rotate_tensor(
    x: Tensor,
    *,
    hue_shift_deg: float,
    chroma_scale: float,
    protect_highlights: bool,
    protect_shadows: bool,
) -> Tensor:
    """Round-trip a ``(B, 3, H, W)`` float[0,1] tensor through Oklch rotation."""

    if x.shape[0] == 0:
        return x.clone()

    device, dtype = x.device, x.dtype
    arr = x.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().byte().permute(0, 2, 3, 1)
    # rotate_hue_oklch broadcasts over leading axes, so we hand it the
    # full (B, H, W, 3) buffer in one call instead of a Python for-loop.
    rotated = rotate_hue_oklch(
        arr.numpy(),
        hue_shift_deg=hue_shift_deg,
        chroma_scale=chroma_scale,
        protect_highlights=protect_highlights,
        protect_shadows=protect_shadows,
        channel_order="rgb",
    )
    rotated_t = torch.from_numpy(rotated).to(device=device, dtype=torch.float32)
    return rotated_t.permute(0, 3, 1, 2).div_(255.0).to(dtype=dtype)


class OklchHueRotation(nn.Module):
    """``nn.Module`` wrapping :func:`oklch_aug.rotate_hue_oklch`.

    Parameters
    ----------
    hue_shift_deg : float, default ``0.0``
        Fixed rotation applied to every sample in the batch. For random
        sampling, draw a value externally and pass via :meth:`forward`'s
        ``hue_shift_deg`` kwarg.
    chroma_scale : float, default ``1.0``
    protect_highlights : bool, default ``True``
    protect_shadows : bool, default ``True``

    Expected input
    --------------
    Floating ``Tensor`` of shape ``(B, 3, H, W)`` or ``(3, H, W)`` with
    values in ``[0, 1]``, RGB channel order, dtype ``float32`` or
    ``float64``. Output matches input shape and dtype. The operation
    is **non-differentiable** (numpy round-trip).

    Numerical contract:

    * Internally the tensor is rounded to ``uint8`` for the canonical
      numpy rotation, so the absolute round-trip error is bounded by
      ``~1/255`` (~0.004 of the [0, 1] unit range) for the L\\*
      coordinate even at zero rotation. Tail error can reach ~0.05 in
      saturated regions where the rotated colour falls outside sRGB
      and gets clipped — see ``paper/figures/fig_L_preservation.pdf``
      in the upstream mosaicraft repo for the empirical distribution.

    Gates (all fail loud, never silent):

    * Integer dtypes raise ``TypeError``.
    * ``float16`` / ``bfloat16`` raise ``TypeError`` (cast to float32).
    * NaN / ±Inf raise ``ValueError``.
    * Values outside ``[0, 1]`` raise ``ValueError``.
    * ``requires_grad=True`` emits a :class:`UserWarning` (gradients
      will not flow through; output's ``requires_grad`` is False).
    """

    def __init__(
        self,
        hue_shift_deg: float = 0.0,
        chroma_scale: float = 1.0,
        protect_highlights: bool = True,
        protect_shadows: bool = True,
    ) -> None:
        super().__init__()
        self.hue_shift_deg = float(hue_shift_deg)
        self.chroma_scale = float(chroma_scale)
        self.protect_highlights = bool(protect_highlights)
        self.protect_shadows = bool(protect_shadows)

    def forward(self, x: Tensor, *, hue_shift_deg: float | None = None) -> Tensor:
        if x.ndim not in (3, 4):
            raise ValueError(f"expected (C,H,W) or (B,C,H,W); got shape {tuple(x.shape)}")
        if x.shape[-3] != 3:
            raise ValueError(f"expected 3 channels (RGB); got shape {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(
                f"OklchHueRotation expects a floating tensor in [0, 1]; got dtype {x.dtype}."
                " Convert with `x.float() / 255.0` first."
            )
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                f"OklchHueRotation expects float32 or float64; got dtype {x.dtype}."
                " float16 / bfloat16 round-trip through uint8 amplifies quantisation"
                " error past 1/255. Cast to float32 first (`x.float()`)."
            )
        if x.numel() > 0:
            if not torch.isfinite(x).all():
                raise ValueError(
                    "OklchHueRotation expects finite values in [0, 1];"
                    " input contains NaN or ±Inf. Sanitise upstream"
                    " (e.g. `torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)`)."
                )
            lo, hi = float(x.min()), float(x.max())
            if lo < 0.0 or hi > 1.0:
                raise ValueError(
                    f"OklchHueRotation expects values in [0, 1]; got [{lo:.4f}, {hi:.4f}]."
                    " Inverse-normalise before the adapter."
                )
        if x.requires_grad:
            warnings.warn(
                "OklchHueRotation is non-differentiable; gradients will not flow through.",
                UserWarning,
                stacklevel=2,
            )

        squeezed = x.ndim == 3
        if squeezed:
            x = x.unsqueeze(0)

        angle = self.hue_shift_deg if hue_shift_deg is None else float(hue_shift_deg)
        out = _rotate_tensor(
            x,
            hue_shift_deg=angle,
            chroma_scale=self.chroma_scale,
            protect_highlights=self.protect_highlights,
            protect_shadows=self.protect_shadows,
        )
        return out.squeeze(0) if squeezed else out
