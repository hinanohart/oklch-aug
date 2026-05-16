"""Kornia / torch adapter for Oklch hue rotation.

Implementation detail: the canonical :func:`oklch_aug.rotate_hue_oklch`
runs on numpy uint8. The torch wrapper therefore round-trips through
CPU numpy per call. This is faster than re-deriving Oklab in
``torch.float32`` for small batches; for large GPU pipelines, a future
release will port the conversion into a torch-native autograd path.

The numpy-touching logic lives in :func:`_rotate_tensor` at module
scope so the :class:`nn.Module` itself stays free of numpy calls
(ONNX exporters and the trailofbits semgrep rule prefer it that way).

Example
-------
>>> import torch
>>> from oklch_aug.adapters.kornia import OklchHueRotation
>>> aug = OklchHueRotation(hue_shift_deg=72.0)
>>> x = torch.rand(4, 3, 64, 64)         # B, C, H, W in [0, 1]
>>> y = aug(x)                            # same shape, same dtype
"""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "OklchHueRotation requires `torch`. Install via `pip install oklch-aug[kornia]`."
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
    """Round-trip a ``(B, 3, H, W)`` float[0,1] tensor through Oklch rotation.

    Module-level so the :class:`nn.Module` subclass stays numpy-free.
    """
    # local import keeps `import oklch_aug.adapters.kornia` light
    import numpy as np

    device, dtype = x.device, x.dtype
    arr = x.detach().cpu().clamp(0.0, 1.0).mul(255.0).round().byte().permute(0, 2, 3, 1)
    np_arr = arr.numpy()
    rotated = np.stack(
        [
            rotate_hue_oklch(
                np_arr[i],
                hue_shift_deg=hue_shift_deg,
                chroma_scale=chroma_scale,
                protect_highlights=protect_highlights,
                protect_shadows=protect_shadows,
                channel_order="rgb",
            )
            for i in range(np_arr.shape[0])
        ],
        axis=0,
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
    ``Tensor`` of shape ``(B, 3, H, W)`` or ``(3, H, W)`` in ``[0, 1]``,
    RGB channel order. Output matches input shape and dtype. The
    operation is **non-differentiable** (numpy round-trip).
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
