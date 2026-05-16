# oklch-aug

> Perceptually-uniform Oklch hue-rotation pool augmentation.
> L-preserving by construction. numpy-core, torch optional.

<p align="center">
  <img src="https://raw.githubusercontent.com/hinanohart/oklch-aug/main/assets/hero_grid.png"
       alt="oklch-aug hue rotation grid: 6 rotations of the same image with their Oklab L channels shown identical underneath"
       width="100%">
</p>

**Top row** — the same photo rotated through six hues with `rotate_hue_oklch`.
**Bottom row** — the Oklab L channel of each rotated image. By construction
they are pixel-for-pixel identical: hue rotation in Oklab fixes L. That is
exactly the invariant a pretrained matcher / retriever / policy needs.

### See the hue sweep

<p align="center">
  <img src="https://raw.githubusercontent.com/hinanohart/oklch-aug/main/assets/hue_sweep.gif"
       alt="animated hue sweep from 0 to 360 degrees with identical Oklab L on every frame"
       width="320">
</p>

Every frame above shares the same Oklab L — only the chroma rotates.

### Why this matters: HSV drifts L, oklch does not

<p align="center">
  <img src="https://raw.githubusercontent.com/hinanohart/oklch-aug/main/assets/oklch_vs_hsv.png"
       alt="HSV 120 degree hue shift produces large luminance deviation while oklch produces near-zero deviation"
       width="100%">
</p>

Same +120° hue shift, two color spaces. The bottom row shows |ΔL| against the
original. HSV's median |ΔL| is in the double digits — the gray-level structure
your matcher learned to rely on has moved. oklch's median is < 1 LSB (the residual
is uint8 round-trip + sRGB gamut clipping, both characterised in `rotate_hue_oklch`'s
docstring).

### Pool augmentation, one call

<p align="center">
  <img src="https://raw.githubusercontent.com/hinanohart/oklch-aug/main/assets/pool_expansion.png"
       alt="HueRotatePool(n_variants=4) emits five evenly-spaced hue rotations of the same image"
       width="100%">
</p>

`HueRotatePool(n_variants=4)` expands each pool image into 5 visibly-different
copies at identical luminance — feeds straight into bipartite-matching /
retrieval pools that want a larger candidate set without the
fidelity-vs-diversity tradeoff of HSV jitter.

> All four figures above are produced by `python scripts/make_readme_demo.py`
> (re-runnable, deterministic, takes seconds).

## Why

Most color augmenters live in HSV/HSL, which is **not** perceptually
uniform: a 30° hue shift through orange looks small, the same shift
through cyan looks large, and on top of that the *lightness* coupled
into V (or L in HSL) drifts as you rotate. That drift is exactly what
trashes edge structure in a downstream policy / matcher / retriever.

Oklch hue rotation (Björn Ottosson, 2020) is different: rotating
chroma at fixed Oklab L is **mathematically** L-preserving — every
pixel keeps its luminance, only its color shifts. Combined with
optional highlight/shadow chroma protection, the result is an
augmentation that produces visibly different colors **without** moving
the gray-level structure a pretrained policy or matcher relies on.

## What

- `rotate_hue_oklch(image, angle_deg, ...)` — single image, L-exact rotation.
- `HueRotatePool(n_variants=4)` — expand an `N`-image pool into
  `N · (1 + n_variants)` images at evenly-spaced hue angles (default:
  72°, 144°, 216°, 288°). Originals come first. Designed for
  bipartite-matching / retrieval pools that benefit from a larger
  candidate set without the usual fidelity-vs-diversity tradeoff.
- `bgr_to_oklab`, `rgb_to_oklab`, and their inverses — numpy-only,
  cv2-free.
- `oklab_distance(a, b)` — pairwise Euclidean perceptual distance.
- Optional `oklab_distance_torch` for differentiable use.
- Optional adapters for AlbumentationsX and torch.

## Install

```bash
pip install oklch-aug                 # numpy only
pip install "oklch-aug[torch]"        # + torch metric and Torch adapter
pip install "oklch-aug[albumentations]"
```

## Quick start

```python
import numpy as np
from oklch_aug import HueRotatePool, rotate_hue_oklch

rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
rotated = rotate_hue_oklch(rgb, hue_shift_deg=120.0)
# rotated.shape == rgb.shape, dtype == uint8, L (Oklab) is unchanged.

pool = HueRotatePool(n_variants=4)
expanded = pool([rgb for _ in range(8)])
# len(expanded) == 8 * 5 == 40
```

## Adapters

### Albumentations / AlbumentationsX

```python
import albumentations as A
from oklch_aug.adapters.albumentations import OklchHueRotation

pipeline = A.Compose([
    OklchHueRotation(hue_shift_range=(-180, 180), chroma_scale=1.0, p=0.5),
    A.HorizontalFlip(p=0.5),
])
out = pipeline(image=img)["image"]
```

### Torch

```python
import torch
from oklch_aug.adapters.torch import OklchHueRotation

aug = OklchHueRotation(hue_shift_deg=72.0)
x = torch.rand(4, 3, 64, 64)            # B, C, H, W float32 in [0, 1] RGB
y = aug(x)                                # same shape / dtype
# Round-trips through CPU numpy; non-differentiable (warns on
# requires_grad=True; rejects integer dtypes / out-of-range values).
```

This is a plain `nn.Module`, not a `kornia.augmentation.AugmentationBase2D`
subclass — use it inside torch / torchvision / kornia pipelines as a
fixed-policy transform, but expect no autograd, no per-batch parameter
generation, and no `same_on_batch`-style coupling.

## Provenance

Extracted from
[hinanohart/mosaicraft](https://github.com/hinanohart/mosaicraft)
(`color_augment.py`, `color.py`) where the technique was first used
to expand photomosaic tile pools fed to a Hungarian assignment.
Verified absent (as of 2026-05-16) from albumentations, AlbumentationsX,
kornia, torchvision, and DALI. The torch adapter exposes a plain
`nn.Module`; no `kornia.augmentation.AugmentationBase2D` subclass is
shipped (deliberately — see the Torch section above).

## License

[MIT](LICENSE). Matches `mosaicraft` upstream.
