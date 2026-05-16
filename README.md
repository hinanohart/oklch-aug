# oklch-aug

> Perceptually-uniform Oklch hue-rotation pool augmentation.
> L-preserving by construction. numpy-core, torch optional.

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
- Optional adapters for AlbumentationsX and kornia.

## Install

```bash
pip install oklch-aug                 # numpy only
pip install "oklch-aug[torch]"        # + torch metric
pip install "oklch-aug[albumentations]"
pip install "oklch-aug[kornia]"       # + torch + kornia
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

### Torch / Kornia

```python
import torch
from oklch_aug.adapters.kornia import OklchHueRotation

aug = OklchHueRotation(hue_shift_deg=72.0)
x = torch.rand(4, 3, 64, 64)            # B, C, H, W in [0, 1] RGB
y = aug(x)                                # same shape / dtype
# Round-trips through CPU numpy; non-differentiable.
```

## Provenance

Extracted from
[hinanohart/mosaicraft](https://github.com/hinanohart/mosaicraft)
(`color_augment.py`, `color.py`) where the technique was first used
to expand photomosaic tile pools fed to a Hungarian assignment.
Verified absent (as of 2026-05-16) from albumentations, AlbumentationsX,
kornia, torchvision, and DALI.

## License

[MIT](LICENSE). Matches `mosaicraft` upstream.
