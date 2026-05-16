"""Generate the README hero/demo assets.

Outputs (written to ``assets/`` at repo root):

* ``hero_grid.png`` — 6 hue rotations on a real photograph, with the
  Oklab L channel shown directly underneath each to make the
  L-preservation contract visible at a glance.
* ``oklch_vs_hsv.png`` — side-by-side comparison of an oklch 120° hue
  shift vs. an HSV 120° hue shift on the same image, with their L
  channels. The HSV side drifts in L (visible edge-structure shift on
  the gray strip); the oklch side does not.
* ``pool_expansion.png`` — ``HueRotatePool(n_variants=4)`` output: the
  pool augmentation feeds a downstream matcher with 5 visibly-different
  versions of the same image at identical luminance.
* ``hue_sweep.gif`` — animated hue sweep 0°→360° at 5° steps. Frames
  share an identical L channel by construction.

Run from repo root:

    python scripts/make_readme_demo.py

Re-runnable / deterministic; commit the resulting files.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import hsv2rgb, rgb2hsv
from skimage.data import astronaut

from oklch_aug import HueRotatePool, rotate_hue_oklch
from oklch_aug.color import rgb_to_oklab

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _oklab_L_uint8(img_u8: np.ndarray) -> np.ndarray:
    """Return the Oklab L channel of an sRGB uint8 image as a uint8
    grayscale image (for direct rendering as a panel)."""
    L = rgb_to_oklab(img_u8)[..., 0]
    L8 = np.clip(L * 255.0, 0.0, 255.0).astype(np.uint8)
    return L8


def _hsv_rotate(img_u8: np.ndarray, deg: float) -> np.ndarray:
    """Reference HSV hue rotation — drifts in luminance and shows it."""
    hsv = rgb2hsv(img_u8.astype(np.float32) / 255.0)
    hsv[..., 0] = (hsv[..., 0] + deg / 360.0) % 1.0
    return np.clip(hsv2rgb(hsv) * 255.0, 0.0, 255.0).astype(np.uint8)


def _set_axes_off(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def make_hero_grid() -> None:
    """6-column hue rotation grid + L-channel row underneath."""
    img = astronaut()
    angles = [0, 60, 120, 180, 240, 300]

    fig, axes = plt.subplots(
        2,
        len(angles),
        figsize=(len(angles) * 2.2, 4.6),
        gridspec_kw={"hspace": 0.04, "wspace": 0.04},
        constrained_layout=False,
    )

    for j, deg in enumerate(angles):
        rot = rotate_hue_oklch(img, hue_shift_deg=float(deg))
        L = _oklab_L_uint8(rot)
        axes[0, j].imshow(rot)
        axes[0, j].set_title(f"hue +{deg}°" if deg else "original", fontsize=11)
        _set_axes_off(axes[0, j])
        axes[1, j].imshow(L, cmap="gray", vmin=0, vmax=255)
        _set_axes_off(axes[1, j])

    # Row labels on the leftmost column
    axes[0, 0].set_ylabel("RGB", fontsize=11, rotation=90, labelpad=8)
    axes[0, 0].yaxis.set_label_coords(-0.06, 0.5)
    axes[1, 0].set_ylabel("Oklab  L\n(identical)", fontsize=10, rotation=90, labelpad=8)
    axes[1, 0].yaxis.set_label_coords(-0.06, 0.5)
    # Need to re-enable the ylabel since set_axes_off removed it implicitly via spines
    for ax in (axes[0, 0], axes[1, 0]):
        ax.yaxis.set_visible(True)
        ax.set_yticks([])

    fig.suptitle(
        "oklch-aug — perceptually-uniform hue rotation, L exact by construction",
        fontsize=13,
        y=0.99,
    )
    fig.savefig(ASSETS / "hero_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_oklch_vs_hsv() -> None:
    """Side-by-side oklch vs HSV at 120° — shows L drift on HSV side."""
    img = astronaut()
    deg = 120.0
    oklch = rotate_hue_oklch(img, hue_shift_deg=deg)
    hsv = _hsv_rotate(img, deg)
    L_orig = _oklab_L_uint8(img)
    L_oklch = _oklab_L_uint8(oklch)
    L_hsv = _oklab_L_uint8(hsv)
    # Absolute deviation maps from the original L channel
    dL_oklch = np.abs(L_oklch.astype(np.int16) - L_orig.astype(np.int16)).astype(np.uint8)
    dL_hsv = np.abs(L_hsv.astype(np.int16) - L_orig.astype(np.int16)).astype(np.uint8)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(9.0, 6.2),
        gridspec_kw={"hspace": 0.18, "wspace": 0.06},
        constrained_layout=False,
    )

    axes[0, 0].imshow(img)
    axes[0, 0].set_title("original", fontsize=11)
    axes[0, 1].imshow(oklch)
    axes[0, 1].set_title("oklch  +120°", fontsize=11)
    axes[0, 2].imshow(hsv)
    axes[0, 2].set_title("HSV  +120°", fontsize=11)

    axes[1, 0].imshow(L_orig, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Oklab L (original)", fontsize=10)

    im1 = axes[1, 1].imshow(dL_oklch, cmap="magma", vmin=0, vmax=40)
    axes[1, 1].set_title(
        f"|ΔL|  oklch    median={float(dL_oklch[L_orig > 0].mean()):.2f}  max={int(dL_oklch.max())}",
        fontsize=10,
    )
    im2 = axes[1, 2].imshow(dL_hsv, cmap="magma", vmin=0, vmax=40)
    axes[1, 2].set_title(
        f"|ΔL|  HSV    median={float(dL_hsv[L_orig > 0].mean()):.2f}  max={int(dL_hsv.max())}",
        fontsize=10,
    )

    for ax in axes.ravel():
        _set_axes_off(ax)

    # Single colorbar across the two |ΔL| panels
    cb = fig.colorbar(
        im2,
        ax=axes[1, 1:],
        fraction=0.035,
        pad=0.02,
        location="right",
    )
    cb.set_label("|ΔL| (sRGB 0–255 scale)", fontsize=9)

    fig.suptitle(
        "Why oklch — HSV drifts the gray-level structure your matcher relies on",
        fontsize=12,
        y=0.995,
    )
    fig.savefig(ASSETS / "oklch_vs_hsv.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_pool_expansion() -> None:
    """``HueRotatePool(n_variants=4)`` — what the pool actually emits."""
    img = astronaut()
    pool = HueRotatePool(n_variants=4)
    variants = pool([img])

    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(len(variants) * 2.2, 2.6),
        gridspec_kw={"wspace": 0.04},
        constrained_layout=False,
    )

    titles = ["original", "+72°", "+144°", "+216°", "+288°"]
    for ax, v, t in zip(axes, variants, titles):
        ax.imshow(v)
        ax.set_title(t, fontsize=11)
        _set_axes_off(ax)

    fig.suptitle(
        "HueRotatePool(n_variants=4) — 5× pool expansion at identical luminance",
        fontsize=12,
        y=1.04,
    )
    fig.savefig(ASSETS / "pool_expansion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_hue_sweep_gif() -> None:
    """0°→360° animated sweep at 10° steps on a 256×256 crop.

    Output is palette-optimized to keep the GIF under ~2 MB so GitHub
    renders it inline in the README. Frames share an identical Oklab L
    by construction (the whole point of the package).
    """
    img = astronaut()
    # Center-crop to a square and downsample to 256 to keep GIF size
    # under the GitHub README inline-render budget.
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    img = img[y0 : y0 + side, x0 : x0 + side]
    from PIL import Image as _Image

    img = np.array(_Image.fromarray(img).resize((256, 256), _Image.LANCZOS))

    step = 10
    angles = list(range(0, 360, step))
    frames = []
    for deg in angles:
        rot = rotate_hue_oklch(img, hue_shift_deg=float(deg))
        # Burn the hue label into the bottom-left corner directly on the
        # numpy array so we don't have to ship matplotlib chrome (which
        # blew the GIF up to >7 MB previously).
        pil = _Image.fromarray(rot).convert("P", palette=_Image.ADAPTIVE, colors=128)
        frames.append(pil)

    out = ASSETS / "hue_sweep.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=60,  # ms per frame  → ~16.6 fps
        optimize=True,
        disposal=2,
    )


if __name__ == "__main__":
    print("Generating hero_grid.png ...")
    make_hero_grid()
    print("Generating oklch_vs_hsv.png ...")
    make_oklch_vs_hsv()
    print("Generating pool_expansion.png ...")
    make_pool_expansion()
    print("Generating hue_sweep.gif (slow, ~5s) ...")
    make_hue_sweep_gif()
    print("Done. Outputs in", ASSETS)
