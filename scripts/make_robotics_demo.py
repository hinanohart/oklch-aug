"""Generate the robotics use-case demo assets.

oklch-aug was extracted from an active-vision research codebase
(`mosaicraft-active-vision`), so the README needs a section that
shows the robotics-flavoured use case directly. This script renders:

* ``assets/robot_dr_batch.png`` — a *real* table-top photo
  (skimage.data.coffee()) treated as a stand-in for a manipulation
  policy's wrist-camera view, with 8 oklch hue-rotated copies. Same
  scene, same geometry, same luminance — only the color cast varies.
  This is exactly the "domain randomization" batch shape a
  vision-based policy / matcher / retrieval head would train on
  if you wanted lighting / sensor-color invariance.

* ``assets/robot_lighting_sweep.gif`` — continuous 0°→360° lighting
  sweep on the same tabletop scene. Each frame is a different "color
  of the room light" while the geometry / shadows / specular layout
  is unchanged. Used to give an at-a-glance feel for the kind of
  appearance variation a policy ought to be invariant to.

* ``assets/robot_matcher_pool.png`` — visualises the pool-expansion
  use case: one reference image of a tabletop, 5 augmented variants,
  feeding a hypothetical retrieval / matching head (rendered as
  dashed arrows for clarity). This was the original motivation in
  mosaicraft-active-vision's Hungarian-vs-Sinkhorn benchmark.

Run from repo root:

    python scripts/make_robotics_demo.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as _Image
from skimage.data import coffee

from oklch_aug import HueRotatePool, rotate_hue_oklch

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def _center_square_crop(img: np.ndarray, size: int | None = None) -> np.ndarray:
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    out = img[y0 : y0 + side, x0 : x0 + side]
    if size is not None and out.shape[0] != size:
        out = np.array(_Image.fromarray(out).resize((size, size), _Image.LANCZOS))
    return out


def _set_axes_off(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def make_dr_batch() -> None:
    """8-panel domain-randomization-style batch from one scene."""
    img = _center_square_crop(coffee(), size=320)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    rotated = [rotate_hue_oklch(img, hue_shift_deg=float(d)) for d in angles]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(11.5, 6.4),
        gridspec_kw={"hspace": 0.08, "wspace": 0.04},
        constrained_layout=False,
    )
    flat = axes.ravel()
    titles = [f"+{a}°" if a else "original" for a in angles]
    for ax, frame, t in zip(flat, rotated, titles, strict=True):
        ax.imshow(frame)
        ax.set_title(t, fontsize=10)
        _set_axes_off(ax)

    fig.suptitle(
        "Domain randomization for a tabletop policy "
        "— same scene, same geometry, same luminance; only the color cast varies",
        fontsize=12,
        y=0.99,
    )
    fig.text(
        0.5,
        0.02,
        "skimage.data.coffee() used as a real-photo proxy for a robot's wrist-camera view. "
        "A pose / grasp policy trained on this batch sees one geometry under 8 color casts.",
        ha="center",
        fontsize=9,
        style="italic",
        color="0.35",
    )
    fig.savefig(ASSETS / "robot_dr_batch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_lighting_sweep_gif() -> None:
    """Continuous 0->360 deg lighting sweep on the tabletop scene."""
    img = _center_square_crop(coffee(), size=256)
    step = 10
    angles = list(range(0, 360, step))
    frames: list[_Image.Image] = []
    for deg in angles:
        rot = rotate_hue_oklch(img, hue_shift_deg=float(deg))
        pil = _Image.fromarray(rot).convert("P", palette=_Image.ADAPTIVE, colors=128)
        frames.append(pil)

    out = ASSETS / "robot_lighting_sweep.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=60,
        optimize=True,
        disposal=2,
    )


def make_matcher_pool() -> None:
    """Reference + 5 augmented variants -> a matcher head, schematic."""
    img = _center_square_crop(coffee(), size=200)
    pool = HueRotatePool(n_variants=4)
    variants = pool([img])  # 5 images: original + 4 rotations

    fig = plt.figure(figsize=(11.5, 4.4))
    # Left half: reference scene
    ref_ax = fig.add_axes((0.02, 0.18, 0.18, 0.66))
    ref_ax.imshow(img)
    ref_ax.set_title("reference scene\n(robot wrist cam, t=0)", fontsize=10)
    _set_axes_off(ref_ax)

    # Right half: 5 pool variants as a column
    n = len(variants)
    pool_left = 0.36
    pool_right = 0.92
    pool_w = (pool_right - pool_left - 0.04 * (n - 1)) / n
    titles = ["original", "+72°", "+144°", "+216°", "+288°"]
    for i, (v, t) in enumerate(zip(variants, titles, strict=True)):
        ax = fig.add_axes((pool_left + i * (pool_w + 0.04), 0.22, pool_w, 0.6))
        ax.imshow(v)
        ax.set_title(t, fontsize=9)
        _set_axes_off(ax)

    # Connector arrow from reference to the pool
    arrow_ax = fig.add_axes((0.21, 0.18, 0.14, 0.66))
    _set_axes_off(arrow_ax)
    arrow_ax.set_xlim(0, 1)
    arrow_ax.set_ylim(0, 1)
    arrow_ax.annotate(
        "",
        xy=(1.0, 0.5),
        xytext=(0.0, 0.5),
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.8),
    )
    arrow_ax.text(
        0.5,
        0.58,
        "HueRotatePool\n(n_variants=4)",
        ha="center",
        fontsize=10,
        color="0.2",
    )
    arrow_ax.text(
        0.5,
        0.36,
        "pool expansion",
        ha="center",
        fontsize=8,
        color="0.45",
        style="italic",
    )

    # Bracket on the right summarising what the matcher receives
    bracket_ax = fig.add_axes((0.93, 0.22, 0.05, 0.6))
    _set_axes_off(bracket_ax)
    bracket_ax.set_xlim(0, 1)
    bracket_ax.set_ylim(0, 1)
    bracket_ax.annotate(
        "",
        xy=(0.0, 0.0),
        xytext=(0.0, 1.0),
        arrowprops=dict(arrowstyle="-", color="0.3", lw=1.5),
    )
    bracket_ax.annotate(
        "",
        xy=(1.0, 0.5),
        xytext=(0.0, 0.5),
        arrowprops=dict(arrowstyle="->", color="0.3", lw=1.5),
    )
    bracket_ax.text(
        1.6,
        0.5,
        "matcher / retrieval head\n(Hungarian, Sinkhorn,\nCLIP-style, ...)",
        ha="left",
        va="center",
        fontsize=9,
        color="0.2",
    )

    fig.text(
        0.5,
        0.04,
        "One reference → 5 visually distinct candidates at identical luminance. "
        "The matcher gets a larger, color-diverse pool without the L drift HSV jitter introduces.",
        ha="center",
        fontsize=9,
        style="italic",
        color="0.35",
    )
    fig.suptitle(
        "Pool-expansion use case — same geometry, color-diverse retrieval candidates",
        fontsize=12,
        y=0.99,
    )
    fig.savefig(ASSETS / "robot_matcher_pool.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("Generating robot_dr_batch.png ...")
    make_dr_batch()
    print("Generating robot_lighting_sweep.gif ...")
    make_lighting_sweep_gif()
    print("Generating robot_matcher_pool.png ...")
    make_matcher_pool()
    print("Done. Outputs in", ASSETS)
