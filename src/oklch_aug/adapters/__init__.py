"""Optional integration adapters for popular pipelines.

These submodules import their host library inside their own module
scope, so the bare ``import oklch_aug`` line never pulls torch /
albumentations / kornia in. Install the matching extra to use them:

* ``pip install oklch-aug[albumentations]`` → :mod:`oklch_aug.adapters.albumentations`
* ``pip install oklch-aug[torch]`` → :mod:`oklch_aug.adapters.torch`
"""
