# PyPI upload — runbook (user-only)

The build artefacts are ready under `dist/`. The actual upload step
needs a PyPI API token; per the user's R11 rule, the assistant must
not hold or pipe that token. Run these commands yourself.

## State

```
dist/oklch_aug-0.0.0-py3-none-any.whl   14 KB
dist/oklch_aug-0.0.0.tar.gz             15 KB
```

Both files passed `python -m twine check` (license metadata, README
rendering, classifiers, long-description content-type — all clean).

## Pre-upload sanity (optional)

```bash
# Inspect what is in the wheel before uploading.
python -m zipfile -l dist/oklch_aug-0.0.0-py3-none-any.whl | head -40
```

## Step 1 — upload to TestPyPI first (recommended)

```bash
# 1. Get a TestPyPI token from https://test.pypi.org/manage/account/
#    Scope: "Entire account" the first time; tighten to project-scoped
#    after the project page exists.
# 2. Upload.
python -m twine upload --repository testpypi dist/*

# 3. Verify install from TestPyPI in a throwaway venv:
python -m venv /tmp/oklch-test && \
  /tmp/oklch-test/bin/pip install \
    --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    oklch-aug==0.0.0 && \
  /tmp/oklch-test/bin/python -c \
    "from oklch_aug import rotate_hue_oklch; print('ok')"
```

## Step 2 — upload to real PyPI

```bash
# 1. Get a PyPI token from https://pypi.org/manage/account/
# 2. Upload.
python -m twine upload dist/*
```

## Step 3 — bump CI to use PyPI instead of git+

After the upload succeeds, edit
`/home/runza/oss/mosaicraft-active-vision/.github/workflows/ci.yml`:

```yaml
# Replace this line:
pip install "oklch-aug @ git+https://github.com/hinanohart/oklch-aug.git@7d869330d39885d497d19601df9d1a371a6fd24b"

# With this:
pip install "oklch-aug==0.0.0"
```

Then push to mosaicraft-active-vision main. CI on that repo should
stay green.

## Version policy note (decision pending)

The build is tagged `0.0.0` — placeholder per `pyproject.toml`. A real
"first release" would conventionally be `0.1.0`. Whether to ship as
`0.0.0` (matching the current `__version__`) or to bump to `0.1.0` is
a user call; the assistant did not unilaterally change it (R14
"OSS-公開 / 根本改修" trigger applies to a version bump that lands on
PyPI).

If you choose to bump:

```bash
# 1. Edit pyproject.toml [project] version
# 2. Edit src/oklch_aug/__init__.py __version__
# 3. Re-run:
rm -rf dist/ build/
python -m build
python -m twine check dist/*
# 4. Upload as above.
```

## After upload

* Update `PROGRESS.md` row 20 (oklch-aug PyPI upload) from ⬜ to ✅.
* Update the README install snippet from `pip install
  git+https://github.com/hinanohart/oklch-aug.git` to `pip install
  oklch-aug`.
* (Optional) Tag a git release: `git tag v0.0.0 && git push --tags`.
