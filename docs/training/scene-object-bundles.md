# ObjGauss Scene/Object Bundle

> Status: current
> Last updated: 2026-06-26

ObjGauss training data should be treated as a traceable scene/object bundle, not as loose
`images + masks + ply` files. A usable bundle binds:

- the source asset and posed image set
- the mask manifest and slot definitions
- the trained Gaussian PLY / splat
- the trained Object Field and object-aware PLY
- Gaussian count, slot count, mask frame count, and validation facts

This prevents mixing an Object Field trained for one Gaussian count with a different PLY.

## Level 1 Lego Foreground/Background

For NeRF Synthetic Lego, the first stable target is foreground/background, not part-level editing:

```text
slot 0 = background
slot 1 = foreground Lego
ignore = alpha boundary
```

Generate alpha-derived masks for every train frame:

```bash
uv run objgauss masks from-nerf-alpha-fgbg \
  outputs/assets/training/nerf-synthetic-lego \
  --output outputs/masks/nerf-lego-alpha-fgbg-v1/mask-manifest.json \
  --split train \
  --background-threshold 20 \
  --foreground-threshold 200 \
  --background-confidence 0.05
```

`background-confidence` is intentionally below foreground confidence. The current mask voting path
does not model depth occlusion, so full-frame background masks can otherwise dominate foreground
votes for Gaussians that project into background pixels from many unrelated views.

Validate the mask manifest before training Object Field:

```bash
uv run objgauss masks validate \
  outputs/masks/nerf-lego-alpha-fgbg-v1/mask-manifest.json \
  --dataset outputs/assets/training/nerf-synthetic-lego \
  --summary-output outputs/masks/nerf-lego-alpha-fgbg-v1/validation-summary.json \
  --strict \
  --max-mask-area-fraction 0.995
```

Register an existing trained Gaussian PLY with that mask bundle:

```bash
uv run objgauss training register-output \
  outputs/training/nerf-lego-splatfacto-near1m/export-near1m-cpu-cache-v1/splat.ply \
  --asset-id nerf-lego-alpha-fgbg-v1-local \
  --output-dir outputs/assets/gaussians/nerf-lego-alpha-fgbg-v1 \
  --dataset outputs/assets/training/nerf-synthetic-lego \
  --masks outputs/masks/nerf-lego-alpha-fgbg-v1/mask-manifest.json \
  --slots 2 \
  --iterations 160 \
  --learning-rate 1.0 \
  --no-public-copy
```

Write the traceable sample bundle:

```bash
uv run objgauss training write-sample-bundle \
  --output outputs/samples/objgauss-lego-alpha-fgbg-v1/sample.json \
  --sample-id objgauss-lego-alpha-fgbg-v1 \
  --asset-id nerf-synthetic-lego \
  --dataset outputs/assets/training/nerf-synthetic-lego \
  --masks outputs/masks/nerf-lego-alpha-fgbg-v1/mask-manifest.json \
  --training-manifest outputs/assets/gaussians/nerf-lego-alpha-fgbg-v1/training-output-manifest.json
```

## Current Local Result

The local 168,653-Gaussian Lego candidate can be bound into a Level 1 bundle. The current
recommended conservative run is `bg005-v2`:

- mask manifest: 100 frames, 200 masks, 800x800
- alpha pixels: foreground `843,797`, background `61,517,043`, ignore `1,639,160`
- validation: passed with `max_mask_area_fraction=0.995`, zero overlap
- Object Field: `slots=2`, `supervised_gaussians=149,892`
- projection loss: `1.264038 -> 0.497213`
- object counts: background `133,074`, foreground `35,579`
- high-confidence export at `min_confidence=0.7`: background `100,730`, foreground `16,915`, unknown `51,008`

For comparison, `background-confidence=0.02` is more aggressive:

- projection loss: `1.430821 -> 0.535701`
- object counts: background `88,582`, foreground `80,071`
- high-confidence export at `min_confidence=0.7`: background `64,885`, foreground `41,445`, unknown `62,323`

This proves the traceable bundle and full-frame alpha supervision path works. It does not yet prove
stable part-level object separation. The remaining `vote_conflict_fraction=0.430143` is consistent
with projection voting without depth occlusion; before returning to 4-slot SAM part masks, prefer
adding depth/visibility-aware voting or a reviewed threshold/weight sweep.
