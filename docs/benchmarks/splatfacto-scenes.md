# Splatfacto Cross-Scene Benchmark

This suite compares real Splatfacto scene outputs. It is separate from
`splatfacto-variants`, which compares mask policies on one Lego PLY.

Run all configured scenes:

```bash
npm run benchmark:splatfacto:scenes -- --run
```

Inspect the pipeline without running SAM or Object Field work:

```bash
npm run benchmark:splatfacto:scenes -- --dry-run
```

Check local inputs and outputs:

```bash
npm run benchmark:splatfacto:scenes -- --status
```

## Scenes

The manifest is `docs/benchmarks/splatfacto-scenes.json`.

| Scene | Input | Mask policy | Purpose |
| --- | --- | --- | --- |
| `lego-splatfacto-safe-2000` | `outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply` | 8 SAM frames, 4 masks per frame, max area 0.3 | Current stronger Lego geometry baseline |
| `fern-splatfacto-smoke` | `outputs/training/nerf-fern-splatfacto-smoke/export-smoke-cuda/splat.ply` | 4 SAM frames, 6 masks per frame, max area 0.35, max image size 768 | Second real Splatfacto scene from LLFF/COLMAP Fern |

Outputs:

```text
/tmp/objgauss-splatfacto-scene-suite/summary.json
/tmp/objgauss-splatfacto-scene-suite/summary.csv
/tmp/objgauss-splatfacto-scene-suite/summary.md
/tmp/objgauss-splatfacto-scene-suite/report.html
/tmp/objgauss-splatfacto-scene-suite/<scene>/summary.json
/tmp/objgauss-splatfacto-scene-suite/<scene>/curve.json
```

Each per-scene run delegates to `scripts/benchmark-splatfacto-balanced.mjs`
with scene-specific paths and SAM settings.

## Fern Preparation

Fern comes from the same NeRF example zip but uses LLFF/COLMAP data instead of
Blender transforms. Pulling the asset extracts `nerf_llff_data/fern` and writes
a NeRF-style `transforms_train.json` from `sparse/0/cameras.bin` and
`sparse/0/images.bin`:

```bash
uv run objgauss assets pull nerf-llff-fern
```

Run the resource-safe Splatfacto smoke:

```bash
npm run train:splatfacto:smoke -- \
  --run \
  --asset-id nerf-llff-fern \
  --dataset outputs/assets/training/nerf-llff-fern \
  --output-root outputs/training/nerf-fern-splatfacto-smoke \
  --experiment fern-splatfacto-smoke \
  --timestamp smoke-cuda \
  --export-dir outputs/training/nerf-fern-splatfacto-smoke/export-smoke-cuda \
  --object-field-dir outputs/training/nerf-fern-splatfacto-smoke/object-field-sam \
  --sam-manifest outputs/masks/nerf-fern-sam-smoke/mask-manifest.json \
  --dataparser-transform outputs/training/nerf-fern-splatfacto-smoke/fern-splatfacto-smoke/splatfacto/smoke-cuda/dataparser_transforms.json \
  --data-parser colmap \
  --downscale-factor 1 \
  --images-path images \
  --colmap-path sparse/0 \
  --iterations 100 \
  --steps-per-save 100 \
  --vis tensorboard \
  --cache-images cpu \
  --camera-res-scale-factor 0.25 \
  --cuda-home /tmp/objgauss-cuda13 \
  --max-jobs 2 \
  --device cpu \
  --sam-max-frames 4 \
  --sam-max-masks-per-frame 6 \
  --sam-min-area 256 \
  --sam-max-area-fraction 0.35 \
  --sam-max-image-size 768 \
  --slots 6 \
  --object-iterations 80 \
  --skip-benchmark
```

Then run the scene benchmark:

```bash
npm run benchmark:splatfacto:scenes -- --run --skip-sam
```

Use `--skip-sam` only when the SAM manifests already exist.

## Interpretation Boundary

This suite is a reproducible experiment base, not a final quality claim.
Fern is currently a smoke-level scene. The important regression is whether the
same Object Field metrics, curve generation, and render occlusion probe can run
across more than one real Splatfacto scene.
