# Splatfacto Cross-Scene Benchmark

This suite compares Splatfacto-trained scene outputs. It is separate from
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
| `fern-splatfacto-smoke` | `outputs/training/nerf-fern-splatfacto-smoke/export-smoke-cuda/splat.ply` | 4 SAM frames, 6 masks per frame, max area 0.35, max image size 768 | Second Splatfacto-trained scene from LLFF/COLMAP Fern |
| `chair-splatfacto-smoke` | `outputs/training/polyhaven-chair-splatfacto-smoke/export-smoke-cuda/splat.ply` | 8 SAM frames, 6 masks per frame, max area 0.75 | Third Splatfacto-trained scene from a CC0 Poly Haven mesh-derived NeRF render set |

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

If a scene declares `train_sam_manifest` and `heldout_manifest`, the per-scene
run first splits the source SAM manifest with `objgauss masks split-manifest`.
The split writes lightweight JSON manifests only and keeps the original mask
arrays in place. The current default split holds out every fourth frame:

```json
{
  "heldout": {
    "every": 4,
    "offset": 3
  }
}
```

The render occlusion probe currently uses the ObjGauss deterministic
`scale_aware_cpu_splat_l1` renderer. It projects Gaussian centers into the
mask-manifest cameras, uses Gaussian scale and opacity to rasterize a small CPU
splat footprint, removes each hard object slot, and compares full-vs-removed
RGBA images. This is stronger than the earlier center-point probe, but still not
a full covariance-aware `gsplat` training renderer.

Current local result after refreshing with train/held-out splits and the
scale-aware probe:

| Scene | Train frames | Held-out frames | ARI | Curve OES | Render effect | Held-out loss | Held-out render |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lego-splatfacto-safe-2000` | 6 | 2 | 0.469787 | 0.784051 | 0.229397 | 2.301630 | 0.197505 |
| `fern-splatfacto-smoke` | 3 | 1 | 0.790636 | 0.780132 | 0.235029 | 0.670722 | 0.233851 |
| `chair-splatfacto-smoke` | 6 | 2 | 0.614363 | 0.757609 | 0.248716 | 2.284750 | 0.224084 |

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

## Chair Preparation

The chair scene starts from the CC0 Poly Haven School Chair glTF and uses
`objgauss.mesh_nerf` to render a deterministic NeRF-style RGBA orbit dataset.
It is a reproducible third Splatfacto-trained row, not a real camera-captured
multi-view scene.

Generate the training input:

```bash
uv run objgauss assets pull polyhaven-school-chair-nerf
```

Run the resource-safe Splatfacto smoke:

```bash
npm run train:splatfacto:smoke -- \
  --run \
  --asset-id polyhaven-school-chair-nerf \
  --dataset outputs/assets/training/polyhaven-school-chair-nerf \
  --output-root outputs/training/polyhaven-chair-splatfacto-smoke \
  --experiment chair-splatfacto-smoke \
  --timestamp smoke-cuda \
  --export-dir outputs/training/polyhaven-chair-splatfacto-smoke/export-smoke-cuda \
  --object-field-dir outputs/training/polyhaven-chair-splatfacto-smoke/object-field-sam \
  --sam-manifest outputs/masks/polyhaven-chair-sam-smoke/mask-manifest.json \
  --data-parser blender-data \
  --iterations 100 \
  --steps-per-save 100 \
  --vis tensorboard \
  --cache-images cpu \
  --camera-res-scale-factor 0.5 \
  --cuda-home /tmp/objgauss-cuda13 \
  --max-jobs 2 \
  --device cuda \
  --sam-max-frames 8 \
  --sam-max-masks-per-frame 6 \
  --sam-min-area 64 \
  --sam-max-area-fraction 0.75 \
  --slots 6 \
  --object-iterations 80 \
  --skip-benchmark
```

Then refresh the scene suite:

```bash
npm run benchmark:splatfacto:scenes -- --run --skip-sam
```

## Interpretation Boundary

This suite is a reproducible experiment base, not a final quality claim.
Fern and Chair are currently smoke-level scenes. The important regression is
whether the same Object Field metrics, curve generation, held-out evaluation,
and render occlusion probe can run across multiple Splatfacto-trained scene
rows.

The current scene suite closes the previous paper-gate count gap: three
Splatfacto-trained scene rows and three held-out eval rows are present. The next
gap is quality depth, not table shape: longer or higher-quality reconstructions,
more realistic scenes, and stronger renderer-backed occlusion checks.
