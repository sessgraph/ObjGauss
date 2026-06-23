# Semantic Smoke Benchmark

This benchmark reruns the current ObjGauss object-emergence smoke suite from a
manifest:

```bash
npm run acceptance:semantic
```

Equivalent direct command:

```bash
uv run objgauss object-field emergence-benchmark \
  docs/benchmarks/semantic-smoke.json \
  --output-dir /tmp/objgauss-semantic-smoke-suite \
  --strict
```

The suite writes ignored local outputs only:

```text
/tmp/objgauss-semantic-smoke-suite/summary.json
/tmp/objgauss-semantic-smoke-suite/report.html
/tmp/objgauss-semantic-smoke-suite/<scene>/curve.json
/tmp/objgauss-semantic-smoke-suite/<scene>/curve.csv
```

## Scenes

The manifest currently checks three local smoke scenes:

| Scene | Required inputs | Purpose |
| --- | --- | --- |
| `plush-semantic` | `outputs/assets/converted/plush.ply`, `outputs/demos/plush-semantic-closure/object_field_initial.npz`, `outputs/demos/plush-semantic-closure/mask-manifest.json` | Real Plush 3DGS plus deterministic 2D color semantic masks |
| `lego-alpha-proxy` | `outputs/demos/lego-alpha-closure/lego_proxy_raw.ply`, `outputs/demos/lego-alpha-closure/object_field_initial.npz`, `outputs/demos/lego-alpha-closure/mask-manifest.json` | NeRF Lego posed RGBA proxy plus 2D color masks |
| `lego-splatfacto-smoke` | `outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply`, `outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz`, `outputs/masks/nerf-lego-sam/mask-manifest.json` | Real Splatfacto smoke PLY plus SAM masks |

## Generate Missing Inputs

Generate the Plush semantic scene:

```bash
uv run objgauss assets pull plush-3dgs-local
uv run objgauss demo plush-semantic-closure --iterations 80
```

Generate the Lego alpha proxy scene:

```bash
uv run objgauss assets pull nerf-synthetic-lego
uv run objgauss demo lego-alpha-closure \
  --max-frames 12 \
  --sample-stride 8 \
  --iterations 120
```

Generate the Lego SAM mask manifest when SAM is available locally:

```bash
uv run \
  --with torch \
  --with torchvision \
  --with "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git" \
  objgauss masks from-nerf-sam \
  outputs/assets/training/nerf-synthetic-lego \
  --output outputs/masks/nerf-lego-sam/mask-manifest.json \
  --checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --device cuda \
  --split train \
  --max-frames 2 \
  --max-masks-per-frame 8 \
  --min-area 64
```

The Splatfacto PLY is produced by Nerfstudio, not by ObjGauss. The expected
handoff path is:

```text
outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply
```

After that PLY exists, rebuild its Object Field inputs:

```bash
uv run objgauss object-field init \
  outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply \
  --output outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz \
  --slots 8

uv run objgauss object-field vote-masks \
  outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply \
  --field outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz \
  --masks outputs/masks/nerf-lego-sam/mask-manifest.json \
  --output outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_sam.npz \
  --ply-output outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/lego_splatfacto_sam_objects.ply \
  --iterations 80 \
  --learning-rate 1.0 \
  --colorize
```

`TRAIN-003` will formalize the Nerfstudio training runbook. Until then, this
benchmark treats the Splatfacto smoke PLY as a local prerequisite and reports a
clear missing-output error when it is absent.

## Acceptance Integration

`npm run acceptance:demo` runs this benchmark by default after rebuilding the
closure demos and browser audit. Use this only on machines with the Splatfacto
smoke prerequisite above. For the older demo-only loop:

```bash
npm run acceptance:demo -- --skip-semantic-benchmark
```
