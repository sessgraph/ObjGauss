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

Generate the Lego Splatfacto smoke scene and SAM Object Field handoff:

```bash
SAM_CHECKPOINT=/home/ljy/models/sam/sam_vit_b_01ec64.pth \
npm run train:splatfacto:smoke -- --run --skip-benchmark
```

The Splatfacto PLY is produced by Nerfstudio, not by ObjGauss. The formal
TRAIN-003A runbook is `docs/training/splatfacto-smoke.md`; it records the
Nerfstudio command, CUDA / `gsplat` package notes, SAM checkpoint requirement,
and output contract. Preview the commands before starting the longer run:

```bash
npm run train:splatfacto:smoke -- --dry-run
```

Check whether the local smoke outputs are already present:

```bash
npm run train:splatfacto:smoke -- --status
```

The benchmark still treats `outputs/` as local ignored state. Missing Splatfacto
outputs now have a reproducible generation entrypoint instead of a hand-written
handoff.

## Acceptance Integration

`npm run acceptance:demo` runs this benchmark by default after rebuilding the
closure demos and browser audit. Use this only on machines with the Splatfacto
smoke prerequisite above. For the older demo-only loop:

```bash
npm run acceptance:demo -- --skip-semantic-benchmark
```
