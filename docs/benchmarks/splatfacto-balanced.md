# Splatfacto Safe-2000 Balanced Benchmark

This runbook makes the current safe-2000 / 8-frame / 4-slot SAM candidate
reproducible without committing local training outputs. It keeps Splatfacto
training separate from the benchmark: the benchmark starts from an existing
safe-2000 exported PLY, regenerates balanced SAM masks, registers the Object
Field handoff, and writes emergence metrics and curves.

```bash
npm run benchmark:splatfacto:balanced -- --run
```

Preview the commands:

```bash
npm run benchmark:splatfacto:balanced -- --dry-run
```

Check local prerequisites and generated outputs:

```bash
npm run benchmark:splatfacto:balanced -- --status
```

## Inputs

Required local inputs:

```text
outputs/assets/training/nerf-synthetic-lego/transforms_train.json
outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply
/home/ljy/models/sam/sam_vit_b_01ec64.pth
```

Override the SAM checkpoint with either:

```bash
SAM_CHECKPOINT=/path/to/sam_vit_b_01ec64.pth \
npm run benchmark:splatfacto:balanced -- --run
```

or:

```bash
npm run benchmark:splatfacto:balanced -- \
  --run \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth
```

If the safe-2000 PLY is missing, regenerate it through the TRAIN-003C notes in
`docs/training/splatfacto-smoke.md`. This benchmark does not start a long
Splatfacto train by itself.

## Fixed Candidate Defaults

```text
input_ply=outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply
sam_manifest=outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json
slots=4
sam_max_frames=8
sam_max_masks_per_frame=4
sam_min_area=64
sam_max_area_fraction=0.3
object_iterations=160
curve_iterations=80
eval_every=20
render_size=96
```

The default registration uses `--no-public-copy`, so it does not overwrite
`public/samples/nerf_lego_trained.*`. Use `--publish` only when intentionally
updating the local frontend sample.

## Outputs

The script writes ignored local outputs:

```text
outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json
outputs/assets/gaussians/nerf-lego-trained-safe-2000-sam8f-balanced03-slots4-benchmark/
/tmp/objgauss-splatfacto-balanced-benchmark/emergence.json
/tmp/objgauss-splatfacto-balanced-benchmark/curve.json
/tmp/objgauss-splatfacto-balanced-benchmark/curve.csv
/tmp/objgauss-splatfacto-balanced-benchmark/report.html
/tmp/objgauss-splatfacto-balanced-benchmark/summary.json
```

`summary.json` records:

```text
frames
masks
mask_pixels
object_id_counts
projection_loss
stability_ari
object_emergence_score
render_occlusion_effect_score
```

The summary is considered passed when the expected frame count is present,
masks exist, all four object slots are non-empty, projection loss decreases,
ARI and OES are recorded, and render occlusion effect is recorded.

## Current Local Baseline

The current safe-2000 balanced candidate measured:

```text
frames=8
masks=27
mask_pixels=664780
object_id_counts=126686/40747/34682/53679
supervised_gaussians=70025
projection_loss=2.782336 -> 0.044949
stability_ari=0.468745
object_emergence_score=0.693888
render_occlusion_effect_score=0.195308
```

This is a benchmark candidate, not a final quality claim. It fixes the earlier
near-empty slot failure and produces a stronger render occlusion signal, but it
still needs broader scene coverage and better slot-consistency analysis before
being treated as a paper-grade result.

## Compare Mask Variants

Use the variant suite when comparing this balanced candidate against earlier
safe-2000 mask policies:

```bash
npm run benchmark:splatfacto:variants -- --run --skip-sam
```

The variant runbook is `docs/benchmarks/splatfacto-variants.md`. It compares the
2-frame / 8-slot SAM baseline, the unfiltered 8-frame / 8-slot SAM run, and this
8-frame / 4-slot `max_area_fraction=0.3` candidate on the same safe-2000 PLY.
