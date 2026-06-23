# Splatfacto Safe-2000 Mask Variant Benchmark

This benchmark compares multiple SAM mask / slot policies on the same
safe-2000 NeRF Lego Splatfacto PLY. It is the next step after the single
balanced candidate benchmark: the goal is to produce a small comparison table
before scaling to more scenes.

Run all variants:

```bash
npm run benchmark:splatfacto:variants -- --run
```

Preview without running:

```bash
npm run benchmark:splatfacto:variants -- --dry-run
```

Check local prerequisites and outputs:

```bash
npm run benchmark:splatfacto:variants -- --status
```

Reuse existing SAM manifests when they are already present:

```bash
npm run benchmark:splatfacto:variants -- --run --skip-sam
```

## Variants

| Variant | Mask policy | Slots | Manifest |
| --- | --- | ---: | --- |
| `sam2f-slots8` | 2 frames, 8 largest SAM masks per frame | 8 | `outputs/masks/nerf-lego-sam/mask-manifest.json` |
| `sam8f-slots8-unfiltered` | 8 frames, 8 largest SAM masks per frame, no area cap | 8 | `outputs/masks/nerf-lego-sam-8f/mask-manifest.json` |
| `sam8f-slots4-balanced03` | 8 frames, 4 largest SAM masks per frame after `max_area_fraction=0.3` | 4 | `outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json` |

All variants start from:

```text
outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply
```

If that PLY is missing, regenerate it through the TRAIN-003C notes in
`docs/training/splatfacto-smoke.md`. This benchmark does not train Splatfacto.

## Outputs

The suite writes ignored local outputs only:

```text
/tmp/objgauss-splatfacto-safe-2000-variant-suite/summary.json
/tmp/objgauss-splatfacto-safe-2000-variant-suite/summary.csv
/tmp/objgauss-splatfacto-safe-2000-variant-suite/summary.md
/tmp/objgauss-splatfacto-safe-2000-variant-suite/report.html
/tmp/objgauss-splatfacto-safe-2000-variant-suite/<variant>/summary.json
/tmp/objgauss-splatfacto-safe-2000-variant-suite/<variant>/curve.json
```

Per-variant Object Field outputs are also ignored:

```text
outputs/assets/gaussians/nerf-lego-trained-safe-2000-<variant>-benchmark/
```

Each per-variant run delegates to `scripts/benchmark-splatfacto-balanced.mjs`,
so it uses the same registration, emergence metric, curve, report, and summary
contract as BENCH-001. The suite summary flattens those per-variant summaries
into a table with:

```text
frames
masks
mask_pixels
slots
supervised_gaussians
object_id_counts
registration loss
ARI
OES
render_occlusion_effect_score
```

## Current Interpretation

The expected comparison is not "more masks is always better." SEG-003 already
showed that unfiltered 8-frame SAM increases coverage but can worsen slot
balance and stability because broad/background masks dominate. The comparison
table should make that visible:

- `sam2f-slots8` is the earlier sparse baseline.
- `sam8f-slots8-unfiltered` tests higher coverage without mask balancing.
- `sam8f-slots4-balanced03` tests stricter masks and fewer slots.

Treat the winner as a local benchmark candidate, not as a final object
emergence claim. The next real scaling step is the same suite across additional
scenes or stronger mask sources.

## Cross-Scene Table

Aggregate this variant suite with the broader semantic smoke scenes:

```bash
npm run benchmark:cross-scene -- --run
```

The cross-scene runbook is `docs/benchmarks/cross-scene.md`.
