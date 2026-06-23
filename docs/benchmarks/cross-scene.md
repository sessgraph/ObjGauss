# Cross-Scene Emergence Benchmark

This benchmark aggregates the current ObjGauss emergence evidence into one
table. It combines:

- the three-scene semantic smoke suite from `docs/benchmarks/semantic-smoke.json`;
- the real Splatfacto scene suite from
  `docs/benchmarks/splatfacto-scenes.json`;
- the safe-2000 Splatfacto mask variant suite from
  `docs/benchmarks/splatfacto-variants.md`.

Run the full aggregation:

```bash
npm run benchmark:cross-scene -- --run
```

Preview commands:

```bash
npm run benchmark:cross-scene -- --dry-run
```

Check local outputs:

```bash
npm run benchmark:cross-scene -- --status
```

## Default Behavior

The cross-scene wrapper runs:

```bash
npm run acceptance:semantic -- \
  --manifest docs/benchmarks/semantic-smoke.json \
  --output-dir /tmp/objgauss-semantic-smoke-suite

npm run benchmark:splatfacto:scenes -- \
  --run \
  --manifest docs/benchmarks/splatfacto-scenes.json \
  --suite-output-dir /tmp/objgauss-splatfacto-scene-suite \
  --skip-sam

npm run benchmark:splatfacto:variants -- \
  --run \
  --suite-output-dir /tmp/objgauss-splatfacto-safe-2000-variant-suite \
  --skip-sam
```

`--skip-sam` is the default for the Splatfacto scene and variant suites because
this benchmark is testing aggregation, not regenerating SAM masks. Use
`--refresh-sam` to force the suites to regenerate SAM manifests.

Reuse already generated summaries:

```bash
npm run benchmark:cross-scene -- --run --skip-semantic --skip-scenes --skip-variants
```

## Outputs

The wrapper writes ignored local outputs only:

```text
/tmp/objgauss-cross-scene-benchmark/summary.json
/tmp/objgauss-cross-scene-benchmark/summary.csv
/tmp/objgauss-cross-scene-benchmark/summary.md
/tmp/objgauss-cross-scene-benchmark/summary.html
```

The unified rows include:

```text
suite
scene_id
variant_id
frames
masks
gaussians
slots
supervised_gaussians
object_id_counts
projection loss
ARI
Object Emergence Score
render_occlusion_effect_score
```

## Interpretation Boundary

This is a table-building benchmark, not a new training result. Its purpose is to
make the current evidence comparable across scenes and mask policies:

- `semantic-smoke` rows show the existing Plush semantic, Lego alpha proxy, and
  Splatfacto smoke scenes.
- `splatfacto-scenes` rows compare real Splatfacto scene outputs such as Lego
  safe-2000 and LLFF Fern smoke.
- `splatfacto-safe2000-variants` rows show mask policy changes on one stronger
  Splatfacto reconstruction.

The scene suite is the place to add new real Splatfacto scenes. The variant
suite remains an ablation table for mask and slot policy on a single scene.
