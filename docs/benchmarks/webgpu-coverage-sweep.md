# WebGPU Coverage Sweep

This benchmark compares WebGPU Tile edit-preview tuning variants against the
Spark `.splat` view using the existing browser audit visual residual metrics.

It is for renderer-quality decisions only. It does not change the default
renderer parameters by itself.

Depth-bin alpha compositing can be fixed for a run with
`--webgpu-depth-bins <4-16>`. This is a runtime tuning knob for comparing sorted
alpha approximations; it does not change the default 8-bin baseline unless a
candidate passes the same gate.

Depth alpha mode can be fixed for a run with
`--webgpu-depth-alpha-mode depth-binned|front-top-k`. The default is
`depth-binned`. `front-top-k` keeps the closest K pixel contributors, where K is
controlled by `--webgpu-depth-bins`, then composites them front-to-back.

Camera framing can be fixed for a run with
`--webgpu-camera-mode edit-fixed|spark-frame`. The default is `edit-fixed`.
`spark-frame` matches the Spark viewer framing constants more closely and is a
diagnostic knob for separating camera coverage residual from alpha / SH /
renderer-compositing residual.

Color mode can be fixed for a run with `--webgpu-color-mode source|sh-view`.
The default is `source`. Use `sh-view` when measuring trained outputs that carry
`f_dc_*` and `f_rest_*`, so coverage experiments are not confounded by static
RGB color residual.

## SH-View Coverage Sweep

Use this when the asset has full SH rest coefficients and you want to compare
footprint / anisotropy variants after view-dependent color has already been
enabled:

```bash
npm run audit:webgpu-coverage-sweep -- \
  --asset nerf-lego-trained-output-local \
  --webgpu-color-mode sh-view \
  --output-dir /tmp/objgauss-webgpu-coverage-trained-sh-view
```

Current trained Lego result:

| Variant | Coverage ratio | Luma delta | Chroma delta | SH-view after delete | Tile refs |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 31.205176 | 0.034507 | 0.055774 | 255794 | 581933 |
| compact | 25.958842 | 0.070796 | 0.054231 | 255794 | 540496 |
| tight | 23.164633 | 0.093626 | 0.053189 | 255794 | 525755 |

`tight` wins coverage and tile-reference cost, but luma is `2.71x` the
baseline. Therefore footprint tightening alone should remain diagnostic for the
trained output; it does not justify changing the default renderer parameters.

## Depth-Bin Sweep

Use the depth sweep when comparing sorted-alpha approximations while holding the
coverage tuning fixed:

```bash
npm run audit:webgpu-depth-sweep -- \
  --asset nerf-lego-alpha-closure-local \
  --bins 4,8,12,16 \
  --output-dir /tmp/objgauss-webgpu-depth-sweep
```

The sweep runs the same headed desktop WebGPU full-runtime audit as the coverage
sweep and writes:

```text
/tmp/objgauss-webgpu-depth-sweep/summary.json
/tmp/objgauss-webgpu-depth-sweep/summary.md
```

Current Lego result: 8 bins remains the best Pareto variant; 12 bins gives the
lowest coverage ratio by a tiny margin but worsens chroma. That means simply
raising the bin count does not currently explain the Spark/edit visual residual
on Lego.

## Camera Mode Diagnostic

Use the camera mode knob when comparing fixed edit-camera framing against
Spark-style scene framing:

```bash
npm run audit:webgpu-desktop -- \
  --asset nerf-lego-alpha-closure-local \
  --probes full \
  --webgpu-camera-mode spark-frame
```

Current local result:

| Scene | Camera mode | Coverage ratio | Luma delta | Chroma delta |
| --- | --- | ---: | ---: | ---: |
| NeRF Lego | edit-fixed | 3.784251 | 0.106079 | 0.086537 |
| NeRF Lego | spark-frame | 3.766657 | 0.102396 | 0.087290 |
| Plush semantic | spark-frame | 4.713926 | 0.117382 | 0.016269 |

`spark-frame` slightly improves Lego coverage/luma and substantially improves
Plush coverage relative to the last baseline coverage gate, but it worsens some
color deltas. This makes camera framing a useful diagnostic axis, not enough by
itself to make the edit renderer visually equivalent to Spark.

## Depth Alpha Mode Diagnostic

Use the alpha mode knob when comparing depth-bin compositing against a closer
front-to-back approximation:

```bash
npm run audit:webgpu-desktop -- \
  --asset nerf-lego-alpha-closure-local \
  --probes full \
  --webgpu-depth-alpha-mode front-top-k
```

Current local result:

| Scene | Alpha mode | K | Coverage ratio | Luma delta | Chroma delta |
| --- | --- | ---: | ---: | ---: | ---: |
| NeRF Lego | depth-binned | 8 | 3.784251 | 0.106079 | 0.086537 |
| NeRF Lego | front-top-k | 8 | 3.583371 | 0.208595 | 0.127958 |
| NeRF Lego | front-top-k | 16 | 3.778381 | 0.173505 | 0.113605 |
| Plush semantic | front-top-k | 8 | 6.115472 | 0.245489 | 0.077452 |

`front-top-k` proves a real per-pixel sorted-alpha diagnostic path can run in
the WebGPU tile renderer, but it currently trades lower coverage for worse color
residual. It should remain a diagnostic mode until a later compositing strategy
beats the baseline gate on coverage, luma, and chroma together.

## Smoke Sweep

```bash
npm run audit:webgpu-coverage-sweep -- --assets nerf-lego-alpha-closure-local,plush-semantic-closure-local --output-dir /tmp/objgauss-webgpu-coverage-sweep
```

The default variants are:

| Variant | Footprint scale | Max anisotropy |
| --- | ---: | ---: |
| baseline | 2.2 | 4 |
| compact | 1.9 | 3 |
| tight | 1.7 | 2.5 |

The score is lower-is-better and is normalized per scene against `baseline`:

| Metric | Weight |
| --- | ---: |
| coverage ratio | 0.35 |
| luma delta | 0.25 |
| chroma delta | 0.25 |
| tile references | 0.15 |

Outputs:

```text
/tmp/objgauss-webgpu-coverage-sweep/summary.json
/tmp/objgauss-webgpu-coverage-sweep/summary.md
```

## Default-Parameter Gate

Before changing the default WebGPU coverage tuning, run the gate:

```bash
npm run audit:webgpu-coverage-gate
```

The gate currently checks the `baseline` variant against normalized thresholds:

```text
meanParetoScore <= 1
meanLumaNorm <= 1
meanChromaNorm <= 1
per-scene pareto/luma/chroma <= 1
```

To evaluate a candidate as a possible new default, pass it explicitly:

```bash
npm run audit:webgpu-coverage-sweep -- \
  --assets nerf-lego-alpha-closure-local,plush-semantic-closure-local \
  --output-dir /tmp/objgauss-webgpu-coverage-compact-gate \
  --gate-variant compact \
  --max-mean-pareto-score 1 \
  --max-mean-luma-norm 1 \
  --max-mean-chroma-norm 1 \
  --max-scene-pareto-score 1 \
  --max-scene-luma-norm 1 \
  --max-scene-chroma-norm 1
```

That candidate should not become the default unless it passes the gate and the
resulting report explains the visual tradeoff.

## Current Local Result

The current two-scene headed desktop WebGPU run shows:

| Variant | Mean score | Mean luma norm | Mean chroma norm | Mean tile refs norm |
| --- | ---: | ---: | ---: | ---: |
| baseline | 1 | 1 | 1 | 1 |
| compact | 0.921829 | 1.271571 | 0.579758 | 0.848985 |
| tight | 1.072299 | 1.465639 | 1.097803 | 0.756116 |

`compact` has the best mean score because it improves chroma and tile-reference
cost on Plush, but it worsens luma. `tight` improves coverage/cost but worsens
luma strongly. Therefore the default remains `baseline`.
