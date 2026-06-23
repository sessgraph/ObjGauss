# WebGPU Coverage Sweep

This benchmark compares WebGPU Tile edit-preview tuning variants against the
Spark `.splat` view using the existing browser audit visual residual metrics.

It is for renderer-quality decisions only. It does not change the default
renderer parameters by itself.

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
