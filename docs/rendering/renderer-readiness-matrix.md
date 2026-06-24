# Renderer Readiness Matrix

> Status: RENDER-005T-AT decision record
> Last updated: 2026-06-24

This document defines which renderer route ObjGauss should use for commercial
demo, diagnostics, and the long-term WebGPU tile renderer path.

The current rule is:

```text
Commercial demo default: Spark source/original route
Diagnostics / C-path proof: WebGPU Tile route
Fallback / debug: Gaussian OIT edit route
```

This keeps the product-facing demo on the most faithful available splat display
while WebGPU continues to harden toward the C architecture.

## Route Matrix

| Route | Current role | Default surface | Evidence | Known gaps |
| --- | --- | --- | --- | --- |
| Spark native `.splat` | Commercial no-SH default | `真实查看` and source/original object edit on no-SH assets | `acceptance:spark-commercial-route`, `audit:spark-native-mask-gate`, `audit:splat-index-mapping`, `audit:demo` pixel delta | Depends on generated sample index mapping; arbitrary third-party splats need a mapping check or embedded object metadata. |
| Spark PLY SH source / packed filter | Commercial SH-heavy default | `真实查看` and source/original object edit on SH-heavy assets | `acceptance:spark-commercial-route`, `audit:spark-trained-route`, `audit:spark-pick-report` trained route | Not the original compact `.splat`; filtered subsets can look sparse or grainy near hard object boundaries. |
| WebGPU Tile | C-path renderer candidate | Diagnostics, headed desktop audits, CI/headless compute/readback | `acceptance:webgpu-headless`, `audit:webgpu-desktop`, `audit:webgpu-coverage-gate` | Visual residual and coverage still trail Spark; not the commercial default. |
| Gaussian OIT edit fallback | B-path / fallback | Object-color debug and WebGPU-unavailable edit preview | `audit:webgpu-tile-smoke`, fallback contracts in browser audit | Approximate edit preview, not final splat quality. |

## Product Decision

The commercial demo should prefer Spark whenever the user is in source/original
color mode:

```text
真实查看
        -> Spark original `.splat`
        -> Spark PLY SH source for SH-heavy local training outputs

对象编辑 + 原始颜色
        -> Spark native `.splat` mask for no-SH ObjGauss samples
        -> Spark PLY packed SH mask for SH-heavy samples

对象编辑 + 对象色
        -> diagnostic edit renderer
```

The UI term `原始颜色（编辑预览）` means "source color after object filtering".
It does not mean the scene has been retrained, inpainted, or reoptimized after
delete / isolate. After an object is hidden, the remaining Gaussian subset can
still show grain or holes because:

- object assignment is a hard `object_id` mask;
- boundary Gaussians may belong to the removed object;
- hidden objects are not replaced by newly optimized Gaussians;
- SH-heavy samples may use the PLY packed route to preserve SH coefficients.

This is a renderer/data-quality boundary, not a lost-color bug. If the active
renderer label is `WebGPU Tile 编辑` or `Gaussian OIT 编辑`, the user is seeing a
diagnostic approximation rather than the commercial Spark source/original route.

## WebGPU Default Switch Gate

WebGPU should not become the commercial default until all of these are true on
the same candidate configuration:

1. `npm run acceptance:webgpu-headless` passes.
2. `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --probes full` passes.
3. `npm run audit:webgpu-coverage-gate` passes without loosening thresholds.
4. A trained SH-heavy scene has source/sh-view color parity evidence.
5. Object isolate/delete changes GPU object-state and visible pixels on at
   least Lego proxy, Plush semantic, and one trained local scene.
6. The product route can explain or hide remaining source-color graininess
   without implying a full retrained splat.

Until then, WebGPU remains the C-path proof and diagnostic route.

## Required Gates By Question

| Question | Command |
| --- | --- |
| Does WebGPU compute/storage/object-state work in CI/headless? | `npm run acceptance:webgpu-headless` |
| Does WebGPU present to a desktop canvas? | `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --probes full` |
| Is WebGPU visual tuning still within the current baseline? | `npm run audit:webgpu-coverage-gate` |
| Is Spark native object masking safe for generated no-SH samples? | `npm run audit:spark-native-mask-gate` |
| Do compact `.splat` and object-aware PLY preserve Gaussian index mapping? | `npm run audit:splat-index-mapping` |
| Does Spark canvas selection remain usable after delete? | `npm run audit:spark-pick-report` |
| Does the trained SH-heavy route preserve SH and report packed object masking? | `npm run audit:spark-trained-route` |
| Do the no-SH and SH-heavy Spark commercial routes both pass together? | `npm run acceptance:spark-commercial-route` |

## Product UI Contract

The viewer exposes the renderer decision as compact status rather than long
explanatory copy:

- the color selector shows `自身颜色` and `对象色诊断`;
- the viewport shows a compact route badge such as `商用 Spark / 原生 Splat
  mask`;
- the state panel exposes `展示路线`, `颜色用途`, and `预览边界`;
- the app root exposes machine-readable attributes:

```text
data-renderer-route
data-renderer-route-kind
data-color-mode-role
data-source-preview-boundary
data-preview-quality
```

`npm run audit:demo` validates the key route transitions:

```text
initial view     -> commercial / source-color / source-splat
object color     -> diagnostic-object-color
delete preview   -> source-color / hard-object-mask-no-reoptimize
```

## SH-Heavy Route-Only Gate

Use this after `npm run build` when validating the trained local sample route
without running visual residual, multi-click pick report, or mask restore stress:

```bash
npm run audit:spark-trained-sample
npm run audit:spark-trained-route
```

`audit:spark-trained-sample` is the cheap availability contract. It does not
launch a browser. It checks that the local trained sample is registered, that
`public/samples/nerf_lego_trained.splat` and
`public/samples/nerf_lego_trained_objects.ply` exist, and that the PLY exposes
the properties required by the SH-heavy Spark route: geometry, opacity,
scale/rotation, `object_id`, `f_dc_*`, and degree-3 `f_rest_*` coefficients.

The gate loads `nerf-lego-trained-output-local`, checks the initial route, then
deletes one object and validates:

```text
initial route  -> spark-ply-sh-source / commercial / source-splat
delete route   -> spark-packed-sh-mask / commercial / hard-object-mask-no-reoptimize
Spark source   -> ply-packed / packed-sh-extract-v1
SH rest        -> preserved f_rest_* coefficients, degree 3
object mask    -> object-opacity-texture-v1
```

Full visual/residual validation remains an explicit heavier gate.

## Spark Commercial Route Acceptance

Use this as the product route contract before demo or release review:

```bash
npm run acceptance:spark-commercial-route
```

The command runs:

```text
npm run build
npm run audit:spark-trained-sample
npm run audit:spark-native-mask-gate
npm run audit:spark-trained-route
```

By default it writes:

```text
/tmp/objgauss-spark-commercial-route/summary.json
/tmp/objgauss-spark-commercial-route/summary.md
```

Use `--output-dir <path>` to place the report elsewhere. The report summarizes
the route, source, visible Gaussian counts, SH preservation tuple, contract
boundary, and screenshot paths from the underlying browser gates.

Use `--skip-trained-sample-audit` only when the trained sample has already been
checked in the same run. Missing local trained outputs should be fixed by
rebuilding or publishing the sample, not by silently dropping the SH-heavy route
from the product gate.

`acceptance:demo` keeps this gate opt-in because the SH-heavy route depends on
the local trained `nerf-lego-trained-output-local` sample:

```bash
npm run acceptance:demo -- --include-spark-commercial-route
```

`acceptance:demo` uses the built Vite preview server for browser audit by
default; pass `--browser-audit-mode dev` only when dev-server watch behavior is
specifically needed.

The opt-in path also accepts:

```text
--spark-native-port <port>
--spark-trained-port <port>
--spark-route-output-dir <path>
--skip-spark-route-build
--browser-audit-assets <asset_id[,asset_id]>
--skip-browser-visual-residual
```

It proves that:

- no-SH generated samples use Spark native compact `.splat` object masking;
- SH-heavy trained samples preserve degree-3 SH through the packed route;
- both routes expose `source-color` plus `hard-object-mask-no-reoptimize` after delete.

It does not prove deletion inpainting, reoptimization, WebGPU visual fidelity, or
renderer-native arbitrary third-party `.splat` object metadata.

## Next Product Slice

The next UX-facing slice should make the trained SH-heavy sample portability
story explicit before this gate can become a default CI requirement.
