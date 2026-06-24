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
| Spark native `.splat` | Commercial no-SH default | `真实查看` and source/original object edit on no-SH assets | `audit:spark-native-mask-gate`, `audit:splat-index-mapping`, `audit:demo` pixel delta | Depends on generated sample index mapping; arbitrary third-party splats need a mapping check or embedded object metadata. |
| Spark PLY SH source / packed filter | Commercial SH-heavy default | `真实查看` and source/original object edit on SH-heavy assets | `audit:demo` trained SH route, `audit:spark-pick-report` trained route | Not the original compact `.splat`; filtered subsets can look sparse or grainy near hard object boundaries. |
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

## Next Product Slice

The next UX-facing slice should make the active route and quality boundary
visible without adding explanatory clutter:

- keep commercial demo assets on Spark source/original routes by default;
- demote `对象色` to an explicit diagnostic/debug mode;
- show the active renderer route in compact status text;
- avoid implying that source/original delete preview is a full 3DGS
  reoptimization.
