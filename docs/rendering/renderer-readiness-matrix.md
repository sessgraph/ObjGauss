# Renderer Readiness Matrix

> Status: RENDER-005T-AT decision record
> Last updated: 2026-06-25

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
| Spark native `.splat` | Commercial no-SH default | `真实查看` and source/original object edit on no-SH assets | `audit:renderer-route-contract`, `acceptance:spark-commercial-route`, `audit:spark-native-mask-gate`, `audit:splat-index-mapping`, `audit:demo` pixel delta | Depends on generated sample index mapping; arbitrary third-party splats need a mapping check or embedded object metadata. |
| Spark PLY SH source / packed filter | Commercial SH-heavy default | `真实查看` and source/original object edit on SH-heavy assets | `audit:renderer-route-contract`, `acceptance:spark-commercial-route`, `audit:spark-trained-route`, `audit:spark-pick-report` trained route | Not the original compact `.splat`; filtered subsets can look sparse or grainy near hard object boundaries. |
| WebGPU Tile | C-path renderer candidate | Diagnostics, headed desktop audits, CI/headless compute/readback | `audit:renderer-route-contract`, `audit:webgpu-scale-budget`, `audit:webgpu-edit-cost-budget`, `audit:webgpu-tile-smoke`, `acceptance:webgpu-headless`, `audit:webgpu-desktop`, `audit:webgpu-coverage-gate` | Visual residual and coverage still trail Spark; not the commercial default. Scale / edit-cost budgets prove storage and update shape, not 1M FPS. Object-state-filtered tile list and objectState-only incremental upload are in place for compatible edit updates. |
| Gaussian OIT edit fallback | B-path / fallback | Object-color debug and WebGPU-unavailable edit preview | `audit:renderer-route-contract`, `audit:webgpu-tile-smoke`, fallback contracts in browser audit | Approximate edit preview, not final splat quality. |

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
The product UI exposes this boundary as `预览边界=硬 mask，无补洞` and
`删除结果=源色 mask 预览`, backed by `data-source-preview-result`.

`spark-object-mask-feather=on` and the Web UI `柔化删除边界` checkbox are the
first diagnostic paths for reducing hard mask edge speckle without retraining:
they keep hidden-object opacity at zero but use spatial-neighbor feathering to
lower opacity on visible Gaussians near the hidden boundary. It is not the
default commercial route yet; it is a tunable candidate that must be compared
across scenes before promotion.

`npm run audit:spark-mask-feather-sweep` is that comparison gate. It runs a
route-only browser flow over Lego proxy and Plush semantic by default, compares
`hard:off` with `feather55:0.55`, and writes:

```text
/tmp/objgauss-spark-mask-feather-sweep/summary.json
/tmp/objgauss-spark-mask-feather-sweep/summary.md
```

Current sweep result: `feather55` softens boundary opacity on both default
scenes, but it does not improve coverage ratio, and only Plush shows a small
luma improvement. Therefore feather remains diagnostic / candidate behavior
rather than the default route.

Use the candidate gate before considering any default change:

```bash
npm run audit:spark-mask-feather-candidates
```

It covers Lego proxy, Plush semantic, and the local Poly Haven Chair commercial
sample when available, comparing `hard`, `feather55`, `feather70`, and
`feather55r035`. A candidate is only promotable if coverage, luma, and chroma
do not regress on any included scene. Current result: all feather variants are
`diagnostic-only`; best score is `feather55r035`, but it still increases max
coverage ratio by `0.010209`, so the default remains hard mask.

Pass `--control ui` when the gate should click the Web UI checkbox instead of
using URL params:

```bash
npm run audit:spark-mask-feather-sweep -- --control ui --skip-visual-stats
```

Object-boundary remap is the second diagnostic path for reducing hard-mask
grain. Unlike feathering, it changes only the sampled `object_id` assignment in
a `/tmp` preview PLY and leaves all Gaussian geometry, opacity, color, scale,
rotation, and SH fields untouched.

Use the browser residual gate before considering a cleaned PLY promotion:

```bash
npm run audit:object-boundary-remap-residual
```

Local dev, preview, and remap browser audits use the fixed default port `5395`.
Do not rotate ad hoc ports between runs; if `5395` is occupied, stop the
occupying local dev/preview/browser-audit process and rerun on `5395`.

The gate first runs `audit:object-boundary-remap-preview`, then uploads both
the original object-aware PLY and the remap-preview PLY into the viewer. It
forces the same PLY-packed Spark object-mask route for both files with
`spark-object-source=packed`, deletes the top remap-candidate object, captures
canvas visual stats, and writes:

```text
/tmp/objgauss-object-boundary-remap-residual/summary.json
/tmp/objgauss-object-boundary-remap-residual/summary.md
```

Current default result: Lego proxy, Plush semantic, and Poly Haven Chair all
pass the browser route and residual threshold checks, but the aggregate
recommendation is still `do-not-promote-default-hard-mask`, not promotion. The
remap previews hide `49` / `2786` / `29` fewer target-object Gaussians and keep
after-delete residual max at `0.000784/0.004332/0.019990`. This is evidence that
the sampled remap preview is not obviously harmful on the tested top candidate
object per scene; it does not prove every high-risk object pair is safe.

Use the explicit top-N target sweep to check more than one high-risk target per
scene:

```bash
npm run audit:object-boundary-remap-target-sweep
```

Current top-2 result: all 6 target cases pass route/residual thresholds, but
only 1/6 is promotable. Lego target `3` and Plush target `0` hide more
Gaussians after remap (`+397` and `+4085` hidden delta), so remap cleanup cannot
be promoted as a global default. It needs target-level review or an allowlist.

Use the decision policy gate when a downstream export or QA step needs a
machine-readable answer for each target:

```bash
npm run audit:object-boundary-remap-policy
```

The policy report writes:

```text
/tmp/objgauss-object-boundary-remap-policy/remap-decision-policy.json
/tmp/objgauss-object-boundary-remap-policy/remap-decision-policy.md
```

The policy mode is `object-boundary-remap-decision-policy-v1`. Its default
action is `keep-hard-mask`, and its apply mode is
`manual-target-allowlist-only`. This intentionally separates three cases:

- `allowlist-candidate`: target evidence is strong enough for explicit manual
  review before any sample update.
- `deny-*`: browser residual or hidden-Gaussian evidence says remap is risky.
- `review-only`: route is valid, but evidence is not strong enough to apply.

Current route-only smoke evidence across Lego proxy, Plush semantic, and Poly
Haven Chair classifies 0 targets as allowlist candidates, 2 as risky
`deny-hidden-increase`, and 4 as `review-only`; global recommendation remains
`do-not-apply-remap-globally`.

Use the policy-gated export when a downstream candidate PLY must consume that
decision policy:

```bash
npm run audit:object-boundary-remap-policy-export
```

Policy-gated export applies remaps only when both conditions are true:

1. the target is a policy `allowlist-candidate`;
2. the target is present in the reviewed allowlist manifest
   `docs/rendering/object-boundary-remap-reviewed-allowlist.json`.

Inline `--allow-target asset_id:object_id` remains available for diagnostic
smoke tests, but production candidate export should use the reviewed manifest.
Without a reviewed allowlist entry, or when the requested target is `deny-*` /
`review-only`, the export writes a valid preview PLY but leaves those
`object_id` assignments unchanged. Current three-scene evidence uses an empty
reviewed manifest and has raw candidates=`10012`, applied remaps=`0`, and
blocked remaps=`10012`; a negative smoke with
`--allow-target nerf-lego-alpha-closure-local:3` also keeps applied remaps at
`0` because that target is `deny-hidden-increase`.

Use the reviewed allowlist fixture to prove the positive path without approving
real samples:

```bash
npm run audit:object-boundary-remap-reviewed-allowlist
```

The fixture writes a temporary synthetic policy and reviewed allowlist under
`/tmp`, then requires applied remaps to be greater than zero. Current fixture
evidence on Lego target `2`: applied=`402`, blocked=`741`.

Before adding any real target to the reviewed allowlist, follow the manual
review runbook:

```text
docs/rendering/object-boundary-remap-review-runbook.md
```

The committed allowlist manifest must pass:

```bash
npm run audit:object-boundary-remap-reviewed-allowlist-manifest
```

That gate keeps the manifest empty by default, but if an approved target is
present it requires reviewer metadata, owner approval, `allowlist-candidate`
policy evidence, hidden-delta / residual numbers, and durable
repository-relative report and screenshot paths.

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
| What should default CI run for renderer readiness? | `npm run acceptance:renderer-ci` |
| What should demo / release review run for the full product route? | `npm run acceptance:renderer-product` |
| Is the B -> C renderer route contract still intact? | `npm run audit:renderer-route-contract` |
| Does the WebGPU C-path storage layout fit 100k-1M budgets? | `npm run audit:webgpu-scale-budget` |
| Do object edits avoid full static re-upload inside 100k-1M budgets? | `npm run audit:webgpu-edit-cost-budget` |
| Does WebGPU compute/storage/object-state work in CI/headless? | `npm run acceptance:webgpu-headless` |
| Does WebGPU present to a desktop canvas? | `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --probes full` |
| Is WebGPU visual tuning still within the current baseline? | `npm run audit:webgpu-coverage-gate` |
| Is Spark native object masking safe for generated no-SH samples? | `npm run audit:spark-native-mask-gate` |
| Do compact `.splat` and object-aware PLY preserve Gaussian index mapping? | `npm run audit:splat-index-mapping` |
| Why can source/original delete preview still look grainy? | `npm run audit:object-mask-boundary` |
| Which object boundaries are candidates for assignment cleanup / remap review? | `npm run audit:object-boundary-cleanup` |
| Can we export a cleaned `object_id` preview without changing official samples? | `npm run audit:object-boundary-remap-preview` |
| Does a remap preview preserve browser visual residual after delete? | `npm run audit:object-boundary-remap-residual` |
| Do multiple high-risk remap targets preserve residual before promotion? | `npm run audit:object-boundary-remap-target-sweep` |
| Which remap targets are allowlist candidates, risky, or review-only? | `npm run audit:object-boundary-remap-policy` |
| Does remap export obey the target-level decision policy? | `npm run audit:object-boundary-remap-policy-export` |
| Is the reviewed remap allowlist complete enough for human-approved targets? | `npm run audit:object-boundary-remap-reviewed-allowlist-manifest` |
| Does the reviewed allowlist positive path actually apply remaps? | `npm run audit:object-boundary-remap-reviewed-allowlist` |
| Can hard-mask boundary risk be explained with route and residual artifacts? | `npm run audit:hard-mask-quality` |
| Does Spark source-color object masking have a soft-boundary diagnostic path? | `npm run audit:spark-mask-feather` |
| Does the soft-boundary candidate improve multiple scenes enough to promote? | `npm run audit:spark-mask-feather-sweep` |
| Does Spark canvas selection remain usable after delete? | `npm run audit:spark-pick-report` |
| Does the trained SH-heavy route preserve SH and report packed object masking? | `npm run audit:spark-trained-route` |
| Do the no-SH and SH-heavy Spark commercial routes both pass together? | `npm run acceptance:spark-commercial-route` |

## Renderer Acceptance Profiles

`npm run acceptance:renderer-ci` is the default CI renderer profile. It is
designed to be fresh-clone safe and therefore does not require
`nerf-lego-trained-output-local`:

```text
npm run audit:renderer-route-contract
npm run build
npm run audit:webgpu-tile-smoke
npm run audit:webgpu-scale-budget
npm run audit:webgpu-edit-cost-budget
npm run audit:splat-index-mapping -- --assets nerf-lego-alpha-closure-local,plush-semantic-closure-local
npm run audit:spark-native-mask-gate
```

This profile proves the B/C renderer contracts that can be expected from the
repo's public no-SH samples: route architecture, WebGPU tile smoke, 100k-1M
storage and edit-cost budgets, compact `.splat` / PLY index mapping, and Spark
native object masking. It intentionally does not prove the trained SH-heavy
packed route.

`npm run audit:webgpu-scale-budget` is the storage budget gate for the C-path
scale target. It estimates the full 11-buffer runtime layout for 100k, 300k,
and 1M Gaussian profiles under a default desktop-style budget of `128 MiB` max
storage buffer binding, `256 MiB` total runtime storage, and 12 storage buffers
per shader stage. The current 1M budget row uses a `320px` internal viewport
and 32 compact tile references per Gaussian, producing max buffer `122.07 MiB`
and total storage `173.24 MiB`. This is a scale architecture check, not a
browser FPS or visual-quality proof.

`npm run audit:webgpu-edit-cost-budget` is the object-edit cost companion gate.
It uses the same 100k / 300k / 1M synthetic C-path profiles and checks:

- full first-upload storage remains within the configured budget,
- objectState-only edit upload remains a tiny buffer write,
- object-state upload share stays below the configured threshold,
- pixel resolve workgroups and tile-entry candidate scan upper bounds remain
  within explicit budgets.

The current 1M row uses 256 object slots, a `320px` viewport, and 32 compact
tile references per Gaussian. It reports a `4 KiB` object-state edit upload
versus `173.24 MiB` full storage, with an `8.192G` pixel candidate-check upper
bound. The default total workgroup budget is `4352`, which covers the 512px
interactive row's `4096` pixel workgroups plus tile accumulation / resolve
dispatches. This is an update-shape and cost-envelope check, not an FPS proof.

`npm run audit:webgpu-tile-smoke` also covers the C-path edit-state contract.
The default tile list mode remains `visible-only`, but the WebGPU runtime uses
`object-state-filtered`: the compact tile list can contain all Gaussians while
the shader skips hidden objects through `objectState`. The smoke gate verifies
that isolate/delete changes keep `tileCounts`, `tileOffsets`, and `tileEntries`
stable while changing `objectStateChecksum` and the resolved output checksum.
On top of that contract, the runtime now uses an objectState-only incremental
upload path when the storage reuse signature is compatible: static inputs
including Gaussian geometry, colors, scales, object indices, tile counts,
tile offsets, and tile entries match, while the object-state buffer changes.
In that case the existing storage bundle is reused, only `objectState` receives
`queue.writeBuffer`, and compute is dispatched again. If geometry, color mode,
viewport, point size, tile layout, or other static inputs change, the runtime
falls back to rebuilding the full storage bundle.

The browser-visible contract is:

```text
data-webgpu-storage-update-mode="full-upload" | "object-state-only"
data-webgpu-storage-update-ms="<milliseconds>"
data-webgpu-storage-object-state-byte-size="<bytes>"
data-webgpu-frame-submit-ms="<milliseconds>"
data-webgpu-queue-done-ms="<milliseconds>"
```

`audit:demo` requires WebGPU isolate transitions to report `object-state-only`
with a nonzero object-state byte size and finite update / submit / queue timing.
Delete transitions may report either `object-state-only` or `full-upload`: the
full upload fallback is expected when deletion also changes static inputs such
as source/object color buffers. In both cases timing must be browser-visible.
These timings are runtime observability signals, not FPS guarantees.

`npm run audit:spark-pick-report` validates Spark canvas object selection after
delete. The current product contract is `screen-space-object-pick-v1` with
`hover-confirm-v1`: hover exposes a candidate marker and click confirms that
candidate as the selected object. The report uses fixed local preview port
`5395` by default. Passing proves the screen-space CPU pick over object-aware
PLY metadata is usable; it does not prove Spark-internal ray/object metadata
picking.

`npm run acceptance:renderer-product` is the explicit product/demo profile. It
runs `audit:renderer-route-contract` first, then
`acceptance:spark-commercial-route`, including the trained sample availability
preflight and the SH-heavy browser route. This is the correct gate before
commercial demo review, but it should not be promoted to default CI until the
trained sample becomes a committed, downloadable, or generated fixture in that
environment.

## Product UI Contract

The viewer exposes the renderer decision as compact status rather than long
explanatory copy:

- the color selector shows `自身颜色` and `对象色诊断`;
- the viewport shows a compact route badge such as `商用 Spark / 原生 Splat
  mask`;
- the state panel exposes `展示路线`, `颜色用途`, `预览边界`, and
  `质量解释`;
- the scene panel exposes `柔化删除边界` as an explicit diagnostic toggle;
- the app root exposes machine-readable attributes:

```text
data-renderer-route
data-renderer-route-kind
data-color-mode-role
data-source-preview-boundary
data-preview-quality
data-hard-mask-quality-interpretation
data-hard-mask-quality-source
data-hard-mask-gap-score
data-hard-mask-residual-coverage-ratio
data-hard-mask-deleted-object
data-spark-object-mask-feather-control
data-spark-object-mask-feather-enabled
data-spark-object-mask-feather-opacity
data-spark-object-mask-feather-radius
```

`npm run audit:demo` validates the key route transitions:

```text
initial view     -> commercial / source-color / source-splat / source-splat quality
object color     -> diagnostic-object-color / diagnostic quality
delete preview   -> source-color / hard-object-mask-no-reoptimize / hard-mask quality
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

## Hard Mask Boundary Diagnostic

Use this when the source/original route is correct but delete / isolate preview
still looks sparse, grainy, or rough near object boundaries:

```bash
npm run audit:object-mask-boundary
```

The diagnostic is browser-free. It reads object-aware PLY files and estimates
the hard-mask visual gap from:

- `deletedSubsetCoverageRatio`: how much projected footprint belongs to the
  object being removed;
- `uniqueCoverageLossRatio`: projected footprint that only the removed object
  covers, which is the no-reoptimize hole risk;
- `sharedBoundaryCoverageRatio`: projected footprint where the removed object
  overlaps other object IDs, which is boundary-grain risk;
- `neighborBoundaryRatio`: 3D local neighborhoods that contain another
  `object_id`, which is assignment-mixing risk.

The default report writes:

```text
/tmp/objgauss-object-mask-boundary/summary.json
/tmp/objgauss-object-mask-boundary/summary.md
```

This does not replace Spark / WebGPU visual residual screenshots. It explains
whether the likely source of grain is coverage loss, hard boundary mixing, or
deleted-subset sparsity before tuning shader footprint or alpha presentation.

Use the cleanup candidate wrapper when the next question is not "why is it
grainy?" but "which object assignments should we inspect or remap first?":

```bash
npm run audit:object-boundary-cleanup
```

This writes:

```text
/tmp/objgauss-object-boundary-cleanup/summary.json
/tmp/objgauss-object-boundary-cleanup/summary.md
```

The cleanup layer is still read-only. It reports
`object-boundary-cleanup-candidate-v1` fields such as
`cleanupCandidateGaussianEstimate`, `cleanupDominantTargetObject`,
`cleanupPriorityScore`, and recommendation. The next step is a cleaned
`object_id` preview / browser residual gate; do not promote cleanup into the
default commercial route without that visual proof.

The preview export step is:

```bash
npm run audit:object-boundary-remap-preview
```

It writes a sampled cleaned PLY to:

```text
/tmp/objgauss-object-boundary-remap-preview/*.remap-preview.ply
/tmp/objgauss-object-boundary-remap-preview/summary.json
/tmp/objgauss-object-boundary-remap-preview/summary.md
```

The exporter preserves the source PLY bytes and all Gaussian properties, then
patches only sampled `object_id` values whose local 3D neighborhood is dominated
by another object id. Treat this as an experiment input for the browser residual
gate, not as a promoted object assignment cleanup policy.

## Hard Mask Quality Chain

Use this after producing boundary, Spark route, and Spark reconstruct residual
artifacts:

```bash
npm run audit:object-mask-boundary
npm run acceptance:spark-commercial-route
npm run audit:spark-reconstruct-residual-multiscene
npm run audit:hard-mask-quality
```

`audit:hard-mask-quality` does not launch a browser by itself. It reads existing
summary JSON files, aligns them by asset id, infers the deleted object from the
Spark object-mask hidden Gaussian count, and writes:

```text
/tmp/objgauss-hard-mask-quality/summary.json
/tmp/objgauss-hard-mask-quality/summary.md
```

The output distinguishes:

- `boundary-mixing-dominant`: route and residual evidence exist, coverage holes
  are low, but PLY object boundaries are highly shared or locally mixed;
- `coverage-hole-risk`: the deleted object owns enough unique projected
  footprint that hard masking can create visible holes;
- `browser-residual-dominant`: Spark reconstruction / source mismatch is larger
  than the hard-mask boundary proxy, so renderer/source parity is the immediate
  blocker.

The product route status consumes these explanations directly. Report-backed
samples expose `boundary-mixing-dominant`, `coverage-hole-risk`, or
`browser-residual-dominant`; samples without a matching quality-chain row expose
`hard-mask-quality-unmeasured` instead of inventing a diagnosis.

## Commercial Demo Readiness

Use this after `acceptance:spark-commercial-route` and `audit:hard-mask-quality`
have produced artifacts:

```bash
npm run audit:commercial-demo-readiness
```

The report writes:

```text
/tmp/objgauss-commercial-demo-readiness/summary.json
/tmp/objgauss-commercial-demo-readiness/summary.md
```

This command separates two concerns that should not be mixed in product copy:

- **Product route readiness**: whether the Spark source/original route, object
  mask, screenshot evidence, and hard-mask quality explanation are good enough
  for demo review.
- **Public commercial asset eligibility**: whether the underlying source license
  is clean enough for a public commercial sample.

Current local result:

| Tier | Assets | Product decision |
| --- | --- | --- |
| `商业展示路线可演示` | `nerf-lego-alpha-closure-local`, `plush-semantic-closure-local`, `polyhaven-chair-commercial-demo-local` | Route can be shown, but UI/copy must keep `对象 mask，无补洞 / 边界混合主导` visible. |
| `研究 / 诊断样例` | `nerf-lego-trained-output-local` | Do not present as commercial default while `browser-residual-dominant` remains. |
| `待 route QA` | `plush-3dgs-local`, `plush-v1-closure-local` | Needs route and hard-mask quality evidence before demo claims. |

The current local report with the chair route artifacts has
`publicCommercialCandidateRows=1`. The generated chair sample is the first
public-commercial candidate, but it is still a hard-mask / no-reoptimize object
edit preview rather than an inpainted or retrained post-delete splat.
