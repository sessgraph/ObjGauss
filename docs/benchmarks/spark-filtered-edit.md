# Spark Filtered Edit Preview

> Status: current
> Last updated: 2026-06-24

This benchmark records the first Spark-rendered object filter path for
ObjGauss edit previews.

## Goal

The user-facing problem was that delete / isolate previews returned to
`原始颜色（编辑预览）` but still looked like the approximate edit renderer, not
the real Spark `.splat` view.

RENDER-005T-Y introduces an intermediate route:

```text
object-aware PLY points + object_id
        -> filtered visible Gaussian set
        -> Spark SplatMesh constructSplats / pushSplat
        -> Spark-rendered delete / isolate preview
```

The original unedited view still loads the asset `.splat` directly. When object
editing is active and the color mode is source/original, the app can reconstruct
a filtered Spark `SplatMesh` from the object-aware PLY instead of falling back
to the WebGPU / Gaussian OIT edit renderer.

RENDER-005T-Z adds a direct residual gate for the same reconstruction path:

```text
full .splat Spark render
        -> screenshot coverage / luma / chroma
object-aware PLY reconstructed Spark render
        -> screenshot coverage / luma / chroma
        -> residual gate
```

RENDER-005T-AA changes the construction surface from per-state raw PLY
`constructSplats` loops to a reusable Spark `PackedSplats` base plus
`extractSplats` for each visible object set:

```text
object-aware PLY points
        -> base PackedSplats cache
        -> visible index Uint32Array
        -> PackedSplats.extractSplats(...)
        -> Spark SplatMesh(packedSplats)
```

This is still a PLY reconstruction path, not a native object mask inside the
original `.splat`, but it removes the raw rebuild contract from browser-visible
runtime behavior and exposes timing / SH-preservation facts.

## Current Contract

- Enabled for `renderMode=original` object edit states.
- Disabled for `webgpu-color-mode=sh-view`; SH-view diagnostics continue to use
  WebGPU Tile because the current Spark reconstruction path pushes RGB/SH-DC
  colors and does not preserve full SH rest coefficients.
- Object-color mode and canvas click selection continue to use the existing
  edit renderer path.
- The filtered Spark route reuses a base `PackedSplats` cache and extracts a
  display `PackedSplats` when visible / removed / isolated object state changes.
- The current `PackedSplats.extractSplats` route does not preserve full SH rest
  coefficients. This is exposed as `data-spark-sh-rest-preserved="false"`.

Runtime DOM evidence:

```text
data-renderer="spark-splat"
data-object-filter="spark-filtered-ply-reconstruct"
data-spark-filter-mode="ply-reconstruct"
data-spark-visible-gaussians="..."
data-spark-removed-objects="..."
data-spark-reconstruct-source="packed-extract-v1"
data-spark-packed-base-gaussians="..."
data-spark-packed-visible-indices="..."
data-spark-packed-base-build-ms="..."
data-spark-packed-extract-ms="..."
data-spark-sh-rest-preserved="false"
```

The full-scene reconstruction probe is intentionally URL-gated so normal UI
behavior is unchanged:

```text
?spark-reconstruct-probe=1
data-object-filter="spark-ply-reconstruct"
data-spark-filtered-gaussians="0"
```

## Validation

Use a built static preview to avoid dev-server file watcher limits:

```bash
npm run build
npm run preview -- --port 5294 --strictPort
npm run audit:demo -- \
  --asset nerf-lego-alpha-closure-local \
  --url http://127.0.0.1:5294/ \
  --no-server
```

Current local result:

```text
browser_audit=passed
postDelete="spark-splat":"spark-filtered-ply-reconstruct":3909
sparkPacked="packed-extract-v1":5696/3909:3.9/1.9
sparkShRest=0:"false"
renderModeAfterDelete="原始颜色（编辑预览）"
visibleAfterDelete=3,909
deletedObjects=1
```

This proves the delete preview can now return to a Spark renderer path for the
remaining original-color scene instead of staying on the approximate edit
renderer.

Run the direct full `.splat` versus PLY-reconstructed Spark residual gate:

```bash
npm run build
npm run audit:spark-reconstruct-residual
```

Current default local result:

```text
spark_reconstruct_residual=passed
asset="nerf-lego-alpha-closure-local"
coverageRatio=1.170841
lumaDelta=0.029762
chromaDelta=0.028407
objectFilter="spark-ply-reconstruct"
reconstructSource="packed-extract-v1"
visibleGaussians=5696
filteredGaussians=0
packed=5696/5696:4.2/2.7
shRest=0:false
```

The optional multiscene check includes Plush semantic and is slower because it
reconstructs 281k Gaussians through Spark:

```bash
npm run audit:spark-reconstruct-residual-multiscene
```

Current Plush local result:

```text
asset="plush-semantic-closure-local"
coverageRatio=1.303149
lumaDelta=0.049406
chromaDelta=0.002846
visibleGaussians=281498
```

## Remaining Gaps

- Preserve full SH rest / view-dependent color in the Spark reconstructed path,
  or replace reconstruction with a Spark-side object mask over the original
  packed `.splat` source.
- Add a stricter SH-preserving residual gate for trained SH-heavy samples.
- Add Spark-side selection / raycast-to-object mapping if viewport click
  selection should stay in Spark.
- Avoid per-edit extracted `SplatMesh` allocation for high-frequency brushing;
  the packed extract route is a safer intermediate layer, but native object
  masks remain the terminal architecture.
