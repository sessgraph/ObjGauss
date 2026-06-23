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

## Current Contract

- Enabled for `renderMode=original` object edit states.
- Disabled for `webgpu-color-mode=sh-view`; SH-view diagnostics continue to use
  WebGPU Tile because the current Spark reconstruction path pushes RGB/SH-DC
  colors and does not preserve full SH rest coefficients.
- Object-color mode and canvas click selection continue to use the existing
  edit renderer path.
- The filtered Spark route rebuilds a `SplatMesh` when visible / removed /
  isolated object state changes.

Runtime DOM evidence:

```text
data-renderer="spark-splat"
data-object-filter="spark-filtered-ply-reconstruct"
data-spark-filter-mode="ply-reconstruct"
data-spark-visible-gaussians="..."
data-spark-removed-objects="..."
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
renderModeAfterDelete="原始颜色（编辑预览）"
visibleAfterDelete=3,909
deletedObjects=1
```

This proves the delete preview can now return to a Spark renderer path for the
remaining original-color scene instead of staying on the approximate edit
renderer.

## Remaining Gaps

- Preserve full SH rest / view-dependent color in the Spark reconstructed path.
- Add a visual residual gate comparing full `.splat` Spark against
  PLY-reconstructed Spark before enabling this route for trained SH-heavy
  samples by default.
- Add Spark-side selection / raycast-to-object mapping if viewport click
  selection should stay in Spark.
- Avoid full SplatMesh rebuilds for high-frequency edits; this is acceptable for
  delete / isolate preview but not for continuous brushing.
