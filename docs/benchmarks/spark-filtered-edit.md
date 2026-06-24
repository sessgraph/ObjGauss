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

RENDER-005T-AB preserves SH rest coefficients for the Spark packed
reconstruction route when the object-aware PLY carries `f_rest_*` data:

```text
object-aware PLY f_dc_* + f_rest_*
        -> Spark-compatible extra.sh1 / extra.sh2 / extra.sh3
        -> base PackedSplats cache
        -> visible index Uint32Array
        -> PackedSplats.extractSplats(...)
        -> extracted SH extra copy
        -> Spark SplatMesh(packedSplats)
```

The route also uses `f_dc_*` as Spark base color when SH rest is preserved, so
object-colored `red/green/blue` debug fields do not pollute source-color
filtered previews.

RENDER-005T-AC adds a matching SH-capable full-view source for SH-heavy scenes.
When the loaded scene has complete `f_rest_*` coefficients, `真实查看` uses the
same Spark PLY packed path instead of the compact `.splat` export:

```text
SH-heavy object-aware PLY
        -> Spark PLY SH source (`spark-ply-sh-source`)
        -> full-view source screenshot
        -> Spark PLY reconstruct probe
        -> same-source residual gate
```

No-SH scenes continue to use the compact `.splat` full view, preserving the
older smoke baseline.

RENDER-005T-AD adds a display `PackedSplats` LRU cache over the filtered
visible-index extracts:

```text
base PackedSplats cache
        -> visible index Uint32Array
        -> visible-index cache key
        -> display PackedSplats LRU cache
        -> Spark SplatMesh(packedSplats)
```

Spark `SplatMesh.dispose()` owns and disposes its supplied `PackedSplats`, so the
filtered viewport detaches cached packed data before disposing the temporary
mesh. This avoids repeating `extractSplats(...)` when the UI returns to a recent
visible object set, while still allowing the LRU cache to release entries when
the base source changes.

RENDER-005T-AE keeps the filtered Spark `SplatMesh` alive for the lifetime of a
base packed source and updates that mesh when the visible object set changes:

```text
base PackedSplats cache
        -> display PackedSplats cache / extract
        -> persistent SplatMesh
        -> SplatMesh packed source update
```

This removes the browser-visible contract that every isolate / delete / restore
state creates a fresh temporary Spark mesh. It still does not patch a native
object mask into Spark's original `.splat` renderer: each new visible set still
needs a display `PackedSplats` object, and the current mesh is pointed at that
display source.

## Current Contract

- Enabled for `renderMode=original` object edit states.
- Disabled for `webgpu-color-mode=sh-view`; SH-view diagnostics continue to use
  WebGPU Tile, while Spark source/original reconstruction now preserves SH rest
  directly when the PLY exposes supported `f_rest_*` coefficients.
- Object-color mode and canvas click selection continue to use the existing
  edit renderer path.
- The filtered Spark route reuses a base `PackedSplats` cache and extracts a
  display `PackedSplats` when visible / removed / isolated object state changes.
- Repeated visible object sets can reuse a display `PackedSplats` from a small
  `visible-index-lru-v1` cache. This is a performance cache for PLY
  reconstruction, not a native object mask in the original `.splat`.
- A filtered Spark session keeps one `SplatMesh` instance alive and updates its
  packed source when the visible set changes. Browser audit requires
  `data-spark-mesh-update-mode="persistent-splatmesh-v1"` and a stable mesh id
  across hide / restore.
- No-SH assets continue to expose `data-spark-reconstruct-source="packed-extract-v1"`
  and `data-spark-sh-rest-preserved="false"`.
- SH-heavy assets expose `data-spark-reconstruct-source="packed-sh-extract-v1"`
  and require preserved count to match source SH Gaussian count.
- In `真实查看`, SH-heavy assets expose
  `data-object-filter="spark-ply-sh-source"` with the same
  `packed-sh-extract-v1` / SH preservation telemetry. Use
  `?spark-ply-source=off` to force the legacy compact `.splat` source for
  diagnostics.

Runtime DOM evidence:

```text
data-renderer="spark-splat"
data-object-filter="spark-ply-sh-source|spark-filtered-ply-reconstruct"
data-spark-filter-mode="ply-source|ply-reconstruct"
data-spark-visible-gaussians="..."
data-spark-removed-objects="..."
data-spark-reconstruct-source="packed-extract-v1"
data-spark-packed-base-gaussians="..."
data-spark-packed-visible-indices="..."
data-spark-packed-base-build-ms="..."
data-spark-packed-extract-ms="..."
data-spark-display-cache-mode="visible-index-lru-v1"
data-spark-display-cache-key="..."
data-spark-display-cache-hit="true|false"
data-spark-display-cache-size="..."
data-spark-display-cache-hits="..."
data-spark-display-cache-misses="..."
data-spark-display-cache-evictions="..."
data-spark-mesh-update-mode="persistent-splatmesh-v1"
data-spark-mesh-id="..."
data-spark-mesh-reused="true|false"
data-spark-mesh-updates="..."
data-spark-sh-rest-source-gaussians="..."
data-spark-sh-rest-preserved-gaussians="..."
data-spark-sh-rest-preserved="true|false"
data-spark-sh-rest-coefficients="0|9|24|45"
data-spark-sh-degree="0|1|2|3"
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
sparkPacked="packed-extract-v1":5696/3909:3.8/0
sparkDisplayCache="visible-index-lru-v1":"true":2:2/2/0
sparkMesh="persistent-splatmesh-v1":1:"true":4
sparkShRest=0:0:"false":0:0
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
packed=5696/5696:4.7/2.4
shRest=0:0:false:0:0
```

SH-heavy trained sample interaction result:

```bash
npm run audit:demo -- \
  --asset nerf-lego-trained-output-local \
  --url http://127.0.0.1:5294/ \
  --no-server
```

```text
browser_audit=passed
postDelete="spark-splat":"spark-filtered-ply-reconstruct":129108
sparkPacked="packed-sh-extract-v1":255794/129108:155.5/0
sparkDisplayCache="visible-index-lru-v1":"true":2:2/2/0
sparkMesh="persistent-splatmesh-v1":1:"true":4
sparkShRest=255794:255794:"true":45:3
```

The trained full-reconstruct diagnostic also proves SH preservation telemetry:

```text
reconstructSource="packed-sh-extract-v1"
visibleGaussians=255794
packed=255794/255794:165/71.9
shRest=255794:255794:true:45:3
```

Before RENDER-005T-AC, that diagnostic failed against the registered `.splat`
source (`coverageRatio=15.599172`, `lumaDelta=0.250533`) because the `.splat`
is ObjGauss' compact export and does not carry the PLY's degree-3 SH rest. The
PLY reconstruction preserved more view-dependent appearance information than
the `.splat` baseline, so the gate needed an SH-capable full-view source.

After RENDER-005T-AC, the same trained residual gate passes with the SH-capable
full-view PLY source:

```bash
node scripts/audit-spark-reconstruct-residual.mjs \
  --assets nerf-lego-trained-output-local \
  --output-dir /tmp/objgauss-spark-reconstruct-residual-trained-ac
```

```text
spark_reconstruct_residual=passed
fullSource="spark-ply-sh-source":"packed-sh-extract-v1":255794:255794:true:45:3
reconstructSource="packed-sh-extract-v1"
coverageRatio=1.170018
lumaDelta=0.058189
chromaDelta=0.007036
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

- Replace reconstruction with a Spark-side object mask over the original packed
  source if Spark exposes a stable native masking surface.
- Avoid creating a temporary `SplatMesh` for every distinct visible object set.
  AD avoids repeated extraction for recently visited states, but each new visible
  set still needs a temporary mesh because Spark does not yet expose a stable
  runtime object-state mask for the original packed source.
- Turn the SH-heavy residual check into a first-class npm script / acceptance
  gate once the trained public sample is considered stable enough for CI/local
  acceptance.
- Add Spark-side selection / raycast-to-object mapping if viewport click
  selection should stay in Spark.
