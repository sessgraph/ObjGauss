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

RENDER-005T-AF moves filtered object visibility from per-state display
`PackedSplats.extractSplats(...)` into a Spark `objectModifier` opacity mask over
the stable base `PackedSplats` source:

```text
base PackedSplats cache
        -> Uint32 object visibility DataTexture
        -> Spark Dyno objectModifier
        -> opacity=0 for hidden Gaussian indices
        -> persistent SplatMesh
```

This removes the browser-visible display extract cost for object-state changes:
`data-spark-packed-extract-ms="0.000"`, display cache telemetry is disabled, and
the object mask texture exposes visible / hidden Gaussian counts. It is still
not an object mask inside the original compact `.splat` file; it is a Spark
shader mask over the object-aware PLY-derived packed source.

RENDER-005T-AG adds a pixel-delta guard for the Spark opacity mask route. On
small scenes, browser audit now captures the Spark canvas after delete, after
hiding one remaining object, and after restoring that object:

```text
delete-state Spark canvas
        -> hide one non-deleted object through object opacity mask
        -> visual checksum / coverage / luma / chroma must change
        -> restore object
        -> visual stats must return to the delete-state baseline
```

This proves the mask affects rendered pixels, not only DOM telemetry.

RENDER-005T-AH adds a standalone index-mapping audit for the generated public
sample pairs:

```text
compact .splat rows
        -> position / scale by Gaussian index
object-aware PLY vertices
        -> position / scale / object_id by vertex index
        -> count + per-index delta + rounded-position multiset check
```

This proves the current public/generated samples preserve a stable Gaussian
index between the compact `.splat` source and the object-aware PLY. It is
evidence for using an external object-id mask keyed by Gaussian index on these
assets; it does not mean arbitrary third-party `.splat` files carry object ids
internally.

## Current Contract

- Enabled for `renderMode=original` object edit states.
- Disabled for `webgpu-color-mode=sh-view`; SH-view diagnostics continue to use
  WebGPU Tile, while Spark source/original reconstruction now preserves SH rest
  directly when the PLY exposes supported `f_rest_*` coefficients.
- Object-color mode and canvas click selection continue to use the existing
  edit renderer path.
- The filtered Spark route reuses one base `PackedSplats` cache and a
  `object-opacity-texture-v1` mask texture when visible / removed / isolated
  object state changes.
- Display `PackedSplats.extractSplats(...)` is disabled for object-state
  changes; `data-spark-packed-extract-ms` must stay `0.000`.
- A filtered Spark session keeps one `SplatMesh` instance alive and marks it for
  update when the object mask changes. Browser audit requires
  `data-spark-mesh-update-mode="persistent-splatmesh-v1"` and, for small scenes,
  a stable mesh id across hide / restore.
- Small-scene browser audit also requires
  `spark-object-mask-visual-delta-v1`: hiding a remaining object must change the
  Spark canvas checksum / visual metrics, and restoring it must return to the
  delete-state baseline.
- No-SH assets continue to expose `data-spark-reconstruct-source="packed-extract-v1"`
  and `data-spark-sh-rest-preserved="false"`.
- SH-heavy assets expose `data-spark-reconstruct-source="packed-sh-extract-v1"`
  and require preserved count to match source SH Gaussian count.
- In `真实查看`, SH-heavy assets expose
  `data-object-filter="spark-ply-sh-source"` with the same
  `packed-sh-extract-v1` / SH preservation telemetry. Use
  `?spark-ply-source=off` to force the legacy compact `.splat` source for
  diagnostics.
- Public/generated compact `.splat` and object-aware PLY pairs must pass
  `npm run audit:splat-index-mapping` before relying on Gaussian index keyed
  masks for original-source/native-mask work.

Runtime DOM evidence:

```text
data-renderer="spark-splat"
data-object-filter="spark-ply-sh-source|spark-object-opacity-mask"
data-spark-filter-mode="ply-source|ply-reconstruct"
data-spark-visible-gaussians="..."
data-spark-removed-objects="..."
data-spark-reconstruct-source="packed-extract-v1"
data-spark-packed-base-gaussians="..."
data-spark-packed-visible-indices="..."
data-spark-packed-base-build-ms="..."
data-spark-packed-extract-ms="0.000"
data-spark-display-cache-mode="disabled-by-native-mask-v1"
data-spark-display-cache-key=""
data-spark-display-cache-hit="false"
data-spark-display-cache-size="0"
data-spark-display-cache-hits="0"
data-spark-display-cache-misses="0"
data-spark-display-cache-evictions="0"
data-spark-object-mask-mode="object-opacity-texture-v1"
data-spark-object-mask-size="..."
data-spark-object-mask-updates="..."
data-spark-object-mask-visible-gaussians="..."
data-spark-object-mask-hidden-gaussians="..."
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
postDelete="spark-splat":"spark-object-opacity-mask":3909
sparkPacked="packed-extract-v1":5696/3909:4.4/0
sparkDisplayCache="disabled-by-native-mask-v1":"false":0:0/0/0
sparkObjectMask="object-opacity-texture-v1":"4096x2":3909/1787:4
sparkMaskVisual="spark-object-mask-visual-delta-v1":"4a2ed0e8"/"be002ca4"/"4a2ed0e8":0.000752/0.014063/0.026019:0/0/0
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
coverageRatio=1.177112
lumaDelta=0.03071
chromaDelta=0.027503
objectFilter="spark-ply-reconstruct"
reconstructSource="packed-extract-v1"
visibleGaussians=5696
filteredGaussians=0
packed=5696/5696:4.6/0
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
postDelete="spark-splat":"spark-object-opacity-mask":129108
sparkPacked="packed-sh-extract-v1":255794/129108:155.9/0
sparkDisplayCache="disabled-by-native-mask-v1":"false":0:0/0/0
sparkObjectMask="object-opacity-texture-v1":"4096x63":129108/126686:2
sparkMaskVisual="not-run":""/""/"":0/0/0:0/0/0
sparkMesh="persistent-splatmesh-v1":1:"true":2
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

Run the index-mapping audit for public/generated `.splat` / PLY pairs:

```bash
npm run audit:splat-index-mapping
```

Current local result:

```text
splat_index_mapping=passed
assets=["plush-3dgs-local","plush-v1-closure-local","plush-semantic-closure-local","nerf-lego-alpha-closure-local","nerf-lego-trained-output-local"]
```

All five pairs preserve count and index order exactly:

```text
plush-3dgs-local: 281498/281498, maxPositionDelta=0, maxScaleDelta=0
plush-v1-closure-local: 281498/281498, maxPositionDelta=0, maxScaleDelta=0
plush-semantic-closure-local: 281498/281498, maxPositionDelta=0, maxScaleDelta=0
nerf-lego-alpha-closure-local: 5696/5696, maxPositionDelta=0, maxScaleDelta=0
nerf-lego-trained-output-local: 255794/255794, maxPositionDelta=0, maxScaleDelta=0
```

## Remaining Gaps

- The current mask is over the object-aware PLY-derived Spark packed source, not
  the original compact `.splat` file. Index mapping is now proven for current
  generated public samples, but original-source/native-mask runtime wiring is
  still separate work.
- Prototype an original-source/native mask route that uses the proven index
  mapping, then repeat the existing object-mask pixel-delta and residual gates.
- Turn the SH-heavy residual check into a first-class npm script / acceptance
  gate once the trained public sample is considered stable enough for CI/local
  acceptance.
- Add Spark-side selection / raycast-to-object mapping if viewport click
  selection should stay in Spark.
