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

RENDER-005T-AI adds a URL-gated original-source mask prototype over Spark's
native compact `.splat` loader:

```text
compact .splat source
        + object-aware PLY object_id keyed by proven Gaussian index
        + object-opacity-texture-v1
        -> SplatMesh({ url, objectModifier })
```

This route keeps the Spark source as the compact `.splat` asset and applies the
same opacity mask through `objectModifier`. It initially shipped behind
`?spark-native-mask=on` so native-mask behavior could be audited separately.

RENDER-005T-AJ adds a dedicated native-mask multi-scene gate:

```text
Lego proxy native .splat mask
        -> DOM contract
        -> full audit hide / restore pixel delta

Plush semantic native .splat mask
        -> DOM contract
        -> large-scene screenshot evidence
```

The gate deliberately avoids Spark/edit visual residual screenshots for Plush.
That residual belongs to the packed reconstruction quality gate, while the
native-mask gate only needs to prove original compact `.splat` source,
object-mask telemetry, and persistent mesh update. Small-scene pixel effect is
covered by `audit-demo`.

RENDER-005T-AK promotes native compact `.splat` masking to the automatic default
for no-SH source/original object edit previews:

```text
no-SH sample + object edit active
        -> native compact .splat source
        -> object-opacity-texture-v1

SH-heavy sample + object edit active
        -> PLY packed SH source
        -> object-opacity-texture-v1
```

This keeps `nerf-lego-trained-output-local` on the SH-preserving packed route,
while Lego proxy and Plush semantic use the original compact `.splat` source by
default. `spark-object-source=packed` / `spark-native-mask=off` remain the
diagnostic escape hatch, and `spark-native-mask=on` still forces native mode.

## Current Contract

- Enabled for `renderMode=original` object edit states.
- Disabled for `webgpu-color-mode=sh-view`; SH-view diagnostics continue to use
  WebGPU Tile, while Spark source/original reconstruction now preserves SH rest
  directly when the PLY exposes supported `f_rest_*` coefficients.
- Object-color mode continues to use the existing edit renderer path.
- Source/original Spark filtered edit supports browser-audited canvas selection
  through `screen-space-object-pick-v1`: the viewport projects visible
  object-aware PLY Gaussians through the active Spark camera and selects the
  nearest visible `object_id`.
- Spark selection exposes hit quality telemetry and a non-interactive selection
  marker. `audit-demo` requires a hit, matching selected object, finite
  distance within the pick radius, at least one candidate object, and a visible
  marker. It records ambiguity instead of treating current demo clicks as
  unambiguous.
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
- No-SH assets default to `data-spark-mask-source="native-splat"` and
  `data-spark-reconstruct-source="native-splat-source-v1"` for source/original
  object edit states.
- SH-heavy assets default to `data-spark-mask-source="ply-packed"` and
  `data-spark-reconstruct-source="packed-sh-extract-v1"` so preserved count can
  match source SH Gaussian count.
- In `真实查看`, SH-heavy assets expose
  `data-object-filter="spark-ply-sh-source"` with the same
  `packed-sh-extract-v1` / SH preservation telemetry. Use
  `?spark-ply-source=off` to force the legacy compact `.splat` source for
  diagnostics.
- Public/generated compact `.splat` and object-aware PLY pairs must pass
  `npm run audit:splat-index-mapping` before relying on Gaussian index keyed
  masks for original-source/native-mask work.
- `spark-object-source=packed` or `spark-native-mask=off` switches
  source/original object edit states back to the PLY-derived packed source for
  diagnostics.
- `spark-native-mask=on` forces Spark's native compact `.splat` source even for
  SH-heavy scenes; this is a diagnostic override and can lose SH-rest fidelity.
- `npm run audit:spark-native-mask-gate` must pass for the default native
  candidate. The gate covers Lego + Plush native source / mask contract, while
  `audit-demo` covers the Lego pixel delta.

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
data-spark-mask-source="ply-packed|native-splat"
data-spark-selection-mode="screen-space-object-pick-v1|none"
data-spark-selected-object="..."
data-spark-pick-status="idle|hit|miss"
data-spark-pick-object="..."
data-spark-pick-distance-px="..."
data-spark-pick-candidate-objects="..."
data-spark-pick-ambiguous="true|false"
data-spark-selected-marker-visible="true|false"
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
sparkMaskSource="native-splat"
sparkPacked="native-splat-source-v1":5696/3909:0/0
sparkDisplayCache="disabled-by-native-mask-v1":"false":0:0/0/0
sparkObjectMask="object-opacity-texture-v1":"4096x2":3909/1787:4
sparkCanvasSelectedObject=0
sparkPick="screen-space-object-pick-v1":"hit":"0":3.7:3:"true":"true"
sparkMaskVisual="spark-object-mask-visual-delta-v1":"839479b7"/"6ef6c73f"/"839479b7":0.000786/0.013558/0.000507:0/0/0
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

Run the URL-gated native compact `.splat` mask audit:

```bash
npm run build
npm run preview -- --host 127.0.0.1 --port 5302 --strictPort
npm run audit:demo -- \
  --asset nerf-lego-alpha-closure-local \
  --url http://127.0.0.1:5302/ \
  --no-server \
  --spark-native-mask
```

Current local result:

```text
browser_audit=passed
postDelete="spark-splat":"spark-object-opacity-mask":3909
sparkMaskSource="native-splat"
sparkPacked="native-splat-source-v1":5696/3909:0/0
sparkDisplayCache="disabled-by-native-mask-v1":"false":0:0/0/0
sparkObjectMask="object-opacity-texture-v1":"4096x2":3909/1787:4
sparkMaskVisual="spark-object-mask-visual-delta-v1":"839479b7"/"6ef6c73f"/"839479b7":0.000786/0.013558/0.000507:0/0/0
sparkMesh="persistent-splatmesh-v1":1:"true":4
sparkShRest=0:0:"false":0:0
```

The packed-source diagnostic route can still be forced with
`?spark-object-source=packed`.

Run the native-mask multi-scene gate:

```bash
npm run audit:spark-native-mask-gate
```

Current local result:

```text
native_mask_gate=passed
assets=["nerf-lego-alpha-closure-local","plush-semantic-closure-local"]

nerf-lego-alpha-closure-local:
source="native-splat"
route="native-splat-source-v1"
visible=4960/5696
visual="skipped-contract-gate-v1":0/0/0:0/0/0

plush-semantic-closure-local:
source="native-splat"
route="native-splat-source-v1"
visible=104403/281498
visual="skipped-contract-gate-v1":0/0/0:0/0/0
```

SH-heavy trained sample default route check:

```bash
npm run audit:demo -- \
  --assets nerf-lego-trained-output-local \
  --skip-visual-residual \
  --url http://127.0.0.1:5312/ \
  --no-server
```

```text
browser_audit=passed
sparkMaskSource="ply-packed"
sparkCanvasSelectedObject=3
sparkPick="screen-space-object-pick-v1":"hit":"3":0.892:3:"false":"true"
sparkPacked="packed-sh-extract-v1":255794/129108:156.5/0
sparkShRest=255794:255794:"true":45:3
```

Run the multi-click Spark pick report:

```bash
npm run audit:spark-pick-report
```

Current default local result:

```text
spark_pick_report=passed
asset="nerf-lego-alpha-closure-local"
clicks=15
hits=14
hitRate=0.933333
ambiguousHits=5
ambiguityRate=0.357143
markerHits=14/14
pickStrategy="object-support-score-v1"
scoreMargin=0.171357/0.011
maskSource="native-splat"
route="native-splat-source-v1"
summaryJson="/tmp/objgauss-spark-pick-report/summary.json"
summaryMd="/tmp/objgauss-spark-pick-report/summary.md"
```

Before RENDER-005T-AO, the same deterministic click set reported
`ambiguityRate=0.928571`. The current report gates ambiguity at `<=0.5`, so
this is now a regression check instead of report-only telemetry.

The trained SH-heavy route is intentionally explicit because it loads a 255k
Gaussian local sample:

```bash
npm run audit:spark-pick-report -- \
  --assets nerf-lego-trained-output-local \
  --max-clicks 5 \
  --output-dir /tmp/objgauss-spark-pick-report-trained \
  --port 5316
```

Current trained local result:

```text
spark_pick_report=passed
asset="nerf-lego-trained-output-local"
clicks=5
hits=5
hitRate=1
ambiguousHits=1
ambiguityRate=0.2
markerHits=5/5
pickStrategy="object-support-score-v1"
scoreMargin=0.2034/0.067
distinctHitObjects=["1","2","3"]
maskSource="ply-packed"
route="packed-sh-extract-v1"
summaryJson="/tmp/objgauss-spark-pick-report-trained/summary.json"
summaryMd="/tmp/objgauss-spark-pick-report-trained/summary.md"
```

Before RENDER-005T-AO, the trained 5-click route reported
`ambiguityRate=1`.

Interpretation: the screen-space pick path still uses object-aware PLY metadata
rather than Spark-internal raycast, but the `object-support-score-v1`
disambiguation rule keeps hit-rate stable while reducing ambiguity enough to
gate it. The score mixes nearest distance, local object support and front-depth
priority; remaining ambiguous clicks are true close-boundary cases.

## Remaining Gaps

- The original compact `.splat` mask is now the no-SH default, but SH-heavy
  scenes still need the PLY packed route for SH fidelity.
- The native mask relies on ObjGauss-generated sample pairs that pass the index
  mapping gate. Arbitrary third-party `.splat` files still need an explicit
  mapping check or embedded object metadata before object masking is trusted.
- Spark canvas selection is currently a screen-space CPU pick over the
  object-aware PLY metadata, not a Spark-internal raycast. It is good enough for
  demo interaction and now exposes hit / ambiguity telemetry plus a selection
  marker. `object-support-score-v1` reduces ambiguity to `0.357143` on Lego
  proxy and `0.2` on trained Lego 5-click, but this is still a CPU
  screen-space pick. A clearer hover/confirm UX or Spark-internal ray/object
  metadata path is needed before claiming robust renderer-native picking.
- Turn the SH-heavy residual check into a first-class npm script / acceptance
  gate once the trained public sample is considered stable enough for CI/local
  acceptance.
