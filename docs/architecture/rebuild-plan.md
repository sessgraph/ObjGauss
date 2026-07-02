# ObjGauss Rebuild Plan

> Status: planning baseline
> Last updated: 2026-07-02

This document resets the project framing before the next implementation pass.
It is intentionally a planning artifact: it defines boundaries, core
algorithms, target file structure, and migration order. It does not claim the
repository already follows the target layout.

## Product Split

ObjGauss should be organized around three layers.

### 1. Frontend: experience and rendering

The frontend is the product experience layer. Its job is to let a user load,
inspect, compare, and interact with 3D Gaussian assets.

Frontend responsibilities:

- Render original 3DGS appearance through Spark / splat renderer paths.
- Preserve and continue iterating ObjGauss-owned Gaussian rendering algorithms,
  including Gaussian OIT edit preview, WebGPU tile / compute renderer paths,
  object-state renderer buffers, picking support, and renderer-side
  acceleration / diagnostics.
- Provide fallback point / Gaussian preview when native splat features are not
  enough.
- Support object-level interactions: select, hide, isolate, delete preview,
  switch original color vs object color, inspect object stats.
- Expose clear loading state, route telemetry, render capability telemetry, and
  browser performance evidence.
- Consume stable model artifacts and manifests from the backend or local asset
  registry.

Frontend non-responsibilities:

- It should not train 3D models.
- It should not own backend object segmentation, semantic labeling, training,
  asset conversion, or model publication algorithms.
- It should not own model conversion, compression, or asset publication policy.
- It should not parse large research datasets unless the backend has produced a
  browser-ready artifact.

### 2. Backend: model and pipeline provider

The backend layer provides model artifacts and processing workflows. In the
current repository this is mostly CLI scripts and Python modules, but the target
architecture should allow a service boundary later.

Backend responsibilities:

- Register, validate, and serve 3D Gaussian assets.
- Convert external outputs such as PLY / `.splat` / training directories into
  ObjGauss model artifacts.
- Run object-aware processing pipelines over Gaussian clouds.
- Run or orchestrate training jobs through explicit pipelines and manifests.
- Produce browser-ready artifacts: small demo samples, splats, object-aware
  PLYs, compressed / chunked formats, model manifests, and QA summaries.
- Keep training assets, generated outputs, and public demo assets separated.

Backend non-responsibilities:

- It should not own React UI state.
- It should not depend on browser renderer internals.
- It should not silently publish large or license-unclear assets into
  `public/samples/`.

### 3. Core algorithms: pure, testable model logic

The core algorithms should become the stable center of the repository. They
should be callable by CLI, backend service jobs, notebooks, and tests without
pulling in frontend code or process-specific scripts.

Core algorithm requirements:

- Deterministic by default where random seeds are accepted.
- Data-in/data-out APIs, with filesystem access kept at IO boundaries.
- Small contracts that can be tested independently.
- No browser assumptions.
- No mandatory heavy ML dependencies; SAM / CLIP / torch remain optional
  adapters.

## Current Core Algorithm Inventory

These are the current modules that should be preserved and extracted first.

| Domain | Current files | Core responsibility |
| --- | --- | --- |
| Gaussian data model | `objgauss/gaussians.py` | Structured Gaussian vertex table and field checks. |
| Gaussian IO | `objgauss/ply.py`, `objgauss/splat.py` | Read/write scalar PLY and `.splat` assets. |
| Feature extraction | `objgauss/features.py` | Build position/color/opacity feature vectors. |
| Baseline clustering | `objgauss/clustering.py`, `objgauss/segment.py` | KMeans labels, `object_id` attachment, object colors, object filters. |
| Object Field | `objgauss/object_field.py` | Soft object-slot logits, probabilities, hard label export, metrics. |
| 2D mask manifests | `objgauss/masks.py` | Build, split, validate, and normalize mask manifests. |
| Projection voting | `objgauss/mask_voting.py` | Project Gaussians into frames, vote masks to Gaussians, depth visibility diagnostics, projection loss training. |
| Cross-view semantics | `objgauss/semantic_slots.py`, `objgauss/clip_scoring.py` | Align 2D slots across views, score mask crops, gate semantic naming quality. |
| Evaluation policy | `objgauss/baseline_comparison.py`, `objgauss/emergence.py`, `objgauss/emergence_benchmark.py` | Compare candidates, measure object emergence, block promotion when evidence is weak. |
| Training output handoff | `objgauss/training.py`, `objgauss/sample_bundle.py` | Register external trained outputs and write reproducible handoff manifests. |
| Asset ingestion | `objgauss/assets.py`, `objgauss/mesh_nerf.py` | Pull input assets and create NeRF-style render sets. |

The current frontend algorithm-like files should be preserved as renderer
algorithms, but kept separate from backend training / model-processing
algorithms:

| Domain | Current files | Target role |
| --- | --- | --- |
| Browser Gaussian parsing | `src/ply.js` | Frontend artifact decoder. It may share a formal schema with backend IO but should stay optimized for browser loading. |
| Browser compressed Gaussian parsing | `src/ogcDecoder.js` | Frontend quantized OGC decoder contract. It decodes chunk-local quantized records into renderer-compatible points and preserves object / chunk / LOD metadata without replacing renderer kernels. |
| Renderer paths | `src/SplatViewport.jsx`, `src/PointCloudViewport.jsx`, `src/WebGpuTileViewport.jsx` | Viewer renderer implementations. |
| ObjGauss frontend rendering algorithms | `src/webgpuTileSmoke.js`, `src/webgpuTileStorage.js`, `src/webgpuTileComputeShader.js`, `src/webgpuTileResolveShader.js`, `src/webgpuTextureResolveShader.js`, OIT code in `src/PointCloudViewport.jsx` | Must be preserved and iterated as frontend renderer kernel code. These are not training algorithms, but they are still first-class project algorithms. |
| Spark interaction support | `src/SplatViewport.jsx`, `src/sparkObjectMask.js`, `src/sparkPackedSh.js` | Renderer-side object masks, packed SH handling, picking probes, and native-splat integration. |
| Interaction state | `src/App.jsx`, `src/modelCatalog.js` | Default Three.js world shell, in-world model catalog, object selection and drag state. |
| Legacy asset registry | `src/assetLibrary.js`, `src/modelArtifactManifest.js` | Pipeline / compatibility asset metadata and manifest route derivation for historical audits and handoff scripts; not the default world viewer entrypoint. |

## Target Repository Structure

The target structure should make the product boundary obvious:

```text
apps/
  viewer/
    src/
      app/                 # React shell, state, routes, panels
      assets/              # frontend asset catalog adapter
      renderers/           # Spark, Three, WebGPU renderer paths
      interactions/        # picking, object visibility, delete/isolate preview
      decoders/            # browser PLY / splat / future OGC decoders
      telemetry/           # browser route and performance telemetry
    public/
    package.json

packages/
  objgauss-core/
    objgauss_core/
      gaussian/            # GaussianCloud, field schemas, typed tables
      io/                  # PLY, splat, future compressed Gaussian IO
      features/            # feature extraction
      objects/             # clustering, labels, object colors
      object_field/        # soft slots, metrics, projection loss
      masks/               # manifest schema, validation, builders
      projection/          # camera projection, visibility, mask voting
      semantics/           # slot alignment, CLIP scoring adapters
      evaluation/          # emergence and promotion policy
    tests/

services/
  model-api/
    objgauss_model_api/
      registry/            # model registry and artifact lookup
      jobs/                # job records and pipeline execution hooks
      api/                 # future HTTP/RPC boundary

pipelines/
  training/
    splatfacto/            # wrappers around external trainers
    object_processing/     # mask voting, Object Field training recipes
    publishing/            # sample bundles, public demo export

tools/
  audits/                  # browser, renderer, performance, release checks
  benchmarks/              # benchmark runners
  maintenance/             # one-off migration and data scripts

docs/
  architecture/
  adr/
  state/
  training/
  rendering/
```

This is a target layout. The first implementation should avoid a giant move.
Use compatibility imports and move one domain at a time.

Frontend renderer algorithms can later be split into a separate package if the
viewer grows:

```text
packages/
  objgauss-renderer/
    src/
      gaussian-oit/
      webgpu-tile/
      spark-bridge/
      picking/
      telemetry/
```

That split is optional. The important rule is that renderer algorithms remain
owned by the frontend/rendering layer and are not deleted during backend/core
cleanup.

## Minimal First Extraction

The first code extraction should focus on the algorithm kernel only:

```text
objgauss/
  core/
    gaussian.py        # from gaussians.py
    io_ply.py          # from ply.py
    io_splat.py        # from splat.py
    features.py        # from features.py
    clustering.py      # from clustering.py
    objects.py         # from segment.py
    object_field.py    # from object_field.py
    masks.py           # manifest schema and validation subset
    projection.py      # project_points and depth visibility
    mask_voting.py     # vote_masks_to_gaussians and training from votes
    slots.py           # align_mask_manifest_slots
    evaluation.py      # promotion / emergence metrics
```

Keep the old module paths as thin compatibility wrappers during migration:

```text
objgauss/ply.py             -> imports from objgauss.core.io_ply
objgauss/object_field.py    -> imports from objgauss.core.object_field
objgauss/mask_voting.py     -> imports from objgauss.core.mask_voting
```

The CLI should call core APIs through service/pipeline adapters, not contain
business logic.

## Core Data Contracts

The rebuild should stabilize these contracts before major UI work resumes.

对象状态方向已经冻结在
`docs/architecture/objgauss-v1-kernel-contract.md`。该文档将 v1 kernel 定义为
`PerceptionEvidence -> ObjectState -> GaussianToken`，其中 `ObjectState` 是唯一
reasoning unit；temporal / dynamics 字段只能保存在 object state 或 state history
中。`docs/myobjgausstoken/` 下的原始 token-system 讨论是 research input，不是
architecture contract。

### Gaussian artifact

Required fields:

- `x`, `y`, `z`
- color source: `red`/`green`/`blue` or 3DGS `f_dc_*`
- opacity source: `opacity` or splat alpha equivalent
- optional scale / rotation fields
- optional `object_id`

Target output:

- structured Gaussian table
- field schema metadata
- source format metadata
- count and byte-size metadata

### Object assignment

v1 分阶段实现记录在
`docs/architecture/objgauss-v1-object-emergence-plan.md`。核心规则是：slot assignment、
clustering 和 tracking 都是同一个 assignment matrix `A` 上的约束或视图，不是三套互相
竞争的 object emergence 系统。

Required concepts:

- `object_id`: hard label used by viewer and export formats.
- `ObjectField.logits`: soft `N x K` slot assignment used by training and
  diagnostics.
- `unknown_object_id`: optional label used when confidence is too low.

Target output:

- hard object labels
- soft probabilities
- per-object counts and quality metrics
- deterministic color mapping for inspection

### Mask manifest

Required concepts:

- frame image path and camera intrinsics/extrinsics
- mask path, slot, bbox, confidence
- optional CLIP / semantic scores
- source root and path rewrite behavior

Target output:

- validated manifest
- normalized relative paths
- reproducible mask counts and slot summaries

### Training run manifest

Required concepts:

- external trainer identity and command
- dataset source and license boundary
- input Gaussian artifact
- object processing recipe
- output artifacts and hashes
- QA gates and known limitations

Target output:

- backend-readable model record
- frontend-readable asset record
- reproducible audit evidence

### Model artifact manifest

Schema: `objgauss-model-artifact-manifest-v1`.

This is the backend-to-frontend contract for model delivery. It is currently a
static JSON/file contract, not an HTTP API. The contract is implemented in
`objgauss/model_manifest.py`.

Required top-level fields:

- `manifest_id`
- `asset_id`
- `name`
- `stage`
- `source`
- `license`
- `counts.gaussians`
- `counts.objects`
- `artifacts`
- `quality_evidence`
- `limitations`
- `created_from`

Artifact roles:

- `quick_splat`: browser quick-view `.splat` artifact.
- `object_edit`: browser object-edit artifact, usually object-aware PLY or a
  future chunked format.
- `diagnostic_full`: full diagnostic artifact. The frontend must not request
  this by default.
- `source_gaussian`: raw or normalized source Gaussian artifact used by backend
  processing.
- `object_field`: Object Field training artifact.
- `training_summary`: training / projection-loss summary.
- `quality_report`: audit / benchmark / route evidence.
- `compressed_chunked`: future browser-ready chunked Gaussian delivery.

Delivery tiers:

- `browser_quick`
- `browser_edit`
- `diagnostic`
- `training_internal`
- `quality_evidence`

Safety rule:

- Only `browser_quick` and `browser_edit` are browser-ready tiers.
- `diagnostic_full` and `source_gaussian` must not be marked
  `browser_ready=true`.
- A viewer-facing manifest must contain at least one `browser_ready` artifact.
- A full 4.5M / 1GB+ PLY belongs in `diagnostic_full`, not in a default viewer
  route.

Chunked / compressed browser artifact contract:

- `role="compressed_chunked"` is the reserved path for Object-aware Gaussian
  Codec style assets such as `.ogc`.
- A `browser_ready=true` chunked artifact must use `delivery_tier` of
  `browser_quick` or `browser_edit`.
- A browser-ready chunked artifact must include artifact-level
  `gaussian_count`, `object_count`, `byte_size`, and `sha256`.
- It must include `chunk_index` with schema `objgauss-chunk-index-v1`, `path`,
  positive `chunk_count`, and `sort_key` such as `object_id+morton_xyz`.
- It must include `compression` metadata with `codec`, `version`, and
  `layout`; quantization / VQ / SH policy fields can be added under that
  object without changing the outer manifest schema.
- It must include `lod.levels`, where every level records `level` and
  `gaussian_count`.
- It must include `object_id_coverage` with `has_object_ids=true`, `field`,
  `mode`, and `object_count`. This is required because ObjGauss object editing
  depends on stable object identifiers.
- This contract does not implement the codec. It makes the backend / viewer
  handoff explicit before OGC encoding, streaming, and WebGPU renderer support
  are implemented.

Chunk index generator:

- `objgauss/core/chunk_index.py` implements the pure metadata generator for
  `objgauss-chunk-index-v1`.
- `build_chunk_index(...)` accepts a `GaussianCloud` with `x`, `y`, `z`, and
  `object_id`, sorts by `object_id+morton_xyz`, and emits chunk metadata without
  writing OGC binary payloads.
- Chunks do not cross object boundaries. Each chunk records `chunk_id`,
  `object_id`, `gaussian_count`, sorted index range, AABB, center, radius, and
  deterministic per-chunk LOD metadata.
- The returned `sorted_indices` can be consumed by a later OGC binary writer;
  the JSON index stays metadata-sized and does not include per-Gaussian indices
  unless explicitly requested for small diagnostics.
- `validate_chunk_index(...)`, `write_chunk_index(...)`, and
  `read_chunk_index(...)` provide schema and consistency gates.

Object-aware LOD metadata:

- `objgauss/core/lod.py` implements `objgauss-object-aware-lod-v1` metadata.
- Default levels follow the OGC / OGR plan: LOD0 full, LOD1 50%, LOD2 20%,
  LOD3 5% preview.
- LOD selection is deterministic and object-aware: every positive level keeps
  at least one Gaussian per object, then assigns retained records as chunk-local
  prefixes under the existing `object_id+morton_xyz` ordering.
- The top-level index, every object summary, and every chunk record all carry
  LOD level summaries. Level Gaussian counts must be non-increasing.
- This is metadata only. It does not prune, quantize, VQ, entropy-code, or
  change the frontend renderer. It gives the later OGR/WebGPU loader stable
  chunk ranges for progressive loading.

Chunk-local quantization and prototype writer:

- `objgauss/core/quantization.py` implements
  `objgauss-local-quantization-v1` metadata and
  `objgauss-quantization-estimate-v1` size estimates.
- The current policy is `chunk-aabb-uint16-rgb8-opacity8-v0`.
- It records and can write a chunk-local payload layout where `xyz` is
  quantized as chunk AABB `uint16 x3`, RGB as `uint8 x3`, opacity as `uint8`,
  and `object_id` is stored once in chunk metadata because chunks do not cross
  object boundaries.
- The deterministic estimator compares the current raw OGC record payload size
  against the estimated quantized payload size. JSON index metadata is not
  included in that byte estimate.
- `write_quantized_ogc_payload(...)` writes the first actual quantized `.ogc`
  prototype and `read_quantized_ogc_payload(...)` is a diagnostic reader that
  dequantizes records for bounded-error tests.
- The quantized prototype preserves object ids, chunk boundaries, sort key, and
  LOD metadata. It intentionally does not implement VQ, adaptive SH, entropy
  coding, or a browser decoder yet.

Minimal OGC payload writer:

- `objgauss/core/ogc_payload.py` implements the first object-aware chunked
  payload prototype.
- `write_ogc_payload(...)` uses `build_chunk_index(...)`, writes a raw `.ogc`
  payload as concatenated fixed-size chunk records, and annotates each chunk
  with `byte_offset`, `byte_length`, `record_count`, and `record_format`.
- It also annotates each chunk LOD level with byte offsets and byte lengths for
  prefix-record reads.
- The prototype record format is `objgauss-ogc-record-v0`: `x`, `y`, `z`,
  `red`, `green`, `blue`, `opacity`, and `object_id`.
- It attaches quantization metadata and an estimated quantized payload size to
  the index and compression metadata, while the current `.ogc` bytes remain raw
  fixed-size records.
- It preserves object ids and chunk byte ranges, but it intentionally does not
  implement pruning, quantization, VQ, adaptive SH, entropy coding, or a browser
  streaming loader yet.
- `read_ogc_payload(...)` is a test / diagnostic reader for roundtrip checks,
  not the frontend runtime path.
- `build_compressed_chunked_artifact(...)` in `objgauss/model_manifest.py`
  attaches a `.ogc` payload and `.index.json` to the model artifact manifest as
  `role="compressed_chunked"`, deriving Gaussian/object counts, byte size,
  SHA-256, chunk index summary, compression metadata, LOD metadata, and object id
  coverage from the OGC index.

Current adapters:

- `manifest_from_training_output(...)`: derives a model artifact manifest from
  existing `training-output-manifest.json`.
- `manifest_from_sample_bundle(...)`: derives a model artifact manifest from
  existing sample bundles.
- `manifest_from_asset_library_entry(...)`: maps frontend asset library entry
  dictionaries to the backend artifact contract. Deferred or large object PLYs,
  such as the near-1M full PLY, are mapped to `diagnostic_full` and
  `browser_ready=false`.
- `build_compressed_chunked_artifact(...)`: binds backend-generated OGC payloads
  to the same manifest contract without changing the frontend loader route.
- `decodeQuantizedOgcPayload(...)` / `decodeQuantizedOgcChunk(...)` in
  `src/ogcDecoder.js`: define the browser decoder contract for
  `objgauss-ogc-quantized-payload-v0`, returning renderer-compatible point
  records while preserving object, chunk, and LOD metadata. This is a loader
  contract only; Gaussian OIT, WebGPU tile / compute, Spark bridge, shaders,
  object-state buffers, and picking remain frontend renderer algorithms.

## Requirement Rebuild

### Frontend MVP requirements

1. Load a model asset from a manifest or local asset catalog.
2. Show original 3DGS appearance when a splat artifact is available.
3. Show object-aware edit preview when an object-aware artifact is available.
4. Support object select, hide, isolate, and delete preview.
5. Display object count, Gaussian count, visible count, route, renderer, and
   load state.
6. Never default to a huge diagnostic artifact when a browser-ready artifact is
   available.
7. Keep training controls and model processing out of the first-screen UI.

### Backend MVP requirements

1. Register external 3DGS training output as a model artifact.
2. Convert or normalize Gaussian artifacts into ObjGauss contracts.
3. Run object processing algorithms over a Gaussian cloud.
4. Produce object-aware artifacts for the viewer.
5. Produce manifest records with source, license, command, hashes, counts, and
   limitations.
6. Provide a stable path for future compressed / chunked model delivery.

### Core algorithm MVP requirements

1. Read/write Gaussian tables.
2. Extract stable features.
3. Attach hard `object_id` labels.
4. Initialize and train Object Field from masks.
5. Project Gaussians into camera frames.
6. Vote 2D masks to Gaussian slots with optional depth visibility.
7. Align cross-view slots and evaluate semantic naming quality.
8. Produce metrics that decide whether an output can be promoted.

## Migration Plan

### ARCH-REBUILD-001: Planning baseline

- Create this document.
- Record that frontend/backend/core responsibilities are being reset.
- No source movement.

### CORE-EXTRACT-001: Extract pure algorithm kernel

- Move Gaussian data, IO, feature extraction, clustering, Object Field, mask
  projection, and slot alignment into a core namespace.
- Keep compatibility wrappers at old imports.
- Validation: full Python tests and selected CLI smoke commands.

### BACKEND-CONTRACT-001: Define model artifact manifest

- Introduce a stable model manifest schema.
- Map existing `training-output-manifest.json`, asset library entries, and
  sample bundles to this schema.
- Validation: manifest fixture tests and asset list smoke.

### VIEWER-BOUNDARY-001: Separate viewer app from algorithms

- Move frontend code under `apps/viewer/` or establish that target with build
  aliases.
- Keep viewer consuming manifests and artifacts, not training modules.
- Preserve ObjGauss-owned renderer algorithms as viewer/renderer kernel code:
  Gaussian OIT, WebGPU tile renderer, object-state buffers, shader code,
  picking, and Spark bridge logic.
- Validation: `npm run build`, renderer route audit, browser screenshot.

### PIPELINE-BOUNDARY-001: Move training and audits into pipelines/tools

- Separate external trainer orchestration from algorithm library code.
- Move browser audits and benchmarks out of product app source.
- Validation: existing audit commands still pass or have explicit renamed
  aliases.

### MODEL-DELIVERY-001: Browser-ready model delivery

- Add explicit delivery tiers: quick splat, object edit artifact, diagnostic
  full artifact, and future compressed/chunked artifact.
- Prevent accidental huge model loading from the default UI.
- Validation: route audit proves only the intended artifact is requested.

## Decisions To Keep Pending

- Whether to introduce a real HTTP model service now or keep CLI + static files
  until the artifact contract is stable.
- Whether the final Python import name should be `objgauss_core` or
  `objgauss.core`.
- Whether compressed/chunked Gaussian delivery is implemented before or after
  the viewer app move.
- Whether training orchestration remains Node wrappers around external tools or
  becomes Python pipeline commands.

## Immediate Next Step

Start with `CORE-EXTRACT-001`. Do not move the frontend first. The frontend is
visible and easy to break; extracting pure core algorithms first gives the
backend and future viewer a stable contract to share.
