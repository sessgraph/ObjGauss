# ObjGauss Training Data And Teacher Model Notes

> Status: research input / planning aid
> Last updated: 2026-07-09
> Source: Owner-provided pasted research notes on ObjGauss training data
> sources and teacher models.

This document organizes the attached research notes into a project-facing
reference. It is not an architecture contract, not an asset registry entry, not
a downloaded dataset record, and not proof that any teacher model or external
dataset is already supported by ObjGauss.

Authoritative boundaries remain:

- Kernel contract: `docs/architecture/objgauss-v1-kernel-contract.md`
- Object emergence plan:
  `docs/architecture/objgauss-v1-object-emergence-plan.md`
- State variable gate:
  `docs/architecture/objectstate-state-variable-gate.md`
- Current public/controlled dataset candidate audit:
  `docs/training/objectstate-public-dataset-candidates.md`
- License and asset status: `docs/asset-library.md`

## 1. Project Interpretation

The useful interpretation from the research notes is:

```text
image / video / Gaussian cloud
  -> object separation
  -> object identity
  -> ObjectState
  -> cross-frame stability
  -> prediction / intervention evidence
```

This must be mapped onto the current ObjGauss contract:

```text
PerceptionEvidence -> assignment A[N,K] -> ObjectState[K] -> GaussianArtifact
```

Rules for future work:

- `ObjectState` remains the reasoning unit.
- `object_id` remains a renderer/export address derived from assignment and
  matching; it is not physical identity ground truth.
- Teacher outputs are weak supervision, not ground truth.
- Heavy teachers such as SAM 2, DINOv2, CoTracker, TAPIR, VGGT, MASt3R,
  SAGA or LangSplat must stay optional adapters unless a later ADR or standard
  PR explicitly changes dependency policy.
- Dataset and model license claims in the source notes must be re-audited
  against official sources before download, redistribution, public demo, HF
  release, or commercial wording.

## 2. Data Source Layers

No single public dataset covers object-centric Gaussian reconstruction,
identity persistence, prediction and action-conditioned intervention. Treat
training data as layered evidence.

| Layer | Candidate sources from notes | Useful signal | ObjGauss fit | Boundary |
| --- | --- | --- | --- | --- |
| Static / multi-view 3D reconstruction | Mip-NeRF 360, Tanks and Temples, Deep Blending / Blender synthetic, self-captured image sets | Images, camera poses, sparse points, novel-view reconstruction | Good for renderer, Splatfacto, Gaussian reconstruction, and geometry smoke | Not object identity GT; cannot pass State Variable Gate by itself |
| Object-centric 3D | uCO3D, CO3D / CO3Dv2, OmniObject3D, Objaverse / Objaverse-XL | Object videos or assets, cameras, depth/point clouds, masks or renderable assets depending on source | Best broad source family for object-level pretraining and cross-view identity | License lineage and per-sample fields must be audited before ingestion |
| Video object segmentation | YouTube-VOS, DAVIS, SAM 2 / SA-V style sources | Frame masks, object tubes, temporal mask propagation | Useful for 2D mask and identity consistency supervision | Usually lacks Gaussian evidence and 6DoF object pose |
| Point tracking / dynamic geometry | PointOdyssey, TAP-Vid, Dynamic Replica, Kubric / MOVi | Tracks, depth, masks, flow, synthetic controlled motion | Useful for occlusion recovery, long-term identity, and dynamic geometry | Synthetic/domain bias must be kept visible; not automatically public-demo safe |
| Action / interaction | ManiSkill2, RLBench, DROID, Open X-Embodiment, HOT3D / DexYCB-style clips | Action labels, state transitions, hand/object interaction | Relevant only after identity and prediction gates are reviewable | Needed for intervention rows, but does not prove randomized counterfactuals by itself |

Current repository reality:

- The active Phase 1 evidence route already prioritizes controlled/public rows
  through capture manifests, real evidence bundles, BOP adapters and public
  interaction workspace audits.
- `docs/training/objectstate-public-dataset-candidates.md` is the current
  focused public dataset audit for BOP / HOT3D / DexYCB-style evidence rows.
- The broader sources above are future training and pretraining candidates.
  They should not bypass the existing State Variable Gate.

## 3. Phase Framing

The attached notes recommend a broad v1/v2/v3 ladder. Adapt it to the current
repo state as follows.

### Near-Term: Evidence And Source Audit

Goal:

```text
make future data ingestion reviewable before downloading or training
```

Recommended slices:

- Audit uCO3D / CO3D official fields, size, access path and license lineage.
- Define how a source sequence maps into existing controlled capture or real
  evidence bundle schemas.
- Define a teacher output sidecar for masks, features, tracks, depth and
  confidence without making those teachers required dependencies.
- Keep all downloaded data and generated teacher outputs under ignored
  `outputs/`.

### v1 Candidate: Object Assignment To ObjectState

Goal:

```text
Gaussian / multi-view evidence -> A[N,K] -> ObjectState[K]
```

Suggested data:

- Small audited uCO3D / CO3D subset once source audit passes.
- Existing NeRF / Splatfacto smoke data for renderer and Gaussian export.
- Small self-captured or public rows only when they enter the current manifest
  and file-audit contracts.

Suggested light teacher set:

- SAM 2 or existing mask adapter for candidate masks.
- DINOv2-style visual features for identity embedding.
- CoTracker3 or TAPIR-style tracks for temporal consistency.
- Depth Anything V2 or existing RGB-D evidence for cheap geometry hints.

Non-goals:

- Do not start by training a full world model.
- Do not require action-conditioned prediction.
- Do not promote teacher masks to physical object GT.

### v2 Candidate: Identity Stability And Geometry

Goal:

```text
same physical object across view / occlusion / time -> stable ObjectState
```

Suggested additions:

- PointOdyssey, Dynamic Replica, Kubric / MOVi for controlled track, occlusion
  and synthetic dynamic geometry evidence.
- VGGT / DUSt3R / MASt3R-style geometry teachers only after optional adapter
  boundaries and compute costs are clear.
- SAGA / LangSplat-style 3D Gaussian semantic baselines as posthoc comparison,
  not as default ObjGauss kernel.

Required evaluation:

- identity persistence
- occlusion recovery
- view invariance
- prediction gap versus history baseline
- blocked / fail / pass row separation

### v3 Candidate: Action To Future State

Goal:

```text
ObjectState(t), action -> ObjectState(t+1)
```

Suggested additions:

- ManiSkill2 / RLBench / DROID / HOT3D / DexYCB-style routes only after
  identity and prediction rows are already reviewable.
- Synthetic or controlled action oracle rows before broad real robot data.

Boundary:

- Observed interaction is not the same as randomized counterfactual proof.
- Intervention rows require action GT, pose/state transition GT, and explicit
  action-conditioned predictions.

## 4. Teacher Model Roles

Treat teacher models as independent weak-supervision sources. Each output
should carry source, version, confidence and file path metadata.

| Role | Candidate teachers | Output | Use | Caveat |
| --- | --- | --- | --- | --- |
| Proposal / mask | Grounding DINO, SAM 2 | boxes, phrases, masks, mask tubes | object candidate discovery and mask projection loss | Visual region masks are not physical object truth |
| Identity / semantic feature | DINOv2, CLIP, SigLIP | patch, crop or object embeddings | contrastive identity and open-vocabulary naming | Semantic similarity can merge same-category instances |
| Tracking | CoTracker3, TAPIR | point tracks and visibility | temporal assignment consistency and occlusion recovery | Tracks are not object IDs; visibility uncertainty must weight losses |
| Geometry / depth | VGGT, DUSt3R, MASt3R, Depth Anything V2 | camera, depth, point maps, matches | geometry weighting, point consistency, camera/depth checks | Monocular depth scale and learned geometry can be biased |
| 3D Gaussian segmentation baseline | SAGA, LangSplat, OVGaussian / OpenGaussian-like methods | Gaussian semantic or affinity features | baseline and ablation against ObjGauss object assignment | Usually posthoc/static and may need separate license review |

Recommended minimal teacher chain for a future optional adapter smoke:

```text
Grounding DINO -> SAM 2 -> CoTracker3/TAPIR
  -> DINOv2/CLIP/SigLIP features
  -> cheap geometry/depth teacher
  -> ObjGauss assignment / ObjectState gate
```

The chain should be confidence-gated:

```text
teacher_weight = teacher_score * cross_teacher_agreement * temporal_stability
```

Low-confidence teacher outputs should become weak evidence or be dropped; they
must not silently rewrite object identity.

## 5. Intermediate Data Shape

The notes argue for object-level data, not loose frame files. The current
project already has adjacent contracts: scene/object bundles, controlled
capture manifests, real evidence bundles and reality row ledgers.

A future broad training sample should map into these concepts:

```text
Sample {
  sample_id
  source_dataset
  source_sequence_id
  source_license
  frames[]
  physical_objects[]
  gaussian_refs[]
  teacher_outputs[]
  object_tracks[]
  assignment_evidence[]
  quality_flags
}
```

Minimum per-frame fields:

- RGB path
- timestamp or frame index
- camera intrinsics / extrinsics when available
- optional depth path
- Gaussian evidence ref when available
- teacher mask / box / feature / track / depth refs
- confidence and provenance for every teacher output

Minimum object-level fields:

- physical object id when GT exists
- category / label text when available
- visibility / occlusion metadata
- 6DoF pose when GT exists
- stable track or identity link when available
- action interval when intervention evidence exists

Important distinction:

```text
teacher_outputs != ground_truth
renderer object_id != physical identity GT
```

## 6. Loss And Gate Mapping

The notes list a broad training objective:

```text
L = L_rgb + L_mask + L_id + L_track + L_geo + L_temp + L_pred
```

Map this conservatively to ObjGauss:

| Loss family | Current / future ObjGauss role |
| --- | --- |
| RGB / render | Already aligned with `L_render` in the v1 kernel contract and Splatfacto smoke route |
| Mask projection | Existing mask voting / Object Field path; future teacher masks can enter as `PerceptionEvidence.mask` |
| Identity contrastive | Fits identity encoder and State Variable Gate, but requires explicit candidate predictions |
| Track consistency | Future optional evidence term for assignment / matching, not hard ID equality |
| Geometry / depth | Optional confidence or reconstruction evidence; cannot replace pose GT for reality rows |
| Temporal assignment consistency | Fits object matching and stability diagnostics |
| Prediction | Must be evaluated against history baselines through predictive/reality gates |
| Intervention | Must use action-conditioned candidates and no-action baselines |

Do not add every loss at once. A safer order is:

1. mask projection and object assignment stability
2. identity embedding / track consistency
3. geometry-aware weighting
4. prediction versus history baseline
5. action-conditioned intervention

## 7. Evaluation And Ablations

Evaluation should be object-level.

Priority metrics:

- `idf1`
- `fragmentation_rate`
- `swap_rate`
- `occlusion_recovery_rate`
- `contrastive_margin`
- `multi_view_iou`
- `track_reprojection_error`
- `state_ade`
- `history_ade`
- `prediction_gap_vs_history_model`
- `action_conditioned_ade`
- `counterfactual_outcome_accuracy`
- `wrong_direction_rate`

Baselines to preserve:

| Baseline | Purpose |
| --- | --- |
| Scene-GS only | Proves whether object assignment adds value over plain reconstruction |
| 2D-first object tubes | Compares stitched 2D tracking against object-centric 3D state |
| SAGA-style posthoc segmentation | Compares posthoc 3D segmentation against training-time object assignment |
| Geometry-light | Tests whether expensive geometry teachers are worth the cost |
| No future head | Measures whether prediction hurts or helps object state quality |

Useful ablations:

- SAM 2 alone vs Grounding DINO + SAM 2
- CoTracker3 vs TAPIR vs fused track evidence
- Depth Anything V2 vs VGGT / MASt3R
- DINOv2 vs CLIP / SigLIP vs fused feature evidence
- no confidence weighting vs teacher score vs cross-teacher agreement

## 8. Suggested Backlog Items

These are research / docs / adapter candidates, not automatically approved
implementation tasks.

### TRAINING-SOURCE-AUDIT-UCO3D-001

Audit official uCO3D and CO3D access paths, fields, size, license lineage,
download constraints and minimum safe subset. Output a source-backed markdown
report and do not download data in the first slice.

### TEACHER-OUTPUT-MANIFEST-SPEC-001

Define a manifest for optional teacher outputs:

```text
masks / boxes / features / tracks / depth / point maps / confidence / source
```

The manifest should be consumable by existing mask voting, ObjectState
identity and reality row tooling without making teacher packages required
dependencies.

### OBJECT-LEVEL-SAMPLE-SCHEMA-001

Map the broad object-level sample shape in this document to existing
controlled capture and real evidence bundle schemas. Prefer adapters over a
new parallel dataset contract.

### UCO3D-ADAPTER-SMOKE-001

After source audit, import one tiny ignored local uCO3D / CO3D sequence into
the existing capture/evidence contracts. Keep all media and generated outputs
under `outputs/`.

### LIGHT-TEACHER-ENSEMBLE-SMOKE-001

Run one optional teacher chain on a tiny local sample and produce only
sidecar evidence files. This should not alter the core kernel, default viewer,
or asset registry.

## 9. Risk Notes

- Teacher bias can compound. Agreement and confidence filtering are mandatory.
- SAM-style masks are region proposals, not physical object GT.
- Synthetic datasets are useful for controlled failures but can hide domain
  gaps.
- Objaverse-style asset pools need per-object license lineage, not a single
  dataset-level assumption.
- Robot/action datasets belong to prediction/intervention research, not early
  renderer or static object assignment work.
- Public demo, HF release and commercial wording require a separate license
  audit regardless of training usefulness.
