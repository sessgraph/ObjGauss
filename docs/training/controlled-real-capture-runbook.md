# Controlled Real Capture Runbook

> Status: current
> Last updated: 2026-07-09

This runbook describes the first ObjGauss Phase 1 controlled real capture
session. It exists to move `ObjectState` validation from synthetic fixtures to
real RGB / Gaussian / pose evidence without claiming a world-model result too
early.

The goal is narrow:

```text
real tabletop capture -> controlled capture bundle -> identity handoff
real tabletop capture -> controlled reality candidate rows -> full handoff
```

The first successful row only needs to prove that a real controlled identity
sample can enter the `OBJECTSTATE-REALITY-GATE` as pass or fail evidence. It is
not a public demo, a production dataset, a dynamics model, or a viewer default
promotion. Prediction and intervention rows should be added only after the same
capture bundle has pose GT, action GT, and external candidate outputs.

## 1. Minimum Session

Use a tabletop scene with a fixed object set. Start with one primary object and
one distractor:

```text
cup-001
box-001
```

The minimum Stage 1 identity scenario is:

```text
clear visible -> occluded -> clear visible again
```

Record at least three frames for the primary object:

| Required frame | Purpose |
| --- | --- |
| `f000` | Clear object support before occlusion. |
| `f001` | Partial or strong occlusion, ideally `occlusion_fraction >= 0.5`. |
| `f002` | Reappearance after occlusion. |

For candidate gate quality, prefer six to twelve frames instead of only three:

```text
front clear
front occluded
front reappeared
side clear
side occluded
side reappeared
lighting-a clear
lighting-b clear
```

## 2. Capture Requirements

Each accepted bundle must include:

- RGB frame files referenced by `frames.csv`.
- Per-frame Gaussian evidence referenced by `frames.csv`.
- Physical object declarations in `objects.csv`.
- Per-frame physical identity annotations in `annotations.csv`.
- 6DoF pose annotations for each non-blocked identity row candidate.
- `view_id`, `lighting_id`, and camera pose metadata on frames.
- A trainable ObjectState candidate artifact with explicit
  `identity_evidence` for reconstruction-noise robustness.

The current gates only check declared metadata and file integrity. They do not
infer physical truth from pixels, lighting, or camera motion. Bad annotations
can still produce bad science, so capture notes should be kept next to the
local bundle but not committed if they contain large files or private data.

## 3. Environment Preflight

Before a capture session, run the local environment preflight:

```bash
uv run objgauss object-state audit-controlled-capture-environment \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/environment-summary.json
```

The preflight checks whether the current host can see camera devices under
`/dev`, whether RGB capture tooling such as `ffmpeg` or OpenCV `cv2` is
available, and whether Gaussian reconstruction tooling such as COLMAP /
Nerfstudio is on PATH. It does not capture video, reconstruct Gaussians, create
GT, run training, or prove the reality gate.

If the preflight reports no video device, rerun it on the physical capture host
instead of treating a sandbox result as proof that no camera exists.

## 4. Bundle Layout

Create the local bundle outside committed sample paths:

```bash
uv run objgauss object-state init-controlled-capture-bundle \
  outputs/captures/controlled-tabletop-cup-box-001 \
  --sample-id controlled-tabletop-cup-box-001 \
  --object-category cup_box \
  --capture-device "local-camera" \
  --object cup-001:cup:"blue cup" \
  --object box-001:box:"red box"
```

`outputs/` is ignored and remains the right home for raw captures, Gaussian
reconstructions, candidate artifacts, and handoff outputs.

Expected files:

```text
sample.json
objects.csv
frames.csv
annotations.csv
actions.csv
rgb/
gaussians/
README.md
```

Do not copy large captures, reconstructed Gaussian files, checkpoints, or
candidate artifacts into git.

## 5. File Naming

Use stable names that keep RGB and Gaussian refs easy to audit:

```text
rgb/f000.png
rgb/f001.png
rgb/f002.png
gaussians/f000.ply
gaussians/f001.ply
gaussians/f002.ply
```

RGB files must have recognizable PNG, JPEG, WebP, or PPM signatures. Gaussian
files must be PLY files with a vertex element, or raw `.splat` files with a
non-zero size that is a multiple of 32 bytes.

## 6. CSV Rules

`objects.csv` declares physical objects:

```csv
object_id,category,instance_label,dimension_x_m,dimension_y_m,dimension_z_m
cup-001,cup,blue cup,0.08,0.08,0.10
box-001,box,red box,0.12,0.12,0.08
```

`frames.csv` records timestamped observations:

```csv
frame_id,timestamp,rgb,gaussian,action_id,view_id,lighting_id,camera_x,camera_y,camera_z,camera_qx,camera_qy,camera_qz,camera_qw
f000,0.000,rgb/f000.png,gaussians/f000.ply,,front,lighting-a,0,0,0,0,0,0,1
f001,0.500,rgb/f001.png,gaussians/f001.ply,,front,lighting-a,0.02,0,0,0,0,0,1
f002,1.000,rgb/f002.png,gaussians/f002.ply,,side,lighting-b,0.15,0,0,0,0,0,1
```

`annotations.csv` records physical identity, visibility, occlusion, and pose:

```csv
frame_id,object_id,visible,occlusion_fraction,x,y,z,qx,qy,qz,qw
f000,cup-001,true,0.0,0.10,0.20,0.00,0,0,0,1
f001,cup-001,true,0.7,0.11,0.20,0.00,0,0,0,1
f002,cup-001,true,0.0,0.12,0.21,0.00,0,0,0,1
```

For Stage 1 identity, actions can remain empty. For intervention rows, add
`actions.csv` and reference `action_id` from the affected frames:

```csv
action_id,action_type,object_id,start_timestamp,end_timestamp,actor,target_object_id,vector_x,vector_y,vector_z
push-left-001,push_left,cup-001,1.000,1.500,human,, -0.05,0,0
```

## 6.1 Object Transition Dataset

After `capture-manifest.json` validates and before creating prediction /
intervention candidate files, compile the frame-level capture into object-level
transition rows:

```bash
uv run objgauss object-state compile-objectstate-transitions \
  outputs/captures/controlled-tabletop-cup-box-001/capture-manifest.json \
  --output outputs/captures/controlled-tabletop-cup-box-001/objectstate-transitions.json \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/objectstate-transitions-summary.json \
  --require-action-transition
```

This writes `objgauss-objectstate-transition-dataset-v1`: object episodes and
transition rows shaped as `ObjectState_t + Action_t -> ObjectState_t+1`.
The compiler only consumes the already validated capture manifest. It does not
create GT, infer identity, run a prediction / intervention model, train
dynamics, create replay buffers, emit reality rows, or make a metric-pass
claim.

Then run the read-only transition dataset audit before using the rows for
candidate training or evaluator authoring:

```bash
uv run objgauss object-state audit-objectstate-transition-dataset \
  outputs/captures/controlled-tabletop-cup-box-001/objectstate-transitions.json \
  --min-object-episodes 1 \
  --min-transitions 2 \
  --require-action-transition \
  --require-gaussian-refs \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/objectstate-transitions-audit.json \
  --require-ready
```

This writes `objgauss-objectstate-transition-dataset-audit-v1`, checking object
episode count, transition count, action-conditioned transition count, capture
horizon, pose readiness, and Gaussian ref readiness. Passing this audit only
means the transition dataset is ready for the next candidate/evaluator step.
It is not a dynamics model result, prediction metric pass, intervention metric
pass, replay buffer, or world-model claim.

For a first Real Predictive Gate handoff, export evaluator-ready prediction
candidates from the transition dataset:

```bash
uv run objgauss object-state export-transition-prediction-candidates \
  outputs/captures/controlled-tabletop-cup-box-001/objectstate-transitions.json \
  --output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/prediction-candidates.json \
  --policy constant_velocity \
  --candidate-id controlled-tabletop-cup-box-001-transition-cv \
  --artifact-ref outputs/captures/controlled-tabletop-cup-box-001/objectstate-transitions.json \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/transition-prediction-summary.json
```

For an action-vector baseline on an interaction-capable capture, use
`--policy action_delta --require-action-transition`. The exporter writes the
existing `objgauss-objectstate-controlled-prediction-candidates-v1` schema, so
the result can go directly to `eval-controlled-prediction`. The exporter uses
source pose, prior pose, target timestamp, and optional action vector. It does
not read target pose values to generate predicted positions, does not run an
eval, does not train dynamics, and does not claim a metric pass.

For a first Real Intervention Gate handoff on an action-ready transition
dataset, export evaluator-ready intervention candidates:

```bash
uv run objgauss object-state export-transition-intervention-candidates \
  outputs/captures/controlled-tabletop-cup-box-001/objectstate-transitions.json \
  --output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/intervention-candidates.json \
  --policy action_delta \
  --candidate-id controlled-tabletop-cup-box-001-transition-action-delta \
  --artifact-ref outputs/captures/controlled-tabletop-cup-box-001/objectstate-transitions.json \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/transition-intervention-summary.json
```

This writes the existing
`objgauss-objectstate-controlled-intervention-candidates-v1` schema. The
`action_delta` policy predicts `source_pose + action.vector`; the no-action
baseline holds the source pose. The exporter skips action contexts whose action
interval does not fit within the transition timestamps. It does not read target
pose values to generate action-conditioned predictions, does not run an eval,
does not train dynamics, and does not claim intervention pass.

To run the transition-backed prediction and intervention accounting in one
step, use:

```bash
uv run objgauss object-state transition-reality-handoff \
  outputs/captures/controlled-tabletop-cup-box-001/capture-manifest.json \
  outputs/captures/controlled-tabletop-cup-box-001/objectstate-transitions.json \
  --output-dir outputs/captures/controlled-tabletop-cup-box-001/transition-reality-handoff \
  --prediction-policy action_delta \
  --intervention-policy action_delta \
  --require-gaussian-refs \
  --require-pass
```

This writes:

```text
transition-reality-handoff/transition-dataset-audit.json
transition-reality-handoff/prediction-candidates.json
transition-reality-handoff/transition-prediction-summary.json
transition-reality-handoff/prediction-eval-summary.json
transition-reality-handoff/prediction-controlled-real.json
transition-reality-handoff/intervention-candidates.json
transition-reality-handoff/transition-intervention-summary.json
transition-reality-handoff/intervention-eval-summary.json
transition-reality-handoff/intervention-controlled-real.json
transition-reality-handoff/controlled-real.json
transition-reality-handoff/controlled-real-summary.json
transition-reality-handoff/blocked-rows.md
transition-reality-handoff/transition-reality-handoff-summary.json
```

The handoff merges only prediction and intervention rows into a partial
controlled-real manifest. The identity row intentionally remains a blocked seed
row because identity persistence is evaluated by `controlled-identity-handoff`
or `controlled-reality-bundle-handoff`. A passing
`transition-reality-handoff` means the baseline transition candidates passed
the prediction and intervention evaluators under the selected thresholds; it is
not a learned dynamics model, identity proof, counterfactual proof, replay
buffer, or world-model claim.

After the handoff, audit the transition evidence package and add it to the
Phase 1 ledger:

```bash
uv run objgauss object-state audit-transition-reality-evidence-package \
  outputs/captures/controlled-tabletop-cup-box-001 \
  --handoff-dir transition-reality-handoff \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/transition-reality-evidence-package-summary.json \
  --require-reviewable

uv run objgauss object-state audit-phase1-evidence-ledger \
  --transition-reality-summary outputs/captures/controlled-tabletop-cup-box-001/transition-reality-evidence-package-summary.json \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/phase1-evidence-ledger.json \
  --require-reviewable
```

The transition evidence package is reviewable when all handoff JSON / Markdown
outputs exist, schemas validate, `sample_id` is consistent, standalone outputs
match the embedded handoff summary, the identity row remains blocked, and the
prediction / intervention rows are pass or fail. In the ledger this contributes
`transition_reality_reviewable`: prediction and intervention evidence are
reviewable, but identity and full reality remain unproven until the identity
handoff or full controlled reality bundle handoff also passes reviewability.

## 7. Candidate Artifact

The identity handoff expects a trainable ObjectState artifact JSON compatible
with `objgauss-trainable-kernel-model-artifact-v1`. The artifact must include:

- per-frame `object_states` with positions that can be associated to capture
  annotations;
- candidate `artifact_refs` that include the audited artifact path;
- explicit `identity_evidence` with reconstruction-noise robustness score,
  variant count, and source.

The evaluator does not synthesize robustness evidence from file existence. A
stable slot track still fails if `identity_evidence` is missing.

## 8. Prediction And Intervention Candidates

For full Phase 1 validation, create draft candidate files after the bundle has
pose tracks and action metadata:

```bash
uv run objgauss object-state init-controlled-reality-candidates \
  outputs/captures/controlled-tabletop-cup-box-001 \
  --output-dir outputs/captures/controlled-tabletop-cup-box-001/reality-candidates \
  --candidate-id controlled-tabletop-cup-box-001-candidate-v1 \
  --candidate-source "external objectstate predictor" \
  --artifact-ref outputs/captures/controlled-tabletop-cup-box-001/objectstates.json \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/template-summary.json
```

Fill the generated `.template.json` files with external model or baseline
outputs. Do not copy target GT pose values into predictions.

Then finalize the filled templates into evaluator-ready JSON:

```bash
uv run objgauss object-state finalize-controlled-reality-candidates \
  outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/prediction-candidates.template.json \
  outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/intervention-candidates.template.json \
  --output-dir outputs/captures/controlled-tabletop-cup-box-001/reality-candidates \
  --bundle-root outputs/captures/controlled-tabletop-cup-box-001 \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/finalize-summary.json
```

The finalizer only changes filled draft templates into validated evaluator
input files. It does not run prediction or intervention models and does not
claim pass rows.

## 9. Readiness Loop

Run the tolerant readiness audit while the bundle is being filled:

```bash
uv run objgauss object-state audit-controlled-capture-bundle-readiness \
  outputs/captures/controlled-tabletop-cup-box-001 \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/readiness-summary.json
```

Once the audit reports `capture_bundle_ready=true`, run acceptance:

```bash
uv run objgauss object-state accept-controlled-capture-bundle \
  outputs/captures/controlled-tabletop-cup-box-001 \
  --output outputs/captures/controlled-tabletop-cup-box-001/capture-manifest.json \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/acceptance-summary.json \
  --hash-files \
  --require-pass
```

Then run the identity handoff with the candidate artifact:

```bash
uv run objgauss object-state controlled-identity-bundle-handoff \
  outputs/captures/controlled-tabletop-cup-box-001 \
  outputs/captures/controlled-tabletop-cup-box-001/objectstates.json \
  --output-dir outputs/captures/controlled-tabletop-cup-box-001/identity-handoff \
  --hash-files \
  --require-pass
```

This handoff writes `identity-handoff/identity-evidence-package-summary.json`
after the handoff artifacts. Treat that file as the Stage 1 identity review
entry point: reviewable means the local capture / candidate artifact audits,
scenario challenge audit, identity pass / fail row, and handoff summary are
internally consistent. It does not mean the identity metric passed unless the
identity row itself is `pass`, and it does not claim prediction or intervention
evidence.

For full Phase 1 readiness, add the finalized prediction and intervention
candidate JSON files:

```bash
uv run objgauss object-state audit-controlled-reality-bundle-readiness \
  outputs/captures/controlled-tabletop-cup-box-001 \
  outputs/captures/controlled-tabletop-cup-box-001/objectstates.json \
  outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/prediction-candidates.json \
  outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/intervention-candidates.json \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/full-readiness-summary.json \
  --require-ready
```

If the full readiness audit passes, run the full handoff:

```bash
uv run objgauss object-state controlled-reality-bundle-handoff \
  outputs/captures/controlled-tabletop-cup-box-001 \
  outputs/captures/controlled-tabletop-cup-box-001/objectstates.json \
  outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/prediction-candidates.json \
  outputs/captures/controlled-tabletop-cup-box-001/reality-candidates/intervention-candidates.json \
  --output-dir outputs/captures/controlled-tabletop-cup-box-001/reality-handoff \
  --hash-files \
  --require-pass
```

The expected result is not necessarily a pass. A true fail row is useful
evidence if it preserves the distinction between `pass_rows`, `fail_rows`, and
`blocked_rows`.

After the full handoff, audit the local evidence package before making a
review claim:

```bash
uv run objgauss object-state audit-controlled-reality-evidence-package \
  outputs/captures/controlled-tabletop-cup-box-001 \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/evidence-package-summary.json \
  --require-reviewable
```

This audit only checks that the expected local JSON / Markdown artifacts exist,
validate, use a consistent `sample_id`, and agree with the full handoff summary.
It does not rerun the handoff or turn a failed metric row into a pass.

To summarize the current Phase 1 evidence level from existing package
summaries, run:

```bash
uv run objgauss object-state audit-phase1-evidence-ledger \
  --discover-root outputs/captures/controlled-tabletop-cup-box-001 \
  --max-depth 4 \
  --summary-output outputs/captures/controlled-tabletop-cup-box-001/phase1-evidence-ledger.json \
  --require-reviewable
```

The ledger is a read-only index over existing summaries. It does not generate
rows, rerun metrics, or prove the world-model claim.

## 10. Acceptance Evidence

For the first real identity row, keep these local artifacts together:

```text
readiness-summary.json
capture-manifest.json
acceptance-summary.json
identity-handoff/handoff-summary.json
identity-handoff/identity-eval-summary.json
identity-handoff/controlled-real-summary.json
identity-handoff/blocked-rows.md
```

For full Phase 1 rows, also keep:

```text
reality-candidates/template-summary.json
reality-candidates/finalize-summary.json
reality-candidates/full-readiness-summary.json
reality-candidates/prediction-candidates.json
reality-candidates/intervention-candidates.json
reality-handoff/reality-bundle-handoff-summary.json
reality-handoff/prediction-eval-summary.json
reality-handoff/intervention-eval-summary.json
reality-handoff/controlled-real-summary.json
reality-handoff/blocked-rows.md
evidence-package-summary.json
transition-reality-evidence-package-summary.json
```

A Stage 1 identity claim is only reviewable when:

- capture file audit passed with real RGB and Gaussian files;
- identity scenario audit passed;
- candidate artifact file audit passed;
- candidate artifact ref matched prediction metadata;
- identity eval emitted pass or fail metrics;
- reality gate kept prediction and intervention rows blocked until their GT
  and metrics exist.

A full Phase 1 reality claim is only reviewable when identity, prediction, and
intervention rows all come from the same controlled capture manifest, and the
full reality gate preserves any fail rows instead of converting them into
blocked rows or pass rows. The evidence package audit is the final local
pre-review check for that file set; it is not a substitute for real capture,
pose/action GT, candidate outputs, or metric review.

## 11. Boundaries

This runbook does not:

- create ground truth;
- reconstruct Gaussians;
- train a candidate model;
- infer physical identity from object IDs;
- publish a demo;
- promote viewer or export defaults;
- start replay buffer, identity graph, diffusion, or long-horizon dynamics work.

It only defines the shortest reliable path from real controlled tabletop
evidence to reviewable Phase 1 identity, prediction, and intervention rows.
