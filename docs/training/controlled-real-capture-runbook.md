# Controlled Real Capture Runbook

> Status: current
> Last updated: 2026-07-08

This runbook describes the first ObjGauss Phase 1 controlled real capture
session. It exists to move `ObjectState` validation from synthetic fixtures to
real RGB / Gaussian / pose evidence without claiming a world-model result too
early.

The goal is narrow:

```text
real tabletop capture -> controlled capture bundle -> identity handoff
```

The first successful row only needs to prove that a real controlled identity
sample can enter the `OBJECTSTATE-REALITY-GATE` as pass or fail evidence. It is
not a public demo, a production dataset, a dynamics model, or a viewer default
promotion.

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

## 3. Bundle Layout

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

## 4. File Naming

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

## 5. CSV Rules

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

## 6. Candidate Artifact

The identity handoff expects a trainable ObjectState artifact JSON compatible
with `objgauss-trainable-kernel-model-artifact-v1`. The artifact must include:

- per-frame `object_states` with positions that can be associated to capture
  annotations;
- candidate `artifact_refs` that include the audited artifact path;
- explicit `identity_evidence` with reconstruction-noise robustness score,
  variant count, and source.

The evaluator does not synthesize robustness evidence from file existence. A
stable slot track still fails if `identity_evidence` is missing.

## 7. Readiness Loop

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

The expected result is not necessarily a pass. A true fail row is useful
evidence if it preserves the distinction between `pass_rows`, `fail_rows`, and
`blocked_rows`.

## 8. Acceptance Evidence

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

A Stage 1 identity claim is only reviewable when:

- capture file audit passed with real RGB and Gaussian files;
- identity scenario audit passed;
- candidate artifact file audit passed;
- candidate artifact ref matched prediction metadata;
- identity eval emitted pass or fail metrics;
- reality gate kept prediction and intervention rows blocked until their GT
  and metrics exist.

## 9. Boundaries

This runbook does not:

- create ground truth;
- reconstruct Gaussians;
- train a candidate model;
- infer physical identity from object IDs;
- publish a demo;
- promote viewer or export defaults;
- start replay buffer, identity graph, diffusion, or long-horizon dynamics work.

It only defines the shortest reliable path from real controlled tabletop
evidence to a reviewable Phase 1 identity row.
