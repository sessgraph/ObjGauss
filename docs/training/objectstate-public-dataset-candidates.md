# ObjectState Public Dataset Candidates

> Status: current candidate audit
> Last updated: 2026-07-08
> Scope: Phase 1 controlled real / public data selection only.

This document records public dataset candidates for the ObjectState state
variable gate. It is a selection audit, not a downloaded dataset, not a
training run, not a Gaussian reconstruction, and not a reality-gate pass.

The current capture host preflight is blocked because this session cannot see a
camera device and lacks RGB / Gaussian reconstruction tooling. The practical
fallback is to choose a small public pose dataset subset, adapt it into the
controlled capture manifest contract, and then generate local Gaussian
evidence under ignored `outputs/`.

## Selection Principle

ObjGauss needs evidence for this claim:

```text
ObjectState_t approximates a state variable:
P(S_{t+1} | S_1, ..., S_t, A_t) ~= P(S_{t+1} | S_t, A_t)
```

For Phase 1, the public dataset must support at least:

- physical object identity ground truth;
- timestamp or frame ordering;
- 6DoF pose ground truth;
- camera/view metadata;
- enough view or occlusion change to test identity stability.

Per-frame Gaussian evidence is still required before the candidate can become
a real ObjGauss row. Public RGB / pose data alone is not enough.

## Candidate Ranking

| Rank | Candidate | Role | Why |
| --- | --- | --- | --- |
| 1 | `bop-ycbv-keyframes` | First public identity / prediction adapter | BOP YCB-V gives household objects, RGB-D evidence, labels, masks, object models, and 6D poses. It is the most direct bridge into current controlled manifest rows. |
| 2 | `bop-hopev2` | View / lighting / robustness candidate | HOPE/HOPEv2 adds household/office clutter, lighting variation, and moving-camera data. Good second row after the adapter exists. |
| 3 | `bop-tudl` | Small adapter sanity check | TUD-L is small, with three moving objects and eight lighting conditions. Useful for cheap smoke, but too narrow for the main claim. |
| 4 | `hot3d-clips` | Action-like interaction candidate | HOT3D-Clips has 150-frame hand-object clips with object/hand/camera 3D poses. It can stress prediction and action-conditioned rows, but it is not randomized counterfactual evidence. |
| 5 | `dexycb` | Non-commercial hand-occlusion stress candidate | DexYCB has YCB grasp sequences and 6D object pose tasks, but the full dataset is large and CC BY-NC, so it must never be treated as public commercial demo material. |

## Gate Coverage

| Candidate | Identity | Occlusion | View | Prediction | Counterfactual |
| --- | --- | --- | --- | --- | --- |
| `bop-ycbv-keyframes` | ready with adapter | partial | partial | partial | blocked |
| `bop-hopev2` | ready with adapter | partial | ready with adapter | partial | blocked |
| `bop-tudl` | ready with adapter | blocked | ready with adapter | partial | blocked |
| `hot3d-clips` | ready with adapter | ready with adapter | ready with adapter | ready with adapter | partial |
| `dexycb` | ready with adapter | partial | partial | partial | blocked |

Interpretation:

- `ready with adapter` means the source appears to contain the required GT, but
  ObjGauss still needs a local adapter into `controlled capture manifest` rows.
- `partial` means the source can support a related measurement, but not the
  full gate without restrictions or additional candidate outputs.
- `blocked` means the dataset should not be used to claim that gate.

## Recommended First Slice

Use one small BOP YCB-V subset first:

```text
BOP YCB-V RGB / depth / mask / pose / camera
        |
        v
import-bop-capture-scene adapter
        |
        v
local per-frame Gaussian evidence under outputs/
        |
        v
identity row + prediction row
```

Do not start with HOT3D. It is closer to interaction, but the format, license
agreement, and egocentric multi-stream setup add avoidable complexity before
the simpler pose adapter has been proven.

The adapter entry point is:

```bash
uv run objgauss object-state import-bop-capture-scene \
  outputs/datasets/bop/ycbv/test/000001 \
  --sample-id bop-ycbv-scene-000001 \
  --dataset-id bop-ycbv \
  --output outputs/captures/bop-ycbv-scene-000001/capture-manifest.json \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-adapter-summary.json \
  --controlled-real-output outputs/captures/bop-ycbv-scene-000001/controlled-real-seed.json \
  --require-identity-ready \
  --require-prediction-ready
```

This adapter expects a local BOP scene folder with `scene_camera.json`,
`scene_gt.json`, optional `scene_gt_info.json`, and `rgb/<frame>.png` or JPEG
files. It converts `cam_R_m2c` / `cam_t_m2c` into ObjGauss 6DoF pose, converts
`visib_fract` into `occlusion_fraction`, and uses a conservative
`single_instance_per_bop_obj_id` identity policy. If a selected frame contains
duplicate `obj_id` entries, it fails instead of inventing unstable instance
tracks.

The adapter does not create Gaussian evidence. After importing the BOP scene,
run a file audit and reconstruct per-frame Gaussian files under ignored
`outputs/` before attempting identity / prediction handoff.

The combined adapter + file audit gate is:

```bash
uv run objgauss object-state accept-bop-capture-scene \
  outputs/datasets/bop/ycbv/test/000001 \
  --sample-id bop-ycbv-scene-000001 \
  --dataset-id bop-ycbv \
  --output outputs/captures/bop-ycbv-scene-000001/capture-manifest.json \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-acceptance-summary.json \
  --file-audit-output outputs/captures/bop-ycbv-scene-000001/bop-file-audit.json \
  --missing-files-output outputs/captures/bop-ycbv-scene-000001/bop-missing-files.md \
  --controlled-real-output outputs/captures/bop-ycbv-scene-000001/controlled-real-seed.json \
  --require-gaussian-files \
  --hash-files \
  --require-pass
```

Without `--require-gaussian-files`, this command can verify the RGB / pose BOP
scene files, but it still cannot support Phase 1 identity rows because
ObjGauss has not produced per-frame Gaussian evidence. With
`--require-gaussian-files`, the command expects valid `gaussians/<frame>.ply`
or `.splat` files for every selected frame.

Before choosing a concrete scene manually, scan the local BOP dataset or split
root and let the selector recommend the first scene that can seed Phase 1
identity / prediction rows:

```bash
uv run objgauss object-state select-bop-phase1-subset \
  outputs/datasets/bop/ycbv \
  --dataset-id bop-ycbv \
  --output-root outputs/captures/bop-ycbv-test-000001 \
  --summary-output outputs/captures/bop-ycbv-phase1-subset-selector-summary.json \
  --require-ready
```

The selector is read-only. It scans for BOP scene roots containing
`scene_gt.json` and `scene_camera.json`, validates each candidate through the
existing BOP adapter, checks minimum selected frames / objects / repeated
identities, and prints the next `init-bop-condition-sidecar`,
`accept-bop-capture-scene` and `audit-bop-phase1-local-row` commands for the
recommended scene. It does not download BOP, copy scene files, infer view /
lighting metadata, generate Gaussian evidence or create a pass row.

After a scene is selected, audit the expected per-frame Gaussian evidence before
running handoff or candidate eval:

```bash
uv run objgauss object-state audit-bop-gaussian-evidence \
  outputs/datasets/bop/ycbv/test/000001 \
  --sample-id bop-ycbv-scene-000001 \
  --dataset-id bop-ycbv \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-gaussian-evidence-summary.json \
  --missing-gaussians-output outputs/captures/bop-ycbv-scene-000001/bop-missing-gaussians.md
```

This preflight reuses the BOP acceptance/file-audit path with
`--require-gaussian-files`, lists the expected `gaussians/<frame>.ply` files,
reports missing or invalid files, and separately records whether COLMAP /
Nerfstudio reconstruction tools are visible on the current host. It does not
reconstruct Gaussians or train a model; it only tells you whether the local
scene can pass the Gaussian evidence prerequisite for Phase 1.

If the selected BOP scene includes local `depth/<frame>.png` files, bootstrap
the first local per-frame evidence from RGB-D backprojection before rerunning
the preflight:

```bash
uv run objgauss object-state export-bop-rgbd-gaussian-evidence \
  outputs/datasets/bop/ycbv/test/000001 \
  --sample-id bop-ycbv-scene-000001 \
  --dataset-id bop-ycbv \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-rgbd-gaussian-export-summary.json \
  --require-ready
```

This writes `gaussians/<frame>.ply` under the local BOP scene root by
back-projecting depth pixels through BOP `scene_camera.json` intrinsics and
using RGB only for point color. It does not use object pose GT to place the
geometry, does not run Splatfacto, does not create a checkpoint, and does not
score identity / prediction candidates. Treat this as RGB-D Gaussian evidence
seed for Phase 1 file readiness; model quality and ObjectState rows still need
the route audits and handoff below.

To get one machine-readable status for the whole local BOP row, run the
combined Phase 1 local row readiness audit:

```bash
uv run objgauss object-state audit-bop-phase1-local-row \
  outputs/datasets/bop/ycbv/test/000001 \
  --output-root outputs/captures/bop-ycbv-scene-000001 \
  --sample-id bop-ycbv-scene-000001 \
  --dataset-id bop-ycbv \
  --candidate-artifact outputs/captures/bop-ycbv-scene-000001/objectstates.json \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-phase1-local-row-summary.json
```

This command wraps the identity route and prediction route audits without
running either handoff. Use it as the first local row status check after placing
the BOP subset and any generated Gaussian / candidate artifacts. It reports a
single `blocking_stage`, such as `local_bop_scene`,
`phase1_gaussian_evidence`, `candidate_artifact`,
`identity_scenario_metadata`, `handoff_ready`, or reviewable evidence status.

For Stage 1 identity-state evidence, run the identity route audit before
starting a handoff:

```bash
uv run objgauss object-state audit-bop-identity-route \
  outputs/datasets/bop/ycbv/test/000001 \
  --output-root outputs/captures/bop-ycbv-scene-000001 \
  --sample-id bop-ycbv-scene-000001 \
  --dataset-id bop-ycbv \
  --candidate-artifact outputs/captures/bop-ycbv-scene-000001/objectstates.json \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-identity-route-summary.json
```

This audit checks BOP acceptance, per-frame Gaussian files, candidate
ObjectState artifact validity, `object_states` frame binding, existing
identity evidence package output and the Phase 1 ledger. It also applies a
stricter identity scenario metadata gate: enough frames, occlusion
reappearance, cross-view variation, lighting variation and camera-pose motion.
The default BOP adapter often has enough pose frames for prediction but still
lacks lighting / camera motion metadata for identity-state proof. In that case
the route should stay blocked; do not relax the identity gate just to turn BOP
pose data into a pass row.

When the local BOP subset has explicit scenario condition metadata from the
capture setup, keep it in a sidecar rather than weakening the gate. The helper
command can generate a default sidecar template from selected BOP frames, or
turn a filled condition CSV into an identity-ready sidecar:

```bash
uv run objgauss object-state init-bop-condition-sidecar \
  outputs/datasets/bop/ycbv/test/000001 \
  --condition-csv-template-output outputs/captures/bop-ycbv-scene-000001/bop-conditions.template.csv \
  --condition-csv outputs/captures/bop-ycbv-scene-000001/bop-conditions.csv \
  --output outputs/captures/bop-ycbv-scene-000001/bop-condition-sidecar.json \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-condition-sidecar-summary.json \
  --require-identity-ready
```

The condition CSV columns are:

```text
frame_id,view_id,lighting_id,camera_x,camera_y,camera_z,camera_qx,camera_qy,camera_qz,camera_qw
```

To bootstrap a real BOP scene, run the command once without `--condition-csv`
and with `--condition-csv-template-output`. Fill the generated CSV template
from the capture setup metadata, save it as `bop-conditions.csv`, then rerun
with `--condition-csv` and `--require-identity-ready`.

The generated sidecar uses this schema:

```json
{
  "schema": "objgauss-objectstate-bop-capture-condition-sidecar-v1",
  "kind": "objectstate_bop_capture_condition_sidecar",
  "frames": {
    "0": {
      "view_id": "front",
      "lighting_id": "bright",
      "camera_pose": {
        "position": [0.0, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
      }
    },
    "1": {
      "view_id": "front",
      "lighting_id": "dim",
      "camera_pose": {
        "position": [0.02, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
      }
    },
    "000002": {
      "view_id": "right",
      "lighting_id": "dim",
      "camera_pose": {
        "position": [0.04, 0.0, 0.0],
        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]
      }
    }
  },
  "condition_policy": {
    "sidecar_only": true,
    "does_not_create_ground_truth": true,
    "does_not_infer_from_pixels": true
  }
}
```

If `--condition-csv` is omitted, the sidecar falls back to default per-frame
`view_id` and `lighting_id` values and should usually remain `needs_metadata`
because it lacks lighting variation and camera motion. The optional CSV
template output is only a fillable authoring aid, not inferred metadata. Pass
an identity-ready sidecar to the adapter, acceptance and route audits with
`--condition-sidecar <path>`. The sidecar may override only
`frame.condition.view_id`, `frame.condition.lighting_id` and
`frame.condition.camera_pose`; it does not create identity GT, object pose GT,
Gaussian evidence or a pass row. With RGB / pose / Gaussian files and a bound
candidate artifact already present, a valid sidecar can move the BOP identity
route from `identity_scenario_metadata` blocked to handoff-ready.

Once the local scene has RGB, pose and per-frame Gaussian evidence, the
read-only route audit should report `handoff_ready`:

```bash
uv run objgauss object-state audit-bop-phase1-route \
  outputs/datasets/bop/ycbv/test/000001 \
  --output-root outputs/captures/bop-ycbv-scene-000001 \
  --sample-id bop-ycbv-scene-000001 \
  --dataset-id bop-ycbv \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-phase1-route-summary.json
```

The audit does not write handoff outputs or run prediction eval. It only reports
whether the scene is blocked, ready for `bop-prediction-baseline-handoff`, or
already prediction-reviewable from existing evidence files.

Once the local scene has RGB, pose and per-frame Gaussian evidence, the
prediction-only baseline handoff can run the full reviewable package in one
command:

```bash
uv run objgauss object-state bop-prediction-baseline-handoff \
  outputs/datasets/bop/ycbv/test/000001 \
  --output-root outputs/captures/bop-ycbv-scene-000001 \
  --sample-id bop-ycbv-scene-000001 \
  --dataset-id bop-ycbv \
  --policy constant_velocity \
  --candidate-id bop-ycbv-constant-velocity-baseline \
  --candidate-source controlled-prediction-baseline \
  --artifact-ref outputs/captures/bop-ycbv-scene-000001/objectstates.json \
  --hash-files \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-prediction-baseline-handoff-summary.json \
  --require-reviewable
```

This command writes the accepted capture manifest, BOP file audit, prediction
candidate templates, baseline-filled prediction candidates, prediction eval
summary, prediction evidence package audit and `phase1-evidence-ledger.json`
under `--output-root`.

After acceptance passes, initialize candidate authoring directly from the BOP
controlled capture manifest. The following commands are the expanded manual
version of the one-command handoff above:

```bash
uv run objgauss object-state init-controlled-reality-candidates-from-manifest \
  outputs/captures/bop-ycbv-scene-000001/capture-manifest.json \
  --output-dir outputs/captures/bop-ycbv-scene-000001/reality-candidates \
  --candidate-id bop-ycbv-objectstate-candidate \
  --candidate-source local-objectstate-candidate \
  --artifact-ref outputs/captures/bop-ycbv-scene-000001/objectstates.json \
  --summary-output outputs/captures/bop-ycbv-scene-000001/reality-candidates/template-summary.json
```

BOP pose scenes do not contain action events, so this path should create
prediction draft rows and keep intervention draft rows at zero. To create a
reviewable first prediction candidate without copying target pose GT into the
template, generate a deterministic constant-velocity baseline:

```bash
uv run objgauss object-state generate-controlled-prediction-baseline-candidates \
  outputs/captures/bop-ycbv-scene-000001/capture-manifest.json \
  outputs/captures/bop-ycbv-scene-000001/reality-candidates/prediction-candidates.template.json \
  --output-dir outputs/captures/bop-ycbv-scene-000001/reality-candidates \
  --policy constant_velocity \
  --candidate-id bop-ycbv-constant-velocity-baseline \
  --candidate-source controlled-prediction-baseline \
  --artifact-ref outputs/captures/bop-ycbv-scene-000001/objectstates.json \
  --summary-output outputs/captures/bop-ycbv-scene-000001/reality-candidates/prediction-baseline-summary.json
```

Then run the Real Predictive Gate:

```bash
uv run objgauss object-state eval-controlled-prediction \
  outputs/captures/bop-ycbv-scene-000001/capture-manifest.json \
  outputs/captures/bop-ycbv-scene-000001/reality-candidates/prediction-candidates.json \
  --summary-output outputs/captures/bop-ycbv-scene-000001/reality-candidates/prediction-eval-summary.json \
  --controlled-real-output outputs/captures/bop-ycbv-scene-000001/reality-candidates/controlled-real-prediction.json
```

Finally, audit the local prediction evidence package:

```bash
uv run objgauss object-state audit-controlled-prediction-evidence-package \
  outputs/captures/bop-ycbv-scene-000001 \
  --summary-output outputs/captures/bop-ycbv-scene-000001/reality-candidates/prediction-evidence-package-summary.json \
  --require-reviewable
```

The one-command handoff writes the Phase 1 evidence ledger automatically. For
manual workflows, run the same ledger explicitly:

```bash
uv run objgauss object-state audit-phase1-evidence-ledger \
  --prediction-summary outputs/captures/bop-ycbv-scene-000001/reality-candidates/prediction-evidence-package-summary.json \
  --summary-output outputs/captures/bop-ycbv-scene-000001/phase1-evidence-ledger.json \
  --require-reviewable
```

After handoff, rerun the route audit with `--require-prediction-reviewable` to
confirm the BOP route now exposes reviewable prediction evidence:

```bash
uv run objgauss object-state audit-bop-phase1-route \
  outputs/datasets/bop/ycbv/test/000001 \
  --output-root outputs/captures/bop-ycbv-scene-000001 \
  --sample-id bop-ycbv-scene-000001 \
  --dataset-id bop-ycbv \
  --summary-output outputs/captures/bop-ycbv-scene-000001/bop-phase1-route-summary.json \
  --require-prediction-reviewable
```

This still does not prove the causal / counterfactual gate. It only moves the
BOP route from blocked rows toward a real prediction pass / fail row once
Gaussian evidence, ObjectState candidate output and future-pose predictions are
available. The baseline generator uses source pose, prior pose history and
target timestamp only; it is not a learned model. The prediction evidence
package is reviewable when file acceptance, baseline candidate generation,
candidate finalization and prediction eval outputs are internally consistent;
reviewable does not mean the prediction metric passed.

For a controlled identity-only handoff, the handoff CLIs write
`identity-evidence-package-summary.json` into the output directory. To rerun or
override the read-only local package audit:

```bash
uv run objgauss object-state audit-controlled-identity-evidence-package \
  outputs/captures/controlled-tabletop-cup-box-identity-001/identity-handoff \
  --capture-manifest outputs/captures/controlled-tabletop-cup-box-identity-001/capture-manifest.json \
  --summary-output outputs/captures/controlled-tabletop-cup-box-identity-001/identity-handoff/identity-evidence-package-summary.json \
  --require-reviewable
```

This checks the existing handoff outputs and preserves the Stage 1 boundary:
reviewable means the identity pass / fail row, capture file audit, candidate
artifact audit, scenario challenge audit and handoff summary are internally
consistent. It does not rerun identity eval, require the identity metric to pass,
or claim predictive / intervention / world-model evidence.

## Hard Blockers

- No public candidate directly supplies ObjGauss per-frame Gaussian evidence.
- Public candidates still require a local adapter run and file audit before
  they become controlled capture manifests.
- BOP can support identity and future-pose prediction rows, but its pose scene
  route has no action events and therefore cannot claim the intervention /
  counterfactual gate.
- Counterfactual rows remain blocked until action-conditioned outcomes are
  evaluated against real outcomes.
- Dataset license terms must be reviewed before any public demo,
  redistribution, or commercial claim.

## Source Notes

- BOP dataset page: `https://bop.felk.cvut.cz/datasets/`
- HOT3D project page: `https://facebookresearch.github.io/hot3d/`
- DexYCB project page: `https://dex-ycb.github.io/`

The code-backed audit can be reproduced with:

```bash
uv run objgauss object-state audit-public-dataset-candidates \
  --summary-output /tmp/objgauss-public-dataset-candidates.json \
  --markdown-output /tmp/objgauss-public-dataset-candidates.md
```

This command does not download data or create pass rows. It only records the
candidate selection state.
