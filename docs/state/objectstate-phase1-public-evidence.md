# ObjectState Phase 1 Public Evidence

> 状态: current
> 最近更新: 2026-07-08

本文件记录 ObjGauss Phase 1 controlled real / public route 的本地证据行。
大型 public dataset zip、RGB-D frames、PLY evidence seed、candidate artifacts 和
handoff outputs 留在 ignored `outputs/`，不提交进 git。

## BOP LMO Public RGB-D Baseline Row

- Evidence id: `OBJECTSTATE-BOP-LMO-PUBLIC-ROW-001`
- Source dataset: BOP Benchmark `bop-benchmark/lmo`
- Source URL:
  `https://huggingface.co/datasets/bop-benchmark/lmo`
- License: `cc-by-sa-4.0`
- Downloaded file:
  `outputs/assets/raw/bop-lmo/lmo_test_bop19.zip`
- Downloaded file size: `117550985` bytes
- SHA256:
  `42d7a15f317476ca3980ee7ec0344b691cbadc796835f0b14f72c89a1dcec421`
- Extracted local scene:
  `outputs/assets/raw/bop-lmo/lmo-test-bop19-subset/test/000002`
- Selected frames: `000003`, `000008`, `000017`
- Local evidence output:
  `outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/`

Command:

```bash
uv run objgauss object-state bop-rgbd-baseline-local-row-handoff \
  outputs/assets/raw/bop-lmo/lmo-test-bop19-subset/test/000002 \
  --output-root outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline \
  --sample-id bop-lmo-test-scene-000002-rgbd-baseline \
  --dataset-id bop-lmo \
  --object-category lmo_objects \
  --max-frames 3 \
  --max-points-per-frame 10000 \
  --overwrite-gaussian-evidence \
  --ply-format binary_little_endian \
  --summary-output outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/bop-rgbd-baseline-local-row-summary.json \
  --license-text "BOP LMO dataset license: cc-by-sa-4.0; verify source terms before redistribution" \
  --force
```

Result:

- Summary schema:
  `objgauss-objectstate-bop-rgbd-baseline-local-row-handoff-v1`
- Summary status:
  `objectstate_bop_rgbd_baseline_local_row_handoff_incomplete`
- RGB-D evidence:
  `selected_frames=3`, `exported_frames=3`, `missing_depth_files=0`,
  `rgbd_total_vertices=30000`
- Baseline ObjectState artifact:
  `baseline_frames=3`, `baseline_states=3`,
  `baseline_total_gaussians=30000`
- Identity predictions: `24`
- Prediction candidates: `16`
- Prediction evidence package:
  `objectstate_controlled_prediction_evidence_package_reviewable`
- Phase 1 ledger maturity: `prediction_reviewable`
- Identity evidence package:
  `objectstate_controlled_identity_evidence_package_incomplete`

Metric results:

- Prediction eval:
  `objectstate_controlled_prediction_eval_fail`
- `state_ade=0.29266938346336563`
- `history_ade=0.19507330249123483`
- `prediction_gap_vs_history_model=0.0975960809721308`
- `error_ratio_vs_history_model=1.500304653305985`
- `prediction_count=16`
- Identity eval:
  `objectstate_controlled_identity_eval_fail`
- `idf1=1.0`
- `track_retrieval_recall_at_1=0.125`
- `fragmentation_rate=0.0`
- `swap_rate=0.0`
- `identity_collapse=true`
- `reconstruction_noise_evidence_present=false`

Interpretation:

This row is real public RGB-D + BOP 6D pose evidence, but it is not a pass row.
It is useful negative evidence for the current single-state Gaussian centroid
baseline:

- Prediction is reviewable and fails metric gates.
- Identity evaluation exposes identity collapse, low retrieval and missing
  reconstruction-noise robustness evidence.
- Identity package is not reviewable as a controlled identity scenario because
  the extracted BOP route does not provide explicit lighting-condition or camera
  pose condition metadata.
- No intervention / action evidence is claimed.

The row must not be used to claim ObjectState is a real-world state variable.
It only proves that the BOP public RGB-D route can produce a reviewable
prediction row and a clear identity negative result from actual public data.

## BOP LMO Condition Metadata Gap Audit

- Evidence id: `OBJECTSTATE-BOP-LMO-CONDITION-GAP-001`
- Related row: `OBJECTSTATE-BOP-LMO-PUBLIC-ROW-001`
- Local evidence output:
  `outputs/evidence/objectstate-bop-lmo-public-000002-condition-gap/`
- Files:
  - `bop-condition-sidecar.default.json`
  - `bop-condition-sidecar-summary.json`
  - `bop-conditions.template.csv`
  - `bop-identity-route-audit-summary.json`

Condition sidecar command:

```bash
uv run objgauss object-state init-bop-condition-sidecar \
  outputs/assets/raw/bop-lmo/lmo-test-bop19-subset/test/000002 \
  --output outputs/evidence/objectstate-bop-lmo-public-000002-condition-gap/bop-condition-sidecar.default.json \
  --summary-output outputs/evidence/objectstate-bop-lmo-public-000002-condition-gap/bop-condition-sidecar-summary.json \
  --condition-csv-template-output outputs/evidence/objectstate-bop-lmo-public-000002-condition-gap/bop-conditions.template.csv \
  --max-frames 3
```

Condition sidecar result:

- Status:
  `objectstate_bop_capture_condition_sidecar_needs_metadata`
- Selected frames: `000003`, `000008`, `000017`
- `view_condition_count=3`
- `lighting_condition_count=1`
- `lighting_ids=["bop-default"]`
- `camera_pose_count=0`
- `max_camera_translation_m=0.0`
- `identity_scenario_metadata_ready=false`

Identity route audit command:

```bash
uv run objgauss object-state audit-bop-identity-route \
  outputs/assets/raw/bop-lmo/lmo-test-bop19-subset/test/000002 \
  --output-root outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline \
  --summary-output outputs/evidence/objectstate-bop-lmo-public-000002-condition-gap/bop-identity-route-audit-summary.json \
  --sample-id bop-lmo-test-scene-000002-rgbd-baseline \
  --dataset-id bop-lmo \
  --object-category lmo_objects \
  --candidate-artifact outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/objectstates.json \
  --condition-sidecar outputs/evidence/objectstate-bop-lmo-public-000002-condition-gap/bop-condition-sidecar.default.json \
  --max-frames 3
```

Identity route audit result:

- Status:
  `objectstate_bop_identity_route_audit_blocked`
- `bop_acceptance_pass=true`
- `phase1_gaussian_evidence_ready=true`
- `candidate_artifact_present=true`
- `candidate_artifact_valid=true`
- `candidate_artifact_binding_ready=true`
- `identity_scenario_metadata_ready=false`
- `identity_evidence_package_reviewable=false`
- `phase1_evidence_ledger_identity_reviewable=false`
- `route_ready_for_identity_handoff=false`
- `route_has_reviewable_identity_evidence=false`

Interpretation:

This audit proves that the current LMO public row is blocked by real scenario
metadata, not by missing RGB-D files, Gaussian evidence or candidate artifact
binding. The selected BOP frames provide enough frame/view variation and an
occlusion reappearance signal, but they do not provide explicit lighting
variation or camera-pose motion metadata. The correct next step is to use a
controlled capture or an enriched public manifest with measured condition
metadata; the identity gate must not be relaxed to make this BOP row pass.

## BOP HOPE Public RGB-D Baseline Row

- Evidence id: `OBJECTSTATE-BOP-HOPE-PUBLIC-ROW-001`
- Source dataset: BOP Benchmark `bop-benchmark/hope`
- Source URL:
  `https://huggingface.co/datasets/bop-benchmark/hope`
- License: `cc-by-sa-4.0`
- Downloaded file:
  `outputs/assets/raw/bop-hope/hope_val_realsense.zip`
- Downloaded file size: `153745625` bytes
- SHA256:
  `25c75bb2daad4ad7e143b3f8d5bdff793fadb65463492792e822dbb36245a49f`
- Extracted local scene:
  `outputs/assets/raw/bop-hope/hope-val-realsense-subset/val/000001`
- Selected frames: `000000`, `000001`, `000002`
- Local evidence output:
  `outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline/`

Command:

```bash
uv run objgauss object-state bop-rgbd-baseline-local-row-handoff \
  outputs/assets/raw/bop-hope/hope-val-realsense-subset/val/000001 \
  --output-root outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline \
  --sample-id bop-hope-val-scene-000001-rgbd-baseline \
  --dataset-id bop-hope \
  --object-category hope_objects \
  --license-text "BOP HOPE dataset license: cc-by-sa-4.0; verify source terms before redistribution" \
  --identity-policy pose_track_per_obj_id \
  --max-frames 3 \
  --max-points-per-frame 10000 \
  --overwrite-gaussian-evidence \
  --ply-format binary_little_endian \
  --summary-output outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline/bop-rgbd-baseline-local-row-summary.json \
  --force
```

Result:

- Summary status:
  `objectstate_bop_rgbd_baseline_local_row_handoff_incomplete`
- Adapter identity policy:
  `identity_policy=pose_track_per_obj_id`,
  `duplicate_obj_id_policy=pose_track_per_obj_id`,
  `pose_track_max_distance_m=0.05`,
  `uses_bop_pose_gt_for_identity_import=true`
- RGB-D evidence:
  `selected_frames=3`, `exported_frames=3`, `missing_depth_files=0`,
  `rgbd_total_vertices=30000`
- Baseline ObjectState artifact:
  `baseline_frames=3`, `baseline_states=3`,
  `baseline_total_gaussians=30000`
- Identity predictions: `54`
- Prediction candidates: `36`
- Local row reviewability:
  `local_row_prediction_handoff_reviewable=true`,
  `local_row_identity_handoff_reviewable=false`
- Phase 1 ledger maturity: `prediction_reviewable`
- Pass gates:
  `prediction_eval_pass=true`, `identity_handoff_pass=false`

Interpretation:

This row proves the BOP adapter can now import a multi-instance public scene
when the caller explicitly selects `pose_track_per_obj_id`. The policy uses
existing BOP pose GT only to create stable physical instance ids in the ground
truth manifest; the generated Gaussian centroid baseline still does not use
BOP pose GT or object ids for ObjectState prediction.

The row is still not a Phase 1 identity pass. It is prediction-reviewable
public evidence plus a clear identity blocker, and it must not be used to claim
ObjectState is a real-world state variable.

## BOP HOPE Condition Metadata Gap Audit

- Evidence id: `OBJECTSTATE-BOP-HOPE-CONDITION-GAP-001`
- Related row: `OBJECTSTATE-BOP-HOPE-PUBLIC-ROW-001`
- Local evidence output:
  `outputs/evidence/objectstate-bop-hope-public-000001-condition-gap/`
- Files:
  - `bop-condition-sidecar.default.json`
  - `bop-condition-sidecar-summary.json`
  - `bop-conditions.template.csv`
  - `bop-identity-route-audit-summary.json`

Condition sidecar command:

```bash
uv run objgauss object-state init-bop-condition-sidecar \
  outputs/assets/raw/bop-hope/hope-val-realsense-subset/val/000001 \
  --output outputs/evidence/objectstate-bop-hope-public-000001-condition-gap/bop-condition-sidecar.default.json \
  --summary-output outputs/evidence/objectstate-bop-hope-public-000001-condition-gap/bop-condition-sidecar-summary.json \
  --condition-csv-template-output outputs/evidence/objectstate-bop-hope-public-000001-condition-gap/bop-conditions.template.csv \
  --max-frames 3
```

Condition sidecar result:

- Status:
  `objectstate_bop_capture_condition_sidecar_needs_metadata`
- Selected frames: `000000`, `000001`, `000002`
- `view_condition_count=3`
- `lighting_condition_count=1`
- `lighting_ids=["bop-default"]`
- `camera_pose_count=0`
- `max_camera_translation_m=0.0`
- `identity_scenario_metadata_ready=false`

Identity route audit command:

```bash
uv run objgauss object-state audit-bop-identity-route \
  outputs/assets/raw/bop-hope/hope-val-realsense-subset/val/000001 \
  --output-root outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline \
  --summary-output outputs/evidence/objectstate-bop-hope-public-000001-condition-gap/bop-identity-route-audit-summary.json \
  --sample-id bop-hope-val-scene-000001-rgbd-baseline \
  --dataset-id bop-hope \
  --object-category hope_objects \
  --candidate-artifact outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline/objectstates.json \
  --condition-sidecar outputs/evidence/objectstate-bop-hope-public-000001-condition-gap/bop-condition-sidecar.default.json \
  --identity-policy pose_track_per_obj_id \
  --max-frames 3
```

Identity route audit result:

- Status:
  `objectstate_bop_identity_route_audit_blocked`
- `bop_acceptance_pass=true`
- `phase1_gaussian_evidence_ready=true`
- `candidate_artifact_present=true`
- `candidate_artifact_valid=true`
- `candidate_artifact_binding_ready=true`
- `identity_scenario_metadata_ready=false`
- `identity_evidence_package_reviewable=false`
- `phase1_evidence_ledger_identity_reviewable=false`
- `route_ready_for_identity_handoff=false`
- `route_has_reviewable_identity_evidence=false`

Interpretation:

The HOPE row is blocked by scenario metadata and challenge coverage, not by the
multi-instance adapter, RGB-D evidence, Gaussian file readiness or candidate
artifact binding. The selected frames have three view ids but no explicit
lighting variation, no camera pose motion metadata and no clear
occlusion-reappearance metadata. The next step is to enrich conditions from a
source that actually records them or move to controlled tabletop capture; do
not fabricate metadata to make the identity route pass.

## BOP Reality Rows Gate Audit

- Evidence id: `OBJECTSTATE-BOP-REALITY-ROWS-001`
- Source rows:
  `OBJECTSTATE-BOP-LMO-PUBLIC-ROW-001`,
  `OBJECTSTATE-BOP-HOPE-PUBLIC-ROW-001`
- Output schema: `objgauss-objectstate-bop-reality-rows-v1`
- Reality row schema: `objgauss-objectstate-real-public-row-v1`

Commands:

```bash
uv run objgauss object-state audit-bop-reality-rows \
  outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/bop-rgbd-baseline-local-row-summary.json \
  --summary-output outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/bop-reality-rows-summary.json \
  --blocked-rows-output outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/bop-reality-blocked-rows.md

uv run objgauss object-state audit-bop-reality-rows \
  outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline/bop-rgbd-baseline-local-row-summary.json \
  --summary-output outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline/bop-reality-rows-summary.json \
  --blocked-rows-output outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline/bop-reality-blocked-rows.md
```

Results:

| sample | source_kind | identity row | prediction row | intervention row | full gate |
| --- | --- | --- | --- | --- | --- |
| `bop-lmo-test-scene-000002-rgbd-baseline` | `public_replay` | `fail` | `fail` | `blocked` | `objectstate_reality_gate_fail` |
| `bop-hope-val-scene-000001-rgbd-baseline` | `public_replay` | `fail` | `pass` | `blocked` | `objectstate_reality_gate_fail` |

Interpretation:

This audit converts the existing BOP local-row handoff summaries into first-class
`OBJECTSTATE-REALITY-GATE-001` rows. It does not create new ground truth, rerun
training, create public samples, or claim intervention / world-model evidence.
It makes the current state-variable gap explicit:

- HOPE contributes one public replay prediction pass row, but identity remains a
  fail row because the baseline collapses physical identities and the scenario
  metadata route is not reviewable.
- LMO contributes public replay negative evidence: identity fail, prediction
  fail, intervention blocked.
- Both rows keep intervention blocked because BOP pose replay has no action /
  counterfactual outcome evidence.

## Phase 1 Reality Row Ledger

- Evidence id: `OBJECTSTATE-REALITY-ROW-LEDGER-001`
- Input summaries:
  - `outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/bop-reality-rows-summary.json`
  - `outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline/bop-reality-rows-summary.json`
- Output:
  `outputs/evidence/objectstate-phase1-reality-row-ledger.json`
- Blocked rows:
  `outputs/evidence/objectstate-phase1-reality-row-ledger-blocked.md`
- State-variable experiment matrix:
  `outputs/evidence/objectstate-phase1-state-variable-experiment-matrix.md`
- Next actions:
  `outputs/evidence/objectstate-phase1-reality-row-ledger-next-actions.md`

Command:

```bash
uv run objgauss object-state audit-reality-row-ledger \
  outputs/evidence/objectstate-bop-hope-public-000001-rgbd-baseline/bop-reality-rows-summary.json \
  outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/bop-reality-rows-summary.json \
  --summary-output outputs/evidence/objectstate-phase1-reality-row-ledger.json \
  --blocked-rows-output outputs/evidence/objectstate-phase1-reality-row-ledger-blocked.md \
  --experiment-matrix-output outputs/evidence/objectstate-phase1-state-variable-experiment-matrix.md \
  --next-actions-output outputs/evidence/objectstate-phase1-reality-row-ledger-next-actions.md
```

Result:

- Summary schema:
  `objgauss-objectstate-reality-row-ledger-v1`
- `ledger_status=objectstate_reality_row_ledger_reviewable`
- `summary_count=2`
- `row_count=6`
- `pass_row_count=1`
- `fail_row_count=3`
- `blocked_row_count=2`
- `sample_count=2`
- Full gate:
  `objectstate_reality_gate_fail`
- Missing pass evidence kinds:
  `identity`, `intervention`
- Hard blockers:
  `identity_pass_rows_present`, `intervention_pass_rows_present`,
  `controlled_real_identity_collapse_absent`, `failed_rows_absent`
- State-variable experiment matrix:
  - `identity_persistence: objectstate_state_variable_experiment_fail /
    objectstate_state_variable_challenge_not_required`
  - `occlusion_recovery: objectstate_state_variable_experiment_missing_metric /
    objectstate_state_variable_challenge_present`
  - `view_invariance: objectstate_state_variable_experiment_missing_metric /
    objectstate_state_variable_challenge_present`
  - `predictive_sufficiency: objectstate_state_variable_experiment_pass /
    objectstate_state_variable_challenge_not_required`
  - `counterfactual_action_interface:
    objectstate_state_variable_experiment_blocked /
    objectstate_state_variable_challenge_absent`
- Next actions:
  - `identity: pass_evidence_missing -> controlled_real_identity_handoff`
  - `intervention: pass_evidence_missing -> controlled_reality_bundle_handoff`

Interpretation:

The ledger gives the current Phase 1 public evidence table one authoritative
gate view. It confirms that existing BOP public rows are useful but not enough:
only prediction has a pass row, identity remains failed because baseline
identity collapse is real negative evidence, and intervention remains blocked
because no action-conditioned / counterfactual public row exists.
The experiment matrix makes the narrower scientific claim explicit: current
public replay evidence supports predictive sufficiency only. It does not yet
provide occlusion recovery metrics, view invariance metrics or counterfactual
action evidence. The BOP scenario challenge metrics now show that LMO contributes
an occlusion reappearance challenge and both LMO / HOPE contribute multi-view
challenge metadata, but those are not `occlusion_recovery_rate` or
`contrastive_margin` model metrics. BOP still has no action challenge.
The next-actions file is an operator handoff for the missing state-variable
evidence. It does not create GT, run evaluations, train a model, or convert
these public replay rows into identity / intervention pass evidence.

## Public Interaction Route Preflight

2026-07-09 added a local-only authoring scaffold for action-capable public
interaction evidence:

```bash
uv run objgauss object-state init-public-interaction-route-workspace \
  outputs/captures/hot3d-clip-000001 \
  --sample-id hot3d-clip-000001 \
  --candidate-id hot3d-clips \
  --source-sequence-id <public-dataset-clip-id> \
  --object <object_id:category:label> \
  --summary-output outputs/captures/hot3d-clip-000001/public-interaction-workspace.json
```

The command writes a controlled capture bundle skeleton plus
`PUBLIC_INTERACTION_ROUTE.md`. It is intentionally workspace-only: it does not
download a public dataset, create GT, write frame / annotation / action rows,
generate Gaussian evidence, create candidates, run handoff, run eval, train a
model, or create reality rows. Its purpose is to keep the HOT3D / DexYCB-style
route aligned with the existing controlled capture validators and to remind
operators that final rows must be converted to `source_kind=public_replay`.

2026-07-09 added a read-only action-capable public dataset route audit:

```bash
uv run objgauss object-state audit-public-interaction-route \
  outputs/captures/hot3d-clip-000001 \
  --summary-output outputs/captures/hot3d-clip-000001/public-interaction-route-summary.json \
  --markdown-output outputs/captures/hot3d-clip-000001/public-interaction-route.md \
  --require-ready
```

The audit defaults to `hot3d-clips` and checks whether a local public
interaction clip has been adapted into the existing controlled capture and
controlled reality handoff contracts:

- `capture-manifest.json`
- `objectstates.json`
- `reality-candidates/prediction-candidates.json`
- `reality-candidates/intervention-candidates.json`

This is a route readiness check for the missing intervention evidence, not a
new row in the LMO / HOPE ledger above. It reports handoff-ready only when the
capture manifest is intervention-ready, per-frame Gaussian evidence is
declared, prediction / intervention candidate JSON files validate and all
sample ids match. It does not download HOT3D, create GT, run eval, train a
model, create a pass row or prove counterfactual causality.

2026-07-09 also added a read-only converter for completed public interaction
handoffs:

```bash
uv run objgauss object-state audit-public-interaction-reality-rows \
  outputs/captures/hot3d-clip-000001/reality-handoff/reality-bundle-handoff-summary.json \
  --summary-output outputs/captures/hot3d-clip-000001/public-interaction-reality-rows.json \
  --blocked-rows-output outputs/captures/hot3d-clip-000001/public-interaction-blocked-rows.md \
  --require-pass
```

That converter takes an existing full handoff summary and emits
`objgauss-objectstate-public-interaction-reality-rows-v1` rows with
`source_kind=public_replay`, so the rows can be merged by
`audit-reality-row-ledger`. It does not run handoff or evaluators again; it
only prevents action-capable public data from being accounted as
`controlled_real`.
