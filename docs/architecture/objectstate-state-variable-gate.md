# ObjectState State Variable Gate

> 状态: research architecture spec / v2 gate baseline
> 最近更新: 2026-07-08
> 依赖:
> - `docs/architecture/objgauss-v1-kernel-contract.md`
> - `docs/architecture/objgauss-v1-object-emergence-plan.md`
> - `docs/architecture/object-emergence-model-v1.md`
> - `docs/architecture/objgauss-v2-mvp-world-model-plan.md`

## 0. Purpose

ObjGauss v2 的核心验收不再是“能分割 Gaussian”，而是证明
`ObjectState_t` 可以作为世界模型里的近似状态变量。

工程目标：

```text
ObjGauss v2 proves that ObjectState_t is an approximate sufficient statistic
of X_t: it preserves identity under occlusion, view change, motion, and action
perturbation, and it supports object-level prediction.
```

其中：

```text
X_t = observation at time t
      RGB / video / Gaussian cloud / sensor evidence

S_t = ObjectState_t = f(X_t)

valid state variable:
P(X_future | X_past) ~= P(X_future | S_t)
P(S_{t+1} | S_1, ..., S_t, A_t) ~= P(S_{t+1} | S_t, A_t)
```

本文件定义 ObjGauss 的 **State Variable Gate**。它不改变 v1 kernel contract，
也不把 v2 / v3 world-model 假设提前写成已落地能力。

ObjGauss V2 does not assume ObjectState is a valid world state. The purpose of
V2 is to experimentally determine whether ObjectState satisfies the Markov
sufficiency property required by a world-state representation.

中文口径：

```text
ObjGauss V2 不假设 ObjectState 是真实世界状态，而是通过实验验证
ObjectState 是否满足世界状态表示所需的马尔可夫充分性。
```

## 1. Core Claim

ObjGauss 需要验证的科学假设是：

```text
visible Gaussian != object
ObjectState = latent object belief
```

一个对象被遮挡时，观察里的 Gaussian support 可以变少或消失，但 `ObjectState`
不能被静默删除。正确变化是：

```text
visibility/support decreases
uncertainty increases
identity persists
latent state remains addressable
```

失败表现：

```text
visible support disappears -> object id disappears
reappearance creates a new identity
state history cannot explain future prediction
```

这类失败说明系统学到的是 observation state，而不是 world state。

## 2. State Variable Definition

### Observation State

Observation state 表示“当前看到了什么”：

```text
ObservationState_t = g(X_t)
```

特征：

- 受视角、遮挡、mask support 和 renderer route 影响。
- 对象不可见时容易消失。
- 可以解释当前帧，但不一定能压缩历史。

### World State

World state 表示“世界中真实存在什么”：

```text
WorldState_t = S_t
```

特征：

- 同一对象跨视角和遮挡保持 identity。
- 支持 `P(S_{t+1} | S_t, A_t)`。
- 对不可见对象保留 latent state、uncertainty 和 relation。

### ObjGauss Contract

ObjGauss 的状态变量只能是 `ObjectState`：

```text
PerceptionEvidence -> ObjectState -> GaussianToken
```

`GaussianToken` 是 renderer primitive。`object_id` 是 renderer-facing address，
不是 primary identity truth。`ObjectState.id` / identity belief 必须来自
assignment、matching、state history 和 gate 证据，而不是 hard label equality。

## 3. Required Experiments

State Variable Gate 由五类实验组成。前两类证明 identity，第三类证明 observation
invariance，第四类证明 predictive sufficiency，第五类证明 causal interface。

### 3.1 Identity Persistence

问题：

```text
same physical object across time -> same ObjectState identity
```

输入：

```text
t0 visible object
t1..tn object moves / rotates / changes support
tn reobserved object
```

指标：

- `idf1`
- `fragmentation_rate`
- `swap_rate`
- `identity_survival_rate`
- `new_id_after_reappearance_rate`

失败条件：

```text
cup_001 at t0 -> cup_009 at tn without a birth event
```

### 3.2 Occlusion Recovery

问题：

```text
object remains real while partially or fully hidden
```

遮挡等级：

```text
0%, 30%, 60%, 90%, full temporary disappearance
```

指标：

- `occlusion_recovery_rate = P(id_before == id_after)`
- `latent_state_retention_rate`
- `uncertainty_increase_when_hidden`
- `support_recovery_delta`
- `false_death_rate`
- `false_birth_rate`

最低 contract：

```text
visible support may go to zero
ObjectState must remain latent and matchable
```

### 3.3 View Invariance

问题：

```text
same object from different views -> nearby ObjectState embedding
different objects -> separated ObjectState embedding
```

输入：

```text
front view / side view / back view / multi-camera view
```

指标：

- `same_object_distance`
- `different_object_distance`
- `contrastive_margin`
- `cross_view_id_accuracy`
- `view_conditioned_swap_rate`

验收必须同时记录正负样本。只报告 same-object close 不够；需要证明 different-object
far enough。

### 3.4 Predictive Sufficiency

问题：

```text
ObjectState_t should retain enough information for future prediction
```

对比模型：

```text
History model: X_{t-k:t} -> future
State model:   ObjectState_t -> future
```

指标：

- `centroid_ade`
- `centroid_fde`
- `bbox_iou_future`
- `velocity_error`
- `relation_persistence`
- `prediction_gap_vs_history_model`

解释：

如果 State model 明显弱于 History model，说明 `ObjectState_t` 没有压缩足够历史信息。
v2 初期只要求 one-step / short-horizon prediction；multi-step rollout collapse 应记录为
negative evidence，不作为第一版失败退出。

### 3.5 Counterfactual / Action Interface

问题：

```text
ObjectState_t + action -> different predicted future
```

最小动作集：

- `push_left`
- `push_right`
- `move`
- `hide`
- `reveal`

指标：

- `action_conditioned_ade`
- `counterfactual_outcome_accuracy`
- `wrong_direction_rate`
- `interaction_relation_accuracy`
- `intervention_explanation_available`

边界：

第一版可以使用 synthetic action oracle。真实机器人 / embodied action 不是本 gate 的
前置条件。

## 4. Gate Levels

### Smoke Gate

目标：证明评估闭环能跑，且不会把 observation state 误报为 world state。

数据：

- synthetic identity oracle。
- deterministic occlusion / cross-view / swap scenarios。
- 至少一个 small real/public sample replay，允许 fail。

必需输出：

```text
state_variable_gate.json
failure_table.md
identity_metrics.csv
occlusion_metrics.csv
view_invariance_metrics.csv
```

最低通过条件：

```text
synthetic_identity_idf1 >= 0.95
synthetic_occlusion_recovery_rate >= 0.95
synthetic_swap_rate <= 0.02
cross_view_margin_positive = true
real_sample_rows_present = true
real_sample_failures_recorded = true
```

### Candidate Gate

目标：证明 controlled real/public samples 上没有明显 identity collapse。

数据：

- synthetic oracle suite。
- controlled real/public samples with scripted or replayed occlusion/view changes。
- at least three object categories or scenes。

必需输出：

```text
state_variable_candidate_summary.json
prediction_sufficiency_summary.json
blocked_rows.md
```

最低通过条件：

```text
synthetic_smoke_gate_passed = true
controlled_real_identity_collapse = false
controlled_real_fragmentation_rate reported
controlled_real_swap_rate reported
short_horizon_prediction_gap_vs_history_model reported
blocked_rows_separated_from_pass_rows = true
```

这里不设真实数据 `99%+` 目标。真实数据第一阶段要求可复现、可解释、失败不混入通过行。

### Paper Gate

目标：证明 ObjectState 作为状态变量的证据足够进入论文级 claim。

数据：

- synthetic oracle。
- controlled real/public samples。
- open-world real samples or clearly documented negative rows。
- action-conditioned counterfactual cases, synthetic or controlled real。

必需输出：

```text
state_variable_paper_report.md
identity_persistence_table.csv
occlusion_recovery_table.csv
view_invariance_table.csv
predictive_sufficiency_table.csv
counterfactual_table.csv
negative_evidence.md
```

最低通过条件：

```text
smoke_gate_passed = true
candidate_gate_passed = true
predictive_sufficiency_gap_within_declared_bound = true
counterfactual_direction_accuracy_above_baseline = true
negative_evidence_reported = true
no_open_world_failure_reported_as_pass = true
```

## 5. Metric Definitions

### Identity Metrics

```text
idf1 = identity F1 over matched object tracks
fragmentation_rate = extra identities created per physical object
swap_rate = matched identity changes between physical objects
identity_survival_rate = tracks that keep identity over the required window
```

### Occlusion Metrics

```text
occlusion_recovery_rate = reappeared object matched to pre-occlusion identity
false_death_rate = latent object removed without death event
false_birth_rate = reappeared object assigned new identity without birth event
latent_state_retention_rate = hidden objects retained as ObjectState rows
```

### View Metrics

```text
same_object_distance = distance(S_a, S_b) for same object across views
different_object_distance = distance(S_a, S_b) for different objects
contrastive_margin = different_object_distance - same_object_distance
```

### Prediction Metrics

```text
ADE = mean future centroid error over horizon
FDE = final future centroid error
prediction_gap = metric(ObjectState model) - metric(history model)
```

### Causal Metrics

```text
counterfactual_outcome_accuracy = predicted outcome matches intervention result
wrong_direction_rate = predicted motion direction contradicts action
```

## 6. Implementation Slices

### OBJECTSTATE-STATE-VARIABLE-GATE-001

Freeze this spec and register the gate in project state.

Deliverables:

- This architecture spec.
- `docs/state/pr-queue.md` planned item.
- `docs/state/project-status.md` current direction update.

### OBJECTSTATE-IDENTITY-GATE-001

Build the smoke evaluator for identity persistence, occlusion recovery, and
view invariance.

Required behavior:

- Use existing synthetic identity oracle and scenario fixtures when possible.
- Fail if no explicit prediction/state sequence is provided.
- Output pass and fail rows separately.
- Treat small real/public sample failure as valid negative evidence, not as pass.

Implemented v0.2 facts:

- Core module: `objgauss.core.objectstate_identity_gate`.
- Gate schema: `objgauss-objectstate-identity-gate-v1`.
- Dataset schema: `objgauss-objectstate-identity-dataset-v1`.
- Inputs are explicit candidate predictions: `predicted_slots_by_fixture` or
  `predicted_assignments_by_fixture`.
- Candidate identity embeddings are derived from predicted slots / assignments,
  never from oracle labels by default.
- Metrics include `id_accuracy`, `idf1`, `embedding_retrieval_recall_at_1`,
  `long_term_drift_rate`, `fragmentation_rate`, `occlusion_recovery_rate` and
  `contrastive_margin`.
- Legacy synthetic stability diagnostics / gate now reject missing predictions;
  there is no oracle expected-slot fallback in gate paths.

### OBJECTSTATE-IDENTITY-MODEL-001

Train a small ObjectState identity encoder on the identity rows from
`OBJECTSTATE-IDENTITY-GATE-001`.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_identity_encoder`.
- Training schema: `objgauss-objectstate-identity-encoder-training-v1`.
- State schema: `objgauss-objectstate-identity-encoder-state-v1`.
- Training uses a NumPy linear projection and supervised contrastive identity
  loss: same identity pairs contribute positive distance loss; different
  identity pairs contribute margin loss.
- Summary reports initial / final contrastive loss, positive / negative loss,
  active negative pairs, retrieval recall and non-goals.
- This is a smoke training evaluator. It does not add an identity graph,
  replay buffer, renderer loss, diffusion model or viewer/export default.

### OBJECTSTATE-PREDICTIVE-GATE-001

Build the first synthetic predictive sufficiency smoke gate:

```text
ObjectState_t(pose, velocity) -> ObjectState_{t+n}(pose)
```

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_predictive_gate`.
- Gate schema: `objgauss-objectstate-predictive-gate-v1`.
- The state predictor uses synthetic ObjectState pose + velocity from
  `SyntheticWorldObject.trajectory`.
- The baseline predictor uses short observation history when available.
- Summary reports `state_ade`, `state_fde`, `history_ade`,
  `prediction_error_ratio`, `state_sufficiency_score` and
  `identity_consistency_rate`.
- A wrong / missing velocity candidate fails the gate, so this is not a
  rubber-stamp evaluator.
- Current scope is synthetic smoke only. Controlled real rows, learned
  dynamics, action-conditioned counterfactuals and paper-level Markov
  sufficiency remain future gates.

### OBJECTSTATE-PHYSICAL-STATE-GATE-001

Add pose / centroid / velocity / relation prediction checks.

Required behavior:

- Compare `ObjectState_t -> ObjectState_{t+1}` against a history baseline.
- Report ADE / FDE and prediction gap.
- Keep renderer metrics secondary; PSNR/render score cannot pass this gate.

### OBJECTSTATE-CAUSAL-STATE-GATE-001

Add action-conditioned synthetic or controlled counterfactual checks.

Required behavior:

- Define a minimal action schema.
- Test push / move / hide / reveal.
- Report wrong-direction and relation outcome errors.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_causal_gate`.
- Gate schema: `objgauss-objectstate-causal-gate-v1`.
- Action schema: `objgauss-objectstate-action-v1`.
- Minimal action set: `push_left`, `push_right`, `hold`.
- The synthetic target applies current ObjectState pose + velocity + controlled
  action delta, then compares action-conditioned prediction against a no-action
  baseline.
- Summary reports `action_conditioned_ade`, `action_conditioned_fde`,
  `no_action_ade`, `intervention_gain`, `action_error_ratio`,
  `counterfactual_outcome_accuracy`, `wrong_direction_rate` and
  `identity_consistency_rate`.
- A candidate that ignores action (`candidate_action_scale=0`) fails the gate,
  so causal evidence cannot be replaced by a pure tracker.
- Current scope is synthetic controlled action only. Real action rows, relation
  changes, hide / reveal and learned dynamics remain future gates.

### OBJECTSTATE-REALITY-GATE-001

Add the first controlled real / public row acceptance contract for Phase 1.

Required behavior:

- Treat real / public rows as evidence rows, not as implicit world-model proof.
- Separate `pass`, `fail` and `blocked` rows in the summary.
- Require identity, prediction and intervention rows before a candidate can pass.
- Require ground-truth fields per row type: identity rows need identity GT,
  prediction rows need pose GT, intervention rows need pose + action GT, and
  all non-blocked rows need timestamp GT.
- Report controlled real fragmentation / swap / prediction gap / intervention
  accuracy when the rows are available.
- Keep open-world real rows as negative or blocked evidence; do not mark them
  pass at this gate level.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_reality_gate`.
- Gate schema: `objgauss-objectstate-reality-gate-v1`.
- Row schema: `objgauss-objectstate-real-public-row-v1`.
- Evidence kinds: `identity`, `prediction`, `intervention`.
- Source kinds: `controlled_real`, `public_replay`, `open_world_real`.
- Row statuses: `pass`, `fail`, `blocked`.
- Non-blocked identity rows require `idf1`, `fragmentation_rate`,
  `swap_rate` and `identity_collapse`.
- Non-blocked prediction rows require `state_ade`, `history_ade` and
  `prediction_gap_vs_history_model`.
- Non-blocked intervention rows require `action_conditioned_ade`,
  `counterfactual_outcome_accuracy` and `wrong_direction_rate`.
- Summary reports `controlled_real_identity_collapse`,
  `controlled_real_fragmentation_rate`, `controlled_real_swap_rate`,
  `short_horizon_prediction_gap_vs_history_model`,
  `intervention_counterfactual_outcome_accuracy` and separated
  `pass_rows` / `fail_rows` / `blocked_rows`.
- `open_world_real` rows cannot be marked `pass`; this prevents open-world
  failures from being promoted as candidate evidence.
- Current scope is the row contract / evaluator. It does not collect a real
  tabletop dataset, train a dynamics model, implement memory / replay, use
  diffusion, touch renderer loss, mutate viewer defaults or submit generated
  outputs.

### OBJECTSTATE-REALITY-PUBLIC-ROWS-001

Register the first public artifact evidence rows against
`OBJECTSTATE-REALITY-GATE-001`.

Required behavior:

- Use existing small public / local viewer artifacts as row sources.
- Create rows for identity, prediction and intervention evidence.
- Mark rows `blocked` when the artifact lacks timestamped identity GT, 6DoF
  pose tracks or action / intervention outcome GT.
- State explicitly that `object_id` labels are renderer-facing addresses or
  candidate object assignments, not physical identity ground truth.
- Feed the rows into the reality gate so blocked rows are counted separately
  from pass rows.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_reality_public_rows`.
- Summary schema: `objgauss-objectstate-public-artifact-rows-v1`.
- Default public artifacts:
  - `real-sample-v2-sample-aware-lego`
  - `polyhaven-chair`
  - `nike-real-splat-demo`
  - `plush`
- Each artifact emits three blocked rows: `identity`, `prediction` and
  `intervention`.
- `objectstate_reality_public_rows_summary(...)` returns the artifact list,
  generated rows, embedded reality gate summary and `blocked_rows_markdown`.
- The embedded reality gate fails by design today: `real_or_public_rows_present`
  is true, but identity / prediction / intervention pass rows are absent.
- Current scope is evidence registration only. It does not write
  `public/samples`, submit ignored artifacts, train a Gaussian model, collect
  controlled tabletop data, train dynamics, use replay / diffusion or mutate
  viewer defaults.

### OBJECTSTATE-CONTROLLED-CAPTURE-MANIFEST-001

Add the frame-level contract for actual controlled tabletop capture /
annotation data.

Required behavior:

- Record the raw capture evidence before candidate model metrics exist.
- Require a controlled real sample id, object category, scenario, FPS,
  observation modalities, artifact refs and license.
- Record declared physical objects with stable `object_id`.
- Record frames with strictly increasing timestamps, RGB evidence, optional
  per-frame Gaussian reconstruction refs, per-frame object annotations and
  optional action refs.
- Record optional per-frame capture condition metadata:
  `condition.view_id`, `condition.lighting_id` and `condition.camera_pose`.
- Record 6DoF pose as `position` plus `rotation_xyzw` when available.
- Record action events with object id, time interval, type and optional vector.
- Produce a readiness summary for Stage 1 identity, Stage 2 prediction and
  Stage 3 intervention evidence.
- Produce a controlled-real manifest seed whose identity / prediction /
  intervention rows remain `blocked` until candidate metrics are computed.

Implemented v0.2 facts:

- Core module: `objgauss.core.objectstate_controlled_capture`.
- Manifest schema:
  `objgauss-objectstate-controlled-capture-manifest-v1`.
- Summary schema:
  `objgauss-objectstate-controlled-capture-summary-v1`.
- `read_objectstate_controlled_capture_manifest(...)` reads JSON.
- `validate_objectstate_controlled_capture_manifest(...)` validates schema,
  declared objects, actions, frame references, pose shape and timestamp order.
- Frame `condition` is optional and, when present, validates non-empty
  `view_id`, non-empty `lighting_id` and / or `camera_pose` with
  `position` plus `rotation_xyzw`.
- `objectstate_controlled_capture_summary(...)` reports frame / object / action
  counts, RGB / Gaussian coverage, GT availability and readiness booleans:
  `identity_stage_ready`, `prediction_stage_ready`,
  `intervention_stage_ready` and `real_gaussian_reconstruction_present`.
- `objectstate_controlled_real_manifest_from_capture_manifest(...)` emits a
  `objgauss-objectstate-controlled-real-manifest-v1` seed with blocked rows.
- CLI command:
  `objgauss object-state validate-controlled-capture <capture-manifest.json>`.
- CLI outputs optional `--summary-output` and `--controlled-real-output`, plus
  readiness gates `--require-identity-ready`,
  `--require-prediction-ready` and `--require-intervention-ready`.

Current scope remains manifest / readiness validation only. It does not capture
video, create GT, reconstruct Gaussians, compute model metrics, train Gaussian
or dynamics models, use replay / diffusion, or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-CAPTURE-IMPORT-001

Build a controlled capture manifest from a local tabletop capture bundle.

Required behavior:

- Treat `sample.json`, `objects.csv`, `frames.csv`, `annotations.csv` and
  optional `actions.csv` as operator-provided capture / annotation facts.
- Convert those files into
  `objgauss-objectstate-controlled-capture-manifest-v1`.
- Validate the resulting manifest with the existing controlled capture
  contract.
- Emit the normal controlled capture summary and controlled-real blocked seed.
- Do not create GT, infer missing poses, reconstruct Gaussians or score a
  candidate model.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_controlled_capture_import`.
- Import summary schema:
  `objgauss-objectstate-controlled-capture-import-v1`.
- Bundle files:
  - `sample.json`: controlled sample metadata compatible with
    capture manifest `sample`.
  - `objects.csv`: `object_id`, `category`, optional `instance_label`, optional
    `dimension_x_m`, `dimension_y_m`, `dimension_z_m`.
  - `frames.csv`: `frame_id`, `timestamp`, `rgb`, optional `gaussian`,
    optional `action_id`, optional `view_id`, `lighting_id` and camera pose
    columns `camera_x`, `camera_y`, `camera_z`, `camera_qx`, `camera_qy`,
    `camera_qz`, `camera_qw`.
  - `annotations.csv`: `frame_id`, `object_id`, optional `visible`,
    optional `occlusion_fraction`, optional object pose columns `x`, `y`, `z`,
    `qx`, `qy`, `qz`, `qw`.
  - `actions.csv`: optional action facts with `action_id`, `action_type`,
    `object_id`, `start_timestamp`, `end_timestamp`, optional `actor`,
    optional `target_object_id` and optional vector columns.
- Partial pose vectors fail-fast; if any pose component is present, all
  required pose columns must be present.
- Annotation rows must reference a known frame, and every imported frame must
  have at least one annotation row.
- CLI command:
  `objgauss object-state import-controlled-capture-bundle <bundle-root> --output <capture-manifest.json>`.
- CLI can also write `--summary-output` and `--controlled-real-output`, and
  supports readiness gates `--require-identity-ready`,
  `--require-prediction-ready` and `--require-intervention-ready`.

Current scope remains import / validation only. It does not capture video,
create ground truth, verify file bytes, reconstruct Gaussians, train models,
write public samples, use replay / diffusion or mutate viewer defaults. Use
`audit-controlled-capture-files` after import, or
`accept-controlled-capture-bundle` directly, to prove the referenced RGB /
Gaussian files exist and have recognizable formats.

### OBJECTSTATE-CONTROLLED-CAPTURE-BUNDLE-ACCEPTANCE-001

Bundle import and file audit into the pre-identity-handoff acceptance gate.

Required behavior:

- Import a local controlled capture bundle into the controlled capture
  manifest contract.
- Run the controlled capture file audit against the same bundle root.
- Require identity-stage readiness by default, because the accepted bundle is
  meant to feed Stage 1 identity handoff.
- Optionally require prediction-stage and intervention-stage readiness for
  stricter Phase 1 rows.
- Keep acceptance separate from candidate identity evaluation; a bundle can be
  accepted and still have no candidate pass row.

Implemented v0.1 facts:

- Summary schema:
  `objgauss-objectstate-controlled-capture-bundle-acceptance-v1`.
- Core function:
  `objectstate_controlled_capture_bundle_acceptance_summary(...)`.
- Summary embeds both `objgauss-objectstate-controlled-capture-import-v1` and
  `objgauss-objectstate-controlled-capture-file-audit-v1`.
- Acceptance gates:
  `identity_stage_ready`, `prediction_stage_ready`,
  `intervention_stage_ready` and `capture_file_audit_pass`.
- Default behavior requires identity readiness and Gaussian frame files.
- CLI command:
  `objgauss object-state accept-controlled-capture-bundle <bundle-root> --output <capture-manifest.json>`.
- CLI can write `--summary-output`, `--import-summary-output`,
  `--file-audit-output`, `--missing-files-output` and
  `--controlled-real-output`.
- CLI supports `--require-prediction-ready`,
  `--require-intervention-ready`, `--no-require-identity-ready`,
  `--no-require-gaussian-files`, `--check-artifact-refs`,
  `--min-rgb-bytes`, `--min-gaussian-bytes`, `--hash-files`,
  `--no-require-frame-formats` and `--require-pass`.

Current scope remains import + local file audit only. It does not capture
video, create ground truth, reconstruct Gaussians, run identity handoff,
score a candidate model, train models, write public samples, use replay /
diffusion or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-CAPTURE-FILE-AUDIT-001

Verify that a controlled capture manifest points to an actual local capture
bundle before identity handoff.

Implemented v0.5 facts:

- Core module: `objgauss.core.objectstate_controlled_capture_files`.
- Summary schema:
  `objgauss-objectstate-controlled-capture-file-audit-v1`.
- `objectstate_controlled_capture_file_audit(...)` validates the capture
  manifest, resolves frame-relative paths against a bundle root, checks file
  existence, and requires frame-level RGB / Gaussian refs to be regular files
  meeting configurable minimum byte sizes.
- Frame-level RGB / Gaussian refs also require recognizable file signatures by
  default. RGB accepts PNG, JPEG, WebP and PPM signatures. Gaussian evidence
  accepts PLY headers with a vertex element, or raw `.splat` files whose size
  is a non-zero multiple of 32 bytes.
- RGB frame files are always required.
- Gaussian frame files are required by default; `require_gaussian_files=false`
  allows RGB-only local staging without claiming real Gaussian readiness.
- `check_artifact_refs=true` also checks sample-level `artifact_refs` paths.
- The summary reports per-kind `referenced` / `existing` / `valid` /
  `missing` counts, readiness booleans, full `file_records` and
  `missing_files`.
- `hash_files=true` records SHA256 hashes for valid frame RGB / Gaussian files;
  sample-level artifact refs are not hashed and may remain directories.
- `require_frame_formats=false` / CLI `--no-require-frame-formats` is an
  explicit staging downgrade; default controlled identity handoff keeps the
  format audit enabled.
- `objectstate_controlled_capture_missing_files_markdown(...)` renders missing
  references for handoff reports.
- CLI command:
  `objgauss object-state audit-controlled-capture-files <capture>`.
- CLI defaults `--root` to the manifest directory, and can write
  `--summary-output` JSON plus `--missing-files-output` Markdown.
- CLI supports `--min-rgb-bytes`, `--min-gaussian-bytes`, `--hash-files` and
  `--no-require-frame-formats`.

Current scope remains local file integrity auditing only. It may hash file
bytes and inspect file signatures when requested, but it does not capture
video, create GT, parse image pixels, fully parse Gaussian payloads,
reconstruct Gaussians, train models, write public samples, use replay /
diffusion or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-IDENTITY-EVAL-001

Add the first controlled real Stage 1 identity metric evaluator.

Required behavior:

- Input a controlled capture manifest with timestamped physical object GT.
- Input candidate ObjectState / tracker identity predictions bound to
  `(frame_id, object_id)`.
- Reject candidate predictions whose sample id, frame id or object id does not
  match the capture manifest.
- Compute identity metrics required by `OBJECTSTATE-REALITY-GATE-001`:
  `idf1`, `fragmentation_rate`, `swap_rate` and `identity_collapse`.
- Emit a controlled-real manifest where the identity row becomes `pass` or
  `fail`, while prediction / intervention rows remain `blocked` until their
  metrics exist.
- Keep this as an evaluator / handoff contract only; do not run a tracking
  model or infer GT.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_controlled_identity_eval`.
- Prediction schema:
  `objgauss-objectstate-controlled-identity-predictions-v1`.
- Eval summary schema:
  `objgauss-objectstate-controlled-identity-eval-v1`.
- `read_objectstate_controlled_identity_predictions(...)` reads JSON.
- `validate_objectstate_controlled_identity_predictions(...)` validates
  candidate metadata and per-frame predictions. Candidate metadata may include
  explicit `identity_evidence` for reconstruction noise robustness.
- `evaluate_objectstate_controlled_identity_predictions(...)` compares capture
  GT to candidate identity tracks and outputs pass / fail metrics.
- Metrics now include `idf1`, `track_retrieval_recall_at_1`,
  `long_term_drift_rate`, `fragmentation_rate`, `swap_rate`,
  `identity_collapse`, `track_coverage` and
  `reconstruction_noise_robustness`.
- `track_retrieval_recall_at_1` is a controlled-track retrieval proxy: each
  predicted candidate identity retrieves its majority physical-object owner
  across the capture sequence.
- `long_term_drift_rate` counts same-physical-object transitions where the
  candidate predicted identity changes.
- `reconstruction_noise_robustness` must come from explicit candidate
  `identity_evidence` with a score, variant count and source. Missing evidence
  fails the controlled identity eval; the evaluator does not infer robustness
  from file existence, image pixels or Gaussian headers.
- Threshold defaults:
  `min_idf1=0.95`, `min_track_retrieval_recall_at_1=0.95`,
  `max_fragmentation_rate=0.05`, `max_long_term_drift_rate=0.05`,
  `max_swap_rate=0.0`, `min_reconstruction_noise_robustness=0.95`,
  `min_reconstruction_noise_variants=2`,
  `require_no_identity_collapse=true`.
- CLI command:
  `objgauss object-state eval-controlled-identity <capture.json> <predictions.json>`.
- CLI outputs optional `--summary-output` and `--controlled-real-output`, plus
  threshold args and `--require-pass`.

Current scope remains Stage 1 identity evaluation only. It does not capture
video, create GT, run segmentation / tracking, compute pose prediction,
compute action-conditioned metrics, train Gaussian or dynamics models, use
replay / diffusion, or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-PREDICTION-EVAL-001

Add the first controlled real Stage 2 prediction metric evaluator.

Required behavior:

- Input a controlled capture manifest with timestamped 6DoF pose GT.
- Input candidate future-pose predictions bound to
  `(source_frame_id, target_frame_id, object_id)`.
- Require each prediction to include both `predicted_position` from the
  ObjectState state model and `history_baseline_position` from an explicit
  history baseline.
- Reject candidate predictions whose sample id, frame id, object id or pose GT
  does not match the capture manifest.
- Compute prediction metrics required by `OBJECTSTATE-REALITY-GATE-001`:
  `state_ade`, `history_ade` and `prediction_gap_vs_history_model`.
- Emit a controlled-real manifest where the prediction row becomes `pass` or
  `fail`, while identity / intervention rows remain blocked unless their
  metrics already exist.
- Keep this as an evaluator / handoff contract only; do not run a dynamics
  model, infer pose GT or create history baselines.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_controlled_prediction_eval`.
- Prediction schema:
  `objgauss-objectstate-controlled-prediction-candidates-v1`.
- Eval summary schema:
  `objgauss-objectstate-controlled-prediction-eval-v1`.
- `read_objectstate_controlled_prediction_candidates(...)` reads JSON.
- `validate_objectstate_controlled_prediction_candidates(...)` validates
  candidate metadata and per-object future-pose predictions.
- `evaluate_objectstate_controlled_prediction_candidates(...)` compares
  candidate predictions against capture pose GT and outputs pass / fail
  metrics.
- Metrics include `state_ade`, `history_ade`,
  `prediction_gap_vs_history_model`, `error_ratio_vs_history_model`,
  prediction count and horizon seconds.
- Threshold defaults:
  `max_state_ade=0.05m`,
  `max_prediction_gap_vs_history_model=0.02m`,
  `max_error_ratio_vs_history_model=1.25` and
  `min_prediction_count=1`.
- CLI command:
  `objgauss object-state eval-controlled-prediction <capture.json> <predictions.json>`.
- CLI outputs optional `--summary-output` and `--controlled-real-output`, plus
  threshold args and `--require-pass`.

Current scope remains Stage 2 prediction evaluation only. It does not capture
video, create GT, run a prediction / dynamics model, compute identity or
action-conditioned metrics, train Gaussian or dynamics models, use replay /
diffusion, or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-INTERVENTION-EVAL-001

Add the first controlled real Stage 3 intervention metric evaluator.

Required behavior:

- Input a controlled capture manifest with timestamped 6DoF pose GT and
  timestamped action GT.
- Input candidate action-conditioned future-pose predictions bound to
  `(source_frame_id, target_frame_id, object_id, action_id)`.
- Require each intervention prediction to include both
  `action_conditioned_position` from the ObjectState + Action candidate and
  `no_action_baseline_position` from an explicit no-action baseline.
- Require the referenced action to provide a non-zero action vector so
  `wrong_direction_rate` is measurable instead of inferred.
- Reject candidate predictions whose sample id, frame id, object id, action id,
  action interval or pose GT does not match the capture manifest.
- Compute intervention metrics required by `OBJECTSTATE-REALITY-GATE-001`:
  `action_conditioned_ade`, `counterfactual_outcome_accuracy` and
  `wrong_direction_rate`.
- Emit a controlled-real manifest where the intervention row becomes `pass` or
  `fail`, while identity / prediction rows remain blocked unless their metrics
  already exist.
- Keep this as an evaluator / handoff contract only; do not run a dynamics
  model, infer pose/action GT or create no-action baselines.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_controlled_intervention_eval`.
- Intervention schema:
  `objgauss-objectstate-controlled-intervention-candidates-v1`.
- Eval summary schema:
  `objgauss-objectstate-controlled-intervention-eval-v1`.
- `read_objectstate_controlled_intervention_candidates(...)` reads JSON.
- `validate_objectstate_controlled_intervention_candidates(...)` validates
  candidate metadata and per-object action-conditioned predictions.
- `evaluate_objectstate_controlled_intervention_candidates(...)` compares
  action-conditioned predictions against capture pose/action GT and outputs
  pass / fail metrics.
- Metrics include `action_conditioned_ade`, `no_action_ade`,
  `intervention_gain`, `action_error_ratio`,
  `counterfactual_outcome_accuracy`, `wrong_direction_rate`, intervention
  count and horizon seconds.
- Threshold defaults:
  `max_action_conditioned_ade=0.05m`,
  `min_counterfactual_outcome_accuracy=0.95`,
  `max_wrong_direction_rate=0.0`,
  `min_intervention_gain=0.0` and
  `min_intervention_count=1`.
- CLI command:
  `objgauss object-state eval-controlled-intervention <capture.json> <interventions.json>`.
- CLI outputs optional `--summary-output` and `--controlled-real-output`, plus
  threshold args and `--require-pass`.

Current scope remains Stage 3 intervention evaluation only. It does not capture
video, create GT, run an action-conditioned model, compute identity or
prediction metrics, train Gaussian or dynamics models, use replay / diffusion,
or mutate viewer defaults.

### OBJECTSTATE-IDENTITY-PREDICTION-ADAPTER-001

Bridge candidate ObjectState outputs into the controlled identity evaluator.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_identity_prediction_adapter`.
- `read_trainable_kernel_identity_source(...)` reads and validates a
  `objgauss-trainable-kernel-model-artifact-v1` JSON file.
- `objectstate_identity_predictions_from_trainable_artifact(...)` maps
  per-frame trainable-kernel `object_states` to
  `objgauss-objectstate-controlled-identity-predictions-v1`.
- The adapter requires a controlled capture manifest with per-frame
  `pose.position`. It uses nearest-centroid association only to decide which
  candidate ObjectState slot corresponds to each annotated physical object in
  a frame.
- The emitted `predicted_identity` is the stable candidate slot address
  (`slot-<id>`), not physical identity ground truth.
- If the trainable artifact contains `identity_evidence`, the adapter carries
  it into prediction `candidate.identity_evidence` for the controlled identity
  evaluator.
- Optional `max_centroid_distance` can drop unmatched far associations; an
  all-dropped output fails validation instead of creating empty evidence.
- CLI command:
  `objgauss object-state export-identity-predictions <capture> <objectstates> --output <predictions>`.

Expected handoff chain:

```text
validate-controlled-capture
        ->
export-identity-predictions
        ->
eval-controlled-identity
        ->
controlled-real-gate --identity-only
```

Current scope remains adapter / handoff only. It does not create capture data,
create GT, infer physical identity, train Gaussian or dynamics models, compute
prediction / intervention metrics, use replay / diffusion, or mutate viewer
defaults.

### OBJECTSTATE-CONTROLLED-IDENTITY-HANDOFF-001

Bundle the Stage 1 controlled identity chain into a single reproducible handoff.

Implemented v0.4 facts:

- Core module: `objgauss.core.objectstate_controlled_identity_handoff`.
- Summary schema:
  `objgauss-objectstate-controlled-identity-handoff-v1`.
- `objectstate_controlled_identity_handoff(...)` takes a controlled capture
  manifest plus trainable kernel ObjectState artifact and produces:
  - `capture_file_audit` using
    `objgauss-objectstate-controlled-capture-file-audit-v1`;
  - `candidate_artifact_file_audit` using
    `objgauss-objectstate-controlled-candidate-artifact-file-audit-v1`;
  - `candidate_artifact_ref_match`, proving the audited candidate artifact path
    is present in prediction `candidate.artifact_refs`;
  - `identity_scenario_audit` using
    `objgauss-objectstate-controlled-identity-scenario-audit-v1`;
  - `identity_predictions` using
    `objgauss-objectstate-controlled-identity-predictions-v1`;
  - `identity_eval` using
    `objgauss-objectstate-controlled-identity-eval-v1`;
  - `controlled_real_manifest` using
    `objgauss-objectstate-controlled-real-manifest-v1`;
  - `controlled_real_summary` using
    `objgauss-objectstate-controlled-real-rows-v1`.
- The embedded reality gate is identity-only Stage 1:
  `require_identity_pass_row=true`,
  `require_prediction_pass_row=false`,
  `require_intervention_pass_row=false`.
- The handoff pass condition requires both the capture file audit and candidate
  artifact file audit to pass, and requires the audited candidate artifact path
  to match prediction artifact refs. It also requires the controlled capture
  manifest to contain an identity scenario challenge before identity metrics and
  the identity-only reality gate can make the handoff pass.
- Frame-level RGB / Gaussian refs are checked as non-empty regular files by
  default, must pass the frame format signature audit by default, and optional
  SHA256 hashes can be included for audit evidence.
- The candidate trainable ObjectState artifact is also checked as a non-empty
  regular local file, with optional SHA256 hash evidence.
- The candidate metadata cannot claim a different artifact: the audited file
  path must appear in `identity_predictions.candidate.artifact_refs`.
- The handoff pass condition also inherits controlled identity quality gates:
  track retrieval, long-term drift and explicit reconstruction-noise
  robustness evidence. A stable identity track without candidate
  `identity_evidence` still fails the Stage 1 handoff.
- The identity scenario audit requires at least one object with clear-visible
  before, occluded, and clear-visible after observations. `clear-visible`
  means `visible=true` and `occlusion_fraction` below the configured
  threshold.
- The identity scenario audit also requires declared real-identity coverage by
  default: at least two distinct `frame.condition.view_id` values, at least
  two distinct `frame.condition.lighting_id` values, and at least two
  `frame.condition.camera_pose` values whose max translation is at least
  `0.01m`.
- The audit uses manifest fields `visible`, `occlusion_fraction` and
  `condition`; it does not read image pixels or verify actual lighting /
  physical camera motion beyond the declared metadata.
- Prediction and intervention rows remain visible as blocked rows in the
  controlled-real summary; they are not hidden or promoted.
- CLI command:
  `objgauss object-state controlled-identity-handoff <capture> <objectstates> --output-dir <dir>`.
- CLI writes:
  `capture-file-audit.json`, `capture-missing-files.md`,
  `candidate-artifact-file-audit.json`,
  `identity-scenario-audit.json`,
  `identity-predictions.json`, `identity-eval-summary.json`,
  `controlled-real.json`, `controlled-real-summary.json`, `blocked-rows.md`
  and `handoff-summary.json`.
- CLI defaults `--capture-root` to the manifest directory and supports
  `--min-rgb-bytes`, `--min-gaussian-bytes`, `--hash-files` and
  `--check-artifact-refs`.
- CLI also supports `--no-require-frame-formats` as an explicit staging
  downgrade.
- CLI also supports `--min-candidate-artifact-bytes` and
  `--hash-candidate-artifact`.
- CLI also supports `--min-identity-scenario-frames` and
  `--min-occlusion-fraction`.
- CLI also supports `--min-view-conditions`, `--min-lighting-conditions` and
  `--min-camera-motion-m`.
- CLI also supports `--min-track-retrieval-recall-at-1`,
  `--max-long-term-drift-rate`, `--min-reconstruction-noise-robustness` and
  `--min-reconstruction-noise-variants`.

Current scope remains reproducible handoff only. It does not collect capture
data, create GT, parse image pixels, train Gaussian or dynamics models, compute
prediction / intervention metrics, use replay / diffusion, write public samples
or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-REALITY-BUNDLE-HANDOFF-001

Bundle the full Phase 1 controlled reality chain into a single reproducible
handoff.

Implemented v0.1 facts:

- Core module:
  `objgauss.core.objectstate_controlled_reality_bundle_handoff`.
- Summary schema:
  `objgauss-objectstate-controlled-reality-bundle-handoff-v1`.
- `objectstate_controlled_reality_bundle_handoff(...)` takes a controlled
  capture bundle root, a trainable ObjectState artifact, explicit prediction
  candidate JSON and explicit intervention candidate JSON.
- The handoff first reuses `controlled-identity-bundle-handoff`, so bundle CSV
  import, file audit, candidate artifact audit, identity scenario audit,
  identity predictions and identity eval stay on the established Stage 1 path.
- The handoff then runs `OBJECTSTATE-CONTROLLED-PREDICTION-EVAL-001` and
  `OBJECTSTATE-CONTROLLED-INTERVENTION-EVAL-001` against the same imported
  capture manifest.
- The final controlled-real manifest merges:
  - identity row from the identity evaluator;
  - prediction row from the prediction evaluator;
  - intervention row from the intervention evaluator.
- The final reality gate requires identity, prediction and intervention pass
  rows, with `min_real_or_public_rows=3` by default.
- CLI command:
  `objgauss object-state controlled-reality-bundle-handoff <bundle-root> <objectstates.json> <prediction-candidates.json> <intervention-candidates.json> --output-dir <dir>`.
- CLI writes:
  `capture-manifest.json`, bundle acceptance/import/file-audit artifacts,
  identity handoff artifacts, `prediction-eval-summary.json`,
  `intervention-eval-summary.json`, the merged `controlled-real.json`,
  full `controlled-real-summary.json`, `blocked-rows.md` and
  `reality-bundle-handoff-summary.json`.
- CLI supports the existing identity thresholds, prediction thresholds,
  intervention thresholds, file audit options and `--require-pass`.

Current scope remains full Phase 1 handoff orchestration only. It does not
collect capture data, create GT, create prediction candidates, create
intervention candidates, run prediction / intervention models, train Gaussian
or dynamics models, use replay / diffusion, write public samples or mutate
viewer defaults.

### OBJECTSTATE-CONTROLLED-REALITY-BUNDLE-READINESS-001

Add the preflight audit for the full Phase 1 controlled reality handoff.

Implemented v0.1 facts:

- Core module:
  `objgauss.core.objectstate_controlled_reality_bundle_readiness`.
- Summary schema:
  `objgauss-objectstate-controlled-reality-bundle-readiness-v1`.
- `objectstate_controlled_reality_bundle_readiness(...)` takes a controlled
  capture bundle root, trainable ObjectState artifact path, prediction
  candidates path and intervention candidates path.
- The audit reuses `OBJECTSTATE-CONTROLLED-CAPTURE-READINESS-001` with
  prediction / intervention readiness and candidate artifact checks enabled.
- The audit validates the trainable artifact schema and checks that the
  trainable artifact can bind to the imported capture manifest through the
  identity prediction adapter.
- The audit validates prediction / intervention candidate schemas and checks
  that their sample id, frame ids, object ids, pose references, action ids,
  action vectors and action intervals bind to the same imported capture
  manifest.
- The audit reports `full_reality_handoff_ready`; this means the inputs are
  structurally ready to run `controlled-reality-bundle-handoff`, not that the
  candidate quality metrics will pass.
- CLI command:
  `objgauss object-state audit-controlled-reality-bundle-readiness <bundle-root> <objectstates.json> <prediction-candidates.json> <intervention-candidates.json>`.
- CLI supports `--summary-output`, `--require-ready`, file audit options and
  identity scenario audit thresholds.

Current scope remains pre-handoff readiness only. It does not collect capture
data, create GT, create prediction / intervention candidates, run identity
handoff, run prediction / intervention eval, train Gaussian or dynamics models,
use replay / diffusion, write public samples or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-REALITY-CANDIDATE-TEMPLATE-001

Add a local authoring helper for Phase 1 prediction / intervention candidate
JSON files.

Implemented v0.1 facts:

- Core module:
  `objgauss.core.objectstate_controlled_reality_candidate_template`.
- Summary schema:
  `objgauss-objectstate-controlled-reality-candidate-template-v1`.
- Draft template schemas:
  - `objgauss-objectstate-controlled-prediction-candidates-template-v1`;
  - `objgauss-objectstate-controlled-intervention-candidates-template-v1`.
- `write_objectstate_controlled_reality_candidate_templates(...)` imports a
  controlled capture bundle, enumerates pose-backed future-prediction pairs and
  action-bracketed intervention pairs, then writes:
  - `prediction-candidates.template.json`;
  - `intervention-candidates.template.json`;
  - a local README with the full readiness / handoff command chain.
- The template schemas are deliberately different from the evaluator input
  schemas:
  - `objgauss-objectstate-controlled-prediction-candidates-v1`;
  - `objgauss-objectstate-controlled-intervention-candidates-v1`.
- Template rows keep candidate position fields as TODO strings and omit target
  pose values to reduce GT leakage during authoring.
- Validators require `template_status="draft_not_valid_for_eval"` and preserve
  claim policy that evaluator schemas must only be used after external model /
  baseline outputs replace all TODO values.
- CLI command:
  `objgauss object-state init-controlled-reality-candidates <bundle-root> --output-dir <dir>`.
- CLI supports capture bundle file names, `--candidate-id`,
  `--candidate-source`, `--artifact-ref`, `--summary-output` and `--force`.
- Generated README and summary next commands point to
  `finalize-controlled-reality-candidates` before readiness / handoff, so
  authors do not need to hand-edit evaluator schemas.

Current scope remains candidate authoring only. It does not collect capture
data, create GT, run prediction / intervention models, train Gaussian or
dynamics models, write eval-ready pass candidates, use replay / diffusion,
write public samples or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-REALITY-CANDIDATE-FINALIZE-001

Add a checked transition from filled draft templates to evaluator-ready
candidate JSON.

Implemented v0.1 facts:

- Core function:
  `finalize_objectstate_controlled_reality_candidate_templates(...)`.
- Summary schema:
  `objgauss-objectstate-controlled-reality-candidate-finalize-v1`.
- Inputs are the draft template files from
  `OBJECTSTATE-CONTROLLED-REALITY-CANDIDATE-TEMPLATE-001`.
- Finalize requires:
  - candidate metadata to have non-TODO `candidate_id`, `source` and
    `artifact_refs`;
  - all required candidate positions to be numeric length-3 vectors;
  - TODO values to be absent from required fields;
  - obvious top-level GT leakage fields such as `target_position` or
    `target_pose` to be absent.
- Finalize writes:
  - `prediction-candidates.json` using
    `objgauss-objectstate-controlled-prediction-candidates-v1`;
  - `intervention-candidates.json` using
    `objgauss-objectstate-controlled-intervention-candidates-v1`.
- The outputs are immediately validated by the existing prediction /
  intervention candidate validators.
- CLI command:
  `objgauss object-state finalize-controlled-reality-candidates <prediction-template.json> <intervention-template.json> --output-dir <dir>`.
- CLI supports optional `--bundle-root` for next-command output,
  `--summary-output` and `--force`.

Current scope remains candidate JSON finalization only. It does not collect
capture data, create GT, run prediction / intervention models, train Gaussian
or dynamics models, evaluate metrics, claim pass rows, use replay / diffusion,
write public samples or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-REALITY-CANDIDATE-WORKFLOW-001

Document the full candidate authoring path from capture bundle to full Phase 1
handoff.

Implemented v0.1 facts:

- `docs/training/controlled-real-capture-runbook.md` now covers:
  - Stage 1 identity handoff as the first real row;
  - prediction / intervention candidate template generation;
  - filling external model or baseline outputs into `.template.json` files;
  - finalizing templates into evaluator-ready candidate JSON;
  - full controlled reality readiness;
  - full controlled reality handoff.
- The runbook records local artifact names for full Phase 1 evidence:
  `template-summary.json`, `finalize-summary.json`,
  `full-readiness-summary.json`, `prediction-candidates.json`,
  `intervention-candidates.json`, `reality-bundle-handoff-summary.json`,
  prediction / intervention eval summaries, full `controlled-real-summary.json`
  and `blocked-rows.md`.

Current scope remains workflow documentation and generated command chaining
only. It does not collect capture data, create GT, run candidate models, train
Gaussian or dynamics models, evaluate metrics, claim pass rows, use replay /
diffusion, write public samples or mutate viewer defaults.

### OBJECTSTATE-CONTROLLED-REAL-ROWS-001

Add the import path for real controlled tabletop manifests.

Required behavior:

- Define a manifest schema for controlled real samples that records sample
  metadata, artifact refs, GT availability and evidence rows.
- Import rows into `OBJECTSTATE-REALITY-GATE-001` without creating or inferring
  ground truth.
- Allow identity rows with timestamped identity GT to become pass / fail rows.
- Keep prediction / intervention rows blocked until 6DoF pose tracks, history
  targets, action events and counterfactual outcomes exist.
- Reject non-blocked rows when required GT or metrics are missing.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_controlled_real_rows`.
- Manifest schema: `objgauss-objectstate-controlled-real-manifest-v1`.
- Summary schema: `objgauss-objectstate-controlled-real-rows-v1`.
- Manifest shape:
  - `sample`: `sample_id`, `source_kind=controlled_real`,
    `object_category`, `scenario`, `observation_modalities`, `artifact_refs`
    and `license`.
  - `ground_truth`: boolean availability for `identity`, `pose`, `action` and
    `timestamp`.
  - `evidence_rows`: `identity` / `prediction` / `intervention` rows with
    `pass`, `fail` or `blocked` status.
- `read_objectstate_controlled_real_manifest(...)` reads JSON.
- `objectstate_reality_rows_from_controlled_real_manifest(...)` converts the
  manifest to reality gate rows and validates them immediately.
- `evaluate_controlled_real_manifest_reality_gate(...)` sends imported rows to
  the existing reality gate.
- `objectstate_controlled_real_rows_summary(...)` embeds rows, gate summary and
  blocked rows markdown.
- Current tests use a tiny local manifest fixture, not a captured dataset. This
  PR does not collect video, write `outputs/` / `public/samples`, train
  Gaussian or dynamics models, add replay / diffusion, or mutate viewer
  defaults.

### OBJECTSTATE-CONTROLLED-REAL-CLI-001

Expose the controlled real manifest importer as a reproducible command-line
handoff.

Implemented v0.1 facts:

- CLI command:
  `objgauss object-state controlled-real-gate <manifest.json>`.
- Inputs: a JSON manifest using
  `objgauss-objectstate-controlled-real-manifest-v1`.
- Outputs:
  - stdout summary with sample id, row counts, gate status and hard blockers.
  - optional `--summary-output` JSON using
    `objgauss-objectstate-controlled-real-rows-v1`.
  - optional `--blocked-rows-output` Markdown table derived from blocked rows.
- Default mode runs the full reality gate and still requires identity,
  prediction and intervention pass rows.
- `--identity-only` runs the Stage 1 identity-state gate by requiring only an
  identity pass row while leaving prediction / intervention blocked rows
  visible in the summary.
- `--require-pass` converts a failed gate into a non-zero CLI error for CI or
  handoff checks.

Current scope remains handoff / audit only. The command does not capture video,
create ground truth, write `outputs/` / `public/samples`, train Gaussian or
dynamics models, add replay / diffusion, or mutate viewer defaults.

## 7. Non-Goals

This spec does not authorize:

- Diffusion world model training.
- Self-generated rollout training.
- Large replay buffer implementation.
- New heavy tracking / segmentation dependencies.
- Replacing the renderer.
- Changing public demo or HF release claims.
- Claiming production readiness.
- Treating Gaussian render quality, PSNR, or object-aware PLY size as state-variable proof.

## 8. Relationship To Existing Docs

- `objgauss-v1-kernel-contract.md` defines ObjectState as the v1 reasoning unit.
- `objgauss-v1-object-emergence-plan.md` defines `A[N,K] -> ObjectState`.
- `object-emergence-model-v1.md` defines current solver / decoder training.
- `objgauss-v2-mvp-world-model-plan.md` defines later world-model evolution.

This file adds the missing acceptance layer:

```text
ObjectState is not accepted as world state until State Variable Gate evidence exists.
```

Until then, ObjGauss public language must remain:

```text
development-stage research prototype
object-aware Gaussian / ObjectState candidate
not a proven object-centric world model
```
