# ObjectState State Variable Gate

> 状态: research architecture spec / v2 gate baseline
> 最近更新: 2026-07-07
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

Implemented v0.1 facts:

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
- Record 6DoF pose as `position` plus `rotation_xyzw` when available.
- Record action events with object id, time interval, type and optional vector.
- Produce a readiness summary for Stage 1 identity, Stage 2 prediction and
  Stage 3 intervention evidence.
- Produce a controlled-real manifest seed whose identity / prediction /
  intervention rows remain `blocked` until candidate metrics are computed.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_controlled_capture`.
- Manifest schema:
  `objgauss-objectstate-controlled-capture-manifest-v1`.
- Summary schema:
  `objgauss-objectstate-controlled-capture-summary-v1`.
- `read_objectstate_controlled_capture_manifest(...)` reads JSON.
- `validate_objectstate_controlled_capture_manifest(...)` validates schema,
  declared objects, actions, frame references, pose shape and timestamp order.
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
  candidate metadata and per-frame predictions.
- `evaluate_objectstate_controlled_identity_predictions(...)` compares capture
  GT to candidate identity tracks and outputs pass / fail metrics.
- Threshold defaults:
  `min_idf1=0.95`, `max_fragmentation_rate=0.05`, `max_swap_rate=0.0`,
  `require_no_identity_collapse=true`.
- CLI command:
  `objgauss object-state eval-controlled-identity <capture.json> <predictions.json>`.
- CLI outputs optional `--summary-output` and `--controlled-real-output`, plus
  threshold args and `--require-pass`.

Current scope remains Stage 1 identity evaluation only. It does not capture
video, create GT, run segmentation / tracking, compute pose prediction,
compute action-conditioned metrics, train Gaussian or dynamics models, use
replay / diffusion, or mutate viewer defaults.

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
