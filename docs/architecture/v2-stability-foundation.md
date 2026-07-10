# ObjGauss V2 Stability Foundation

状态：`V2-STABILITY-FOUNDATION-002` contract + scenario / diagnostics / gate extension

Canonical ownership:

- synthetic fixture / observation contract: `objgauss.datasets.v2_stability_foundation`
- diagnostics and stability gates: `objgauss.evaluation.v2_stability_diagnostics`,
  `objgauss.evaluation.v2_stability_gate`
- legacy `objgauss.core` paths are compatibility imports only

## 目标

V2 稳定性评估的第一步不是继续增加指标，而是先冻结 evaluation
invariant：

```text
同一个 object 由 synthetic oracle label 定义，不由 slot / embedding / tracker 推断。
```

这保证后续 cross-view、occlusion、perturbation、adversarial scenario 的指标都能
回答同一个问题：

```text
Object Binding Layer 是否在观测变化下保持同一世界实体？
```

## Contract

### ObjectIdentityOracle

`ObjectIdentityOracle` 是 synthetic scenario 的唯一 identity ground truth。

它冻结：

- `oracle_object_id`
- `lineage_id`
- `canonical_slot`
- per-frame `visible`
- per-frame `expected_slot`
- per-frame `expected_slot_relation`

约束：

- `object_id` 必须唯一。
- `lineage_id` 必须唯一。
- `canonical_slot` 必须从 0 连续且唯一。
- 每一帧必须覆盖所有对象，状态只能是 visible 或 occluded。
- visible object 不能使用 `occluded` relation。
- occluded object 必须使用 `occluded` relation。

### SyntheticWorldState

`SyntheticWorldState` 是 observation 之前的 object-level world。

每一帧包含所有 oracle object：

- `pose_center`
- `appearance_feature`
- `appearance_rgb`
- `trajectory`
- `visible`
- `perturbation`

它不是 data augmentation。它是可复现 world state，用来生成观测并保留 identity
真相。

### ObservationModel

`observe_synthetic_world()` 把 `SyntheticWorldState` 投影为
`SyntheticObservationFrame`。

输出包括：

- `AssignmentEvidenceBatch`
- `oracle_object_ids`
- `lineage_ids`
- `expected_slots`

`AssignmentEvidenceBatch.target_assignment` 使用 oracle slot 构造，因此它可以作为
assignment solver / stability gate 的监督或评估输入。

### SyntheticStabilityScenarioFixture

`SyntheticStabilityScenarioFixture` 是 `V2-STABILITY-SCENARIO-002` 引入的 fixture
层。它把一个 `SyntheticWorldState`、对应的 observation batches、observation config、
oracle identity labels、expected slots 和 visible / occluded transitions 绑定成同一个
可复现样例。

当前 schema：

```text
objgauss-v2-stability-scenario-fixture-v1
```

builder：

- `make_synthetic_stability_scenario_fixture(...)`
- `make_synthetic_stability_scenario_suite(...)`

这些 fixture 可以作为后续 diagnostics / gate 的输入，但本身不是 final stability gate。

### SyntheticStabilityDiagnosticsReport

`SyntheticStabilityDiagnosticsReport` 是 `V2-STABILITY-DIAGNOSTICS-001` 引入的诊断层。
它消费 `SyntheticStabilityScenarioFixture` 和与 observation 对齐的 predicted slots 或
predicted assignments，把 row-level assignment 汇总成 identity-level observations。

当前 schema：

```text
objgauss-v2-stability-diagnostics-v1
```

builder：

- `diagnose_synthetic_stability_fixture(...)`
- `expected_slots_for_synthetic_fixture(...)`

输出包括：

- `slot_transition_matrix`
- `identity_confusion_graph`
- `failure_mode_counts`
- `failure_modes`
- `identity_observations`

`FailureModeClassifier` 当前区分五类 deterministic failure mode：

- `slot_swap`
- `identity_fragmentation`
- `object_merge`
- `background_absorption`
- `temporal_drift`

该层只用于失败定位和 gate 前证据检查。它不会训练 solver，不接 renderer loss，也不把
diagnostics 本身升级成 hard gate。

### SyntheticStabilityGateReport

`SyntheticStabilityGateReport` 是 `V2-STABILITY-GATE-001` 引入的 hard gate 层。它
消费同一个 `SyntheticStabilityScenarioFixture` 和 predicted slots / assignments，先生成
diagnostics，再将 identity invariance 作为唯一 hard gate 来源。

当前 schema：

```text
objgauss-v2-stability-gate-v1
```

suite schema：

```text
objgauss-v2-stability-gate-suite-v1
```

builder：

- `evaluate_synthetic_stability_gate(...)`
- `evaluate_synthetic_stability_suite_gate(...)`

输入 contract：

- Gate 和 diagnostics 都必须消费显式 candidate prediction：`predicted_slots` /
  `predicted_assignments` 或 suite 级对应输入。
- 无 prediction 时必须 fail-fast，不允许回退到 oracle `expected_slots` 形成伪绿灯。
- `predicted_assignments` 的列数必须等于 fixture oracle slot 数，否则 fail-fast。

hard checks：

- `expected_slot_consistency_pass`
- `no_slot_swap_pass`
- `identity_no_cross_slot_drift_pass`
- `adversarial_swap_no_exchange_pass`
- `occlusion_recovery_return_pass`
- `no_object_merge_pass`
- `no_background_absorption_pass`
- `diagnostics_failure_reporting_pass`

soft diagnostics：

- assignment entropy
- assignment purity
- temporal coherence

soft diagnostics 只解释质量和退化风险，不能覆盖 hard gate。`diagnostics` 继续用于失败定位，
但 diagnostics 本身不是 gate truth source。

## Scenario Kinds

当前 contract 支持四类 scenario kind：

- `cross_view`
- `occlusion_recovery`
- `perturbation`
- `adversarial_swap`

`adversarial_swap` 会交换 object 0 / 1 的 appearance feature / RGB，但
`oracle_object_id`、`lineage_id` 和 `expected_slot` 必须保持原身份不变。

本阶段只冻结 world/oracle/observation/fixture contract，不把这些 scenario 升级成
完整 benchmark gate。

## Non-goals

本阶段不做：

- GPU 训练
- rollout model
- dynamic-K 自动 birth / merge / split 更新
- renderer 参数解冻
- CLI gate
- diagnostics 替代 identity hard gate
- TensorBoard / checkpoint 输出
- SAM2 / CoTracker / DETR 之类外部 perception 模型接入

## 后续 PR 队列

1. `V2-STABILITY-SCENARIO-002`
   - 扩展 cross-view / occlusion / perturbation / adversarial synthetic scenario。
2. `V2-STABILITY-DIAGNOSTICS-001`
   - 增加 `FailureModeClassifier`、slot transition matrix、identity confusion graph。
3. `V2-STABILITY-GATE-001`
   - 把 identity invariant 作为 hard gate，assignment / temporal 指标作为 soft diagnostics。
4. `V2-STABILITY-CLI-001`
   - 暴露可复现 CPU synthetic benchmark CLI。
