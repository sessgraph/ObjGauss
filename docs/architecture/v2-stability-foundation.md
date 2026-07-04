# ObjGauss V2 Stability Foundation

状态：`V2-STABILITY-FOUNDATION-002` contract

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

## Scenario Kinds

当前 contract 支持四类 scenario kind：

- `cross_view`
- `occlusion_recovery`
- `perturbation`
- `adversarial_swap`

本阶段只冻结 world/oracle/observation contract，不把这些 scenario 扩成完整
benchmark suite。

## Non-goals

本阶段不做：

- GPU 训练
- rollout model
- dynamic-K 自动 birth / merge / split 更新
- renderer 参数解冻
- CLI gate
- TensorBoard / checkpoint 输出
- SAM2 / CoTracker / DETR 之类外部 perception 模型接入

## 后续 PR 队列

1. `V2-STABILITY-SCENARIO-002`
   - 扩展 cross-view / occlusion / perturbation / adversarial synthetic scenario。
2. `V2-STABILITY-DIAGNOSTICS-001`
   - 增加 `FailureModeClassifier`、slot transition matrix、identity confusion graph。
3. `V2-STABILITY-GATE-001`
   - 把 identity invariant 作为 hard gate，assignment / temporal 指标作为 soft gate。
4. `V2-STABILITY-CLI-001`
   - 暴露可复现 CPU synthetic benchmark CLI。
