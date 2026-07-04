# ObjGauss Assignment Solver v2 Contract

> 状态: frozen contract / ASSIGNMENT-SOLVER-V2-CONTRACT-001
> 日期: 2026-07-04
> 依赖: `docs/architecture/objgauss-v1-kernel-contract.md`,
> `docs/architecture/objgauss-v1-object-emergence-plan.md`
> 范围: 算法模型 contract。本文不修改训练代码、不启动 GPU 训练、不引入新依赖。

## 1. 背景

当前 v1 训练闭环已经跑到：

```text
Gaussian evidence -> A[N,K] -> ObjectStateProjection
  -> Gaussian decode -> gsplat image loss
```

已完成的 renderer thaw 事实：

- `TRAIN-RUN-004`: solver + colors joint GPU run 通过 ObjectState eval。
- `TRAIN-RUN-005`: object-level opacity path 可用，但 image loss 收益很弱。
- `TRAIN-RUN-006`: scale-only path 可用，但 image loss 收益仍很弱。

结论：

```text
renderer thaw path is operational
assignment quality is now the bottleneck
```

所以下一步不继续扩大 renderer 参数解冻，而是回到：

```text
PerceptionEvidence / GaussianEvidence -> assignment A -> ObjectState
```

## 2. v1 Solver 现状

当前代码中的 v1 solver 是：

```text
ObjectEmergenceEvidence
  -> linear logits
  -> softmax temperature
  -> A[N,K]
```

对应实现：

- `ObjectEmergenceEvidence`
- `ObjectEmergenceSolverConfig`
- `ObjectEmergenceSolverState`
- `ObjectEmergenceAssignmentPrediction`

v1 优点：

- fixed-K 明确。
- checkpoint / resume 已有。
- soft assignment `A[N,K]` 已经接入 ObjectState pooling。
- 可被 renderer loss 反传更新。

v1 不足：

- logits 是线性投影，不显式表达 cost function。
- clustering、balance、temporal、matching 还没有作为可审查 cost / loss family 拆开。
- evidence 仍主要来自 Gaussian records，不是真正 frames / masks / tracks evidence。
- solver state 不能清楚表达 slot prototype、temporal memory 或 matching policy。

## 3. v2 核心决策

v2 solver 不改变 kernel 主链路：

```text
Evidence[N] -> C[N,K] -> A[N,K] -> ObjectState[K]
```

v2 只升级 `C -> A` 的表达能力和评估边界。

不允许把以下内容变成并列模块：

```text
Slot module + Cluster module + Tracker module + Renderer module
```

它们只能是同一个 assignment system 的 cost term、loss term、metric 或 downstream
consumer。

## 4. v2 Data Contract

### 4.1 Evidence

v2 的输入统一为 evidence batch：

```text
AssignmentEvidenceBatch = {
  positions: R[N,3]
  features: R[N,D]
  frame_index: int
  mask_votes: R[N,M] optional
  track_hints: R[N] optional
  target_assignment: R[N,K] optional
  source: str
}
```

规则：

- `positions` 和 `features` 是最低必需字段。
- `mask_votes`、`track_hints` 只能作为 optional evidence。
- `target_assignment` 只用于 bootstrap / supervised smoke，不是生产推理依赖。
- `object_id` 不进入 evidence primary key；它只能作为 target 或 export label。

### 4.2 Solver State

v2 state schema 建议：

```text
schema = objgauss-assignment-solver-state-v2

AssignmentSolverV2State = {
  config: AssignmentSolverV2Config
  feature_centers: R[K,D]
  position_centers: R[K,3]
  slot_bias: R[K]
  temporal_memory: R[K,D_t] optional
  step: int
  source: str
}
```

config：

```text
AssignmentSolverV2Config = {
  slots: int
  feature_dim: int
  position_dim: int = 3
  solver_family: "cost-softmax-assignment-v2"
  temperature: float
  cost_terms: list[str]
  balance_policy: str
  temporal_policy: str
  matching_policy: str
}
```

v2 首个实现只允许：

```text
solver_family = cost-softmax-assignment-v2
cost_terms = ["feature", "position", "slot_bias"]
balance_policy = "loss-only-v1"
temporal_policy = "disabled"
matching_policy = "disabled"
```

temporal / matching 只能在后续 PR 中打开。

### 4.3 Prediction

v2 prediction schema 建议：

```text
schema = objgauss-assignment-prediction-v2

AssignmentPredictionV2 = {
  assignment: R[N,K]
  cost: R[N,K] optional
  logits: R[N,K]
  slot_mass: R[K]
  confidence: R[N]
  mean_normalized_entropy: float
  effective_slots: float
  diagnostics: list[str]
}
```

规则：

- `assignment` 是唯一 assignment source。
- hard slot / hard object id 只能由 `argmax(A)` 派生。
- `cost` 可以在 summary 中省略大矩阵，但必须有 cost term diagnostics。

## 5. v2 Cost Function

v2 把 solver 显式写成 cost system：

```text
C[i,k] =
  w_feature  * d_feature(feature_i, feature_center_k)
+ w_position * d_position(position_i, position_center_k)
+ w_bias     * slot_bias_k
+ w_mask     * d_mask(mask_vote_i, slot_k)          optional
+ w_temporal * d_temporal(evidence_i, memory_k)     optional
+ w_balance  * d_balance(slot_mass_k)               optional
```

首个 v2 implementation 只启用：

```text
C[i,k] =
  w_feature  * || normalize(feature_i) - normalize(feature_center_k) ||^2
+ w_position * || normalize(position_i) - normalize(position_center_k) ||^2
+ slot_bias_k
```

assignment：

```text
A[i,k] = softmax_k(-C[i,k] / temperature)
```

说明：

- 这是对当前 linear-softmax solver 的可解释替代。
- v2 先使用 center / prototype 表达 object slot，不直接上 Slot Attention。
- Sinkhorn / OT 只作为后续 balanced solver proposal，不进入首个 v2。

## 6. v2 Loss Contract

v2 object loss 拆成明确 family：

```text
L_object_v2 =
  lambda_cluster  * L_cluster
+ lambda_entropy  * L_entropy
+ lambda_balance  * L_balance
+ lambda_temporal * L_temporal
+ lambda_matching * L_matching
+ lambda_supervised * L_assignment_supervised optional
```

首个 v2 training MVP 只允许：

```text
L_cluster + L_entropy + L_balance + optional L_assignment_supervised
```

暂不启用：

- `L_temporal`
- `L_matching`
- renderer geometry loss 对 solver v2 的强依赖
- dynamic-K birth / merge / split

定义：

```text
L_cluster = mean_i sum_k A[i,k] * C_cluster[i,k]
L_entropy = mean_i H(A[i,:])
L_balance = mean_k (slot_mass_k / N - 1/K)^2
L_assignment_supervised = CE(A, target_assignment)
```

`L_entropy` 不能单独优化。它必须和 `L_balance` 或 supervised target 一起使用，避免
slot collapse。

## 7. ObjectState Pooling

v2 不改变 pooling ABI：

```text
ObjectStateProjection = pool(evidence, A)
```

最低输出仍然是：

```text
ObjectState_k = {
  id: derived persistent_id
  slot_prob: A[:,k]
  centroid: weighted position
  bbox: weighted / threshold bbox
  feature: weighted feature
  confidence: function(slot_mass, entropy, purity)
}
```

新增要求：

- inactive slot 必须显式标记。
- mixed slot 必须通过 diagnostics 暴露。
- hard `object_id` 只允许在 export / renderer address 阶段生成。

## 8. Metrics And Gates

v2 assignment evaluation 至少输出：

```text
mean_normalized_entropy
assignment_confidence
effective_slots
max_dominant_slot_mass_fraction
slot_collapse
object_purity
temporal_mean_drift
id_stability optional
```

首个 v2 gate 沿用当前 ObjectState eval 阈值：

```text
mean_normalized_entropy <= 0.60
object_purity >= 0.80
slot_collapse = false
```

promotion gate 目标：

```text
mean_normalized_entropy <= 0.30
object_purity >= 0.85
slot_collapse = false
```

v2 solver 只有在 assignment metrics 变好或至少不退化时，才允许重新接 renderer joint
training。

## 9. Checkpoint And Migration

v2 checkpoint 必须包含：

```text
schema = objgauss-assignment-solver-v2-checkpoint

checkpoint = {
  solver_state
  training
  evaluation
  source
  export_policy
}
```

迁移规则：

- v1 `ObjectEmergenceSolverState` 不自动升级为 v2。
- 可以提供 explicit adapter：

  ```text
  v1 linear weights -> v2 slot centers / slot bias init
  ```

- adapter 输出必须把 `source` 标记为 `v1_linear_softmax_adapter`。
- 旧 v1 checkpoint loader 继续存在，不被 v2 替换。

## 10. Implementation Order

后续 PR 顺序固定为：

```text
1. OBJECT-LOSS-V2-001
2. ASSIGNMENT-FRAMES-EVIDENCE-001
3. TRAIN-ASSIGNMENT-MVP-001
4. EVAL-ASSIGNMENT-STABILITY-001
5. ASSIGNMENT-RENDER-JOINT-001
6. DYNAMIC-K-PROPOSAL-001
```

解释：

- 先拆 loss，再扩大 evidence。
- 先 fixed-K assignment MVP，再接 renderer joint loss。
- dynamic-K 最后，只做 proposal，不自动改写状态。

## 11. Non-Goals

本 contract 明确不做：

- 不启动 GPU 训练。
- 不解冻 Gaussian means / per-Gaussian scales / quats / cameras。
- 不引入 dynamic-K birth / merge / split。
- 不把 SAM / CLIP / CoTracker / optical flow 作为 kernel 默认依赖。
- 不引入 Slot Attention、Mamba、diffusion 或 Transformer world model。
- 不把 hard `object_id` 作为训练主状态。
- 不替换 browser viewer 或 renderer delivery path。

## 12. Completion Criteria

`ASSIGNMENT-SOLVER-V2-CONTRACT-001` 完成后，仓库应满足：

- 有一份可审查的 v2 solver contract。
- 当前 v1 linear-softmax solver 的保留边界清楚。
- v2 的 evidence、state、prediction、loss、metrics、checkpoint 和 migration
  都有明确形状。
- 后续 PR 能按本文顺序实施，不需要重新讨论 renderer thaw。
