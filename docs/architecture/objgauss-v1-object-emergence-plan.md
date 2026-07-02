# ObjGauss v1 Object Emergence Plan

> 状态: v1 implementation plan / KERNEL-001 草案
> 最近更新: 2026-07-02
> 依赖: `docs/architecture/objgauss-v1-kernel-contract.md`
> 目的: 将 `PerceptionEvidence -> ObjectState` 明确为 assignment optimization，
> 避免 slot attention、clustering、tracking 和动态对象数演化成互相竞争的模块集合。

## 核心结论

v1 的 object emergence 是一个 assignment solver 问题：

```text
f(PerceptionEvidence) -> A
A -> ObjectState
ObjectState -> GaussianToken
```

其中：

```text
A = assignment matrix, shape N x K
N = observation evidence units，或当前 bootstrap 路线里的 Gaussian records
K = object slots
```

当前仓库已经有 bootstrap 版本的 `A`：

```text
ObjectField.probabilities() -> A[N,K]
```

v1 先复用这条 Gaussian-level soft assignment 路线，再逐步接入真正的 image /
perception evidence。不要在第一阶段引入 DINOv2、SAM2、CoTracker、Mamba 或新的
renderer 路线。

## 系统规则

不要把 slot attention、clustering、tracking 实现成三套独立 object emergence 模块。
v1 中它们只能是同一个 assignment matrix `A` 的不同来源、约束或视图。

| 概念 | v1 角色 |
| --- | --- |
| Slot assignment | `A` matrix。 |
| Clustering | `A` 的初始化或 feature compactness objective。 |
| Tracking | `A` 或 pooled `ObjectState` 上的 temporal matching。 |
| ObjectState | 由 `A` 加权池化得到的世界状态。 |
| object_id | 从 `A` / matching / export policy 派生的 renderer 地址。 |

## Fixed K First

v1 从 fixed `K` 开始。

原因：

- 当前 `ObjectField` 已经是 fixed slot count。
- Fixed `K` 更容易写 deterministic tests。
- Dynamic `K` 同时混合 birth、merge、split、identity 和 confidence policy，过早引入会让
  v1 不可解释。

动态对象数只在 fixed-K 稳定后进入 proposal 层：

```text
fixed K -> effective K metrics -> inactive slot policy -> merge/split proposals
```

v1 不允许 dynamic-K proposal 自动改写 `object_id` 或已有 Gaussian artifact。

## KERNEL-001: Object Emergence Solver

### 输入

第一版输入可以是 Gaussian-level evidence：

```text
E = {
  feature_i
  position_i
  opacity_i optional
  color_i optional
  mask_vote_i optional
}
```

未来 image-level evidence 可以接入同一接口：

```text
PerceptionEvidence = {
  feature_i
  mask_i
  track_id_i optional
}
```

### Cost Function

Solver 先构造 cost matrix：

```text
C[i,k] =
  w_feature  * d_feature(feature_i, center_k)
+ w_spatial  * d_spatial(position_i, centroid_k)
+ w_mask     * d_mask(mask_or_vote_i, slot_k)
+ w_temporal * d_temporal(evidence_i, ObjectState_{t-1,k})
+ w_balance  * d_balance(slot_mass_k)
```

v1 最小实现可以只使用：

```text
C[i,k] = w_feature * d_feature + w_spatial * d_spatial
```

后续阶段再加入 mask / vote / temporal / balance 项。

### Assignment Solver

`A` 是唯一 assignment source。

允许的 v1 solver 顺序：

1. Bootstrap solver: 复用 `ObjectField.probabilities()`。
2. Cost-softmax solver:

   ```text
   A[i,k] = softmax_k(-C[i,k] / tau)
   ```

3. Balanced solver proposal: Sinkhorn / optimal transport，只在后续 PR 中显式引入。
4. Hard export: `object_id_i = argmax_k A[i,k]`，仅用于 renderer / artifact / debug。

Hungarian matching 只用于跨帧 `ObjectState` matching，不用于直接替代 soft assignment。

### Object Pooling

从 assignment 到对象状态：

```text
mass_k = sum_i A[i,k]
ObjectState_k.feature = sum_i A[i,k] * feature_i / mass_k
ObjectState_k.centroid = sum_i A[i,k] * position_i / mass_k
ObjectState_k.bbox = bbox({position_i | A[i,k] > threshold})
ObjectState_k.confidence = confidence(A[:,k], mass_k)
```

低质量 slot 必须显式输出：

```text
inactive_slot
low_confidence_slot
mixed_slot
```

不要静默删除 slot，也不要静默改写 object ids。

### Object Loss / Objective

v1 的 object objective 是：

```text
L_object =
  L_cluster
+ lambda_track   * L_track
+ lambda_entropy * L_entropy
+ lambda_balance * L_balance
+ lambda_purity  * L_purity
```

其中：

```text
L_cluster = sum_i sum_k A[i,k] * || feature_i - center_k ||
L_track = matching_cost(ObjectState_t, ObjectState_{t-1})
L_entropy = mean_i H(A[i,:])
L_balance = collapse_penalty(slot_mass)
L_purity = disagreement(slot, mask_or_vote_evidence)
```

解释：

- `L_cluster` 让同一对象内部 feature / spatial evidence 更一致。
- `L_track` 让跨帧或跨观测的对象身份稳定。
- `L_entropy` 不是越低越好；它用于暴露 assignment sharpness，需要配合
  `L_balance` 防止全部 collapse 到一个 slot。
- `L_balance` 防止 slot collapse 和大量 inactive slots。
- `L_purity` 只在有 mask / vote evidence 时启用。

### Training / Non-Training Loop

v1 先实现非训练闭环，再考虑 trainable solver。

非训练闭环：

```text
GaussianCloud / PerceptionEvidence
  -> feature extraction
  -> bootstrap A
  -> ObjectState pooling
  -> stability metrics
  -> hard object_id export
  -> Gaussian artifact / chunk delivery
```

可训练闭环只在非训练闭环通过后进入：

```text
frames
  -> perceive()
  -> solver(C -> A)
  -> ObjectState pooling
  -> decode_gaussian()
  -> render()
  -> L_render + L_object
```

v1 不要求 trainable solver 才算完成。稳定、可解释、可审计的 assignment pipeline 优先。

## v1 阶段计划

### Phase 0: Kernel Contract

状态: done。

交付物：

- `docs/architecture/objgauss-v1-kernel-contract.md`。

边界：

- 不改代码。
- 不改 renderer。
- 不引入新 ML 依赖。

### Phase 1: Assignment Matrix And Object Pooling

状态: done / `OBJECT-ASSIGNMENT-001`。

目标：

建立最小 object emergence kernel：

```text
A[N,K] + evidence[N] -> ObjectState[K]
```

实现目标：

- 已新增 `objgauss/core/object_state.py`。
- 已接受 `ObjectField.probabilities()` 作为第一版 `A` 来源。
- 已先对 Gaussian evidence 做 object pooling；image-token evidence 留给后续阶段。
- 已明确 `object_id` 是由 `A` 派生的 export address。

必须输出：

- `ObjectState.id`。
- `ObjectState.slot_prob` 或 slot mass summary。
- `ObjectState.centroid`。
- `ObjectState.bbox`。
- `ObjectState.feature`。
- `ObjectState.confidence`。
- inactive / low-confidence / mixed slot diagnostics。

验证：

- 小型 Gaussian fixture unit tests: done。
- deterministic output ordering: done。
- empty / low-confidence / mixed / uniform / noisy assignment handling: done。
- hard `object_id` export 可复现: done。
- 不依赖 SAM2、DINOv2、CoTracker、Mamba 或 renderer: done。

非目标：

- 不做 temporal matching。
- 不做 dynamic `K`。
- 不做 trainable slot attention。

### Phase 2: Object Stability Metrics

状态: done / `OBJECT-STABILITY-001`。

目标：

判断 pooled object states 是否足够稳定，可以交给后续 pipeline。

指标：

```text
assignment_confidence = 1 - normalized_entropy(A)
slot_mass = sum_i A[i,k]
effective_slots = exp(entropy(slot_mass))
object_purity = agreement between slot and mask / vote evidence
slot_collapse = mass concentration above threshold
inactive_slots = slots with mass below threshold
mixed_slots = slots with high assignment entropy
```

实现目标：

- 已新增 `ObjectStabilityReport` 和 `object_state_stability_report(...)`。
- 已基于 Phase 1 的 `ObjectStateProjection.assignment` 计算 assignment confidence、
  slot mass、effective slots、inactive / low-confidence / mixed slots、slot collapse
  和可选 `purity_labels` object purity。
- `temporal_drift = || centroid_t - centroid_{t-1} ||` 留给 Phase 3 matching；本阶段不伪造
  跨观测 identity 信号。

验证：

- 指标 bounded and deterministic: done。
- collapse case 可见：单 slot 吞噬、uniform assignment、empty slots、noisy slots: done。
- 报告不能把弱 semantic route promote 成稳定能力: done；report 只输出 diagnostics，不改
  promotion threshold。

非目标：

- 不自动修改 `K`。
- 不放宽 promotion threshold。

### Phase 3: Temporal Matching

状态: done / `OBJECT-TEMPORAL-MATCH-001`。

目标：

通过 matching 稳定跨帧或跨观测对象身份，而不是依赖 hard ID equality loss。

核心操作：

```text
match(ObjectState_t, ObjectState_{t-1}) -> matched pairs + unmatched states
```

匹配 cost：

```text
M[p,q] =
  a_centroid * || centroid_p - centroid_q ||
+ a_bbox     * (1 - IoU(bbox_p, bbox_q))
+ a_feature  * (1 - cosine(feature_p, feature_q))
+ a_mass     * | mass_p - mass_q |
+ a_track    * track_mismatch_penalty
```

允许的第一版：

- 已实现 dependency-free greedy matching。
- `ObjectTemporalMatchReport` 显式输出 matched pairs、unmatched previous /
  current、ignored inactive states、mean / max temporal drift 和 cost matrix。
- 当前不引入 Hungarian；只有后续大规模 matching 证明需要时再单独立项。

验证：

- slot permutation 后 identity 仍稳定: done。
- unmatched states 显式输出: done。
- temporal drift 报告出来，不被静默吞掉: done。

非目标：

- 不引入 CoTracker 依赖。
- 不做 learned tracker。
- 不做 dynamics model。

### Phase 4: Gaussian Decode / Delivery Binding

状态: done / `OBJECT-GAUSSIAN-BINDING-001`。

目标：

让 pooled `ObjectState` 能服务渲染与交付，但不改变 renderer。

当前 bootstrap 路线：

```text
ObjectState.id -> derived object_id -> GaussianToken children -> chunk / LOD metadata
```

实现目标：

- 已新增 `object_state_delivery_summary(...)`，将 `ObjectStateProjection` 汇总为
  renderer-facing metadata：derived `object_id` 来源、Gaussian children count、
  per-state summary、stability 摘要和可选 chunk binding。
- 已新增 `bind_object_states_to_artifact(...)`，把上述 summary 附到现有 Gaussian
  artifact dict，不替换 manifest schema、不改变 viewer route。
- hard `object_id` 仍只作为 renderer/export address。
- `ObjectState` 保持 reasoning state，`GaussianToken` 保持 render child。

验证：

- object-state summary 能绑定 Gaussian children 和 chunk metadata: done。
- viewer routes 继续消费 Gaussian artifact，不消费 ObjectState tensor: done；本阶段只附加
  metadata，不改前端 route。
- full diagnostic PLY 不会变成默认 browser route: done；本阶段不修改 manifest validator
  的 browser-ready 规则。

非目标：

- 不替换 renderer。
- 不建立 dense `B x T x K x N x D` tensor API。

### Phase 5: Dynamic K Proposal Policy

状态: done / `OBJECT-DYNAMIC-K-PROPOSAL-001`。

目标：

fixed-K assignment 和 temporal matching 可测稳定后，再引入 object birth / merge /
split。

proposal policy：

```text
inactive slot -> candidate removal
high-entropy mixed slot -> split proposal
near-duplicate objects -> merge proposal
new unmatched evidence -> birth proposal
```

规则：

- v1 只输出 proposal；实现为 `DynamicKProposalReport`。
- 自动 state mutation 需要后续 ADR 或明确 PR。
- 每个 proposal 必须包含 evidence、metric threshold 和 rejection path。
- proposal 不能静默改写已有 Gaussian artifact 使用的 object ids。

验证：

- proposal generation deterministic: done。
- merge / split decisions auditable: done。
- remove inactive、split mixed、merge duplicate、birth unmatched proposal 均有测试覆盖: done。
- 拒绝 proposal 后当前 fixed-K 状态仍可继续运行: done；report 不修改 projection /
  assignment / artifact。

非目标：

- 不做 open-ended dynamic slot attention。
- 不从黑盒 temporal model 隐式 birth object。

## 实施顺序

1. `KERNEL-SOLVER-SPEC-001`: 已完成。冻结 Object Emergence Solver 的 cost、
   assignment、pooling、export address 和非训练闭环。
2. `OBJECT-ASSIGNMENT-001`: done。实现 assignment matrix 与 ObjectState pooling
   contract。
3. `OBJECT-STABILITY-001`: done。实现 object-state stability metrics 和 collapse
   diagnostics。
4. `OBJECT-TEMPORAL-MATCH-001`: done。实现跨观测 state matching。
5. `OBJECT-GAUSSIAN-BINDING-001`: done。将 object-state summaries 绑定到现有
   Gaussian artifact / chunk delivery metadata。
6. `OBJECT-DYNAMIC-K-PROPOSAL-001`: done。产出 birth / merge / split proposals，
   不做 automatic mutation。

## v1 接受标准

v1 object emergence 可接受，当且仅当：

- 单一 assignment matrix `A` 驱动 object pooling。
- Fixed-K 行为 deterministic and tested。
- `object_id` 是从 `A` / matching / export policy 派生的 renderer address。
- Object stability metrics 能暴露 collapse、drift、purity、inactive slots。
- Temporal identity 使用 matching，不使用 hard ID equality。
- Dynamic `K` 在证据充分前只作为 proposal layer。
- Renderer-facing 数据仍是通过 `object_id` 寻址的 Gaussian artifacts。
