# ObjGauss Core Model Train / Validate Plan

> 状态: planning / CORE-MODEL-TRAIN-VALIDATE-001 target
> 日期: 2026-07-06
> 范围: 算法模型训练验证路线。本文只定义近期阶段，不实现训练代码、不启动 GPU
> 训练、不引入 diffusion / world model。

## 1. Core Model Boundary

当前核心模型先定义为 object binding / assignment 链路：

```text
Gaussian / AssignmentEvidence
  -> Assignment Solver v2
  -> A[N,K]
  -> ObjectState
  -> Gaussian decoder / renderer loss validation
```

本阶段先验证 `Evidence -> A -> ObjectState` 是否能稳定绑定同一世界对象，再把
checkpoint 接回现有 Gaussian decoder / renderer loss contract。

不把以下内容纳入近期核心模型：

- rollout model
- weak identity graph
- replay buffer
- diffusion / transformer world model
- self-generated data loop
- dynamic-K 自动更新
- per-Gaussian geometry 解冻

## 2. 为什么先做诊断和 Gate

renderer thaw 路线已经证明 opacity / scale path 可运行，但收益很弱。下一阶段的瓶颈是
assignment quality，而不是继续扩大 renderer 参数。

训练 loss 下降不能证明 object identity 学对了。必须先让系统能回答：

```text
失败是 slot swap、identity fragmentation、object merge、background absorption，
还是 temporal drift？
```

因此训练前必须补齐 diagnostics 和 identity-invariant gate。

## 3. Phase Plan

### Phase 0: 收口当前工作区

目标：

- 将 `V2-STABILITY-SCENARIO-002` 单独提交。
- 将既有 `objgauss-v2-mvp-world-model-plan.md` roadmap 改动单独提交或明确继续挂起。
- 避免 diagnostics / training PR 混入旧 planning diff。

完成条件：

- `docs/state/pr-queue.md` 中 `V2-STABILITY-SCENARIO-002` 的 `完成 commit` 回填。
- 工作区没有与下一阶段无关的 unstaged diff，或无关 diff 已明确记录为挂起。

### Phase 1: V2-STABILITY-DIAGNOSTICS-001

目标：

- 增加 deterministic diagnostics。
- 让 stability eval 能区分具体 failure mode。

产物：

```text
schema = objgauss-v2-stability-diagnostics-v1
```

最小对象：

- `slot_transition_matrix`
- `identity_confusion_graph`
- `FailureModeClassifier`

必须区分：

- `slot_swap`
- `identity_fragmentation`
- `object_merge`
- `background_absorption`
- `temporal_drift`

输入：

- `SyntheticStabilityScenarioFixture`
- 与 observation 对齐的 predicted assignment 或 predicted slots

边界：

- 不训练 solver。
- 不接 renderer loss。
- 不把 diagnostics 变成 hard gate。

### Phase 2: V2-STABILITY-GATE-001

目标：

- 将 identity invariance 设为 hard gate。
- 将 assignment entropy / purity / temporal coherence 保持为 soft diagnostics。

hard gate 通过条件：

- oracle identity 不跨槽漂移。
- adversarial swap 不导致 expected slot 互换。
- occlusion recovery 后 identity 能回到原 expected slot。
- merge / fragmentation / background absorption 可被明确报告。

边界：

- 不做 rollout model。
- 不改变 dynamic-K proposal-only 约束。
- 不用多指标投票替代 identity invariant。

### Phase 3: ASSIGNMENT-SOLVER-V2-TRAIN-001

目标：

- 实现可训练 `AssignmentSolverV2State`。
- 先只在 synthetic fixtures 上验证 fixed-K assignment training。

最小 state：

```text
AssignmentSolverV2State = {
  feature_centers: R[K,D]
  position_centers: R[K,3]
  slot_bias: R[K]
  step: int
}
```

assignment：

```text
Evidence[N] -> C[N,K] -> softmax(-C / temperature) -> A[N,K]
```

loss 使用现有 v2 helper：

- cluster
- entropy
- balance
- optional supervised CE

边界：

- 不启用 temporal / matching loss。
- 不接 GPU。
- 不接 renderer。
- 不引入 Slot Attention / Sinkhorn / OT。

### Phase 4: ASSIGNMENT-SOLVER-V2-EVAL-001

目标：

- 用 synthetic stability suite 验证训练前后。
- 证明 loss 下降不能替代 identity gate。

产物：

- training summary
- eval summary
- checkpoint roundtrip
- diagnostics before / after

成功条件：

- assignment training loss 下降。
- hard gate 通过，或失败可被 diagnostics 定位。
- diagnostics 不退化。
- checkpoint load 后 eval 结果可复现。

### Phase 5: ASSIGNMENT-V2-RENDER-JOINT-001

目标：

- 将 v2 solver checkpoint 接回现有 renderer validation path。

链路：

```text
A[N,K] -> ObjectState -> Gaussian decoder -> renderer loss
```

验收顺序：

1. CPU point renderer smoke。
2. gsplat smoke。
3. renderer-loss-contract 消费 v2 checkpoint。

成功条件：

- assignment gate 不退化。
- ObjectState eval 通过。
- renderer loss contract 可消费 v2 checkpoint。
- checkpoint / resume 可用。

边界：

- 不追求大幅 image loss。
- 不解冻 per-Gaussian geometry。
- 不把 renderer loss 作为绕过 identity gate 的理由。

### Phase 6: CORE-MODEL-TRAIN-VALIDATE-001

目标：

- 将核心模型推进到可训练、可验证、可失败定位的阶段。

成功标准：

- v2 assignment training loss 下降。
- synthetic stability hard gate 通过。
- failure diagnostics 可解释失败样例。
- real sample small smoke 不退化。
- ObjectState eval 通过。
- renderer joint smoke 通过。
- checkpoint / summary / boundary report 都可复现。

失败也可接受，但必须满足：

- failure mode 被明确分类。
- 失败 evidence 可复现。
- 下一步是修 assignment model / evidence / loss，而不是跳到 diffusion 或 world model。

## 4. PR Queue After This Plan

建议近期队列：

1. `V2-STABILITY-DIAGNOSTICS-001`
2. `V2-STABILITY-GATE-001`
3. `ASSIGNMENT-SOLVER-V2-TRAIN-001`
4. `ASSIGNMENT-SOLVER-V2-EVAL-001`
5. `ASSIGNMENT-V2-RENDER-JOINT-001`
6. `CORE-MODEL-TRAIN-VALIDATE-001`

`MODEL-V2-TRAINING-ROADMAP-001` 继续保持 post-core-model-validation，不应抢在以上队列
之前实施 diffusion / rollout / replay buffer。

## 5. Non-goals

本路线不能用来合理化：

- diffusion / rollout model 提前进入。
- weak identity graph 作为 truth source。
- replay buffer 存 generated futures。
- dynamic-K 自动 birth / merge / split。
- self-generated training loop。
- per-Gaussian geometry / camera 解冻。
- SAM2 / CoTracker / optical flow 成为 kernel 默认依赖。
- 改变 public demo / HF release 口径。
