# ObjGauss Object Emergence Model v1

> 状态: algorithm model mainline / ABI freeze candidate
> 最近更新: 2026-07-03
> 依赖:
> - `docs/architecture/objgauss-v1-kernel-contract.md`
> - `docs/architecture/objgauss-v1-object-emergence-plan.md`

## 目标

本阶段开始构建 ObjGauss 算法模型主线。当前训练阻塞不是 object
emergence 本身，而是 full renderer training 依赖的 torch / gsplat / CUDA /
NVIDIA driver 环境不可用。因此模型主线先拆成两个层级：

```text
Object Emergence Solver: PerceptionEvidence -> A[N,K] -> ObjectState
Full Renderer Loss: ObjectState -> Gaussian decode -> gsplat image loss
```

前者必须先在 dependency-free 环境中可训练、可导出、可审计；后者继续保持
`TRAIN-GSPLAT-MVP-001` suspended，直到环境满足 torch / gsplat / CUDA。

## 非目标

- 不把 full renderer training 伪装为已完成。
- 不引入 torch、gsplat、CUDA、SAM、DINO、CoTracker 或 Mamba。
- 不把 dynamic K proposal 直接变成每 step 自动改写 slot count。
- 不让 semantic promotion 绕过 quality gate。
- 不替换现有 Three.js / Spark / WebGPU viewer renderer。

## v1 模型定义

v1 的核心模型是：

```text
f_theta(EvidenceBatch) -> logits[N,K] -> softmax -> A[N,K]
```

其中：

```text
EvidenceBatch = {
  positions: R^{N x 3}
  features: R^{N x D}
  target_assignment: R^{N x K} optional
  frame_index: int
}

SolverState = {
  feature_weights: R^{D x K}
  position_weights: R^{3 x K}
  bias: R^K
  temperature: float
}

AssignmentPrediction = {
  logits: R^{N x K}
  assignment: R^{N x K}
  slot_mass: R^K
  top_slots: R^N
  confidence: R^N
}
```

`A[N,K]` 仍是唯一 assignment source。`object_id` 继续是从 `A` /
matching / export policy 派生的 renderer address，不是 primary state。

## 第一阶段实现顺序

### ALGOMODEL-SPEC-001

冻结本文件，明确算法主线与 full renderer training 的边界。

完成标准：

- 文档中能直接看出当前要训练的是 Object Emergence Solver。
- full renderer training 仍被明确记录为环境挂起。

### SOLVER-ABI-001

新增 dependency-free solver ABI：

```text
evidence_from_gaussian_cloud(...)
initialize_object_emergence_solver(...)
predict_object_emergence_assignment(...)
project_object_emergence_prediction(...)
```

完成标准：

- `predict_object_emergence_assignment(...)` 输出 normalized `A[N,K]`。
- `project_object_emergence_prediction(...)` 复用现有 ObjectState projection。
- 测试覆盖 shape validation、softmax row sum、slot mass diagnostics 和
  GaussianCloud -> ObjectState bridge。

### TRAINABLE-SOLVER-NP-001

在 `SolverState` 上实现 NumPy 训练器：

```text
L = L_render + lambda_object L_object + lambda_temporal L_temporal + lambda_balance L_balance
```

第一版只更新 solver weights，不训练 Gaussian geometry / opacity / rotation。

### DYNAMIC-K-UPDATE-001

把现有 proposal 抬升为 gated state update，但只允许在 epoch / artifact 边界改 K。

允许规则：

```text
birth: unmatched evidence mass > threshold
split: high entropy + spatial fragmentation
merge: high feature similarity + bbox overlap
death: persistent low mass
```

禁止规则：

- 不在每个 gradient step 中改变 K。
- 不静默改写已有 object id。
- 不删除低质量 slot 而不留 diagnostics。

### SEMANTIC-PROMOTION-001

semantic promotion 必须经过 quality gate：

```text
assignment confidence
object purity
temporal stability
fragmentation status
CLIP / naming score when available
```

未通过 gate 的 slot 保持 anonymous ObjectState，不晋升为 semantic object。

## 成功标准

阶段成功不以“渲染好看”为标准，而以以下证据为标准：

- `A[N,K]` 由 solver state 推理产生，而不是手写 object id。
- 训练前后 loss 和 stability metrics 可比较。
- 输出 artifact 能被现有 ObjectState Debug OS 审计。
- dynamic-K 和 semantic promotion 都有 gate，不绕过 evidence。

