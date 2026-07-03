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
L_solver =
  lambda_assignment * L_assignment
+ lambda_entropy    * L_entropy
+ lambda_balance    * L_balance
+ lambda_temporal   * L_temporal
```

第一版只更新 solver weights，不训练 Gaussian geometry / opacity / rotation。

实际落地顺序：

```text
object-aware GaussianCloud
  -> object_id one-hot target
  -> ObjectEmergenceEvidence
  -> train_object_emergence_solver()
  -> ObjectEmergenceSolverState
  -> AssignmentPrediction A[N,K]
  -> ObjectState projection / Debug OS handoff
```

当前 `TRAINABLE-SOLVER-NP-001` 是 CPU / NumPy 训练管道，不占用 GPU，不调用 torch /
gsplat，也不运行 full 3DGS renderer。它优化的是 `SolverState` 的 feature weights、
position weights 和 bias，而不是直接优化每帧 assignment logits。

`L_render` / `image_render_loss` 仍属于后续 full renderer loss producer，不在本阶段
伪装成已完成。

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

## 训练管道资源策略

### 当前可执行阶段

允许在本环境执行：

- dependency-free NumPy solver training。
- 小型 object-aware PLY smoke。
- summary / artifact contract 写入 `/tmp` 或 ignored `outputs/`。
- Debug OS 加载和审计训练结果。

禁止在本环境直接执行：

- 长时间 full renderer training。
- 需要 torch / gsplat / CUDA 的真实 3DGS rasterizer loss。
- 写入 git 的 checkpoint、训练输出、rendered image 或大资产。

### GPU 恢复前置条件

当后续进入 full renderer training 或 GPU solver training 时，必须先通过 preflight：

```text
torch import ok
gsplat import ok
torch.cuda.is_available() == true
nvidia-smi 可读取 total / free memory
```

显存策略固定为：

```text
reserve_vram_gb = 1
usable_vram_gb = max(total_vram_gb - reserve_vram_gb, 0)
```

调度规则：

- 必须至少保留 1GB GPU 显存给系统和桌面进程。
- 除预留 1GB 外，其余显存可以用于训练 batch / chunk / tile / cache。
- 如果 preflight 无法证明可用显存，停止，不启动训练。
- 如果训练预计进入长时间 GPU run，停止在 handoff 点，由 Owner 明确选择运行窗口。
- `--skip-gpu-preflight` 只允许用于诊断，不允许作为正常训练入口。

当前 Codex 会话到达 GPU full renderer training 前必须暂停；不能把
`TRAIN-GSPLAT-MVP-001` 从 suspended 改成 done。

## 成功标准

阶段成功不以“渲染好看”为标准，而以以下证据为标准：

- `A[N,K]` 由 solver state 推理产生，而不是手写 object id。
- 训练前后 loss 和 stability metrics 可比较。
- 输出 artifact 能被现有 ObjectState Debug OS 审计。
- dynamic-K 和 semantic promotion 都有 gate，不绕过 evidence。
