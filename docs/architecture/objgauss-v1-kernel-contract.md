# ObjGauss v1 Kernel Contract

> 状态: frozen direction / ABI 级约束
> 最近更新: 2026-07-02
> 目的: 将 token-system 讨论压缩为仓库级 kernel contract，约束当前 v1
> 只实现对象状态内核，不把 v2 / v3 的研究假设提前带进工程架构。

## 结论

ObjGauss 是对象中心的世界状态系统，不是 Gaussian-first 系统。

v1 kernel 的定义是：

```text
Object State Kernel + Gaussian Renderer
```

本文是 Architecture Contract / Kernel ABI。`docs/myobjgausstoken/` 下的原始
token-system 讨论保留为 Research Spec。两者冲突时，以本文为准。

## 系统形态

v1 只有三层：

```text
Observation Layer -> World State Layer -> Rendering Layer
```

唯一允许的依赖方向是：

```text
PerceptionEvidence -> ObjectState -> GaussianToken
                          |
                          +-> temporal fields / state history
```

不要把 perception、object、Gaussian、dynamics 建成四套并列 token 系统。v1
kernel 只有一个 reasoning unit：

```text
ObjectState
```

## 观测层

观测层只负责把图像或视频转成结构化证据。

```text
PerceptionEvidence = {
  feature: R^D optional
  mask: R^{H x W}
  track_id: int optional
}
```

规则：

- 不负责对象身份。
- 不负责 3D 状态。
- 不负责 dynamics。
- 可以来自颜色 mask、SAM / SAM2、DINO / CLIP feature、CoTracker / TAPIR track，
  或更简单的 deterministic adapter。
- 重型 ML 依赖只能是 optional adapter，不能成为 kernel 强依赖。

## 世界状态层

世界状态层是系统核心。

```text
ObjectState = {
  id: persistent_id
  slot_prob: R^K
  centroid: R^3
  bbox: R^6
  feature: R^D
  velocity: R^3 optional
  acceleration: R^3 optional
  latent_motion: R^D optional
  confidence: float
}
```

规则：

- `ObjectState` 是最小稳定世界单元。
- `object_id` 是派生变量，不是 primary state。它由 assignment / matching /
  export policy 生成，用于 renderer、artifact 和 debug 寻址。
- `slot_prob` 是 soft assignment 的状态表达；hard `object_id` 不能和
  `slot_prob` 形成双真相。
- 时间通过 state history、replay buffer 和 `ObjectState` 上的 temporal fields 表达。
- 不引入与 `ObjectState` 平级的 `DynamicsToken`。
- 时间身份必须用 assignment / matching 处理，例如 soft matching、greedy
  matching 或 Hungarian matching；不能用 hard ID equality loss。

实现层可以把 `ObjectState` 拆成三个内部子状态，但 contract 对外仍保持一个对象：

```text
ObjectState = {
  semantic_state
  geometric_state
  temporal_state
}
```

这个拆分是为了避免 feature、bbox、velocity、confidence 和 latent motion 的 learning
signal 互相污染；不要把它暴露成三套外部 kernel。

## 渲染层

渲染层保存和展示可渲染 3D 表达。

```text
GaussianToken = {
  mu: R^3
  covariance: R^6
  color: R^3 or SH
  opacity: R^1
  object_id: int
}
```

规则：

- Gaussian 是 render primitive，不是 reasoning unit。
- 渲染只发生在 Gaussian artifact 层。
- 对象编辑通过派生的 `object_id` 和后续 object-aware chunk metadata 寻址
  Gaussian children。
- 每个对象的 Gaussian 数量天然是 ragged。不要要求全局 dense
  `R[B, T, K, N, D]` tensor contract。

## Kernel 函数

v1 kernel 对外只有三个概念函数：

```text
perceive(input_frames) -> list[PerceptionEvidence]
infer_object(evidence, previous_state=None) -> list[ObjectState]
decode_gaussian(objects, source_artifact=None) -> GaussianArtifact
```

边界：

- `perceive` 只返回观测事实。
- `infer_object` 创建或更新对象状态。
- `decode_gaussian` 产出可渲染 Gaussian artifact，或 object-aware compressed
  delivery artifact。

`infer_object` 的分阶段实现计划见
`docs/architecture/objgauss-v1-object-emergence-plan.md`。该计划把 object emergence
定义为 assignment optimization：

```text
f(PerceptionEvidence) -> A
A -> ObjectState
```

其中 `A` 是 soft assignment matrix。v1 从 fixed `K` 开始；object birth / merge /
split 在 fixed-K 稳定性可测之后，先作为 proposal 产出，不自动改写状态。

## Object Emergence Solver

`infer_object` 不能停留在抽象接口，v1 必须把它约束成 optimization problem。Solver
的正式规划见 `docs/architecture/objgauss-v1-object-emergence-plan.md`，核心 ABI 是：

```text
evidence[N] -> cost C[N,K] -> assignment A[N,K] -> ObjectState[K]
```

最低要求：

- `A` 是唯一 assignment source。
- clustering、slot attention、tracking 都只能作为 `A` 的初始化、cost 项或
  matching 约束。
- hard `object_id` 必须由 `A`、matching 和 export policy 派生。
- v1 bootstrap 可以复用 `ObjectField.probabilities()` 作为 Gaussian-level `A`。
- Sinkhorn、Hungarian、Gumbel-softmax、Slot Attention 等只能按 PR 明确引入，不能
  作为隐式依赖。

## v1 Loss Contract

如果实现 trainable v1 path，loss 只分三类：

```text
L_render = || I - render(decode_gaussian(ObjectState)) ||
L_object = matching(ObjectState_t, ObjectState_{t-1})
L_temporal = || centroid_t - centroid_{t-1} ||
```

总形式：

```text
L_v1 = L_render + lambda_object L_object + lambda_temporal L_temporal
```

规则：

- Rendering loss 是唯一必须 loss family。
- Object consistency 必须基于 assignment / matching，不基于 hard ID equality。
- Temporal smoothness 是弱正则，不是独立 dynamics model。
- 物理模型、diffusion world model、Mamba、temporal Transformer decoder 都在 v1 范围外；
  只有后续 ADR 批准后才能提升。

## 当前仓库映射

| Contract 概念 | 当前位置 | 说明 |
| --- | --- | --- |
| `PerceptionEvidence.mask` | `objgauss/core/masks.py` | Mask manifest 与 optional SAM adapter。 |
| `PerceptionEvidence.feature` | `objgauss/core/clip_scoring.py`，未来 feature adapter | CLIP / DINO 风格语义特征保持 optional。 |
| `ObjectState.slot_prob` | `objgauss/core/object_field.py` | 当前 Object Field soft slots。 |
| `ObjectState.id` / hard labels | `objgauss/core/objects.py`，`objgauss/core/clustering.py` | 当前 hard label 与 baseline clustering；它们应视为派生地址。 |
| Cross-view object evidence | `objgauss/core/projection.py`，`objgauss/core/semantic_slots.py` | Projection voting、slot alignment、semantic gates。 |
| `GaussianToken` | `objgauss/core/gaussian.py` | 当前 Gaussian table 与 field schema。 |
| Browser delivery | `objgauss/core/chunk_index.py`，`objgauss/core/quantization.py`，`src/ogcDecoder.js` | Object-aware chunk、LOD、quantized payload 与 browser decoder contract。 |

未来只有在代码真正需要共享状态对象时，才引入 `objgauss/core/object_state.py`。不要为了
占位创建空抽象。

## 非目标

本文不能用来合理化以下事项：

- 巨型目录重构。
- 强制引入 DINOv2 / SAM2 / CoTracker / Mamba。
- 全局 dense `B x T x K x N x D` tensor API。
- 替换 renderer。
- 发布 public demo asset。
- 把训练输出或大生成产物放进 git。

## v1 接受标准

一个 v1 implementation slice 符合本文，当且仅当：

- Observation evidence、ObjectState、Gaussian artifact 有独立 contract。
- Renderer-facing 数据仍是 Gaussian artifact，并可通过派生 `object_id` 寻址。
- `object_id` 不是 primary state，不和 `slot_prob` 形成双真相。
- 涉及时序时，对象一致性使用 assignment / matching。
- Dynamics 是 object temporal fields 或 state history，不是平级 token system。
- 大模型交付继续受 manifest、chunk、LOD 和 browser-ready artifact 边界约束。
