# ObjGauss v1 Renderer Field Unfreeze Plan

> 状态: frozen plan / RENDER-FIELD-UNFREEZE-PLAN-001
> 日期: 2026-07-04
> 范围: 算法模型规划。本文只定义第一个最小 renderer 参数解冻切片，不修改训练代码，不启动 GPU 训练，不提交 checkpoint / summary / TensorBoard / rendered image 产物。

## 1. 当前事实

当前可训练闭环已经到达：

```text
PerceptionEvidence -> A[N,K] -> ObjectStateProjection
  -> Gaussian decode -> gsplat image_render_loss
```

`TRAIN-RUN-004` 的事实输入：

- run-level total loss: `0.219158 -> 0.170798`
- run-level image render loss: `0.018028 -> 0.017223`
- run-level object loss: `0.278156 -> 0.208460`
- ObjectState eval: `objectstate_eval_pass`
- `mean_normalized_entropy=0.237402`
- `assignment_confidence=0.762598`
- `object_purity=0.868178`
- renderer boundary: `full_3dgs_solver_decoder_joint_training_ready`
- `upgrade_blockers=[]`

当前 trainable / frozen contract：

```text
trained:
  solver.feature_weights
  solver.position_weights
  solver.bias
  decoder.object_colors

frozen:
  means
  quats
  scales
  opacities
  cameras
  dynamic_k
```

因此下一步不是直接解冻 geometry，而是先给 renderer 增加一个最小、可回滚、可诊断的 object-level 标量参数。

## 2. 候选字段审查

| 字段 | 粒度 | 结论 | 原因 |
| --- | --- | --- | --- |
| `decoder.object_opacity_logits` | object-level `K` | 第一优先级 | 只影响 alpha / visibility 强度，不移动几何、不改相机、不改拓扑；参数量是 `K`，不是 Gaussian 数 `N`。 |
| `decoder.object_scale_logits` | object-level `K` 或 `K x 3` | 第二优先级 | 会改变 splat footprint，容易把 geometry / visibility 问题混进 render loss；应在 opacity gate 通过后再做。 |
| per-Gaussian opacity | Gaussian-level `N` | 暂缓 | 参数量随点数增长，容易过拟合 target image，并绕开 ObjectState reasoning unit。 |
| means / positions | Gaussian-level `N x 3` | 禁止作为第一刀 | 会直接训练几何，和 assignment / camera / scale 的 credit assignment 混在一起。 |
| scales / quats | Gaussian-level | 暂缓 | 会改变 footprint / covariance，需要独立的 stability 和 artifact gate。 |
| cameras | frame-level | 暂缓 | 当前目标图像和多视角约束太弱，容易用 camera drift 掩盖模型问题。 |
| dynamic-K | topology-level | 暂缓 | 会改变 ObjectState 数量和 checkpoint 结构，必须等 renderer 字段 gate 稳定后再进入 birth / merge / split。 |

## 3. 决策

第一解冻字段：

```text
decoder.object_opacity_logits: R^K
```

对外解码：

```text
object_opacity_scale = clamp(sigmoid(object_opacity_logits), min=0.05, max=1.0)
gaussian_opacity_i = default_opacity * sum_k A[i,k] * object_opacity_scale[k]
```

实现约束：

- `ObjectState` 仍是唯一 reasoning unit。
- `object_opacity_logits` 属于 Gaussian decoder state，不属于 solver state。
- `object_id` 仍是 derived renderer address，不成为 primary state。
- source Gaussian opacity 继续冻结；本切片只学习 object-level multiplier。
- geometry、camera、dynamic-K 继续冻结。

## 4. 需要的代码切片

### DECODER-OPACITY-CONTRACT-001

目标：把 object-level opacity 写进 decoder state 和 checkpoint ABI。

需要变更：

- `ObjectStateGaussianDecoderState` 增加可选 `object_opacity_logits`，旧 checkpoint 缺字段时默认全 `0.0`。
- `as_dict()` / `object_state_gaussian_decoder_state_from_dict(...)` 保持向后兼容。
- `decode_gaussian_from_object_state(...)` 支持用 `assignment @ object_opacity_scale` 生成 per-Gaussian opacity。
- `ObjectStateGaussianDecode.as_dict()` 将 `decoder.object_opacity_logits` 标为 differentiable field，仅在显式启用时从 frozen list 移除 `opacities`。

### TRAIN-DECODER-OPACITY-001

目标：让 renderer API 暴露 opacity 梯度，但仍保持最小训练面。

需要变更：

- CPU renderer 和 gsplat renderer 的 loss result 增加 `gradient_decoder_opacity_logits` 或等价 object-level opacity gradient。
- `solver-decoder-mvp` 增加显式开关：`--train-decoder-opacity`。
- 增加独立 learning rate：`--decoder-opacity-learning-rate`。
- 增加 opacity regularization：默认把 object opacity scale 拉回 `1.0`，避免通过透明化逃避重建。
- 首个 smoke 建议冻结 solver，或把 solver learning rate 降到 near-zero gate，避免 assignment sharpening 和 opacity 同时变化。

### TRAIN-RUN-005-OPACITY-SMOKE

目标：从 run-004 final checkpoint resume，做一次受控 gsplat smoke。

建议配置：

```text
resume_checkpoint:
  outputs/training/train-run-004-solver-temp05-gsplat/final-checkpoint.json

renderer:
  gsplat
  max_points=128
  frames=2
  image_size=16x16
  vram_reserve_gb=1

optimization:
  train decoder.object_opacity_logits only
  keep solver assignment fixed if implementation supports freeze
  keep decoder.object_colors fixed for first isolated smoke if implementation supports freeze
  iterations=60..100
  checkpoint_every=20
  opacity_lr=0.01..0.03
  opacity_reg_weight=0.01
```

如果当前 CLI 暂时不支持冻结 solver / colors，则本计划要求先补冻结开关，不允许把 opacity 解冻和 solver/color 更新混成同一个不可解释实验。

## 5. 成功门槛

训练 smoke 必须同时满足：

- `renderer-loss-contract` 使用 run-level evidence，状态保持 `full_3dgs_solver_decoder_joint_training_ready` 或进入更明确的 `renderer_opacity_training_ready`。
- `run_image_render_loss_decreased=true`。
- image render loss 至少比 run-004 final `0.017223` 有可测下降；建议首轮目标为 `>= 0.0001` absolute improvement 或 `>= 0.5%` relative improvement。
- `eval-objectstate --require-pass` 继续通过。
- `object_purity >= 0.85`。
- `mean_normalized_entropy <= 0.30`。
- `slot_collapse=false`。
- opacity scale 不饱和：`opacity_scale_min >= 0.05`，`opacity_scale_max <= 1.0`，且落在 clamp 边界的 slot 数不能超过 1 个。
- TensorBoard 至少写出 `loss/image_render`、`loss/object`、`loss/entropy`、`loss/balance`、`decoder/opacity_scale_mean`、`decoder/opacity_scale_min`、`decoder/opacity_scale_max`。

## 6. 回滚条件

出现任一条件就回滚该字段解冻，不继续扩大到 scale / geometry：

- `image_render_loss` 没有 run-level 下降。
- ObjectState eval 从 pass 退化为 fail。
- `object_purity < 0.85`。
- `mean_normalized_entropy > 0.30`。
- 出现 `slot_collapse=true`。
- opacity 通过透明化逃避 loss：多个 slot 被压到 `0.05` 附近。
- `renderer-loss-contract` 出现新的 `upgrade_blockers`。
- before / after render 出现明显 visibility artifact，但 loss 下降。

## 7. 下一步顺序

```text
1. DECODER-OPACITY-CONTRACT-001
2. TRAIN-DECODER-OPACITY-001
3. TRAIN-RUN-005-OPACITY-SMOKE
4. RENDER-FIELD-SCALE-PLAN-001
```

在 `TRAIN-RUN-005-OPACITY-SMOKE` 通过前，不进入 means / scales / quats / cameras / dynamic-K。

## 8. 非目标

- 不训练 Gaussian means / positions。
- 不训练 rotation / quaternion。
- 不训练 covariance / scale。
- 不优化 camera。
- 不引入 dynamic-K birth / merge / split。
- 不把 per-Gaussian opacity 作为第一批 trainable field。
- 不替换 browser viewer renderer。
- 不提交 ignored `outputs/` 或 `/tmp` 训练产物。
