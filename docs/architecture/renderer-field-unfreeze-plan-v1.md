# ObjGauss v1 Renderer Field Unfreeze Plan

> 状态: frozen plan / RENDER-FIELD-UNFREEZE-PLAN-001 + RENDER-FIELD-SCALE-PLAN-001
> 日期: 2026-07-04
> 范围: 算法模型规划。本文定义 renderer 参数逐步解冻切片，不修改训练代码，不启动 GPU 训练，不提交 checkpoint / summary / TensorBoard / rendered image 产物。

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

`TRAIN-RUN-005-OPACITY-SMOKE` 的事实输入：

- 正式 run: `outputs/training/train-run-005-opacity-gsplat-image/`
- run-level image render loss: `0.0172239579 -> 0.0172238834`
- run-level object loss: `0.2084604055 -> 0.2084604204`
- ObjectState eval: `objectstate_eval_pass`
- `mean_normalized_entropy=0.237402`
- `assignment_confidence=0.762598`
- `object_purity=0.868178`
- opacity scale: `0.99752742 -> 0.99752766`
- renderer boundary: `full_3dgs_solver_decoder_joint_training_ready`
- `upgrade_blockers=[]`

结论：run-005 证明 opacity training path / checkpoint / TensorBoard / eval gate 可用，
但收益很弱。两次带 object loss 权重的 opacity 配置未通过 image render loss decrease
gate，因此它不能作为直接解冻 geometry 的证据。

run-005 accepted smoke 的 trainable / frozen contract：

```text
trained:
  solver.feature_weights
  solver.position_weights
  solver.bias
  decoder.object_colors
  decoder.object_opacity_logits

frozen:
  means
  quats
  scales
  source_opacities
  cameras
  dynamic_k
```

因此下一步仍不是直接解冻 geometry，而是给 renderer 增加第二个最小、可回滚、
可诊断的 object-level 标量参数。

## 2. 候选字段审查

| 字段 | 粒度 | 结论 | 原因 |
| --- | --- | --- | --- |
| `decoder.object_opacity_logits` | object-level `K` | 第一优先级 | 只影响 alpha / visibility 强度，不移动几何、不改相机、不改拓扑；参数量是 `K`，不是 Gaussian 数 `N`。 |
| `decoder.object_scale_log_offsets` | object-level `K` | 第二优先级 | 只学习每个 ObjectState 的 isotropic footprint multiplier，参数量是 `K`；比 per-Gaussian scale 小得多，但仍会影响 visibility / covariance，所以必须在 opacity path smoke 后单独规划。 |
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

- `ObjectStateGaussianDecoderState` 增加可选 `object_opacity_logits`，旧 checkpoint 缺字段时按 disabled / `constant-opacity-v1` 加载，避免历史 checkpoint 被隐式改成半透明。
- `as_dict()` / `object_state_gaussian_decoder_state_from_dict(...)` 保持向后兼容。
- `decode_gaussian_from_object_state(...)` 支持在显式传入 `object_opacity_logits` 时用 `assignment @ object_opacity_scale` 生成 per-Gaussian opacity。
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
1. DECODER-OPACITY-CONTRACT-001      done
2. TRAIN-DECODER-OPACITY-001        done
3. TRAIN-RUN-005-OPACITY-SMOKE      done / weak pass
4. RENDER-FIELD-SCALE-PLAN-001      current / docs-only
5. DECODER-SCALE-CONTRACT-001       next code PR
6. TRAIN-DECODER-SCALE-001          renderer gradient + CLI gate
7. TRAIN-RUN-006-SCALE-SMOKE        controlled GPU smoke
```

即使 `TRAIN-RUN-005-OPACITY-SMOKE` 通过，也不进入 per-Gaussian means / scales /
quats / cameras / dynamic-K。下一刀只允许 object-level scale multiplier。

## 8. 非目标

- 不训练 Gaussian means / positions。
- 不训练 rotation / quaternion。
- 不训练 per-Gaussian covariance / scale；object-level scale multiplier 必须走
  Section 9 之后的独立 gate。
- 不优化 camera。
- 不引入 dynamic-K birth / merge / split。
- 不把 per-Gaussian opacity 作为第一批 trainable field。
- 不替换 browser viewer renderer。
- 不提交 ignored `outputs/` 或 `/tmp` 训练产物。

## 9. RENDER-FIELD-SCALE-PLAN-001 决策

第二解冻字段：

```text
decoder.object_scale_log_offsets: R^K
```

命名选择：

- 使用 `log_offsets`，不是 `logits`，因为 scale multiplier 的 identity init 应该是
  `0.0 -> 1.0`。
- 第一版只允许 isotropic object-level multiplier `R^K`，不允许 `R^{K x 3}`。
- per-Gaussian scale、quaternion、mean、camera 继续冻结。

对外解码：

```text
object_scale_multiplier =
  exp(clamp(object_scale_log_offsets, log(0.75), log(1.25)))

gaussian_scale_i =
  base_scale_i * sum_k A[i,k] * object_scale_multiplier[k]
```

其中：

- `base_scale_i` 是当前 decoder 的 frozen synthetic isotropic scale 或未来 source
  Gaussian scale。
- `A[i,k]` 仍来自 Object Emergence Solver assignment。
- `object_scale_log_offsets=0` 必须严格等价于当前 frozen scale path。

## 10. Scale Contract 约束

`DECODER-SCALE-CONTRACT-001` 只允许做 ABI 和 decode contract：

- `ObjectStateGaussianDecoderState` 增加可选 `object_scale_log_offsets`。
- `as_dict()` 输出 `object_scale_log_offsets`、`object_scale_multipliers`、
  `scale_policy`、`available_fields` 和 `frozen_fields`。
- 旧 checkpoint 缺该字段时按 disabled / `constant-scale-v1` 加载。
- `decode_gaussian_from_object_state(...)` 支持显式传入
  `object_scale_log_offsets`，并用 `assignment @ object_scale_multiplier` 生成
  per-Gaussian scale。
- decode summary 将 `decoder.object_scale_log_offsets` 标为 differentiable field；
  启用 scale multiplier 后，frozen field 应从 `scales` 改为 `base_scales` 或
  `source_scales`。
- 不新增 GPU run，不改 optimizer，不动 CLI training gate。

## 11. Scale Training 约束

`TRAIN-DECODER-SCALE-001` 才允许接训练路径：

- CPU point renderer 和 gsplat renderer 的 loss result 增加
  `gradient_decoder_scale_log_offsets`。
- `solver-decoder-mvp` 新增显式开关 `--train-decoder-scale`。
- 新增独立 learning rate：`--decoder-scale-learning-rate`。
- 新增 scale init：`--decoder-scale-init-log-offset`，默认 `0.0`。
- 新增 multiplier bounds：默认 first smoke 使用 `[0.75, 1.25]`，不允许更宽。
- 新增 scale regularization：默认把 `object_scale_log_offsets` 拉回 `0.0`。
- 新增 freeze controls 或等价 near-zero gates，至少要能解释 solver / color /
  opacity / scale 哪些字段实际更新。

如果 CLI 仍只能同时更新 solver、colors、opacity 和 scale，则不能启动
`TRAIN-RUN-006-SCALE-SMOKE`。scale 比 opacity 更容易改变 visibility 和 footprint，
必须先解决 field isolation。

## 12. TRAIN-RUN-006-SCALE-SMOKE 建议

首个 scale smoke 应从 run-005 final checkpoint resume，但只训练 scale field：

```text
resume_checkpoint:
  outputs/training/train-run-005-opacity-gsplat-image/final-checkpoint.json

renderer:
  gsplat
  max_points=128
  frames=2
  image_size=16x16
  vram_reserve_gb=1

optimization:
  train decoder.object_scale_log_offsets only
  freeze or near-freeze solver assignment
  freeze decoder.object_colors
  freeze decoder.object_opacity_logits
  scale_lr=0.01..0.05
  scale_log_offset_init=0.0
  scale_multiplier_bounds=[0.75, 1.25]
  scale_reg_weight=0.01
  iterations=60..100
  checkpoint_every=20
```

如果实现层无法冻结 colors / opacity，则允许以 near-zero learning rate 做过渡 smoke，
但 summary 必须显式记录这是 fallback，不允许把结果标为 promotion evidence。

## 13. Scale 成功门槛

scale smoke 必须同时满足：

- `renderer-loss-contract` 无 `upgrade_blockers`。
- run-level `image_render_loss_decreased=true`。
- image render loss 相比 run-005 final `0.0172238834` 至少下降 `0.0001`
  absolute 或 `0.5%` relative。
- ObjectState eval 继续 pass。
- `object_purity >= 0.85`。
- `mean_normalized_entropy <= 0.30`。
- `slot_collapse=false`。
- object loss 不应明显退化；建议 `final_object_loss <= run005_final_object_loss + 0.01`。
- scale multiplier 不饱和：`scale_multiplier_min >= 0.75`、
  `scale_multiplier_max <= 1.25`，且贴边 slot 数不能超过 1 个。
- TensorBoard 至少写出 `decoder/scale_multiplier_min`、
  `decoder/scale_multiplier_mean`、`decoder/scale_multiplier_max`。
- before / after render 不得出现明显 splat bloom、hole expansion 或 silhouette collapse。

## 14. Scale 回滚条件

出现任一条件就回滚 scale 解冻，不进入 per-Gaussian geometry：

- image loss 未达到成功门槛。
- ObjectState eval 从 pass 退化为 fail。
- object purity 跌破 `0.85`。
- normalized entropy 超过 `0.30`。
- 出现 slot collapse。
- scale multiplier 多个 slot 贴 `[0.75, 1.25]` 边界。
- render 通过扩大 footprint 掩盖颜色 / assignment 错误。
- renderer boundary 出现新 blocker。
- 训练结果只能在 image-only weights 下成立，带 object loss 权重时失败。
