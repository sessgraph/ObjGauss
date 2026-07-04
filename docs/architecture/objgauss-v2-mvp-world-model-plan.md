# ObjGauss MVP -> v2 -> v3 World Model Roadmap

> 状态: research spec / planning baseline
> 最近更新: 2026-07-04
> 依赖:
> - `docs/architecture/objgauss-v1-kernel-contract.md`
> - `docs/architecture/objgauss-v1-object-emergence-plan.md`
> - `docs/architecture/object-emergence-model-v1.md`
>
> 目的: 将 ObjGauss world-model 方向拆成能跑、能验收、能解释失败原因的工程演进路线。
> 这个文档约束后期模型训练迭代，不改变当前 v1 kernel contract 或近期 renderer field
> thaw 队列。

## 0. 总体演进哲学

三阶段不是功能堆叠，而是逐步引入复杂度来解决上一阶段必然出现的崩溃点：

```text
MVP = stable object prediction
v2  = stabilized object dynamics with weak structure
v3  = self-evolving object-centric world simulator
```

对应的核心问题：

```text
MVP: can the model predict O_t+1 from O_t?
v2:  can the model predict stably when identity drifts?
v3:  can the model generate, validate, and reuse future worlds safely?
```

近期主线仍是 v1:

```text
PerceptionEvidence -> A[N,K] -> ObjectState
  -> Gaussian decode -> renderer loss
```

world-model 路线只能依赖这条链路输出的稳定 ObjectState 序列和 renderer validation，
不能替换 v1 kernel，不能把 Gaussian 变成 reasoning unit，也不能让 diffusion 直接操作
Gaussian records。

## 1. MVP: 最小可跑版本

### 1.1 目标

MVP 只做一个稳定的 object-level temporal prediction model：

```text
p(O_t+1 | O_t)
```

它的本质是：

```text
object-level video prediction model
```

### 1.2 架构

```text
Video / multi-view frames
  -> Gaussian reconstruction or registered Gaussian artifact
  -> object clustering / assignment A[N,K]
  -> ObjectState encoder
  -> temporal model
  -> next ObjectState prediction
  -> Gaussian decoder / renderer validation
```

### 1.3 MVP 必须删除的系统

MVP 不包含：

- identity graph。
- replay buffer。
- synthetic data。
- canonical object space。
- self-generated rollout training。
- strong interaction graph reasoning。

这些不是永久不要，而是必须先故意留空，避免第一版训练被多系统耦合拖崩。

### 1.4 MVP 学什么

MVP 只学习短期对象状态转移：

```text
O_t -> O_t+1
```

候选预测字段：

- centroid / position delta。
- bbox / shape summary delta。
- confidence / slot stability。
- optional appearance / shape latent delta。

第一版建议先做 deterministic baseline，再进入 object latent diffusion：

1. Deterministic regression baseline。
2. Probabilistic delta baseline。
3. Lightweight object-token diffusion。
4. Transformer diffusion, only after baseline gates pass。

### 1.5 为什么 MVP 不容易崩

因为它没有复杂耦合结构：

- 没有 graph noise。
- 没有 hard identity truth。
- 没有 self-generated feedback loop。
- 数据来自 real trajectory。
- 训练目标只看短期 prediction。

### 1.6 MVP 必然存在的问题

MVP 应该允许这些问题暴露出来：

- identity drift。
- object swap。
- long-term error accumulation。
- multi-step rollout collapse。

这些不是 MVP 失败本身；它们是进入 v2 的依据。MVP 的验收重点是短期预测能跑、
训练曲线稳定、collapse 可见，而不是长时段世界模拟。

### 1.7 MVP 成功标准

MVP 成功必须满足：

```text
one_step_prediction_loss_decreased = true
training_curve_stable = true
objectstate_eval_pass = true
renderer_boundary_upgrade_blockers = []
```

同时必须输出 negative evidence：

```text
identity_drift_report_available = true
object_swap_report_available = true
multi_step_rollout_report_available = true
```

## 2. v2: 结构增强版

### 2.1 目标

v2 引入弱结构约束，解决 MVP 暴露出的 identity drift、object swap 和 temporal
instability。

v2 的本质是：

```text
stable object-centric dynamics model
```

### 2.2 架构升级

```text
MVP temporal model
  -> weak identity graph
  -> real-only memory / replay buffer
  -> soft conditioning dynamics / diffusion
```

### 2.3 Weak Identity Graph

Weak identity graph 不是 truth graph，而是 similarity graph：

```text
w(i,j) = similarity(ObjectState_i_t, ObjectState_j_t+1)
```

允许来源：

- motion similarity。
- appearance similarity。
- spatial continuity。
- bbox overlap / centroid distance。
- assignment confidence compatibility。

用途：

- stabilize tracking。
- reduce object swap。
- provide soft conditioning summary。

禁止：

- 把 graph edge 当 ground truth。
- 用 hard graph matching 作为主 loss。
- 用 graph 静默改写 `object_id`、slot identity 或 artifact。
- 让 graph 成为训练代码的硬依赖。

### 2.4 Real-only Replay Buffer

v2 replay buffer 只保存真实 object episodes：

```text
ObjectEpisode = {
  source_asset_id
  frame_range
  temporary_object_id
  object_state_sequence
  observations
  confidence_score
  filter_policy
  license_note
}
```

写入条件：

- 来源是真实 video / multi-view / reconstruction output。
- ObjectState eval pass。
- confidence 超过明确阈值。
- license 和 local path 记录清楚。
- 输出在 ignored `outputs/` 或 `/tmp`，不进 git。

禁止：

- synthetic data。
- self-generated rollout training。
- generated futures 进入 training replay。
- 无 filter policy 的 episode 进入 buffer。

### 2.5 Identity Embedding

v2 可以引入软身份 embedding：

```text
z_id = learned embedding
```

规则：

- `z_id` 是 embedding，不是 identity truth。
- `z_id` 服务 smoothing / re-identification aid / contrastive regularizer。
- 缺少 `z_id` 时，MVP temporal model 仍必须能跑。
- `ObjectState.id` 和 renderer-facing `object_id` 继续由 assignment / matching /
  export policy 派生。

### 2.6 v2 Training Objective

v2 可以在 MVP prediction loss 上加入轻量稳定项：

```text
L_v2 =
  L_prediction_or_diffusion
+ lambda_motion   L_motion
+ lambda_identity L_identity_smooth
```

注意：

- `L_identity_smooth` 只能是 regularizer。
- graph loss 不能成为主约束。
- ambiguous match 必须 down-weight 或输出 diagnostics。

### 2.7 v2 解决的问题

v2 应该改善：

- object swap 减少。
- tracking 稳定。
- short-term consistency 提升。

v2 仍然不解决：

- 真正 world model memory。
- self-generated data loop。
- long-horizon planning。
- partial world imagination。

## 3. v3: 世界模型版本

### 3.1 目标

v3 才允许进入自我生成、自我验证、自我扩展世界的方向。

v3 的本质是：

```text
object-centric world simulator
```

### 3.2 架构升级

```text
v2 stabilized dynamics
  -> strong identity graph
  -> object episode DB + graph DB
  -> multi-step diffusion world model
  -> self-generated rollout + filtering
```

### 3.3 Strong Identity Graph

从 v2 的 similarity graph 升级为 persistent identity graph。

能力目标：

- object lifetime tracking。
- cross-view binding。
- canonical identity mapping。

v3 之前禁止把 strong identity graph 提前塞进训练主线。

### 3.4 Object Replay Buffer / Graph DB

v3 replay system 可以从 real-only episode buffer 升级为：

```text
ObjectEpisode DB + Graph DB
```

能力目标：

- store full object histories。
- store interaction graphs。
- store canonical forms。
- support multi-step rollout validation。

### 3.5 Self-generated World Loop

v3 才允许讨论：

```text
model predicts future
  -> render
  -> validate consistency
  -> add to buffer
  -> retrain
```

这是最高风险模块。开启前必须已有：

- v2 stable dynamics gate。
- identity graph noise audit。
- generated rollout filter metrics。
- renderer validation。
- explicit rollback policy。

### 3.6 v3 Training Objective

v3 才允许更完整的 loss family：

```text
L_v3 =
  L_diffusion
+ L_identity_consistency
+ L_graph_consistency
+ L_replay_consistency
+ L_self_generated_filtering
```

这些 loss 不能提前出现在 MVP / v2 主训练目标里。

## 4. 三阶段对比

| 阶段 | 学什么 | 新增复杂度 | 主要风险 | 禁止事项 |
| --- | --- | --- | --- | --- |
| MVP | prediction: `O_t -> O_t+1` | temporal model | long-term drift | graph / replay / synthetic data |
| v2 | stability | weak graph + real-only buffer | over-constraint / graph noise | hard graph truth / self-generated data |
| v3 | world simulation | strong graph + generated loop | self-generated data collapse | 无 filter 的 generated training |

简化理解：

```text
MVP: learn prediction
v2:  learn stable prediction
v3:  learn world expansion
```

## 5. 崩溃点演进

### MVP 崩溃点

- long-term drift。
- object swap。
- multi-step error accumulation。

解决方式：

- 只记录 negative evidence。
- 不急着加 graph。
- 用 v2 weak identity graph 专门解决。

### v2 崩溃点

- over-constraint。
- graph noise。
- identity smoothing 反向污染 dynamics。

解决方式：

- graph edge 永远 soft。
- graph removal must not crash training。
- ambiguous matches down-weight。
- identity loss 只能是 regularizer。

### v3 崩溃点

- self-generated data collapse。
- model trains on its own error。
- generated futures 污染 replay buffer。

解决方式：

- generated data 必须经过 filter。
- generated training 必须有 ADR 和 rollback policy。
- generated futures 默认不得进入 v2 replay buffer。

## 6. 推荐真实开发路径

不要跳级：

1. `OBJECT-ROLLOUT-BASELINE-001`
   - 跑通 deterministic object prediction。
   - 记录 drift / swap / rollout collapse negative evidence。

2. `OBJECT-LATENT-DIFFUSION-MVP-001`
   - 只在 object latent space 做 diffusion。
   - 不加 graph，不加 replay buffer。

3. `WEAK-IDENTITY-GRAPH-001`
   - 加 local weak graph。
   - 只做 soft stabilizer。

4. `OBJECT-EPISODE-BUFFER-001`
   - 加 real-only replay buffer。
   - 写 verifier 和 filter policy。

5. `V2-STABILITY-GATE-001`
   - 验证 object swap、short-term consistency 和 graph noise。

6. `V3-SELF-GENERATED-ADR-001`
   - 只有 v2 stability gate 通过后，才讨论 self-generated loop。

## 7. 与当前 v1 主线的关系

当前近期优先级不变：

- `TRAIN-DECODER-SCALE-001`
- `TRAIN-RUN-006-SCALE-SMOKE`
- renderer field promotion gates
- ObjectState eval / renderer-loss-contract

MVP / v2 / v3 world-model 路线是后期训练迭代规划。它依赖 v1 输出稳定的
ObjectState 序列和 Gaussian decoder / renderer validation，而不是替换它们。

## 8. 非目标

本文不能用来合理化以下事项：

- 立即引入 SAM2 / CoTracker / optical flow / PyTorch diffusion 默认依赖。
- 在 MVP 阶段加入 identity graph 或 replay buffer。
- 在 v2 阶段加入 self-generated rollout training。
- 训练或提交 generated futures。
- 把 diffusion world model 记为当前已落地能力。
- 改变 v1 kernel contract。
- 替换 viewer renderer。
- 改变 public demo / HF release 口径。
