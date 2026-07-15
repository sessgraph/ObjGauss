# ADR-006：PR-02C Trainer/Baselines 实施边界

- 状态：accepted
- 日期：2026-07-15
- 范围：PR-02C Trainer/Baselines
- 动作授权：Owner 已于 2026-07-15 授权 PR-02C 并要求先完成规划

## 背景

PR-02A 已冻结 `0.3.0` dynamics contract，PR-02B 已用 clean pilot 冻结 horizon、归一化尺度、
阈值、split、training seeds、hyperparameter grid 和资源预算。PR-02C 的职责不是裁决模型是否
胜过 baseline，而是证明一个与 simulator 隔离的 clean GPU runtime 能按冻结输入公平、可复现地
产生四个 arms 的 predictions，以及两个 learned arms 的完整 trial/attempt/checkpoint lineage。

PR-02D 才建立独立 evaluator；PR-02E 才读取 final GT 并给出科学 verdict。因此 PR-02C 不能用
test 指标选配置，也不能把训练 loss、validation 表现或 Demo 描述成 PR-02 假设已获支持。

## 已冻结输入

- Runtime：CPython `3.10.20`、`uv`、`torch==2.13.0+cu130`、CUDA `13.0`；纯 PyTorch，不引入
  PyG、Lightning 或 Hydra。
- 数据：48/12/12 train/validation/test sibling groups；PR-01 和 PR-02B pilot 全部排除。
- Horizon：`1.1 s`；评分点 `[0.1, 0.2, 0.5, 1.1] s`。
- Normalization：position `0.014703245192199387`、orientation
  `0.00033644301922148125`、linear velocity `0.09083031897380531`、angular velocity
  `0.011206615557372438`，四项等权。
- Learned arms：`action_free`、`action_conditioned`；training seeds 为 `2026071501`、
  `2026071502`、`2026071503`。
- Grid：hidden width `{64, 128}` × learning rate `{1e-3, 3e-4}`，其余参数固定，共 4 configs；
  每个 learned arm 使用相同 grid、seeds、数据曝光、最大 20,000 updates 和 200 epochs。
- HPO：2 arms × 4 configs × 3 seeds = 24 tasks，基础上限 6 GPU-hours；正式训练：2 arms ×
  3 seeds = 6 tasks，基础上限 4 GPU-hours；计入 5% retry reserve 后合计不得超过 10.5 GPU-hours。
- GPU：单进程峰值不超过 12 GiB，并始终保留至少 1 GiB 实际空闲显示显存。

以上值由 PR-02B evidence 和冻结 fixture 决定，PR-02C 不得调整。

## 已冻结的 PR-02C 规划决策

### Final test 延迟物化

Owner 已选择方案 A：PR-02C 只生成 48 个 train 和 12 个 validation sibling groups。PR-02B
冻结的 12 个 test group IDs、object identities、layouts 和 seeds 继续保留在 formal spec 中，
但 PR-02C 不生成对应 episode、trajectory、contact ledger 或任何 GT future。只有 PR-02E 在
config 与 checkpoints 冻结、PR-02D evaluator 就绪后，才能按同一 spec 物化 test source。

具体控制如下：

- PR-02C source CLI 只接受 `train`、`validation`，收到 `test` 必须 fail closed；
- `generated/pr02c/` 中不得出现 test dataset root、test GT artifact 或可解析的 test future；
- trainer、HPO、checkpoint selector 和 prediction writer 仍必须独立拒绝 `split=test`，形成纵深
  防御，不能因为 test 尚未物化而省略负例；
- PR-02C acceptance verifier 若发现 test 数据产物，verdict 为 `invalid`；
- PR-02E 物化 test 时不得改变 frozen spec、source runtime、对象、layout、seed 或 action support。

该选择优先最小化 final leakage；代价是 PR-02E 必须单独验证 test source 的生成和 source audit，
不能声称 PR-02C 已生成完整 72-group cohort。

### 四评分区间 variable-`Δt` residual rollout

Owner 已选择 rollout 方案 A。Learned arms 从初态 `S_0` 开始，只在冻结时间边界
`[0.0, 0.1, 0.2, 0.5, 1.1] s` 的四个相邻区间调用同一个共享 transition：

```text
S_(t_next) = normalize(S_t + residual_theta(S_t, commanded_schedule[t:t_next], Δt))
```

具体控制如下：

- `Δt` 是显式标量 feature，依次为 `0.1/0.1/0.3/0.6 s`，不能用 simulator step index 代替；
- commanded-action encoder 只读取裁剪到当前区间的预注册 command、target mask、作用方向/大小、
  作用点和 active duration/fraction；executed action 只可进入事后 telemetry；
- action-free 使用相同形状、相同 action encoder 和 learned constant mask token，不删除模块或
  减少参数；
- 只在 `S_0` teacher-force，后续四个状态全部消费前一步预测；label builder 可读取
  train/validation GT 计算 loss，但模型调用边界不得读取区间终点 GT；
- 每步输出后规范化 quaternion 并验证有限值；不得用 fallback GT 修复非有限状态；
- loss 与 predictions 只在四个冻结评分时刻记账，不增加七个未预注册的 `0.1 s` 中间 rollout
  steps，也不把其 secondary loss 混入 primary training objective。

该选择直接对齐 endpoint 并减少无监督中间递归；代价是 transition 必须学习 variable `Δt`，因此
tiny/golden fixtures 必须覆盖两个相同 `0.1 s` 区间以及 `0.3/0.6 s` 长区间，验证共享参数而非
按时刻建立四个 heads。

### Validation-group/seed 两级等权 HPO 聚合

Owner 已选择 HPO 聚合方案 A。四个 configs 分别在两个 learned arms 和全部三个冻结 training
seeds 上运行。对每个 `arm × config`：

1. 每个 seed 先在 12 个 validation sibling groups 上计算 group-first 等权 primary error；同组
   五个 branches 和四个评分时刻按冻结 endpoint 计算，不能按 episode/frame 数隐式加权；
2. 再对三个 seed-level validation errors 等权算术平均，得到该 config 的唯一 selection score；
3. 只有三个 seeds 的 trial/attempt/checkpoint/validation prediction 均完整、contract-valid 且
   checksum-valid 的 config 才可入选；缺失或耗尽合法重试的 config 保留原始失败证据并失去资格；
4. 每个 arm 选择 score 最低的单一 config；完全相等时按从 canonical config JSON 冻结的 config
   ID 字典序选择，不能以运行顺序、checkpoint 大小或单 seed 表现破平局；
5. 不使用跨 seeds 中位数，不挑 best seed，也不删除失败 trial。选定 config 后，formal training
   仍对全部三个 seeds 从头运行，并逐 seed 只按 validation primary error 选择 checkpoint。

该规则让每个 seed 都对选择产生同等影响；代价是任一 seed 缺失会使该 config 不可入选。若某个
arm 没有任何完整 config，PR-02C 为 `blocked` 或按失败分类保留 `rejected/invalid`，不得缩减为
两个 seeds 或扩大 grid。

## 单一假设与 primary acceptance endpoint

可证伪假设：在不导入 ManiSkill/SAPIEN、不可读取 final GT、资源和重试规则不变的条件下，独立
`learning/` runtime 能公平、可复现地运行 copy-state、constant-velocity、action-free 和
action-conditioned 四个预注册 arms，并输出 contract-valid、checksum-valid、lineage-complete
的训练与预测 artifacts。

PR-02C 的 primary acceptance endpoint 是固定 golden training group 上的 clean repeat：

1. 两个确定性 baselines 的 prediction semantic hashes 完全相同；
2. 两个 learned arms 在同一 config/seed 下的排序后 tensor-state semantic hash、validation
   prediction semantic hash、optimizer update 数和 checkpoint selection 完全相同；
3. action-free/action-conditioned 的 backbone parameter count、updates、数据顺序和可见字段相同；
4. 所有 trial、attempt、checkpoint 和 prediction records 通过 `0.3.0` schema、lineage、checksum、
   资源、retry 和 final-isolation checks。

Checkpoint 文件字节 hash 继续作为 artifact identity，但 repeat 裁决使用按参数名排序并包含
dtype/shape/tensor bytes 的 semantic hash，避免把容器元数据差异误写成模型不确定性。容差比较
不得替代上述同宿主、同 runtime 的 semantic repeat 门。该 semantic hash 写入 immutable training
log 与 PR-02C acceptance report，由 verifier 重算；不为此修改已冻结的 `0.3.0` 公共 schema。

## 责任边界

### `sim/`

- 只按 PR-02B 的 formal data spec 生成和审计不可变 `0.2.0` episodes。
- 不包含 dataset loader、feature builder、模型、loss、optimizer 或 checkpoint selector。
- PR-02C 不修改已冻结的 simulator 动作、对象、layout、seed、horizon 或 source gates。

### `learning/`

- 建立独立 package、精确 `uv.lock`、数据 loader、四个 arms、训练器、原子 artifact writer 和 CLI。
- 只消费 checksum-valid `0.2.0` episodes、`0.3.0` experiment 与冻结 manifest；不得导入
  `mani_skill`、`sapien` 或 `objgauss_sim`。
- 不修改 `0.3.0` schemas；trial/attempt/checkpoint/prediction 使用现有 contract，PR-02C acceptance
  report 只是仓库内 verifier 输出，不冒充独立 scientific evaluation report。
- Trainer/HPO/checkpoint selector 遇到 `split=test` 或 future-only 输入必须 fail closed。
- 所有输出写入 ignored `generated/pr02c/`；Git 只提交源码、lock、tiny fixtures 和测试。

### PR-02D / PR-02E

- PR-02D 独立实现 primary endpoint、bootstrap、baseline comparisons、shuffle/direction 和
  mutation audit；不得导入 PR-02C loss。
- PR-02E 在配置与 checkpoints 冻结后才运行 final inference/evaluation。PR-02C 不产生科学
  `supported/rejected` verdict。

## 实施顺序

1. **C0 Runtime/contract gate**：先在实现 HEAD 重跑 PR-02B clean gate 并核验 formal spec hash，
   再创建最小 `learning/` package、精确 lock 和 clean-install smoke；冻结 CLI 输入输出、source
   tree/runtime hashes、退出码与 ignored output root。
2. **C1 Data boundary**：从 formal spec 只生成并独立审计 48 train + 12 validation groups；test
   仅保留 spec。Loader 校验 checksum、lineage、object/layout/group 隔离、commanded action
   presence，并以行为测试证明 test/future fail closed。
3. **C2 Deterministic baselines**：实现 copy-state 与 constant-velocity，输出统一
   `objgauss.dynamics_prediction`，覆盖四个评分时刻；不创建虚构 checkpoint/trial。
4. **C3 Minimal Object GNN**：共享两层 object encoder、两层 pairwise message MLP、mean
   aggregation、一次 message passing、两层 shared residual head；action 只注入 target object。
   Action-free 使用同一 action encoder 和 learned constant mask token，必须与 conditioned arm
   保持同等可训练参数量；四个物理时间区间复用同一显式 `Δt` transition。
5. **C4 Loss/trainer/ledger**：初态后完全自回归；四分量按冻结尺度归一化并等权；实现 AdamW、
   gradient clipping、early stopping、validation-only checkpoint selection、技术失败重试和原子
   trial/attempt/checkpoint publication。
6. **C5 Tiny golden acceptance**：CPU tiny fixture 覆盖 CI smoke；宿主 GPU clean repeat 覆盖
   semantic reproducibility、参数/updates 公平性、显存显示保留和负例矩阵。
7. **C6 HPO/config freeze**：运行 24 个预注册 HPO tasks；每个 arm 按三个 seeds 的 group-first
   validation primary error 聚合选一个 config，平局按冻结 config ID 排序；保留全部失败和未选
   trials，不读取 test。
8. **C7 Formal training/checkpoint freeze**：以每个 arm 的选定 config 对三个 seeds 从头正式训练
   6 个 tasks，冻结每个 seed 的 validation-selected checkpoint 和 manifests；生成 train/validation
   predictions，发布仓库内 PR-02C acceptance report 与 checksums；不得复用
   `objgauss.evaluation_report` 冒充 PR-02D。

实现可以分成可审查提交，但仍属于一个 PR 和一个假设；任一步失败不得通过拆 PR、换 seed、
扩大 grid 或修改阈值绕过。

### C6 机器执行 contract

[`learning/hpo-manifest.json`](../../learning/hpo-manifest.json) 是 C6 task、pair、config、selector、
资源、retry、输出和声明边界的唯一机器源。它把 ADR 已批准的矩阵展开为 12 个固定 fairness
pairs / 24 个固定 task IDs，并明确以下执行语义：

- 每个 `config_id + training_seed` 配对 `action_free` 与 `action_conditioned`；二者共享初始化
  seed/算法、数据与 batch/group 顺序、updates、训练预算和 checkpoint policy；只要求公共参数
  子树的初始化 digest 相同，arm-specific 子树各自确定性，不能要求整个 checkpoint 字节相同；
- `4498bd6` 是 trainer contract commit；C3 的 `dd5994a3…1a30` data index 与两个 semantic
  indexes 只作为 reference。C6 必须在 clean runner commit 下原子生成一次新的
  `hpo-data-index.json`，24 个 tasks 共同消费，单 task 不得重建或替换数据；
- selector 不导入 trainer 排名逻辑，只按 12 validation groups 的 group-first score、三个 seeds
  等权平均和精确分数平局时的 `config_id` 字典序，为两个 learned arms 各选一个 config；不设
  performance promotion threshold，也不以近似平局、资源或运行顺序改变选择；
- canonical/reverse 是同一组冻结结果的 selector 输入顺序重放，不是额外 HPO arm 或 task；
- formal training 只用 train 拟合，validation 只选择 checkpoint；HPO checkpoint 不晋升为 formal
  checkpoint。

C6 的唯一交付是确定性的双配置映射及完整 task/pair/checksum/data lineage。它不关闭 PR-02C，
不产生模型性能 verdict；C7 仍须从头运行六个 formal tasks。

## 最低测试与验收

- Feature/action encoding、quaternion normalization、symmetry-aware loss、irregular physical-time
  rollout、copy/velocity baselines 和 parameter-count parity 的纯单元测试。
- Loader 对 test split、GT future、executed-action feature、缺失 commanded action、坏 checksum、
  跨 split identity/layout、PR-01/pilot lineage 的负例。
- Attempt 分类矩阵：只有 process crash、I/O、transient OOM 可用相同 seed/config 重试；NaN、
  不收敛、schema/lineage 和显示显存违规不得伪装成成功。
- 两进程 golden repeat、clean-install GPU smoke、24-task HPO ledger 完整性、6-task formal ledger
  完整性、checkpoint/prediction 原子发布和 checksum 重算。
- `npm run check` 继续覆盖旧 contracts；新增 PR-02C 独立命令必须在实现时同步写入 `AGENTS.md`
  和 CI。无 GPU runner 的 CI 只运行 CPU tiny fixture，不声称复现本机 GPU 结果。

## Verdict 与停止条件

- `supported`：primary acceptance endpoint 与所有 contract、isolation、fairness、resource、retry
  gates 全部通过。
- `rejected`：协议有效，但 learned runtime 无法满足预注册 reproducibility 或公平性 endpoint。
- `blocked`：冻结依赖/GPU/显存/数据缺失，或预算内无法完成必需 tasks。
- `invalid`：test/future leakage、split/lineage/checksum 损坏、非法重试、任务缺失或资源账本不全。

PR-02C 合并或训练 loss 下降都不自动等于 `supported`；未通过不得进入 PR-02D。

## 决策状态

Final test 物化、rollout integrator 与 HPO config 聚合三个规划问题均已由 Owner 冻结。本 ADR
状态为 `accepted`，PR-02C 可以从 C0 Runtime/contract gate 开始实现；实现仍须逐门产生证据，
不能因规划获批而预先标记 `supported`。
