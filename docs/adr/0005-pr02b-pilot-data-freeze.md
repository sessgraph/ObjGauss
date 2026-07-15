# ADR-005：PR-02B 隔离 Pilot 与 Data Freeze

- 状态：accepted
- 日期：2026-07-15
- 范围：PR-02B Pilot/Data Freeze

## 背景

PR-02A 只证明 `0.3.0` 能表达 dynamics evidence，尚未提供可执行的 horizon、归一化尺度、
最小实际增益、正式 group/seed 数、训练配置或资源调度。PR-02C 又必须消费这些冻结值，不能
在看到 HPO、模型或 final test 后再调整。PR-01 evidence 只能作 runtime/contract 回归，不能
进入 PR-02 pilot 或统计。

## 决策

1. Pilot 使用 PR-01 已批准的 ManiSkill `3.0.1` programmatic CPU primitive source，不下载
   外部数据；对象、layout、reset seed 和 experiment lineage 与 PR-01、PR-02 正式 cohort
   全部隔离。Canonical/reverse 两遍各生成 12 sibling groups / 60 episodes，并分别通过不依赖
   simulator/writer 的 PR-01D auditor。
2. Horizon 按物理时间冻结为覆盖 `0.1 s` push 与固定 settling 的 `1.1 s`，评分点为
   `[0.1, 0.2, 0.5, 1.1] s`。位置、显式对象对称性校正朝向、线速度和角速度分别使用 pilot
   effect 的 75% robust quantile、两遍 evaluator noise 的 95% quantile及预注册 epsilon 三者
   的最大值作为尺度；四分量在后续 primary scalar 中等权。
3. `δ` 使用 `clamp(0.10 * median_normalized_effect, 0.05, 0.10)`，`δ_shuffle` 使用
   `max(0.03, 0.60 * δ)`。Power design 只把 sibling-group effect 的相对异质性投影到 `δ`
   尺度，并以固定 seed bootstrap group means 作为 training-seed sensitivity proxy；它不是已
   观察的 GNN 跨 seed 性能。若预注册候选在该保守代理和硬预算内没有达到 0.8 power，结果必须
   为 `blocked`。
4. 最小满足候选冻结为 12 test groups 与 3 training seeds；正式 data spec 因此固定为
   48/12/12 train/validation/test sibling groups。三个 split 的 object identities、layouts 和
   group IDs 两两不交，pilot 全部排除于训练和 final 统计；trainer/final loader 的代码级隔离
   留给 PR-02C/PR-02D 实现。
5. PR-02B 只提交纯 PyTorch 最小 Object GNN 的有限 grid 与任务上限，不创建 `learning/`
   package、trainer、checkpoint 或模型结果。两个学习 arms 的 4-config HPO × 3 seeds 最多
   6 GPU-hours；正式训练基础调度最多 4 GPU-hours。再为最多 5% 技术重试分别保留后为
   6.3/4.2 GPU-hours，总调度 10.5 GPU-hours，分别低于 8/16/24 小时硬门。
6. CUDA preflight 只做 16 MiB allocation probe；训练 cap 取 12 GiB 与“实际空闲显存减 1 GiB”
   的较小值。若无法为桌面显示保留 1 GiB，PR-02B 为 `blocked`，不得抢占显示或扩大预算。
7. `scripts/check-pr02b-pilot` 是唯一验收入口，必须在 clean checkout、精确 Node/uv/lock、离线和
   空只读 asset 目录下运行。报告绑定 analyzer 字节哈希、当前 HEAD、repeat/audit 输入哈希和
   frozen inputs；dirty diagnostic 不能成为权威 freeze。

## 排障负证据

- 第一版 calibration object B 的质量超出 weak-push 支持范围，独立 source audit 正确以
  `paired_effect_valid` 拒绝；没有放宽 `0.0014 m` 门，而是在正式预注册前把 pilot/formal
  object catalog 限制回相同动作支持范围并完整重跑。
- 第二版 source audit 通过后，首版 power analyzer 因把 normalized physical effect 的绝对方差
  与 error reduction 直接相加而 `blocked`。该量纲错误在 freeze 前改为相对异质性投影；旧
  结果只保留为 ignored diagnostic，不得描述成统计功效结论。

## 后果

- PR-02C 只能消费 clean PR-02B evidence 中已冻结的 experiment/data/grid；不得重新选择
  horizon、尺度、`δ`、seed/group 数或搜索空间。
- 代理功效不支持模型表现声明。实际 trial/seed 方差必须在 PR-02C ledger 中完整报告，正式
  scientific verdict 仍由 PR-02D 独立 evaluator 和 PR-02E 一次性 final experiment 决定。
- 当前 dirty-worktree diagnostic 的 source audits、GPU probe 与 21 项 freeze verifier 为
  `supported`，但 clean acceptance 尚未运行，因此 PR-02B 动态状态仍是
  `implemented_pending_clean_acceptance`。

## 验证

```bash
npm run test:pr02b
PYTHONPATH=sim/src python3 -m unittest sim.tests.test_pr02_pilot
./scripts/check-pr02b-pilot
```

前两项已在当前工作区通过；最后一项按设计拒绝 dirty checkout，必须在提交后的 clean HEAD
重跑。该边界不支持 trainer、模型性能、Gaussian dynamics、外部数据或机器人控制声明。
