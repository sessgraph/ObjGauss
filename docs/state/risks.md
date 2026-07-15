# 当前风险

> 动态风险源；稳定风险类别与失败策略见 [`../PRD.md`](../PRD.md) 和
> [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)。
> 日期：2026-07-15

| 风险 | 当前状态 | 控制措施 | 解除证据 |
| --- | --- | --- | --- |
| 已发布 `0.1.0`/`0.2.0` 被静默改变或跨版本误分派 | active | 固定 5 个旧文件 SHA、精确 version+kind 分派、禁止 migrator | PR-02A 本地门冻结旧 hashes 并通过 39 个负例 |
| PR-02A schema-valid 被误写成 pilot/模型能力 | active | machine report 声明边界、状态源分账、PR-02B 独立授权 | 当前只记录 local contract supported；无数据/模型/指标 claim |
| Writer 与 evaluator 共享逻辑导致伪独立 | active | PR-01D 禁止导入 writer；从 raw artifacts 重算 | mutation matrix 与源码依赖审计通过 |
| 未审计定义就批量生成导致 cohort 返工 | mitigated_local | PR-01D 先于 PR-01E；preflight 与 formal 分账 | D audit、冻结 preflight 与 48-group formal audit 均 supported |
| Pilot runtime 无法在 clean CI 重建 | mitigated_remote | 精确 lock、clean-venv wheel-only install、真实 smoke；缺 runtime fail/blocked | `234ba00` 远端 runtime、writer、cohort 与 delivery 均成功 |
| Dirty worktree 把旧 HEAD 冒充最终 lineage | mitigated_remote | spec 只冻结 source-commit policy；runtime 注入当前 HEAD；accept/builder/verifier 都要求 clean checkout | `234ba00` 本地与远端 delivery supported |
| 重试选择性删除科学失败 | active | attempt 独立保留；科学失败不可重试或换 seed | attempt audit 与 5% 总额外上限通过 |
| Demo 被误当机器证据 | mitigated_remote | Demo 只消费已审计 episode；report 为事实源；浏览器重验 report 与 artifact checksum | 本地行为测试/Delivery verifier 与 `234ba00` 远端 delivery supported |
| 资源预算尚未由 preflight 实测冻结 | mitigated_local | 12-group preflight 独立运行 | 300s / 50MiB / 8GiB 正式预算已冻结且 formal run 在预算内 |
| Provisional effect threshold 不覆盖第二 object spec | closed_negative | 保留 rejected report，不重试/换 seed；由隔离 preflight 冻结阈值 | `0.0014 m` 下完整 preflight 与 formal cohort 独立 audit supported |
| PR-02B 排障结果被误当 clean freeze | mitigated_local | 唯一验收脚本先检查 clean worktree；报告绑定 analyzer bytes、HEAD、repeat/audit 与冻结输入 hashes | `04ddb18` clean gate、21 checks 与 checksums supported；后续代码 HEAD 必须重跑 |
| Pilot object 超出既定 weak-push 动作支持 | closed_negative | 不降低 `0.0014 m` 门；保留 rejected diagnostic，预注册前限制 pilot/formal catalog 质量范围并完整重跑 | `04ddb18` clean 两份 source audit supported |
| Power proxy 与 error-reduction 量纲或声明边界漂移 | active | 使用相对 effect 异质性投影到 `δ`；源码/hash 固定；明确它不是 observed GNN seed variance | 量纲错误版 blocked 已保留；`04ddb18` clean freeze 通过，PR-02C 仍须报告真实 trial/seed variance |
| PR-02 训练抢占桌面显示显存 | active | 12 GiB 进程 cap 与 1 GiB 实际空闲保留取更严者；不足即 blocked | `04ddb18` 与 PR-02C `fc20023` clean 16 MiB probes 均 supported；C3 dirty diagnostic 训练峰值 68,277,760 bytes、最低空闲 15,470,034,944 bytes，仍须由 clean gate 复核并在 HPO/formal 持续监控 |
| PR-02C learning runtime 意外携带 simulator 或在线依赖 | mitigated_local | 独立精确 lock 仅含 Torch；锁定且校验哈希的 install 是唯一可联网阶段；执行代码前显式离线；AST/lock/import 三层拒绝 ManiSkill、SAPIEN、objgauss-sim，并由独立 verifier 复核 | `fc20023` C0 与 `adb1a62` C1 clean gates supported；C1 首轮 offline install 因 wheel cache miss 失败后修正阶段边界，未放宽 runtime isolation；尚无远端 CI |
| C1 source 误用 RES-001 legacy venv 冒充冻结 simulator runtime | closed_negative | Source producer 强制复核 Python、ManiSkill、SAPIEN、Torch distribution/runtime 与 CUDA 六项精确版本；只接受 `sim/uv.lock` 环境 | 首次 dirty 诊断因 legacy venv 的 Torch distribution metadata 为 `2.13.0+cu130` 被 invalid；改用精确 sim venv 后 60 groups / 300 branches supported，未放宽断言 |
| PR-02C 在 config/checkpoint 冻结前泄漏 final GT | mitigated_local | Owner 已决定 PR-02C 只物化 train/validation；source CLI、loader、trainer 与 verifier 对 test/future fail closed，HPO/selector 后续仍须重复防御 | `adb1a62` clean C1 无 test artifact；C3 test split 以 exit 4 被 trainer 拒绝且 verifier 检查无 test artifact；C3 clean gate 与后续 HPO/selector 仍须复核 |
| C1 source checksum/lineage 漂移或跨 split identity 复用 | mitigated_local | 冻结 formal spec hash；producer、loader、独立 verifier 三方重算 data index；完整 group/branch/checksum/source-plan/HEAD 与初始化 lineage 检查 | `adb1a62` clean 300-branch evidence 三方 index `2501ebc2…17a81b5` 一致；尚无远端 CI |
| C2 producer 与 verifier 因对象键序或跨语言浮点算法产生伪一致/伪漂移 | mitigated_local | prediction payload 使用 Node canonical JSON；verifier 使用同一 canonical 比较但独立重算状态数学，并让平方和/归一化运算顺序与规范一致；canonical/reverse index 与 120 份 predictions 必须字节级一致 | 首轮 diagnostic 正确暴露 verifier 的键序与 `hypot` 差异；`9ea2b92` clean 18 checks/repeat supported，corruption mutation 被四层拒绝；尚无远端 CI |
| C2 baseline 偷读 source trajectory/future GT 或意外物化 final | mitigated_local | C1 单独发布 sanitized validation bundle；baseline 进程只读取 bundle；独立 Node verifier 从 raw source 重算投影、禁止 future/executed/test，并核对完整文件集 | `9ea2b92` clean 60 validation branches / 120 predictions、18 checks supported，无 test prediction；尚无远端 CI |
| Action-free 与 action-conditioned 容量或训练预算不公平 | mitigated_diagnostic | 共享 backbone/action encoder、参数量、updates、数据顺序、grid 和 seeds；独立 verifier 重算 checkpoint structure 与账本；HPO 对 validation groups/3 seeds 两级等权 | C3 dirty GPU repeat 两 arm 各 35,734 参数且 24 checks supported，篡改参数账本被 exit 4 拒绝；clean gate 与 24-task HPO/6-task formal audit 尚未运行 |
| Variable-`Δt` rollout 静默变成按时刻独立 heads 或读取 executed/GT | mitigated_diagnostic | 四区间共享 transition；显式 `Δt`、区间裁剪 commanded schedule 与 COM 作用点；初态后无 teacher forcing | Unit/CPU tiny 与 C3 dirty GPU repeat 覆盖 `0.1/0.1/0.3/0.6 s`、共享参数和 feature visibility；仍待 clean gate，不能外推为模型性能 |
