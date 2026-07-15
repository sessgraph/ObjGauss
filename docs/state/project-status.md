# 项目当前状态

> 动态事实摘要；需求见 [`../PRD.md`](../PRD.md)，资源事实见 [`../../REFERENCES.md`](../../REFERENCES.md)。
> 日期：2026-07-15

PR-01 严格成对干预里程碑已关闭。Stage-0 与 PR-00 已提交；ManiSkill `3.0.1` 的
snapshot/RNG 和程序化 CPU primitive action/contact source gates 已本地支持，但相关脚本和文档
最初随 PR-01A–F 由提交 `71d4e39` 固化，最终代码承载验收 SHA 为 `234ba00`。

Owner 已确定 PR-01 使用 `0.2.0` episode/experiment/attempt/invariance-report 四层 contract、
隔离 `sim` production optional extra，以及无 RGB 五联状态回放。里程碑内部按 PR-01A–F 推进，
独立审计器必须先于正式 cohort。

PR-01A Contract 已在精确 Node `24.18.0` 本地门得到 `supported`。PR-01B Runtime 也已从全新
临时 venv 完成精确 lock 安装、10 个行为测试及 canonical/reverse 真实五分支 smoke，稳定
evidence 为 `8a2013f1…71cb0`。PR-01C Writer 已通过 22 个 Python 行为/负例测试，并由两个真实进程
在同一 clean source 下产出一致 golden evidence；该 digest 纳入 source commit/tree，不作为跨提交
常量。PR-01D 独立审计的 14 个 hard gates 与 11 个 mutations 全部通过。PR-01E 保留 provisional
threshold rejection 后完成冻结 preflight，并以
48 groups / 240 episodes、24/12/12 split、0 failed/extra attempts 通过正式独立审计。

PR-01F 的五联无 RGB Demo、Delivery report/checksums、`accept-pr01`、CI job 与 clean-head lineage
guard 已实现。正式 spec 不硬编码提交，而由运行时注入当前 clean HEAD。代码承载验收 SHA
`234ba00` 的 clean `accept-pr01` 已重建 48 groups / 240 episodes、24/12/12 split、0 failed/extra
attempts；独立 audit、1210-entry checksum index 与 Delivery verifier 均为 `supported`，source
commit 与 HEAD 一致。该 SHA 的 PR-00、runtime、writer、independent audit、frozen cohort 与
acceptance delivery 六项远端 Actions 全部成功。任何后续 `main` HEAD 若未保持这些门成功，
PR-01 状态自动重开。PR-02C C0 之外的模型、训练、Gaussian dynamics、外部数据、RGB/GPU
renderer 和机器人控制均未实现。

PR-02 的主要预注册决策已经完成：唯一 primary endpoint 是 held-out sibling groups 上 target
object 的多步 `effect-vs-hold` ObjectState error；使用全新隔离 ManiSkill cohort、最小 Object
GNN、独立 `learning/`/纯 PyTorch 栈、全门联合 verdict 和 `02A`–`02F` 串行切片。硬资源上限为
24 GPU-hours、12 GiB 训练峰值且始终为桌面显示保留至少 1 GiB 实际可用显存、8 CPU
wall-hours 和 100 GiB ignored artifacts。PR-02A 已在本地建立 7 个 `0.3.0` schema 文件、6 种
精确分派记录、6 个正向 fixtures 与 39 个负例；旧 5 个 contract 文件哈希保持冻结，machine
report `3b1e64a0…acccca3f` 和 `npm run check` 均为 `supported`。该实现已提交但尚无远端 CI
证据；仍没有模型代码或训练证据。

PR-02B 已获授权、实现并提交：冻结输入 manifest、与 PR-01/正式 cohort 隔离的 12-group source spec、
canonical/reverse cohort order、两份独立 source audit、对称性校正 calibration、功效/正式 data
freeze、4-config 最小 Object GNN grid、1 GiB 桌面显存保留探针和 clean-head verifier。排障中首版
过重 object 被既有 source gate 正确拒绝，随后首版 power 计算又因量纲错误得到 blocked；两条
负证据均未通过放宽阈值解决。

实现与路径修复分别由 `b99b5f1`、`04ddb18` 提交。代码承载 SHA `04ddb18` 的 clean
`./scripts/check-pr02b-pilot` 完整通过：canonical/reverse 各 12 groups / 60 episodes、0 failed/
extra attempts，两份独立 source audit、跨顺序语义一致性、GPU 1 GiB 显示保留、21 项 freeze
verification 与 evidence checksums 均为 `supported`。权威 pilot report SHA-256 为
`47ad53c6…944cc`，冻结 48/12/12 formal groups、3 seeds、`δ=0.1`、`δ_shuffle=0.06` 和含 5%
retry reserve 的 10.5 GPU-hours 调度。当前状态为 `committed_local_supported`，尚无远端 CI
证据。PR-02C 的前置依赖已满足，Owner 已于 2026-07-15 单独授权；accepted
ADR-006 已定义 runtime/data/model/ledger 边界、24 个 HPO tasks、6 个 formal training tasks、
golden repeat 和停止条件。Owner 已选择延迟物化 final test：PR-02C 只生成 48 train + 12
validation groups，12 个 test groups 到 PR-02E 前仅保留冻结 spec；learned rollout 复用四个
评分区间的 variable-`Δt` residual transition，只读取 commanded-action schedule；HPO config
按 12 validation groups group-first 后对全部 3 seeds 等权平均，缺失 seed 的 config 不可入选。
ADR-006 已 accepted。C0 已建立独立 `learning/` package、精确 `uv.lock`、纯 PyTorch 离线
runtime、simulator isolation、clean HEAD/lock/grid lineage guard、12 GiB cap/1 GiB display
reserve probe、独立 verifier 和失败语义。提交 `fc20023` 的 clean
`./scripts/check-pr02c-runtime` 已在 Node `24.18.0` 下通过 77 项全库测试、12 个 Python 测试、
14 项 verification checks 和真实 RTX 5060 Ti probe；最低观测空闲显存超过 1 GiB，training
allocation cap 为 12 GiB。C0 状态是 `c0_committed_local_supported`，尚无远端 CI；其后已进入
C1 train/validation cohort 与 fail-closed loader，C0 状态继续保持。

C1 已提交冻结 source plan、train/validation-only simulator producer、独立纯 Python
checksum/lineage loader 和不导入两侧实现的 Node verifier。Loader 的 model input 只含初态、
commanded action、target object 和四个物理 rollout times；future GT 单独进入 labels，executed
action 不进入 feature。代码承载 SHA `adb1a62` 的 clean gate 生成 48 train + 12 validation groups、
300 branches、0 failed attempts，producer、loader 与 16 项独立 checks 对 data index
`2501ebc2…17a81b5` 一致。当前状态为 `c1_committed_local_supported`；尚未物化 test，也没有
baseline、模型、trainer、checkpoint、模型指标、Gaussian dynamics 或机器人控制。下一切片是
C2 deterministic baselines。
