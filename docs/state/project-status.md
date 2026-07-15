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
PR-01 状态自动重开。
模型、训练、Gaussian dynamics、外部数据、
RGB/GPU renderer 和机器人控制均未实现。

PR-02 的主要预注册决策已经完成：唯一 primary endpoint 是 held-out sibling groups 上 target
object 的多步 `effect-vs-hold` ObjectState error；使用全新隔离 ManiSkill cohort、最小 Object
GNN、独立 `learning/`/纯 PyTorch 栈、全门联合 verdict 和 `02A`–`02F` 串行切片。硬资源上限为
24 GPU-hours、12 GiB 训练峰值且始终为桌面显示保留至少 1 GiB 实际可用显存、8 CPU
wall-hours 和 100 GiB ignored artifacts。PR-02A 已在本地建立 7 个 `0.3.0` schema 文件、6 种
精确分派记录、6 个正向 fixtures 与 39 个负例；旧 5 个 contract 文件哈希保持冻结，machine
report `3b1e64a0…acccca3f` 和 `npm run check` 均为 `supported`。该实现已提交但尚无远端 CI
证据；仍没有模型代码、pilot、cohort 或训练证据。下一门 PR-02B 必须另行授权。
