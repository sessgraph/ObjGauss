# 项目当前状态

> 动态事实摘要；需求见 [`../PRD.md`](../PRD.md)，资源事实见 [`../../REFERENCES.md`](../../REFERENCES.md)。
> 日期：2026-07-14

当前处于 PR-01 严格成对干预里程碑。Stage-0 与 PR-00 已本地提交；ManiSkill `3.0.1` 的
snapshot/RNG 和程序化 CPU primitive action/contact source gates 已本地支持，但相关脚本和文档
现已随 PR-01A–F 由提交 `71d4e39` 固化。

Owner 已确定 PR-01 使用 `0.2.0` episode/experiment/attempt/invariance-report 四层 contract、
隔离 `sim` production optional extra，以及无 RGB 五联状态回放。里程碑内部按 PR-01A–F 推进，
独立审计器必须先于正式 cohort。

PR-01A Contract 已在精确 Node `24.18.0` 本地门得到 `supported`。PR-01B Runtime 也已从全新
临时 venv 完成精确 lock 安装、10 个行为测试及 canonical/reverse 真实五分支 smoke，稳定
evidence 为 `8a2013f1…71cb0`。PR-01C Writer 已通过 22 个 Python 行为/负例测试，并由两个真实进程
产出一致 golden evidence `d25a635c…2dfb4`。PR-01D 独立审计的 14 个 hard gates 与 11 个
mutations 全部通过。PR-01E 保留 provisional threshold rejection 后完成冻结 preflight，并以
48 groups / 240 episodes、24/12/12 split、0 failed/extra attempts 通过正式独立审计。

PR-01F 的五联无 RGB Demo、Delivery report/checksums、`accept-pr01`、CI job 与 clean-head lineage
guard 已实现。正式 spec 不再硬编码旧提交，而由运行时注入当前 clean HEAD。实现提交 `71d4e39`
的 clean `accept-pr01` 已重建 48 groups / 240 episodes、24/12/12 split、0 failed/extra attempts；
独立 audit、1210-entry checksum index 与 Delivery verifier 均为 `supported`，source commit 与 HEAD
一致。A–F 均未运行远端 CI，PR-01 仍未关闭；下一门是状态同步后的最终 SHA 本地复跑和远端 CI。
模型、训练、Gaussian dynamics、外部数据、
RGB/GPU renderer 和机器人控制均未实现。
