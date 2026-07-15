# 开放事项

> 只记录会影响近期执行、但尚未由本地或外部证据关闭的事项。
> 日期：2026-07-15

当前没有 PR-01 范围内的开放事项。PR-02 的主要假设、primary endpoint、训练栈、硬资源上限、
数据隔离、统计方法和 `02A`–`02F` 切片已经由 Owner 确认并写入
[`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)。PR-02A 已获授权、本地实现、通过项目门并提交，但
尚未获得远端 CI 证据。PR-02B 已由 `b99b5f1`、`04ddb18` 提交，并在代码承载 SHA `04ddb18`
完成 clean `./scripts/check-pr02b-pilot`：两遍 source、独立 audits、GPU reserve、21 项 freeze
verification、lineage 与 checksums 均为 `supported`。权威 freeze 是 horizon `1.1 s`、
`δ=0.1`、`δ_shuffle=0.06`、48/12/12 groups、3 seeds 与含 retry reserve 的 10.5 GPU-hours
调度；pilot report SHA-256 为 `47ad53c6…944cc`。

Owner 已单独授权 PR-02C Trainer/Baselines。Accepted
[`ADR-006`](../adr/0006-pr02c-trainer-baselines.md) 已冻结单一假设、责任边界、24 个 HPO tasks、
6 个 formal training tasks、golden repeat、资源和失败语义。

Owner 已冻结 final test 延迟物化：PR-02C 只生成 48 train + 12 validation groups，12 个 test
groups 在 PR-02E 前仅保留 spec；若 PR-02C 发现任何 test GT 产物则为 `invalid`。

Owner 已冻结四评分区间 variable-`Δt` residual rollout：transition 依次覆盖
`0→0.1→0.2→0.5→1.1 s`，显式读取 `Δt` 与区间 commanded-action schedule，初态后不再
teacher-force。

Owner 已冻结 HPO config 聚合：每个 config 先对 12 validation groups group-first 等权，再对
3 seeds 等权平均；缺失任一 seed 的 config 不可入选，平局按冻结 config ID。

三个规划问题均已关闭。C0 已由提交 `fc20023` 实现独立 `learning/` package、精确 uv lock、
离线/无 simulator runtime、clean HEAD/lineage guard、GPU reserve probe、独立 verifier 与行为
测试。`./scripts/check-pr02c-runtime` 在 Node `24.18.0` 下通过 77 项全库测试、12 个 Python
测试、14 项独立 checks 和真实 RTX 5060 Ti probe，C0 为本地 `supported`。

C1 已提交只物化 48 train + 12 validation groups 的 source producer、checksum/lineage loader、
独立 Node verifier、两层 test-split 拒绝和行为负例。代码承载 SHA `adb1a62` 的 clean gate 产生
60 groups / 300 branches、0 failed attempts；producer、loader 与 16 项独立 checks 的 data
index 均为 `2501ebc2…17a81b5`，C1 为本地 `supported`。

当前开放动作是完成 C2 clean acceptance。C2 已实现 validation-only sanitized bundle、copy-state、
constant-velocity、独立 Node verifier、canonical/reverse repeat 和 clean gate；基于既有 ignored C1
source 的 dirty diagnostic 生成 60 branches / 120 predictions，18 项 checks 全部通过，semantic
index 为 `17488a15…7c647`。它尚未提交，不能替代 clean HEAD acceptance；下一步是提交后运行
`./scripts/check-pr02c-baselines`。当前仍没有 test source/prediction、learned model、trainer、HPO、
checkpoint 或模型性能证据。

已关闭：Contract 采用 `0.2.0` 四层记录；runtime 使用隔离 `sim` optional extra；本地 clean-venv
安装与真实五分支 smoke 可复现；preflight 阈值与正式资源预算已冻结，48-group cohort 已通过；Demo 采用无 RGB
五联状态回放；source commit 使用 runtime current-clean-HEAD policy，dirty worktree fail closed；
Owner 已授权提交边界；代码承载验收 SHA `234ba00` 的 clean 一键验收与六项远端 Actions 全部
成功，PR-01A–F 里程碑关闭。
以上决策由 [`../adr/0003-pr01-sibling-evidence-stack.md`](../adr/0003-pr01-sibling-evidence-stack.md)
记录，不在本文件复制语义细节。
