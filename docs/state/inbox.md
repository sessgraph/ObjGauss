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

当前唯一开放动作是 PR-02C Trainer/Baselines 的 Owner 单独授权。授权前不得创建 `learning/`
package、安装训练依赖或运行训练；PR-02B supported 也不构成模型性能或 Gaussian dynamics 证据。

已关闭：Contract 采用 `0.2.0` 四层记录；runtime 使用隔离 `sim` optional extra；本地 clean-venv
安装与真实五分支 smoke 可复现；preflight 阈值与正式资源预算已冻结，48-group cohort 已通过；Demo 采用无 RGB
五联状态回放；source commit 使用 runtime current-clean-HEAD policy，dirty worktree fail closed；
Owner 已授权提交边界；代码承载验收 SHA `234ba00` 的 clean 一键验收与六项远端 Actions 全部
成功，PR-01A–F 里程碑关闭。
以上决策由 [`../adr/0003-pr01-sibling-evidence-stack.md`](../adr/0003-pr01-sibling-evidence-stack.md)
记录，不在本文件复制语义细节。
