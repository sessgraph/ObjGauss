# 开放事项

> 只记录会影响近期执行、但尚未由本地或外部证据关闭的事项。
> 日期：2026-07-15

当前没有 PR-01 范围内的开放事项。PR-02 的主要假设、primary endpoint、训练栈、硬资源上限、
数据隔离、统计方法和 `02A`–`02F` 切片已经由 Owner 确认并写入
[`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)。PR-02A 已获授权、本地实现、通过项目门并提交，但
尚未获得远端 CI 证据。当前开放项是是否另行授权 PR-02B；不能因 contract supported 就创建
`learning/` package、安装训练依赖或运行 pilot/训练。

`PR-02B` 仍须在隔离 calibration/power pilot 中实测并冻结 horizon、scales、`δ`、
`δ_shuffle`、group/seed 数和训练配置；这些是待运行证据，不是可在 formal 结果后调整的开放
产品选择。PR-02A 当前仅为 local `supported`；PR-02B 在动作级授权前不得运行。

已关闭：Contract 采用 `0.2.0` 四层记录；runtime 使用隔离 `sim` optional extra；本地 clean-venv
安装与真实五分支 smoke 可复现；preflight 阈值与正式资源预算已冻结，48-group cohort 已通过；Demo 采用无 RGB
五联状态回放；source commit 使用 runtime current-clean-HEAD policy，dirty worktree fail closed；
Owner 已授权提交边界；代码承载验收 SHA `234ba00` 的 clean 一键验收与六项远端 Actions 全部
成功，PR-01A–F 里程碑关闭。
以上决策由 [`../adr/0003-pr01-sibling-evidence-stack.md`](../adr/0003-pr01-sibling-evidence-stack.md)
记录，不在本文件复制语义细节。
