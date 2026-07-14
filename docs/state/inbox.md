# 开放事项

> 只记录会影响近期执行、但尚未由本地或外部证据关闭的事项。
> 日期：2026-07-14

| 项目 | 状态 | 所需证据/动作 | 影响切片 |
| --- | --- | --- | --- |
| GitHub runner 是否能满足固定 simulator runtime 与 writer | open | 不 skip 的远端 job 结果；缺 runtime 必须失败或 blocked | PR-01B / PR-01C / PR-01F |
| PR-01 最终 SHA 远端验收 | open | 当前最终 clean HEAD 的 `./scripts/accept-pr01` 已 supported；要求相同 SHA 的远端 delivery job supported | PR-01F |

已关闭：Contract 采用 `0.2.0` 四层记录；runtime 使用隔离 `sim` optional extra；本地 clean-venv
安装与真实五分支 smoke 可复现；preflight 阈值与正式资源预算已冻结，48-group cohort 已通过；Demo 采用无 RGB
五联状态回放；source commit 使用 runtime current-clean-HEAD policy，dirty worktree fail closed；
Owner 已授权提交边界，PR-01A–F 由 `71d4e39` 固化且 implementation HEAD 的 clean 一键验收通过。
以上决策由 [`../adr/0003-pr01-sibling-evidence-stack.md`](../adr/0003-pr01-sibling-evidence-stack.md)
记录，不在本文件复制语义细节。
