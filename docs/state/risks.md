# 当前风险

> 动态风险源；稳定风险类别与失败策略见 [`../PRD.md`](../PRD.md) 和
> [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md)。
> 日期：2026-07-14

| 风险 | 当前状态 | 控制措施 | 解除证据 |
| --- | --- | --- | --- |
| `0.1.0` 被后续 contract 静默改变 | active | 固定字节 SHA、版本分派测试、禁止 migrator | PR-01A 门持续验证 hash `b619…a9c` |
| Writer 与 evaluator 共享逻辑导致伪独立 | active | PR-01D 禁止导入 writer；从 raw artifacts 重算 | mutation matrix 与源码依赖审计通过 |
| 未审计定义就批量生成导致 cohort 返工 | mitigated_local | PR-01D 先于 PR-01E；preflight 与 formal 分账 | D audit、冻结 preflight 与 48-group formal audit 均 supported |
| Pilot runtime 无法在 clean CI 重建 | mitigated_local | 精确 lock、clean-venv wheel-only install、真实 smoke；缺 runtime fail/blocked | 最终 commit SHA 的远端 CI supported |
| Dirty worktree 把旧 HEAD 冒充最终 lineage | mitigated_local | spec 只冻结 source-commit policy；runtime 注入当前 HEAD；accept/builder/verifier 都要求 clean checkout | `71d4e39` 本地 delivery supported；最终 SHA 远端 delivery 待运行 |
| 重试选择性删除科学失败 | active | attempt 独立保留；科学失败不可重试或换 seed | attempt audit 与 5% 总额外上限通过 |
| Demo 被误当机器证据 | mitigated_local | Demo 只消费已审计 episode；report 为事实源；浏览器重验 report 与 artifact checksum | 本地浏览器行为测试与 Delivery verifier supported；最终 SHA 远端 delivery 待运行 |
| 资源预算尚未由 preflight 实测冻结 | mitigated_local | 12-group preflight 独立运行 | 300s / 50MiB / 8GiB 正式预算已冻结且 formal run 在预算内 |
| Provisional effect threshold 不覆盖第二 object spec | closed_negative | 保留 rejected report，不重试/换 seed；由隔离 preflight 冻结阈值 | `0.0014 m` 下完整 preflight 与 formal cohort 独立 audit supported |
