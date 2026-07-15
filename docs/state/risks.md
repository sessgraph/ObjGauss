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
