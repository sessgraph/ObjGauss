# 状态文件维护规则

`docs/state/` 是 ObjGauss 当前状态的事实源。新 AI 会话开工前先读：

1. `docs/development-flow.md`
2. `docs/state/project-status.md`
3. `docs/state/pr-queue.md`

## 文件职责

- `project-state.md`: 项目目标、边界、非目标。
- `project-status.md`: 当前可运行能力、验证状态、最近阶段。
- `pr-queue.md`: 可独立验收的工作队列。
- `action-queue.md`: 跨主题行动项和非 PR 型工作。
- `risks.md`: 当前风险、缓解措施、关闭条件。
- `inbox.md`: 未整理输入，只追加，整理后转入队列或风险。

## 更新规则

- 行为或阶段变化后更新 `project-status.md`。
- 标准 PR 完成后更新 `pr-queue.md`。
- 新风险或风险状态变化更新 `risks.md`。
- 素材来源、许可、训练/Demo 分层变化同步 `docs/asset-library.md`。
- 汇总文件和底层事实冲突时，以具体任务或代码为准，并修正汇总。
