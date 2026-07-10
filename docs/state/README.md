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
- `archive/`: 已完成任务和历史状态的只读归档；不作为当前状态入口。

## 更新规则

- 行为或阶段变化后更新 `project-status.md`。
- 标准 PR 完成后更新 `pr-queue.md`。
- 新风险或风险状态变化更新 `risks.md`。
- 素材来源、许可、训练/Demo 分层变化同步 `docs/asset-library.md`。
- 汇总文件和底层事实冲突时，以具体任务或代码为准，并修正汇总。
- `project-status.md` 目标控制在约 200 行以内；`pr-queue.md` 不保留完整 Done 流水账。
- 历史内容移动到 `archive/`，active 文件只回答“现在是什么、卡在哪里、下一步是什么”。
- `risks.md` 只保留当前风险表；逐次缓解历史同样进入 `archive/`。
