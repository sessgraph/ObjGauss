# ObjGauss AI 统一开发流程

> 状态: current
> 最近更新: 2026-06-22
> 适用对象: 在本仓库工作的 Codex、Claude Code 等 AI 会话与人类贡献者。
> 本文件只定义稳定流程：工作如何进入、如何执行、如何退出。
> 项目的当前状态、队列和风险应写在 `docs/state/`，不要写进本文件。

## 0. 角色分工

- **Owner**: 定方向、确认范围、拍板重大技术和产品决策、验收结果。
- **AI 会话**: 执行者、整理器、冲突检查器。AI 不发明任务，不擅自扩大范围，不替 Owner 做重大决策。
- **事实源优先**: 多个 AI 会话互相不可见，协调必须通过仓库文件完成，不依赖聊天记忆。
- **用户明示指令优先**: 如果用户指令和本流程冲突，先指出冲突，再按用户最新指令执行。

## 1. 第一性原理

1. **注意力比代码更贵。** 让人类更难审查的大 diff、重复实现、巨型文件和模糊状态都是负资产。
2. **只为已存在的需求写代码。** ObjGauss 当前主线是对象级 Gaussian 场景预览、素材管线、训练/Demo 数据管理。不为假想需求预留复杂抽象。
3. **同一事实只有一个权威来源。** 代码逻辑一处、状态一处、素材来源一处。发现重复事实时先收敛再继续。
4. **进度的最小单位是已提交。** 未提交改动是风险。当前仓库如果还没有 baseline commit，应优先固化可运行基线。
5. **训练素材和 Demo 素材必须分层。** 研究数据、训练输出、公开展示样例不能混在一起，许可不清楚的素材只能本地测试。

## 2. 事实源地图

| 想知道 | 去哪看 |
| --- | --- |
| 项目目标与边界 | `docs/state/project-state.md` |
| 当前状态总览 | `docs/state/project-status.md` |
| 当前任务队列 | `docs/state/pr-queue.md` |
| 跨主题行动项 | `docs/state/action-queue.md` |
| 当前风险 | `docs/state/risks.md` |
| 未整理输入 | `docs/state/inbox.md` |
| 素材来源、许可、训练/Demo 分层 | `docs/asset-library.md` |
| 前端素材卡片数据 | `src/assetLibrary.js` |
| CLI 素材拉取 registry | `objgauss/assets.py` |
| 正式技术决策 | `docs/adr/` |
| AI 稳定开发流程 | `docs/development-flow.md` |

如果这些文件尚不存在，AI 在第一次流程化项目时可以创建最小版本，但不要把状态信息写进本流程文档。

冲突规则：

- 具体任务文件压倒汇总状态文件。
- 代码和测试压倒过期文档。
- 素材许可信息不明确时，按最保守方式处理。

## 3. 会话启动协议

每个新 AI 会话开工前：

1. 读 `docs/development-flow.md`。
2. 如果存在，读 `docs/state/project-status.md` 和 `docs/state/pr-queue.md`。
3. 看 `git status --short`。
4. 若工作区有大量未提交改动，先提醒 Owner 分批提交或确认继续基于当前脏工作区工作。
5. 用户请求不在队列中时，先做粒度判定，不直接扩大范围。

## 4. 立项粒度

任何文件改动前，AI 必须先说明：

- 要改哪些文件。
- 目标是什么。
- 范围外是什么。
- 准备怎么验证。

在当前 Codex 执行模式下，如果用户明确要求实现，可以直接给出简短说明后执行；如果涉及重大变更、外部依赖、训练管线或不确定取舍，必须先等 Owner 确认。

| 级别 | 判定标准 | 流程 |
| --- | --- | --- |
| 微改动 | 小文案、文档补充、窄 bugfix，通常不改变对外行为 | 简述范围和验证，直接执行 |
| 标准 PR | 一个可独立验收的功能切片 | 进入 `pr-queue.md`，必要时配任务文档 |
| 重大变更 | 新服务/包、新外部依赖、训练架构、渲染器替换、素材许可策略变化、公开发布策略变化 | 先写 ADR，Owner 确认后再动代码 |

判定不清楚时按高一级处理。

## 5. 执行循环

1. 一次只做一个目标，不把顺手优化混进来。
2. 预计 diff 超过约 800 行时，先停下来拆分。
3. 行为变化必须配行为级测试或说明为何不可测。
4. 涉及前端可视行为，必须做浏览器验证并保留截图到 `/tmp/`。
5. 涉及素材，必须记录来源、许可、本地路径、转换命令和训练/Demo 用途。
6. 涉及训练管线，必须区分原始素材、中间产物、训练素材、训练输出、Demo 样例。

## 6. 验证

默认验证：

```bash
uv run --extra dev pytest
npm run build
```

按改动增加验证：

- CLI / Python 行为变化：增加或更新 `tests/`。
- 前端 UI 变化：用浏览器或 Playwright 验证页面身份、非空、无框架报错、关键交互、截图证据。
- 素材拉取变化：至少跑 `uv run objgauss assets list`；自动拉取素材需跑一次 `uv run objgauss assets pull <asset_id>`。
- PLY / splat 转换变化：跑 `uv run objgauss stats <output.ply>`。

不要为通过验证而调松门禁或改弱测试断言。门禁本身有问题时单独立项。

## 7. 提交

- 一个提交一件事，推荐 conventional commits。
- 不提交大型数据集、训练输出、缓存、`node_modules/`、`.venv/`、`dist/`。
- `outputs/` 下的素材和训练产物默认不提交。
- `public/samples/` 只放小型可直接演示样例；许可不清楚时不要用于公开发布。
- 当前无 baseline commit 时，先做一个可运行 MVP 基线提交，再继续拆功能。

## 8. 状态回写

完成标准 PR 后：

1. 更新 `docs/state/pr-queue.md` 的状态、验收方式、完成 commit。
2. 阶段变化同步 `docs/state/project-status.md`。
3. 新风险写入 `docs/state/risks.md`。
4. 未消化输入写入 `docs/state/inbox.md`。
5. 素材变化同步 `docs/asset-library.md`、`src/assetLibrary.js`、`objgauss/assets.py`。

不回写状态，不算真正完成。

## 9. ObjGauss 专用边界

- 当前渲染器是点云预览，不是完整 3DGS splat renderer。替换为真正 splat renderer 属于重大变更。
- 当前对象分组是 KMeans MVP，不是语义分割。引入 SAM/CLIP/Gaussian Grouping 属于标准 PR 或重大变更，取决于依赖和训练成本。
- 素材分为：
  - `outputs/assets/raw/`: 原始下载。
  - `outputs/assets/converted/`: 转换中间产物。
  - `outputs/assets/training/<asset_id>/`: 训练素材。
  - `outputs/assets/gaussians/<asset_id>/`: 训练输出。
  - `public/samples/`: 小型 Demo 样例。
- 研究数据训练出的模型仍继承原始数据许可，不自动变成可商用 Demo。
- 新增外部大数据源前先确认下载规模、许可、最小子集和验收方式。

## 10. 红线

- 不写入 token、账号、客户数据或私有数据。
- 不把大型素材或训练输出提交进仓库。
- 不为未知需求新增复杂抽象、兼容层或空壳目录。
- 不在未确认许可的素材上做公开 Demo 承诺。
- 不把聊天里的临时结论当事实源，必须写入仓库文档。

## 11. 完成定义

任务可以声称完成，当且仅当：

1. 相关测试和构建已通过，或失败原因已明确说明。
2. 新行为有测试或浏览器验证证据。
3. 素材/训练/Demo 边界已记录。
4. 必要状态文件已回写。
5. 未提交改动已有清晰提交计划，或已经按粒度提交。
