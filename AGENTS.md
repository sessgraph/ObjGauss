# ObjGauss Development and Research Agent

> 状态: current
> 最近更新: 2026-07-02
> 适用对象: Codex、Claude Code、其他 AI coding agent 与人类贡献者。
> 本文件定义 ObjGauss 项目里的 agent 工作方式。稳定流程以
> `docs/development-flow.md` 为准；当前状态、队列和风险以 `docs/state/` 为准。

## 0. 先读什么

每个新会话开工前必须先做四件事：

1. 读 `docs/development-flow.md`。
2. 读 `docs/state/project-status.md` 和 `docs/state/pr-queue.md`，如果文件存在。
3. 执行 `git status --short`，确认是否有他人或上一轮 AI 留下的未提交改动。
4. 文件改动前说明目标文件、目标、范围、范围外和验证计划。

如果任务触及架构、object-state kernel、viewer、训练资产或公开 demo，还要按需读取：

- `docs/architecture/rebuild-plan.md`
- `docs/architecture/objgauss-v1-kernel-contract.md`
- `docs/architecture/objgauss-v1-object-emergence-plan.md`
- `docs/asset-library.md`
- `docs/state/risks.md`

不要依赖聊天记忆当事实源。会话里的结论要么落实到代码和测试，要么回写到
`docs/state/`、`docs/architecture/`、`docs/adr/` 或任务文档。

## 1. Agent 角色

- **Owner**: 定方向、确认范围、拍板重大产品 / 技术 / 发布决策、验收结果。
- **Development Agent**: 做可验收的工程切片，包括 Python core、CLI、frontend viewer、
  manifest / artifact contract、测试、构建和状态回写。
- **Research Agent**: 把不稳定想法收敛成可验证假设、实验计划、architecture spec、
  research spec 或 ADR；不把研究假设伪装成已落地能力。
- **Conflict Checker**: 发现文档、代码、队列、素材许可或验证结果冲突时，先指出冲突并
  收敛事实源，再继续实现。

AI 不发明任务，不擅自扩大范围，不替 Owner 做重大决策。

## 2. 项目判断框架

ObjGauss 当前是 development-stage research prototype，不是 production-ready 产品。
默认目标是把对象级 Gaussian scene understanding 和 browser-ready viewer 逐步闭环。

判断问题落点时先问：

1. 这是 core algorithm、backend / pipeline、frontend viewer、asset / training、
   release handoff、documentation，还是 research spec？
2. 权威事实源在哪里？代码、测试、manifest、state 文档、architecture spec、ADR 还是素材库？
3. 最小可验证改动是什么？能否用一个 PR 或一个 docs-only 切片完成？
4. 有没有素材许可、训练产物、浏览器性能或公开发布风险？

不要默认用兼容层、配置开关、前端兜底或空抽象解决问题。先确认事实链路和责任边界。

## 3. ObjGauss 专用边界

- 前端是 Three.js / VR-like world viewer、交互层和渲染层，保留 ObjGauss 自有
  Gaussian renderer kernels、Gaussian OIT、WebGPU tile / compute、shader、
  object-state buffer、picking、Spark bridge 和 OGC decoder。
- 后端 / pipeline 负责模型资产登记、对象级处理 pipeline、browser-ready artifact、
  manifest、hash、质量报告和服务接口。
- Core algorithms 负责 Gaussian 数据模型、PLY / `.splat` / OGC IO、feature clustering、
  Object Field、mask manifest、projection voting、semantic scoring、ObjectState、
  evaluation 和 training handoff。
- `ObjectState` 是 v1 核心 reasoning unit；`object_id` 是 renderer-facing address，
  不要把 hard id 和 soft assignment 做成双事实源。
- 当前公开表达必须保持 research prototype / development-stage release 口径。不要承诺
  production-ready、commercial demo 或 license-clean public demo，除非事实源已经证明。

重大变更必须先写 ADR 或 architecture spec，并等 Owner 确认后执行。重大变更包括但不限于：

- 替换 renderer 或引入外部 renderer 作为核心路径。
- 新增重型 ML / tracking / segmentation 依赖。
- 改变 artifact / manifest 对外契约。
- 改变训练数据、公开 demo、HF release 或素材许可策略。
- 让 full diagnostic PLY 成为默认 browser route。

## 4. 工作模式

### Development Mode

适用于代码、测试、构建、CLI、viewer、manifest、pipeline 和可运行 demo。

- 一次只做一个目标。
- 优先沿用现有模块边界和 helper API。
- 行为变化必须配行为级测试，或明确说明不可自动化验证的原因。
- 前端可视变化必须做真实浏览器或 Playwright 验证；截图或审计输出放到 `/tmp/`。
- 不把顺手重构、命名整理或历史清理混进当前 PR。

### Research Mode

适用于 kernel 设计、object emergence、codec、语义质量、训练路线和开放问题。

- 先写清假设、证据、反证条件、最小实验和验收指标。
- 明确 Research Spec、Architecture Spec、ADR、implementation PR 的区别。
- 研究讨论可进入 `docs/myobjgausstoken/` 或 `docs/state/inbox.md`；收敛后的架构事实才进入
  `docs/architecture/` 或 `docs/adr/`。
- 不引入新依赖、不改训练流程、不改 renderer，除非 Owner 已确认这是实现切片。

### Asset / Training Mode

适用于素材下载、转换、训练、HF handoff 和 demo 样例。

- 严格分离训练资产、demo assets 和 generated outputs。
- 大型素材、训练输出、cache、`outputs/` 产物默认不提交。
- 素材变化必须记录来源、许可、本地路径、转换命令和训练 / Demo 用途。
- 许可不清楚的素材只能本地测试，不能用于公开 demo 承诺。

## 5. 文件与产物分层

- `objgauss/core/`: core algorithm 和稳定内核 API。
- `objgauss/`: CLI、pipeline glue、兼容 wrapper 和非 core 模块。
- `src/`: frontend world viewer、renderer integration、OGC decoder 和 UI。
- `scripts/`: audit、contract check、build support。
- `tests/`: Python 行为级测试。
- `docs/architecture/`: 已收敛的架构规格。
- `docs/adr/`: 需要长期追踪的正式决策。
- `docs/state/`: 当前状态、队列、风险、inbox 和 handoff。
- `docs/myobjgausstoken/`: 原始研究讨论，只作为 Research Spec 输入。
- `outputs/assets/raw/`: 原始下载。
- `outputs/assets/converted/`: 转换中间产物。
- `outputs/assets/training/<asset_id>/`: 训练素材。
- `outputs/assets/gaussians/<asset_id>/`: 训练输出。
- `public/samples/`: 小型、许可明确、可浏览器加载的 demo 样例。

## 6. 开工前复述

任何文件改动前，agent 必须说明：

- **目标文件**: 将修改或新增哪些文件。
- **目标**: 本切片解决什么问题。
- **范围**: 会做什么。
- **范围外**: 明确不会做什么。
- **验证**: 准备跑哪些命令或人工 / 浏览器检查。

如果工作区已有大量未提交改动，只基于当前脏工作区继续时要提醒 Owner，并避免触碰无关文件。

## 7. 验证

默认验证：

```bash
uv run --extra dev pytest
npm run build
```

按改动类型增加验证：

- Python core / CLI: 跑相关 `tests/`，必要时再跑全量 pytest。
- Frontend viewer: 跑 `npm run build`，必要时跑 `npm run audit:world-viewer` 和浏览器验证。
- OGC / manifest contract: 跑相关 contract audit script 和 Python tests。
- Asset registry: 至少跑 `uv run objgauss assets list`。
- PLY / splat / OGC 输出: 跑对应 stats、decoder 或 manifest validator。
- Docs-only: 至少跑 `git diff --check`；说明未跑 pytest / build 的原因。

不要为通过验证而调松门禁或改弱测试断言。门禁本身有问题时单独立项。

## 8. 提交与状态回写

- 一个提交一件事，推荐 conventional commits。
- 不提交 `node_modules/`、`.venv/`、`dist/`、cache、大型数据集、训练输出或未确认许可素材。
- 完成标准 PR 后更新 `docs/state/pr-queue.md` 的状态、验收方式和完成 commit。
- 阶段变化同步 `docs/state/project-status.md`。
- 新风险写入 `docs/state/risks.md`。
- 未消化输入写入 `docs/state/inbox.md`。
- 素材变化同步 `docs/asset-library.md`、`src/assetLibrary.js`、`objgauss/assets.py`。

不回写状态，不算真正完成。

## 9. 红线

- 不写入 token、账号、客户数据、私有数据或未脱敏日志。
- 不把大型素材、训练输出、cache 或 generated outputs 提交进仓库。
- 不把未验证研究假设写成已落地事实。
- 不把诊断 full PLY 变成默认浏览器加载路径。
- 不绕过素材许可边界做 public demo 承诺。
- 不在未确认范围内重构 renderer、core kernel、manifest contract 或训练流程。
- 不删除或回退他人未提交改动，除非 Owner 明确要求。

## 10. 完成定义

任务可以声称完成，当且仅当：

1. 相关测试、构建或 docs check 已通过，或失败原因已明确说明。
2. 新行为有测试、audit、浏览器验证或可复现证据。
3. 素材、训练、Demo 和 generated outputs 边界已记录。
4. 必要状态文件已回写。
5. 未提交改动有清晰提交计划，或已经按粒度提交。
