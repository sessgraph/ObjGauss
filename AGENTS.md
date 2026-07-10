# AGENTS.md — ObjGauss Development and Research

本文件定义 Codex、Claude Code、其他 AI coding agent 与人类贡献者在 ObjGauss 仓库中的长期工作约定。这里只保存稳定的项目事实、工程边界、工作协议和验收规则；操作细节参考 `docs/development-flow.md`，当前状态、任务队列和风险以 `docs/state/` 为准。若流程文档中的能力描述或命令与代码、测试、accepted ADR、CI 或本文件的事实源地图冲突，先按证据定位漂移并修正权威投影，不沿用过期描述。

## 1. 项目罗盘（Project Compass）

- **长期目标**：验证并构建对象级可编辑、可持续推理的 3D Gaussian 场景。
- **当前阶段**：research-first development prototype（研究优先的开发阶段原型）；先用真实 controlled scenes 证明 `ObjectState` 的 identity persistence、predictive sufficiency 和 action-conditioned transition，再讨论产品扩张。
- **主要交付物**：可复现的 Gaussian/ObjectState 研究内核、数据与评估工具、browser-ready artifact/manifest，以及只展示已闭环能力的 evidence viewer。
- **核心推理单元**：`ObjectState`；`object_id` 只是 renderer-facing address（渲染寻址）。
- **明确非目标**：当前不声称 production-ready、已验证 world model、commercial demo 或 license-clean public demo。
- **默认方法**：以一个 PR 可完成的最小可验证切片推进；不新增用来掩盖责任边界或未接通路径的 wrapper、fallback、配置开关、前端兜底或空抽象。

### 事实源地图

| 想知道 | 权威来源 |
| --- | --- |
| 当前实际行为 | 代码、测试和可复现运行结果 |
| 项目目标、研究主张与能力边界 | `docs/state/project-state.md` |
| 当前阶段与已验证/未验证能力 | `docs/state/project-status.md` |
| 当前任务、范围与验收 | `docs/state/pr-queue.md`、`docs/state/action-queue.md` 及任务直接引用的 spec |
| 公共 artifact/manifest/schema 与 kernel contract | 对应 canonical 实现、validator 和 contract tests；稳定语义见 `docs/architecture/` |
| 稳定架构与长期决策 | `docs/architecture/`、`docs/adr/` |
| 风险与未整理输入 | `docs/state/risks.md`、`docs/state/inbox.md` |
| 素材来源、许可与 training/demo 分层 | `docs/asset-library.md`；代码 registry 是消费镜像，必须与其一致 |
| 原始研究讨论 | `docs/myobjgausstoken/`；仅作为 Research Spec 输入，不是已落地事实 |

代码和运行证据回答“现在实际发生什么”；已批准的 contract、task 和 ADR 回答“规范要求什么”；status 是这些事实的投影。来源冲突时先复现、定位责任源并修正错误投影，不静默选择方便实现的一方。

### 第一性原则

- **注意力稀缺**：代码生成便宜，理解与审查昂贵；保持 diff 小而内聚，避免重复实现、巨型文件和无关重构。
- **先找责任层**：实现前判断问题属于 core algorithm、datasets、evaluation、pipeline/backend、viewer/frontend、asset/training、release handoff、documentation 还是 research spec。
- **只为真实需求实现**：没有当前需求和真实调用方，不新增服务、包、占位目录或假想扩展点。
- **一个事实一个来源**：schema、指标、状态、命令和许可各有权威来源；必要镜像必须标明消费关系并有一致性验证。
- **证据匹配结论**：不把计划、candidate、accounting、mock、fixture、simulation、baseline、oracle 或单次指标描述成尚未实际运行并验证的能力。
- **完成状态分层**：已实现、已验证、已提交和已集成是不同状态，只能声明实际达到的层级。
- **显性与隐性知识分开**：仓库保存已确认知识；Owner 掌握但尚未编码的语义必须通过第 7 节收敛，不由 Agent 脑补。

处理非平凡问题时先明确：真实研究或用户结果、权威事实源、当前状态、核心不变量、失败/错误代价、最小反证或验证、明确非目标，以及是否涉及数据泄漏、素材许可、训练产物、浏览器性能或公开发布风险。

## 2. 指令加载与开工阅读

- Codex 本地自动加载项目根到当前工作目录的指令；修改更深子树前，主动读取目标路径上的 `AGENTS.override.md` / `AGENTS.md`。局部文件只写该子树相对根规则的差异。
- 每个新会话开始时读取 `docs/development-flow.md`、`docs/state/project-status.md` 和 `docs/state/pr-queue.md`（存在时），然后运行 `git status --short`。
- 工作区存在大量未提交改动时，先提醒 Owner；继续工作时把现有改动视为用户或其他工作流的成果，不覆盖、回退、暂存、格式化或整理无关文件。
- 按任务补充读取：
  - 架构与重建：`docs/architecture/rebuild-plan.md`
  - 内核契约：`docs/architecture/objgauss-v1-kernel-contract.md`
  - object emergence：`docs/architecture/objgauss-v1-object-emergence-plan.md`
  - 素材与许可：`docs/asset-library.md`
  - 风险：`docs/state/risks.md`
- 只读取与任务相关的最小上下文，优先使用 `rg` / `rg --files` 定位真实入口、调用方、contract、状态所有权和测试。
- 聊天记忆不能替代仓库事实；需要成为项目事实的结论必须进入代码/测试、contract、architecture、ADR 或明确的 `docs/state/` 事实源。

## 3. 仓库地图与所有权

| 路径 | 唯一责任 |
| --- | --- |
| `objgauss/core/` | canonical 边界是 Gaussian 表示与 I/O、assignment、`ObjectState`、decoder 和 metric primitives；当前仍暂存待队列切片外移的 legacy orchestration/compatibility modules，其物理位置不代表长期 ownership |
| `objgauss/datasets/` | controlled dataset/capture 的 canonical schema、manifest、校验与 authoring workflow |
| `objgauss/evaluation/` | identity、prediction、intervention evaluator，reality gate 与 canonical evaluation rows |
| `objgauss/` | CLI、pipeline glue、asset registry、剩余非 core orchestration 与迁移期 compatibility wrapper |
| `src/` | Three.js/Spark evidence viewer、renderer integration、OGC decoder、交互与 UI |
| `scripts/` | audit、contract check、训练/构建支持与可复现命令入口 |
| `tests/` | Python 行为级、contract、兼容性与回归测试 |
| `docs/architecture/` | 已收敛、可实施的架构规格 |
| `docs/adr/` | 需要长期追踪的正式决策 |
| `docs/state/` | 当前状态、任务队列、风险、inbox 与 handoff |
| `docs/myobjgausstoken/` | 原始研究讨论，只作为 Research Spec 输入 |
| `outputs/assets/raw/` | 原始下载素材，不提交 |
| `outputs/assets/converted/` | 转换中间产物，不提交 |
| `outputs/assets/training/<asset_id>/` | 训练素材，不默认提交 |
| `outputs/assets/gaussians/<asset_id>/` | 训练输出，不默认提交 |
| `public/samples/` | 小型、许可明确、可由浏览器加载的 demo 样例 |

## 4. 架构与契约不变量

- Frontend/viewer 负责 Three.js/Spark world viewer、交互与渲染，保留 ObjGauss 自有 Gaussian renderer kernels、Gaussian OIT、WebGPU tile/compute、shader、object-state buffer、picking、Spark bridge 与 OGC decoder。
- Backend/pipeline 负责模型资产登记、对象级处理 pipeline、browser-ready artifact、manifest、hash、质量报告和服务接口，不依赖浏览器 renderer internals。
- Core algorithms 的 canonical 边界只包括 Gaussian、assignment、`ObjectState`、decoder 与 metric primitives；仍位于 core 的 dataset、filesystem workflow、report、handoff 和 orchestration 是 `docs/state/pr-queue.md` 跟踪的迁移债务，不得据其现状继续扩张 core，也不得将其描述成已经全部外移。
- `objgauss.datasets` 与 `objgauss.evaluation` 是当前 controlled-data 和 evaluation 实现的 canonical 包；旧 `objgauss.core` 路径仅可保留明确批准、对象身份不变且有 import/object-identity contract tests 的兼容导入。
- `ObjectState` 是核心 reasoning unit；`object_id` 是由 soft assignment、matching 与 export policy 派生的 renderer-facing address。不得让 hard id 与 soft assignment 演化成两个冲突事实源。
- 对外 artifact/manifest/schema 的字段、版本、单位、坐标系、哈希和兼容语义必须唯一且可测试；viewer 不得通过猜测弥补 contract 缺失。
- Full diagnostic PLY 是诊断产物，不得成为默认 browser route。
- 新依赖与跨层调用必须尊重当前所有权；不为了套用架构模式而重构，也不新增 compatibility path 来伪装边界迁移已经完成。

以下重大变更若没有当前任务的明确授权或既有已批准决策，必须先形成 ADR 或 Architecture Spec，由 Owner 确认后再实现：

- 替换 renderer，或把外部 renderer 引入核心路径；
- 新增顶层 package、service、middleware 或基础设施边界；
- 新增重型 ML、tracking 或 segmentation 依赖；
- 改变 artifact/manifest 的对外 contract；
- 改变 `ObjectState` kernel contract，或 hard id 与 soft assignment 的事实关系；
- 改变训练数据、公开 demo、Hugging Face release 或素材许可策略；
- 让 full diagnostic PLY 进入默认浏览器路径。

## 5. 授权、任务粒度与领域模式

- **Owner**：确定方向和范围，审批重大产品、技术、依赖、许可与发布决策，验收结果。
- **Development Agent**：交付可验收的代码、测试、CLI、viewer、manifest、pipeline、构建和必要状态回写。
- **Research Agent**：把想法收敛成可证伪假设、实验、Research Spec、Architecture Spec 或 ADR，不把假设伪装成能力。
- **Conflict Checker**：发现事实冲突时先定位证据与责任源，再继续工作。

AI 不自行新增产品目标、研究结论或独立 PR，不替 Owner 作重大决策，也不把相关但超出范围的问题顺手并入当前切片；完成当前验收所需的正常实现、测试和必要文档同步可自主推进。

当前会话中有权管理本仓库并发出明确任务的用户即 Owner；其明确请求视为该任务范围内可逆仓库修改与本地验证的授权。授权模式与领域模式是两个维度：

- **解释 / 评审 / 状态报告**：默认只读；未经要求不修改。
- **诊断**：定位根因、影响和验证方法；诊断本身不等于获准修复。
- **规划**：明确边界、风险、依赖、验收和最小切片；未经要求不提前实现。
- **修改 / 修复 / 创建 / 构建**：完成当前范围内的实现、适用验证、自审和必要事实源同步。
- **commit / amend / rebase / push / merge / tag / 发布 / 部署 / 生产写入 / 外部消息**：始终需要动作级明确授权。

| 级别 | 判定 | 处理 |
| --- | --- | --- |
| 微改动 | 局部、低风险、易回滚，不改变公共 contract、数据/许可或发布边界 | 已有实现授权时直接完成，运行最相关检查 |
| 标准切片 | 一个可独立验收的功能或修复，可能跨多个文件或模块 | 使用现有 task/queue，完成实现、验证和触发式回写 |
| 重大变更 | 核心架构、破坏性 contract、训练/数据策略、安全、许可、生产或不可逆边界变化 | 先做只读调研与 ADR/Architecture Spec，Owner 决策后实现 |

风险、不可逆性、contract 与证据影响比代码行数更重要。

### Development Mode

- 一次只交付一个可观察目标，沿用现有模块边界和 helper API。
- 行为变化应有行为级测试；无法自动化时说明原因并提供可复现证据。
- Viewer 的用户可见行为或布局变化必须做真实浏览器或 Playwright 验证；截图和审计临时输出放 `/tmp/`。
- 不把命名整理、历史清理、全库格式化或无关重构混进当前切片。

### Research Mode

- 先写假设、已有证据、反证条件、最小实验、指标、基线与通过/停止条件。
- 区分 Research Spec、Architecture Spec、ADR 和 implementation PR；讨论不等于批准，spec 不等于实现。
- 经 Owner 要求持久化或会改变本次实现事实的原始讨论进入 `docs/myobjgausstoken/` 或 `docs/state/inbox.md`；收敛后的稳定事实才进入 `docs/architecture/` 或 `docs/adr/`。纯讨论不自动写仓库。
- 明确区分真实 model inference、预生成 candidate consumption、matching/accounting、baseline 与 oracle；指标不能替代未实际运行的能力。
- 显式检查数据、教师信息、oracle、identity 与 future-state leakage；每个指标应可追溯到输入、分母、过滤、阈值和产物。
- 当前任务未明确授权时，不引入依赖、不改训练流程、不改 renderer；需要实现时先由 Owner 批准对应切片。

### Asset / Training Mode

- 严格分离 training assets、demo assets 和 generated outputs。
- 大型素材、训练输出、cache 和 `outputs/` 产物默认不提交。
- 素材变更必须记录来源、许可、本地路径、转换命令和 training/demo 用途。
- 许可不清的素材只能用于本地研究，不得进入 public demo、release 或对外承诺。

## 6. Change Contract 与执行纪律

任何文件改动前，先给出简短 Change Contract；微改动可压缩成一行：

- **目标文件**：预计修改/新增的文件；
- **目标**：本切片的用户或研究价值、可观察结果；
- **范围 / 范围外**：会做什么、明确不做什么；
- **验证**：相关测试、构建、audit 或浏览器检查；
- **提交粒度**：仅在 Owner 要求提交时说明预计意图和边界。

执行时：

- 复杂任务先明确目标、边界、状态、不变量、失败路径和验收证据；简单任务直接完成。
- 优先修复根因并保持 diff 小而内聚；兼容迁移只允许第 4 节所述的已批准例外。
- 未经要求，不改变公开 API、CLI、schema、manifest、默认 browser route、序列化格式、默认值、持久化或错误语义。
- 当前任务未明确授权时，新增生产依赖、外部服务、存储、基础设施或重型模型前必须获得 Owner 确认。
- 不手工编辑可由正式生成器重建的产物；修改权威源并运行已有生成命令。
- 新发现但超出范围的问题不顺手修复；实现任务需要回写时按现有 inbox/risk/task 路由记录，否则在最终回复中报告 Owner。

修改公共 contract 时，按适用项明确 producer、consumer、版本、必填/可选字段、默认值、单位、坐标系、时间语义、唯一标识、hash、兼容策略、迁移和回滚；同一概念只保留一个 canonical schema。

新增或修改异步、可重试、持久化、跨系统或控制类流程时，按适用项明确合法状态与转换、幂等、并发与顺序、取消/超时/有限重试、部分失败与恢复、审计证据，以及可注入的时间、随机数和外部依赖。“请求已接受”不等于“业务状态已生效”，除非 contract 明确保证。

涉及研究数据或 evidence 时，同时定义来源/Owner、适用范围、单位/坐标系、时间与粒度、精度/缺失值、质量或置信度、敏感与许可级别、消费方，以及指标的分母、过滤和 leakage 边界。

## 7. `grill-me` 与 Knowledge Capture

用户明确提出 `grill me` 或 `grill-me` 时进入知识质询模式。否则，仅当任务依赖仓库中未编码的产品/研究语义，且多个合理解释会改变 contract、架构、安全、证据或验收时，先说明知识缺口并询问是否进入；可查问题和不改变语义的细节不阻塞执行。

进入后：

1. 先查代码、contract、文档和状态源，整理“已知 / 冲突 / 未知 / 受影响决策”，不把可查问题交回 Owner。
2. 每轮只问一个主问题；给出 Agent 推荐答案、证据、取舍，以及答案不同会改变什么。
3. 优先收敛目标用户或研究结果、事实源、核心不变量、异常/错误代价、数据与指标语义、权限边界、验收证据和明确非目标。
4. 对前后矛盾主动指出；非阻塞细节可采用显式标注的默认值，改变语义或验收的未知项必须确认或标记 blocked。
5. 信息足以支持下一项 plan、ADR、spec 或实现决策时退出，并输出收敛摘要。

候选知识按以下状态区分：

- `confirmed_fact`：Owner 明确确认的当前事实，或仓库权威证据直接证明的事实；
- `decision`：Owner 作出的取舍，包含理由和适用范围；
- `working_assumption`：为继续工作暂时采用但尚未确认的假设；
- `open_question`：仍可能改变语义、contract、证据或验收的问题；
- `superseded`：已被新事实替代，需链接替代项而不是静默覆盖。

知识写入遵循现有事实源：研究目标/边界进入 `docs/state/project-state.md`，当前能力进入 `project-status.md`，任务范围进入 queue/spec，稳定取舍进入 ADR/architecture，未确认研究输入进入 `docs/myobjgausstoken/` 或 inbox，风险进入 risks，素材与许可进入 asset library。公共 contract 写入 canonical schema/validator/tests，文档只链接或解释，不复制出第二套 contract。

纯讨论、评审或规划没有写入授权时，只在回复中输出候选结论，不静默修改仓库。用户明确要求记录/沉淀，或实现任务中新确认的语义直接影响本次行为时，必要文档同步属于授权范围。不得沉淀 token、凭据、PII、真实客户原始数据或未脱敏生产 payload。

## 8. 权威命令与风险驱动验证

下列命令已与当前锁文件、脚本或 CI 核对，均从仓库根目录运行：

| 目的 | 命令 |
| --- | --- |
| 同步 Python 开发环境 | `uv sync --locked --extra dev` |
| 同步前端依赖 | `npm ci` |
| 启动固定端口开发服务 | `npm run dev` |
| Python 完整 CI 等价门禁 | `uv run --locked --extra dev pytest` |
| Viewer production build | `npm run build` |
| Viewer audit | `npm run audit:world-viewer` |
| OGC decoder contract audit | `npm run audit:ogc-decoder-contract` |
| Asset registry smoke | `uv run objgauss assets list` |
| 文档/whitespace 检查 | `git diff --check` |

当前没有独立的 lint 或 typecheck 权威脚本；不得为补齐模板而发明命令。`npm run dev` 的固定 `127.0.0.1:5395` 端口与占用处理以 `docs/development-flow.md` 为准。

| 风险 | 典型改动 | 最低证据 |
| --- | --- | --- |
| 低 | 文档、注释、无行为变化配置、局部纯函数 | 定向 diff、`git diff --check`、最相关检查 |
| 中 | Python/CLI、viewer、内部数据流或单模块行为 | 行为回归测试、相关 integration test、适用 build/audit |
| 高 | 公共 contract、ObjectState ABI、持久化、训练/数据策略、renderer、发布或控制路径 | 完整相关门禁、兼容/迁移/回滚验证和端到端证据 |

迭代时先跑最小相关检查，完成前按影响面扩大：

- Python core / CLI：运行相关 `tests/`；行为或公共路径变化时再跑完整 Python 门禁。
- Frontend viewer：运行 `npm run build`；用户可见变化再跑 `npm run audit:world-viewer` 和真实浏览器/Playwright 验证。
- OGC / manifest contract：运行相关 audit、Python contract tests 与 decoder/validator。
- Asset registry：至少运行 `uv run objgauss assets list`；自动拉取变化按 development flow 验证 pull。
- PLY / `.splat` / OGC 输出：运行相应 stats、decoder、round-trip 或 manifest validator。
- Docs-only：至少运行 `git diff --check`；说明未跑 pytest/build 的原因。
- 单栈、低风险代码改动可不运行明确无关的另一栈门禁，但必须说明判断依据和未覆盖风险。

验证纪律：

- Bug 可稳定复现时，先确认回归测试在修复前失败。
- 行为变化必须有行为级测试；无法自动化时说明原因并给出可复现证据。
- 不删除测试、skip、放宽断言、吞掉异常或未经审查更新阈值/基线来制造通过。
- 失败门禁本身有问题时单独立项，不混入当前功能切片。
- 完成前检查 `git diff --check`、任务相关 diff、工作区状态和意外生成物。
- 无法运行某项检查时，如实记录准确命令、失败原因、替代证据和残余风险。

## 9. 状态、文档、素材与 Git

- 标准实现任务改变队列事实时，更新 `docs/state/pr-queue.md` 的状态和验收证据；已有完成 commit 时记录真实 hash，尚未提交时不得编造。若 queue schema 规定 `done` 必须有 commit，则保持真实中间状态，不临时发明新状态。
- 阶段或能力边界变化同步 `docs/state/project-status.md`；新风险写入 `docs/state/risks.md`；未消化输入写入 `docs/state/inbox.md`。
- 行为、架构、contract、迁移、运维或发布方式变化时更新对应权威文档；不把愿景或未验证研究结果写成当前能力。
- 新增或修改已登记的 demo/training asset 元数据时，以 `docs/asset-library.md` 的来源与许可记录为准，并按实际消费方同步 `src/assetLibrary.js` 和 `objgauss/assets.py`；临时下载、转换中间物和训练输出不触发 registry 回写。
- 只读评审、诊断、纯讨论或未改变项目状态的任务不为形式而回写状态文档。
- 不编造版本、完成状态、测试结果、日期或 commit hash。
- 不默认 commit、amend、rebase、push、merge、tag、发布或部署。Owner 明确要求提交时，一个提交只表达一个意图，优先 conventional commits，并精确暂存相关文件而非 `git add .`。
- 不提交 `node_modules/`、`.venv/`、`dist/`、cache、大型数据集、训练输出或许可未确认素材。

## 10. 安全与红线

- 不主动读取、输出、写入或提交 token、账号、凭据、客户数据、私有数据和未脱敏日志。
- 不把研究假设、candidate accounting、oracle 上限或单次指标提升描述成已落地模型能力。
- 不绕过素材许可边界作 public demo、release 或商业可用承诺。
- 不在未确认范围内重构 renderer、core kernel、manifest contract 或训练流程。
- 不让测试触达真实用户、生产服务、生产数据、设备控制或未授权外部系统。
- 禁止 `git reset --hard`、`git clean -fd`、覆盖式恢复和强制推送，除非 Owner 明确要求且风险已确认。
- 产品语义冲突、破坏性迁移、秘密/额外权限、重大范围扩张或外部副作用会实质改变结果时，暂停并请求最小必要决策；普通任务内步骤自主推进。

## 11. 多 Agent 协作

若当前工具支持 Subagents（子代理）：

- 仅在子任务独立、边界清晰且并行有实际收益时委派，优先只读调研、日志分析、独立测试和专项审查。
- 主 Agent 保留需求解释、架构决策、最终集成、知识/状态回写和验证责任，并复核子 Agent 结论与仓库证据。
- 并行写任务必须划分文件、模块和共享事实源所有权，不让多个 Agent 修改同一 contract 或重叠文件。
- 子 Agent 不得扩大原任务授权，也不得自行 commit、改外部系统或发布。
- 简单任务不递归扇出；汇总前等待必要结果并检查共享工作区状态。

## 12. 完成定义与完成层级

只读任务完成时说明：检查范围、关键证据、事实与推断、未检查项和残余风险。

修改任务只有满足以下条件才能声明相应完成层级：

1. 用户要求的可观察结果已实现，且未扩大范围。
2. 行为变化有测试、audit、浏览器验证或其他可复现证据。
3. 适用测试、构建和门禁已通过；未运行项及原因已明确。
4. 素材、训练、demo、generated output 与许可边界保持清晰。
5. 应更新的 contract、architecture、state、risk、knowledge 或 asset 事实源已同步。
6. 最终 diff 无无关改动、冲突标记、调试残留、秘密或意外产物。
7. 外部副作用只发生在明确授权范围内；commit、工作区余留和残余风险已准确报告。

完成层级：

1. **实现完成**：行为或文档已实现，相关验证和必要事实源同步已完成；可以尚未提交，但不能称为持久进度或提交完成。
2. **提交完成**：Owner 已明确授权 commit，且存在真实、范围正确的 commit hash。
3. **集成完成**：Owner 授权的 push、merge、部署或发布已经执行并验证。

只能声明实际达到的层级；未提交不阻塞本地“实现完成”，也不能被表述为“提交完成”或“集成完成”。

## 13. 输出与维护

- 默认使用中文沟通；技术术语首次出现时可附英文名称。代码、命令、路径和标识符保持原样。
- 最终回复先给结果，再列关键变更、实际验证、未验证项、知识/状态回写和残余风险。
- 不声称未运行的命令已通过，不隐藏 warning、fallback、skip 或不确定性。
- 根 `AGENTS.md` 只保留跨任务稳定、可执行、可验证的规则；当前 Sprint、测试计数、scene 数、冻结期和实时风险留在 `docs/state/`。
- 项目地图、事实源和命令必须保持真实；发现漂移时更新权威来源，不复制一个临时修补版本。
- 新规则只来自重复错误、重复评审意见或稳定项目约束，并写成“触发条件 + 动作 + 验证结果”。
- 复杂流程放专题文档或 Skill；机械约束交给 formatter、linter、类型检查、hooks、contract tests 和 CI。
- 删除过时、重复或不可验证规则，使根文件保持为项目导航与不变量，并为局部指令留出 Codex 默认 32 KiB 合并预算。
