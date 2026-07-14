# AGENTS.md — ObjGauss 新项目协作规则

本文件定义当前仓库中 AI coding agent 与贡献者的长期工作约定。它只保存稳定的项目事实、
工程边界、工作协议和验收规则，不承担实时任务看板职责。

## 1. 项目罗盘

### 当前事实

- 当前阶段：Demo A 的 Stage-0 可见切片；核心 `PR-00` contract 尚未建立或通过。
- 当前主要交付物：PRD、实施计划、资源台账、可执行协作规则，以及可复现的本地数据预览。
- 拟议研究方向：对象中心、动作条件的 Gaussian 世界模型；方向、完成层级和阈值仍是
  working_assumption，只有 Owner 明确确认后才是 decision。
- Owner 已确认实现路径按“一个 PR 只引入一个可证伪假设”拆分，并由 PR-04 实验裁决
  Gaussian 是否进入 dynamics 核心；准确队列以 docs/PROJECT_PLAN.md 为准。
- 当前分支已有无生产依赖的静态 `viewer/`、确定性 synthetic Gaussian world、Node 内置测试
  和固定审计样例下载脚本；仍没有模型代码、依赖锁文件、原始训练数据、checkpoint 或已批准
  的机器 schema。
- Owner 已明确要求 Stage-0 不使用 notebook，而是直接看到渲染后的 3D Gaussian；已批准把
  一个固定 `.splat` 下载到 ignored `data/` 并构建 WebGL2 页面。准确来源、大小、哈希与
  声明边界以 REFERENCES.md 为准，该授权不延伸到训练数据或其他 3DGS 资产。
- 旧 ObjGauss 完整恢复点为标签 archive/objgauss-final-2026-07-14；它不是当前项目。

### 当前用户与可观察结果

- 当前直接用户是 Owner 和参与立项的研究/工程贡献者。
- 当前可观察结果是：项目目标、证据边界、资源状态、阶段门和下一批 PR 能被独立理解与评审；
  贡献者还能在本地打开沉浸式 Stage-0 页面，从环境内部环绕、平移、移动和缩放由 8,523 个
  splat 组成的确定性 synthetic Gaussian world，并可切换到通过固定哈希校验的
  103,060-splat 外部审计样例。
- 长期目标用户、产品形态和首个对外成果仍待 Owner 在 PRD 评审中确认。

### 当前不变量

1. 计划、候选、synthetic、fixture、reviewable 或 archive fact 不得描述成新项目已实现能力。
2. 在许可证、技术栈、数据预算和验收命令获批前，不预建框架、不新增生产依赖、不下载大型数据。
3. 不整体恢复旧项目；只读参考或经 Owner 对具体文件批准后做最小移植。
4. 不修改或删除归档标签 archive/objgauss-final-2026-07-14。
5. 数据、模型、指标和声明必须保留来源、版本、许可、lineage 和失败证据。
6. 工作区既有改动默认属于 Owner 或其他工作流，不覆盖、回退、暂存或整理。
7. 数据集只是证据来源，不是 PR 拆分边界；每个实现 PR 只引入一个可证伪假设和一个预注册
   primary endpoint，secondary metrics 不得事后替代裁决。

### 当前明确不做

- 除已批准的 Stage-0 静态预览外，不实现模型、服务、公共 API、持久化或机器人控制。
- 不把旧 M2、BOP、RBO、NeRF Lego 或 Hugging Face 结果当作新项目当前基线。
- 不在 Identity Gate 之前扩展复杂因果头、diffusion、replay buffer 或自生成训练。
- 不按“适配一个数据集”打包模型、评估和 Demo 等多个无法独立裁决的目标。
- 不提交数据、训练输出、缓存、构建产物、checkpoint、凭据或未脱敏日志。

### 关键风险

- 产品目标、技术栈、许可证和资源预算尚未由 Owner 确认。
- Stage-0 固定 `.splat` 的直接上游位置、大小和哈希已核验，但其资产生成 provenance 仍未
  核验；大多数外部数据集字段、版本、规模和许可也尚未用上游一手资料复核。
- 身份、坐标、future leakage 和 sibling split 错误会污染后续动力学与因果结论。
- 旧项目代码与兼容层体量较大，整体迁移会恢复已知耦合和技术债。
- Gaussian 可能只改善渲染，不一定提高规划价值。

## 2. 事实源地图

| 想知道 | 当前权威来源 |
| --- | --- |
| 工作区定位与文档入口 | README.md |
| 拟议目标、角色、需求、概念数据语义、阶段门、风险与开放决策 | docs/PRD.md |
| 当前建议工作流、PR 队列和各切片验收 | docs/PROJECT_PLAN.md |
| 候选数据、外部地址、归档资源、许可状态与历史负证据 | REFERENCES.md |
| 当前实际文件与改动 | 文件系统、git status --short 和可复现命令结果 |
| 稳定协作与授权规则 | 本文件 |
| 公共机器 contract | 尚未建立；docs/PRD.md 第 6 节仅是概念草案，由 PR-00 建立唯一机器源 |
| 架构与 ADR | docs/adr/；ADR-001 只冻结 Stage-0 本地预览栈 |
| 已实现行为 | viewer/、scripts/ 和 tests/ 的实际代码与可复现命令结果 |

代码决定“现在实际发生什么”；Owner 已批准的 PRD、contract、task 和 ADR 决定“规范要求
什么”；README 或状态摘要只是上述事实的投影。来源冲突时先复现和定位，不静默选择方便
实现的一方，并修正错误投影。

一个事实只保留一个权威来源：

- 需求和声明门改 docs/PRD.md，计划只链接。
- PR 队列和切片验收改 docs/PROJECT_PLAN.md，README 只链接。
- 数据、资产、归档和许可状态改 REFERENCES.md，其他文档不复制细节。
- 稳定执行规则改本文件，动态进度不得写入本文件。

## 3. 仓库地图

| 路径 | 唯一责任 |
| --- | --- |
| README.md | 当前阶段摘要和事实源入口 |
| docs/PRD.md | 拟议产品/研究需求与声明门 |
| docs/PROJECT_PLAN.md | 阶段依赖、PR 队列和评审清单 |
| REFERENCES.md | 资源、外部地址、归档、许可和历史证据台账 |
| AGENTS.md | 根级协作、授权、验证和安全规则 |
| docs/adr/ | 已批准的长期技术取舍；当前 ADR-001 只覆盖 Stage-0 预览栈 |
| viewer/ | Stage-0 WebGL2 3D Gaussian 世界页面、synthetic scene generator、严格 `.splat` 解析、深度排序与渲染器 |
| scripts/ | 固定来源、大小和哈希的本地资源获取脚本 |
| tests/ | 当前已批准切片的 Node 内置行为测试 |
| data/ | Git ignored 本地数据；不得提交 |
| .git 中的归档标签 | 旧项目只读恢复点，不是当前源码目录 |

当前没有核心包、contract 目录、adapter 目录或 generated 目录。Stage-0 之外的技术栈确定后
必须先更新本节，再创建真实需要的目录；不要预建空结构或假想扩展点。

## 4. 当前权威命令

所有命令从仓库根目录 /home/ljy/gitpo/ObjGauss 执行。

| 目的 | 命令或状态 |
| --- | --- |
| 查看工作区 | git status --short |
| 定位文件 | rg --files -g '!**/.git/**' |
| 搜索内容 | 按任务使用 rg，并传入真实搜索模式和路径 |
| 检查 tracked diff 空白错误 | git diff --check |
| 只读查看归档文件示例 | git show archive/objgauss-final-2026-07-14:README.md |
| 安装/同步依赖 | 尚未定义；当前禁止推断或执行 |
| 获取 Stage-0 固定预览 | bash scripts/fetch-gaussian-preview.sh |
| 启动 Stage-0 页面 | python3 -m http.server 8000 --bind 127.0.0.1；打开 http://127.0.0.1:8000/viewer/ |
| Stage-0 行为测试 | node --test tests/*.test.mjs |
| Stage-0 语法检查 | node --check viewer/app.mjs；node --check viewer/splat-format.mjs；node --check viewer/splat-renderer.mjs；node --check viewer/splat-sort.mjs；node --check viewer/splat-sort-worker.mjs；bash -n scripts/fetch-gaussian-preview.sh |
| 完整测试或 CI 等价门 | 尚未定义 |
| 格式、lint、类型检查、构建 | Stage-0 无构建步骤；项目级命令尚未定义 |
| contract/codegen/audit | 尚未定义 |
| 文档专用检查 | 当前至少运行 git diff --check，再核对本地链接、冲突标记和范围 diff |

没有真实命令时明确写“尚未定义”，不得借用旧归档命令、临时脚本或未安装工具伪造门禁。
Stage-0 局部决策见 docs/adr/0001-stage-0-preview-stack.md；模型/训练技术栈确定后，后续 ADR
必须把本表更新为与锁文件、任务脚本和 CI 一致的可复制命令。

## 5. 会话启动协议

每个任务开始时：

1. 完整阅读本文件，并检查目标子树是否存在更深的 AGENTS.override.md 或 AGENTS.md。
2. 阅读任务直接引用的事实源；项目方向任务至少读 README.md、docs/PRD.md、
   docs/PROJECT_PLAN.md 和 REFERENCES.md 中相关部分。
3. 运行 git status --short；现有改动属于 Owner 或其他工作流。
4. 使用 rg / rg --files 定位真实入口、调用方、contract、状态所有权和测试。
5. 修改前给出简短 Change Contract：
   - 目标文件；
   - 用户价值或可观察结果；
   - 范围与明确范围外；
   - 验证方式；
   - 预计提交粒度。

简单任务直接执行；复杂或高风险任务拆成可验证步骤。开工说明不是机械的二次审批；用户明确
要求修改、修复、创建或构建时，已授权当前范围内可逆的仓库修改和相关本地验证。

## 6. 修改纪律

- 一个实现 PR 只有一个可观察目标和一个可证伪假设，不混入命名整理、历史清理、全库格式化
  或无关重构；纯文档/治理 PR 必须只有一个明确决策目标。
- 先判断责任属于数据、contract、领域逻辑、算法、tool、Agent 表达、UI、基础设施还是文档。
- 没有真实调用方，不新增抽象、服务、包、配置开关、兼容分支或占位目录。
- 优先修复根因；不用 wrapper、fallback、mock 或空实现制造“已接通”的假象。
- 未经明确授权，不改变公共 API、CLI、schema、默认值、序列化、持久化或错误语义。
- 新增生产依赖、外部服务、存储、基础设施或公共 contract 前取得 Owner 动作级确认。
- 不手工编辑生成物；修改权威源并运行正式生成器。
- 超出范围的问题写入已有 PRD、计划或资源台账的正确位置，不顺手扩张当前切片。
- 行为变化必须有行为级测试；无法自动化时提供固定输入、复现步骤和残余风险。
- 不删除测试、skip、放宽断言、吞异常或擅自更新阈值/基线来制造通过。

当前只扩大到 Owner 明确批准的 Stage-0 页面和固定小型预览。除非 Owner 再次明确扩大范围，
不得借该授权提前实现机器 contract、adapter、模型或后续 PR。

## 7. 数据、Contract 与研究证据

### 数据与资产

- REFERENCES.md 是资源状态唯一台账；候选入口不等于已核验或获准使用。
- 使用外部数据前，从官方一手来源核验版本、字段、许可、大小、用途、下载范围和校验和；
  Stage-0 `.splat` 预览的已核验范围和未核验 provenance 以 REFERENCES.md 为准。
- 下载数据、模型或 checkpoint 前取得 Owner 对来源、范围、磁盘和许可的确认。
- 原始数据、转换数据、训练输出、缓存和 checkpoint 放在 ignored 或批准的外部存储，禁止进 Git。
- 派生产物必须保留 source checksum、producer/version、transform/config hash、license review 和 lineage。

### Contract

- docs/PRD.md 第 6 节只定义概念语义，不能当作已冻结机器 schema。
- 实现公共 contract 时必须明确 producer、consumer、版本、必填/可选、默认值、单位、时区、
  唯一标识、缺失语义、兼容策略、迁移与回滚。
- 同一概念只保留一个权威 schema；第二份副本视为漂移风险。
- 物理对象身份、模型 slot 和 renderer address 不得静默共用一个真值字段。
- prediction-before-observation 与 correction-after-observation 必须分账；GT 或未来输入不得
  泄漏到推理 feature。
- commanded action、executed action 和缺失 action 必须分开表达。
- sibling group 必须整体隔离 train/validation/test；无 GT 或 measured action 时结果是
  blocked，不得伪装成零值或 pass。

以上 contract 约束在 PR-00 的机器 contract 获批前是 PRD 驱动的实现门，不代表字段名或
序列化已批准。

### 证据与声明

- fixture、synthetic、public replay、controlled real 和 closed-loop evidence 分开统计。
- pass、fail、blocked 和 not-run 分开；reviewable/schema-valid 不等于 metric pass。
- 假设裁决统一使用 supported、rejected、blocked、invalid；代码合并不自动等于 supported。
- 每个模型与批准的简单 baseline 同跑；指标从 raw prediction 与 GT 独立重算。
- split 以完整 scene、sequence、layout 或 sibling group 为单位；禁止 row/frame 泄漏。
- 失败 scene、负结果、warning、fallback 和 skip 必须可见。
- 阈值在 pilot 前或按预注册流程冻结，不能看完最终结果后放宽。

## 8. 归档边界

- 默认通过 git show 只读查看归档；不为普通核对恢复整个 worktree。
- REFERENCES.md 中的 reference 只复用思想，candidate-port 仍需许可、技术栈和具体文件批准，
  do-not-migrate 不得整体恢复。
- 旧根 LICENSE 是 all rights reserved；新项目许可证未确定前不复制旧实现。
- 旧代码、测试、viewer、CLI、数据合同、阈值和研究结论都不自动成为新项目事实。
- 不改写、删除或移动归档标签，不执行破坏性历史操作。

## 9. Knowledge Capture 与 grill-me

### 触发

持续识别 Owner 提供的用户角色、目标、术语、规则、不变量、阈值、例外、错误代价、数据语义、
审批边界、决策理由、假设、开放问题和风险。

用户明确提出 grill me 时直接进入主动质询。否则，仅当任务依赖未编码的产品语义，且多个
合理解释会改变 contract、架构、安全边界或验收时，先说明知识缺口并询问是否进入。

### 方法

1. 先查事实源，建立“已知 / 冲突 / 未知 / 受影响决策”的 Knowledge Gap Map。
2. 每轮只问一个主问题，可给 2–3 个互斥选项和 Agent 推荐答案。
3. 说明依据、取舍以及答案不同会改变什么。
4. 优先确认用户、业务结果、事实源、不变量、例外、错误代价、权限和验收。
5. 不阻塞的细节可标记推荐默认值；会改变语义、安全或验收的未知项必须继续确认或标 blocked。

候选知识使用以下状态：

- confirmed_fact：Owner 明确表述，或仓库/运行证据已验证。
- decision：Owner 已作出取舍，含理由和适用范围。
- working_assumption：暂时采用但未确认。
- open_question：仍会改变语义、contract 或验收。
- superseded：已由新事实替代，必须链接替代项。

### 写入路由

| 知识类型 | 当前写入位置 |
| --- | --- |
| 已确认目标、角色、术语、规则与例外 | docs/PRD.md |
| 公共字段与机器 contract | PR-00 建立的唯一 contract 源；当前尚未创建 |
| 当前任务、非目标、验收和队列 | docs/PROJECT_PLAN.md |
| 长期架构取舍 | docs/adr/；ADR-001 只覆盖 Stage-0，其余架构使用后续 ADR |
| 候选数据、资产、许可和归档事实 | REFERENCES.md |
| 稳定协作与授权边界 | AGENTS.md |
| 开放问题 | 当前写入 docs/PRD.md 第 13 节或计划评审清单 |

写入前搜索已有事实源。冲突时列出旧事实、新陈述、影响和建议裁决；Owner 确认后更新权威项，
并把旧项标为 superseded 或删除错误投影。纯讨论没有文档写入授权时，只输出候选知识账本。
不写入真实客户数据、PII、token、凭据或生产 payload。

## 10. 风险驱动验证

| 风险 | 典型改动 | 当前最低证据 |
| --- | --- | --- |
| 低 | 文档、注释、无行为变化配置 | 本地链接/结构核对、git diff --check、范围 diff |
| 中 | 模块行为、CLI、UI、内部数据流 | 回归测试、相关集成测试、适用 lint/typecheck/build |
| 高 | 公共 contract、持久化、安全、并发、迁移、发布或控制 | 完整相关门、兼容/迁移/回滚和端到端证据 |

当前只有 Stage-0 的 Node 行为/语法检查和 shell 语法检查，没有项目级 lint、typecheck、build
或 CI 命令。不得把局部检查描述成完整项目门通过。技术栈落地后，每个实现 PR 必须先补真实
命令与相应测试，再按影响面扩大验证。

完成前始终检查：

- git diff --check；
- 任务相关 diff；
- git status --short；
- 冲突标记、调试残留、秘密和意外生成物；
- 未运行项、原因、替代证据与残余风险。

## 11. Git、外部操作与安全

- 未经 Owner 动作级明确授权，不执行 commit、amend、rebase、push、merge、tag、发布、部署、
  生产写入或外部消息。
- 用户要求提交时，一个提交只表达一个意图，精确暂存相关文件，不使用 git add .。
- 禁止 git reset --hard、git clean -fd、覆盖式恢复和强制推送，除非 Owner 明确要求且风险已确认。
- 不覆盖、回退、暂存、格式化或整理无关现有改动。
- 网络核验优先使用官方一手来源；核验不等于获准下载或产生外部写入。
- 不读取、输出、写入或提交密钥、凭据、客户数据和未脱敏日志。
- 测试不得触达真实用户、生产数据、设备或未授权系统。

## 12. 多 Agent 路由

- 主 Agent 保留需求解释、Owner 意图、产品/架构决策、关键写入、最终集成、验证和最终答复责任。
- 仅在子任务独立、边界清晰且并行有收益时委派，优先只读调研、日志分析、测试和专项审查。
- 并行写任务必须划分文件和事实源所有权，不让多个 Agent 修改同一 contract 或重叠文件。
- 子 Agent 不得扩大授权；输出必须包含证据、假设、未检查项和建议，由主 Agent 复核。
- 简单任务不创建子 Agent，不递归扇出；汇总前等待必要结果。
- 模型路由不能降低测试、证据、权限、安全和验收标准；不得虚构已切换模型。

## 13. 完成定义与沟通

修改任务只有满足以下条件才能声明“实现完成”：

1. 请求的可观察结果已实现，且没有扩大范围。
2. 适用验证已完成；未运行项和原因已明确。
3. 事实源、contract、风险和知识文档已按实际变化同步。
4. 最终 diff 无无关改动、冲突标记、秘密或意外产物。
5. 外部副作用只发生在明确授权范围内。
6. commit、工作区余留和残余风险被准确报告。

“提交完成”还要求 Owner 已授权 commit 且存在真实、范围正确的 commit hash；“集成完成”还
要求 Owner 授权的 push、merge、部署或发布已执行并验证。只能声明真实达到的层级。

默认使用中文沟通；代码、命令、路径和标识符保持原样。最终回复先给结果，再说明关键变更、
实际验证、知识/状态回写、未验证项、工作区余留和残余风险。

## 14. 维护本文件

- 本文件只保留跨任务稳定、可执行、可验证的规则，不写 Sprint、实时进度或聊天记忆。
- 项目地图、事实源和命令必须真实；技术栈、目录或门禁变化时同步更新。
- 新规则应来自重复错误、重复评审意见或稳定架构约束，并写成触发条件、动作和验证结果。
- 复杂流程放专题文档或 Skill；动态状态放对应事实源；机械约束交给未来的工具和 CI。
- 删除过时、重复或不可验证的规则，为局部指令保留足够上下文预算。
