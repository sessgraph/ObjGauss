# AGENTS.md — ObjGauss 新项目协作规则

本文件定义当前仓库中 AI coding agent 与贡献者的长期工作约定。它只保存稳定的项目事实、
工程边界、工作协议和验收规则，不承担实时任务看板职责。

## 1. 项目罗盘

### 当前事实

- 当前阶段：Demo A 的 PR-01 严格成对干预里程碑已经关闭；PR-02A `0.3.0` Contract 与 PR-02B
  pilot/data freeze 均已在本地实现、通过各自项目门并提交；PR-02C Trainer/Baselines 已获动作
  授权，ADR-006 规划决策已冻结，C0 独立 runtime/contract gate 已由提交 `fc20023` 实现并通过
  clean GPU 验收；C1 train/validation source 与 fail-closed loader 已实现、等待 clean HEAD
  验收，模型与 trainer 尚未实现。准确动态状态以 `docs/state/` 为准。
- 当前已完成的证据底座：PR-00 `0.1.0` contract、PR-01 `0.2.0` sibling evidence contract
  family、隔离 simulator runtime、原子 writer、独立 audit、正式 cohort、无 RGB 五联 Demo、
  机器报告和远端验收，以及本地 supported 的 PR-02A `0.3.0` contract family 与 PR-02B
  calibration/power freeze。
- 拟议研究方向：对象中心、动作条件的 Gaussian 世界模型；方向、完成层级和阈值仍是
  working_assumption，只有 Owner 明确确认后才是 decision。
- Owner 已确认实现路径按“一个 PR 只引入一个可证伪假设”拆分，并由 PR-04 实验裁决
  Gaussian 是否进入 dynamics 核心；准确队列以 docs/PROJECT_PLAN.md 为准。
- 当前工作区已有 Stage-0 `viewer/`、`0.1.0`/`0.2.0`/`0.3.0` JSON Schema、Node/npm 锁文件、
  synthetic producer、独立 evaluator、Web consumer、行为测试和 GitHub Actions workflow；仍
  没有模型代码、原始训练数据或 checkpoint。
- Owner 已明确要求 Stage-0 不使用 notebook，而是直接看到渲染后的 3D Gaussian；已批准把
  一个固定 `.splat` 下载到 ignored `data/` 并构建 WebGL2 页面。准确来源、大小、哈希与
  声明边界以 REFERENCES.md 为准，该授权不延伸到训练数据或其他 3DGS 资产。
- Owner 已决定新项目当前保持私有并按 all rights reserved 管理；对外发布前必须重新做
  许可证决策。该决定不授权复制旧归档中的任何实现。
- Owner 已决定以 JSON Schema Draft 2020-12 作为机器 contract 源。`0.1.0` episode 只属于
  PR-00 synthetic audit；PR-01 使用 `0.2.0` episode、experiment、attempt 和 invariance report
  四份唯一 schema；PR-02 使用 `0.3.0` dynamics experiment、training trial/attempt、checkpoint
  manifest、raw prediction 和 independent evaluation report 六种记录。Consumer 必须按精确
  `schema_version + contract_kind` 分派，不存在隐式迁移。
- Owner 已批准 `PR-00` 使用 JavaScript ESM、Node 24 LTS、npm、Ajv 8、esbuild 和 `node:test`；
  不引入服务端框架。精确依赖版本由实现时的 lockfile 固定；该决定不冻结未来 Python/训练
  技术栈。
- Owner 已批准 PR-01 生产仿真边界：固定 pilot 已验证的 CPython/ManiSkill/SAPIEN/Torch 版本，
  作为独立 `sim` optional extra，只用于离线 primitive episode 生成；允许 adapter、原子 writer、
  CLI、行为测试、真实 smoke 和 CI。禁止 runtime 网络、外部 asset、RGB/GPU renderer、训练、
  模型、Gaussian dynamics 和机器人控制；独立 auditor 不得依赖 ManiSkill 或 writer 逻辑。
- Owner 已决定 PR-01 Contract 使用 `0.2.0 + experiment manifest`：单 branch episode、完整
  experiment、失败/retry attempt 与独立 invariance report 分账；`0.1.0` 字节冻结，不存在自动
  `0.1.0 -> 0.2.0` 迁移，Schema 不固定具体 fixture ID。
- Owner 已决定 PR-01 Demo 使用无 RGB 五联状态回放：只消费已审计 episode，以 Canvas/SVG
  显示对象、动作、轨迹、contact 与 settling；浏览器不运行 simulator，不使用 CDN、外部资产
  或 GPU。Machine report 是事实源，Demo 只解释证据。
- PR-01 最终 lineage 不在冻结 spec 中硬编码提交 SHA；spec 只记录
  `runtime-current-clean-git-head` 策略，生成进程注入当前 HEAD。`accept-pr01`、Delivery builder
  与 verifier 必须拒绝 staged、tracked 或非 ignored untracked 改动；只有 clean checkout 才能
  生成绑定最终 commit 的验收证据。
- Owner 已批准以 GitHub Actions 作为 `PR-00` CI，并以 `npm run check` 作为本地与 CI 的唯一
  验收入口；PR 和 `main` 均使用 Node `24.18.0`、`npm ci`、contract audit、测试、Web build
  和语法检查。真实脚本、lockfile 与 workflow 已建立；该授权不包含 push、部署或发布。
- Owner 已冻结 `PR-00` Robotics/OpenCV 坐标约定：`T_AB · p_B = p_A`，列向量左乘；World
  为右手系、`+Z` 向上、meter；Camera 为 `+X` 右、`+Y` 下、`+Z` 前；`T_WC` 表示 Camera
  到 World，投影使用 `T_CW = inverse(T_WC)`；WebGL bridge 只属于 Viewer consumer。
- Owner 已冻结 `PR-00` 姿态与时间语义：quaternion 为 `[w, x, y, z]`、有限且归一化，序列化
  使用确定性符号，插值前按相邻点积处理双覆盖；`episode_time_s` 从首个 Observation 的 `0.0`
  开始，Observation 严格递增，同步字段共时，事件可同刻但不得倒退。
- Owner 已冻结 `PR-00` 缺失语义：所有可能缺失的 contract 值使用 `availability` tagged union；
  `present` 必须且只能携带有效 `value`，`missing` 必须携带 `not_measured`、`not_provided`、
  `not_applicable`、`redacted` 或 `invalidated` 之一且禁止 `value`。`null`、零/单位哨兵、空串、
  NaN 和默认置信度不得冒充缺失；`hold` 是实际动作，不是 missing action。
- Owner 已冻结 `PR-00` 唯一 primary endpoint：固定 `synthetic-audit-v0` 全部 primary points 的
  2D Euclidean 像素误差最大值 `max_camera_reprojection_error_px < 1.0`。Evaluator 必须独立；
  任一点未达门为 `rejected`，零有效点或 evaluator 复用被测投影逻辑为 `invalid`；相机后方、
  越界和奇异内参属于必须拒绝的负例，不进入 primary 统计。
- Owner 已冻结 `PR-00` fixture/资源边界：固定 seed/config 的仓库内 JavaScript producer 生成
  `synthetic-audit-v0`；Git 只收 producer、fixture spec 和预期 checksum manifest，派生资源
  ignored。RGB/depth 等数组只用含 `uri`、`media_type`、`dtype`、`shape`、`sha256` 的 descriptor
  引用；记录 producer version、config hash 和 lineage；PR-00 不联网、不下载数据、不读旧归档。
- Owner 已冻结 schema 版本/兼容策略：从 `0.1.0` 开始使用严格 SemVer；每份 schema 的版本化
  `$id` 与记录的精确 `schema_version` 匹配，所有 object schema 拒绝未知字段。已发布版本不可
  原地修改；`0.x` 中改变合法实例集合必须升级 MINOR，PATCH 不得改变 contract；跨版本只允许
  显式、可测试的迁移并记录迁移前后 checksum/lineage，禁止静默自动升级。
- Owner 已冻结 canonical object frame/symmetry：object frame 由 producer 在对象创建时定义并在
  episode 内保持不变；synthetic 原点为刚体质心，轴为 producer 声明的右手语义轴，禁止从
  观测、PCA 或 mesh 外观推断。`T_WO · p_O = p_W`。Symmetry 显式为 `none`、单位 `wxyz`
  有限旋转集合或绕归一化 object-frame axis 的连续旋转；未知时为 `missing:not_provided`，
  姿态指标 `blocked`，不得默认 `none`。
- Owner 已冻结 `PR-00` claim/archive 边界：全部实现从零编写，不复制旧归档的代码、schema、
  测试、Viewer 或阈值；旧归档仅可用 `git show` 做只读思想参考，未来移植须逐文件重新授权并
  记录许可审查。PR-00 最多声明 synthetic contract、坐标链和独立重投影门得到支持，不得声明
  真实数据、Gaussian 重建、世界模型、动力学或规划价值。
- 旧 ObjGauss 完整恢复点为标签 archive/objgauss-final-2026-07-14；它不是当前项目。

### 当前用户与可观察结果

- 当前直接用户是 Owner 和参与立项的研究/工程贡献者。
- 当前可观察结果是：项目目标、证据边界、资源状态、阶段门和下一批 PR 能被独立理解与评审；
  贡献者还能在本地打开沉浸式 Stage-0 页面，从环境内部环绕、平移、移动和缩放由 8,523 个
  splat 组成的确定性 synthetic Gaussian world，并可切换到通过固定哈希校验的
  103,060-splat 外部审计样例；也能打开 `PR-00` Web 证据页，同步查看已验证 contract 的
  RGB、对象、相机、坐标轴、轨迹和机器裁决；还能从 PR-01 已审计 formal episode 打开五联
  无 RGB 状态回放。
- 长期目标用户、产品形态和首个对外成果仍待 Owner 在 PRD 评审中确认。

### 当前不变量

1. 计划、候选、synthetic、fixture、reviewable 或 archive fact 不得描述成新项目已实现能力。
2. 在技术栈、数据预算和验收命令获批前，不预建框架、不新增生产依赖、不下载大型数据；
   当前项目不得对外发布或以开放许可证分发。
3. 不整体恢复旧项目；只读参考或经 Owner 对具体文件批准后做最小移植。
4. 不修改或删除归档标签 archive/objgauss-final-2026-07-14。
5. 数据、模型、指标和声明必须保留来源、版本、许可、lineage 和失败证据。
6. 工作区既有改动默认属于 Owner 或其他工作流，不覆盖、回退、暂存或整理。
7. 数据集只是证据来源，不是 PR 拆分边界；每个实现 PR 只引入一个可证伪假设和一个预注册
   primary endpoint，secondary metrics 不得事后替代裁决。

### 当前明确不做

- 除已批准的 Stage-0、PR-00、PR-01A–F 严格 sibling evidence 里程碑与 ADR-006 界定的 PR-02C
  Trainer/Baselines 外，不实现模型、服务、公共 API、长期持久化或机器人控制。
- 不把旧 M2、BOP、RBO、NeRF Lego 或 Hugging Face 结果当作新项目当前基线。
- 不在 Identity Gate 之前扩展复杂因果头、diffusion、replay buffer 或自生成训练。
- 不按“适配一个数据集”打包模型、评估和 Demo 等多个无法独立裁决的目标。
- 不提交数据、训练输出、缓存、构建产物、checkpoint、凭据或未脱敏日志。

### 关键风险

- 长期产品目标、PR-03 之后的技术栈/资源预算和未来对外发布许可证尚未由 Owner 确认；
  PR-02 的硬上限、阈值、group/seed 数与训练配置已由隔离 pilot 冻结，但尚无训练证据。
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
| 公共机器 contract | `contracts/objgauss/0.1.0/`、`0.2.0/` 与 `0.3.0/`；必须精确版本分派 |
| 架构与 ADR | docs/adr/；ADR-001 覆盖 Stage-0，ADR-002 覆盖 PR-00，ADR-003 覆盖 PR-01，ADR-004 覆盖 PR-02A contract，ADR-005 覆盖 PR-02B pilot/data freeze，ADR-006 覆盖 PR-02C trainer/baselines |
| 已实现行为 | src/pr00/、src/pr01/、sim/、learning/、viewer/、scripts/、tests/ 与可复现命令结果 |

代码决定“现在实际发生什么”；Owner 已批准的 PRD、contract、task 和 ADR 决定“规范要求
什么”；README 或状态摘要只是上述事实的投影。来源冲突时先复现和定位，不静默选择方便
实现的一方，并修正错误投影。

一个事实只保留一个权威来源：

- 需求和声明门改 docs/PRD.md，计划只链接。
- 稳定 PR 依赖和切片验收改 docs/PROJECT_PLAN.md；动态队列、状态、风险和 inbox 改
  docs/state/；README 只链接。
- 数据、资产、归档和许可状态改 REFERENCES.md，其他文档不复制细节。
- 稳定执行规则改本文件，动态进度不得写入本文件。

## 3. 仓库地图

| 路径 | 唯一责任 |
| --- | --- |
| README.md | 当前阶段摘要和事实源入口 |
| docs/PRD.md | 拟议产品/研究需求与声明门 |
| docs/PROJECT_PLAN.md | 阶段依赖、PR 队列和评审清单 |
| docs/state/ | 动态 PR 执行状态、项目状态、风险与开放事项 |
| REFERENCES.md | 资源、外部地址、归档、许可和历史证据台账 |
| AGENTS.md | 根级协作、授权、验证和安全规则 |
| docs/adr/ | 已批准的长期技术取舍；ADR-001–006 分别覆盖 Stage-0、PR-00、PR-01、PR-02A、PR-02B 与 PR-02C |
| contracts/objgauss/0.1.0/ | PR-00 唯一 JSON Schema contract；已发布版本不得原地修改 |
| contracts/objgauss/0.2.0/ | PR-01 episode/experiment/attempt/invariance-report 四份 JSON Schema |
| contracts/objgauss/0.3.0/ | PR-02 dynamics experiment、training trial/attempt、checkpoint、prediction、evaluation report 与 shared definitions |
| contracts/fixtures/ | 固定 fixture spec、正负例、producer identity 与预期 checksum manifest |
| src/pr00/ | frame math、validator、producer、evaluator、verdict 与 browser consumer 源码 |
| src/pr01/ | PR-01 contract dispatch 与独立 auditor；auditor 不得导入 simulator 或 writer |
| sim/ | PR-01 隔离 Python package、精确 uv lock、程序化 primitive runtime、adapter、原子 writer、PR-02B calibration/power freeze producer 与 PR-02C train/validation source producer；只供离线证据生成，不含 loader/trainer |
| learning/ | PR-02C 独立纯 PyTorch package、精确 uv lock、C0 runtime/isolation/GPU probe 与 C1 checksum/lineage loader；当前不含模型或 trainer，禁止导入 simulator |
| viewer/ | Stage-0 WebGL2 Gaussian 世界、PR-00 同步 contract 证据页与 PR-01 无 RGB 五联回放源码 |
| scripts/ | PR-00/PR-01/PR-02A contract audit、build/syntax、Stage-0 预览获取、RES-001/PR-01 source pilots、PR-01 clean-install gates、PR-02B clean pilot/freeze verifier 与 PR-02C C0/C1 clean gates |
| tests/、sim/tests/、learning/tests/ | Stage-0、PR-00、PR-01、PR-02A contract、PR-02B pilot/freeze 与 PR-02C runtime/data-boundary 的行为和负例测试 |
| generated/pr00/ | Git ignored 的确定性 episode、数组、报告与 browser bundle；不得手改或提交 |
| generated/pr01b/ | Git ignored 的 canonical/reverse runtime smoke 报告；可重建、不得提交 |
| generated/pr01c/ | Git ignored 的 golden group 与 writer 报告；可重建、不得提交 |
| generated/pr01d/ | Git ignored 的独立 audit/mutation 输出；可重建、不得提交 |
| generated/pr01e/ | Git ignored 的 preflight/formal cohort 与 audit 输出；可重建、不得提交 |
| generated/pr02a/ | Git ignored 的 PR-02A contract machine report；由正式 audit 重建，不得提交 |
| generated/pr02b/ | Git ignored 的 PR-02B repeat/audit/freeze evidence；只有 clean gate 生成的 `evidence/` 可用于验收，仍不得提交 |
| generated/pr02c/ | Git ignored 的 PR-02C runtime、source/loader、训练与后续证据根；C0/C1 clean gates 分别原子发布 `runtime/` 与 `data/`，不得提交 |
| artifacts/pr01/ | Git ignored 的最终 Delivery 投影；只由 clean-checkout 验收生成，不得提交 |
| package.json、package-lock.json、.node-version | PR-00 命令、精确依赖解析与 Node runtime 事实源 |
| .github/workflows/ | PR-00 Node 门与 PR-01 runtime/writer/audit/cohort/delivery 门；不部署或发布 |
| data/ | Git ignored 本地数据；不得提交 |
| .git 中的归档标签 | 旧项目只读恢复点，不是当前源码目录 |

`learning/` 当前实现 PR-02C C0 的精确 runtime/隔离/资源门与 C1 的 checksum/lineage loader；
loader 只暴露初态、commanded action 与 rollout times，future GT 单独作为 train/validation labels，
不暴露 executed action。仍没有模型、trainer 或 checkpoint；后续只按 ADR-006 的串行门增加
真实调用方，不预建空结构或假想扩展点。

## 4. 当前权威命令

所有命令从仓库根目录 /home/ljy/gitpo/ObjGauss 执行。

| 目的 | 命令或状态 |
| --- | --- |
| 查看工作区 | git status --short |
| 定位文件 | rg --files -g '!**/.git/**' |
| 搜索内容 | 按任务使用 rg，并传入真实搜索模式和路径 |
| 检查 tracked diff 空白错误 | git diff --check |
| 只读查看归档文件示例 | git show archive/objgauss-final-2026-07-14:README.md |
| 运行时 | Node 24.18.0；版本源为 .node-version |
| 安装/同步依赖 | npm ci |
| 获取 Stage-0 固定预览 | bash scripts/fetch-gaussian-preview.sh |
| 验证 Stage-0 固定外部预览 | `npm run test:preview`；显式要求已下载的 ignored 文件，不属于无外部资产的 clean-checkout `npm run check` |
| 启动 Web 页面 | python3 -m http.server 8000 --bind 127.0.0.1；PR-00 打开 http://127.0.0.1:8000/viewer/?mode=contract，Stage-0 打开 /viewer/，PR-01 验收后打开 /artifacts/pr01/demo/ |
| 行为与负例测试 | npm test |
| 语法检查 | npm run syntax |
| 构建确定性 fixture 与 Web consumer | npm run build |
| contract/codegen/audit | npm run contract:audit |
| 完整测试或 CI 等价门 | npm run check |
| RES-001 snapshot pilot | `env -u MS_SKIP_ASSET_DOWNLOAD_PROMPT MS_ASSET_DIR="$PWD/data/res001/no-assets" XDG_CACHE_HOME="$PWD/data/res001/xdg-cache" MPLCONFIGDIR="$PWD/data/res001/mpl-cache" data/res001/venv-py310/bin/python scripts/res001_snapshot_pilot.py`；只使用 ignored 固定 runtime |
| PR-01 primitive sibling action baseline | `env -u MS_SKIP_ASSET_DOWNLOAD_PROMPT MS_ASSET_DIR="$PWD/data/res001/no-assets" XDG_CACHE_HOME="$PWD/data/res001/xdg-cache" MPLCONFIGDIR="$PWD/data/res001/mpl-cache" data/res001/venv-py310/bin/python scripts/pr01_sibling_action_pilot.py --order canonical --output data/res001/evidence/sibling-action-pilot-run1.json`；预期先停在 `pending_repeat` |
| PR-01 primitive sibling action repeat | 上一命令后运行 `env -u MS_SKIP_ASSET_DOWNLOAD_PROMPT MS_ASSET_DIR="$PWD/data/res001/no-assets" XDG_CACHE_HOME="$PWD/data/res001/xdg-cache" MPLCONFIGDIR="$PWD/data/res001/mpl-cache" data/res001/venv-py310/bin/python scripts/pr01_sibling_action_pilot.py --order reverse --compare data/res001/evidence/sibling-action-pilot-run1.json --output data/res001/evidence/sibling-action-pilot-run2.json`；只有稳定 evidence 相同且执行顺序相反才 `supported` |
| PR-01A contract audit | `npm run contract:pr01a`；写入 ignored `generated/pr01a/contract-report.json` |
| PR-01A contract tests | `node --test tests/pr01a-contracts.test.mjs`；同时由 `npm test` / `npm run check` 覆盖 |
| PR-01B clean-install runtime gate | `./scripts/check-pr01b-runtime`；uv `0.11.17` 从 `sim/uv.lock` 创建全新临时 venv，外部依赖只装 wheel，随后在 offline/空资产门下运行测试和 canonical/reverse 五分支 smoke |
| PR-01C clean-install writer gate | `./scripts/check-pr01c-writer`；在 PR-01B 相同隔离门下运行 writer 负例与 canonical/reverse 真实 golden group，并独立复核 raw artifacts |
| PR-01D independent audit gate | `./scripts/check-pr01d-audit`；重新生成真实 golden group，由不依赖 simulator/writer 的 evaluator 重算 hard gates 并运行完整 mutation matrix |
| PR-01E preflight gate | `./scripts/check-pr01e-preflight`；先复跑 PR-01D，再生成与正式 cohort 隔离的 12 groups / 60 episodes，独立审计并产出预算 freeze candidate |
| PR-01E formal cohort gate | `./scripts/check-pr01e-cohort`；先复跑 preflight，再按冻结 split/预算生成 48 groups / 240 episodes 并独立审计 |
| PR-01 完整本地验收 | `./scripts/accept-pr01`；要求 clean checkout，从锁定依赖、全库 Node 门、独立 runtime smoke、真实 writer、preflight/formal cohort、独立 audit 到无 RGB Demo/checksums，只有最终 delivery supported 才退出 0 |
| PR-02A contract audit | `npm run contract:pr02a`；冻结旧 contract 哈希，验证 `0.3.0` schemas、fixtures、精确分派和负例，写入 ignored `generated/pr02a/contract-report.json` |
| PR-02A contract tests | `node --test tests/pr02a-contracts.test.mjs`；同时由 `npm test` / `npm run check` 覆盖 |
| PR-02B pilot unit tests | `npm run test:pr02b` 与 `PYTHONPATH=sim/src python3 -m unittest sim.tests.test_pr02_pilot`；不需要 simulator asset 或 GPU |
| PR-02B clean pilot/data freeze | `./scripts/check-pr02b-pilot`；要求 clean checkout、Node 24.18.0、uv 0.11.17、冻结 runtime、离线/空资产，生成 canonical/reverse source、独立 audits、GPU 1 GiB 显示保留 probe、`0.3.0` experiment 与 checksum/lineage verification |
| PR-02C C0 unit tests | `npm run test:pr02c` 与 `uv run --project learning --frozen --no-dev python -m unittest discover -s learning/tests -p 'test_runtime.py'`；分别覆盖 manifest/lock 静态门与 Python runtime 失败语义 |
| PR-02C C0 clean runtime gate | `./scripts/check-pr02c-runtime`；要求 clean checkout、Node 24.18.0、uv 0.11.17、CPython 3.10.20、精确 `learning/uv.lock`、离线且无 simulator 的新 venv，并验证 CUDA 13.0、12 GiB cap、至少 1 GiB 显示显存保留、HEAD/lock/grid lineage 与独立 verifier |
| PR-02C C1 unit tests | `npm run test:pr02c-data`、`OBJGAUSS_REPO_ROOT="$PWD" PYTHONPATH=sim/src python3 -m unittest sim.tests.test_pr02_data` 与 `uv run --project learning --frozen --no-dev python -m unittest discover -s learning/tests -p 'test_data.py'`；不物化正式 source |
| PR-02C C1 clean data gate | `./scripts/check-pr02c-data`；要求 clean checkout，先重建 PR-02B freeze 与 C0 runtime，再仅物化/审计 48 train + 12 validation groups，运行无 simulator loader 和独立 300-branch verifier；发现 test artifact、坏 checksum/lineage、executed/future model feature 或跨 split identity 即 `invalid` |
| 文档专用检查 | 当前至少运行 git diff --check，再核对本地链接、冲突标记和范围 diff |

没有真实命令时明确写“尚未定义”，不得借用旧归档命令、临时脚本或未安装工具伪造门禁。
Stage-0 局部决策见 docs/adr/0001-stage-0-preview-stack.md，PR-00 决策见
docs/adr/0002-pr00-contract-stack.md；PR-02C C0/C1 的真实命令已与 `learning/uv.lock` 和脚本同步，
baselines/trainer 与 PR-02D–F 命令仍尚未定义。PR-03 之后的模型/表示栈仍须另行决定。

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

Owner 对 Stage-0、PR-00、PR-01A–F、PR-02A Contract、PR-02B Pilot/Data Freeze 和 PR-02C
Trainer/Baselines 的实现授权已经给出。PR-02A 不得外推为 pilot、数据或模型证据；PR-02B
不得外推为 trainer 或模型性能证据；PR-02C 授权只覆盖 accepted ADR-006 冻结的独立
`learning/` package、精确纯 PyTorch 依赖、四个预注册 arms 与本地训练证据。不得借此开始
PR-02D、PR-02E、Gaussian dynamics、外部数据、RGB/GPU renderer 或机器人控制。

## 7. 数据、Contract 与研究证据

### 数据与资产

- REFERENCES.md 是资源状态唯一台账；候选入口不等于已核验或获准使用。
- 使用外部数据前，从官方一手来源核验版本、字段、许可、大小、用途、下载范围和校验和；
  Stage-0 `.splat` 预览的已核验范围和未核验 provenance 以 REFERENCES.md 为准。
- 下载数据、模型或 checkpoint 前取得 Owner 对来源、范围、磁盘和许可的确认。
- 原始数据、转换数据、训练输出、缓存和 checkpoint 放在 ignored 或批准的外部存储，禁止进 Git。
- 派生产物必须保留 source checksum、producer/version、transform/config hash、license review 和 lineage。

### Contract

- docs/PRD.md 第 6 节只定义概念语义；已冻结机器 schema 是 PR-00 的
  `contracts/objgauss/0.1.0/episode.schema.json`，以及 PR-01 的
  `contracts/objgauss/0.2.0/` 四份记录与 PR-02A 的 `contracts/objgauss/0.3.0/` 六种记录。
- 实现公共 contract 时必须明确 producer、consumer、版本、必填/可选、默认值、单位、时区、
  唯一标识、缺失语义、兼容策略、迁移与回滚。
- 同一概念只保留一个权威 schema；第二份副本视为漂移风险。
- 物理对象身份、模型 slot 和 renderer address 不得静默共用一个真值字段。
- prediction-before-observation 与 correction-after-observation 必须分账；GT 或未来输入不得
  泄漏到推理 feature。
- commanded action、executed action 和缺失 action 必须分开表达。
- sibling group 必须整体隔离 train/validation/test；无 GT 或 measured action 时结果是
  blocked，不得伪装成零值或 pass。

PR-00、PR-01 与 PR-02 字段和序列化分别以对应精确版本的唯一机器 schema 为准；PRD 概念约束仍
适用于尚未进入机器 contract 的下游语义，不能用 Viewer 私有字段或第二份 schema 绕过。

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
- 旧根 LICENSE 和新项目当前政策均为 all rights reserved，但二者是独立授权边界；未经 Owner
  对具体文件批准，仍不得复制旧实现。
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
| 公共字段与机器 contract | contracts/objgauss/ 下对应精确版本的唯一 schema；当前为 `0.1.0`、`0.2.0` 与 `0.3.0` |
| 当前任务、非目标、验收和队列 | docs/PROJECT_PLAN.md |
| 长期架构取舍 | docs/adr/；ADR-001–006 已 accepted，分别覆盖 Stage-0 至 PR-02C 当前切片 |
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

当前 `npm run check` 覆盖 Stage-0、PR-00、PR-01、PR-02A contract 与 PR-02B 的纯
calibration/power/manifest 测试；`./scripts/accept-pr01` 是 PR-01 含 simulator、formal cohort 和
Delivery 的完整验收门，`./scripts/check-pr02b-pilot` 是 PR-02B 含真实 source、独立 audit、GPU
reserve 和 freeze lineage 的完整验收门。当前尚无 lint、typecheck 或模型训练门，不得把既有
命令外推为 PR-02C 之后的完整验证；每个后续实现 PR 必须先补与其风险匹配的真实命令和测试。

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
