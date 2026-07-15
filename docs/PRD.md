# ObjGauss 对象中心 Gaussian 世界模型 PRD

> 状态：Draft；`PR-00` 已固化，PR-01 source gate 与 `PR-01A`–`PR-01F` 已提交；代码承载
> 验收 SHA `234ba00` 的本地完整门与六项远端 Actions 均为 `supported`
> 版本：0.12
> 日期：2026-07-15
> 当前验收范围：文档、Stage-0 本地预览、`PR-00` synthetic contract 证据与无渲染 primitive
> sibling action/contact pilot、`0.2.0` contract、隔离 production runtime、原子 writer、独立
> audit、冻结 formal cohort 与无 RGB Delivery；PR-01 严格 sibling evidence 里程碑已关闭，但不
> 包含模型、原始训练数据或下游研究结论
>
> 知识状态：“一个实现 PR 只引入一个可证伪假设”、`PR-04` Gaussian 生死门，以及
> “面向研究评审的 Demo A 是首个验收目标”已经由 Owner 确认为 `decision`；`PR-04` 的
> primary endpoint 已确定为 held-out sibling groups 上的多步 `effect-vs-hold` ObjectState
> 预测误差，并采用 group-first 的配对聚合；稳定增益要求 paired hierarchical bootstrap
> 95% 置信区间下界超过预先冻结的最小实际增益 `δ`。`δ` 只能由与最终 test 隔离的 pilot
> 基于 evaluator 噪声和跨 seed 波动产生。完整项目完成层级、技术栈和 `δ` 数值仍是
> `working_assumption`。Owner 另已决定先交付一个直接渲染 3D Gaussian 的可见页面，不使用
> notebook，并允许下载 REFERENCES.md 登记的固定小型 `.splat`；该 Stage-0 切片不是
> `PR-00` contract、训练好的 ObjGauss 输出或任何研究假设的通过证据。
> Owner 随后确认该切片只做 Web，默认体验必须像可自由浏览的环境级 3D scene，而不是单物体
> 展台或从外部俯看的有限底板；默认相机必须位于环境内部，地面不能在首屏暴露完整边界。
> synthetic world 只能作为 viewer fixture，不能伪装成 `PR-00` episode。
> Owner 已完成 `PR-00` Decision Freeze。当前本地实现使用唯一 JSON Schema `0.1.0`、固定
> `synthetic-audit-v0`、独立 evaluator 与 Web consumer，在 36 个 primary points 上得到
> `max_camera_reprojection_error_px = 1.005e-14 < 1.0` 的 `supported` 窄裁决；它不支持真实
> 数据、Gaussian 重建、世界模型、动力学或规划价值声明；代码承载验收 SHA `234ba00` 的
> PR-00 远端 check 已成功。
> 固定 ManiSkill 3.0.1 的 CPU/no-render primitive pilot 进一步支持了同 snapshot/RNG 的五个
> external-force sibling action/contact outcomes，并通过相反 branch 顺序的独立进程复跑；它只
> 批准 PR-01 primitive push source，不代表 robot controller、renderer/GPU 或完整 PR-01 已实现。
> Owner 已决定把 PR-01 作为一个验收里程碑，内部依次拆成 Contract、Runtime、Writer、
> Independent Audit、Cohort 和 Delivery 六个切片；独立审计先于正式 cohort。Contract 使用
> `0.2.0` episode/experiment/attempt/invariance-report 四层记录，production simulator 只作为
> 隔离 `sim` optional extra，Demo 使用无 RGB 五联状态回放。完整决策见 ADR-003。
> PR-01B 已用精确 uv lock 从全新临时 venv 完成 wheel-only 外部依赖安装，并在显式 offline、
> 空只读 asset 门下由 canonical/reverse 两个真实进程得到一致五分支 evidence；这只支持
> production runtime 可复现。代码承载验收 SHA `234ba00` 的远端 runtime smoke 已成功。
> PR-01C 已由真实 canonical/reverse 进程生成语义一致的五分支 golden group，并通过原子发布、
> 幂等 replay、冲突拒绝、失败 attempt 与非有限值负例；这只支持单 group writer，后续
> audit、正式 cohort 与 Delivery 仍由各自切片独立裁决。
> PR-01D 已在重新生成的真实 golden group 上独立重算 14 个 hard gates，并让 11 类预注册
> mutation 命中稳定四态与 reason code；这只支持审计器及 fixture，不表示 preflight、正式
> cohort 或最终交付已完成。
> PR-01E 已保留一次 provisional effect threshold rejection，再用 preflight 冻结阈值与资源预算；
> 正式 48 groups / 240 episodes、24/12/12 split、零额外 attempt 通过独立审计，且该 SHA 的
> frozen cohort 远端门成功。该结果仍不表示因果模型理解或 Gaussian dynamics 已完成。
> PR-01F 的五联状态 Demo、Delivery report/checksums、clean-checkout guard 与 `accept-pr01` 已实现；
> source commit 在运行时绑定当前 clean HEAD，不能在冻结 spec 中自引用。代码承载验收 SHA
> `234ba00` 的一键验收重建 48 groups / 240 episodes、0 failed/extra attempts，独立 audit 与
> Delivery verifier 均为 `supported`；该 SHA 的六项远端 Actions 全部成功，PR-01 里程碑关闭。
> 归档事实和候选资源状态由 [`../REFERENCES.md`](../REFERENCES.md) 统一维护。

## 1. 决策摘要

新项目拟验证一个严格命题：**持久对象状态与对象级 Gaussian 表示，能否在动作条件下
预测未来，并最终提高具身任务的闭环规划成功率。**

项目不寻找“万能数据集”，而采用三层数据金字塔：

1. 真值密集数据负责把坐标、身份、位姿和规范对象 Gaussian 做对。
2. 可控仿真数据负责动作、干预、隐藏物理和同起点反事实分支。
3. 真实机器人数据负责执行偏差、接触域差异和最终规划价值。

实现队列不按这三层或数据集拆分，而按“一个 PR 只引入一个可证伪假设”拆分。数据集只是
某个假设的证据来源。`PR-04` 必须用受控 ablation 裁决 Gaussian 是否对预测或简单规划有
稳定增益；无增益时 Gaussian 降级为视觉记忆和 renderer，不进入 dynamics 核心。

首个必须说服的受众是研究评审，首个验收目标是 Demo A。它必须让 `PR-04` 对 Gaussian 的
增量价值给出可复现的 `supported` 或 `rejected` 结论；漂亮渲染或后续 Demo 不能替代该裁决。
该裁决的唯一 primary endpoint 是 held-out sibling groups 上的多步 `effect-vs-hold`
ObjectState 预测误差；数据效率、简单规划和渲染质量只作 secondary metrics。
primary endpoint 只评估被直接干预的 target object。裁决值先在每个 sibling group 内跨
冻结的 horizons 聚合 target 的归一化轨迹误差，再用相同 group 的模型差值做 paired
comparison；episode 长度和场景内非目标对象数量不得改变 group 权重。

为避免把 Demo、研究证据和最终价值混成一个 MVP，本 PRD 定义四个完成层级：

| 完成层级 | 对应阶段 | 可以声称什么 | 不能声称什么 |
| --- | --- | --- | --- |
| Foundation Release | 0–1 | 数据、坐标和 Oracle Object Gaussian 管线正确 | 在线身份、动作理解或世界模型成立 |
| Demo Release | 2 | 真实视频中可在线维护对象身份与不确定性 | 学到了干预或隐藏因果 |
| Research Release | 3–4 | 动作影响预测，且隐藏物性试探在受控反事实上有效 | 对真实机器人规划有价值 |
| Value Release | 5–6 | 真实校准后，Gaussian rollout 提高闭环规划成功率 | 超出已测机器人、任务和安全边界的泛化 |

这些层级是 **声明阶梯**，不是 PR 的线性分组。首个核心计划目标是
[`PROJECT_PLAN.md`](PROJECT_PLAN.md) 中的 Demo A（`PR-00`–`PR-04`）：先建立 contract、
成对干预、ObjectState-only 基线和 Canonical Gaussian，再直接验证 Gaussian 的增量价值。
Demo Release 是首个在线产品化演示目标，Research Release 是首个有因果说服力的目标，
Value Release 是项目最终成功标准。

## 2. 背景与问题

目标系统需要同时处理：

- 跨时间、跨视角、遮挡后的物理对象身份。
- 可独立变换、逐视角补全的对象级 Gaussian 表示。
- 在下一帧到来前，根据动作预测一个或多个未来。
- 从主动试探中更新质量、摩擦等隐藏物理信念。
- 从严格相同的初始状态生成 sibling interventions，支持控制变量和反事实评估。
- 在真实执行偏差和扰动下重新规划。

现有候选数据源各只覆盖其中一部分。若没有统一语义、坐标和 lineage，身份错误会被误报
为动力学错误，执行动作偏差会被误报为因果关系，随机视频切分还可能产生未来帧或同组
分支泄漏。

旧归档提供了重要负证据：旧对象模型未胜简单 3D connected-components 基线，真实
identity/prediction/intervention 证据也未闭环。详细数值及来源只在
[`../REFERENCES.md`](../REFERENCES.md) 维护，不能当作新项目当前指标；本 PRD 因而要求
Identity Gate 通过前不训练复杂因果头。

## 3. 用户与核心任务

| 用户 | 核心任务 | 需要的输出 |
| --- | --- | --- |
| 数据/平台工程师 | 把异构数据转为一致且可审计的 episode | schema 校验、坐标报告、lineage、数据卡 |
| 3D/视觉研究者 | 构建并跟踪对象级 Gaussian | 对象身份、规范几何、位姿、可见性、不确定性 |
| 世界模型研究者 | 比较动作条件未来与严格反事实 | sibling groups、预测、基线、误差和统计报告 |
| 机器人研究者 | 校准仿真到真实并进行闭环选择 | commanded/executed action、rollout、成功率与失败分类 |
| 演示使用者 | 直观看到预测与修正的区别 | 同步 2D/3D 视图、预测未来、GT、误差矢量和置信度 |

## 4. 产品原则

1. **ObjectState 是推理单元。** Gaussian 是对象的可渲染几何/外观证据，不默认等于身份
   真值，也不能只因画面更好看就宣称规划更好。
2. **预测和修正分开。** `prediction_before_observation` 不得读取目标帧；
   `correction_after_observation` 必须单独记录和评分。
3. **物理身份和渲染地址分开。** 数据集中的持久 `object_id` 是物理身份 GT；渲染器的
   slot/index 是派生地址。
4. **指令和执行分开。** `commanded_action` 与 `executed_action` 不得混为一个字段。
5. **缺失必须显式。** 数据源不具备动作、深度、力或物性时，记录 capability，并用
   `availability` tagged union 区分 `present` 与带受控 `reason` 的 `missing`；不能用 `null`、
   零值或其他哨兵伪装。
6. **证据分级。** fixture、synthetic、public replay、controlled real 和 closed-loop
   evidence 分开统计；schema-valid 或 reviewable 不等于指标通过。
7. **真实在线声明先过身份门。** `PR-01`–`PR-04` 可以在 oracle/simulation ObjectState 上形成
   明确受限的动作与表示结论；`PR-06` Identity Gate 未通过时，不得形成真实在线
   predict-correct、隐藏物理或主动因果声明。
8. **一 PR 一假设。** 实现 PR 必须预先写明可证伪假设、falsifier、基线、split、裁决状态
   和失败路径；不得用“适配一个数据集”代替研究问题。

## 5. Stage-0、PR-00 当前实现与 Demo A 计划范围（PR-00–PR-04）

进入核心 `PR-00` 前，当前已有一个纯 Web 的 Stage-0 入口切片：页面默认把确定性生成的严格
`.splat` records 组成环境级 synthetic Gaussian world，并可读取 Git ignored 的固定外部审计
样例；两条路径都校验大小、SHA-256 与 32-byte records，再用 WebGL2 投影 3D covariance、
计算 Gaussian alpha、按深度合成。它只证明环境级输入的本地渲染链路可见，不满足
FR-001/FR-002、机器 schema、坐标门、episode 同步、3DGS 训练/重建或动力学能力；准确外部
资源事实见 [`../REFERENCES.md`](../REFERENCES.md)。

`PR-00` 当前工作区已经建立唯一
[`episode.schema.json`](../contracts/objgauss/0.1.0/episode.schema.json)、固定 seed/config 的
仓库内 producer、独立重投影 evaluator、机器 report 与同步 Web consumer。派生 episode、RGB、
depth、mask、report 和 browser bundle 均位于 ignored `generated/pr00/`，可由 `npm run check`
确定性重建。该门验证 contract、坐标与证据链，不把 Stage-0 splat 或任何外部数据提升为
`PR-00` 输入。

### 5.1 必须交付

- 新项目当前保持私有并按 all rights reserved 管理；对外发布前重新决策许可证。
- `PR-00`：已在当前工作区实现 Owner 批准的 schema `0.1.0`、坐标约定、合成 episode、
  重投影门和 Web 证据页；提交/合并状态与窄假设 verdict 分账。
- `PR-01`：经资源审计批准的仿真器生成可重复、无同组泄漏、只改变声明变量的 sibling
  branches；ManiSkill 只是当前候选。
- `PR-02`：完全不使用 Gaussian 的 copy、constant-velocity、action-free 和
  action-conditioned ObjectState dynamics 基线。
- `PR-03`：使用 GT identity/mask/depth/pose 的 `canonical_gaussians`，以及 held-out view、
  背景污染、刚性移动和补全评估。
- `PR-04`：ObjectState、point cloud、Gaussian 的受控增量实验和明确职责决策。
- 每个 PR 的固定 fixture、行为测试、机器报告、失败样例和可复现 Demo。

### 5.2 明确不包含

- `PR-05` 之后的 HO-Cap 适配、在线对象发现、重识别或真实视频身份结论。
- 隐藏物理、主动试探、真实校准和闭环 MPC。
- diffusion、replay buffer、自生成未来、graph DB 或长期记忆平台。
- RH20T、DROID、CALVIN 等大型数据的全量下载或训练。
- 真实机器人部署、持久化服务、公共 API 或不可逆数据迁移。
- 直接恢复旧归档的框架、前端或 824 个测试。

## 6. 概念数据协议

本节是 PRD 级概念语义，不是可执行 schema。`PR-00` 的 synthetic audit 字段由
[`0.1.0 episode`](../contracts/objgauss/0.1.0/episode.schema.json) 建立；PR-01 sibling evidence
分别由 [`0.2.0 episode`](../contracts/objgauss/0.2.0/episode.schema.json)、experiment、attempt
和 invariance-report schema 建立。发生冲突时，实例按精确 `schema_version + contract_kind`
选择的机器 schema 为准；不得使用 `latest`、静默升级或第二份字段真值。

### 6.1 公共信封

每个记录至少可追溯到：

```text
schema_version
source_dataset, source_version, source_uri
license_review_id
episode_id, split
episode_time_s
coordinate_convention_id, unit_system
raw_checksum, transform_version
capabilities
```

`0.1.0` 固定为 PR-00 synthetic episode，字节不可变；`0.2.0` 固定为 PR-01 sibling evidence
family。所有 object schema 使用 `unevaluatedProperties: false` 拒绝未知字段。已发布 schema
不得原地修改；`0.x` 中任何改变合法实例集合的修改升级 MINOR，PATCH 只能修改不影响 contract
的说明、示例或错误文本。不存在自动 `0.1.0 -> 0.2.0` 迁移，因为 snapshot、RNG、contact、
attempt 和 lineage 无法推断；其他跨版本迁移也必须显式、可测试并记录前后 checksum。

所有可能缺失的 contract 值使用以下唯一语义：

```text
present := { availability: "present", value: <valid value> }
missing := { availability: "missing", reason: <controlled reason> }
reason  := not_measured | not_provided | not_applicable | redacted | invalidated
```

`present` 必须且只能携带有效 `value`；`missing` 禁止携带 `value`。`null`、零/单位矩阵、空串、
NaN 和默认置信度均不是缺失表示。

RGB、depth 和其他数组资源不内嵌 JSON，只使用以下 descriptor；URI 解析相对于 fixture
manifest，consumer 必须在使用前核对 shape、dtype 和完整 SHA-256：

```text
ArrayResource:
  uri
  media_type
  dtype
  shape
  sha256
```

### 6.2 Observation

```text
Observation:
  rgb
  depth
  K
  T_WC
  episode_time_s
```

- `PR-00` 合成 fixture 中上述字段均为必需；后续 GT 数据源必须逐项声明 present/missing。
- 其他数据源缺少字段时必须使用上述 `missing` variant，不能用单位矩阵、全零深度等哨兵伪装。
- `T_WC · p_C = p_W`，使用列向量左乘；World 为右手系、`+Z` 向上、meter。投影使用
  `T_CW = inverse(T_WC)` 转入 OpenCV Camera frame。

### 6.3 ObjectBelief

```text
ObjectBelief:
  id
  canonical_gaussians
  T_WO
  velocity, angular_velocity
  visibility, existence
  mask
  covariance_or_confidence
  estimate_phase
```

- `estimate_phase` 至少区分 `oracle`、`prediction_before_observation` 和
  `correction_after_observation`。
- Canonical object frame 由 producer 在对象创建时定义并在 episode 内保持不变；synthetic
  原点为刚体质心，轴为 producer 声明的右手语义轴。禁止从观测、PCA 或 mesh 外观推断；
  `T_WO · p_O = p_W`。
- Symmetry 必须可追溯并显式表示为 `none`、有限旋转集合或绕 object-frame axis 的连续旋转。
  有限旋转使用单位 `[w,x,y,z]`，连续轴必须归一化。未知 symmetry 使用
  `missing:not_provided`，姿态指标状态为 `blocked`，不得默认成 `none`。
- 阶段 0–1 允许 GT mask/pose，仅能标记为 `oracle`。
- 阶段 2 的在线推理不得读取当前时刻之后的 mask、pose 或图像；GT 只用于训练监督或离线评估。

### 6.4 Intervention

```text
Intervention:
  actor, target_id, type
  contact_point, direction, magnitude, duration
  coordinate_frame
  commanded_action, executed_action
```

- 仅在数据源真实提供或仿真器明确生成时出现。
- `hold` 不是“缺失动作”；它是显式零干预对照。
- 没有 measured action 的视频不能进入 causal pass/fail 统计。

### 6.5 CausalLineage

```text
CausalLineage:
  episode_id
  snapshot_id
  sibling_group_id
  branch_id
  changed_variable
  reset_seed
```

`snapshot_id + sibling_group_id` 是反事实分组主键。拆分必须按 sibling group 隔离，不能把
同一起点的不同分支分散到 train/validation/test。

## 7. 数据金字塔

下表来自用户研究材料，是**候选职责分配**而非已核验资产。候选入口、资源状态和归档
边界的唯一台账是 [`../REFERENCES.md`](../REFERENCES.md)。每个来源都必须先完成上游版本、
许可、下载范围、校验和、字段语义和存储预算审查。

| 层 | 主候选 | 辅助候选 | 在本项目中的职责 | 禁止替代的证据 |
| --- | --- | --- | --- | --- |
| 真值密集 | Kubric/MOVi | HO-Cap、HOT3D | 坐标、实例、规范对象；真实遮挡/重识别外测 | 真实动作因果 |
| 可控仿真 | ManiSkill 3 | CausalWorld、Physion++ | reset、动作分支、物性干预和 OOD | sim-to-real 或真实规划价值 |
| 真实机器人 | RH20T | DROID、CALVIN、少量自采 | 接触、执行偏差、长时规划和最终价值 | 严格 sibling 反事实，除非另行构造 |

BridgeData、RoboNet、Open X-Embodiment 仅保留为后期预训练候选，不作为开局因果真值。

## 8. 功能需求

| ID | 需求 | 首次要求阶段 |
| --- | --- | --- |
| FR-001 | 数据 adapter 输出统一记录、capabilities、lineage 和校验报告 | 0 |
| FR-002 | 自动验证单位、时间戳、坐标往返和 2D/3D 重投影 | 0 |
| FR-003 | 同步显示 RGB、深度、3D、对象 ID、轨迹并支持跨视角高亮 | 0 |
| FR-004 | 从 GT ID/pose 构建、补全、独立变换和渲染对象 Gaussian | 1 |
| FR-005 | 无推理期 GT 地维护在线身份、位姿、遮挡信念和不确定性 | 2 |
| FR-006 | 从同一 snapshot 可重复生成并审计 sibling branches | 3 |
| FR-007 | 在下一观察前输出动作条件的多未来及误差 | 3 |
| FR-008 | 通过主动试探更新隐藏物性 posterior，并在未见组合上外测 | 4 |
| FR-009 | 分离并比较真实 `commanded_action` 与 `executed_action` | 5 |
| FR-010 | 用 rollout 选动作、执行、检测扰动并重新规划 | 6 |

## 9. 非功能需求

| ID | 约束 |
| --- | --- |
| NFR-001 | 所有指标能追溯到原始输入、固定 split、模型/配置 hash 和独立重算脚本 |
| NFR-002 | adapter、坐标、拆分、future-leakage、reset 和 lineage 必须有行为级测试 |
| NFR-003 | 数据、训练输出、缓存、checkpoint 和未脱敏日志不得提交 Git |
| NFR-004 | 外部数据与模型在许可审核、大小预算和校验和批准前不得下载或再分发；Stage-0 仅允许 REFERENCES.md 登记的固定预览 |
| NFR-005 | 仿真 reset 对同一 snapshot/seed 必须可重复，分支只改变声明变量 |
| NFR-006 | 阶段 2 起的在线路径只能读取当前及历史输入，并记录端到端延迟 |
| NFR-007 | 真实机器人阶段必须先冻结工作空间、速度/力限制、急停和失败恢复策略 |
| NFR-008 | 新生产依赖、服务、持久化或公共 contract 继续需要 Owner 动作级批准 |
| NFR-009 | 每个实现 PR 只引入一个可证伪假设和一个预注册 primary endpoint，并输出 supported/rejected/blocked/invalid 裁决 |

延迟、吞吐、显存、磁盘和真实机器人安全阈值目前没有可靠输入，不能伪设；应在相应阶段
进入实现队列前冻结。

## 10. 评估与声明门槛

| 声明层级 | 对应实现 PR | 最小证据 | 必须通过的门 | 通过后允许的声明 |
| --- | --- | --- | --- | --- |
| 0 | `PR-00` | 固定 seed/config 的仓库内 JavaScript producer；只提交 producer、fixture spec 与预期 checksum manifest，派生资源 ignored | `synthetic-audit-v0` 全部 primary points 的 `max_camera_reprojection_error_px < 1.0`；独立 evaluator；坐标往返一致；错误 convention 可检测 | 数据与坐标管线可用 |
| 1 | `PR-03` | 固定的 GT identity/mask/depth/pose 对象集 | held-out view、背景污染、刚性变换和几何补全协议通过 | Oracle Object Gaussian 可用 |
| 2 | `PR-05`–`PR-06` | 批准的真实多视角小子集，HO-Cap 为当前候选 | 无 future leakage；胜同集重跑的 3D connected-components；位姿阈值先冻结 | 在线 ObjectBelief Demo 可用 |
| 3 | `PR-01`–`PR-02` | 固定 sibling groups 与 action baselines | 优于 copy/velocity/action-free；action shuffle 显著退化 | 模型使用了动作 |
| Gaussian 生死门 | `PR-04` | ObjectState/point cloud/Gaussian 受控 ablation | held-out sibling groups 上 target object 的多步 `effect-vs-hold` ObjectState 预测误差有稳定增益 | Gaussian 可进入 dynamics；否则仅视觉使用 |
| 4 | `PR-08`–`PR-09` | 物性 sibling/OOD 组合与主动 probe 对照 | hidden posterior 胜无历史；主动 probe 胜随机/固定并提高任务成功率 | 有受控主动因果证据 |
| 5 | `PR-10` | 批准的一个真实机器人配置/任务，RH20T 为当前候选 | 候选 0/10/50/200 ladder 待预算冻结；residual 胜 commanded-only 和同预算 fine-tune | 真实域校准有效 |
| 6 | `PR-11` | 统一任务、动作预算和安全约束 | 预注册 primary endpoint 或组合效用超过全部必需基线 | 世界模型提高已测闭环规划价值 |

阶段 1 的“稳定”“不污染”和阶段 2 的“厘米/数度级”仍不够可执行。对应实现 PR 必须先用
小 pilot 生成基线，冻结指标、阈值、置信区间和失败判定，再扩大数据；不能在看到最终结果
后改阈值。

`PR-04` 的 primary endpoint 使用 group-first paired aggregation：每个 held-out sibling group
先形成一个归一化多步轨迹误差，再比较同一 group 上 Gaussian 与非 Gaussian 模型的差值。
对 ObjectState-only 与 ObjectState + point cloud 分别计算配对改进，并按冻结的数据层级和
训练 seed 做 paired hierarchical bootstrap；两项 95% 置信区间下界都必须超过预先冻结的
最小实际增益 `δ`，`PR-04` 才能判为 `supported`。正式 experiment spec 仍须冻结 horizon、
状态分量、归一化尺度、组内权重和 bootstrap 细节。`δ` 由与最终 test 完全隔离的 pilot
估计 evaluator 噪声与跨 seed 波动后冻结；pilot 不进入最终统计，最终结果可见后不得调整
`δ`。具体估计器和数值仍须在 experiment spec 中预注册。

多步 horizon 按物理时间定义，而不是 simulator steps：从干预开始，覆盖动作执行和固定的
post-action settling，并在预注册的物理时间点评分。动作时长、settling 时长和采样点由与
最终 test 隔离的 pilot 冻结；最终结果可见后不得延长、缩短或选择性删除时间点。

primary error 覆盖完整刚体 ObjectState：位置、按对象 symmetry metadata 校正的朝向、
线速度和角速度。每类误差除以独立 pilot 估计、且不低于 evaluator noise floor 的 robust
scale；四类归一化误差各占 primary scalar 的 25%。robust scale 与 noise-floor estimator
必须在 final test 前冻结。可见性、外观/渲染质量和接触事件仅作 secondary metrics，
不得进入 primary verdict。

多对象场景只把 `Intervention.target_id` 指向的直接干预对象纳入 primary endpoint。其他对象
的状态误差、碰撞传播和 collateral effects 必须单独报告为 secondary metrics；非目标对象
数量不得稀释或放大 target 的 primary error。

final test 必须完整留出 object identities、scene layouts 及其全部 sibling groups。primary
push 类型、方向和力度范围须在 train 中有覆盖并跨 split 分层保持支持，因此本阶段只主张
跨对象与跨布局泛化，不主张未见动作外推。任何 object、layout 或 sibling group 跨 split
复用都使结果 `invalid`。

held-out 对象的 Gaussian 与 point cloud 必须由同一份冻结的 pre-intervention 多视角 oracle
context 构建，并共享原始帧、视角、深度、GT identity/mask/pose 和采样预算。context manifest
必须记录时间边界与校验和；任何 post-action frame、目标 rollout future 或额外视角预算进入
任一表示，都使比较 `invalid`。具体视角数与采样预算在独立 pilot 后预注册。

`PR-03` 的 Gaussian/point-cloud builders 在进入 `PR-04` 前按版本和 hash 冻结，并离线生成
immutable representation artifacts。`PR-04` 可为两种表示训练容量匹配的 adapters，但必须
共享同一 dynamics backbone、训练协议和输出头；梯度不得回传 builders。端到端重训结果只能
作为独立 secondary experiment，不能进入 primary verdict。

ObjectState-only、ObjectState + point cloud 与 ObjectState + Gaussian 三个 primary arms
必须在预注册容差内匹配 adapter + backbone 的总可训练参数量，并使用相同的数据曝光量、
optimizer updates、batch policy 和调参预算。FLOPs、峰值显存和推理延迟完整报告为资源
secondary metrics，但不作为本阶段的强制等价条件。

阶段 3 的数据量由 `PR-01` determinism pilot 和资源预算冻结；每个起点候选生成以下分支：

```text
hold
push(+x, weak)
push(+x, strong)
push(-x, weak)
push(+y, weak)
```

PR-01 正式 cohort 固定为 `2 object specs × 3 layouts × 2 start poses × 4 reset seeds = 48 sibling
groups / 240 episodes`。每个 `(object, layout, start)` stratum 内按 seed 的稳定 SHA-256 排序，
用 `2/1/1` 分配为 `24/12/12` train/validation/test groups；禁止 Python `hash()` 或按 episode
row 随机切分。正式 cohort 前运行完全隔离的 `12 groups / 60 episodes` preflight，只用于冻结
timeout、effect/contact/settling thresholds、p95 runtime/size 和正式资源预算，不进入训练或统计。
正式 episode 的 `source_commit` 必须由验收进程从当前 clean Git HEAD 注入并与 Delivery/CI SHA
一致；冻结 experiment spec 只记录 `runtime-current-clean-git-head` 策略，不得硬编码一个未来
提交无法自引用的旧 SHA。Dirty worktree 必须 `invalid`，不能生成最终 Delivery 声明。

以上 `hold + push` 是 `PR-04` primary endpoint 的唯一 action cohort。`grasp/lift` 使用同一
类起点但必须拆成独立 secondary cohort，并配置与其动作时长匹配的 hold 对照；其结果不得
进入或救活 `PR-04` primary verdict。

## 11. 必须保留的对照

- `ObjectState-only` vs `ObjectState + point cloud` vs `ObjectState + Gaussian` vs
  `Gaussian-only`
- action-free vs action-conditioned
- `prediction_before_observation` vs `correction_after_observation`
- 随机视频样本 vs 同初始状态 sibling interventions
- 阶段 2 身份模型 vs 同一固定评估集重跑的 3D connected-components；归档数值只作诊断
- 阶段 3 copy-state、constant-velocity 和 action-shuffle
- 阶段 6 随机/无动作预测器、纯 BC 和 `ObjectState-only` planner

## 12. 主要风险与停止条件

| 风险 | 早期信号 | 停止/降级条件 |
| --- | --- | --- |
| 身份错误污染动力学 | IDF1、swap、遮挡恢复不达标 | 不进入复杂动作/因果头训练 |
| 坐标或标定错误 | 重投影、往返或单位检查失败 | 不生成训练集，回到 adapter |
| sibling 泄漏或不等价 | 初态 hash、seed 或非目标变量不同 | 整组作废，不进入指标 |
| Gaussian 仅改善画面 | 渲染更好但规划无提升 | 降级为可视化输出，不主张决策价值 |
| 物性不可辨识 | posterior 不收敛或多组解释等价 | 报告不可辨识，不伪造单一物性 GT |
| sim-to-real 失配 | executed/commanded 差异主导误差 | 先做校准，不扩大真实训练 |
| 数据/许可/成本失控 | 条款不清、磁盘或预处理超预算 | 不下载或只用更小已批准子集 |
| 真实机器人不安全 | 超限、接触异常、恢复不确定 | 禁止闭环执行，仅离线 replay |

## 13. Owner 必须确认的决策

实现路径原则和首个验收目标已经确认：先面向研究评审交付 Demo A，并由 `PR-04` 裁决
Gaussian 的增量价值。`PR-00` 的许可证、contract、坐标与验证栈决策已经冻结；以下列表同时
保留已确认决策和仍会阻塞后续 PR 的开放项：

1. 是否批准“Foundation → Demo → Research → Value”的完整声明层级和 Demo B–D 定义。
2. `decision`：新项目当前保持私有并按 all rights reserved 管理；对外发布前重新决策许可证。
   `PR-00` 全新实现且不移植独立 all-rights-reserved 归档中的任何文件；未来移植必须逐文件
   重新取得 Owner 授权并记录许可审查。
3. `decision`：`PR-00` 使用 JavaScript ESM、Node 24 LTS、npm、Ajv 8、esbuild 和
   `node:test`，不引入服务端框架；GitHub Actions 在 PR 和 `main` 上以 `npm run check` 执行
   Node 24、`npm ci`、contract audit、测试与 Web build。未来 Python/训练栈、长期 Viewer 和
   数据落盘格式仍待确认。
4. `decision`：`PR-00` 以 JSON Schema Draft 2020-12 作为唯一机器 contract 源，并使用
   Robotics/OpenCV 坐标约定：`T_AB · p_B = p_A`，列向量左乘；World 为右手系、`+Z`
   向上、meter；Camera 为 `+X` 右、`+Y` 下、`+Z` 前；`T_WC` 表示 Camera → World，投影使用
   `T_CW = inverse(T_WC)`，WebGL bridge 只属于 Viewer consumer。Quaternion 使用有限、
   归一化的 `[w, x, y, z]` 和确定性符号；`episode_time_s` 从首个 Observation 的 `0.0` 开始，
   Observation 严格递增，同步字段共时，事件可同刻但不得倒退。Canonical object frame 由
   producer 定义且 synthetic 原点为刚体质心；symmetry 按第 6.3 节显式表达。缺失值使用
   第 6.1 节的 tagged union。Schema 从 `0.1.0`
   开始使用严格 SemVer、精确版本匹配、未知字段拒绝和显式可追溯迁移。
5. 许可审核标准、磁盘/下载预算、算力与可接受的最长验收时间。
6. 每个 PR 的正式指标、阈值、统计检验、seed 数和资源预算。
7. 目标机器人、动作空间、控制频率、延迟预算和安全规范。

Gaussian 在动力学中的角色不再要求 Owner 预先选择，由 `PR-04` 的预注册实验裁决。

这些决策及建议 PR 顺序见 [`PROJECT_PLAN.md`](PROJECT_PLAN.md)。

## 14. 来源与证据边界

- 本 PRD 的研究路线、候选数据源和声明门来自 Owner 提供的 2026-07-14 研究材料。
- “一 PR 一可证伪假设”、`PR-00`–`PR-11` 路径和 `PR-04` 生死门来自 Owner 提供并于
  2026-07-14 确认的重新规划实现路径；准确队列只在 [`PROJECT_PLAN.md`](PROJECT_PLAN.md)
  维护。
- 旧项目只读恢复点、候选移植资源、历史负证据、外部数据和 Hugging Face 入口统一见
  [`../REFERENCES.md`](../REFERENCES.md)。
- Stage-0 固定 `.splat` 只复核了直接上游位置、版本、大小、记录数与校验和；其资产生成
  provenance 仍未核验，且只准用于本地渲染预览。`PR-00` synthetic fixture 不使用该文件或
  其他外部数据。准确范围见
  [`../REFERENCES.md`](../REFERENCES.md)。其他外部数据的字段、版本、大小和许可仍须由资源
  审计 PR 确认。
