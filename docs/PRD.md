# ObjGauss 对象中心 Gaussian 世界模型 PRD

> 状态：Draft，等待 Owner 评审
> 版本：0.2
> 日期：2026-07-14
> 当前验收范围：文档与立项，不包含代码、依赖、数据下载或研究结论
>
> 知识状态：“一个实现 PR 只引入一个可证伪假设”和 `PR-04` Gaussian 生死门已经由
> Owner 确认为 `decision`；项目完成层级、技术栈和数值阈值仍是 `working_assumption`。
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

为避免把 Demo、研究证据和最终价值混成一个 MVP，本 PRD 定义四个完成层级：

| 完成层级 | 对应阶段 | 可以声称什么 | 不能声称什么 |
| --- | --- | --- | --- |
| Foundation Release | 0–1 | 数据、坐标和 Oracle Object Gaussian 管线正确 | 在线身份、动作理解或世界模型成立 |
| Demo Release | 2 | 真实视频中可在线维护对象身份与不确定性 | 学到了干预或隐藏因果 |
| Research Release | 3–4 | 动作影响预测，且隐藏物性试探在受控反事实上有效 | 对真实机器人规划有价值 |
| Value Release | 5–6 | 真实校准后，Gaussian rollout 提高闭环规划成功率 | 超出已测机器人、任务和安全边界的泛化 |

这些层级是 **声明阶梯**，不是 PR 的线性分组。当前执行范围是
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
5. **缺失必须显式。** 数据源不具备动作、深度、力或物性时，记录 capability 和
   `missing_reason`，不能伪造零值。
6. **证据分级。** fixture、synthetic、public replay、controlled real 和 closed-loop
   evidence 分开统计；schema-valid 或 reviewable 不等于指标通过。
7. **真实在线声明先过身份门。** `PR-01`–`PR-04` 可以在 oracle/simulation ObjectState 上形成
   明确受限的动作与表示结论；`PR-06` Identity Gate 未通过时，不得形成真实在线
   predict-correct、隐藏物理或主动因果声明。
8. **一 PR 一假设。** 实现 PR 必须预先写明可证伪假设、falsifier、基线、split、裁决状态
   和失败路径；不得用“适配一个数据集”代替研究问题。

## 5. 当前实现范围：Demo A（PR-00–PR-04）

### 5.1 必须交付

- 完成许可证、首批资源和最小技术栈的前置决策；未通过前不创建源码框架。
- `PR-00`：经过 Owner 批准的 schema v0、坐标约定、合成 episode 和重投影门。
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

本节是 PRD 级语义，不是已经批准的公共 contract。字段类型、坐标约定、序列化格式和
版本迁移策略必须在 `PR-00` 的唯一机器 contract 中冻结。

### 6.1 公共信封

每个记录至少可追溯到：

```text
schema_version
source_dataset, source_version, source_uri
license_review_id
episode_id, split
timestamp, time_unit
coordinate_convention_id, unit_system
raw_checksum, transform_version
capabilities
```

### 6.2 Observation

```text
Observation:
  rgb
  depth
  K
  T_WC
  timestamp
```

- `PR-00` 合成 fixture 中上述字段均为必需；后续 GT 数据源必须逐项声明 present/missing。
- 其他数据源缺少字段时必须给出 `missing_reason`，不能用单位矩阵、全零深度等哨兵伪装。
- `T_WC` 的方向、左右手系、矩阵作用方向和单位在 schema PR 前均为待决策。

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
| NFR-004 | 外部数据与模型在许可审核、大小预算和校验和批准前不得下载或再分发 |
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
| 0 | `PR-00` | 合成 contract fixture 与批准的 GT pilot | 重投影候选门 `< 1 px`，正式阈值待 pilot/Owner 冻结；坐标往返一致；错误 convention 可检测 | 数据与坐标管线可用 |
| 1 | `PR-03` | 固定的 GT identity/mask/depth/pose 对象集 | held-out view、背景污染、刚性变换和几何补全协议通过 | Oracle Object Gaussian 可用 |
| 2 | `PR-05`–`PR-06` | 批准的真实多视角小子集，HO-Cap 为当前候选 | 无 future leakage；胜同集重跑的 3D connected-components；位姿阈值先冻结 | 在线 ObjectBelief Demo 可用 |
| 3 | `PR-01`–`PR-02` | 固定 sibling groups 与 action baselines | 优于 copy/velocity/action-free；action shuffle 显著退化 | 模型使用了动作 |
| Gaussian 生死门 | `PR-04` | ObjectState/point cloud/Gaussian 受控 ablation | 预注册 primary endpoint 上有稳定增益 | Gaussian 可进入 dynamics；否则仅视觉使用 |
| 4 | `PR-08`–`PR-09` | 物性 sibling/OOD 组合与主动 probe 对照 | hidden posterior 胜无历史；主动 probe 胜随机/固定并提高任务成功率 | 有受控主动因果证据 |
| 5 | `PR-10` | 批准的一个真实机器人配置/任务，RH20T 为当前候选 | 候选 0/10/50/200 ladder 待预算冻结；residual 胜 commanded-only 和同预算 fine-tune | 真实域校准有效 |
| 6 | `PR-11` | 统一任务、动作预算和安全约束 | 预注册 primary endpoint 或组合效用超过全部必需基线 | 世界模型提高已测闭环规划价值 |

阶段 1 的“稳定”“不污染”和阶段 2 的“厘米/数度级”仍不够可执行。对应实现 PR 必须先用
小 pilot 生成基线，冻结指标、阈值、置信区间和失败判定，再扩大数据；不能在看到最终结果
后改阈值。

阶段 3 的数据量由 `PR-01` determinism pilot 和资源预算冻结；每个起点候选生成以下分支：

```text
hold
push(+x, weak)
push(+x, strong)
push(-x, weak)
push(+y, weak)
grasp/lift
```

其中 `grasp/lift` 与 push 的控制器、时长和成功语义不同。reset 可重复性通过后，应先判断
它能否与 push 共用 sibling 评估；否则保留相同起点，但拆成独立 action cohort。

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

实现路径原则已确认，以下事项在代码 PR 前仍是硬门：

1. 是否批准“Foundation → Demo → Research → Value”的声明层级及首个对外 Demo。
2. 新项目许可证，以及是否允许从 all-rights-reserved 归档移植具体代码。
3. Python/训练、前端/查看器、数据落盘格式和包管理技术栈。
4. schema 版本化、坐标方向、单位、canonical frame 和缺失值语义。
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
- 本文没有联网复核外部数据集的当前字段、版本、大小或许可；这些内容必须由资源审计 PR
  使用上游一手资料确认。
