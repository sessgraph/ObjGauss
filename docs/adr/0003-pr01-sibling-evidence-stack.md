# ADR-003：PR-01 Sibling Evidence Contract、Runtime 与无 RGB Demo 边界

> 状态：Accepted
> 日期：2026-07-14
> 决策者：Owner（PR-01 三项边界）
> 适用范围：`PR-01A`–`PR-01F` 严格成对干预里程碑

## 背景

RES-001 已支持 ManiSkill `3.0.1` 的 snapshot/RNG fork，并在程序化 CPU primitive 上支持五个
external-force sibling 的 action/contact/settling 跨进程复现。这只证明 source gate 可行，尚未
形成可版本化 episode、失败 attempt、完整 cohort、独立 invariance report 或可从 clean checkout
复现的验收交付。

现有 `episode 0.1.0` 专门冻结 `synthetic-audit-v0` 与重投影门。它不能表达 snapshot、RNG、
branch、attempt、physics、contact 和 sibling split，且作为已发布 contract 不得原地修改。

## 决策

### PR-01 是一个里程碑，内部按依赖拆成 A–F

```text
PR-01A Contract ─┐
                 ├─> PR-01C Adapter/Writer -> PR-01D Independent Audit
PR-01B Runtime ──┘                                  |
                                                    v
                                           PR-01E Cohort/Split
                                                    |
                                                    v
                                           PR-01F Demo/Report/CI
```

独立审计器必须先于正式 cohort。Preflight 可以为冻结预算和阈值生成隔离证据，但不得混入正式
cohort；在 PR-01D 负例矩阵通过前，不允许批量生成 PR-01E。

### Contract 使用 `0.2.0` 四层记录

- `episode.schema.json`：单个完整成功 branch 的事实记录。
- `experiment.schema.json`：完整 sibling cohort 的 action set、split、preflight、retry、预算、
  阈值、runtime 和 claim boundary。
- `attempt.schema.json`：每次执行及其失败/重试，不把失败包装成半合法 episode。
- `invariance-report.schema.json`：独立审计器的四态 machine verdict。

`episode 0.1.0` 的字节级 SHA-256 固定为
`b619618706a1bd8da370c465fb36ba8e8edb08ada3406663fc2e2ed2dfa0da9c`。Validator 必须按
`schema_version + contract_kind` 精确分派；禁止 `latest`、未知版本和静默升级。不存在
`0.1.0 -> 0.2.0` 自动迁移，因为 snapshot、RNG、contact、attempt 和 lineage 无法从旧记录推断。

Schema 约束格式，具体 `fixture_id`、action、cohort、split、预算和阈值由 experiment manifest
冻结。重投影审计仍属于 `0.1.0`；sibling audit 使用独立 `audit_kind=sibling_invariance`。
`source_commit` 不能在 tracked spec 中硬编码“最终提交”，因为提交无法包含自身未来 SHA；spec
只冻结 `runtime-current-clean-git-head` policy，generator 从当前 clean HEAD 注入实际值，并由
Delivery 与 CI 复核。Preflight 的历史测量 spec 单独保留，provenance policy 迁移不得改变物理、
动作、seed、split、阈值或预算。

同一 sibling group 唯一允许变化的输入是
`/intervention/commanded_action`。Trajectory、contact、terminal state 与 settling 是干预结果，
允许且预期随 branch 变化；审计器不得把结果差异误报为控制变量污染。

### 生产仿真栈保持隔离

- ManiSkill `3.0.1`、SAPIEN `3.0.3`、CPython `3.10.20` 和 Torch `2.13.0+cu130` 以精确 lock
  固定，放入独立 `sim` optional extra，只用于离线 episode 生成。
- 允许实现 adapter、原子 writer、CLI、行为测试、真实 runtime smoke 和 CI job。
- Simulator 运行时只用内置解析 primitive，禁止外部 asset 和网络。
- 审计器不得依赖 ManiSkill，也不得导入 writer 的比较或 hash 结果。
- CI 缺少批准 runtime 时必须明确 `blocked` 或失败；不得 skip 后继续宣称生产验证通过。
- 该授权不包含 RGB/GPU renderer、训练、模型、Gaussian dynamics、外部数据或机器人控制。

当前通过的 source gate 实测 backend 为 `physx_cpu`。若未来切换 `physx_cuda`，必须作为新
source gate 单独预注册和裁决，不能沿用 CPU 证据。

### Writer、重试与 evidence 语义

- 逻辑幂等键是 `experiment_id + group_id + branch_id`；每次执行有独立 `attempt_id`。
- Writer 先写临时目录，完成所有 checksum、Schema 与本地完整性验证后再原子 rename。
- 已成功 episode 不允许覆盖：相同 digest 为 no-op，不同 digest 为 `invalid` conflict。
- JSON 采用固定规范化和 SHA-256，拒绝 NaN/Inf。
- Writer 可以输出便利 hash，但独立 evaluator 必须从原始字段和 artifacts 重算。
- 成功 attempt 与 episode 在同一原子 branch 目录发布；没有成功 episode 的失败 attempt 进入
  独立 `attempts/`，两类账本都必须进入 audit 与 Delivery checksum。
- 每 branch 最多重试一次；只有 simulator crash、startup timeout 或 atomic write failure 可重试。
  科学失败不得换 seed、删 group、截断或重试成“成功”。

### 正式 cohort 与 split

正式设计为 `2 object specs × 3 layouts × 2 start poses × 4 reset seeds = 48 groups`，每组五个
actions，共 `240 episodes`。每个 `(object, layout, start)` stratum 内对四个 seed 使用稳定
SHA-256 排序，按 `2/1/1` 分到 train/validation/test，即 `24/12/12 groups`。禁止 Python
`hash()` 和 episode-row 随机切分。

Preflight 使用 `2 objects × 3 layouts × 1 start × 2 reserved seeds = 12 groups / 60 episodes`，
与正式 seed 隔离，只用于冻结 timeout、contact/settling/effect thresholds、p95 runtime、p95
artifact size 和正式资源预算。

### Demo 使用无 RGB 五联状态回放

不扩大 renderer 授权。PR-01F 从已审计 episode artifacts 读取五个 branch，在同一坐标系和时间轴
上以 Canvas/SVG 显示对象、action vector、trajectory、contact point/normal/impulse、settling 与
snapshot/RNG hash；支持统一播放、暂停和拖动。浏览器不运行 simulator，不使用 CDN、外部资产
或 GPU。Demo 负责解释，machine invariance report 才是验收事实源。

## 验收与退出码

独立审计状态为 `supported`、`rejected`、`blocked`、`invalid`，聚合优先级固定为
`invalid > rejected > blocked > supported`。建议 CLI 退出码为：`0 supported`、`1 evaluator
internal error`、`2 rejected`、`3 blocked`、`4 invalid`。

正式入口为 `./scripts/accept-pr01`。它必须从 clean checkout 完成依赖冻结、contract/writer/
audit/negative/Demo tests、真实五 branch smoke、正式 cohort、独立 audit、report/checksum/Demo
生成；入口、builder 与 verifier 都必须拒绝 staged、tracked 或非 ignored untracked 改动。只有
最终 `supported` 才退出 `0`。本地成功不能替代绑定最终 commit SHA 的远端 CI。

## 目录责任

| 路径 | 唯一责任 |
| --- | --- |
| `contracts/objgauss/0.2.0/` | PR-01 四份机器 Schema |
| `contracts/fixtures/pr01a/` | PR-01A 正负 contract fixtures 与 checksum manifest |
| `src/pr01/contract-dispatch.mjs` | `0.1.0` / `0.2.0` 精确版本分派与 contract-local 语义门 |
| `sim/` | PR-01B–E 隔离 runtime lock/package、adapter/writer 与 cohort producer；不得成为核心依赖 |
| `artifacts/pr01/` | ignored、可重建的 episode/attempt/report/demo 产物 |
| `docs/state/` | 动态 PR 队列、项目状态、风险和待办；不复制稳定 contract 语义 |

## 回滚与影响

`0.2.0` 尚未发布时可删除本 ADR、新 schema、fixtures、dispatcher 和 tests，回到 PR-00 +
source-gate 状态；不得借回滚修改 `0.1.0`。一旦 `0.2.0` 随 PR-01A 提交，后续改变合法实例集合
必须新增版本，不得原地编辑。

该决策最多支持“项目可以生产并独立审计没有被初态、随机数、切分、重试和血缘污染的受控
反事实证据”。它不证明模型理解因果，不支持 Gaussian dynamics 或机器人规划价值。
