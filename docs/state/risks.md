# ObjGauss 当前风险登记

> 最近更新: 2026-07-10
> 历史缓解流水见 `docs/state/archive/risks-through-2026-07-09.md`。

| ID | 风险 | 当前证据 | 关闭条件 | 状态 |
| --- | --- | --- | --- | --- |
| R-002 | 默认对象分组尚非稳定语义对象分割 | CLIP 路线仍 `do-not-promote`；native Gaussian identity evidence 未通过 real gate | held-out real scenes 上超过简单 baseline 且无 teacher/GT leakage | open |
| R-006 | Viewer 首屏与 bundle 过重 | catalog 已改为只加载 selected/staged evidence；production JS 仍约 5.95 MB | code splitting 并通过目标设备首屏 SLA | open |
| R-012 | full 4.5M PLY 不能直接交互 | full route 约 4.4 FPS；sampled 1M 才达到本机 SLA | LOD/streaming/chunking 通过明确硬件 SLA | accepted / deferred |
| R-014 | `audit:world-viewer` 过宽且条件漂移 | 历史 full audit 会在旧 probe/wait 条件超时 | 收敛已有 audit 到当前 evidence viewer 行为；不得新增平行 audit | open |
| R-015 | supervised CE clip 梯度与 forward 不一致 | clip plateau derivative 已归零；zero assignment、finite gradient 与 non-clipped 路径进入 required CI | 已满足 | closed |
| R-017 | ObjectState 可能只是 observation slot，不是 persistent world state | 三段 BOP 为 0 pass / 6 fail / 3 blocked；首批三段 RBO 虽有 RGB-D/6DoF/camera motion/wrench，但严格 V-O-V 为 `0/3`，action sign/target link 未闭环；RRC 不适配现行 3D action | 至少 3 个真实 controlled scenes 独立通过 identity/prediction/intervention 比较 | open / P0 |
| R-018 | Teacher/GT evidence 泄漏可能制造 identity 假通过 | audit 已绑定实际 feature digest/content，训练与各 eval 复用同批 evidence；label one-hot、shuffle、random 与 provenance 伪装均有负路径 | 已满足；后续新增 teacher source 必须沿用同一检查 | closed |
| R-019 | 公开代码与研究环境不可复现或许可不清 | 已补 CI、all-rights-reserved LICENSE、core lock 与顶层训练依赖 pin；GPU/driver/transitive wheel 仍按运行记录 | CI required；Owner 明确最终代码许可；外部训练运行保存完整环境与硬件记录 | mitigating |
| R-020 | `core`/CLI/App/状态面失控 | controlled/BOP/transition/public/real-sample authoring、handoff、package、row ledger、assignment/v2 experiment、artifact/temporal/long-smoke、identity benchmark/gate、renderer/training SCC、mask/CLIP/semantic、teacher-evidence contract、ObjectField filesystem tooling 与 projection orchestration 已外移；`objgauss/core/*.py` 约 12.0k physical LOC，兼容壳使其仍有 146 模块 / 678 lazy exports | 在 breaking window 收缩 compatibility/public exports；继续拆分 CLI/App | mitigating |
| R-021 | Viewer 控件与 Spark source 行为不一致 | hidden source 不首屏 fetch；Spark visibility/translate texture 已同步；rotate/scale 等未闭环控件已移除；CLI handoff 明示 | 已满足；后续交互必须有 source-level browser evidence | closed |
| R-022 | metric-level gate 不能单独证明 raw artifact 语义 | gate 会重算 status/gap/gain，但 raw prediction/GT 到 metrics 由 canonical evaluator 负责 | 研究结论必须保存 raw refs 并经 canonical evaluator；手工 metrics 不得计作证据 | accepted boundary |
| R-023 | `pipelines` 与 `evaluation` 存在 package-level 双向耦合 | BOP reality rows 消费 pipeline handoff validator，而 BOP bundle pipeline 消费该 row evaluator；当前模块 DAG 和双顺序冷启动均通过 | 冻结期后提取稳定 summary contract/validator，使 package dependency 单向且不改 schema | open / deferred |
| R-024 | `objgauss.core` root ABI 依赖导入顺序 | 32 个 root export 与 legacy 子模块同名；root-first 得 function，导入同名子模块后变 module；678-key surface 当前不能无破坏收缩 | 冻结新增；breaking window 先退役 32 个歧义 alias，再按包分批退役其余 canonical aliases/wrappers | open / deferred |

## 风险处理规则

- Fixture/schema/reviewability 只能关闭 plumbing 风险，不能关闭模型能力风险。
- `accepted` 不等于 production-ready；必须保留适用硬件、资产层级和负证据。
- 新缓解更新本表当前证据，不再追加长篇时间流水；详细历史进入归档或具体任务。
