# ObjGauss 当前风险登记

> 最近更新: 2026-07-11
> 历史缓解流水见 `docs/state/archive/risks-through-2026-07-09.md`。

| ID | 风险 | 当前证据 | 关闭条件 | 状态 |
| --- | --- | --- | --- | --- |
| R-002 | 默认对象分组尚非稳定语义对象分割 | CLIP 路线仍 `do-not-promote`；native Gaussian identity evidence 未通过 real gate | held-out real scenes 上超过简单 baseline 且无 teacher/GT leakage | open |
| R-006 | Viewer 首屏与 bundle 过重 | catalog 已改为只加载 selected/staged evidence；production JS 仍约 5.95 MB | code splitting 并通过目标设备首屏 SLA | open |
| R-012 | full 4.5M PLY 不能直接交互 | full route 约 4.4 FPS；sampled 1M 才达到本机 SLA | LOD/streaming/chunking 通过明确硬件 SLA | accepted / deferred |
| R-014 | `audit:world-viewer` 过宽且条件漂移 | 已纠正旧 pills、隐藏模型入口、rotate/scale、StabilityDashboard、catalog/load-count 与跨时刻断言；artifact flows 使用独立浏览器，full/truth audit 均通过，显式 failed status 会阻断 | 已满足；不得新增平行 audit | closed |
| R-015 | supervised CE 训练链出现巨大中间梯度 | probability-space `-target/p` 保留数学语义；两个训练 caller 已改用解析 softmax-logit VJP，EPS 两侧 finite-difference 与一步下降通过 | required Python CI 持续覆盖 logits-space VJP | closed |
| R-017 | ObjectState 可能只是 observation slot，不是 persistent world state | 三段 BOP 为 0 pass / 6 fail / 3 blocked；首批 RBO official-chain visibility 为 `0/3`，action-ready 仅 2 scenes，新 ledger 为 0 pass / 7 fail / 2 blocked；follow-up P0 未解锁，且单 interaction 只有一种 lighting | 至少 3 个严格 controlled scenes 独立完成 identity/prediction/intervention 比较，并超过简单 baseline | open / P0 |
| R-018 | Teacher/GT evidence 泄漏可能制造 identity 假通过 | feature tensor 已绑定实际内容并进入训练/eval；raw identity 要求有限 association threshold；旧 GT-row 键控路线不能 pass。当前仍只有 fixture teacher 运行，没有本地真实 DINO/CLIP teacher 证据 | 真实 teacher extraction 保存原始 feature/provenance，并通过同一 anti-leakage 与 held-out 检查 | mitigating |
| R-019 | 公开代码与研究环境不可复现或许可不清 | 已补 CI、all-rights-reserved LICENSE、core lock 与顶层训练依赖 pin；GPU/driver/transitive wheel 仍按运行记录 | CI required；Owner 明确最终代码许可；外部训练运行保存完整环境与硬件记录 | mitigating |
| R-020 | `core`/CLI/App/状态面失控 | controlled/BOP/transition/public/real-sample authoring、handoff、package、row ledger、assignment/v2 experiment、artifact/temporal/long-smoke、identity benchmark/gate、renderer/training SCC、mask/CLIP/semantic、teacher-evidence contract、ObjectField filesystem tooling 与 projection orchestration 已外移；`objgauss/core/*.py` 约 12.0k physical LOC，兼容壳使其仍有 146 模块 / 678 lazy exports | 在 breaking window 收缩 compatibility/public exports；继续拆分 CLI/App | mitigating |
| R-021 | Viewer 控件与 Spark source 行为不一致 | hidden source 不首屏 fetch；Spark visibility/translate texture 已同步；rotate/scale 等未闭环控件已移除；CLI handoff 明示 | 已满足；后续交互必须有 source-level browser evidence | closed |
| R-022 | metric-level gate 不能单独证明 raw artifact 语义 | generic row gate 已明确仅作诊断；canonical controlled full handoff validator 会从 capture、identity predictions 与 eval summary 保留的 prediction/intervention records 重跑三个 evaluator，并拒绝篡改 IDF1/ADE/accuracy/GT | 研究结论只接受 live canonical handoff 与已保存 candidate records；generic summary-only pass 不得计证据 | accepted boundary |
| R-023 | `pipelines` 与 `evaluation` 存在 package-level 双向耦合 | BOP reality rows 消费 pipeline handoff validator，而 BOP bundle pipeline 消费该 row evaluator；当前模块 DAG 和双顺序冷启动均通过 | 冻结期后提取稳定 summary contract/validator，使 package dependency 单向且不改 schema | open / deferred |
| R-024 | `objgauss.core` root ABI 依赖导入顺序 | 32 个 root export 与 legacy 子模块同名；root-first 得 function，导入同名子模块后变 module；678-key surface 当前不能无破坏收缩 | 冻结新增；breaking window 先退役 32 个歧义 alias，再按包分批退役其余 canonical aliases/wrappers | open / deferred |

## 风险处理规则

- Fixture/schema/reviewability 只能关闭 plumbing 风险，不能关闭模型能力风险。
- `accepted` 不等于 production-ready；必须保留适用硬件、资产层级和负证据。
- 新缓解更新本表当前证据，不再追加长篇时间流水；详细历史进入归档或具体任务。
