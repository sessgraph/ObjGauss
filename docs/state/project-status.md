# ObjGauss 当前状态总览

> 最近更新: 2026-07-09

## 当前阶段

MVP 原型可运行，已完成流程化基线提交，已接入真实 3DGS splat renderer，并具备可复现的 ObjGauss v1 闭环验收 demo。

项目整体仍是 development-stage research prototype / 开发阶段研究原型。当前 API、
CLI、资产布局、指标、模型产物和文档都可能在 stable release 前变化；HF 资产与本地
ignored `outputs/` 产物用于研究复现和 handoff，不能表述为 production-ready 或
commercial demo release。

2026-07-09 docs-only research intake: Owner 提供的 ObjGauss 训练数据来源与教师模型
资料已整理为 `docs/training/objgauss-training-data-teacher-models.md`。该文档把
uCO3D / CO3D / VOS / tracking / dynamic geometry / action data、SAM2 /
Grounding DINO / DINOv2 / CoTracker / TAPIR / geometry teachers / SAGA-style
baseline 等内容收敛成 project-facing research notes 和后续 backlog 候选；它不是
architecture contract，不登记新 asset，不下载数据，不引入默认依赖，也不改变当前
State Variable Gate 优先级。所有外部数据集、教师模型、许可和可用性事实在实际
ingestion 前仍需官方来源复核。

随后完成 `OBJECTSTATE-CONTROLLED-DATASET-CONTRACT-001`：新增
`docs/dataset/controlled-reality-contract.md` 和
`objgauss.core.controlled_schema`，把 Phase 2 controlled real 数据语言冻结为
Episode / Object Instance / Action Event / State Transition，并复用现有
`objgauss-objectstate-controlled-capture-manifest-v1` 作为唯一机器事实源。
新增 `objgauss-objectstate-controlled-dataset-contract-summary-v1` summary，可审计
identity / prediction / causal 三个 invariant：同一 physical `object_id` 跨时间存在、
`S(t) -> S(t+n)` 具备 timestamped 6DoF pose GT，以及非零 action interval 覆盖同对象连续
pose transition。该切片不采集真实数据、不创建 GT、不重建 Gaussian、不训练模型、不运行
identity / prediction / intervention evaluator，也不声明 reality gate pass 或 world model。

随后按新阶段规划完成 `OBJECTSTATE-MODEL-CONTRACT-001`：新增
`docs/architecture/objectstate-model-contract.md`，冻结 `Gaussian / AssignmentEvidence
-> ObjectState Model -> ObjectState -> Gate` 的模型接口。该 contract 明确当前 MVP 不创建
并行模型事实源，而是沿用既有 `AssignmentSolverV2 -> A[N,K] -> ObjectStateProjection`
主线；`A[N,K]` 是唯一 object assignment source，renderer-facing hard `object_id` 只能由
assignment / matching / export policy 派生。文档定义 GaussianToken / AssignmentEvidence
输入、assignment matrix 中间表示、ObjectState 输出、MVP loss family、gate handoff 和模型
artifact 必需 metadata，并明确 Transformer / Slot Attention、replay buffer、diffusion、
dynamics、torch/CUDA 重依赖和真实训练都不在本切片内。

Viewer 主流程已明确收敛为 Three.js-first：所有分割、对象化和移动能力都建立在
Three.js 先加载并展示高斯云 / 模型之后。当前对象层已支持多模型版本展示、选中
ObjectState group、移动 / 旋转 / 缩放 gizmo、undo / redo / cancel、Shift snap，
并新增 `projected-object-centroid-picker-v1` screen-space object picker，使鼠标选择
对象不再直接依赖 Gaussian renderer raycast hit-test。训练证据、Gaussian probe 和协议
诊断仍保留在高级系统区，不作为主流程第一入口。

Viewer 已把 `sourceLayer` 的真实 `.splat` 接进主 Three.js world：默认真实样例 V2 使用
`spark-source-splat-stage-v1` 在同一个 Three.js scene / camera 内加载 `Lego 原始 splat`，
顶部 HUD 和 `Three.js 世界` 面板显示 `高斯展示=完整 splat`。处理后的 object-aware PLY
仍作为可交互对象层叠加，用于 ObjectState 选择、bbox / centroid 叠层、TransformControls
移动和调试证据；它不再作为主视觉的完整高斯替代品。2026-07-07 已新增
`source-splat-object-translate-v1`：当 source `.splat` 和 object-aware PLY 满足
`sourceCount == pointCount` 时，主 `ThreeWorld` 会用 Spark Dyno modifier 将
`gsplat.index -> object_id -> object translate` 接到 source splat 子集。real-sample v2
targeted browser check 证明移动 `object-0` 后 native source splat motion
`active=true`、`transformedObjects=1`、`sourceCount=pointCount=5696`。
随后完成 `NATIVE-SPLAT-MOTION-HARDEN-001`：右侧选中摘要和 `Three.js 世界` 面板现在展示
`真实绑定` / `绑定点数`，root/debug panel 也暴露 `ready/active/countMatches/sourceCount`
telemetry；bbox-projected Playwright check 证明移动 selected source splat 子集时，
desktop / mobile 的 selected screen delta 分别为 `28.708px` / `26.922px`，同模型 peer
对象 `peerScreenDelta=0`、`peerWorldDelta=0`，UI 会从 `已绑定` 刷新为 `已随动`。当前实现不生成
新的训练 / Demo 产物，不改变 ObjectState / manifest / training contract；rotate / scale、
任意第三方 `.splat` object id 和 Gaussian 重优化仍不在默认能力内。

Viewer 对象交互随后补齐 `projected-object-bbox-picker-v2`：主鼠标选择现在只按
ObjectState bbox 的 screen-space 投影命中，空白区域不会再被中心点半径误选；选中对象会强制
显示高亮 bbox、centroid / glow / selection ring，并把 object-aware Gaussian overlay 提亮，
方便确认“选中了哪个对象”和“移动的是对象层”。模型版本列表改为两行布局，避免完整高斯、
点预览、对象层和处理按钮挤在一行。随后完成 `GAUSSIAN-OBJECT-PROCESS-FLOW-001`：
viewer catalog 新增默认 `lego-alpha-raw-source`，只加载未分割 source `.splat`，对象层状态为
`raw-gaussian / 未生成`；点击 `生成` 会进入 `objgauss-gaussian-object-process-flow-v1`
handoff，展示可复跑的本地 CLI 命令、viewer path 和目标 object-aware model。浏览器不运行重型
分割 / 训练模型；本地命令生成 ignored `public/samples/objgauss-real-sample-v2-sample-aware-lego.ply`
后，`加载结果` 会 fetch 该 PLY、切换到 `real-sample-v2-sample-aware-lego`，并把 stage 收敛为
生成后的对象层模型。Playwright + system Chrome 验证 desktop raw -> handoff -> load result ->
source splat object motion：result model `object-layer-ready`，stage 只保留
`real-sample-v2-sample-aware-lego`，移动 `object-0` 后 native source splat motion
`active=true`、`transformedObjects=1`、`selectedScreenDelta=28.513px`、peer world delta 为 `0`；
mobile 验证 handoff panel 宽度 `284px < 390px`、`命令就绪` / `加载结果` 可见且无框架 overlay。
full `audit:world-viewer` 当前仍有旧 trainable Gaussian probe / stability 等待过宽问题，最新失败点
为 `scripts/audit-world-viewer.mjs:595`；本流程继续采用 targeted browser check 作为验收事实源。

随后按 Owner 要求完成 `DEMO-CATALOG-REAL-SPLAT-001` 的前置整理：新增
`nike-3dgs-local` 真实 Gaussian asset，从 `cakewalk/splat-data` 下载 `nike.splat`，
经既有 `splat-to-objgauss-ply` 管线生成 ignored `public/samples/nike.splat` 和
`public/samples/nike_objects.ply`。该样例为 `270,491` Gaussians / 4 个 object，counts
为 `84,781 / 69,968 / 74,734 / 41,008`，仅作为本地测试素材，不作为公开商用 demo 承诺。
Viewer catalog 同步新增 `nike-real-splat-demo`，并把首屏 dock / 默认 stage 收敛成 5 个
curated demos：`lego-alpha-raw-source`、`real-sample-v2-sample-aware-lego`、
`polyhaven-chair`、`nike-real-splat-demo`、`plush`。near-1M、OGC、trainable artifact 和旧
closure 仍保留在高级模型版本 / URL 调试路径，不再抢首屏。Playwright + system Chrome
desktop / mobile 验证 `catalogModelCount=10`、`dockModelCount=5`、`stageVisibleCount=5`，
点击 `nike-real-splat-demo` 后 selected / object layer 都进入 loaded 状态，截图在
`/tmp/objgauss-demo-catalog-desktop.png` 和 `/tmp/objgauss-demo-catalog-mobile.png`。

随后按“找一个更好的、构建训练数据”的方向补齐
`TRAINING-DATA-POLYHAVEN-DENSE-001`：新增
`polyhaven-school-chair-nerf-dense`，复用 Poly Haven School Chair CC0 glTF，但把
训练输入从原来的 16-frame / 256px 提升为 32-frame / 384px NeRF-style RGBA orbit
dataset。已执行 `uv run objgauss assets pull polyhaven-school-chair-nerf-dense`，本地
ignored 产物为 `outputs/assets/training/polyhaven-school-chair-nerf-dense/` 和
`outputs/assets/converted/polyhaven-school-chair-nerf-dense/training-manifest.json`。
manifest 记录 `frames=32`、`image_size=384`、`triangles=5072`、`files=35`；
`inspect-nerf` 验证 train / val / test 各 32 frames，总计 `frames=96`、
`missing_images=0`、`invalid_transforms=0`；PNG 检查为 shape `(384,384,4)`，
alpha coverage `min=0.182231`、`mean=0.286609`、`max=0.359138`。该步骤只构建
训练数据，不表示已经训练出新的 Gaussian 模型；下一步若继续模型产出，应在 ignored
`outputs/` 上跑 Splatfacto candidate / smoke training，再决定是否进入 viewer/export
默认策略。

随后完成 `POLYHAVEN-DENSE-SPLATFACTO-SMOKE-001`：使用
`polyhaven-school-chair-nerf-dense` 跑通 100-step Nerfstudio Splatfacto smoke，并生成新的
ignored Gaussian candidate。输出路径为
`outputs/training/polyhaven-chair-dense-splatfacto-smoke/`，其中 checkpoint
`step-000000099.ckpt` 约 `44M`，exported Gaussian PLY
`export-smoke-cuda/splat.ply` 为 `50,000` Gaussians / 约 `12M`，object-aware PLY 为
`object-field-sam/polyhaven-school-chair-nerf-dense_splatfacto_sam_objects.ply` / 约 `13M`。
训练前因 `/tmp/objgauss-cuda13` 临时 CUDA wrapper 已消失，首次 run 在 `gsplat: No CUDA
toolkit found` 处失败；重建 wrapper symlink 后同一命令通过。dense 相比旧 chair smoke 的
Splatfacto train loss 更低 `0.359148 vs 0.389867`、PSNR 略高
`11.108979 vs 11.075611`，Object emergence 也更好：
`object_emergence_score=0.796839 vs 0.709509`、`stability_ari=0.763848 vs 0.581991`。
但当前 SAM vote 质量回退：supervised fraction `0.161200 vs 0.242340`，
vote conflict fraction `0.859677 vs 0.743914`，final vote loss
`0.999969 vs 0.849624`。结论：dense candidate 已生成且值得保留，但暂不推进为
viewer/export 默认；当时的下一步是先为 dense Chair 补 mask policy / benchmark row 复验，
再做发布或默认策略决定。

随后完成 `POLYHAVEN-DENSE-BENCH-001`：`chair-dense-splatfacto-smoke` 已加入
`docs/benchmarks/splatfacto-scenes.json` 并通过 4-row scene suite 复验。全量
`npm run benchmark:splatfacto:scenes -- --run --skip-sam` 通过，`status=ready missing=0`，
scene suite 扩展为 Lego / Fern / Chair / dense Chair 四行。dense Chair 使用相同
SAM 8 frames / 6 masks / `max_area_fraction=0.75` policy，train / held-out split 为
`6 / 2` frames；ARI=`0.786356`、curve OES=`0.759438`、render=`0.185040`、
held-out loss=`2.002325`、held-out render=`0.178836`。相对旧 Chair，dense 的
assignment stability 更好，但 render 和 held-out render 更弱（旧 Chair 分别为
`0.248716` / `0.224084`）。结论保持不 publish、不设为 viewer/export 默认；若继续，
应先尝试更保守的 dense Chair mask policy 或扩更多许可清晰的小型样本行。

随后冻结 `OBJECTSTATE-STATE-VARIABLE-GATE-001` 的研究验收定义：
`docs/architecture/objectstate-state-variable-gate.md` 明确 ObjGauss v2 的核心 claim
不是“能分割 Gaussian”，而是证明 `ObjectState_t` 是 `X_t` 的近似充分统计量。该 gate 将
验收拆成 Identity State、Physical State、Causal State 三层，并要求五类证据：
identity persistence、occlusion recovery、view invariance、predictive sufficiency 和
counterfactual / action interface。当前阈值策略是 synthetic oracle 使用高阈值
（例如 identity / occlusion recovery `>= 0.95`），controlled real / public rows 先要求
可复现、可解释、无明显 identity collapse，open-world real rows 只记录失败，不伪装成通过。
因此下一阶段算法质量优先级改为实现 state-variable smoke gate，而不是继续追
renderer 指标、diffusion、replay buffer 大系统或 viewer/export 默认模型。

随后完成 `OBJECTSTATE-IDENTITY-GATE-001` 的第一版可证伪实现：新增
`objgauss.core.objectstate_identity_gate`，冻结
`objgauss-objectstate-identity-gate-v1` 和
`objgauss-objectstate-identity-dataset-v1`。该 smoke evaluator 复用 synthetic identity
oracle / cross-view / occlusion recovery / perturbation / adversarial swap fixtures，但要求
显式 candidate prediction 输入；`predicted_slots_by_fixture` 或
`predicted_assignments_by_fixture` 缺失时 fail-fast，不再允许 oracle expected-slot fallback。
输出指标包括 `id_accuracy`、`idf1`、`embedding_retrieval_recall_at_1`、
`long_term_drift_rate`、`fragmentation_rate`、`occlusion_recovery_rate` 和
`contrastive_margin`。同步收紧旧 `v2_stability_diagnostics` / `v2_stability_gate`：
无 prediction 不能 pass，assignment matrix 列数必须等于 fixture slot 数。当前结论只表示
Identity State smoke gate 可运行且可拒绝坏候选；Physical State / Causal State 和真实
controlled real rows 仍未完成，不能把 ObjGauss 表述为已证明的 object-centric world model。

随后补齐目标文件建议的 `OBJECTSTATE-IDENTITY-MODEL-001` 和
`OBJECTSTATE-PREDICTIVE-GATE-001` smoke slices：`objgauss.core.objectstate_identity_encoder`
新增 NumPy 线性 ObjectState identity encoder 训练摘要，schema 为
`objgauss-objectstate-identity-encoder-training-v1`，使用 supervised contrastive identity
loss 记录 initial / final loss、positive / negative loss、active negatives 和 retrieval
recall；它不引入 identity graph、replay buffer、diffusion 或 renderer loss。
`objgauss.core.objectstate_predictive_gate` 新增
`objgauss-objectstate-predictive-gate-v1`，用 synthetic ObjectState pose + velocity 预测
`ObjectState(t+n)`，并和 history baseline 比较 `state_ade`、`history_ade`、
`prediction_error_ratio`、`state_sufficiency_score` 和 `identity_consistency_rate`。当前
predictive gate 的 velocity 来源明确是 synthetic world oracle trajectory，属于可失败的
smoke evaluator；controlled real rows、learned dynamics 和 counterfactual action gate 仍未完成。

随后按新评审建议完成 `OBJECTSTATE-CAUSAL-GATE-001` synthetic controlled action smoke：
新增 `objgauss.core.objectstate_causal_gate`，schema 为
`objgauss-objectstate-causal-gate-v1`，action schema 为 `objgauss-objectstate-action-v1`。
最小 action set 为 `push_left` / `push_right` / `hold`，gate 用
`ObjectState(t).pose + velocity + action_delta` 预测 counterfactual target，并和 no-action
baseline 比较 `action_conditioned_ade`、`no_action_ade`、`intervention_gain`、
`counterfactual_outcome_accuracy`、`wrong_direction_rate` 和 `identity_consistency_rate`。
`candidate_action_scale=0` 的负路径会 fail，证明该 gate 不能被纯 tracker / no-action
predictor 代替。当前仍是 synthetic controlled action evidence；controlled real action rows、
relation change、hide / reveal 和 learned dynamics 仍未完成。

随后完成 `OBJECTSTATE-REALITY-GATE-001` 的第一版 controlled real / public row
acceptance contract：新增 `objgauss.core.objectstate_reality_gate`，schema 为
`objgauss-objectstate-reality-gate-v1`，row schema 为
`objgauss-objectstate-real-public-row-v1`。该 gate 把 Phase 1 真实验证拆成
`identity` / `prediction` / `intervention` 三类 evidence rows，并强制分离
`pass_rows` / `fail_rows` / `blocked_rows`。非 blocked identity 行必须有 identity GT 和
`idf1` / `fragmentation_rate` / `swap_rate` / `identity_collapse`；prediction 行必须有
pose GT 和 `state_ade` / `history_ade` / `prediction_gap_vs_history_model`；intervention
行必须有 pose + action GT 和 `action_conditioned_ade` /
`counterfactual_outcome_accuracy` / `wrong_direction_rate`；所有非 blocked 行都要求
timestamp GT。`open_world_real` 行不能被标记为 `pass`，避免把开放真实失败伪装成通过。
当前完成的是 row contract / evaluator，不表示已经采集到 controlled real tabletop 数据或
证明真实世界 ObjectState；下一步应把实际 small real / public rows 从 artifact 生成到该
gate，而不是推进 diffusion、replay buffer 或 viewer/export 默认模型。

随后完成 `OBJECTSTATE-REALITY-PUBLIC-ROWS-001`：新增
`objgauss.core.objectstate_reality_public_rows`，schema 为
`objgauss-objectstate-public-artifact-rows-v1`。它把当前已存在的 public / local viewer
artifacts 登记成第一批 reality gate rows：`real-sample-v2-sample-aware-lego`、
`polyhaven-chair`、`nike-real-splat-demo` 和 `plush`。每个 artifact 生成
`identity` / `prediction` / `intervention` 三条 blocked rows，共 12 rows，并嵌入
`OBJECTSTATE-REALITY-GATE-001` summary。当前 gate 结果按设计为 fail：
`real_or_public_rows_present=true`，但 identity / prediction / intervention pass rows 均不存在。
blocked 原因明确写入：现有 `object_id` 只是 renderer-facing address / candidate
assignment，不是 timestamped physical identity GT；现有 public artifacts 也缺 6DoF pose
tracks、history-vs-state future targets、action events 和 counterfactual outcomes。因此
这一步只把真实缺口显性化，不声明 ObjectState 已通过真实世界状态变量验证。

随后完成 `OBJECTSTATE-CONTROLLED-REAL-ROWS-001`：新增
`objgauss.core.objectstate_controlled_real_rows`，manifest schema 为
`objgauss-objectstate-controlled-real-manifest-v1`，summary schema 为
`objgauss-objectstate-controlled-real-rows-v1`。该 importer 定义真实 controlled tabletop
样本进入 reality gate 的最小 JSON 契约：`sample` 记录 `sample_id`、`source_kind=controlled_real`、
对象类别、场景、观察模态、artifact refs 和许可；`ground_truth` 记录 identity / pose /
action / timestamp GT 是否存在；`evidence_rows` 记录 identity / prediction /
intervention 的 `pass` / `fail` / `blocked` 行和指标。导入时不会创建 GT，非 blocked
row 会立即走 `ObjectStateRealityRow` 严格校验；测试 fixture 证明有 timestamped identity
GT 与 identity metrics 时，identity row 可以从 blocked 进入可评估 pass / fail，而
prediction / intervention 可继续保持 blocked。当前仍没有提交真实采集数据；下一步是用
实际 controlled tabletop capture / annotation 生成 manifest。

随后完成 `OBJECTSTATE-CONTROLLED-REAL-CLI-001`：`objgauss object-state
controlled-real-gate <manifest.json>` 已成为 controlled real manifest 的可复跑验收入口。
命令默认执行完整 reality gate，输出 sample / row count / gate status / hard blockers；
`--summary-output` 写入 `objgauss-objectstate-controlled-real-rows-v1` JSON，
`--blocked-rows-output` 写入 blocked rows Markdown。`--identity-only --require-pass`
支持 Stage 1 identity-state 验收：只要求 timestamped identity GT + identity metrics
形成 pass row，同时保留 prediction / intervention blocked rows，不把它们伪装成通过。
当前仍没有采集或提交真实 controlled tabletop 数据，也不训练 Gaussian / dynamics、不写
`outputs/` / `public/samples`、不改 viewer/export 默认。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-MANIFEST-001`：新增
`objgauss.core.objectstate_controlled_capture`，冻结 frame-level controlled tabletop
capture / annotation manifest schema
`objgauss-objectstate-controlled-capture-manifest-v1` 和 summary schema
`objgauss-objectstate-controlled-capture-summary-v1`。该 manifest 位于 reality rows
之前，用来记录真实采集的一手事实源：sample id、对象类别、场景、FPS、artifact refs、
license、declared physical objects、严格递增 timestamp 的 frames、RGB refs、可选
per-frame Gaussian refs、per-frame object annotations、6DoF pose 和 action events。
`objectstate_controlled_capture_summary(...)` 输出 identity / prediction /
intervention readiness、RGB / Gaussian coverage、GT availability 和 issues；
`objectstate_controlled_real_manifest_from_capture_manifest(...)` 可生成
`objgauss-objectstate-controlled-real-manifest-v1` seed，但三类 evidence rows 仍保持
`blocked`，直到候选模型指标被计算。CLI 新增
`objgauss object-state validate-controlled-capture <capture-manifest.json>`，支持写
summary JSON 和 controlled-real seed manifest。当前仍没有提交真实视频、图像、
Gaussian 或 GT 标注文件；这一步只是让真实采集结果进入 gate 前有可验证数据契约。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-IMPORT-001`：新增
`objgauss.core.objectstate_controlled_capture_import`，schema 为
`objgauss-objectstate-controlled-capture-import-v1`。新增 CLI
`objgauss object-state import-controlled-capture-bundle <bundle-root> --output <capture-manifest.json>`，
可从本地 controlled tabletop bundle 的 `sample.json`、`objects.csv`、`frames.csv`、
`annotations.csv` 和可选 `actions.csv` 生成
`objgauss-objectstate-controlled-capture-manifest-v1`，并同步输出 import summary
和 controlled-real blocked seed。CSV importer 支持 per-frame RGB / Gaussian refs、
timestamp、view / lighting / camera pose condition、per-object visibility /
occlusion、6DoF pose 和 action event metadata；partial pose 会 fail-fast，未知 frame
annotation 也会 fail-fast。该步骤只把已采集 / 已标注事实导入现有 manifest / audit /
handoff 链路，不采集视频、不创建 GT、不重建 Gaussian、不训练模型、不写
`public/samples`、不改 viewer/export 默认；当前仍缺实际 controlled tabletop 文件和
真实 candidate artifact 作为通过证据。完成 commit: `01ed42f`。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-BUNDLE-ACCEPTANCE-001`：新增
`objgauss-objectstate-controlled-capture-bundle-acceptance-v1`，把 controlled capture
bundle import 和 controlled capture file audit 合成一条 pre-identity-handoff gate。
新增 CLI `objgauss object-state accept-controlled-capture-bundle <bundle-root> --output <capture-manifest.json>`，
默认要求 imported manifest 达到 identity-stage ready，且 RGB / Gaussian frame refs
通过本地文件存在性与格式签名审计；可选再要求 prediction / intervention readiness。
CLI 可同时写出 acceptance summary、import summary、file audit、missing-files Markdown
和 controlled-real blocked seed。该步骤只验收本地 bundle 可进入 Stage 1 identity
handoff 前置流程，不运行 candidate identity handoff、不创建 GT、不重建 Gaussian、
不训练模型、不写 `public/samples`、不改 viewer/export 默认；当前仍缺实际 controlled
tabletop bundle 和真实 candidate artifact 作为通过证据。完成 commit: `cd0cf1b`。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-FILE-AUDIT-001`：新增
`objgauss.core.objectstate_controlled_capture_files`，schema 为
`objgauss-objectstate-controlled-capture-file-audit-v1`。它在 capture manifest
结构验证之外，检查 frame-level RGB / Gaussian refs 是否真的存在于本地 capture bundle；
RGB 文件始终要求存在，Gaussian 文件默认要求存在，也可用 `--no-require-gaussian-files`
只做 RGB-only staging。新增 CLI
`objgauss object-state audit-controlled-capture-files <capture>`，默认以 manifest
所在目录作为 `--root`，可写 `--summary-output` JSON 和 `--missing-files-output`
Markdown，并支持 `--check-artifact-refs` 与 `--require-pass`。这一步不读取图像像素、
不采集视频、不创建 GT、不重建 Gaussian、不训练模型、不写 `public/samples`；
它只防止只有 JSON 而没有真实文件的 capture bundle 进入后续 handoff。完成 commit:
`d6bd5db`。

随后补强 controlled capture file audit：frame-level RGB / Gaussian refs 现在默认必须是
非空 regular files，`file_counts` 增加 `valid` 计数，summary 增加完整
`file_records`，并可用 `--hash-files` 为有效 RGB / Gaussian frame 文件记录 SHA256；
CLI 同时支持 `--min-rgb-bytes` / `--min-gaussian-bytes` 调整最低字节数。sample-level
`artifact_refs` 仍只在 `--check-artifact-refs` 时检查存在性，目录可作为 bundle refs。
这一步只提升本地 bundle 证据完整性，不采集真实数据、不解析图像像素、不训练或发布模型。
完成 commit: `51fd551`。

随后继续补强 controlled capture file audit：frame-level RGB / Gaussian refs 默认还必须
通过文件签名审计。RGB 支持 PNG / JPEG / WebP / PPM；Gaussian 支持带 vertex element
的 PLY header 或 raw `.splat` 非零 32-byte 倍数文件。非空文本占位文件现在会导致
file audit fail，从而让 `controlled-identity-handoff` fail；`--no-require-frame-formats`
只作为显式 staging 降级。该检查只读取文件 header / signature，不解析图像像素，
不完整解析 Gaussian payload，也不证明真实采集或重建质量。当前仍没有实际 controlled
real capture 文件作为通过证据。完成 commit: `7b6e761`。

随后完成 `OBJECTSTATE-CONTROLLED-IDENTITY-EVAL-001`：新增
`objgauss.core.objectstate_controlled_identity_eval`，把 controlled capture GT 和候选
ObjectState / tracker identity predictions 连接起来，第一次让真实 Stage 1 identity row
可以从 `blocked` 进入 `pass` / `fail`。新增 prediction schema
`objgauss-objectstate-controlled-identity-predictions-v1` 和 eval summary schema
`objgauss-objectstate-controlled-identity-eval-v1`；候选预测按 `(frame_id, object_id)`
绑定 `predicted_identity`。Evaluator 会拒绝 sample / frame / object 不匹配，计算
`idf1`、`fragmentation_rate`、`swap_rate`、`identity_collapse`、track coverage 和
missing prediction count，并生成带 identity pass/fail row 的
`objgauss-objectstate-controlled-real-manifest-v1`。CLI 新增
`objgauss object-state eval-controlled-identity <capture.json> <predictions.json>`，支持
summary JSON、controlled-real manifest 输出、阈值参数和 `--require-pass`。当前仍没有
提交真实 capture / candidate prediction 文件；测试使用本地 fixture，只证明 evaluator
可以拒绝坏 identity tracks 并生成 pass/fail rows。

随后完成 `OBJECTSTATE-IDENTITY-PREDICTION-ADAPTER-001`：新增
`objgauss.core.objectstate_identity_prediction_adapter`，把已有
`objgauss-trainable-kernel-model-artifact-v1` 中的 per-frame `object_states`
转换为 `objgauss-objectstate-controlled-identity-predictions-v1`。新增 CLI
`objgauss object-state export-identity-predictions <capture> <objectstates> --output <predictions>`；
adapter 要求 controlled capture 里有
per-frame `pose.position`，用 nearest-centroid association 将真实标注物体关联到候选
ObjectState slot，并把稳定 slot address (`slot-<id>`) 作为
`predicted_identity`。这一步只打通候选模型输出 -> identity evaluator 的 handoff；
不采集真实数据、不创建 GT、不训练 Gaussian / dynamics、不计算 prediction /
intervention 指标、不改 viewer/export 默认。完成 commit: `cc644e8`。

随后完成 `OBJECTSTATE-CONTROLLED-IDENTITY-HANDOFF-001`：新增
`objgauss.core.objectstate_controlled_identity_handoff`，把 controlled capture manifest
+ trainable kernel ObjectState artifact 一次性跑完整 Stage 1 identity handoff。新增
schema `objgauss-objectstate-controlled-identity-handoff-v1` 和 CLI
`objgauss object-state controlled-identity-handoff <capture> <objectstates> --output-dir <dir>`。
该命令会写出 `identity-predictions.json`、`identity-eval-summary.json`、
`controlled-real.json`、`controlled-real-summary.json`、`blocked-rows.md` 和
`handoff-summary.json`；内部使用 identity-only reality gate，只要求 identity pass row，
并保留 prediction / intervention blocked rows。当前仍不采集真实数据、不创建 GT、
不训练 Gaussian / dynamics、不计算 prediction / intervention 指标、不写
`public/samples`、不改 viewer/export 默认。完成 commit: `47c2754`。

随后补强 `controlled-identity-handoff`：handoff 现在先执行 controlled capture file
audit，并把 `capture_file_audit` 嵌入 summary；handoff pass 条件要求 file audit、
identity eval 和 identity-only reality gate 三者都通过。CLI 默认用 capture manifest
所在目录作为 `--capture-root`，会额外写出 `capture-file-audit.json` 和
`capture-missing-files.md`，并支持 `--min-rgb-bytes`、`--min-gaussian-bytes`、
`--hash-files` 与 `--check-artifact-refs`。这一步防止 manifest-only 或空文件 bundle
获得 identity handoff pass；当前仍没有实际 controlled real capture / true candidate
artifact 作为通过证据。完成 commit: `2690196`。

随后继续补强 `controlled-identity-handoff` 的 candidate 侧证据：handoff 现在会对本地
trainable ObjectState artifact 文件执行 `candidate_artifact_file_audit`，要求它是非空
regular file，且可用 `--hash-candidate-artifact` 写 SHA256。CLI 额外写出
`candidate-artifact-file-audit.json`，并支持 `--min-candidate-artifact-bytes`。
handoff pass 现在同时要求 capture file audit、candidate artifact file audit、
identity eval 和 identity-only reality gate 全部通过；这一步仍不训练新模型、不创建
GT、不写 `public/samples`。完成 commit: `f8b37e4`。

随后进一步把 candidate file audit 和 prediction metadata 绑定：handoff summary 新增
`candidate_artifact_ref_match`，要求被审计的本地 candidate artifact 路径必须出现在
`identity_predictions.candidate.artifact_refs` 中。handoff pass 现在会拒绝“审计 A
文件、metadata 声明 B 文件”的不一致情况。CLI 输出新增
`candidate_artifact_ref_match=true|false`。当前仍没有实际 controlled real capture /
true candidate artifact 作为通过证据。完成 commit: `32d4a5d`。

随后补强 identity scenario challenge：`controlled-identity-handoff` 新增
`identity_scenario_audit`，要求 controlled capture manifest 至少包含一个对象的
clear-visible / occluded / clear-visible 轨迹；clear-visible 表示 `visible=true`
且 `occlusion_fraction` 低于阈值，避免全程可见、无遮挡挑战或全程高遮挡的 trivial
sequence 拿到 Stage 1 handoff pass。该 audit 只读取 manifest 的 `visible` 与
`occlusion_fraction`，不解析图像像素，也不声称验证光照变化或相机运动。CLI 会写出
`identity-scenario-audit.json`，并支持 `--min-identity-scenario-frames` 与
`--min-occlusion-fraction`。当前仍没有实际 controlled real capture / true candidate
artifact 作为通过证据。完成 commit: `c810b75`。

随后把 Real Identity Gate 的 view / lighting / camera motion 要求落到 handoff
可审计 metadata：controlled capture frame 现在可选记录 `condition.view_id`、
`condition.lighting_id` 和 `condition.camera_pose`；`controlled-identity-handoff`
默认要求至少 2 个 view condition、2 个 lighting condition，以及
`frame.condition.camera_pose` 最大平移至少 `0.01m`。缺少这些 declared condition
coverage 时，即使 file audit、candidate artifact、identity eval 和 identity-only
reality gate 都通过，Stage 1 handoff 也只能 fail。该 audit 仍只读取 manifest
metadata，不读取图像像素，也不证明实际光照变化或真实相机运动；当前仍没有实际
controlled real capture / true candidate artifact 作为通过证据。完成 commit:
`0368b0e`。

随后补强 controlled real identity quality gate：`eval-controlled-identity` /
`controlled-identity-handoff` 现在除 `idf1`、fragmentation、swap 和 collapse 外，
还输出并门禁 `track_retrieval_recall_at_1` 与 `long_term_drift_rate`。candidate
metadata 可携带 `identity_evidence`，其中必须显式声明
`reconstruction_noise_robustness`、`reconstruction_noise_variant_count` 和 source；
缺少该证据时，即使 identity track 稳定，Stage 1 identity eval / handoff 也会 fail。
`objectstate_identity_prediction_adapter` 会把 trainable artifact 里的
`identity_evidence` 传入 predictions。该切片只强化真实 identity gate 的证据契约，
不采集真实数据、不训练新 Gaussian / dynamics 模型、不写 `public/samples`、不改
viewer/export 默认；当前仍缺实际 controlled tabletop capture 和真实 candidate
artifact 作为通过证据。完成 commit: `a9b0007`。

随后完成 `OBJECTSTATE-CONTROLLED-IDENTITY-BUNDLE-HANDOFF-001`：新增
`objgauss.core.objectstate_controlled_identity_bundle_handoff`，schema 为
`objgauss-objectstate-controlled-identity-bundle-handoff-v1`，并新增 CLI
`objgauss object-state controlled-identity-bundle-handoff <bundle-root> <objectstates> --output-dir <dir>`。
该入口从真实 controlled tabletop bundle 根目录开始，先导入 `sample.json`、
`objects.csv`、`frames.csv`、`annotations.csv` 和可选 `actions.csv`，再执行 bundle
acceptance / file audit，最后复用现有 `controlled-identity-handoff` 生成 predictions、
identity eval、controlled-real manifest、identity-only gate summary 和 blocked rows。
顶层 pass 同时要求 bundle acceptance pass 和 identity handoff pass；缺 RGB / Gaussian
真实文件、缺 candidate artifact、artifact ref mismatch、缺 view / lighting /
camera-motion metadata、缺 reconstruction-noise evidence 或 identity gate fail 都不会被
顶层 summary 隐藏。CLI 会写出 `capture-manifest.json`、
`bundle-acceptance-summary.json`、`bundle-import-summary.json`、`bundle-file-audit.json`、
`bundle-missing-files.md`、`controlled-real-seed.json` 以及 handoff 的所有 identity
输出。该切片仍不采集视频、不创建 GT、不重建 Gaussian、不训练新模型、不写
`public/samples`、不改 viewer/export 默认；它只是把下一次真实采集目录 + candidate
artifact 的 Stage 1 identity 验收收敛成一条可复跑命令。完成 commit: `67b45e0`。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-TEMPLATE-001`：新增
`objgauss.core.objectstate_controlled_capture_template`，schema 为
`objgauss-objectstate-controlled-capture-bundle-template-v1`，并新增 CLI
`objgauss object-state init-controlled-capture-bundle <bundle-root> --sample-id <id>`。
该命令为真实 controlled tabletop 采集生成本地 skeleton：`sample.json`、
`objects.csv`、`frames.csv`、`annotations.csv`、`actions.csv`、`README.md`、
`rgb/` 和 `gaussians/`。默认只写 CSV headers，不写 frame / annotation / action
数据行；可用 `--object object_id:category[:label]` 预写 physical object declarations。
模板 README 记录最小 Stage 1 identity 场景要求和后续 import / acceptance /
`controlled-identity-bundle-handoff` 命令。该切片不采集视频、不创建 GT、不生成伪造
RGB / Gaussian / pose / action row、不重建 Gaussian、不训练模型、不写 `public/samples`、
不改 viewer/export 默认；它只把真实采集目录的文件布局和下一步验收命令固定下来。
完成 commit: `873aec1`。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-FRAMES-001`：新增
`objgauss.core.objectstate_controlled_capture_frames`，schema 为
`objgauss-objectstate-controlled-capture-frames-v1`，并新增 CLI
`objgauss object-state populate-controlled-capture-frames <bundle-root>`。
该命令扫描 controlled capture bundle 中已存在的 `rgb/` 和 `gaussians/` 文件，按同名
stem 配对，写出 timestamped `frames.csv` rows；默认要求每个 RGB frame 都有同名
`.ply` 或 `.splat` Gaussian evidence，并且只覆盖空的 `frames.csv`，非空表需要显式
`--force`。`init-controlled-capture-bundle` 生成的 next commands / README 和
controlled real capture runbook 已补入该步骤。该切片只消除真实文件到 frame rows 的
authoring blocker，不采集视频、不创建 pose / action GT、不生成 annotation / action rows、
不重建 Gaussian、不运行 identity / prediction / intervention gate、不声明 metric pass 或
world model。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-ANNOTATIONS-001`：新增
`objgauss.core.objectstate_controlled_capture_annotations`，schema 为
`objgauss-objectstate-controlled-capture-annotation-template-v1` 和
`objgauss-objectstate-controlled-capture-annotation-finalize-v1`，并新增 CLI
`objgauss object-state init-controlled-capture-annotations <bundle-root>` /
`finalize-controlled-capture-annotations <bundle-root>`。第一步从已有 `frames.csv` 和
`objects.csv` 写出 draft-only `annotations.template.csv`，第二步只在人工或外部填写的
visible、occlusion 和完整 6DoF pose 没有空值 / TODO、frame/object binding 合法且默认覆盖
所有 frame/object pairs 时，才写正式 `annotations.csv`。该切片减少 pose GT 标注表进入
controlled capture bundle 的路径漂移，但不推断 pose、不创建 GT、不写 action rows、不运行
handoff/eval、不创建 pass row、不声明 ObjectState 已通过真实世界状态变量验证。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-ACTIONS-001`：新增
`objgauss.core.objectstate_controlled_capture_actions`，schema 为
`objgauss-objectstate-controlled-capture-action-template-v1` 和
`objgauss-objectstate-controlled-capture-action-finalize-v1`，并新增 CLI
`objgauss object-state init-controlled-capture-actions <bundle-root>` /
`finalize-controlled-capture-actions <bundle-root>`。第一步从已有 `frames.csv` 和
`objects.csv` 写出 draft-only `actions.template.csv`，第二步只在人工或外部填写的
action id / type / object binding / time interval / vector 没有空值或 TODO、object /
target refs 合法、动作区间覆盖至少一个 frame timestamp，且默认 action vector 非零时，
才写正式 `actions.csv`。可用 `--require-frame-action-refs` 要求每个 `action_id` 已被
`frames.csv` 引用。该切片减少 intervention GT action 表进入 controlled capture bundle
的路径漂移，但不推断动作、不创建 GT、不写 annotation rows、不运行 handoff/eval、不创建
pass row、不声明 counterfactual proof 或 ObjectState 已通过真实世界状态变量验证。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-READINESS-001`：新增
`objgauss.core.objectstate_controlled_capture_bundle_readiness`，schema 为
`objgauss-objectstate-controlled-capture-bundle-readiness-v1`，并新增 CLI
`objgauss object-state audit-controlled-capture-bundle-readiness <bundle-root>`。
该 audit 可在 skeleton 或半填充 bundle 上运行，不再要求先通过 import；它会检查
layout、`sample.json` metadata、CSV headers、object / frame / annotation rows、
frame-action-object 引用、pose columns、真实 RGB / Gaussian 文件、identity scenario
metadata，以及可选 candidate artifact。输出区分 `capture_bundle_ready` 和
`identity_bundle_handoff_ready`：前者只代表真实采集 bundle 足够进入 Stage 1 identity
验收，后者在 `--require-candidate-artifact` 时还要求 candidate artifact 文件存在。
空 skeleton 会返回 hard blockers 和 next actions，不会被误报为 ready。该切片仍不采集
视频、不创建 GT、不生成伪造行、不重建 Gaussian、不训练模型、不运行 handoff、不写
`public/samples`、不改 viewer/export 默认。完成 commit: `63c384f`。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-INTERVENTION-READINESS-001`：
扩展 `audit-controlled-capture-bundle-readiness` 的 intervention preflight。此前
`--require-intervention-ready` 主要依赖 capture summary 的 `action_count > 0`；现在新增
`intervention_action_gt_ready` 和 `intervention_action_gt` summary，要求至少一个 action
具备非零 vector，并且 action interval 能完整落在被引用对象的连续 pose transition 区间内。
CLI 会打印 `intervention_action_gt_ready`，hard blockers 会暴露缺失 / 零向量或无可用
action transition 的具体原因。该切片只收紧 bundle readiness，不改 capture manifest
schema、不运行 intervention eval、不创建 GT、不训练 dynamics、不生成 pass row，也不声明
counterfactual proof 或 world model。

随后完成 `OBJECTSTATE-CONTROLLED-CAPTURE-RUNBOOK-001`：新增
`docs/training/controlled-real-capture-runbook.md`，把 Phase 1 controlled real
capture 的最小桌面场景、帧要求、文件命名、CSV 填写、pose / action 记录、
candidate artifact `identity_evidence` 要求、readiness / acceptance /
`controlled-identity-bundle-handoff` 命令链和验收证据清单固定成可复跑规程。
`init-controlled-capture-bundle` 生成的 bundle README 现在会指向该 runbook，并先建议
运行 `audit-controlled-capture-bundle-readiness`。该切片仍不采集视频、不创建 GT、
不生成 RGB / Gaussian 文件、不训练 candidate、不运行 handoff、不写 `public/samples`、
不改 viewer/export 默认；它只把真实采集执行口径从聊天和模板提示收敛到仓库事实源。
完成 commit: `ecae41d`。

随后完成 `OBJECTSTATE-CONTROLLED-PREDICTION-EVAL-001`：新增
`objgauss.core.objectstate_controlled_prediction_eval`，schema 为
`objgauss-objectstate-controlled-prediction-candidates-v1` 和
`objgauss-objectstate-controlled-prediction-eval-v1`，并新增 CLI
`objgauss object-state eval-controlled-prediction <capture.json> <predictions.json>`。
该 evaluator 要求 controlled capture manifest 已具备 timestamped 6DoF pose GT，
候选预测显式提供 `(source_frame_id, target_frame_id, object_id)`、`predicted_position`
和 `history_baseline_position`；它计算 `state_ade`、`history_ade`、
`prediction_gap_vs_history_model`、error ratio 和 horizon 统计，并把 controlled-real
manifest 的 prediction row 从 blocked 推进为 pass / fail。该切片仍不采集视频、
不创建 GT、不运行预测 / dynamics 模型、不训练 Gaussian 或 dynamics、不计算 identity /
intervention 指标、不写 `public/samples`、不改 viewer/export 默认。完成 commit: `354920a`。

随后完成 `OBJECTSTATE-CONTROLLED-INTERVENTION-EVAL-001`：新增
`objgauss.core.objectstate_controlled_intervention_eval`，schema 为
`objgauss-objectstate-controlled-intervention-candidates-v1` 和
`objgauss-objectstate-controlled-intervention-eval-v1`，并新增 CLI
`objgauss object-state eval-controlled-intervention <capture.json> <interventions.json>`。
该 evaluator 要求 controlled capture manifest 已具备 timestamped 6DoF pose GT 和
action GT，候选 intervention 显式提供
`(source_frame_id, target_frame_id, object_id, action_id)`、
`action_conditioned_position` 和 `no_action_baseline_position`；action 必须提供非零
vector，才能计算 `wrong_direction_rate`。它计算 `action_conditioned_ade`、
`no_action_ade`、`intervention_gain`、`counterfactual_outcome_accuracy`、
`wrong_direction_rate` 和 horizon 统计，并把 controlled-real manifest 的 intervention
row 从 blocked 推进为 pass / fail。该切片仍不采集视频、不创建 GT、不运行
action-conditioned / dynamics 模型、不训练 Gaussian 或 dynamics、不计算 identity /
prediction 指标、不写 `public/samples`、不改 viewer/export 默认。完成 commit: `26cf3ec`。

随后完成 `OBJECTSTATE-CONTROLLED-REALITY-BUNDLE-HANDOFF-001`：新增
`objgauss.core.objectstate_controlled_reality_bundle_handoff`，schema 为
`objgauss-objectstate-controlled-reality-bundle-handoff-v1`，并新增 CLI
`objgauss object-state controlled-reality-bundle-handoff <bundle-root> <objectstates.json> <prediction-candidates.json> <intervention-candidates.json> --output-dir <dir>`。
该 handoff 复用现有 controlled identity bundle handoff 完成 bundle import、file audit、
candidate artifact audit、identity scenario audit、identity predictions 和 identity eval；
随后在同一个 imported capture manifest 上运行 controlled prediction eval 和 controlled
intervention eval，并把 identity / prediction / intervention 三条 row 合并成最终
controlled-real manifest。最终 reality gate 默认要求 3 条 controlled real rows 且
identity、prediction、intervention 都是 pass。该切片仍不采集视频、不创建 GT、
不创建 prediction / intervention candidates、不运行预测或动作模型、不训练 Gaussian /
dynamics、不写 `public/samples`、不改 viewer/export 默认。完成 commit: `6dd84df`。

随后完成 `OBJECTSTATE-CONTROLLED-REALITY-BUNDLE-READINESS-001`：新增
`objgauss.core.objectstate_controlled_reality_bundle_readiness`，schema 为
`objgauss-objectstate-controlled-reality-bundle-readiness-v1`，并新增 CLI
`objgauss object-state audit-controlled-reality-bundle-readiness <bundle-root> <objectstates.json> <prediction-candidates.json> <intervention-candidates.json>`。
该 audit 复用 capture bundle readiness，并额外检查 trainable ObjectState artifact schema /
identity binding、prediction candidates schema / capture binding 和 intervention
candidates schema / action binding。`full_reality_handoff_ready=true` 只表示输入结构足够运行
`controlled-reality-bundle-handoff`，不表示 identity / prediction / intervention 指标会 pass。
该切片仍不采集视频、不创建 GT、不生成候选 prediction / intervention、不运行 handoff /
eval、不训练 Gaussian 或 dynamics、不写 `public/samples`、不改 viewer/export 默认。
完成 commit: `c3b22ef`。

随后完成 `OBJECTSTATE-CONTROLLED-REALITY-CANDIDATE-TEMPLATE-001`：新增
`objgauss.core.objectstate_controlled_reality_candidate_template`，schema 为
`objgauss-objectstate-controlled-reality-candidate-template-v1`，并新增 CLI
`objgauss object-state init-controlled-reality-candidates <bundle-root> --output-dir <dir>`。
该命令从已填写的 controlled capture bundle 导入 manifest，枚举 pose-backed
prediction draft rows 和 action-bracketed intervention draft rows，写出
`prediction-candidates.template.json`、`intervention-candidates.template.json`
和本地 README。两个模板使用独立 draft schema：
`objgauss-objectstate-controlled-prediction-candidates-template-v1` 和
`objgauss-objectstate-controlled-intervention-candidates-template-v1`，并保持
`template_status=draft_not_valid_for_eval`；正式 evaluator 会拒绝这些模板 schema，防止
TODO 模板被误当作真实 pass evidence。模板不写 target pose values，不运行预测 /
动作模型，不训练 Gaussian 或 dynamics，不创建 GT，不写 `public/samples`、不改
viewer/export 默认。当前仍缺实际 controlled tabletop capture 文件和真实
objectstates / prediction / intervention candidate JSON 作为通过证据。完成 commit:
`443344d`。

随后完成 `OBJECTSTATE-CONTROLLED-REALITY-CANDIDATE-FINALIZE-001`：同一模块新增
schema `objgauss-objectstate-controlled-reality-candidate-finalize-v1` 和 CLI
`objgauss object-state finalize-controlled-reality-candidates <prediction-template.json> <intervention-template.json> --output-dir <dir>`。
该命令只接受已填完的 draft templates：candidate metadata 不能还是 TODO，所有必需
position 字段必须是 numeric length-3 vectors，且 row 顶层不能出现
`target_position` / `target_pose` 等明显 GT 泄漏字段。通过后写出正式 evaluator
输入 `prediction-candidates.json` 和 `intervention-candidates.json`，并立即用现有
prediction / intervention candidate validators 校验。该切片仍不运行预测 / 动作模型，
不创建 GT、不训练 Gaussian 或 dynamics、不评估指标、不声明 pass rows、不写
`public/samples`、不改 viewer/export 默认；它只把外部真实 candidate 输出从模板转成
full readiness / handoff 可消费的 JSON。当前仍缺实际 controlled tabletop capture 文件和
真实 objectstates / prediction / intervention candidate 内容作为通过证据。完成 commit:
`5a10915`。

随后完成 `OBJECTSTATE-CONTROLLED-REALITY-CANDIDATE-WORKFLOW-001`：candidate template
生成的 README 和 summary `next_commands` 现在会先指向
`finalize-controlled-reality-candidates`，再进入
`audit-controlled-reality-bundle-readiness` 和 `controlled-reality-bundle-handoff`，
避免作者手工 rename / 改 schema。`docs/training/controlled-real-capture-runbook.md`
也从 identity-only handoff 扩展为完整 Phase 1 链路：初始化 candidate templates、填入外部
model / baseline 输出、finalize 成正式 evaluator JSON、跑 full readiness、再跑 full
reality handoff。该切片仍不采集视频、不创建 GT、不运行预测 / 动作模型、不训练
Gaussian 或 dynamics、不评估指标、不声明 pass rows、不写 `public/samples`、不改
viewer/export 默认。完成 commit: `f064801`。

随后完成 `OBJECTSTATE-CONTROLLED-REALITY-EVIDENCE-PACKAGE-001`：新增
`objgauss.core.objectstate_controlled_reality_evidence_package`，schema 为
`objgauss-objectstate-controlled-reality-evidence-package-v1`，并新增 CLI
`objgauss object-state audit-controlled-reality-evidence-package <package-root>`。
该 read-only audit 检查 full Phase 1 本地证据包：candidate template summary、
finalize summary、full readiness summary、prediction / intervention candidates、
reality handoff summary、prediction / intervention eval summary、controlled-real
summary 和 blocked rows Markdown 是否存在、schema 可验证、`sample_id` 一致，且
standalone eval / controlled-real outputs 与 handoff summary 嵌入内容一致。
reviewable 只表示文件包可审查，不表示 metric pass；fail rows 仍应作为真实负证据保留。
本切片同时修正 controlled identity / reality handoff validators 的 JSON roundtrip 比较，
避免 CLI 写盘后 tuple/list 差异导致误报 mismatch。当前仍没有采集或提交真实 controlled
tabletop RGB / Gaussian / pose / action 文件，也没有真实 candidate artifact /
prediction / intervention 输出；不训练 Gaussian 或 dynamics，不声明 ObjectState 已通过真实
世界状态变量验证，不写 `public/samples`、不改 viewer/export 默认。

随后创建本地 ignored capture skeleton
`outputs/captures/controlled-tabletop-cup-box-001/`，用于下一次真实桌面采集。命令为
`uv run objgauss object-state init-controlled-capture-bundle ... --sample-id controlled-tabletop-cup-box-001`
并输出 `template-summary.json`、`sample.json`、`objects.csv`、空的 `frames.csv` /
`annotations.csv` / `actions.csv`、`rgb/` 和 `gaussians/`。随后运行
`uv run objgauss object-state audit-controlled-capture-bundle-readiness outputs/captures/controlled-tabletop-cup-box-001 --summary-output outputs/captures/controlled-tabletop-cup-box-001/readiness-summary.json --require-prediction-ready --require-intervention-ready`，
结果为 `objectstate_controlled_capture_bundle_readiness_blocked`。当前硬 blocker 为：
缺 timestamped frame rows、缺 frame/object pose rows、identity / prediction /
intervention stage 均未 ready、capture file audit 不能运行、capture manifest 还不能导入。
下一步仍必须放入真实 RGB / Gaussian 文件、填写真实 timestamp / view / lighting /
camera pose / object pose / action rows，并生成真实 ObjectState candidate artifact；该本地
skeleton 不进入 git，也不构成真实 Phase 1 通过证据。

随后新增 `OBJECTSTATE-CONTROLLED-CAPTURE-ENVIRONMENT-001`：新增
`objgauss.core.objectstate_controlled_capture_environment`，schema 为
`objgauss-objectstate-controlled-capture-environment-v1`，并新增 CLI
`objgauss object-state audit-controlled-capture-environment`。该 preflight 检查当前 host
是否可见 `/dev/video*` / `/dev/media*`，是否有 RGB capture 工具 `ffmpeg` 或 Python
`cv2`，以及 Gaussian 重建所需 `colmap`、`ns-process-data`、`ns-train` 和 `ns-export`。
本地运行
`uv run objgauss object-state audit-controlled-capture-environment --summary-output outputs/captures/controlled-tabletop-cup-box-001/environment-summary.json`
结果为 `objectstate_controlled_capture_environment_blocked`：`video_devices=0`，
`ffmpeg=false`、`cv2=false`、`colmap=false`、`ns-process-data=false`、`ns-train=false`、
`ns-export=false`。这说明当前会话环境不能直接采集 / 重建真实 controlled tabletop
数据；需要在物理采集主机上暴露摄像头和采集 / 重建工具后重跑。该 preflight 只记录环境
blocker，不采集视频、不创建 GT、不重建 Gaussian、不训练模型、不声明 reality gate pass。

随后新增 `OBJECTSTATE-PUBLIC-DATASET-CANDIDATES-001`：在当前 capture host 不可用时，
新增 public pose / interaction dataset candidate audit 作为 Phase 1 替代数据路线前置
审计。`objgauss.core.objectstate_public_dataset_candidates` 冻结
`objgauss-objectstate-public-dataset-candidates-v1` summary schema，CLI
`objgauss object-state audit-public-dataset-candidates` 可写 JSON / Markdown 审计输出。
当前排序为 `bop-ycbv-keyframes` -> `bop-hopev2` -> `bop-tudl` -> `hot3d-clips` ->
`dexycb`；建议第一步用一个小型 BOP YCB-V subset 做 controlled capture manifest
adapter，再在 ignored `outputs/` 下生成 per-frame Gaussian evidence，之后才生成
identity / prediction rows。HOT3D 只作为后续 action-like interaction candidate；
所有 public candidates 当前仍报告 `has_direct_phase1_ready_dataset=false` 和
`has_direct_gaussian_evidence=false`，因此不能声明 reality gate pass、public demo 或
world model。

随后新增 `OBJECTSTATE-BOP-CAPTURE-ADAPTER-001`：新增
`objgauss.core.objectstate_bop_capture_adapter`，schema 为
`objgauss-objectstate-bop-capture-adapter-v1`，并新增 CLI
`objgauss object-state import-bop-capture-scene`。该 adapter 读取本地 BOP scene 的
`scene_camera.json`、`scene_gt.json`、可选 `scene_gt_info.json` 和 `rgb/` 帧文件，
将 `cam_R_m2c` / `cam_t_m2c` 转成 ObjGauss controlled capture 6DoF pose，将
`visib_fract` 转成 `occlusion_fraction`，并输出 capture manifest、adapter summary 和
controlled-real blocked seed。当前 identity policy 为
`single_instance_per_bop_obj_id`，若同一选中帧内出现重复 `obj_id` 会 fail-fast，避免把不稳定
instance tracking 当作 physical identity GT。该 adapter 只推进 BOP YCB-V 小子集进入
manifest 的第一步；仍不下载数据、不重建 Gaussian、不训练模型、不创建 pass rows、不声明
reality gate pass 或 public demo。

随后新增 `OBJECTSTATE-BOP-CAPTURE-ACCEPTANCE-001`：在 BOP adapter 后接入既有
controlled capture file audit，新增 CLI
`objgauss object-state accept-bop-capture-scene`。该命令可一次性输出 capture
manifest、acceptance summary、file audit、missing-files Markdown 和 controlled-real
blocked seed。默认可验证 RGB / pose BOP scene 文件；使用 `--require-gaussian-files`
时会把 expected `gaussians/<frame>.ply` refs 变成硬要求，只有真实 per-frame Gaussian
文件存在且格式通过时 `phase1_gaussian_evidence_ready=true`。当前仍不下载 BOP 数据、不生成
Gaussian、不训练模型、不评分 ObjectState candidates、不创建 pass rows。

随后新增 `OBJECTSTATE-BOP-PREDICTION-CANDIDATE-HANDOFF-001`：`objgauss.core.objectstate_controlled_reality_candidate_template`
现在支持 manifest-first candidate authoring。新增
`write_objectstate_controlled_reality_candidate_templates_from_manifest(...)` 和 CLI
`objgauss object-state init-controlled-reality-candidates-from-manifest <capture-manifest.json>`，
可从 BOP acceptance 输出的 controlled capture manifest 直接生成
`prediction-candidates.template.json` / `intervention-candidates.template.json` 和 README。
对于无 action 的 BOP pose scene，prediction draft rows 会生成，intervention draft rows
保持 0 并记录 issue。新增
`finalize_objectstate_controlled_prediction_candidate_template(...)` 和 CLI
`objgauss object-state finalize-controlled-prediction-candidates`，可把已填写的
prediction template 转成 evaluator-ready `prediction-candidates.json`，再交给
`eval-controlled-prediction` 形成真实 prediction pass / fail row。该切片不创建 GT、不运行
prediction model、不重建 Gaussian、不训练模型、不创建 pass row、不声明 intervention /
counterfactual gate 或 world model。

随后新增 `OBJECTSTATE-CONTROLLED-PREDICTION-EVIDENCE-PACKAGE-001`：新增
`objgauss.core.objectstate_controlled_prediction_evidence_package`，schema 为
`objgauss-objectstate-controlled-prediction-evidence-package-v1`，并新增 CLI
`objgauss object-state audit-controlled-prediction-evidence-package`。该 audit 面向
BOP / manifest-first prediction-only 证据包，默认检查 `capture-manifest.json`、
`bop-acceptance-summary.json`、`bop-file-audit.json`、`bop-missing-files.md`、
`reality-candidates/template-summary.json`、`prediction-finalize-summary.json`、
`prediction-candidates.json`、`prediction-eval-summary.json` 和
`controlled-real-prediction.json`。reviewable 要求 BOP acceptance 已有
`phase1_gaussian_evidence_ready=true`、所有 JSON schema 有效、`sample_id` 一致、
prediction row 为 pass 或 fail，且 controlled-real prediction manifest 与 prediction eval
内嵌 manifest 一致；reviewable 不要求 prediction 指标 pass，也不声明 intervention /
counterfactual gate。

随后新增 `OBJECTSTATE-CONTROLLED-IDENTITY-EVIDENCE-PACKAGE-001`：新增
`objgauss.core.objectstate_controlled_identity_evidence_package`，schema 为
`objgauss-objectstate-controlled-identity-evidence-package-v1`，并新增 CLI
`objgauss object-state audit-controlled-identity-evidence-package`。该 audit 面向
`controlled-identity-handoff` 之后的 identity-only Stage 1 本地证据包，默认检查
`capture-manifest.json`、`capture-file-audit.json`、`capture-missing-files.md`、
`candidate-artifact-file-audit.json`、`identity-scenario-audit.json`、
`identity-predictions.json`、`identity-eval-summary.json`、`controlled-real.json`、
`controlled-real-summary.json`、`blocked-rows.md` 和 `handoff-summary.json`。
reviewable 要求 capture / candidate artifact file audit 通过、candidate artifact ref
match、identity scenario audit 通过、identity row 为 pass 或 fail，且 standalone outputs
与 handoff summary 内嵌结果一致；reviewable 不要求 identity 指标 pass，也不声明
prediction、intervention、counterfactual gate 或 world model。
随后补强 handoff 集成：`controlled-identity-handoff` 现在会把输入
`capture-manifest.json` 复制到 output dir，并在写完 handoff artifacts 后自动写出
`identity-evidence-package-summary.json`；`controlled-identity-bundle-handoff` 也会在
同一 output dir 自动写出该 summary。该 summary 只是 read-only evidence package audit
结果，不重新运行 identity eval，不改变 handoff pass / fail 判定，也不声明 prediction /
intervention 或 world model。

随后新增 `OBJECTSTATE-PHASE1-EVIDENCE-LEDGER-001`：新增
`objgauss.core.objectstate_phase1_evidence_ledger`，schema 为
`objgauss-objectstate-phase1-evidence-ledger-v1`，并新增 CLI
`objgauss object-state audit-phase1-evidence-ledger`。该 ledger 只读取已有
identity / prediction / full reality evidence package summary JSON，复用各自 validator，
输出 maturity（`identity_reviewable`、`prediction_reviewable`、
`identity_prediction_reviewable`、`full_reality_reviewable` 等）、sample scope、阶段
row count 和 issues。它不运行 handoff 或 eval，不创建 GT，不生成 pass rows，也不声明
world model。
随后补强 ledger 发现能力：CLI 现在支持 `--discover-root` / `--max-depth`，可在
controlled capture 或 public dataset handoff root 下自动发现标准文件名
`identity-evidence-package-summary.json`、`prediction-evidence-package-summary.json`、
`evidence-package-summary.json` 和 `reality-evidence-package-summary.json`，再与显式
summary 路径合并去重。缺失或非法 root 会记录为 discovery issue 和 ledger gate failure。

随后新增 `OBJECTSTATE-CONTROLLED-PREDICTION-BASELINE-CANDIDATES-001`：新增
`objgauss.core.objectstate_controlled_prediction_baseline`，schema 为
`objgauss-objectstate-controlled-prediction-baseline-candidates-v1`，并新增 CLI
`objgauss object-state generate-controlled-prediction-baseline-candidates`。该命令读取
controlled capture manifest 和 draft `prediction-candidates.template.json`，用
`hold` 或 `constant_velocity` policy 生成
`prediction-candidates.baseline-filled.template.json`，再复用现有 prediction finalizer
写出 evaluator-ready `prediction-candidates.json` 和
`prediction-finalize-summary.json`。baseline 只使用 source pose、prior pose history 和
target timestamp，不读取 target pose values；它不是 learned dynamics model，不运行 eval，
不创建 GT、不训练 Gaussian / dynamics、不创建 pass row、不改 viewer/export 默认策略。

随后新增 `OBJECTSTATE-BOP-PREDICTION-BASELINE-HANDOFF-001`：新增
`objgauss.core.objectstate_bop_prediction_baseline_handoff`，schema 为
`objgauss-objectstate-bop-prediction-baseline-handoff-v1`，并新增 CLI
`objgauss object-state bop-prediction-baseline-handoff`。该命令对本地 BOP scene 执行
`--require-gaussian-files` acceptance，写出 `capture-manifest.json`、BOP acceptance /
file audit / missing files、manifest-first candidate templates、baseline-filled
prediction candidates、prediction eval summary、`controlled-real-prediction.json`
和 prediction evidence package audit summary，并同步写出
`phase1-evidence-ledger.json`。该 ledger 会把 BOP prediction-only evidence 记录为
`prediction_reviewable` maturity。它把 BOP prediction-only 路线从多条手工命令收敛为一个
可复验本地 handoff；仍不下载 BOP 数据、不生成 Gaussian evidence、不训练或发布新模型、
不创建 GT、不声明 intervention / counterfactual gate、不改 viewer/export 默认。
随后新增 BOP route 只读审计：`objgauss.core.objectstate_bop_phase1_route_audit`，
schema 为 `objgauss-objectstate-bop-phase1-route-audit-v1`，CLI 为
`objgauss object-state audit-bop-phase1-route`。该命令对本地 BOP scene 运行内存态
acceptance（要求 per-frame Gaussian files），并检查 output root 下已有
`prediction-evidence-package-summary.json` 和 `phase1-evidence-ledger.json`，输出 route
状态：blocked、handoff-ready 或 prediction-reviewable。它不运行 handoff / eval，不下载
数据，不生成 Gaussian，不创建 GT，不训练模型，也不声明 identity、intervention 或 world
model。
随后新增 BOP identity route 只读审计：
`objgauss.core.objectstate_bop_identity_route_audit`，schema 为
`objgauss-objectstate-bop-identity-route-audit-v1`，CLI 为
`objgauss object-state audit-bop-identity-route`。该命令对本地 BOP scene 运行内存态
acceptance（要求 per-frame Gaussian files），并检查 trainable ObjectState candidate
artifact 是否存在、schema 是否有效、`object_states` frame index 是否与 accepted BOP
frames 一一绑定；同时检查已有 `identity-evidence-package-summary.json` 和
`phase1-evidence-ledger.json` 是否已经提供 reviewable identity evidence。额外的
identity scenario metadata audit 要求 frame count、occlusion reappearance、cross-view、
lighting variation 和 camera-pose motion。默认 BOP adapter 往往有 pose / visibility
序列但缺 lighting / camera_pose 条件，因此 route 可以在 RGB / pose / Gaussian /
candidate artifact 都就绪时仍 blocked；这是 Stage 1 identity-state 门禁的预期行为，
不应通过放松 identity gate 来声明真实 ObjectState 状态变量证据。该命令不运行 identity
handoff / eval，不下载数据，不生成 Gaussian，不创建 GT，不训练模型，也不声明
prediction、intervention 或 world model。
随后新增 BOP Phase 1 local row 组合只读审计：
`objgauss.core.objectstate_bop_phase1_local_row_readiness`，schema 为
`objgauss-objectstate-bop-phase1-local-row-readiness-v1`，CLI 为
`objgauss object-state audit-bop-phase1-local-row`。该命令在同一 scene / output root /
sample id 上同时运行 identity route audit 和 prediction route audit，并输出一个
`blocking_stage`：可区分 local BOP scene 缺失、BOP acceptance 失败、
per-frame Gaussian evidence 缺失、candidate ObjectState artifact 缺失 / 未绑定、
identity scenario metadata 不足、handoff-ready 或已有 reviewable identity /
prediction evidence。它只做 route accounting，不运行 handoff / eval、不下载数据、
不重建 Gaussian、不训练模型，也不声明 intervention / counterfactual / world model。
随后新增 `OBJECTSTATE-BOP-CONDITION-SIDECAR-001`：BOP adapter / acceptance /
identity route / prediction route / combined local-row readiness 和 prediction baseline
handoff 现在都支持 `--condition-sidecar`。sidecar schema 为
`objgauss-objectstate-bop-capture-condition-sidecar-v1`，只允许为选中 frame 显式覆盖
`condition.view_id`、`condition.lighting_id` 和 `condition.camera_pose`。这让本地 BOP /
public pose subset 在已有 RGB / pose / per-frame Gaussian / ObjectState candidate
artifact 的前提下，能够满足 Stage 1 identity scenario metadata gate，而不是因为默认
adapter 缺 lighting / camera motion 元数据而永久 blocked。该 sidecar 不创建 identity GT、
不改 BOP pose GT、不生成 Gaussian、不训练或发布新模型、不声明 prediction /
intervention / world model，也不改变 viewer/export 默认策略。
随后补齐 `OBJECTSTATE-BOP-CONDITION-SIDECAR-AUTHORING-001`：新增
`objgauss object-state init-bop-condition-sidecar` CLI 和
`objgauss-objectstate-bop-capture-condition-sidecar-summary-v1` summary。该命令读取本地
BOP scene 的选中 frame，并可选消费 `bop-conditions.csv`（`frame_id`、view /
lighting、camera pose columns），写出符合
`objgauss-objectstate-bop-capture-condition-sidecar-v1` 的 sidecar，同时报告 view /
lighting condition count、camera pose count、最大 camera translation 和
`identity_scenario_metadata_ready`。无 CSV 时会写出默认模板但通常保持
`needs_metadata`；带齐 lighting variation 和 camera motion 的 CSV 可让 sidecar authoring
达到 `identity_ready`。该步骤仍不下载 BOP、不创建 GT、不生成 Gaussian、不训练模型、不运行
handoff，也不声明 pass row 或 world model。
随后补齐 `OBJECTSTATE-BOP-CONDITION-CSV-TEMPLATE-001`：同一个 summary 现在会生成
`condition_csv_template` rows，CLI 新增 `--condition-csv-template-output`，可把选中 BOP
frame id、默认 view / lighting metadata 和已有 camera pose metadata 写成可填写 CSV。
推荐流程变为先导出 `bop-conditions.template.csv`，人工填入真实 capture condition，再 rerun
`init-bop-condition-sidecar --condition-csv ... --require-identity-ready`。这只是 authoring
aid，不从 pixels 推断条件、不创建 identity / pose GT、不生成 Gaussian、不训练模型，也不把
BOP row 标记为 reality gate pass。
随后补齐 `OBJECTSTATE-BOP-PHASE1-SUBSET-SELECTOR-001`：新增
`objgauss.core.objectstate_bop_phase1_subset_selector`，schema 为
`objgauss-objectstate-bop-phase1-subset-selector-v1`，CLI 为
`objgauss object-state select-bop-phase1-subset`。该命令只读扫描本地 BOP dataset /
split root，发现包含 `scene_gt.json` 和 `scene_camera.json` 的 scene roots，并复用现有
BOP adapter 验证每个候选；ready 判定要求足够 selected frames、objects、repeated
identities、identity stage 和 prediction stage seed。输出推荐 scene、sample id、
blocked candidate issues 和后续 `init-bop-condition-sidecar` /
`accept-bop-capture-scene` / `audit-bop-phase1-local-row` 命令。它不下载 BOP、不复制数据、
不创建 GT、不推断 condition metadata、不生成 Gaussian、不运行 handoff、不训练模型，也不把
任何 row 标记为 reality gate pass。
随后完成 `OBJECTSTATE-BOP-BATCH-CSV-TEMPLATE-001`：`select-bop-phase1-subset`
新增 `--batch-samples-csv-template-output` 和 `--batch-sample-artifact-root`，可把扫描结果中
ready 的 BOP scene 写成 `init-bop-local-row-batch-spec` 可消费的 CSV。CSV 包含
`sample_id`、`scene_root`、默认 `candidate_artifact` / `condition_sidecar` 路径、
`output_root`、`dataset_id`、`object_category`、`scenario`、`max_frames` 和
`frame_step`；它只写 authoring 模板，不创建 candidate artifact 或 sidecar，不运行
readiness / handoff，不生成 Gaussian、不训练模型、不声明 pass row 或 world model。
随后完成 `OBJECTSTATE-BOP-PHASE1-BATCH-WORKSPACE-001`：
新增 `objgauss.core.objectstate_bop_phase1_batch_workspace`，schema 为
`objgauss-objectstate-bop-phase1-batch-workspace-v1`，CLI 为
`objgauss object-state init-bop-phase1-batch-workspace <dataset-root> --workspace-root <dir>`。
该命令把本地 BOP subset selector、`samples.csv` authoring 和
`objgauss-objectstate-bop-local-row-batch-spec-v1` batch spec authoring 合成一个本地
workspace，写出 `selector-summary.json`、`samples.csv`、
`bop-local-row-batch.json`、`batch-spec-authoring-summary.json` 和 `README.md`。
它只初始化 authoring workspace，不创建 candidate artifact / condition sidecar，不生成
Gaussian evidence，不运行 readiness / handoff，不训练模型，也不声明 metric pass /
intervention gate / world model。
随后完成 `OBJECTSTATE-BOP-PHASE1-SAMPLE-WORKSPACES-001`：
新增 `objgauss.core.objectstate_bop_phase1_sample_workspace`，schema 为
`objgauss-objectstate-bop-phase1-sample-workspaces-v1`，CLI 为
`objgauss object-state init-bop-phase1-sample-workspaces <batch-spec.json>`。
该命令从 native BOP local-row batch spec 解析每个 sample 的 scene root、目标
`objectstates.json` 和目标 `bop-condition-sidecar.json`，在每个 sample 的 authoring
目录写出 `bop-conditions.template.csv`、`bop-condition-sidecar.draft.json` 和
`README.md` / next commands。它不创建真实 target sidecar 或 candidate artifact，不生成
Gaussian evidence，不运行 readiness / handoff，不训练模型，也不声明 metric pass /
intervention gate / world model。
随后完成 `OBJECTSTATE-BOP-PHASE1-AUTHORING-PROGRESS-001`：
新增 `objgauss.core.objectstate_bop_phase1_authoring_progress`，schema 为
`objgauss-objectstate-bop-phase1-authoring-progress-v1`，CLI 为
`objgauss object-state audit-bop-phase1-authoring-progress <batch-spec.json>`。
该命令只读检查每个 sample 的 helper files、target `bop-condition-sidecar.json`、
per-frame `gaussians/<frame>.ply` evidence 和 target `objectstates.json`，并输出
per-sample issues、next commands 和 Markdown table。只有 target sidecar schema 有效、
Gaussian evidence 齐全且 candidate artifact 至少是正式 trainable artifact schema 时，
才标记 ready for batch readiness input；它不运行 batch readiness / handoff，不生成或训练
模型，也不声明 metric pass / intervention gate / world model。
随后完成 `OBJECTSTATE-BOP-BASELINE-CANDIDATE-001`：新增
`objgauss.core.objectstate_bop_baseline_candidate`，schema 为
`objgauss-objectstate-bop-baseline-candidate-v1`，CLI 为
`objgauss object-state generate-bop-objectstate-baseline-candidate <scene-root> --output <objectstates.json>`。
该命令在本地 BOP scene 和 per-frame `gaussians/<frame>.ply` evidence 已存在时运行
BOP acceptance，然后对每个 selected frame 的 Gaussian `xyz` 计算一个全局 centroid
和 bbox，写出当前 identity route 可审计的
`objgauss-trainable-kernel-model-artifact-v1` candidate artifact。该 artifact 明确标记
为 `gaussian_centroid_single_state` baseline candidate，预期在多对象 identity 场景中
形成可审阅负证据；它不读取 BOP pose GT 或 object ids 来放置预测 ObjectState，不训练
Gaussian / tracking / dynamics 模型，不运行 identity handoff / eval，不创建 pass row，
也不改 viewer/export 默认策略。`init-bop-phase1-sample-workspaces` 已把该 baseline
命令插到 RGB-D Gaussian evidence export 之后、手工 template authoring 之前；
`audit-bop-phase1-authoring-progress` 在 Gaussian evidence 存在但 target candidate
artifact 缺失时也会优先提示这条命令。
随后完成 `OBJECTSTATE-BOP-BASELINE-LOCAL-ROW-HANDOFF-001`：新增
`objgauss.core.objectstate_bop_baseline_local_row_handoff`，schema 为
`objgauss-objectstate-bop-baseline-local-row-handoff-v1`，CLI 为
`objgauss object-state bop-baseline-local-row-handoff <scene-root> --output-root <dir>`。
该命令在本地 BOP scene 和 per-frame Gaussian evidence 已存在时，默认写
`<output-root>/objectstates.json` baseline candidate，再复用
`bop-local-row-handoff` 串起 identity handoff、prediction baseline handoff 和
Phase 1 evidence ledger。reviewable 与 metric pass 继续分离：single-state baseline
预期可形成 identity fail / negative evidence，而不是被包装成通过。该步骤不下载 BOP、
不创建 GT、不使用 BOP pose GT 或 object ids 来放置预测 ObjectState、不重建 Gaussian、
不训练模型、不声明 intervention / counterfactual gate 或 world model，也不改 viewer/export
默认策略。
随后完成 `OBJECTSTATE-BOP-RGBD-BASELINE-LOCAL-ROW-HANDOFF-001`：新增
`objgauss.core.objectstate_bop_rgbd_baseline_local_row_handoff`，schema 为
`objgauss-objectstate-bop-rgbd-baseline-local-row-handoff-v1`，CLI 为
`objgauss object-state bop-rgbd-baseline-local-row-handoff <scene-root> --output-root <dir>`。
该命令把本地 BOP RGB-D route 串成一条：先用 `depth/<frame>.png` 和
`scene_camera.json` 写 per-frame `gaussians/<frame>.ply` evidence seed，再生成
single-state baseline `objectstates.json`，最后复用 baseline local-row handoff 生成
identity / prediction packages 和 Phase 1 evidence ledger。RGB-D export、baseline
reviewability 和 metric pass 继续分离；depth 缺失时只返回 incomplete / blocked summary，
不会创建伪 Gaussian evidence 或伪 local rows。该步骤不下载 BOP、不创建 GT、不使用 object
pose GT 生成 RGB-D geometry、不使用 BOP pose GT 或 object ids 放置预测 ObjectState、不运行
Splatfacto / 3DGS optimization、不训练模型、不声明 metric pass、intervention /
counterfactual gate 或 world model，也不改 viewer/export 默认策略。
随后生成第一条真实 public BOP LMO RGB-D baseline evidence row：
`OBJECTSTATE-BOP-LMO-PUBLIC-ROW-001`。官方来源为 Hugging Face
`bop-benchmark/lmo`，license 为 `cc-by-sa-4.0`；本地 ignored zip 为
`outputs/assets/raw/bop-lmo/lmo_test_bop19.zip`，大小 `117550985` bytes，
SHA256 为
`42d7a15f317476ca3980ee7ec0344b691cbadc796835f0b14f72c89a1dcec421`。
抽取 `test/000002` 的 `000003` / `000008` / `000017` 三帧后，运行
`bop-rgbd-baseline-local-row-handoff` 产出本地 ignored evidence package：
`outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/`。结果为
`selected_frames=3`、`exported_frames=3`、`rgbd_total_vertices=30000`、
`baseline_total_gaussians=30000`、`prediction_candidates=16`。prediction evidence
package 已达到 `objectstate_controlled_prediction_evidence_package_reviewable`，
Phase 1 ledger maturity 为 `prediction_reviewable`，但 metric gate 明确 fail：
`state_ade=0.292669`、`history_ade=0.195073`、gap `0.097596`、
error ratio `1.500305`。identity eval 也 fail，并暴露
`identity_collapse=true`、`track_retrieval_recall_at_1=0.125`、缺
reconstruction-noise evidence；identity package 仍因缺 explicit lighting /
camera-pose condition metadata 而 incomplete。该行是实际 public data negative
evidence，不是 pass row，不声明 intervention / world model。
同时修复 `OBJECTSTATE-BOP-PREDICTION-PACKAGE-RELATIVE-PATH-001`：当
`bop-prediction-baseline-handoff` 使用相对 `output_root` 时，prediction evidence package
此前会把 `candidate_dir` 拼成双重路径，真实 LMO run 中表现为 package audit 报
required files missing；现在 handoff 传入相对 `reality-candidates`，并补相对路径回归测试。
随后完成 `OBJECTSTATE-BOP-LMO-CONDITION-GAP-001`：对同一 LMO row 运行
`init-bop-condition-sidecar` 和 `audit-bop-identity-route`，把 identity blocker 收敛为
可复验 condition metadata gap，而不是继续停留在口头判断。ignored 输出为
`outputs/evidence/objectstate-bop-lmo-public-000002-condition-gap/`。sidecar summary 为
`objectstate_bop_capture_condition_sidecar_needs_metadata`，选中帧仍为 `000003` /
`000008` / `000017`，`view_condition_count=3`，但 `lighting_condition_count=1`、
`camera_pose_count=0`、`max_camera_translation_m=0.0`，
`identity_scenario_metadata_ready=false`。route audit 为
`objectstate_bop_identity_route_audit_blocked`，同时证明 `bop_acceptance_pass=true`、
`phase1_gaussian_evidence_ready=true`、`candidate_artifact_valid=true` 且
`candidate_artifact_binding_ready=true`；因此当前阻塞不是文件、Gaussian evidence 或
candidate binding，而是缺真实 lighting / camera-pose condition metadata。该切片不放松
identity gate、不伪造 condition CSV、不创建 pass row、不训练模型、不声明 intervention /
world model。
随后完成 `OBJECTSTATE-BOP-MULTI-INSTANCE-IDENTITY-POLICY-001` 和
`OBJECTSTATE-BOP-HOPE-PUBLIC-ROW-001`：BOP adapter 的默认 identity policy 仍为
`single_instance_per_bop_obj_id`，重复 `obj_id` 继续 fail-fast；新增显式
`pose_track_per_obj_id` policy，用 BOP pose GT 的帧间连续性只在 ground-truth manifest
导入阶段生成稳定 physical instance ids，并在实例数变化、匹配歧义或超过
`pose_track_max_distance_m=0.05` 时失败。该策略已透传到 RGB-D export、baseline
candidate、identity / prediction handoff、route audit 和 candidate template CLI。使用
官方 BOP HOPE `hope_val_realsense.zip`（license `cc-by-sa-4.0`，大小 `153745625`
bytes，SHA256
`25c75bb2daad4ad7e143b3f8d5bdff793fadb65463492792e822dbb36245a49f`）抽取
`val/000001` 的 `000000` / `000001` / `000002` 三帧后，显式运行
`bop-rgbd-baseline-local-row-handoff --identity-policy pose_track_per_obj_id`。
结果为 `selected_frames=3`、`exported_frames=3`、`rgbd_total_vertices=30000`、
`baseline_total_gaussians=30000`、`identity_predictions=54`、
`prediction_candidates=36`；prediction package 和 Phase 1 ledger prediction 侧
reviewable，`prediction_eval_pass=true`，但 identity 侧仍不 reviewable / 不 pass。
随后完成 `OBJECTSTATE-BOP-HOPE-CONDITION-GAP-001`：condition sidecar 和 identity route
audit 证明 HOPE row 的阻塞已从 adapter multi-instance 限制转移到真实 scenario metadata：
`bop_acceptance_pass=true`、`phase1_gaussian_evidence_ready=true`、
`candidate_artifact_binding_ready=true`，但 `lighting_condition_count=1`、
`camera_pose_count=0`、缺 clear occlusion-reappearance metadata，
`identity_scenario_metadata_ready=false`。该结果是 public multi-instance route 的
prediction-reviewable / identity-blocked evidence，不是 Phase 1 identity pass row；
不使用 pose GT 生成 candidate prediction，不训练模型，不声明 intervention 或 world-model
证明。
随后完成 `OBJECTSTATE-BOP-REALITY-ROWS-001`：新增
`objgauss.core.objectstate_bop_reality_rows`，schema 为
`objgauss-objectstate-bop-reality-rows-v1`，CLI 为
`objgauss object-state audit-bop-reality-rows <local-row-summary.json>`。该命令把现有
BOP local-row / baseline local-row / RGB-D baseline local-row summary 中已经生成的
controlled-real manifest evidence rows 转成 `source_kind=public_replay` 的
`OBJECTSTATE-REALITY-GATE-001` rows，并写出 summary JSON 和 blocked rows Markdown；
它不重跑 handoff、不创建 GT、不重建 Gaussian、不训练模型、不写 public samples、不改
viewer/export 默认，也不声明 intervention 或 world model。当前已对 ignored LMO / HOPE
public evidence 执行：LMO 输出 `identity=fail`、`prediction=fail`、
`intervention=blocked`，HOPE 输出 `identity=fail`、`prediction=pass`、
`intervention=blocked`；两者 full gate 都是 `objectstate_reality_gate_fail`，因为缺
identity pass、intervention pass，且 baseline identity collapse 被保留为 fail evidence。
随后完成 `OBJECTSTATE-REALITY-ROW-LEDGER-001`：新增
`objgauss.core.objectstate_reality_row_ledger`，schema 为
`objgauss-objectstate-reality-row-ledger-v1`，CLI 为
`objgauss object-state audit-reality-row-ledger <summary...>`。该 ledger 读取已存在的
BOP reality rows、controlled real rows、public artifact rows 或 raw reality gate summary，
重新校验每条 `objgauss-objectstate-real-public-row-v1` row，并合并运行一个全局
`OBJECTSTATE-REALITY-GATE-001` report。当前对 ignored LMO / HOPE public summaries 运行
后得到 `summary_count=2`、`row_count=6`、`pass_row_count=1`、`fail_row_count=3`、
`blocked_row_count=2`、`sample_count=2`，全局 gate 仍为
`objectstate_reality_gate_fail`；缺失 pass evidence kinds 为 `identity` 和
`intervention`。该步骤是 read-only accounting，不创建 GT、不训练模型、不声明 world model。
随后完成 `OBJECTSTATE-REALITY-ROW-LEDGER-NEXT-ACTIONS-001`：扩展同一个
`audit-reality-row-ledger` summary，新增机器可读 `next_actions` 和
operator-facing `next_actions_markdown`。当前 LMO / HOPE ledger 仍为
`summary_count=2`、`row_count=6`、`pass_row_count=1`、`fail_row_count=3`、
`blocked_row_count=2`、full gate=`objectstate_reality_gate_fail`；新增 next actions
只列出两个 P0 缺口：`identity -> controlled_real_identity_handoff` 和
`intervention -> controlled_reality_bundle_handoff`。CLI 新增
`--next-actions-output`，本地 ignored 输出为
`outputs/evidence/objectstate-phase1-reality-row-ledger-next-actions.md`。该切片仍是
read-only handoff planning：不采集数据、不创建 GT、不运行 identity / prediction /
intervention eval、不训练模型、不声明 ObjectState 已经是 world model。
随后完成 `OBJECTSTATE-STATE-VARIABLE-MATRIX-001`：同一个
`audit-reality-row-ledger` summary 新增 `state_variable_evidence_matrix` 和
`state_variable_evidence_matrix_markdown`，把 reality rows 映射到 State Variable Gate
五个实验：identity persistence、occlusion recovery、view invariance、predictive
sufficiency 和 counterfactual / action interface。当前 LMO / HOPE public replay 的矩阵为：
`identity_persistence=fail`，`occlusion_recovery=missing_metric`，
`view_invariance=missing_metric`，`predictive_sufficiency=pass`，
`counterfactual_action_interface=blocked`。CLI 新增
`--experiment-matrix-output`，本地 ignored 输出为
`outputs/evidence/objectstate-phase1-state-variable-experiment-matrix.md`。该矩阵只做
read-only evidence accounting，不创建新 row、不放松 gate、不把 prediction pass 解释成
identity / counterfactual pass，也不声明 world model。
随后完成 `OBJECTSTATE-BOP-SCENARIO-CHALLENGE-METRICS-001`：扩展
`audit-bop-reality-rows`，从已有 BOP local-row identity scenario audit 里把场景挑战覆盖
写入 identity row metrics，例如 `occlusion_challenge_present`、
`view_challenge_present`、`lighting_challenge_present`、
`camera_motion_challenge_present`、condition counts 和
`identity_scenario_metadata_ready`；intervention blocked row 也显式记录
`action_challenge_present=false`。重新生成 LMO / HOPE ignored BOP reality rows 后，
全局 ledger 的五实验矩阵变为：`occlusion_recovery=missing_metric/challenge_present`，
`view_invariance=missing_metric/challenge_present`，
`counterfactual_action_interface=blocked/challenge_absent`。这说明 public replay 已含
部分 occlusion / view 挑战场景，但仍缺真正的 `occlusion_recovery_rate`、
`contrastive_margin` 和 action/counterfactual evidence；该切片不创建 pass row、不放松
identity gate、不声明 world model。
随后完成 `OBJECTSTATE-PUBLIC-INTERACTION-ROUTE-AUDIT-001`：扩展
`audit-public-dataset-candidates` 所在模块，新增
`objgauss.core.objectstate_public_interaction_route_audit`，schema 为
`objgauss-objectstate-public-interaction-route-audit-v1`，CLI 为
`objgauss object-state audit-public-interaction-route`。该命令默认审计 `hot3d-clips`
这类 action-capable public interaction candidate，并检查本地 bundle root 是否具备
`capture-manifest.json`、`objectstates.json`、
`reality-candidates/prediction-candidates.json` 和
`reality-candidates/intervention-candidates.json`。只有 controlled capture manifest
intervention-ready、声明 per-frame Gaussian evidence、prediction / intervention
candidate JSON valid 且 `sample_id` 绑定一致时，才报告
`objectstate_public_interaction_route_handoff_ready`。该切片只是把 BOP 缺失的
action / counterfactual 路线变成机器可审计 preflight；不下载 HOT3D、不适配原始
egocentric streams、不创建 GT、不运行 eval、不训练模型、不新增 pass row，也不声明
counterfactual proof 或 world model。
随后完成 `OBJECTSTATE-PUBLIC-INTERACTION-ROUTE-STATUS-001`：同一路由 audit
新增 `accounting_route_status` 四态：`handoff_ready`、`prediction_ready`、
`identity_ready` 和 `evidence_incomplete`。原有 strict `status` /
`controlled_reality_handoff_ready` 不变，仍要求 usable action GT、prediction /
intervention candidates、Gaussian refs 和 sample binding 全部满足才允许 full handoff；
新增四态只表示 public interaction route 当前最多能进入哪类真实 row accounting。
例如 action row 存在但 vector 为零时，route 仍保持
`objectstate_public_interaction_route_intervention_gt_required`，但如果 identity / pose
transition、Gaussian refs、ObjectState artifact 和 prediction candidates 已齐，会报告
`accounting_route_status=prediction_ready`，避免把可用 prediction evidence 一刀切废掉，
也避免把 intervention 伪装成 ready。
随后完成 `OBJECTSTATE-PUBLIC-INTERACTION-REALITY-ROWS-001`：新增
`objgauss.core.objectstate_public_interaction_reality_rows`，schema 为
`objgauss-objectstate-public-interaction-reality-rows-v1`，CLI 为
`objgauss object-state audit-public-interaction-reality-rows <reality-bundle-handoff-summary.json>`。
该命令读取已经完成的 full controlled reality handoff summary，重新登记 identity /
prediction / intervention 三条 row 为 `source_kind=public_replay`，并重新运行
`OBJECTSTATE-REALITY-GATE-001` accounting，使 HOT3D 这类 public interaction evidence
可以进入 `audit-reality-row-ledger`，而不是被计入 `controlled_real`。intervention row
会显式保留 `action_challenge_present=true`，用于 state-variable matrix 的
counterfactual/action challenge accounting。该切片不运行 handoff、不运行 eval、不下载或适配
HOT3D、不创建 GT、不训练模型、不改变 metric 结果，也不把 observed action 解释成 randomized
counterfactual proof 或 world model。
随后完成 `OBJECTSTATE-PUBLIC-INTERACTION-WORKSPACE-001`：新增
`objgauss.core.objectstate_public_interaction_workspace`，schema 为
`objgauss-objectstate-public-interaction-workspace-v1`，CLI 为
`objgauss object-state init-public-interaction-route-workspace <workspace-root>`。
该命令为 HOT3D / DexYCB-style public interaction clip 初始化 local-only controlled
capture authoring workspace：`sample.json`、CSV headers、`rgb/`、`gaussians/` 和
`PUBLIC_INTERACTION_ROUTE.md`，并把 route audit、bundle import / acceptance、
candidate templates、full handoff、`public_replay` rows converter 和 ledger 命令串成
同一条 operator handoff。该切片只降低 public action evidence authoring 漏填和路径漂移风险；
不下载或适配 public dataset、不创建 GT / frame / annotation / action rows、不生成 Gaussian
evidence、不创建 candidate、不运行 handoff/eval、不训练模型、不创建 pass row，也不声明
counterfactual proof 或 world model。
随后完成 `OBJECTSTATE-PUBLIC-INTERACTION-WORKSPACE-PROGRESS-001`：同一模块新增
`objgauss-objectstate-public-interaction-workspace-progress-v1` 和 CLI
`objgauss object-state audit-public-interaction-workspace-progress <workspace-root>`。
该 read-only audit 把 public interaction workspace 从 source sequence binding 到
controlled capture import/file readiness、intervention-ready pose/action rows、ObjectState
artifact、prediction / intervention candidates、full handoff summary、`public_replay`
rows 和 ledger summary 的每一段缺口显性化，输出 `hard_blockers`、`next_actions` 和
`evidence_chain_reviewable`。该标志只表示 row accounting 文件链可审阅，不是 metric pass；
本切片仍不下载数据、不创建 GT/rows/candidates、不生成 Gaussian、不运行 handoff/eval、不训练模型，
也不声明 intervention pass、counterfactual proof 或 world model。
随后完成 `OBJECTSTATE-PUBLIC-INTERACTION-CLIP-CSV-ADAPTER-001`：同一模块新增
`objgauss-objectstate-public-interaction-clip-csv-adapter-v1` 和 CLI
`objgauss object-state import-public-interaction-clip-csv <source.csv> <workspace-root>`。
该 adapter 面向已经被外部规范化的一行一 frame/object 标注表，把 timestamp、physical
`object_id`、6DoF pose、action metadata、RGB ref 和 per-frame Gaussian ref 转写为
controlled capture bundle 的 `sample.json`、`objects.csv`、`frames.csv`、
`annotations.csv` 和 `actions.csv`，并立即跑 existing controlled capture import summary。
默认要求 pose/action/Gaussian refs 齐全，使 bundle 可进入 identity / prediction /
intervention-ready 状态；但它仍只是 GT row authoring adapter，不下载 HOT3D / DexYCB、
不复制媒体、不推断 GT、不重建 Gaussian、不创建 candidates、不运行 handoff/eval、不生成
`public_replay` rows、不训练模型，也不声明 intervention pass、counterfactual proof 或
world model。
随后完成 `OBJECTSTATE-TRANSITION-DATASET-001`：新增
`objgauss.core.objectstate_transition_dataset`，schema 为
`objgauss-objectstate-transition-dataset-v1`，row schema 为
`objgauss-objectstate-transition-row-v1`，CLI 为
`objgauss object-state compile-objectstate-transitions <capture-manifest.json> --output <objectstate-transitions.json>`。
该 compiler 从已验证 controlled capture manifest 生成 object-level episodes 和
`ObjectState_t + action_context -> ObjectState_t+1` transition rows，并统计
action-conditioned / no-action transitions、pose transition readiness 和 Gaussian ref
coverage。它是把 frame-level GT 收敛为目标文件所要求 Object Transition Dataset 的数据
契约；不采集数据、不创建 GT、不推断 identity、不重建 Gaussian、不运行 prediction /
intervention eval、不训练 dynamics、不创建 replay buffer、不生成 reality rows，也不声明
metric pass 或 world model。
随后完成 `OBJECTSTATE-TRANSITION-DATASET-AUDIT-001`：同一模块新增
`objgauss-objectstate-transition-dataset-audit-v1` 和 CLI
`objgauss object-state audit-objectstate-transition-dataset <objectstate-transitions.json>`。
该只读 audit 检查 object episode 数、transition 数、action-conditioned transition 数、
object horizon、pose readiness 和 real Gaussian ref readiness，并输出 hard blockers /
next actions。它只说明 transition dataset 是否具备进入候选训练或 evaluator authoring 的
最低数据条件；不训练 dynamics、不创建 replay buffer、不运行 prediction / intervention eval、
不生成 reality rows、不声明 metric pass 或 world model。
随后完成 `OBJECTSTATE-TRANSITION-PREDICTION-CANDIDATES-001`：新增
`objgauss.core.objectstate_transition_prediction_candidates`，schema 为
`objgauss-objectstate-transition-prediction-candidates-v1`，CLI 为
`objgauss object-state export-transition-prediction-candidates <objectstate-transitions.json>`。
该 exporter 从 object-level transition rows 写出现有
`objgauss-objectstate-controlled-prediction-candidates-v1` evaluator JSON，支持 `hold`、
`constant_velocity` 和 `action_delta` baseline policy。预测生成只使用 source pose、
prior pose、target timestamp 和可选 action vector；测试覆盖修改 target pose 不改变导出预测。
它不运行 prediction eval、不训练 dynamics、不创建 replay buffer、不生成 reality rows，也不声明
metric pass、learned model 或 world model。
随后完成 `OBJECTSTATE-TRANSITION-INTERVENTION-CANDIDATES-001`：新增
`objgauss.core.objectstate_transition_intervention_candidates`，schema 为
`objgauss-objectstate-transition-intervention-candidates-v1`，CLI 为
`objgauss object-state export-transition-intervention-candidates <objectstate-transitions.json>`。
该 exporter 从 action-conditioned transition rows 写出现有
`objgauss-objectstate-controlled-intervention-candidates-v1` evaluator JSON，支持
`action_delta` 和 `hold_action` baseline policy。action-conditioned 预测只使用 source pose
和 action vector，no-action baseline 使用 source pose；测试覆盖修改 target pose 不改变导出预测。
它不运行 intervention eval、不训练 dynamics、不创建 replay buffer、不生成 reality rows，也不声明
metric pass、learned model、counterfactual proof 或 world model。
随后完成 `OBJECTSTATE-TRANSITION-REALITY-HANDOFF-001`：新增
`objgauss.core.objectstate_transition_reality_handoff`，schema 为
`objgauss-objectstate-transition-reality-handoff-v1`，CLI 为
`objgauss object-state transition-reality-handoff <capture-manifest.json> <objectstate-transitions.json>`。
该 handoff 对 Object Transition Dataset 先跑 readiness audit，再导出 action-delta /
constant-velocity 等 baseline prediction candidates 和 action-delta / hold-action
intervention candidates，随后复用既有 controlled prediction / intervention evaluator，
写出 prediction / intervention eval summaries、controlled-real manifest、partial
controlled-real summary 和 blocked rows。它会把 identity row 保持为 blocked seed，只要求
prediction / intervention rows 进入 pass / fail accounting；不训练 dynamics、不创建 replay
buffer、不声明 learned model、identity proof、counterfactual proof 或 world model。
随后完成 `OBJECTSTATE-TRANSITION-REALITY-EVIDENCE-PACKAGE-001`：新增
`objgauss.core.objectstate_transition_reality_evidence_package`，schema 为
`objgauss-objectstate-transition-reality-evidence-package-v1`，CLI 为
`objgauss object-state audit-transition-reality-evidence-package <package-root>`。
该只读 audit 检查 `transition-reality-handoff/` 下的 transition audit、prediction /
intervention candidates、eval summaries、controlled-real manifest / summary、blocked rows
和 handoff summary 是否存在、schema valid、sample id 一致、standalone outputs 与 handoff
嵌入内容一致，并要求 identity row 保持 blocked、prediction / intervention row 为 pass 或
fail。Phase 1 ledger 新增 `transition_reality` stage 和
`transition_reality_reviewable` maturity；该 stage 可贡献 prediction / intervention
reviewability，但不会贡献 identity reviewability 或 full reality reviewability，不训练
dynamics、不创建 replay buffer、不声明 learned model、identity proof、counterfactual proof 或
world model。
随后补齐 `OBJECTSTATE-BOP-GAUSSIAN-EVIDENCE-PREFLIGHT-001`：新增
`objgauss.core.objectstate_bop_gaussian_evidence_preflight`，schema 为
`objgauss-objectstate-bop-gaussian-evidence-preflight-v1`，CLI 为
`objgauss object-state audit-bop-gaussian-evidence`。该命令对已选 BOP scene 复用
`accept-bop-capture-scene` 的 adapter / file audit 路径并强制 `require_gaussian_files`，
列出 expected / missing per-frame `gaussians/<frame>.ply` evidence，同时复用 controlled
capture environment preflight 记录 COLMAP / Nerfstudio Gaussian reconstruction tool
readiness。它只做 read-only preflight，不下载 BOP、不创建 GT、不推断 condition metadata、
不重建 Gaussian、不运行 handoff、不训练模型，也不声明任何 Phase 1 pass row。
随后补齐 `OBJECTSTATE-BOP-RGBD-GAUSSIAN-EXPORT-001`：新增
`objgauss.core.objectstate_bop_rgbd_gaussian_export`，schema 为
`objgauss-objectstate-bop-rgbd-gaussian-export-v1`，CLI 为
`objgauss object-state export-bop-rgbd-gaussian-evidence`。该命令在本地 BOP scene 已有
`depth/<frame>.png` 时，用 `scene_camera.json` 的相机内参和 `depth_scale` 把 depth
pixels 反投影成 per-frame `gaussians/<frame>.ply` evidence seed，并用 RGB 只做点颜色。
它会写本地 scene root 下的 ignored evidence 文件，但不下载 BOP、不使用 object pose GT
放置几何、不训练 Splatfacto、不创建 checkpoint、不运行 identity / prediction handoff，
也不声明 Phase 1 pass row 或 world-model 证明。
随后补齐 `OBJECTSTATE-BOP-RGBD-READINESS-HINT-001`：`audit-bop-phase1-local-row`
新增 `rgbd_gaussian_export_hint`，在 read-only local row audit 中统计 selected BOP frames
的 `depth/<frame>.png` 覆盖和 missing `gaussians/<frame>.ply` 数量；当 depth 齐全但
Gaussian evidence 缺失时，CLI 会打印 `rgbd_export_candidate=true` 和可直接运行的
`export-bop-rgbd-gaussian-evidence` 命令。该 hint 不自动写 PLY、不重建 / 训练模型、
不运行 handoff，也不把 local row 或 Phase 1 reality gate 标记为 pass。
随后补齐 `OBJECTSTATE-BOP-CANDIDATE-ARTIFACT-TEMPLATE-001`：
新增 `objgauss.core.objectstate_bop_candidate_artifact_template`，schema 为
`objgauss-objectstate-bop-candidate-artifact-template-v1`，summary schema 为
`objgauss-objectstate-bop-candidate-artifact-template-summary-v1`，CLI 为
`objgauss object-state init-bop-objectstate-artifact-template`。该命令从本地 BOP
acceptance manifest 生成 draft-only `objectstates.template.json`，列出每个 selected
frame 的 Gaussian ref、object ids 和待填写的 ObjectState centroid / bbox /
confidence placeholder；模板 schema 与 `objgauss-trainable-kernel-model-artifact-v1`
刻意不同，因此会被 identity route / trainable artifact validator 拒绝。该步骤只帮助作者
填写真正的本地模型输出 `objectstates.json`，不复制 BOP pose GT 到 candidate centroid、
不训练模型、不运行 identity eval、不创建 pass row，也不改 viewer/export 默认策略。
随后补齐 `OBJECTSTATE-BOP-CANDIDATE-ARTIFACT-FINALIZE-001`：
同一模块新增 `objgauss-objectstate-bop-candidate-artifact-finalize-v1` summary，
CLI 为 `objgauss object-state finalize-bop-objectstate-artifact-template`。该命令读取
已填写的 `objectstates.template.json`，要求 BOP acceptance / per-frame Gaussian
evidence 通过，检查 frame / Gaussian ref / object id 与 accepted manifest 绑定一致，
拒绝残留 TODO，拒绝 candidate centroid 与 BOP pose GT 精确匹配的明显泄漏，然后写出当前
identity route 可审计的 `objgauss-trainable-kernel-model-artifact-v1`
`objectstates.json`。输出 artifact policy 标记为 candidate packaging / not a training
run；可选写入真实模型提供的 reconstruction-noise evidence，但不伪造默认值。该步骤不训练
模型、不运行 identity handoff / eval、不创建 Phase 1 pass row，也不改 viewer/export 默认策略。
随后完成 `OBJECTSTATE-BOP-IDENTITY-HANDOFF-001`：新增
`objgauss.core.objectstate_bop_identity_handoff`，schema 为
`objgauss-objectstate-bop-identity-handoff-v1`，CLI 为
`objgauss object-state bop-identity-handoff <scene-root> --output-root <dir> --candidate-artifact <objectstates.json>`。
该命令把 BOP acceptance、finalized trainable ObjectState artifact、controlled identity
handoff / eval、identity evidence package audit 和 Phase 1 evidence ledger 串成一条本地
Stage 1 identity evidence 命令。输出中 `reviewability_gates` 与 `pass_gates` 分离：
证据包完整可审阅不等于 identity metric 通过，identity fail row 也可作为负证据进入
ledger。命令仍要求 per-frame Gaussian evidence、candidate artifact file audit、artifact ref
match、occlusion reappearance、view / lighting / camera motion metadata；它不下载 BOP、
不创建 GT、不重建 Gaussian、不训练模型、不声明 prediction / intervention gate 或 world model。
为兼容 BOP finalizer 的 candidate artifact，identity prediction adapter 会把 artifact-level
`identity_evidence` 归一化为 evaluator 需要的带来源 evidence，而不修改 artifact 本身。
随后完成 `OBJECTSTATE-BOP-LOCAL-ROW-HANDOFF-001`：新增
`objgauss.core.objectstate_bop_local_row_handoff`，schema 为
`objgauss-objectstate-bop-local-row-handoff-v1`，CLI 为
`objgauss object-state bop-local-row-handoff <scene-root> --output-root <dir> --candidate-artifact <objectstates.json>`。
该命令用同一 BOP scene / sample id / selected frames / Gaussian evidence / condition
sidecar 跑 `bop-identity-handoff` 和 `bop-prediction-baseline-handoff`，然后重写合并的
`phase1-evidence-ledger.json`。当 identity package 和 prediction package 都完整可审阅时，
ledger maturity 可达到 `identity_prediction_reviewable`；`intervention_evidence_reviewable`
仍按设计为 false，因为 BOP pose scene 没有 action outcome / counterfactual evidence。
顶层继续分离 `reviewability_gates` 与 `pass_gates`，不把可审阅证据等同于 metric pass。
该步骤仍不下载 BOP、不创建 GT、不重建 Gaussian、不训练模型、不声明 intervention gate 或
world model，也不改 viewer/export 默认策略。
随后完成 `OBJECTSTATE-BOP-CROSS-SAMPLE-LEDGER-001`：新增
`objgauss.core.objectstate_bop_cross_sample_ledger`，schema 为
`objgauss-objectstate-bop-cross-sample-ledger-v1`，CLI 为
`objgauss object-state audit-bop-cross-sample-ledger`。该 read-only audit 读取一个或多个
`bop-local-row-handoff-summary.json`，或从显式 root 发现这些 summary，先用 local-row
validator 复验，再输出 per-sample identity / prediction reviewability、row status、
metric pass booleans、sample / scene / category / scenario 覆盖和 Markdown 表。
`candidate_cross_sample_ready` 只表示 BOP identity+prediction 本地行在配置阈值下达到
cross-sample reviewable coverage；它不要求 metric pass，不声明 intervention /
counterfactual evidence，也不声明 ObjectState 已证明为 world state。该步骤不运行
handoff、不下载 BOP、不创建 GT、不重建 Gaussian、不训练模型、不写 public samples、不改
viewer/export 默认。
随后完成 `OBJECTSTATE-BOP-LOCAL-ROW-BATCH-SPEC-AUTHORING-001`：新增
`objgauss.core.objectstate_bop_local_row_batch_spec`，summary schema 为
`objgauss-objectstate-bop-local-row-batch-spec-authoring-v1`，CLI 为
`objgauss object-state init-bop-local-row-batch-spec --samples-csv <samples.csv> --output <batch-spec.json>`。
该命令把含 `sample_id`、`scene_root`、`candidate_artifact` 和可选
`condition_sidecar` / sample options 的 CSV 写成原生
`objgauss-objectstate-bop-local-row-batch-spec-v1`，并用现有 validator 复验；
默认把 CSV 相对输入路径改写成相对 batch spec 的路径，方便后续
`audit-bop-local-row-batch-readiness` 和 `bop-local-row-batch-handoff` 复用同一 spec。
authoring summary 会检查本地 scene root、candidate artifact 和 sidecar 是否存在，
但不验证 metric pass，不运行 readiness / handoff，不下载数据、不创建 GT、不重建 Gaussian、
不训练模型、不声明 intervention / world model。
随后完成 `OBJECTSTATE-BOP-LOCAL-ROW-BATCH-HANDOFF-001`：新增
`objgauss.core.objectstate_bop_local_row_batch_handoff`，batch spec schema 为
`objgauss-objectstate-bop-local-row-batch-spec-v1`，summary schema 为
`objgauss-objectstate-bop-local-row-batch-handoff-v1`，CLI 为
`objgauss object-state bop-local-row-batch-handoff <batch-spec.json>`。该命令用显式
batch spec 列出多个 BOP sample 的 `scene_root`、`sample_id`、`candidate_artifact`
和可选 `condition_sidecar`，逐个运行既有 `bop-local-row-handoff`，写出每个 sample 的
`bop-local-row-handoff-summary.json`，再自动生成 `bop-cross-sample-ledger.json`
和 `bop-cross-sample-table.md`。batch reviewable 与
`candidate_cross_sample_ready` 分离：前者表示所有 local rows 和 ledger 可审阅，后者才表示
达到配置的多 sample / scene / category coverage 阈值；二者都不声明 metric pass、
intervention gate 或 world model。该步骤仍不下载 BOP、不创建 GT、不重建 Gaussian、
不训练模型、不写 public samples、不改 viewer/export 默认。
随后完成 `OBJECTSTATE-BOP-LOCAL-ROW-BATCH-READINESS-001`：新增
`objgauss.core.objectstate_bop_local_row_batch_readiness`，summary schema 为
`objgauss-objectstate-bop-local-row-batch-readiness-v1`，CLI 为
`objgauss object-state audit-bop-local-row-batch-readiness <batch-spec.json>`。该
read-only preflight 使用同一个 batch spec，逐个复用 single-sample
`audit-bop-phase1-local-row` readiness，聚合 Gaussian evidence、candidate artifact
binding、identity scenario metadata、ready / reviewable sample count 和 scene /
category / scenario coverage。`ready` 只表示该样本可进入 `bop-local-row-handoff` 或已具备
identity+prediction reviewable evidence，不表示 metric pass；batch readiness 也不运行
handoff、不创建 GT、不重建 Gaussian、不训练模型、不声明 intervention / world model、不改
viewer/export 默认。下一步若要扩大 cross-sample 表，应先在实际 ignored BOP subset 上跑
该 readiness，使缺口显性化，再决定是否运行 batch handoff。

账面状态更新：训练模型主线 `TRAIN-GSPLAT-MVP-001` 已从
`suspended / current-env-missing-torch-gsplat-cuda` 恢复并完成最小 full renderer smoke。
真实 host 环境具备 RTX 5060 Ti、NVIDIA driver `595.71.05`、CUDA `13.2`、
`torch 2.12.1+cu130` 和 `gsplat 1.5.3`；需要在 host shell 或提权命令中执行 GPU
preflight，因为默认沙箱 `/dev` 视图不暴露 `/dev/nvidia*`，会误报 `nvidia-smi` /
CUDA 不可用。

算法模型主线已重新启动，但目标已从 full renderer training 拆分为先训练
`Object Emergence Solver`。当前事实源为
`docs/architecture/object-emergence-model-v1.md`：先在 dependency-free 环境中推进
`PerceptionEvidence -> A[N,K] -> ObjectState`，再在 torch / gsplat / CUDA 环境恢复后
把 full renderer loss 接回训练目标。
当前 torch / gsplat / CUDA 环境已经恢复到可运行 smoke 的状态，算法模型主线已从
“环境阻塞”转入真正训练路径设计。`OBJECTSTATE-GAUSSIAN-DECODER-001` 已将
`ObjectStateProjection -> Gaussian decode -> gsplat/image loss` 固化为可测代码路径：
v1 decoder 用 `ObjectStateProjection.assignment` 和 object-level decoder colors 生成
per-Gaussian color，means / quats / scales / opacities / cameras 仍冻结；短程 gsplat
smoke 已验证 renderer API、image render loss 和 decoder handoff 均可用。本切片不启动
长时间训练，不提交 `/tmp` summary、checkpoint、rendered image 或 ignored `outputs/`
产物，也不训练 Gaussian geometry / opacity / rotation。下一阶段进入 solver checkpoint
到 decoder 参数的训练绑定与 full loop 设计。

随后完成 `SOLVER-DECODER-TRAIN-001`：新增
`objgauss-object-state-gaussian-decoder-training-v1`，把
`ObjectStateProjection.assignment -> object_colors -> Gaussian decode -> image renderer loss`
变成最小可训练闭环。当前 trainer 只更新 `object_colors`，冻结 assignment、means、
quats、scales、opacities 和 cameras；CLI 新增
`objgauss training decoder-mvp`，可使用 object_id one-hot assignment，也可通过
`--solver-checkpoint` 读取 Object Emergence Solver checkpoint 生成 `A[N,K]`。CPU point
renderer smoke 验证 `image_render_loss=0.029535 -> 0.004649`；host GPU / gsplat smoke
验证 `image_render_loss=0.075436 -> 0.061272`，`renderer-loss-contract` 对应输出
`status=full_3dgs_decoder_training_ready`、`decoder_handoff_status=full_renderer_decoder_training_ready`。
该步骤仍不训练 Gaussian geometry / opacity / rotation，不提交 `/tmp` summary、
checkpoint、rendered image 或 ignored `outputs/` 产物。

随后完成 `SOLVER-DECODER-JOINT-001`：新增
`objgauss-solver-decoder-joint-training-v1`，把 Object Emergence Solver 的
`feature_weights / position_weights / bias` 与 Gaussian decoder 的 `object_colors`
放进同一个最小训练 loop。renderer API 的 `gradient_assignments` 会通过 softmax
assignment 反传到 solver 权重，`gradient_decoder_colors` 直接更新 decoder colors；
geometry、opacity、scale、camera 和 dynamic-K 仍冻结。CLI 新增
`objgauss training solver-decoder-mvp`，支持 object_id one-hot targets 和可选
`--solver-checkpoint` 初始状态。CPU point smoke 验证
`image_render_loss=0.052319 -> 0.049218`、`object_loss=1.368271 -> 1.344811`；
host GPU / gsplat smoke 验证
`image_render_loss=0.054193 -> 0.052948`、`object_loss=1.368271 -> 1.358761`，
`renderer-loss-contract` 对应输出
`status=full_3dgs_solver_decoder_joint_training_ready`、
`decoder_handoff_status=full_renderer_solver_decoder_joint_training_ready`。该步骤仍不启动
长训练，不提交 `/tmp` summary、checkpoint、rendered image 或 ignored `outputs/`
产物。

随后完成 `SOLVER-DECODER-EXPORT-001`：新增
`objgauss-solver-decoder-joint-checkpoint-v1`，把 joint training 的最终
`ObjectEmergenceSolverState` 和 `ObjectStateGaussianDecoderState` 保存为可
roundtrip / resume 的 checkpoint。CLI `objgauss training solver-decoder-mvp`
新增 `--checkpoint-output` 与 `--resume-checkpoint`；resume 会同时恢复 solver 与
decoder，并让两者 `step` 从 checkpoint 继续递增。`--solver-checkpoint` 仍保留旧
Object Emergence Solver checkpoint 初始化路径，并兼容只读取 joint checkpoint 中的
solver state。`renderer-loss-contract` 现在可直接消费 joint checkpoint，输出
`status=solver_decoder_joint_training_ready` 和
`decoder_handoff_status=solver_decoder_joint_training_ready`。CPU point export smoke
验证 `image_render_loss=0.052319 -> 0.049218`，resume smoke 验证
`image_render_loss=0.049218 -> 0.048663`；所有 checkpoint / summary / boundary 输出
仍只写入 `/tmp` 或 ignored `outputs/`，不提交训练产物。Gaussian geometry / opacity /
rotation / camera 和 dynamic-K 继续冻结。

随后完成 `TRAIN-SCALE-001`：新增 `objgauss-training-scale-plan-v1`，把
solver-decoder joint training 从单次 smoke 推进到可分段、可恢复、可审计的训练 run
布局。CLI `objgauss training solver-decoder-mvp` 新增 `--run-output-dir`、
`--checkpoint-every`、`--loss-log-every` 和 `--vram-reserve-gb`；传入
`--run-output-dir` 时会写出 `training-scale-plan.json`、每段 summary / checkpoint、
`final-summary.json`、`final-checkpoint.json` 和 `renderer-loss-boundary.json`。分段
训练通过现有 joint checkpoint 恢复下一段状态，保持 solver / decoder `step` 递增；
`--checkpoint-every` 只在 run output 模式下启用，避免无输出路径的伪 checkpoint。
CPU point scale smoke 验证 4 iteration / checkpoint every 2 的 run：
`run_initial_total_loss=0.189146 -> run_final_total_loss=0.183699`，并确认 1GB
显存预留策略进入 plan / checkpoint。该步骤不启动长时间训练，不提交 `/tmp` run 输出
或 ignored `outputs/` 产物，不训练 Gaussian geometry / opacity / rotation / camera /
dynamic-K。

随后完成 `TRAIN-RUN-TB-001`：`solver-decoder-mvp` 新增可选
`--tensorboard-logdir`，在 `--run-output-dir` 分段训练结束后写出 TensorBoard scalar
event。该 writer 只在显式传入 TensorBoard logdir 时导入 `torch.utils.tensorboard`；
基础依赖不新增 torch / tensorboard。已写入的 scalar tags 包括 `loss/total`、
`loss/image_render`、`loss/object`、`loss/entropy`、`loss/balance` 和
`run/final_total_loss`。真实 smoke 使用
`uv run --with torch --with tensorboard ... --tensorboard-logdir
/tmp/objgauss-tensorboard-smoke-run/tensorboard` 生成
`events.out.tfevents.*`，并用 TensorBoard event accumulator 读回上述 scalar tags。
6006 UI 必须指向实际 run logdir，例如 `$RUN_DIR/tensorboard`；指向 `/tensorboard`
这类空目录会显示 inactive。

随后完成 `EVAL-OBJECTSTATE-001`：新增
`objgauss-objectstate-checkpoint-eval-v1`，把 solver-decoder joint checkpoint 重新投影为
`ObjectStateProjection` 并输出 assignment entropy、confidence、effective slots、slot
mass / purity、collapse gate 和 temporal drift。CLI 新增
`objgauss training eval-objectstate <ply> --checkpoint <final-checkpoint.json>`，默认从
checkpoint 推断 slots / frame count / sampled gaussians，可写 summary 到 `/tmp` 或
ignored `outputs/`。对当前 GPU run
`outputs/training/train-run-003-solver-decoder-gsplat-sharpen/final-checkpoint.json` 的
只读评估结果为：`mean_normalized_entropy=0.601689`、
`assignment_confidence=0.398312`、`effective_slots=3.440100`、
`max_dominant_slot_mass_fraction=0.438636`、`slot_collapse=false`、
`object_purity=0.719425`，状态为 `objectstate_eval_fail`。结论：run-003 已经接近
entropy gate 且没有 collapse，但 purity 未达 0.8，不能据此解冻 Gaussian geometry；
下一步优先 `SOLVER-TEMP-001` / assignment sharpening，而不是继续盲目加长训练。

随后完成 `SOLVER-TEMP-001`：`solver-decoder-mvp` 新增 `--solver-temperature`，可在
初始化或 resume joint checkpoint 时覆盖 linear-softmax solver 的 temperature，并把该
config 写回 summary / checkpoint；`eval-objectstate` 也新增同名只读覆盖，用于快速扫描
assignment sharpening gate。该参数只改变 `A[N,K]` 的 softmax 温度，不训练或解冻
Gaussian means / quats / scales / opacities / cameras / dynamic-K。对 run-003 checkpoint 的
只读扫描显示 `--solver-temperature 0.5` 能让 eval 从 fail 变成
`objectstate_eval_pass`：`mean_normalized_entropy=0.192517`、
`assignment_confidence=0.807483`、`effective_slots=2.955224`、
`max_dominant_slot_mass_fraction=0.469030`、`slot_collapse=false`、
`object_purity=0.866342`。下一步应做一次受控 GPU resume run，把 temperature=0.5 固化
进正式 checkpoint，再评估是否进入 renderer 参数解冻计划。

随后完成 `TRAIN-RUN-004` 受控 GPU resume run：从
`outputs/training/train-run-003-solver-decoder-gsplat-sharpen/final-checkpoint.json`
resume，使用 `--solver-temperature 0.5`、gsplat renderer、128 sampled Gaussians、
2 frames、100 total iterations、checkpoint every 20、1GB VRAM reserve，输出到 ignored
`outputs/training/train-run-004-solver-temp05-gsplat/`。run-level loss 从
`0.219158 -> 0.170798`，image render loss 从 `0.018028 -> 0.017223`，object loss 从
`0.278156 -> 0.208460`；最后一个 segment 的 final loss 为
`total=0.170798`、`image_render=0.017223`、`object=0.208460`、`entropy=0.237402`。
final checkpoint 固化 `solver_temperature=0.5`，solver / decoder step 均为 500，trained
fields 仍只有 solver assignment weights 和 decoder object colors，Gaussian geometry /
opacity / camera / dynamic-K 继续冻结。`eval-objectstate --require-pass` 对 run-004 final
checkpoint 通过：`mean_normalized_entropy=0.237402`、`assignment_confidence=0.762598`、
`effective_slots=3.178765`、`max_dominant_slot_mass_fraction=0.466883`、
`slot_collapse=false`、`object_purity=0.868178`。注意：当前
`renderer-loss-boundary.json` 对 segmented run 仍读取最后一个 segment 的
`image_render_loss_decreased=false`，因此会显示 `point_render_smoke_blocked`，但
`decoder_handoff_status=full_renderer_decoder_ready` 且 run-level image loss 已下降；下一步
应先修正 segmented run boundary gate，再进入 renderer 参数解冻。

随后完成 `RENDER-LOSS-RUN-GATE-001`：`renderer-loss-contract` 在读取
solver-decoder segmented `final-summary.json` 时优先使用 `run_loss` 作为 readiness
evidence，同时保留最后一个 segment 的 loss 作为 `segment_*` 诊断字段。对
`outputs/training/train-run-004-solver-temp05-gsplat/final-summary.json` 重新生成 boundary
后，状态从 `point_render_smoke_blocked` 修正为
`full_3dgs_solver_decoder_joint_training_ready`，`point_smoke_blockers=[]`、
`upgrade_blockers=[]`、`decoder_handoff_status=full_renderer_solver_decoder_joint_training_ready`，
evidence 使用 run-level delta：`image_render_loss=0.018028 -> 0.017223`、
`object_loss=0.278156 -> 0.208460`。该修正不改变训练数学、不改 checkpoint schema、不解冻
Gaussian geometry / opacity / camera / dynamic-K。

随后完成 `RENDER-FIELD-UNFREEZE-PLAN-001`：新增
`docs/architecture/renderer-field-unfreeze-plan-v1.md`，把第一个 renderer 参数解冻切片
冻结为 `decoder.object_opacity_logits` 这类 object-level opacity multiplier，而不是直接
解冻 per-Gaussian opacity、means、scales、quats、camera 或 dynamic-K。规划要求下一步先做
`DECODER-OPACITY-CONTRACT-001`，将 opacity 写入 decoder state / checkpoint ABI，并保持旧
checkpoint 向后兼容；再做 `TRAIN-DECODER-OPACITY-001`，让 CPU / gsplat renderer API 暴露
object-level opacity 梯度和显式 `--train-decoder-opacity` 开关。首个训练 run 仍应从 run-004
final checkpoint resume，保留 1GB VRAM reserve，优先冻结 solver / colors 以隔离 opacity
效果；成功门槛包括 run-level image loss 下降、ObjectState eval 继续通过、`object_purity >=
0.85`、`mean_normalized_entropy <= 0.30`、无 slot collapse、opacity scale 不在 clamp 边界
大面积饱和。该步骤是 docs-only planning，不改训练数学、不启动 GPU 训练、不提交 ignored
`outputs/` 或 `/tmp` 产物。

随后完成 `DECODER-OPACITY-CONTRACT-001`：`ObjectStateGaussianDecoderState` 现在可携带可选
`object_opacity_logits`，并在 `as_dict()` / checkpoint roundtrip 中输出
`object_opacity_logits`、`object_opacity_scales`、`available_fields`、`frozen_fields` 和
`opacity_policy`。旧 decoder / joint checkpoint 缺该字段时会按 disabled /
`constant-opacity-v1` 加载，保持现有训练结果不被隐式改成半透明；显式传入 logits 时
`decode_gaussian_from_object_state(...)` 会用 `assignment @ sigmoid(logits)` 生成
per-Gaussian opacity，并在 decode summary 中把 `decoder.object_opacity_logits` 标为
differentiable field，同时把 frozen opacity 改写为 `source_opacities`。当前
`decoder-mvp` 和 `solver-decoder-mvp` 仍只训练 `object_colors` / solver assignment 参数；
本切片只建立 ABI，不接入 opacity gradient、不新增 CLI 训练开关、不启动 GPU 训练、不提交
ignored `outputs/` 产物。下一步是 `TRAIN-DECODER-OPACITY-001`：让 CPU / gsplat renderer API
暴露 object-level opacity gradient，并加显式 `--train-decoder-opacity` gate。

随后完成 `TRAIN-DECODER-OPACITY-001`：CPU point renderer 和 gsplat training renderer 的
`TrainingRendererLossResult` 现在暴露 `gradient_decoder_opacity_logits`，默认不传 logits 时
仍保持旧的 color / assignment 梯度路径和 constant opacity。`solver-decoder-mvp` 新增显式
`--train-decoder-opacity`、`--decoder-opacity-learning-rate` 和
`--decoder-opacity-init-logit`；未启用 gate 时 opacity 继续 frozen，启用时会初始化或读取
`decoder.object_opacity_logits`，通过 renderer loss 更新 object-level opacity multiplier，并在
summary / checkpoint 的 trained fields 中记录 `decoder.object_opacity_logits`。分段训练
summary 现在可记录 `final_decoder_opacity_scale_min/mean/max`，TensorBoard writer 可写入
`decoder/opacity_scale_min`、`decoder/opacity_scale_mean` 和
`decoder/opacity_scale_max`。本切片只接训练 contract 和 CPU smoke，不启动长时间 GPU run，
不提交 ignored `outputs/` 产物，Gaussian means / scales / quats / camera / dynamic-K 继续冻结。
下一步是 `TRAIN-RUN-005-OPACITY-SMOKE`：从 run-004 final checkpoint resume 做受控 GPU /
gsplat opacity smoke。

随后完成 `TRAIN-RUN-005-OPACITY-SMOKE` 受控 GPU resume run：从
`outputs/training/train-run-004-solver-temp05-gsplat/final-checkpoint.json` resume，开启
`--train-decoder-opacity`，使用 gsplat renderer、128 sampled Gaussians、2 frames、16x16
image target、100 total iterations、checkpoint every 20、1GB VRAM reserve，正式通过的
输出目录为 ignored `outputs/training/train-run-005-opacity-gsplat-image/`。该 run 使用
image-dominant smoke 配置：`object_weight=0.0`、`entropy_weight=0.0`、`balance_weight=0.0`、
`solver_learning_rate=0.001`、`decoder_learning_rate=0.5`、
`decoder_opacity_learning_rate=10.0`、`decoder_opacity_init_logit=6.0`。run-level image
loss 仅小幅下降：`0.0172239579 -> 0.0172238834`；`object_loss` 基本不变：
`0.2084604055 -> 0.2084604204`。final checkpoint 的 solver / decoder step 均为 600，
trained fields 包含
`solver.feature_weights, solver.position_weights, solver.bias, decoder.object_colors,
decoder.object_opacity_logits`，frozen fields 仍为
`means, quats, scales, source_opacities, cameras, dynamic_k`。TensorBoard logdir 为
`outputs/training/train-run-005-opacity-gsplat-image/tensorboard`，已写入
`decoder/opacity_scale_min`、`decoder/opacity_scale_mean`、`decoder/opacity_scale_max`；
opacity scale 位于 `0.99752742 -> 0.99752766`，未出现大面积 clamp 边界饱和。
`eval-objectstate --require-pass` 通过：`mean_normalized_entropy=0.237402`、
`assignment_confidence=0.762598`、`effective_slots=3.178765`、
`max_dominant_slot_mass_fraction=0.466883`、`slot_collapse=false`、
`object_purity=0.868178`。`renderer-loss-contract` 输出
`status=full_3dgs_solver_decoder_joint_training_ready`，`upgrade_blockers=[]`。
注意：两次带 object loss 权重的 opacity 配置未通过 image render loss decrease gate，因此
run-005 只能证明 opacity training path / checkpoint / TensorBoard / eval gate 可用，不能证明
object-level opacity 已经带来稳定可推广收益；下一步进入 renderer scale thaw 前应先做规划和
更严格的 promotion gate。

随后完成 `RENDER-FIELD-SCALE-PLAN-001`：在
`docs/architecture/renderer-field-unfreeze-plan-v1.md` 中把第二个 renderer 参数解冻切片
冻结为 `decoder.object_scale_log_offsets: R^K`。该字段是 object-level isotropic scale
multiplier，`0.0` 初始化必须严格等价于当前 frozen scale path；首个实现只允许
`[0.75, 1.25]` multiplier bound，不允许 `R^{K x 3}`、per-Gaussian scale、means、quats、
camera 或 dynamic-K。规划明确下一步先做 `DECODER-SCALE-CONTRACT-001`，只把 scale
offset 写入 decoder state / checkpoint ABI 和 Gaussian decode summary；之后才允许
`TRAIN-DECODER-SCALE-001` 接 renderer gradient / CLI gate，最后再进入
`TRAIN-RUN-006-SCALE-SMOKE`。run-005 的弱收益被记录为 promotion 风险：scale smoke 需要
更严格的 image loss、ObjectState eval、object loss、scale saturation 和 before / after
render gate。本步骤是 docs-only planning，不改训练数学、不启动 GPU 训练、不提交 ignored
`outputs/` 或 `/tmp` 产物。

随后完成 `DECODER-SCALE-CONTRACT-001`：`ObjectStateGaussianDecoderState` 现在可携带可选
`object_scale_log_offsets`，并在 `as_dict()` / checkpoint roundtrip 中输出
`object_scale_log_offsets`、`object_scale_multipliers`、`scale_policy`、
`available_fields` 和 `frozen_fields`。旧 decoder / joint checkpoint 缺该字段时会按
disabled / `constant-scale-v1` 加载，保持历史 checkpoint 的 frozen scale path 不变。
`decode_gaussian_from_object_state(...)` 新增显式 `object_scale_log_offsets` 输入，使用
`assignment @ exp(clamp(log_offsets, log(0.75), log(1.25)))` 生成 per-Gaussian
isotropic scale multiplier；启用时 decode summary 将
`decoder.object_scale_log_offsets` 标为 differentiable field，并把 frozen scale 改写为
`base_scales`。joint checkpoint 会保留 scale offsets，但当前 trainer 仍不更新该字段。
本切片只建立 ABI，不接 renderer scale gradient、不新增 CLI 训练开关、不启动 GPU 训练、
不提交 ignored `outputs/` 产物。下一步是 `TRAIN-DECODER-SCALE-001`：让 CPU / gsplat
renderer API 暴露 object-level scale gradient，并加显式 `--train-decoder-scale` gate。

随后完成 `TRAIN-DECODER-SCALE-001`：CPU point renderer 和 gsplat training renderer 的
`TrainingRendererLossResult` 现在暴露 `gradient_decoder_scale_log_offsets`。默认不传
`decoder_scale_log_offsets` 时仍保持 frozen scale path；显式启用时，CPU point renderer
使用 object-level scale multiplier 作为最小可微 smoke proxy，gsplat renderer 则通过
torch autograd 对 differentiable `scales = default_scale * (assignment @ scale_multiplier)`
回传真实 renderer scale 梯度。`solver-decoder-mvp` 新增 `--train-decoder-scale`、
`--decoder-scale-learning-rate` 和 `--decoder-scale-init-log-offset`；summary / checkpoint
会记录 `train_decoder_scale`、`learning_rates.decoder_scale`、`decoder_scale` summary、
`decoder.object_scale_log_offsets` trained field 和 `base_scales` frozen field。分段 run
summary 可写 `final_decoder_scale_multiplier_min/mean/max`，TensorBoard writer 可写
`decoder/scale_multiplier_min`、`decoder/scale_multiplier_mean` 和
`decoder/scale_multiplier_max`。本切片不启动 run-006 GPU smoke，不提交 ignored
`outputs/` 或 `/tmp` 产物，不训练 per-Gaussian means / scales / quats / camera /
dynamic-K。下一步先做 `FIELD-FREEZE-CONTROLS-001`，使 solver / colors / opacity / scale
可以独立冻结或训练，再进入 `TRAIN-RUN-006-SCALE-SMOKE`，否则 scale smoke 的结果不可解释。

随后完成 `FIELD-FREEZE-CONTROLS-001`：`train_solver_decoder_joint(...)` 现在支持
`train_solver` 和 `train_decoder_colors`，默认保持 `True` 以兼容旧命令；CLI 新增
`--freeze-solver`、`--freeze-decoder-colors`、`--freeze-decoder-opacity` 和
`--freeze-decoder-scale`。solver 冻结时不会更新 solver weights / bias / step；decoder
colors 冻结时不会更新 `decoder.object_colors`。如果 checkpoint 已经带有
`decoder.object_opacity_logits` 或 `decoder.object_scale_log_offsets`，冻结状态下这些字段仍会
参与 forward，只是不更新，避免 frozen 被误解为 disabled。summary / checkpoint 现在记录
`train_solver`、`train_decoder_colors`，并让 `trained_fields` / `frozen_fields` 精确反映
实际更新字段。CPU scale-only smoke 已验证冻结 solver 和 colors 后只训练
`decoder.object_scale_log_offsets`。本切片不启动 GPU 训练、不提交 ignored `outputs/` 或
`/tmp` 产物，不训练 per-Gaussian means / scales / quats / camera / dynamic-K。下一步是
`TRAIN-RUN-006-SCALE-SMOKE`：从 run-005 final checkpoint resume，使用 gsplat renderer 和
1GB VRAM reserve 做 scale-only GPU smoke。

随后完成 `TRAIN-RUN-006-SCALE-SMOKE`：从
`outputs/training/train-run-005-opacity-gsplat-image/final-checkpoint.json` resume，使用
gsplat renderer、128 sampled Gaussians、2 frames、16x16 image target、60 total
iterations、checkpoint every 20、`solver_temperature=0.5`、`freeze_solver=true`、
`freeze_decoder_colors=true`、`train_decoder_scale=true`、`decoder_scale_learning_rate=5.0`、
`object_weight=0.0`、`entropy_weight=0.0`、`balance_weight=0.0`、`vram_reserve_gb=1`，
输出到 ignored `outputs/training/train-run-006-scale-gsplat-smoke/`。run-level image loss
小幅下降：`0.0172238834 -> 0.0172227919`；object loss 保持
`0.2084604204 -> 0.2084604204`。trained fields 只有
`decoder.object_scale_log_offsets`；frozen fields 包含 solver 参数、`decoder.object_colors`、
`decoder.object_opacity_logits`、`base_scales`、`source_opacities`、cameras 和 dynamic-K。
final scale multiplier 位于 `0.9982541799 -> 1.0072258711`，未贴近 `[0.75, 1.25]`
边界。TensorBoard logdir 为
`outputs/training/train-run-006-scale-gsplat-smoke/tensorboard`，包含
`decoder/scale_multiplier_min`、`decoder/scale_multiplier_mean` 和
`decoder/scale_multiplier_max`。`eval-objectstate --require-pass` 通过：
`mean_normalized_entropy=0.237402`、`assignment_confidence=0.762598`、
`effective_slots=3.178765`、`max_dominant_slot_mass_fraction=0.466883`、
`slot_collapse=false`、`object_purity=0.868178`。`renderer-loss-contract` 输出
`status=full_3dgs_solver_decoder_joint_training_ready`，`upgrade_blockers=[]`。结论：
scale-only path / freeze controls / checkpoint / TensorBoard / eval gate 可用，但收益仍很弱；
renderer thaw 路线已足够证明管线可运行，下一步应回到 object assignment 主线，进入
`ASSIGNMENT-SOLVER-V2-CONTRACT-001`。

随后完成 `ASSIGNMENT-SOLVER-V2-CONTRACT-001`：新增
`docs/architecture/assignment-solver-v2-contract.md`，将下一代 Object Emergence Solver
冻结为显式 cost-softmax assignment system：
`Evidence[N] -> C[N,K] -> A[N,K] -> ObjectState[K]`。v2 首个 contract 使用
`AssignmentEvidenceBatch`，state 包含 `feature_centers`、`position_centers` 和
`slot_bias`，prediction 输出 `assignment`、`slot_mass`、`confidence`、
`mean_normalized_entropy`、`effective_slots` 和 diagnostics。首个 v2 cost 只启用
feature / position / slot bias；mask、temporal、matching、Sinkhorn / OT 和 dynamic-K 均延后。
loss family 被定义为
`L_cluster + L_entropy + L_balance + L_temporal + L_matching + optional supervised CE`，
首个 MVP 只允许 cluster / entropy / balance / optional supervised CE。v1 checkpoint 不自动升级；
后续只能通过 explicit adapter 迁移。下一步进入 `OBJECT-LOSS-V2-001`，先把 object loss 拆成
可独立测试的 loss helper，再实现新的 solver state。

随后完成 `OBJECT-LOSS-V2-001`：新增 `objgauss/core/assignment_losses.py`，将 assignment
object loss 拆成可独立计算和测试的 helper：`assignment_cluster_loss_and_gradient(...)`、
`assignment_entropy_loss_and_gradient(...)`、`assignment_balance_loss_and_gradient(...)` 和
`supervised_assignment_loss_and_gradient(...)`。新增
`assignment_loss_v2_breakdown(...)`，summary 显式输出 cluster、entropy、balance、temporal、
matching、supervised；temporal / matching 当前保持 disabled terms。现有
`solver_decoder_training` 的 supervised CE、entropy、balance 已改为复用 v2 helper，默认训练行为
保持兼容。`objgauss.core` 已暴露 v2 loss helper 和
`validate_assignment_loss_v2_summary(...)`。本切片不启动 GPU 训练、不实现 solver v2 state、不引入
frame-level perception adapter，也不启用 temporal / matching 优化。下一步进入
`ASSIGNMENT-FRAMES-EVIDENCE-001`，补 frame / mask / feature 到 `AssignmentEvidenceBatch`
的最小 adapter contract。

随后完成 `ASSIGNMENT-FRAMES-EVIDENCE-001`：新增 `objgauss/core/assignment_evidence.py`，
定义 `AssignmentEvidenceBatch` 和 `objgauss-assignment-evidence-batch-v1` schema。该 batch
承载 `positions`、`features`、`frame_index`、optional `mask_votes`、optional `track_hints`、
optional `target_assignment` 和 `source`，并提供
`assignment_evidence_from_trainable_frame(...)`、
`assignment_evidence_sequence_from_trainable_frames(...)`、
`assignment_evidence_from_object_emergence(...)`、`validate_assignment_evidence_batch(...)` 和
`validate_assignment_evidence_summary(...)`。现有 `TrainableKernelFrame` 与
`ObjectEmergenceEvidence` 可以无损转换到 v2 evidence contract；mask votes / track hints 只是
optional adapter 字段，不引入 SAM / CLIP / CoTracker 默认依赖。本切片不启动 GPU 训练、不训练
solver v2、不接 renderer loss、不做 dynamic-K。该切片之后进入 `TRAIN-ASSIGNMENT-MVP-001`，
训练 fixed-K assignment MVP，只验证 `A[N,K]` 的 assignment/object loss 能下降。

随后完成 `TRAIN-ASSIGNMENT-MVP-001`：`object-emergence-solver` summary 已嵌入
`objgauss-assignment-mvp-training-v1`，把 fixed-K `AssignmentEvidenceBatch` 和 v2 loss helper
接成 assignment MVP 账面训练闭环。新增 `assignment_mvp_training_summary(...)`，会 replay 初始 /
最终 solver state，并输出 initial / final `assignment_loss_v2_breakdown(...)`；CLI stdout
新增 `assignment_mvp_schema`、`assignment_mvp_loss_decreased` 和
`assignment_mvp_supervised_loss_decreased`。该切片仍不接 gsplat renderer loss、不解冻 renderer
fields、不引入 dynamic-K，也不引入重型 perception 依赖。完成 commit: `3bb9345`。

随后完成 `EVAL-ASSIGNMENT-STABILITY-001`：新增
`objgauss-assignment-stability-eval-v1`，为 Object Emergence Solver checkpoint 提供 assignment
专用只读评估。CLI 新增
`objgauss training eval-assignment <ply> --checkpoint <solver-checkpoint.json>`，会从输入样例构造
`AssignmentEvidenceBatch` 序列，replay solver 得到 `A[N,K]`，投影为 ObjectState，并输出
entropy、assignment confidence、effective slots、slot collapse、object purity、temporal drift
和 ID stability gate。小规模真实样例 smoke 已通过：
`eval_status=assignment_stability_eval_pass`、`slot_collapse=false`、`id_stability=1.0`。该切片不启动
GPU 训练、不接 renderer loss、不做 dynamic-K，eval 输出只写 `/tmp` 或 ignored `outputs/`。
完成 commit: `0375524`。下一步进入 `ASSIGNMENT-RENDER-JOINT-001`，把 assignment stability
before / after gate 接入 renderer joint run。

随后完成 `ASSIGNMENT-RENDER-JOINT-001`：`eval-assignment` 现在可读取
`objgauss-solver-decoder-joint-checkpoint-v1`，`solver-decoder-mvp` summary 新增
`objgauss-solver-decoder-assignment-stability-gate-v1`。普通 joint summary 会写入
`assignment_stability.before` / `assignment_stability.after`；分段 run final summary 会写入
`run_assignment_stability`，覆盖整段 run 的 before / after。CLI stdout 新增 assignment
stability status、before / after status、degraded、after entropy / purity / ID stability，
并新增可选门禁 `--require-assignment-stability-not-degrade`。小规模真实样例 CPU joint smoke
通过：total loss `0.195038 -> 0.190573`，image render loss `0.057252 -> 0.053990`，
assignment stability gate 为 `assignment_stability_gate_ok`，`status_degraded=false`；
同一个 joint checkpoint 可被 `eval-assignment` 单独读取。本切片不改 optimizer、不解冻
per-Gaussian geometry / camera、不引入 dynamic-K。完成 commit: `e8071c6`。下一步进入
`DYNAMIC-K-PROPOSAL-001`，但仍只允许 proposal，不允许自动改写 K。

随后完成 `DYNAMIC-K-PROPOSAL-001`：`assignment_stability` summary 新增 `dynamic_k` 章节，
将现有 `dynamic_k_proposal_report(...)` 接入 assignment eval。输出保持 proposal-only：
`dynamic_k.mode=proposal_only`、`auto_update=false`、`checkpoint_k_mutation=forbidden`。每个
proposal 会记录 `kind`、`source_ids`、`target_id`、`score`、`threshold`、`reason`、
`action` 和 evidence；`eval-assignment` stdout 新增 `dynamic_k_mode`、
`dynamic_k_auto_update`、`dynamic_k_proposal_count` 和 `dynamic_k_proposal_kinds`。真实样例 smoke
通过并输出 `dynamic_k_proposal_kinds=merge_duplicate,split_mixed`。该切片不自动
birth / merge / split、不改变 checkpoint K、不引入 Slot Attention / OT / Sinkhorn、不启动
GPU 训练。完成 commit: `648f723`。至此 object assignment v2 阶段队列已完成到
proposal-only dynamic-K gate；下一阶段需 Owner 确认进入 v2 stability gate / world-model
rollout baseline，或继续扩大真实 perception 输入。

随后完成 `V2-STABILITY-FOUNDATION-002`：新增
`objgauss-v2-stability-foundation-v1`，在 stability gate 前先冻结 evaluation invariant。
核心结论是：同一个 object 由 synthetic oracle label 定义，不由 slot、embedding 或 tracker
推断。新增 `ObjectIdentityOracle`、`ObjectIdentityRecord`、
`ObjectIdentityObservation`，冻结 `oracle_object_id`、`lineage_id`、`canonical_slot`、
per-frame `visible` 和 `expected_slot_relation`；新增 `SyntheticWorldState` /
`SyntheticWorldFrame` / `SyntheticWorldObject`，把 scenario 建模为 observation 前的
object-level world；新增 `ObservationModelConfig`、`SyntheticObservationFrame` 和
`observe_synthetic_world(...)`，将 world 投影为 `AssignmentEvidenceBatch`，同时保留
`oracle_object_ids`、`lineage_ids` 和 `expected_slots`。本切片不启动 GPU 训练、不做
rollout model、不接外部 perception 模型、不自动 birth / merge / split。完成 commit:
`051d667`。下一步进入 `V2-STABILITY-SCENARIO-002`，扩展 cross-view、occlusion recovery、
perturbation 和 adversarial swap 的可复现 synthetic fixtures。

随后完成 `V2-STABILITY-SCENARIO-002`：新增
`objgauss-v2-stability-scenario-fixture-v1`，把 `SyntheticWorldState`、oracle
identity labels、expected slots、visible / occluded transitions 和
`SyntheticObservationFrame` batches 绑定成可复现 scenario fixture。新增
`make_synthetic_stability_scenario_fixture(...)` 和
`make_synthetic_stability_scenario_suite(...)`，覆盖 `cross_view`、
`occlusion_recovery`、`perturbation` 和 `adversarial_swap` 四类场景；其中
`adversarial_swap` 会交换 object 0 / 1 的 appearance feature / RGB，但保留
`oracle_object_id`、`lineage_id` 和 `expected_slot` 不变。`objgauss.core` 已暴露
scenario fixture schema、kind tuple、builder 和 validator。该切片不启动 GPU 训练、
不接 rollout model、不引入外部 perception 依赖、不把 scenario 指标做成最终 gate。
下一步进入 `V2-STABILITY-DIAGNOSTICS-001`，补 failure mode classifier、slot transition
matrix 和 identity confusion graph。

随后完成 `CORE-MODEL-TRAIN-VALIDATE-PLAN-001` docs-only planning：新增
`docs/architecture/core-model-train-validate-plan.md`，把近期算法路线收敛到
“核心模型可训练、可验证、可失败定位”。当前核心模型边界定义为
`Gaussian / AssignmentEvidence -> Assignment Solver v2 -> A[N,K] -> ObjectState ->
Gaussian decoder / renderer loss validation`，先验证 object binding / assignment，不跳到
rollout、identity graph、replay buffer、diffusion 或 self-generated world loop。近期阶段固定为
`V2-STABILITY-DIAGNOSTICS-001 -> V2-STABILITY-GATE-001 ->
ASSIGNMENT-SOLVER-V2-TRAIN-001 -> ASSIGNMENT-SOLVER-V2-EVAL-001 ->
ASSIGNMENT-V2-RENDER-JOINT-001 -> CORE-MODEL-TRAIN-VALIDATE-001`；
`MODEL-V2-TRAINING-ROADMAP-001` 后移到 core model validation 完成之后。本步骤不实现训练代码、
不启动 GPU / renderer training、不引入 diffusion / rollout / replay buffer。

随后完成 `V2-STABILITY-DIAGNOSTICS-001`：新增
`objgauss-v2-stability-diagnostics-v1`，为 synthetic stability fixture 增加
deterministic failure diagnostics。`diagnose_synthetic_stability_fixture(...)` 可消费
`SyntheticStabilityScenarioFixture` 和与 observation 对齐的 predicted slots 或 predicted
assignments，输出 identity-level observations、slot transition matrix、identity confusion
graph、failure mode counts 和 failure events。`FailureModeClassifier` 当前能区分
`slot_swap`、`identity_fragmentation`、`object_merge`、`background_absorption` 和
`temporal_drift`。该切片只做诊断，不训练 solver、不接 renderer loss、不把 diagnostics
变成 hard gate、不改变 dynamic-K proposal-only 约束。下一步进入
`V2-STABILITY-GATE-001`，把 identity invariance 做成 hard gate。

随后完成 `V2-STABILITY-GATE-001`：新增 `objgauss-v2-stability-gate-v1` 和
`objgauss-v2-stability-gate-suite-v1`，将 synthetic oracle identity invariance 固化为
hard gate。`evaluate_synthetic_stability_gate(...)` 复用 diagnostics 输出，并把
expected slot consistency、slot swap、cross-slot drift、adversarial swap exchange、
occlusion recovery return、object merge、background absorption 和 diagnostics reporting
作为 hard checks；assignment entropy、assignment purity 和 temporal coherence 只作为
soft diagnostics，明确不能覆盖 hard gate。该切片不训练 solver、不接 renderer loss、不做
rollout model、不改变 dynamic-K proposal-only 约束。下一步进入
`ASSIGNMENT-SOLVER-V2-TRAIN-001`，在 synthetic fixtures 上实现 fixed-K cost-softmax
assignment solver v2 training。

随后完成 `ASSIGNMENT-SOLVER-V2-TRAIN-001`：新增 dependency-free
`objgauss-assignment-solver-state-v2` / `objgauss-assignment-prediction-v2` /
`objgauss-assignment-solver-v2-training-v1`。`AssignmentSolverV2State` 使用
`feature_centers`、`position_centers` 和 `slot_bias` 计算 cost-softmax assignment
`A[N,K]`；`train_assignment_solver_v2(...)` 复用现有 v2 loss helper，支持 cluster、
entropy、balance 和 supervised CE 权重。synthetic fixture smoke 将 swapped 初始化的
supervised loss 从 `8.569320` 降到 `0.002646`，final assignment 匹配 oracle expected
slots。该切片只做 fixed-K synthetic / CPU NumPy training，不接 GPU、不接 renderer loss、
不启用 temporal / matching loss、不引入 Slot Attention / Sinkhorn / OT、不改变 dynamic-K
proposal-only 约束。

随后完成 `ASSIGNMENT-SOLVER-V2-EVAL-001`：新增
`objgauss-assignment-solver-v2-checkpoint` 和
`objgauss-assignment-solver-v2-stability-eval-v1`。`evaluate_assignment_solver_v2_stability(...)`
会对同一组 synthetic stability fixtures 分别运行训练前 / 训练后 hard gate，并输出 loss
decrease、before / after diagnostics delta、hard blockers、checkpoint summary 和 final state
roundtrip。当前 two-object stability suite smoke 证明 swapped 初始化训练后可从
`synthetic_stability_suite_gate_fail` 变成 `synthetic_stability_suite_gate_pass`，slot swap
failure count 从非零降到 0；eval status 只有在 loss 下降、after hard gate 通过且 checkpoint
roundtrip 通过时才 pass，明确 loss 下降不能替代 identity gate。本切片不接 renderer loss、
不接 GPU、不做 rollout / replay buffer、不改变 dynamic-K proposal-only 约束。下一步进入
`ASSIGNMENT-V2-RENDER-JOINT-001`，把 v2 assignment checkpoint 接回
`A[N,K] -> ObjectState -> Gaussian decoder -> renderer loss` 验证。

随后完成 `ASSIGNMENT-V2-RENDER-JOINT-001`：新增
`objgauss-assignment-v2-render-joint-validation-v1`，把
`objgauss-assignment-solver-v2-checkpoint` 接回
`A[N,K] -> ObjectStateProjection -> ObjectStateGaussianDecoderState -> renderer_api image_render_loss`。
该 validation bridge 使用 uniform assignment 作为 renderer baseline，用 v2 checkpoint final
state 产生 final assignment；默认 decoder colors 从 `target_assignment / target_rgb` 拟合，
避免把 decoder 未训练误判为 assignment failure。summary 输出 initial / final renderer loss、
assignment loss、ObjectState eval、checkpoint roundtrip、identity gate policy 和 frozen fields；
`renderer-loss-contract` 已能消费该 summary，并输出
`assignment_v2_renderer_joint_validation_ready`。本切片不启动 optimizer、不接 GPU 长训、
不解冻 Gaussian geometry / camera / dynamic-K、不把 renderer loss 作为绕过 identity hard gate
的理由。下一步进入 `CORE-MODEL-TRAIN-VALIDATE-001`，汇总核心模型可训练、可验证、可失败定位的
milestone 证据。

随后完成 `CORE-MODEL-TRAIN-VALIDATE-001`：新增
`objgauss-core-model-train-validate-v1` milestone summary，把
`ASSIGNMENT-SOLVER-V2-TRAIN-001`、`ASSIGNMENT-SOLVER-V2-EVAL-001` 和
`ASSIGNMENT-V2-RENDER-JOINT-001` 的 evidence 聚合为同一份可审计报告。gate 覆盖
assignment training loss 下降、synthetic stability hard gate 通过、failure diagnostics
可用、assignment / renderer checkpoint roundtrip、ObjectState eval、renderer joint smoke、
small sample smoke 不退化、renderer-loss-contract 消费 summary，以及 identity gate 未被
renderer loss 覆盖。该 milestone 明确当前 small sample smoke 仍是 dependency-free fixture /
contract proof，promotion 前必须在小型 real / public sample 上重复；同时继续禁止长 GPU 训练、
diffusion / rollout / replay buffer、dynamic-K 自动 mutation 和 Gaussian geometry 解冻。至此
近期路线已推进到“核心模型可训练、可验证、可失败定位”的 development-stage 阶段。

随后完成 `REAL-SAMPLE-V2-SMOKE-001`：新增
`objgauss-real-sample-v2-smoke-v1`，把小型 real / public `object_id` 样例接入
`trainable_kernel_sample_from_cloud -> AssignmentSolverV2 training -> checkpoint ->
AssignmentV2RendererJointValidation -> renderer-loss-contract`。该 smoke 明确把
`object_id` labels 作为训练 target，不声称语义 ground truth，也不使用 fixture oracle；
Gaussian geometry、camera、dynamic-K、rollout / replay buffer 和 GPU 长训仍冻结。仓库内
`public/samples/lego_alpha_v1_objects.ply` 的真实采样 smoke 已暴露核心差距：
v2 supervised loss `1.588851 -> 0.503807`、renderer image loss `0.038943 -> 0.008694`
且 checkpoint roundtrip 通过，但 renderer joint summary 仍为
`assignment_v2_renderer_joint_validation_fail`，失败原因是 ObjectState gate：
`mean_normalized_entropy=0.694399`、`assignment_confidence=0.305601`、
`object_purity=0.623420`，diagnostics 为 `low_assignment_confidence` /
`low_object_purity`。结论：fixture proof 已经推进到真实样例 smoke 入口，但真实样例不能
promote；下一步应先做 real-sample diagnostics / sharpening，而不是解冻 geometry 或进入
diffusion / replay / rollout。

随后完成 `REAL-SAMPLE-V2-DIAGNOSTICS-001`：新增
`objgauss-real-sample-v2-diagnostics-v1`，对同一 public sample 固定采样后扫描 solver
temperature，并把每个候选的 v2 training loss、renderer image loss、ObjectState entropy /
confidence / purity、diagnostics 和 checkpoint roundtrip 写入同一份报告。结论明确：
默认 `solver_temperature=1.0` 失败，`0.75` 仍失败，`0.5` 是当前扫描中最高的通过温度。
`temperature=0.5` 的训练模型验证结果为
`assignment_v2_renderer_joint_validation_pass`、`objectstate_eval_pass`，
`mean_normalized_entropy=0.423937`、`assignment_confidence=0.576063`、
`object_purity=0.815958`，相对 baseline 的 confidence / purity 分别提升
`+0.270462` / `+0.192537`。diagnostics recommendation 为
`temperature_sharpening_sufficient`，建议固化 `solver_temperature=0.5`；当前不需要先做
evidence normalization，也不解冻 geometry / camera / dynamic-K，不进入 diffusion /
replay / rollout。diagnostics summary 已携带 best v2 solver state 小矩阵 checkpoint，可用于
下一步 handoff / resume 验证；训练输出仍不进 git。

随后完成 `REAL-SAMPLE-V2-MODEL-HANDOFF-001`：新增
`objgauss-real-sample-v2-model-handoff-v1` 和
`objgauss-real-sample-v2-effect-preview-v1`，把 diagnostics 选出的
`solver_temperature=0.5` 训练模型导出为可复跑 handoff。CLI 新增
`objgauss training real-sample-v2-handoff`，可从
`public/samples/lego_alpha_v1_objects.ply` 生成 summary、assignment v2 checkpoint 和
HTML/SVG effect preview 到 `/tmp` 或 ignored `outputs/`。handoff 会对 checkpoint 做 JSON
roundtrip，再从 checkpoint restore 后重新运行
`AssignmentV2RendererJointValidation`，验证结果为
`real_sample_v2_model_handoff_pass`、`assignment_v2_renderer_joint_validation_pass`、
`objectstate_eval_pass`，关键指标仍为 `mean_normalized_entropy=0.423937`、
`assignment_confidence=0.576063`、`object_purity=0.815958`。效果展示文件
`/tmp/objgauss-real-sample-v2-handoff-preview.html` 显示 baseline `temperature=1.0`
fail 与 trained `temperature=0.5` pass 的采样 assignment 对比；浏览器验证使用系统
Chrome 截图到 `/tmp/objgauss-real-sample-v2-handoff-preview.png`，确认页面标题正确且有
2 个 SVG 面板。本切片仍不提交 checkpoint / summary / preview 产物，不解冻 geometry /
camera / dynamic-K，不进入 GPU 长训、diffusion、replay 或 rollout。

随后完成 `REAL-SAMPLE-V2-VIEWER-PREVIEW-001`：新增
`objgauss-real-sample-v2-viewer-preview-v1`，把 handoff checkpoint restore 后的
assignment 投影回全量 real public Gaussian PLY，并导出 viewer 可加载的 object-aware
debug PLY。CLI 新增 `objgauss training real-sample-v2-viewer-preview`，可从
`public/samples/lego_alpha_v1_objects.ply` 写出派生 PLY 和 summary 到 `/tmp` 或 ignored
`outputs/`；派生 PLY 保留原始 Gaussian geometry / opacity / scale，新增
`target_object_id`、`target_slot`、`assignment_confidence`、`assignment_entropy`，并把
renderer-facing `object_id` 写成 `argmax(A)` 预测 slot。frontend viewer 新增同源
`?ply=/... .ply` debug route，`/?ply=/samples/objgauss-real-sample-v2-viewer-preview.ply`
可直接进入 ObjectState Debug OS 查看派生点云。本轮验证的 `/tmp` 预览 PLY 覆盖
`5696` 个 Gaussian、`4` 个 predicted objects，`solver_temperature=0.5`、
`mean_normalized_entropy=0.481849`、`assignment_confidence=0.518151`、
`direct_slot_match=0.900281`，但 full-cloud `object_purity=0.758462`，仍触发
`low_object_purity`。结论：训练模型已经能被 3D viewer 看到效果，但全量高斯质量还没有达到
promotion gate；下一步应诊断 sample-to-full-cloud purity gap，而不是解冻 geometry 或进入
GPU 长训、diffusion、replay、rollout。

随后完成 `REAL-SAMPLE-V2-FULL-CLOUD-PURITY-001`：新增
`objgauss-real-sample-v2-full-cloud-purity-v1`，把 full-cloud viewer preview 的对象分割
目标覆盖做成可复跑扫描。CLI 新增
`objgauss training real-sample-v2-full-cloud-purity`，固定 public
`lego_alpha_v1_objects.ply` 时默认比较 `max_points=24/64/128`，并把最佳候选导出为
viewer 可加载 debug PLY。当前结果显示：`max_points=24` 仍为
`object_purity=0.758462`、`direct_slot_match=0.900281`、`low_object_purity`；
`max_points=64` 小幅改善到 `object_purity=0.765805`、`direct_slot_match=0.912746`；
`max_points=128` 通过 full-cloud gate，`solver_temperature=0.35`、
`mean_normalized_entropy=0.325897`、`assignment_confidence=0.674103`、
`object_purity=0.853079`、`direct_slot_match=0.989642`。结论：本 public sample 的主要差距
来自采样覆盖不足，下一步应把阶段分割对象目标临时设为 `max_points=128` 后做视觉质量检查和
误分割定位；当前不需要 evidence normalization，不解冻 geometry / camera，不进入 GPU 长训、
diffusion、replay 或 rollout。

随后完成 `REAL-SAMPLE-V2-SEGMENTATION-QUALITY-001`：新增
`objgauss-real-sample-v2-segmentation-quality-v1`，把 `max_points=128` 的 full-cloud
对象分割结果拆成 per-object counts、target-vs-predicted confusion、confidence / entropy
分布和优化建议。CLI 新增
`objgauss training real-sample-v2-segmentation-quality`，可生成 viewer PLY 与 summary 到
`/tmp` 或 ignored `outputs/`。当前真实样例 hard argmax 分割已通过：
`direct_slot_match=0.989642`、`hard_argmax_object_purity=0.989642`、
`min_predicted_object_purity=0.971724`、`min_target_recall=0.910499`、`mixed_gaussians=59`。
具体弱点是 `target_slot=1` 有 `52` 个 Gaussian 漏到 `predicted_object=2`；同时
`predicted_object=1` 置信度均值只有 `0.583306`、熵均值 `0.599941`，触发
`low_confidence_predicted_object` / `high_entropy_predicted_object`。包含
`temperature=0.25` 的复查仍选择 `0.35`，结论是不继续 sharpening，也不继续加覆盖；下一步应
围绕 slot 1/2 边界做 evidence normalization 或局部 export policy 诊断。仍不提交 generated
PLY / summary / screenshot，不解冻 geometry / camera，不进入 GPU 长训、diffusion、replay
或 rollout。

随后完成 `REAL-SAMPLE-V2-WEAK-BOUNDARY-OPT-001`：新增
`objgauss-real-sample-v2-weak-boundary-opt-v1`，在固定 `max_points=128` 和
`solver_temperature=0.35` 下比较 baseline cost weights 与一个最小候选
`feature_weight=2.0, position_weight=1.0`。该候选只改变 assignment v2 预测时的
cost 权重配置，不改 checkpoint，不使用 target labels 做推理。真实 public sample 结果为：
baseline `mixed_gaussians=59`、`direct_slot_match=0.989642`、
`min_target_recall=0.910499`；candidate `mixed_gaussians=0`、
`direct_slot_match=1.0`、`hard_argmax_object_purity=1.0`、
`min_predicted_object_purity=1.0`、`min_target_recall=1.0`。变更的 `59` 个 Gaussian 为
`52` 个 `baseline_object_id=2 -> candidate_object_id=1` 和 `7` 个
`baseline_object_id=3 -> candidate_object_id=0`；导出的候选 PLY `object_id` counts 为
`736/581/1787/2592`，并附带 `baseline_object_id`、
`baseline_assignment_confidence`、`baseline_assignment_entropy`、
`weak_boundary_candidate` 和 `boundary_changed` audit 字段。CLI
`objgauss training real-sample-v2-weak-boundary-opt` 可生成 `/tmp` preview PLY / summary；
Playwright + system Chrome 已验证 `/?ply=/samples/objgauss-real-sample-v2-weak-boundary-opt.ply`
desktop / mobile 可加载 `ply-url-artifact`，ObjectState source 为
`derived_from_object_id`，4 个对象开关分别为 `736/581/1787/2592`，对象 #1 的 `581`
个 Gaussian 可隐藏并恢复。结论：下一步应把该 weighted candidate 作为受控 viewer
preview / handoff promotion 路径，而不是继续 coverage / sharpening 或进入 geometry /
camera unfreeze、GPU 长训、diffusion、rollout、replay buffer、dynamic-K mutation。

随后完成 `REAL-SAMPLE-V2-WEIGHTED-VIEWER-PREVIEW-001`：`real-sample-v2-viewer-preview`
现在默认使用 promoted full-cloud assignment weights
`feature_weight=2.0, position_weight=1.0`，并把 `max_points` 默认提升到 `128`。
底层 builder 保留受控参数；未显式传入时仍可使用 checkpoint 原权重，避免影响
`full-cloud-purity` 等历史诊断路径。viewer preview summary 新增
`assignment_weight_policy` 与 `projection.hard_segmentation`，显式记录 baseline weights、
promoted weights、`uses_target_labels_for_prediction=false`、`mutates_checkpoint=false`、
object counts 和 `mixed_gaussians`。真实 public sample CLI 默认输出：
`recommended_solver_temperature=0.35`、`mixed_gaussians=0`、`object_id_counts=0:736,1:581,2:1787,3:2592`、
`full_cloud_entropy=0.120526`、`full_cloud_confidence=0.879474`、
soft `object_purity=0.951687`、hard `direct_slot_match=1.0`，quality diagnostics 为
`none`。Playwright + system Chrome 已验证
`/?ply=/samples/objgauss-real-sample-v2-weighted-viewer-preview.ply` desktop / mobile 可加载
`ply-url-artifact`，4 个对象开关为 `736/581/1787/2592`，object #1 的 `581` 个 Gaussian
可隐藏并恢复。该步骤不改 handoff checkpoint schema，不提交 generated PLY / summary /
screenshot，不改变 public demo / HF release 口径，不解冻 geometry / camera，不进入 GPU 长训、
diffusion、rollout、replay buffer 或 dynamic-K mutation。

随后完成 `REAL-SAMPLE-V2-PROMOTED-WEIGHTS-CROSS-SAMPLE-001`：新增
`objgauss-real-sample-v2-promoted-weights-cross-sample-v1`，把 promoted weights
`feature_weight=2.0, position_weight=1.0` 与 baseline `1.0/1.0` 固化为可复跑
cross-sample diagnostic。CLI 新增
`objgauss training real-sample-v2-promoted-weights-cross-sample`，复用同一个
real-sample v2 handoff checkpoint，对第二样例做 baseline / promoted full-cloud viewer
preview 对比，并导出 promoted PLY audit fields：
`baseline_object_id`、`baseline_assignment_confidence`、`baseline_assignment_entropy`、
`promotion_changed`、`promotion_hard_fix`、`promotion_hard_regression`。Polyhaven Chair
样例 `public/samples/polyhaven_chair_demo_objects.ply` 为 `50,000` Gaussians / 6 个
`object_id`，baseline hard metrics 为 `mixed_gaussians=3840`、
`direct_slot_match=0.923200`，promoted hard metrics 为 `mixed_gaussians=3918`、
`direct_slot_match=0.921640`；虽然 soft metrics 改善
`object_purity=0.844726 -> 0.905148`、confidence `0.792748 -> 0.918182`、
entropy `0.207252 -> 0.081818`，但 hard delta 为
`mixed_gaussians_delta=78`、`direct_slot_match_delta=-0.001560`、
`hard_fix_count=1736`、`hard_regression_count=1814`。补充本地 Plush 复查同样显示
soft metrics 改善但 hard boundary 回退：baseline `mixed_gaussians=3849`、
`direct_slot_match=0.986327`，promoted `mixed_gaussians=6283`、
`direct_slot_match=0.977680`。结论：promoted weights 是当前 Lego viewer preview 的
sample-specific 候选，不是已通过跨样例 hard-boundary 非回归的全局默认策略；下一步应做
sample-aware weight policy / evidence normalization gate，不应解冻 geometry / camera 或进入
GPU 长训、diffusion、rollout、replay buffer、dynamic-K mutation。本步骤验证通过：
`uv run --extra dev pytest` 256 passed，`npm run build` passed，`git diff --check` passed；
代码完成 commit 为 `d8d6b2f`。

随后完成 `REAL-SAMPLE-V2-SAMPLE-AWARE-WEIGHT-POLICY-001`：新增
`objgauss-real-sample-v2-sample-aware-weight-policy-v1`，把上一张的 cross-sample
negative evidence 收敛为 per-sample gate。CLI 新增
`objgauss training real-sample-v2-sample-aware-weight-policy`，在同一个 handoff
checkpoint 下评估 baseline `feature_weight=1.0, position_weight=1.0` 与 promoted
`feature_weight=2.0, position_weight=1.0`，只有 promoted 同时满足 hard mixed 非回归、
direct slot match 非回归、对象数稳定和 soft purity 非回归时才选 promoted；否则选 baseline
并把 `evidence_normalization_gate.status` 标为
`required_before_global_weight_promotion`。该 CLI 导出 selected policy PLY，并添加
`sample_aware_baseline_object_id`、`sample_aware_baseline_confidence`、
`sample_aware_baseline_entropy`、`sample_aware_selected_index`、`sample_aware_changed`、
`sample_aware_hard_fix`、`sample_aware_hard_regression` audit fields。Polyhaven Chair
结果选择 baseline：baseline `mixed_gaussians=3840`、`direct_slot_match=0.923200`；
promoted 被 gate 阻断，因 `mixed_delta=78`、`direct_delta=-0.001560`、
`hard_fix=1736`、`hard_regression=1814`，虽然 soft purity / confidence 改善。Lego
结果选择 promoted：baseline `mixed_gaussians=59`、`direct_slot_match=0.989642`；
promoted `mixed_gaussians=0`、`direct_slot_match=1.000000`、`hard_fix=59`、
`hard_regression=0`。本步骤不改变既有 `real-sample-v2-viewer-preview` 默认、不改 public
demo / HF release / renderer / manifest / checkpoint schema，也不表示 evidence normalization
数学已实现。验证通过：targeted pytest 14 passed，`uv run --extra dev pytest` 259 passed，
`npm run build` passed，`git diff --check` passed；代码完成 commit 为 `45afd2d`。

随后完成 `BOUNDED-EVIDENCE-NORMALIZATION-001`：`real-sample-v2-sample-aware-weight-policy`
在 baseline / promoted 之外新增 `bounded-normalized` 候选。该候选根据 promoted 相对
baseline 的 hard fix / hard regression 计算 `feature_weight_blend` /
`position_weight_blend`，把权重限制在 baseline 和 promoted 之间，并在 summary 中记录
bounded confidence gain、entropy reduction 和 purity gain；prediction 仍不使用 target
labels，target labels 只用于 gate / summary 的 hard-boundary 验收。Lego 仍选择
`promoted`：`feature_weight=2.0`、`mixed_gaussians=0`、
`direct_slot_match=1.000000`、`hard_fix=59`、`hard_regression=0`。Polyhaven 现在选择
`bounded-normalized`：`feature_weight=1.0`、`position_weight=1.0`、
`selection_reason=bounded_evidence_normalization_safe_fallback`、
`evidence_normalization_status=satisfied_by_bounded_normalization`、selected
`hard_regression=0`，并记录 promoted soft evidence：
`bounded_confidence_gain=0.125433`、`bounded_entropy_reduction=0.125433`；blocked
promoted candidate 仍记录 `mixed_delta=78`、`direct_delta=-0.001560`、`hard_fix=1736`、
`hard_regression=1814`。补充只读扫描显示
Polyhaven `feature_weight=1.05..1.25` 虽可改善总体 mixed/direct，但仍产生 hard
regression，因此当前 viewer export 不选择这些中间权重。本步骤不改变 ObjectState /
manifest / checkpoint ABI，不解冻 geometry / camera / dynamic-K，不引入 diffusion /
rollout / replay buffer。

随后完成 `REAL-SAMPLE-V2-AUTO-LOAD-VIEWER-001`：viewer catalog 新增本机
`real-sample-v2-sample-aware-lego` 预览模型，默认指向 ignored
`public/samples/objgauss-real-sample-v2-sample-aware-lego.ply`，让根路径 UI 优先展示
sample-aware promoted Lego 分割效果，而不是停在 Trainable MVP fixture 日志。该本地
PLY 由
`objgauss training real-sample-v2-sample-aware-weight-policy public/samples/lego_alpha_v1_objects.ply`
生成，本轮结果为 selected `promoted`、`mixed_gaussians=0`、`direct_slot_match=1.000000`、
`hard_fix=59`、`hard_regression=0`；生成 PLY 被现有 `*.ply` ignore 规则排除，不提交。
缺少本地 PLY 时，该 catalog 条目会标记为 `skipped` 并回退选中 `lego-alpha`，避免其他
开发机首屏坏默认。Playwright + system Chrome 已验证 `http://127.0.0.1:5395/` 根路径
桌面与移动端首屏均选中 `real-sample-v2-sample-aware-lego`，模型 dock 可切到
`lego-alpha` 再切回，canvas 元素截图像素非空；截图保存在
`/tmp/objgauss-real-sample-v2-auto-load-{desktop,canvas,mobile}.png`。验证通过：
`uv run --extra dev pytest` 259 passed，`npm run build` passed（保留既有 Vite chunk
size warning），`git diff --check` passed；代码完成 commit 为 `36731a8`。

随后完成 `REAL-SAMPLE-V2-BOUNDED-NORM-CROSS-SAMPLE-001`：新增
`objgauss-real-sample-v2-bounded-normalization-cross-sample-v1` 汇总报告和
`objgauss training real-sample-v2-bounded-normalization-cross-sample` CLI。该报告复用
单样例 `real_sample_v2_sample_aware_weight_policy_from_cloud`，按多样例 rows 汇总
selected policy、promoted / bounded-normalized candidate、evidence normalization status
和 aggregate gate；gate 要求样例数达到 `min_samples`、所有单样例 policy pass，且
selected policy 的 hard regression 总和为 `0`。Lego + Polyhaven 复验通过：Lego selected
`promoted`，`mixed_gaussians=0`、`hard_fix=59`、selected `hard_regression=0`；Polyhaven
selected `bounded-normalized`，`mixed_gaussians=3840`、selected `hard_regression=0`，
同时保留 promoted blocked evidence `hard_regression=1814`。Aggregate 结果为
`selected_policy_counts={"bounded-normalized":1,"promoted":1}`、
`selected_hard_regression_count=0`、`blocked_promoted_samples=["polyhaven"]`；结论是继续使用
sample-aware policy 并增加小型 real / public sample 行，而不是把单一 promoted weight
设为全局默认。本步骤不改 ObjectState / manifest / checkpoint ABI，不解冻 geometry /
camera / dynamic-K，不引入 diffusion / rollout / replay buffer，不导出新的 demo PLY。
验证通过：targeted pytest 14 passed，CLI cross-sample smoke passed，
`uv run --extra dev pytest` 261 passed，`npm run build` passed（保留既有 Vite chunk
size warning），`git diff --check` passed。

随后完成 `REAL-SAMPLE-V2-CROSS-SAMPLE-EXPANSION-002`：cross-sample 表从 Lego +
Polyhaven 扩展到 Lego + Polyhaven + Nike 三行，并收紧 sample-aware candidate gate。非
baseline 候选现在必须满足 `hard_regression_count == 0` 才能 eligible，candidate gate
新增 `hard_regression_free`；`feature_weight_blend=0` / `position_weight_blend=0` 的
`bounded-normalized` 不再 eligible，因为它与 baseline 等价，不能算 evidence
normalization 已满足；`evidence_normalization_gate` 也把任意 hard regression 视为 soft
sharpening blocker，不再只在 `hard_regression > hard_fix` 时阻断全局 promotion。
3-row pass 表通过：Lego selected `promoted`；Polyhaven selected `baseline`；Nike selected
`baseline`，`source_gaussians=270491`、selected `mixed_gaussians=16721`、selected
`hard_regression=0`，promoted candidate 被阻断并记录 `hard_regression=6671`。Aggregate 结果为
`selected_policy_counts={"baseline":2,"promoted":1}`、
`selected_hard_regression_count=0`、`blocked_promoted_samples=["polyhaven","nike"]`。
Plush KMeans 作为 blocked negative evidence：旧 gate 会选择 promoted，但其
`hard_regression=2746`；strict gate 后该样例没有安全 selected policy，CLI 返回
`no sample-aware candidate passed the gate`，不进入 pass 表。结论仍是继续 sample-aware
policy + 增加样例行，不把单一 promoted weight 设为全局默认。验证通过：targeted pytest
15 passed，`uv run --extra dev pytest` 262 passed，`npm run build` passed（保留既有
Vite chunk size warning），`git diff --check` passed。

随后完成 `MODEL-CATALOG-LATEST-SLOT-ORDER-001`：viewer catalog 新增
`displaySlot`、`displayOrder` 和 `updatedAt` 规则，把同一模型定位下的候选按最新在前
排序。当前 `real-sample-v2-sample-aware-lego` 与 `lego-alpha` 同属
`lego-object-segmentation-preview` 定位，dock 顺序固定为最新的 `真实样例 V2` 在前，
旧的 `Lego alpha` 紧随其后；默认模型也从该定位下按 `updatedAt` 取最新，而不是硬编码
具体 id。显式 URL 注入的 trainable / PLY / manifest / OGC route 仍保持更高优先级。
验证通过：Node catalog 断言通过，`uv run --extra dev pytest` 259 passed，
`npm run build` passed（保留既有 Vite chunk size warning），`git diff --check` passed；
Playwright + system Chrome 验证 `http://127.0.0.1:5395/` 首屏选中最新模型，dock 前两项
为 `真实样例 V2` / `Lego alpha`；截图为 `/tmp/objgauss-latest-model-order.png`。
代码完成 commit 为 `b7c1d35`。

随后完成本地 `OBJECT-EDIT-UX-PRIMARY-FLOW-001` 切片：viewer 首屏从开发者调试入口收敛为
训练阶段的高斯云展示台。多个模型被摆在同一 stage 上用于查看训练 / 分割效果，而不是被表述为
最终生产编辑器；顶栏口径改为 `高斯云训练展示台`，指标收敛为
`Three.js / 展示版本 / 对象层`，明确所有对象处理都先建立在 Three.js 加载并展示模型之上。
顶栏不再展示 `导入训练`、`导入模型`、`导入OGC` 和 `A[N,K]`，只保留 `重置视角`；
artifact / model / OGC 导入下沉到 `协议与归档` 的高级导入区，原有 `data-*` audit hooks
保持不变。左侧主面板改为 `世界操作`，只默认展开 `Three.js 世界`、`对象交互` 和
`模型版本`；assignment、Gaussian probe、对象诊断、对象开关、协议归档、训练 / 基准证据统一
下沉到 `系统工具` 高级抽屉。底部状态条已删除，右侧 inspector 只保留必要的模型 / 对象元数据。
选中对象后 Three.js `TransformControls` 自动挂载
`three-transform-controls-v1` 平移 gizmo；方向按钮只作为对象交互面板内的辅助移动控件，
`object-group-position-v1` audit contract 继续保留。`ThreeWorld` 同时新增
`object-transform-state-v1` 交互状态栈：按钮移动、直接拖拽和 TransformControls 拖拽都会记录
object group transform 前后状态；TransformControls 支持移动 / 旋转 / 缩放模式切换，并支持
undo / redo、active transform cancel 和 Shift 临时 snap。对象交互面板新增移动 / 旋转 / 缩放
模式图标按钮，以及 undo / redo / cancel 图标按钮，快捷键能力不作为主界面文案展示，但通过 audit
handle 暴露可验证状态。该切片同时补入训练阶段的
`模型版本处理` 面板：每个模型版本可用复选框加入 / 移出展示台，批量按钮支持
`对象层`、`同定位`、`全部`；已加载对象层的版本显示 `跳过`，已有对象层但未加载时显示
`加载`，未分割输入保留 `生成` 状态入口。`ThreeWorld` 现在支持模型级 stage visibility，
`window.__OBJGAUSS_WORLD__` 暴露 `stageModelIds`、`visibleModelCount`、
`modelVisibilitySamples`、`objectTransformContract`、`objectTransformEngine`、
`transformGizmoObject`、`transformHistoryDepth` 和 `transformRedoDepth` 以审计多版本展示
和 3D 对象交互状态。UX 审计截图保存在 `/tmp/objgauss-ux-audit/`，训练展示台模型版本验证截图为
`/tmp/objgauss-training-stage-version-panel.png`；本轮 Three.js-first 对象交互验证截图为
`/tmp/objgauss-object-interaction-desktop.png`、`/tmp/objgauss-object-interaction-mobile.png`、
`/tmp/objgauss-object-interaction-canvas-desktop.png` 和
`/tmp/objgauss-object-interaction-canvas-mobile.png`。验证通过：`npm run build` passed（保留既有
Vite chunk size warning）、Playwright + system Chrome desktop/mobile targeted check passed，
并验证 `object-transform-state-v1` 的 move / undo / redo / snap / cancel 合同；PNG 像素统计确认隐藏 HUD 后 canvas 非空；此前同一切片已跑
`uv run --extra dev pytest` 259 passed 和 `git diff --check` passed。该切片只做 viewer
handoff UI，未接真实后端分割任务执行；随后 `GAUSSIAN-OBJECT-PROCESS-FLOW-001` 已把
`未分割高斯云 -> 生成对象层 -> 选中/移动对象` 收敛为 viewer 内 CLI handoff 主流程入口。

## 架构重梳理基线

2026-07-02 已按 Owner 新方向建立重构规划基线，事实源为
`docs/architecture/rebuild-plan.md`。新的产品边界是：

- 前端作为体验、交互和渲染层，负责加载模型、展示 3DGS 外观、对象选择 /
  隐藏 / 隔离 / 删除预览、渲染路线和性能 telemetry；同时保留并继续迭代
  ObjGauss 自有 Gaussian 渲染算法，包括 Gaussian OIT、WebGPU tile / compute
  renderer、object-state buffer、shader、picking 和 Spark bridge。
- 后端 / pipeline 层负责提供 3D 模型资产、登记外部训练输出、运行对象级处理
  pipeline、产出 browser-ready artifact、manifest、hash、质量报告和后续服务接口。
- 核心算法层优先抽取 Gaussian 数据模型、PLY / `.splat` IO、feature clustering、
  Object Field、mask manifest、projection voting、cross-view slot alignment、
  CLIP / semantic scoring adapter 和 promotion / evaluation policy。

同日 Owner 将 token-system 讨论收敛为 ObjGauss v1 Kernel Contract，事实源为
`docs/architecture/objgauss-v1-kernel-contract.md`。该 contract 冻结三层链路：
`PerceptionEvidence -> ObjectState -> GaussianToken`，其中 `ObjectState` 是唯一
核心 reasoning unit；dynamics 降级为 `ObjectState` 的时间字段或状态历史，不再作为
并列 token 系统。`docs/myobjgausstoken/` 下的原始讨论保留为 Research Spec，不作为
Architecture Spec。该归档不引入新训练依赖、不重构 renderer、不发布素材。

同日进一步将 v1 的 `PerceptionEvidence -> ObjectState` 阶段规划为 Object Emergence
Plan，事实源为 `docs/architecture/objgauss-v1-object-emergence-plan.md`。该规划把
slot / clustering / tracking 收敛为同一个 assignment matrix `A` 的不同约束或视图；
v1 先走 fixed-K 的 `A[N,K] + evidence[N] -> ObjectState[K]`，再做 stability metrics、
temporal matching、Gaussian delivery binding，最后才进入 birth / merge / split proposal。
该规划仍为 docs-only，不新增 `objgauss/core/object_state.py`，不引入新 ML 依赖。

同日 Owner 进一步要求中文化并补齐 KERNEL-001 solver 规范。上述两个 kernel 文档现已
以中文写作，并在 Object Emergence Plan 中明确 cost matrix `C[N,K]`、assignment
solver 顺序、object pooling、`L_object` 组成、非训练闭环和后续 trainable loop 边界。
关键约束同步为：`object_id` 是从 `A` / matching / export policy 派生的 renderer
address，不是 primary state；`ObjectState` 实现层可拆成 semantic / geometric /
temporal 子状态，但对外仍是单一 reasoning unit。本步骤仍未改代码、未引入 Sinkhorn /
Hungarian / Gumbel-softmax 依赖、未改 renderer 或素材。

随后已完成 `ALGOMODEL-SOLVER-ABI-001`：`docs/architecture/object-emergence-model-v1.md`
明确算法模型主线先训练 Object Emergence Solver，而 full renderer training 继续等待
torch / gsplat / CUDA 环境。`objgauss/core/object_emergence_solver.py` 新增
dependency-free solver ABI，定义 `ObjectEmergenceEvidence`、
`ObjectEmergenceSolverConfig`、`ObjectEmergenceSolverState` 和
`ObjectEmergenceAssignmentPrediction`，并提供 `evidence_from_gaussian_cloud(...)`、
`initialize_object_emergence_solver(...)`、`predict_object_emergence_assignment(...)` 和
`project_object_emergence_prediction(...)`。当前 solver 是线性 softmax assignment
state，可从 Gaussian evidence 生成 normalized `A[N,K]` 并复用现有 ObjectState
projection；它还不是 optimizer，不更新权重，不自动执行 dynamic-K，不引入 torch /
gsplat / CUDA，不改 viewer renderer 或 artifact schema。下一步算法 PR 是
`TRAINABLE-SOLVER-NP-001`。

随后已完成 `TRAINABLE-SOLVER-NP-001`：`objgauss/core/object_emergence_solver.py`
现在可用 dependency-free NumPy 有限差分训练 `ObjectEmergenceSolverState` 的 feature
weights、position weights 和 bias。训练 loss 是 pre-render solver loss：
`L_assignment + L_entropy + L_balance + L_temporal`，其中 `L_render` /
`image_render_loss` 仍保留给后续 full renderer loss producer。新增
`object_id_targets_from_cloud(...)` 和 CLI
`objgauss training object-emergence-solver <ply>`，可从 object-aware PLY 生成 one-hot
assignment target 并训练 solver weights。已用
`public/samples/lego_alpha_v1_objects.ply` 的 16 点采样验证：
`initial_total_loss=1.386400 -> final_total_loss=0.767572`，且 summary 明确
`gpu_used=false`、`vram_reserve_gb=1`。同一架构文档已记录 GPU 训练管道策略：
后续进入 GPU / full renderer training 前必须 preflight torch / gsplat / CUDA /
`nvidia-smi`，并固定预留 1GB GPU 显存；当前会话不把
`TRAIN-GSPLAT-MVP-001` 从 suspended 改为 done。本切片不训练 Gaussian geometry /
opacity / rotation，不自动执行 dynamic-K，不提交 checkpoint / artifact / rendered
image 或大资产。

同日已完成 `DYNAMIC-K-UPDATE-001`：`objgauss/core/object_state.py` 新增
`DynamicKUpdateAction`、`DynamicKUpdatePlan` 和 `dynamic_k_update_plan(...)`，将
`dynamic_k_proposal_report(...)` 的 remove / merge / split / birth proposal 转成
epoch-boundary 才能应用的 gated update plan。该 plan 会输出 accepted / blocked action、
`slot_delta`、`next_slot_count`、block reason 和 diagnostics；它不在 gradient step 中
修改 `K`，不静默改写 object id，也不直接改变当前 projection / artifact。dynamic-K
因此仍是可审计的状态更新计划，不是自动在线 slot policy。

同日已完成 `SOLVER-LOSS-UI-001`：`public/models/trainable-mvp-debug/model-artifact.json`
现在携带最小 `objgauss-object-emergence-solver-training-v1` CPU solver training summary；
`src/App.jsx` 在 ObjectState Debug OS 中新增 Solver loss 面板，并把 total /
assignment loss delta、slots、sample count、`gpu_used=false` 和 `vram_reserve_gb=1`
同步到 root telemetry、Debug panel、debug snapshot 和 `window.__OBJGAUSS_WORLD__`。
`scripts/audit-world-viewer.mjs` 已校验 solver loss 下降、assignment loss 下降、GPU 未使用
和 1GB 显存预留策略。该步骤只做训练结果 handoff / 可视审计，不启动 torch / gsplat /
CUDA full renderer training，不提交 checkpoint 或大训练产物。

同日推进 `SOLVER-CHECKPOINT-EXPORT-001`：`objgauss/core/object_emergence_solver.py`
新增 `objgauss-object-emergence-solver-checkpoint-v1` contract，可将 CPU solver 训练后的
`ObjectEmergenceSolverState` 权重、config、loss、source metadata 和 GPU policy 导出为
可 roundtrip 的 checkpoint JSON。`objgauss training object-emergence-solver` 新增
`--checkpoint-output`，训练 smoke 可同时写 summary 和 checkpoint 到 `/tmp` 或 ignored
`outputs/`。`renderer-loss-contract` 现在能识别 solver training summary / checkpoint，
输出 `status=object_emergence_solver_ready`，并明确下一阶段仍被
`solver_checkpoint_not_bound_to_gaussian_decoder` 和
`solver_checkpoint_not_bound_to_renderer_loss` 阻塞。本切片仍不启动真正 torch / gsplat /
CUDA full renderer training，也不提交 checkpoint 产物。

随后完成 `DECODER-HANDOFF-CONTRACT-001`：`renderer-loss-contract` 输出现在包含
`objgauss-decoder-renderer-handoff-v1`，把 solver checkpoint 到 full renderer loss 的
边界固定为 `solver_checkpoint -> PerceptionEvidence -> A[N,K] -> ObjectStateProjection
-> GaussianToken decode -> renderer_api image_render_loss`。CLI 会打印
`decoder_handoff_status` 和 `decoder_handoff_starts_real_training=false`，测试覆盖
`awaiting_solver_checkpoint`、`solver_checkpoint_ready`、`renderer_api_decoder_smoke_ready`
和 `full_renderer_decoder_ready` 四类状态。该合约只是训练前置 handoff，不启动 GPU
训练，不绑定真实 Gaussian decoder 参数，也不把 point / CPU smoke 伪装成 full 3DGS
renderer training。

随后完成 `TRAIN-GSPLAT-MVP-001` 环境恢复与最小 full renderer smoke：host GPU preflight
通过，`nvidia-smi` 显示 RTX 5060 Ti / driver `595.71.05` / CUDA `13.2`；临时 uv
环境加载 `torch 2.12.1+cu130`、`gsplat 1.5.3` 并确认 CUDA 可用。由于默认 Codex
沙箱 `/dev` 不暴露 `/dev/nvidia*`，GPU 命令需要 host shell 或提权命令执行。复用
`/tmp/objgauss-cuda13` wrapper 和 CUDA 13.0 uv package set 后，显式
`--image-renderer gsplat` 的 2-iteration / 4-point `kernel-sample` smoke 通过：
`renderer_api_status=ready`，`renderer_name=gsplat-rasterization-v1`，
`initial_total_loss=1.651442 -> final_total_loss=1.584998`，
`initial_image_render_loss=0.319775 -> final_image_render_loss=0.319773`。对应
`renderer-loss-contract` 输出 `status=full_3dgs_renderer_ready`、`upgrade_blockers=[]`、
`decoder_handoff_status=full_renderer_decoder_ready`。本步骤不把 torch / gsplat 加入基础
dependencies，不提交 `/tmp` summary / checkpoint / rendered image / ignored
`outputs/` 产物，不训练 Gaussian geometry / opacity / rotation，也不替换 viewer
renderer。

同日已完成 `DEBUG-UI-HIERARCHY-001`：ObjectState Debug OS 左侧调试面板从平铺信息流
收敛为可折叠目录结构，默认打开概览、常用操作、Assignment / Gaussian、对象诊断和
对象开关，低频的协议归档、质量 / 训练 / 基准默认折叠；可见字段同步中文化。右侧
floating inspector 改为窄版，并新增收起 / 展开机制，收起后仅保留对象标题和 kind。
本切片只改 viewer UI 与 Playwright audit 对折叠目录的展开步骤，不改变 ObjectState
算法、assignment projection、manifest / artifact contract、OGC loader 或 renderer。

同日已完成 `OBJECT-ASSIGNMENT-001`：`objgauss/core/object_state.py` 新增 Phase 1
Object Field Projection Layer。该实现验证 normalized `A[N,K]`，复用
`ObjectField.probabilities()`，将 Gaussian evidence weighted reduction 为
`ObjectState[K]`，输出 `slot_mass` / `confidence`、`mass_fraction`、
`assignment_entropy`、`centroid`、`bbox`、`feature`、status diagnostics 和派生
`object_id` export address。`objgauss.core` lazy namespace 已暴露 `ObjectState`、
`ObjectStateProjection`、`project_object_states(...)`、
`project_object_states_from_field(...)` 和 `validate_assignment_matrix(...)`。测试覆盖
empty、uniform、sparse、single-dominant 和 noisy assignment failure modes。该切片不做
temporal matching、不做 dynamic K、不引入 Slot Attention / Sinkhorn / SAM / DINO /
CoTracker / Mamba，不改 renderer 或素材。

同日已完成 `OBJECT-STABILITY-001`：`objgauss/core/object_state.py` 在 Phase 1
`ObjectStateProjection` 上新增 `ObjectStabilityReport` 和
`object_state_stability_report(...)`。该报告计算 `assignment_confidence`、
`mean_normalized_entropy`、`slot_mass`、`slot_mass_fraction`、`effective_slots`、
inactive / low-confidence / mixed slots、single-slot collapse、dominant slot 和可选
`purity_labels` 的 object purity / per-slot purity。`objgauss.core` lazy namespace 已暴露
`ObjectStabilityReport` 与 `object_state_stability_report(...)`。测试覆盖 healthy sparse
assignment、uniform collapse risk、single dominant slot、noisy high-entropy assignment、
empty evidence 和 purity label validation。本切片仍不做 temporal matching、不自动修改
dynamic K、不放宽 semantic promotion threshold、不改 renderer 或素材。

同日已完成剩余 v1 kernel 非训练阶段：`OBJECT-TEMPORAL-MATCH-001`、
`OBJECT-GAUSSIAN-BINDING-001` 和 `OBJECT-DYNAMIC-K-PROPOSAL-001`。`objgauss/core/object_state.py`
现在提供 `match_object_states(...)` / `ObjectTemporalMatchReport`，使用 centroid、bbox、
feature 和 mass cost 做 dependency-free greedy matching，显式输出 matched pairs、
unmatched previous / current、ignored inactive states、cost matrix 和 temporal drift；
slot permutation 不再依赖 hard id equality。该模块同时提供
`object_state_delivery_summary(...)` 和 `bind_object_states_to_artifact(...)`，把
`ObjectStateProjection` 汇总为 renderer-facing metadata：derived `object_id` 来源、
Gaussian children count、per-state summary、stability 摘要和可选 chunk binding；viewer
仍消费 Gaussian artifact，不消费 ObjectState tensor，也不改变 diagnostic full PLY 的
browser-ready 规则。最后新增 `dynamic_k_proposal_report(...)` /
`DynamicKProposalReport`，只输出 remove inactive、split mixed、merge duplicate 和
birth unmatched proposal，不自动修改 `K`、不改写 object ids、不启动模型训练。本轮仍不引入
CoTracker / learned tracker / Sinkhorn / Slot Attention，不改 renderer、不移动素材。

同日已完成 `WORLD-PRUNE-001`：默认前端入口已收敛到 Three.js / VR-like world viewer。
`package.json` 现在只保留 `dev` / `build` / `preview`、`audit:world-viewer`、OGC /
WebGPU 核心算法验证、semantic acceptance、资产采样、训练和 benchmark 入口；旧产品 UI 的
`audit:demo`、renderer-route、commercial demo、object-boundary、Spark route 等 npm
快捷入口已从默认 package scripts 移除。历史 audit 脚本文件暂保留在 `scripts/`，用于
旧报告复现和后续有边界的迁移；默认 viewer catalog 已改为 `src/modelCatalog.js`，
`src/assetLibrary.js` 只作为 pipeline / compatibility registry，不再是默认 world viewer
入口。`scripts/audit-world-viewer.mjs` 默认端口同步为 fixed `5395`。

同日已完成 `VIEWER-MANIFEST-CONSUME-001`：前端素材库现在以最小改动消费 /
暴露 backend model artifact manifest 的 browser-ready 路线，同时保留现有 renderer
和对象交互行为。

同日已完成最小代码切片：新增 `objgauss/core/` facade namespace，按领域暴露
Gaussian 数据模型、IO、features、objects、Object Field、masks、projection /
voting、semantics、evaluation 和 training handoff 入口。当前 facade 复用历史模块，
不改变 CLI / UI 行为。

随后已完成底层 kernel 移动：`GaussianCloud`、PLY / `.splat` IO、feature extraction、
baseline clustering 和 hard object label 操作已经由 `objgauss/core/` 承载实现，
旧 `objgauss/gaussians.py`、`ply.py`、`splat.py`、`features.py`、`clustering.py`
和 `segment.py` 只作为兼容 wrapper。前端自有 Gaussian renderer 算法未移动、未删除。

Object Field 与 projection / voting 也已迁移到 core：`objgauss/core/object_field.py`
承载 Object Field soft slots、metrics、save/load、hard label export 和 NeRF dataset
inspect；`objgauss/core/projection.py` 承载 `project_points`、mask voting、
depth visibility diagnostic、projection loss training 和 vote quality gate。旧
`objgauss/object_field.py` 与 `objgauss/mask_voting.py` 只作为兼容 wrapper。

Mask / semantic / evaluation algorithms 也已迁移到 core：`objgauss/core/masks.py`
承载 mask manifest build / validate、image readers、Lego RGBA classification 和
SAM manifest adapter；`objgauss/core/clip_scoring.py` 承载 CLIP scoring adapter
与 naming quality gate；`objgauss/core/semantic_slots.py` 承载 cross-view slot
alignment、slot naming policy 和 slot support rebalance；`objgauss/core/baseline_comparison.py`
与 `objgauss/core/emergence.py` 承载 comparison / promotion policy 和 object
emergence metrics。旧 `objgauss/masks.py`、`clip_scoring.py`、`semantic_slots.py`、
`baseline_comparison.py` 和 `emergence.py` 只作为兼容 wrapper。

后端模型交付契约已定义：`objgauss/model_manifest.py` 实现
`objgauss-model-artifact-manifest-v1`，用于描述后端提供给前端的 quick splat、
object-edit artifact、diagnostic full artifact、training/internal artifacts、
quality evidence、source / license / hash / Gaussian count / object count 和
browser-ready delivery tier。契约明确禁止 `diagnostic_full` / `source_gaussian`
被标成 `browser_ready=true`，防止 viewer 默认误拉 full PLY。

同一模块也已提供 adapter：`manifest_from_training_output(...)`、
`manifest_from_sample_bundle(...)` 和 `manifest_from_asset_library_entry(...)`。
其中 asset library adapter 会把 deferred / large object PLY（例如 near-1M full PLY）
映射为 `diagnostic_full` / `browser_ready=false`，只把 `.splat` quick view 和安全的
object-edit artifact 暴露为 browser-ready。

前端 manifest 消费已接入：`src/modelArtifactManifest.js` 按同一 schema 从本地
asset library 派生 viewer-side manifest / routes；`src/App.jsx` 的加载入口先解析
`quick_splat`、`object_edit` 和 `diagnostic_full` artifact，再选择 quick view 或显式
对象 / 诊断 PLY 路线。near-1M quick view 只激活 `quick_splat` /
`browser_quick`，full object PLY 暴露为 `diagnostic_full` /
`browser_ready=false`，不会作为默认 quick route 请求。前端 Spark、Gaussian OIT、
WebGPU tile / compute、shader、object-state buffer 和 picking 代码未重写。

下一步若继续处理“加载慢”，应进入 `PERF-BUNDLE-001` 或后续 LOD / streaming /
chunked artifact 设计；不要把 4.5M / 1GB+ full PLY 的显式诊断加载误当成默认交互
路线。

`PERF-BUNDLE-001` 已完成首轮 bundle 拆分：`src/App.jsx` 使用 `React.lazy`
按需加载 `SplatViewport`、`PointCloudViewport` 和 `WebGpuTileViewport`；纯配置
`sparkObjectMaskConfig.js` 从 Spark object mask renderer 中拆出，避免 App 首屏
静态引入 `@sparkjsdev/spark` / `three`。`npm run build` 结果显示主入口 JS 从
约 `5,850.87 kB` 降到 `254.72 kB` / gzip `79.77 kB`。Vite 仍会提示 lazy 的
`SplatViewport` route chunk 约 `5,017.81 kB`，这是 Spark renderer 算法仍被保留并
按需加载，不再是首屏主包。

加载速度后续剩余问题分两类：首屏 bundle 已明显下降；near-1M full PLY 的显式对象 /
诊断加载仍需要后端产出 LOD / streaming / chunked / compressed browser artifact 才能
真正优化。

`CHUNKED-GAUSSIAN-ARTIFACT-001` 已完成 contract 层：`objgauss/model_manifest.py`
新增 `objgauss-chunk-index-v1` 常量，并允许 `compressed_chunked` artifact 记录
`chunk_index`、`compression`、`lod` 和 `object_id_coverage` metadata。validator
要求 browser-ready chunked artifact 必须带 artifact-level Gaussian/object count、
byte size、sha256、chunk index、codec metadata、LOD levels 和完整 object id 覆盖；
同时继续禁止把 diagnostic full PLY 标成 browser-ready。该 contract 能容纳 Owner
提供的 Object-aware Gaussian Codec / OGC 方向；剪枝、量化、VQ、adaptive SH、
entropy coding 和 WebGPU streaming loader 仍未实现，最小 chunk binary writer
已在后续 OGC payload prototype 中落地。

`OGC-CHUNK-INDEX-001` 已完成 metadata 生成层：`objgauss/core/chunk_index.py`
基于 `GaussianCloud` 的 `x/y/z/object_id` 生成 `objgauss-chunk-index-v1`，排序键为
`object_id+morton_xyz`，chunk 不跨 object，记录每个 chunk 的 object id、count、
sorted range、AABB、center、radius 和 LOD metadata。该步骤仍不写 OGC binary、
不做剪枝 / 量化 / VQ、不改变前端 renderer；它为下一步 OGC binary writer 和后续
browser streaming loader 提供可测试 index。

`OGC-BINARY-WRITER-001` 已完成最小 payload prototype：`objgauss/core/ogc_payload.py`
使用 chunk index 和 sorted indices 写 `.ogc` raw chunk payload，并在 index chunk
metadata 中记录 `byte_offset`、`byte_length`、`record_count` 和
`record_format=objgauss-ogc-record-v0`。当前 record 保存 `x/y/z`、RGB、opacity 和
`object_id`，用于证明 chunk byte ranges、metadata roundtrip 和 object id preservation。
该步骤仍不做剪枝 / 实际量化写入 / VQ / adaptive SH / entropy coding，也未接入前端
streaming loader。

`OGC-MANIFEST-ADAPTER-001` 已完成 manifest 绑定层：`objgauss/model_manifest.py`
新增 `build_compressed_chunked_artifact(...)`，可从 `.ogc` payload 和
`objgauss-chunk-index-v1` `.index.json` 派生 `compressed_chunked` model artifact。
该 adapter 自动登记 `gaussian_count`、`object_count`、`byte_size`、`sha256`、
`chunk_index` summary、`compression`、`lod` 和 `object_id_coverage`，并继续要求
diagnostic full PLY 保持 `browser_ready=false`。本步骤不改前端 loader、不发布大
public demo asset、不移动训练产物。

`OGC-LOD-SAMPLING-001` 已完成 deterministic object-aware LOD metadata：
`objgauss/core/lod.py` 新增 `objgauss-object-aware-lod-v1` helper，默认 LOD ratios
为 full / 50% / 20% / 5%。`build_chunk_index(...)` 现在会给 top-level index、
object summary 和 chunk summary 写入 LOD levels，并保证每个正向 level 对每个
object 至少保留 1 个 Gaussian，level counts 单调不增。`write_ogc_payload(...)`
还会给 chunk-level LOD records 标注 `byte_offset` / `byte_length`，为后续 OGR /
WebGPU streaming loader 提供 prefix-record 读取窗口。该步骤仍不改前端 loader、
不做剪枝 / 量化 / VQ / entropy coding，也不移动前端 Gaussian OIT、WebGPU tile /
compute、Spark bridge 等自有渲染算法。

`OGC-QUANTIZATION-SCHEMA-001` 已完成 chunk-local quantization metadata 和 size
estimator：`objgauss/core/quantization.py` 新增 `objgauss-local-quantization-v1`
和 `objgauss-quantization-estimate-v1`，当前 policy 为
`chunk-aabb-uint16-rgb8-opacity8-v0`。metadata 记录未来 payload 中 `xyz` 使用
chunk AABB 内 `uint16 x3`、RGB 使用 `uint8 x3`、opacity 使用 `uint8`，`object_id`
保存在 chunk metadata。`write_ogc_payload(...)` 现在会把该估算写入 index 和
compression metadata，并证明估算 quantized payload 小于当前 raw OGC record payload；
当前 `.ogc` 文件本身仍是 raw prototype，不是实际量化二进制。本步骤不改前端 loader、
不做 VQ / adaptive SH / entropy coding，也不移动前端 Gaussian OIT、WebGPU tile /
compute、Spark bridge 等自有渲染算法。

`OGC-QUANTIZED-PAYLOAD-WRITER-001` 已完成第一个实际 chunk-local quantized OGC
payload prototype：`objgauss/core/quantization.py` 新增
`write_quantized_ogc_payload(...)` 和 `read_quantized_ogc_payload(...)`。quantized
record 当前为 `objgauss-ogc-quantized-record-v0`，每个 Gaussian 写 `xyz uint16x3`
、RGB `uint8x3` 和 opacity `uint8`，`object_id` 继续保存在 chunk metadata。测试已
证明 quantized payload 字节数小于 raw OGC payload，diagnostic roundtrip 的位置 /
opacity 误差有界，object id / chunk / LOD metadata 不丢失。本步骤仍不改前端 loader、
不做 VQ / adaptive SH / entropy coding / WebGPU decoder，也不移动前端 Gaussian OIT、
WebGPU tile / compute、Spark bridge 等自有渲染算法。

`OGR-BROWSER-DECODER-001` 已完成 quantized OGC browser decoder contract：
`src/ogcDecoder.js` 可把 `objgauss-ogc-quantized-payload-v0` chunk-local records
解码为现有 renderer-compatible points，并保留 object / chunk / LOD metadata；
`src/modelArtifactManifest.js` 现在会把 browser-ready `compressed_chunked`
artifact 暴露为独立 route；`scripts/audit-ogc-decoder-contract.mjs` 用小 fixture
验证 x/y/z、RGB、opacity、object_id、chunk id、LOD metadata 和 manifest route
contract。该步骤不接入真实 near-1M 大资产、不发布 public demo，也不改写或替换
Gaussian OIT、WebGPU tile / compute、Spark bridge 等前端自有渲染算法。

`OGC-BROWSER-STREAMING-001` 已完成 browser delivery 第一闭环：默认 Three.js
world viewer 现在能消费 browser-ready `compressed_chunked` artifact，把 quantized
OGC chunk payload 解码为现有 renderer-compatible points，并继续按 `object_id`
派生 ObjectState Debug OS 的对象 render targets、assignment heatmap 和 Gaussian
probe。`src/modelCatalog.js` 新增小型 `ogc-debug` inline fixture，用于验证
manifest route / chunk index / payload decode / ObjectState overlay 全链路；`src/App.jsx`
新增 OGC loader，后续真实 artifact 可走 `chunk_index.path + payload path` fetch 路线。
`scripts/audit-world-viewer.mjs` 现在会选中 `ogc-debug`，要求 `ogcLoaded=1`、
至少 2 个 OGC object render targets、`assignmentSlots=2` 和 Gaussian probe 成功。
该步骤仍不提交 near-1M / 4.5M 大资产，不把 full diagnostic PLY 恢复为默认 route，
也不替换 Gaussian OIT、WebGPU tile / compute、Spark bridge、shader、object-state
buffer 或 picking 代码。

`OGC-URL-ARTIFACT-001` 已完成 browser delivery 的运行时 URL OGC 注入入口：
`src/modelCatalog.js` 现在支持同源
`?ogcIndex=/path/to/scene.index.json&ogcPayload=/path/to/scene.ogc`，并追加
`ogc-url-artifact` 临时模型；无 trainable URL artifact 时会默认选中该模型。该入口
继续走 `compressed_chunked` manifest route、现有 quantized OGC decoder、ObjectState
Debug OS、assignment heatmap 和 Gaussian probe，并支持可选 `ogcLod` / `ogcChunks`。
`src/App.jsx` 为 selected OGC artifact 暴露 `fetch-ogc`、index path、payload path 和
LOD telemetry。新增 `public/models/ogc-url-fixture/` 下的小型 index / `.ogc` fixture
只用于验证同源 fetch path，不是训练输出或大资产；`scripts/audit-world-viewer.mjs`
现在额外打开 URL OGC route，验证模型数变为 8、默认选中 URL OGC artifact、
URL OGC route telemetry 可审计，且 LOD 1 解码后仍有 2 个 ObjectState render targets。

`OGC-URL-MANIFEST-ARTIFACT-001` 已把 URL OGC 注入进一步收敛为单 manifest handoff：
`src/modelCatalog.js` 现在支持同源
`?ogcManifest=/path/to/model-artifact.json`（兼容 `ogc-manifest`、
`modelArtifactManifest`、`model-artifact-manifest`），并追加 `ogc-manifest-artifact`
临时模型；该路径仍限制为 same-origin `.json` 且不接受 query / hash。`src/App.jsx`
会 fetch `objgauss-model-artifact-manifest-v1`，选出 browser-ready `compressed_chunked`
artifact，并把 manifest-relative `chunk_index.path` 与 payload `path` 解析为同源绝对
路径，再走现有 OGC range loader、LOD / chunk selector、ObjectState render targets、
assignment heatmap、Gaussian probe、snapshot 和 event trace。新增
`public/models/ogc-url-fixture/model-artifact.json` 小型 fixture 只引用同目录
`scene.index.json` / `scene.ogc`，不是训练输出或大资产；`scripts/audit-world-viewer.mjs`
现在验证 `urlOgcManifest=url-manifest-range-lod-chunk-ui`。

`ALGO-MANIFEST-BUNDLE-001` 已把算法产物 handoff 从“单一 OGC manifest”升级为组合
`modelArtifactManifest` 路径：`objgauss/model_manifest.py` 现在承认
`trainable_kernel` browser-ready role；`src/modelArtifactManifest.js` 可解析
`trainableKernel` route；`src/modelCatalog.js` 将
`?modelArtifactManifest=/path/model-artifact.json` 作为组合入口，而 `?ogcManifest=...`
继续保留 OGC-only 入口。`src/App.jsx` 新增 `model-artifact-manifest` load mode，会
fetch `objgauss-model-artifact-manifest-v1` 后自动展开为 `Manifest Train` 和
`Manifest OGC` 两个 runtime 模型：前者复用 trainable artifact loader，展示
`A[N,K]`、ObjectState、training loss 和 Gaussian probe；后者复用 OGC range loader、
LOD selector、chunk selector 与 ObjectState Debug OS。新增
`public/models/algorithm-bundle-fixture/model-artifact.json` 只引用既有 trainable debug
artifact 与 tiny OGC fixture，不复制 payload、不提交训练输出；同轮修复多模型 dock 被
floating inspector 遮挡的问题。`scripts/audit-world-viewer.mjs` 现在验证
`algorithmManifest=manifest-trainable-ogc-debug-os`。

`LOCAL-MODEL-MANIFEST-BUNDLE-001` 已把上述组合 handoff 扩展到本地 package 导入：
`src/App.jsx` top HUD 新增 `导入模型` 多文件入口，可一次选择
`objgauss-model-artifact-manifest-v1 + trainable artifact JSON + .index.json + .ogc`。
前端会按 manifest 的 `trainable_kernel` role 匹配训练 artifact JSON，按
`compressed_chunked` role 匹配 OGC chunk index 与 payload，然后自动创建
`Local Manifest`、`Local Train` 和 `Local OGC` 三个 runtime 模型。默认进入
`Local Train` 检查 `A[N,K]`、training loss、ObjectState 和 Gaussian probe；`Local OGC`
继续复用 local-file OGC loader、LOD selector、chunk selector、assignment heatmap、
Gaussian probe 和 event trace。root shell 新增 `data-model-manifest-import-*` telemetry；
`scripts/audit-world-viewer.mjs` 现在验证
`localModelManifest=local-manifest-trainable-ogc-debug-os`。该路径只读取浏览器会话中的
本地文件，不复制 payload 到 `public/`，不提交 ignored `outputs/` 或训练产物。

`QUALITY-REPORT-HANDOFF-001` 已把 Phase 1 “UI + metrics 双系统”继续收敛进同一个
model manifest handoff：`src/modelArtifactManifest.js` 现在可解析 browser-ready
`quality_report` artifact；`public/models/algorithm-bundle-fixture/quality-report.json`
新增小型 `objgauss-object-state-quality-report-v1` fixture，记录 assignment entropy、
slot utilization、object purity、temporal drift、assignment jitter、bbox stability 和
quality gates；同目录 `model-artifact.json` 现在同时包含 `trainable_kernel`、
`compressed_chunked` 和 `quality_report` 三个 browser-ready artifact。`src/App.jsx`
会在 URL 组合 manifest 和本地组合 package 导入时加载 quality report，并挂到 Train /
OGC 子模型与父 manifest model；Debug panel 新增 `Quality` 证据面板，root shell 和
snapshot protocol 暴露 `data-quality-report-*` / `quality` telemetry。
`scripts/audit-world-viewer.mjs` 现在验证 `qualityReport=warn`，包括 quality schema、
gate count、object purity 与 snapshot quality 状态。该步骤不生成质量报告、不改 promotion
policy、不提交训练输出或大资产。

`DEBUG-SNAPSHOT-EXPORT-001` 已把 Phase 1 Debug OS 的 snapshot protocol 做成可导出的
browser handoff：`src/App.jsx` 的 `Protocol` 面板新增 `JSON` 导出按钮，导出内容保持
top-level `objgauss-object-state-debug-snapshot-v1` / `object-state-debug-os-v1` 协议，
并附带 `objgauss-debug-snapshot-export-v1` export metadata。root shell 新增
`data-debug-snapshot-export-*` telemetry，浏览器会话中也保留
`window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SNAPSHOT__` 与 JSON 文本，便于 audit / 人工复查。
`scripts/audit-world-viewer.mjs` 现在在组合算法 manifest 场景中点击导出，验证导出的
model、quality status、chunk scope、schema 和 `export-snapshot` event trace。该步骤
不训练模型、不写训练输出、不复制 OGC payload 或 ignored `outputs/` 到仓库。

`DEBUG-SESSION-EXPORT-001` 已把单帧 snapshot export 扩展为 browser-local debug session
handoff：`src/App.jsx` 的 `Protocol` 面板新增 `SESSION` 导出按钮，导出
`objgauss-object-state-debug-session-v1`，其中包含当前 snapshot、compact model delivery
summary、recent debug events、quality gate evidence 和 browser-local-only export policy。
root shell 同步暴露 `data-debug-session-*` telemetry，浏览器会话中保留
`window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SESSION__` 与 JSON 文本。`scripts/audit-world-viewer.mjs`
现在在组合 algorithm manifest 场景中验证 `debugSessionExport=exported`、session schema、
model list、`export-snapshot` / `ogc-chunks` trace 和 `export-session` event。该步骤仍不改
训练 loop、artifact schema、OGC decoder 或 renderer，不写训练输出或大资产。

`DEBUG-SESSION-IMPORT-001` 已把 debug session handoff 补成前端可审计闭环：
`src/App.jsx` 的 `Protocol` 面板新增 `LOAD` 按钮，可导入
`objgauss-object-state-debug-session-v1` 并显示为只读 `Archive` 面板。导入路径会校验
session / snapshot schema，压缩 models / events / quality gates 为 debug metadata，
并暴露 `data-debug-session-import-*`、`data-debug-session-archive-*` 与
`window.__OBJGAUSS_IMPORTED_DEBUG_SESSION__`。`scripts/audit-world-viewer.mjs` 现在用刚导出的
session JSON 重新导入，验证 `debugSessionImport=loaded`、archive model、quality warn、
model count、event count 和 `import-session` trace。该步骤不做场景 replay，不重新拉
OGC payload，不改训练 loop、不写训练输出或大资产。

`DEBUG-SESSION-DIFF-001` 已把导入的 debug session 进一步变成 live-vs-archive 比较视图：
`src/App.jsx` 新增 `debugSessionSnapshotDiff(...)`，在当前 live snapshot 与导入 archive
snapshot 之间比较 model、object、assignment source、slot count、entropy、confidence、
quality / training / stability status 和 delivery route。root shell 与 `Archive` 面板现在暴露
`data-debug-session-diff-*` telemetry，并显示 diff status、match flags、slot delta、entropy
delta、confidence delta 和 event delta。`scripts/audit-world-viewer.mjs` 在刚导出再导入的
同一 session 上验证 `debugSessionDiff=match`，证明该视图可作为后续不同训练 run 的 drift
检查基础。该步骤不改 session schema、不做场景 replay、不拉 payload、不写训练输出。

`DEBUG-SESSION-DRIFT-001` 已补齐 live-vs-archive diff 的 changed-field 证据：
`debugSessionSnapshotDiff(...)` 现在输出 `changedFields` / `changedFieldNames`，覆盖
model、object、assignment source、quality、training、stability、delivery、slots、entropy、
confidence 和 quality entropy。root shell 与 `Archive` 面板新增
`data-debug-session-diff-field-count` / `data-debug-session-diff-fields` telemetry，并在 UI 中显示
compact field list；切换模型时会清除旧 Gaussian probe，避免当前 snapshot 继承前一个模型的
probe source。`scripts/audit-world-viewer.mjs` 现在同时验证同一 session 的
`debugSessionDiff=match` 和切换到 trainable route 后的 `debugSessionDrift=changed`，
changed fields 至少包含 `model/source/training/delivery`。该步骤不改 session schema、
不做场景 replay、不拉 payload、不写训练输出。

`OBJECTSTATE-BENCH-001` 已完成训练前 ObjectState stability gate：
`objgauss/core/object_state_benchmark.py` 新增
`objgauss-object-state-stability-benchmark-v1` synthetic pressure suite，固定覆盖 clean sparse、
uniform mixed、single-slot collapse、soft noise、slot permutation、temporal jitter、birth
unmatched 和 duplicate fragment。该 suite 复用现有 `project_object_states(...)`、
`object_state_stability_report(...)`、`match_object_states(...)` 与
`dynamic_k_proposal_report(...)`，输出 assignment confidence、entropy、effective slots、
object purity、label fragmentation、bbox diagonal、raw assignment jitter、matched temporal
drift、bbox drift、temporal matches 和 dynamic-K proposal kinds。CLI 入口为
`objgauss object-state stability-benchmark --output <json> --strict`；默认 gate 当前输出
`status=pass`、`cases=8`、`observed_warn_count=6`、`warn_count=0`。该步骤不改 solver /
renderer / trainable kernel loop，不触碰 viewer UI，不引入 torch / gsplat / CUDA / SAM /
CLIP，也不提交训练输出或大资产。

`OBJECTSTATE-BENCH-HANDOFF-001` 已将上述训练前 stability benchmark 接入 browser
handoff：`objgauss-model-artifact-manifest-v1` 新增 browser-ready
`object_state_benchmark` artifact role，algorithm bundle fixture 现在包含
`public/models/algorithm-bundle-fixture/object-state-benchmark.json`，schema 为
`objgauss-object-state-stability-benchmark-v1`，状态为 `pass`、case 数 `8`、
observed warn 数 `6`、warn 数 `0`。`src/App.jsx` 会随 model artifact manifest 加载
benchmark report，并在 ObjectState Debug 面板显示 Benchmark 卡片，同时将摘要写入 root
telemetry 和 debug snapshot / session archive。`scripts/audit-world-viewer.mjs` 同时验证
远程 manifest 和本地 package import 的 benchmark telemetry，输出
`objectStateBenchmark=pass`。该切片不改变 solver、renderer、训练 loop，不引入 torch /
gsplat / CUDA，也不提交训练输出或大资产。

`OBJECTSTATE-BENCH-CASE-INSPECT-001` 已把 benchmark 从 summary evidence 升级为
case-level inspector：`src/App.jsx` 的 Benchmark 卡片现在默认选中首个
`observed_status=warn` case，并支持点击具体 case 查看 assignment confidence、
normalized entropy、object purity、temporal drift、dynamic-K proposal count、
failure modes 和 diagnostics。active case 同步写入 root telemetry 和 debug snapshot /
session archive；`scripts/audit-world-viewer.mjs` 验证默认 `uniform_mixed` case、点击
`temporal_jitter` 后的 active telemetry，以及 snapshot / session export-import 中的
active case 摘要。该切片只增强 browser Debug OS，不改变 benchmark 生成逻辑、
solver、renderer、训练 loop，不引入 torch / gsplat / CUDA。

`OBJECTSTATE-OVERLAY-CONTROLS-001` 已将已有 centroid sphere / bbox wireframe 升级为
可控、可审计的 ObjectState overlay 层：`src/App.jsx` 新增 `full` / `bbox` /
`centroid` / `off` overlay selector，控制 Three.js scene 中 `object-state-bbox` 和
`core-point` 的可见性，同时保持 Gaussian 点云、selection ring 和 object toggle 行为不变。
overlay mode 写入 root telemetry、`window.__OBJGAUSS_WORLD__` audit handle 和
debug snapshot / session archive；`scripts/audit-world-viewer.mjs` 验证四种模式下 bbox /
centroid child visibility，输出 `objectOverlay=full`。该切片只强化 Phase 1 Layer 2
调试仪器，不改变 renderer、solver、artifact schema、训练 loop，也不引入 torch /
gsplat / CUDA。

`OBJECTSTATE-OPACITY-LENS-001` 已将 Phase 1 Layer 1 的 opacity debugging 升级为
一等 Debug OS lens：`src/App.jsx` 现在把 `opacity` 加入 lens selector，并为普通
point cloud、compressed placeholder 和 trainable artifact Gaussian cloud 写入
`opacityColor` buffer 与 `gaussianOpacityMean`。`colorAttributeForDebugLens(...)` 与
`opacityForDebugLens(...)` 支持 opacity lens，`window.__OBJGAUSS_WORLD__.lensOpacitySamples`
继续暴露 active color / opacity lens 和实际 material opacity。`scripts/audit-world-viewer.mjs`
已验证 trainable fixture 在切换到 opacity lens 后，root、Debug panel、selector 和 scene
samples 均进入 `opacity` lens。该切片只强化 Gaussian scene 的调试仪器属性，不改变
assignment solver、ObjectState projection、renderer artifact schema、OGC decoder 或
trainable kernel loop；训练模型主线仍保持
`suspended / current-env-missing-torch-gsplat-cuda`。

`OBJECTSTATE-PROBE-DIAGNOSTIC-001` 已将 Gaussian assignment probe 从单纯 heatmap
升级为可审计诊断：`src/App.jsx` 现在从当前 `A[n,:]` 派生 top-1 / top-2、
margin、ambiguous 和 collapse-risk summary，并同步写入 ObjectState Debug 面板、
root telemetry、`window.__OBJGAUSS_WORLD__`、debug snapshot、session archive 和
live-vs-archive diff changed fields。`scripts/audit-world-viewer.mjs` 验证 trainable
artifact、local import、algorithm manifest、OGC route、snapshot export 和 session
import 中的 probe 状态与 margin 保真。该切片仍是 Phase 1 非训练 Debug OS 工作，
不改变 assignment solver、ObjectState projection、renderer、artifact schema 或训练
loop；`TRAIN-GSPLAT-MVP-001` 继续因 torch / gsplat / CUDA 环境缺失保持挂起。

`OBJECTSTATE-HOVER-HIGHLIGHT-001` 已将 object hover 升级为可审计的 Gaussian cluster
focus：Three.js 状态层现在在 hover 某个 ObjectState render target 时把该对象的
Gaussian cloud 标记为 `highlighted`，并将其它未选 cluster 压暗为 `dimmed`；root
telemetry、ObjectState Debug panel 和 `window.__OBJGAUSS_WORLD__` 同步暴露
`hoverHighlightActive`、highlighted / dimmed object count、Gaussian count 和 per-object
opacity samples。`scripts/audit-world-viewer.mjs` 验证 hover target 的 assigned
Gaussians 被高亮、非目标 cluster 被压暗，同时保持 debug lens、overlay、snapshot、
session 和 benchmark handoff 通过。该切片只强化 Phase 1 交互调试能力，不改变
assignment solver、ObjectState projection、renderer artifact schema 或训练 loop。

`OBJECTSTATE-VISIBILITY-CONTRACT-001` 已将 object toggle 升级为可审计的 Gaussian
cluster visibility contract：React/root 从 `hiddenObjects + model.objects` 派生
`objgauss-object-visibility-summary-v1`，Three.js audit handle 从真实 `object.visible`
派生 visible / hidden object count、Gaussian count、hidden object ids 和 per-object
visibility samples。ObjectState Debug panel、debug snapshot 和 session archive 现在同步
记录 hidden object / hidden Gaussian 证据；`scripts/audit-world-viewer.mjs` 通过真实 UI
toggle 按钮验证 root telemetry、Debug panel、Three.js scene 和 snapshot 的 hidden /
visible Gaussian 统计一致。该切片只强化 Phase 1 的 hide/show cluster 调试闭环，不改变
assignment solver、ObjectState projection、renderer artifact schema 或训练 loop。

`OBJECTSTATE-CONTINUITY-001` 已将 per-object spatial continuity 从隐式 bbox/centroid
字段升级为可审计诊断：`src/App.jsx` 新增
`objgauss-object-continuity-summary-v1`，从 ObjectState 的 `bbox`、`centroid`、
Gaussian count 和可选 `spatialCompactness` 派生 `continuous` / `fragmented` /
`degenerate` / `centroid-outside` 等状态、bbox diagonal、Gaussian density 和
centroid-contained 证据。root `.worldShell`、ObjectState Debug panel、
`window.__OBJGAUSS_WORLD__`、debug snapshot、hover preview 和 session archive 现在同步
记录 selected / hovered ObjectState continuity；`scripts/audit-world-viewer.mjs` 验证
trainable fixture 的 root / panel / scene handle / snapshot / session handoff 都保留
continuity schema 和 selected / hover spatial contract。该切片仍是 Phase 1 非训练 Debug
OS 工作，不改变 assignment solver、ObjectState projection、renderer artifact schema、
OGC decoder 或 trainable kernel loop；训练模型主线继续保持
`suspended / current-env-missing-torch-gsplat-cuda`。

`OBJECTSTATE-TEMPORAL-INSPECT-001` 已将 selected / hovered ObjectState 的时间稳定性从
模型级 dashboard 均值下钻到单对象 inspector：`src/App.jsx` 新增
`objgauss-object-temporal-summary-v1`，复用 trainable artifact loader 已计算的
`temporalDrift`、`assignmentJitter` 和 `bboxStability`，派生 `stable` /
`assignment-jitter` / `temporal-drift` / `bbox-unstable` 等状态。root `.worldShell`、
ObjectState Debug panel、`window.__OBJGAUSS_WORLD__`、debug snapshot、hover preview 和
session archive 现在同步记录 selected / hovered temporal stability；session diff 也会把
temporal status 变化列为 changed field。该切片继续只强化 Phase 1 非训练 Debug OS，
不改变 assignment solver、ObjectState projection、renderer artifact schema、OGC decoder
或 trainable kernel loop；训练模型主线仍保持
`suspended / current-env-missing-torch-gsplat-cuda`。

`OBJECTSTATE-EXPLAINABILITY-001` 已将 selected / hovered ObjectState 的可解释性压成
单一 Debug OS contract：`src/App.jsx` 新增
`objgauss-object-explainability-summary-v1`，把 assignment probe 的 confidence /
margin / entropy、spatial continuity 和 temporal stability 合并为 `explainable`、
score 和 reason list。root `.worldShell`、ObjectState Debug panel、
`window.__OBJGAUSS_WORLD__`、debug snapshot、hover preview 和 session archive 现在同步
记录 selected / hovered explainability summary；session diff 也会把 explainability
status 变化列为 changed field。`scripts/audit-world-viewer.mjs` 已扩展为验证 trainable
fixture 的 root / panel / scene handle / snapshot / session handoff 都保留
explainability schema 和 selected / hover explainability contract。该切片继续只强化
Phase 1 非训练 Debug OS，不改变 assignment solver、ObjectState projection、renderer
artifact schema、OGC decoder 或 trainable kernel loop；训练模型主线仍保持
`suspended / current-env-missing-torch-gsplat-cuda`。

`OBJECTSTATE-VERDICT-PANEL-001` 已将上述 explainability contract 从隐藏 telemetry /
meta 字段升级为可读 inspector：`src/App.jsx` 新增 `ObjectStateVerdictPanel`，在
Assignment Heatmap 后显示 selected ObjectState 的 verdict、score、assignment margin、
A confidence / entropy、spatial / temporal 子状态和 clear / warning reason rows；hover
ObjectState 激活时同一面板显示 hover verdict，并暴露 `data-object-verdict-*` 与
`data-hover-verdict-*` telemetry。`scripts/audit-world-viewer.mjs` 已验证 trainable
fixture 中 selected / hover verdict panel 与 root、Debug panel、scene handle、snapshot
和 session handoff 的 explainability contract 保持一致。该切片只强化 Phase 1 Debug OS
的可读性，不改变 assignment solver、ObjectState projection、renderer artifact schema、
OGC decoder 或 trainable kernel loop；训练模型主线仍保持
`suspended / current-env-missing-torch-gsplat-cuda`。

`OBJECTSTATE-HOVER-ASSIGNMENT-001` 已将 hover 从纯视觉 focus 升级为 ObjectState
assignment preview：`objectTarget(...)` 现在携带 hovered object 的 compact assignment
vector、confidence、entropy、status、centroid 和 bbox；root telemetry、
ObjectState Debug panel、`window.__OBJGAUSS_WORLD__`、debug snapshot 和 session archive
同步记录 hover assignment source、slot count、top slot、margin、confidence、entropy 和
probe status。`scripts/audit-world-viewer.mjs` 验证 hover trainable ObjectState 时 root /
panel / scene handle / snapshot 四层都能看到 `trainable_kernel_model_artifact` 的 2-slot
assignment preview，并保持 hover highlight、visibility、snapshot/session handoff 通过。
该切片只强化 Phase 1 “hover 显示 A / ObjectState 快速预览”，不改变 solver、
projection、renderer artifact schema 或训练 loop。

`OBJECTSTATE-HOVER-HEATMAP-001` 已将 hover assignment preview 进一步升级为可见
`A[n,k]` heatmap：`src/App.jsx` 新增 `HoverAssignmentHeatmap`，在 selected
Assignment Heatmap 后显示 hovered ObjectState 的 compact assignment vector、top slot、
top probability、margin 和 per-slot probability bars；`src/styles.css` 复用现有
assignment row / probe meta 样式，只增加轻量 hover 边框。`scripts/audit-world-viewer.mjs`
已验证 trainable fixture hover 后，hover heatmap 的 target、model、source、slot count、
status、margin 和 row count 与 `window.__OBJGAUSS_WORLD__` 的 hover assignment contract
一致。该切片只强化 Phase 1 Layer 3 的交互显微镜，不改变 assignment solver、
ObjectState projection、renderer artifact schema、OGC decoder 或 trainable kernel loop；
训练模型主线仍保持 `suspended / current-env-missing-torch-gsplat-cuda`。

`OBJECTSTATE-GAUSSIAN-PROBE-PANEL-001` 已将 `gaussian click -> show A[n,:]` 从
heatmap meta 升级为可见 Gaussian Probe inspector：`src/App.jsx` 在 selected
Assignment Heatmap 后新增 `GaussianProbePanel`，显示 clicked Gaussian index、source、
top slot、margin、confidence、position、opacity、entropy、top / second probability
和 ambiguous / collapse-risk flag；`src/styles.css` 为该面板增加紧凑四列 metrics、
source / position metadata 和 flag rows。`scripts/audit-world-viewer.mjs` 现在验证
trainable fixture 点击 Gaussian 后，该面板的 source、index、status、margin、
confidence、entropy、opacity、position 和 flags 与 root / heatmap / world telemetry
一致，并在 audit 输出中记录 `gaussianProbe=confident`。该切片只强化 Phase 1
“点击 Gaussian 查看 assignment vector”调试闭环，不改变 assignment solver、
ObjectState projection、renderer artifact schema、OGC decoder 或 trainable kernel loop；
训练模型主线仍保持 `suspended / current-env-missing-torch-gsplat-cuda`。

`OBJECTSTATE-ASSIGNMENT-TIMELINE-001` 已将 assignment stability 从单值 jitter 指标
升级为可见跨帧时间线：`src/App.jsx` 新增 `objgauss-assignment-timeline-v1` 派生 summary，
从 trainable artifact 的 `assignments` 中抽取同一个 Gaussian row 在各 frame 的
`A[n,k]`、top slot、margin、entropy 和 adjacent delta；Debug panel 在 selected
Assignment Heatmap 后显示 `Assignment Timeline`，同时将 timeline summary 写入 root
telemetry、Debug panel telemetry、debug snapshot / session archive 和 live-vs-archive
diff changed fields。`src/styles.css` 为 timeline 增加稳定三列行布局和 slot probability
bars；`scripts/audit-world-viewer.mjs` 现在验证 trainable fixture 点击 Gaussian 后，
timeline 为 2-frame `stable`，`meanDelta=0.02`，并在 audit 输出中记录
`assignmentTimeline=stable`。该切片强化 Phase 1 “assignment 是否 jitter”调试证据，
不改变 artifact schema、assignment solver、ObjectState projection、renderer、OGC
decoder 或 trainable kernel loop；训练模型主线仍保持
`suspended / current-env-missing-torch-gsplat-cuda`。

`OBJECTSTATE-FRAGMENTATION-PANEL-001` 已将 spatial continuity / fragmentation 从
summary telemetry 升级为可见 Object Fragmentation inspector：`src/App.jsx` 新增
`ObjectFragmentationPanel`，复用现有 `objgauss-object-continuity-summary-v1`，显示
selected ObjectState 的 Gaussian 数、compactness、bbox diagonal、density、bbox、
centroid 和 fragmented / centroid-contained / bbox-valid flag；hover ObjectState 时
同一面板也暴露 hover fragmentation metadata。`scripts/audit-world-viewer.mjs` 已验证
trainable fixture 的 panel、root telemetry、Debug panel 和 snapshot continuity contract
一致，并在 audit 输出中记录 `objectFragmentation=continuous`。该切片继续只强化 Phase 1
非训练 Debug OS，不改变 assignment solver、ObjectState projection、artifact schema、
renderer、OGC decoder 或 trainable kernel loop；训练模型主线仍保持
`suspended / current-env-missing-torch-gsplat-cuda`。

`OGC-RANGE-LOADER-001` 已完成 browser delivery 的 byte-range chunk loader：
`src/ogcDecoder.js` 新增 `quantizedOgcReadWindows(...)` 和
`decodeQuantizedOgcPayloadWindows(...)`，让前端可以从同一个 chunk / LOD window
contract 解码全量 payload 或 range payload。`src/App.jsx` 对 fetchable OGC artifact
会先按 `Range: bytes=start-end` 拉取 `chunk_index` 选中的 LOD byte windows；成功时
delivery route 记录为 `range-ogc`，不支持 range 的环境才回退 `fetch-ogc`。selected
OGC telemetry 和 inspector 现在显示 fetched bytes、requested bytes、decoded windows、
`OGC route` 与 `OGC bytes`。`scripts/audit-world-viewer.mjs` 在 URL OGC fixture 上验证
LOD 1 只请求 2 个 byte windows / 20 bytes，并仍能生成 2 个 ObjectState render targets、
assignment heatmap 和 Gaussian probe。该切片不改变 OGC writer、record format、
manifest contract，不实现 VQ / entropy / WebGPU decoder，也不提交大资产。

`OGC-LOD-DEBUG-UI-001` 已完成 OGC range loader 的 Debug OS 交互层：`src/App.jsx`
在 OGC artifact 加载后从 `chunk_index.lod.levels` 记录可用 `lodLevels`，并在 selected
OGC 模型的 ObjectState Debug panel 中显示紧凑 `lod` selector。点击 LOD 按钮会重新
调用现有 OGC range loader，按新的 `lodLevel` 重新 fetch byte windows、decode points、
upsert Three.js object render targets，并清除该模型旧隐藏状态。selected OGC telemetry
和 inspector 继续显示 `OGC route`、`OGC bytes`、fetched / requested bytes 和 decoded
windows。`scripts/audit-world-viewer.mjs` 现在在 URL OGC route 下先验证 LOD 1 为
`20 / 20` bytes，再点击 `L0` 并验证 `range-ogc`、`40 / 40` bytes、2 个 decoded
windows、assignment heatmap 和 Gaussian probe 仍可用。该切片只是把已存在 range loader
变成可交互调试控件，不改变 OGC writer / manifest / renderer，不提交大资产。

`OGC-CHUNK-DEBUG-UI-001` 已完成 OGC range loader 的 chunk scope 调试层：
`src/App.jsx` 将 OGC reload 逻辑收敛为共享 helper，LOD 与 chunk scope 都复用同一条
`loadOgcModel -> range windows -> decode -> upsertModel` 路径。OGC artifact load /
reload 后记录 `chunkIds` 与 `availableChunkIds`，root shell、inspector 和 Debug panel
暴露 `data-ogc-artifact-chunk-scope`、`OGC scope` 和 `data-ogc-chunk-selector`。
selected OGC 模型现在可以在 `all` 与单个 object-aware chunk（如 `c0`）之间切换：
单 chunk route 只请求对应 byte window，ObjectState / assignment heatmap 会随 decoded
chunk scope 收缩。`scripts/audit-world-viewer.mjs` 在 URL OGC route 下验证 LOD1 初始
`20 / 20` bytes、LOD0 全量 `40 / 40` bytes、`c0` 单 chunk `20 / 20` bytes / 1 个
decoded window / 1 个 heatmap slot，再切回 `all` 恢复 2 个 slots。该切片不改变 OGC
writer、chunk index schema、manifest、renderer 或训练系统，不提交大资产。

`OGC-LOCAL-ARTIFACT-IMPORT-001` 已完成 Debug OS 的本地 OGC file-pair 导入入口：
`src/App.jsx` 的 top HUD 新增 `导入OGC` 多文件入口，可直接读取本地
`.index.json + .ogc` 文件对，并包装为运行时 `ogc-local-artifact` 模型。该模型复用
现有 `compressed_chunked` manifest route、quantized OGC decoder、ObjectState render
targets、assignment heatmap、Gaussian probe、LOD selector 和 chunk selector；delivery
route 明确标为 `local-file`，index / payload path 标为 `local://<file>`。root
telemetry 新增 `data-ogc-import-*`，并继续暴露 `data-ogc-artifact-*` route / bytes /
window 信息。`scripts/audit-world-viewer.mjs` 现在会用小型 OGC fixture 执行
Playwright `setInputFiles(...)` 导入，验收本地 file route、LOD1、单 chunk scope、
assignment heatmap、Gaussian probe 和 `import-ogc` event trace。该切片不改变 OGC
writer、chunk index schema、quantized record format、manifest validator 或训练系统；
本地 OGC artifact 只作为 browser-session debug input，不写入 `public/` 或 git。

`OGC-LOCAL-MANIFEST-PACKAGE-001` 已将上述本地 OGC 导入升级为 manifest package 入口：
同一个 `导入OGC` 文件选择器现在支持
`objgauss-model-artifact-manifest-v1 + .index.json + .ogc`。前端会优先识别 model
artifact manifest，选出 browser-ready `compressed_chunked` artifact，再根据
artifact 的 `chunk_index.path` 与 payload `path` 匹配用户选择的本地文件，并把 artifact
本地化为 `local://<file>` route、内联 chunk index 和内存 payload buffer。裸
`.index.json + .ogc` 路径保持兼容；manifest package 和裸 file-pair 最终都走
`loadOgcModel -> decode -> upsertModel`，并继续支持 LOD / chunk selector、assignment
heatmap、Gaussian probe、snapshot 和 event trace。`scripts/audit-world-viewer.mjs`
现在运行时生成 `/tmp/objgauss-local-ogc-model-artifact.json` 小型 manifest fixture，
并验证 `localOgcManifest=local-manifest-file-lod-chunk-ui`。该切片不改变 OGC writer /
schema / manifest validator，不提交新的 payload 或训练产物。

Owner 随后把 viewer 目标更新为“打开即进入 Three.js / VR-like 3D 世界”：不再以
侧栏工作台作为默认入口，所有模型以展品方式出现在三维场景中，对象可拖动，模型 /
对象信息通过磨砂玻璃浮层显示。`WORLD-REBUILD-001` 已完成默认前端入口替换：
`src/App.jsx` 现在只渲染全屏 Three.js world、底部模型胶囊和浮动 inspector；
`src/modelCatalog.js` 记录模型展品、核心点、load mode 和 per-object corepoint
chunk 压缩计划；`scripts/audit-world-viewer.mjs` 验证 no sidebar、Three.js canvas、
draggable model count、frosted-glass UI 和 selected model telemetry。near-1M
diagnostic asset 不再作为首屏全量 PLY 拉取，而是作为 compressed-placeholder 展品
和后续单对象压缩块加载计划出现。

开源参考已写入 `docs/architecture/open-source-reference-map.md`。当前借鉴方向：
GaussianSplats3D 的 Three.js splat world / `.ksplat` compressed scene / WebXR 思路；
antimatter15/splat 的极简 browser viewer 和透明排序问题说明；SuperSplat 的 browser
editor / inspect / optimize / publish workflow；A-Frame 与 Hubs 的 WebXR / in-world UI
模型。边界仍然明确：不照搬外部 renderer，不删除 ObjGauss 自有前端 Gaussian OIT、
WebGPU tile / compute、shader、object-state buffer、picking、Spark bridge 和 OGC
decoder 等前端核心渲染算法；下一步清理的是旧产品 UI、旧审计入口和未再服务新世界
入口的非核心代码。

`WORLD-OBJECT-RENDER-001` 已完成对象级 viewer 优化：默认 Three.js 世界不再把每个
PLY 作为单个 draggable model mesh，而是按 `objectId` 拆成独立 `THREE.Group` /
`THREE.Points` render target。每个对象都有自己的 selection id、核心点、选中环、
拖拽句柄、viewer-side 位置和 per-object chunk path；near-1M 仍不拉 full diagnostic
PLY，只用 compressed-placeholder 为每个对象生成独立占位 Gaussian cloud。前端 HUD
现在暴露 object count、selected object 和 per-object corepoint chunk 路线；小对象
仅在 viewer 层做最小显示跨度放大，训练数据、压缩 contract 和后端 artifact 不变。
该切片继续保留 ObjGauss 自有 Gaussian OIT、WebGPU tile / compute、Spark bridge、
shader、object-state buffer、picking 和 OGC decoder 等核心渲染算法，未引入外部
renderer。

`OBJECT-DEBUG-UI-001` 已完成 Phase 1 ObjectState Debug OS 第一切片：默认 Three.js
world viewer 现在不仅展示对象，还显示 `objgauss-object-state-debug-v1` 调试协议。
`src/App.jsx` 为每个 object render target 派生 ObjectState debug metadata，显示
centroid / bbox overlay、confidence、normalized assignment entropy、slot mass、
mass fraction、centroid、bbox 和 per-object visibility toggle。Gaussian click / audit
probe 能显示单个 Gaussian 的 `A[n,k]` vector，assignment heatmap 当前明确标记为
`derived_from_object_id` 或 `compressed_placeholder_assignment`，用于 Phase 1
visual sanity check，不伪装成 trainable solver 输出。`scripts/audit-world-viewer.mjs`
已经把 Debug OS 纳入 world viewer 验收，检查 debug panel、assignment heatmap、
Gaussian probe、debug protocol 和 object visibility toggle。

`TRAIN-MVP-001` 已完成 ObjGauss v1 trainable kernel smoke loop：
`objgauss/core/trainable_kernel.py` 新增 dependency-free CPU MVP，用 `numpy`
数值梯度优化 assignment logits 和 Gaussian color decoder。该闭环显式走
`frames -> PerceptionEvidence(features/positions) -> A -> ObjectStateProjection ->
Gaussian decode -> point render -> L_render + L_object + L_temporal`，其中
`L_render` 是 point RGB reconstruction MSE，`L_object` 是 evidence-derived
pseudo assignment cross entropy / balance，`L_temporal` 是跨帧 ObjectState centroid
smoothness。`objgauss training kernel-mvp` 可运行内置 toy fixture 并输出
initial / final total loss、render loss、object loss 和 temporal loss；测试证明
total loss 与 render loss 均下降。该步骤不引入 torch / SAM / CLIP / real video
training，不改 viewer renderer，不提交训练产物或大资产，也不把 CPU point-render
MVP 表述为完整 3DGS rasterizer。

`TRAIN-SAMPLE-ADAPTER-001` 已完成 trainable kernel 的真实样例入口：
`objgauss/core/trainable_kernel.py` 新增 `TrainableKernelSample`、
`trainable_kernel_sample_from_cloud(...)` 和 `train_kernel_mvp_from_cloud(...)`，
可从 `GaussianCloud` / object-aware PLY 生成 trainable frames。若输入带
`object_id` 字段，adapter 会映射为 one-hot assignment targets；若没有 object id，
则要求显式 `slots` 并回退到 feature-quantile pseudo targets。`objgauss training
kernel-sample <ply>` 现在可直接读取小型 PLY，按 `max_points` 做 deterministic /
object-aware sampling，再运行同一 `L_render + L_object + L_temporal` smoke loop。
已用 `public/samples/lego_alpha_v1_objects.ply` 验证真实仓库样例路径：
`source_gaussians=5696`、`sampled_gaussians=8`、`slots=4`、`target_source=object_id_one_hot_targets`、
`initial_total_loss=1.516573 -> final_total_loss=1.309766`。该步骤仍不接 near-1M /
4.5M 大资产，不提交训练输出，不引入 torch / SAM / CLIP，也不把 point-render smoke
升级表述为真实 renderer loss。

`TRAIN-RENDER-LOSS-001` 已完成 renderer-loss upgrade boundary contract：
`objgauss/core/renderer_loss.py` 新增 `objgauss-renderer-loss-boundary-v1` 报告，
用于区分当前 `cpu-point-rgb-smoke` 和后续 `differentiable-gaussian-image-renderer`
目标。该 contract 明确 input frame、image-space render target、loss telemetry 和
viewer / training renderer 分工：viewer renderer 继续作为 debug visualization /
browser audit，training renderer 必须作为独立 loss producer 接入，不能默认替换
Three.js / viewer renderer。`objgauss training renderer-loss-contract` 可读取
`kernel-sample` summary，检查 point smoke 是否 ready，并列出升级 blockers：
`image_space_targets_not_bound`、`differentiable_gaussian_renderer_not_selected`、
`renderer_gradient_path_not_defined` 和 `camera_visibility_policy_not_bound`。已用
`public/samples/lego_alpha_v1_objects.ply` 的 kernel summary 验证：
`status=point_render_smoke_ready`、`point_smoke_ready=true`、
`evidence_initial_render_loss=0.084134 -> evidence_final_render_loss=0.061294`。
该步骤仍不引入 torch / GPU renderer / differentiable rasterizer，不提交训练输出，
不把 point-render smoke 宣称为完整 3DGS training。

`TRAIN-IMAGE-TARGET-001` 已完成 image / camera target ABI 绑定：
`objgauss/core/trainable_kernel.py` 新增 `objgauss-train-image-target-contract-v1`
和 `objgauss-train-image-target-v1`，在 `TrainableKernelFrame` 上挂载可选
`TrainableKernelImageTarget`，包含 `H x W x 3` float32 image target、visibility
mask / policy、orthographic debug camera intrinsics 和 `camera_to_world`。`kernel-sample`
新增 `--bind-image-targets`、`--image-width`、`--image-height`、`--point-radius`
和 `--visibility-policy`，可从小型 object-aware PLY 生成 deterministic image-space
target summary。`renderer_loss_boundary_report(...)` 现在能识别
`image_target_contract.status=image_targets_bound`，并移除
`image_space_targets_not_bound` 与 `camera_visibility_policy_not_bound` blockers；
剩余 blockers 明确收敛为 `differentiable_gaussian_renderer_not_selected` 和
`renderer_gradient_path_not_defined`。该步骤仍不引入 torch / GPU renderer /真实
differentiable rasterizer，不替换 Three.js / Spark / WebGPU viewer renderer，不提交
训练输出或大资产。

`TRAIN-RENDERER-API-001` 已完成 dependency-free training renderer API 第一版：
`objgauss/core/training_renderer.py` 新增 `objgauss-training-renderer-api-v1`，
提供 `cpu-image-point-splat-differentiable-v1` renderer loss producer。该模块消费
`TrainableKernelFrame + A[N,K] + decoder_colors`，按已绑定的 image / camera target
生成 image-space render、计算 visibility-masked `image_render_loss`，并暴露
`analytic-color-assignment-gradient-v1` 梯度路径，覆盖 `decoder_colors` 与
`assignments`，同时明确 `positions`、`camera`、`visibility_mask` 和 `point_radius`
仍是 frozen fields。`kernel-sample` 在 image targets 已绑定时会自动写入
`renderer_api` summary，CLI 输出 `renderer_api_status=ready`、renderer name、
gradient path 和 `image_render_loss`。`renderer_loss_boundary_report(...)` 现在把这类
summary 升级为 `status=renderer_api_ready`，移除
`renderer_gradient_path_not_defined` / `differentiable_gaussian_renderer_not_selected`
blockers，剩余 blocker 收敛为 `full_3dgs_renderer_not_selected`。该步骤仍不引入
torch / GPU rasterizer，不替换 viewer renderer，不把 CPU point splat stub 宣称为完整
3DGS renderer；下一步应把 `image_render_loss` 接入训练 objective，或在 ADR 后替换为
真实 3DGS differentiable renderer。

`TRAIN-IMAGE-LOSS-OPTIM-001` 已完成 image-space renderer loss 的训练接入：
`TrainableKernelLoss` 新增 `image_render_loss`，`train_kernel_mvp(...)` /
`train_kernel_mvp_from_cloud(...)` 新增 `image_render_weight`，总损失现在可配置为
`point L_render + image_render_loss + L_object + L_temporal` 的加权组合。默认
`image_render_weight=0.0`，保持旧 smoke 行为；当该权重大于 0 时，每一帧必须绑定
`TrainableKernelImageTarget`。`objgauss training kernel-sample` 新增
`--image-render-weight`，并输出 initial / final image render loss 与
`image_render_loss_decreased`。已用 `public/samples/lego_alpha_v1_objects.ply` 验证：
`initial_image_render_loss=0.082205 -> final_image_render_loss=0.053806`，
同时 `initial_total_loss=1.557675 -> final_total_loss=1.332860`。该步骤仍使用
dependency-free CPU point splat renderer API，不引入 torch / GPU rasterizer，不提交
训练输出或大资产，也不替换 viewer renderer。

`TRAIN-MVP-MODEL-ARTIFACT-001` 已完成 trainable kernel MVP model artifact：
`objgauss/core/trainable_artifact.py` 新增
`objgauss-trainable-kernel-model-artifact-v1`，可把训练后的 assignments、
ObjectState summary、decoder colors、rendered RGB、loss telemetry、renderer API summary
和 source sample provenance 写成可追踪 JSON artifact。`objgauss training kernel-sample`
新增 `--model-output`，默认写到用户指定路径；训练输出仍不提交进 git。已用
`public/samples/lego_alpha_v1_objects.ply` 验证 `/tmp/objgauss-trainable-kernel-model.json`
写出成功，summary 中记录 `model_artifact_schema=objgauss-trainable-kernel-model-artifact-v1`，
同时保留 `renderer_api_status=ready`、`image_render_loss_decreased=true` 和 source
sample provenance。该步骤仍不引入 torch / GPU renderer，不替换 viewer renderer，
不把该 JSON 产物标记为 browser artifact。

`TRAINABLE-MANIFEST-OUTPUT-001` 已完成 trainable kernel MVP 的 viewer-ready manifest
handoff：`objgauss/model_manifest.py` 新增
`manifest_from_trainable_kernel_model_artifact(...)`，可把
`objgauss-trainable-kernel-model-artifact-v1` 包装成
`objgauss-model-artifact-manifest-v1`，其中 `trainable_kernel` artifact 使用
`browser_edit / browser_ready=true`，并记录 gaussian / object counts、byte size、
sha256、training summary 和 renderer API quality evidence。`objgauss training
kernel-sample` 新增 `--manifest-output`、`--manifest-asset-id`、`--manifest-name` 和
`--manifest-license`；当同时传 `--model-output` 时，会写出同目录相对 artifact path，
使输出目录可直接作为 `?modelArtifactManifest=...` 或本地 `导入模型` package 输入。
`scripts/audit-world-viewer.mjs` 新增 trainable-only local manifest package audit，输出
`localTrainableManifest=local-trainable-manifest-debug-os`。该步骤仍不引入 torch /
gsplat / CUDA，不提交 `/tmp` 或 ignored `outputs/` 训练产物，不改变 viewer renderer。

`TRAINABLE-QUALITY-REPORT-001` 已完成 trainable kernel MVP 的 metrics report handoff：
`objgauss/core/trainable_quality.py` 新增
`objgauss-object-state-quality-report-v1` 生成器，直接从
`objgauss-trainable-kernel-model-artifact-v1` 计算 assignment entropy、slot utilization、
object purity proxy、temporal drift、assignment jitter、bbox stability 和 spatial
compactness，并输出 slot utilization / assignment entropy / temporal drift gates。
`objgauss training kernel-sample` 新增 `--quality-report-output` 和 `--quality-report-id`；
当同时写 `--manifest-output` 时，manifest 会把该报告登记为 browser-ready
`quality_report` artifact。`scripts/audit-world-viewer.mjs` 现在验证本地 trainable-only
manifest package 可以同时导入训练 artifact 与 quality report，输出
`localTrainableManifest=local-trainable-manifest-quality-debug-os`，并确认 Quality 面板与
snapshot quality 状态存在。该步骤仍不引入 torch / gsplat / CUDA，不提交 `/tmp` 或
ignored `outputs/` 训练产物，不改变 viewer renderer。

`QUALITY-GATE-UI-001` 已完成 Debug OS 的 quality gate 可视化：`src/App.jsx` 的
`Quality` 面板现在不仅显示 report status 与核心 metrics，还会逐条显示 gate
name、status、value 和 threshold；root / panel telemetry 新增 failing gate names，
`objgauss-object-state-debug-snapshot-v1.quality` 也携带 compact gate rows。`scripts/audit-world-viewer.mjs`
现在验证 algorithm manifest、本地组合 manifest 与 trainable-only local manifest package
中的 `assignment_entropy=warn`、`slot_utilization=pass` / `temporal_drift=pass` 等具体
gate 判定，并确认 snapshot export 中保留 gate 信息。该步骤只增强 viewer-side metrics
debugging，不改变 quality report schema、不改训练算法、不引入 torch / gsplat / CUDA。

`TRAIN-MODEL-VIEWER-BINDING-001` 已完成 trainable MVP model artifact 的 Debug OS
前端绑定：`src/modelCatalog.js` 新增小型 `trainable-mvp-debug` static fixture，
明确标记为 `objgauss-trainable-kernel-model-artifact-v1` debug fixture，不是真实训练
输出发布物。`src/App.jsx` 现在支持 `loadMode="trainable-artifact"`，可从 artifact
中的 assignment matrix、ObjectState summary、decoder colors 和 renderer API telemetry
生成 ObjectState Debug OS render targets、Gaussian probe、bbox overlay、2-slot
assignment heatmap、renderer loss 和 source metadata。前端 inspector / debug panel
会显示 `assignment=trainable_kernel_model_artifact`、renderer name 和
`renderer loss=0.053806`。`scripts/audit-world-viewer.mjs` 已扩展为先验证 OGC
仍是 `derived_from_object_id`，再选择 `trainable-mvp-debug`，断言
`assignmentSource=trainable_kernel_model_artifact`、`assignmentSlots=2`、
`trainableArtifacts=1` 和 Gaussian probe 成功。该步骤不替换 Three.js / Spark /
WebGPU viewer renderer，不提交真实训练输出，不默认加载 near-1M / 4.5M 大资产。

`TRAINABLE-ARTIFACT-FETCH-001` 已完成 Debug OS 的 fetchable trainable artifact delivery：
`src/modelCatalog.js` 不再内联 `trainable-mvp-debug` artifact，而是指向
`/models/trainable-mvp-debug/model-artifact.json`；`src/App.jsx` 在
`loadMode="trainable-artifact"` 路径上会先 fetch 并校验
`objgauss-trainable-kernel-model-artifact-v1`，再把 artifact 注入 Three.js ObjectState
Debug OS。前端 inspector、`.worldShell` telemetry 和
`scripts/audit-world-viewer.mjs` 现在暴露并验证 `loadRoute=fetch-json` 与 artifact path。
新增的 public JSON 仍是 4KB 小型 browser fixture，用于证明“算法处理后的 artifact
文件可以被 viewer 加载”；不提交真实训练输出、不发布大模型、不改变 gsplat / CUDA
blocker。

`TRAINABLE-FRAME-DEBUG-001` 已完成 Debug OS 的 trainable artifact 多帧调试：
`src/App.jsx` 在 `trainable-mvp-debug` 上新增紧凑 frame selector，可在同一 fetched
`objgauss-trainable-kernel-model-artifact-v1` 内切换 `object_states[n]` /
`assignments[n]`，并重新注入 Three.js ObjectState render targets、assignment heatmap
和 stability telemetry。`.worldShell`、Debug panel 和 `window.__OBJGAUSS_WORLD__`
同步暴露 frame index / frame count；`scripts/audit-world-viewer.mjs` 现在在桌面与移动
viewport 中选择 Trainable MVP、点击 `f1`，并验证 `trainableFrame=1/2`、Gaussian probe
和 hover highlight 仍来自 `trainable_kernel_model_artifact`。该切片只增强已加载 artifact
的可交互调试，不改变训练算法、不引入 torch / gsplat / CUDA、不提交训练输出。

`TRAINABLE-URL-ARTIFACT-001` 已完成 Debug OS 的 URL trainable artifact 注入入口：
`src/modelCatalog.js` 新增运行时 catalog builder，当前同源 URL 参数
`?trainableArtifact=/path/to/model-artifact.json` 会追加 `trainable-url-artifact` 临时模型；
`src/App.jsx` 使用运行时 catalog，并在 URL artifact 存在时默认选中该模型。该入口仍复用
`loadMode="trainable-artifact"` 的 schema 校验、frame selector、ObjectState render
targets、heatmap、stability telemetry 和 Gaussian probe。`scripts/audit-world-viewer.mjs`
现在额外打开 URL 注入路径，验证模型数变为 8、默认选中 URL artifact、`urlArtifact=fetch-json`
且 frame 1 可审计。该切片只支持同源 `.json` 路径，不支持远程 URL，不提交训练输出或大模型。

`TRAINABLE-LOCAL-ARTIFACT-IMPORT-001` 已完成 Debug OS 的本地 trainable artifact 导入入口：
`src/App.jsx` 的 top HUD 新增 `导入训练` 文件入口，可直接读取本地
`objgauss-trainable-kernel-model-artifact-v1` JSON，复用同一 schema 校验和
trainable artifact hydrate / upsert helper，把本地文件作为运行时
`trainable-local-artifact` 注入 Three.js ObjectState Debug OS。该模型会生成同样的
ObjectState render targets、assignment heatmap、frame selector、Training evidence、
snapshot 和 event trace；delivery route 明确标为 `local-file`，artifact path 标为
`local://<file>`。root telemetry 新增 `data-trainable-import-*`，并将
`data-model-count` / `window.__OBJGAUSS_WORLD__.modelCount` 改为运行时 scene model count，
同时保留 `data-catalog-model-count`。`scripts/audit-world-viewer.mjs` 现在用
Playwright `setInputFiles(...)` 导入小型 fixture，验收输出 `localArtifact=local-file`。
该入口服务 ignored `outputs/` 或 `/tmp` 中的新训练产物调试，不把训练产物复制进
`public/` 或 git；不改变训练算法、artifact schema、renderer API 或 gsplat / CUDA
blocker。

`TRAINABLE-LOSS-DEBUG-UI-001` 已完成 trainable artifact 的训练证据可视化：
`src/App.jsx` 从 `objgauss-trainable-kernel-model-artifact-v1.training` 与
`renderer_api` 只读生成 evidence summary，并把 `loss_down`、iterations、final total loss、
loss delta、final image loss 和 image loss delta 暴露到 `.worldShell` 的
`data-trainable-training-*` telemetry。ObjectState Debug panel 现在新增 `Training`
evidence 面板，显示 total / image / object / temporal loss、delta、iteration、
renderer 和 gradient path；floating inspector 同步显示 `train loss` 与 `loss delta`。
`scripts/audit-world-viewer.mjs` 在默认 trainable fixture 与 URL injected trainable artifact
两条 browser path 下断言 training panel 存在、renderer 为
`cpu-image-point-splat-differentiable-v1`、image loss decreased、loss delta 为正。该切片
不改变训练算法、artifact schema、renderer API 或 `TRAIN-GSPLAT-MVP-001` 的 optional
deps / CUDA blocker，不提交新的训练输出。

`OBJECT-DEBUG-LENS-001` 已完成 Phase 1 Debug OS 的 lens system：`src/App.jsx`
将原先二元 `A[N,K]` color toggle 扩展为 `assignment / confidence / entropy` 三种
debug lens。Debug panel 新增紧凑 `lens` selector；root `.worldShell`、Debug panel 和
`window.__OBJGAUSS_WORLD__` 同步暴露当前 lens。Three.js world 不重建模型，而是在每个
Gaussian cloud 上保留 assignment、confidence、entropy 三套 color buffer，切换 lens 时
只替换 active color attribute；confidence / entropy lens 还会根据 ObjectState
confidence 或 normalized entropy 调整非选中 cloud opacity，形成可审计的 opacity
debugging。`scripts/audit-world-viewer.mjs` 在 trainable artifact 路径下点击 `conf` 和
`H`，断言 DOM、audit handle、active lens 和 opacity samples 均同步，最终截图保留
entropy lens。该切片不改变训练算法、artifact schema、OGC schema 或 renderer 依赖，
也不解除 gsplat / CUDA blocker。

`OBJECT-DEBUG-SNAPSHOT-001` 已完成 ObjectState Debug OS 的 snapshot protocol：
`src/App.jsx` 新增 `objgauss-object-state-debug-snapshot-v1`，从当前 selected model /
object、Gaussian probe、debug lens、assignment vector、ObjectState、stability、
training evidence 和 delivery 摘要生成只读调试快照。该 snapshot 同步暴露到
`window.__OBJGAUSS_DEBUG_SNAPSHOT__`、root `.worldShell` 的 `data-debug-snapshot-*`
attributes，以及 Debug panel 的 `Protocol` 面板。`scripts/audit-world-viewer.mjs`
现在在 trainable artifact + entropy lens 路径下断言 global snapshot、root attributes
和 Protocol panel 三者一致，并验证 snapshot 绑定
`trainable_kernel_model_artifact`、2-slot assignment、`loss_down` training evidence 和
当前 stability 状态。该切片不写出 snapshot 文件、不改变训练 / OGC artifact schema、
不替换 viewer renderer，也不解除 gsplat / CUDA blocker。

`OBJECT-DEBUG-TRACE-001` 已完成 ObjectState Debug OS 的 profiler-style event trace：
`src/App.jsx` 现在维护最近 12 条 `objgauss-debug-event-v1`，覆盖 model / object select、
Gaussian probe、debug lens、trainable frame、hover、visibility toggle 和 OGC LOD /
chunk scope 等调试操作。事件同步暴露到
`window.__OBJGAUSS_DEBUG_EVENTS__`、root `.worldShell` telemetry、Debug panel 的
`Trace` 面板，以及 `objgauss-object-state-debug-snapshot-v1.events`。world viewer audit
现在要求 `gaussian-probe`、`debug-lens`、`frame-select`、`hover-object` 和
`toggle-visibility` 同时出现在 global / root / panel / snapshot 四个入口中，并输出
`debugEvents=12`。该切片只补可审计调试协议，不改变训练算法、trainable artifact schema、
OGC schema、renderer 依赖或 gsplat / CUDA blocker。

`TRAIN-FULL-3DGS-RENDERER-ADR-001` 已完成 renderer 依赖路径冻结：新增
`docs/adr/0006-full-3dgs-training-renderer.md`，选择 `gsplat-rasterization-v1`
作为 ObjGauss v1 full differentiable 3DGS training renderer 的第一优先实验路径。
ADR 明确 `torch` / `gsplat` 只能作为 optional extra 或 import-guarded adapter 引入，
不进入基础 dependencies；第一版 full renderer 只训练 `assignments` 和
`decoder_colors`，冻结 Gaussian geometry / opacity / cameras；Three.js / Spark /
WebGPU viewer renderer 继续作为 ObjectState Debug OS consumer，不被训练 renderer
替换。下一步代码切片是 `TRAIN-GSPLAT-ADAPTER-001`，只做可选 gsplat adapter 的
smoke path，不接入 optimizer、不改 CLI 默认行为、不提交训练输出。

`TRAIN-GSPLAT-ADAPTER-001` 已完成 optional gsplat training renderer adapter 的
import-guarded contract：新增 `objgauss/core/gsplat_training_renderer.py`，定义
`gsplat-rasterization-v1`、`torch-autograd-gsplat-rasterization-v1`、availability
summary 和 `objgauss-gsplat-training-input-v1`。该模块可把现有
`TrainableKernelFrame + A[N,K] + decoder_colors` 映射为 gsplat 所需的 means /
quats / scales / opacities / colors / viewmats / intrinsics / target image /
visibility mask，并在真实调用 gsplat 前检查 `torch`、`gsplat` 和 CUDA blockers。
当前仍不修改 `pyproject.toml`、不安装依赖、不改 CLI 默认行为、不接 optimizer、不替换
viewer renderer；无 optional dependency 环境下测试覆盖 unavailable blockers 和 input
contract。下一步是 `TRAIN-GSPLAT-LOSS-001`，把 trainable kernel image renderer loss
producer 做成可选 `point|gsplat`，默认继续使用 CPU point path。

`TRAIN-GSPLAT-LOSS-001` 已完成 optional gsplat loss route：`train_kernel_mvp(...)`
和 `train_kernel_mvp_from_cloud(...)` 新增 `image_renderer="point|gsplat"`，默认仍为
CPU point renderer；CLI `objgauss training kernel-mvp` / `kernel-sample` 新增
`--image-renderer point|gsplat` 并在 summary / stdout 中记录 `image_renderer`。
image render loss producer 现在按该字段路由到 CPU point renderer 或
`evaluate_gsplat_training_renderer_loss(...)`；显式 gsplat 不可用时暴露 adapter
blockers，不静默 fallback。`renderer_loss_boundary_report(...)` 同时识别
`gsplat-rasterization-v1` 为 full 3DGS renderer evidence，该 evidence 下会关闭
`full_3dgs_renderer_not_selected` blocker。本步骤仍不安装 torch / gsplat、不训练
Gaussian geometry / opacity / rotation、不替换 viewer renderer、不提交训练输出。
下一步是 `TRAIN-GSPLAT-MVP-001`：在可用 optional deps / CUDA 环境下跑第一个小规模
full renderer training MVP；如果当前环境不可用，必须记录 blockers。

当前本机环境已验证无法执行 `TRAIN-GSPLAT-MVP-001`：`gsplat_renderer_availability()`
返回 `available=False`，blockers 为 `optional_dependency_missing:torch` 和
`optional_dependency_missing:gsplat`；`nvidia-smi` 无法连接 NVIDIA driver。显式运行
`objgauss training kernel-sample ... --image-renderer gsplat` 会按设计失败并输出同样
blockers，没有静默 fallback 到 point renderer。该状态说明 full renderer MVP 需要转到
具备 torch / gsplat / CUDA 的环境，或先由 Owner 明确批准安装 / 配置 optional training
renderer 依赖；训练模型主线当前已在 PR 队列标记为
`suspended / current-env-missing-torch-gsplat-cuda`，挂起期间不在当前环境重复尝试
full renderer training MVP。

`OBJECT-STABILITY-DASHBOARD-001` 已完成 Phase 1 Debug OS 的 stability dashboard：
`src/App.jsx` 新增 `objgauss-stability-dashboard-v1` summary，按 selected model 的
ObjectState metadata 计算 slot utilization、mean / max entropy、mean / min
confidence、mixed slots、low-confidence slots，并在 metadata 存在时显示 purity /
temporal drift。Debug panel 现在在 assignment heatmap 下方显示 Stability dashboard；
移动端布局优先保留 Debug OS 面板，隐藏会重叠的 floating inspector / bottom status；
`.worldShell` 和 `window.__OBJGAUSS_WORLD__` 暴露 stability status / slot utilization /
mean entropy / mixed slots / low-confidence slots。`scripts/audit-world-viewer.mjs` 已验证
trainable artifact 路径下 dashboard status 与 audit handle 一致，`slotUtil=1`，
`mixedSlots=2`，截图为 `/tmp/objgauss-world-viewer.png` 和
`/tmp/objgauss-world-viewer-mobile.png`。该切片不改变训练依赖、
不替换 viewer renderer、不提交训练输出；缺失的 purity / temporal drift 会明确显示为
`n/a`，不伪装成已计算指标。

`OBJECT-HOVER-HIGHLIGHT-001` 已完成 Phase 1 Debug OS 的 object-hover 高亮审计：
`src/App.jsx` 现在把 hovered ObjectState target 暴露为 `.worldShell` telemetry 和
`window.__OBJGAUSS_WORLD__` audit handle，包括 hovered model / object / selection id、
assigned Gaussian count 和 assignment source。对象 hover 时对应 Gaussian cloud 会提高
opacity / point size，并显示 ObjectState bbox / selection ring / core glow；Debug panel
的 hover 行会显示当前 hover 对象和 Gaussian 数量。`scripts/audit-world-viewer.mjs`
新增 `hoverObjectForAudit(...)` 路径验证，在 `trainable-mvp-debug` artifact 上断言
hover 后 `data-hovered-target`、`hoveredGaussianCount` 和
`hoveredAssignmentSource=trainable_kernel_model_artifact` 一致。该切片不引入 torch /
gsplat / CUDA，不改变训练 blocker，不替换 viewer renderer，不提交训练输出或大资产。

`OBJECT-STABILITY-METRICS-001` 已完成 Phase 1 Debug OS 的 purity / temporal
metric evidence：`trainable-mvp-debug` fixture 现在是 2-frame
`objgauss-trainable-kernel-model-artifact-v1`，携带相邻帧 assignment matrix、
derived object ids 和 ObjectState centroid。`src/App.jsx` 在加载 trainable artifact
时从 dominant assignment rows 与 `derived_object_ids` 推导 per-object purity，并从相邻
ObjectState centroid 推导 temporal drift；Stability dashboard、`.worldShell` telemetry
和 `window.__OBJGAUSS_WORLD__.stabilitySummary` 都暴露 `meanPurity` 与
`meanTemporalDrift`。`scripts/audit-world-viewer.mjs` 现在要求 trainable artifact 路径
下 purity / temporal metrics 均可用，验收输出为 `purity=1`、
`temporalDrift=0.018`。该切片不改变 Python 训练算法、不引入 torch / gsplat / CUDA、
不替换 viewer renderer，不提交 `/tmp` 训练产物。

`OBJECT-SPATIAL-COMPACTNESS-001` 已完成 Phase 1 Debug OS 的空间连续性指标：
`src/App.jsx` 现在在每个 object render target 生成时，从对象 Gaussian positions 和
bbox 派生 `spatialCompactness`，并把该字段写入 ObjectState debug metadata。Stability
dashboard 新增 compactness bar 和 `spatial` meta，`.worldShell` telemetry 与
`window.__OBJGAUSS_WORLD__.stabilitySummary` 同步暴露 `meanSpatialCompactness`。
`scripts/audit-world-viewer.mjs` 现在要求 trainable artifact 路径下 compactness metric
可用，验收输出为 `compactness=0.5`；桌面和移动截图证明新增指标没有遮挡现有
ObjectState Debug panel。该切片只补 viewer-side debug evidence，不改 Python 训练算法、
不引入 torch / gsplat / CUDA、不替换 viewer renderer。

`OBJECT-ASSIGNMENT-JITTER-001` 已完成 Phase 1 Debug OS 的 assignment stability
指标：`src/App.jsx` 在 trainable artifact 路径上比较当前帧和相邻帧同一 Gaussian
row 的 A[n,k] vector，使用 half-L1 distance 派生 per-object `assignmentJitter`，并在
`objgauss-stability-dashboard-v1` 中汇总为 `meanAssignmentJitter`。Stability dashboard
新增 jitter bar / meta，缺少相邻 assignment frame 的普通模型明确显示 `n/a`，不会把缺失
evidence 伪装成 0。`.worldShell` telemetry 与 `window.__OBJGAUSS_WORLD__.stabilitySummary`
同步暴露该指标。`scripts/audit-world-viewer.mjs` 现在要求 trainable artifact 路径下
jitter metric 可用，验收输出为 `assignmentJitter=0.023`。该切片只补 A[N,K] stability
debug evidence，不改变 Python 训练算法、不引入 torch / gsplat / CUDA、不替换 viewer
renderer。

`OBJECT-BBOX-STABILITY-001` 已完成 Phase 1 Debug OS 的 bbox convergence 指标：
`src/App.jsx` 在 trainable artifact 路径上比较当前帧和相邻帧同一 ObjectState 的
3D bbox IoU，派生 per-object `bboxStability`，并在
`objgauss-stability-dashboard-v1` 中汇总为 `meanBboxStability`。Stability dashboard
新增 `bbox` meta；缺少相邻 ObjectState frame 的普通模型继续显示 `n/a`，不把缺失
evidence 伪装成稳定。`.worldShell` telemetry、`window.__OBJGAUSS_WORLD__` 和
`scripts/audit-world-viewer.mjs` 均暴露并验证该指标，验收输出为
`bboxStability=0.899`。该切片只补 viewer-side bbox stability evidence，不改变 Python
训练算法、不引入 torch / gsplat / CUDA、不替换 viewer renderer。

## Hugging Face 开发阶段发布

2026-06-29 已为 near-1M NeRF Lego trained candidate 建立 Hugging Face
development-stage release 记录，事实源见 `docs/state/huggingface-release.md`。

- Dataset: `https://huggingface.co/datasets/jianyong365/objgauss-nerf-lego-near1m`
- Model: `https://huggingface.co/jianyong365/objgauss-nerf-lego-near1m-model`
- Dataset README development-stage commit: `9c202e5393c6dea70b1a077e5b70766205d83c87`
- Model README development-stage commit: `e4364f247644062bd22bbeceda974dd69a86f06d`

该 HF 发布仅用于开发阶段研究复现和下载 handoff。当前不能标记为 stable release、
production ready 或 commercial demo；large object-aware PLY / checkpoint 已完成远端上传
和 metadata 核对，Dataset verified head 为 `295b13f8bac09bc302019ab6c9d238d11d2d6538`，
Model verified head 为 `82b700392699852c62dca70ac4274dc722d82282`。

## Near-1M 终局证据状态

2026-06-29 已用 HF 远端已核对的 `4,503,634`-Gaussian object-aware PLY
派生出 deterministic sampled1m 本地审计输入，并通过 WebGPU C-path production
SLA。该 sampled1m 产物在 ignored `outputs/` 下，不进入 git，也尚未作为 HF stable
asset 发布：

- Sampled PLY: `outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m/object_aware_gaussians.ply`
- Gaussians: `1,000,000`
- Byte size: `255,001,677`
- sha256: `354440011354a80aeb23a357466e65ccb9c2f2ace07c3756590ab7e203271ea1`
- Object counts: `0=450284`、`1=112243`、`2=360042`、`3=77431`
- 生成命令: `npm run assets:sample-ply -- --input outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-candidate/object_aware_gaussians.ply --output outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m/object_aware_gaussians.ply --target-gaussians 1000000 --manifest-output outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m/sample-manifest.json --label near1m-random1300k-v1-deterministic-sampled1m --overwrite`
- SLA report: `/tmp/objgauss-webgpu-cpath-production-sla-sampled1m-v2/summary.md`
- strict route-goal report: `/tmp/objgauss-renderer-route-goal-production-ready-sampled1m-v2/summary.md`

关键验收结果：`webgpu_cpath_production_sla=passed`，`trainedGaussians=1000000`，
`realTrainedBrowserRuntime1m=passed`，`fpsSla=passed`，sustained trained
min approx FPS 为 `33.804`，`audit:near1m-production-gap -- --require-ready`
返回 `ready`，`audit:renderer-route-goal -- --require-production-ready` 在显式绑定
sampled1m candidate / SLA 目录时返回 `ready`。

边界：HF Dataset 中的全量 object-aware PLY 仍是 `4,503,634` Gaussians。该全量
PLY 的直接 browser runtime 审计失败，min approx FPS 为 `4.412`；因此不能把
HF 全量 PLY 误称为 production-interactive。下一步若要让全量 HF PLY 直接交互，
需要 LOD、streaming、分块加载或专门的全量性能优化。

## Large Model Viewer 默认路线

2026-06-29 `RENDER-ROUTE-032` 已把 near-1M trained asset 的默认查看路线整理为：

- 素材库支持 `全部` / `训练模型` / `near-1M` / `商用` 筛选，`?asset-filter=trained`
  可直接显示 `nerf-lego-trained-near1m-random1300k-local`。
- near-1M card 默认 `快速查看` 只加载
  `public/samples/nerf_lego_trained_near1m_random1300k.splat`，不请求
  `1.15GB` object-aware PLY。
- 只有点击 `加载对象 PLY` 或进入对象编辑时，才加载
  `public/samples/nerf_lego_trained_near1m_random1300k_objects.ply`。
- root DOM 暴露 active asset、筛选数量、object PLY load state / mode / path，
  由 `npm run audit:large-model-viewer-route` 审计。

本地验证结果：`audit:large-model-viewer-route` 通过，训练筛选 card 数 `4`，
quick view `.splat` 请求数 `1`、quick object PLY 请求数 `0`，对象编辑和直接按钮
各触发 `1` 次 object PLY 请求。该审计使用 Playwright fulfill 小型 PLY 来证明请求
策略，不下载 HF 全量 PLY；截图在 `/tmp/objgauss-large-model-viewer-route.png`。

## Depth-Aware Mask Vote 诊断

2026-06-29 `TRAIN-QUALITY-002` 已落地 depth / visibility-aware mask voting 诊断：

- 默认 Object Field 训练仍使用 legacy projected voting，不改变现有 public samples。
- `vote_masks_to_gaussians(..., visibility_mode="depth-buffer")` 可用每帧像素级 z-buffer
  只让最前方 Gaussian 消费 mask vote，并记录 `depth_culled_matched`。
- `objgauss object-field vote-diagnostics` 会同时跑 projected baseline 与 depth-buffer
  diagnostic，输出 `objgauss-depth-visibility-vote-diagnostic-v1` summary。

真实 Lego 样例诊断结果：

- SAM balanced safe-2000: conflict `0.021192 -> 0.018820`，slot balance
  `0.001571 -> 0.001748`，`depth_culled_matched=8271`。
- Alpha foreground/background: conflict `0.430143 -> 0.402118`，slot balance
  `0.130996 -> 0.133745`，`depth_culled_matched=904903`。

结论：depth-buffer diagnostic 在真实 Lego trained / alpha fgbg 样例上能降低 vote
conflict 并轻微改善 slot balance，但还没有 promotion 为默认训练策略。下一步应基于该
summary 做跨视角 slot alignment、CLIP 命名和 promotion gate，而不是只增加训练步数。

## Cross-View Slot Alignment / CLIP 命名

2026-06-30 `SEG-CLIP-001` 已落地 manifest-level 跨视角 slot alignment 和
CLIP-score 命名入口；`CLIP-SCORE-001` 已补上 mask crop 的 CLIP score cache 入口：

- 新增 `objgauss masks align-slots`，按每个 2D mask 投影命中的 Gaussian support
  聚类，重写稳定的跨视角 `slot` / `slot_id`，并保留 `source_slot`、
  `source_label` 和 `aligned_slot`。`CLIP-SLOT-QUALITY-002` 已补 slot-level
  naming quality gate、低面积 mask 过滤和 CLIP top-label 背景过滤。
- 新增 `objgauss masks score-clip`，按 mask crop 对一组文本 label 写入
  `clip_scores` 与 `clip` metadata；真实推理走可选 `transformers` backend，
  测试 / contract smoke 走 `hash` diagnostic backend。`CLIP-QUALITY-001`
  已补 label preset、prompt template 聚合、mask background fill 和
  `objgauss-clip-naming-quality-v1` summary gate。
- 如果 manifest 已包含预计算 `clip_scores` / `clip_label_scores` /
  `semantic_scores` / `clip.scores`，alignment 会按 cluster 聚合分数并给 slot 写
  `label` / `name` / `semantic_name_source="clip_scores"`。
- 输出 manifest 会重写相对 `mask_path` 和 frame 级 `ignore_mask_path`，因此可安全
  写到 `/tmp` 或其他目录后继续被 `masks validate`、`vote-diagnostics` 和
  `vote-masks` 消费。

真实 Lego SAM balanced safe-2000 top-4 诊断结果：

- Alignment: `/tmp/objgauss-sam-aligned-top4-mask-manifest.json`，`frames=6`，
  `masks=11`，`aligned_slots=4`，`remapped_masks=6`，`dropped_masks=16`，
  `named_slots=0`。
- Manifest validate: `/tmp/objgauss-sam-aligned-top4-validation.json`，passed，
  zero overlap，slots `0,1,2,3`。
- Depth-aware diagnostic: `/tmp/objgauss-sam-aligned-top4-vote-diagnostic.json`，
  conflict `0.015662 -> 0.014064`，`depth_culled_matched=8101`，
  recommendation `depth-aware-diagnostic-improved`。
- Object Field vote-masks smoke:
  `/tmp/objgauss-sam-aligned-top4-object-aware-gaussians.ply`，`255,794`
  Gaussians，loss `2.737449 -> 0.595588`，object counts
  `0=80479`、`1=50125`、`2=57008`、`3=68182`。
- CLIP score cache contract smoke:
  `/tmp/objgauss-sam-clip-hash-mask-manifest.json`，`backend=hash-diagnostic`，
  `frames=8`，`masks=27`，`scored_masks=27`，`named_masks=27`。
- CLIP-score downstream alignment smoke:
  `/tmp/objgauss-sam-clip-hash-aligned-top4-mask-manifest.json`，`aligned_slots=4`，
  `named_slots=4`，slot labels 来自 `clip_scores`。
- Real CLIP scoring run:
  `uv run --with torch --with transformers --with pillow --with socksio objgauss masks score-clip ... --backend transformers --device cuda`
  已用 `openai/clip-vit-base-patch32` 跑通真实 CLIP inference；summary 为
  `/tmp/objgauss-sam-clip-real-summary.json`，`frames=8`，`masks=27`，
  `scored_masks=27`，`named_masks=27`。本次临时依赖环境为
  `torch 2.12.1+cu130`、`transformers 5.12.1`，CUDA 可用；GPU preflight 显示
  RTX 5060 Ti 空闲约 `15193MiB`。
- Real CLIP naming quality gate:
  `/tmp/objgauss-sam-clip-real-mask-manifest.json` validation passed；
  `/tmp/objgauss-sam-clip-real-aligned-top4-mask-manifest.json` 得到
  `aligned_slots=4`、`named_slots=4`。但 mask top labels 分布为
  `lego tread=23`、`table surface=3`、`white background=1`；aligned top-4 labels 为
  slot 0 `white background`、slot 1/2/3 `lego tread`。该结果证明真实 CLIP
  运行链路可用，但当前 label set / crop 策略不能 promotion。
- CLIP quality v1 gate:
  `objgauss masks score-clip` 已支持 `--label-preset nerf-lego-v1|lego-parts-v1`、
  repeatable `--prompt-template`、`--background-fill white|black|gray|mean|image`
  和 `--require-naming-quality`。真实 run
  `/tmp/objgauss-sam-clip-quality-v1-summary.json` 使用 `nerf-lego-v1`、3 个 prompt
  template 与 `background_fill=mean`，mask top labels 从旧 run 的
  `lego tread=23/27` 改善为 5 类：`white background=11`、
  `lego wheel tread=10`、`cast shadow=3`、`black rubber tire=2`、
  `table surface=1`。但 gate 仍因 `background-label-dominant` 失败。
- CLIP quality v1 alignment:
  `/tmp/objgauss-sam-clip-quality-v1-aligned-top4-mask-manifest.json` 得到
  `aligned_slots=4`、`named_slots=4`，slot labels 仍是
  `white background`、`lego wheel tread`、`lego wheel tread`、`white background`。
  object-only strict diagnostic
  `/tmp/objgauss-sam-clip-quality-objectonly-strict-v1-summary.json` 在
  `--min-unique-top-labels 3` 下失败，top labels 为
  `lego wheel tread=19`、`black rubber tire=8`。
- CLIP slot quality v1 gate:
  `objgauss masks align-slots` 已支持 `--require-slot-quality`、
  `--min-mask-area`、`--min-mask-area-fraction`、`--exclude-top-labels` 和
  `--exclude-background-top-labels`。未过滤的 quality v1 alignment 在
  `--require-slot-quality` 下失败，slot labels 为 `white background=2`、
  `lego wheel tread=2`，blocker=`background-slot-dominant`。使用
  `--exclude-background-top-labels --min-mask-area 500` 后，
  `/tmp/objgauss-sam-clip-slot-quality-filtered-v1-aligned-top4-mask-manifest.json`
  validation passed，但 `filtered_low_area=10`、`filtered_top_label=9` 后剩余
  slot labels 全部为 `lego wheel tread=4`，blockers 为
  `not-enough-unique-slot-labels` 和 `slot-label-dominant:lego wheel tread`。
  object-only slot gate 同样失败，slot labels 为
  `lego wheel tread=3`、`black rubber tire=1`。
- CLIP baseline comparison / promotion policy:
  `objgauss masks compare-baselines` 已将 CLIP slot naming 与 SAM、alpha、
  color-mask 和 KMeans / Object Field 初始 PLY reference 放到同一份
  `objgauss-clip-baseline-comparison-v1` summary；输出内置 development-stage notice。
  真实 comparison summary 为
  `/tmp/objgauss-clip-baseline-comparison-v1-summary.json`，Markdown report 为
  `/tmp/objgauss-clip-baseline-comparison-v1-summary.md`，候选包含
  `clip_unfiltered`、`clip_filtered`、`clip_objectonly`、`sam`、`alpha`、`color`
  和 `kmeans`，结论为 `promotion_policy=do-not-promote`。
  关键阻断：未过滤 CLIP slots 为 `lego wheel tread=2`、`white background=2`；
  filtered slots 全部为 `lego wheel tread=4`；object-only slots 为
  `lego wheel tread=3`、`black rubber tire=1`；三组 CLIP 语义候选都缺少
  downstream vote quality / training summary evidence。Baseline reference 行记录了
  SAM conflict `0.021192` / slot balance `0.001571`、alpha conflict `0.430143`
  / slot balance `0.130996`、color-mask loss `1.386294 -> 0.390825`、
  KMeans active slots `8` / slot balance `0.875182`。
- CLIP slot naming diversity policy:
  `objgauss masks align-slots` 已支持 `--foreground-only-slot-names`、
  `--unique-slot-names` 和 `--slot-name-diversity-penalty`，并在输出 manifest 的
  `slot_alignment.naming_policy`、slot 定义和 mask entry 上记录使用的 naming policy。
  真实 quality v1 run 在
  `/tmp/objgauss-sam-clip-slot-quality-diverse-v1-aligned-top4-mask-manifest.json`
  通过 slot naming gate：slot labels 从 filtered 旧策略的 `lego wheel tread=4`
  改为 `lego wheel tread`、`black rubber tire`、`gray wheel hub`、
  `yellow lego vehicle body` 各 1 个；manifest validate summary 为
  `/tmp/objgauss-sam-clip-slot-quality-diverse-v1-validation.json`，passed。
  该 aligned manifest 的 downstream `vote-masks` smoke 写出
  `/tmp/objgauss-sam-clip-slot-quality-diverse-v1-mask-training-summary.json`，
  loss `3.245392 -> 0.219241`，active slots `4`，conflict `0.009012`。
  但 comparison summary
  `/tmp/objgauss-clip-quality-004-comparison-summary.json` 仍是
  `promotion_policy=do-not-promote`，blockers 为 mask-level
  `background-label-dominant`、supervised fraction `0.114960 < 0.200000`、
  slot balance `0.006349 < 0.010000`。
- CLIP slot support rebalance policy:
  `objgauss masks align-slots` 已支持 `--min-slot-support-gaussians`、
  `--min-slot-support-ratio` 和 `--min-balanced-slots`，可显式丢弃 aligned 后
  Gaussian support 太弱的 slot，并在 manifest 的 `slot_alignment.slot_rebalance`
  与 `compare-baselines` candidate summary 里保留 kept / dropped 证据。真实 quality v1
  balanced run 写出
  `/tmp/objgauss-sam-clip-slot-balance-v1-aligned-mask-manifest.json`，使用
  `--min-slot-support-ratio 0.01 --min-balanced-slots 3` 后得到 `frames=3`、
  `masks=4`、`aligned_slots=3`，slot labels 为 `lego wheel tread`、
  `black rubber tire`、`gray wheel hub`；丢弃弱 support slot `yellow lego vehicle body`
  的 source order `3`，support `225 < 317`。kept support 为
  `[31637, 1248, 1089]`，manifest support balance score `0.034422`。
  Manifest validate summary 为
  `/tmp/objgauss-sam-clip-slot-balance-v1-validation.json`，passed，errors /
  warnings `0`。Downstream `vote-masks` smoke 写出
  `/tmp/objgauss-sam-clip-slot-balance-v1-mask-training-summary.json`，loss
  `3.246389 -> 0.218255`，supervised fraction `0.114283`，conflict `0.008244`，
  active winner slot balance `0.028220`。Comparison summary
  `/tmp/objgauss-clip-balance-001-comparison-summary.json` 仍是
  `promotion_policy=do-not-promote`，但 blockers 已收敛为
  `mask-naming:background-label-dominant` 和
  `supervised_fraction-below-threshold:0.114283<0.200000`，slot balance blocker
  已清除。
- CLIP foreground coverage recovery:
  `CLIP-COVERAGE-001` 已在 `objgauss masks align-slots` 增加显式
  `--recover-foreground-coverage` 开关。该机制只处理被 slot support rebalance 丢弃的弱
  slot mask：当其 top label 为非背景，且能通过相同语义 label 或 Gaussian overlap 映射到
  已保留 slot 时，输出 manifest 会把该 mask 标记为 `coverage_only=true` 并作为
  downstream `vote-masks` 的 foreground coverage supervision 保留；它不增加
  `slot_count`，不参与 slot 命名，也不放宽 promotion threshold。fixture 验证中，被
  rebalance 丢弃的 1 个 foreground mask 恢复后，downstream supervised fraction 从不完整
  coverage 提升到 `1.0`，winner balance 为 `[4,3]`。真实 Lego SAM / CLIP balanced route
  已用现有真实 CLIP score cache 重跑
  `align-slots --recover-foreground-coverage -> vote-masks -> compare-baselines`：aligned
  manifest 写到 `/tmp/objgauss-clip-coverage-recovery-aligned-mask-manifest.json`，恢复
  1 个 `coverage_only` foreground mask，`recovered_gaussian_support=225`，frames / masks
  从 `3 / 4` 提升到 `4 / 5`。Manifest validate passed；downstream `vote-masks`
  写到 `/tmp/objgauss-clip-coverage-recovery-mask-training-summary.json`，supervised
  fraction 从 `0.114283` 仅提升到 `0.114960`，conflict 为 `0.008196`，slot balance 为
  `0.028042`。Comparison summary
  `/tmp/objgauss-clip-coverage-recovery-comparison-summary.json` 仍为
  `promotion_policy=do-not-promote`，blockers 仍是
  `mask-naming:background-label-dominant` 和
  `supervised_fraction-below-threshold:0.114960<0.200000`。

边界：真实 CLIP inference 已通过临时 `uv --with` 依赖环境跑通，但仓库默认依赖仍不包含
torch / transformers，也不提交 CLIP 权重或模型 cache。当前已落地 mask-level CLIP
命名质量 gate、slot-level gate、baseline comparison、slot naming diversity policy 和 slot
support rebalance policy，以及显式 foreground coverage recovery 机制；但真实 Lego SAM
balanced safe-2000 的语义路线仍结论为
`do-not-promote`：slot-level 命名塌缩已经被 diversity / foreground-only policy 缓解，
slot balance blocker 也已被 support rebalance 清除；剩余关键问题是 mask-level 背景占比
仍高，且真实 filtered / balanced run 的 supervised fraction 仍需用新 recovery 机制重跑
验证。下一步不是把当前 CLIP labels 作为默认语义质量策略或放宽 promotion threshold，而是
继续降低 mask-level background dominant，并从 SAM / CLIP mask selection 或 crop /
label policy 上扩大真实 foreground coverage；当前 recovery 机制本身有效，但现有真实
输入只恢复了 225 个 Gaussian，不足以清除 supervised fraction blocker。

## 阶段最终目标

当前阶段的最终目标不是先追求完整科研级训练质量，而是把 ObjGauss v1 的最小闭环变成可重复验收的工程事实：

```text
多视角数据 / 3DGS 场景
  -> Gaussian 场
  -> 每个 Gaussian 的 Object Field 概率
  -> 2D mask / 语义线索修正 object_logits
  -> 导出 object_id
  -> 前端可选择、隔离、删除对象
```

验收视角：一条命令能重新生成 Plush semantic、Plush v1 与 NeRF Lego 三个闭环样例，机器检查产物，再打开浏览器验证真实 splat 外观和对象级交互。

## 已完成能力

- Python CLI:
  - `objgauss convert-splat`
  - `objgauss cluster`
  - `objgauss colorize`
  - `objgauss filter`
  - `objgauss stats`
  - `objgauss assets list/pull`
  - `objgauss masks from-nerf-alpha/from-nerf-alpha-fgbg/from-nerf-rgba-colors/from-nerf-sam/validate/score-clip/align-slots/compare-baselines`
  - `objgauss training register-output/write-sample-bundle`
  - `objgauss demo v1-closure/verify-v1-closure/plush-semantic-closure/verify-plush-semantic-closure/lego-alpha-closure/verify-lego-alpha-closure/audit-v1-goal`
  - `objgauss object-field init/export/stats/emergence/emergence-curve/emergence-report/emergence-benchmark/inspect-nerf/vote-masks/vote-diagnostics`
- 前端:
  - 中文 UI。
  - Spark / Three.js 真实 3DGS splat 预览。
  - Three.js Gaussian Shader 对象编辑 fallback。
  - 已拆分 `真实查看` / `对象编辑` 两个工作模式；对象操作会显式进入点云编辑预览，不再伪装成真实 splat 内直接编辑。
  - 原始颜色（编辑预览） / 对象色（编辑预览）切换；UI 不再把编辑预览文案伪装成真实 3DGS 重渲染。
  - 对象列表、点击 Gaussian OIT 画布选中对象、隔离、删除预览；选中对象在 shader 编辑模式下有高亮层，删除预览会退出隔离并切回原始颜色（编辑预览）显示剩余整体场景，且可一键清除编辑状态返回真实 Splat。
  - 对象编辑 renderer 已从 `PointsMaterial` 升级为 screen-space Gaussian kernel `ShaderMaterial`，消费 PLY `scale_0/1/2`、`rot_0..3` 和 `opacity` attributes，并通过 RGBA half-float accumulation / fullscreen resolve 实现 weighted blended OIT。
  - 对象过滤已进入 shader 路径：每个 Gaussian 上传 dense object index attribute，隐藏 / 隔离 / 删除通过 GPU object-state `DataTexture` 控制；WebGPU tile renderer 仍是后续任务。
  - RENDER-004 WebGPU tile renderer 已补 ADR 设计入口；RENDER-004A capability detection / renderer boundary、RENDER-004B buffer packing / binning smoke contract、RENDER-004C tile accumulation / resolve smoke contract、RENDER-004D object-state buffer smoke contract 与 RENDER-004E overflow gate / fallback hardening 已完成。
  - RENDER-005A 已落地 WebGPU device-backed first-frame skeleton：zero-overflow + WebGPU available 时会切到 `WebGPU Tile 编辑`，上传 tile resolve texture 并绘制 fullscreen triangle；当前本机 headless Chrome 无 WebGPU adapter，真实 runtime first-frame audit 仍 pending，Gaussian OIT 继续作为 fallback。
  - RENDER-005B 已落地 WebGPU storage-buffer upload contract：WebGPU route 会创建并写入 `webgpu-tile-storage-v1` buffers，覆盖 Gaussian geometry/color/object-state/tile counts/tile accumulation/resolve payload。
  - RENDER-005C 已落地 WebGPU storage-buffer resolve shader：WebGPU first-frame display path 会从 `tileResolvedRgba` storage buffer 读取并 fullscreen resolve。
  - RENDER-005D 已落地 WebGPU compute resolve shader：WebGPU route 会先 dispatch compute，把 `tileAccumulation` resolve 到 `tileResolvedRgba`，再由 storage-buffer fullscreen pass 显示。
  - RENDER-005E 已落地 WebGPU tile-center accumulation shader：WebGPU route 会先读取 `tileEntries`、Gaussian buffers 和 object-state storage，在 GPU compute 中写出 `tileAccumulation`。
  - RENDER-005F 已落地 WebGPU covariance-aware tile sampling：accumulation shader 消费 Gaussian scale / rotation，并在 tile 内 2x2 sample points 上近似椭圆高斯 footprint。
  - RENDER-005G 已落地 WebGPU viewport pixel output contract：WebGPU route 会把 tile accumulation resolve 到 `tileResolvedRgba`，再写入 viewport-sized `pixelResolvedRgba`，最后由 `webgpu-pixel-storage-resolve-v1` fullscreen shader 显示。
  - RENDER-005H 已落地 WebGPU per-pixel Gaussian accumulation：`webgpu-compute-pixel-accumulation-v1` 不再复制 `tileResolvedRgba`，而是每个像素读取所属 tile 的 Gaussian entries、object state、scale / rotation 和 color / opacity，直接计算椭圆高斯 weighted OIT 并写入 `pixelResolvedRgba`。
  - RENDER-005I 已落地 WebGPU compact tile entry list：tile entry storage 从 fixed-cap stride 推进到 `compact-offset-list`，新增 `tileOffsets` buffer；Plush 级大场景不再因为 fixed-cap tile overflow 被 capacity gate 阻塞，后续 WebGPU 可用时会进入真实 tile renderer 路径。
  - RENDER-005J 已落地 WebGPU storage/device-limit gate：WebGPU route 进入前会预测 runtime 11-buffer storage 规模，并用 `maxBufferSize` / `maxStorageBufferBindingSize` 阻断超限场景；当前 headless Chrome 仍无 WebGPU adapter，真实 runtime audit 继续 pending。
  - RENDER-005K 已落地强制 WebGPU runtime audit 入口：`npm run audit:webgpu-runtime` 会要求浏览器真实进入 `webgpu-tile`，常规 `audit:demo` 仍保留 fallback 验收。
  - RENDER-005L 已落地 WebGPU device-lost telemetry split：first-frame submission / accumulation / compute / pixel dispatch 与 `device.lost` 作为独立 runtime facts 暴露，强制 audit 可区分“已提交渲染命令”和“提交后 device lost”。
  - RENDER-005M 已落地 WebGPU requiredLimits + backend-loss diagnostics：capability detection 不再创建 probe device，runtime `requestDevice` 显式请求 `maxStorageBuffersPerShaderStage=9`，storage binding limit 会 blocked 于 `webgpu-binding-limit`，并新增 `uncapturederror` 与 queue `onSubmittedWorkDone()` telemetry；当前 blocker 已收敛为 headless unsafe WebGPU 的 queue/backend loss。
  - RENDER-005N 已落地 WebGPU runtime pass probes：`?webgpu-probe=accumulation-only|resolve-only|pixel-output-only` 和 `npm run audit:webgpu-probe` 可单独提交 runtime pass；当前诊断显示 accumulation-only 与 resolve-only queue done，pixel-output-only first frame 后 device lost。
  - RENDER-005O 已将 pixel-output probe 继续拆成 `pixel-compute-only`、`display-only` 和 `tiny-pixel-output`：当前 headless unsafe WebGPU 下 pixel compute / storage write 可 queue done，单独 fullscreen pixel-storage display pass 会 device lost，tiny 32px 合并路径仍 device lost。
  - RENDER-005P 已新增 texture-backed display 和 `clear-only` probes：sampled texture、buffer-to-texture copy display 以及无 draw 的 canvas clear pass 都会在当前 headless unsafe WebGPU 下 device lost；blocker 已从 shader/storage/texture 路径收敛为 canvas render pass / presentation backend loss。
  - RENDER-005Q 已完成 desktop WebGPU audit：headless diagnostic 仍分类为 presentation backend loss，但 headed desktop Chrome/WebGPU 下 `clear-only`、`texture-display-only` 和 `full` runtime audit 均通过；NeRF Lego proxy 已证明 WebGPU tile runtime 可以完成 first frame、compute dispatch、object-state 过滤、选择、隔离和删除预览。
  - RENDER-005R 已完成大场景 desktop WebGPU runtime audit：Plush semantic、Plush v1 与本机 safe-2000 Splatfacto sample 均以 255k-281k Gaussian 进入 `WebGPU Tile 编辑`，storage gate / compact tile list / full compute pipeline / object-state 交互均通过。
  - RENDER-005S 已完成 WebGPU runtime visual fidelity audit 第一轮：full runtime 内部 pixel output 从 `128x128` 提升到 `256x256`，并暴露 viewport / pixel-count telemetry；NeRF Lego proxy 与 Plush semantic 大场景均在 headed desktop Chrome/WebGPU 下通过 256px full runtime audit。
  - RENDER-005T-A 已完成 WebGPU pixel-storage bilinear resolve：full runtime display pass 不再用最近邻放大 `pixelResolvedRgba`，而是使用 bilinear storage sampling，并通过 audit 暴露 `resolveSource=webgpu-pixel-storage-resolve-v1:bilinear-storage`。
  - RENDER-005T-B 已完成 WebGPU aspect-fit runtime viewport：full runtime 会根据实际 viewer display size 计算 area-preserving internal viewport，projection bounds 改为 aspect-fit + 8% 留白，减少固定方形 viewport 和 x/z 独立拉伸导致的贴边 / 比例偏差。
  - RENDER-005T-C 已完成 WebGPU edit-camera perspective projection：WebGPU Tile 编辑预览现在按固定编辑相机在 CPU 端打包 screen-space center / depth / sigma，GPU accumulation / pixel resolve 和 canvas 点击命中不再按旧 x/z 正交 bounds 做二次投影，并通过 audit 暴露 `projection=edit-perspective-camera-v1:52`。
  - RENDER-005T-D 已完成 WebGPU front-weighted OIT depth contract：WebGPU Tile 编辑预览现在会记录 edit-camera depth range，并在 tile accumulation / per-pixel resolve 中使用 `front-weighted-oit-v1`，减少纯 weighted OIT 把前后层 Gaussian 直接混色的问题。
  - RENDER-005T-E 已完成 WebGPU camera-Jacobian screen covariance：前端 PLY parser 保留三轴 scale / quaternion，WebGPU Tile 按 edit-camera projection Jacobian 将 3D covariance 投影成 screen-space ellipse，并以 4:1 anisotropy clamp 降低低分辨率 tile preview 的针状 streak。
  - RENDER-005T-F 已完成 WebGPU adaptive runtime quality：默认 full runtime 从固定 256px 输出升级为按场景规模和显示比例自适应，小场景可到 `adaptive-high-512`，Plush 级大场景走 `adaptive-medium-384`，并通过 audit 暴露质量档和 pixel budget。
  - RENDER-005T-G 已完成 WebGPU source-color fidelity audit：前端 PLY parser 保留 RGB / SH DC / fallback 颜色来源，WebGPU Tile 暴露 `source-color-fidelity-v1`，browser audit 证明 Plush 与 Lego 删除预览后均回到 100% RGB 原始颜色而非对象调试色或 fallback 色。
  - RENDER-005T-H 已完成 WebGPU front-depth gated pixel resolve：per-pixel resolve 现在先找每像素最近有效 Gaussian contributor，再用 `front-depth-gated-oit-v1` 抑制后层 contributor 混入；这是比纯 front-weighted OIT 更强的遮挡近似，但仍不是完整 per-pixel sorted alpha 或 Spark 真实 `.splat` 重渲染。
  - RENDER-005T-I 已完成 Spark vs edit visual residual audit：browser audit 会采集 Spark canvas 与“对象编辑 / 原始颜色”canvas 的 coverage、luma、chroma 和 checksum，并输出 `spark-edit-visual-residual-v1`；NeRF Lego WebGPU full audit 当前显示编辑预览 coverage 是 Spark 的约 4.47x，说明下一步应优先校准 alpha / footprint coverage 或继续拆 view-dependent SH 差距。
  - RENDER-005T-J 已完成 WebGPU footprint coverage calibration：WebGPU Tile 编辑预览暴露 `footprint-weight-floor-calibrated-v1`，pixel resolve 使用 `0.004` 权重 floor，footprint scale 校准到 `2.2`；NeRF Lego Spark/edit coverage ratio 从 `4.469421` 降到 `3.271989`，但 luma / chroma 未同步改善，说明“自身颜色不像真实高斯”的剩余主因是排序 alpha / SH / Spark 合成路径差距，而不是 RGB 原色丢失。
  - RENDER-005T-K 已完成 WebGPU depth-binned alpha compositing：pixel resolve 升级为 `depth-binned-alpha-composite-v1`，每像素固定 8 个 depth bins 做 front-to-back alpha compositing；NeRF Lego luma / chroma delta 从 T-J 的 `0.207570 / 0.133965` 降到 `0.109000 / 0.087808`，Plush 大场景也通过 desktop WebGPU full audit，但 coverage ratio 未同步改善，说明后续需把 coverage 和 shading 分线治理。
  - RENDER-005T-L 已完成 WebGPU alpha presentation edge gate：fullscreen storage resolve 新增 `alpha-edge-gated-presentation-v1:0.035`，只在最终显示阶段压掉低 alpha halo，不改 compute resolve buffer；NeRF Lego coverage ratio 从 T-K 的 `3.856920` 小幅降到 `3.784251`，Plush 从 `6.680406` 降到 `6.448639`，说明 halo 有贡献但 coverage 主问题仍需 footprint / covariance / threshold sweep。
  - RENDER-005T-M 已完成 WebGPU coverage tuning sweep：WebGPU Tile 支持 runtime `webgpu-footprint-scale` / `webgpu-covariance-max-anisotropy`，新增 `npm run audit:webgpu-coverage-sweep`；Lego sweep 显示 coverage 可从 baseline `3.784251` 降到 tight `3.346752`，但 luma / chroma 从 `0.106079 / 0.086537` 恶化到 `0.142279 / 0.102668`，下一步需 Pareto scoring / multi-scene sweep。
  - RENDER-005T-N 已完成 WebGPU coverage Pareto multi-scene sweep：`npm run audit:webgpu-coverage-sweep` 支持 `--assets` 多场景、解析 `tileReferences`，并按 coverage / luma / chroma / tile reference cost 的 `0.35 / 0.25 / 0.25 / 0.15` 权重输出每场景和跨场景 variant score；Lego best Pareto 仍是 baseline，Plush best Pareto 是 compact，tight 只在 coverage / cost 上最好但 luma 代价明显，因此默认渲染参数暂不切到 tight。
  - RENDER-005T-O 已完成 WebGPU coverage report / threshold gate：`audit:webgpu-coverage-sweep` 支持 `--output-dir` 写 `summary.json` / `summary.md`，并支持 `--gate-variant` 与 mean / per-scene pareto、luma、chroma、tile-reference 阈值；新增 `npm run audit:webgpu-coverage-gate` 作为默认参数变更前的 baseline gate，当前 2-scene headed WebGPU gate 通过并写出 `/tmp/objgauss-webgpu-coverage-sweep-gate/summary.*`。
  - RENDER-005T-P 已完成 WebGPU runtime depth-bin tuning：WebGPU Tile 的 depth-binned alpha composite 不再把 8 bins 硬编码在 shader / smoke / audit 三处，新增 `runtime-depth-sort-tuning-v1` 和 URL / audit 参数 `webgpu-depth-bins`，运行时可在 4-16 bins 间调参；默认仍保持 8 bins，coverage gate 证明 baseline 未变化，12-bin headed WebGPU audit 证明 tuned shader 可真实进入 runtime。
  - RENDER-005T-Q 已完成 WebGPU depth-bin sweep：新增 `npm run audit:webgpu-depth-sweep`，可对 `4/8/12/16` bins 跑同一套 headed desktop WebGPU full-runtime visual residual audit，并输出 `summary.json` / `summary.md`；Lego sweep 显示 8 bins 仍是 best Pareto，12 bins 只让 coverage ratio 从 `3.784251` 微降到 `3.784235` 但 chroma 变差，因此单纯提高 bin 数不是当前 Spark/edit 残差主因。
  - RENDER-005T-R 已完成 WebGPU runtime camera framing diagnostic：新增 `runtime-camera-tuning-v1` 和 URL / audit 参数 `webgpu-camera-mode=edit-fixed|spark-frame`，默认仍保持 `edit-fixed`；`spark-frame` 对齐 Spark viewer 的 58° FOV、scene-center target 和 `distance=maxDim*1.7` framing。Lego `spark-frame` headed audit 将 coverage ratio 从 `3.784251` 降到 `3.766657`、luma 从 `0.106079` 降到 `0.102396` 但 chroma 从 `0.086537` 略升到 `0.087290`；Plush `spark-frame` 大场景 audit 也通过并把 coverage ratio 降到 `4.713926`，但 luma/chroma 未同步改善。因此 camera alignment 是 coverage 残差项之一，不是“原始颜色编辑预览不像 Spark”的完整主因。
  - RENDER-005T-S 已完成 WebGPU front-top-k sorted-alpha diagnostic：`runtime-depth-sort-tuning-v1` 新增 URL / audit 参数 `webgpu-depth-alpha-mode=depth-binned|front-top-k`，默认仍保持 `depth-binned`；`front-top-k` 每像素保留最近 K 个 contributor 并前到后 alpha composite，K 复用 `webgpu-depth-bins`。Lego K=8 将 coverage ratio 从 `3.784251` 降到 `3.583371`，但 luma/chroma 恶化到 `0.208595/0.127958`；K=16 coverage 接近 baseline 但 luma/chroma 仍弱于 baseline；Plush K=8 大场景也通过但 luma/chroma 明显变差。因此 per-pixel sorted-alpha 诊断路径可运行，但当前不是默认候选。
  - RENDER-005T-T 已完成 WebGPU SH-rest presence audit：前端 PLY parser 记录 `f_rest_*` 系数数量与推断 SH degree，WebGPU Tile / Gaussian OIT fallback contract 暴露 `shRest=count/maxCoeffs/maxDegree`，browser audit 验证该 telemetry 合法且不改变默认 RGB/SH-DC 渲染路径；本机 `NeRF Lego 训练输出样例` headed WebGPU full audit 显示 `shRest=255794/45/3`，删除预览后仍回到 `colorAfterDelete=255794/0/0/0` RGB 原始色，证明 trained sample 存在完整 degree-3 view-dependent SH 但当前编辑预览尚未利用。
  - RENDER-005T-U 已完成 WebGPU SH-view color diagnostic：前端 PLY parser 在保留 RGB 原始色的同时保存 raw `f_dc`，并把 `f_rest_*` 系数打包为 typed array；WebGPU Tile 新增 URL / audit 参数 `webgpu-color-mode=source|sh-view`，默认仍是 `source`。`sh-view` 按 edit camera 方向评估 degree-3 SH，仅在“原始颜色（编辑预览）”路径生效，不影响对象调试色；headed WebGPU audit 证明本机 trained Lego 删除预览后 `shViewAfterDelete=255794`，luma/chroma residual 从 source 的 `0.090165/0.071164` 降到 `0.034507/0.055774`，但 coverage ratio 仍约 `31x`，说明“自身颜色不像高斯”的颜色主因已被确认并部分缓解，剩余颗粒/膨胀主要来自编辑 renderer 的 footprint / alpha / presentation 与 Spark 真实 splat 合成差距。
  - RENDER-005T-V 已完成 WebGPU SH-view coverage sweep：`audit:webgpu-coverage-sweep` 支持 `--webgpu-color-mode source|sh-view` 并在 report 中记录 `colorMode` / `shViewAfterDelete`，可以在 view-dependent color 生效后继续比较 footprint / anisotropy variants；trained Lego SH-view sweep 显示 baseline 仍是 best Pareto，tight 可将 coverage ratio 从 `31.205176` 降到 `23.164633` 并降低 tile refs，但 luma delta 从 `0.034507` 恶化到 `0.093626`。因此 footprint 收紧只能作为诊断，不能作为默认修复；下一步应优先看 presentation coverage threshold / alpha path 或 Spark object filter feasibility。
  - RENDER-005T-W 已完成 WebGPU alpha presentation floor diagnostic：`webgpu-alpha-presentation-floor` 可在 `0-0.2` 范围内 runtime tuning，默认仍保持 `0.035`；coverage sweep variants 支持 `id:footprint:maxAnisotropy:alphaFloor`。trained Lego + SH-view alpha sweep 显示 `0.1` floor 将 coverage ratio 从 `31.205176` 降到 `24.248059`，luma delta 从 `0.034507` 降到 `0.00276`，chroma 基本持平，因此 presentation threshold 是比 footprint tightening 更强的候选轴；但目前仍是单 scene 证据，不能默认切换，下一步需要多场景 alpha-floor gate 或 Spark renderer object filter feasibility。
  - RENDER-005T-X 已完成 WebGPU alpha presentation floor multi-scene candidate gate：新增 `npm run audit:webgpu-alpha-floor-sweep` 和 `npm run audit:webgpu-alpha-floor-candidate-gate`，在 NeRF Lego proxy + Plush semantic 上复现 `alpha10` 候选。`alpha10` 是 best mean Pareto (`0.965287`) 且同时降低两个 scene 的 coverage / luma，但 strict gate 失败：mean chroma norm=`1.178616`、Plush per-scene Pareto=`1.07908`、Plush chroma norm=`1.485213`。因此 alpha floor 仍是候选轴，默认 `0.035` 不变。
  - RENDER-005T-Y 已完成 Spark filtered edit feasibility implementation：`SplatViewport` 可从 object-aware PLY points 重建 filtered Spark `SplatMesh`；`App` 在 source/original object edit 状态下优先用 `Spark 过滤 Splat` 显示隔离/删除后的剩余场景，SH-view diagnostics 与对象色/点击选择仍走 WebGPU / Gaussian OIT 编辑路径；`audit-demo` 已能验证删除后 `postDelete="spark-splat":"spark-filtered-ply-reconstruct"`。
  - RENDER-005T-Z 已完成 Spark PLY reconstruction residual gate：新增 URL probe `spark-reconstruct-probe=1` 和 `npm run audit:spark-reconstruct-residual`，可直接比较 full `.splat` Spark 与 object-aware PLY reconstructed Spark 的 coverage / luma / chroma；Lego proxy 默认 gate 通过 (`coverageRatio=1.170841`, `lumaDelta=0.029762`, `chromaDelta=0.028407`)，Plush semantic 可选 multiscene 复查也通过 (`coverageRatio=1.303149`, `lumaDelta=0.049406`, `chromaDelta=0.002846`)，但 281k Gaussian 重建耗时明显，仍需 SH-rest preservation 与性能治理。
  - RENDER-ROUTE-003 已新增 WebGPU 100k-1M scale budget audit：`npm run audit:webgpu-scale-budget` 用 synthetic 100k / 300k / 1M `tileSmoke` 估算完整 11-buffer runtime storage，默认在 128 MiB 单 buffer / 256 MiB 总 storage 预算下通过；该 gate 已纳入 `acceptance:renderer-ci`，证明 C-path storage budget 形态，不证明真实 1M FPS。
  - RENDER-ROUTE-004 已新增 WebGPU object-state-filtered tile list mode：WebGPU runtime tile smoke 可保持全量 compact tile list，由 object-state buffer 控制隐藏 / 隔离 / 删除；`audit:webgpu-tile-smoke` 证明编辑前后 tileCounts / tileOffsets / tileEntries checksum 稳定，同时 objectState / resolve checksum 变化。
  - RENDER-ROUTE-005 已新增 WebGPU objectState-only incremental upload：当 tile/storage reuse signature 兼容时，WebGPU runtime 会复用既有 11-buffer storage bundle，只对 `objectState` buffer 执行 `queue.writeBuffer`，再重新 dispatch compute；静态输入变化时仍安全重建完整 bundle。
  - RENDER-ROUTE-006 已新增 WebGPU object-state update telemetry gate：WebGPU viewport 暴露 `data-webgpu-storage-update-mode` 与 `data-webgpu-storage-object-state-byte-size`；browser audit 在 WebGPU isolate / delete transition 后要求 `object-state-only` 增量更新。
  - RENDER-ROUTE-007 已新增 WebGPU edit cost budget audit：`npm run audit:webgpu-edit-cost-budget` 用 100k / 300k / 1M synthetic C-path profiles 检查 objectState-only edit upload、workgroups 和 tile-entry candidate scan 上界；该 gate 已纳入 `acceptance:renderer-ci`，证明编辑更新成本形态，不证明真实 FPS。
  - RENDER-ROUTE-008 已新增 WebGPU runtime timing telemetry gate：WebGPU viewport 暴露 storage update、queue submit 和 queue done 毫秒级 DOM telemetry；browser audit 在 WebGPU object isolate / delete transition 后等待 storage checksum 变化并检查 timing 合法。Isolate 要求 objectState-only；delete 允许 source/object color buffer 改变时的 full-upload fallback。
  - RENDER-ROUTE-009 已新增 WebGPU runtime performance smoke gate：`npm run audit:webgpu-runtime-performance` 复用 offscreen object-transition suite，聚合 Lego proxy 与 Plush 281k 场景的 browser storage update / submit / queue done timing，并按当前 smoke envelope 阻断明显回归。该 gate 仍不是 FPS 或 1M interactive SLA。
  - RENDER-ROUTE-010 已新增 WebGPU headed presentation performance smoke gate：`npm run audit:webgpu-presentation-performance` 在真实 headed browser 中强制 WebGPU full canvas presentation，默认覆盖 Lego proxy 与 Plush semantic 大场景，检查 first-frame pixels、device / queue 状态、pixel dispatch、storage timing、tile overflow 和截图产物。该 gate 证明当前 presentation path 在 smoke envelope 内，不证明 FPS 或 1M interactive SLA。
  - RENDER-ROUTE-011 已将 WebGPU presentation smoke 纳入 product renderer acceptance：`npm run acceptance:renderer-product` 会在 Spark commercial route 前显式验收 headed WebGPU presentation performance；CI profile 仍不跑 headed presentation，继续保持 fresh-clone/headless 友好。
  - RENDER-ROUTE-012 已新增 headed WebGPU presentation object-transition gate：`npm run audit:webgpu-presentation-transition` 强制 full canvas WebGPU C-path 执行选中、隔离、删除，检查 object-state checksum、storage timing、post-delete renderer 和截图；product renderer acceptance 已纳入该 gate。
  - RENDER-ROUTE-013 已新增 WebGPU C-path readiness audit 初版：`npm run audit:webgpu-cpath-readiness` 原始版本串起 build、1M scale budget、1M edit-cost budget 和 fixed-port headed presentation transition，输出 combined evidence / remaining gap report。初版本机通过：1M scale budget max/total=`122.07/173.24 MiB`，1M edit upload=`4 KiB`，headed browser 最大场景=`281498` Gaussians；当时报告仍保留 `browserRuntime1m=not-proven` 与 `fpsSla=not-proven`。
  - RENDER-ROUTE-014 已新增 headed WebGPU frame pacing smoke：`npm run audit:webgpu-frame-pacing` 强制 WebGPU Tile C-path，在 idle / isolate / delete 三段采样 `requestAnimationFrame` 间隔。当前本机覆盖 Lego proxy 与 Plush semantic，Plush `281498` Gaussians 通过，min approx FPS=`26.471`、max mean frame=`37.777ms`、max p95 frame=`16.8ms`、max long-frame ratio=`0.013`；该 gate 证明当前真实场景浏览器响应 smoke，不证明持续 FPS SLA 或真实 1M runtime。
  - RENDER-ROUTE-015 已新增 synthetic 1M headed browser runtime audit：`npm run audit:webgpu-synthetic-1m-runtime` 会在 `/tmp` 生成 binary PLY，通过真实文件上传控件加载 `1,000,000` Gaussians，并在 WebGPU Tile C-path 执行 first-frame、选中、隔离、删除和 rAF frame pacing 检查。当前本机完整 readiness 通过：tile refs=`1,709,862`、tile overflow=`0`、upload wall=`3150.075ms`、isolate object-state update=`513.9ms`、delete full-upload update=`551.5ms`、min approx FPS=`15.429`。`audit:webgpu-cpath-readiness` 已纳入该 gate；剩余 gap 是真实训练 1M scene runtime、持续 FPS SLA 和视觉质量。
  - RENDER-ROUTE-016 已新增 WebGPU sustained frame pacing baseline：`npm run audit:webgpu-sustained-frame-pacing` 会用 `120` rAF samples per phase 复跑当前真实场景 frame pacing 和 synthetic 1M upload/runtime。当前本机通过：真实场景最大 `281498` Gaussians，min approx FPS=`33.964`、max mean frame=`29.443ms`、max p95=`16.8ms`、long-frame ratio=`0.008`；synthetic 1M min approx FPS=`28.917`、max mean frame=`34.582ms`、max p95=`16.8ms`、long-frame ratio=`0.008`。这把 sustained frame pacing 从 smoke 推进为 baseline evidence；生产 FPS SLA 和真实训练 1M scene 仍未完成。
  - RENDER-ROUTE-017 已新增 real/trained PLY WebGPU runtime gate：`npm run audit:webgpu-ply-runtime -- --input-ply <path> --scene-kind trained --min-gaussians <n>` 可上传任意 object-aware PLY，通过真实文件输入进入 WebGPU Tile C-path，并检查 first-frame、对象选择、隔离、删除、tile overflow、更新模式和 rAF pacing。当前本机 trained Lego 样例通过：`255794` Gaussians、tile refs=`581933`、min approx FPS=`30.51`、max mean frame=`32.775ms`、upload wall=`1777.187ms`、isolate object-state update=`143.4ms`、delete object-state update=`142.8ms`。这证明真实/训练 PLY 验收入口可用；near-1M trained scene 仍需实际 PLY 产物后用 `--min-gaussians 1000000` 复跑。
  - RENDER-ROUTE-018 已将 trained PLY runtime 证据接入 WebGPU C-path readiness 聚合器：`npm run audit:webgpu-cpath-readiness -- --trained-ply <path> --trained-min-gaussians <n>` 现在会调用 `audit:webgpu-ply-runtime`，在同一份 readiness report 中输出 `trainedPlyRuntime` 和 `realTrainedBrowserRuntime1m`。当前本机可选路径用 `public/samples/nerf_lego_trained_objects.ply` / `--trained-min-gaussians 250000` 通过：`255794` Gaussians、tile refs=`581933`、min approx FPS=`26.215`、upload wall=`1738.952ms`、isolate update=`142.3ms`、delete update=`141.6ms`。结论仍是 `realTrainedBrowserRuntime1m=not-proven`；最终 gap 是拿到 near-1M trained object-aware PLY 后用 `--trained-min-gaussians 1000000` 复跑。
  - RENDER-ROUTE-019 已将 sustained frame-pacing baseline 证据接入 WebGPU C-path readiness 聚合器：`npm run audit:webgpu-cpath-readiness -- --include-sustained-frame-pacing` 现在会调用 `audit:webgpu-sustained-frame-pacing`，或通过 `--sustained-frame-pacing-summary <summary.json>` 读取已有 baseline，并在 readiness report 中输出 `sustainedFramePacing`。本次短采样聚合验证通过：real scenes min approx FPS=`5.405`、synthetic 1M min approx FPS=`4.138`。这把“当前真实场景 + synthetic 1M 长采样”纳入同一份 C-path readiness 判断；`fpsSla` 仍保持 `not-proven`，直到在目标硬件和真实训练 1M scene 上审定阈值。
  - RENDER-ROUTE-020 已将 trained PLY 接入 sustained frame-pacing baseline：`npm run audit:webgpu-sustained-frame-pacing -- --trained-ply <path> --trained-min-gaussians <n>` 现在会在 current real scenes 与 synthetic 1M 之外追加 `trainedPly` 长采样行；`audit:webgpu-cpath-readiness -- --include-sustained-frame-pacing --trained-ply <path>` 会把同一个 trained PLY 透传给 sustained baseline，并在 readiness terminal summary 中输出 `sustainedTrainedPly` / `sustainedTrainedMinApproxFps`。本次短采样 trained Lego PLY 通过：`255794` Gaussians、`trainedMinApproxFps=11.322`，readiness 同时保持 `realTrainedBrowserRuntime1m=not-proven` 与 `fpsSla=not-proven`。这把未来 near-1M trained scene 的“真实训练输入 + 长采样响应性”验收入口合并起来；当前仍未证明 near-1M trained scene 或生产 FPS SLA。
  - RENDER-ROUTE-021 已给 C-path readiness 增加 reviewed FPS SLA promotion contract：`fpsSla` 不再只是硬编码文字，而是只有在显式 `--fps-sla-reviewed`、提供 `--fps-sla-target-hardware`、真实 trained PLY 达到 1M browser runtime proof、sustained baseline 中 trained PLY 行通过、且 `--fps-sla-min-trained-approx-fps` 达标时才会变成 `passed`。当前 255k trained Lego summary 仍输出 `fpsSla=not-proven`，并列出 blockers：缺少 reviewed flag / target hardware、不是 real trained 1M proof、Gaussian 数不足 1M、trained sustained min FPS 低于默认 `24`。
  - RENDER-ROUTE-022 已新增严格 production SLA wrapper：`npm run audit:webgpu-cpath-production-sla -- --trained-ply <near-1m-trained-objects.ply> --target-hardware <label>` 会先读取 PLY header，拒绝低于 `1,000,000` Gaussians 的 trained PLY 和 summary shortcut，然后强制运行完整 C-path readiness、sustained trained PLY row 和 reviewed FPS SLA gate。当前 255k trained Lego dry-run 被正确挡在 preflight：`trainedGaussians=255794 < 1000000`、`readiness=not-run`、`fpsSla=not-run`。这把最终 SLA 证明入口固化了；真实 near-1M trained PLY 和目标硬件生产 FPS 仍是剩余 gap。
  - RENDER-005T-AA 已完成 Spark packed extract reconstruction route：filtered Spark 路径从 raw PLY `constructSplats` 推进到 base `PackedSplats` cache + visible-index `extractSplats`，浏览器 contract 暴露 `data-spark-reconstruct-source="packed-extract-v1"`、base / visible counts、build / extract timing 和 `data-spark-sh-rest-preserved="false"`；Lego delete preview 通过 `sparkPacked="packed-extract-v1":5696/3909:3.9/1.9` 验收。该路径仍不是原始 `.splat` 内部 object mask，native mask / SH-capable full-view baseline 仍是后续任务。
  - RENDER-005T-AB 已完成 Spark packed SH-rest preservation：filtered Spark PLY reconstruction 现在会把 scene-level `f_rest_*` 编码为 Spark `extra.sh1/sh2/sh3`，并在 visible-index extract 后保留 SH extra；SH-heavy route 暴露 `data-spark-reconstruct-source="packed-sh-extract-v1"` 和 `data-spark-sh-rest-preserved="true"`。本机 `NeRF Lego 训练输出样例` 删除预览通过：`sparkPacked="packed-sh-extract-v1":255794/129108:171.2/41.7`、`sparkShRest=255794:255794:"true":45:3`。剩余外观 gap 是 registered compact `.splat` viewer source 不携带 degree-3 SH rest，导致 trained full reconstruct visual residual 不能直接以 `.splat` 为完整 SH baseline。
  - RENDER-005T-AC 已完成 Spark PLY SH full-view source baseline：SH-heavy 场景在 `真实查看` source/original 且无对象编辑状态时自动使用 Spark PLY SH source，暴露 `data-object-filter="spark-ply-sh-source"` 和 `packed-sh-extract-v1`；`?spark-ply-source=off` 可回到 legacy compact `.splat` 诊断路径。本机 trained Lego same-source residual gate 通过：`fullSource="spark-ply-sh-source":"packed-sh-extract-v1":255794:255794:true:45:3`、`coverageRatio=1.170018`、`lumaDelta=0.058189`、`chromaDelta=0.007036`。trained browser interaction audit 通过，真实查看非背景像素提升到 `70188`，删除后继续保留 `sparkShRest=255794:255794:"true":45:3`。
  - RENDER-005T-AD 已完成 Spark display `PackedSplats` cache telemetry：filtered Spark route 新增 `visible-index-lru-v1` display cache，并在临时 `SplatMesh` dispose 前摘掉 cached packed 引用，避免回到同一 visible-index set 时重复 `extractSplats`；UI HUD 区分 `过滤重建` / `缓存过滤` / `PLY SH 源`，browser audit 会通过隐藏/恢复对象证明 cache hit。Lego proxy 与 trained Lego 删除预览均通过，trained route 保持 `packed-sh-extract-v1` 和完整 SH rest preservation。这仍不是原始 `.splat` 内部 native object mask，每个全新 visible set 仍会创建临时 `SplatMesh`。
  - RENDER-005T-AE 已完成 Spark filtered persistent `SplatMesh` update surface：filtered Spark session 现在会保留同一个 `SplatMesh`，在 isolate / delete / restore 等 object-state 变化时更新其 display `PackedSplats` source，并通过 `data-spark-mesh-update-mode="persistent-splatmesh-v1"`、mesh id、reuse 和 update count 暴露 browser contract。Lego proxy 与 trained Lego 删除预览均证明同一个 mesh 在 hide / restore 后复用并更新：`sparkMesh="persistent-splatmesh-v1":1:"true":4`。这一步减少交互重建感，但仍不是原始 `.splat` 内部 native object mask；每个全新 visible set 仍需要 display `PackedSplats` extract 或 cache hit。
  - RENDER-005T-AF 已完成 Spark object opacity mask over packed source：filtered Spark route 现在用 `object-opacity-texture-v1` mask texture + Spark Dyno `objectModifier` 控制每个 Gaussian opacity，object-state 变化不再做 display `PackedSplats.extractSplats(...)`，browser contract 要求 `data-object-filter="spark-object-opacity-mask"`、`data-spark-packed-extract-ms="0.000"` 和 `data-spark-display-cache-mode="disabled-by-native-mask-v1"`。Lego proxy 删除预览通过：`sparkObjectMask="object-opacity-texture-v1":"4096x2":3909/1787:4`；trained Lego 删除预览通过：`sparkObjectMask="object-opacity-texture-v1":"4096x63":129108/126686:2`、`sparkShRest=255794:255794:"true":45:3`。因此当前“原始颜色 / 自身颜色”在对象编辑后已是 Spark 高斯路径，剩余颗粒感主要来自隔离 / 删除后的 object_id 子集稀疏、边界 assignment 噪声，以及尚未实现 original compact `.splat` 内部 object mask。
  - RENDER-005T-AG 已完成 Spark object opacity mask visual delta guard：`audit-demo` 小场景 hide / restore stress 现在会截图 delete baseline、hide-one-object 和 restored Spark canvas，要求隐藏对象后 checksum 与 coverage/luma/chroma 发生实际变化，恢复后回到 baseline。Lego proxy 通过：`sparkMaskVisual="spark-object-mask-visual-delta-v1":"4a2ed0e8"/"be002ca4"/"4a2ed0e8":0.000752/0.014063/0.026019:0/0/0`。这把 Spark mask 从“DOM telemetry 正确”推进到“像素级可见变化已验收”；original compact `.splat` 与 object-aware PLY packed source 的 index mapping 仍是下一步。
  - RENDER-005T-AH 已完成 compact `.splat` / object-aware PLY index mapping audit：新增 `npm run audit:splat-index-mapping`，对 5 个 public/generated Gaussian 样例检查 count、逐 index position / scale delta、rounded-position multiset 和 object_id 范围；当前 Plush、Plush v1、Plush semantic、Lego proxy、trained Lego 全部 `maxPositionDelta=0`、`maxScaleDelta=0`、`positionMultisetCoverage=1`，证明这些样例可用 Gaussian index 作为外部 object mask key。该结论只覆盖 ObjGauss 生成/登记的 public samples，不等价于任意第三方 `.splat` 内部携带 object_id。
  - RENDER-005T-AI 已完成 Spark native compact `.splat` object mask prototype：新增 URL gate `?spark-native-mask=on`，source/original 对象编辑状态可直接加载原始 compact `.splat`，并用已验证 index mapping 的 object-aware PLY `object_id` 构建 `object-opacity-texture-v1`，通过 Spark `SplatMesh({ url, objectModifier })` 控制删除 / 隔离。Lego proxy browser audit 通过：`sparkMaskSource="native-splat"`、`sparkPacked="native-splat-source-v1":5696/3909:0/0`、`sparkMesh="persistent-splatmesh-v1":1:"true":4`；默认 packed-source 路径仍保持 `sparkMaskSource="ply-packed"`。该路径目前仍是诊断开关，尚未作为默认商业展示路线。
  - RENDER-005T-AJ 已完成 Spark native compact `.splat` object mask 多场景候选 gate：新增 `scripts/audit-spark-native-mask-gate.mjs` 和 `npm run audit:spark-native-mask-gate`，默认覆盖 Lego proxy + Plush semantic。Gate 验证 native source / object mask / persistent mesh contract；Lego pixel delta 由完整 `audit-demo` 覆盖。Plush 281k 大场景通过：`source="native-splat"`、`route="native-splat-source-v1"`、`visible=104403/281498`。该 gate 证明 native route 不是单场景偶然，但默认商业展示路线仍未切换。
  - RENDER-005T-AK 已完成 Spark native compact `.splat` mask 安全默认化：source/original 对象编辑状态下，无 SH-rest 样例默认走 native compact `.splat` source；SH-heavy 样例默认保留 PLY packed SH route，避免丢失 degree-3 SH 外观。Lego 默认 audit 通过：`sparkMaskSource="native-splat"`、`sparkPacked="native-splat-source-v1":5696/3909:0/0`、`sparkMaskVisual="spark-object-mask-visual-delta-v1"`；trained SH-heavy audit 通过：`sparkMaskSource="ply-packed"`、`sparkPacked="packed-sh-extract-v1":255794/129108:160.4/0`、`sparkShRest=255794:255794:"true":45:3`。诊断开关保留：`spark-object-source=packed` / `spark-native-mask=off` 强制 PLY packed，`spark-native-mask=on` 强制 native。
  - RENDER-005T-AL 已完成 Spark canvas object selection product path：source/original Spark filtered edit viewport 现在暴露 `screen-space-object-pick-v1`，点击 Spark 高斯画布会用 object-aware PLY Gaussian 投影选择最近可见 `object_id`，并通过 `data-spark-selection-mode` / `data-spark-selected-object` 验收。Lego 默认 audit 通过：删除后 `sparkCanvasSelectedObject=0`，同时保持 `sparkMaskSource="native-splat"`；trained SH-heavy audit 通过：删除后 `sparkCanvasSelectedObject=3`，同时保持 `sparkMaskSource="ply-packed"` 与完整 SH rest。
  - RENDER-005T-AM 已完成 Spark pick hit telemetry 与选中 marker：Spark canvas pick 现在暴露 `data-spark-pick-status/object/distance/candidate-objects/ambiguous` 和 `data-spark-selected-marker-visible`，画布内会显示非交互选中标记；`audit-demo` 要求 hit、选中对象匹配、距离在半径内、候选数大于 0、marker 可见。当前 Lego proxy 输出 `sparkPick="screen-space-object-pick-v1":"hit":"0":3.7:3:"true":"true"`，trained Lego 输出 `sparkPick="screen-space-object-pick-v1":"hit":"3":0.892:3:"true":"true"`；两者都有效命中但都 `ambiguous=true`。
  - RENDER-005T-AN 已完成 Spark pick 多点击 hit-rate / ambiguity report：新增 `npm run audit:spark-pick-report`，默认对 Lego proxy 的 Spark 删除预览执行 15 个画布点击点并输出 `/tmp/objgauss-spark-pick-report/summary.{json,md}`。当前 Lego proxy `14/15` hit、hit rate `0.933333`、ambiguous hit rate `0.928571`；本机 trained Lego 5-click 显式报告 `5/5` hit、ambiguous hit rate `1`，同时保持 `maskSource="ply-packed"` 与 `packed-sh-extract-v1`。结论是 Spark canvas pick 可用且 marker/selected 状态一致，但 screen-space object 邻近歧义很高，不能宣称 robust renderer-native picking。
  - RENDER-005T-AO 已完成 Spark pick object-support disambiguation：`screen-space-object-pick-v1` 不再只按最近 Gaussian 和第二 object 距离差判歧义，而是用 `object-support-score-v1` 聚合每个候选 object 的最近距离、局部支持占比和前景深度得分，并暴露 `data-spark-pick-strategy/score/score-margin/second-object/second-score`。默认 Lego proxy report 仍 `14/15` hit，但 ambiguity rate 从 `0.928571` 降到 `0.357143`；trained Lego 5-click 从 `1` 降到 `0.2`。`audit-demo` 两个样例的单次 Spark pick 均变为 `ambiguous=false`，同时保持 marker 可见和 selected object match。
  - RENDER-005T-AP 已完成 WebGPU offscreen readback probe：新增 `webgpu-probe=offscreen-readback` 和 `npm run audit:webgpu-offscreen-readback`，该 probe 只 dispatch pixel compute、把 `pixelResolvedRgba` copy 到 `MAP_READ` buffer 并暴露 `data-webgpu-readback-*`，不创建 canvas render pass。Lego proxy 本地 audit 通过：`firstFrame="readback":253952`、`queue="done"`、`deviceLost="active"`、`readback="mapped":"webgpu-compute-depth-binned-alpha-composite-v1":"897e852d":4063232:1015808/1015808:533740`，证明 headless/CI 可单独验证 WebGPU compute/storage/readback，而不把 presentation backend loss 误判为 renderer compute failure。
  - RENDER-005T-AQ 已完成 WebGPU offscreen readback 多场景 suite：新增 `scripts/audit-webgpu-offscreen-readback.mjs`，`npm run audit:webgpu-offscreen-readback` 现在默认覆盖 Lego proxy + Plush semantic，并写出 `/tmp/objgauss-webgpu-offscreen-readback/summary.{json,md}`。默认双场景 gate 通过：Lego `readback="mapped":"897e852d":4063232:1015808/1015808:533740`，Plush `readback="mapped":"0f87864a":2359296:589824/589824:254524`、`packedGaussians=281498`、`tileReferences=1190026`。
  - RENDER-005T-AR 已完成 WebGPU offscreen object-state transition gate：`audit-demo --webgpu-object-transition` 会在 `offscreen-readback` 下通过 `spark-filtered-edit=off` 留在 WebGPU Tile 编辑路径，执行选中、隔离和删除，并要求 object-state checksum 与 readback checksum 都三段变化。默认双场景 gate 通过：Lego readback `897e852d -> 3bd507d9 -> 916a5fc9`，Plush readback `0f87864a -> 0bdb3b09 -> 9660bc47`；对应 object-state 分别为 `7243475b -> f72fa1f4 -> 35652440` 和 `362760d7 -> fc48aab0 -> 637142bc`。
  - RENDER-005T-AS 已完成 WebGPU headless acceptance / presentation split：新增 `npm run acceptance:webgpu-headless`，一键执行 build、WebGPU tile smoke 和 offscreen object-transition suite；`docs/rendering/webgpu-headless-acceptance.md` 明确该命令证明 compute/storage/object-state/readback，不证明 canvas presentation，headed presentation 仍由 `npm run audit:webgpu-desktop` 覆盖，visual tuning 仍由 coverage gate 覆盖。
  - RENDER-005T-AT 已完成 renderer readiness matrix：`docs/rendering/renderer-readiness-matrix.md` 明确商业 Demo 默认继续走 Spark source/original route，WebGPU Tile 作为 C-path proof / diagnostics，Gaussian OIT 作为 fallback；同时把“原始颜色 / 自身颜色删除后仍可能颗粒”界定为 hard object mask、边界 assignment、未重优化补洞和 SH-heavy packed route 的质量边界，而不是颜色丢失 bug。
  - RENDER-005T-AU 已完成产品 renderer route UI contract：颜色下拉改为 `自身颜色` / `对象色诊断`，viewport 增加 route badge，状态面板显示 `展示路线`、`颜色用途`、`预览边界`；root DOM 暴露 route / color role / preview boundary attributes，并由 `audit-demo` 验证首屏 commercial Spark、对象色 diagnostic、删除后 `hard-object-mask-no-reoptimize`。
  - RENDER-005T-AV 已完成 SH-heavy Spark route-only audit：新增 `npm run audit:spark-trained-route`，默认用静态 preview 验证 `nerf-lego-trained-output-local` 的 `spark-ply-sh-source` 初始 route、删除后的 `spark-packed-sh-mask` / `packed-sh-extract-v1`、`object-opacity-texture-v1`、`hard-object-mask-no-reoptimize` 和完整 degree-3 SH rest preservation。
  - RENDER-005T-BD 已完成产品 hard-mask quality status：状态面板新增 `质量解释`，root DOM 暴露 `data-hard-mask-quality-*`，`audit-demo` 可验证并打印 hard-mask quality contract；`自身颜色` 现在只表示颜色来源，颗粒感来源由 `预览边界` 和 `质量解释` 区分为原始 Spark、高斯 hard mask 边界混合、重建残差或待审计。
  - RENDER-005T-BE 已完成 commercial demo readiness QA：新增 `npm run audit:commercial-demo-readiness`，可把 Spark route summary 与 hard-mask quality summary 合并成样例准入表，区分“商业展示路线可演示”“研究 / 诊断样例”“待 route QA”和 public-commercial license eligibility；当前本地报告显示 public-commercial candidate 为 0。
  - DEMO-004 已登记 Poly Haven Chair 商用展示样例：`polyhaven-chair-commercial-demo-local` 指向本地生成的 `polyhaven_chair_demo.splat` / `polyhaven_chair_demo_objects.ply`，发布命令为 `npm run publish:polyhaven-chair-demo`；当前本地 commercial readiness 报告显示 public-commercial candidate 为 1，但它仍是 hard-mask / no-reoptimize 删除预览。
  - DEMO-005A 已强化 `自身颜色` hard-mask 预览 UX contract：删除后 root DOM 暴露 `data-source-preview-result="hard-mask-no-inpaint"`，状态面板显示 `删除结果=源色 mask 预览`，避免把源色 hard-mask 误解为补洞后的完整高斯重渲染。
  - DEMO-005B 已新增 Spark object mask feather 诊断路径：`spark-object-mask-feather=on` 会保持 hidden object opacity 为 0，同时对靠近 hidden boundary 的 visible Gaussian 降低 opacity；默认仍关闭，并由 `npm run audit:spark-mask-feather` 单场景验收。
  - DEMO-005C 已新增 Spark object mask feather 多场景 sweep/report：`npm run audit:spark-mask-feather-sweep` 默认比较 Lego proxy 与 Plush semantic 的 hard mask / `feather55`，输出 route、opacity texture、coverage / luma / chroma、截图和 summary；当前结果显示 feather 可软化边界但未改善 coverage ratio，因此默认仍关闭。
  - DEMO-005D 已新增 Spark object mask feather UI toggle：对象编辑控制面板提供 `柔化删除边界` checkbox，默认关闭；开启后通过显式 `objectMaskFeathering` prop 驱动 Spark object opacity texture，root DOM 和状态面板暴露 enabled / opacity / radius，可由 `audit:spark-mask-feather-sweep -- --control ui` 验收。
  - DEMO-005E 已新增 Spark mask feather candidate gate：`npm run audit:spark-mask-feather-candidates` 默认覆盖 Lego proxy、Plush semantic 和 Poly Haven Chair commercial sample，比较 hard / feather55 / feather70 / feather55r035，并输出 promotion recommendation；当前三个 feather candidates 均为 `diagnostic-only`，默认 hard mask 不变。
  - DEMO-005F 已新增 object boundary cleanup candidate report：`npm run audit:object-boundary-cleanup` 默认覆盖 Lego proxy、Plush semantic 和 Poly Haven Chair commercial sample，在 hard-mask boundary diagnostic 中输出 `object-boundary-cleanup-candidate-v1`，定位值得做 assignment cleanup / remap review 的 object 子集；该报告只读，不自动改 PLY。
  - DEMO-005G 已新增 object boundary remap preview export：`npm run audit:object-boundary-remap-preview` 可生成保留原始 PLY 属性、仅 patch sampled `object_id` 的 `/tmp` preview PLY，用于下一步 browser residual gate；该实验不改默认样例、不证明视觉质量改善。
  - DEMO-005H 已新增 object boundary remap browser residual gate：`npm run audit:object-boundary-remap-residual` 会先生成 sampled remap preview PLY，再用 Playwright 上传原始 PLY 与 remap-preview PLY，强制同一 PLY-packed Spark object-mask route，删除 top remap candidate object 并比较 before/after canvas stats。Lego proxy gate 通过：target object=`2`，remap-preview 少隐藏 `49` 个 Gaussian，after-delete residual=`0.999216/0.004332/0.019990`，recommendation=`browser-evidence-only`，因此仍不默认替换样例。
  - DEMO-005I 已将 object boundary remap residual gate 扩展为默认三场景 promotion table：Lego proxy、Plush semantic、Poly Haven Chair 全部通过同一 PLY-packed Spark route browser gate，remap-preview 分别少隐藏 `49` / `2786` / `29` 个 Gaussian，after-delete residual max=`0.000784/0.004332/0.019990`；aggregate recommendation=`do-not-promote-default-hard-mask`，说明 remap preview 当前可作为证据，不作为默认样例替换。
  - DEMO-005J 已新增 top-N object boundary remap target sweep：`npm run audit:object-boundary-remap-target-sweep` 默认 `--target-count 2`，覆盖三场景 6 个高风险 target object。Top-2 gate 全部通过 route/residual 阈值，但只有 1/6 target 满足 promotion 条件；Lego target `3` 和 Plush target `0` 的 hidden delta 分别为 `+397` / `+4085`，证明 sampled remap 不能按全局默认启用，只能进入 target-level review / allowlist。
  - DEMO-005K 已新增 target-level object boundary remap decision policy：`npm run audit:object-boundary-remap-policy` 会在 top-N browser residual gate 后额外写出 `remap-decision-policy.json/md`，将 target 分为 `allowlist-candidate`、`deny-*` 和 `review-only`。Policy 默认动作仍是 `keep-hard-mask`，应用模式是 `manual-target-allowlist-only`，禁止把 sampled remap 作为全局默认；Lego smoke 中 target `3` 被归为 `deny-hidden-increase`，target `2` 为 `review-only`。
  - DEMO-005L 已让 remap export / QA 消费 decision policy：`npm run audit:object-boundary-remap-policy-export` 读取 `remap-decision-policy.json`，并且只有同时满足 policy `allowlist-candidate` 与命令行显式 `--allow-target asset_id:object_id` 的 target 才会 patch `object_id`。当前三场景 policy-export raw candidates=`10012`，实际 applied remaps=`0`，blocked=`10012`；显式传入 denied target `nerf-lego-alpha-closure-local:3` 仍 remapped=`0`，确认 risky/review-only target 保持 hard mask。
  - DEMO-005M 已新增 reviewed remap allowlist manifest 与正向 fixture gate：`docs/rendering/object-boundary-remap-reviewed-allowlist.json` 当前为空，`npm run audit:object-boundary-remap-policy-export` 同时要求 policy candidate 和 reviewed allowlist 命中，因此真实三场景仍 raw candidates=`10012`、applied=`0`、blocked=`10012`。`npm run audit:object-boundary-remap-reviewed-allowlist` 使用 `/tmp` synthetic policy + reviewed allowlist 证明正向路径可工作：Lego fixture target=`2`，applied=`402`，blocked=`741`。
  - DEMO-005N 已新增 reviewed allowlist 人工评审 runbook 与 manifest schema gate：`docs/rendering/object-boundary-remap-review-runbook.md` 定义进入 allowlist 前必须查看 fixed-port browser screenshots、hidden delta、non-target residual 和 owner approval；`npm run audit:object-boundary-remap-reviewed-allowlist-manifest` 会校验 committed allowlist，当前 `targets=0` 通过。Export 现在会拒绝缺少 reviewer / owner approval / allowlist-candidate evidence 的 approved target；真实三场景 policy export 仍 applied=`0`、blocked=`10012`。
  - DEMO-005O 已新增 Spark canvas hover-confirm selection UX：Spark source/original object edit 现在先在 hover 时显示候选 marker，再由 click 确认选中；root DOM 暴露 `data-spark-pick-interaction="hover-confirm-v1"`、hover pick object / marker 和 confirmed pick marker。`npm run audit:spark-pick-report -- --port 5395` 已验证 Lego proxy 6/6 hover hits、6/6 confirmed hits、markerHits=6/6；`audit:demo` 单样例也通过同一 fixed-port `5395` hover-confirm contract。
  - DEMO-005P 已新增 Spark native pick feasibility audit：`SplatViewport` 暴露 `spark-native-pick-feasibility-v1` telemetry，`npm run audit:spark-native-pick-feasibility -- --port 5395` 在 Lego proxy 删除预览下验证 Spark `SplatMesh.raycast` 可 hit，但 intersection payload 只有 `distance,object,point`，没有 splat index / object id。结论是 recommendation=`keep-screen-space-hover-confirm`，blocker=`raycast-intersection-missing-splat-index`；当前不能安全迁移到 renderer-native object picking。
  - 素材库卡片只展示当前 viewer 可直接加载/交互的本地 Gaussian 样例，并支持训练模型 / near-1M / 商用筛选。
  - Web 内已有 Benchmark tab，展示 SEMANTIC-003 smoke / candidate / paper gates 和三场景 Splatfacto 指标。
  - 移动端已改为 viewport 优先的纵向堆叠布局。
  - `NeRF Lego 训练输出样例` 与 near-1M trained card 已接入；near-1M 默认 quick `.splat` 查看，object-aware PLY 走显式按需加载。
  - `npm run audit:demo` 可启动临时 Vite 服务并浏览器验收三个闭环样例。
- 素材:
  - `plush-3dgs-local` 可自动拉取。
  - Plush `.splat` 用于真实 renderer，`plush_objects.ply` 用于对象级编辑。
  - `polyhaven-school-chair-1k` 可自动拉取到 mesh Demo 输入目录。
  - `polyhaven-school-chair-nerf` 可从 Poly Haven School Chair glTF 离线渲染 NeRF-style RGBA orbit dataset，用作第三个 Splatfacto-trained benchmark scene row。
  - `nerf-synthetic-lego` 可自动拉取到训练素材目录。
  - `nerf-llff-fern` 可从 NeRF example zip 自动抽取 LLFF/COLMAP Fern，并生成 ObjGauss mask/vote 可用的 `transforms_train.json`。
  - ARKitScenes、ScanNet、OmniObject3D、Google Scanned Objects、Poly Haven、Mip-NeRF 360、Tanks and Temples 已登记为候选来源。
- Object Field:
  - 已有 `object_logits: (N, K)` 软分区文件格式。
  - 可从现有 Gaussian PLY warm start，并导出 hard `object_id` PLY 复用前端。
  - 可检查 NeRF-style `transforms_*.json` 训练素材完整性。
  - 可从 NeRF Synthetic RGBA alpha 通道生成真实图片 mask manifest。
  - 可从 NeRF Synthetic RGBA alpha 通道生成 foreground/background + ignore boundary mask manifest，并用 foreground/background confidence 调整训练监督权重；`masks validate` 可检查 image/mask shape、slot 连续性、overlap、空 mask、过大 mask 和 ignore mask。
  - 可从 NeRF Synthetic Lego RGBA 颜色生成多 slot 真实 2D mask manifest。
  - 可在本机提供 `segment-anything` 和 checkpoint 时生成 SAM automatic mask manifest，支持 JPEG 输入和 `--max-image-size` 资源安全降采样。
  - 可消费预计算 SAM / CLIP / 2D mask manifest，并投影投票到 Gaussian。
  - 可用 `objgauss masks score-clip` 将 mask crop 与文本 label 的 CLIP scores 缓存进 manifest；真实 CLIP 推理通过可选 `transformers` backend 提供，仓库默认依赖不包含 torch / transformers。
  - 可用 `objgauss masks align-slots` 将逐帧 SAM area-rank slot 重写为跨视角稳定 slot，并在 manifest 已有预计算 CLIP 分数时聚合命名。
  - 可通过 projection loss 更新 Object Field logits。
  - 可在 hard `object_id` 导出、mask voting PLY 输出和训练输出登记时按 max-probability threshold 将低置信度 Gaussian 归入 background / unknown object，并在登记 manifest 记录该策略。
  - `vote-masks` / `training register-output` 支持可选 background slot 训练：每帧投影可见但未命中任何前景 mask 的 Gaussian 会作为 background/unknown slot 的训练投票，而不是仅在导出阶段硬过滤。
  - `vote-masks` 支持显式 `--visibility-mode depth-buffer` 诊断路径；`vote-diagnostics` 可比较 projected baseline 与 depth-buffer voting 的 conflict / slot balance / coverage delta。
  - 可输出 mask vote quality audit，检查监督覆盖率、每槽覆盖、冲突比例、target entropy 和观测权重。
  - 可输出 Object Emergence observability metrics，检查 assignment entropy、effective slots、空间紧致度、reference stability / ARI 和 partial OES。
  - 可输出 Object Emergence benchmark curves，跟踪 projection loss、entropy、effective slots、ARI、空间紧致度、mask-proxy occlusion delta 和 scale-aware CPU splat render occlusion delta 随 mask-vote training iteration 的变化。
  - 可将多个 emergence curve JSON 聚合为 HTML/SVG benchmark report artifact，用于横向比较多场景曲线。
  - 可从 benchmark manifest 一键重跑多场景 emergence curves、CSV、HTML report 和 summary，并执行阈值检查。
  - `emergence-benchmark` 支持可选 `heldout_masks`，可用同一训练参数生成最终 Object Field 后在 held-out mask manifest 上评估 projection loss、监督覆盖和 render occlusion effect。
  - `emergence-benchmark` 和 cross-scene 聚合会写 failure report，用于记录失败 checks 和 paper-readiness gap。
  - 可机器检查 mask guidance 是否实际改变 Object Field hard labels。
- 训练输出接入:
  - `objgauss training register-output` 可登记外部成熟 3DGS 训练器产出的 `.ply` / `.splat`。
  - 登记时可生成 viewer `.splat`、标准 Gaussian PLY、Object Field、mask 投票 summary 和 `object_id` PLY。
  - 带 mask 登记时，Object Field 初始场使用 Gaussian 几何 warm start，避免全零 logits 在稀疏 mask vote 下坍缩到少数对象槽。
  - 带 mask 登记时可传 `--background-slot` / `--background-weight`，将 projected-unmatched Gaussian 作为背景槽监督写入 `training.background_training` 和顶层 manifest 证据；未显式给 `--slots` 时，`--background-slot` 可自动扩展 slot count。
  - `objgauss training write-sample-bundle` 可写 `objgauss-sample-bundle-v1` 顶层 `sample.json`，把 dataset / transforms、mask manifest、training-output-manifest、Gaussian PLY、Object Field 和 slot 定义绑定，防止旧 Object Field 与新 PLY 混用。
  - 本机已验证 Nerfstudio Splatfacto 可读取 `nerf-synthetic-lego` 的 `blender-data` 格式，完成 100-step CUDA smoke 训练、导出 Gaussian PLY，并接入 Object Field / SAM mask voting。
  - `npm run train:splatfacto:smoke` 已固化为 TRAIN-003A smoke 生成入口，支持 `--dry-run`、`--status` 和 `--run`。
  - 本机已完成 NeRF Lego Splatfacto 500-step resource-safe candidate，导出 47168 个 Gaussian，并通过 `training register-output` 登记为本机 `NeRF Lego 训练输出样例` public sample；该产物在 ignored `outputs/` / `public/samples/`，不进入 git。
  - 本机已完成 NeRF Lego Splatfacto 2000-step resource-safe candidate，导出 255794 个 Gaussian；几何/渲染指标强于 safe-500，但 2-frame SAM supervision 下 object slots 仍不平衡，暂不作为最终语义样例结论。
  - `objgauss masks from-nerf-sam` 支持 `--max-area-fraction` 过滤过大的 SAM masks；safe-2000 当前最佳语义候选是 8-frame / 4-slot / `max_area_fraction=0.3`，已消除近空 object slots 并提升 render occlusion effect。
  - `npm run train:splatfacto:near1m-candidate` 已固化为 TRAIN-003D/003E/003F/003G/003H/003I/003J/003K/003L/003M/003N near-1M candidate handoff：dry-run/status/run 三模式串起 10000-step Splatfacto candidate、balanced Object Field 登记和 `audit:webgpu-cpath-production-sla`，`--status-json` 可写出 `objgauss-near1m-candidate-status-v1` 机器可读 readiness report，并会在 dry-run、成功和已知失败路径留下当前 readiness、`launchReadiness`、`lastExit` 与 `lastFailure`。`npm run train:splatfacto:near1m-gpu-preflight` 可单独执行 `1GB` GPU memory reserve preflight，不启动 Splatfacto。`npm run train:splatfacto:near1m-background` 可 preflight / dry-run / status / guarded stop / 显式 confirmed 后台启动 near-1M 长训，并写 `/tmp` 下的 launcher manifest、status JSON、nested candidate status JSON 和 log；`--preflight` 会刷新 nested candidate status 并只在启动输入与 GPU preflight ready 时返回 0，`--status` 会汇总 nested candidate readiness、launch readiness、last exit 和 last failure kind，`--stop --confirm-stop` 会向记录的 detached process group 发送 `SIGTERM`。Pipeline 默认要求 exported PLY / object-aware PLY 均 `>=1000000` Gaussians；低于阈值会在登记或 SLA 前输出 `near1m_scale_gate=failed` 并停止。真正启动长训还必须显式传 `--confirm-long-run`，否则 `--run` 会输出 `near1m_long_run_guard=failed`；确认后的 CUDA 长训还会先用 `nvidia-smi` 执行默认 `1GB` GPU memory reserve preflight，失败时在启动 Splatfacto 前输出 `near1m_gpu_preflight=failed`。宿主机提权只读 preflight 已通过：RTX 5060 Ti，free=`15215MiB`，reserve=`1024MiB`，`launchReadiness=ready`；当前 random1300k-v1 本地训练证据已形成，sampled1m derivative 通过 strict production SLA，完整 4.5M PLY 仍需单独 LOD / streaming / 全量性能优化。
  - `train:splatfacto:near1m-candidate -- --status` 现在会直接输出 `near1m_goal_gap` 和逐项 `near1m_goal_blocker_<n>`，并把同样结构写入 status JSON 的 `goalGap` 字段；production SLA summary 必须读到 `status="passed"` 才算 terminal evidence，存在但 failed / unreadable 的 summary 会保持 `incomplete` 并给出 `next_action=run-production-sla`。
  - `train:splatfacto:near1m-background` 的 dry-run/status/preflight/confirmed start 现在会输出 `objgauss-near1m-background-handoff-v1`，包含 `near1m_next_action`、下一条可执行命令、是否会启动长训，以及剩余 production SLA 证据缺口；默认还会写 `<output-dir>/handoff.md` 作为 operator-facing Markdown 交接文件。`--status` / `--stop` 已补 process identity 诊断：新 manifest 记录 launch process start time / process group / cwd / command line，状态输出 `near1m_process_identity=running|stale-pid|running-unverified`，避免 PID 复用或旧 manifest 把状态误报成 running，也避免 stop 误杀不匹配进程。该入口仍用于未来长训 / 重训 handoff；当前 terminal proof 已由 random1300k-v1 的 deterministic sampled1m derivative 和 production SLA report 关闭。
- Demo:
  - `objgauss demo v1-closure` 可生成当前 v1 闭环验收包。
  - `objgauss demo verify-v1-closure` 可重新读取产物并机器检查闭环证据。
  - `objgauss demo plush-semantic-closure` 可在真实 Plush `.splat` 上生成非 KMeans 的 2D color mask manifest、训练 Object Field，并导出保留原色的 `object_id` PLY。
  - `objgauss demo verify-plush-semantic-closure` 可检查真实 splat、2D color masks、Object Field、loss、`object_id` PLY、public assets 和前端素材注册。
  - `objgauss demo lego-alpha-closure` 可从 NeRF Lego 真实多视角 RGBA + pose 生成轻量 Gaussian proxy、2D color mask manifest 和 object-aware PLY。
  - `objgauss demo verify-lego-alpha-closure` 可检查 Lego proxy demo 的源图像、mask 文件、Object Field、loss、`object_id` PLY、public assets 和前端素材注册。
  - 前端素材库已有 `Plush 2D 语义 Mask 闭环样例`，加载后可查看真实 splat 外观并执行对象隔离/删除预览。
  - 前端素材库已有 `ObjGauss v1 闭环样例`，加载后可查看真实 splat 外观并执行对象隔离/删除预览。
  - 前端素材库已有 `NeRF Lego 闭环代理样例`，运行 demo 命令后可加载 proxy splat 和对象 PLY。
- 流程:
  - `docs/development-flow.md` 已建立。
  - `AGENTS.md` 和 `CLAUDE.md` 已指向统一流程。
  - `npm run acceptance:demo` 已固化为一键闭环总验收命令；browser audit 默认走 built preview 以避开 dev watcher 上限，Spark commercial route gate 可通过 `--include-spark-commercial-route` 显式纳入，但默认不要求本机 trained SH-heavy sample。
  - `npm run acceptance:semantic` 已固化为 SEMANTIC benchmark suite 验收命令。
  - `npm run acceptance:webgpu-headless` 已固化为 WebGPU CI/headless acceptance 命令；默认构建 viewer、跑 WebGPU tile smoke、再跑 Lego proxy + Plush semantic 的 offscreen object-transition suite。
  - `docs/training/splatfacto-smoke.md` 已记录 Splatfacto smoke 训练 / 导出 / SAM / Object Field 的 runbook 和输出 contract。
  - `npm run train:splatfacto:near1m-candidate` 已固化为 near-1M trained candidate 到 production SLA 的本地编排入口：只有 `--run --confirm-long-run` 才会启动长训，后续输出仍在 ignored `outputs/` 和 `/tmp`；默认 scale gate 会拒绝低于 1M Gaussians 的 exported / object-aware PLY。
  - `npm run audit:near1m-production-gap` 已固化为非训练 near-1M terminal evidence gap artifact：默认写 `/tmp/objgauss-near1m-production-gap/summary.{json,md}` 并在 incomplete 时 exit `0` 用作进度报告；加 `--require-ready` 时会把同一套检查变成 final gate。绑定 random1300k-v1 sampled1m candidate / SLA 目录时已返回 `ready`，completed evidence `8`，missing evidence `0`。
  - `npm run benchmark:splatfacto:balanced` 已固化为 safe-2000 balanced candidate 的一键本地 benchmark 入口，可重跑 balanced SAM、`training register-output`、emergence metrics、curve、report 和 summary。
  - `npm run benchmark:splatfacto:variants` 已固化为 safe-2000 同场景多 mask / slot policy 对比入口，可生成三变体 summary、CSV、Markdown 表格和 HTML 曲线报告。
  - `npm run benchmark:splatfacto:scenes` 已固化为 Splatfacto-trained scene suite，可比较 Lego safe-2000、LLFF Fern smoke 与 Poly Haven Chair smoke 三个 scene rows，并支持 train / held-out SAM manifest split。
  - `npm run benchmark:cross-scene` 已固化为跨场景 / 跨变体汇总入口，可聚合 semantic smoke suite、Splatfacto scene suite 和 safe-2000 variant suite 到同一张表，并输出 smoke / candidate / paper stage gates。
  - `npm run audit:webgpu-coverage-gate` 已固化为 WebGPU 编辑预览 coverage/luma/chroma/cost 的多场景 baseline gate，并输出可复查 summary report。
  - `npm run audit:webgpu-alpha-floor-sweep` / `npm run audit:webgpu-alpha-floor-candidate-gate` 已固化为 alpha presentation floor 候选的多场景复现实验和 strict gate。
  - `npm run audit:webgpu-cpath-readiness` 已固化为 WebGPU C-path readiness 聚合审计：一条命令重跑 build、1M storage/edit budget、headed browser object transition、可选 `--trained-ply` runtime evidence、可选 `--include-sustained-frame-pacing` baseline evidence 和显式 reviewed FPS SLA promotion check，并写出哪些证据已完成、哪些 1M browser/FPS gap 仍未完成。
  - `npm run audit:webgpu-cpath-production-sla` 已固化为 WebGPU C-path production SLA 终局 wrapper：强制 real trained near-1M PLY preflight、目标硬件标签、完整 readiness 运行、sustained trained PLY row 和 reviewed FPS SLA，拒绝 summary shortcut 和低于 1M 的 trained PLY。
  - `npm run audit:webgpu-frame-pacing` 已固化为 headed browser C-path frame pacing smoke：默认覆盖 Lego proxy 与 Plush semantic，采样 idle / isolate / delete 三段 rAF interval，输出 summary report 和截图。
  - `npm run audit:webgpu-ply-runtime` 已固化为 real/trained object-aware PLY browser runtime gate：调用者提供 `--input-ply` 和 `--min-gaussians`，可直接用于后续 near-1M trained scene runtime 验收。
  - `npm run audit:webgpu-sustained-frame-pacing` 已固化为 headed browser C-path sustained frame-pacing baseline：默认覆盖当前真实场景和 synthetic 1M upload/runtime，各 phase 采样 `120` rAF interval，并可通过 `--trained-ply` 增加 real/trained object-aware PLY 长采样行。
  - `npm run audit:webgpu-offscreen-readback` 已固化为 WebGPU compute/storage/readback 多场景 object-transition suite；默认不依赖 canvas presentation 成功，覆盖 Lego proxy 与 Plush semantic，验证首帧、隔离、删除三段 readback checksum 和 object-state checksum，并输出 summary report，适合在 CI/headless 环境区分计算管线和 presentation backend loss。
  - `npm run audit:spark-reconstruct-residual` / `npm run audit:spark-reconstruct-residual-multiscene` 已固化为 Spark full `.splat` 与 PLY reconstructed Spark 的 visual residual gate。
  - `npm run audit:object-mask-boundary` 已固化为 hard object mask 质量诊断，可从 object-aware PLY 输出 deleted-subset coverage、unique coverage loss、shared boundary coverage 和 3D neighbor boundary risk，解释删除后 source/original 预览仍有颗粒感的可能来源。
  - `npm run audit:object-boundary-cleanup` 已固化为 hard-mask 边界清理候选报告，可输出 cleanup candidate Gaussian 估算量、dominant target object、priority score 和 recommendation，用于决定下一轮是否做 cleaned object_id / remap preview 实验。
  - `npm run audit:object-boundary-remap-preview` 已固化为 sampled cleaned `object_id` preview 导出入口，默认输出 `/tmp/objgauss-object-boundary-remap-preview/*.remap-preview.ply` 和 summary，用于后续对比删除后 visual residual / non-target damage。
  - `npm run audit:hard-mask-quality` 已固化为 hard-mask 质量解释链聚合器，可把 PLY boundary diagnostic、Spark route summary 和 browser visual residual summary 按 asset 对齐，区分 boundary mixing、coverage hole risk 与 browser/source residual 主导问题。
  - `npm run audit:commercial-demo-readiness` 已固化为商用展示 QA 准入报告，可复用 Spark route 与 hard-mask quality artifacts，列出 route tier、质量解释、素材许可范围、required copy 和截图路径，避免把研究/本地样例误标成 public commercial demo。
  - `npm run audit:splat-index-mapping` 已固化为 compact `.splat` 与 object-aware PLY 的 Gaussian index mapping gate，用于 native source / original `.splat` object mask 原型前置验收。
  - `npm run audit:spark-native-mask-gate` 已固化为 native compact `.splat` object mask 的 Lego + Plush 多场景默认 route contract gate。
  - `npm run audit:spark-trained-sample` 已固化为本机 trained SH-heavy sample availability preflight，可在浏览器 route gate 前检查 `nerf_lego_trained.*` public sample、`object_id`、degree-3 `f_rest_*` 和 object 数量。
  - `npm run audit:spark-trained-route` 已固化为本机 trained SH-heavy route-only browser gate，低成本验证 `spark-packed-sh-mask` / SH preservation / hard-object-mask boundary。
  - `npm run audit:spark-mask-feather-sweep` 已固化为 Spark object mask feather 多场景报告，可比较 hard mask 与 feather variants 对 opacity texture、coverage / luma / chroma 和截图的影响，并支持 `--control ui` 验证 Web 内显式 soft-boundary toggle。
  - `npm run audit:spark-mask-feather-candidates` 已固化为 soft-boundary promotion 前置 gate，覆盖三场景与多个 opacity / radius candidates；当前报告给出 `promotionCandidate=false`，说明 feather 仍只是诊断开关。
  - `npm run acceptance:renderer-ci` 已固化为 fresh-clone-safe renderer 默认 CI profile，覆盖 B -> C renderer route contract、build、WebGPU tile smoke、WebGPU 100k-1M scale/edit budget、renderer route goal progress artifact、no-SH public sample index mapping 和 Spark native object mask gate，不要求本机 trained SH-heavy sample，也不把 near-1M production proof 当作默认阻塞项。
  - `npm run acceptance:renderer-product` 已固化为显式产品 / Demo route profile，会调用完整 `acceptance:spark-commercial-route`，包括 trained sample availability preflight 和 SH-heavy packed SH route；该 profile 现在还会写 renderer route goal 和 near-1M production gap progress artifacts，并把 B/C foundation、gap status / next action / evidence counts 嵌入顶层 renderer acceptance summary，默认不因 incomplete 失败，作为产品/演示 review 的 C-path production 缺口说明；需要把产品 profile 本身提升为 terminal gate 时可显式加 `--require-near1m-production-ready`。
  - `npm run acceptance:spark-commercial-route` 已固化为 Spark commercial route 总验收命令，一次覆盖 trained sample availability、no-SH native compact `.splat` object mask 与 SH-heavy packed SH object mask；该命令会写 `/tmp/objgauss-spark-commercial-route/summary.{json,md}`，证明 route contract，不证明删除后补洞或重优化。
  - `npm run audit:spark-pick-report` 已固化为 Spark canvas `screen-space-object-pick-v1` 的多点击 hit-rate / ambiguity report；默认跑 Lego proxy，小成本 gate，trained 大场景可用 `--assets nerf-lego-trained-output-local --max-clicks 5` 显式复查。当前 report 默认要求 ambiguity rate `<=0.5`，用于防止 pick 消歧回退。
  - `npm run audit:spark-native-pick-feasibility` 已固化为 Spark native ray/object metadata feasibility report；默认 fixed port `5395`，只在 URL probe 下执行 raycast sample，当前结论是 Spark raycast 可作为 depth probe，但不能返回 splat index / object id，因此不能替代 `hover-confirm-v1` screen-space object picking。
  - PORT-001 已将本地 browser audit / acceptance 默认端口收敛到 fixed `5395`：`audit:demo`、Spark audits、WebGPU browser audits、renderer / demo / Spark commercial / WebGPU headless acceptance 默认都走 `5395`。如端口占用，应停止占用 `5395` 的本地 preview/audit 进程后重跑，不再轮换随机端口；显式 override 参数仍保留用于特殊诊断。
  - PORT-002 已将裸跑 `npm run dev` 与 `npm run preview` 也固定到 `127.0.0.1:5395 --strictPort`；日常 Web 查看和 browser audit 使用同一端口，端口占用时先停止占用进程再重跑。
  - `npm run audit:renderer-route-contract` 已固化为 B -> C renderer 路线静态合约审计：检查 WebGL Gaussian OIT fallback、WebGPU tile terminal path、Spark commercial source route、browser audit telemetry 和 fixed `5395` 端口策略仍保持一致。
  - `npm run audit:renderer-route-goal` 已固化为 B -> C renderer 路线目标级进度审计：聚合 route contract、WebGPU 100k-1M scale/edit budgets 和 near-1M production gap 到一份 summary；默认可作为进度报告，显式加 `--require-production-ready` 时会作为 terminal gate。该命令现在还会在 evidence 表中区分 C-path runtime readiness：默认只报告 `not-collected` 且不阻塞 CI，可用 `--cpath-readiness-summary <summary.json>` 复用已有 `audit:webgpu-cpath-readiness` 结果，或用 `--include-cpath-readiness` 主动运行；加 `--require-cpath-readiness` 时会要求 headed object transition 与 synthetic 1M runtime 证据齐备，但仍不会把 255k trained PLY 或 synthetic 1M 误计为 real trained near-1M production proof。本次新增 `--near1m-*` passthrough，绑定 sampled1m candidate / SLA 目录时 `--require-production-ready` 已返回 `ready`。
  - `docs/benchmarks/spark-filtered-edit.md` 已记录 Spark filtered edit preview 的 runtime contract、验证命令和剩余 gap。
  - `objgauss demo audit-v1-goal --allow-incomplete` 已固化为阶段目标完成度审计命令。
  - baseline commit: `c8dcef7`.

## 最近验证

2026-06-25:

```bash
node --check scripts/audit-renderer-route-goal.mjs
node --check scripts/acceptance-renderer-profile.mjs
npm run audit:renderer-route-goal -- --output-dir /tmp/objgauss-renderer-route-goal-cpath-default-smoke-2
npm run audit:renderer-route-goal -- --cpath-readiness-summary /tmp/objgauss-webgpu-cpath-readiness-trained-sustained-summary/summary.json --output-dir /tmp/objgauss-renderer-route-goal-cpath-summary-smoke
npm run audit:renderer-route-goal -- --require-cpath-readiness --cpath-readiness-summary /tmp/objgauss-webgpu-cpath-readiness/summary.json --output-dir /tmp/objgauss-renderer-route-goal-cpath-require-positive-smoke-2
# expected failed with exit 1: provided summary skipped synthetic 1M runtime, so strict C-path readiness blocks
npm run audit:renderer-route-goal -- --require-cpath-readiness --cpath-readiness-summary /tmp/objgauss-webgpu-cpath-readiness-trained-sustained-summary/summary.json --output-dir /tmp/objgauss-renderer-route-goal-cpath-require-expected-fail-2
node --check scripts/audit-webgpu-cpath-readiness.mjs
node --check scripts/audit-renderer-route-contract.mjs
npm run audit:webgpu-cpath-readiness -- --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-trained-ply --skip-synthetic-1m-runtime --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000
node --check scripts/audit-webgpu-cpath-readiness.mjs
node --check scripts/audit-renderer-route-contract.mjs
npm run audit:webgpu-sustained-frame-pacing -- --port 5395 --output-dir /tmp/objgauss-webgpu-sustained-frame-pacing-readiness-short --skip-build --frame-count 10 --min-real-approx-fps 1 --min-synthetic-approx-fps 1 --max-real-mean-frame-ms 300 --max-synthetic-mean-frame-ms 300 --max-p95-frame-ms 2500 --max-long-frame-ratio 1
npm run audit:webgpu-cpath-readiness -- --skip-run --skip-synthetic-1m-runtime --scale-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/scale-budget/summary.json --edit-cost-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/edit-cost-budget/summary.json --transition-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/presentation-transition/summary.json --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --trained-ply-runtime-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/trained-ply-runtime/summary.json --sustained-frame-pacing-summary /tmp/objgauss-webgpu-sustained-frame-pacing-readiness-short/summary.json --sustained-min-real-approx-fps 1 --sustained-min-synthetic-approx-fps 1 --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-sustained-summary
node --check scripts/audit-webgpu-sustained-frame-pacing.mjs
node --check scripts/audit-webgpu-cpath-readiness.mjs
node --check scripts/audit-renderer-route-contract.mjs
npm run audit:webgpu-sustained-frame-pacing -- --port 5395 --output-dir /tmp/objgauss-webgpu-sustained-frame-pacing-trained-ply-short --skip-build --frame-count 10 --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --min-real-approx-fps 1 --min-synthetic-approx-fps 1 --min-trained-approx-fps 1 --max-real-mean-frame-ms 300 --max-synthetic-mean-frame-ms 300 --max-trained-mean-frame-ms 300 --max-p95-frame-ms 2500 --max-long-frame-ratio 1
npm run audit:webgpu-cpath-readiness -- --skip-run --skip-synthetic-1m-runtime --scale-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/scale-budget/summary.json --edit-cost-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/edit-cost-budget/summary.json --transition-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/presentation-transition/summary.json --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --trained-ply-runtime-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/trained-ply-runtime/summary.json --sustained-frame-pacing-summary /tmp/objgauss-webgpu-sustained-frame-pacing-trained-ply-short/summary.json --sustained-min-real-approx-fps 1 --sustained-min-synthetic-approx-fps 1 --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-trained-sustained-summary
npm run audit:webgpu-cpath-readiness -- --skip-run --skip-synthetic-1m-runtime --scale-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/scale-budget/summary.json --edit-cost-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/edit-cost-budget/summary.json --transition-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/presentation-transition/summary.json --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --trained-ply-runtime-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/trained-ply-runtime/summary.json --sustained-frame-pacing-summary /tmp/objgauss-webgpu-sustained-frame-pacing-trained-ply-short/summary.json --sustained-min-real-approx-fps 1 --sustained-min-synthetic-approx-fps 1 --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-fps-sla-review
npm run audit:webgpu-cpath-readiness -- --allow-failures --skip-run --skip-synthetic-1m-runtime --scale-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/scale-budget/summary.json --edit-cost-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/edit-cost-budget/summary.json --transition-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/presentation-transition/summary.json --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --trained-ply-runtime-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/trained-ply-runtime/summary.json --sustained-frame-pacing-summary /tmp/objgauss-webgpu-sustained-frame-pacing-trained-ply-short/summary.json --sustained-min-real-approx-fps 1 --sustained-min-synthetic-approx-fps 1 --fps-sla-reviewed --fps-sla-target-hardware local-rtx5060ti --fps-sla-min-trained-approx-fps 1 --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-fps-sla-negative
node --check scripts/audit-webgpu-cpath-production-sla.mjs
node --check scripts/audit-renderer-route-contract.mjs
npm run audit:webgpu-cpath-production-sla -- --trained-ply public/samples/nerf_lego_trained_objects.ply --target-hardware local-rtx5060ti --dry-run --allow-failures --output-dir /tmp/objgauss-webgpu-cpath-production-sla-dry-run
npm run audit:webgpu-cpath-production-sla -- --trained-ply public/samples/nerf_lego_trained_objects.ply --target-hardware local-rtx5060ti --skip-run --dry-run --allow-failures --output-dir /tmp/objgauss-webgpu-cpath-production-sla-forbidden-dry-run
node --check scripts/train-splatfacto-near1m-candidate.mjs
node --check scripts/launch-splatfacto-near1m-background.mjs
npm run train:splatfacto:near1m-background -- --dry-run --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-handoff-dry-run
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-handoff-dry-run
npm run train:splatfacto:near1m-background -- --preflight --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --skip-gpu-preflight --output-dir /tmp/objgauss-near1m-handoff-preflight-ready-smoke
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-handoff-preflight-ready-smoke
# host-elevated read-only handoff preflight: passed, gpu_preflight=passed, free_mib=15215, near1m_next_action=start-background-long-run, final candidate incomplete with 5 blockers
npm run train:splatfacto:near1m-background -- --preflight --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-handoff-preflight-host
npm run train:splatfacto:near1m-background -- --dry-run --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-handoff-md-dry-run
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-handoff-md-dry-run
npm run train:splatfacto:near1m-background -- --preflight --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --skip-gpu-preflight --output-dir /tmp/objgauss-near1m-handoff-md-preflight-readable-smoke
npm run train:splatfacto:near1m-background -- --dry-run --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-handoff-md-custom-text --handoff-md /tmp/objgauss-near1m-handoff-md-custom-text/custom-handoff.md
# expected failed with exit 2: background near-1M training requires --confirm-long-run
npm run train:splatfacto:near1m-background -- --run --output-dir /tmp/objgauss-near1m-start-handoff-guard
# fake npm smoke: exercises confirmed start status/handoff path without real training
PATH=/tmp/objgauss-fake-npm:$PATH node scripts/launch-splatfacto-near1m-background.mjs --run --confirm-long-run --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-start-handoff-fake-run
npm run train:splatfacto:near1m-background -- --dry-run --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-start-handoff-dry-run
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-start-handoff-dry-run
npm run train:splatfacto:near1m-background -- --preflight --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --skip-gpu-preflight --output-dir /tmp/objgauss-near1m-start-handoff-preflight-smoke
pgrep -af "nerfstudio|splatfacto|train:splatfacto|ns-train|ns-export"
# expected failed with exit 2: GPU preflight unavailable in sandbox, launchReadiness=not-ready
npm run train:splatfacto:near1m-background -- --preflight --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-background-preflight-not-ready
# host-elevated read-only preflight: passed, gpu_preflight=passed, free_mib=15163, launchReadiness=ready
npm run train:splatfacto:near1m-background -- --preflight --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-background-preflight-host
npm run train:splatfacto:near1m-background -- --preflight --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --skip-gpu-preflight --output-dir /tmp/objgauss-near1m-background-preflight-ready-smoke
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-background-preflight-ready-smoke
npm run train:splatfacto:near1m-candidate -- --dry-run --target-hardware local-rtx5060ti --skip-pull --status-json /tmp/objgauss-near1m-candidate-status-json-smoke/summary.json
npm run train:splatfacto:near1m-background -- --dry-run --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-background-status-json-smoke
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-background-status-json-smoke --candidate-status-json /tmp/objgauss-near1m-candidate-status-json-smoke/summary.json
# expected failed with exit 2: near-1M long training requires --confirm-long-run, but still writes status JSON
npm run train:splatfacto:near1m-candidate -- --run --skip-sla --target-hardware local-rtx5060ti --status-json /tmp/objgauss-near1m-candidate-run-guard-status/summary.json
# expected failed with exit 2: writes lastExit=failed and lastFailure=long-run-confirmation-required
npm run train:splatfacto:near1m-candidate -- --run --skip-sla --target-hardware local-rtx5060ti --status-json /tmp/objgauss-near1m-candidate-failure-status/summary.json
npm run train:splatfacto:near1m-candidate -- --dry-run --target-hardware local-rtx5060ti --skip-pull --status-json /tmp/objgauss-near1m-candidate-passed-status/summary.json
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-background-failure-status --candidate-status-json /tmp/objgauss-near1m-candidate-failure-status/summary.json
npm run train:splatfacto:near1m-candidate -- --dry-run --target-hardware local-rtx5060ti --skip-pull
npm run train:splatfacto:near1m-candidate -- --status
npm run train:splatfacto:near1m-candidate -- --status --status-json /tmp/objgauss-near1m-status/summary.json
npm run train:splatfacto:near1m-gpu-preflight -- --gpu-memory-reserve-gb 1 --status-json /tmp/objgauss-near1m-gpu-preflight-host/summary.json
npm run train:splatfacto:near1m-background -- --dry-run --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-background-smoke
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-background-smoke
# expected failed with exit 2: stopping background near-1M training requires --confirm-stop
npm run train:splatfacto:near1m-background -- --stop --output-dir /tmp/objgauss-near1m-background-stop-guard
npm run train:splatfacto:near1m-background -- --stop --confirm-stop --output-dir /tmp/objgauss-near1m-background-stop-empty
# expected failed with exit 2: background near-1M training requires --confirm-long-run
npm run train:splatfacto:near1m-background -- --run --output-dir /tmp/objgauss-near1m-background-guard
# expected failed with exit 2: near-1M long training requires --confirm-long-run
npm run train:splatfacto:near1m-candidate -- --run --skip-sla --target-hardware local-rtx5060ti
# expected failed with exit 2 before training: nvidia-smi GPU reserve preflight unavailable/failed in this environment
npm run train:splatfacto:near1m-candidate -- --run --confirm-long-run --skip-sla --target-hardware local-rtx5060ti
# expected failed with exit 2: safe-2000 has 255794 Gaussians, below near-1M gate
npm run train:splatfacto:near1m-candidate -- --run --skip-train --skip-sla --skip-gpu-preflight --export-dir outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1 --target-hardware local-rtx5060ti
npm run audit:renderer-route-contract
npm run build
uv run --extra dev pytest
git diff --check
```

2026-06-24:

```bash
node --check scripts/publish-polyhaven-chair-demo.mjs
npm run publish:polyhaven-chair-demo -- --output-dir /tmp/objgauss-polyhaven-chair-demo-publish
uv run objgauss stats public/samples/polyhaven_chair_demo_objects.ply
npm run audit:spark-trained-route -- --assets polyhaven-chair-commercial-demo-local --port 5367
npm run acceptance:spark-commercial-route -- --skip-build --skip-trained-sample-audit --trained-assets polyhaven-chair-commercial-demo-local --native-port 5368 --trained-port 5369 --output-dir /tmp/objgauss-spark-commercial-route-chair
npm run audit:object-mask-boundary -- --assets polyhaven-chair-commercial-demo-local --output-dir /tmp/objgauss-object-mask-boundary-chair
npm run audit:spark-reconstruct-residual -- --assets polyhaven-chair-commercial-demo-local --output-dir /tmp/objgauss-spark-reconstruct-residual-chair --port 5370 --allow-failures
npm run audit:hard-mask-quality -- --boundary-summary /tmp/objgauss-object-mask-boundary-chair/summary.json --route-summary /tmp/objgauss-spark-commercial-route-chair/summary.json --residual-summary /tmp/objgauss-spark-reconstruct-residual-chair/summary.json --output-dir /tmp/objgauss-hard-mask-quality-chair --require-route --require-residual
npm run audit:commercial-demo-readiness -- --output-dir /tmp/objgauss-commercial-demo-readiness-with-chair
node --check scripts/audit-demo.mjs
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5375
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5376 --spark-object-mask-feather --spark-object-mask-feather-opacity 0.55
npm run audit:spark-mask-feather
node --check scripts/audit-spark-mask-feather-sweep.mjs
npm run audit:spark-mask-feather-sweep -- --assets nerf-lego-alpha-closure-local --variants hard:off,feather55:0.55 --skip-build --port 5387 --output-dir /tmp/objgauss-spark-mask-feather-sweep-smoke
npm run audit:spark-mask-feather-sweep -- --skip-build --port 5388 --output-dir /tmp/objgauss-spark-mask-feather-sweep
npm run audit:spark-mask-feather-sweep -- --assets nerf-lego-alpha-closure-local --variants hard:off,feather55:0.55 --control ui --skip-build --skip-visual-stats --port 5389 --output-dir /tmp/objgauss-spark-mask-feather-ui-toggle
npm run audit:spark-mask-feather-sweep -- --assets nerf-lego-alpha-closure-local --variants hard:off,feather55:0.55 --control url --skip-build --skip-visual-stats --port 5390 --output-dir /tmp/objgauss-spark-mask-feather-url-regression
npm run audit:spark-mask-feather-candidates -- --skip-build --skip-visual-stats --port 5391 --output-dir /tmp/objgauss-spark-mask-feather-candidates-telemetry
npm run audit:spark-mask-feather-candidates -- --skip-build --port 5392 --output-dir /tmp/objgauss-spark-mask-feather-candidates
node --check scripts/audit-object-mask-boundary.mjs
npm run audit:object-boundary-cleanup
node --check scripts/export-object-boundary-remap-preview.mjs
npm run audit:object-boundary-remap-preview
uv run objgauss stats /tmp/objgauss-object-boundary-remap-preview/nerf-lego-alpha-closure-local.remap-preview.ply
node scripts/export-object-boundary-remap-preview.mjs --assets plush-semantic-closure-local --max-remap-samples 80000 --output-dir /tmp/objgauss-object-boundary-remap-preview-plush
uv run objgauss stats /tmp/objgauss-object-boundary-remap-preview-plush/plush-semantic-closure-local.remap-preview.ply
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5372
npm run build
uv run --extra dev pytest
git diff --check
node --check scripts/audit-commercial-demo-readiness.mjs
npm run audit:commercial-demo-readiness -- --output-dir /tmp/objgauss-commercial-demo-readiness
node --check scripts/audit-demo.mjs
npm run build
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5365
uv run --extra dev pytest
git diff --check
node --check scripts/audit-hard-mask-quality.mjs
npm run audit:hard-mask-quality -- --boundary-summary /tmp/objgauss-object-mask-boundary/summary.json --route-summary /tmp/objgauss-spark-commercial-route-availability/summary.json --residual-summary /tmp/objgauss-spark-reconstruct-residual-multiscene/summary.json,/tmp/objgauss-spark-reconstruct-residual-trained/summary.json --output-dir /tmp/objgauss-hard-mask-quality
npm run audit:hard-mask-quality -- --output-dir /tmp/objgauss-hard-mask-quality-default
npm run build
uv run --extra dev pytest
git diff --check
node --check scripts/audit-object-mask-boundary.mjs
npm run audit:object-mask-boundary -- --output-dir /tmp/objgauss-object-mask-boundary
npm run build
uv run --extra dev pytest
git diff --check
node --check scripts/acceptance-renderer-profile.mjs
npm run acceptance:renderer -- --profile ci --dry-run --output-dir /tmp/objgauss-renderer-profile-ci-dry-run
npm run acceptance:renderer-product -- --dry-run --output-dir /tmp/objgauss-renderer-profile-product-dry-run
npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-nonbrowser
uv run --extra dev pytest
git diff --check
node --check scripts/acceptance-demo.mjs
node --check scripts/audit-demo.mjs
node --check scripts/acceptance-spark-commercial-route.mjs
npm run acceptance:demo -- --skip-semantic-benchmark --browser-audit-assets nerf-lego-alpha-closure-local --skip-browser-visual-residual --include-spark-commercial-route --spark-native-port 5351 --spark-trained-port 5352 --spark-route-output-dir /tmp/objgauss-acceptance-demo-spark-route --skip-spark-route-build
node -e "const r=require('/tmp/objgauss-acceptance-demo-spark-route/summary.json'); console.log(JSON.stringify({status:r.status, steps:r.steps.length, native:r.routes.native.length, trained:r.routes.trained.length}, null, 2))"
uv run --extra dev pytest
git diff --check
node --check scripts/acceptance-spark-commercial-route.mjs
npm run acceptance:spark-commercial-route -- --native-port 5349 --trained-port 5350 --output-dir /tmp/objgauss-spark-commercial-route-report
node -e "const r=require('/tmp/objgauss-spark-commercial-route-report/summary.json'); console.log(JSON.stringify({status:r.status, steps:r.steps.length, native:r.routes.native.length, trained:r.routes.trained.length}, null, 2))"
uv run --extra dev pytest
git diff --check
node --check scripts/acceptance-spark-commercial-route.mjs
npm run acceptance:spark-commercial-route -- --native-port 5347 --trained-port 5348
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
node --check scripts/audit-spark-trained-route.mjs
npm run build
npm run audit:spark-trained-route -- --port 5346
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --url http://127.0.0.1:5341/ --no-server
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
node --check scripts/acceptance-webgpu-headless.mjs
node --check scripts/audit-webgpu-offscreen-readback.mjs
node --check scripts/audit-webgpu-desktop.mjs
npm run audit:webgpu-tile-smoke
npm run build
npm run acceptance:webgpu-headless -- --port 5330 --output-dir /tmp/objgauss-webgpu-headless-acceptance
npm run audit:webgpu-offscreen-readback -- --assets nerf-lego-alpha-closure-local --port 5323 --output-dir /tmp/objgauss-webgpu-offscreen-readback-transition-single
npm run audit:webgpu-offscreen-readback -- --port 5324 --output-dir /tmp/objgauss-webgpu-offscreen-readback-transition
node --check scripts/audit-spark-pick-report.mjs
npm run build
npm run audit:spark-pick-report
npm run audit:spark-pick-report -- --assets nerf-lego-trained-output-local --max-clicks 5 --output-dir /tmp/objgauss-spark-pick-report-trained --port 5316
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5317/ --no-server
npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5317/ --no-server
node --check scripts/audit-demo.mjs
npm run build
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5314/ --no-server
npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5314/ --no-server
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
npm run audit:spark-reconstruct-residual
npm run audit:spark-native-mask-gate
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
npm run build
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5313/ --no-server
npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5313/ --no-server
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
npm run audit:spark-reconstruct-residual
npm run audit:spark-native-mask-gate
uv run --extra dev pytest
git diff --check
node --check scripts/audit-spark-native-mask-gate.mjs
node --check scripts/audit-demo.mjs
npm run build
npm run audit:spark-native-mask-gate
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5312/ --no-server
npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5312/ --no-server
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
npm run audit:spark-reconstruct-residual
uv run --extra dev pytest
git diff --check
node --check scripts/audit-spark-native-mask-gate.mjs
node --check scripts/audit-demo.mjs
npm run build
npm run audit:spark-native-mask-gate
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --spark-native-mask --skip-visual-residual --port 5311
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
npm run build
npm run preview -- --host 127.0.0.1 --port 5302 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5302/ --no-server --spark-native-mask
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5302/ --no-server
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
node --check scripts/audit-splat-index-mapping.mjs
npm run audit:splat-index-mapping
node --check scripts/audit-demo.mjs
npm run build
npm run preview -- --host 127.0.0.1 --port 5301 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5301/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5301/ --no-server
npm run audit:spark-reconstruct-residual
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
git diff --check
node --check src/sparkObjectMask.js
node --check scripts/audit-demo.mjs
npm run build
npm run preview -- --host 127.0.0.1 --port 5300 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5300/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5300/ --no-server
npm run audit:spark-reconstruct-residual
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
npm run build
npm run preview -- --port 5299 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5299/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5299/ --no-server
npm run audit:spark-reconstruct-residual
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5298/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5298/ --no-server
npm run audit:spark-reconstruct-residual
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
node --check scripts/audit-demo.mjs
node --check scripts/audit-spark-reconstruct-residual.mjs
npm run build
npm run audit:spark-reconstruct-residual
npm run preview -- --port 5294 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5294/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5294/ --no-server
node scripts/audit-spark-reconstruct-residual.mjs --assets nerf-lego-trained-output-local --output-dir /tmp/objgauss-spark-reconstruct-residual-trained-ac
node --check src/sparkPackedSh.js
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run build
uv run --extra dev pytest
npm run audit:webgpu-alpha-floor-candidate-gate -- --port 5292 --allow-failures
node --check scripts/audit-demo.mjs
npm run audit:webgpu-tile-smoke
npm run preview -- --port 5294 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5294/ --no-server
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5295 --probes full
node --check scripts/lib/visual-stats.mjs
node --check scripts/audit-spark-reconstruct-residual.mjs
npm run audit:spark-reconstruct-residual
npm run audit:spark-reconstruct-residual-multiscene
node --check src/webgpuTileResolveShader.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-coverage-sweep -- --asset nerf-lego-trained-output-local --port 5290 --webgpu-color-mode sh-view --variants baseline:2.2:4:0.035,alpha05:2.2:4:0.05,alpha075:2.2:4:0.075,alpha10:2.2:4:0.1 --output-dir /tmp/objgauss-webgpu-alpha-floor-trained-sh-view
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run audit:webgpu-coverage-sweep -- --asset nerf-lego-trained-output-local --port 5289 --webgpu-color-mode sh-view --output-dir /tmp/objgauss-webgpu-coverage-trained-sh-view
node --check src/ply.js
node --check src/webgpuTileSmoke.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5287 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5288 --probes full --webgpu-color-mode sh-view
node --check src/ply.js
node --check src/webgpuTileSmoke.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5286 --probes full
node --check src/webgpuCameraTuning.js
node --check src/webgpuDepthTuning.js
node --check src/webgpuTileComputeShader.js
node --check src/webgpuTileSmoke.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-coverage-sweep.mjs
node --check scripts/audit-webgpu-depth-sweep.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5281 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5282 --probes full --webgpu-depth-alpha-mode front-top-k
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5283 --probes full --webgpu-depth-alpha-mode front-top-k --webgpu-depth-bins 16
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5284 --probes full --webgpu-depth-alpha-mode front-top-k
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5278 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5279 --probes full --webgpu-camera-mode spark-frame
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5280 --probes full --webgpu-camera-mode spark-frame
uv run --extra dev pytest
npm run build
node --check scripts/audit-webgpu-depth-sweep.mjs
npm run audit:webgpu-depth-sweep -- --asset nerf-lego-alpha-closure-local --bins 4,8,12,16 --port 5276 --output-dir /tmp/objgauss-webgpu-depth-sweep
npm run build
uv run --extra dev pytest
node --check src/webgpuDepthTuning.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5274 --probes full --webgpu-depth-bins 12
npm run audit:webgpu-coverage-gate -- --port 5275
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run audit:webgpu-coverage-gate -- --port 5270
npm run audit:webgpu-coverage-sweep -- --assets nerf-lego-alpha-closure-local,plush-semantic-closure-local --port 5268
node --check src/webgpuTileSmoke.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-coverage-sweep.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-coverage-sweep -- --port 5266
node --check src/webgpuTileResolveShader.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5264/ --no-server --probes full
node scripts/audit-webgpu-desktop.mjs --asset plush-semantic-closure-local --url http://127.0.0.1:5264/ --no-server --probes full
node --check src/webgpuTileSmoke.js
node --check src/webgpuTileComputeShader.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5263/ --no-server --probes full
node scripts/audit-webgpu-desktop.mjs --asset plush-semantic-closure-local --url http://127.0.0.1:5263/ --no-server --probes full
node --check src/webgpuTileSmoke.js
node --check src/webgpuTileComputeShader.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5262/ --no-server --probes full
node scripts/audit-webgpu-desktop.mjs --asset plush-semantic-closure-local --url http://127.0.0.1:5262/ --no-server --probes full
node --check scripts/audit-demo.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
node scripts/audit-demo.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5260/ --no-server
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5260/ --no-server --probes full
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5260/ --no-server --probes pixel-compute-only --allow-device-lost-probes
node --check src/webgpuTileComputeShader.js
node --check src/webgpuTileSmoke.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5258 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5259 --probes full
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5245 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5246 --probes full
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5247 --probes full
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5252 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5253 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5254 --probes full
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5255 --probes full
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5256 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5257 --probes full
```

2026-06-23:

```bash
npm run build
npm run audit:demo -- --url http://127.0.0.1:5194/
npm run audit:demo -- --url http://127.0.0.1:5193/
npm run audit:demo -- --url http://127.0.0.1:5192/
npm run audit:demo -- --asset plush-v1-closure-local --url http://127.0.0.1:5191/
npm run audit:demo -- --asset plush-v1-closure-local --url http://127.0.0.1:5190/
uv run objgauss assets pull polyhaven-school-chair-nerf
npm run train:splatfacto:smoke -- --run --asset-id polyhaven-school-chair-nerf --dataset outputs/assets/training/polyhaven-school-chair-nerf --output-root outputs/training/polyhaven-chair-splatfacto-smoke --experiment chair-splatfacto-smoke --timestamp smoke-cuda --export-dir outputs/training/polyhaven-chair-splatfacto-smoke/export-smoke-cuda --object-field-dir outputs/training/polyhaven-chair-splatfacto-smoke/object-field-sam --sam-manifest outputs/masks/polyhaven-chair-sam-smoke/mask-manifest.json --data-parser blender-data --iterations 100 --steps-per-save 100 --vis tensorboard --cache-images cpu --camera-res-scale-factor 0.5 --cuda-home /tmp/objgauss-cuda13 --max-jobs 2 --device cuda --sam-max-frames 8 --sam-max-masks-per-frame 6 --sam-min-area 64 --sam-max-area-fraction 0.75 --slots 6 --object-iterations 80 --skip-benchmark
node scripts/benchmark-splatfacto-scenes.mjs --run --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth
node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-scenes --skip-variants
node scripts/benchmark-splatfacto-scenes.mjs --status
node scripts/benchmark-cross-scene.mjs --status
uv run objgauss assets pull nerf-llff-fern
npm run train:splatfacto:smoke -- --run --asset-id nerf-llff-fern --dataset outputs/assets/training/nerf-llff-fern --output-root outputs/training/nerf-fern-splatfacto-smoke --experiment fern-splatfacto-smoke --timestamp smoke-cuda --export-dir outputs/training/nerf-fern-splatfacto-smoke/export-smoke-cuda --object-field-dir outputs/training/nerf-fern-splatfacto-smoke/object-field-sam --sam-manifest outputs/masks/nerf-fern-sam-smoke/mask-manifest.json --dataparser-transform outputs/training/nerf-fern-splatfacto-smoke/fern-splatfacto-smoke/splatfacto/smoke-cuda/dataparser_transforms.json --data-parser colmap --downscale-factor 1 --images-path images --colmap-path sparse/0 --iterations 100 --steps-per-save 100 --vis tensorboard --cache-images cpu --camera-res-scale-factor 0.25 --cuda-home /tmp/objgauss-cuda13 --max-jobs 2 --device cpu --sam-max-frames 4 --sam-max-masks-per-frame 6 --sam-min-area 256 --sam-max-area-fraction 0.35 --sam-max-image-size 768 --slots 6 --object-iterations 80 --skip-benchmark
node scripts/benchmark-splatfacto-scenes.mjs --run --scene fern-splatfacto-smoke --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth
node scripts/benchmark-splatfacto-scenes.mjs --run --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth
node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-scenes --skip-variants
node scripts/benchmark-splatfacto-scenes.mjs --status
node scripts/benchmark-cross-scene.mjs --status
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "asset_registry or nerf_pull or fern_pull or splatfacto_scene or cross_scene or splatfacto_variant or splatfacto_balanced or splatfacto_smoke or nerf_sam" -q
npm run benchmark:cross-scene -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth
node scripts/benchmark-cross-scene.mjs --run
node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-variants
node scripts/benchmark-cross-scene.mjs --status
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "cross_scene or splatfacto_variant or splatfacto_balanced or splatfacto_smoke" -q
npm run benchmark:splatfacto:variants -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth
node scripts/benchmark-splatfacto-variants.mjs --run --skip-sam
node scripts/benchmark-splatfacto-variants.mjs --status
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "splatfacto_variant or splatfacto_balanced or splatfacto_smoke" -q
npm run benchmark:splatfacto:balanced -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth
node scripts/benchmark-splatfacto-balanced.mjs --run
node scripts/benchmark-splatfacto-balanced.mjs --run --skip-sam
node scripts/benchmark-splatfacto-balanced.mjs --status
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "splatfacto_balanced or splatfacto_smoke" -q
uv run --extra dev pytest
npm run build
uv run --with torch --with torchvision --with "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git" objgauss masks from-nerf-sam outputs/assets/training/nerf-synthetic-lego --output outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json --checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth --model-type vit_b --device cuda --split train --max-frames 8 --max-masks-per-frame 4 --min-area 64 --max-area-fraction 0.3
uv run objgauss training register-output outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --asset-id nerf-lego-splatfacto-safe-2000-sam8f-balanced03-slots4-local --output-dir outputs/assets/gaussians/nerf-lego-trained-safe-2000-sam8f-balanced03-slots4-public --dataset outputs/assets/training/nerf-synthetic-lego --masks outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json --slots 4 --public-name nerf_lego_trained --iterations 160 --learning-rate 1.0
uv run objgauss stats public/samples/nerf_lego_trained_objects.ply
uv run objgauss object-field emergence-curve outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --field outputs/assets/gaussians/nerf-lego-trained-safe-2000-sam8f-balanced03-slots4-warmstart/object_field_initial.npz --masks outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json --output /tmp/objgauss-lego-splatfacto-safe-2000-sam8f-balanced03-slots4-emergence-curve.json --csv-output /tmp/objgauss-lego-splatfacto-safe-2000-sam8f-balanced03-slots4-emergence-curve.csv --iterations 80 --learning-rate 1.0 --eval-every 20 --render-size 96
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "nerf_sam" -q
uv run --extra dev pytest
npm run build
env CUDA_HOME=/tmp/objgauss-cuda13 PATH=/tmp/objgauss-cuda13/bin:$PATH LD_LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LD_LIBRARY_PATH LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LIBRARY_PATH MAX_JOBS=2 uv run --offline --with nerfstudio --with torch --with torchvision --with gsplat --with nvidia-cuda-nvcc==13.0.* --with nvidia-cuda-cccl==13.0.* --with nvidia-nvvm==13.0.* --with nvidia-cuda-crt==13.0.* ns-train splatfacto --max-num-iterations 2000 --steps-per-save 500 --output-dir outputs/training/nerf-lego-splatfacto-long --experiment-name lego-splatfacto-safe --timestamp safe-2000-cpu-cache-v1 --vis tensorboard --pipeline.datamanager.cache-images cpu --pipeline.datamanager.camera-res-scale-factor 0.5 blender-data --data outputs/assets/training/nerf-synthetic-lego
env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_HOME=/tmp/objgauss-cuda13 PATH=/tmp/objgauss-cuda13/bin:$PATH LD_LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LD_LIBRARY_PATH LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LIBRARY_PATH MAX_JOBS=2 uv run --offline --with nerfstudio --with torch --with torchvision --with gsplat --with nvidia-cuda-nvcc==13.0.* --with nvidia-cuda-cccl==13.0.* --with nvidia-nvvm==13.0.* --with nvidia-cuda-crt==13.0.* ns-export gaussian-splat --load-config outputs/training/nerf-lego-splatfacto-long/lego-splatfacto-safe/splatfacto/safe-2000-cpu-cache-v1/config.yml --output-dir outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1
uv run objgauss training register-output outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --asset-id nerf-lego-splatfacto-safe-2000-local --output-dir outputs/assets/gaussians/nerf-lego-trained-safe-2000-cpu-cache-v1-warmstart --dataset outputs/assets/training/nerf-synthetic-lego --masks outputs/masks/nerf-lego-sam/mask-manifest.json --slots 8 --public-name nerf_lego_trained --iterations 160 --learning-rate 1.0
uv run objgauss object-field emergence outputs/assets/gaussians/nerf-lego-trained-safe-2000-cpu-cache-v1-warmstart/object_field_trained.npz --cloud outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --reference outputs/assets/gaussians/nerf-lego-trained-safe-2000-cpu-cache-v1-warmstart/object_field_initial.npz --output /tmp/objgauss-lego-splatfacto-safe-2000-emergence.json
uv run objgauss object-field emergence-curve outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --field outputs/assets/gaussians/nerf-lego-trained-safe-2000-cpu-cache-v1-warmstart/object_field_initial.npz --masks outputs/masks/nerf-lego-sam/mask-manifest.json --output /tmp/objgauss-lego-splatfacto-safe-2000-emergence-curve.json --csv-output /tmp/objgauss-lego-splatfacto-safe-2000-emergence-curve.csv --iterations 80 --learning-rate 1.0 --eval-every 20 --render-size 96
uv run objgauss object-field emergence-report /tmp/objgauss-lego-splatfacto-safe-500-emergence-curve.json /tmp/objgauss-lego-splatfacto-safe-2000-emergence-curve.json --label safe-500 --label safe-2000 --output /tmp/objgauss-lego-splatfacto-safe-500-vs-2000-report.html --title "ObjGauss NeRF Lego Splatfacto Safe 500 vs 2000"
npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5186
npm run audit:demo -- --port 5187
uv run --extra dev pytest
npm run build
uv run objgauss training register-output outputs/training/nerf-lego-splatfacto-long/export-safe-500-cpu-cache-v2/splat.ply --asset-id nerf-lego-splatfacto-safe-500-local --output-dir outputs/assets/gaussians/nerf-lego-trained-safe-500-cpu-cache-v2-warmstart --dataset outputs/assets/training/nerf-synthetic-lego --masks outputs/masks/nerf-lego-sam/mask-manifest.json --slots 8 --public-name nerf_lego_trained --iterations 160 --learning-rate 1.0
uv run objgauss stats public/samples/nerf_lego_trained_objects.ply
npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5182
npm run audit:demo -- --port 5183
uv run --extra dev pytest
npm run build
node scripts/train-splatfacto-smoke.mjs --dry-run --sam-checkpoint /tmp/sam-vit-b.pth --skip-benchmark
node scripts/train-splatfacto-smoke.mjs --status
npm run train:splatfacto:smoke -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth --skip-benchmark
uv run --extra dev pytest
uv run objgauss object-field emergence outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_sam.npz --cloud outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply --reference outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz --output /tmp/objgauss-lego-splatfacto-emergence.json
uv run objgauss object-field emergence-curve outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply --field outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz --masks outputs/masks/nerf-lego-sam/mask-manifest.json --output /tmp/objgauss-lego-splatfacto-render-emergence-curve.json --csv-output /tmp/objgauss-lego-splatfacto-render-emergence-curve.csv --iterations 80 --learning-rate 1.0 --eval-every 20 --render-size 128
uv run objgauss object-field emergence-report /tmp/objgauss-benchmark-plush-semantic.json /tmp/objgauss-benchmark-lego-alpha.json /tmp/objgauss-benchmark-lego-splatfacto.json --label plush-semantic --label lego-alpha-proxy --label lego-splatfacto-smoke --output /tmp/objgauss-emergence-benchmark-report.html --title "ObjGauss Emergence Benchmark Smoke"
uv run objgauss object-field emergence-benchmark docs/benchmarks/semantic-smoke.json --output-dir /tmp/objgauss-semantic-smoke-suite --strict
npm run acceptance:semantic
npm run acceptance:demo
npm run build
npm run audit:demo -- --url http://127.0.0.1:5197/
uv run --extra dev pytest
npm run build
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run build
npm run audit:demo -- --url http://127.0.0.1:5199/
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run build
npm run audit:demo -- --url http://127.0.0.1:5201/
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run build
npm run audit:demo -- --url http://127.0.0.1:5202/
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run build
npm run audit:demo -- --url http://127.0.0.1:5203/
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:demo -- --url http://127.0.0.1:5204/
```

结果：

- RENDER-005A implementation progress: 新增 `src/WebGpuTileViewport.jsx`，在 WebGPU route 中创建 adapter/device/context/render pipeline，将 `tileResolvedRgba` 上传为 `rgba8unorm` texture，并用 fullscreen triangle 绘制第一帧；同时保留 CPU canvas pick fallback。`editRendererContract` 在 `webgpu-device-ready + tileCapacityGate=pass` 时切到 `rendererId="webgpu-tile"` 和 `objectFilter="gpu-object-state-buffer"`；overflow 或 capability failure 继续 fallback。
- RENDER-005A validation: `npm run audit:webgpu-tile-smoke` 通过，包含 simulated WebGPU available + roomy no-overflow contract 切到 `webgpu-tile`，overflow contract blocked 于 `tile-overflow`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5206/` 三样例通过，删除后均显示 `renderModeAfterDelete="原始颜色（编辑预览）"`；定向 Playwright QA 验证删除预览 banner 的 `真实 Splat` 动作会清除编辑状态并返回 Spark renderer。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以实际 WebGPU first frame 未执行。额外 Playwright probe 加 `--enable-unsafe-webgpu` / Vulkan flags 后 Chrome 在当前容器 SIGTRAP 退出，runtime WebGPU audit pending。
- RENDER-005B storage-buffer upload contract: 新增 `src/webgpuTileStorage.js`，定义 `webgpu-tile-storage-v1`，将 `positionRadius`、`colorOpacity`、`scaleRotation`、`objectIndices`、`objectState`、`tileCounts`、`tileAccumulation`、`tileResolvedRgba` 和可选 `tileEntries` 描述为 WebGPU storage buffers；`WebGpuTileViewport` 会在 first-frame / tileSmoke update path 中创建、写入并销毁 storage buffer bundle，并暴露 `data-webgpu-storage-*`。`audit-demo` 的 WebGPU route 现在要求 storage upload 成功，并要求隔离/删除改变 storage checksum。
- RENDER-005B validation: `npm run audit:webgpu-tile-smoke` 通过，fake device 验证 9 个 storage buffers，输出 `storage=86cb35c1:9`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5207/` 三样例通过。当前本机仍 fallback，`storage=null:null` 是预期，因为没有进入 WebGPU route。
- RENDER-005C storage-buffer resolve shader: 新增 `src/webgpuTileResolveShader.js`，定义 `webgpu-storage-resolve-v1` WGSL，fragment shader 直接读取 `tileResolvedRgba` storage buffer 和 16-byte `ResolveMeta` uniform；`WebGpuTileViewport` 不再为 first frame 创建 sampled resolve texture，而是绑定 storage buffer + uniform 绘制 fullscreen triangle。`audit-demo` 的 WebGPU route 现在要求 `data-webgpu-resolve-source="webgpu-storage-resolve-v1"`。
- RENDER-005C validation: `npm run audit:webgpu-tile-smoke` 通过，验证 shader 无 `textureSample` 依赖并输出 `resolveSource=webgpu-storage-resolve-v1`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5208/` 三样例通过。当前本机仍 fallback，`resolveSource=null` 是预期，因为没有进入 WebGPU route。
- RENDER-005D compute resolve shader: 新增 `src/webgpuTileComputeShader.js`，定义 `webgpu-compute-resolve-v1` WGSL compute shader，从 `tileAccumulation` 读取 weighted OIT accumulation 并写入 `tileResolvedRgba`；`WebGpuTileViewport` 会创建 compute pipeline，在 render pass 前 dispatch compute，并暴露 `data-webgpu-compute-*`。`audit-demo` 的 WebGPU route 现在要求 compute 已 dispatch。
- RENDER-005D validation: `npm run audit:webgpu-tile-smoke` 通过，验证 compute shader contract 并输出 `compute=webgpu-compute-resolve-v1:64`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5209/` 三样例通过。当前本机仍 fallback，`compute=null:null:0` 是预期，因为没有进入 WebGPU route。
- RENDER-005E tile-center accumulation shader: `src/webgpuTileComputeShader.js` 新增 `webgpu-compute-accumulation-v1` WGSL compute shader，从 `tileEntries`、`tileCounts`、Gaussian storage buffers 和 `objectState` 读取每个 tile 的 Gaussian list，并在 GPU compute 中写入 `tileAccumulation`；`WebGpuTileViewport` 会先 dispatch accumulation pass，再 dispatch resolve pass，最后由 storage-buffer fullscreen pass 显示。WebGPU route 现在要求 storage 中包含 `tileEntries`。
- RENDER-005E validation: `npm run audit:webgpu-tile-smoke` 通过，验证 accumulation shader contract 并输出 `accumulation=webgpu-compute-accumulation-v1:64`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5212/ --no-server` 三样例通过。当前本机仍 fallback，`accumulation=null:null:0` 是预期，因为没有进入 WebGPU route。
- RENDER-005F covariance-aware tile sampling: `src/webgpuTileComputeShader.js` 将 accumulation source 升级为 `webgpu-compute-covariance-accumulation-v1`，绑定 `scaleRotation` storage buffer，并使用 Gaussian scale / rotation 在 tile 内 2x2 sample points 上计算椭圆高斯 weighted OIT contribution；`src/webgpuTileSmoke.js` 的 CPU reference 同步切到 `tile-2x2-covariance-weighted-oit`。
- RENDER-005F validation: `npm run audit:webgpu-tile-smoke` 通过，验证 covariance accumulation shader contract 并输出 `accumulation=webgpu-compute-covariance-accumulation-v1:64`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5213/ --no-server` 三样例通过。当前本机仍 fallback，`accumulation=null:null:0` 是预期，因为没有进入 WebGPU route。
- RENDER-005G viewport pixel output: `src/webgpuTileComputeShader.js` 新增 `webgpu-compute-pixel-resolve-v1`，WebGPU route 现在按 `covariance accumulation -> tile resolve -> pixel resolve -> pixel-storage fullscreen resolve` 执行；`src/webgpuTileStorage.js` 新增可选 `pixelResolvedRgba` storage buffer，runtime WebGPU tile route 会包含 10 个 storage buffers。
- RENDER-005G validation: `npm run audit:webgpu-tile-smoke` 通过，输出 `storage=5561b7fd:10 pixel=webgpu-compute-pixel-resolve-v1:16384 resolveSource=webgpu-pixel-storage-resolve-v1`；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5214/ --no-server` 三样例通过。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以 browser audit 是 fallback 验收，不宣称真实 WebGPU runtime 证据。
- RENDER-005H per-pixel Gaussian accumulation: `src/webgpuTileComputeShader.js` 将 pixel stage 升级为 `webgpu-compute-pixel-accumulation-v1`，pixel shader 直接读取 Gaussian storage buffers、tile entries、object-state 和 covariance scale / rotation，在每个像素计算 Gaussian kernel weighted OIT 后写入 `pixelResolvedRgba`；`src/webgpuTileSmoke.js` 的 Node smoke reference 同步计算 direct pixel Gaussian output，浏览器 runtime 只分配 GPU 写入用 pixel buffer，避免主线程 CPU 全帧 reference。
- RENDER-005H validation: `npm run audit:webgpu-tile-smoke` 通过，输出 `storage=243af027:10 pixel=webgpu-compute-pixel-accumulation-v1:16384 resolveSource=webgpu-pixel-storage-resolve-v1`；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5215/ --no-server` 三样例通过。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以 browser audit 是 fallback 验收，不宣称真实 WebGPU runtime 证据。
- RENDER-005I compact tile list: `src/webgpuTileSmoke.js` 默认使用 `compact-offset-list` capacity strategy，新增 per-tile `tileOffsets` prefix offsets 和 compact `tileEntries`；fixed-cap layout 仍保留为 audit 对照。`src/webgpuTileComputeShader.js` 的 tile accumulation / pixel accumulation shader 改为从 `tileOffsets[tileIndex]` 读取 entry base，不再假设 `tileIndex * maxEntriesPerTile` stride。
- RENDER-005I validation: `npm run audit:webgpu-tile-smoke` 通过，输出 `storage=de5eaf8f:11 capacity=pass pixel=webgpu-compute-pixel-accumulation-v1:16384`；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5216/ --no-server` 三样例通过，Plush semantic / Plush v1 均为 `tileCapacity="compact-offset-list":"ok":0`。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以 browser audit 是 fallback 验收，不宣称真实 WebGPU runtime 证据。
- RENDER-005J storage/device-limit gate: `src/webgpuTileStorage.js` 新增 WebGPU runtime 11-buffer storage estimate；`src/webgpuCapability.js` 在 target gate 中加入 `maxBufferSize` / `maxStorageBufferBindingSize` 检查，超限时 fallback 为 `webgpu-buffer-limit`；两个 viewport 和 `audit-demo` 暴露/检查 `data-webgpu-storage-limit-*` 与 estimated storage telemetry。
- RENDER-005J validation: `npm run audit:webgpu-tile-smoke` 通过，覆盖 compact pass、fixed overflow block 和模拟小 binding 的 `webgpu-buffer-limit` block；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5217/ --no-server` 三样例通过。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以 storage gate 为 `unknown:webgpu-capability`；Plush estimated max buffer 为 `tileEntries:42053252`，Lego estimated max buffer 为 `pixelResolvedRgba:16777216`。
- RENDER-005K runtime audit entry: `scripts/audit-demo.mjs` 新增 `--require-webgpu` 和 `--webgpu-flags none|unsafe|vulkan`；`package.json` 新增 `npm run audit:webgpu-runtime`。常规 fallback audit 默认不加 WebGPU flags；强制 runtime audit 要求 `data-renderer="webgpu-tile"`、target gate pass、无 fallback、first frame 经过 accumulation / compute / pixel / storage resolve。
- RENDER-005K validation: `npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5218/ --no-server` 三样例通过；`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5218/ --no-server` 单样例通过。`npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5218/ --no-server` 在当前 headless Chrome + `--webgpu-flags unsafe` 下进入 WebGPU route，`accumulation=dispatched`、`compute=dispatched`、`pixel=dispatched`，但 first frame 失败为 `webgpu-device-lost-destroyed`；因此真实 WebGPU runtime audit 仍 pending。
- RENDER-005L device-lost telemetry split: `WebGpuTileViewport` 新增 `data-webgpu-device-lost-status/reason/message`，`device.lost` 不再覆盖 `data-webgpu-first-frame-status`；`audit-demo` 先验 first-frame accumulation / compute / pixel / storage resolve，再单独以 device-lost blocker 失败。
- RENDER-005L validation: `npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --url http://127.0.0.1:5221/ --no-server` 三样例通过；`npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5221/ --no-server` 在当前 headless unsafe WebGPU 下 expected failed，并明确报告 `WebGPU device was lost after first-frame submission: reason=webgpu-device-lost-destroyed`。
- RENDER-005M validation: `npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5222/ --no-server` 单样例通过，覆盖 Spark 真实查看、Gaussian OIT 编辑、画布选择、隔离、删除预览和 `原始颜色（编辑预览）`；`npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5222/ --no-server` 在当前 headless unsafe WebGPU 下 expected failed，失败收敛为 `deviceError=none:: queue=failed:webgpu-queue-submitted-work-failed:A valid external Instance reference no longer exists`。最小 localhost WebGPU 空提交可稳定完成，说明这不是普通 requestDevice / empty submit 失败。本轮 `npm run audit:demo -- --url http://127.0.0.1:5222/ --no-server` 全量 3-asset audit 在 Plush/Spark 大场景的 headless SwiftShader GPU process 上长时间满载，已中止，未作为 005M 验收证据。
- RENDER-005N runtime pass probes: 新增 `src/webgpuRuntimeProbe.js`，`WebGpuTileViewport` 支持 `full`、`accumulation-only`、`resolve-only`、`pixel-output-only` runtime probes；`scripts/audit-demo.mjs` 新增 `--webgpu-probe` 和 `--allow-webgpu-device-lost`，`package.json` 新增 `npm run audit:webgpu-probe`。
- RENDER-005N validation: `git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5223/ --no-server` 单样例通过；`accumulation-only` probe queue done、device active；`resolve-only` probe queue done、device active；`pixel-output-only` probe first frame rendered 后 device lost、queue failed；strict full runtime audit 仍 expected failed with `probe=full` and `A valid external Instance reference no longer exists`。
- RENDER-005O runtime probe split: 新增 `pixel-compute-only`、`display-only`、`tiny-pixel-output` runtime probes，并让 audit 在读取 telemetry 前等待 queue 进入 done/failed 或 device lost；`tiny-pixel-output` 将 WebGPU runtime viewport 降到 32px，用于排除纯 workload size 问题。
- RENDER-005O validation: `git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5224/ --no-server` 单样例通过。`pixel-compute-only` probe 为 queue done、device active、pixel workgroups=256；`display-only` probe 没有任何 compute dispatch，但 first frame 后 device lost、queue failed；`tiny-pixel-output` probe 为 32px viewport / pixel workgroups=16，仍 device lost、queue failed；strict full runtime audit 继续 expected failed with `probe=full` and `A valid external Instance reference no longer exists`。
- RENDER-005P texture/display probes: 新增 `src/webgpuTextureResolveShader.js`、`texture-display-only`、`texture-copy-display` 和 `clear-only` runtime probes；sampled texture display 使用 CPU 生成的 `rgba8unorm` texture，copy display 使用 pixel compute 后 `copyBufferToTexture` 到 `rgba32float` texture，clear-only 只提交 canvas render pass clear、不 draw。
- RENDER-005P validation: `git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5225/ --no-server` 单样例通过。`texture-display-only` probe 为 all compute stages skipped、resolveSource=`webgpu-sampled-texture-resolve-v1`、device lost；`texture-copy-display` 为 pixel workgroups=256、resolveSource=`webgpu-buffer-copy-texture-resolve-v1`、device lost；`clear-only` 为 all compute stages skipped、resolveSource=`webgpu-clear-pass-v1`、device lost。clear-only no-draw fix 后，`npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5226/ --no-server --webgpu-probe clear-only` 复跑通过，`accumulation/compute/pixel=skipped`。strict full runtime audit 仍 expected failed with `probe=full` and `A valid external Instance reference no longer exists`。
- RENDER-005Q desktop audit runner: `scripts/audit-demo.mjs` 新增 `--headed` / `--browser-channel` / `--executable-path`；`scripts/audit-webgpu-desktop.mjs` 和 `npm run audit:webgpu-desktop` 将 `clear-only`、`texture-display-only`、`full` 组合成一条 RENDER-005Q 桌面 runtime audit 命令，并在 `docs/rendering/webgpu-desktop-audit.md` 记录运行方式。
- RENDER-005Q validation: `git diff --check` 通过；`node --check scripts/audit-demo.mjs` 通过；`node --check scripts/audit-webgpu-desktop.mjs` 通过；`npm run audit:webgpu-desktop -- --headless --allow-failures --port 5232` 在当前 headless unsafe WebGPU 下完整收集三项 failure，suite classification=`desktop-webgpu-presentation-backend-loss`；preview server start path fix 后，`npm run audit:webgpu-desktop -- --headless --allow-failures --probes clear-only --port 5233` 复跑通过分类收集；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5234 --allow-failures` headed 通过；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5235` strict headed 通过，classification=`desktop-webgpu-runtime-passed`，full probe 中 accumulation / compute / pixel dispatched，queue done，device active，对象选择 / 隔离 / 删除通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed。
- RENDER-005R large-scene desktop runtime audit: `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5236` 通过，281498 Gaussians、tileReferences=724881、maxTileOccupancy=38792、storage max buffer=`positionRadius:4503968`、visibleAfterIsolate=98770、visibleAfterDelete=182728；`npm run audit:webgpu-desktop -- --asset plush-v1-closure-local --port 5237` 通过，281498 Gaussians、tileReferences=724881、visibleAfterIsolate=85041、visibleAfterDelete=196457；`npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5238` 通过，255794 Gaussians、tileReferences=436816、visibleAfterIsolate=126686、visibleAfterDelete=129108。三者均为 `desktop-webgpu-runtime-passed`，`full` probe dispatch accumulation / compute / pixel stages，queue done，device active。
- RENDER-005S visual fidelity audit: WebGPU full runtime 默认内部 viewport 从 `128x128` 提升到 `256x256`；`App` 新增 `?webgpu-viewport-size=<n>` 配置入口，`audit-demo` 支持 `OBJGAUSS_WEBGPU_VIEWPORT_SIZE` / `--webgpu-viewport-size`，`audit-webgpu-desktop` 会透传该参数；`WebGpuTileViewport` 暴露 `data-webgpu-viewport-width/height` 和 `data-webgpu-pixel-count`。验证：`node --check scripts/audit-demo.mjs` 通过；`node --check scripts/audit-webgpu-desktop.mjs` 通过；`git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5241 --probes full` 通过，`webgpuViewport=256x256:65536`、firstFrame=65536、pixel workgroups=1024；`npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5242 --probes full` 通过，281498 Gaussians、tileReferences=1458084、maxTileOccupancy=21717、visibleAfterIsolate=98770、visibleAfterDelete=182728。
- RENDER-005T-A bilinear display resolve: `src/webgpuTileResolveShader.js` 的 `webgpu-pixel-storage-resolve-v1` 从最近邻 `floor()` storage read 改为 bilinear storage sampling，并新增 `WEBGPU_TILE_RESOLVE_FILTER="bilinear-storage"`；`WebGpuTileViewport` 暴露 `data-webgpu-resolve-filter`，`audit-demo` / `audit-webgpu-tile-smoke` 检查 full storage path 的 resolve filter。验证：`node --check scripts/audit-demo.mjs` 通过；`node --check scripts/audit-webgpu-desktop.mjs` 通过；`git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过，`resolveSource=webgpu-pixel-storage-resolve-v1:bilinear-storage`；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5243 --probes full` 通过，device active、queue done、对象交互通过。
- RENDER-005T-B aspect-fit viewport: `App` 在默认 full runtime 下按实际 viewer display size 计算 area-preserving internal viewport，保留 explicit square viewport override 和 tiny probe；`WebGpuTileViewport` 用 `ResizeObserver` 暴露 display size / viewport aspect mode；`webgpuTileSmoke` 的 bounds projection 改为 `aspect-fit-padding`，按 viewport aspect 扩展短轴并加 8% 留白；`audit-demo` 检查 display size、`display-aspect-area`、`aspect-fit-padding` 和 bounds aspect consistency。验证：`node --check scripts/audit-demo.mjs` 通过；`node --check scripts/audit-webgpu-desktop.mjs` 通过；`git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过，Node smoke 覆盖 1:1 和 2:1 viewport bounds fit；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5244 --probes full` 通过，`webgpuViewport=256x256:65536:"display-aspect-area"`、`display=784x812`、`boundsFit="aspect-fit-padding":1/1`、device active、queue done。
- RENDER-004E overflow gate / fallback hardening: fixed-capacity tile smoke 现在输出 overflow tile count、overflow ratio、max excess、stored references、entry capacity/utilization、capacity mode/status/gate；WebGPU target gate 区分 `webgpu-capability`、`tile-overflow` 和 `renderer-not-implemented`。Browser audit 不再只检查 `tileOverflowCount`，而是验证 overflow 场景被 blocked，非 overflow 场景为 pass/ok。
- RENDER-004E validation: `npm run audit:webgpu-tile-smoke` 通过，内置 sample packed=5800、refs=157323、resolved=2301、overflow=40114、overflowTiles=1056、capacity=blocked；`uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5203/` 三样例通过。Plush semantic / Plush v1 为 `tileCapacity="overflow":169`，Lego 为 `tileCapacity="ok":0`，三者 targetGate 均为 `blocked:webgpu-capability`。
- RENDER-004D object-state buffer smoke: `src/webgpuTileSmoke.js` 现在输出 `webgpu-object-state-v1`，用 stride=4 的 `vec4u`-style buffer 编码 object flags、dense object index、Gaussian count 和 reserved slot；flags 覆盖 visible、selected、removed、isolated 和 enabled，并生成 visible/hidden/removed/selected/isolated object counts 与 checksum。`PointCloudViewport` 暴露 `data-webgpu-object-state-*`，浏览器 audit 会检查隔离 / 删除后的 checksum 和计数变化。
- RENDER-004D validation: `npm run audit:webgpu-tile-smoke` 通过，内置 sample packed=5800、tiles=2362/4096、refs=157323、resolved=2301、objectState=72aeff5e；`uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5202/` 三样例通过。Plush semantic objectState 362760d7 -> 637142bc，Plush v1 e1cdb2e4 -> b0b19f1f，Lego 7243475b -> 7ca4643c。
- RENDER-004C tile resolve smoke: `src/webgpuTileSmoke.js` 现在在 16x16 tile binning 后执行 deterministic tile-center weighted OIT accumulation / resolve，输出 resolved RGBA buffer、resolved tile count、resolve weight、alpha/luma mean 和 checksum；前端状态面板与 DOM contract 暴露 `webgpu-tile-resolve-v1` / `tile-center-weighted-oit`。
- RENDER-004C validation: `npm run audit:webgpu-tile-smoke` 通过，内置 sample packed=5800、tiles=2362/4096、refs=157323、resolved=2301、checksum=c8567887；`uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5201/` 三样例通过并检查 resolve contract。Plush semantic resolvedTiles=3051 / checksum=9feb3736，Plush v1 resolvedTiles=3051 / checksum=4e86df13，Lego resolvedTiles=3881 / checksum=2b4d3d8e。
- RENDER-004B tile smoke packing: 新增 `src/webgpuTileSmoke.js`，把当前 ObjGauss scene 打包成 future WebGPU storage-buffer layout，包括 `positionRadius`、`colorOpacity`、`scaleRotation`、`objectIndices`、`objectState`、`tileCounts` 和可选 `tileEntries` typed arrays；同时生成 deterministic 16x16 tile occupancy、tile references、max occupancy 和 overflow telemetry。
- RENDER-004B validation: `npm run audit:webgpu-tile-smoke` 通过，内置 sample packed=5800、tiles=2362/4096、refs=157323；`uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5199/` 三样例通过并检查 `tileSmokeLayout="webgpu-tile-smoke-v1"`、positive pack/bin counts 和 `objectFilterTarget="gpu-object-state-buffer"`。Plush 当前 telemetry: packed=281498、activeTiles=3119/4096、tileReferences=10513313、maxTileOccupancy=11026、tileOverflowCount=196038，说明 fixed-capacity smoke path 仍需要后续 prefix-sum / overflow hardening。
- RENDER-004A renderer boundary: 前端现在检测 `navigator.gpu` / adapter / device capability，状态面板显示目标 renderer、WebGPU 状态、fallback reason 和 tile overflow；Spark 真实查看暴露 `data-renderer="spark-splat"`，编辑 fallback 暴露 `data-renderer="gaussian-oit"`、`data-renderer-target="webgpu-tile"`、`data-renderer-fallback-reason`、`data-webgpu-status` 和 `data-tile-overflow-count`。
- RENDER-004A validation: `uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5197/` 三样例通过，当前 headless Chrome 为 `webgpuStatus="unavailable"`、`fallbackReason="webgpu-adapter-unavailable"`、`tileOverflowCount=0`，并继续通过画布选中、隔离和删除预览。
- RENDER-003 object-state filtering: Gaussian OIT 编辑 renderer 现在保留全量 Gaussian geometry，使用 dense object index GPU attribute + `gpu-object-state-texture` 控制对象隐藏、隔离和删除；画布拾取会跳过当前 object-state 不可见对象。
- RENDER-004/005 design: `docs/adr/0005-webgpu-tile-renderer.md` 已定义 WebGPU tile renderer 的 staged delivery、data contract、tile binning、per-tile accumulation、object-state buffer、fallback contract 和验收标准；当前下一步是诊断 `webgpu-device-lost-destroyed`，或在真实桌面 WebGPU 浏览器中重跑 runtime audit。
- RENDER-003 validation: `npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5194/` 三样例通过并检查 `objectFilter="gpu-object-state-texture"`；targeted Playwright QA 保存到 `/tmp/objgauss-gpu-filter-*.png`，验证 `initialVisible=281498 -> isolatedVisible=48066 -> deletedVisible=233432`，且无 shader/framebuffer/texture console error。
- RENDER-002 Weighted OIT: 对象编辑 renderer 现在使用 RGBA half-float accumulation render target；RGB 累加 `sum(w*c)`，Alpha 累加 `sum(w)`，fullscreen resolve 后混回基础 grid / axes 场景。Phase 3 WebGPU tile renderer 尚未完成。
- RENDER-002 validation: `npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5193/` 三样例通过，分别检查 `editRenderer="Gaussian OIT 编辑"`、画布点选、隔离、删除后 `renderModeAfterDelete="自身颜色"`；targeted Playwright QA 保存到 `/tmp/objgauss-oit-edit-*.png`，断言真实 Splat -> Gaussian OIT 编辑 -> 画布选中 -> 删除预览全链路，过滤已知 Spark `Worker terminate` 噪声后无 shader/framebuffer/render target console error。
- WEB-002 renderer route: ADR 0004 已接受 B -> C 渐进路线；Phase 1 将对象编辑 renderer 从 `PointsMaterial` / soft sprite 过渡到 screen-space Gaussian kernel `ShaderMaterial`，并把 Gaussian scale / opacity / rotation 传入 GPU attributes。
- WEB-001 UX repair: 顶部假工具按钮已替换为真实 `真实查看` / `对象编辑` 模式；无 `.splat` 场景默认对象编辑，带 `.splat` 样例加载后默认真实查看；删除预览会退出 `只看所选` 隔离并切回 `原始颜色（编辑预览）`，显示删除后的剩余整体场景；对象编辑 banner 可直接清除编辑状态并返回 `真实 Splat`；素材库只显示 5 个可加载样例；Benchmark tab 显示 smoke/candidate/paper pass 和 Lego/Fern/Chair 三场景指标。
- SEMANTIC-003A: render occlusion probe 已从旧的 center/depth point-splat probe 升级为 `scale_aware_cpu_splat_l1`，使用 Gaussian `scale_0/1/2` 与 opacity rasterize 小 footprint，再做 full-vs-object-removed RGBA delta；仍明确不是 covariance-aware `gsplat` training renderer。
- SEMANTIC-003B: `objgauss object-field emergence-benchmark` manifest 支持 `heldout_masks` / `heldout` 配置，summary 可记录 held-out projection loss、supervised Gaussians 和 held-out render occlusion effect；Splatfacto scene suite 现在可从 source SAM manifest split 出 train / held-out manifests，并把 held-out projection loss 与 held-out render occlusion 写入 per-scene / cross-scene summary。
- SEMANTIC-003B current rows: Lego safe-2000 split 为 train 6 frames / held-out 2 frames，held-out supervised_gaussians=459，held-out projection_loss=2.301630，held-out render=0.197505；Fern smoke split 为 train 3 frames / held-out 1 frame，held-out supervised_gaussians=1011，held-out projection_loss=0.670722，held-out render=0.233851；Chair smoke split 为 train 6 frames / held-out 2 frames，held-out supervised_gaussians=6463，held-out projection_loss=2.284750，held-out render=0.224084。
- SEMANTIC-003C: 新增 Poly Haven School Chair NeRF render set 自动素材源，使用纯 Python/NumPy glTF rasterizer 生成 16-frame NeRF-style RGBA dataset；100-step Splatfacto smoke 导出 50000 Gaussians，SAM 生成 8 frames / 48 masks，register-output 监督 10499 Gaussians，projection loss `3.330907 -> 0.774314`。
- SEMANTIC-003D/E: cross-scene summary 新增 smoke / candidate / paper gates，并生成 `/tmp/objgauss-cross-scene-benchmark/failure-report.md`。当前 smoke、candidate 和 paper gate 均通过；paper gate 证据为 `real_splatfacto_scenes=3/3`、`heldout_eval_rows=3/3`、failure report 已写出。
- SEMANTIC-003A validation: `npm run acceptance:semantic` 通过，3 个 semantic smoke scenes 均使用 `scale_aware_cpu_splat_l1`；render effect 分别为 Plush semantic 0.242028、Lego alpha proxy 0.274398、Lego Splatfacto smoke 0.137784。
- BENCH refresh with scale-aware renderer and held-out split: `node scripts/benchmark-splatfacto-scenes.mjs --run --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth` 通过；Lego safe-2000 render=0.229397 / held-out render=0.197505，Fern smoke render=0.235029 / held-out render=0.233851。
- Variant refresh with scale-aware renderer: `node scripts/benchmark-splatfacto-variants.mjs --run --skip-sam` 通过；safe-2000 最佳仍为 `sam8f-slots4-balanced03`，ARI=0.468745、curve OES=0.780806、render=0.221535。
- Cross-scene refresh: `node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-scenes --skip-variants` 通过，rows=9，heldout_eval_rows=3；best render 为 `lego-alpha-proxy/default` 0.274398；stage gates 为 smoke=true、candidate=true、paper=true。

- SEMANTIC-003C scene result: `/tmp/objgauss-splatfacto-scene-suite/summary.json` 含 3 scenes。Lego safe-2000: ARI=0.469787、curve OES=0.784051、render=0.229397、held-out render=0.197505；Fern smoke: ARI=0.790636、curve OES=0.780132、render=0.235029、held-out render=0.233851；Chair smoke: ARI=0.614363、curve OES=0.757609、render=0.248716、held-out render=0.224084。
- SEMANTIC-003C cross-scene result: `/tmp/objgauss-cross-scene-benchmark/summary.json` 从 8 rows 扩展为 9 rows，新增 `splatfacto-scenes/chair-splatfacto-smoke/default`；failure report 显示 `Overall passed: true` 和 `Paper gate passed`。
- SEMANTIC-003C validation: focused tests 5 passed；full Python suite 41 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。

- BENCH-004 real Splatfacto scene suite: 新增 `nerf-llff-fern` 自动素材源、`scripts/benchmark-splatfacto-scenes.mjs`、`npm run benchmark:splatfacto:scenes`、`docs/benchmarks/splatfacto-scenes.json` 和 `docs/benchmarks/splatfacto-scenes.md`，将真实 Splatfacto scene comparison 从 Lego 单场景推进到 Lego safe-2000 + LLFF Fern smoke 两场景。
- BENCH-004 COLMAP handoff: Fern asset pull 从 NeRF example zip 抽取 `nerf_llff_data/fern`，解析 COLMAP `cameras.bin` / `images.bin` 生成 `transforms_train.json`；Nerfstudio COLMAP dataparser 的 `dataparser_transforms.json` 已通过 `scripts/apply-mask-dataparser-transform.mjs` 乘进 mask manifest，解决 raw COLMAP camera 与导出 PLY 坐标不一致的问题。
- BENCH-004 Fern smoke: 100-step Splatfacto smoke 导出 10091 Gaussians；SAM 使用 CPU + `max_image_size=768` 生成 4 frames / 24 masks；register-output 监督 1247 Gaussians，projection loss `3.778366 -> 0.670971`。
- BENCH-004 scene result: `/tmp/objgauss-splatfacto-scene-suite/summary.json` 含 2 scenes。Lego safe-2000: ARI=0.468745、OES=0.693888、curve OES=0.775560、render=0.195308；Fern smoke: ARI=0.783070、OES=0.824959、curve OES=0.772515、render=0.193574。
- BENCH-004 cross-scene result: `/tmp/objgauss-cross-scene-benchmark/summary.json` 从 6 rows 扩展为 8 rows，新增 `splatfacto-scenes/lego-splatfacto-safe-2000/default` 与 `splatfacto-scenes/fern-splatfacto-smoke/default`；全表 best render 仍为 `lego-alpha-proxy/default` 0.236530。
- BENCH-004 validation: scene suite `--status` 与 cross-scene `--status` 均为 `status=ready missing=0`；focused tests 10 passed；full Python suite 39 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- BENCH-003 cross-scene suite: 新增 `scripts/benchmark-cross-scene.mjs` 和 `npm run benchmark:cross-scene`，聚合 semantic smoke 三场景与 safe-2000 三 mask variants 到统一 summary / CSV / Markdown / HTML 表。
- BENCH-003 runbook: 新增 `docs/benchmarks/cross-scene.md`；semantic-smoke 和 splatfacto-variants runbooks 均已链接到 cross-scene 入口。
- BENCH-003 validation: `node scripts/benchmark-cross-scene.mjs --run` 重新跑 semantic smoke suite 和 safe-2000 variant suite，生成 `/tmp/objgauss-cross-scene-benchmark/summary.json`，rows=6；`--status` 输出 `status=ready missing=0`。
- BENCH-003 result: semantic rows 为 Plush semantic、Lego alpha proxy、Lego Splatfacto smoke；safe-2000 rows 为 `sam2f-slots8`、`sam8f-slots8-unfiltered`、`sam8f-slots4-balanced03`。全表 best render 当前为 `lego-alpha-proxy/default` 0.236530；safe-2000 内最佳仍为 `sam8f-slots4-balanced03`，ARI=0.468745、OES=0.775560、render=0.195308。
- BENCH-003 tests: focused script tests 4 passed；full Python suite 36 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- BENCH-002 variant suite: 新增 `scripts/benchmark-splatfacto-variants.mjs` 和 `npm run benchmark:splatfacto:variants`，编排 `sam2f-slots8`、`sam8f-slots8-unfiltered`、`sam8f-slots4-balanced03` 三个 safe-2000 mask policy 变体。
- BENCH-002 runbook: 新增 `docs/benchmarks/splatfacto-variants.md`；BENCH-001 runbook 已链接到 variant suite。
- BENCH-002 validation: `node scripts/benchmark-splatfacto-variants.mjs --run --skip-sam` 复用已有 SAM manifests，重新登记三组 Object Field、生成三条 emergence curve、三变体 HTML report、suite summary / CSV / Markdown；`--status` 输出 `status=ready missing=0`。
- BENCH-002 result: `sam8f-slots4-balanced03` 当前最好，ARI=0.468745、OES=0.693888、render_occlusion_effect_score=0.195308；`sam2f-slots8` 为 ARI=0.388430、OES=0.671132、render=0.123359；`sam8f-slots8-unfiltered` 虽有 frames=8、masks=44、supervised_gaussians=185949，但 ARI=0.113853、OES=0.531374、render=0.108884，证明更多 unfiltered SAM masks 会引入背景/slot 噪声。
- BENCH-002 tests: focused script tests 3 passed；full Python suite 35 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- BENCH-001 reproducible benchmark: 新增 `scripts/benchmark-splatfacto-balanced.mjs` 和 `npm run benchmark:splatfacto:balanced`，支持 `--dry-run`、`--status`、`--run`、`--skip-sam` 和显式 `--publish`；默认不会覆盖 `public/samples/`。
- BENCH-001 runbook: 新增 `docs/benchmarks/splatfacto-balanced.md`，记录 safe-2000 balanced 的输入、固定参数、输出 contract、summary 字段和缺失输入处理；`docs/benchmarks/semantic-smoke.md` 指向该本地 benchmark。
- BENCH-001 validation: full run 重新生成 balanced SAM manifest 并完成 register-output、single-point emergence、emergence curve、HTML report、object PLY stats 和 summary；复用 SAM 的 `--run --skip-sam` 也通过，`--status` 输出 `status=ready missing=0`。
- BENCH-001 summary: `/tmp/objgauss-splatfacto-balanced-benchmark/summary.json` 记录 frames=8、masks=27、mask_pixels=664780、object_id_counts=126686/40747/34682/53679、stability_ari=0.468745、object_emergence_score=0.693888、render_occlusion_effect_score=0.195308，summary_status=passed。
- BENCH-001 tests: focused script tests 2 passed；full Python suite 34 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- SEG-003 SAM filter: `objgauss masks from-nerf-sam` 新增 `--max-area-fraction`，默认 1.0 保持兼容；测试覆盖过大 SAM mask 过滤。
- SEG-003 multi-view finding: unfiltered 8-frame / 8-slot SAM 提升 coverage 到 185949 Gaussians，但 slot0/1 和背景 mask 主导，effective_slots=4.191789，stability_ari=0.113853，弱于 2-frame baseline。
- SEG-003 balanced candidate: 8-frame / 4-slot / `max_area_fraction=0.3` SAM manifest 生成 27 masks、664780 mask pixels；safe-2000 登记后 `supervised_gaussians=70025`，projection loss `2.782336 -> 0.044949`，object_id counts=126686/40747/34682/53679，effective_slots=3.509020，stability_ari=0.468745，partial OES=0.693888，render_occlusion_effect_score=0.195308。
- SEG-003 public sample: 当前本机 `public/samples/nerf_lego_trained.*` 已覆盖为 safe-2000 + balanced 8-frame SAM + 4-slot Object Field；`uv run objgauss stats public/samples/nerf_lego_trained_objects.ply` 通过。
- SEG-003 browser audit: Browser MCP 未暴露可用工具，使用 Playwright fallback；Vite dev server 因系统 inotify watcher 上限 `ENOSPC` 失败后，改用 `npm run preview -- --port 5188 --strictPort` 服务静态 `dist/`，`npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5188/` 通过，splatPixels=3256，editPixels=74388，隔离后可见 126686，删除预览为 1。
- SEG-003 validation: `uv run --extra dev pytest` 33 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- TRAIN-003C resource-safe 2000 candidate: Splatfacto 2000-step 在 `vis=tensorboard`、CPU image cache、0.5 camera scale 和 `MAX_JOBS=2` 下完成；TensorBoard final train loss=0.022640，train PSNR=25.625683，gaussian_count=255795，GPU memory=941.883MB，train total time=18.331932s。
- TRAIN-003C export: `outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply` 导出 255794 / 255795 个 Gaussian，PLY 大小约 61MB。
- TRAIN-003C registration: `training register-output` 使用同一 2-frame SAM manifest 和 8 slots 登记 safe-2000 PLY，`supervised_gaussians=85349`，projection loss `4.467615 -> 0.288167`，public local outputs 覆盖为 `public/samples/nerf_lego_trained.splat` 和 `public/samples/nerf_lego_trained_objects.ply`。
- TRAIN-003C Object Field distribution: safe-2000 object_id counts 为 84464/64455/111/14821/27910/23159/15867/25007，assignment_confidence=0.819222，effective_slots=5.996345，spatial_compactness_score=0.980746，stability_ari=0.388430，partial OES=0.671132。
- TRAIN-003C emergence curve: projection loss `4.467615 -> 0.302584`，final render_occlusion_effect_score=0.123359；与 safe-500 的 render occlusion 同量级，说明几何密度提升没有自动解决对象语义质量。
- TRAIN-003C browser audit: `npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5186` 通过，splatPixels=3256，editPixels=74388，隔离后可见 84464，删除预览为 1；截图 `/tmp/objgauss-audit-nerf-lego-trained-output-local.png`。
- UI regression audit: `npm run audit:demo -- --port 5187` 通过 Plush semantic、Plush v1、NeRF Lego proxy 三个默认闭环样例。
- SplatViewport 修复: 真实 splat 视口 fog 已随 bounding box 自适应，避免 denser / larger Splatfacto sample 在真实 splat 模式下被固定 fog 盖成背景。
- TRAIN-003C 判断: safe-2000 是更好的几何/渲染候选，但不是最终语义样例；下一步应扩展多视角 SAM / slot balancing，而不是盲目增加训练步数。
- TRAIN-003B resource-safe candidate: Nerfstudio Splatfacto 500-step 在 `vis=tensorboard`、CPU image cache、0.5 camera scale 和 `MAX_JOBS=2` 下完成；导出 `outputs/training/nerf-lego-splatfacto-long/export-safe-500-cpu-cache-v2/splat.ply`，47168 / 50000 Gaussian 通过 opacity filter。
- TRAIN-003B public sample registration: `training register-output` 使用 2-frame SAM manifest 和 8 slots 登记 safe-500 PLY，`supervised_gaussians=7676`，projection loss `3.047123 -> 0.321066`，public local outputs 为 `public/samples/nerf_lego_trained.splat` 和 `public/samples/nerf_lego_trained_objects.ply`。
- TRAIN-003B Object Field distribution: `nerf_lego_trained_objects.ply` 含 8 个 object_id，counts=9127/5528/5661/5815/6073/3923/5995/5046，避免了登记阶段 uniform init 造成的少槽坍缩。
- TRAIN-003B browser audit: `npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5182` 通过，splatPixels=408，editPixels=86577，隔离后可见 9127，删除预览为 1。
- UI regression audit: `npm run audit:demo -- --port 5183` 通过 Plush semantic、Plush v1、NeRF Lego proxy 三个默认闭环样例。
- Validation: `uv run --extra dev pytest` 32 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- TRAIN-003A script smoke: dry-run 输出完整 Nerfstudio Splatfacto、`ns-export gaussian-splat`、SAM manifest、Object Field init / vote-masks 和 PLY stats 命令；`--status` 在本机检查 9 项输入/输出，`status=ready missing=0`。
- Python 测试: 32 passed。
- Object Emergence smoke: assignment_confidence=0.797826，effective_slots=7.323355，spatial_compactness_score=0.968811，stability_ari=0.642209，matched_label_agreement=0.825040，partial OES=0.772490。
- Object Emergence curve smoke: 5 points，projection_loss 4.384474 -> 0.308315，assignment_confidence 0.791077 -> 0.797826，effective_slots 7.994654 -> 7.323355，ari_to_initial 1.000000 -> 0.642209，spatial_compactness_score 0.979225 -> 0.968811，mask_proxy_occlusion_mean_delta_loss 1.428752 -> 1.927487；当前 scale-aware CPU splat probe 在 semantic smoke acceptance 中 Lego Splatfacto smoke render_occlusion_effect_score=0.137784。
- Object Emergence benchmark report smoke: Plush semantic、Lego alpha proxy、Lego Splatfacto smoke 三条本地曲线聚合为 `/tmp/objgauss-emergence-benchmark-report.html`，curves=3，charts=7；最终 render_occlusion_effect_score 分别为 0.227482、0.236530、0.124240。
- Object Emergence benchmark suite smoke: `docs/benchmarks/semantic-smoke.json` 严格模式通过，输出 `/tmp/objgauss-semantic-smoke-suite/summary.json` 和 `report.html`；3 scenes 全部 passed，projection loss 分别为 1.386294 -> 1.346402、1.386294 -> 0.235765、4.384474 -> 0.339695。
- Semantic benchmark acceptance: `npm run acceptance:semantic` 通过，输出 `acceptance_semantic_benchmark=passed`。
- Full demo acceptance: `npm run acceptance:demo` 通过，已在闭环 demo 生成、浏览器 audit 后执行 SEMANTIC benchmark suite，输出 `acceptance_demo=passed`。
- 前端构建: 通过，仍有 bundle size warning。

2026-06-22:

```bash
uv run --extra dev pytest
uv run objgauss demo audit-v1-goal
npm run build
npm run acceptance:demo
```

结果：

- Python 测试: 24 passed。
- 前端构建: 通过。
- 浏览器验证: 桌面 1440x920 与移动端 390x844 均渲染非空、无前端错误。
- OBJECT-FIELD-BG-TRAIN-001: Object Field mask voting 新增训练级 background slot 机制；`vote-masks` / `training register-output` 可把投影可见但未命中前景 mask 的 Gaussian 作为背景槽投票训练，并在 summary / manifest 记录 `background_training`。本地单元测试覆盖底层投票、CLI summary、登记 manifest 和 object-aware PLY 导出。
- OBJECT-FIELD-BG-TRAIN-001 local Lego registration: 使用现有 168,653-Gaussian near-1M candidate PLY 重新登记 background slot 4，输出 ignored `outputs/assets/gaussians/nerf-lego-trained-near1m-sam8f-balanced03-slots4-bgslot4/object_aware_gaussians.ply` 和 public local copy `public/samples/nerf_lego_trained_near1m_bgslot4_objects.ply`；`slots=5`，`supervised_gaussians=118729`，`background_matched=496056`，projection loss `2.245413 -> 0.604945`，object counts `15810/9080/15376/25100/103287`。该结果证明背景槽已作为训练信号生效，但前景 slot 1/2/3 的 mask winners 仍少，不能单独证明对象语义已稳定分离。
- SCENE-BUNDLE-001: 新增 scene/object bundle 训练数据层；`from-nerf-alpha-fgbg` 可生成 K=2 foreground/background + ignore alpha mask，并通过 `--background-confidence` / `--foreground-confidence` 把 full-frame background supervision 降权，避免无 depth/visibility occlusion 的投票路径被大背景面积压倒；`masks validate` 可做训练前一致性检查，`training write-sample-bundle` 可写顶层 `sample.json` 绑定 dataset、masks、Gaussian PLY、Object Field 和 slot 定义。当前推荐本机 ignored bundle 为 `outputs/samples/objgauss-lego-alpha-fgbg-bg005-v2/sample.json`；Lego alpha fgbg manifest 为 100 frames / 200 masks / foreground `843797` / background `61517043` / ignore `1639160`，validation passed，K=2 Object Field `supervised_gaussians=149892`，projection loss `1.264038 -> 0.497213`，object counts background/foreground=`133074/35579`，high-confidence export at `min_confidence=0.7` 为 background/foreground/unknown=`100730/16915/51008`。`background_confidence=0.02` 对照可把 foreground 扩到 `80071`，但更可能吸收背景 / 桌面；两组训练均为 CPU/Object Field 路径，本机 GPU 仍约 `596MiB` used / `15246MiB` free。该结果证明可追溯 bundle 和 alpha fgbg 训练目标跑通，但 baseline `vote_conflict_fraction=0.430143` 仍高，不是 part-level 稳定分离结论；TRAIN-QUALITY-002 的 depth-buffer diagnostic 可将 alpha fgbg conflict 降到 `0.402118`，但尚未 promotion 为默认训练策略。
- ASSET-001: Poly Haven School Chair 实际拉取 5 个文件；NeRF Synthetic Lego 实际抽取 805 个文件。
- OBJFIELD-001: Plush PLY 可初始化 6-slot Object Field；NeRF Lego 检查 400 frames、缺图 0、无效 pose 0。
- SEG-001 / OBJFIELD-002: synthetic projection mask vote 可训练 Object Field，并输出 `object_id` PLY。
- DEMO-001: Plush v1 闭环 demo 生成 281498 个 Gaussian、6 个对象、3 个投影视角、18 个 masks；projection loss 1.791760 -> 1.201637；浏览器验证可加载 `ObjGauss v1 闭环样例` 并执行对象选择、隔离、删除预览。
- VERIFY-001: `objgauss demo verify-v1-closure` 通过，检查真实 splat、mask manifest、Object Field shape、loss 下降、`object_id` PLY、public copy 和前端素材注册。
- MASK-001: NeRF Lego 真实 RGBA 图片 alpha 生成 mask manifest，8 frames / 8 masks / 800x800 / 299242 foreground pixels。
- LEGO-001: `objgauss demo lego-alpha-closure --max-frames 12 --sample-stride 8 --iterations 120` 生成 5696 个 Gaussian proxy、4 个对象、12 个真实视角、48 个 2D color masks；projection loss 1.386294 -> 0.538856；浏览器验证可加载 `NeRF Lego 闭环代理样例` 并执行对象选择、隔离、删除预览。
- VERIFY-002: `objgauss demo verify-lego-alpha-closure` 通过，17 项检查全部通过，包括源图像和 mask 文件存在、Object Field shape、loss 下降、`object_id` PLY、public assets 和前端素材注册。
- UI-AUDIT-001: `npm run audit:demo` 通过，加载 `Plush 2D 语义 Mask 闭环样例`、`ObjGauss v1 闭环样例` 与 `NeRF Lego 闭环代理样例`，检查 splat / 点云编辑 canvas 非空，并执行对象选择、只看所选、预览删除。
- ACCEPT-001: `npm run acceptance:demo` 通过，重新生成并验证 Plush v1 closure、Plush semantic closure、NeRF Lego proxy closure，然后执行浏览器闭环验收；输出 `acceptance_demo=passed`。
- MASK-002: `objgauss masks from-nerf-rgba-colors` 在 NeRF Lego 真实 RGBA 上生成 8 frames / 32 masks / 4 slots；独立 `vote-masks` 消费该 manifest，3423 个 Gaussian 被监督，projection loss 1.386294 -> 0.390825，并输出 `object_id` PLY。
- TRAIN-002: `objgauss training register-output` 接入 Gaussian PLY smoke 通过，生成 viewer splat、Object Field 和 `object_id` PLY；使用真实 Lego color mask manifest 时 supervised_gaussians=4806，projection loss 1.386294 -> 0.375765。
- SEG-002A: `objgauss masks from-nerf-sam --help` 可用；SAM manifest 生成逻辑由 fake generator 测试覆盖，输出 `sam-automatic-mask-generator` manifest 和 boolean `.npy` masks。
- VERIFY-003: `npm run acceptance:demo` 已检查 `mask_guidance_changed_object_field`；Plush changed_gaussians=196457，Lego proxy changed_gaussians=4960，证明 mask supervision 实际改变 Object Field labels。
- SEMANTIC-001: `objgauss demo plush-semantic-closure` 在真实 Plush 3DGS 上生成 3 views / 12 masks / 4 objects；281498 个 Gaussian 全部被监督，104403 个 hard labels 被 2D mask guidance 改变，projection loss 1.386294 -> 1.345684。
- AUDIT-001: `objgauss demo audit-v1-goal` 严格模式通过，当前证据为 unified，completion_blockers=`-`。
- VERIFY-004: `objgauss object-field vote-masks` summary、闭环 demo manifest 和 verifier 已包含 mask vote quality audit；本地测试覆盖 per-slot coverage、conflict fraction、normalized target entropy 和 verifier 检查。
- SEG-002: 真实 SAM checkpoint 小场景验收通过；`from-nerf-sam` 在 NeRF Lego 2 帧上生成 8 个 SAM masks，`vote-masks` 监督 5567 / 5696 个 Gaussian，supervised_fraction=0.977353，vote_conflict_fraction=0.064308，projection loss 3.902681 -> 0.120758，并输出带 `object_id` 的 PLY。
- TRAIN-001: Nerfstudio Splatfacto smoke 训练通过。`ns-train splatfacto ... blender-data --data outputs/assets/training/nerf-synthetic-lego` 完成 100 iterations，checkpoint 为 `outputs/training/nerf-lego-splatfacto-smoke/lego-splatfacto-smoke/splatfacto/smoke-cuda/nerfstudio_models/step-000000099.ckpt`；`ns-export gaussian-splat` 导出 `outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply`，ObjGauss 读取为 50000 gaussians。
- TRAIN-001 环境结论: 当前 RTX 5060 Ti / PyTorch `2.12.1+cu130` / CUDA 13.0 环境需要为 `gsplat` JIT 显式加入 `nvidia-cuda-nvcc==13.0.*`、`nvidia-cuda-cccl==13.0.*`、`nvidia-nvvm==13.0.*`、`nvidia-cuda-crt==13.0.*`；未对齐时会出现 no `nvcc`、CUDA 13.3 header/compiler mismatch、PTX version mismatch 或 `-lcudart` 链接失败。导出本地可信 checkpoint 时需设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 兼容 PyTorch 2.6+ 的 `torch.load` 默认行为。
- TRAIN-001 Object Field smoke: 对导出的 `splat.ply` 执行 8-slot init 和 SAM mask vote，`supervised_gaussians=8887 / 50000`，`supervised_fraction=0.177740`，`vote_conflict_fraction=0.268707`，projection loss `4.384474 -> 0.308315`，最终 `outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/lego_splatfacto_sam_objects.ply` 含 `object_id` 和 RGB 字段。
- SEMANTIC-002: `objgauss object-field emergence` 已提供 object emergence 观测指标。Synthetic 测试覆盖 assignment confidence、effective slots、spatial compactness、permutation-invariant ARI 和 partial OES；在 NeRF Lego Splatfacto smoke 上输出 assignment_confidence=0.797826，effective_slots=7.323355，spatial_compactness_score=0.968811，stability_ari=0.642209，matched_label_agreement=0.825040，partial OES=0.772490。
- SEMANTIC-003: `objgauss object-field emergence-curve` 已提供随 mask-vote training iteration 变化的 benchmark 曲线，输出 JSON 和 CSV。
- SEMANTIC-004: `emergence-curve` 已新增 scale-aware CPU splat render occlusion delta；默认从 mask manifest 的相机位姿做 CPU 重渲染 probe，输出 `render_occlusion_delta`、CSV render 列、target/non-target/locality 字段和曲线内 occlusion-effect OES component。当前 probe 仍不是 covariance-aware 3DGS / gsplat renderer。
- SEMANTIC-005: `objgauss object-field emergence-report` 已可将多个 curve JSON 聚合为 HTML/SVG 报告；本地 smoke 已覆盖 Plush semantic、Lego alpha proxy 和 Lego Splatfacto smoke 三个场景曲线。
- SEMANTIC-006: `objgauss object-field emergence-benchmark` 已可从 `docs/benchmarks/semantic-smoke.json` 一键重跑 3-scene semantic smoke suite，生成 per-scene curve JSON/CSV、summary JSON、HTML report，并在 `--strict` 下执行阈值检查。
- SEMANTIC-007: `npm run acceptance:semantic` 已作为独立 benchmark acceptance；`npm run acceptance:demo` 默认纳入 SEMANTIC benchmark suite，并提供 `--skip-semantic-benchmark` 保留 demo-only 验收。`docs/benchmarks/semantic-smoke.md` 记录缺失 `outputs/` 时的生成命令和 Splatfacto smoke 边界。
- TRAIN-003A: `npm run train:splatfacto:smoke` 已将 NeRF Lego Splatfacto 100-step smoke 的生成过程固化为 dry-run/status/run 三模式脚本；`docs/training/splatfacto-smoke.md` 记录 CUDA / `gsplat` 环境、SAM checkpoint、输出 contract 和验证命令。
- 已知提示: Vite 报 Spark / Three.js chunk 超过 500KB，不影响当前预览。

## 当前限制

- Hugging Face 上的 near-1M NeRF Lego Dataset / Model 当前只是 development-stage release。它记录研究复现资产和模型产物 handoff，不代表稳定公开版本；远端大文件已核对通过，sampled1m derivative 已通过 `audit:webgpu-cpath-production-sla` 形成 terminal proof，但 HF 全量 `4,503,634`-Gaussian object-aware PLY 直接 browser runtime 仍未达到 production-interactive。
- 对象聚类色和部分诊断仍走 `Gaussian OIT 编辑` fallback 或 WebGPU tile route；`原始颜色（编辑预览）` 在 object edit active 后 no-SH 样例可走 Spark native compact `.splat` object mask，SH-heavy 样例保留 PLY packed route 以保存 degree-3 SH rest。剩余颗粒感主要来自 object_id 子集稀疏、边界 assignment 噪声、透明混合中被隐藏对象不再贡献，以及删除后没有补洞 / 重优化。WebGPU full runtime 内部输出、bilinear resolve、aspect-fit viewport、camera-Jacobian covariance、depth-binned alpha composite、Spark-frame camera diagnostic 和 front-top-k diagnostic 已把 coverage / sorting / color 残差拆成可审计项。当前 headless unsafe WebGPU failure 已归类为 canvas render pass / presentation backend limitation；headed desktop Chrome/WebGPU 已通过 NeRF Lego proxy、Plush 和 safe-2000 Splatfacto 的 full WebGPU tile runtime audit。
- Spark `SplatMesh.raycast` 当前可命中 splat depth，但 intersection 不暴露 splat index / object id，且不能证明 object opacity mask filter-aware；因此对象选择仍应使用已验收的 `hover-confirm-v1` screen-space pick，不能宣称已经是 renderer-native object picking。
- `plush-semantic-closure` 已证明真实 3DGS + 非 KMeans 2D color masks + Object Field + 前端对象编辑的统一闭环；但它仍是确定性颜色规则，不等价于 SAM / CLIP 实例语义分割。
- 当前 v1 闭环 demo 的 Plush mask manifest 由已有对象标签派生，用于回归验收；NeRF Lego alpha/color masks 已能从真实图片生成，但仍是确定性 alpha/颜色规则，不等价于 SAM / CLIP 实例语义分割。
- SAM 入口已用真实 checkpoint 跑通小场景 manifest 和 `vote-masks` 验收；`objgauss masks score-clip --backend transformers --device cuda` 已用临时 `uv --with` 依赖跑通真实 CLIP inference，并已补 mask-level / slot-level naming quality gate，但当前 aligned slot labels 仍未通过语义质量 gate，不能 promotion 为默认语义命名策略。
- Object Emergence Score 的单点 `emergence` CLI 仍是 partial OES；`emergence-curve` 在提供 cloud 和 mask manifest 时已覆盖 assignment / stability / spatial compactness / scale-aware CPU splat render occlusion。`emergence-benchmark` 当前是本地 smoke suite，依赖 ignored `outputs/` 产物；缺失输入时按 `docs/benchmarks/semantic-smoke.md` 与 `docs/benchmarks/splatfacto-scenes.md` 生成。本 suite 仍不是 CI 固定 public benchmark。gradient coherence 和 covariance-aware 3DGS renderer occlusion 仍未实现，不能据此单独宣称 object emergence 完成。
- 当前已有 solver / decoder / gsplat image loss 的最小训练 smoke，但仍不是可推广的长程
  full renderer training 结论。opacity-only 和 scale-only GPU path 已证明 checkpoint /
  TensorBoard / eval gate 可用，但收益很弱；继续训练前需要先证明 ObjectState 是稳定
  latent variable。
- 当前近期算法主线已从 renderer field thaw 转向 V2 stability foundation：先冻结
  identity oracle、synthetic world generator、scenario diagnostics 和 invariant-first gate，
  再讨论 rollout model 或更大规模训练。
- NeRF Lego 闭环代理样例仍是 posed RGBA 生成的轻量 Gaussian proxy；另有 Nerfstudio Splatfacto 100-step smoke 产物和 TRAIN-003A runbook/script 证明本机可复现真实 3DGS optimization PLY，但尚未作为前端公开样例固化。
- 外部训练输出接入命令已完成，本机已产出真实 NeRF Lego Splatfacto smoke PLY、500-step resource-safe public sample candidate 和 2000-step higher-quality geometry candidate；safe-2000 经过 8-frame balanced SAM 后已消除近空 object slots、提升 render occlusion effect，并通过当前 public sample 浏览器 audit。
- Poly Haven mesh Demo 还不能直接进入现有 3DGS viewer；当前已具备 mesh -> NeRF-style render set -> Splatfacto smoke 的 benchmark 链路，但不是公开前端 demo。
- 训练素材目录已接入 NeRF Lego、LLFF Fern 与 Poly Haven Chair NeRF render set；Fern 和 Chair 当前只是 100-step smoke，不代表高质量 reconstruction。

## 下一步主线

1. 产品 viewer 线：`GAUSSIAN-OBJECT-PROCESS-FLOW-001` 已补 raw source -> CLI handoff -> object layer ready 主入口；`DEMO-CATALOG-REAL-SPLAT-001` 已下载 Nike 真实 `.splat` 并把首屏模型入口收敛成 5 个 curated demos；下一步若继续 viewer 聚焦全量 4.5M PLY 的 LOD / streaming / 分块加载，以及任意第三方 `.splat` object id / rotate-scale native motion / Gaussian 重优化等仍未承诺边界。
2. 语义质量线：depth-aware mask voting、manifest-level 跨视角 slot alignment、CLIP score cache contract、真实 `transformers` CLIP run、mask-level gate、slot-level gate、baseline comparison 和 promotion policy 已落地；真实 CLIP 语义路线仍保持 `do-not-promote`，下一步应扩大 foreground coverage / mask selection 证据，而不是把更多训练步数当作语义质量解释。
3. 算法模型线：`CORE-MODEL-TRAIN-VALIDATE-001` 已聚合 v2 assignment training、synthetic
   stability hard gate、failure diagnostics、ObjectState eval、renderer joint smoke、
   checkpoint roundtrip 和 renderer-loss-contract evidence。当前 public sample 已跑到
   可训练、可验证、可 3D 查看对象分割效果阶段；`feature_weight=2.0` 的 weak-boundary
   candidate 已把 `max_points=128` full-cloud hard segmentation 修到 `mixed_gaussians=0`，
   最新 strict sample-aware gate 已让 3-row 表中 Lego 选择 promoted、Polyhaven / Nike 回退
   baseline，并避免 selected hard regression；Plush KMeans 暴露为无安全候选的负证据。
   下一步若继续算法质量，应先在有本地 / ignored BOP subset 的环境运行
   `init-bop-phase1-batch-workspace <dataset-root> --workspace-root <dir>`，再运行
   `init-bop-phase1-sample-workspaces <batch-spec.json>` 生成每个 sample 的 condition CSV
   模板和 README；随后填齐真实 `bop-condition-sidecar.json` 和 per-frame Gaussian
   evidence。若还没有真正模型输出，可先运行
   `bop-rgbd-baseline-local-row-handoff` 从 depth 生成 Gaussian evidence seed、
   single-state baseline `objectstates.json` 和 identity+prediction local-row package；
   若 Gaussian evidence 已经存在，也可直接跑 `bop-baseline-local-row-handoff`；若已有
   真实模型输出，则继续使用 template / finalize 路径，再跑 `bop-local-row-handoff`。之后先跑
   `audit-bop-phase1-authoring-progress` 或 `audit-bop-local-row-batch-readiness` 确认
   target files 与 package 已经可进入 batch / cross-sample 表，再按 readiness 缺口决定
   是否运行 `bop-local-row-batch-handoff` 扩大 cross-sample 表；不要直接跳到 rollout、
   replay buffer、diffusion 或 geometry / camera unfreeze。若继续 action / intervention
   route，`accept-controlled-capture-bundle --require-intervention-ready`、full controlled
   reality readiness、`controlled-reality-bundle-handoff` 和
   `transition-reality-handoff` 现在都会要求 `intervention_action_gt_ready=true`：
   action 必须有非零 vector，并且 action interval 必须覆盖被引用对象的连续 pose
   transition，弱 `actions.csv` 不能进入 full 或 transition-backed Phase 1 intervention
   evidence。`audit-public-interaction-route` 和
   `audit-public-interaction-workspace-progress` 也已同步该 gate：public interaction
   route 只有在 `capture_intervention_action_gt_ready=true` 时才会进入
   handoff-ready，零向量或无法覆盖 pose transition 的公开交互 action rows 会停在
   `objectstate_public_interaction_route_intervention_gt_required`。
   `OBJECTSTATE-REAL-BUNDLE-SCHEMA-001` 已进一步把真实证据从“准入门禁”推进到
   “证据记账合同”：新增 `objgauss-objectstate-real-evidence-bundle-v1`，
   `validate-real-evidence-bundle` 可验证 observation、object pose、identity link、
   action interval、state transition 和 gate accounting rows，并强制 intervention
   pass / fail accounting 引用同一对象上时间重叠的 action / transition pair；
   summary 同时输出 `action_transition_coverage_rate`，并把
   `static_scene_evidence` 与 `state_variable_evidence` 分开。
   `OBJECTSTATE-REAL-IDENTITY-ROWS-001` 已把 bundle 内 `identity` accounting rows
   接入 identity-only reality gate：`pass` / `fail` 进入真实 row accounting，
   `evidence_incomplete` / `unsupported` 映射为 blocked，不算模型失败。
   `OBJECTSTATE-REAL-PREDICTION-ROWS-001` 已把 bundle 内 `prediction` accounting rows
   接入 prediction-only reality gate：`pass` / `fail` 必须引用同一对象的
   `StateTransitionRow`，并保留 `state_ade` / `history_ade` /
   `prediction_gap_vs_history_model` 和 `state_vs_history_error_ratio` 比较；
   `evidence_incomplete` / `unsupported` 继续保持 blocked。
   `OBJECTSTATE-REAL-INTERVENTION-ROWS-001` 已把 bundle 内 `intervention`
   accounting rows 接入 intervention-only reality gate：`pass` / `fail` 必须引用
   同一对象上时间重叠的 `ActionIntervalRow` / `StateTransitionRow`，并要求 transition
   source / target identity link 保持同一 `physical_identity_id`；summary 保留
   `action_conditioned_ade` / `no_action_ade` / `intervention_gain` /
   `counterfactual_outcome_accuracy` / `wrong_direction_rate` /
   `identity_consistency_rate` 和 `action_transition_coverage_rate`。
   `OBJECTSTATE-REALITY-ROW-LEDGER-REAL-SUMMARIES-001` 已扩展
   `audit-reality-row-ledger`，可直接消费 real identity / prediction / intervention
   summary JSON，并把其中的标准 reality rows 汇入同一个 full reality gate、gap summary
   和 state-variable evidence matrix。`OBJECTSTATE-REAL-BUNDLE-LEDGER-HANDOFF-001`
   已新增 `audit-real-evidence-bundle-ledger`，可从一个或多个 real evidence bundle
   一条命令写出 bundle summary、三类 real row summary、full reality ledger、blocked rows、
   state-variable evidence matrix 和 next actions；full `reality-row-ledger` 仍是 pass /
   fail / blocked 的权威输出。下一步应让真实 controlled/public bundle 产物进入这条
   handoff，而不是继续新增大模型。`OBJECTSTATE-REAL-BUNDLE-LEDGER-ACCOUNTING-STATUS-001`
   已把 row summary 中的 `pass` / `fail` / `evidence_incomplete` / `unsupported`
   accounting status counts 汇总到 bundle-ledger wrapper、每条 record、row_counts 和
   package audit 输出；因此 `evidence_incomplete` / `unsupported` 在总账层也不会被
   混同为模型 fail，只会同时表现为 reality row 的 blocked。`OBJECTSTATE-REAL-BUNDLE-LEDGER-PACKAGE-AUDIT-001`
   已新增 `audit-real-evidence-bundle-ledger-package`，可对 handoff output root 做只读
   reviewability audit：检查 wrapper、standalone `reality-row-ledger.json`、per-bundle
   summaries、blocked rows、state-variable evidence matrix、next actions、row counts 和
   static/state evidence 分账一致性。`OBJECTSTATE-REAL-BUNDLE-LEDGER-PHASE1-ACCEPTANCE-001`
   继续把目标文件第 7 节的 Phase 1 通过条件做成 package audit 的
   `phase1_acceptance_*` 输出：package reviewability、至少一个 controlled/public
   real bundle loaded、三类 row 进入 accounting、至少一个可评估 pass/fail row、缺 GT 的
   `evidence_incomplete` / `unsupported` 不混成 fail、synthetic / real gate 分账和
   static scene / state-variable evidence 分账。该 status 只表示证据系统 acceptance，
   不声明 metric pass 或 world-model proof。`OBJECTSTATE-BOP-REAL-BUNDLE-ADAPTER-001`
   继续把 BOP/public replay 产物接入这条新 handoff：新增
   `objgauss-objectstate-bop-real-evidence-bundle-adapter-v1` 和
   `objgauss object-state bop-real-evidence-bundle <bop-acceptance.json>
   <bop-reality-rows.json> --bundle-output <real-bundle.json>`，可从已有 BOP
   acceptance summary 的 controlled capture manifest 生成 observation / object pose /
   identity link / state transition rows，并把 BOP reality rows 映射为 real bundle
   gate accounting rows。BOP blocked intervention row 会进入
   `accounting_status=evidence_incomplete`，不会伪造成 action-conditioned pass；若没有
   非零 action interval 和 action/transition overlap，adapter 也不会创建 intervention
   pass / fail。测试中的 RGB-D BOP public replay bundle 进入
   `audit-real-evidence-bundle-ledger` 后形成 `pass=1`、`fail=1`、
   `evidence_incomplete=1` 的可审计 Phase 1 accounting，并通过 package
   `phase1_acceptance_status=objectstate_phase1_evidence_system_acceptance_pass`；
   这仍不声明 reality gate pass 或 world-model proof。
   2026-07-09 随后在本地 ignored HOPE + LMO public replay outputs 上复跑真实链路：
   两个 `bop-real-evidence-bundle.json` 均为 ready，HOPE 为
   `pass=1 / fail=1 / evidence_incomplete=1`，LMO 为
   `pass=0 / fail=2 / evidence_incomplete=1`。首次双 bundle ledger 暴露
   `bop-real-accounting:identity|prediction|intervention` 跨样本 row id 重复，
   已将 BOP adapter 生成的 observation / pose / identity / action / transition /
   gate accounting row id 改为 sample-scoped。复跑
   `audit-real-evidence-bundle-ledger` 后 package 变为 reviewable：
   `row_count=6`、`pass=1`、`fail=3`、`blocked=2`、
   `evidence_incomplete=2`、`unsupported=0`、`duplicate_row_ids=[]`；
   `audit-real-evidence-bundle-ledger-package --require-phase1-acceptance`
   通过，`phase1_acceptance_status=objectstate_phase1_evidence_system_acceptance_pass`。
   Full reality gate 仍按设计为 fail：identity persistence 是真实 fail evidence，
   predictive sufficiency 只有 HOPE pass / LMO fail，counterfactual action interface
   仍 blocked，`state_variable_intervention_ready_bundle_count=0`。
   若只是想扩展现成静态 Gaussian 场景，`docs/asset-library.md` 已新增候选表：
   优先本地审计 cakewalk `room.splat` / `train.splat`，再考虑 `truck`、`garden`、
   `bicycle`、`stump`、`treehill` 或 GraphDECO / Inria official large results；
   `cakewalk-room-3dgs-local`、`cakewalk-train-3dgs-local`、`cakewalk-truck-3dgs-local`、
   `cakewalk-garden-3dgs-local`、`cakewalk-bicycle-3dgs-local`、
   `cakewalk-stump-3dgs-local` 和 `cakewalk-treehill-3dgs-local` 已进入 assets
   registry，可通过 `objgauss assets pull <asset_id>` 走现有
   `splat-to-objgauss-ply` 管线拉取和转换。`garden` / `bicycle` / `stump` /
   `treehill` 是大体量本地研究候选，适合 LOD / renderer pressure 和静态分割负例，
   不能替代带 timestamped identity / 6DoF pose / action GT 的 Phase 1 rows，也未进入
   viewer/export 默认策略。`room` / `train` 已补非默认 viewer catalog 入口；拉取
   `cakewalk-room-3dgs-local` / `cakewalk-train-3dgs-local` 并生成 ignored
   `public/samples/room_objects.ply` / `public/samples/train_objects.ply` 后，可在
   viewer 模型版本列表中作为本地静态候选加载，但不会出现在首屏 featured dock。
   2026-07-09 本地已执行这两个 P0 小场景 pull：`room.splat` 写入
   `public/samples/room.splat` / `public/samples/room_objects.ply`，约 1,593,376
   Gaussians / 8 clusters，文件大小约 49 MB / 60 MB；`train.splat` 写入
   `public/samples/train.splat` / `public/samples/train_objects.ply`，约 1,026,508
   Gaussians / 5 clusters，文件大小约 32 MB / 39 MB。这些仍是 ignored local
   static-scene artifacts，只用于 viewer / segmentation / cross-sample smoke，不提交
   git，也不进入首屏 featured dock、viewer/export 默认或 State Variable Gate pass row。
   `GAUSSIAN-SCENE-EXPANDED-SOURCE-TRIAGE-001` 继续补充
   cakewalk 现成 Gaussian PLY、GaussianSplats3D sample archive 和 Niantic SPZ
   samples 作为后续格式适配候选；这些需要 `.ply` / `.ksplat` / `.spz` import
   contract 或本地解包审计，暂不进入 pullable registry。
4. 后续 SEG: CLIP / color-mask / KMeans baseline comparison，alignment 质量指标和 promotion policy。
5. 将 Poly Haven mesh -> NeRF-style render set -> Splatfacto smoke 链路升级为可审计的公开 demo 候选前，先补许可说明、质量阈值和浏览器验收。
6. 后续 renderer 优化: Spark 按需加载或拆包，降低首屏 bundle。
