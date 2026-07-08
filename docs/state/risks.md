# ObjGauss 风险登记

> 最近更新: 2026-07-08

| ID | 风险 | 影响 | 当前缓解 | 关闭条件 | 状态 |
| --- | --- | --- | --- | --- | --- |
| R-001 | 当前不是完整 3DGS renderer，只是高斯中心点云预览 | Demo 视觉效果和真实 3DGS 有差距 | 已接入 `@sparkjsdev/spark`，Plush `.splat` 已通过桌面和移动端浏览器验证 | 真实 splat renderer 接入并通过浏览器验证 | closed |
| R-002 | 当前默认对象分组不是端到端语义级对象分割 | 对象边界和语义一致性不稳定 | 已落地 `SEMANTIC-001`：真实 Plush 3DGS + 非 KMeans 2D color masks + Object Field + `object_id` + 前端对象编辑统一闭环；`VERIFY-003` 已检查 mask guidance 实际改变 Object Field labels；`VERIFY-004` 已固化 mask vote quality audit，检查监督覆盖、每槽覆盖和投票冲突；`SEG-002` 已用真实 SAM checkpoint 跑通小场景 manifest 和 `vote-masks`；`UI-AUDIT-001` / `ACCEPT-001` 已固化浏览器和一键总验收；默认 KMeans 仍保留为 baseline；`CLIP-BASELINE-003` 已完成 CLIP / color-mask / alpha / KMeans comparison policy，`CLIP-QUALITY-004` 已缓解 slot-level label collapse，`CLIP-BALANCE-001` 已清除当前 CLIP route 的 slot balance blocker；但当前 CLIP 仍因 mask-level 背景 dominant 和 supervised fraction 低保持 `do-not-promote` | SAM / CLIP 结果通过 slot naming、vote quality、training summary 和 baseline comparison promotion policy | open |
| R-003 | 只有 Plush 自动拉取，其他素材无转换管线 | 训练和 Demo 数据不足 | 已接入 `polyhaven-school-chair-1k` 和 `nerf-synthetic-lego` 自动拉取管线 | 至少一个 Demo 源和一个训练源跑通转换 | closed |
| R-004 | 仓库尚无 baseline commit | 进度不可追踪，后续 AI 会话难以协作 | 已创建 baseline commit `c8dcef7` 并回填状态文件 | baseline commit 存在且状态文件回填 | closed |
| R-005 | Plush 来源许可混合 | 不适合公开发布或商用 Demo | `docs/asset-library.md` 已标明仅本地测试 | 首个公开 Demo 改用许可明确素材 | open |
| R-006 | Three.js / Spark bundle 超 500KB warning | 后续页面加载可能变慢 | 当前只记录，不影响 MVP；RENDER-001 后主 JS 约 5.6MB / gzip 1.94MB | 引入 code splitting 或按需加载 Spark renderer | accepted |
| R-007 | Poly Haven mesh 还不是可直接 viewer 打开的 3DGS Demo | 许可干净素材已接入，但公开演示仍需要训练转换 | 已记录 mesh -> 多视角渲染 -> 3DGS 后续链路 | School Chair 训练出 `.splat` / ObjGauss PLY 并可前端加载 | open |
| R-008 | SAM / CLIP 仍是可选外部模型能力 | 语义分割效果依赖本地 checkpoint / 模型缓存，仓库不能无依赖端到端复现实例分割 | 已定义稳定 mask manifest 和 `vote-masks` 消费命令；`objgauss masks from-nerf-alpha` 已能从 NeRF Lego 真实图片生成前景 mask manifest；`objgauss masks from-nerf-rgba-colors` 已能从真实图片生成多 slot color mask manifest；`objgauss masks from-nerf-sam` 已作为可选 SAM 入口接入，且真实 `sam_vit_b` checkpoint 小场景 manifest 已被 `vote-masks` 消费；`objgauss masks score-clip` 已跑通真实 `transformers` CLIP inference，并已落地 label preset / prompt template / background fill / mask-level naming quality gate；`align-slots` 已落地 slot-level naming quality gate、背景 / 低面积过滤、foreground-only / unique / diversity naming policy 和 slot support rebalance policy；`objgauss masks compare-baselines` 已把 CLIP slot naming、slot rebalance 与 color-mask / alpha / KMeans reference 放入同一 promotion policy；当前 CLIP slot-level naming 和 slot balance 可通过，但 mask-level 背景 dominant 与 supervised fraction `0.114283 < 0.200000` 仍让整体保持 `do-not-promote` | 真实 CLIP 通过 mask-level / slot-level 命名质量 gate，并补齐 downstream vote quality / training summary 后通过 comparison promotion policy | open |
| R-009 | NeRF Lego 还没有真实训练出的 Gaussian PLY / `.splat` | Lego 路线仍依赖 proxy，不能完全代表真实 3DGS optimization 输出 | 已完成真实 NeRF Lego Splatfacto safe-500 / safe-2000 / near-1M tuned candidate；near-1M 本地 object-aware PLY 为 `4,503,634` Gaussians，并已接入前端快速查看 / 按需加载对象 PLY | 真实 Lego Gaussian 已产出、登记并可前端加载 | closed |
| R-010 | 此前 near-1M trained object-aware PLY 已存在但 production SLA 未通过 | 不能把 HF 发布或本地 near-1M PLY 误称为 terminal proof / production ready | deterministic sampled1m derivative 已通过 `audit:webgpu-cpath-production-sla`；`audit:near1m-production-gap -- --require-ready` 和 `audit:renderer-route-goal -- --require-production-ready` 在绑定 sampled1m evidence 时均为 ready | production SLA summary 为 `status="passed"`，且 renderer route goal strict gate 不再缺 near-1M proof | closed |
| R-011 | Hugging Face 公开仓库处于开发阶段，远端大文件状态可能与本地记录不一致 | 其他人下载时可能缺少 object-aware PLY 或 checkpoint，复现失败 | HF README 已加 development-stage 注释；Dataset object-aware PLY 和 Model checkpoint 已上传并远端核对，大小 / checksum / commit 已写入 `docs/state/huggingface-release.md` | Dataset object-aware PLY 和 Model checkpoint 均远端可见，大小 / checksum / commit 已写入 `docs/state/huggingface-release.md` | closed |
| R-012 | HF 全量 `4,503,634`-Gaussian object-aware PLY 不能直接达到 production-interactive browser runtime | 其他人下载 HF 全量 PLY 后可能误以为 viewer 可无优化直接流畅交互 | HF / 项目文档明确区分 development-stage full PLY 与 sampled1m terminal proof；默认 viewer 仍优先 `.splat` 快速查看和按需加载 PLY；`audit:large-model-viewer-route` 已证明 near-1M quick view 不请求 object-aware PLY，只有对象编辑 / `加载对象 PLY` 才请求；当前 full PLY runtime min approx FPS=`4.412` 已记录为 negative evidence | LOD、streaming、分块加载或全量性能优化通过 full PLY production SLA | open |
| R-013 | 训练模型主线受当前 torch / gsplat / CUDA / NVIDIA driver 环境阻塞 | `TRAIN-GSPLAT-MVP-001` 无法在当前 Codex 环境证明 full renderer training MVP；若继续重复尝试会浪费队列并可能把 point renderer / deterministic Debug OS 误记为 gsplat 训练成功 | 已确认此前 `nvidia-smi` 失败来自默认沙箱 `/dev` 视图不暴露 `/dev/nvidia*`，不是 host driver 不可用；host / 提权环境中 RTX 5060 Ti、driver `595.71.05`、CUDA `13.2` 可用。临时 uv 环境已验证 `torch 2.12.1+cu130`、`gsplat 1.5.3`，并复用 `/tmp/objgauss-cuda13` + CUDA 13.0 uv package set 跑通显式 `--image-renderer gsplat` 2-iteration smoke；`renderer-loss-contract` 输出 `status=full_3dgs_renderer_ready`、`upgrade_blockers=[]` | `TRAIN-GSPLAT-MVP-001` 以显式 `--image-renderer gsplat` 完成小规模 smoke 并产出验收 summary | closed |
| R-014 | `audit:world-viewer` full script 覆盖过宽且部分等待条件已落后于当前精简 UI | 新 viewer 切片可能已通过 targeted browser evidence，但 full audit 仍在旧 trainable Gaussian probe / stability 条件超时，导致验收信号混杂 | `NATIVE-SPLAT-OBJECT-TRANSFORM-001` 和 `NATIVE-SPLAT-MOTION-HARDEN-001` 使用 targeted Playwright + system Chrome 验证 source splat motion；最新 full audit source motion 段已越过，失败点记录为 `scripts/audit-world-viewer.mjs:595`，不作为本切片完成门槛 | 将 full audit 拆成 source motion、object picking、trainable diagnostics、import routes 等独立脚本，或更新旧 trainable probe / stability 等待条件与当前 UI 一致 | open |
| R-015 | supervised assignment CE clip 边界梯度可放大到约 `1e8` | early training 中零质量 slot 遇到正 target 时可能造成训练发散或 NaN，尤其在 mixed precision / AMP 环境下风险更高 | 已通过审查日志验证和 runtime probe 复现；已登记 `ACTION-026` | clip 边界梯度定义修正并有单元测试覆盖 finite gradient / zero assignment 正 target 负路径 | open |
| R-016 | v2 stability gate 无 prediction 时可回退 oracle 并 pass | solver 未真正运行时，synthetic stability gate 可能给出误导性绿灯；assignment slot 数不匹配也缺少输入层 fail-fast | `OBJECTSTATE-IDENTITY-GATE-001` 已修复：diagnostics / gate 无显式 prediction 时 fail-fast；`predicted_assignments` 列数必须等于 fixture slot 数；负向测试覆盖 no-prediction 和 slot-count mismatch | gate 要求显式 prediction 或无 prediction fail；slot-count mismatch fail fast；负向测试覆盖 | closed |
| R-017 | ObjectState 可能只是 observation state，而不是 world state | 如果 `ObjectState` 只跟随可见 Gaussian / mask support，遮挡、视角变化或对象重新出现时会发生 identity fragmentation，ObjGauss 会停留在 Gaussian segmentation 工具层，不能支撑 world-model claim | 已冻结 `docs/architecture/objectstate-state-variable-gate.md`；`OBJECTSTATE-IDENTITY-GATE-001` 已落地 synthetic identity smoke evaluator；`OBJECTSTATE-IDENTITY-MODEL-001` 已补 contrastive identity encoder training summary；`OBJECTSTATE-PREDICTIVE-GATE-001` 已补 synthetic state-vs-history predictive sufficiency smoke gate；`OBJECTSTATE-CAUSAL-GATE-001` 已补 synthetic controlled action gate；`OBJECTSTATE-REALITY-GATE-001` 已补 controlled real / public row contract 与 pass / fail / blocked 分离；`OBJECTSTATE-REALITY-PUBLIC-ROWS-001` 已把当前 public artifacts 生成 12 条 blocked rows，明确 `object_id` 不是 physical identity GT；`OBJECTSTATE-CONTROLLED-REAL-ROWS-001` 已补 controlled real manifest importer，可让带 timestamped identity GT 的 row 进入 pass/fail；`OBJECTSTATE-CONTROLLED-REAL-CLI-001` 已把 importer 暴露成 `objgauss object-state controlled-real-gate`，可写 summary JSON 和 blocked rows Markdown，并支持 identity-only Stage 1 gate；`OBJECTSTATE-CONTROLLED-CAPTURE-MANIFEST-001` 已补 frame-level capture / annotation manifest validator 和 `validate-controlled-capture` CLI，可验证 RGB / Gaussian refs、timestamp、object_id、6DoF pose 和 action readiness，并支持可选 `condition.view_id`、`condition.lighting_id` 和 `condition.camera_pose`；`OBJECTSTATE-CONTROLLED-CAPTURE-FILE-AUDIT-001` 已补 `audit-controlled-capture-files` CLI，可检查 capture manifest 引用的 RGB / Gaussian 文件是否真实存在于本地 bundle，并已强化为 frame refs 非空 regular files、格式签名审计、per-kind valid counts、完整 `file_records` 和可选 SHA256；`OBJECTSTATE-CONTROLLED-IDENTITY-EVAL-001` 已补 candidate identity track evaluator 和 `eval-controlled-identity` CLI，可计算 `idf1` / fragmentation / swap / collapse 并输出 identity pass/fail row；`OBJECTSTATE-CONTROLLED-PREDICTION-EVAL-001` 已补 candidate future-pose evaluator 和 `eval-controlled-prediction` CLI，可计算 `state_ade` / `history_ade` / `prediction_gap_vs_history_model` 并输出 prediction pass/fail row；`OBJECTSTATE-CONTROLLED-INTERVENTION-EVAL-001` 已补 candidate action-conditioned evaluator 和 `eval-controlled-intervention` CLI，可计算 `action_conditioned_ade` / `counterfactual_outcome_accuracy` / `wrong_direction_rate` 并输出 intervention pass/fail row；`OBJECTSTATE-CONTROLLED-REALITY-BUNDLE-HANDOFF-001` 已补 full Phase 1 handoff，可从真实 bundle root、trainable ObjectState artifact、prediction candidates 和 intervention candidates 一次性生成 merged controlled-real manifest 与 full reality gate；`OBJECTSTATE-CONTROLLED-REALITY-BUNDLE-READINESS-001` 已补 full handoff preflight，可在运行 handoff 前检查 bundle、trainable artifact、prediction candidates 和 intervention candidates 的 schema 与 capture binding；`OBJECTSTATE-CONTROLLED-REALITY-CANDIDATE-TEMPLATE-001` 已补 `init-controlled-reality-candidates` CLI，可从已填写 bundle 生成 draft-only prediction / intervention candidate templates，且模板 schema 会被正式 evaluator 拒绝，避免 TODO 模板被误当作 pass evidence；`OBJECTSTATE-IDENTITY-PREDICTION-ADAPTER-001` 已补 `export-identity-predictions` CLI，可把 trainable kernel `object_states` 经 capture pose association 转成 evaluator 输入；`OBJECTSTATE-CONTROLLED-IDENTITY-HANDOFF-001` 已补 `controlled-identity-handoff` CLI，可一次性生成 capture file audit、candidate artifact file audit、identity scenario audit、predictions、identity eval、controlled-real manifest、identity-only gate summary 和 blocked rows markdown，且 handoff pass 现在要求 capture file audit、candidate artifact file audit、candidate artifact ref match、clear-visible / occluded / clear-visible identity scenario、declared view / lighting coverage 与 camera-motion metadata 同时通过；`OBJECTSTATE-BOP-IDENTITY-HANDOFF-001` 已补 `bop-identity-handoff` CLI，可把 local BOP acceptance、finalized ObjectState artifact、identity eval、identity evidence package 和 Phase 1 ledger 串成 reviewable Stage 1 identity evidence，且 reviewable 与 metric pass 分离；`OBJECTSTATE-BOP-LOCAL-ROW-HANDOFF-001` 已补 `bop-local-row-handoff` CLI，可把同一 BOP sample 的 identity package 与 prediction baseline package 合成 `identity_prediction_reviewable` ledger，同时明确 intervention 不在 BOP pose route 中声明；`OBJECTSTATE-BOP-CROSS-SAMPLE-LEDGER-001` 已补 `audit-bop-cross-sample-ledger` CLI，可把多个 BOP local-row summary 汇总为 cross-sample identity+prediction reviewability 表，并把 sample / scene / category / scenario 覆盖和 candidate gate 缺口显性化，同时不把 metric pass 或 BOP pose route 伪装为 intervention / world-model evidence。当前仍缺实际 controlled real capture 文件以及真实 intervention candidate 文件作为完整 Phase 1 通过证据。 | state-variable gate 接入实际 controlled real identity / prediction / intervention rows，至少 identity、prediction 和 intervention rows 从真实采集 manifest 进入可评估 pass/fail，identity collapse 没有被误报为 pass，blocked rows 和 open-world failures 与 pass rows 分离 | open |

R-017 update 2026-07-08: `OBJECTSTATE-BOP-LOCAL-ROW-BATCH-HANDOFF-001`
新增 `bop-local-row-batch-handoff`，可从显式 batch spec 连续运行多个本地
BOP local-row handoff，并自动生成 cross-sample ledger / Markdown table；该缓解
仍只编排本地已有 scene、Gaussian evidence 和 candidate artifact，不创建 GT、不训练
模型、不声明 intervention / world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-LOCAL-ROW-BATCH-READINESS-001`
新增 `audit-bop-local-row-batch-readiness`，可在 batch handoff 前复用每个
BOP sample 的 local-row readiness，显式报告 Gaussian evidence、candidate artifact
binding、identity scenario metadata、ready/reviewable sample count 和
scene/category/scenario coverage 缺口；该缓解仍是 read-only preflight，不运行 handoff、
不创建 GT、不训练模型、不声明 metric pass、intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-LOCAL-ROW-BATCH-SPEC-AUTHORING-001`
新增 `init-bop-local-row-batch-spec`，可从本地 BOP sample CSV 写出原生 batch spec，
并检查 scene root、candidate artifact 和 declared condition sidecar 路径缺口；该缓解
只减少手写 batch spec 和路径漂移风险，不运行 readiness / handoff、不创建 GT、不重建
Gaussian、不训练模型、不声明 metric pass、intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-BATCH-CSV-TEMPLATE-001`
扩展 `select-bop-phase1-subset`，可把 ready BOP scene 写成
`init-bop-local-row-batch-spec` 可消费的 CSV 模板，并指向 expected candidate artifact /
condition sidecar 本地路径；该缓解只减少从 selector 到 batch authoring 的人工错漏，不创建
candidate artifact 或 sidecar，不运行 readiness / handoff、不创建 GT、不训练模型、不声明
metric pass、intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-PHASE1-BATCH-WORKSPACE-001`
新增 `init-bop-phase1-batch-workspace`，可从本地 BOP dataset / split root 初始化
selector summary、samples CSV、native batch spec、batch spec authoring summary 和
README / next commands；该缓解只把本地 BOP subset 到 batch readiness 的 authoring
路径收敛成可复跑 workspace，不创建 candidate artifact、condition sidecar、Gaussian
evidence 或 GT，不运行 readiness / handoff、不训练模型、不声明 metric pass、
intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-PHASE1-SAMPLE-WORKSPACES-001`
新增 `init-bop-phase1-sample-workspaces`，可从 native BOP local-row batch spec 为每个
sample 写出 condition CSV 模板、condition sidecar draft 和 README / next commands，
帮助作者填真实 `bop-condition-sidecar.json`、per-frame Gaussian evidence 和
`objectstates.json`；该缓解只初始化 per-sample authoring helper，不创建真实 target
sidecar、candidate artifact、Gaussian evidence 或 GT，不运行 readiness / handoff、不训练
模型、不声明 metric pass、intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-PHASE1-AUTHORING-PROGRESS-001`
新增 `audit-bop-phase1-authoring-progress`，可从 native BOP local-row batch spec 只读检查
每个 sample 的 helper files、target `bop-condition-sidecar.json`、per-frame
`gaussians/<frame>.ply` 和 target `objectstates.json` 是否已经填到可进入 batch readiness
input；该缓解只减少 sample authoring 到 batch readiness 之间的漏填 / 路径漂移风险，不创建
target files、不生成 Gaussian、不运行 readiness / handoff、不训练模型、不声明 metric pass、
intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-BASELINE-CANDIDATE-001`
新增 `generate-bop-objectstate-baseline-candidate`，可在本地 BOP scene 已有 per-frame
Gaussian evidence 时写出单个全局 Gaussian centroid / bbox 的 trainable ObjectState
baseline artifact，减少 sample authoring 卡在缺 `objectstates.json` 的风险；该 artifact
预期作为可审阅负证据候选，不读取 BOP pose GT / object ids 来放置预测，不训练模型、不运行
handoff、不创建 pass row，也不声明 intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-BASELINE-LOCAL-ROW-HANDOFF-001`
新增 `bop-baseline-local-row-handoff`，可在本地 BOP scene 和 per-frame Gaussian evidence
已存在时，一次生成 single-state baseline `objectstates.json`，再运行 identity handoff、
prediction baseline handoff 和 Phase 1 ledger；该缓解让真实 BOP subset 可先产出
reviewable negative evidence，但仍不下载数据、不创建 GT、不重建 Gaussian、不训练模型、
不声明 metric pass、intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-RGBD-BASELINE-LOCAL-ROW-HANDOFF-001`
新增 `bop-rgbd-baseline-local-row-handoff`，可在本地 BOP RGB-D scene 已存在时，
先从 `depth/<frame>.png` 反投影写 per-frame `gaussians/<frame>.ply` evidence seed，
再生成 single-state baseline `objectstates.json` 并运行 identity / prediction local-row
package；该缓解减少真实 BOP subset 从 RGB-D 到 reviewable negative evidence 的手工链路，
但仍不下载数据、不创建 GT、不使用 pose GT 生成 candidate prediction、不运行 Splatfacto /
optimized 3DGS reconstruction、不训练模型、不声明 metric pass、intervention gate 或
world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-LMO-PUBLIC-ROW-001`
使用官方 BOP LMO public RGB-D scene 生成第一条真实 public baseline evidence row。本地
ignored zip 为 `outputs/assets/raw/bop-lmo/lmo_test_bop19.zip`，大小 `117550985`
bytes，SHA256 为
`42d7a15f317476ca3980ee7ec0344b691cbadc796835f0b14f72c89a1dcec421`；抽取
`test/000002` 的 `000003` / `000008` / `000017` 三帧后，prediction evidence
package 已达到 reviewable，Phase 1 ledger maturity 为 `prediction_reviewable`。该 row
同时给出明确负证据：prediction metric fail，identity eval fail 且
`identity_collapse=true`，identity package 因缺 explicit lighting / camera-pose
condition metadata 仍 incomplete。此次还修复相对 `output_root` 下 prediction package
audit 双重拼接 `candidate_dir` 的问题。该缓解只证明 public RGB-D route 可以生成可复验
prediction row 和 identity negative result，不训练模型、不提交数据集或 evidence outputs、
不声明 metric pass、intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-LMO-CONDITION-GAP-001`
对同一 LMO public row 运行 condition sidecar authoring 和 identity route audit，确认
BOP acceptance、per-frame Gaussian evidence、candidate artifact validity 和 frame binding
均 ready，但 identity route 仍因 `lighting_condition_count=1`、`camera_pose_count=0`、
`identity_scenario_metadata_ready=false` 而 blocked。该缓解把 identity blocker 从口头判断
变成可复验 evidence，同时保持风险 open：不能放松 identity scenario gate，不能伪造
lighting / camera-pose sidecar，也不能把该 public BOP row 记为 identity-reviewable、
metric pass、intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-MULTI-INSTANCE-IDENTITY-POLICY-001`
新增显式 `pose_track_per_obj_id` BOP identity import policy，解决 HOPE 这类同一
`obj_id` 多实例 public scene 无法进入 manifest 的问题；默认
`single_instance_per_bop_obj_id` 策略仍 fail-fast。`OBJECTSTATE-BOP-HOPE-PUBLIC-ROW-001`
在 BOP HOPE `val/000001` 三帧上生成 RGB-D Gaussian evidence seed 和 baseline local row：
prediction package reviewable / pass，但 identity handoff 不 reviewable / 不 pass；
`OBJECTSTATE-BOP-HOPE-CONDITION-GAP-001` 进一步证明阻塞来自缺 lighting、camera pose 和
clear occlusion-reappearance metadata，而不是 adapter、Gaussian evidence 或 candidate
binding。该缓解扩大了真实 public multi-instance evidence 覆盖，但风险仍 open：尚无真实
controlled identity pass row、无真实 intervention row，也不能把 BOP pose import policy
误称为 ObjectState 已经是 world state。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-REALITY-ROWS-001`
新增 `audit-bop-reality-rows`，可把现有 BOP local-row summary 转成
`OBJECTSTATE-REALITY-GATE-001` 的 `public_replay` rows，并保留 pass / fail / blocked
状态。当前 HOPE rows 为 identity fail / prediction pass / intervention blocked，LMO rows
为 identity fail / prediction fail / intervention blocked；两个 full gate 都保持 fail。
该缓解把 public evidence 纳入 state-variable gate accounting，但不创建 GT、不训练模型、
不声明 identity pass、intervention gate 或 world-model evidence。

R-017 update 2026-07-08: `OBJECTSTATE-REALITY-ROW-LEDGER-001`
新增 `audit-reality-row-ledger`，可把多个 reality row summaries 聚合成一个全局
`OBJECTSTATE-REALITY-GATE-001` report。当前 LMO + HOPE 总表为 2 summaries / 6 rows /
1 pass / 3 fail / 2 blocked，missing pass evidence kinds 为 identity 和 intervention；
full gate 仍 fail。该缓解让 Phase 1 缺口可跨样本审计，但风险继续 open：还需要真实
controlled identity pass row 和 action-conditioned intervention pass row。

R-017 update 2026-07-08: `OBJECTSTATE-REALITY-ROW-LEDGER-NEXT-ACTIONS-001`
扩展 `audit-reality-row-ledger`，在 summary 中新增机器可读 `next_actions` 和
operator-facing `next_actions_markdown`。当前 LMO + HOPE ledger 仍 fail，但 CLI 会明确输出
两个 P0 缺口：`identity -> controlled_real_identity_handoff` 和
`intervention -> controlled_reality_bundle_handoff`，并把所需 GT、指标、命令链和 claim
boundary 一起记录。该缓解只做 read-only gap planning，不创建 GT、不采集数据、不运行
eval、不训练模型、不声明 identity / intervention pass 或 world-model evidence；风险继续
open，直到真实 controlled/public rows 形成 identity、prediction 和 intervention pass/fail
证据。

R-017 update 2026-07-08: `OBJECTSTATE-STATE-VARIABLE-MATRIX-001`
扩展 `audit-reality-row-ledger`，把现有 reality rows 映射到五个 State Variable Gate
实验。当前 LMO + HOPE public replay matrix 为：identity persistence fail、occlusion
recovery missing_metric、view invariance missing_metric、predictive sufficiency pass、
counterfactual / action interface blocked。该缓解防止把单个 prediction pass row 误读为
状态变量通过，也把下一步真实证据缺口从三类 row 拆到五类实验；风险继续 open，因为仍缺
真实 identity pass、occlusion recovery / view invariance metrics 和 action-conditioned
counterfactual row。

R-017 update 2026-07-08: `OBJECTSTATE-BOP-SCENARIO-CHALLENGE-METRICS-001`
扩展 BOP reality rows 和 ledger matrix，把 BOP identity scenario audit 的 challenge
coverage 与模型指标分开记录。当前 LMO 提供 occlusion reappearance challenge，LMO / HOPE
提供 multi-view challenge，BOP action challenge 仍 absent；但这些只说明 public replay
含部分测试场景，不等于 `occlusion_recovery_rate`、`contrastive_margin` 或 counterfactual
metric pass。风险继续 open：identity collapse 仍是真实 fail evidence，且真实 controlled
identity pass / view-invariant metric / action-conditioned intervention row 仍缺。
