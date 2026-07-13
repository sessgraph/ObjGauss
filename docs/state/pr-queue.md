# ObjGauss 当前任务队列

> 最近更新: 2026-07-13
> 只保留 active、blocked、planned 和最近完成项；完整历史已归档。

## In Progress

### CORE-BOUNDARY-001: Move orchestration out of core

- 已完成 foundation：controlled schema、capture contract、real manifest 与 intervention
  action-GT readiness 迁至 `objgauss.datasets`；identity、prediction、intervention evaluator、
  reality gate、controlled real rows 与 identity prediction adapter 迁至
  `objgauss.evaluation`。
- 已完成 controlled authoring：capture template、frames、annotations、actions、files、
  environment、import、bundle readiness 全部迁至 `objgauss.datasets`。
- 已建立 `objgauss.pipelines`，controlled capture handoff 已迁入；旧 core 路径仅兼容。
- BOP capture adapter、local-row batch spec/authoring、subset selector 与 batch/sample
  workspace 已迁入 `objgauss.datasets`；legacy public surface 受显式集合与对象身份测试保护。
- BOP identity、prediction-baseline、local-row、baseline local-row、RGB-D local-row 与
  batch handoff 已迁入 `objgauss.pipelines`；identity/prediction route audit 与 local/batch
  readiness 也已迁入；controlled identity/prediction/reality 的 candidate/bundle/package
  链、transition chain、phase1 ledger、temporal runner、trainable artifact/adapter 与
  long-smoke/contract 与 BOP candidate authoring 已迁入；benchmark/report/leakage audit/
  identity gate/ablation 已迁入 evaluation。public dataset candidates/workspace 已迁入
  pipelines；real evidence bundle 数据合同已迁入 datasets；BOP/real/public reality-row
  adapters 与聚合 ledger 已迁入 evaluation；BOP bundle、real bundle ledger/audit 已迁入
  pipelines；controlled-real bundle adapter 已迁入 datasets，readiness/identity/prediction
  eval 已迁入 evaluation；assignment MVP/train/generalization/ablation 与 BOP Gaussian
  preflight 已迁入 pipelines，baseline comparison 与 ObjectState benchmark 已迁入 evaluation；
  v2 synthetic foundation/gate chain 与 identity encoder 已迁出，assignment checkpoint/eval
  已拆回 solver primitive 与 canonical evaluation；training helpers/emergence 与完整
  real-sample-v2 workflow 已迁出；trainable kernel、CPU/gsplat training renderer、Gaussian/
  solver-decoder training、assignment-renderer validation 与 renderer-loss report 已归
  pipelines，ObjectState checkpoint eval 与 assignment stability 已归 evaluation；mask
  manifest/SAM adapter 与 teacher-evidence contract 已归 datasets，CLIP scoring 与 semantic
  slot alignment 已归 pipelines；ObjectField 邻接的 NeRF inspection/JSON output 已拆至
  datasets/pipelines；projection/loss/quality audit primitive 留 core，manifest voting/
  training/depth diagnostic 已归 pipelines，acceptance check 已归 evaluation。
- canonical `datasets` 仅依赖允许的 core assignment primitive；`evaluation`/`pipelines`
  只依赖保留 primitives；projection summary 已归入 `object_state` primitive，不再跨层引用
  assignment MVP 私有 helper。
- 旧 `objgauss.core` 路径只做显式、对象身份不变的 compatibility import。
- 当前冻结期无剩余可无 schema/ABI 变更整迁的 orchestration；`object_emergence_solver`
  作为 shared assignment primitive 暂留 core。下一阶段只在明确 breaking/deprecation
  window 收缩 compatibility modules 与 root exports。
- 保留目标：Gaussian、assignment、ObjectState、decoder、metric primitives。
- 当前验证：806 个 Python tests；本轮 17 个迁移/拆分模块的 canonical-first / legacy-first
  冷启动、精确导出面、对象身份与逆向依赖检查通过；`objgauss/core/*.py` 为 11,956
  physical LOC（约 12.0k，含 compatibility wrappers）。wheel/sdist 在 714-test 检查点通过；
  最新批次因审批额度耗尽未重跑；Vite production build 已通过。
- core root 仍有 678 个 lazy exports，其中同名 legacy 子模块会制造 32 个 import-order
  ambiguity；冻结期不新增 root export，退役需单独 breaking/deprecation window。

## Blocked External Evidence

### CONTROLLED-REAL-3SCENE-001: Run three real controlled scenes

- 需要: physical identity、timestamped 6DoF pose、occlusion、view change、真实非零 action。
- 已有: 两段 HOPE 与一段 LMO pose/public replay，均不含可用 intervention action；首批三段
  RBO 已形成 physical link identity、6DoF、camera motion 与 measured force 原始链路。
- 当前结果: official playback-chain full-frame visibility 仍为 `0/3`；三个重建 bundle 的
  capture/file validation 通过，但 action GT 只覆盖 tripod/cabinet 两段，cardboard action
  明确缺失。旧 direct-marker `0 pass / 9 fail / 0 blocked` ledger 已 supersede，新 canonical
  ledger 为 `0 pass / 7 fail / 2 blocked`；单一真实 lighting 同样保持 full readiness blocked。
- 缺口: 合格 scene 仍为 `0/3`；仍缺严格 occlusion/reappearance、第三个 official-chain
  action-ready scene，以及超过简单 baseline 的 candidate。
- 已核验: H2O/HOI4D/HOT3D/DexYCB 等公开候选可支持部分 identity/prediction，
  但都不原生提供现行合同要求的独立非零 3D action/control vector；当前 host 也无 capture
  设备与重建工具链。2026-07-13 追加核验 YCBInEOAT 与 RH20T：前者有真实 RGB-D、逐帧
  object 6DoF GT、显著遮挡与 robot FK，但相机静态，dataset 许可和独立 action provenance
  未明确；
  后者有多视角 RGB-D、TCP/command 与 6DoF F/T，却没有逐对象 6DoF GT，且最小完整
  depth 分卷约 71.3 GB。追加筛查 ObjectInHand/VinT-6D/Robo360：分别缺 controller
  action 与 moving camera、episode action contract、逐对象 6DoF GT。上述来源当前都不能
  直接解锁本任务。
- acquisition/semantic audit: RBO/RRC 官方索引及各 3 条最小候选已于 2026-07-10 本地
  下载并完成完整性与字段语义审计。RBO 三段有同步 RGB-D、逐 link 6DoF、相机运动和
  wrench；按 base marker + 同刻 joint state + matching URDF 的严格全帧 mesh/depth 重算为
  `0/3` V-O-V，最大遮挡为 `0.421140/0.467351/0.476498`。ATI sign 已解决；official-chain
  target 重算接受 `5/8` windows、`3/7` 50 Hz intervals，只覆盖 2 个 scenes；
  RRC 三段有真实 9D desired/applied action 与 tracker pose，但固定相机、无 depth，不能
  直接进入现行 3D action contract。
- follow-up acquisition: `scripts/download-rbo-occlusion-followup.sh` 固定 P0
  `treasurebox25/laptop26/globe25` 与官方 `ftSensor` model。P0 已完整下载并逐帧复核：
  treasurebox/globe 均为可信 `0` V-O-V；laptop camera calibration 无效且
  `event_count=null`。P1 的 `treasurebox24/tripod24/pliers24/ikeasmall23` 与 models
  已于 2026-07-13 完成全量完整性校验、official-chain preflight 和同阈值逐帧重算；
  接受 `309/358/368/257` 帧，四段均为 `0` V-O-V，独立重算 `3,199/3,199` 个逐 link
  CSV 样本一致。因此 P1 是可信 `0/4` 负证据，没有进入 action-target 复核，也没有新增
  合格 scene。全量 19 段 camera-motion + F/T 库存对账后，新增 P2 只含 `globe23`
  （`393,459,489` bytes）：它与已验证 `globe25` 同 object、同 `2017-07-18` marker-set
  date，补齐 artificial/natural lighting 配对。通用 official-chain audit 已与专用 globe
  calibration 在 `746/746` 个样本上零差异；P2 archive 完整性通过后实际接受 `640/670`
  个 RGB 帧，`rb0/rb1` 最大遮挡为 `0.392906/0.007348`，两 link 均为 `0` V-O-V；独立
  CSV 复核 `1,280/1,280` 个逐 link 样本一致。因此双 lighting 配对仍未解锁 scene，
  不进入 action-target 复核。
- 禁止: 新增 wrapper、伪造 action、复制 target GT、把 fixture 标成 real。
- 路线决定: RBO eligibility audit 已完成并停止继续投入；其工具与负证据保留。Phase M
  闭环后优先自建 controlled capture，不再从已审计 RBO route 挖字段或修 adapter。
- 解锁条件: 新 capture 或新数据源提供至少 3 个许可明确、严格 V-O-V 可复核且同时含
  RGB-D/6DoF/view change/measured action 的 scene，并让 candidate 与简单 baseline 完成比较。

## Planned After Stabilization

### REAL-EVIDENCE-BASELINES-001: Compare on held-out real scenes

- identity 必须超过不使用 GT association 的简单 baseline。
- prediction 必须超过 hold-last、constant velocity / Kalman。
- intervention 必须超过 no-action baseline。

## Recently Completed

- 2026-07-13: `OBJECTSTATE-MODEL-DIAGNOSTIC-001` 实现完成。新增原始 M2 reproduction
  与 13-observation hard-case cohort，对 XYZ、RGB、XYZ+RGB、XYZ+RGB+class-semantic 做
  固定 split、3-seed 输入消融，并输出 merge/split/frozen-map identity swap taxonomy、
  ablation/hard-case/error CSV、双数据集 manifest 和 portable technical report。原始 M2
  native seed 0 精确保持 `0.754572` 对 connected-components `0.825829`；M2 最佳变体
  `0.758923` 仍未胜基线。hard cases 最佳变体 `0.877924` 虽高于 `0.786994`，跨视角
  swap rate 仍为 `0.5`；semantic proxy 相对 XYZ+RGB 仅 `+0.006625`。结论保持
  `no_model_variant_beats_recorded_segmentation_baselines`，不进入 Transformer/长训练。
- 2026-07-13: `OBJECTSTATE-MODEL-DEMO-001` 实现完成。M1 NeRF Lego
  的 `1.0` ARI/IoU/Purity 只作为单物体四色 wiring smoke，不再描述成对象分割效果。
  M2 新增 12-scene 程序化四物体数据、8/4 完整 scene holdout、独立实例 GT、同色同形
  对象、接触/局部观测、XYZ/RGB KMeans、3D connected-components 和 Model v0；报告
  Hungarian mIoU/ARI/count/merge/split/recall。leakage gate 通过，Model v0 mIoU
  `0.754572` 低于最佳基线 `0.825829`，verdict 为 `model_not_better_than_recorded_baselines`。
  Browser checkpoint inference 与 467-row prediction 一致；Raw/Prediction/GT、指标、对象
  hide/show 与 Spark stage isolation 通过专项 Playwright 和截图审查。820 tests、build、M1
  专项与 viewer truth audit 通过；完整串行 world audit 本轮 7/9 flow 后受外部 SIGTERM。
- 2026-07-11: `95dfe2a`；将 real-sample-v2 required-CI 覆盖迁到 license-safe、固定 seed
  的程序化边界场景；保留 temperature sharpening、coverage、hard fix/regression、
  no-safe-candidate 与 CLI/PLY roundtrip 分支，clean archive 全量 806 tests 通过。
- 2026-07-11: `3acfe73`；对齐 world viewer audit 与 5-pill evidence UI、lazy artifact
  flows、translate-only/Stability 合同；full/truth 浏览器审计通过。
- 2026-07-10: `506b793`；约束 controlled identity association 的可接受来源与匹配边界。
- 2026-07-10: `ca22c71`；从 canonical evaluator 输出独立重算 controlled reality handoff metrics。
- 2026-07-10: `21f2744`；阻止 GT-preassociated identity 输入形成通过结果。
- 2026-07-10: `b054e96`；保留 probability-space CE 导数，并为训练消费者融合有界的
  analytic softmax-logit VJP。
- 2026-07-10: `c60b069`；强制执行 ObjectState trainable artifact ABI。
- 2026-07-10: `STABILIZE-CORE-001`；CE clip gradient 与 ObjectState ABI 修复，行为测试覆盖。
- 2026-07-10: `STABILIZE-VIEWER-001`；catalog lazy load、Spark hide/translate truth 与 UI 收口。
- 2026-07-10: `STABILIZE-GATES-001`；teacher content binding、raw identity association、
  reality status/gap/gain 独立派生。
- 2026-07-10: `STABILIZE-DELIVERY-001`；required CI、保守 LICENSE、版本/依赖锁。
- 2026-07-10: `STATE-SLIM-001`；active state 归档减量并启动 datasets/evaluation 边界迁移。
- 2026-07-10: public interaction correctness；纠正 HOT3D action GT，解除 identity/prediction
  对 action 声明的错误依赖，并阻止 fixture intervention 转成 public pass。
- 2026-07-10: RBO/RRC 首批 archive 字段语义审计；确认 RBO 严格 V-O-V `0/3` 与 action
  GT blocker，确认 RRC 固定相机/无 depth/9D action 边界，并冻结 RBO occlusion follow-up。
- 2026-07-10: RBO official playback-chain visibility 与 action-target 复核；首批仍为
  `0/3` V-O-V，action GT 收缩为 3 intervals / 2 scenes；official bundle/ledger 重建为
  `0 pass / 7 fail / 2 blocked`，旧 direct-marker 结果已 supersede；follow-up P0 未新增 scene。
- 2026-07-09: controlled real prediction evaluator plumbing。
- 2026-07-09: controlled real identity evaluator plumbing。

这些完成项只表示工具链存在，不表示 real metric gate 或 world-model claim 成立。

## Archive

- `docs/state/archive/pr-queue-through-2026-07-09.md`
- `docs/state/archive/project-status-through-2026-07-09.md`
