# ObjGauss 行动队列

> 最近更新: 2026-07-10
> 只保留 active、frozen/planned 与最近完成项；历史见 archive。

## Active

### CONTROLLED-REAL-3SCENE-001: 取得真实动作证据

- 需要至少 3 个真实 controlled scenes，具备 physical identity、timestamped 6DoF pose、
  occlusion、view change 与测量得到的非零 action。
- 当前三段 BOP public replay 仅能提供 RGB-D/pose 负证据；identity/prediction gate 未通过，
  且没有 intervention action GT。
- RBO/RRC 首批子集已下载并完成字段审计。RBO official-chain strict V-O-V 为 `0/3`；
  measured-force target/interval 仅留下 tripod/cabinet 两个 ready scenes。follow-up P0 也未
  新增 scene；重建 ledger 为 `0 pass / 7 fail / 2 blocked`。RRC 固定相机、无 depth，
  9D control 不能冒充现行 3D vector。
- 下一步下载并逐帧审计 P1 `treasurebox24/tripod24/pliers24/ikeasmall23`，继续寻找第三个
  official-chain action-ready 且有严格 V-O-V 的 scene；然后重跑 canonical bundle/ledger。
  不要在聊天中发送凭据，也不要新增 wrapper、复制 target GT 或伪造 action。

### CORE-BOUNDARY-001: 继续外移 core orchestration

- 已迁出 controlled schema/capture/real-manifest/action-GT、四个 evaluator/gate、
  controlled real rows 与 identity prediction adapter 的 canonical 实现。
- 已迁出 capture template/frames/annotations/actions/files/environment/import/readiness；
  capture handoff 已迁入 `objgauss.pipelines`。
- BOP capture adapter、local-row batch spec/authoring、subset selector 与两级 workspace
  已迁入 `objgauss.datasets`。
- BOP identity/prediction、local-row、baseline、RGB-D 与 batch handoff 已迁入 pipelines；
  route audit、两级 readiness 与 controlled identity/prediction/reality package 链也已迁入。
- transition chain、phase1 ledger、temporal/artifact/long-smoke chain 与 identity benchmark/
  leakage audit/gate/ablation、BOP candidate authoring 已迁出。
- public dataset/workspace、real evidence bundle 数据合同、BOP/real/public reality-row
  adapters 与聚合 ledger 已迁出。
- projection summary 已归入 ObjectState primitive，evaluation 不再引用 assignment MVP
  私有 helper。
- BOP/controlled-real bundle adapter、real bundle ledger/audit 与 controlled-real readiness/
  identity/prediction eval 已迁出。
- assignment MVP/train/generalization/ablation、BOP Gaussian preflight、baseline comparison 与
  ObjectState benchmark 已迁出。
- v2 synthetic foundation、diagnostics/stability/identity/predictive/causal gates 与 identity
  encoder 已迁出；assignment checkpoint/evaluator 已按 primitive/evaluation 拆分。
- training scale/TensorBoard、quality/core-model reports、emergence 与 10 模块 real-sample-v2
  workflow 已迁出。
- trainable kernel、CPU/gsplat training renderer、Gaussian/solver-decoder training、assignment-
  renderer validation 与 renderer-loss report 已归 pipelines；ObjectState checkpoint eval 与
  assignment stability 已归 evaluation。
- mask manifest/SAM adapter 与 teacher-evidence contract 已归 datasets；CLIP scoring 与
  semantic slot alignment 已归 pipelines。
- ObjectField 邻接的 NeRF inspection 已归 datasets，JSON output helper 已归 pipelines；
  core primitive import 不会加载外层包，旧名称仅惰性兼容。
- projection/loss/quality audit primitive 留 core；manifest voting/training/depth diagnostic
  已归 pipelines，acceptance check 已归 evaluation，core-only import 不加载外层包。
- 当前冻结期不再做 ownership 搬迁；Gaussian/assignment/ObjectState/decoder/PLY 与 metric
  primitives 保留 core，混合 shared ABI 的 `object_emergence_solver` 不做整文件搬迁。
- 冻结期不新增 `objgauss.core` root alias；32 个同名 alias 的退役留到明确 breaking window。
- 后续只按 domain 小切片迁移，不做巨型搬迁。

## Frozen / Planned After Stabilization

- `ACTION-006`: 改进真实 SAM/CLIP foreground coverage；当前 promotion 仍是 do-not-promote。
- `ACTION-004`: Poly Haven mesh 到 3DGS demo 转换链；冻结期不扩 Viewer/demo 产品面。

## Recently Completed

### ACTION-026: 修复 supervised assignment CE clip 边界梯度

- clip plateau 的导数与 `log(clip(...))` forward 一致；zero assignment + positive target
  不再产生约 `1e8` 梯度。
- 覆盖 finite gradient、clipped plateau 与正常 non-clipped 路径；进入 required Python CI。
- 已包含在 research-first stabilization commit `08ec78d`。

## Archive

- `docs/state/archive/action-queue-through-2026-07-09.md`
- `docs/state/archive/pr-queue-through-2026-07-09.md`
