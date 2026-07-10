# ObjGauss 当前状态总览

> 最近更新: 2026-07-10
> 阶段: research-first stabilization
> 决策: `docs/adr/0007-research-first-stabilization.md`

## 一句话状态

ObjGauss 是可运行的 Gaussian object-slot 研究与证据平台；当前尚未证明
`ObjectState` 是持续的真实世界状态，也不是 production-ready 对象编辑产品。

## 当前唯一主线

2026-07-10 至 2026-07-24 冻结新 schema、audit、handoff、renderer route 和 Viewer
产品功能。现阶段只允许：

- 修 correctness、数值稳定、ABI、虚假交互和复现问题；
- 让既有 gate 从原始预测和 GT 独立计算；
- 使用既有工具取得至少 3 个真实 controlled scenes 的结果；
- 做状态与接口减法。

Viewer 暂时是 evidence viewer，不是产品扩张主线。

## 已验证能力

- Gaussian PLY / `.splat` IO、对象色、过滤与 object-aware PLY。
- `A[N,K]` soft assignment、ObjectState pooling、训练 smoke、checkpoint roundtrip。
- Spark 真实 splat 展示与 Three.js 对象证据层；点数一致时支持 source splat 子集平移。
- controlled capture / BOP public replay 的现有数据合同、evaluator 和 ledger 工具链。
- 真实 3DGS / SAM / CLIP / gsplat 本地实验产物，但重依赖仍不是默认可复现环境。
- 当前工作树本地基线：778 个 Python 测试与 Vite production build 通过；锁文件检查在
  此前 Viewer/lock 检查点通过。Python wheel/sdist 在 714-test 检查点通过；最新 pipeline
  批次因审批额度耗尽未重跑打包。

## 当前真实负证据

2026-07-10 使用现有命令直接重跑三段 BOP public replay：HOPE scene 000001、
LMO scene 000002、HOPE scene 000002。每段各产生 identity、prediction、intervention
三行，当前直接结果为：

- rows: 9；pass: 0；fail: 6；blocked: 3；
- identity: 三段均 fail；prediction: 三段均由 reality gate 派生为 fail；
- intervention: 三段均因没有真实 action GT 而 blocked；
- full reality gate: 三段均 fail。

本地聚合事实源为
`outputs/evidence/objectstate-bop-3scene-current-ledger/reality-row-ledger-summary.json`
（ignored evidence output，不提交仓库）。

HOPE 的局部 prediction evaluator 会在 `state_ade == history_ade == 0` 时给出旧式 pass，
但 reality gate 要求严格优于 history baseline，因此独立派生为 fail。Synthetic、fixture、
manual teacher evidence 和 schema/reviewability 状态均不计作 real gate pass。

RBO `cardboardbox22`、`tripod25`、`cabinet20` 已完成实际字段与像素级审计。三段均有
同步 RGB-D、逐刚体 MoCap 6DoF、`0.109–0.173m` camera displacement 和非零 100 Hz
wrench；但使用官方 mesh、完整 camera TF、registered depth 独立重算后，严格
clear → `occlusion_fraction >= 0.5` → clear 为 `0/3`。最大严格遮挡比例分别为
`0.369 / 0.167 / 0.461`，不能手写成 pass。Wrench contact windows 可检测，但 sensor sign
与 target-link attribution 未确认，action GT 继续 blocked。本地事实源为 ignored
`outputs/evidence/rbo-objectstate-3scene/`。

## P0 稳定化结果

已完成：

1. supervised assignment CE 在 clip plateau 的导数与 forward 一致，不再产生约 `1e8` 梯度；
2. ObjectState persistent `id`、renderer `slot/object_id` 与 `[0,1]` confidence ABI 已分离；
3. Viewer catalog 按 stage/selection 加载，Spark hide/translate 与 source splat 同步，未闭环控件已移除；
4. 最小 CI、保守 `LICENSE`、Python/Node 版本与依赖锁已落地；
5. teacher audit 绑定实际 feature 内容；identity evaluator 从 raw track observations 关联，
   不再用 GT pose 预关联；gate 不信任 caller status/gap/gain；
6. active state 已压缩；controlled schema/capture/real manifest/action-GT，以及 capture
   template/frames/annotations/actions/files/environment/import/readiness authoring 已移至
   `objgauss.datasets`；四个 evaluator/gate、controlled real rows 与 identity prediction
   adapter 移至 `objgauss.evaluation`；BOP capture adapter 与 local-row batch spec 已移至
   `objgauss.datasets`；BOP batch authoring、subset selector 与两级 workspace 也已移至
   `objgauss.datasets`；controlled capture、BOP identity/prediction、local-row、baseline、
   RGB-D、batch handoff、两条 route audit 与两级 readiness，以及 identity/prediction/
   reality 的 handoff、baseline、candidate template、bundle 与 evidence package，transition
   candidates/handoff/package、phase1 ledger、artifact adapter/contract、temporal runner 与
   long-smoke/contract 已移至 `objgauss.pipelines`。transition dataset 已移至
   `objgauss.datasets`；identity benchmark/report 与 leakage audit 已移至
   `objgauss.evaluation`；model identity gate/ablation 与 benchmark/report 也已归 evaluation，
   BOP candidate template/authoring progress、public dataset candidates/workspace 已归
   pipelines；real evidence bundle 数据合同已归 datasets；BOP、真实 bundle、公开 artifact
   的 reality-row adapter 与聚合 ledger 已归 evaluation；projection summary 已提升为
   ObjectState primitive，不再从 assignment MVP 借用私有 helper；BOP/controlled-real bundle
   adapters、real bundle ledger/audit 已归 pipelines/datasets，controlled-real readiness、
   identity/prediction eval 已归 evaluation；assignment MVP/train/generalization/ablation、
   BOP Gaussian preflight 已归 pipelines，baseline comparison 与 ObjectState benchmark 已归
   evaluation；v2 synthetic foundation 已归 datasets，diagnostics 与 stability/identity/
   predictive/causal gates 已归 evaluation，identity encoder 已归 pipelines；assignment
   checkpoint 留在 solver primitive，stability evaluator 已归 evaluation；training scale/
   TensorBoard、quality/core-model report、emergence 与完整 real-sample-v2 workflow 已归
   pipelines/evaluation；trainable kernel、CPU/gsplat training renderer、Gaussian/solver-
   decoder training、assignment-renderer validation 与 renderer-loss report 已归 pipelines，
   ObjectState checkpoint eval 与 assignment stability 已归 evaluation；mask manifest/SAM
   adapter 与 teacher-evidence contract 已归 datasets，CLIP mask scoring 与 semantic slot
   alignment 已归 pipelines；ObjectField 邻接的 NeRF dataset inspection 已归 datasets，
   JSON output helper 已归 pipelines；projection/loss/quality audit primitive 留在 core，
   manifest voting/training/depth diagnostic 已归 pipelines，acceptance check 已归 evaluation。
   旧 core 路径仅保留显式或惰性兼容导入。
7. HOT3D-Clips 的原生 action GT 已纠正为 false；public interaction route 仍允许
   identity/prediction accounting，但 reality-row 转换会把无原生 action GT 的 intervention
   强制保持 blocked，fixture action 不能越权形成 public pass。

仍未完成：真实 action/counterfactual evidence。当前冻结期可在不改 schema/ABI 的 core
ownership 减法已完成；`object_emergence_solver` 同时承载 shared assignment state/predict
ABI，冻结期不为压缩 LOC 整体外移。146 个 core module files（compatibility wrappers 仍
保留）与 678 个 root exports 只能在明确 breaking/deprecation window 中收缩。

## 真实数据状态

- 已有 public replay: BOP HOPE、BOP LMO；有 pose GT，但没有可用真实 action，不能满足
  intervention exit criterion。
- 本地 controlled tabletop skeleton 不是证据；不得填 fixture action 或复制 target GT。
- [H2O](https://doi.org/10.3929/ethz-b-000685070) 已有匿名 Open Access / CC BY-NC 4.0
  官方副本，最小 RGB-D 分卷约 13.57 GB；旧注册入口条款单独治理。它和 HOT3D 都没有
  现行 intervention contract 要求的独立 3D action/control vector，当前没有本地文件。
- [HOI4D](https://hoi4d.github.io/) 也提供 RGB-D、category-level object pose 与 hand action，许可为 CC BY-NC 4.0；
  action 只有类别/时间区间而非独立 control vector；当前没有本地 raw sequence。
- RBO Articulated Objects 与 RRC 2020 的官方索引及各 3 条最小 acquisition candidate
  已于 2026-07-10 下载、完整性验证并完成字段语义审计。RBO 三段的 RGB-D、link/camera
  pose 与 wrench 时间覆盖合格，但严格 V-O-V 为 `0/3`，wrench sign/target link 未闭环；
  RRC 三段的三路 RGB、tracker pose 与 desired/applied action overlap 合格，但相机固定、
  无 depth，9D joint control 不能直接当现行 3D vector。
- 新增 `scripts/download-rbo-occlusion-followup.sh`：P0 冻结
  `treasurebox25/laptop26/globe25`（约 1.024 GiB），P1 再加
  `treasurebox24/laptop25`；它只解决 acquisition，下载后仍须用同一严格可见性方法复核。
- 仍需至少 3 个 scene，每个包含 physical identity、timestamped 6DoF pose、明确
  occlusion/view change 和测量得到的非零 action interval/vector。

## 当前阻塞与风险

- 实际 capture host / public interaction data 尚未提供任何同时满足全部条件的 scene；
  当前合格数为 `0/3`，三段 BOP 只是部分负证据。
- 当前 host 检测到 0 个视频设备，且无 ffmpeg/cv2/COLMAP/Nerfstudio capture/reconstruction
  工具链；首批 RBO/RRC archives 已完成审计，但 RBO 严格遮挡回环为 `0/3` 且 action
  sign/target-link 未闭环，RRC 又缺 temporal camera motion/depth/canonical 3D action。两者
  当前都不能满足完整 controlled-scene contract。
- 根目录已有保守的 all-rights-reserved `LICENSE`；若要开放复用，仍需 Owner 明确
  选择并替换为合适的开源许可证。
- Python 默认依赖未覆盖 torch / SAM / transformers / gsplat / nerfstudio 复现实验。
- Viewer 主 bundle 约 5.95 MB，full catalog eager load 会进一步放大首屏成本。
- `objgauss.core`、CLI 和 App 仍超出易审查规模；`objgauss/core/*.py` 已降至 11,956
  physical LOC（约 12.0k，含 compatibility wrappers 与 2,599-line lazy namespace），但
  146 个模块与 678 个 lazy exports 尚未减少，因为兼容壳仍保留。
- 32 个 root export 与 legacy 子模块同名；导入子模块会把同名 root function 覆盖为 module。
  冻结期只能停止新增，不能在无 breaking window 时删除既有 alias/wrapper。

## 验收口径

研究结论只接受：原始输入、原始候选输出、GT、独立重算 metrics、固定 threshold、
held-out scene 与简单 baseline 的完整链路。Reviewable、ready、schema-valid 或 synthetic
pass 都不能替代 metric pass。

通用 reality-row gate 只负责从规范 metrics 与 thresholds 独立派生行状态；raw artifact
到 metrics 的重算由 canonical identity/prediction/intervention evaluator 负责。绕过这些
evaluator 手工喂 metrics 不能形成可接受研究证据。

历史状态见：

- `docs/state/archive/project-status-through-2026-07-09.md`
- `docs/state/archive/pr-queue-through-2026-07-09.md`
