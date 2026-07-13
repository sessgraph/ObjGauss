# ObjGauss 当前状态总览

> 最近更新: 2026-07-13
> 阶段: research-first stabilization；Phase M M2 synthetic evidence reviewable，模型未胜基线
> 决策: `docs/adr/0007-research-first-stabilization.md`、
> `docs/adr/0008-objectstate-model-demo-phase.md`、
> `docs/adr/0009-objectstate-multi-object-evidence.md`

## 一句话状态

ObjGauss 是可运行的 Gaussian object-slot 研究与证据平台；当前尚未证明
`ObjectState` 是持续的真实世界状态，也不是 production-ready 对象编辑产品。

## 当前路线

RBO strict qualified scenes 为 `0/3`，后续 field mining 和 adapter 投入停止。Phase M
现分成两层：NeRF Lego M1 只证明 checkpoint/inference/viewer 接线；其一个物理对象上的
四色规则不计作 objectness。M2 以 12 个程序化四物体 scene、8/4 完整 scene holdout、
独立 `gt_instance_id`、两个同色同形立方体、接触和局部观测建立实例分割证据。

M2 leakage gate 通过，但 Model v0 的 Hungarian mIoU `0.754572` 低于 3D connected
components 基线 `0.825829`，差值 `-0.071257`；当前结论是模型未胜已记录简单基线，
不是 Phase M 模型通过。Viewer 已把 Raw / Prediction / GT 与同源 metrics 接通。下一阶段
仍需改进模型并回到 controlled capture；synthetic M2 不计作真实 identity/prediction/intervention pass。

## 已验证能力

- Gaussian PLY / `.splat` IO、对象色、过滤与 object-aware PLY。
- `A[N,K]` soft assignment、ObjectState pooling、训练 smoke、checkpoint roundtrip。
- Phase M M1 CLI 已把登记的 5,696-Gaussian NeRF Lego proxy 接成同 run Model v0 bundle；完整
  `source_frame` 划分为 9 train / 3 held-out，frame/row overlap 均为 0。loss
  `1.657089 -> 0.103907`；held-out ARI/mean-best-IoU/Purity 从
  `0.060393/0.237195/0.507981` 到 `1.0/1.0/1.0`。这只证明同一 Lego scene 的颜色标签
  assignment 与接线，不证明对象分割。Node/Viewer inference 与预计算 PLY `5,696/5,696`
  一致；artifact hash、layer switch、选择、hide/show 与 4/4 wiring gates 已核验。
- Phase M M2 已对四个 held-out multi-object scenes 运行 XYZ/RGB KMeans、3D connected
  components 与 Model v0；报告 Hungarian mIoU、ARI、count error、merge/split 与 Recall@IoU，
  失败 verdict 保留。浏览器从 checkpoint 重推理并与 467-row 预计算 prediction 一致，
  Raw/Prediction/GT、对象显隐、指标面板与独立 GT provenance 已通过专项 Playwright 和截图复核。
- Spark 真实 splat 展示与 Three.js 对象证据层；点数一致时支持 source splat 子集平移。
- controlled capture / BOP public replay 的现有数据合同、evaluator 和 ledger 工具链。
- 真实 3DGS / SAM / CLIP / gsplat 本地实验产物，但重依赖仍不是默认可复现环境。
- 当前工作树本地基线：820 个 Python 测试、Vite production build、M1/M2 专项 Playwright
  与 viewer truth audit 通过。完整串行 world audit 本轮通过前 7/9 artifact flows 后受 10 分钟
  外部 SIGTERM；并发重试触发资源 SIGTERM，未声明本轮完整 audit 通过。
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
同步 RGB-D、逐刚体 MoCap 6DoF、camera motion 和非零 100 Hz wrench。权威 pose 路径已
纠正为 base marker → `rb0` + 同刻 `joint_states` + matching-date URDF；在完整 camera TF、
registered depth 与冻结阈值下，全帧严格 clear → `occlusion_fraction >= 0.5` → clear
仍为 `0/3`，最大遮挡分别为 `0.421140 / 0.467351 / 0.476498`。

ATI load sign、加速度补偿与 force 语义仍独立成立：输出是 `/map`、N、
`measured_tool_on_object_force`，不是 controller command，也禁止当 `action_delta`。但
official-chain target 重算只接受 8 个 force validation windows 中的 5 个，以及 7 个
50 Hz geometry intervals 中的 3 个；三条 interval 只覆盖 `tripod25` 与 `cabinet20`，共使用
`1,172/1,172` 个 raw wrench samples，`cardboardbox22` 没有合格 target interval。三个
official-chain 3-frame bundle 的 capture/file validation 已通过，但 cardboard action 被明确
移除。旧 direct-marker `5 intervals / 3 bundles / 9 rows / 0-9-0 ledger` 已被 supersede，
不得作为当前 gate 证据。新 canonical ledger 为 `0 pass / 7 fail / 2 blocked`：cardboard
identity fail，独立 prediction-only eval 也 fail；但 full handoff 因无 action 不可运行，
所以主 ledger 不手工合并该结果，prediction/intervention 保持 blocked。tripod/cabinet 的
identity、prediction、intervention 均未过 reality gate。严格 scene 仍为 `0/3`，因为三段
都缺 V-O-V 且每段只有一个真实 lighting condition。事实源为 ignored
`outputs/evidence/rbo-objectstate-3scene/` 下的 `official-kinematic-audit`、
`action-target-official-chain-full`、`action-gt-official-chain` 与 `official-bundle-rebuild`。

## P0 稳定化结果

已完成：

1. supervised assignment CE 保留数学正确的 probability-space `-target/p` 导数；训练 caller
   改用解析、有界的 softmax-logit VJP，避免 materialize `1/p` 中间量及二次 Jacobian；
2. trainable ObjectState artifact validator 已强制 persistent `id == persistent_id`、renderer
   `slot == object_id`、帧内唯一非负地址，以及显式有限 `[0,1]` confidence；
3. Viewer catalog 按 stage/selection 加载，Spark hide/translate 与 source splat 同步，未闭环
   控件已移除；full/truth audit 已对齐 5-pill evidence UI、独立 artifact flows 与 translate-only 合同；
4. 最小 CI、保守 `LICENSE`、Python/Node 版本与依赖锁已落地；
5. teacher audit 在代码路径上绑定并实际消费 feature tensor；新 identity evaluator 从 raw
   track observations 关联，且只有显式有限 association distance 才能通过；旧 GT
   `object_pose_row_id` 键控路线已降为 diagnostic-only，不能产生 pass。canonical controlled
   reality handoff validator 会从 capture manifest、identity predictions 与 eval summary
   中保留的 prediction/intervention records 重跑三个 evaluator；通用 row gate 只保留
   threshold/accounting 诊断语义；
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

仍未完成：满足严格 identity scenario 的 controlled scene，以及超过 history/no-action
baseline 的预测与 intervention candidate。RBO measured action 目前只有两个 official-chain
ready scenes；新 canonical ledger 为 `0 pass / 7 fail / 2 blocked`，不构成能力通过。
当前冻结期可在不改 schema/ABI 的 core ownership 减法已完成；`object_emergence_solver`
同时承载 shared assignment state/predict
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
  pose 与 wrench 时间覆盖合格；official-chain visibility 仍为 `0/3`。ATI sign 已解决，
  但 target/interval 只留下 2 个 ready scenes，旧三场景 action 结果已 supersede；
  RRC 三段的三路 RGB、tracker pose 与 desired/applied action overlap 合格，但相机固定、
  无 depth，9D joint control 不能直接当现行 3D vector。
- RBO follow-up P0/P1/P2 均已下载并按 official chain 逐帧复核。P0 的 treasurebox/globe、
  P1 四段与 P2 `globe23` 均为可信零 V-O-V；laptop26 因 camera TF/RGB 矛盾保持
  `event_count=null`。P1 `3,199/3,199`、P2 `1,280/1,280` 个逐 link 样本独立重算一致；
  双 lighting 配对也未解锁 scene。19 条 interaction 已对账完毕，路线停止。
- 仍需至少 3 个 scene，每个包含 physical identity、timestamped 6DoF pose、明确
  occlusion/view change 和测量得到的非零 action interval/vector。

## 当前阻塞与风险

- 当前没有任何 scene 同时满足严格 V-O-V、两种 lighting 与完整 controlled 条件；合格数
  仍为 `0/3`。首批 RBO 只有 tripod/cabinet 两个 official-chain action-ready scenes；
  cardboard action 缺失，新 ledger 为 `0 pass / 7 fail / 2 blocked`，不能替代严格 scene。
- 当前 host 无视频设备与 capture/reconstruction 工具链；RBO 无严格遮挡回环，RRC 缺
  temporal camera motion/depth/canonical 3D action，均不能满足 controlled-scene contract。
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

通用 reality-row gate 只负责 threshold/accounting 诊断，不能单独形成可引用研究通过。
controlled full handoff 必须从 capture manifest、identity predictions 与 eval summary 中
保留的 prediction/intervention records 重跑 canonical evaluator；绕过该 live 链手工喂
summary/metrics 不计证据。

历史状态见：

- `docs/state/archive/project-status-through-2026-07-09.md`
- `docs/state/archive/pr-queue-through-2026-07-09.md`
