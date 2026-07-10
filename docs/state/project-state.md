# ObjGauss 项目状态边界

> 最近更新: 2026-07-10

## 项目目标

ObjGauss 的长期目标是验证和构建“对象级可编辑、可持续推理的 3D Gaussian
场景”。当前阶段明确采用 research-first：先证明 ObjectState 在真实 controlled
scene 中具有 identity persistence、predictive sufficiency 和 action-conditioned
state transition，再讨论产品扩张或 world-model 叙事。

基础产品边界仍包括：

- 从 3DGS Gaussian PLY 或 `.splat` 样例进入。
- 为每个 Gaussian 附加 `object_id`。
- 使用真实 3DGS splat renderer 预览原始高斯外观。
- 在界面中预览原始颜色（编辑预览）和对象色（编辑预览）。
- 长期支持对象隔离、删除和 6DoF 编辑；当前 evidence viewer 只暴露已与 Spark
  source 真正同步的能力。
- 建立训练素材与 Demo 素材分层管理。

当前 Viewer 仅作为 evidence viewer：展示原始 Gaussian、派生对象层、原始预测、GT、
metrics 和 failure evidence。它不是本阶段的对象编辑产品主线。

## 当前研究主张

唯一待证明主张是：在不使用 target identity / target pose 泄漏的前提下，
`Gaussian / PerceptionEvidence -> A[N,K] -> ObjectState` 能在 held-out 真实场景中：

- 维持 physical identity；
- 在遮挡和视角变化后恢复；
- 比 history-only baseline 更好地预测状态；
- 使用真实 action 后优于 no-action baseline。

在真实数据通过前，`ObjectState` 只可表述为 object-slot observation state，不能表述为
已验证 world state。

## 开发阶段说明

ObjGauss 当前仍是 development-stage research prototype / 开发阶段研究原型。
当前 API、CLI、资产布局、指标、模型产物和文档都可能在 stable release 前变化。
HF 资产与本地 ignored `outputs/` 产物用于研究复现和 handoff，不能表述为
production-ready release 或 commercial demo release。

## 当前 MVP 边界

已接受的 MVP 能力：

- Gaussian PLY 读写。
- antimatter15/cakewalk `.splat` 转 PLY。
- 基于 `[x, y, z, r, g, b, opacity]` 的 KMeans 聚类。
- Object Field v1-lite：`object_logits: (N, K)` 软对象槽位、指标和 hard `object_id` 导出。
- 预计算 2D mask 投票到 Object Field。
- Object Field projection loss 训练 smoke。
- `AssignmentSolverV2 -> A[N,K] -> ObjectStateProjection` 固定槽位训练与评估。
- CPU / optional gsplat renderer-loss smoke；不等于完整端到端研究结论。
- `object_id` 写回 PLY。
- React + Spark / Three.js splat 预览。
- React + Three.js 点云编辑 fallback。
- 素材库登记和 Plush 示例自动拉取。
- Poly Haven CC0 mesh Demo 输入源自动拉取。
- NeRF Synthetic Lego 多视角训练素材自动拉取。
- NeRF Synthetic Lego 轻量 Gaussian proxy 闭环样例。

## 明确非目标

当前阶段不声称：

- 已交付可替代成熟 3DGS renderer 的 production renderer。仓库拥有实验性
  WebGPU/OIT kernels 和 Spark bridge，但默认主路径仍依赖 Spark。
- 已完整实现 splat source 的对象级隐藏、隔离、删除、旋转和缩放；这些能力只有在
  source truth 与 UI 同步并通过浏览器证据后才能声明。
- 已对所有 3DGS PLY / `.splat` / SH 格式做完整兼容矩阵。
- 已实现语义级对象分割。
- 已将 SAM / CLIP 作为默认仓库依赖、提交其权重或完成稳定语义级对象分割。
- 已证明完整 3DGS render loss 联合训练可泛化到 held-out real scenes。
- NeRF Lego proxy 已等价于完整 3DGS optimization 训练输出。
- 已完成 ARKitScenes / OmniObject3D 转换管线。
- 当前 Plush 测试素材可商用或可公开发布。

## 设计边界

- 完整 3DGS renderer 替换已按 ADR `0001-3dgs-renderer` 落地；后续 shader 级对象编辑仍需单独立项。
- SAM/CLIP/Gaussian Grouping 等语义分割依赖属于重大或标准 PR；当前统一接口是预计算 mask manifest -> Object Field。SAM / CLIP 可作为本地可选模型运行，但权重和 cache 不进入 git，当前 CLIP 命名质量仍未 promotion。
- 大型素材、训练素材、训练输出默认不提交仓库。
- 小型 Demo 样例进入 `public/samples/` 前必须记录来源和许可。
