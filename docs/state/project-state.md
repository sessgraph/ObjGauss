# ObjGauss 项目状态边界

> 最近更新: 2026-06-22

## 项目目标

ObjGauss 的目标是验证和构建“对象级可编辑 3D Gaussian 场景”：

- 从 3DGS Gaussian PLY 或 `.splat` 样例进入。
- 为每个 Gaussian 附加 `object_id`。
- 使用真实 3DGS splat renderer 预览原始高斯外观。
- 在界面中预览自身颜色和对象聚类色。
- 支持对象隔离、删除预览和后续编辑流程。
- 建立训练素材与 Demo 素材分层管理。

## 当前 MVP 边界

已接受的 MVP 能力：

- Gaussian PLY 读写。
- antimatter15/cakewalk `.splat` 转 PLY。
- 基于 `[x, y, z, r, g, b, opacity]` 的 KMeans 聚类。
- Object Field v1-lite：`object_logits: (N, K)` 软对象槽位、指标和 hard `object_id` 导出。
- 预计算 2D mask 投票到 Object Field。
- Object Field projection loss 训练 smoke。
- `object_id` 写回 PLY。
- React + Spark / Three.js splat 预览。
- React + Three.js 点云编辑 fallback。
- 素材库登记和 Plush 示例自动拉取。
- Poly Haven CC0 mesh Demo 输入源自动拉取。
- NeRF Synthetic Lego 多视角训练素材自动拉取。

## 明确非目标

当前阶段不声称：

- 已自研 3DGS renderer。
- 已在 splat renderer 内实现对象级隐藏、隔离、删除或对象聚类色 shader。
- 已对所有 3DGS PLY / `.splat` / SH 格式做完整兼容矩阵。
- 已实现语义级对象分割。
- 已在本仓库内运行 SAM / CLIP 模型或下载其权重。
- 已实现完整 3DGS render loss 联合训练。
- 已完成 ARKitScenes / OmniObject3D / Poly Haven 转换管线。
- 已将 Poly Haven mesh 自动转换成可前端加载的 3DGS Demo。
- 当前 Plush 测试素材可商用或可公开发布。

## 设计边界

- 完整 3DGS renderer 替换已按 ADR `0001-3dgs-renderer` 落地；后续 shader 级对象编辑仍需单独立项。
- SAM/CLIP/Gaussian Grouping 等语义分割依赖属于重大或标准 PR；当前统一接口是预计算 mask manifest -> Object Field。
- 大型素材、训练素材、训练输出默认不提交仓库。
- 小型 Demo 样例进入 `public/samples/` 前必须记录来源和许可。
