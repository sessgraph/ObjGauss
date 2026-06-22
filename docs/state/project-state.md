# ObjGauss 项目状态边界

> 最近更新: 2026-06-22

## 项目目标

ObjGauss 的目标是验证和构建“对象级可编辑 3D Gaussian 场景”：

- 从 3DGS Gaussian PLY 或 `.splat` 样例进入。
- 为每个 Gaussian 附加 `object_id`。
- 在界面中预览自身颜色和对象聚类色。
- 支持对象隔离、删除预览和后续编辑流程。
- 建立训练素材与 Demo 素材分层管理。

## 当前 MVP 边界

已接受的 MVP 能力：

- Gaussian PLY 读写。
- antimatter15/cakewalk `.splat` 转 PLY。
- 基于 `[x, y, z, r, g, b, opacity]` 的 KMeans 聚类。
- `object_id` 写回 PLY。
- React + Three.js 点云预览。
- 素材库登记和 Plush 示例自动拉取。

## 明确非目标

当前阶段不声称：

- 已实现完整 3DGS 椭圆 splat renderer。
- 已实现 SH view-dependent shading。
- 已实现 depth sort / alpha compositing。
- 已实现语义级对象分割。
- 已完成 ARKitScenes / OmniObject3D / Poly Haven 转换管线。
- 当前 Plush 测试素材可商用或可公开发布。

## 设计边界

- 完整 3DGS renderer 替换属于重大变更，先走 ADR。
- SAM/CLIP/Gaussian Grouping 等语义分割依赖属于重大或标准 PR，先明确第一个调用方和验收数据。
- 大型素材、训练素材、训练输出默认不提交仓库。
- 小型 Demo 样例进入 `public/samples/` 前必须记录来源和许可。
