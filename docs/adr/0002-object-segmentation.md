# ADR 0002: Object Segmentation Strategy

> 状态: Accepted / v1-lite implemented
> 日期: 2026-06-22

## 背景

当前 ObjGauss 使用 KMeans 对 `[x, y, z, r, g, b, opacity]` 聚类，并写入 `object_id`。这能验证对象级编辑 runtime，但不等价于语义/实例分割。

主要问题：

- 聚类不理解对象语义。
- 复杂场景中对象边界不稳定。
- 不同视角、遮挡、同色物体会影响结果。

## 决策

保留 KMeans 作为 baseline；语义级对象分组作为独立能力推进，输出仍统一为 ObjGauss PLY with `object_id`。

在引入 SAM / CLIP / Gaussian Grouping 之前，先落地 `Object Field v1-lite`：

- 每个 Gaussian 维护 `object_logits: (N, K)`。
- 当前用 KMeans 结果 warm start 成软 object-slot 分布。
- CLI 保存 `.npz` Object Field，并可导出 hard `object_id` PLY 供现有前端复用。
- NeRF Synthetic Lego 作为第一套多视角训练烟测数据，先验证 transforms 和图像完整性。
- 不新增深度学习依赖；完整 optimizer / view consistency / semantic guidance 后续单独 PR。

候选路线：

1. 图像侧 2D segmentation + 多视角投票到 Gaussian。
2. CLIP / SAM 特征融合后对 Gaussian 分组。
3. Gaussian Grouping 类方法。
4. 使用带实例标注的数据集做监督或评估。

推荐第一步：选一个小验证集，建立 KMeans baseline 与语义分组输出对比，不直接追求全自动最优。

## 实施结果

`OBJFIELD-001` 已实现最小训练骨架：

- `objgauss object-field init`: 从 Gaussian PLY 初始化软 Object Field。
- `objgauss object-field stats`: 输出 entropy、normalized entropy、sharpness、active slots。
- `objgauss object-field export`: 将软分布导出为现有 viewer 可用的 `object_id` PLY。
- `objgauss object-field inspect-nerf`: 检查 NeRF-style `transforms_*.json`、图像引用和 4x4 pose。

这一步降低了从 KMeans baseline 走向可学习对象分区的接口风险，但还没有解决语义分割质量问题。

`SEG-001` / `OBJFIELD-002` 已实现第一条语义接入路径：

- `objgauss object-field vote-masks`: 将预计算 2D masks 投影到 Gaussian。
- mask manifest 支持 `rect` 和 boolean `.npy` mask，slot 由上游 SAM / CLIP / 人工标注结果指定。
- 对同一个 Gaussian 聚合多视角 mask votes，形成 projection supervision。
- 用 NumPy 训练循环更新 `object_logits`，以投影交叉熵降低为验收指标。
- 可直接导出 hard `object_id` PLY 给现有 viewer 使用。

仍不在本仓库内运行 SAM / CLIP 模型；新增模型依赖和权重下载单独立项。

## 验收标准

- 输入同一个 scene，输出带 `object_id` 的 PLY。
- 至少有一个小场景能展示语义对象比 KMeans 更稳定。
- 前端对象隔离/删除预览无需改接口即可使用。
- 明确依赖、模型权重、运行成本和许可。

## 风险

- SAM/CLIP 等依赖可能显著增加安装和运行成本。
- 数据集许可可能限制公开 Demo。
- 语义分割结果需要相机位姿、多视角图像或额外特征，不一定适用于所有已有 PLY。

## 后续任务

- `SEG-002`: 接入可选 SAM / CLIP mask 生成器，输出本 ADR 定义的 mask manifest。
- `TRAIN-001`: 训练 NeRF Lego Gaussian PLY，用真实 3DGS 输出验收 mask voting。
