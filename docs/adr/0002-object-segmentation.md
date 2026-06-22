# ADR 0002: Object Segmentation Strategy

> 状态: Proposed
> 日期: 2026-06-22

## 背景

当前 ObjGauss 使用 KMeans 对 `[x, y, z, r, g, b, opacity]` 聚类，并写入 `object_id`。这能验证对象级编辑 runtime，但不等价于语义/实例分割。

主要问题：

- 聚类不理解对象语义。
- 复杂场景中对象边界不稳定。
- 不同视角、遮挡、同色物体会影响结果。

## 决策

保留 KMeans 作为 baseline；语义级对象分组作为独立能力推进，输出仍统一为 ObjGauss PLY with `object_id`。

候选路线：

1. 图像侧 2D segmentation + 多视角投票到 Gaussian。
2. CLIP / SAM 特征融合后对 Gaussian 分组。
3. Gaussian Grouping 类方法。
4. 使用带实例标注的数据集做监督或评估。

推荐第一步：选一个小验证集，建立 KMeans baseline 与语义分组输出对比，不直接追求全自动最优。

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

- `SEG-001`: 建立语义级对象分组方案。
