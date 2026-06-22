# ADR 0001: 3DGS Renderer Strategy

> 状态: Proposed
> 日期: 2026-06-22

## 背景

当前 ObjGauss 前端使用 Three.js `PointsMaterial` 渲染 Gaussian 中心点。这足够验证：

- PLY 加载。
- 自身颜色 / 对象聚类色切换。
- 对象隔离和删除预览。

但它不是完整 3D Gaussian Splatting renderer，缺少：

- 椭圆 splat 投影。
- scale / rotation covariance 渲染。
- alpha compositing。
- depth sort。
- SH view-dependent color。

## 决策

短期保留当前点云预览作为 fallback；新增真实 3DGS renderer 必须独立成 PR，不混入素材管线或语义分割。

候选路线：

1. 接入成熟 WebGL splat renderer。
2. 在 Three.js 中接入现成 3DGS renderer 包。
3. 自研 shader renderer。

推荐顺序：先评估成熟 WebGL / Three.js renderer，避免自研透明排序和 splat shader。

## 验收标准

- Plush 样例在新 renderer 下呈现 splat 外观，而非稀疏点云。
- 保留或明确替换当前点云 fallback。
- 浏览器验证桌面和移动端非空、可交互、无框架错误。
- 对象可见性、隔离、颜色模式仍可工作。

## 风险

- 引入外部 renderer 可能带来 bundle 增大和格式适配成本。
- `.splat`、PLY、标准 3DGS PLY 字段可能需要多格式适配。
- SH 渲染可能不是第一步必须项，应先跑通 ellipse splat + alpha。

## 后续任务

- `RENDER-001`: 评估并接入完整 3DGS renderer。
