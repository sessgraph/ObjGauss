# ADR 0001: 3DGS Renderer Strategy

> 状态: Accepted / Implemented
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

## 实施结果

选择 `@sparkjsdev/spark` 2.1.0 作为前端真实 3DGS renderer，并将 `three` 升级到 0.180.0 以满足 peer dependency。

落地策略：

- Plush 示例真实渲染读取 `public/samples/plush.splat`。
- 对象编辑 fallback 继续读取 `public/samples/plush_objects.ply`。
- 默认优先使用“真实 Splat”；当用户切换到对象聚类色、对象隐藏、隔离或删除预览时，自动回落到“点云编辑”。
- 用户上传 PLY 时保留原始 `ArrayBuffer` 副本，若格式被 Spark 支持可进入真实 renderer；否则仍可用点云预览。

没有选择自研 renderer。没有把对象编辑、语义分割或训练 pipeline 混入本 PR。

## 验收标准

- Plush 样例在新 renderer 下呈现 splat 外观，而非稀疏点云。
- 保留或明确替换当前点云 fallback。
- 浏览器验证桌面和移动端非空、可交互、无框架错误。
- 对象可见性、隔离、颜色模式仍可工作。

## 风险

- 引入外部 renderer 可能带来 bundle 增大和格式适配成本。
- `.splat`、PLY、标准 3DGS PLY 字段可能需要多格式适配。
- SH 渲染可能不是第一步必须项，应先跑通 ellipse splat + alpha。
- 当前 ObjGauss RGB PLY 不是标准 `f_dc_0` 3DGS PLY，Spark 不能直接作为真实 renderer 输入；因此真实 renderer 读取 `.splat`，对象编辑读取带 `object_id` 的 PLY。

## 后续任务

- `RENDER-001`: 评估并接入完整 3DGS renderer。
