# ObjGauss 风险登记

> 最近更新: 2026-06-22

| ID | 风险 | 影响 | 当前缓解 | 关闭条件 | 状态 |
| --- | --- | --- | --- | --- | --- |
| R-001 | 当前不是完整 3DGS renderer，只是高斯中心点云预览 | Demo 视觉效果和真实 3DGS 有差距 | 已立 ADR `0001-3dgs-renderer` 和 PR `RENDER-001` | 真实 splat renderer 接入并通过浏览器验证 | open |
| R-002 | 当前对象分组是 KMeans，不是语义级对象分割 | 对象边界和语义一致性不稳定 | 已立 ADR `0002-object-segmentation` 和 PR `SEG-001` | 语义/实例分组方案输出 `object_id` 并有基准对比 | open |
| R-003 | 只有 Plush 自动拉取，其他素材无转换管线 | 训练和 Demo 数据不足 | 已立 ADR `0003-asset-ingestion` 和 PR `ASSET-001` | 至少一个 Demo 源和一个训练源跑通转换 | open |
| R-004 | 仓库尚无 baseline commit | 进度不可追踪，后续 AI 会话难以协作 | 本轮将创建 `docs/state/` 并提交 baseline | baseline commit 存在且状态文件回填 | mitigating |
| R-005 | Plush 来源许可混合 | 不适合公开发布或商用 Demo | `docs/asset-library.md` 已标明仅本地测试 | 首个公开 Demo 改用许可明确素材 | open |
| R-006 | Three.js bundle 超 500KB warning | 后续页面加载可能变慢 | 当前只记录，不影响 MVP | 引入 code splitting 或 renderer 方案时处理 | accepted |
