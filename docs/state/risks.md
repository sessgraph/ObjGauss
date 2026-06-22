# ObjGauss 风险登记

> 最近更新: 2026-06-22

| ID | 风险 | 影响 | 当前缓解 | 关闭条件 | 状态 |
| --- | --- | --- | --- | --- | --- |
| R-001 | 当前不是完整 3DGS renderer，只是高斯中心点云预览 | Demo 视觉效果和真实 3DGS 有差距 | 已接入 `@sparkjsdev/spark`，Plush `.splat` 已通过桌面和移动端浏览器验证 | 真实 splat renderer 接入并通过浏览器验证 | closed |
| R-002 | 当前默认对象分组不是端到端语义级对象分割 | 对象边界和语义一致性不稳定 | 已落地 `DEMO-001`：真实 Plush splat + Object Field + 多视角 mask voting + 前端对象编辑闭环可复现；`VERIFY-001` 已固化机器验收；`MASK-001` 可从 NeRF Lego 真实 RGBA 图片生成前景 mask manifest；`MASK-002` 可从 NeRF Lego 真实 RGBA 图片生成 4-slot color mask manifest 并独立监督 Object Field；`LEGO-001` 用真实 NeRF Lego 2D color masks 跑通同一 proxy scene 闭环；`VERIFY-002` 已固化 Lego proxy 机器验收；`VERIFY-003` 已检查 mask guidance 实际改变 Object Field labels；`UI-AUDIT-001` 已固化前端交互验收；`ACCEPT-001` 已固化一键总验收；默认 KMeans 仍保留为 baseline | 仓库内可对真实小场景生成 SAM / CLIP mask manifest，并与 KMeans 基线对比 | open |
| R-003 | 只有 Plush 自动拉取，其他素材无转换管线 | 训练和 Demo 数据不足 | 已接入 `polyhaven-school-chair-1k` 和 `nerf-synthetic-lego` 自动拉取管线 | 至少一个 Demo 源和一个训练源跑通转换 | closed |
| R-004 | 仓库尚无 baseline commit | 进度不可追踪，后续 AI 会话难以协作 | 已创建 baseline commit `c8dcef7` 并回填状态文件 | baseline commit 存在且状态文件回填 | closed |
| R-005 | Plush 来源许可混合 | 不适合公开发布或商用 Demo | `docs/asset-library.md` 已标明仅本地测试 | 首个公开 Demo 改用许可明确素材 | open |
| R-006 | Three.js / Spark bundle 超 500KB warning | 后续页面加载可能变慢 | 当前只记录，不影响 MVP；RENDER-001 后主 JS 约 5.6MB / gzip 1.94MB | 引入 code splitting 或按需加载 Spark renderer | accepted |
| R-007 | Poly Haven mesh 还不是可直接 viewer 打开的 3DGS Demo | 许可干净素材已接入，但公开演示仍需要训练转换 | 已记录 mesh -> 多视角渲染 -> 3DGS 后续链路 | School Chair 训练出 `.splat` / ObjGauss PLY 并可前端加载 | open |
| R-008 | SAM / CLIP 仍是外部预计算 mask 来源 | 语义分割效果依赖外部脚本，仓库不能端到端复现实例分割 | 已定义稳定 mask manifest 和 `vote-masks` 消费命令；`objgauss masks from-nerf-alpha` 已能从 NeRF Lego 真实图片生成前景 mask manifest；`objgauss masks from-nerf-rgba-colors` 已能从真实图片生成多 slot color mask manifest；`objgauss masks from-nerf-sam` 已作为可选 SAM 入口接入，权重和依赖由本机提供 | 用真实 SAM checkpoint 生成小场景 manifest，并被 `vote-masks` 消费；CLIP 语义命名另行立项 | open |
| R-009 | NeRF Lego 还没有真实训练出的 Gaussian PLY / `.splat` | 当前 Lego 闭环仍依赖 proxy，不能完全代表真实 3DGS optimization 输出 | `TRAIN-002` 已固化外部训练输出接入命令，真实 trainer 产物一旦存在即可登记、跑 mask voting、导出 `object_id` PLY；`AUDIT-001` 会把该缺口报告为 `unified_real_3dgs_mask_demo_available` blocker | `TRAIN-001` 产出真实 Lego Gaussian，并用 `training register-output` 完成前端可加载验收 | open |
