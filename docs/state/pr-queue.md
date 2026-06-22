# ObjGauss PR 队列

> 最近更新: 2026-06-22

## 队列规则

- 一次只执行一个 PR。
- 重大变更先补 ADR，Owner 确认后执行。
- 每个 PR 完成后更新验证结果和完成 commit。

## Ready

### SEG-001: 建立语义级对象分组方案

- 状态: ready-for-ADR-review
- 类型: 重大变更或标准 PR，取决于依赖选择
- ADR: `docs/adr/0002-object-segmentation.md`
- 目标: 从 KMeans MVP 走向语义/实例对象分组。
- 范围外: 不在第一步追求全自动高质量分割。
- 验收:
  - 明确第一套验证数据。
  - 输出仍为 ObjGauss PLY with `object_id`。
  - 与 KMeans 基线可对比。

### ASSET-001: 建立 Demo/训练素材转换管线

- 状态: ready
- 类型: 标准 PR
- ADR: `docs/adr/0003-asset-ingestion.md`
- 目标: 将至少一个 Demo 素材源和一个训练素材源拉通到本地目录规范。
- 首选:
  - Demo: Poly Haven 小模型。
  - 训练: OmniObject3D 或 ARKitScenes 最小子集。
- 验收:
  - `objgauss assets list` 能看到 pipeline stage 和 use cases。
  - 新素材记录来源、许可、下载命令、转换命令。
  - Demo 样例可在前端加载或明确说明为何只作为训练输入。

## Done

### RENDER-001: 评估并接入完整 3DGS renderer

- 状态: done
- 类型: 重大变更
- ADR: `docs/adr/0001-3dgs-renderer.md`
- 目标: 从点云预览升级到真实 3DGS splat 渲染，支持椭圆 splat、透明度合成、视角交互。
- 范围外: 不同时实现训练 pipeline；不把语义分割混入 renderer PR。
- 验收:
  - Plush 样例使用 `@sparkjsdev/spark` 读取 `.splat` 并显示真实 splat 外观。
  - 点云编辑 fallback 保留，用于对象聚类色、隐藏、隔离和删除预览。
  - 桌面 1440x920 与移动端 390x844 浏览器验证均非空、无前端错误。
- 验证:
  - `uv run --extra dev pytest`: 5 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
  - Playwright: 桌面 canvas `nonBackground=9422`，移动端 canvas `nonBackground=4559`。
- 完成 commit: `e34b7de`。

### BASE-001: MVP 原型与流程基线

- 状态: done
- 类型: 基线固化
- 目标: 固化当前 CLI、前端、素材库、AI 流程和状态事实源。
- 完成 commit: `c8dcef7`
- 验收:
  - `uv run --extra dev pytest` 通过。
  - `npm run build` 通过。
  - 已创建 `docs/state/`。
  - baseline commit 已完成。
