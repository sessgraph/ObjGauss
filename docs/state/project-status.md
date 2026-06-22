# ObjGauss 当前状态总览

> 最近更新: 2026-06-22

## 当前阶段

MVP 原型可运行，已完成流程化基线提交，并已接入真实 3DGS splat renderer。

## 已完成能力

- Python CLI:
  - `objgauss convert-splat`
  - `objgauss cluster`
  - `objgauss colorize`
  - `objgauss filter`
  - `objgauss stats`
  - `objgauss assets list/pull`
- 前端:
  - 中文 UI。
  - Spark / Three.js 真实 3DGS splat 预览。
  - Three.js 高斯中心点云编辑 fallback。
  - 自身颜色 / 对象聚类色切换。
  - 对象列表、隔离、删除预览。
  - 素材库卡片和本地 Plush 样例加载。
- 素材:
  - `plush-3dgs-local` 可自动拉取。
  - Plush `.splat` 用于真实 renderer，`plush_objects.ply` 用于对象级编辑。
  - ARKitScenes、ScanNet、OmniObject3D、Google Scanned Objects、Poly Haven、Mip-NeRF 360、Tanks and Temples 已登记为候选来源。
- 流程:
  - `docs/development-flow.md` 已建立。
  - `AGENTS.md` 和 `CLAUDE.md` 已指向统一流程。
  - baseline commit: `c8dcef7`.

## 最近验证

2026-06-22:

```bash
uv run --extra dev pytest
npm run build
```

结果：

- Python 测试: 5 passed。
- 前端构建: 通过。
- 浏览器验证: 桌面 1440x920 与移动端 390x844 均渲染非空、无前端错误。
- 已知提示: Vite 报 Spark / Three.js chunk 超过 500KB，不影响当前预览。

## 当前限制

- 对象聚类色、隐藏、隔离、删除预览仍通过点云编辑 fallback 完成，不是对象级 splat shader。
- 对象分组仍是 KMeans，不是语义级分割。
- 只有 Plush 样例具备自动拉取管线。
- 训练素材和训练输出目录规范已定义，但实际训练管线尚未接入。

## 下一步主线

1. 执行 SEG-001: 建立语义对象分组方案和最小验证集。
2. 执行 ASSET-001: 建立 Poly Haven / OmniObject3D / ARKitScenes 的最小转换管线。
3. 后续 renderer 优化: Spark 按需加载或拆包，降低首屏 bundle。
