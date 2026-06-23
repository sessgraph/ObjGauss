# ADR 0004: Object-Aware Gaussian Renderer Roadmap

> 状态: Accepted / object-state filtering implemented
> 日期: 2026-06-23

## 背景

ObjGauss 前端现在有两条渲染路径：

- `真实查看`: Spark 读取 `.splat`，用于真实 3DGS 外观预览。
- `对象编辑`: 读取带 `object_id` 的 Gaussian PLY，用于对象选择、隔离、隐藏和删除预览。

早期对象编辑使用 Three.js `PointsMaterial` 或 soft sprite 点云。它能验证对象绑定，但不是 object-aware Gaussian renderer：

- 没有消费 Gaussian scale / rotation / opacity。
- 没有 screen-space Gaussian kernel。
- 没有 weighted blended OIT。
- 没有 tile-based splat accumulation。

这会让删除后的自身颜色预览带明显颗粒感，也无法作为 Obj-level Gaussian representation 的长期渲染架构。

## 决策

采用 B -> C 渐进路线：

1. **Phase 1 / B: Three.js Gaussian Shader 编辑预览。**
   - 保留 `THREE.Points` 几何和 raycast 选中能力。
   - 替换 `PointsMaterial` 为 `ShaderMaterial`。
   - 从 PLY 解析 `scale_0/1/2`、`rot_0..3` 和 `opacity`。
   - 将 per-Gaussian scale / rotation / opacity 作为 GPU attributes。
   - 在 fragment shader 里用 `gl_PointCoord` 计算 screen-space ellipse Gaussian kernel。

2. **Phase 2 / B+: Weighted Blended OIT。**
   - 使用 RGBA half-float accumulation render target。
   - RGB 累加 `sum(w * c)`，Alpha 累加 `sum(w)`。
   - fullscreen resolve pass 输出 `sum(w * c) / sum(w)`，再混回基础 grid / axes 场景。
   - 目标是 demo 级 object-aware splat preview，不要求 WebGPU。

3. **Phase 3 / C: Tile-based WebGPU Gaussian renderer。**
   - 以 object-aware runtime editing 为终局。
   - 支持 100k 到 1M Gaussian 的 binning、per-tile accumulation 和 object-id filtering。
   - WebGL2 shader renderer 保留为 fallback/debug path。

## 当前实施

Phase 1 / Phase 2 已落地到前端对象编辑 renderer：

- `src/ply.js` 解析 Gaussian scale、opacity 和 quaternion rotation 的 screen-space 近似角。
- `src/PointCloudViewport.jsx` 使用 `ShaderMaterial` 渲染 Gaussian kernel，而不是默认 `PointsMaterial`。
- `src/PointCloudViewport.jsx` 使用 weighted OIT accumulation / resolve 管线，降低普通透明混合排序伪影。
- `src/PointCloudViewport.jsx` 将 `object_id` 映射为 dense GPU attribute，并通过 object-state `DataTexture` 在 shader 内执行隐藏 / 隔离 / 删除过滤。
- `src/sampleScene.js` 为内置 demo 提供 fallback scale / rotation。
- `scripts/audit-demo.mjs` 验证编辑 renderer 为 `Gaussian OIT 编辑`，并覆盖画布选中、隔离和删除后的自身颜色预览。

## 明确非目标

当前 Phase 2 不是完整 3DGS renderer：

- 没有完整 3D covariance 投影。
- 没有 depth sort；当前使用 weighted blended OIT approximation。
- 没有 SH view-dependent color。
- 没有 tile binning 或 WebGPU compute。
- Spark `.splat` 真实查看路径还不能直接按 `object_id` 过滤；object-state filtering 当前落在 Gaussian OIT 编辑 renderer。

因此 UI 使用 `Gaussian OIT 编辑`，不把它称为最终真实 splat 删除。

## 验收

Phase 2 验收标准：

- Browser audit 显示 `editRenderer="Gaussian OIT 编辑"`。
- Browser audit 显示 `objectFilter="gpu-object-state-texture"`。
- Plush / Plush semantic / Lego proxy 三个样例均可进入 shader edit renderer。
- 点击编辑画布可选中 object。
- 删除预览后仍显示剩余整体场景，并回到 `自身颜色`。
- 没有 shader compile/link console error。
- 没有 framebuffer / render target console error。

## 后续任务

- `RENDER-004`: WebGPU tile-based object-aware Gaussian renderer.
