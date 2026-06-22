# ObjGauss PR 队列

> 最近更新: 2026-06-22

## 队列规则

- 一次只执行一个 PR。
- 重大变更先补 ADR，Owner 确认后执行。
- 每个 PR 完成后更新验证结果和完成 commit。

## Ready

### SEG-002: 接入可选 SAM / CLIP mask 生成器

- 状态: ready-for-ADR-review
- 类型: 重大变更或标准 PR，取决于依赖选择和模型权重策略
- ADR: `docs/adr/0002-object-segmentation.md`
- 目标: 从图片生成当前 `vote-masks` 命令可消费的 mask manifest。
- 范围外: 不改变 Object Field 文件格式；不把模型权重提交仓库。
- 验收:
  - 明确 SAM / CLIP 依赖、权重下载方式、许可和运行成本。
  - 对一个小场景输出 mask manifest。
  - `objgauss object-field vote-masks` 可消费该 manifest。

### TRAIN-001: 训练 NeRF Lego Gaussian PLY

- 状态: ready-for-ADR-review
- 类型: 重大变更
- 目标: 基于 `nerf-synthetic-lego` 得到可供 Object Field / mask voting 使用的 Gaussian PLY。
- 范围外: 不自研完整 3DGS trainer，优先封装成熟训练器。
- 验收:
  - 产出 Lego `gaussians.ply` 或 `.splat`。
  - 可用 `objgauss object-field init` 和 `vote-masks` 跑通最小验收。

## Done

### MASK-001: 生成 NeRF Lego 真实图像 alpha mask manifest

- 状态: done
- 类型: 标准 PR
- ADR: `docs/adr/0002-object-segmentation.md`
- 目标: 在不引入 SAM / CLIP 依赖的前提下，从 NeRF Synthetic Lego 真实 RGBA 图片生成 `vote-masks` 可消费的 mask manifest。
- 实施:
  - 新增 `objgauss masks from-nerf-alpha`。
  - 直接解析 8-bit RGBA PNG alpha 通道，生成 boolean `.npy` mask。
  - 输出沿用现有 mask manifest，保留 `transform_matrix`、`camera_angle_x`、width/height 和 mask path。
- 范围外:
  - 不实现 SAM / CLIP 模型推理。
  - 不做实例级多对象分割；当前 alpha mask 是 Lego 前景监督。
- 验收:
  - NeRF Lego 真实图片可生成 mask manifest。
  - 生成的 manifest 可作为 `objgauss object-field vote-masks` 的输入格式。
- 验证:
  - `uv run --extra dev pytest`: 14 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
  - `uv run objgauss masks from-nerf-alpha outputs/assets/training/nerf-synthetic-lego --output outputs/masks/nerf-lego-alpha/mask-manifest.json --split train --max-frames 8 --threshold 1 --slot 0 --label foreground`: 8 frames，8 masks，800x800，299242 foreground pixels。
- 完成 commit: `e96b024`。

### DEMO-001: 固化 ObjGauss v1 闭环验收 demo

- 状态: done
- 类型: 标准 PR
- 目标: 把当前阶段终点压成一个可复现结果画面：真实 3DGS 外观 + Object Field + mask 投票 + 前端对象隔离/删除。
- 实施:
  - 新增 `objgauss demo v1-closure`。
  - 基于 Plush 真实 `.splat` 和 `plush_objects.ply` 生成闭环验收包。
  - 自动生成 3 个投影视角、18 个 mask votes、训练后 Object Field 和 `plush_v1_objects.ply`。
  - 前端素材库新增 `ObjGauss v1 闭环样例`。
- 范围外:
  - 不声称 SAM / CLIP 已在仓库内运行。
  - 不声称 NeRF Lego 已训练出 Gaussian PLY。
- 验收:
  - `outputs/demos/v1-closure/v1-closure-manifest.json` 记录闭环证据。
  - `public/samples/plush_v1_objects.ply` 可由前端加载。
  - projection loss 下降，Object Field 输出 6 个 active object slots。
- 验证:
  - `uv run --extra dev pytest`: 13 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
  - `uv run objgauss demo v1-closure --iterations 80`: 281498 gaussians，6 objects，loss 1.791760 -> 1.201637。
  - Playwright + system Chrome: 素材库可见并可加载 `ObjGauss v1 闭环样例`，可选择对象、只看所选、预览删除。
- 完成 commit: `6802e7f`。

### SEG-001: 建立语义级对象分组方案

- 状态: done
- 类型: 标准 PR
- ADR: `docs/adr/0002-object-segmentation.md`
- 目标: 在 Object Field 接口上接入第一种语义/实例对象分组路径。
- 实施:
  - 新增预计算 2D mask manifest 接口。
  - 支持 mask `rect` 和 boolean `.npy` mask。
  - 通过相机 pose 将 Gaussian 投影到 2D mask，聚合多视角 votes。
  - 输出仍为 Object Field，并可导出 ObjGauss PLY with `object_id`。
- 范围外:
  - 不在本 PR 中运行 SAM / CLIP 模型。
  - 不新增模型权重或深度学习依赖。
- 验收:
  - 2D mask votes 能改变 Object Field labels。
  - hard labels 可导出为现有 viewer 可读的 `object_id` PLY。
- 验证:
  - `uv run --extra dev pytest`: 12 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `af825f8`。

### OBJFIELD-002: 引入 projection loss 训练循环

- 状态: done
- 类型: 标准 PR
- ADR: `docs/adr/0002-object-segmentation.md`
- 目标: 让 Object Field logits 可以通过 2D mask projection supervision 实际更新。
- 实施:
  - 新增 `train_object_field_from_votes`。
  - 使用 multi-view mask votes 生成 Gaussian-level targets。
  - 用 NumPy softmax cross-entropy 更新 `object_logits`。
  - CLI `objgauss object-field vote-masks` 输出训练后 field、summary 和可选 PLY。
- 范围外:
  - 不实现完整 differentiable 3DGS render loss。
  - 不实现 NeRF Lego 的 3DGS 训练器。
- 验收:
  - projection loss 下降。
  - Object Field labels 按 mask supervision 改变。
- 验证:
  - `uv run --extra dev pytest`: 12 passed。
  - `uv run objgauss object-field inspect-nerf outputs/assets/training/nerf-synthetic-lego`: 400 frames，缺图 0，无效 pose 0。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `af825f8`。

### OBJFIELD-001: 建立 Object Field 最小训练骨架

- 状态: done
- 类型: 标准 PR
- ADR: `docs/adr/0002-object-segmentation.md`
- 目标: 基于 NeRF Lego 和现有 Gaussian PLY，建立 soft object-slot 的文件格式、指标和 CLI 骨架。
- 实施:
  - 新增 `ObjectField`，保存 `object_logits: (N, K)`。
  - 支持从 KMeans baseline warm start 为软 object-slot 分布。
  - 支持导出 hard `object_id` PLY，继续复用前端对象预览。
  - 支持检查 NeRF-style `transforms_*.json`、图像引用和 pose 矩阵。
- 范围外:
  - 不实现 SAM / CLIP / Gaussian Grouping。
  - 不实现真实 3DGS / Object Field 联合训练。
  - 不把 object-slot 控制接入 Spark shader。
- 验收:
  - `objgauss object-field init` 能从 Gaussian PLY 生成 `.npz` soft field。
  - `objgauss object-field export` 能输出带 `object_id` 的 PLY。
  - `objgauss object-field inspect-nerf` 能检查 NeRF Lego 训练输入。
- 验证:
  - `uv run --extra dev pytest`: 10 passed。
  - `uv run objgauss object-field inspect-nerf outputs/assets/training/nerf-synthetic-lego`: 400 frames，缺图 0，无效 pose 0。
  - `uv run objgauss object-field init public/samples/plush_objects.ply --output /tmp/plush_object_field.npz --slots 6 --smoothness`: 281498 gaussians，6 active slots。
- 完成 commit: `2962af4`。

### ASSET-001: 建立 Demo/训练素材转换管线

- 状态: done
- 类型: 标准 PR
- ADR: `docs/adr/0003-asset-ingestion.md`
- 目标: 将至少一个 Demo 素材源和一个训练素材源拉通到本地目录规范。
- 实施:
  - Demo: `polyhaven-school-chair-1k`，Poly Haven CC0 School Chair 1K glTF。
  - 训练: `nerf-synthetic-lego`，NeRF 官方示例 Lego 多视角数据。
- 范围外:
  - 不在本 PR 中实现 mesh -> 多视角渲染 -> 3DGS 训练。
  - 不提交下载后的大素材。
- 验收:
  - `objgauss assets list --pullable` 能看到 Plush、Poly Haven School Chair、NeRF Synthetic Lego。
  - Poly Haven 拉取输出 glTF、bin、textures 和 manifest。
  - NeRF 拉取输出 `outputs/assets/training/nerf-synthetic-lego/` 和 manifest。
  - Demo 样例当前明确为 mesh 输入源，尚不能直接由现有 3DGS viewer 打开。
- 验证:
  - `uv run --extra dev pytest`: 7 passed。
  - `uv run objgauss assets list --pullable`: 通过。
  - `uv run objgauss assets pull polyhaven-school-chair-1k`: 5 files。
  - `uv run objgauss assets pull nerf-synthetic-lego`: 805 files。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `9c88666`。

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
- 完成 commit: `f4aa2f1`。

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
