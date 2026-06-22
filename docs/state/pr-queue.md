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

### MASK-002: 生成 NeRF Lego 多 slot 真实 2D color mask manifest

- 状态: done
- 类型: 标准 PR
- ADR: `docs/adr/0002-object-segmentation.md`
- 目标: 将 Lego demo 内部的真实 2D color mask 生成逻辑提升为可复用 CLI，让语义线索生成不再只藏在 demo 命令里。
- 实施:
  - 新增 `objgauss masks from-nerf-rgba-colors`。
  - 从 NeRF Synthetic Lego RGBA 图片生成 `yellow`、`red`、`dark`、`other` 四类 boolean `.npy` masks。
  - 输出沿用现有 mask manifest，可直接被 `objgauss object-field vote-masks` 消费。
  - `lego-alpha-closure` 复用同一 mask 生成逻辑。
- 范围外:
  - 不声称该规则等价于 SAM / CLIP 实例语义分割。
  - 不改变 Object Field 文件格式。
  - 不完成 NeRF Lego 的完整 3DGS optimization 训练。
- 验收:
  - 真实 NeRF Lego 多视角 RGBA 可生成多 slot mask manifest。
  - 该 manifest 可直接监督 Object Field logits 并导出 `object_id` PLY。
- 验证:
  - `uv run objgauss masks from-nerf-rgba-colors outputs/assets/training/nerf-synthetic-lego --output outputs/masks/nerf-lego-rgba-colors/mask-manifest.json --split train --max-frames 8 --alpha-threshold 16`: 8 frames，32 masks，foreground_pixels=209891。
  - `uv run objgauss object-field vote-masks outputs/demos/lego-rgba-cli-smoke/lego_proxy_raw.ply --field outputs/demos/lego-rgba-cli-smoke/object_field_initial.npz --masks outputs/masks/nerf-lego-rgba-colors/mask-manifest.json --output outputs/demos/lego-rgba-cli-smoke/object_field_cli_masks.npz --summary-output outputs/demos/lego-rgba-cli-smoke/cli-mask-training-summary.json --ply-output outputs/demos/lego-rgba-cli-smoke/lego_cli_mask_objects.ply --iterations 80 --learning-rate 1.0 --colorize`: supervised_gaussians=3423，loss 1.386294 -> 0.390825，active_slots=4。
  - `uv run --extra dev pytest`: 16 passed。
- 完成 commit: `5302cfe`。

### ACCEPT-001: 固化一键闭环总验收命令

- 状态: done
- 类型: 标准 PR
- 目标: 把当前阶段最终目标压成一个可重复运行的本地总验收命令。
- 实施:
  - 新增 `npm run acceptance:demo`。
  - 命令重新生成并验证 Plush v1 closure。
  - 命令重新生成并验证 NeRF Lego proxy closure。
  - 命令最后执行 `npm run audit:demo`，用浏览器验收两个闭环素材卡片。
- 范围外:
  - 默认不下载素材；需要时通过 `npm run acceptance:demo -- --pull-assets` 显式拉取。
  - 不声称 NeRF Lego proxy 是完整 3DGS optimization 输出。
  - 不替代后续 SAM / CLIP 端到端语义分割。
- 验收:
  - 单条命令能重建、验证并浏览器审计两个闭环样例。
- 验证:
  - `npm run acceptance:demo`: passed，Plush loss 1.791760 -> 1.201637；Lego proxy loss 1.386294 -> 0.538856；浏览器审计两个素材均通过。
  - `uv run --extra dev pytest`: 15 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `81f1d0b`。

### UI-AUDIT-001: 固化闭环 demo 浏览器交互验收

- 状态: done
- 类型: 标准 PR
- 目标: 将“打开页面、看到闭环样例、选择/隔离/删除对象”的浏览器验证固化为可重复命令。
- 实施:
  - 新增 `npm run audit:demo`。
  - 命令自动启动临时 Vite server，并用 Playwright 加载 `ObjGauss v1 闭环样例` 和 `NeRF Lego 闭环代理样例`。
  - 检查页面身份、素材卡片、真实 splat canvas 非空、点云编辑 canvas 非空、对象选择、只看所选和预览删除状态。
- 范围外:
  - 不新增仓库内截图报告文件。
  - 不替代 Python 侧 manifest verifier。
- 验收:
  - 两个闭环素材都能通过浏览器交互验证。
- 验证:
  - `npm run audit:demo`: passed，Plush splatPixels=38400 / editPixels=60294；Lego splatPixels=78043 / editPixels=55465；两个样例 delete preview 均更新为 1。
  - 截图证据: `/tmp/objgauss-audit-plush-v1-closure-local.png`、`/tmp/objgauss-audit-nerf-lego-alpha-closure-local.png`。
  - `uv run --extra dev pytest`: 15 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `f3e5c62`，截图输出补充 commit: `f1b1190`。

### VERIFY-002: 固化 NeRF Lego proxy 闭环验收检查器

- 状态: done
- 类型: 标准 PR
- 目标: 让 `LEGO-001` 的“真实多视角数据 + 2D mask + Object Field + 前端对象编辑”有独立机器验收命令。
- 实施:
  - 新增 `objgauss demo verify-lego-alpha-closure`。
  - 重新读取 `lego-alpha-closure-manifest.json`、NeRF 源图像、mask `.npy`、proxy `.splat`、Object Field `.npz`、导出 PLY、public assets 和 `src/assetLibrary.js`。
  - 检查 mask manifest 至少使用多视角、Object Field shape 匹配、projection loss 下降、导出 PLY 含 `object_id`。
- 范围外:
  - 不声称 NeRF Lego proxy 是完整 3DGS optimization 结果。
  - 不替代浏览器交互测试。
- 验收:
  - 真实 Lego proxy demo verifier 通过。
  - 测试覆盖 verifier CLI。
- 验证:
  - `uv run objgauss demo verify-lego-alpha-closure`: passed=true，17 项检查全部通过。
  - `uv run --extra dev pytest`: 15 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `7a250d9`。

### LEGO-001: 生成 NeRF Lego 闭环代理样例

- 状态: done
- 类型: 标准 PR
- 目标: 把真实 NeRF Lego 多视角数据、真实 2D mask manifest、Object Field 投票和前端对象编辑压到同一个可加载样例里。
- 实施:
  - 新增 `objgauss demo lego-alpha-closure`。
  - 从 NeRF Lego RGBA 图片和 camera pose 采样生成轻量 Gaussian proxy PLY。
  - 写出 `lego_proxy.splat`，可走前端真实 Splat renderer。
  - 基于真实 2D 图像颜色规则生成 4-slot color mask manifest，并通过 `vote-masks` 更新 Object Field。
  - 导出 `lego_v1_objects.ply`，并复制到 `public/samples/lego_alpha_v1_objects.ply`。
  - 前端素材库新增 `NeRF Lego 闭环代理样例`。
- 范围外:
  - 不声称完成 NeRF Lego 的完整 3DGS optimization 训练。
  - 不声称 SAM / CLIP 已接入。
- 验收:
  - 同一个 Lego proxy scene 有 `.splat`、Object Field、真实 2D masks、`object_id` PLY。
  - 前端可以加载新素材并执行对象选择、隔离、删除预览。
- 验证:
  - `uv run objgauss demo lego-alpha-closure --max-frames 12 --sample-stride 8 --iterations 120`: 5696 gaussians，4 objects，12 frames，48 masks，loss 1.386294 -> 0.538856。
  - `uv run objgauss stats public/samples/lego_alpha_v1_objects.ply`: object_id 0/1/2/3 counts = 736 / 581 / 1787 / 2592。
  - Playwright + system Chrome: `NeRF Lego 闭环代理样例` 可加载，canvas nonBackground=78043，可执行只看所选和预览删除。
  - `uv run --extra dev pytest`: 15 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `db3441a`。

### VERIFY-001: 固化 v1 闭环验收检查器

- 状态: done
- 类型: 标准 PR
- 目标: 让当前阶段最终目标有可重复的机器验收命令，不再只依赖人工口头检查。
- 实施:
  - 新增 `objgauss demo verify-v1-closure`。
  - 重新读取 `v1-closure-manifest.json`、真实 `.splat`、mask manifest、Object Field `.npz`、导出 PLY、public viewer copy 和 `src/assetLibrary.js`。
  - 检查 Object Field shape 是否匹配 Gaussian 数量和对象数，projection loss 是否下降，导出 PLY 是否含 `object_id`。
- 范围外:
  - 不替代浏览器交互测试。
  - 不声称 SAM / CLIP 已接入。
- 验收:
  - 真实 Plush v1 closure 产物 verifier 通过。
  - 测试覆盖 verifier CLI。
- 验证:
  - `uv run objgauss demo v1-closure --iterations 80`: 281498 gaussians，6 objects，loss 1.791760 -> 1.201637。
  - `uv run objgauss demo verify-v1-closure`: passed=true，13 项检查全部通过。
  - `uv run --extra dev pytest`: 14 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `b6236bd`。

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
