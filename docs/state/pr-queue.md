# ObjGauss PR 队列

> 最近更新: 2026-06-23

## 队列规则

- 一次只执行一个 PR。
- 重大变更先补 ADR，Owner 确认后执行。
- 每个 PR 完成后更新验证结果和完成 commit。

## Ready

### BENCH-001: Stabilize safe-2000 balanced benchmark

- 状态: ready-for-owner-confirmation
- 类型: 标准 PR
- 目标: 将 safe-2000 balanced 8-frame / 4-slot SAM candidate 纳入可复现 benchmark/runbook，减少手工命令和 ignored outputs 依赖。
- 范围外: 不重新训练 Splatfacto；不提交 checkpoint、SAM checkpoint 或大体积训练输出。
- 验收:
  - 一条命令或 manifest 可重跑 balanced SAM -> register-output -> emergence curve。
  - benchmark summary 记录 frames、masks、object_id counts、ARI、OES 和 render occlusion effect。
  - 失败时输出缺失输入和生成命令。

## Done

### ACCEPT-002: Browser audit for balanced Splatfacto sample

- 状态: done
- 类型: 标准 PR
- 目标: 在获得本地服务 / Playwright 提权授权后，重跑当前 `NeRF Lego 训练输出样例` 的浏览器验收。
- 实施:
  - 先尝试 `npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5188`，Vite dev server 因系统 inotify watcher 上限 `ENOSPC` 崩溃。
  - 改用 `npm run preview -- --port 5188 --strictPort` 服务已构建的 `dist/` 静态产物，避免 dev watcher。
  - 使用 `npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5188/` 完成同一浏览器验收流。
- 验证:
  - Browser plugin: 当前 MCP 搜索未暴露 Browser 工具，按 Playwright fallback 执行。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5188/`: passed。
  - audit result: splatPixels=3256，editPixels=74388，visibleAfterIsolate=126686，deletedObjects=1。
  - screenshot: `/tmp/objgauss-audit-nerf-lego-trained-output-local.png`。
- 完成 commit: `<pending>`.

### SEG-003: Multi-view SAM supervision for Splatfacto candidates

- 状态: done
- 类型: 重大变更
- 目标: 为 safe-2000 Splatfacto candidate 生成更多 NeRF Lego SAM views，并补 slot balancing / 多视角一致性检查，降低 2-frame supervision 导致的 object slot 不平衡。
- 范围外: 不继续盲目增加 Splatfacto 训练步数；不提交 SAM checkpoint、训练 checkpoint 或大体积训练输出。
- 已实施:
  - `objgauss masks from-nerf-sam` 新增 `--max-area-fraction`，用于过滤过大的 SAM masks；默认 1.0 保持兼容。
  - 生成并比较 8-frame / 8-slot unfiltered、8-frame / 8-slot filtered、8-frame / 4-slot filtered 多个 safe-2000 SAM supervision 变体。
  - 当前最佳候选为 8-frame / 4-slot / `max_area_fraction=0.3`，已登记到本机 `public/samples/nerf_lego_trained.*`。
- 验收:
  - 生成 8-frame SAM manifest，frames=8，masks=27，mask_pixels=664780，slots=4。
  - safe-2000 上 `training register-output` 的 object_id 分布不再出现接近空槽，counts=126686/40747/34682/53679。
  - 与 2-frame SAM baseline 比较：supervised_gaussians=70025，effective_slots=3.509020，ARI=0.468745，render_occlusion_effect_score=0.195308。
  - 浏览器 audit 已通过，splatPixels=3256，editPixels=74388，visibleAfterIsolate=126686，deletedObjects=1。
- 验证:
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "nerf_sam" -q`: 2 passed。
  - `uv run --extra dev pytest`: 33 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5188/`: passed。
- 完成 commit: `9e22765`.

### TRAIN-003C: Higher-quality NeRF Lego Splatfacto candidate

- 状态: done
- 类型: 重大变更
- 目标: 在不影响本机交互的资源窗口内，将 safe-500 candidate 推进到更高质量 NeRF Lego Splatfacto 训练结果，并形成质量验收记录。
- 实施:
  - 使用 Nerfstudio Splatfacto 2000 iterations、`vis=tensorboard`、CPU image cache、0.5 camera scale 和 `MAX_JOBS=2` 完成 resource-safe candidate。
  - 复用 `/tmp/objgauss-cuda13` CUDA wrapper 和已有 `gsplat` JIT cache，避免重新触发长时间编译。
  - 导出 `outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply`，255794 / 255795 Gaussian 通过 opacity filter。
  - 使用 `training register-output` 将 safe-2000 登记为本机 ignored `NeRF Lego 训练输出样例` public sample。
  - 生成 safe-500 vs safe-2000 emergence report：`/tmp/objgauss-lego-splatfacto-safe-500-vs-2000-report.html`。
  - 修正 `SplatViewport` 的 fog 为随 splat bounding box 自适应，避免 denser / larger Splatfacto sample 在真实 splat 模式下被固定 fog 盖成背景。
- 范围外:
  - 不提交 `outputs/`、`public/samples/*.ply`、`public/samples/*.splat`、checkpoint、SAM checkpoint 或训练日志。
  - 不声称 safe-2000 已经是最终语义样例。
  - 不把该本机样例纳入默认 CI/public benchmark。
- 验收:
  - 记录训练配置、GPU 占用、耗时、输出路径和失败/恢复策略。
  - 导出的 Lego `splat.ply` 有可比较的 Gaussian 数、opacity filter 结果和前端外观截图。
  - 与 safe-500 candidate 比较 Object Field vote loss、emergence curve 和浏览器对象交互表现。
- 验证:
  - Splatfacto train: final train loss=0.022640，train PSNR=25.625683，gaussian_count=255795，TensorBoard GPU memory=941.883MB，train total time=18.331932s。
  - `ns-export gaussian-splat`: exported 255794 / 255795 Gaussian，PLY 大小约 61MB。
  - `training register-output`: `gaussians=255794`，`slots=8`，`supervised_gaussians=85349`，projection loss `4.467615 -> 0.288167`。
  - `objgauss stats public/samples/nerf_lego_trained_objects.ply`: object_id counts `84464/64455/111/14821/27910/23159/15867/25007`。
  - `objgauss object-field emergence`: assignment_confidence=0.819222，effective_slots=5.996345，spatial_compactness_score=0.980746，stability_ari=0.388430，partial OES=0.671132。
  - `objgauss object-field emergence-curve`: final projection loss=0.302584，render_occlusion_effect_score=0.123359。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5186`: passed，splatPixels=3256，editPixels=74388，visibleAfterIsolate=84464，deletedObjects=1。
  - `npm run audit:demo -- --port 5187`: passed，默认 3 个闭环样例仍可加载和交互。
  - `uv run --extra dev pytest`: 32 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `0751984`.

### TRAIN-003B: NeRF Lego Splatfacto public training sample

- 状态: done
- 类型: 重大变更
- 目标: 基于 TRAIN-003A runbook 跑一个比 100-step smoke 更有意义、但仍可在本机安全资源预算内完成的 Splatfacto 训练样例，并登记为前端 `NeRF Lego 训练输出样例`。
- 实施:
  - 使用 Nerfstudio Splatfacto 500 iterations、`vis=tensorboard`、CPU image cache、0.5 camera scale 和 `MAX_JOBS=2` 完成 resource-safe candidate。
  - 使用临时 `/tmp/objgauss-cuda13` CUDA wrapper 解决 uv CUDA wheel 只提供 `libcudart.so.13`、`gsplat` JIT 链接需要 `libcudart.so` 的问题。
  - 导出 `outputs/training/nerf-lego-splatfacto-long/export-safe-500-cpu-cache-v2/splat.ply`，47168 / 50000 Gaussian 通过 opacity filter。
  - 修正 `training register-output` 的 mask 分支，从 Gaussian 几何 warm start Object Field，避免全零 logits 在稀疏 mask vote 下坍缩到少数对象槽。
  - 登记 safe-500 PLY 到 `outputs/assets/gaussians/nerf-lego-trained-safe-500-cpu-cache-v2-warmstart/`，并生成本机 ignored public sample `public/samples/nerf_lego_trained.*`。
  - 扩展 `scripts/audit-demo.mjs`，允许单独验收 `nerf-lego-trained-output-local`，并等待 WebGL canvas 出现非背景像素。
  - 修正点云编辑视口 fog 为随点云尺度自适应，避免训练输出尺度较大时被固定 6-14 fog 完全盖成背景。
- 范围外:
  - 不提交 `outputs/`、`public/samples/*.ply`、`public/samples/*.splat`、checkpoint、SAM checkpoint 或训练日志。
  - 不声称 safe-500 是最终高质量 Lego reconstruction。
  - 不把该本机样例纳入默认 CI/public benchmark。
- 验收:
  - safe-500 Splatfacto candidate 导出 PLY 可被 ObjGauss 读取。
  - `training register-output` 生成 viewer `.splat`、Object Field、mask summary 和 8-slot `object_id` PLY。
  - 前端素材库卡片可加载训练输出样例并完成对象选择、隔离、删除预览。
- 验证:
  - `uv run objgauss training register-output ... --public-name nerf_lego_trained --iterations 160 --learning-rate 1.0`: `gaussians=47168`，`slots=8`，`supervised_gaussians=7676`，projection loss `3.047123 -> 0.321066`。
  - `uv run objgauss stats public/samples/nerf_lego_trained_objects.ply`: object_id counts `9127/5528/5661/5815/6073/3923/5995/5046`。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5182`: passed，splatPixels=408，editPixels=86577，visibleAfterIsolate=9127，deletedObjects=1。
  - `npm run audit:demo -- --port 5183`: passed，默认 3 个闭环样例仍可加载和交互。
  - `uv run --extra dev pytest`: 32 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `c3de487`.

### TRAIN-003A: Splatfacto smoke runbook and script

- 状态: done
- 类型: 标准 PR
- 目标: 将 TRAIN-001 的 NeRF Lego Splatfacto 100-step smoke 从“本机已有 outputs”固化成可复现 runbook / script，供 SEMANTIC benchmark 缺失输入时生成本地 handoff。
- 实施:
  - 新增 `scripts/train-splatfacto-smoke.mjs`，支持 `--dry-run`、`--status` 和 `--run`。
  - 新增 `npm run train:splatfacto:smoke`。
  - 新增 `docs/training/splatfacto-smoke.md`，记录 Nerfstudio Splatfacto 训练、`ns-export gaussian-splat`、SAM manifest、Object Field init / vote-masks、CUDA / `gsplat` 包要求和输出 contract。
  - `docs/benchmarks/semantic-smoke.json` 的 Splatfacto scene `prepare` 提示改为引用新脚本。
  - README 和 `docs/benchmarks/semantic-smoke.md` 指向 TRAIN-003A runbook。
  - 测试覆盖脚本 dry-run 输出的核心 pipeline。
- 范围外:
  - 不运行新的长训练，不登记 public sample。
  - 不提交 `outputs/`、checkpoint、SAM checkpoint 或训练日志。
  - 不替换当前 point-splat render probe。
- 验收:
  - `npm run train:splatfacto:smoke -- --dry-run` 输出完整训练 / 导出 / SAM / Object Field pipeline。
  - `npm run train:splatfacto:smoke -- --status` 能机器检查本地 smoke inputs / outputs 是否齐全。
  - `npm run acceptance:semantic` 仍通过。
- 验证:
  - `node scripts/train-splatfacto-smoke.mjs --dry-run --sam-checkpoint /tmp/sam-vit-b.pth --skip-benchmark`: passed。
  - `node scripts/train-splatfacto-smoke.mjs --status`: `status=ready missing=0`。
  - `npm run train:splatfacto:smoke -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth --skip-benchmark`: passed。
  - `uv run --extra dev pytest`: 32 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
  - `npm run acceptance:semantic`: passed，输出 `acceptance_semantic_benchmark=passed`。
- 完成 commit: `1647fa3`。

### SEMANTIC-007: Acceptance integration and missing-output runbook

- 状态: done
- 类型: 标准 PR
- 目标: 将 SEMANTIC-006 benchmark suite 接入 acceptance，并为缺失 ignored `outputs/` 的环境提供明确生成命令。
- 实施:
  - 新增 `scripts/acceptance-semantic-benchmark.mjs` 和 `npm run acceptance:semantic`。
  - `npm run acceptance:demo` 默认在闭环 demo 生成、verify 和浏览器 audit 后执行 SEMANTIC benchmark suite。
  - `acceptance:demo` 新增 `--skip-semantic-benchmark`，用于只跑 demo/UI 闭环验收。
  - `docs/benchmarks/semantic-smoke.json` 增加 `help` 和 per-scene `prepare` 命令。
  - `objgauss.emergence_benchmark` 对缺失输入输出 scene、缺失路径、prepare 命令和 runbook 路径。
  - 新增 `docs/benchmarks/semantic-smoke.md`，记录三场景输入、输出目录、缺失输入生成命令和 Splatfacto smoke 边界。
- 范围外:
  - 不把 `/tmp`、`outputs/`、checkpoint 或训练产物提交进 git。
  - 不把 Splatfacto 训练过程完全固化为 TRAIN-003；这里只记录 SEMANTIC benchmark 所需的本地 handoff。
  - 不替换 point-splat render probe 为 covariance-aware 3DGS / gsplat renderer。
- 验收:
  - `npm run acceptance:semantic` 能一键跑 strict semantic benchmark suite。
  - `npm run acceptance:demo` 默认包含 SEMANTIC benchmark suite。
  - 缺失 benchmark 输入时，CLI 错误信息包含 prepare 命令和 runbook。
- 验证:
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "emergence_benchmark"`: 2 passed。
  - `npm run acceptance:semantic`: passed，输出 `acceptance_semantic_benchmark=passed`。
  - `npm run acceptance:demo`: passed，输出 `acceptance_demo=passed`，并包含 SEMANTIC benchmark suite。
  - `uv run --extra dev pytest`: 31 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `c88f4ad`。

### SEMANTIC-006: Reproducible emergence benchmark suite

- 状态: done
- 类型: 标准 PR
- 目标: 将 SEMANTIC-005 的本地 smoke report 固化为 manifest 驱动的一键 benchmark suite，包含固定 scene、输出目录、summary、HTML report 和阈值检查。
- 实施:
  - 新增 `objgauss.emergence_benchmark`，读取 `object_emergence_benchmark` manifest，批量运行 emergence curves。
  - 新增 `objgauss object-field emergence-benchmark` CLI，输出 per-scene `curve.json` / `curve.csv`、顶层 `summary.json` 和 `report.html`。
  - 支持 manifest `defaults`、全局 / scene-specific `thresholds`、scene `max_frames`，以及 `--strict` 失败即报错。
  - 新增 `docs/benchmarks/semantic-smoke.json`，固定 Plush semantic、Lego alpha proxy、Lego Splatfacto smoke 三个本地 smoke scene。
- 范围外:
  - 不提交 `/tmp` 或 `outputs/` benchmark 产物。
  - 不生成缺失的 demo / training outputs；当前 suite 依赖本地已有 ignored outputs。
  - 不替换 point-splat render probe 为 covariance-aware 3DGS / gsplat renderer。
- 验收:
  - 一条命令可从 manifest 重新生成三场景 curve JSON/CSV、summary JSON 和 HTML report。
  - `--strict` 会检查 projection loss decrease、最小 points 和最小 render occlusion effect。
  - Summary 可机器判断每个 scene 和整体 suite 是否 passed。
- 验证:
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "emergence_benchmark or emergence_report or emergence_curve"`: 4 passed。
  - `uv run objgauss object-field emergence-benchmark docs/benchmarks/semantic-smoke.json --output-dir /tmp/objgauss-semantic-smoke-suite --strict`: passed=true，scenes=3；Plush semantic loss 1.386294 -> 1.346402，Lego alpha proxy loss 1.386294 -> 0.235765，Lego Splatfacto smoke loss 4.384474 -> 0.339695。
  - `uv run --extra dev pytest`: 30 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `1d9aa23`。

### SEMANTIC-005: Emergence benchmark report artifact

- 状态: done
- 类型: 标准 PR
- 目标: 将多个 `emergence-curve` JSON 聚合为可检查的 HTML/SVG benchmark 曲线报告，支持 2-3 个 scene 横向对比。
- 实施:
  - 新增 `objgauss.emergence_report`，生成静态 HTML/SVG 报告，不引入 matplotlib、plotly 或前端运行时依赖。
  - 新增 `objgauss object-field emergence-report` CLI，支持多个 curve JSON、重复 `--label` 和自定义 `--title`。
  - 报告包含 summary table 和 7 条曲线图：projection loss、assignment confidence、normalized entropy、ARI、spatial compactness、render occlusion effect 和 OES。
  - 本地 smoke 使用 Plush semantic、Lego alpha proxy、Lego Splatfacto smoke 三个场景 curve JSON 生成 `/tmp/objgauss-emergence-benchmark-report.html`。
- 范围外:
  - 不提交 `/tmp` 或 `outputs/` 中的报告产物。
  - 不固定正式 benchmark suite、阈值或 public dataset。
  - 不替换 point-splat render probe 为 covariance-aware 3DGS / gsplat renderer。
- 验收:
  - CLI 能读取一个或多个 `object_emergence_curve` JSON 并生成 HTML。
  - HTML 包含 summary table、SVG charts、scene labels 和 render occlusion metric。
  - 三个本地 scene 曲线可聚合为同一份 report artifact。
- 验证:
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "emergence_report or emergence_curve"`: 3 passed。
  - `uv run objgauss object-field emergence-report /tmp/objgauss-benchmark-plush-semantic.json /tmp/objgauss-benchmark-lego-alpha.json /tmp/objgauss-benchmark-lego-splatfacto.json --label plush-semantic --label lego-alpha-proxy --label lego-splatfacto-smoke --output /tmp/objgauss-emergence-benchmark-report.html --title "ObjGauss Emergence Benchmark Smoke"`: curves=3，charts=7。
  - `uv run --extra dev pytest`: 29 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `e42e66f`。

### SEMANTIC-004: Render occlusion delta probe

- 状态: done
- 类型: 标准 PR
- 目标: 将 SEMANTIC-003 的 occlusion benchmark 从 mask-vote loss proxy 升级为可复现的图像重渲染差分 probe。
- 实施:
  - 新增 `objgauss.render_probe`，从 mask manifest 读取相机位姿，执行 CPU point-splat/depth render probe。
  - `objgauss object-field emergence-curve` 默认输出 `render_occlusion_delta`，并保留 `mask_proxy_occlusion_delta` 作为对照。
  - CSV 新增 `render_occlusion_mean_delta_l1`、`render_occlusion_mean_relative_delta_l1`、`render_occlusion_mean_affected_fraction` 和 `render_occlusion_effect_score`。
  - 曲线内 Object Emergence Score 使用 render occlusion effect 补齐 occlusion component。
- 范围外:
  - 当前 probe 不是 covariance-aware 3DGS / gsplat renderer。
  - 不实现 gradient coherence probe。
  - 不新增曲线图 artifact 或多 scene benchmark。
- 验收:
  - Synthetic camera test 中删除 slot 后 render delta 大于 0。
  - Splatfacto smoke 曲线输出 `occlusion_delta_kind=point_splat_render_l1`。
  - JSON / CSV 同时保留 proxy occlusion 和 render occlusion。
- 验证:
  - `uv run --extra dev pytest`: 28 passed。
  - `uv run objgauss object-field emergence-curve outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply --field outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz --masks outputs/masks/nerf-lego-sam/mask-manifest.json --output /tmp/objgauss-lego-splatfacto-render-emergence-curve.json --csv-output /tmp/objgauss-lego-splatfacto-render-emergence-curve.csv --iterations 80 --learning-rate 1.0 --eval-every 20 --render-size 128`: points=5，projection_loss 4.384474 -> 0.308315，render_occlusion_mean_relative_delta_l1=0.124603，render_occlusion_effect_score=0.124603。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `073adcf`。

### SEMANTIC-003: Object emergence benchmark curves

- 状态: done
- 类型: 标准 PR
- 目标: 将 SEMANTIC-002 的单点观测指标扩展为随 mask-vote training iteration 变化的 benchmark 曲线。
- 实施:
  - 新增 `objgauss object-field emergence-curve` CLI。
  - 曲线采样 projection loss、assignment confidence、mean normalized entropy、effective slots、ARI to initial、ARI to previous、spatial compactness 和 mask-proxy occlusion delta。
  - 输出 JSON 曲线，并可选输出 CSV，便于后续画图。
  - `mask_vote_targets` 和 `projection_loss` 提升为公共 helper，保证曲线和 `vote-masks` 使用同一监督目标。
- 范围外:
  - 当前 occlusion delta 是 `mask_proxy_projection_loss`，不是 3DGS renderer 重渲染遮挡差分。
  - 不实现 gradient coherence probe。
  - 不新增图表 artifact 或前端可视化。
- 验收:
  - CLI 能从 Gaussian PLY、初始 Object Field 和 mask manifest 生成曲线 JSON。
  - CSV 至少包含 step、loss、entropy、ARI、compactness 和 mask-proxy occlusion delta。
  - 曲线训练后 projection loss 下降，occlusion proxy 可观测。
- 验证:
  - `uv run --extra dev pytest`: 28 passed。
  - `uv run objgauss object-field emergence-curve outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply --field outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz --masks outputs/masks/nerf-lego-sam/mask-manifest.json --output /tmp/objgauss-lego-splatfacto-emergence-curve.json --csv-output /tmp/objgauss-lego-splatfacto-emergence-curve.csv --iterations 80 --learning-rate 1.0 --eval-every 20`: points=5，projection_loss 4.384474 -> 0.308315，mask_proxy_occlusion_mean_delta_loss 1.428752 -> 1.927487。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `fe828f6`。

### SEMANTIC-002: Object Emergence observability metrics

- 状态: done
- 类型: 标准 PR
- 目标: 建立 ObjGauss v1 的 object emergence 最小观测系统，避免只靠视觉主观判断。
- 实施:
  - 新增 `objgauss/emergence.py`。
  - 新增 `objgauss object-field emergence` CLI。
  - 输出 assignment confidence、mean normalized entropy、effective slots、low/high entropy fraction。
  - 支持传入 Gaussian PLY 后计算空间紧致度。
  - 支持传入 reference Object Field 后计算 permutation-invariant ARI、matched label agreement 和 slot matching。
  - 输出 partial Object Emergence Score，显式标记 `occlusion_effect` missing 和 `gradient_coherence` unsupported。
- 范围外:
  - 不实现真实 renderer occlusion delta。
  - 不实现 gradient coherence probe。
  - 不改变训练逻辑、Object Field 文件格式或 SAM / CLIP slot 对齐。
  - 不声称 partial OES 已经证明 object emergence 完成。
- 验收:
  - CLI 能对单个 Object Field 输出结构化 emergence summary。
  - 有 cloud 时输出 spatial compactness；有 reference 时输出 ARI 和 label agreement。
  - permutation 后的相同分群应得到稳定性满分。
- 验证:
  - `uv run --extra dev pytest`: 26 passed。
  - `uv run objgauss object-field emergence outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_sam.npz --cloud outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply --reference outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz --output /tmp/objgauss-lego-splatfacto-emergence.json`: assignment_confidence=0.797826，effective_slots=7.323355，spatial_compactness_score=0.968811，stability_ari=0.642209，matched_label_agreement=0.825040，partial OES=0.772490。
- 完成 commit: `d217d81`。

### TRAIN-001: 训练 NeRF Lego Gaussian PLY

- 状态: done
- 类型: 重大变更
- 目标: 基于 `nerf-synthetic-lego` 得到可供 Object Field / mask voting 使用的 Gaussian PLY。
- 实施:
  - 使用 Nerfstudio `splatfacto` 和 `blender-data` dataparser 读取 `outputs/assets/training/nerf-synthetic-lego`。
  - 在 RTX 5060 Ti / PyTorch `2.12.1+cu130` 上完成 100-step CUDA smoke 训练。
  - 通过 `ns-export gaussian-splat` 导出 `outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply`。
  - 使用 `objgauss object-field init` 和 `vote-masks` 消费真实 SAM mask manifest，导出带 `object_id` 的训练 PLY。
- 环境结论:
  - `gsplat` PyPI JIT 需要本地 `nvcc`；系统未安装 CUDA toolkit 时，可用 `uv --with` 临时加入 `nvidia-cuda-nvcc==13.0.*`、`nvidia-cuda-cccl==13.0.*`、`nvidia-nvvm==13.0.*`、`nvidia-cuda-crt==13.0.*`。
  - 需要为 PyPI CUDA runtime 的 `libcudart.so.13` 提供临时 `libcudart.so` 链接路径。
  - 导出本地可信 checkpoint 时需设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 兼容 PyTorch 2.6+ 默认 `weights_only=True`。
- 范围外:
  - 不提交 `outputs/` 训练产物、checkpoint、TensorBoard event 或 SAM checkpoint。
  - 不声称 100-step smoke 是高质量 Lego reconstruction。
  - 不在本项中固化 runbook/script 或前端公共样例；后续 TRAIN-003 处理。
- 验收:
  - 产出 Lego `splat.ply`。
  - 可用 `objgauss object-field init` 和 `vote-masks` 跑通最小验收。
- 验证:
  - `ns-train splatfacto ... blender-data --data outputs/assets/training/nerf-synthetic-lego`: 100 iterations completed，checkpoint step `000000099`。
  - `ns-export gaussian-splat ...`: 导出 `splat.ply`，`uv run objgauss stats` 读取为 50000 gaussians。
  - `uv run objgauss object-field init ... --slots 8`: active_slots=8。
  - `uv run objgauss object-field vote-masks ... --masks outputs/masks/nerf-lego-sam/mask-manifest.json`: supervised_gaussians=8887 / 50000，supervised_fraction=0.177740，vote_conflict_fraction=0.268707，loss 4.384474 -> 0.308315，active_slots=8。
  - `uv run objgauss stats outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/lego_splatfacto_sam_objects.ply`: 50000 gaussians，包含 `object_id` 和 RGB 字段。
- 完成记录 commit: `39d4f6c`。

### SEG-002: 接入可选 SAM mask 生成器并完成 checkpoint 验收

- 状态: done
- 类型: 标准 PR
- ADR: `docs/adr/0002-object-segmentation.md`
- 目标: 从真实 NeRF Lego 图片生成当前 `vote-masks` 命令可消费的 SAM mask manifest，并证明 Object Field 能消费真实 SAM 输出。
- 实施:
  - `SEG-002A` 已新增 `objgauss masks from-nerf-sam`，运行时动态加载 `segment-anything`，不提交模型权重。
  - 使用本地 `sam_vit_b_01ec64.pth` / `vit_b` checkpoint 和 CUDA 生成 NeRF Lego 2 帧 SAM manifest。
  - 使用 8-slot Object Field 消费 SAM manifest，导出 `outputs/demos/lego-sam-smoke/lego_sam_objects.ply`。
- 范围外:
  - 不提交 checkpoint、torch 环境或 `outputs/` 产物。
  - 不声称 SAM 输出等价于高质量实例语义分割。
  - 不实现 CLIP 语义命名或跨视角 SAM slot 对齐；这些另行立项。
- 验收:
  - `objgauss masks from-nerf-sam` 输出真实 SAM manifest。
  - `objgauss object-field vote-masks` 可消费该 manifest，并输出带 `object_id` 的 PLY。
- 验证:
  - `segment_anything=ok`，`torch 2.12.1+cu130`，GPU: NVIDIA GeForce RTX 5060 Ti。
  - `objgauss masks from-nerf-sam ... --max-frames 2 --max-masks-per-frame 8 --min-area 64`: frames=2，masks=8，width=800，height=800，mask_pixels=1199536，slots=8。
  - `objgauss object-field vote-masks ... --iterations 80 --learning-rate 1.0`: supervised_gaussians=5567 / 5696，supervised_fraction=0.977353，vote_conflict_fraction=0.064308，loss 3.902681 -> 0.120758，active_slots=8。
  - `objgauss stats outputs/demos/lego-sam-smoke/lego_sam_objects.ply`: 5696 gaussians，`object_id` 8 slots。
- 实现 commit: `8c3c80e`。
- checkpoint 验收记录 commit: `18ac234`。

### VERIFY-004: 固化 mask vote quality audit

- 状态: done
- 类型: 标准 PR
- 目标: 在不引入 mIoU、SAM 质量结论或渲染差分指标的前提下，把“2D mask 投票到底监督了什么”写入可机器检查的 summary 和 demo verifier。
- 实施:
  - `MaskVoteResult.as_dict()` 输出 `vote_quality`。
  - `objgauss object-field vote-masks` CLI 打印监督覆盖率、冲突 Gaussian 数、冲突比例和 normalized target entropy。
  - `training_summary()`、外部训练输出登记和三个闭环 demo manifest 自动包含 vote quality audit。
  - `verify-v1-closure`、`verify-plush-semantic-closure`、`verify-lego-alpha-closure` 检查 `mask_vote_quality_audit_available`。
  - 单测覆盖 per-slot coverage、重叠 mask conflict fraction、normalized target entropy 和 verifier 输出。
- 范围外:
  - 不声称 mIoU / pixel accuracy 已完成；当前没有统一 ground truth。
  - 不评估 non-target object damage；当前对象编辑仍是点云 fallback，不是 splat shader 删除。
  - 不改变 Object Field 文件格式或 mask voting 优化逻辑。
- 验收:
  - vote-masks summary 能解释 supervised fraction、per-slot coverage 和 vote conflict。
  - 闭环 verifier 不只检查 loss 下降和 changed labels，还检查 vote quality audit 存在。
- 验证:
  - `uv run --extra dev pytest`: 24 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `6a32018`。

### SEMANTIC-001: 生成真实 3DGS + 2D 语义 mask 统一闭环样例

- 状态: done
- 类型: 标准 PR
- 目标: 把阶段目标压成一个统一真实 demo：真实 3DGS 场景、非 KMeans 2D mask guidance、Object Field、`object_id` 导出和前端对象级交互。
- 实施:
  - 新增 `objgauss demo plush-semantic-closure`。
  - 新增 `objgauss demo verify-plush-semantic-closure`。
  - 从原始 Plush Gaussian PLY 投影生成 `red-subject`、`straw-frame`、`dark-detail`、`other-surface` 四类 2D color masks。
  - 训练 Object Field logits，导出保留原始颜色的 `plush_semantic_objects.ply`。
  - 前端素材库新增 `Plush 2D 语义 Mask 闭环样例`。
  - `npm run acceptance:demo` 和 `npm run audit:demo` 纳入该样例。
  - `objgauss demo audit-v1-goal` 接受该 unified semantic demo 作为阶段目标完成证据。
- 范围外:
  - 不声称该颜色规则等价于 SAM / CLIP。
  - 不产出 NeRF Lego 的真实 3DGS training output。
  - 不实现对象级 splat shader；对象编辑仍走点云编辑 fallback。
- 验收:
  - 严格阶段审计通过，completion_blockers=`-`。
  - 浏览器可加载新样例并执行对象选择、隔离、删除预览。
- 验证:
  - `uv run objgauss demo plush-semantic-closure --iterations 80`: 281498 gaussians，4 objects，supervised_gaussians=281498，loss 1.386294 -> 1.345684。
  - `uv run objgauss demo verify-plush-semantic-closure`: passed=true，changed_gaussians=104403。
  - `uv run objgauss demo audit-v1-goal`: passed=true，current_evidence=unified。
  - `uv run --extra dev pytest`: 23 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
  - `npm run acceptance:demo`: passed，浏览器审计 3 个闭环素材。
- 完成 commit: `ae83594`。

### AUDIT-001: 固化 ObjGauss v1 阶段目标完成度审计

- 状态: done
- 类型: 标准 PR
- 目标: 用机器检查把“当前是否已经证明 ObjGauss v1 闭环成立”说清楚，避免把 proxy 或分散证据误报为最终完成。
- 实施:
  - 新增 `objgauss demo audit-v1-goal`。
  - 审计真实 3DGS splat、mask guidance、`object_id` 导出、前端素材注册、固定复现命令和统一真实训练 demo。
  - 默认严格模式未完成会失败；`--allow-incomplete` 用于输出当前证据报告。
- 范围外:
  - 不生成真实 SAM checkpoint 结果。
  - 不生成真实 NeRF Lego 训练 Gaussian。
- 验收:
  - 当前仓库能报告 split evidence 已满足哪些要求，并明确剩余 blocker。
- 验证:
  - 初始验证 `uv run objgauss demo audit-v1-goal --allow-incomplete`: passed=false，completion_blockers=`unified_real_3dgs_mask_demo_available`。
  - `SEMANTIC-001` 后严格验证 `uv run objgauss demo audit-v1-goal`: passed=true，completion_blockers=`-`。
  - `uv run --extra dev pytest`: 21 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
  - `npm run acceptance:demo`: passed。
- 完成 commit: `85943d4`。

### VERIFY-003: 固化 mask guidance 改变 Object Field 的验收

- 状态: done
- 类型: 标准 PR
- 目标: 让闭环验收不仅检查 loss 下降，还直接证明 2D mask guidance 改变了 Object Field 的 hard labels。
- 实施:
  - 新增 `object_field_label_delta`。
  - Plush v1 closure、NeRF Lego proxy closure、training register-output manifest 写入 `object_field_delta`。
  - `verify-v1-closure` 和 `verify-lego-alpha-closure` 增加 `mask_guidance_changed_object_field` 检查。
- 范围外:
  - 不改变 Object Field 文件格式。
  - 不解决 SAM checkpoint / 真实 3DGS 训练产物缺口。
- 验收:
  - `npm run acceptance:demo` 输出并检查 changed_gaussians。
- 验证:
  - `uv run --extra dev pytest`: 19 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
  - `npm run acceptance:demo`: passed；Plush changed_gaussians=196457，Lego proxy changed_gaussians=4960。
- 完成 commit: `e5e5154`。

### SEG-002A: 接入可选 SAM automatic mask manifest 生成器

- 状态: done
- 类型: 标准 PR
- ADR: `docs/adr/0002-object-segmentation.md`
- 目标: 在不提交模型权重、不强制安装深度学习依赖的前提下，提供仓库内 SAM mask manifest 生成入口。
- 实施:
  - 新增 `objgauss masks from-nerf-sam`。
  - 运行时动态加载 `segment-anything`，要求用户显式提供本地 checkpoint。
  - 对 NeRF-style RGBA 图片运行 SAM automatic mask generator，并输出现有 `vote-masks` manifest。
  - slot 按单帧 mask 面积排序为 `sam-area-rank-*`，每个 mask 写为 boolean `.npy`。
- 范围外:
  - 不提交或下载 SAM 权重。
  - 不默认安装 `segment-anything` / torch。
  - 不实现 CLIP 语义命名或跨视角 SAM slot 对齐。
  - 未在本机用真实 SAM checkpoint 跑小场景。
- 验收:
  - 命令可发现，且无 checkpoint 时不会伪造结果。
  - manifest 生成逻辑有 fake SAM generator 测试覆盖。
- 验证:
  - `uv run objgauss masks from-nerf-sam --help`: passed。
  - `uv run --extra dev pytest`: 18 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
  - `npm run acceptance:demo`: passed。
- 完成 commit: `8c3c80e`。

### TRAIN-002: 固化外部 3DGS 训练输出接入

- 状态: done
- 类型: 标准 PR
- 目标: 让成熟 3DGS 训练器产出的 `point_cloud.ply` / `.splat` 能直接进入 ObjGauss v1 闭环。
- 实施:
  - 新增 `objgauss training register-output`。
  - 支持读取 `.ply` / `.splat`，写出标准 `gaussians.ply` 和 viewer `.splat`。
  - 支持传入 mask manifest 后直接跑 Object Field projection supervision，输出 `object_field_trained.npz`、summary 和带 `object_id` 的 PLY。
  - 前端素材库预留 `NeRF Lego 训练输出样例` 卡片，公共文件名为 `nerf_lego_trained.splat` / `nerf_lego_trained_objects.ply`。
- 范围外:
  - 不在本 PR 中自研或封装实际 3DGS trainer。
  - 不声称 smoke 使用的 proxy PLY 是真实训练输出。
  - 不替代 `TRAIN-001` 的真实 NeRF Lego 训练产物。
- 验收:
  - 外部 Gaussian PLY 可被登记为 ObjGauss 训练输出。
  - 给定真实 mask manifest 后可导出 `object_id` PLY。
- 验证:
  - `uv run objgauss training register-output outputs/demos/lego-alpha-closure/lego_proxy_raw.ply --asset-id nerf-lego-trained-output-local --output-dir outputs/assets/gaussians/nerf-lego-register-smoke --dataset outputs/assets/training/nerf-synthetic-lego --masks outputs/masks/nerf-lego-rgba-colors/mask-manifest.json --public-name nerf_lego_trained --iterations 80 --learning-rate 1.0 --no-public-copy`: 5696 gaussians，slots=4，supervised_gaussians=4806，loss 1.386294 -> 0.375765。
  - `uv run --extra dev pytest`: 17 passed。
  - `npm run build`: 通过，仍有 bundle size warning。
- 完成 commit: `721ac49`。

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
