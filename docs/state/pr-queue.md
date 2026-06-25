# ObjGauss PR 队列

> 最近更新: 2026-06-25

## 队列规则

- 一次只执行一个 PR。
- 重大变更先补 ADR，Owner 确认后执行。
- 每个 PR 完成后更新验证结果和完成 commit。

## Ready

### RENDER-004: WebGPU tile-based Gaussian renderer

- 状态: ready
- 类型: 重大变更 / 渲染器
- 目标: 以 WebGPU tile binning + per-tile accumulation 作为 ObjGauss object-aware Gaussian renderer 终局架构。
- 设计: `docs/adr/0005-webgpu-tile-renderer.md`
- 下一步:
  - Renderer-native object picking 暂不迁移：Spark `SplatMesh.raycast` 目前只返回 `distance/object/point`，没有 splat index / object id；后续要么等待/扩展 Spark intersection metadata，要么继续使用已审计的 `hover-confirm-v1` screen-space pick。
- 验收底线:
  - WebGPU 可用环境中暴露 `data-renderer="webgpu-tile"` 和 `data-object-filter="gpu-object-state-buffer"`。
  - 不支持 WebGPU 或初始化失败时明确 fallback 到当前 `Gaussian OIT 编辑`，不静默伪装成功。
  - 隔离 / 删除后 `visibleCount` 与 object-state 一致，并记录 `tileOverflowCount`。

## In Progress

当前无进行中 PR。

## Done

### RENDER-ROUTE-015: Synthetic 1M browser runtime proof

- 状态: done / synthetic-1m-browser-runtime
- 类型: 标准 PR / WebGPU C-path scale runtime observability
- 目标: 在 1M storage/edit budget 和 281k 真实场景 browser transition 之间补一条真实浏览器上传运行证据，证明 WebGPU Tile C-path 能通过 UI 上传 synthetic 1M PLY 并完成对象级交互。
- 已实施:
  - 新增 `scripts/audit-webgpu-synthetic-1m-runtime.mjs` 和 `npm run audit:webgpu-synthetic-1m-runtime`。
  - Gate 默认在 `/tmp` 生成 `1,000,000` Gaussian / `256` object 的 binary PLY，不提交任何生成数据。
  - Gate 通过真实文件上传控件加载 PLY，使用 `uploaded-ply-splat-source=off` 直接进入 WebGPU Tile 对象编辑路线，检查 first frame、tile overflow、对象选中、隔离、删除和三段 rAF frame pacing。
  - `src/App.jsx` 新增 `uploaded-ply-splat-source=off` 诊断参数；默认行为不变，普通上传仍保留 `splatSource`。
  - `audit:webgpu-cpath-readiness` 已纳入 synthetic 1M runtime 子步骤，并把剩余 gap 改为真实训练 1M scene runtime 和 sustained FPS SLA。
  - `audit:renderer-route-contract`、renderer readiness matrix 和 WebGPU runbook 已登记该 gate。
- 结论:
  - 当前本机 synthetic 1M runtime 通过：uploadedGaussians=`1000000`，tileReferences=`1709862`，tileOverflow=`0`。
  - 最新 C-path readiness 通过：scale1m=`passed`，edit1m=`passed`，headedTransition=`passed`，browserRuntime1m=`passed` / proof=`proven-synthetic-upload`。
  - 这证明 synthetic 1M browser runtime shape；仍不证明真实训练 1M scene、论文级视觉质量或 sustained FPS SLA。
- 验证:
  - `node --check scripts/audit-webgpu-synthetic-1m-runtime.mjs`: passed。
  - `npm run audit:webgpu-synthetic-1m-runtime -- --gaussians 50000 --objects 32 --port 5395 --output-dir /tmp/objgauss-webgpu-synthetic-1m-runtime-small --frame-count 10 --max-mean-frame-ms 200 --max-p95-frame-ms 300 --min-approx-fps 1 --allow-failures`: expected failed only on 1M suite count; browser upload / WebGPU interaction passed。
  - `npm run audit:webgpu-synthetic-1m-runtime -- --port 5395 --output-dir /tmp/objgauss-webgpu-synthetic-1m-runtime`: passed；min approx FPS=`15.429`。
  - `npm run audit:webgpu-cpath-readiness -- --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness`: passed；realTrainedBrowserRuntime1m=`not-proven`，fpsSla=`not-proven`。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-014: Headed WebGPU frame pacing smoke

- 状态: done / frame-pacing-smoke
- 类型: 标准 PR / WebGPU C-path browser responsiveness observability
- 目标: 在 storage / queue timing 和 headed presentation transition 之外，补一条真实浏览器 `requestAnimationFrame` frame-pacing smoke，持续观察当前 C-path 场景在 idle、isolate、delete 后是否仍保持基本交互响应。
- 已实施:
  - 新增 `scripts/audit-webgpu-frame-pacing.mjs` 和 `npm run audit:webgpu-frame-pacing`。
  - Gate 默认覆盖 Lego proxy 与 Plush semantic，启动 fixed-port `5395` preview，强制 `spark-filtered-edit=off`，进入 WebGPU Tile 编辑路径。
  - 每个 scene 采样 idle / after-isolate / after-delete 三段 rAF intervals，并检查 mean frame、p95 frame、long-frame ratio 和 approximate FPS。
  - Report 输出 `/tmp/objgauss-webgpu-frame-pacing/summary.json`、`summary.md` 和每个 scene 的 `/tmp` 截图。
  - `audit:renderer-route-contract`、renderer readiness matrix 和 WebGPU presentation runbook 已登记该 gate。
- 结论:
  - 当前本机 frame pacing smoke 通过：2/2 scenes，largestGaussians=`281498`，maxTileReferences=`1190026`。
  - Plush semantic 大场景通过：min approx FPS=`26.471`，max mean frame=`37.777ms`，max p95 frame=`16.8ms`，max long-frame ratio=`0.013`。
  - 该 gate 证明当前真实场景 headed browser C-path 响应 smoke；它不是 sustained renderer FPS benchmark，也不是真实 1M browser runtime proof。
- 验证:
  - `node --check scripts/audit-webgpu-frame-pacing.mjs`: passed。
  - `npm run audit:webgpu-frame-pacing -- --port 5395 --output-dir /tmp/objgauss-webgpu-frame-pacing`: passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-013: WebGPU C-path readiness evidence aggregator

- 状态: done / cpath-readiness-report
- 类型: 标准 PR / WebGPU C-path evidence reporting
- 目标: 将 1M static scale budget、1M edit-cost budget 和真实 headed browser object transition 证据收敛到一条本地命令，明确哪些已证明、哪些仍是 gap，避免把 budget audit 误读成 1M browser FPS。
- 已实施:
  - 新增 `scripts/audit-webgpu-cpath-readiness.mjs` 和 `npm run audit:webgpu-cpath-readiness`。
  - Readiness audit 默认执行 build、`audit:webgpu-scale-budget`、`audit:webgpu-edit-cost-budget` 和 fixed-port `5395` headed `audit:webgpu-presentation-transition`。
  - Report 输出 `/tmp/objgauss-webgpu-cpath-readiness/summary.json` 和 `summary.md`，合并 scale / edit / presentation transition evidence，并单独列出 remaining gaps。
  - `audit:renderer-route-contract`、renderer readiness matrix 和 WebGPU headless/presentation runbook 已登记该命令。
- 结论:
  - 当前本机 readiness 通过：1M scale budget max / total storage=`122.07 / 173.24 MiB`。
  - 1M edit-cost row 通过：object-state edit update=`4 KiB`，full storage=`173.24 MiB`，pixel candidate upper bound=`8.192G`。
  - Headed browser transition 当前覆盖到 Plush semantic `281498` Gaussians，2/2 scenes 保持 `postDelete="webgpu-tile":"gpu-object-state-buffer"`，max queue done=`1836.8ms`。
  - 报告仍明确标注 `browserRuntime1m=not-proven` 和 `fpsSla=not-proven`；本 PR 不是 1M interactive SLA。
- 验证:
  - `node --check scripts/audit-webgpu-cpath-readiness.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run audit:webgpu-cpath-readiness -- --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness`: passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-012: Headed WebGPU presentation object-transition gate

- 状态: done / presentation-transition-gate
- 类型: 标准 PR / WebGPU C-path browser transition observability
- 目标: 在 RENDER-ROUTE-010 的 first-frame presentation smoke 之上，补一条 headed full canvas WebGPU object-transition gate，证明对象选中、隔离、删除三段交互不会切到 Spark fallback，而是留在 WebGPU Tile C-path 并暴露 storage timing / object-state checksum。
- 已实施:
  - `audit-demo --webgpu-object-transition` 现在支持 full WebGPU probe；offscreen readback checksum 断言仍只在 `--webgpu-probe offscreen-readback` 下启用。
  - 新增 `scripts/audit-webgpu-presentation-transition.mjs` 和 `npm run audit:webgpu-presentation-transition`。
  - Gate 默认覆盖 Lego proxy 与 Plush semantic，启动 fixed-port `5395` preview，强制 headed WebGPU full canvas runtime，执行 canvas select -> isolate -> delete。
  - Report 输出 `/tmp/objgauss-webgpu-presentation-transition/summary.json` 和 `summary.md`，记录 first frame、selected object、visible counts、post-delete renderer、initial / isolate / delete timing、object-state checksum 和截图。
  - Product renderer acceptance 现在在 presentation performance 后追加 `WebGPU presentation object transition`；CI profile 不跑该 headed gate。
  - `audit:renderer-route-contract` 和 renderer readiness docs 已登记该 gate。
- 结论:
  - 当前本机 fixed-port headed transition gate 通过：Lego proxy 与 Plush semantic 均在删除后保持 `postDelete="webgpu-tile":"gpu-object-state-buffer"`。
  - Plush semantic 大场景通过：281498 Gaussians、tileReferences=1190026、selected object=`0`、visibleAfterIsolate=`177095`、visibleAfterDelete=`104403`、maxUpdateMs=`178.9`、maxQueueDoneMs=`1847.4`。
  - Product profile 内复跑同一 transition gate 也通过：2/2 scenes，largestGaussians=281498，maxUpdateMs=`180.3`，maxQueueDoneMs=`2456.9`，仍在当前 `2500ms` smoke envelope 内。
  - 该 gate 证明 headed browser 下 C-path object edit transition 可用；它仍不是 FPS benchmark，也不是 1M interactive SLA。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-presentation-transition.mjs`: passed。
  - `node --check scripts/acceptance-renderer-profile.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run acceptance:renderer-product -- --dry-run --skip-build --output-dir /tmp/objgauss-renderer-product-transition-dry-run`: passed；steps=4，包含 WebGPU presentation transition。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-presentation-transition -- --port 5395 --output-dir /tmp/objgauss-webgpu-presentation-transition`: passed；2/2 scenes，largestGaussians=281498，maxUpdateMs=178.9，maxQueueDoneMs=1847.4。
  - `npm run acceptance:renderer-product -- --output-dir /tmp/objgauss-renderer-product-transition-profile`: passed；steps=5。
  - `git diff --check`: passed。

### RENDER-ROUTE-011: Product renderer acceptance includes WebGPU presentation smoke

- 状态: done / product-presentation-acceptance
- 类型: 标准 PR / renderer acceptance profile hardening
- 目标: 将 RENDER-ROUTE-010 的 headed WebGPU full canvas presentation smoke 接入显式 product / demo renderer acceptance，同时保持默认 CI profile fresh-clone/headless 友好，避免把本机 WebGPU presentation 依赖误放进 CI。
- 已实施:
  - `scripts/acceptance-renderer-profile.mjs` 的 product profile 现在按 `Renderer route contract -> Build viewer -> WebGPU presentation performance smoke -> Spark commercial route acceptance` 执行。
  - Product profile 的 WebGPU presentation step 写出 `/tmp/.../webgpu-presentation-performance/summary.json|md`，并固定使用 `5395`。
  - Spark commercial route 在 product profile 中改为消费顶层 build 后的 `dist/`，通过 `--skip-build` 避免重复构建。
  - 新增 `--skip-webgpu-presentation-performance` 诊断开关；默认 product profile 不跳过。
  - `audit:renderer-route-contract` 现在检查 product profile 包含 `audit:webgpu-presentation-performance`。
  - Renderer readiness matrix 已说明 product/local 承担 headed presentation，CI 继续使用 offscreen / headless C-path gates。
- 结论:
  - 当前本机 `acceptance:renderer-product` 通过 4 步：route contract、build、WebGPU presentation performance、Spark commercial route。
  - Product profile 内 WebGPU presentation 通过：2/2 scenes，largestGaussians=281498，maxUpdateMs=181.3，maxQueueDoneMs=1742.3。
  - CI profile 未被误加 headed WebGPU presentation；dry-run 只保留 route contract 等 CI 可控步骤。
- 验证:
  - `node --check scripts/acceptance-renderer-profile.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run acceptance:renderer-product -- --dry-run --skip-build --output-dir /tmp/objgauss-renderer-product-presentation-dry-run`: passed；steps=3，包含 WebGPU presentation performance smoke。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run acceptance:renderer-product -- --output-dir /tmp/objgauss-renderer-product-presentation-profile`: passed；steps=4。
  - `npm run acceptance:renderer-ci -- --dry-run --skip-build --skip-webgpu-tile-smoke --skip-webgpu-scale-budget --skip-webgpu-edit-cost-budget --skip-splat-index-mapping --skip-native-route --output-dir /tmp/objgauss-renderer-ci-presentation-split-dry-run`: passed；steps=1。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-010: WebGPU headed presentation performance smoke gate

- 状态: done / presentation-performance-smoke
- 类型: 标准 PR / WebGPU C-path browser presentation observability
- 目标: 在 runtime timing smoke 之外补一条 headed browser full canvas presentation gate，持续证明 WebGPU C-path 不只是 compute / offscreen readback 可跑，也能真实提交到 canvas、产出非空首帧、写截图，并在当前 timing envelope 内完成小场景和 281k 大场景的 presentation。
- 已实施:
  - 新增 `scripts/audit-webgpu-presentation-performance.mjs` 和 `npm run audit:webgpu-presentation-performance`。
  - Presentation smoke 默认启动 fixed-port `5395` preview，强制 `audit-demo` 进入 WebGPU full runtime，并用 `--webgpu-presentation-only` 只验收 presentation path，避免 Spark residual / object mask stress 干扰该 gate。
  - `audit-demo` 新增 `webGpuPresentationOnly` 开关；presentation-only 早退结果会保留 storage update / submit / queue done timing，并把截图写到 `/tmp/objgauss-audit-*-webgpu-presentation.png`。
  - Report 输出 `/tmp/objgauss-webgpu-presentation-performance/summary.json` 和 `summary.md`，按 asset 汇总 first-frame pixels、packed Gaussians、tile references、storage timing、device / queue 状态和截图路径。
  - `audit:renderer-route-contract`、renderer readiness matrix 和 WebGPU acceptance docs 已登记该 gate。
- 结论:
  - 当前本机 fixed-port headed smoke 通过：Lego proxy 5696 Gaussians、tileReferences=40389、firstFramePixels=253952、storage update=16.3ms、queue done=73.3ms；Plush semantic 281498 Gaussians、tileReferences=1190026、firstFramePixels=147456、storage update=180.3ms、queue done=1867.5ms。
  - 该 gate 证明 WebGPU full canvas presentation 在当前 headed desktop smoke envelope 内可用；它不是 FPS benchmark，也不是 1M interactive SLA。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-presentation-performance.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-presentation-performance -- --port 5395 --output-dir /tmp/objgauss-webgpu-presentation-performance`: passed；2/2 scenes，largestGaussians=281498，maxUpdateMs=180.3，maxQueueDoneMs=1867.5。
  - `git diff --check`: passed。

### RENDER-ROUTE-009: WebGPU runtime performance smoke gate

- 状态: done / runtime-performance-smoke
- 类型: 标准 PR / WebGPU C-path browser performance observability
- 目标: 在 RENDER-ROUTE-008 的 browser-visible timing telemetry 上补一条可复现的 runtime smoke envelope，持续记录 object edit transition 的 storage update、queue submit 和 queue done timing，避免 C-path 只停留在静态 budget 或单行日志证据。
- 已实施:
  - 新增 `scripts/audit-webgpu-runtime-performance.mjs` 和 `npm run audit:webgpu-runtime-performance`。
  - Performance smoke 默认复用 `audit:webgpu-offscreen-readback` 的 Lego proxy + Plush semantic object-transition suite，并读取 `/tmp/objgauss-webgpu-runtime-performance/offscreen-readback/summary.json`。
  - Report 输出 `/tmp/objgauss-webgpu-runtime-performance/summary.json` 和 `summary.md`，按 asset 汇总 packed Gaussians、tile references、initial / isolate / delete timing、最大 update 和最大 queue done。
  - 默认 smoke envelope：objectState update <= 300ms、full upload update <= 500ms、queue submit <= 25ms、queue done <= 2500ms、至少包含一个 >=250k Gaussian 大场景。
  - `audit:renderer-route-contract` 和 renderer readiness docs 已登记该 gate。
- 结论:
  - 当前本机 fixed-port browser smoke 通过：Lego proxy 5696 Gaussians、tileReferences=40389，最大 update=19.4ms、最大 queue done=68.9ms；Plush semantic 281498 Gaussians、tileReferences=1190026，最大 update=181ms、最大 queue done=1679.1ms。
  - 该 gate 证明当前 WebGPU C-path object edit 的浏览器 runtime timing 仍在 smoke envelope 内；它不是 FPS benchmark，也不是 1M interactive SLA。
- 验证:
  - `node --check scripts/audit-webgpu-runtime-performance.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-runtime-performance -- --port 5395 --output-dir /tmp/objgauss-webgpu-runtime-performance`: passed；2/2 scenes，largestGaussians=281498，maxUpdateMs=181，maxQueueDoneMs=1679.1。
  - `git diff --check`: passed。

### RENDER-ROUTE-008: WebGPU runtime timing telemetry gate

- 状态: done / runtime-timing-telemetry
- 类型: 标准 PR / WebGPU C-path browser audit contract
- 目标: 将 RENDER-ROUTE-006 的 objectState-only 更新事实继续升级为浏览器 runtime 可观测 timing：storage update 耗时、queue submit CPU 耗时、queue done 等待耗时。
- 已实施:
  - `WebGpuTileViewport.jsx` 新增 `data-webgpu-storage-update-ms`、`data-webgpu-frame-submit-ms` 和 `data-webgpu-queue-done-ms`。
  - storage timing 包住真实 full-upload / objectState-only `queue.writeBuffer` 路径；submit timing 包住 `device.queue.submit()`；queue done timing 来自 `queue.onSubmittedWorkDone()`。
  - `scripts/audit-demo.mjs` 在 WebGPU route 初始帧、isolate transition 和 delete transition 中检查 timing 为有限非负数。
  - `scripts/audit-demo.mjs` 在 browser audit 输出中写出 `storageTiming`、`storageTimingAfterIsolate` 和 `storageTimingAfterDelete`，使日志 / report 能直接看到 update mode 与耗时。
  - `scripts/audit-demo.mjs` 新增 `waitForWebGpuStorageUpdate`，在 object isolate / delete 后等待 storage checksum 改变和 timing settled，再读取 telemetry。
  - `scripts/audit-webgpu-offscreen-readback.mjs` 解析 storage timing 输出，并对 isolate / delete transition 建立 timing gate；isolate 只接受 `object-state-only`，delete 接受 `object-state-only` 或 `full-upload`。
  - Isolate transition 仍要求 `object-state-only`；delete transition 允许 `object-state-only` 或 `full-upload`，因为删除预览可能同时从对象色切回源色，导致静态 color buffers 改变并触发正确的 full-upload fallback。
  - `audit:renderer-route-contract` 将 runtime timing DOM attrs 和 browser audit helper 纳入 C-path contract。
- 结论:
  - C-path object edit 现在不仅能证明“只写 objectState 小 buffer”，还能在浏览器 audit 中记录这次 update / submit / queue done 的 runtime timing。
  - 这些字段是 observability，不是 FPS 或交互延迟 SLA；真实 1M FPS 仍需要 headed WebGPU performance run。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-offscreen-readback.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-runtime-timing-final`: passed，steps=6。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-offscreen-readback -- --assets nerf-lego-alpha-closure-local --port 5395 --output-dir /tmp/objgauss-webgpu-offscreen-readback-runtime-timing-lego`: passed；readback initial/isolate/delete checksums 均变化；`storageTimingAfterIsolate="object-state-only":17.8/0/68.8`，`storageTimingAfterDelete="object-state-only":16.3/0/68.9`。
  - `npm run audit:webgpu-offscreen-readback -- --assets plush-semantic-closure-local --port 5395 --output-dir /tmp/objgauss-webgpu-offscreen-readback-runtime-timing-plush`: passed；281498 Gaussians、tileReferences=1190026；isolate 为 `object-state-only`，delete 为 `full-upload` fallback：`storageTimingAfterDelete="full-upload":179.7/0.1/0`。

### RENDER-ROUTE-007: WebGPU edit cost budget audit

- 状态: done / edit-cost-budget-gate
- 类型: 标准 PR / renderer scale audit
- 目标: 在 RENDER-ROUTE-003 storage budget 和 RENDER-ROUTE-005 objectState-only 增量上传之上，补一个 fresh-clone-safe 的编辑更新成本 gate，量化 100k / 300k / 1M C-path profile 下对象编辑需要写多少数据、dispatch 多少 workgroups、以及 pixel resolve 可能扫描多少 tile-entry candidates。
- 已实施:
  - 新增 `scripts/audit-webgpu-edit-cost-budget.mjs` 与 `npm run audit:webgpu-edit-cost-budget`。
  - Audit 复用 `estimateWebGpuTileRuntimeStorage`、`webGpuAccumulationWorkgroups`、`webGpuComputeWorkgroups` 和 `webGpuPixelResolveWorkgroups`，不构造真实 1M points、不启动浏览器。
  - 默认三档 profile 与 scale budget 对齐：100k / 512px / 64 objects，300k / 384px / 128 objects，1M / 320px / 256 objects。
  - 报告同时输出 full first-upload MiB、objectState-only edit upload KiB、objectState upload share、accumulation / resolve / pixel workgroups、tile references、accumulation sample checks 和 pixel candidate-check upper bound。
  - `acceptance:renderer-ci` 默认加入 `WebGPU edit cost budget` 步骤；新增 `--skip-webgpu-edit-cost-budget` 诊断开关。
  - `audit:renderer-route-contract` 检查 npm script 和 renderer CI profile 均包含 edit-cost budget gate。
- 结论:
  - 默认三档预算均通过：100k edit upload=`1 KiB` / full=`18.15 MiB` / candidates=`0.614G`；300k edit upload=`2 KiB` / full=`49.19 MiB` / candidates=`2.150G`；1M edit upload=`4 KiB` / full=`173.24 MiB` / candidates=`8.192G`。
  - 这证明 C-path 对象编辑的兼容更新可以避免重传静态 geometry / colors / tile entries，并把成本集中到 objectState 小 buffer + compute redispatch；它不是真实浏览器 FPS 或视觉质量证明。
- 验证:
  - `node --check scripts/audit-webgpu-edit-cost-budget.mjs`: passed。
  - `node --check scripts/acceptance-renderer-profile.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run audit:webgpu-edit-cost-budget`: passed；100k / 300k / 1M rows passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-edit-cost-budget`: passed；steps=6。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### PORT-002: Dev and preview fixed port defaults

- 状态: done / fixed-dev-preview-port
- 类型: 标准 PR / local workflow ergonomics
- 目标: 将裸跑 `npm run dev` / `npm run preview` 也固定到 `127.0.0.1:5395 --strictPort`，避免手动看 Web 效果时 Vite 自动换端口。
- 已实施:
  - `package.json` 的 `dev` 和 `preview` scripts 均显式带 `--port 5395 --strictPort`。
  - `audit:renderer-route-contract` 的 fixed-port contract 扩展为同时检查 package scripts、browser audits 和 acceptance defaults。
  - 文档说明更新为 dev / preview / browser audit 都复用 `5395`；端口被占用时停止占用进程后重跑，不换新端口。
- 结论:
  - 默认本地使用方式统一为一个端口：`http://127.0.0.1:5395/`。
  - 特殊诊断仍可直接显式传端口给具体 audit 脚本，但日常流程不再漂移端口。
- 验证:
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-006: WebGPU object-state update telemetry gate

- 状态: done / object-state-update-telemetry
- 类型: 标准 PR / WebGPU C-path browser audit contract
- 目标: 将 RENDER-ROUTE-005 的 objectState-only 增量上传从内部 runtime 行为升级为可 DOM / browser audit 检查的事实，避免后续误回归成全量 storage upload 而不被验收发现。
- 已实施:
  - `WebGpuTileViewport.jsx` 新增 `data-webgpu-storage-update-mode` 与 `data-webgpu-storage-object-state-byte-size`。
  - 首次完整上传标记为 `updateMode="full-upload"`；reuse path 标记为 `updateMode="object-state-only"`，并暴露实际 object-state buffer byte size。
  - `scripts/audit-demo.mjs` 读取 storage update mode / object-state byte size，并在 WebGPU isolate / delete transition 后要求 `status="object-state-updated"`、`updateMode="object-state-only"` 且 object-state bytes > 0。
  - `audit:renderer-route-contract` 将 storage update telemetry 和 browser audit 字符串纳入 C-path contract。
- 结论:
  - WebGPU C-path 的对象编辑增量更新现在不仅由 Node smoke 证明，也能由浏览器 WebGPU object transition audit 验证。
  - 这仍不改变 Spark 商用默认路线；它加强的是 WebGPU 终局架构的编辑 runtime 可审计性。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-object-state-update-telemetry`: passed；steps=5。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-005: WebGPU objectState-only incremental upload

- 状态: done / object-state-incremental-upload
- 类型: 标准 PR / WebGPU C-path edit runtime
- 目标: 在 object-state-filtered tile list contract 之上，让 WebGPU runtime 在 tile/storage signature 兼容时复用既有 GPU storage bundle，只对 `objectState` storage buffer 执行 `queue.writeBuffer`，避免对象隐藏 / 隔离 / 删除时重传全量 Gaussian geometry / tile entries。
- 已实施:
  - `src/webgpuTileStorage.js` 新增 `webGpuTileStorageReuseSignature`、`canReuseWebGpuTileStorageBuffers` 和 `updateWebGpuTileObjectStateBuffer`。
  - Reuse signature 会比较所有 storage buffer allocation，并对静态输入 buffers（position / color / scale / object indices / tile counts / tile offsets / tile entries）做 checksum；`objectState` 与 GPU 输出 buffers 只检查布局大小。
  - `WebGpuTileViewport.jsx` 在 `tileListMode="object-state-filtered"` 且 compute path 会重写输出 buffers 时复用当前 storage bundle；兼容时只写 `objgauss-object-state`，否则仍安全重建完整 storage bundle。
  - `audit:webgpu-tile-smoke` 新增 fake device 断言：object-state tile list 的 isolate / delete 更新不会创建新 GPU buffers，只追加一次 `objgauss-object-state` write，且 storage checksum 更新到新 `tileSmoke` 描述。
  - `audit:renderer-route-contract` 将 objectState-only reuse / update helper 纳入 C-path 静态合约。
- 结论:
  - C-path 已从“tile list 语义稳定”推进到“对象编辑可走 objectState-only 增量上传”的 runtime contract。
  - 当 geometry、颜色模式、viewport、point size、tile layout 或其他静态输入变化时，runtime 仍会重建完整 bundle；这是正确 fallback，不是回归。
- 验证:
  - `node --check src/webgpuTileStorage.js`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-offscreen-readback.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；fake device 断言 objectState-only update 不创建新 buffers，只追加 `objgauss-object-state` write。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-object-state-incremental`: passed；steps=5。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-004: WebGPU object-state-filtered tile list mode

- 状态: done / object-state-tile-list-mode
- 类型: 标准 PR / WebGPU C-path edit runtime
- 目标: 让 WebGPU Tile runtime 具备“全量 tile list + object-state 过滤”的编辑模式，使对象隐藏 / 隔离 / 删除不必在 tile list 语义上依赖重新按可见对象裁剪；这是后续 object-state 小 buffer 更新和避免大场景全量重传的前置 contract。
- 已实施:
  - `src/webgpuTileSmoke.js` 新增 `WEBGPU_TILE_LIST_MODE_VISIBLE` 与 `WEBGPU_TILE_LIST_MODE_OBJECT_STATE`，默认保持 visible-only；object-state-filtered 模式会把隐藏对象也写入 compact tile list，但 CPU/GPU accumulation 继续由 `objectState` 跳过隐藏 Gaussian。
  - `App.jsx` 的 WebGPU runtime tile smoke 启用 `object-state-filtered` tile list；Spark 商用 route 与 Gaussian OIT fallback 不受影响。
  - `WebGpuTileViewport.jsx` 暴露 `data-webgpu-tile-list-mode` telemetry。
  - `audit:webgpu-tile-smoke` 新增断言：object-state-filtered 模式下，隔离 / 删除前后的 `tileCounts`、`tileOffsets`、`tileEntries` checksum 保持一致，而 `objectStateChecksum` 与 render resolve checksum 发生变化。
  - `audit:renderer-route-contract` 将 object-state-filtered tile list 与 DOM telemetry 纳入 C-path 静态合约。
- 结论:
  - C-path 现在有明确的 object-state edit runtime 语义：tile list 可以覆盖全量 Gaussian，编辑状态由 object-state buffer 决定。
  - 后续 RENDER-ROUTE-005 已在该 contract 上补齐 objectState-only `queue.writeBuffer` 增量上传。
- 验证:
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；新增 object-state-filtered tile list stability assertions。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-object-state-tile-list`: passed；steps=5。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-003: WebGPU 100k-1M scale budget audit

- 状态: done / scale-budget-gate
- 类型: 标准 PR / renderer scale audit
- 目标: 将 Phase 3 “WebGPU tile splatting 支持 100k-1M Gaussians”拆出一个 fresh-clone-safe 的 storage / tile-entry budget gate，避免 C-path 只证明小样例 smoke 而没有大规模预算证据。
- 已实施:
  - 新增 `scripts/audit-webgpu-scale-budget.mjs` 与 `npm run audit:webgpu-scale-budget`。
  - Audit 使用现有 `estimateWebGpuTileRuntimeStorage` 和 `editRendererContract`，不构造 1M 个 point、不启动浏览器，按 100k / 300k / 1M 三档 synthetic `tileSmoke` 检查完整 11-buffer WebGPU runtime layout。
  - 默认预算为单 storage buffer binding `128 MiB`、总 runtime storage `256 MiB`、storage buffers per shader stage `12`；1M 档使用 `320px` internal viewport 和每 Gaussian `32` 个 compact tile refs 的预算假设。
  - `acceptance:renderer-ci` 默认加入 `WebGPU scale budget` 步骤；新增 `--skip-webgpu-scale-budget` 诊断开关。
- 结论:
  - 默认三档预算均通过：100k=`9.16/18.15 MiB` max/total，300k=`32.04/49.19 MiB`，1M=`122.07/173.24 MiB`。
  - 这证明当前 C-path storage layout / compact tile entries / object-state buffer 在 1M 预算上没有先天超出常见 desktop WebGPU storage binding 约束；它不是 FPS、视觉质量或真实浏览器 1M runtime pass 的证明。
- 验证:
  - `node --check scripts/audit-webgpu-scale-budget.mjs`: passed。
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `node --check scripts/acceptance-renderer-profile.mjs`: passed。
  - `npm run audit:webgpu-scale-budget`: passed；100k / 300k / 1M rows passed，1M max/total=`122.07/173.24 MiB`。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run acceptance:renderer-ci -- --dry-run --skip-build --skip-webgpu-tile-smoke --skip-splat-index-mapping --skip-native-route`: passed；dry-run steps=2，包含 route contract 和 scale budget。
  - `npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-scale-budget-nonbrowser`: passed；steps=5，覆盖 route contract、build、WebGPU tile smoke、WebGPU scale budget 和 no-SH public sample index mapping。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-002: Renderer acceptance includes route contract

- 状态: done / acceptance-route-contract
- 类型: 标准 PR / renderer acceptance hardening
- 目标: 将 `audit:renderer-route-contract` 纳入默认 renderer acceptance profile，使 B -> C renderer 主线不再只是手动 audit，而是 CI / product route 验收的前置合约检查。
- 已实施:
  - `scripts/acceptance-renderer-profile.mjs` 在 `ci` 和 `product` profile 的第一步加入 `Renderer route contract`。
  - 新增 `--skip-route-contract` 诊断开关；默认不跳过。
  - `docs/rendering/renderer-readiness-matrix.md`、`docs/state/project-status.md` 同步说明 renderer acceptance 现在覆盖路线合约。
- 结论:
  - `npm run acceptance:renderer-ci` 现在先证明 WebGL Gaussian OIT fallback、WebGPU tile C-path、Spark commercial source route、browser telemetry 和 fixed `5395` port policy 仍一致，再继续 build / WebGPU smoke / index mapping / Spark native mask gate。
  - `npm run acceptance:renderer-product` 也会先跑同一合约，再进入 Spark commercial route acceptance。
- 验证:
  - `node --check scripts/acceptance-renderer-profile.mjs`: passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run acceptance:renderer-ci -- --dry-run --skip-build --skip-webgpu-tile-smoke --skip-splat-index-mapping --skip-native-route`: passed；dry-run steps=1，generated command=`npm run audit:renderer-route-contract`。
  - `npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-route-contract-nonbrowser`: passed；steps=4，覆盖 route contract、build、WebGPU tile smoke 和 no-SH public sample index mapping。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-ROUTE-001: B-to-C renderer route contract audit

- 状态: done / route-contract-audit
- 类型: 标准 PR / renderer architecture audit
- 目标: 将 B -> C renderer 主线从聊天记忆和长文档描述固化成可机器检查的静态合约，防止后续 UI / audit / renderer 改动把 WebGL Gaussian OIT fallback、WebGPU tile terminal path、Spark commercial source route 或 fixed port policy 拆散。
- 已实施:
  - 新增 `scripts/audit-renderer-route-contract.mjs` 与 `npm run audit:renderer-route-contract`。
  - 审计分三层检查：`B-webgl-gaussian-oit` 验证 `ShaderMaterial` screen-space Gaussian kernel、half-float weighted OIT resolve 和 GPU object-state texture；`C-webgpu-tile` 验证 ADR、storage buffers、tile entries / offsets、object-state buffer、compute accumulation 和 pixel resolve；`bridge-route-contract` 验证 `App` route boundary、browser audit telemetry、npm acceptance commands、Spark pick contract 和 fixed `5395` browser audit 默认端口。
  - 报告输出 `/tmp/objgauss-renderer-route-contract/summary.{json,md}`，失败时以非零退出码阻断。
- 结论:
  - 这条 audit 证明当前仓库的 renderer 路线仍是：Spark 负责商用源色查看 / source-color edit route，Gaussian OIT 作为 WebGL fallback/debug 过渡层，WebGPU tile renderer 作为 C-path 终局架构。
  - 这不是视觉质量 gate；WebGPU/Spark 视觉残差、真实 occlusion、边界质量和高质量商用编辑仍由对应 browser / benchmark audits 继续约束。
- 验证:
  - `node --check scripts/audit-renderer-route-contract.mjs`: passed。
  - `npm run audit:renderer-route-contract`: passed，16/16 checks。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### PORT-001: Browser audit fixed port defaults

- 状态: done / fixed-port-defaults
- 类型: 标准 PR / local browser audit ergonomics
- 目标: 将本地 browser audit / acceptance 默认端口统一到 fixed `5395`，避免每条命令不断换端口；保留显式 `--port` / `--native-port` / `--trained-port` override 能力。
- 已实施:
  - `audit:demo`、Spark route / pick / feather / reconstruct audits、WebGPU desktop / coverage / depth / offscreen audits 的 `DEFAULT_PORT` 均收敛到 `5395`。
  - `acceptance:demo`、`acceptance:spark-commercial-route`、`acceptance:renderer-*`、`acceptance:webgpu-headless` 的默认 browser audit ports 均收敛到 `5395`；多 step acceptance 仍顺序执行，每个子 audit 自己启动并停止 `--strictPort` preview。
  - `package.json` 的 `audit:spark-mask-feather` shortcut 改为 `--port 5395`。
  - 当前 runbook 中仍用于“照命令跑”的旧端口示例已更新到 `5395`：`docs/training/splatfacto-smoke.md`、`docs/rendering/webgpu-desktop-audit.md`、`docs/rendering/webgpu-headless-acceptance.md`、`docs/benchmarks/spark-filtered-edit.md`。
- 结论:
  - 新默认行为是“同一端口反复使用”：如果 `5395` 被占用，`--strictPort` 会失败，应停止占用该端口的本地 preview/audit 进程再重跑，而不是换到新端口。
  - 历史验证记录中的旧端口保留为历史事实，不作为当前 runbook 默认。
- 验证:
  - `node --check` changed browser audit / acceptance scripts: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --server-mode preview --skip-visual-residual`: passed，默认 preview URL=`http://127.0.0.1:5395/`。
  - `npm run audit:spark-native-mask-gate -- --assets nerf-lego-alpha-closure-local`: passed，默认 preview URL=`http://127.0.0.1:5395/`。
  - `npm run acceptance:renderer-ci -- --dry-run --skip-build --skip-webgpu-tile-smoke --skip-splat-index-mapping`: passed，generated command uses `--port 5395`。
  - `git diff --check`: passed。

### DEMO-005P: Spark native pick feasibility audit

- 状态: done / native-pick-feasibility-blocked
- 类型: 标准 PR / renderer UX + browser audit
- 目标: 评估 Spark-internal ray/object metadata path，判断能否从 `screen-space-object-pick-v1` 进一步收敛到 renderer-native object picking。
- 已实施:
  - `src/SplatViewport.jsx` 新增 `spark-native-pick-feasibility-v1` telemetry：暴露 raycast function / raycastable / sample hit / intersection keys / splat-index / object-id / object-filter-aware / object metadata / recommendation / blocker。
  - Native pick probe 只在 URL `spark-native-pick-probe=1` 下执行，probe 延后到 Spark frame 后，且只临时降低 `minRaycastOpacity` 后立即恢复，不改变产品渲染或默认选择路径。
  - 新增 `scripts/audit-spark-native-pick-feasibility.mjs` 与 `npm run audit:spark-native-pick-feasibility`，默认 fixed port `5395` + `--strictPort`，输出 `/tmp/objgauss-spark-native-pick-feasibility/summary.{json,md}`。
  - `package.json` 增加 audit 命令入口。
- 结论:
  - Lego proxy 删除预览下 Spark raycast 可用且能 hit：`raycast=true:true:hit:2`。
  - Intersection payload 只有 `distance,object,point`，没有 `splatIndex` / `objectId`，且 raycast 本身不证明 object opacity mask filter-aware。
  - Recommendation=`keep-screen-space-hover-confirm`，blocker=`raycast-intersection-missing-splat-index`。也就是说当前不能安全迁移到 renderer-native object picking；继续保留 `hover-confirm-v1` screen-space pick 是正确产品路径。
- 验证:
  - `node --check scripts/audit-spark-native-pick-feasibility.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-native-pick-feasibility -- --port 5395 --output-dir /tmp/objgauss-spark-native-pick-feasibility`: passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### DEMO-005O: Spark hover-confirm object picking UX

- 状态: done / hover-confirm-pick
- 类型: 标准 PR / renderer UX + browser audit
- 目标: 让 Spark canvas object selection 从 click-only 变成 hover candidate + click confirm，降低对象编辑中误选的不确定性，并把交互 contract 纳入现有 audit。
- 已实施:
  - `src/SplatViewport.jsx` 新增 `hover-confirm-v1` pick interaction：pointer hover 先计算候选 object、显示候选 marker；pointer up 重新确认同一 screen-space pick 并调用 `onSelectObject`。
  - Spark viewport root DOM 新增 hover pick telemetry：`data-spark-pick-interaction`、`data-spark-hover-pick-*` 和 `data-spark-hover-marker-visible`；原有 confirmed pick telemetry 保持兼容。
  - `src/styles.css` 新增候选 marker 样式：hover candidate 为 cyan dashed ring，confirmed selected marker 保持 yellow ring。
  - `scripts/audit-spark-pick-report.mjs` 默认端口改为 fixed `5395`，并改为每个点先 hover 再 click；gate 要求 hover hits 覆盖 confirmed hits、interaction=`hover-confirm-v1`、hover marker 和 selected marker 都可见。
  - `scripts/audit-demo.mjs` 同步验证 hover-confirm contract，常规 Spark canvas selection gate 不再只接受 click-only pick。
- 结论:
  - Lego proxy Spark pick report 通过：clicks=`6`，hoverHits=`6`，hits=`6`，validHits=`6`，markerHits=`6/6`，interaction=`hover-confirm-v1`，fixed URL=`http://127.0.0.1:5395/`。
  - `audit:demo` 单样例通过，日志输出 `sparkPick="screen-space-object-pick-v1":"hover-confirm-v1":"hit"...`，证明完整 Demo 流也消费了新 contract。
  - 这改善的是产品确认 UX；底层仍是 object-aware PLY metadata 的 screen-space CPU pick，不是 Spark-internal raycast。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-spark-pick-report.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-pick-report -- --port 5395 --max-clicks 6 --output-dir /tmp/objgauss-spark-pick-report-hover-confirm`: passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --server-mode preview --port 5395 --skip-visual-residual`: passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### DEMO-005N: Reviewed remap allowlist review runbook and manifest audit

- 状态: done / reviewed-allowlist-review-gate
- 类型: 标准 PR / renderer quality export gate
- 目标: 给 reviewed allowlist 增加人工评审 runbook / checklist，定义 target 进入 allowlist 前必须查看的 screenshots、hidden delta、non-target damage 和 owner approval 字段，并让 manifest 可机器校验。
- 已实施:
  - 新增 `docs/rendering/object-boundary-remap-review-runbook.md`，记录 fixed-port `5395` policy gate、证据复制、人工评审 checklist、拒绝条件和 approved target JSON 模板。
  - 新增 `scripts/lib/remap-reviewed-allowlist.mjs`，集中校验 reviewed allowlist manifest、approved target reviewer / owner approval / evidence / residual threshold 字段。
  - `scripts/export-object-boundary-remap-preview.mjs` 改为使用 shared reviewed allowlist validator；approved target 缺少人工审批字段时会被 export 拒绝，而不是只凭 `assetId/objectId` 应用 remap。
  - 新增 `scripts/audit-object-boundary-remap-reviewed-allowlist-manifest.mjs` 与 `npm run audit:object-boundary-remap-reviewed-allowlist-manifest`；默认要求 approved target 的 evidence report / screenshot 路径为仓库相对路径且存在。
  - 更新 synthetic positive fixture，使 `/tmp` allowlist 也携带完整 synthetic owner approval / evidence 字段，但不批准真实 repo target。
- 结论:
  - Committed reviewed allowlist 仍保持 `targets=[]`，manifest audit 通过：targets=`0`，evidence=`required`。
  - 真实三场景 policy-export 仍 raw candidates=`10012`，applied=`0`，blocked=`10012`。
  - Synthetic positive fixture 仍通过：Lego fixture target=`2`，applied=`402`，blocked=`741`。
  - 这一步把 reviewed allowlist 从“空安全阀”升级为“可审计人工批准流程”；仍没有任何真实 remap target 进入默认样例。
- 验证:
  - `node --check scripts/lib/remap-reviewed-allowlist.mjs`: passed。
  - `node --check scripts/export-object-boundary-remap-preview.mjs`: passed。
  - `node --check scripts/audit-object-boundary-remap-reviewed-allowlist.mjs`: passed。
  - `node --check scripts/audit-object-boundary-remap-reviewed-allowlist-manifest.mjs`: passed。
  - `npm run audit:object-boundary-remap-reviewed-allowlist-manifest`: passed。
  - `npm run audit:object-boundary-remap-reviewed-allowlist`: passed。
  - `npm run audit:object-boundary-remap-policy-export`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### DEMO-005M: Reviewed remap allowlist manifest and positive fixture

- 状态: done / reviewed-allowlist-gate
- 类型: 标准 PR / renderer quality export gate
- 目标: 增加 reviewed remap allowlist manifest / positive fixture gate，把“policy allowlist-candidate”和“人工批准可应用 target”分成两个可审计文件，再允许后续候选产物引用。
- 已实施:
  - 新增 `docs/rendering/object-boundary-remap-reviewed-allowlist.json`，mode=`object-boundary-remap-reviewed-allowlist-v1`，当前 `targets=[]`，默认 `keep-hard-mask`。
  - `scripts/export-object-boundary-remap-preview.mjs` 新增 `--reviewed-allowlist`，有 policy 时只应用同时满足 policy `allowlist-candidate` 与 reviewed allowlist 的 target；inline `--allow-target` 保留为诊断 override，并在 summary 中单独记录。
  - `scripts/audit-object-boundary-remap-residual.mjs` 新增 `--reviewed-allowlist` 透传到 preview export，让 browser QA 与 export gate 使用同一 allowlist contract。
  - 新增 `scripts/audit-object-boundary-remap-reviewed-allowlist.mjs` 与 `npm run audit:object-boundary-remap-reviewed-allowlist`，用 `/tmp` synthetic policy + synthetic reviewed allowlist 做正向 fixture，不批准真实 repo target。
  - `npm run audit:object-boundary-remap-policy-export` 默认读取 committed reviewed allowlist manifest。
- 结论:
  - 真实三场景 policy-export 通过，reviewed allowlist targetCount=`0`，raw candidates=`10012`，applied=`0`，blocked=`10012`。
  - Synthetic positive fixture 通过：Lego fixture target=`2`，raw candidates=`1143`，applied=`402`，blocked=`741`，证明只有 policy + reviewed allowlist 同时命中时才会 patch。
  - Denied target 负向 smoke 仍通过：`nerf-lego-alpha-closure-local:3` 即使作为 inline diagnostic target 传入，也因 policy `deny-hidden-increase` 保持 applied=`0`。
- 验证:
  - `node --check scripts/export-object-boundary-remap-preview.mjs`: passed。
  - `node --check scripts/audit-object-boundary-remap-residual.mjs`: passed。
  - `node --check scripts/audit-object-boundary-remap-reviewed-allowlist.mjs`: passed。
  - `npm run audit:object-boundary-remap-policy-export`: passed。
  - `npm run audit:object-boundary-remap-reviewed-allowlist`: passed。
  - `node scripts/export-object-boundary-remap-preview.mjs --assets nerf-lego-alpha-closure-local --policy /tmp/objgauss-object-boundary-remap-policy/remap-decision-policy.json --reviewed-allowlist docs/rendering/object-boundary-remap-reviewed-allowlist.json --allow-target nerf-lego-alpha-closure-local:3 --output-dir /tmp/objgauss-object-boundary-remap-reviewed-allowlist-denied-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### DEMO-005L: Policy-gated remap preview export

- 状态: done / policy-gated-export
- 类型: 标准 PR / renderer quality export gate
- 目标: 让 remap preview export / QA 消费 `remap-decision-policy`，只允许显式 allowlist target 进入后续候选产物；risky / review-only target 继续保留 hard mask。
- 已实施:
  - `scripts/export-object-boundary-remap-preview.mjs` 新增 `--policy` / `--allow-target asset_id:object_id`。
  - 有 policy 时，export 只 patch 同时满足 policy `allowlist-candidate` 和显式 `--allow-target` 的 `fromObject` remaps；所有其他 candidate remaps 保持原始 `object_id`。
  - Export summary 同时记录 raw candidate remaps、实际 applied remaps、policy blocked remaps、blocked decision breakdown、raw remap pairs 和 policy-gated remap pairs。
  - `scripts/audit-object-boundary-remap-residual.mjs` 新增 `--remap-policy` / `--allow-target` 透传到 preview 生成阶段，让后续 browser QA 可以消费同一 policy gate。
  - 新增 `npm run audit:object-boundary-remap-policy-export`，默认读取 `/tmp/objgauss-object-boundary-remap-policy/remap-decision-policy.json`。
- 结论:
  - 当前三场景 policy-export 通过，raw candidates=`10012`，applied remaps=`0`，blocked remaps=`10012`，输出仍可由 `objgauss stats` 读取。
  - 对 denied target 的负向 smoke 通过：显式传入 `--allow-target nerf-lego-alpha-closure-local:3` 后，Lego raw candidates=`1143`，applied remaps=`0`，blocked remaps=`1143`。
  - 这把“policy 分类”升级为“产物门禁”：没有人工显式 allow-target 时，后续 remap-preview PLY 保持 hard-mask assignment，不会误把 review-only / risky target 应用到候选产物。
- 验证:
  - `node --check scripts/export-object-boundary-remap-preview.mjs`: passed。
  - `node --check scripts/audit-object-boundary-remap-residual.mjs`: passed。
  - `npm run audit:object-boundary-remap-policy -- --skip-build --skip-visual-stats`: passed on fixed default port `5395` with Playwright fallback；Browser plugin absent；本地 preview server 需要提权，沙箱内监听端口会 `EPERM`。
  - `npm run audit:object-boundary-remap-policy-export`: passed。
  - `node scripts/export-object-boundary-remap-preview.mjs --assets nerf-lego-alpha-closure-local --policy /tmp/objgauss-object-boundary-remap-policy/remap-decision-policy.json --allow-target nerf-lego-alpha-closure-local:3 --output-dir /tmp/objgauss-object-boundary-remap-policy-export-denied-smoke`: passed。
  - `uv run objgauss stats /tmp/objgauss-object-boundary-remap-policy-export/nerf-lego-alpha-closure-local.remap-preview.ply`: passed，并与原始 Lego object counts 一致。

### DEMO-005K: Object boundary remap decision policy

- 状态: done / target-level-policy
- 类型: 标准 PR / renderer quality browser audit policy
- 目标: 基于 top-N remap target sweep 生成 target-level decision policy，把 promotable target、risky target 和 review-only target 分开；默认 hard mask 继续保持，禁止按全局 remap preview 直接替换 public samples。
- 已实施:
  - `scripts/audit-object-boundary-remap-residual.mjs` 在 promotion summary 后新增 `object-boundary-remap-decision-policy-v1`。
  - Policy 输出 `defaultAction="keep-hard-mask"`、`applyMode="manual-target-allowlist-only"`、global recommendation、decision counts、allowlist candidates、risky targets、review-only targets 和每个 target 的 residual/hidden-delta evidence。
  - `summary.md` 新增 Decision Policy 摘要；`writeReport` 额外写出 `remap-decision-policy.json` 和 `remap-decision-policy.md`。
  - 新增 `npm run audit:object-boundary-remap-policy`，默认复用 top-2 target sweep。
- 结论:
  - Lego policy smoke 通过，rows=`4`，comparisons=`2`，policy 文件已写入 `/tmp/objgauss-object-boundary-remap-policy-lego-smoke/remap-decision-policy.json`。
  - 默认三场景 route-only policy smoke 通过，rows=`12`，comparisons=`6`，policy 文件已写入 `/tmp/objgauss-object-boundary-remap-policy-route-smoke/remap-decision-policy.json`。
  - 当前 policy 明确不全局启用 remap：recommendation=`do-not-apply-remap-globally`，default action=`keep-hard-mask`。
  - 三场景 route-only policy 结果为 allowlist=`0`、risky=`2`、review-only=`4`；Lego target `3` 和 Plush target `0` 因 hidden delta 增加被归为 `deny-hidden-increase`。这说明 policy 会把“route/residual 通过”与“可进入 allowlist”分开。
- 验证:
  - `node --check scripts/audit-object-boundary-remap-residual.mjs`: passed。
  - `npm run audit:object-boundary-remap-policy -- --skip-build --skip-visual-stats`: passed on fixed default port `5395` with Playwright fallback；Browser plugin absent；本地 preview server 需要提权，沙箱内监听端口会 `EPERM`。

### DEMO-005J: Top-N object boundary remap target sweep

- 状态: done / top-n-target-evidence
- 类型: 标准 PR / renderer quality browser audit
- 目标: 将 remap browser residual gate 从每场景 top-1 candidate 扩展为 top-N target sweep，覆盖每个场景多个高风险 remap pair；只有 top-N sweep 和多场景 promotion table 都稳定时，才考虑 cleaned preview PLY 的默认化或公开样例替换。
- 已实施:
  - `scripts/audit-object-boundary-remap-residual.mjs` 新增 `--target-count`，按 `remapPairs` 的 top unique `fromObject` 为每个 scene 选择多个 target object。
  - 比较维度从 per-asset 升级为 `assetId + targetObjectId`，截图文件名也包含 target id，避免多 target 覆盖。
  - 对象选择改为严格匹配 target object；target 在 remap-preview 中缺失时 gate 失败，不再回退到第一个对象。
  - 新增 `npm run audit:object-boundary-remap-target-sweep`，默认 `--target-count 2`。
- 结论:
  - 默认 top-2 三场景 sweep 通过，rows=`12`，comparisons=`6`，skipped=`0`。
  - 所有 6 个 target case 都通过 route/residual 阈值，但只有 1/6 是 promotion candidate。
  - Negative evidence: Lego target `3` hidden delta=`+397`，Plush target `0` hidden delta=`+4085`，说明 sampled remap 可能让某些 target 删除更激进。
  - Aggregate recommendation=`do-not-promote-default-hard-mask`，promotion=`false`；下一步应做 target-level allowlist / policy，而不是全局默认启用 remap preview。
- 验证:
  - `node --check scripts/audit-object-boundary-remap-residual.mjs`: passed。
  - `npm run audit:object-boundary-remap-target-sweep -- --skip-build`: covered by the fixed-port `5395` policy sweep with Playwright fallback；Browser plugin absent；本地 preview server 需要提权，沙箱内监听端口会 `EPERM`。

### DEMO-005I: Multi-scene object boundary remap residual promotion table

- 状态: done / multi-scene-browser-evidence
- 类型: 标准 PR / renderer quality browser audit
- 目标: 将 remap browser residual gate 扩展到 Plush semantic 与 Poly Haven Chair commercial sample，形成多场景 promotion table；只有多场景残差和 non-target preservation 都稳定时，才考虑 cleaned preview PLY 的默认化或公开样例替换。
- 已实施:
  - `scripts/audit-object-boundary-remap-residual.mjs` 默认资产扩展为 `nerf-lego-alpha-closure-local,plush-semantic-closure-local,polyhaven-chair-commercial-demo-local`。
  - 新增 `minSceneCount`、可选 `--max-remap-samples`、skipped asset 记录和 aggregate promotion summary。
  - 浏览器 console 过滤补充上传 PLY source preload 阶段的已知 Spark worker 噪声 `Missing f_dc_0 property`；真正的编辑 route contract 仍要求 `spark-splat` / `ply-packed` / `object-opacity-texture-v1`。
- 结论:
  - 默认三场景 gate 通过，rows=`6`，comparisons=`3`，skipped=`0`。
  - Lego proxy: target object=`2`，hidden delta=`-49`，after residual=`0.999216/0.004332/0.019990`。
  - Plush semantic: target object=`1`，hidden delta=`-2786`，after residual=`0.999355/0.000070/0.000117`。
  - Poly Haven Chair: target object=`1`，hidden delta=`-29`，after residual=`0.999800/0.000048/0.000059`。
  - Aggregate recommendation=`do-not-promote-default-hard-mask`，promotion=`false`；当前证据说明 sampled remap preview 没有明显伤害 top candidate object 的删除预览，但还不能默认替换 public samples。
- 验证:
  - `node --check scripts/audit-object-boundary-remap-residual.mjs`: passed。
  - `npm run audit:object-boundary-remap-residual -- --output-dir /tmp/objgauss-object-boundary-remap-residual-multiscene --skip-build`: use fixed default port `5395`; Playwright fallback；Browser plugin absent；本地 preview server 需要提权，沙箱内监听端口会 `EPERM`。

### DEMO-005H: Object boundary remap browser residual gate

- 状态: done / browser-evidence-only
- 类型: 标准 PR / renderer quality browser audit
- 目标: 基于 sampled remap preview PLY，做 browser residual gate，对比原始 object-aware PLY 与 remap-preview PLY 的删除后 visual stats / non-target damage；默认 hard mask 继续保持，除非 browser residual gate 证明 cleanup 不伤害非目标对象。
- 已实施:
  - 新增 `scripts/audit-object-boundary-remap-residual.mjs`，自动生成 `/tmp` remap preview PLY，再用 Playwright 上传原始 PLY 与 remap-preview PLY。
  - Gate 强制 `spark-object-source=packed` / `spark-reconstruct-probe=1`，确保两个文件走同一 PLY-packed Spark object-mask route，变量只剩 sampled `object_id` remap。
  - 脚本删除 top remap candidate object，采集 before/after canvas visual stats、Spark route telemetry、object opacity mask stats 和截图，并输出 `summary.json` / `summary.md`。
  - 新增 `npm run audit:object-boundary-remap-residual`。
- 结论:
  - Lego proxy browser residual gate 通过：target object=`2`，original hidden=`1787`，remap-preview hidden=`1738`，少隐藏 `49` 个 Gaussian。
  - After-delete residual 保持在阈值内：coverage ratio=`0.999216`，luma delta=`0.004332`，chroma delta=`0.019990`。
  - recommendation=`browser-evidence-only`，不是 promotion；单场景证据不足以默认替换样例或宣称视觉质量改善。
- 验证:
  - `node --check scripts/audit-object-boundary-remap-residual.mjs`: passed。
  - `npm run audit:object-boundary-remap-residual -- --assets nerf-lego-alpha-closure-local --output-dir /tmp/objgauss-object-boundary-remap-residual-smoke --skip-build`: use fixed default port `5395`; Playwright fallback；Browser plugin absent；本地 preview server 需要提权，沙箱内监听端口会 `EPERM`。

### DEMO-005G: Object boundary remap preview export

- 状态: done / sampled-remap-preview
- 类型: 标准 PR / renderer quality experiment
- 目标: 基于 `DEMO-005F` 的 cleanup candidate report，生成保留原始 PLY 属性、仅 patch `object_id` 的 cleaned preview PLY，用于下一步 browser residual gate。
- 已实施:
  - 新增 `scripts/export-object-boundary-remap-preview.mjs`，读取 object-aware PLY，按 `object-boundary-remap-preview-v1` 采样邻域规则挑出被其他 object id 支配的 Gaussian，并只修改这些 sampled Gaussian 的 `object_id`。
  - 脚本支持 binary / ascii scalar vertex PLY，默认保留原始 PLY 字节和所有 SH / scale / rotation / color 属性，只 patch `object_id` 字段。
  - 新增 `npm run audit:object-boundary-remap-preview`，默认导出 Lego proxy preview 到 `/tmp/objgauss-object-boundary-remap-preview/summary.{json,md}` 和 `/tmp/objgauss-object-boundary-remap-preview/*.remap-preview.ply`。
- 结论:
  - Lego proxy 全量采样通过：gaussians=`5696`，sampled=`5696`，remapped=`1143`，estimated=`1143`，输出 PLY 可由 `objgauss stats` 读取。
  - Plush semantic 大场景采样 smoke 通过：gaussians=`281498`，sampled=`70375`，step=`4`，remapped=`5701`，estimated=`22804`，输出 PLY 可由 `objgauss stats` 读取。
  - 这仍是 sampled preview / export experiment；它没有证明删除后视觉质量改善，也没有进入默认 commercial route。
- 验证:
  - `node --check scripts/export-object-boundary-remap-preview.mjs`: passed。
  - `npm run audit:object-boundary-remap-preview`: passed。
  - `uv run objgauss stats /tmp/objgauss-object-boundary-remap-preview/nerf-lego-alpha-closure-local.remap-preview.ply`: passed。
  - `node scripts/export-object-boundary-remap-preview.mjs --assets plush-semantic-closure-local --max-remap-samples 80000 --output-dir /tmp/objgauss-object-boundary-remap-preview-plush`: passed。
  - `uv run objgauss stats /tmp/objgauss-object-boundary-remap-preview-plush/plush-semantic-closure-local.remap-preview.ply`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### DEMO-005F: Object boundary cleanup candidate report

- 状态: done / read-only-cleanup-candidates
- 类型: 标准 PR / renderer quality reporting
- 目标: 基于 feather candidate gate 的 `diagnostic-only` 结论，转向 object assignment 边界清理方向，输出哪些 object / Gaussian 子集值得做 remap review，而不是继续把 opacity feather 当默认视觉修复。
- 已实施:
  - `scripts/audit-object-mask-boundary.mjs` 在原 hard-mask boundary diagnostic 基础上新增 `object-boundary-cleanup-candidate-v1` 只读候选层。
  - 邻域采样现在会统计被另一个 `object_id` 支配的本地 Gaussian 子集，输出 `cleanupCandidateRatio`、`cleanupCandidateGaussianEstimate`、`cleanupDominantTargetObject`、`cleanupPriorityScore` 和 recommendation。
  - 新增 `npm run audit:object-boundary-cleanup`，默认覆盖 Lego proxy、Plush semantic 和 Poly Haven Chair commercial sample，并写 `/tmp/objgauss-object-boundary-cleanup/summary.{json,md}`。
- 结论:
  - 三场景 cleanup report 通过，assets=3，skipped=0。
  - Lego proxy 估算 cleanup candidates=`1138`，top object=`1 -> 3`，recommendation=`boundary-remap-review`。
  - Plush semantic 估算 cleanup candidates=`23335`，top object=`2 -> 0`，recommendation=`boundary-remap-review`。
  - Poly Haven Chair 估算 cleanup candidates=`983`，top object=`1 -> 0`，recommendation=`low-priority-boundary-cleanup`，说明商用 chair 当前更适合先维持 hard mask + 明确 no-inpaint copy。
- 验证:
  - `node --check scripts/audit-object-mask-boundary.mjs`: passed。
  - `npm run audit:object-mask-boundary -- --assets nerf-lego-alpha-closure-local --output-dir /tmp/objgauss-object-mask-boundary-regression`: passed。
  - `npm run audit:object-boundary-cleanup`: passed；assets=3，skipped=0。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### DEMO-005E: Spark mask feather candidate gate

- 状态: done / candidate-gate-diagnostic-only
- 类型: 标准 PR / renderer quality reporting
- 目标: 基于 UI toggle 和 sweep 报告，扩展更多 opacity / radius / commercial-chair variants，给出是否可默认启用 soft boundary 的机器判断。
- 已实施:
  - `scripts/audit-spark-mask-feather-sweep.mjs` 新增 candidate recommendations：按 variant 聚合 hard→feather comparison，输出 `promotionCandidate`、recommendation、mean/max coverage / luma / chroma deltas 和 score。
  - 脚本新增 `--skip-missing-assets`，三场景候选 gate 在缺少本地 generated public sample 时会明确跳过，而不是把 chair 缺失误报为 renderer failure。
  - 新增 `npm run audit:spark-mask-feather-candidates`，默认覆盖 Lego proxy、Plush semantic 和 Poly Haven Chair commercial sample；variants 为 `hard`、`feather55`、`feather70`、`feather55r035`。
- 结论:
  - 三场景四变体 visual-stats candidate gate 通过，rows=12、comparisons=9、skippedAssets=0。
  - 三个 feather candidate 全部为 `diagnostic-only`，没有一个满足 promotion criteria。
  - 当前 best score 是 `feather55r035`，但仍 `promotionCandidate=false`：mean coverage delta=`0.006688`，max coverage delta=`0.010209`，max chroma delta=`0.001512`。
  - 结论是默认 hard mask 继续保持；feather 只保留为显式诊断开关。下一步应该看 object boundary cleanup / remap，而不是继续把 opacity feather 当默认修复。
- 验证:
  - `node --check scripts/audit-spark-mask-feather-sweep.mjs`: passed。
  - `npm run audit:spark-mask-feather-candidates -- --skip-build --skip-visual-stats --port 5391 --output-dir /tmp/objgauss-spark-mask-feather-candidates-telemetry`: passed；rows=12。
  - `npm run audit:spark-mask-feather-candidates -- --skip-build --port 5392 --output-dir /tmp/objgauss-spark-mask-feather-candidates`: passed；rows=12，recommendations=3。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### DEMO-005D: Spark mask feather UI toggle

- 状态: done / explicit-soft-boundary-toggle
- 类型: 标准 PR / renderer UX + browser audit
- 目标: 将 Spark object mask feather 从 URL-only 诊断能力接入 Web UI，让用户能在对象编辑里直接切换 hard mask 与 soft-boundary 候选，同时保持默认关闭。
- 已实施:
  - `src/App.jsx` 新增 `柔化删除边界` checkbox，默认关闭；URL `spark-object-mask-feather=on` 仍可初始化为开启。
  - App root 暴露 `data-spark-object-mask-feather-control="ui-v1"`、enabled、opacity 和 radius，状态面板新增 `边界柔化`。
  - `src/SplatViewport.jsx` 接收显式 `objectMaskFeathering` prop；未传入时仍兼容原 URL 参数。
  - `scripts/audit-spark-mask-feather-sweep.mjs` 新增 `--control ui|url`，可通过 Playwright 真正点击 UI toggle 验证 telemetry，同时保留 URL sweep 兼容路径。
- 结论:
  - UI toggle 可用，但它仍是显式诊断开关；默认 route 仍是 hard mask。
  - Lego UI audit 中 hard row 为 `feather="off"`，点击 toggle 后 `feather="spatial-neighbor-feather-v1":2932:0.07:0.55:0.901044/0.560784`。
- 验证:
  - `node --check scripts/audit-spark-mask-feather-sweep.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-mask-feather-sweep -- --assets nerf-lego-alpha-closure-local --variants hard:off,feather55:0.55 --control ui --skip-build --skip-visual-stats --port 5389 --output-dir /tmp/objgauss-spark-mask-feather-ui-toggle`: passed。
  - `npm run audit:spark-mask-feather-sweep -- --assets nerf-lego-alpha-closure-local --variants hard:off,feather55:0.55 --control url --skip-build --skip-visual-stats --port 5390 --output-dir /tmp/objgauss-spark-mask-feather-url-regression`: passed。

### DEMO-005C: Spark mask feather sweep report

- 状态: done / multi-scene-feather-report
- 类型: 标准 PR / renderer UX + browser audit report
- 目标: 将 `spark-object-mask-feather` 从单场景诊断扩展成多场景 sweep / report，比较 hard mask 与 soft-boundary variant 对 route、opacity texture、coverage / luma / chroma 和截图的影响，再决定是否默认启用。
- 已实施:
  - 新增 `scripts/audit-spark-mask-feather-sweep.mjs` 与 `npm run audit:spark-mask-feather-sweep`。
  - 默认覆盖 `nerf-lego-alpha-closure-local` 与 `plush-semantic-closure-local`，默认 variants 为 `hard:off` 和 `feather55:0.55`。
  - 脚本使用轻量 Playwright route-only flow：加载样例、进入对象编辑、选择首个 object、执行 `预览删除`，读取 Spark route / object opacity texture / feather telemetry，并采集 before/after canvas visual stats 与截图。
  - 输出 `/tmp/objgauss-spark-mask-feather-sweep/summary.{json,md}`，包含每场景 hard / feather rows 和 hard→feather comparison。
- 结论:
  - 双场景 route contract 通过，Lego 与 Plush 都保持 `spark-native-mask` / `hard-object-mask-no-reoptimize` / `hard-mask-no-inpaint`。
  - `feather55` 在 Lego 软化 2932 个 Gaussian，在 Plush 软化 65273 个 Gaussian，mean opacity 分别降到 `0.901044` 和 `0.845783`。
  - 当前结果不支持默认启用：Lego coverage ratio 从 `1.281243` 升到 `1.286964`，Plush 从 `1.511632` 升到 `1.521993`；Plush luma 略好但 chroma 略差。因此 feather 保持诊断 / 候选策略。
- 验证:
  - `node --check scripts/audit-spark-mask-feather-sweep.mjs`: passed。
  - `npm run audit:spark-mask-feather-sweep -- --assets nerf-lego-alpha-closure-local --variants hard:off,feather55:0.55 --skip-build --port 5387 --output-dir /tmp/objgauss-spark-mask-feather-sweep-smoke`: passed。
  - `npm run audit:spark-mask-feather-sweep -- --skip-build --port 5388 --output-dir /tmp/objgauss-spark-mask-feather-sweep`: passed；rows=4，comparisons=2。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### DEMO-005B: Spark object mask feather diagnostic

- 状态: done / soft-boundary-diagnostic
- 类型: 标准 PR / renderer UX + browser audit
- 目标: 回应 `自身颜色` hard-mask 删除后颗粒感问题，提供一条可运行、可审计的 soft boundary 诊断路径，而不是只靠文案解释。
- 已实施:
  - `src/sparkObjectMask.js` 将 Spark object opacity texture 从二值 0/1 扩展为 0..255 opacity scale；hidden object 仍为 0，visible Gaussian 默认为 255。
  - 新增 `spark-object-mask-feather=on` URL 开关；开启后用 3D spatial hash 查找靠近 hidden Gaussian 的 visible Gaussian，并按距离降低 opacity。
  - `src/SplatViewport.jsx` 暴露 `data-spark-object-mask-feather-*` telemetry，包括 mode、radius、opacity、softened Gaussian 数、mean opacity 和 min opacity。
  - `scripts/audit-demo.mjs` 支持 `--spark-object-mask-feather` / `--spark-object-mask-feather-opacity` / `--spark-object-mask-feather-radius`，默认仍要求 feather=`off`，显式开启时要求 `spatial-neighbor-feather-v1` 和非零 softened Gaussian。
  - 新增 `npm run audit:spark-mask-feather` 作为可复现单场景 browser gate。
- 结论:
  - 默认 hard-mask route 保持兼容；Lego 默认 audit 输出 `sparkObjectMaskFeather="off":0:0:1:1/1`。
  - Feather 诊断路径在 Lego 上通过：`3214` 个 Gaussian 被软化，auto radius=`0.07`，requested opacity=`0.55`，mean opacity=`0.832853`，min opacity=`0.560784`。
  - 这仍不是补洞或重优化；它只是降低 visible boundary Gaussian 的 opacity，是否默认启用要等多场景 sweep。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5375`: passed。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5376 --spark-object-mask-feather --spark-object-mask-feather-opacity 0.55`: passed。
  - `git diff --check`: passed。

### DEMO-005A: Source-color hard-mask preview UX contract

- 状态: done / source-preview-copy
- 类型: 微改动 / renderer UX contract
- 目标: 回应 `自身颜色` 删除后仍有颗粒感的反馈，让 UI 明确区分颜色来源和删除结果，避免把 hard-mask 预览误解为补洞后的完整高斯重渲染。
- 已实施:
  - `src/App.jsx` 新增 `data-source-preview-result`，删除 / 隔离后的 source-color route 暴露为 `hard-mask-no-inpaint`。
  - 状态面板新增 `删除结果`，删除后显示 `源色 mask 预览`；`预览边界` 改为更直接的 `硬 mask，无补洞`。
  - `scripts/audit-demo.mjs` 验证删除后同时具备 `source-color`、`hard-object-mask-no-reoptimize` 和 `hard-mask-no-inpaint`。
  - `docs/rendering/renderer-readiness-matrix.md` 记录 UI contract。
- 结论:
  - `自身颜色` 不再承担“删除后完整高斯效果”的含义；它只表示颜色来源，删除结果由 `删除结果` 和 `质量解释` 共同说明。
  - 这不是视觉质量修复；真正降低颗粒感仍需要 soft mask、边界重分配、补洞或重优化。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5372`: passed；删除后 `postDeleteRoute="spark-native-mask":"commercial":"source-color":"hard-object-mask-no-reoptimize":"hard-mask-no-inpaint":"boundary-mixing-dominant"`。
  - `git diff --check`: passed。

### DEMO-004: Poly Haven Chair commercial Gaussian sample registration

- 状态: done / commercial-sample-registered
- 类型: 标准 PR / demo asset registration
- 目标: 基于 `audit:commercial-demo-readiness` 的结论，补一个 license-clean、viewer 可直接交互的 Gaussian demo sample。
- 已实施:
  - 新增 `polyhaven-chair-commercial-demo-local` 前端素材与 CLI asset registry 记录，指向本地生成的 `public/samples/polyhaven_chair_demo.splat` 和 `public/samples/polyhaven_chair_demo_objects.ply`。
  - 新增 `scripts/publish-polyhaven-chair-demo.mjs` 与 `npm run publish:polyhaven-chair-demo`，把已有 Splatfacto chair 输出发布到 ignored public sample 路径，并写 `/tmp/objgauss-polyhaven-chair-demo-publish/summary.{json,md}`。
  - `acceptance:spark-commercial-route` 支持 `--native-assets` / `--trained-assets`，可以只跑指定样例的 route gate。
  - `audit-commercial-demo-readiness` 可合并多个 route / quality summaries，把 chair 样例纳入 public-commercial eligibility 表。
  - `docs/asset-library.md` 和 `docs/rendering/renderer-readiness-matrix.md` 记录本地生成边界、命令和当前准入结论。
- 结论:
  - Chair 本地样例有 50,000 个 Gaussian 和 6 个 object ids，Spark trained route / hard-mask quality / commercial readiness 均有本地报告证据。
  - 当前 public-commercial candidate 从 `0` 增至 `1`，但删除后仍是 `hard-object-mask-no-reoptimize`，不是补洞或重优化后的完整编辑结果。
  - 完整 generic `audit:demo` 对 chair 本轮超过合理等待后中止，未作为 gate；后续需要 `DEMO-005` 或专门的 heavy-sample audit 优化。
- 验证:
  - `node --check scripts/publish-polyhaven-chair-demo.mjs`: passed。
  - `npm run publish:polyhaven-chair-demo -- --output-dir /tmp/objgauss-polyhaven-chair-demo-publish`: passed。
  - `uv run objgauss stats public/samples/polyhaven_chair_demo_objects.ply`: passed；`gaussians=50000`，`objects=6`。
  - `npm run audit:spark-trained-route -- --assets polyhaven-chair-commercial-demo-local --port 5367`: passed。
  - `npm run acceptance:spark-commercial-route -- --skip-build --skip-trained-sample-audit --trained-assets polyhaven-chair-commercial-demo-local --native-port 5368 --trained-port 5369 --output-dir /tmp/objgauss-spark-commercial-route-chair`: passed。
  - `npm run audit:object-mask-boundary -- --assets polyhaven-chair-commercial-demo-local --output-dir /tmp/objgauss-object-mask-boundary-chair`: passed。
  - `npm run audit:spark-reconstruct-residual -- --assets polyhaven-chair-commercial-demo-local --output-dir /tmp/objgauss-spark-reconstruct-residual-chair --port 5370 --allow-failures`: passed。
  - `npm run audit:hard-mask-quality -- --boundary-summary /tmp/objgauss-object-mask-boundary-chair/summary.json --route-summary /tmp/objgauss-spark-commercial-route-chair/summary.json --residual-summary /tmp/objgauss-spark-reconstruct-residual-chair/summary.json --output-dir /tmp/objgauss-hard-mask-quality-chair --require-route --require-residual`: passed；interpretation=`boundary-mixing-dominant`。
  - `npm run audit:commercial-demo-readiness -- --output-dir /tmp/objgauss-commercial-demo-readiness-with-chair`: passed；`publicCommercial=1`。

### RENDER-005T-BE: Commercial demo readiness QA

- 状态: done / commercial-demo-readiness
- 类型: 标准 PR / renderer QA reporting
- 目标: 基于产品 route status 与 hard-mask quality 解释，整理商用展示 QA 截图 / 样例准入表，明确哪些样例能标“商业展示默认路线”、哪些只能作为诊断或研究样例。
- 已实施:
  - 新增 `scripts/audit-commercial-demo-readiness.mjs` 与 `npm run audit:commercial-demo-readiness`。
  - 读取 `acceptance:spark-commercial-route` 的 route summary 和 `audit:hard-mask-quality` 的 quality summary，输出 `/tmp/objgauss-commercial-demo-readiness/summary.{json,md}`。
  - 把产品 route readiness 与 public-commercial license eligibility 分开：route 通过不等于素材许可干净。
  - 报告会列出 route tier、quality interpretation、route kind、license scope、public-commercial eligibility、required copy 和 screenshot path。
  - `docs/rendering/renderer-readiness-matrix.md` 增加 Commercial Demo Readiness 说明和当前本地结论。
- 结论:
  - `nerf-lego-alpha-closure-local` 与 `plush-semantic-closure-local` 是 `商业展示路线可演示`，但必须显示 `对象 mask，无补洞 / 边界混合主导`，且许可分别是 research-only / local-test-only，不是 public commercial candidate。
  - `nerf-lego-trained-output-local` 是 `研究 / 诊断样例`，因为当前质量解释为 `browser-residual-dominant`，不得标成商业展示默认效果。
  - `plush-3dgs-local` 与 `plush-v1-closure-local` 仍是 `待 route QA`。
  - 当前 `publicCommercialCandidateRows=0`，下一步如果要“可商用展示”，需要生成或登记 license-clean 且 viewer 可交互的 Gaussian 样例。
- 验证:
  - `node --check scripts/audit-commercial-demo-readiness.mjs`: passed。
  - `npm run audit:commercial-demo-readiness -- --output-dir /tmp/objgauss-commercial-demo-readiness`: passed；rows=5，routeReady=3，showcaseCaveated=2，researchDiagnostic=1，publicCommercial=0。

### RENDER-005T-BD: Product hard-mask quality status

- 状态: done / quality-status-ui
- 类型: 标准 PR / renderer UX + browser audit
- 目标: 将 hard-mask quality chain 的解释结果接入产品 route status / QA copy，让用户区分 hard object mask 边界问题、coverage hole risk 和 source reconstruction residual。
- 已实施:
  - `src/App.jsx` 在加载素材时保留 `assetId`，并按当前 route 输出 `hardMaskQuality`。
  - 状态面板新增 `质量解释`，首屏显示 `原始 Spark 高斯`，对象色显示 `对象色诊断`，删除后按证据显示 `边界混合主导`、`重建残差主导` 或 `硬 mask 待审计`。
  - App root 新增 `data-hard-mask-quality-interpretation`、`data-hard-mask-quality-source`、`data-hard-mask-gap-score`、`data-hard-mask-residual-coverage-ratio`、`data-hard-mask-deleted-object`。
  - `scripts/audit-demo.mjs` 验证首屏、对象色和删除后的 hard-mask quality contract，并在 audit 日志输出 `hardMaskQuality=...`。
  - `docs/rendering/renderer-readiness-matrix.md` 将 Product UI Contract 更新为包含质量解释字段。
- 结论:
  - `自身颜色` 现在在 UI 上只表示颜色来源；`预览边界` 和 `质量解释` 共同说明它是否是原始 Spark 高斯，还是 hard object mask / no reoptimize 预览。
  - Lego proxy 与 Plush semantic 使用 report-backed `boundary-mixing-dominant`；trained Lego 使用 `browser-residual-dominant`；缺少 quality-chain row 的样例显示 `hard-mask-quality-unmeasured`，不伪造结论。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5365`: passed；删除后 `hardMaskQuality="boundary-mixing-dominant":"hard-mask-quality-chain-v1":0.524659:1.170841:"0"`。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。
  - `npm run audit:demo -- --assets plush-v1-closure-local --skip-visual-residual --server-mode preview --port 5366`: 本轮超过合理等待时间后手动中止，未作为 gate；前两次 dev-server audit 因本机 watcher 上限 `ENOSPC` 失败，preview 模式已覆盖新增 contract。

### RENDER-005T-BC: Hard mask quality chain report

- 状态: done / hard-mask-quality-chain
- 类型: 标准 PR / renderer quality reporting
- 目标: 将 object-mask boundary diagnostic 与 browser visual residual / Spark route report 对齐，形成 hard-mask 质量解释链，而不是只看单个 PLY-level proxy。
- 已实施:
  - 新增 `scripts/audit-hard-mask-quality.mjs` 与 `npm run audit:hard-mask-quality`。
  - Aggregator 读取 `audit:object-mask-boundary` 的 boundary summary、`acceptance:spark-commercial-route` 的 route summary 和 `audit:spark-reconstruct-residual*` 的 browser residual summary。
  - 按 asset id 对齐证据，并从 Spark object mask hidden Gaussian count 反推实际删除的 object_id，再读取该 object 的 boundary metrics，而不是只使用 worst object。
  - 输出 `/tmp/objgauss-hard-mask-quality/summary.{json,md}`，区分 `boundary-mixing-dominant`、`coverage-hole-risk` 和 `browser-residual-dominant`。
  - `docs/rendering/renderer-readiness-matrix.md` 增加 Hard Mask Quality Chain，明确该命令本身不启动浏览器，而是消费已有 browser artifacts。
- 结论:
  - Lego proxy 与 Plush semantic 在 route + residual 证据齐全时均为 `boundary-mixing-dominant`，说明 hard mask 后的粗糙主要来自对象边界共享 / 局部混合，而不是 PLY unique coverage 大面积丢失。
  - Trained Lego 当前为 `browser-residual-dominant`，因为 Spark reconstruct residual coverage ratio=`15.599172`、luma delta=`0.250533` 明显大于 hard-mask proxy；该场景优先问题是 SH-heavy source / reconstruction residual，而不是单纯 object boundary。
- 验证:
  - `node --check scripts/audit-hard-mask-quality.mjs`: passed。
  - `npm run audit:hard-mask-quality -- --boundary-summary /tmp/objgauss-object-mask-boundary/summary.json --route-summary /tmp/objgauss-spark-commercial-route-availability/summary.json --residual-summary /tmp/objgauss-spark-reconstruct-residual-multiscene/summary.json,/tmp/objgauss-spark-reconstruct-residual-trained/summary.json --output-dir /tmp/objgauss-hard-mask-quality`: passed，rows=3，missing route/residual assets=0。
  - `npm run audit:hard-mask-quality -- --output-dir /tmp/objgauss-hard-mask-quality-default`: passed，默认路径可发现现有 boundary / route / residual artifacts。
  - Summary inspect: `nerf-lego-alpha-closure-local` evidence=`boundary+route+residual`，deleted object=`0`，gap=`0.524659`，coverageRatio=`1.170841`，interpretation=`boundary-mixing-dominant`。
  - Summary inspect: `plush-semantic-closure-local` evidence=`boundary+route+residual`，deleted object=`0`，gap=`0.513937`，coverageRatio=`1.303149`，interpretation=`boundary-mixing-dominant`。
  - Summary inspect: `nerf-lego-trained-output-local` evidence=`boundary+route+residual`，deleted object=`0`，gap=`0.377656`，coverageRatio=`15.599172`，residual=`failed`，interpretation=`browser-residual-dominant`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-BB: Object mask boundary diagnostic

- 状态: done / hard-mask-boundary-diagnostic
- 类型: 标准 PR / renderer quality diagnostic
- 目标: 针对 Spark source/original 删除后仍可能颗粒的问题，建立 object-boundary / deleted-subset coverage diagnostic，把 hard object mask 的视觉缺口从主观观感变成可量化报告。
- 已实施:
  - 新增 `scripts/audit-object-mask-boundary.mjs` 与 `npm run audit:object-mask-boundary`。
  - 诊断读取 object-aware PLY，不启动浏览器，不修改素材，默认报告写到 `/tmp/objgauss-object-mask-boundary/summary.{json,md}`。
  - 三组正交投影估算 `deletedSubsetCoverageRatio`、`uniqueCoverageLossRatio`、`visibleAfterDeleteCoverageRatio` 和 `sharedBoundaryCoverageRatio`。
  - 3D 邻域采样估算 `neighborBoundaryRatio` / `neighborMixedRatio`，用于解释 object assignment 边界混合风险。
  - `docs/rendering/renderer-readiness-matrix.md` 增加 Hard Mask Boundary Diagnostic，明确该脚本解释 hard-mask grain 来源，但不替代 browser visual residual。
- 结论:
  - 默认三资产诊断通过：Lego proxy、Plush semantic、trained Lego 都可生成 hard-mask gap report。
  - 当前结果显示三个样例的 worst-object `uniqueCoverageLossRatio` 都较低（约 `0.0015-0.0200`），但 `sharedBoundaryCoverageRatio` 与部分 `neighborBoundaryRatio` 很高，说明删除后颗粒感更像 hard object 边界混合 / 子集稀疏问题，而不是大面积 coverage 被删空。
- 验证:
  - `node --check scripts/audit-object-mask-boundary.mjs`: passed。
  - `npm run audit:object-mask-boundary -- --output-dir /tmp/objgauss-object-mask-boundary`: passed；assets=3，skipped=0。
  - Summary inspect: Lego proxy worst object `1`，gap=`0.528368`，uniqueLoss=`0.019980`，sharedBoundary=`0.941537`，neighborBoundary=`0.946644`。
  - Summary inspect: Plush semantic worst object `2`，gap=`0.541380`，uniqueLoss=`0.001512`，sharedBoundary=`0.995673`，neighborBoundary=`0.972603`。
  - Summary inspect: trained Lego worst object `2`，gap=`0.402209`，uniqueLoss=`0.017939`，sharedBoundary=`0.959457`，neighborBoundary=`0.514241`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-BA: Renderer CI / product route profile split

- 状态: done / renderer-profile-split
- 类型: 标准 PR / acceptance profile
- 目标: 基于 trained sample availability audit，决定 Spark commercial route gate 的默认 CI 策略，并把 CI-safe renderer gate 与显式 product-route gate 分开。
- 已实施:
  - 新增 `scripts/acceptance-renderer-profile.mjs` 与 `npm run acceptance:renderer`。
  - 新增 `npm run acceptance:renderer-ci`，默认执行 build、WebGPU tile smoke、no-SH public sample index mapping 和 Spark native object mask gate。
  - 新增 `npm run acceptance:renderer-product`，显式调用完整 `acceptance:spark-commercial-route`，包括 trained sample availability preflight 与 SH-heavy packed SH route。
  - `docs/rendering/renderer-readiness-matrix.md` 增加 Renderer Acceptance Profiles，明确默认 CI 不要求 `nerf-lego-trained-output-local`，产品 / Demo review 才要求该本机 SH-heavy sample。
- 结论:
  - Spark commercial route gate 不提升为默认 fresh-clone CI requirement。
  - 默认 renderer CI profile 只覆盖 repo public no-SH 样例和 C-path smoke；trained SH-heavy route 保持显式 product/demo gate。
  - 只有当 trained sample 变成 committed、downloadable 或 CI-generated fixture 时，才重新评估默认 CI 纳入。
- 验证:
  - `node --check scripts/acceptance-renderer-profile.mjs`: passed。
  - `npm run acceptance:renderer -- --profile ci --dry-run --output-dir /tmp/objgauss-renderer-profile-ci-dry-run`: passed；dry-run steps=4，且 `includesTrainedShHeavySample=false`。
  - `npm run acceptance:renderer-product -- --dry-run --output-dir /tmp/objgauss-renderer-profile-product-dry-run`: passed；dry-run steps=1，且 `includesTrainedShHeavySample=true`。
  - `npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-nonbrowser`: passed；覆盖 build、WebGPU tile smoke 和 no-SH public sample index mapping，report 为 `status=passed`、steps=3。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。
  - 本轮完整 `npm run acceptance:renderer-ci -- --native-port 5357 --output-dir /tmp/objgauss-renderer-profile-ci` 在 sandbox 内执行到 Spark native browser gate 时因 localhost `fetch failed` 失败；按规则提权重跑被审批服务断连拒绝。该失败未进入页面逻辑，不推翻此前 `acceptance:spark-commercial-route` 对 native route 的通过证据，但完整 profile browser rerun 仍需在允许 localhost 的环境执行。

### RENDER-005T-AZ: Trained SH-heavy sample availability contract

- 状态: done / trained-sample-availability-audited
- 类型: 标准 PR / acceptance preflight
- 目标: 明确 `nerf-lego-trained-output-local` 的 portability / availability 策略，让 Spark commercial route gate 在启动浏览器前先能判断当前环境是否具备 SH-heavy trained sample。
- 已实施:
  - 新增 `scripts/audit-spark-trained-sample.mjs` 与 `npm run audit:spark-trained-sample`。
  - Audit 检查 `src/assetLibrary.js` 中的 `nerf-lego-trained-output-local` 注册、`public/samples/nerf_lego_trained.splat`、`public/samples/nerf_lego_trained_objects.ply`。
  - Audit 解析 PLY header 和 `object_id`，检查 geometry、opacity、scale/rotation、`object_id`、`f_dc_*`、degree-3 `f_rest_*`、最小 Gaussian 数和最小 object 数。
  - `acceptance:spark-commercial-route` 默认前置运行 `audit:spark-trained-sample`，并把报告写到 `<output-dir>/trained-sample/`；可用 `--skip-trained-sample-audit` 显式跳过同轮重复检查。
  - `docs/rendering/renderer-readiness-matrix.md` 记录 availability audit、prepare 命令和 skip 边界。
- 结论:
  - Trained SH-heavy sample 仍是本机 / public sample contract，不是 fresh clone 必然具备的 CI fixture。
  - 但缺失或结构不合格现在会在浏览器 gate 前 fail fast，并给出 `benchmark:splatfacto:balanced` / `audit:spark-trained-route` prepare path。
- 验证:
  - `node --check scripts/audit-spark-trained-sample.mjs`: passed。
  - `node --check scripts/acceptance-spark-commercial-route.mjs`: passed。
  - `npm run audit:spark-trained-sample -- --output-dir /tmp/objgauss-spark-trained-sample-audit`: passed；`gaussians=255794`、`objects=4`、`shRest=45`、`shDegree=3`。
  - `npm run acceptance:spark-commercial-route -- --native-port 5353 --trained-port 5354 --output-dir /tmp/objgauss-spark-commercial-route-availability`: passed；`summary.json` 为 `status=passed`、`steps=4`、`native=2`、`trained=1`、`skipTrainedSampleAudit=false`。
  - Summary inspect: trained sample preflight 通过，native routes 覆盖 `nerf-lego-alpha-closure-local` 与 `plush-semantic-closure-local`，trained route 覆盖 `nerf-lego-trained-output-local`，delete route 为 `spark-packed-sh-mask / commercial / hard-object-mask-no-reoptimize`，SH rest 为 `255794 / 255794 / true / 45 / 3`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AY: Spark route acceptance demo opt-in

- 状态: done / demo-opt-in-gate
- 类型: 标准 PR / acceptance integration
- 目标: 评估 report-backed `acceptance:spark-commercial-route` 是否应默认纳入 broader `acceptance:demo` / CI，并落地一个不破坏新环境的集成方式。
- 已实施:
  - `scripts/acceptance-demo.mjs` 保持默认行为不变，不默认要求本机 trained SH-heavy sample。
  - 新增 `--include-spark-commercial-route`，显式把 `acceptance:spark-commercial-route` 追加到 demo acceptance。
  - 新增 Spark route 透传参数：`--spark-native-port`、`--spark-trained-port`、`--spark-route-output-dir`、`--skip-spark-route-build`。
  - 新增 browser audit 窄化参数：`--browser-audit-assets`、`--skip-browser-visual-residual`，便于在复查 opt-in 编排时避免全量视觉残差长跑。
  - `docs/rendering/renderer-readiness-matrix.md` 记录 opt-in 命令和为什么不默认纳入：SH-heavy route 依赖本机 `nerf-lego-trained-output-local`。
- 结论:
  - Spark commercial route gate 可以从 broader demo acceptance 显式调用，但不会成为默认 CI / fresh-env requirement。
  - `acceptance:demo` 的 browser audit 默认改走 built preview，避免 Vite dev watcher 上限在本机 / CI 中阻断验收；旧 dev 模式仍可用 `--browser-audit-mode dev` 显式选择。
  - 下一步要先明确 trained SH-heavy sample 的 portability / availability 策略，再决定是否默认纳入 CI。
- 验证:
  - `node --check scripts/acceptance-demo.mjs`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/acceptance-spark-commercial-route.mjs`: passed。
  - `npm run acceptance:demo -- --skip-semantic-benchmark --browser-audit-assets nerf-lego-alpha-closure-local --skip-browser-visual-residual --include-spark-commercial-route --spark-native-port 5351 --spark-trained-port 5352 --spark-route-output-dir /tmp/objgauss-acceptance-demo-spark-route --skip-spark-route-build`: passed；输出 `acceptance_demo=passed`。
  - Summary inspect: `/tmp/objgauss-acceptance-demo-spark-route/summary.json` 为 `status=passed`、`native=2`、`trained=1`，trained delete route 为 `spark-packed-sh-mask / commercial / hard-object-mask-no-reoptimize`，SH rest 为 `255794 / 255794 / true / 45 / 3`。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AX: Spark commercial route report artifact

- 状态: done / route-report-artifact
- 类型: 标准 PR / acceptance reporting
- 目标: 在 `acceptance:spark-commercial-route` 已能统一跑 no-SH native 与 SH-heavy packed route 后，补 compact route report artifact，避免只靠终端长日志审查商业展示路线。
- 已实施:
  - `scripts/acceptance-spark-commercial-route.mjs` 现在会 tee 子命令输出并解析关键 browser gate 行。
  - 默认写出 `/tmp/objgauss-spark-commercial-route/summary.json` 与 `summary.md`。
  - 支持 `--output-dir <path>` 指定报告目录。
  - Report 汇总 native no-SH route、SH-heavy initial/delete route、Spark source、visible/base Gaussian、object mask、SH rest preservation tuple、contract boundary 和截图路径。
- 结论:
  - 商用 Spark route gate 现在既能 fail fast，也能留下可审查 artifact。
  - 当前仍不直接纳入 `acceptance:demo`；是否放进 broader CI 需要先决定本机 trained SH-heavy sample 的可用性策略。
- 验证:
  - `node --check scripts/acceptance-spark-commercial-route.mjs`: passed。
  - `npm run acceptance:spark-commercial-route -- --native-port 5349 --trained-port 5350 --output-dir /tmp/objgauss-spark-commercial-route-report`: passed。
  - Summary inspect: `status=passed`、`steps=3`、`native=2`、`trained=1`，trained delete route 为 `spark-packed-sh-mask / commercial / hard-object-mask-no-reoptimize`，SH rest 为 `255794 / 255794 / true / 45 / 3`。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AW: Spark commercial route acceptance

- 状态: done / commercial-route-accepted
- 类型: 标准 PR / acceptance integration
- 目标: 将 no-SH native route gate 与 SH-heavy packed route gate 合并成一条 Spark commercial route acceptance 命令，避免商业展示路线验收依赖人工串命令。
- 已实施:
  - 新增 `scripts/acceptance-spark-commercial-route.mjs` 与 `npm run acceptance:spark-commercial-route`。
  - 默认先执行 `npm run build`，再跑 `audit:spark-native-mask-gate` 和 `audit:spark-trained-route`。
  - 支持 `--skip-build`、`--native-port` 和 `--trained-port`，便于本地避开端口冲突。
  - `docs/rendering/renderer-readiness-matrix.md` 增加 Spark commercial route acceptance 说明，并明确该 gate 不证明删除补洞、重优化、WebGPU visual fidelity 或任意第三方 `.splat` object metadata。
- 结论:
  - 商用 Spark source/original route 现在有一条统一验收命令，同时覆盖 no-SH native compact `.splat` object mask 和 SH-heavy packed SH object mask。
  - 这条 gate 专门证明 renderer route contract；删除后的 `自身颜色` 仍是 hard object mask / no reoptimize preview，不被解释成最终编辑重渲染。
- 验证:
  - `node --check scripts/acceptance-spark-commercial-route.mjs`: passed。
  - `npm run acceptance:spark-commercial-route -- --native-port 5347 --trained-port 5348`: passed；no-SH native gate 覆盖 `nerf-lego-alpha-closure-local` 和 `plush-semantic-closure-local`，trained SH-heavy gate 覆盖 `nerf-lego-trained-output-local`。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AV: SH-heavy Spark route-only audit

- 状态: done / trained-route-audited
- 类型: 标准 PR / browser audit
- 目标: 增加 SH-heavy trained route-only browser audit，低成本验证 `spark-packed-sh-mask` / SH preservation / hard-object-mask boundary，不每次跑完整 visual residual。
- 已实施:
  - 新增 `scripts/audit-spark-trained-route.mjs` 与 `npm run audit:spark-trained-route`。
  - 默认覆盖 `nerf-lego-trained-output-local`，使用静态 `vite preview`，避免 dev watcher 上限。
  - Gate 加载 trained SH-heavy 样例，检查初始 `spark-ply-sh-source` commercial route，再删除一个对象并验证 `spark-packed-sh-mask`、`ply-packed`、`packed-sh-extract-v1`、`object-opacity-texture-v1`、`hard-object-mask-no-reoptimize` 和完整 `f_rest_*` degree-3 preservation。
  - `docs/rendering/renderer-readiness-matrix.md` 增加 SH-heavy route-only gate 说明。
- 结论:
  - Trained SH-heavy route 现在有低成本正式 gate，不再需要用完整 `audit:demo` 作为 route contract 证据。
  - 当前本机结果：`initial="spark-ply-sh-source":"commercial":"source-splat"`，`delete="spark-packed-sh-mask":"commercial":"hard-object-mask-no-reoptimize"`，`spark="ply-packed":"packed-sh-extract-v1"`，`shRest=255794:255794:"true":45:3`。
- 验证:
  - `node --check scripts/audit-spark-trained-route.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-trained-route -- --port 5346`: passed；截图 `/tmp/objgauss-spark-trained-route-nerf-lego-trained-output-local.png`。
  - `git diff --check`: passed。
  - `uv run --extra dev pytest`: 41 passed。

### RENDER-005T-AU: Product renderer route UI contract

- 状态: done / route-ui-audited
- 类型: 标准 PR / 前端 UX + browser audit
- 目标: 将 renderer readiness matrix 反映到产品 UI，精简商业 Demo 的颜色/调试入口，并让 source/original delete preview 的质量边界更清晰。
- 已实施:
  - 顶部和侧栏颜色下拉改为 `自身颜色` / `对象色诊断`，把对象色明确降级为 debug/diagnostic 入口。
  - Viewport 增加紧凑 route badge，状态面板增加 `展示路线`、`颜色用途`、`预览边界`。
  - App root 暴露 `data-renderer-route`、`data-renderer-route-kind`、`data-color-mode-role`、`data-source-preview-boundary`、`data-preview-quality`。
  - `audit-demo` 新增 route contract：首屏必须是 commercial / source-color / source-splat；对象色必须是 diagnostic-object-color；删除预览必须回到 source-color 且暴露 `hard-object-mask-no-reoptimize`。
  - `docs/rendering/renderer-readiness-matrix.md` 新增 Product UI Contract。
- 结论:
  - UI 现在把商用 Spark route、WebGPU C-path 诊断 route、Fallback route 分开呈现。
  - 删除后的自身颜色不会再暗示“重新优化后的完整 3DGS”；状态面板明确显示 `对象 mask，无补洞`。
  - Trained SH-heavy route 在手动复查中打印了 `browser_audit=passed` 与 `postDeleteRoute="spark-packed-sh-mask":"commercial":"source-color":"hard-object-mask-no-reoptimize"`，但本次命令被中止信号打断返回 130，因此只作为补充观察，不记为正式 gate。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --url http://127.0.0.1:5341/ --no-server`: passed；route 输出 `spark-original-view -> diagnostic-object-color -> spark-native-mask`，删除后 boundary 为 `hard-object-mask-no-reoptimize`。
  - Playwright screenshot: `/tmp/objgauss-audit-nerf-lego-alpha-closure-local.png`，视觉检查 route badge / banner / 状态面板未遮挡主操作。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AT: Renderer readiness matrix

- 状态: done / route-decision-recorded
- 类型: 标准 PR / 渲染路线决策
- 目标: 建立 WebGPU / Spark renderer readiness matrix，明确商业 Demo 默认路线、WebGPU 诊断路线和何时允许切换默认渲染。
- 已实施:
  - 新增 `docs/rendering/renderer-readiness-matrix.md`。
  - 明确当前商业展示默认走 Spark source/original route：no-SH 样例走 native `.splat` mask，SH-heavy 样例走 PLY packed SH mask。
  - 明确 WebGPU Tile 是 C-path proof / diagnostics，不是当前商业默认 renderer。
  - 将“原始颜色 / 自身颜色删除后仍可能颗粒”的原因记录为 hard object mask、边界 assignment、未重优化补洞和 SH-heavy packed route 的质量边界，而不是颜色丢失 bug。
  - 记录 WebGPU 切默认前必须通过的 headless acceptance、desktop presentation、coverage gate、trained SH parity 和多场景 object edit pixel evidence。
- 结论:
  - 当前产品路线应继续优先 Spark filtered edit，避免把 WebGPU 诊断近似展示成商业默认效果。
  - 下一步 UX 切片应把 renderer route / quality boundary 反映到界面，并把 `对象色` 明确降级为诊断模式。
- 验证:
  - `git diff --check`: passed。
  - 文档引用检查：`docs/state/project-status.md` 和 `docs/state/pr-queue.md` 均指向 `docs/rendering/renderer-readiness-matrix.md`。

### RENDER-005T-AS: WebGPU headless acceptance and presentation split

- 状态: done / headless-acceptance-runbooked
- 类型: 标准 PR / WebGPU acceptance integration
- 目标: 将 WebGPU offscreen object-transition suite 纳入更高层 acceptance / CI-headless runbook，并明确与 headed presentation gate 的分工。
- 已实施:
  - 新增 `scripts/acceptance-webgpu-headless.mjs` 与 `npm run acceptance:webgpu-headless`。
  - Headless acceptance 顺序执行 `npm run build`、`npm run audit:webgpu-tile-smoke` 和 `npm run audit:webgpu-offscreen-readback`。
  - 新增 `docs/rendering/webgpu-headless-acceptance.md`，记录命令、输出目录、可证明内容、不可证明内容，以及与 `audit:webgpu-desktop` / `audit:webgpu-coverage-gate` 的分工。
  - `docs/rendering/webgpu-desktop-audit.md` 增加 headless acceptance 交叉引用，防止把 offscreen readback 误当 canvas presentation 通过。
- 结论:
  - `acceptance:webgpu-headless` 是 CI/headless compute/storage/object editing gate。
  - `audit:webgpu-desktop` 仍是 headed browser canvas presentation gate。
  - `audit:webgpu-coverage-gate` 仍是 visual fidelity / tuning regression gate。
  - 该接入不改变默认商业展示 Spark filtered edit route，也不改 WebGPU shader / visual 参数。
- 验证:
  - `node --check scripts/acceptance-webgpu-headless.mjs`: passed。
  - `npm run acceptance:webgpu-headless -- --port 5330 --output-dir /tmp/objgauss-webgpu-headless-acceptance`: passed；Lego / Plush object-transition readback report 写入 `/tmp/objgauss-webgpu-headless-acceptance/offscreen-readback/summary.*`。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AR: WebGPU offscreen object-state transition gate

- 状态: done / offscreen-transition-audited
- 类型: 标准 PR / WebGPU CI-headless object editing diagnostics
- 目标: 在不创建 WebGPU canvas render pass 的情况下，验证隔离 / 删除会同时改变 GPU object-state buffer 和 offscreen readback checksum。
- 已实施:
  - `audit-demo` 新增 `--webgpu-object-transition`，仅允许搭配 `--webgpu-probe offscreen-readback`。
  - transition audit 会通过 URL 诊断开关 `spark-filtered-edit=off` 暂时禁用 Spark filtered edit route，让删除后保留在 WebGPU Tile 编辑路径；默认产品 / 商业展示路线不变。
  - `audit-demo` 现在输出 `readbackAfterIsolate`、`readbackAfterDelete` 和 `objectStateAfterIsolate` telemetry。
  - `npm run audit:webgpu-offscreen-readback` 默认启用 object transition gate，仍可用 `--skip-object-transition` 回到首帧 readback suite。
  - Suite report 记录初始 / 隔离 / 删除三段 readback checksum 和 object-state checksum。
- 结论:
  - Lego proxy 通过：readback `897e852d -> 3bd507d9 -> 916a5fc9`，object-state `7243475b -> f72fa1f4 -> 35652440`，可见数 `2592` isolate / `3104` delete。
  - Plush semantic 281k 大场景通过：readback `0f87864a -> 0bdb3b09 -> 9660bc47`，object-state `362760d7 -> fc48aab0 -> 637142bc`，可见数 `177095` isolate / `104403` delete。
  - 这证明 WebGPU object-state buffer 的隔离 / 删除状态不只更新 DOM telemetry，也会改变真实 compute pixel output 和 MAP_READ readback。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-offscreen-readback.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run audit:webgpu-offscreen-readback -- --assets nerf-lego-alpha-closure-local --port 5323 --output-dir /tmp/objgauss-webgpu-offscreen-readback-transition-single`: passed。
  - `npm run audit:webgpu-offscreen-readback -- --port 5324 --output-dir /tmp/objgauss-webgpu-offscreen-readback-transition`: passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AQ: WebGPU offscreen readback multi-scene suite

- 状态: done / multiscene-offscreen-readback-audited
- 类型: 标准 PR / WebGPU CI-headless diagnostics
- 目标: 将 `offscreen-readback` 从单场景 probe 扩展为可复现的多场景 CI/headless gate 和 summary report。
- 已实施:
  - 新增 `scripts/audit-webgpu-offscreen-readback.mjs`，默认启动 built `dist/` 的 Vite preview，并逐场景调用 `audit-demo --webgpu-probe offscreen-readback`。
  - `npm run audit:webgpu-offscreen-readback` 现在默认覆盖 `nerf-lego-alpha-closure-local` 与 `plush-semantic-closure-local`，并写出 `/tmp/objgauss-webgpu-offscreen-readback/summary.json` 与 `summary.md`。
  - Suite 解析 `firstFrame`、`queue`、`deviceLost`、`pixel`、`readback`、`storage`、`objectFilter`、`packedGaussians` 和 `tileReferences` telemetry，并对每个场景执行 gate checks。
  - 单场景仍可用 `--assets <asset_id>` 显式复查。
- 结论:
  - Lego proxy 通过：`firstFrame="readback":253952`、`queue="done"`、`deviceLost="active"`、`readback="mapped":"webgpu-compute-depth-binned-alpha-composite-v1":"897e852d":4063232:1015808/1015808:533740`。
  - Plush semantic 281k 大场景通过：`firstFrame="readback":147456`、`queue="done"`、`deviceLost="active"`、`readback="mapped":"webgpu-compute-depth-binned-alpha-composite-v1":"0f87864a":2359296:589824/589824:254524`、`tileReferences=1190026`。
  - 这证明 WebGPU storage upload、pixel compute、buffer copy、queue completion 和 MAP_READ readback 在小场景与 Plush 级大场景上都可脱离 canvas presentation 验收。
- 验证:
  - `node --check scripts/audit-webgpu-offscreen-readback.mjs`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run audit:webgpu-offscreen-readback -- --assets nerf-lego-alpha-closure-local --port 5321 --output-dir /tmp/objgauss-webgpu-offscreen-readback-single`: passed。
  - `npm run audit:webgpu-offscreen-readback -- --port 5322 --output-dir /tmp/objgauss-webgpu-offscreen-readback`: passed。

### RENDER-005T-AP: WebGPU offscreen readback probe

- 状态: done / offscreen-readback-audited
- 类型: 标准 PR / WebGPU runtime diagnostics
- 目标: 为 CI/headless 环境补 WebGPU compute-only / offscreen readback probe，避免把 canvas presentation backend failure 误判为 renderer compute/storage failure。
- 已实施:
  - 新增 `webgpu-probe=offscreen-readback`，只 dispatch pixel compute，不创建 WebGPU canvas render pass。
  - Runtime 在 compute 后执行 `copyBufferToBuffer(pixelResolvedRgba -> MAP_READ staging)`，并把 mapped GPU buffer telemetry 暴露为 `data-webgpu-readback-status/reason/source/checksum/byte-size/float-count/finite-floats/nonzero-floats`。
  - `audit-demo` 验证 offscreen probe 必须 `pixel=dispatched`、`firstFrame=readback`、`resolveFilter=offscreen-map-read`、readback checksum 与 first-frame checksum 一致、finite float 全覆盖且 nonzero floats 大于 0。
  - 新增 `npm run audit:webgpu-offscreen-readback`；`audit:webgpu-desktop` 默认 probe 列表加入 `offscreen-readback`，使 desktop/headless classification 先证明 compute/readback 再判断 presentation。
- 结论:
  - Lego proxy 本地 offscreen audit 通过：`firstFrame="readback":253952`、`queue="done"`、`deviceLost="active"`、`readback="mapped":"webgpu-compute-depth-binned-alpha-composite-v1":"897e852d":4063232:1015808/1015808:533740`。
  - 这条 probe 不证明 canvas display path 成功；它专门证明 WebGPU storage upload、pixel compute、buffer copy 和 GPU readback 成功。presentation 仍由 `clear-only` / texture display / full headed desktop probes 覆盖。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:webgpu-offscreen-readback -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5320/ --no-server`: passed，使用本地 Vite preview；直接 dev server 在 sandbox 下未能 ready，改用 built preview。

### RENDER-005T-AO: Spark pick object-support disambiguation

- 状态: done / pick-disambiguation-audited
- 类型: 标准 PR / 前端对象交互
- 目标: 在 `RENDER-005T-AN` 证明 hit-rate 高但 ambiguity-rate 也高之后，改进 Spark `screen-space-object-pick-v1` 的消歧策略，并把 ambiguity rate 变成回归门禁。
- 已实施:
  - `SplatViewport` 的 Spark pick 从“最近 Gaussian + 第二 object 距离差”升级为 `object-support-score-v1`。
  - 新 scoring 聚合每个候选 object 的最近距离、点击半径内局部支持占比和前景深度优先级，按 score margin 判定 `ambiguous`。
  - Spark viewport 新增 `data-spark-pick-strategy`、`data-spark-pick-score`、`data-spark-pick-score-margin`、`data-spark-pick-second-object`、`data-spark-pick-second-score`。
  - `audit:spark-pick-report` 读取 score telemetry，并默认要求 ambiguity rate `<=0.5`，防止消歧回退。
- 结论:
  - Lego proxy 默认 report 保持 `14/15` hit、marker hits `14/14`，ambiguity rate 从 `0.928571` 降到 `0.357143`，mean score margin `0.171357`。
  - Trained SH-heavy 5-click report 保持 `5/5` hit、marker hits `5/5`，ambiguity rate 从 `1` 降到 `0.2`，mean score margin `0.2034`。
  - `audit-demo` 小场景和 trained 样例的单次 Spark pick 都变成 `ambiguous=false`，同时保持 `spark-object-opacity-mask`、native / packed route 和 SH-rest preservation contract。
  - 这仍是 screen-space CPU pick over object-aware PLY metadata，不是 Spark-internal raycast；剩余 close-boundary ambiguity 后续可通过 hover/confirm UX 或 renderer-native object metadata path 继续处理。
- 验证:
  - `node --check scripts/audit-spark-pick-report.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-pick-report`: passed。
  - `npm run audit:spark-pick-report -- --assets nerf-lego-trained-output-local --max-clicks 5 --output-dir /tmp/objgauss-spark-pick-report-trained --port 5316`: passed。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5317/ --no-server`: passed。
  - `npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5317/ --no-server`: passed。
  - `npm run audit:splat-index-mapping`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run audit:spark-native-mask-gate`: passed。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `uv run --extra dev pytest`: 41 passed。

### RENDER-005T-AN: Spark pick hit-rate / ambiguity report

- 状态: done / pick-report-audited
- 类型: 标准 PR / 前端对象交互验收
- 目标: 将 Spark `screen-space-object-pick-v1` 从单点击 hit telemetry 推进到多点击 hit-rate / ambiguity-rate report，避免把高歧义点击误判为 robust renderer-native picking。
- 已实施:
  - 新增 `scripts/audit-spark-pick-report.mjs`，默认启动静态 Vite preview，进入 Spark 删除预览，并对画布执行 15 个 deterministic 点击点。
  - 新增 `npm run audit:spark-pick-report`，默认覆盖 Lego proxy；trained 大场景可用 `--assets nerf-lego-trained-output-local --max-clicks 5` 显式复查。
  - Report 写出 `/tmp/objgauss-spark-pick-report/summary.json`、`summary.md` 和素材截图，并把 `hitRate`、`ambiguityRate`、`markerHits`、`distinctHitObjects`、route / mask source 一并记录。
  - Report gate 要求至少 1 个 hit、hit rate >= `0.2`、所有 hit 都有合法 marker 和 selected object match；`ambiguous=true` 只记录不失败。
- 结论:
  - Lego proxy 默认 route 通过：`14/15` hit、hit rate `0.933333`、ambiguous hits `13/14`、ambiguity rate `0.928571`、marker hits `14/14`，route 为 `native-splat-source-v1`。
  - Trained SH-heavy 显式 5-click route 通过：`5/5` hit、ambiguity rate `1`、marker hits `5/5`，route 为 `packed-sh-extract-v1`，mask source 为 `ply-packed`。
  - 这证明 Spark canvas pick 对 demo 交互已经可用，但高 ambiguity 是硬事实；下一步应做 pick 消歧策略，而不是宣称无歧义 renderer-native picking。
- 验证:
  - `node --check scripts/audit-spark-pick-report.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-pick-report`: passed。
  - `npm run audit:spark-pick-report -- --assets nerf-lego-trained-output-local --max-clicks 5 --output-dir /tmp/objgauss-spark-pick-report-trained --port 5316`: passed。

### RENDER-005T-AM: Spark pick hit telemetry and selected marker

- 状态: done / pick-telemetry-and-marker-audited
- 类型: 标准 PR / 前端对象交互
- 目标: 在 `screen-space-object-pick-v1` 已能选中对象后，补上可审计 hit / ambiguity telemetry 和画布内选中视觉反馈。
- 已实施:
  - Spark viewport 新增 `data-spark-pick-status/object/distance-px/candidate-objects/ambiguous/radius-px`。
  - Spark viewport 新增 `data-spark-selected-marker-visible`，命中选中对象后在画布上显示非交互选中 marker。
  - `audit-demo` 在 Spark 删除预览后要求 pick 命中、选中对象一致、距离在 pick radius 内、候选对象数大于 0、marker 可见，并输出 `sparkPick=...`。
- 结论:
  - Lego no-SH native route 通过：`sparkCanvasSelectedObject=0`、`sparkPick="screen-space-object-pick-v1":"hit":"0":3.7:3:"true":"true"`。
  - Trained SH-heavy route 通过：`sparkCanvasSelectedObject=3`、`sparkPick="screen-space-object-pick-v1":"hit":"3":0.892:3:"true":"true"`，同时保持 `sparkMaskSource="ply-packed"` 与完整 SH rest。
  - 这证明 Spark canvas pick 有可见反馈和可机器审计 hit 质量；但两个样例当前都是 `ambiguous=true`，下一步需要多点击 hit-rate / ambiguity report，而不是直接宣称 renderer-native robust picking。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5314/ --no-server`: passed。
  - `npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5314/ --no-server`: passed。
  - `npm run audit:splat-index-mapping`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `npm run audit:spark-native-mask-gate`: passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AL: Spark canvas object selection product path

- 状态: done / spark-canvas-selection-audited
- 类型: 标准 PR / 前端对象交互
- 目标: 在 no-SH native `.splat` mask 默认化之后，补上 Spark source/original edit 路径里的画布选择交互，避免商业 demo 只能依赖对象列表选择。
- 已实施:
  - `SplatViewport` 新增 `screen-space-object-pick-v1`：点击 Spark canvas 时，把 object-aware PLY Gaussian 投影到当前 Spark camera，选择最近的可见 `object_id`。
  - Spark viewport 暴露 `data-spark-selection-mode` 和 `data-spark-selected-object`，browser audit 可直接检查选中状态。
  - `App` 只在 Spark filtered object edit 路径传入 `selectedId` / `onSelectObject`，真实查看不改变行为。
  - `audit-demo` 在删除后继续点击 Spark canvas，要求所选对象从已删除对象切到一个可见对象。
- 结论:
  - Lego no-SH native route 删除后通过 Spark canvas 重新选中对象：`sparkCanvasSelectedObject=0`，同时保持 `sparkMaskSource="native-splat"`。
  - Trained SH-heavy packed route 删除后也通过 Spark canvas 重新选中对象：`sparkCanvasSelectedObject=3`，同时保持 `sparkMaskSource="ply-packed"` 和 `sparkShRest=255794:255794:"true":45:3`。
  - 这是一条基于 object-aware PLY metadata 的产品级 screen-space pick，不是 Spark 内部 raycast；下一步如果要宣称 renderer-native picking，需要补 hit-rate / ambiguity gate。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5313/ --no-server`: passed。
  - `npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5313/ --no-server`: passed。
  - `npm run audit:splat-index-mapping`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `npm run audit:spark-native-mask-gate`: passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AK: Safe native splat mask default route

- 状态: done / native-mask-defaulted-for-no-sh
- 类型: 标准 PR / 前端渲染默认路线
- 目标: 基于 AJ 的多场景 gate，将 source/original object edit 的 no-SH 样例默认切到 native compact `.splat` mask，同时保护 SH-heavy 样例继续使用 PLY packed SH route。
- 已实施:
  - `readSparkNativeMaskMode()` 默认从 `off` 改为 `auto`。
  - `auto` 仅在 scene 无完整 SH-rest source 时启用 native compact `.splat` mask。
  - SH-heavy sample 继续默认走 `ply-packed` + `packed-sh-extract-v1`，保留 SH rest。
  - 诊断开关保留：`spark-object-source=packed` / `spark-native-mask=off` 强制 PLY packed，`spark-native-mask=on` 强制 native。
  - `audit-demo` 默认 expectation 更新为：no-SH 期望 `native-splat`，SH-heavy 期望 `ply-packed`。
  - `audit:spark-native-mask-gate` 改用静态 Vite preview，避免 dev file watcher 上限，并验证默认 route 而非强制 URL 参数。
- 结论:
  - Lego proxy 默认 source/original 删除预览已走 native compact `.splat`：`sparkMaskSource="native-splat"`、`sparkPacked="native-splat-source-v1":5696/3909:0/0`，且 `audit-demo` pixel delta 通过。
  - Plush semantic 默认 native contract gate 通过：`source="native-splat"`、`route="native-splat-source-v1"`、`visible=104403/281498`。
  - Trained SH-heavy sample 未被 native 抢走：`sparkMaskSource="ply-packed"`、`sparkPacked="packed-sh-extract-v1":255794/129108:160.4/0`、`sparkShRest=255794:255794:"true":45:3`。
  - 剩余 UX 决策：默认 native route 下 Spark 画布点击选中还未实现，商业 demo 当前应依赖对象列表选择或继续保留编辑 renderer 选中路径。
- 验证:
  - `node --check scripts/audit-spark-native-mask-gate.mjs`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-native-mask-gate`: passed。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5312/ --no-server`: passed。
  - `npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5312/ --no-server`: passed。
  - `npm run audit:splat-index-mapping`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AJ: Spark native compact splat multi-scene gate

- 状态: done / native-mask-multiscene-gated
- 类型: 标准 PR / 前端渲染验收
- 目标: 将 URL-gated native compact `.splat` object mask 从单场景 prototype 推进到 Lego + Plush 多场景候选 gate，并验证它具备默认候选资格，但不直接切默认。
- 已实施:
  - 新增 `scripts/audit-spark-native-mask-gate.mjs`，默认覆盖 `nerf-lego-alpha-closure-local` 与 `plush-semantic-closure-local`。
  - 新增 `npm run audit:spark-native-mask-gate`，使用独立 5310 端口启动本地 Vite，并打开 `?spark-native-mask=on`。
  - Gate 通过对象列表选择对象并触发删除，验证 `data-spark-mask-source="native-splat"`、`data-spark-reconstruct-source="native-splat-source-v1"`、`object-opacity-texture-v1`、visible / hidden Gaussian count 和 persistent mesh contract。
  - Plush 281k 大场景跳过重截图 delta，只验证 native source / mask / mesh contract 并保留截图证据；Lego pixel-delta 由完整 `audit-demo` 覆盖。
  - `audit-demo` 增加 `--assets` 多 asset 选择和 `--skip-visual-residual`，用于 native gate 复查时跳过无关 Spark/edit residual 截图。
- 结论:
  - Native mask 不再只是 Lego 单场景原型；Lego + Plush 均可在原始 compact `.splat` source 上使用外部 object-id mask。
  - Lego 通过 native contract，且完整 `audit-demo` 路径继续覆盖像素变化验收。
  - Plush 通过大场景 contract：`source="native-splat"`、`route="native-splat-source-v1"`、`visible=104403/281498`、`objectMask="object-opacity-texture-v1":"4096x69":104403/177095:3`。
  - 仍不默认切换：下一步需要决定 selection UX，保留 PLY packed route 诊断开关，并确认 native route 默认化不降低商业展示稳定性。
- 验证:
  - `node --check scripts/audit-spark-native-mask-gate.mjs`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-native-mask-gate`: passed。
  - `npm run audit:demo -- --assets nerf-lego-alpha-closure-local --spark-native-mask --skip-visual-residual --port 5311`: passed。
  - `npm run audit:splat-index-mapping`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AI: Spark native compact splat object mask prototype

- 状态: done / native-mask-prototype-audited
- 类型: 标准 PR / 前端渲染验收
- 目标: 基于 RENDER-005T-AH 已证明的 compact `.splat` / object-aware PLY index mapping，把 source/original object edit 从 PLY-derived packed source 进一步原型化到 Spark native compact `.splat` source + object opacity mask。
- 已实施:
  - `SplatViewport` 新增 URL-gated native mask route：`?spark-native-mask=on` 时直接创建 `SplatMesh({ url, objectModifier })`，对象显隐仍使用 `object-opacity-texture-v1`。
  - Native route 暴露 `data-spark-mask-source="native-splat"`、`data-spark-filter-mode="native-splat-mask"`、`data-spark-reconstruct-source="native-splat-source-v1"`。
  - `audit-demo` 新增 `--spark-native-mask`，要求 native route 命中原始 `.splat` source、保持 persistent mesh update contract，并继续跑 object-mask pixel-delta guard。
  - 默认 route 仍保持 PLY-derived packed source，browser audit 通过 `sparkMaskSource="ply-packed"` 与 native route 区分。
- 结论:
  - Lego proxy native audit 通过：`postDelete="spark-splat":"spark-object-opacity-mask":3909`、`sparkMaskSource="native-splat"`、`sparkPacked="native-splat-source-v1":5696/3909:0/0`、`sparkObjectMask="object-opacity-texture-v1":"4096x2":3909/1787:4`、`sparkMesh="persistent-splatmesh-v1":1:"true":4`。
  - 这一步解释了“自身颜色为什么颗粒感”的核心边界：默认 packed-source Spark mask 已不是点云 fallback，native route 进一步证明原始 compact `.splat` 也能挂 object mask；剩余是否可商用默认化取决于多场景质量、index gate 和 selection UX。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5302/ --no-server --spark-native-mask`: passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5302/ --no-server`: passed。
  - `npm run audit:splat-index-mapping`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。

### RENDER-005T-AH: Splat / PLY index mapping audit

- 状态: done / index-mapping-audited
- 类型: 标准 PR / 前端渲染验收
- 目标: 评估 original compact `.splat` 与 object-aware PLY packed source 的稳定 Gaussian index mapping，判断 Spark object mask 是否能从 PLY-derived packed source 继续推进到 original source / native `.splat` object masking。
- 已实施:
  - 新增 `scripts/audit-splat-index-mapping.mjs`，直接解析 compact 32-byte `.splat` rows 和 object-aware PLY vertices。
  - Audit 检查 count、逐 index position delta、逐 index scale delta、rounded-position multiset coverage、重复 position key 和 object_id 范围。
  - 新增 `npm run audit:splat-index-mapping`，默认输出 `/tmp/objgauss-splat-index-mapping/summary.json` 与 `summary.md`。
  - `docs/benchmarks/spark-filtered-edit.md` 记录该 gate 和 native-mask 解释边界。
- 结论:
  - 5 个 public/generated Gaussian 样例全部保序：`plush-3dgs-local`、`plush-v1-closure-local`、`plush-semantic-closure-local`、`nerf-lego-alpha-closure-local`、`nerf-lego-trained-output-local` 均 `indexMatches=count`、`maxPositionDelta=0`、`maxScaleDelta=0`、`positionMultisetCoverage=1`。
  - 对当前 ObjGauss 生成/登记 public samples，可以把 object-aware PLY 的 `object_id` 作为按 Gaussian index keyed 的外部 mask 输入。
  - 该结论不证明任意第三方 compact `.splat` 自带 object_id，也不等价于 original source/native mask runtime 已接线。
- 验证:
  - `node --check scripts/audit-splat-index-mapping.mjs`: passed。
  - `npm run audit:splat-index-mapping`: passed。

### RENDER-005T-AG: Spark object opacity mask visual delta guard

- 状态: done / visual-delta-audited
- 类型: 标准 PR / 前端渲染验收
- 目标: 补上 AF 后的关键证据缺口：不只检查 `data-spark-object-mask-*` telemetry，还要证明 Spark object opacity mask 的显隐变化真的改变 canvas 像素。
- 已实施:
  - `audit-demo` 在小场景 Spark mask restore stress 中新增三帧 `canvasVisualStats`：delete baseline、hide-one-object、restore。
  - 新增 `spark-object-mask-visual-delta-v1` 校验：hide 后 checksum 必须变化，coverage / luma / chroma 至少一个达到最小 delta；restore 后必须回到 delete baseline 或非常接近。
  - Browser audit 输出新增 `sparkMaskVisual`，记录 before / hidden / restored checksum、hide delta 和 restore delta。
  - Heavy trained scene 继续跳过该像素压力循环，保留 SH-heavy delete contract，避免把重场景 browser audit 变成高成本性能压力测试。
- 结论:
  - Lego proxy 小场景证明 opacity mask 影响真实渲染像素：`sparkMaskVisual="spark-object-mask-visual-delta-v1":"4a2ed0e8"/"be002ca4"/"4a2ed0e8":0.000752/0.014063/0.026019:0/0/0`。
  - Trained Lego 继续通过 SH-heavy contract：`sparkObjectMask="object-opacity-texture-v1":"4096x63":129108/126686:2`、`sparkShRest=255794:255794:"true":45:3`。
  - 剩余问题不是 telemetry 可信度，而是 original compact `.splat` 与 object-aware PLY packed source 的 stable index mapping / native source mask 可行性。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5301/ --no-server`: passed。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5301/ --no-server`: passed。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AF: Spark object opacity mask over packed source

- 状态: done / object-opacity-mask-audited
- 类型: 标准 PR / 前端渲染性能与 UX 诊断
- 目标: 回应“自身颜色为什么有颗粒感、像没走高斯”的问题，把 source/original object edit 的显隐从每个 visible set 的 display `PackedSplats.extractSplats(...)` 推进到 Spark shader/object-state 层，并明确剩余颗粒感来自 object 子集稀疏、边界 assignment 和仍非 original `.splat` 内部 mask。
- 已实施:
  - 新增 `src/sparkObjectMask.js`，构建 `object-opacity-texture-v1` `Uint32` mask texture，并通过 Spark Dyno `objectModifier` 按 Gaussian index 将隐藏对象 opacity 置零。
  - `SplatViewport` 的 filtered Spark route 现在保留同一个 base `PackedSplats` 和 `SplatMesh`，object-state 变化只更新 mask texture 并标记 `splat.needsUpdate`。
  - display `PackedSplats` LRU cache / per-state extract 在 native mask route 下禁用，浏览器 contract 要求 `data-spark-packed-extract-ms="0.000"` 和 `data-spark-display-cache-mode="disabled-by-native-mask-v1"`。
  - Browser telemetry 新增 `data-spark-object-mask-*`，记录 mask size、updates、visible / hidden Gaussian counts；small-scene audit 继续跑 hide / restore stress，heavy trained audit 跳过该压力循环，专注 SH-preserved delete contract。
  - `audit-demo` 增加 browser close timeout，避免重场景 WebGL/Spark 关闭阶段阻塞验收输出。
- 结论:
  - 当前“原始颜色 / 自身颜色”在 object edit active 后已经可以走 `spark-splat` + `spark-object-opacity-mask`，不是点云 fallback；trained sample 删除后仍保留 `packed-sh-extract-v1` 和完整 degree-3 SH rest。
  - 视觉上仍可能有颗粒感，因为隔离 / 删除后显示的是 object_id 子集，隐藏对象不再参与透明混合，边界和错误 assignment 会直接暴露稀疏 Gaussian；这不是“没有高斯”，而是 object-level mask 质量与子集密度问题。
  - 这一步消除了 object-state 变化时的 display extract contract，但仍不是 original compact `.splat` 内部 object mask；下一步需要 pixel-delta guard 和 original `.splat` index mapping 评估。
- 验证:
  - `node --check src/sparkObjectMask.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5300/ --no-server`: passed；`postDelete="spark-splat":"spark-object-opacity-mask":3909`、`sparkObjectMask="object-opacity-texture-v1":"4096x2":3909/1787:4`、`sparkPacked="packed-extract-v1":5696/3909:4.4/0`。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5300/ --no-server`: passed；`postDelete="spark-splat":"spark-object-opacity-mask":129108`、`sparkObjectMask="object-opacity-texture-v1":"4096x63":129108/126686:2`、`sparkShRest=255794:255794:"true":45:3`。
  - Focused Playwright probe verified trained isolate/delete route: isolate `40747/215047`, delete `215047/40747`, `route="packed-sh-extract-v1"`, `extractMs="0.000"`, `sh=255794:255794:true:45:3`。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `git diff --check`: passed。

### RENDER-005T-AE: Spark filtered persistent SplatMesh update surface

- 状态: done / persistent-splatmesh-audited
- 类型: 标准 PR / 前端渲染性能与 UX 诊断
- 目标: 在 Spark filtered edit route 已有 display `PackedSplats` cache 的基础上，去掉每次 object-state 变化都创建临时 `SplatMesh` 的 browser-visible contract，改为保留一个 filtered Spark mesh 并更新其 packed source。
- 已实施:
  - `SplatViewport` 拆分普通 source `SplatMesh` lifecycle 与 filtered Spark lifecycle；filtered route 只在 base `PackedSplats` source 变化时创建 mesh。
  - visible / removed / isolated object-state 变化时，filtered route 从 display cache 或 `extractSplats(...)` 取得 display `PackedSplats`，并更新现有 `SplatMesh` 的 packed source、mapping version 和 generator。
  - `SplatMesh.dispose()` 前继续摘掉 cached display packed 引用，避免释放 LRU cache 持有的数据。
  - Browser contract 新增 `data-spark-mesh-update-mode="persistent-splatmesh-v1"`、`data-spark-mesh-id`、`data-spark-mesh-reused` 和 `data-spark-mesh-updates`。
  - `audit-demo` 在删除后隐藏 / 恢复一个未删除对象，要求 Spark display cache hit，同时要求 mesh id 保持一致、reused 为 true、updates 增长。
- 结论:
  - Lego proxy 删除预览通过：`sparkPacked="packed-extract-v1":5696/3909:3.8/0`、`sparkDisplayCache="visible-index-lru-v1":"true":2:2/2/0`、`sparkMesh="persistent-splatmesh-v1":1:"true":4`。
  - Trained Lego SH-heavy 删除预览通过：`sparkPacked="packed-sh-extract-v1":255794/129108:155.5/0`、`sparkDisplayCache="visible-index-lru-v1":"true":2:2/2/0`、`sparkMesh="persistent-splatmesh-v1":1:"true":4`、`sparkShRest=255794:255794:"true":45:3`。
  - 这一步减少 UI 交互中的重建感，并把 Spark filtered source 更新变成可审计事实；它仍不是 Spark 原生 object mask，每个全新 visible set 仍需要 display `PackedSplats` extract 或 cache hit。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5299/ --no-server`: passed。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5299/ --no-server`: passed。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `uv run --extra dev pytest`: 41 passed。

### RENDER-005T-AD: Spark display PackedSplats cache telemetry

- 状态: done / display-cache-audited
- 类型: 标准 PR / 前端渲染性能与 UX 诊断
- 目标: 回应“自身颜色有颗粒感、不像高斯”后续问题里暴露的 Spark filtered edit 开销：在不改默认视觉参数、不伪装 native `.splat` object mask 的前提下，减少回到同一 visible-index set 时重复 `PackedSplats.extractSplats(...)` 的成本，并把路径事实暴露给 UI / browser audit。
- 已实施:
  - `SplatViewport` 为 filtered Spark route 增加 `visible-index-lru-v1` display `PackedSplats` cache，默认保留最近 4 个 visible-index set。
  - 由于 Spark `SplatMesh.dispose()` 会释放传入的 `PackedSplats`，filtered viewport 在清理临时 mesh 前会摘掉 cached packed 引用，由 cache 自己统一释放。
  - Spark HUD 增加路径提示：`PLY SH 源` / `PLY 源` / `过滤重建` / `缓存过滤` / `PLY 重建`，减少用户把“原始颜色（编辑预览）”误解成原始 `.splat` 内部删除。
  - Browser contract 新增 `data-spark-display-cache-*` telemetry：mode、key、hit、size、hits、misses、evictions。
  - `audit-demo` 删除后会隐藏并恢复一个未删除对象，要求回到同一 visible-index set 时 `data-spark-display-cache-hit="true"` 且 hit 计数增长。
- 结论:
  - Lego proxy 删除预览通过：`sparkPacked="packed-extract-v1":5696/3909:3.9/1.7`、`sparkDisplayCache="visible-index-lru-v1":"true":2:1/2/0`。
  - Trained Lego SH-heavy 删除预览通过：`sparkPacked="packed-sh-extract-v1":255794/129108:153.8/37.5`、`sparkDisplayCache="visible-index-lru-v1":"true":2:1/2/0`、`sparkShRest=255794:255794:"true":45:3`。
  - 这一步减少重复 extract，并把“缓存过滤 vs 过滤重建”做成可见/可审计事实；它仍不是 Spark 原生 object mask，每个全新 visible set 仍会创建临时 `SplatMesh`。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5298/ --no-server`: passed。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5298/ --no-server`: passed。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `uv run --extra dev pytest`: 41 passed。

### RENDER-005T-AC: Spark PLY SH full-view source baseline

- 状态: done / sh-source-baseline-audited
- 类型: 标准 PR / 前端渲染质量与验收
- 目标: 为 trained SH-heavy demo 建立 SH-capable full-view source / residual baseline，避免继续用不携带 degree-3 SH rest 的 compact `.splat` 作为完整外观对照。
- 已实施:
  - `App` 在 `真实查看`、source/original、无 object edit active 且场景有完整 SH rest 时，自动使用 Spark PLY packed source，暴露 `data-object-filter="spark-ply-sh-source"`；`?spark-ply-source=off` 可强制回到 legacy compact `.splat` 诊断路径。
  - `SplatViewport` 新增 `reconstructRole="source"`，区分 full-view PLY source 与 edit/filter reconstruction，source 路径仍复用 `packed-sh-extract-v1` 和 SH extra preservation。
  - `audit-spark-reconstruct-residual` 允许 full source 为 `none` / `spark-ply-source` / `spark-ply-sh-source`，并对 SH full source 校验 route、visible count 和 preserved SH telemetry。
  - `audit-demo` 的 canvas visual stats 改用 page clip + 60s timeout，避免大场景 SH source 在 element screenshot 稳定等待阶段误超时。
- 结论:
  - 默认 no-SH Lego proxy 不变：`fullSource="none":"none":0:0:false:0:0`，`reconstructSource="packed-extract-v1"`，residual gate 继续通过。
  - 本机 trained Lego full-view source 已变成 SH-capable PLY source：`fullSource="spark-ply-sh-source":"packed-sh-extract-v1":255794:255794:true:45:3`。
  - Trained same-source residual gate 通过：`coverageRatio=1.170018`、`lumaDelta=0.058189`、`chromaDelta=0.007036`，解决了 AB 后“PLY reconstruction 比 compact `.splat` baseline 保留更多 SH，导致 residual 失败”的基准不一致问题。
  - Trained browser interaction audit 通过，真实查看非背景像素提升到 `70188`，删除后继续保留 `sparkPacked="packed-sh-extract-v1"`、`sparkShRest=255794:255794:"true":45:3`。
- 验证:
  - `node --check scripts/audit-spark-reconstruct-residual.mjs`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `node scripts/audit-spark-reconstruct-residual.mjs --assets nerf-lego-trained-output-local --output-dir /tmp/objgauss-spark-reconstruct-residual-trained-ac`: passed。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5294/ --no-server`: passed。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5294/ --no-server`: passed。

### RENDER-005T-AB: Spark packed SH-rest preservation

- 状态: done / sh-rest-preserved-audited
- 类型: 标准 PR / 前端渲染质量与验收
- 目标: 解释并修复“删除后的自身颜色有颗粒感、不像高斯”的核心数据路径问题：Spark filtered PLY reconstruction 之前只保留 base packed splat，不保留 PLY `f_rest_*` SH rest。
- 已实施:
  - 新增 `src/sparkPackedSh.js`，复用 Spark `utils.encodeSh1Rgb/encodeSh2Rgb/encodeSh3Rgb`，按 Spark PLY parser 的 channel-major 到 basis-major 映射把 `f_rest_*` 打包为 `extra.sh1/sh2/sh3`。
  - `SplatViewport` 在 filtered Spark 路径中接收 scene-level `shRestCoefficients` / `shRestCoefficientCount`，构建 base `PackedSplats.extra`，并在 visible-index extract 后拷贝对应 SH extra。
  - SH-preserved 路径优先用 `f_dc_*` 作为 Spark base color，避免 object-colored `red/green/blue` debug fields 污染“原始颜色” filtered preview。
  - Browser contract 区分 `packed-extract-v1` 和 `packed-sh-extract-v1`，并暴露 `data-spark-sh-rest-source-gaussians`、`data-spark-sh-rest-preserved-gaussians`、`data-spark-sh-rest-coefficients`、`data-spark-sh-degree`。
  - `audit-demo` / `audit-spark-reconstruct-residual` 支持 no-SH asset 继续走旧 route，同时要求 SH-heavy asset preserved count 等于 source count。
  - `audit:webgpu-tile-smoke` 增加 Spark SH helper smoke，直接验证 degree-3 SH encode 与 extract 后 extra preservation。
- 结论:
  - 默认 Lego proxy 仍走 no-SH 兼容路径：`sparkPacked="packed-extract-v1":5696/3909:5.3/4.8`，`sparkShRest=0:0:"false":0:0`。
  - 本机 trained Lego 删除预览已走 SH-preserved Spark filtered route：`sparkPacked="packed-sh-extract-v1":255794/129108:171.2/41.7`，`sparkShRest=255794:255794:"true":45:3`。
  - Trained full reconstruct diagnostic 证明 SH telemetry 正确：`reconstructSource="packed-sh-extract-v1"`、`shRest=255794:255794:true:45:3`；但它相对 registered `.splat` 的 visual residual 仍失败，因为 ObjGauss compact `.splat` viewer source 不携带 degree-3 SH rest，PLY reconstruction 现在比 `.splat` baseline 保留更多 view-dependent appearance。
- 验证:
  - `node --check src/sparkPackedSh.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-spark-reconstruct-residual.mjs`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:spark-reconstruct-residual`: passed；沙箱内本地 fetch 被拒，提权重跑通过。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5294/ --no-server`: passed。
  - `npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5294/ --no-server`: passed。

### RENDER-005T-AA: Spark packed extract reconstruction route

- 状态: done / packed-extract-audited
- 类型: 标准 PR / 前端渲染性能与验收
- 目标: 把 Spark filtered edit 从每次 raw object-aware PLY `constructSplats` 重建推进到可复用 `PackedSplats` base + visible-index extraction 的过渡层，并把 SH-rest 未保留事实暴露成 browser contract。
- 已实施:
  - `SplatViewport` 在 filtered Spark 路径中构建 base `PackedSplats` cache，再用 `visibleIndices -> PackedSplats.extractSplats(...)` 创建当前显示用 `SplatMesh(packedSplats)`。
  - 新增 runtime telemetry：`data-spark-reconstruct-source="packed-extract-v1"`、base Gaussian 数、visible index 数、base build / extract 毫秒，以及 `data-spark-sh-rest-preserved="false"`。
  - `audit-demo` 删除预览验收现在要求 Spark filtered route 命中 `packed-extract-v1`，并校验 visible indices、timing 和 SH preservation contract。
  - `audit-spark-reconstruct-residual` 的 full PLY reconstruction gate 也要求 `packed-extract-v1`，并把 packed / SH telemetry 写入 summary report。
  - `docs/benchmarks/spark-filtered-edit.md` 更新当前 runtime contract、验证结果和剩余 gap。
- 结论:
  - Lego proxy full reconstruct gate 继续通过：`coverageRatio=1.170841`、`lumaDelta=0.029762`、`chromaDelta=0.028407`，且 `reconstructSource="packed-extract-v1"`、`packed=5696/5696:4.2/2.7`、`shRest=0:false`。
  - Lego delete preview 通过静态 preview browser audit：`postDelete="spark-splat":"spark-filtered-ply-reconstruct":3909`，`sparkPacked="packed-extract-v1":5696/3909:3.9/1.9`。
  - 这一步解决 raw rebuild contract，但仍不是原始 `.splat` 内部 object mask；`PackedSplats.extractSplats` 仍不保留 SH rest，所以 trained SH-heavy sample 的商业展示级外观还需要 RENDER-005T-AB。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-spark-reconstruct-residual.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5294/ --no-server`: passed；默认 dev-server audit 因系统 watcher `ENOSPC` 失败，改用 built static preview 复查通过。

### RENDER-005T-Z: Spark PLY reconstruction residual gate

- 状态: done / spark-reconstruct-residual-audited
- 类型: 标准 PR / 前端渲染质量与验收
- 目标: 把 full `.splat` Spark 与 object-aware PLY reconstructed Spark 的视觉差距变成可重复、可阈值化的 browser gate，避免凭主观观感判断 filtered Spark 路线是否可继续扩大。
- 已实施:
  - 新增 URL probe `spark-reconstruct-probe=1`，只在显式 probe 下让 `对象编辑 / 原始颜色` 进入全量 PLY reconstructed Spark；正常用户路径不变。
  - `SplatViewport` 区分 `data-object-filter="spark-ply-reconstruct"` 和 `spark-filtered-ply-reconstruct`，全量重建时要求 `filteredGaussians=0`。
  - 新增共享 `scripts/lib/visual-stats.mjs`，让 `audit-demo` 和 Spark reconstruction gate 共用同一套 PNG coverage / luma / chroma 统计。
  - 新增 `scripts/audit-spark-reconstruct-residual.mjs`、`npm run audit:spark-reconstruct-residual` 和可选 `npm run audit:spark-reconstruct-residual-multiscene`，输出 `/tmp/objgauss-spark-reconstruct-residual*/summary.*`。
- 结论:
  - Lego proxy 默认 gate 通过：`coverageRatio=1.170841`、`lumaDelta=0.029762`、`chromaDelta=0.028407`、`objectFilter="spark-ply-reconstruct"`、`visibleGaussians=5696`。
  - Plush semantic 可选复查也通过：`coverageRatio=1.303149`、`lumaDelta=0.049406`、`chromaDelta=0.002846`、`visibleGaussians=281498`，但耗时约 70 秒，说明大场景 full rebuild 不应成为高频交互路径。
  - 当前 filtered Spark 路线已具备 residual gate；下一步主要缺口是 SH-rest preservation 和大场景重建性能。
- 验证:
  - `node --check scripts/lib/visual-stats.mjs`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-spark-reconstruct-residual.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:spark-reconstruct-residual`: passed。
  - `npm run audit:spark-reconstruct-residual-multiscene`: passed。

### RENDER-005T-Y: Spark filtered edit feasibility implementation

- 状态: done / spark-filtered-edit-audited
- 类型: 标准 PR / 前端渲染质量与交互
- 目标: 回应“删除后的自身颜色仍不像真实高斯”的 UX 问题，评估并落地把 object filter 接入 Spark renderer 的最小可用路径。
- 已实施:
  - `SplatViewport` 新增 filtered-points 构造模式：使用 Spark `constructSplats` / `pushSplat` 从 object-aware PLY points 重建 filtered `SplatMesh`。
  - `App` 在 `renderMode=original`、object edit active、source color tuning 时优先用 `Spark 过滤 Splat` 渲染隔离/删除后的剩余场景。
  - SH-view diagnostics、对象色模式和 canvas click selection 继续走 WebGPU Tile / Gaussian OIT 编辑路径，避免把当前 RGB/SH-DC-only Spark reconstruction 误当成完整 SH route。
  - `audit-demo` 支持删除后 Spark filtered contract，验证 `data-object-filter="spark-filtered-ply-reconstruct"`、剩余 Gaussian 数和 source color 状态。
  - 新增 `docs/benchmarks/spark-filtered-edit.md` 记录 runtime contract、验证命令和剩余 gap。
- 结论:
  - Spark object filter 的最小可行路线不是直接 patch Spark 内部 shader，而是从 ObjGauss object-aware PLY 生成 filtered Spark `SplatMesh`。
  - 该路线已经解决删除/隔离后“原始颜色预览仍留在近似编辑 renderer”的核心 UX gap，但仍不保留 trained SH-heavy sample 的完整 SH rest，也不支持 Spark canvas click-to-object。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5294/ --no-server`: passed；`postDelete="spark-splat":"spark-filtered-ply-reconstruct":3909`。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5295 --probes full`: passed，headed desktop WebGPU route；隔离/选择仍走 WebGPU Tile，删除后 `postDelete="spark-splat":"spark-filtered-ply-reconstruct":3104`。

### RENDER-005T-X: WebGPU alpha floor multi-scene candidate gate

- 状态: done / alpha-floor-multiscene-gate-audited
- 类型: 标准 PR / 前端渲染质量诊断
- 目标: 基于 RENDER-005T-W 里 `alpha10` 在 trained Lego 上同时改善 coverage 和 luma 的事实，把 alpha presentation floor 候选扩展到稳定的 NeRF Lego proxy + Plush semantic 多场景 gate，判断是否可作为默认候选。
- 已实施:
  - 新增 `npm run audit:webgpu-alpha-floor-sweep`，固定两场景、四个 alpha floor variants 和报告输出目录 `/tmp/objgauss-webgpu-alpha-floor-sweep`。
  - 新增 `npm run audit:webgpu-alpha-floor-candidate-gate`，以 `alpha10` 为 strict candidate gate，并要求 mean / per-scene pareto、luma、chroma 全部不劣于 baseline。
  - `docs/benchmarks/webgpu-coverage-sweep.md` 新增 alpha floor multi-scene gate 用法、当前结果表和失败解释。
- 结论:
  - NeRF Lego proxy `alpha10`: coverage ratio `3.190749`、luma/chroma `0.079933/0.075462`、Pareto `0.851494`，优于 baseline `3.784251`、`0.106079/0.086537`。
  - Plush semantic `alpha10`: coverage ratio `6.082743`、luma `0.102588` 优于 baseline `6.448639`、`0.112667`，但 chroma 从 `0.010651` 恶化到 `0.015819`。
  - `alpha10` 是 best mean Pareto variant (`0.965287`)，但 strict gate 失败：mean chroma norm=`1.178616`、Plush per-scene Pareto=`1.07908`、Plush chroma norm=`1.485213`。
  - 因此 alpha presentation floor 仍是候选/诊断轴，默认 `0.035` 不变；下一步应转向 Spark renderer object filter feasibility 或 chroma-aware alpha presentation。
- 验证:
  - `npm run audit:webgpu-coverage-sweep -- --assets nerf-lego-alpha-closure-local,plush-semantic-closure-local --port 5291 --variants baseline:2.2:4:0.035,alpha05:2.2:4:0.05,alpha075:2.2:4:0.075,alpha10:2.2:4:0.1 --output-dir /tmp/objgauss-webgpu-alpha-floor-multiscene --gate-variant alpha10 --max-mean-pareto-score 1 --max-mean-luma-norm 1 --max-mean-chroma-norm 1 --max-scene-pareto-score 1 --max-scene-luma-norm 1 --max-scene-chroma-norm 1 --allow-failures`: completed，suite passed，strict gate failed as expected，headed desktop WebGPU 2 scenes x 4 variants。
  - `npm run audit:webgpu-alpha-floor-candidate-gate -- --port 5292 --allow-failures`: completed，复现同一 strict gate failure，报告写入 `/tmp/objgauss-webgpu-alpha-floor-candidate-gate/summary.*`。

### RENDER-005T-W: WebGPU alpha presentation floor diagnostic

- 状态: done / alpha-presentation-floor-audited
- 类型: 标准 PR / 前端渲染质量诊断
- 目标: 在 SH-view coverage sweep 证明 footprint tightening 降 coverage 但伤 luma 后，将 presentation alpha floor 从硬编码常量推进成 runtime tuning，并判断低 alpha halo 是否解释 trained Lego 的 coverage 膨胀。
- 已实施:
  - `webgpuTileResolveShader` 新增 `runtime-alpha-presentation-tuning-v1`，支持 `webgpu-alpha-presentation-floor`，合法范围 `0-0.2`，默认仍为 `0.035`。
  - `WebGpuTileViewport` 按 URL tuning 生成 resolve shader，并暴露 `data-webgpu-alpha-presentation-tuning-mode` / `data-webgpu-alpha-presentation-floor`。
  - `audit-demo` / `audit-webgpu-desktop` 校验 requested floor 与浏览器 runtime telemetry 一致。
  - `audit-webgpu-coverage-sweep` 支持固定 floor，也支持 variant 格式 `id:footprint:maxAnisotropy:alphaFloor`。
  - `docs/benchmarks/webgpu-coverage-sweep.md` 新增 alpha presentation floor sweep 用法和 trained Lego 结果。
- 结论:
  - trained Lego + SH-view baseline `0.035`: coverage ratio=`31.205176`、luma/chroma=`0.034507/0.055774`。
  - floor `0.05`: coverage ratio=`29.156993`、luma/chroma=`0.024444/0.055686`。
  - floor `0.075`: coverage ratio=`26.456439`、luma/chroma=`0.010019/0.055521`。
  - floor `0.1`: coverage ratio=`24.248059`、luma/chroma=`0.00276/0.055336`，best Pareto score=`0.690001`。
  - 与 footprint tightening 不同，alpha presentation floor 在该 trained scene 上同时改善 coverage 和 luma；这是更强候选轴，但仍需多场景 gate，不能直接改默认。
- 验证:
  - `node --check src/webgpuTileResolveShader.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `node --check scripts/audit-webgpu-coverage-sweep.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-coverage-sweep -- --asset nerf-lego-trained-output-local --port 5290 --webgpu-color-mode sh-view --variants baseline:2.2:4:0.035,alpha05:2.2:4:0.05,alpha075:2.2:4:0.075,alpha10:2.2:4:0.1 --output-dir /tmp/objgauss-webgpu-alpha-floor-trained-sh-view`: passed，headed desktop WebGPU 4 variants。

### RENDER-005T-V: WebGPU SH-view coverage sweep

- 状态: done / sh-view-coverage-sweep-audited
- 类型: 标准 PR / 前端渲染质量诊断
- 目标: 在 SH-view 已显著降低 trained Lego luma/chroma residual 后，把 coverage sweep 扩展到 `webgpu-color-mode=sh-view`，判断颜色已修正后 footprint / anisotropy 收紧是否能解决颗粒感 / 膨胀感。
- 已实施:
  - `audit-webgpu-coverage-sweep` 新增 `--webgpu-color-mode source|sh-view` 透传到 headed desktop WebGPU audit。
  - Sweep parser 记录 `colorMode`、`shViewGaussians` 和 `shViewAfterDelete`，并写入 `summary.json` / `summary.md` rows。
  - `docs/benchmarks/webgpu-coverage-sweep.md` 新增 SH-view coverage sweep 用法和当前 trained Lego 结果表。
- 结论:
  - trained Lego SH-view baseline: coverage ratio=`31.205176`、luma/chroma=`0.034507/0.055774`、`shViewAfterDelete=255794`。
  - compact: coverage ratio=`25.958842`，但 luma delta 恶化到 `0.070796`。
  - tight: coverage ratio=`23.164633`、tile refs 降到 `525755`，但 luma delta 恶化到 `0.093626`，约为 baseline 的 `2.71x`。
  - 因此 footprint tightening 只能作为诊断轴，不能作为默认修复；“有颗粒感 / 不像高斯”的下一步应看 presentation coverage threshold、alpha path，或 Spark renderer object filter feasibility。
- 验证:
  - `node --check scripts/audit-webgpu-coverage-sweep.mjs`: passed。
  - `npm run audit:webgpu-coverage-sweep -- --asset nerf-lego-trained-output-local --port 5289 --webgpu-color-mode sh-view --output-dir /tmp/objgauss-webgpu-coverage-trained-sh-view`: passed，headed desktop WebGPU 3 variants。

### RENDER-005T-U: WebGPU SH-view color diagnostic

- 状态: done / sh-view-color-audited
- 类型: 标准 PR / 前端渲染质量诊断
- 目标: 在 SH-rest presence audit 已证明 trained Lego 带完整 degree-3 SH 后，增加可切换的 view-dependent SH 颜色诊断模式，判断“原始颜色（编辑预览）不像 Spark”的颜色残差是否来自未使用 SH rest。
- 已实施:
  - 前端 PLY parser 在 RGB 原始色存在时仍保留 raw `f_dc`，并把 `f_rest_*` 打包成 scene 级 typed array，避免把 255k+ Gaussian 的完整 SH rest 塞进每个 point 对象。
  - WebGPU Tile 新增 `runtime-color-tuning-v1`，支持 URL / audit 参数 `webgpu-color-mode=source|sh-view`；默认保持 `source`。
  - `sh-view` 只在 `原始颜色（编辑预览）` 生效，对象调试色仍使用 object palette，避免把 object interaction 审计和 SH 颜色诊断混在一起。
  - `WebGpuTileViewport` / `PointCloudViewport` 暴露 color tuning telemetry；`audit-demo` 校验删除预览切回自身颜色后 `shViewAfterDelete` 生效。
  - `audit:webgpu-tile-smoke` 新增合成 SH-view buffer 差异检查，确保 source 默认不变、SH-view 可改变 packed color buffer。
- 结论:
  - source baseline headed audit：`colorTuning=runtime-color-tuning-v1:source:0`，删除预览后 `colorAfterDelete=255794/0/0/0`、`shViewAfterDelete=0`。
  - `sh-view` headed audit：删除预览后 `shViewAfterDelete=255794`，说明 trained Lego 的全部 Gaussian 都走了 degree-3 SH view-dependent color。
  - `sh-view` 将 trained Lego 的 luma/chroma residual 从 source 的 `0.090165/0.071164` 降到 `0.034507/0.055774`，证明颜色差距中 SH 是真实主因之一。
  - 但 coverage ratio 仍从 source 的 `31.102403` 到 `31.205176`，没有解决颗粒感 / 膨胀感；下一步应转向 footprint / alpha / presentation 或 Spark object filter feasibility。
- 验证:
  - `node --check src/ply.js`: passed。
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check src/webgpuCapability.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5287 --probes full`: passed，headed desktop WebGPU source baseline。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5288 --probes full --webgpu-color-mode sh-view`: passed，headed desktop WebGPU SH-view diagnostic。

### RENDER-005T-T: WebGPU SH-rest presence audit

- 状态: done / sh-rest-presence-audited
- 类型: 标准 PR / 前端渲染质量诊断
- 目标: 在 front-top-k sorted-alpha 证明 per-pixel 排序路径可运行但不是默认候选后，把 “原始颜色（编辑预览）不像 Spark” 的 SH / view-dependent color 假设变成可审计事实，先判断当前素材是否含有未使用的 SH rest 信息。
- 已实施:
  - 前端 PLY parser 解析 `f_rest_*` presence 元数据，为每个 point 记录 `shRestCoefficientCount` 和推断 `shDegree`；为避免 255k+ Gaussian 场景额外内存膨胀，本切片不把完整 SH rest 系数复制到 per-point JS array。
  - `buildWebGpuTileSmoke` / renderer contract 暴露 `colorShRestGaussians`、`colorShRestCoefficientMax` 和 `colorShDegreeMax`，并保持 RGB / SH-DC / fallback / object-palette 色源统计不变。
  - `WebGpuTileViewport` 与 `PointCloudViewport` 均暴露 `data-webgpu-color-sh-rest-*`，确保 WebGPU route 和 Gaussian OIT fallback audit 口径一致。
  - `audit-demo` 日志新增 `shRest=count/maxCoeffs/maxDegree`，并校验该 telemetry 与 packed Gaussian 数一致。
  - `audit:webgpu-tile-smoke` 新增内存 PLY parser smoke 和合成 SH-rest point smoke，验证 `9 -> degree 1` 与 `45 -> degree 3` 两条路径。
- 结论:
  - 默认 sample smoke 仍为 `shRest=0/0/0`，说明新增 telemetry 不改变默认渲染输出。
  - 本机 `NeRF Lego 训练输出样例` headed WebGPU full audit 显示 `shRest=255794/45/3`，而删除预览后 `colorAfterDelete=255794/0/0/0`；因此该 trained sample 确实带完整 degree-3 view-dependent SH，但当前编辑预览仍只使用 RGB / SH-DC 派生颜色。
  - 当前“自身颜色编辑预览不像 Spark”的下一步应优先做 `sh-view-color` 诊断模式或 Spark object filter feasibility，而不是继续盲目调 footprint/depth bins。
- 验证:
  - `node --check src/ply.js`: passed。
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check src/webgpuCapability.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；输出 `shRest=0/0/0`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5286 --probes full`: passed，headed desktop WebGPU full runtime；输出 `shRest=255794/45/3`、`colorAfterDelete=255794/0/0/0`。

### RENDER-005T-S: WebGPU front-top-k sorted-alpha diagnostic

- 状态: done / sorted-alpha-diagnostic-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 camera-mode 诊断显示 Spark-frame 主要改善 coverage 但未解决 luma/chroma 后，把真实 per-pixel sorted alpha 推进成可审计 WebGPU runtime mode，判断排序方向是否能解释 “原始颜色（编辑预览）不像 Spark”。
- 已实施:
  - `runtime-depth-sort-tuning-v1` 新增 `depthAlphaMode=depth-binned|front-top-k`，URL / audit 参数为 `webgpu-depth-alpha-mode`；默认仍保持 `depth-binned`。
  - `front-top-k` 在每个像素扫描 tile entries 后，保留最近 K 个有效 Gaussian contributor，并按前到后 alpha compositing；K 复用 `webgpu-depth-bins`，合法范围仍为 4-16。
  - WebGPU WGSL pixel resolve 与 CPU smoke reference 使用同一 top-K 插入排序逻辑，`buildWebGpuTileSmoke` 暴露 `pixelDepthAlphaMode` 和 `front-top-k-alpha-composite-v1` contract。
  - `WebGpuTileViewport` / `PointCloudViewport` DOM 暴露 `data-webgpu-pixel-depth-alpha-mode`。
  - `audit-demo` / `audit-webgpu-desktop` / coverage sweep / depth sweep 支持 `--webgpu-depth-alpha-mode`，并校验 DOM telemetry 命中 requested alpha mode。
- 结论:
  - 默认 Lego headed WebGPU full audit 保持 baseline：coverage ratio=`3.784251`、luma/chroma=`0.106079/0.086537`。
  - Lego `front-top-k` K=8 通过 headed WebGPU full audit：coverage ratio=`3.583371`，但 luma/chroma 恶化到 `0.208595/0.127958`。
  - Lego `front-top-k` K=16 通过 headed WebGPU full audit：coverage ratio=`3.778381`，luma/chroma=`0.173505/0.113605`，仍弱于 baseline。
  - Plush semantic `front-top-k` K=8 通过大场景 headed WebGPU full audit：coverage ratio=`6.115472`，luma/chroma=`0.245489/0.077452`。
  - 因此 front-top-k sorted alpha 是可运行诊断路径，但当前不是默认候选；“自身颜色编辑预览不像 Spark”的下一主因更可能在 SH/view-dependent color、Spark 合成路径差异或 object filter 未接入 Spark renderer。
- 验证:
  - `node --check src/webgpuDepthTuning.js`: passed。
  - `node --check src/webgpuTileComputeShader.js`: passed。
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check src/webgpuCapability.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `node --check scripts/audit-webgpu-coverage-sweep.mjs`: passed。
  - `node --check scripts/audit-webgpu-depth-sweep.mjs`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5281 --probes full`: passed，headed desktop WebGPU full runtime。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5282 --probes full --webgpu-depth-alpha-mode front-top-k`: passed，headed desktop WebGPU full runtime。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5283 --probes full --webgpu-depth-alpha-mode front-top-k --webgpu-depth-bins 16`: passed，headed desktop WebGPU full runtime。
  - `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5284 --probes full --webgpu-depth-alpha-mode front-top-k`: passed，headed desktop WebGPU full runtime。

### RENDER-005T-R: WebGPU runtime camera framing diagnostic

- 状态: done / camera-framing-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 depth-bin sweep 已显示单纯提高 bins 不是 Lego 主因后，把 Spark vs edit 残差中的真实 camera alignment 拆成 runtime-tunable contract，避免继续凭主观观感调参数。
- 已实施:
  - 新增 `runtime-camera-tuning-v1` 和 URL / audit 参数 `webgpu-camera-mode=edit-fixed|spark-frame`；默认仍保持 `edit-fixed`。
  - `spark-frame` 使用和 Spark viewer `frameSplat` 一致的 framing 常量：FOV 58、distance=maxDim*1.7、height multiplier 0.58、target=scene center。
  - `buildWebGpuTileSmoke`、runtime WebGPU smoke、projection bounds、click hit projection 和 storage contract 共用同一 camera tuning。
  - `WebGpuTileViewport` / `PointCloudViewport` DOM 暴露 camera tuning mode、camera mode、projection mode、fov、position、target、distance 和 frame max dimension。
  - `audit-demo` / `audit-webgpu-desktop` / coverage sweep / depth sweep 支持 `--webgpu-camera-mode`，并校验 DOM telemetry 命中 requested camera contract。
- 结论:
  - 默认 Lego headed WebGPU full audit 保持 `edit-fixed`：coverage ratio=`3.784251`、luma/chroma=`0.106079/0.086537`。
  - Lego `spark-frame` 通过 headed WebGPU full audit：coverage ratio=`3.766657`、luma/chroma=`0.102396/0.087290`，coverage/luma 小幅改善但 chroma 略差。
  - Plush semantic `spark-frame` 通过大场景 headed WebGPU full audit：coverage ratio=`4.713926`、luma/chroma=`0.117382/0.016269`；相对历史 baseline coverage 有明显改善，但颜色 delta 不同步改善。
  - 因此 camera alignment 是“自身颜色编辑预览不像 Spark”的 coverage 贡献项之一，但不是完整主因；下一步应转向 SH/view-dependent color、真实 per-pixel sorted alpha，或把 object filter 接入 Spark renderer。
- 验证:
  - `node --check src/webgpuCameraTuning.js`: passed。
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check src/webgpuCapability.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `node --check scripts/audit-webgpu-coverage-sweep.mjs`: passed。
  - `node --check scripts/audit-webgpu-depth-sweep.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5278 --probes full`: passed，headed desktop WebGPU full runtime。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5279 --probes full --webgpu-camera-mode spark-frame`: passed，headed desktop WebGPU full runtime。
  - `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5280 --probes full --webgpu-camera-mode spark-frame`: passed，headed desktop WebGPU full runtime。

### RENDER-005T-Q: WebGPU depth-bin sweep

- 状态: done / depth-bin-sweep-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 基于 T-P 的 runtime depth-bin tuning，把 4 / 8 / 12 / 16 bins 变成一键可复现 browser audit sweep，判断单纯提高 alpha depth bins 是否能解释 “原始颜色（编辑预览）不像 Spark”。
- 已实施:
  - 新增 `npm run audit:webgpu-depth-sweep`，默认使用 headed desktop WebGPU full-runtime audit。
  - Sweep 固定 coverage tuning，逐个传入 `--webgpu-depth-bins`，并校验实际 DOM telemetry 命中 requested bins。
  - 输出每个 depth-bin variant 的 coverage ratio、luma delta、chroma delta、tile reference count 和 normalized Pareto score。
  - 支持 `--assets`、`--bins`、`--output-dir`、`--webgpu-viewport-size`、`--webgpu-footprint-scale`、`--webgpu-covariance-max-anisotropy`。
  - `summary.json` / `summary.md` 报告和 `docs/benchmarks/webgpu-coverage-sweep.md` 已记录用法。
- 结论:
  - Lego 4 / 8 / 12 / 16 bins 全部通过 headed desktop WebGPU full runtime。
  - 8 bins 仍是 best Pareto：score=`1`、luma=`0.106079`、chroma=`0.086537`。
  - 12 bins 的 coverage ratio 仅从 `3.784251` 微降到 `3.784235`，但 chroma 变差到 `0.086874`；16 bins 也没有实质改善。
  - 因此当前 Lego 上继续提高 depth bins 不是高优先级修复方向，下一步应转向 SH / view-dependent color 或真实 camera alignment。
- 验证:
  - `node --check scripts/audit-webgpu-depth-sweep.mjs`: passed。
  - `npm run audit:webgpu-depth-sweep -- --asset nerf-lego-alpha-closure-local --bins 4,8,12,16 --port 5276 --output-dir /tmp/objgauss-webgpu-depth-sweep`: passed；报告写入 `/tmp/objgauss-webgpu-depth-sweep/summary.json` 和 `summary.md`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。

### RENDER-005T-P: WebGPU runtime depth-bin tuning

- 状态: done / depth-bin-tunable
- 类型: 标准 PR / 前端渲染质量
- 目标: 将 WebGPU Tile 的 depth-binned alpha composite 从硬编码 8 bins 推进成 runtime-tunable contract，为后续 sorted-alpha 近似 sweep 提供可复现入口，同时保持默认视觉参数不变。
- 已实施:
  - 新增 `runtime-depth-sort-tuning-v1`，统一 normalize `webgpu-depth-bins`，合法范围为 4-16，默认 8。
  - `buildWebGpuTileSmoke`、CPU pixel reference 和 WebGPU pixel resolve shader 共用同一 depth-bin count。
  - `WebGpuTileViewport` 改为按 runtime smoke 生成 pixel resolve WGSL shader，解决 WGSL array length 必须是编译期常量的问题。
  - `audit-demo` / `audit-webgpu-desktop` / `audit-webgpu-coverage-sweep` 支持 `--webgpu-depth-bins`，并校验 DOM telemetry 命中 requested tuning。
  - DOM renderer contract 暴露 `data-webgpu-pixel-depth-tuning-mode` 和 `data-webgpu-pixel-depth-bin-count`。
- 结论:
  - 12-bin Lego headed WebGPU full audit 通过，日志显示 `pixelDepthSort="depth-binned-alpha-composite-v1":"runtime-depth-sort-tuning-v1":12/0.06:12`，说明 tuned shader 进入真实 runtime。
  - 默认 baseline coverage gate 通过，Lego / Plush 全部默认显示 `bins=8`；默认参数没有被切到 12 或其他值。
  - 一次中间 gate 抓到 `URLSearchParams.get()` 缺失值被 `Number(null)` 解释为 0 并 clamp 到 4 的 bug，已修复为 `null` / `undefined` / `""` 回到默认 8。
- 验证:
  - `node --check src/webgpuDepthTuning.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `node --check scripts/audit-webgpu-coverage-sweep.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5274 --probes full --webgpu-depth-bins 12`: passed，headed desktop WebGPU full runtime。
  - `npm run audit:webgpu-coverage-gate -- --port 5275`: passed，2 scenes x 3 variants headed desktop WebGPU gate。

### RENDER-005T-O: WebGPU coverage report and threshold gate

- 状态: done / report-gated
- 类型: 标准 PR / 前端渲染质量
- 目标: 将 T-N 的多场景 Pareto sweep 固化为可持久化 report / threshold gate，让默认 coverage 参数变更必须先通过多场景 score 和 luma/chroma 门禁。
- 已实施:
  - `audit-webgpu-coverage-sweep` 新增 `--output-dir`，写出 `summary.json` 和 `summary.md`。
  - Sweep summary 记录 mode、assets、variants、score weights、best mean Pareto、scene summaries、variant summaries、raw rows 和 gate result。
  - 新增 `--gate-variant` 与 mean / per-scene pareto、luma、chroma、tile-reference norm 阈值；gate 失败会让命令失败，除非显式 `--allow-failures`。
  - 新增 `npm run audit:webgpu-coverage-gate`，默认用 Lego + Plush 多场景 baseline gate，输出到 `/tmp/objgauss-webgpu-coverage-sweep-gate`。
  - 新增 `docs/benchmarks/webgpu-coverage-sweep.md`，记录 smoke sweep、gate 用法、输出文件和当前本地结论。
- 结论:
  - 当前 baseline gate 9 项通过：mean pareto/luma/chroma 均为 `1`，Lego 和 Plush per-scene pareto/luma/chroma 均为 `1`。
  - Report 再次显示 compact mean Pareto=`0.921829` 但 mean luma norm=`1.271571`，tight mean luma norm=`1.465639`；因此默认参数仍保持 baseline。
  - 后续任何默认参数替换都应先跑 gate，并在 report 中说明 luma/chroma tradeoff。
- 验证:
  - `node --check scripts/audit-webgpu-coverage-sweep.mjs`: passed。
  - `npm run audit:webgpu-coverage-gate -- --port 5270`: passed；2 scenes x 3 variants headed desktop WebGPU full audit，`webgpu_coverage_sweep_gate=passed`，报告写入 `/tmp/objgauss-webgpu-coverage-sweep-gate/summary.json` 和 `summary.md`。

### RENDER-005T-N: WebGPU coverage Pareto multi-scene sweep

- 状态: done / pareto-multi-scene-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 基于 T-M 的 runtime tuning sweep，把 coverage、luma、chroma 和 tile reference cost 合成可比较的多场景表，避免只按单一 coverage ratio 选择牺牲 shading 的参数。
- 已实施:
  - `audit-webgpu-coverage-sweep` 支持 `--assets` 多场景输入，默认仍保持 Lego 单场景以控制日常审计耗时。
  - Sweep 解析 `visualResidual` 与 `tileReferences`，对每个 scene 按 baseline 归一化，并用 coverage / luma / chroma / tile reference cost 的 `0.35 / 0.25 / 0.25 / 0.15` 权重输出 `paretoScore`。
  - 输出每场景 winner、coverage winner、luma / chroma winner、lowest-cost winner，以及跨场景 variant summary 的 mean score / mean normalized metrics。
- 结论:
  - Lego: best Pareto=`baseline:1`，best coverage=`tight:3.346752`，lowest cost=`tight:29641`；tight 降低 coverage / cost，但 luma/chroma 明显恶化。
  - Plush: best Pareto=`compact:0.813147`，best coverage=`tight:6.015767`，lowest cost=`tight:926251`；compact 的 chroma 和 cost 改善拉低综合 score，但 luma 仍比 baseline 差。
  - 跨场景 mean Pareto: baseline=`1`，compact=`0.921829`，tight=`1.072299`。由于 scene winner 不一致且 compact / tight 都牺牲 luma，默认渲染参数暂不切换；下一步需要持久化 report、阈值 gate 和继续拆 alpha / SH / camera 残差。
- 验证:
  - `node --check scripts/audit-webgpu-coverage-sweep.mjs`: passed。
  - `npm run audit:webgpu-coverage-sweep -- --assets nerf-lego-alpha-closure-local,plush-semantic-closure-local --port 5268`: passed；2 scenes x 3 variants headed desktop WebGPU full audit，Lego / Plush 均进入 `WebGPU Tile 编辑`，删除后均回到 RGB 原始色。

### RENDER-005T-M: WebGPU coverage tuning sweep

- 状态: done / coverage-sweep-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 T-L 证明低 alpha halo 只解释一部分 coverage 残差后，把 footprint / covariance 调参从“手改常量”推进成可复用 browser audit sweep。
- 已实施:
  - `buildWebGpuTileSmoke` 新增 `runtime-coverage-tuning-v1`，支持 runtime `footprintScale` 与 `maxAnisotropy`，默认仍为 `2.2 / 4`。
  - `App.jsx` 从 URL 读取 `webgpu-footprint-scale` 与 `webgpu-covariance-max-anisotropy`，并传入 base smoke 与 runtime smoke，保证 gate telemetry 和真实 WebGPU 渲染一致。
  - `audit-demo` 和 `audit-webgpu-desktop` 支持同名参数，并校验浏览器 telemetry 命中 requested tuning。
  - 新增 `npm run audit:webgpu-coverage-sweep`，默认跑 baseline / compact / tight 三组 Lego headed desktop WebGPU full audits，并输出 coverage / luma / chroma。
- 结论:
  - baseline `2.2 / 4`: coverage ratio=`3.784251`，luma/chroma=`0.106079/0.086537`。
  - compact `1.9 / 3`: coverage ratio=`3.536942`，luma/chroma=`0.127400/0.096173`。
  - tight `1.7 / 2.5`: coverage ratio=`3.346752`，luma/chroma=`0.142279/0.102668`。
  - 收紧 footprint / anisotropy 可以稳定降低 coverage ratio，但会牺牲 luma / chroma；因此不应直接把 tight 设为默认，下一步需要 Pareto scoring 或 multi-scene sweep。
- 验证:
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `node --check scripts/audit-webgpu-coverage-sweep.mjs`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`pixelCoverage=footprint-weight-floor-calibrated-v1:runtime-coverage-tuning-v1:0.004:2.2`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-coverage-sweep -- --port 5266`: passed；best coverage variant=`tight:3.346752`，but luma/chroma tradeoff worsened versus baseline。

### RENDER-005T-L: WebGPU alpha presentation edge gate

- 状态: done / alpha-presentation-gated
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 T-K 已证明 depth-binned alpha 能改善 luma / chroma、但 coverage ratio 仍偏大后，把 coverage 残差从 shading 残差里拆出来，先对最终显示阶段的低 alpha halo 做窄门控。
- 已实施:
  - `webgpu-pixel-storage-resolve-v1` fullscreen display shader 新增 `alpha-edge-gated-presentation-v1`，对 `alpha < 0.035` 的最终显示像素直接回落到背景色。
  - 该门控只影响 presentation，不修改 compute resolve 的 straight RGB / alpha buffer，保留后续排序 / SH / debug 的底层数据。
  - `WebGpuTileViewport` 暴露 `data-webgpu-alpha-presentation-mode` 与 `data-webgpu-alpha-presentation-floor`。
  - `audit-demo` 和 tile smoke audit 校验 storage resolve 使用 `alpha-edge-gated-presentation-v1:0.035`。
- 结论:
  - NeRF Lego desktop WebGPU full audit 通过；相对 T-K，coverage ratio 从 `3.856920` 降到 `3.784251`，luma / chroma delta 从 `0.109000 / 0.087808` 小幅降到 `0.106079 / 0.086537`。
  - Plush semantic desktop WebGPU full audit 通过；coverage ratio 从 `6.680406` 降到 `6.448639`，luma delta 从 `0.119591` 降到 `0.112667`，chroma delta 从 `0.007786` 到 `0.010651`。
  - 这证明低 alpha halo 确实贡献了一部分 coverage 残差，但幅度有限；下一步 coverage 线应做 footprint / covariance / threshold sweep，不能把 T-L 当作覆盖问题完成。
- 验证:
  - `node --check src/webgpuTileResolveShader.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`resolveSource=webgpu-pixel-storage-resolve-v1:bilinear-storage:alpha-edge-gated-presentation-v1:0.035`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5264/ --no-server --probes full`: passed；coverage ratio=`3.784251`、luma/chroma=`0.106079/0.086537`。
  - `node scripts/audit-webgpu-desktop.mjs --asset plush-semantic-closure-local --url http://127.0.0.1:5264/ --no-server --probes full`: passed；coverage ratio=`6.448639`、luma/chroma=`0.112667/0.010651`。

### RENDER-005T-K: WebGPU depth-binned alpha compositing

- 状态: done / depth-binned-alpha-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 T-J 已证明单纯缩小 footprint 只能降低过覆盖、不能改善 luma / chroma 后，把 WebGPU Tile pixel resolve 从 weighted average / nearest-depth gate 推进到更接近最终 C 架构的 front-to-back alpha compositing 近似。
- 已实施:
  - `WEBGPU_PIXEL_DEPTH_SORT_MODE` 升级为 `depth-binned-alpha-composite-v1`。
  - Pixel resolve 不再做两遍 nearest-depth gate + weighted average；改为每像素固定 8 个 depth bins，先按 Gaussian depth 累积 bin 内颜色 / 权重，再按前到后进行 alpha compositing。
  - WebGPU WGSL shader 和 CPU smoke reference 同步实现同一合成逻辑，输出仍保持 straight RGB + alpha，兼容现有 fullscreen storage resolve。
  - `WebGpuTileViewport`、`PointCloudViewport`、renderer contract、browser audit 和 tile smoke audit 均暴露 / 校验 `pixelDepthSort=depth-binned-alpha-composite-v1:...:8`。
- 结论:
  - NeRF Lego desktop WebGPU full audit 通过；相对 T-J，luma / chroma delta 从 `0.207570 / 0.133965` 降到 `0.109000 / 0.087808`，说明 alpha compositing 路线确实在修复颜色 / 亮度合成残差。
  - 同一 audit 的 coverage ratio 从 `3.271989` 回升到 `3.856920`，说明 coverage 和 shading 是两个独立问题；T-K 不应被解读为 footprint 完成。
  - Plush semantic 大场景 281498 Gaussians 也通过 desktop WebGPU full audit；luma / chroma delta 为 `0.119591 / 0.007786`，device / queue active，删除后仍回到 `281498/0/0/0` RGB 原色。
  - 下一步应分线推进：coverage 继续校准 footprint / alpha threshold，shading 继续评估 SH 或 Spark edit handoff。
- 验证:
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check src/webgpuTileComputeShader.js`: passed。
  - `node --check src/webgpuCapability.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`pixelDepthSort=depth-binned-alpha-composite-v1:12/0.06:8`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5263/ --no-server --probes full`: passed；coverage ratio=`3.856920`、luma/chroma=`0.109000/0.087808`。
  - `node scripts/audit-webgpu-desktop.mjs --asset plush-semantic-closure-local --url http://127.0.0.1:5263/ --no-server --probes full`: passed；coverage ratio=`6.680406`、luma/chroma=`0.119591/0.007786`。

### RENDER-005T-J: WebGPU footprint coverage calibration

- 状态: done / footprint-coverage-calibrated
- 类型: 标准 PR / 前端渲染质量
- 目标: 回应“原始颜色 / 自身颜色有颗粒感、没有真实高斯感”的问题，在 T-I 已证明 RGB source 未丢失后，先校准 WebGPU Tile 编辑预览的 alpha footprint / coverage，并把该校准变成可审计 contract。
- 已实施:
  - WebGPU Tile pixel resolve 新增 `footprint-weight-floor-calibrated-v1` coverage contract。
  - Pixel candidate / accumulation 统一使用 `PIXEL_COVERAGE_WEIGHT_FLOOR=0.004`，和 Three Gaussian OIT fallback 的弱尾部剔除策略对齐。
  - WebGPU 编辑预览 footprint scale 从 `4.8` 校准到 `2.2`，降低过大的半透明覆盖和烟雾化残影。
  - `WebGpuTileViewport`、`PointCloudViewport`、renderer contract、`audit-demo` 和 tile smoke audit 均暴露 / 校验 `pixelCoverage=mode:weightFloor:footprintScale`。
- 结论:
  - NeRF Lego WebGPU full audit 的 Spark/edit coverage ratio 从 T-I 的 `4.469421` 降到 `3.271989`，说明原始色编辑预览的过覆盖已明显收敛。
  - 单独尝试 weight floor 时 ratio 为 `4.547073`，未解决问题；真正有效的是 footprint scale 校准。
  - 但 luma / chroma delta 未同步改善，NeRF Lego 校准后为 `0.207570 / 0.133965`；因此“自身颜色不像真实高斯”的剩余主因不再是颜色源或单纯 footprint，而是编辑 renderer 仍不是 Spark 的排序 alpha / SH / 真实 `.splat` 合成路径。
  - Plush semantic 大场景也通过 desktop WebGPU full audit，删除预览后仍回到 `281498/0/0/0` RGB 原色；但 Spark/edit coverage ratio 仍为 `6.472669`，后续不应继续只靠缩 footprint 硬调。
- 验证:
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check src/webgpuTileComputeShader.js`: passed。
  - `node --check src/webgpuCapability.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-tile-smoke.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`pixelCoverage=footprint-weight-floor-calibrated-v1:0.004:2.2`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5262/ --no-server --probes full`: passed；coverage ratio=`3.271989`。
  - `node scripts/audit-webgpu-desktop.mjs --asset plush-semantic-closure-local --url http://127.0.0.1:5262/ --no-server --probes full`: passed；coverage ratio=`6.472669`。

### RENDER-005T-I: Spark vs edit visual residual audit

- 状态: done / visual-residual-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 在确认 RGB source 未丢失、并加入 front-depth gate 后，把“为什么编辑预览仍不像 Spark 真实 `.splat`”从主观观感变成可审计残差，支撑后续 SH / sorting / footprint 校准。
- 已实施:
  - `audit-demo` 新增 `spark-edit-visual-residual-v1`，对 Spark canvas 和“对象编辑 / 原始颜色”canvas 做无依赖 PNG screenshot 解析，输出 coverage、luma、chroma、checksum 和 Spark-vs-edit residual。
  - Browser audit 流程调整为：加载 Spark 真实查看 -> 采集 Spark visual stats -> 进入对象编辑原始色 -> 采集 edit visual stats -> 再切对象色继续原有选择 / 隔离 / 删除验收。
  - `PointCloudViewport` fallback 也暴露 color-source telemetry，避免非 WebGPU 环境下 after-delete source-color audit 误读为 0。
  - `pixel-compute-only` probe 的 expected pixel source 改为跟随 `WEBGPU_PIXEL_RESOLVE_SOURCE`，避免 T-H 后旧字符串导致诊断 probe 误报。
- 结论:
  - NeRF Lego fallback audit 显示 Spark coverage=0.121827、原始色 Gaussian OIT 编辑 coverage=0.218147，coverage ratio=1.790629，luma delta=0.040493，chroma delta=0.054075。
  - NeRF Lego headed WebGPU full audit 显示 Spark coverage=0.121799、原始色 WebGPU Tile 编辑 coverage=0.544371，coverage ratio=4.469421，luma delta=0.152456，chroma delta=0.111168。
  - 因此当前最大可观测残差更像 WebGPU Tile 原始色编辑预览 coverage / alpha footprint 过铺开，而不是单纯“自身颜色没回来”；下一步应优先校准 coverage/alpha 或继续拆 SH 残差。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `node scripts/audit-demo.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5260/ --no-server`: passed；fallback visual residual ratio=1.790629。
  - `node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5260/ --no-server --probes full`: passed；WebGPU full visual residual ratio=4.469421。
  - `node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5260/ --no-server --probes pixel-compute-only --allow-device-lost-probes`: passed；pixel source=`webgpu-compute-front-depth-pixel-accumulation-v1`。

### RENDER-005T-H: WebGPU front-depth gated pixel resolve

- 状态: done / front-depth-gated-oit-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 回应“自身颜色有颗粒感、不像高斯”的后续合成问题，在已确认 RGB source 未丢失后，把 WebGPU Tile per-pixel resolve 从纯 front-weighted weighted OIT 推进到每像素最近深度门控，降低前后层 Gaussian 混色。
- 已实施:
  - `webgpu-compute-pixel-accumulation-v1` 升级为 `webgpu-compute-front-depth-pixel-accumulation-v1`。
  - Pixel resolve shader 对每个像素先扫描 tile Gaussian entries，找出 visible / in-kernel / non-trivial alpha contributor 的 nearest depth，再在第二遍 weighted accumulation 中叠加 `frontDepthGate`。
  - CPU smoke reference 与 WGSL contract 同步，新增 `front-depth-gated-oit-v1`、gate strength / floor telemetry。
  - `WebGpuTileViewport`、renderer contract 和 browser audit 暴露并校验 pixel depth gate contract。
- 结论:
  - 这一步是比上一版 `front-weighted-oit-v1` 更强的 GPU-friendly 前层遮挡近似，目标是减少编辑预览的后层颜色透混和颗粒感。
  - 它仍不是完整 per-pixel sorted alpha，不是 Spark 真实 `.splat` renderer，也没有解决 view-dependent SH 视觉差距；下一步应进入 Spark visual residual / SH 或真正排序设计。
- 验证:
  - `node --check src/webgpuTileComputeShader.js`: passed。
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check src/webgpuCapability.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`pixelDepthSort=front-depth-gated-oit-v1:12/0.06`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5258 --probes full`: passed；`pixel="dispatched":"webgpu-compute-front-depth-pixel-accumulation-v1":2304`、`colorAfterDelete=281498/0/0/0`。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5259 --probes full`: passed；`pixel="dispatched":"webgpu-compute-front-depth-pixel-accumulation-v1":3968`、`colorAfterDelete=5696/0/0/0`。

### RENDER-005T-G: WebGPU source-color fidelity audit

- 状态: done / source-color-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 回应“自身颜色为啥还是不像高斯”的问题，先把 WebGPU Tile 编辑预览的颜色来源做成可审计事实，区分 RGB/SH source 丢失、fallback 默认色、对象调试色和后续合成/排序问题。
- 已实施:
  - PLY parser 在生成 `point.color` 时同步保留 `colorSource`：`rgb`、`sh-dc` 或 `fallback`；内置 sample scene 标记为 `rgb`。
  - `webgpuTileSmoke` 增加 `source-color-fidelity-v1` telemetry，统计 `rgb/sh-dc/fallback/object-palette` Gaussian 数量和 opacity mean。
  - `WebGpuTileViewport` / renderer contract / browser audit 暴露并验证颜色来源；audit 起点允许对象色模式全量走 `object-palette`，但删除预览切回 `原始颜色（编辑预览）` 后必须命中 RGB/SH source 且无 fallback/object 色。
- 结论:
  - Plush semantic audit 起点是对象色：`object=281498`；删除预览后切回原始色：`rgb=281498`、`shDc=0`、`fallback=0`、`object=0`。
  - NeRF Lego proxy audit 起点是对象色：`object=5696`；删除预览后切回原始色：`rgb=5696`、`shDc=0`、`fallback=0`、`object=0`。
  - 因此当前“原始颜色不像 Spark”的主要剩余差距不再是 RGB/SH source 丢失，而更可能来自 alpha/depth compositing、view-dependent SH 项和 Spark renderer 的排序/抗锯齿差异。
- 验证:
  - `node --check src/ply.js`: passed。
  - `node --check src/sampleScene.js`: passed。
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check src/webgpuCapability.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`colorFidelity=source-color-fidelity-v1:5800/0/0/0:0.955`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5256 --probes full`: passed；`colorAfterDelete=281498/0/0/0`。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5257 --probes full`: passed；`colorAfterDelete=5696/0/0/0`。

### RENDER-005T-F: WebGPU adaptive runtime quality

- 状态: done / adaptive-quality-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 回应“原始颜色（编辑预览）颗粒感强”的问题，把 WebGPU Tile full runtime 从固定 256px 内部输出推进到可审计的自适应内部分辨率，同时保留大场景资源安全。
- 已实施:
  - `App.jsx` 新增 WebGPU runtime quality policy：小场景走 `adaptive-high-512`，中等 / 大场景走 `adaptive-medium-384`，超大场景保守走 `adaptive-safe-320`；显式 `--webgpu-viewport-size` 和 `tiny-pixel-output` 诊断 probe 不变。
  - 默认 full runtime 继续匹配 viewer display aspect，但 aspect mode 从固定面积 `display-aspect-area` 升级为 `display-aspect-adaptive`，并按质量档暴露 `viewportQuality` / `pixelBudget`。
  - `WebGpuTileViewport` 暴露 `data-webgpu-viewport-quality` 与 `data-webgpu-viewport-pixel-budget`，`audit-demo` 将默认 WebGPU full runtime 视觉验收地板从 256px 提升到 320px，并要求 adaptive quality contract。
- 结论:
  - NeRF Lego proxy 5696 Gaussian 在 desktop WebGPU full runtime 下使用 `496x512` 输出，quality=`adaptive-high-512`，first frame、queue、device 和对象选择 / 隔离 / 删除预览通过。
  - Plush semantic 281498 Gaussian 在 desktop WebGPU full runtime 下使用 `384x384` 输出，quality=`adaptive-medium-384`，first frame、queue、device 和对象选择 / 隔离 / 删除预览通过。
  - 这一步降低的是低内部分辨率放大造成的颗粒 / 格子感；仍不是 Spark 真实 `.splat` 的 SH 颜色、严格 alpha 排序或完整高质量 3DGS renderer。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5254 --probes full`: passed；`webgpuViewport=496x512:253952:"display-aspect-adaptive":"adaptive-high-512"`、device active、queue done。
  - `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5255 --probes full`: passed；`webgpuViewport=384x384:147456:"display-aspect-adaptive":"adaptive-medium-384"`、device active、queue done。

### RENDER-005T-E: WebGPU camera-Jacobian screen covariance

- 状态: done / screen-covariance-calibrated
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 front-weighted OIT 后，让 WebGPU Tile 编辑预览不再只用二维 scale + yaw 近似 Gaussian footprint，而是消费真实 3DGS 的三轴 scale / quaternion covariance 并投影成屏幕椭圆。
- 已实施:
  - PLY parser 保留 `scale3` 和 normalized `rotationQuaternion`，同时继续提供原有二维 `scale` / `rotation` 给 Three Gaussian OIT fallback。
  - `webgpuTileSmoke` 使用 edit-camera projection Jacobian 将 3D covariance 投影为 screen-space 2D covariance，再分解为 shader 已消费的 `sigmaMajor / sigmaMinor / rotation`。
  - 对缺少 quaternion 的 proxy 数据保留 legacy fallback；对真实 covariance path 增加 4:1 anisotropy clamp，避免低分辨率 WebGPU Tile 预览出现过长针状 streak。
  - renderer contract / DOM / browser audit 暴露 `screenCovariance=camera-jacobian-covariance-v1:full/fallback/clamped:maxAnisotropy:sigmaMean`。
- 结论:
  - Node smoke 的内置 sample 走 full covariance path：`screenCovariance=camera-jacobian-covariance-v1:5800/0/0:4`。
  - Plush semantic 大场景全量 281498 个 Gaussian 走 full covariance path，127733 个 Gaussian 被 4:1 anisotropy clamp 校准，desktop WebGPU full runtime 通过。
  - NeRF Lego proxy 因缺少 quaternion 走 fallback path，desktop WebGPU full runtime 仍通过，说明缺字段样例不会回归。
  - 这一步解决的是 footprint covariance 投影和极端各向异性校准；仍不是完整 per-pixel depth sort，也不等于 Spark 真实 `.splat` 渲染。
- 验证:
  - `node --check src/ply.js`: passed。
  - `node --check src/sampleScene.js`: passed。
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`screenCovariance=camera-jacobian-covariance-v1:5800/0/0:4`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5252 --probes full`: passed；281498 full covariance Gaussians、127733 clamped、device active、queue done。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5253 --probes full`: passed；0 full / 5696 fallback covariance Gaussians、device active、queue done。

### RENDER-005T-D: WebGPU front-weighted OIT depth contract

- 状态: done / front-weighted-oit-depth
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 edit-camera perspective projection 后，减少 WebGPU Tile 编辑预览中前后层 Gaussian 被纯 weighted OIT 混成一团的视觉问题。
- 已实施:
  - `webgpuTileSmoke` 记录 edit-camera projection depth range，并在 CPU tile resolve / per-pixel reference 中加入 `front-weighted-oit-v1` 深度权重。
  - WebGPU accumulation shader 与 per-pixel resolve shader 使用同一 depth weight：近处 Gaussian 保持高权重，远处 Gaussian 按归一化相机深度指数衰减但保留 floor，避免远处对象被直接抹掉。
  - renderer contract / DOM / browser audit 暴露 `depthWeight=front-weighted-oit-v1:min/max/span`，Node smoke 检查 uniform meta 中的 `depthMin/depthSpan` 和 WGSL `frontWeight` contract。
- 结论:
  - NeRF Lego proxy 与 Plush semantic 大场景均在 headed desktop Chrome/WebGPU full runtime 下通过，说明该 depth contract 在小样例和 281k Gaussian 场景上都可编译、可提交、可完成对象选择 / 隔离 / 删除预览。
  - 这一步是 alpha-order 的 GPU-friendly 近似，不是完整 per-pixel depth sort，也不是最终 3D covariance 投影；下一步仍应处理 screen-space covariance / splat scale calibration。
- 验证:
  - `node --check src/webgpuTileSmoke.js`: passed。
  - `node --check src/webgpuTileComputeShader.js`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`depthWeight=front-weighted-oit-v1`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5246 --probes full`: passed；`depthWeight="front-weighted-oit-v1"`、device active、queue done、object interactions passed。
  - `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5247 --probes full`: passed；281498 Gaussians、`depthWeight="front-weighted-oit-v1"`、compact tile list、device active、queue done。

### RENDER-005T-C: WebGPU edit-camera perspective projection

- 状态: done / edit-camera-projection
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 aspect-fit runtime viewport 后，减少 WebGPU Tile 编辑预览因为简化 x/z 正交投影导致的“自身颜色像颗粒/不像高斯”视觉差距。
- 已实施:
  - `webgpuTileSmoke` 改为 CPU 端按固定编辑相机投影 Gaussian，打包 screen-space center / depth / sigma，WGSL accumulation 和 pixel resolve 直接消费 screen-space Gaussian。
  - WebGPU canvas 点击命中复用同一套 edit-camera projection，并按内部 viewport 映射到实际 canvas 尺寸，避免渲染画面和选中区域继续按旧 x/z 正交逻辑错位。
  - 新增 `projectionMode=edit-perspective-camera-v1` 与 `projectionCameraFovDegrees=52` runtime telemetry，前端 DOM 和 audit 会检查 WebGPU full path 使用该投影 contract。
  - Node smoke contract 更新为验证 GPU shader 不再按旧 world bounds 做二次投影，pixel resolve 继续使用 bilinear storage display。
- 结论:
  - NeRF Lego proxy 的 headed desktop WebGPU full audit 通过，选择、隔离、删除预览仍能更新 object-state buffer。
  - 这一步解释并缓解了“原始颜色（编辑预览）颗粒/不像高斯”的一部分原因：WebGPU 编辑预览以前不是 Spark 真实 `.splat` 重渲染，而是低分辨率 tile renderer + 简化投影。
  - 剩余视觉差距仍来自 depth / alpha-order、screen-space covariance 和最终 tile renderer 质量，不应宣称已经等同 Spark 真实查看。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`projection=edit-perspective-camera-v1:52`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5245 --probes full`: passed；`projection="edit-perspective-camera-v1":52`、device active、queue done、object interactions passed。

### RENDER-005T-B: WebGPU aspect-fit runtime viewport

- 状态: done / aspect-fit-viewport
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 bilinear storage resolve 后，减少 WebGPU Tile 编辑因为固定方形 viewport 和 x/z 独立拉伸导致的比例偏差、撑满画布和过度正交感。
- 已实施:
  - 默认 WebGPU full runtime 根据实际 viewer 显示尺寸计算 area-preserving internal viewport；显式 `--webgpu-viewport-size` 仍保留 square override，`tiny-pixel-output` 仍保留 32px。
  - `webgpuTileSmoke` 的 projection bounds 改为 `aspect-fit-padding`：按 viewport aspect 扩展短轴，并加入 8% 单边留白。
  - `WebGpuTileViewport` 暴露 display size、viewport aspect mode 和 bounds-fit telemetry；`audit-demo` 会检查 `display-aspect-area`、`aspect-fit-padding` 和 bounds world aspect / viewport aspect 一致。
- 结论:
  - 这一步把 WebGPU 编辑视图从“方形低分辨率正交 blob”推进到“按显示区域比例 fit 的平滑正交 blob”，减少对象贴边和非等比拉伸。
  - 它仍不是 Spark 真实相机 / depth-order / 3D covariance 的对象级重渲染；下一步应处理真实 camera transform、depth / alpha order 和 screen-space covariance。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `git diff --check`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；Node smoke 覆盖 1:1 和 2:1 viewport bounds fit。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5244 --probes full`: passed；`webgpuViewport=256x256:65536:"display-aspect-area"`、`display=784x812`、`boundsFit="aspect-fit-padding":1/1`、device active、queue done。

### RENDER-005T-A: WebGPU pixel-storage bilinear resolve

- 状态: done / storage-resolve-smoothed
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 RENDER-005S 已把 full runtime 内部输出提升到 256px 后，降低该内部图放大到主画布时的最近邻颗粒 / 格子感。
- 已实施:
  - `webgpu-pixel-storage-resolve-v1` fullscreen fragment shader 从 `floor()` 最近邻读取 `pixelResolvedRgba` 改为 bilinear storage sampling。
  - 新增 `WEBGPU_TILE_RESOLVE_FILTER="bilinear-storage"`，`WebGpuTileViewport` 通过 `data-webgpu-resolve-filter` 暴露当前 display filter。
  - `audit-demo` 和 `audit-webgpu-tile-smoke` 会输出 / 检查 storage full path 的 `resolveSource=webgpu-pixel-storage-resolve-v1:bilinear-storage`。
- 结论:
  - NeRF Lego proxy 的 headed desktop WebGPU full audit 通过，说明 bilinear storage resolve 在真实浏览器 WebGPU runtime 中可编译、可提交、可完成对象选择 / 隔离 / 删除预览。
  - 这一步解决的是放大颗粒感，不等于 Spark 真实 3DGS 对象级重渲染；截图显示画面更平滑，但仍存在相机 / 投影 / alpha-order 视觉差距。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `git diff --check`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed；`resolveSource=webgpu-pixel-storage-resolve-v1:bilinear-storage`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5243 --probes full`: passed；`webgpuViewport=256x256:65536`、firstFrame=65536、resolve filter=`bilinear-storage`、device active、queue done。

### RENDER-005S: WebGPU runtime visual fidelity audit

- 状态: done / viewport-resolution-audited
- 类型: 标准 PR / 前端渲染质量
- 目标: 在 desktop WebGPU runtime 已通过大场景后，提升和审计 WebGPU tile 编辑视图的显示分辨率、非空像素覆盖和用户可感知视觉质量。
- 已实施:
  - WebGPU full runtime 的默认内部 pixel output 从 `128x128` 提升到 `256x256`，`tiny-pixel-output` 诊断 probe 继续保持 `32x32`。
  - 新增 `?webgpu-viewport-size=<n>`、`OBJGAUSS_WEBGPU_VIEWPORT_SIZE` 和 `--webgpu-viewport-size` audit 参数，用于对比不同内部输出分辨率。
  - `WebGpuTileViewport` 暴露 `data-webgpu-viewport-width`、`data-webgpu-viewport-height` 和 `data-webgpu-pixel-count`，`audit-demo` / `audit-webgpu-desktop` 会输出并检查 full runtime 默认不低于 256。
- 结论:
  - 256px full runtime 在 NeRF Lego proxy 和 Plush semantic 大场景上通过 headed desktop Chrome/WebGPU audit，且对象选择、隔离、删除预览仍更新 object-state buffer。
  - 当前“原始颜色（编辑预览）”仍不是 Spark 真实 splat 的对象级重渲染；剩余视觉差距主要来自正交简化投影、近似 covariance / blending 和未匹配真实查看相机，而不是 Object Field 颜色本身。
- 验证:
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `git diff --check`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5241 --probes full`: passed；`webgpuViewport=256x256:65536`、pixel workgroups=1024、firstFrame=65536。
  - `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5242 --probes full`: passed；281498 Gaussians、`webgpuViewport=256x256:65536`、tileReferences=1458084、maxTileOccupancy=21717、visibleAfterIsolate=98770、visibleAfterDelete=182728。

### RENDER-005R: Large-scene desktop WebGPU runtime audit

- 状态: done / large-scene-runtime-passed
- 类型: 标准 PR / 前端渲染验收
- 目标: 在 WebGPU-capable 桌面 Chrome 中把强制 WebGPU runtime audit 从 NeRF Lego proxy 扩展到 Plush / Splatfacto 级 100k+ Gaussian 场景，验证 C 路线在大场景上的对象编辑稳定性。
- 结论:
  - Plush semantic closure、Plush v1 closure 和本机 safe-2000 Splatfacto public sample 均在 headed desktop Chrome/WebGPU 下通过 `clear-only`、`texture-display-only` 和 `full` probes。
  - 三个大场景均进入 `data-renderer="webgpu-tile"`，`targetGate="pass"`，`storageLimit="pass"`，`tileCapacity="compact-offset-list":"ok"`。
  - `full` probe 均 dispatch accumulation / resolve / pixel stages，queue done，device active，并完成对象选择、隔离和删除预览。
- 验收底线:
  - `data-renderer="webgpu-tile"`、`targetGate="pass"`、`storageLimit="pass"`。
  - `full` probe 中 accumulation / resolve / pixel stages 均 dispatched，queue done，device active。
  - 对象选择、隔离、删除预览仍能更新 object-state checksum 和 visible counts。
- 验证:
  - `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5236`: passed；281498 packed/binned Gaussians，tileReferences=724881，maxTileOccupancy=38792，storage max buffer=`positionRadius:4503968`，visibleAfterIsolate=98770，visibleAfterDelete=182728。
  - `npm run audit:webgpu-desktop -- --asset plush-v1-closure-local --port 5237`: passed；281498 packed/binned Gaussians，tileReferences=724881，maxTileOccupancy=38792，storage max buffer=`positionRadius:4503968`，visibleAfterIsolate=85041，visibleAfterDelete=196457。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5238`: passed；255794 packed/binned Gaussians，tileReferences=436816，maxTileOccupancy=37934，storage max buffer=`positionRadius:4092704`，visibleAfterIsolate=126686，visibleAfterDelete=129108。

### RENDER-005Q: Desktop WebGPU presentation audit

- 状态: done / desktop-runtime-passed
- 类型: 标准 PR / 前端渲染验收
- 目标: 在 WebGPU-capable 桌面 Chrome 中重跑 `clear-only`、`texture-display-only` 和 `full` runtime audit，判断当前 failure 是否只属于 headless unsafe WebGPU presentation backend。
- 已实施:
  - `scripts/audit-demo.mjs` 新增 `--headed`、`--browser-channel`、`--executable-path` 和 `--slow-mo`，使同一套 WebGPU runtime audit 可在系统 Chrome / headed desktop session 中运行。
  - 新增 `scripts/audit-webgpu-desktop.mjs` 和 `npm run audit:webgpu-desktop`，默认启动 built `dist/` 的 `vite preview`，依次运行 `clear-only`、`texture-display-only`、`full` 三个 probe，并输出 suite classification。
  - 新增 `docs/rendering/webgpu-desktop-audit.md`，记录桌面 audit 命令、浏览器选择参数、headless diagnostic 参数和结果解释。
- 结论:
  - 本机 headless diagnostic 运行 `npm run audit:webgpu-desktop -- --headless --allow-failures --port 5232` 可完整跑完三项，并分类为 `desktop-webgpu-presentation-backend-loss`。
  - headed desktop Chrome/WebGPU 运行 `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5235` 严格通过，classification=`desktop-webgpu-runtime-passed`。
  - 因此当前 headless unsafe WebGPU failure 正式归类为 presentation backend limitation；不再视为 ObjGauss WebGPU compute/storage/display shader 的 runtime blocker。
- 验证:
  - `git diff --check`: passed。
  - `node --check scripts/audit-demo.mjs`: passed。
  - `node --check scripts/audit-webgpu-desktop.mjs`: passed。
  - `npm run audit:webgpu-desktop -- --headless --allow-failures --port 5232`: expected failed probes collected；`clear-only`、`texture-display-only`、`full` 均分类为 `webgpu-presentation-or-backend-loss`，suite classification=`desktop-webgpu-presentation-backend-loss`。
  - preview server start path fix 后，`npm run audit:webgpu-desktop -- --headless --allow-failures --probes clear-only --port 5233` 复跑通过分类收集。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5234 --allow-failures`: passed；`clear-only` / `texture-display-only` queue done、device active；`full` accumulation / compute / pixel dispatched，queue done，device active，对象选择 / 隔离 / 删除通过。
  - `npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5235`: strict passed；classification=`desktop-webgpu-runtime-passed`，`allowWebGpuDeviceLost=false`。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。

### RENDER-005K: WebGPU runtime audit entry

- 状态: audit-entry-landed / browser-webgpu-pending
- 类型: 标准 PR / 前端渲染验收
- 目标: 固化一条强制 WebGPU runtime route 的浏览器 audit 命令，用于在 WebGPU-capable 浏览器中证明 Plush / Lego first frame 真进入 `webgpu-tile`。
- 已实施:
  - `scripts/audit-demo.mjs` 新增 `--require-webgpu`，显式要求 `webgpuStatus=available`、`data-renderer="webgpu-tile"`、target gate pass、无 fallback reason。
  - `scripts/audit-demo.mjs` 新增 `--webgpu-flags none|unsafe|vulkan`，默认不影响常规 fallback audit；runtime audit 可用 `unsafe` 或 `vulkan` flags 启动 Chromium。
  - `package.json` 新增 `npm run audit:webgpu-runtime`，默认使用 `--require-webgpu --webgpu-flags unsafe`。
- 当前阻塞:
  - 常规 headless Chrome 仍返回 `webgpu-adapter-unavailable`。
  - 使用 `--webgpu-flags unsafe` 时，NeRF Lego proxy 可进入 WebGPU route，accumulation / resolve / pixel compute 均 dispatch，first-frame submission 已记录；但 device 随后变为 `webgpu-device-lost-destroyed`，这不是最终 runtime pass 证据。
- 待验证:
  - 在 WebGPU-capable 浏览器 / 桌面 Chrome 环境中运行 Plush / Lego first-frame runtime audit。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5218/ --no-server`: passed，assets=3，常规 fallback audit 未回归。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5218/ --no-server`: passed，修改后的 audit ordering 在 fallback 单样例上通过。
  - `npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5218/ --no-server`: expected failed；当前 headless Chrome + unsafe WebGPU flags 下 compute stages dispatched，但 first frame 为 `webgpu-device-lost-destroyed`。

### RENDER-005A: WebGPU device-backed renderer skeleton

- 状态: implementation-landed / browser-webgpu-pending
- 类型: 标准 PR / 前端渲染架构
- 目标: 在 zero-overflow scene 上进入真实 WebGPU device-backed renderer，并渲染非空 first frame。
- 已实施:
  - 新增 `src/WebGpuTileViewport.jsx`，在 WebGPU route 中创建 adapter/device/context/pipeline，把 `tileResolvedRgba` 上传为 WebGPU texture，并用 fullscreen triangle 画出第一帧。
  - WebGPU route 只在 `webGpuStatus=available` 且 `tileCapacityGate=pass` 时启用；Plush overflow 场景继续 fallback 到 `Gaussian OIT 编辑`。
  - `editRendererContract` 现在在 zero-overflow + WebGPU available 时切到 `rendererId="webgpu-tile"` 和 `objectFilter="gpu-object-state-buffer"`。
  - `audit-demo` 已支持 WebGPU Tile / Gaussian OIT 双路径；WebGPU 路径会检查 `data-webgpu-first-frame-status="rendered"`、positive first-frame pixels 和 checksum。
  - `audit-webgpu-tile-smoke` 已用模拟 `webgpu-device-ready` capability 验证 zero-overflow contract 切到 `webgpu-tile`，overflow contract 仍 blocked 于 `tile-overflow`。
- 当前阻塞:
  - 当前 headless Chrome 常规 audit 返回 `webgpu-adapter-unavailable`，所以真实 WebGPU first frame 没有在本环境执行。
  - 带 `--enable-unsafe-webgpu` / Vulkan flags 的 Playwright probe 在当前容器中 SIGTRAP 退出，不能作为 runtime 证据。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，包含 simulated available + roomy no-overflow contract。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --url http://127.0.0.1:5204/`: passed，assets=3；当前仍全量 fallback，三者 `targetGate="blocked":"webgpu-capability"`。
- 完成 commit: runtime audit pending；implementation commit `12f5fc8`.

### RENDER-005P: WebGPU texture-backed display probes

- 状态: done / canvas-presentation-backend-loss-isolated
- 类型: 标准 PR / 前端渲染诊断
- 目标: 将 fullscreen pixel-storage display pass 替换/对照为 texture-backed display probes，区分 storage-buffer fragment read、buffer-to-texture copy、sampled-texture presentation 和 canvas render pass/presentation。
- 范围外:
  - 不声明 full WebGPU tile renderer 已稳定通过。
  - 不改变默认 `npm run audit:webgpu-runtime` 的严格失败语义。
  - 不把 headless unsafe WebGPU probe 结果等同于 WebGPU-capable 桌面 Chrome 结果。
- 实施:
  - 新增 `src/webgpuTextureResolveShader.js`，提供 sampled texture resolve shader 和 float texture load resolve shader。
  - 新增 `texture-display-only` probe：CPU 生成 `rgba8unorm` sampled texture，直接 fullscreen display，不跑 compute。
  - 新增 `texture-copy-display` probe：pixel compute 写 `pixelResolvedRgba`，再 `copyBufferToTexture` 到 `rgba32float` texture 并 fullscreen display。
  - 新增 `clear-only` probe：只提交 canvas render pass clear，不绑定 pipeline、不 draw，用来隔离 presentation backend。
  - `scripts/audit-demo.mjs` 和 Node smoke audit 覆盖新增 probe source、stage dispatch/skipped contract 和 shader contract。
- 诊断结论:
  - `texture-display-only`: all compute stages skipped，但 sampled texture display 后 device lost、queue failed。
  - `texture-copy-display`: pixel compute dispatched，但 texture copy/display 后 device lost、queue failed。
  - `clear-only`: all compute stages skipped、无 draw，canvas clear pass 仍 device lost、queue failed。
  - 当前 blocker 已收敛到 headless unsafe WebGPU canvas render pass / presentation backend loss，不是 pixel compute、storage write、storage-buffer fragment read、sampled texture display 或 buffer-to-texture copy 本身。
- 验证:
  - `git diff --check`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5225/ --no-server`: passed，Browser plugin absent，使用 Playwright fallback + built `dist/` static server。
  - `npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5225/ --no-server --webgpu-probe texture-display-only`: passed with allowed device lost，`resolveSource=webgpu-sampled-texture-resolve-v1`、all compute stages skipped。
  - `npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5225/ --no-server --webgpu-probe texture-copy-display`: passed with allowed device lost，`resolveSource=webgpu-buffer-copy-texture-resolve-v1`、`pixelWorkgroups=256`。
  - `npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5225/ --no-server --webgpu-probe clear-only`: passed with allowed device lost，`resolveSource=webgpu-clear-pass-v1`、all compute stages skipped。
  - clear-only no-draw fix 后，`npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5226/ --no-server --webgpu-probe clear-only` 复跑通过，`accumulation/compute/pixel=skipped`。
  - `npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5225/ --no-server`: expected failed，仍报告 `probe=full` 和 `A valid external Instance reference no longer exists`。

### RENDER-005O: WebGPU pixel output sub-probes

- 状态: done / display-pass-backend-loss-isolated
- 类型: 标准 PR / 前端渲染诊断
- 目标: 将 `pixel-output-only` 继续拆成 pixel-compute-only / display-only / tiny-pixel-output probes，区分 pixel accumulation shader、pixel storage write、fullscreen display pass 和 workload size。
- 范围外:
  - 不声明 full WebGPU tile renderer 已稳定通过。
  - 不改变默认 `npm run audit:webgpu-runtime` 的严格失败语义。
  - 不把 headless unsafe WebGPU probe 结果等同于桌面 Chrome / production WebGPU 结果。
- 实施:
  - `src/webgpuRuntimeProbe.js` 新增 `pixel-compute-only`、`display-only`、`tiny-pixel-output` 和 32px tiny viewport constant。
  - `App` 在 `tiny-pixel-output` URL probe 下生成 32px WebGPU runtime smoke，用于降低 pixel output workload。
  - `WebGpuTileViewport` 可独立运行 pixel compute、独立运行 fullscreen display，且 display-only 不再创建空 compute pass。
  - `scripts/audit-demo.mjs` 的 WebGPU route 在读取 telemetry 前等待 queue done/failed 或 device lost，并验证新增 probe 的 dispatched/skipped stage contract。
  - Node smoke audit 覆盖新增 probe modes 和 normalize 行为。
- 诊断结论:
  - `pixel-compute-only`: pixel accumulation compute / storage write queue done，device active。
  - `display-only`: 无 compute dispatch，仅 fullscreen pixel-storage display pass 后 device lost、queue failed。
  - `tiny-pixel-output`: 32px viewport / 16 pixel workgroups 仍 device lost、queue failed，说明问题不是普通 viewport workload size。
  - 当前 blocker 已收敛到 fullscreen pixel-storage resolve/display path；下一步应对照 texture-backed display，而不是继续拆 pixel compute。
- 验证:
  - `git diff --check`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5224/ --no-server`: passed，Browser plugin absent，使用 Playwright fallback + built `dist/` static server。
  - `npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5224/ --no-server --webgpu-probe pixel-compute-only`: passed，`queue=done`、`deviceLost=active`、`pixel=dispatched`。
  - `npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5224/ --no-server --webgpu-probe display-only`: passed with allowed device lost，`queue=failed`、`deviceLost=lost`、all compute stages skipped。
  - `npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5224/ --no-server --webgpu-probe tiny-pixel-output`: passed with allowed device lost，`queue=failed`、`deviceLost=lost`、`pixelWorkgroups=16`。
  - `npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5224/ --no-server`: expected failed，仍报告 `probe=full` 和 `A valid external Instance reference no longer exists`。

### RENDER-005N: WebGPU runtime pass probes

- 状态: done / pixel-output-backend-loss-isolated
- 类型: 标准 PR / 前端渲染诊断
- 目标: 将 WebGPU runtime route 拆成 `accumulation-only`、`resolve-only` 和 `pixel-output-only` 三个可审计 probe，定位 headless unsafe WebGPU 的 queue/device loss 出现在哪条 pass path。
- 范围外:
  - 不声明 full WebGPU tile renderer 已稳定通过。
  - 不改变默认 `npm run audit:webgpu-runtime` 的严格失败语义。
  - 不把 probe 结果等同于完整对象编辑 runtime 验收。
- 实施:
  - 新增 `src/webgpuRuntimeProbe.js`，统一定义 `full`、`accumulation-only`、`resolve-only`、`pixel-output-only` probe modes。
  - `WebGpuTileViewport` 读取 `?webgpu-probe=...`，按 probe mode 只 dispatch 指定 stage；未运行的 stage 暴露为 `status=skipped`，并通过 `data-webgpu-runtime-probe` 暴露当前 probe。
  - `scripts/audit-demo.mjs` 新增 `--webgpu-probe` 和 `--allow-webgpu-device-lost`；`npm run audit:webgpu-probe` 用于收集诊断 telemetry，默认 full runtime audit 仍在 device lost 时失败。
  - 非 full probe audit 只验证 stage / queue / storage telemetry，不继续执行对象选择、隔离、删除交互，避免把 runtime 诊断混入 UI 交互失败。
  - Node smoke audit 覆盖 probe mode 常量和 normalize 行为。
- 诊断结论:
  - `accumulation-only`: queue done，device active；说明 covariance accumulation compute pass 本身不是当前 blocker。
  - `resolve-only`: queue done，device active；说明 tile accumulation -> tile resolved compute pass 本身不是当前 blocker。
  - `pixel-output-only`: first frame rendered 后 device lost，queue failed；当前 blocker 已收窄到 pixel-output path，即 pixel accumulation shader / pixel storage write / fullscreen pixel storage resolve 之一。
- 验证:
  - `git diff --check`: passed。
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5223/ --no-server`: passed，Browser plugin absent，使用 Playwright fallback + built `dist/` static server。
  - `npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5223/ --no-server --webgpu-probe accumulation-only`: passed，`queue=done`。
  - `npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5223/ --no-server --webgpu-probe resolve-only`: passed，`queue=done`。
  - `npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5223/ --no-server --webgpu-probe pixel-output-only`: passed with allowed device lost，`deviceLost=lost`、`queue=failed`。
  - `npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5223/ --no-server`: expected failed，仍报告 `probe=full`、`A valid external Instance reference no longer exists`。

### RENDER-005M: WebGPU required-limit and backend-loss diagnostics

- 状态: done / browser-webgpu-still-pending
- 类型: 标准 PR / 前端渲染诊断
- 目标: 消除 WebGPU route 中已知的 storage-buffer binding limit 和 JS cleanup 干扰项，并把当前 runtime blocker 定位到更具体的 queue/backend failure。
- 范围外:
  - 不声明 WebGPU tile renderer 已在浏览器 runtime 稳定通过。
  - 不实现最终 depth-sort / 真实 3D covariance 投影。
  - 不把编辑态 `原始颜色（编辑预览）` 伪装为 Spark / gsplat 真实对象级重渲染。
- 实施:
  - `detectWebGpuCapability()` 改为 adapter-only capability detection，不再创建并销毁 probe `GPUDevice`，避免 capability probe 自身污染 `device.lost` 诊断。
  - WebGPU tile runtime 在 `requestDevice()` 时显式请求 `requiredLimits.maxStorageBuffersPerShaderStage=9`；shader 当前需要 9 个 storage buffers/stage，当前 adapter 报告可支持 10。
  - `editRendererContract` 和 DOM contract 新增 storage-buffer binding limit telemetry；低于 9 时 blocked 于 `webgpu-binding-limit`。
  - `WebGpuTileViewport` cleanup 不再显式调用 `GPUDevice.destroy()`；只销毁 transient/storage buffers，并保留 device-lost 由浏览器 runtime 报告。
  - WebGPU init 增加短延迟，并在 capability pending 时渲染无 WebGL canvas 的 pending viewport，避免 fallback WebGL context 与 WebGPU 初始化交叉。
  - 强制 runtime audit 新增 `uncapturederror`、queue submit / `onSubmittedWorkDone()` telemetry；browser route 的内部 viewport output 暂降为 128，用于降低 headless runtime gate 负载，Node smoke 仍覆盖 1024 contract。
- 诊断结论:
  - 最小 localhost WebGPU 空提交可稳定 requestAdapter / requestDevice / submit / `onSubmittedWorkDone()`。
  - ObjGauss WebGPU route 已消除 “storage buffers 9 > default per-stage limit 8” warning，也不再由显式 JS `device.destroy()` 触发 lost。
  - 当前强制 runtime audit 仍在 first-frame submission 后失败为 `webgpu-device-lost-destroyed`；`deviceError=none`，queue telemetry 为 `webgpu-queue-submitted-work-failed: A valid external Instance reference no longer exists`。
- 验收:
  - WebGPU available 时必须请求 renderer 所需 storage-buffer per-stage limit。
  - storage-buffer per-stage limit 不足时必须明确 fallback 为 `webgpu-binding-limit`。
  - device-lost failure 必须同时报告 `deviceError` 和 queue state，便于区分 WGSL validation error、JS cleanup、queue/backend loss。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5222/ --no-server`: passed，assets=1，Browser plugin absent，使用 Playwright fallback + built `dist/` static server；覆盖 Spark 真实查看、Gaussian OIT 编辑、画布选择、隔离、删除预览和 `原始颜色（编辑预览）`。
  - `npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5222/ --no-server`: expected failed；当前 headless unsafe WebGPU 报告 `deviceError=none:: queue=failed:webgpu-queue-submitted-work-failed:A valid external Instance reference no longer exists`。
  - `npm run audit:demo -- --url http://127.0.0.1:5222/ --no-server`: attempted；本轮全量 3-asset audit 在 Plush/Spark 大场景的 headless SwiftShader GPU process 上长时间满载，已中止，未作为 005M 验收证据。

### RENDER-005L: WebGPU device-lost telemetry split

- 状态: done / browser-webgpu-pending
- 类型: 标准 PR / 前端渲染验收
- 目标: 将 WebGPU first-frame submission telemetry 与 `device.lost` telemetry 分离，避免 device loss 覆盖已完成的 accumulation / compute / pixel dispatch 证据。
- 实施:
  - `WebGpuTileViewport` 新增 `data-webgpu-device-lost-status`、`data-webgpu-device-lost-reason` 和 `data-webgpu-device-lost-message`。
  - `device.lost` 不再覆盖 `data-webgpu-first-frame-status`；first-frame 和 device-lost 作为两个独立 runtime facts 暴露。
  - `audit-demo` 的 WebGPU route 先检查 first-frame accumulation / compute / pixel / storage resolve，再单独阻断 `deviceLost=lost`。
- 验收:
  - 常规 fallback audit 不受 device-lost telemetry 影响。
  - 强制 WebGPU runtime audit 在当前 headless unsafe WebGPU 下必须报告 `WebGPU device was lost after first-frame submission`，而不是误报 first-frame blank。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5221/ --no-server`: passed，assets=3，Browser plugin absent，使用 Playwright fallback + built `dist/` static server。
  - `npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5221/ --no-server`: expected failed with `webgpu-device-lost-destroyed`.

### RENDER-005J: WebGPU storage/device-limit gate

- 状态: done / browser-webgpu-pending
- 类型: 标准 PR / 前端渲染架构
- 目标: 在进入 WebGPU tile renderer 前预测 runtime storage buffer 分配规模，并用 adapter/device limits 阻断超过 `maxBufferSize` / `maxStorageBufferBindingSize` 的场景。
- 范围外:
  - 不实现 GPU-side binning / prefix-sum。
  - 不实现 buffer chunking 或 viewport 分辨率降级。
  - 不把当前 headless Chrome fallback 宣称为真实 WebGPU runtime 证据。
- 实施:
  - `src/webgpuTileStorage.js` 新增 runtime storage estimate，按 WebGPU route 的 11 个 buffers 估算 `positionRadius`、`colorOpacity`、`scaleRotation`、`objectIndices`、`objectState`、`tileCounts`、`tileOffsets`、`tileAccumulation`、`tileResolvedRgba`、`pixelResolvedRgba` 和 `tileEntries`。
  - `src/webgpuCapability.js` 将 storage estimate 接入 `editRendererContract`，在 WebGPU available 且 tile capacity pass 后检查最大单 buffer 是否超过设备限制；超过时 fallback reason 为 `webgpu-buffer-limit`，target blocker 为 `webgpu-buffer-limit`。
  - `PointCloudViewport` 与 `WebGpuTileViewport` 暴露 storage-limit DOM telemetry；状态面板新增 `存储门禁`。
  - `audit-webgpu-tile-smoke` 覆盖 roomy pass、fixed overflow block、compact pass 和模拟小 binding 的 `webgpu-buffer-limit` block。
  - `audit-demo` 读取并验证 storage-limit telemetry；当前 WebGPU unavailable 时必须表现为 `unknown / webgpu-capability`。
- 验收:
  - Node smoke audit 必须证明 estimated runtime storage 与实际 11-buffer storage contract 一致。
  - 模拟小设备限制必须 blocked 于 `webgpu-buffer-limit`，不能进入 `webgpu-tile`。
  - 当前 headless Chrome 无 WebGPU adapter 时仍明确 fallback 到 `Gaussian OIT 编辑`，并暴露 capability blocker。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，覆盖 compact pass、fixed overflow block 和模拟小 binding 的 `webgpu-buffer-limit` block。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5217/ --no-server`: passed，assets=3；当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，storage gate 为 `unknown:webgpu-capability`。
- 完成 commit: 本次提交。

### RENDER-005I: WebGPU compact tile entry list

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 将 tile entries 从 fixed-cap per-tile stride 推进到 compact offset list，消除 Plush 级大场景因 fixed-cap overflow 被 capacity gate 长期阻塞的问题。
- 范围外:
  - 不实现 GPU-side binning / prefix-sum；当前 compact list 仍由 CPU smoke/runtime builder 生成后上传 WebGPU。
  - 不实现 depth-sorted compositing。
  - 不把当前 headless Chrome fallback 宣称为真实 WebGPU runtime 证据。
- 实施:
  - `src/webgpuTileSmoke.js` 新增 `compact-offset-list` 默认 capacity strategy，使用 `tileOffsets[tileIndex]` 作为 compact `tileEntries` 起点，`tileEntryCapacity=tileReferenceCount`，capacity gate 为 pass。
  - fixed-cap layout 仍保留为 `fixed-cap-smoke`，并在 smoke audit 中继续验证 overflow 会被 gate blocked，避免隐藏旧风险。
  - `src/webgpuTileStorage.js` 新增可选 `tileOffsets` storage buffer；WebGPU runtime route 现在上传 11 个 buffers。
  - `src/webgpuTileComputeShader.js` 的 tile accumulation 和 pixel accumulation shader 都从 `tileOffsets` 读取 entry base，不再依赖 `tileIndex * maxEntriesPerTile`。
  - `WebGpuTileViewport` bind group 新增 `tileOffsets`，并暴露 `data-webgpu-storage-tile-offsets`；两个 viewport 均暴露 `data-webgpu-tile-entry-layout` / `data-webgpu-tile-entry-offset-count`。
  - `audit-demo` 接受 `compact-offset-list` capacity telemetry，并要求真实 WebGPU route 上传 tile entries、tile offsets 和 pixel output。
- 验收:
  - Node smoke audit 默认 compact layout 必须 `capacity=pass`、`tileOverflowCount=0`、`tileEntryStoredCount=tileEntryCapacity=tileReferenceCount`。
  - Node smoke audit 的 fixed-cap 对照必须仍然 overflow/block。
  - WebGPU 可用环境中 Plush 级大场景不再因为 tile overflow 被 target gate blocked；若仍 fallback，必须是 capability 或 future device-limit blocker。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，输出 `storage=de5eaf8f:11 capacity=pass pixel=webgpu-compute-pixel-accumulation-v1:16384`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5216/ --no-server`: passed，assets=3；Plush semantic / Plush v1 均为 `tileCapacity="compact-offset-list":"ok":0`。当前 headless Chrome 仍 fallback，`targetGate="blocked":"webgpu-capability"` 是预期。

### RENDER-005H: WebGPU per-pixel Gaussian accumulation

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 将 `pixelResolvedRgba` 的来源从 tile resolved color 展开升级为每像素直接遍历所属 tile 的 Gaussian entries，并在 GPU compute 中执行 covariance-aware weighted OIT。
- 范围外:
  - 不实现完整 3D covariance 投影和深度排序。
  - 不解决 fixed-cap tile entries 在 Plush 级大场景上的 overflow blocker。
  - 不把当前 headless Chrome fallback 宣称为真实 WebGPU runtime 证据。
- 实施:
  - `src/webgpuTileComputeShader.js` 将 pixel stage source 升级为 `webgpu-compute-pixel-accumulation-v1`，pixel shader 不再读取 `tileResolvedRgba`，而是读取 `positionRadius`、`colorOpacity`、`objectIndices`、`objectState`、`tileCounts`、`tileEntries`、`scaleRotation` 和 pixel meta。
  - pixel shader 对每个像素定位 tile，遍历该 tile 中 stored Gaussian list，按 Gaussian scale / rotation 计算椭圆核，并把 weighted color / opacity resolve 到 `pixelResolvedRgba`。
  - `WebGpuTileViewport` 的 pixel bind group 改为绑定 Gaussian/object/tile-entry buffers；display pass 仍从 `pixelResolvedRgba` fullscreen resolve。
  - `src/webgpuTileSmoke.js` 的 Node smoke reference 同步计算 direct per-pixel Gaussian output；浏览器 runtime route 只分配 GPU 写入用 pixel buffer，不在主线程做 CPU 全帧 reference。
  - `audit-webgpu-tile-smoke` 断言 pixel shader 不再依赖 `tileResolvedRgba`，并验证 direct Gaussian pixel source / 48-byte pixel meta / positive pixel reference。
- 验收:
  - Node smoke audit 可在无 WebGPU adapter 环境中验证 pixel shader 读取 Gaussian/object/tile-entry storage 并输出 direct Gaussian pixel reference。
  - WebGPU 可用且 zero-overflow 的环境中，`webgpu-tile` first frame 必须经过 `webgpu-compute-pixel-accumulation-v1 -> webgpu-pixel-storage-resolve-v1`。
  - 当前 headless WebGPU 不可用环境仍明确 fallback 到 `Gaussian OIT 编辑`，三样例 browser audit 继续通过。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，输出 `storage=243af027:10 pixel=webgpu-compute-pixel-accumulation-v1:16384 resolveSource=webgpu-pixel-storage-resolve-v1`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5215/ --no-server`: passed，assets=3；当前 headless Chrome 仍 fallback，`pixel=null:null:0` 是预期，因为没有进入 WebGPU route。

### RENDER-005G: WebGPU viewport pixel output contract

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 在 covariance-aware tile sampling 基础上，增加 viewport-sized pixel output buffer，使 WebGPU display path 从 tile storage resolve 推进到 pixel storage resolve。
- 范围外:
  - 不实现最终逐像素 Gaussian accumulation；当前 pixel buffer 仍由 tile resolved color 展开生成。
  - 不实现完整 3D covariance 投影和深度排序。
  - 不把当前 headless Chrome fallback 宣称为真实 WebGPU runtime 证据。
- 实施:
  - `src/webgpuTileComputeShader.js` 新增 `webgpu-compute-pixel-resolve-v1` WGSL compute shader，把 `tileResolvedRgba` 展开到 viewport-sized `pixelResolvedRgba`。
  - `src/webgpuTileResolveShader.js` 升级为 `webgpu-pixel-storage-resolve-v1`，fullscreen fragment shader 直接读取 `pixelResolvedRgba`。
  - `src/webgpuTileStorage.js` 新增可选 `pixelResolvedRgba` storage buffer，WebGPU runtime route 现在包含 `positionRadius`、`colorOpacity`、`scaleRotation`、`objectIndices`、`objectState`、`tileCounts`、`tileAccumulation`、`tileResolvedRgba`、`pixelResolvedRgba` 和 `tileEntries` 共 10 个 buffers。
  - `WebGpuTileViewport` 在 accumulation 和 tile compute resolve 后 dispatch pixel resolve，并暴露 `data-webgpu-pixel-*` 与 `data-webgpu-storage-pixel-output` DOM contract。
  - `App` 只在实际进入 `webgpu-tile` route 时分配 pixel output buffer，fallback 状态不分配 viewport-sized pixel payload。
  - `audit-webgpu-tile-smoke` 和 `audit-demo` 更新为 pixel resolve route contract。
- 验收:
  - Node smoke audit 可在无 WebGPU adapter 环境中验证 10-buffer storage contract、pixel resolve shader、pixel workgroups 和 pixel-storage fullscreen resolve source。
  - WebGPU 可用且 zero-overflow 的环境中，`webgpu-tile` first frame 必须经过 `webgpu-compute-covariance-accumulation-v1 -> webgpu-compute-resolve-v1 -> webgpu-compute-pixel-resolve-v1 -> webgpu-pixel-storage-resolve-v1`。
  - 当前 headless WebGPU 不可用环境仍明确 fallback 到 `Gaussian OIT 编辑`，三样例 browser audit 继续通过。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，输出 `storage=5561b7fd:10 pixel=webgpu-compute-pixel-resolve-v1:16384 resolveSource=webgpu-pixel-storage-resolve-v1`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5214/ --no-server`: passed，assets=3；当前 headless Chrome 仍 fallback，`pixel=null:null:0` 是预期，因为没有进入 WebGPU route。

### RENDER-005F: WebGPU covariance-aware tile sampling

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 将 WebGPU accumulation 从径向 tile-center kernel 推进到消费 Gaussian scale / rotation 的 covariance-aware tile traversal，并用 2x2 tile samples 近似 tile footprint。
- 范围外:
  - 不实现 viewport-sized per-pixel accumulation buffer。
  - 不实现完整 3D covariance 投影和深度排序。
  - 不把当前 headless Chrome fallback 宣称为真实 WebGPU runtime 证据。
- 实施:
  - `src/webgpuTileComputeShader.js` 将 accumulation source 升级为 `webgpu-compute-covariance-accumulation-v1`，绑定 `scaleRotation` storage buffer，使用 Gaussian scale 和 rotation 计算椭圆高斯核。
  - accumulation shader 对每个 tile 遍历 stored Gaussian list，并在 tile 内 2x2 sample points 上累积 weighted OIT contribution。
  - `src/webgpuTileSmoke.js` 的 CPU reference 同步切到 `tile-2x2-covariance-weighted-oit`，保证 smoke telemetry、checksum 和 WebGPU shader contract 表示同一条近似路径。
  - `WebGpuTileViewport` 的 accumulation bind group 新增 `scaleRotation` buffer。
  - `audit-webgpu-tile-smoke` 和 `audit-demo` 更新为 covariance accumulation source 与新 resolve mode。
- 验收:
  - Node smoke audit 可在无 WebGPU adapter 环境中验证 covariance accumulation contract。
  - WebGPU 可用且 zero-overflow 的环境中，`webgpu-tile` first frame 必须经过 `webgpu-compute-covariance-accumulation-v1`。
  - 当前 headless WebGPU 不可用环境仍明确 fallback 到 `Gaussian OIT 编辑`，三样例 browser audit 继续通过。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，输出 `accumulation=webgpu-compute-covariance-accumulation-v1:64`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5213/ --no-server`: passed，assets=3；当前仍 fallback，`accumulation=null:null:0` 是预期，因为没有进入 WebGPU route。

### RENDER-005E: WebGPU per-tile Gaussian accumulation shader

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 将 WebGPU route 从“CPU 生成 `tileAccumulation` 后 GPU resolve”推进到“GPU compute 读取 tile entries 与 Gaussian/object storage buffers，按 tile-center weighted OIT 写出 `tileAccumulation`”。
- 范围外:
  - 不实现完整 per-pixel tile traversal。
  - 不实现 covariance-aware elliptical footprint rasterization。
  - 不把当前 headless Chrome fallback 宣称为真实 WebGPU runtime 证据。
- 实施:
  - `src/webgpuTileComputeShader.js` 新增 `webgpu-compute-accumulation-v1` WGSL compute shader：读取 `positionRadius`、`colorOpacity`、`objectIndices`、`objectState`、`tileCounts` 与 `tileEntries`，每个 tile 遍历 stored Gaussian list 并写入 `tileAccumulation`。
  - `WebGpuTileViewport` 创建 accumulation compute pipeline，并在 resolve compute 前 dispatch accumulation pass；WebGPU DOM contract 新增 `data-webgpu-accumulation-*`。
  - `App` 在真正进入 `webgpu-tile` route 时生成带 `tileEntries` 的 runtime tile smoke，并使用紧凑的 `maxEntriesPerTile=maxTileOccupancy`，避免 fallback 场景无意义分配大型 tile-entry buffer。
  - `audit-webgpu-tile-smoke` 验证 accumulation shader bindings、48-byte accumulation meta、workgroup 计算和 shader 写入 contract。
  - `audit-demo` 的 WebGPU route 现在要求 accumulation dispatch、resolve compute dispatch、storage tile entries 和 storage-buffer fullscreen resolve 同时成立。
- 验收:
  - Node smoke audit 可在无 WebGPU adapter 环境中验证 accumulation compute contract。
  - WebGPU 可用且 zero-overflow 的环境中，`webgpu-tile` first frame 必须经过 `webgpu-compute-accumulation-v1` -> `webgpu-compute-resolve-v1` -> `webgpu-storage-resolve-v1`。
  - 当前 headless WebGPU 不可用环境仍明确 fallback 到 `Gaussian OIT 编辑`，三样例 browser audit 继续通过。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，输出 `accumulation=webgpu-compute-accumulation-v1:64`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5212/ --no-server`: passed，assets=3；当前仍 fallback，`accumulation=null:null:0` 是预期，因为没有进入 WebGPU route。

### RENDER-005D: WebGPU compute resolve shader

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 将 WebGPU route 从“CPU 写好 `tileResolvedRgba` 后直接显示”推进到“GPU compute 从 `tileAccumulation` 写 `tileResolvedRgba`，再由 storage-buffer resolve shader 显示”。
- 范围外:
  - 不实现 per-Gaussian tile traversal。
  - 不实现 tile list 上的完整 per-pixel accumulation。
  - 不把当前 headless Chrome fallback 宣称为真实 WebGPU runtime 证据。
- 实施:
  - 新增 `src/webgpuTileComputeShader.js`，定义 `webgpu-compute-resolve-v1` WGSL compute shader：读取 `tileAccumulation`，按 weighted OIT resolve 规则写入 `tileResolvedRgba`。
  - `WebGpuTileViewport` 创建 compute pipeline，并在同一个 command encoder 中先 dispatch compute，再执行 fullscreen storage-buffer resolve render pass。
  - WebGPU viewport 新增 `data-webgpu-compute-*` DOM contract：source、status、reason 和 workgroup count。
  - `audit-webgpu-tile-smoke` 验证 compute shader contract、16-byte compute meta 和 workgroup 计算。
  - `audit-demo` 的 WebGPU route 现在要求 compute 已 dispatch，且 first frame 仍来自 `webgpu-storage-resolve-v1`。
- 验收:
  - Node smoke audit 可在无 WebGPU adapter 环境中验证 compute resolve contract。
  - WebGPU 可用环境中，`webgpu-tile` route 的 first frame 必须经过 `webgpu-compute-resolve-v1`。
  - 当前 headless WebGPU 不可用环境仍明确 fallback 到 `Gaussian OIT 编辑`，三样例 browser audit 继续通过。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，输出 `compute=webgpu-compute-resolve-v1:64`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5209/`: passed，assets=3；当前仍 fallback，`compute=null:null:0` 是预期，因为没有进入 WebGPU route。

### RENDER-005C: WebGPU storage-buffer resolve shader

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 将 WebGPU first-frame display path 从 CPU resolved texture sampling 推进到从 `tileResolvedRgba` storage buffer 直接 fullscreen resolve，为后续 compute accumulation 输出接入同一个 buffer 铺路。
- 范围外:
  - 不实现 WebGPU compute accumulation shader。
  - 不实现 per-pixel tile traversal；当前 `tileResolvedRgba` 内容仍来自 CPU tile-center smoke resolve。
  - 不把当前 headless Chrome fallback 宣称为真实 WebGPU runtime 证据。
- 实施:
  - 新增 `src/webgpuTileResolveShader.js`，定义 `webgpu-storage-resolve-v1` WGSL：fragment shader 读取 `var<storage, read> tileResolvedRgba: array<vec4f>` 和 `ResolveMeta` uniform，不再依赖 sampled texture。
  - `WebGpuTileViewport` 使用 storage-buffer resolve shader 创建 render pipeline，bind `tileResolvedRgba` storage buffer 与 16-byte resolve meta uniform 后绘制 fullscreen triangle。
  - WebGPU viewport 新增 `data-webgpu-resolve-source`，真实 WebGPU route 会暴露 `webgpu-storage-resolve-v1`。
  - `audit-webgpu-tile-smoke` 验证 resolve shader contract、16-byte resolve meta、storage bundle `getBuffer("tileResolvedRgba")` 和无 `textureSample` 依赖。
  - `audit-demo` 的 WebGPU route 现在要求 first frame 来自 `webgpu-storage-resolve-v1`。
- 验收:
  - Node smoke audit 可在无 WebGPU adapter 环境中验证 storage-buffer resolve contract。
  - WebGPU 可用环境中，`webgpu-tile` route 的 first frame 必须来自 storage-buffer resolve shader。
  - 当前 headless WebGPU 不可用环境仍明确 fallback 到 `Gaussian OIT 编辑`，三样例 browser audit 继续通过。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，输出 `resolveSource=webgpu-storage-resolve-v1`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5208/`: passed，assets=3；当前仍 fallback，`resolveSource=null` 是预期，因为没有进入 WebGPU route。

### RENDER-005B: WebGPU storage-buffer upload contract

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 将 WebGPU device-backed skeleton 从“只上传 CPU resolved texture”推进到真实创建并写入 tile/object storage buffers，使 final tile renderer 的核心数据布局进入 WebGPU runtime 边界。
- 范围外:
  - 不实现 WebGPU compute accumulation shader。
  - 不替换 CPU tile-center resolve texture 的 first-frame display path。
  - 不把当前 headless Chrome fallback 宣称为真实 WebGPU runtime 证据。
- 实施:
  - 新增 `src/webgpuTileStorage.js`，定义 `webgpu-tile-storage-v1` storage contract，覆盖 `positionRadius`、`colorOpacity`、`scaleRotation`、`objectIndices`、`objectState`、`tileCounts`、`tileAccumulation`、`tileResolvedRgba` 和可选 `tileEntries`。
  - `WebGpuTileViewport` 在 first-frame / tileSmoke update path 中创建 WebGPU storage buffers、写入 typed-array payload、保留 bundle 并在下一次更新/卸载时销毁。
  - WebGPU viewport 暴露 `data-webgpu-storage-*` DOM contract：layout、status、reason、buffer count、byte size、checksum 和 tileEntries presence。
  - `audit-webgpu-tile-smoke` 使用 fake WebGPU device 验证 buffer descriptors、writeBuffer 调用、byte size、checksum、destroy 行为，以及隔离/删除后 storage checksum 变化。
  - `audit-demo` 的 WebGPU route 增加 storage upload 断言：真实进入 `webgpu-tile` 时必须有 `storageStatus="uploaded"`、positive byte size 和 8 位 checksum，并要求隔离/删除更新 checksum。
- 验收:
  - Node smoke audit 可在无 WebGPU adapter 环境中验证 storage contract。
  - WebGPU 可用环境中，`webgpu-tile` route 不仅要 first-frame rendered，还必须上传 storage buffers。
  - 当前 headless WebGPU 不可用环境仍明确 fallback 到 `Gaussian OIT 编辑`，三样例 browser audit 继续通过。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，输出 `storage=86cb35c1:9`。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run audit:demo -- --url http://127.0.0.1:5207/`: passed，assets=3；当前仍 fallback，`storage=null:null` 是预期，因为没有进入 WebGPU route。

### RENDER-004E: WebGPU overflow gate and fallback hardening

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 将 fixed-capacity tile smoke 的 overflow 从单个计数升级为可审计 capacity gate，并强化 WebGPU target fallback reason，避免 overflow 场景被误认为可直接切换到 WebGPU renderer。
- 范围外:
  - 不创建 WebGPU compute/render pipeline。
  - 不把实际编辑 renderer 从 `Gaussian OIT 编辑` 切到 WebGPU。
  - 不把 fixed-capacity tile list 宣称为最终方案。
- 实施:
  - `src/webgpuTileSmoke.js` 新增 tile capacity summary：overflow tile count、overflow ratio、max excess、stored reference count、entry capacity、entry utilization、capacity mode/status/gate。
  - `src/webgpuCapability.js` 新增 WebGPU target gate，区分 `webgpu-capability`、`tile-overflow` 和 `renderer-not-implemented` 三类 blocker。
  - `App` 状态面板展示目标状态与 Tile capacity。
  - `PointCloudViewport` 暴露 `data-webgpu-target-gate-*` 与 `data-webgpu-tile-capacity-*` DOM contract。
  - `audit-webgpu-tile-smoke` 同时验证小 cap overflow blocked 与大 cap pass/ok。
  - `audit-demo` 校验浏览器中的 target gate、capacity gate，并确保 overflow 场景不会通过高质量 WebGPU readiness。
- 验收:
  - Plush semantic / Plush v1 暴露 `tileCapacity="overflow"`，并记录 overflow tile count。
  - Lego 暴露 `tileCapacity="ok"`，overflow tile count 为 0。
  - 当前 headless WebGPU 不可用时，target gate 明确为 `blocked:webgpu-capability`。
  - 若 WebGPU 可用且 tile overflow 存在，audit 要求 blocker 为 `tile-overflow`；若无 overflow，则仍 blocked 于 `renderer-not-implemented`，直到真实 WebGPU renderer 完成。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，内置 sample packed=5800、refs=157323、resolved=2301、overflow=40114、overflowTiles=1056、capacity=blocked。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --url http://127.0.0.1:5203/`: passed，assets=3；Plush semantic / Plush v1 为 `tileCapacity="overflow":169`，Lego 为 `tileCapacity="ok":0`，三者 targetGate 均为 `blocked:webgpu-capability`。
- 完成 commit: `8d233a1`.

### RENDER-004D: WebGPU object-state buffer smoke

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 将 WebGPU tile smoke 的对象可见性从隐含 0/1 数组升级为可审计的 storage-buffer contract，使隐藏、隔离、删除和选中状态都能进入 future WebGPU renderer。
- 范围外:
  - 不创建 WebGPU compute/render pipeline。
  - 不把实际编辑 renderer 从 `Gaussian OIT 编辑` 切到 WebGPU。
  - 不声明 Spark `.splat` 真实查看已支持 object-state 过滤。
- 实施:
  - `src/webgpuTileSmoke.js` 新增 `webgpu-object-state-v1`，使用 `vec4u`-style stride 4 buffer：flags、dense object index、Gaussian count、reserved。
  - object-state flags 覆盖 visible、selected、removed、isolated 和 enabled，并输出 dense object id mapping、visible/hidden/removed/selected/isolated object counts 与 checksum。
  - `App` 将 `selectedId` 输入 WebGPU tile smoke，并在渲染状态面板展示 Object state。
  - `PointCloudViewport` 暴露 `data-webgpu-object-state-*` DOM contract。
  - `audit-webgpu-tile-smoke` 和 `audit-demo` 校验 object-state layout、stride、checksum，以及隔离 / 删除后的状态变更。
- 验收:
  - 初始样例暴露 `objectStateLayout="webgpu-object-state-v1"`、stride=4、visible objects 为正、hidden/removed/selected/isolated 均为 0。
  - 画布选中后点击 `只看所选`，object-state checksum 变化，visible objects=1，selected/isolated objects=1。
  - 点击 `预览删除` 后 checksum 再次变化，removed objects=1，isolated objects=0，visible objects 减 1。
  - 当前 headless WebGPU 不可用时仍明确 fallback 到 `Gaussian OIT 编辑`，浏览器交互仍通过。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，内置 sample packed=5800、tiles=2362/4096、refs=157323、resolved=2301、objectState=72aeff5e。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --url http://127.0.0.1:5202/`: passed，assets=3；Plush semantic objectState 362760d7 -> 637142bc，Plush v1 e1cdb2e4 -> b0b19f1f，Lego 7243475b -> 7ca4643c。
- 完成 commit: `9c1c0a2`.

### RENDER-004C: WebGPU tile accumulation and resolve smoke

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 在真实 WebGPU tile renderer 前，先把 tile binning 结果推进到 deterministic tile-center accumulation / resolve smoke contract，并暴露可审计的 resolve telemetry。
- 范围外:
  - 不创建 WebGPU compute/render pipeline。
  - 不把实际编辑 renderer 从 `Gaussian OIT 编辑` 切到 WebGPU。
  - 不声明删除后的自身颜色已经是完整 3DGS 重渲染。
- 实施:
  - `src/webgpuTileSmoke.js` 新增 `webgpu-tile-resolve-v1`，对每个 active tile 做 tile-center weighted OIT accumulation，并生成 resolved RGBA、resolved tile count、weight sum、alpha/luma mean 和 checksum。
  - `src/webgpuCapability.js`、`src/App.jsx` 和 `src/PointCloudViewport.jsx` 将 resolve layout / mode / telemetry 暴露到状态面板和 `data-webgpu-resolve-*` DOM contract。
  - `scripts/audit-webgpu-tile-smoke.mjs` 校验 tile accumulation / resolved RGBA buffer shape、positive resolve telemetry，以及隔离 / 删除后 checksum 变化。
  - `scripts/audit-demo.mjs` 校验三样例浏览器 DOM 中的 resolve layout、mode、resolved tile count、weight、alpha/luma 和 checksum。
- 验收:
  - 三个默认闭环样例均暴露 `resolveLayout="webgpu-tile-resolve-v1"` 和 `resolveMode="tile-center-weighted-oit"`。
  - resolved tile count、resolve weight、alpha/luma mean 为正，checksum 为 8 位 hex。
  - 当前 headless WebGPU 不可用时仍明确 fallback 到 `Gaussian OIT 编辑`，对象选择、隔离和删除预览保持通过。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，内置 sample packed=5800、tiles=2362/4096、refs=157323、resolved=2301、checksum=c8567887。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --url http://127.0.0.1:5201/`: passed，assets=3；Plush semantic resolvedTiles=3051 / checksum=9feb3736，Plush v1 resolvedTiles=3051 / checksum=4e86df13，Lego resolvedTiles=3881 / checksum=2b4d3d8e。
- 完成 commit: `359d930`.

### RENDER-004B: WebGPU tile smoke packing and binning contract

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 在实现 WebGPU accumulation shader 前，先把 ObjGauss Gaussian scene 打包为 future WebGPU storage-buffer layout，并生成 deterministic tile occupancy / overflow telemetry。
- 范围外:
  - 不创建 WebGPU compute/render pipeline。
  - 不把实际编辑 renderer 从 `Gaussian OIT 编辑` 切到 WebGPU。
  - 不隐藏 fixed-capacity tile list 的 overflow 风险。
- 实施:
  - 新增 `src/webgpuTileSmoke.js`，输出 `positionRadius`、`colorOpacity`、`scaleRotation`、`objectIndices`、`objectState`、`tileCounts` 和可选 `tileEntries` typed arrays。
  - 使用 `16x16` tile、orthographic smoke projection 和 fixed-capacity tile entry cap 生成 pack/binning telemetry。
  - `App` 将当前 scene、颜色模式、可见 / 隔离 / 删除 object-state 输入 smoke builder，并在状态面板展示 pack、tile bins 和 overflow。
  - `PointCloudViewport` 暴露 `data-webgpu-pack-layout`、packed / visible / binned Gaussian counts、tile size/count、active tiles、tile references、max tile occupancy、overflow 和目标 object-state buffer。
  - `audit-demo` 校验三样例的 WebGPU tile smoke DOM contract。
  - 新增 `npm run audit:webgpu-tile-smoke`，验证 typed array layout 和隔离 / 删除时 binning 计数跟随 object-state 变化。
- 验收:
  - 三个默认闭环样例均暴露 `tileSmokeLayout="webgpu-tile-smoke-v1"`。
  - `packedGaussians`、`binnedGaussians`、active tiles 和 tile references 为正。
  - `objectFilterTarget="gpu-object-state-buffer"` 已作为 RENDER-004D 目标 contract 暴露。
  - 当前 headless WebGPU 不可用时仍明确 fallback 到 `Gaussian OIT 编辑`。
- 验证:
  - `npm run audit:webgpu-tile-smoke`: passed，内置 sample packed=5800、tiles=2362/4096、refs=157323。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --url http://127.0.0.1:5199/`: passed，assets=3；Plush packed=281498、activeTiles=3119/4096、tileReferences=10513313、maxTileOccupancy=11026、tileOverflowCount=196038。
- 完成 commit: `5a98ba8`.

### RENDER-004A: WebGPU renderer boundary and capability contract

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 在真正实现 WebGPU tile renderer 前，先建立 renderer boundary、WebGPU capability detection 和可审计 fallback contract。
- 范围外:
  - 不实现 tile binning、WebGPU storage buffer packing 或 accumulation shader。
  - 不替换当前 `Gaussian OIT 编辑` fallback。
  - 不声称删除后的自身颜色已经是完整 3DGS 重渲染。
- 实施:
  - 新增 `src/webgpuCapability.js`，检测 `navigator.gpu`、adapter 和 device，并输出稳定 capability reason。
  - `App` 维护 WebGPU capability 状态，明确显示当前 edit renderer、目标 renderer、WebGPU 状态、fallback reason 和 tile overflow。
  - `SplatViewport` 暴露 `data-renderer="spark-splat"`。
  - `PointCloudViewport` 暴露 `data-renderer="gaussian-oit"`、`data-renderer-target="webgpu-tile"`、`data-renderer-fallback-reason`、`data-webgpu-status`、`data-tile-overflow-count` 和 object filter。
  - `audit-demo` 验证 Spark 真实查看 renderer id、WebGPU tile target、resolved WebGPU status、fallback reason、tile overflow 和现有 object-state filtering。
- 验收:
  - WebGPU 不可用环境中明确 fallback 到 `Gaussian OIT 编辑`，不伪装成 WebGPU renderer。
  - Browser audit 能看到 `rendererTarget="webgpu-tile"`、`editRendererId="gaussian-oit"` 和非空 fallback reason。
  - 三个默认闭环样例的真实查看、编辑画布选中、隔离和删除预览仍通过。
- 验证:
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --url http://127.0.0.1:5197/`: passed，assets=3；当前 headless Chrome 输出 `webgpuStatus="unavailable"`、`fallbackReason="webgpu-adapter-unavailable"`、`tileOverflowCount=0`。
- 完成 commit: `ad82813`.

### RENDER-003: GPU object-state filtering for edit renderer

- 状态: done
- 类型: 标准 PR / 前端渲染架构
- 目标: 接入 object-id filtering，使 shader preview 能按对象隐藏、隔离和删除，而不是只靠 React 先过滤 PLY 点数组。
- 范围外:
  - 不要求一次完成 WebGPU compute pipeline。
  - 不替换 Spark `.splat` 真实查看 renderer。
  - 不实现真实 Spark splat shader 内对象过滤。
- 实施:
  - `PointCloudViewport` 保留全量 Gaussian geometry 常驻 GPU。
  - 每个 Gaussian 上传 dense object index attribute。
  - `buildObjectFilter` 将可见 / 删除 / 隔离状态编码为 object-state `DataTexture`。
  - Gaussian fragment shader 通过 `uObjectState` texture 判断 object 是否可见，不可见则 `discard`。
  - 画布 raycast 会跳过当前 object-state 不可见的 object。
  - Audit 增加 `objectFilter="gpu-object-state-texture"` contract。
- 验收:
  - 三个默认 Web demo 样例均暴露 `gpu-object-state-texture` object filter。
  - 隔离和删除只更新 object-state，可见计数与画面状态一致。
  - 画布点选、隔离、删除预览仍工作。
  - 定向 Playwright QA 无 shader / framebuffer / texture console error。
- 验证:
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --url http://127.0.0.1:5194/`: passed，assets=3。
  - Targeted Playwright QA: passed，`281498 -> 48066 -> 233432` 可见计数链路成立，截图位于 `/tmp/objgauss-gpu-filter-*.png`。
- 完成 commit: `de45d90`.

### RENDER-002: Weighted blended OIT for object edit renderer

- 状态: done
- 类型: 标准 PR / 前端渲染
- 目标: 在 `Gaussian Shader 编辑` renderer 上增加 weighted blended accumulation，降低当前普通透明混合带来的排序伪影。
- 范围外:
  - 不引入 WebGPU tile renderer。
  - 不替换 Spark 真实查看 renderer。
  - 不实现完整 3D covariance projection、SH view-dependent color 或真实 splat shader 内对象删除。
- 实施:
  - `PointCloudViewport` 增加 RGBA half-float accumulation render target。
  - Gaussian fragment shader 输出 `vec4(color * weight, weight)`。
  - 使用 additive custom blending 累加 `sum(w*c)` 与 `sum(w)`。
  - fullscreen resolve pass 输出 `sum(w*c) / sum(w)`，再以 normal blending 混回基础 grid / axes scene。
  - UI / audit renderer label 更新为 `Gaussian OIT 编辑`。
- 验收:
  - 三个默认 Web demo 样例均能进入 `Gaussian OIT 编辑`。
  - 画布点选 object、隔离和删除预览仍工作。
  - 删除后回到 `自身颜色` 并显示剩余整体场景。
  - 定向 Playwright QA 无 shader / framebuffer / render target console error；已知 Spark `Worker terminate` 切换噪声单独过滤。
- 验证:
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --url http://127.0.0.1:5193/`: passed，assets=3。
  - Targeted Playwright QA: passed，截图位于 `/tmp/objgauss-oit-edit-*.png`。
- 完成 commit: `a7e40f6`.

### WEB-002: Gaussian shader edit renderer phase 1

- 状态: done
- 类型: 标准 PR / 前端渲染
- 目标: 按 B -> C 路线完成 Phase 1，把对象编辑 renderer 从 `PointsMaterial` / soft sprite 升级为 screen-space Gaussian kernel shader。
- 范围外:
  - 不实现 weighted blended OIT。
  - 不实现 WebGPU tile renderer。
  - 不实现完整 3D covariance projection、SH view-dependent color 或真实 splat shader 内对象删除。
- 实施:
  - 新增 ADR 0004，明确 `ShaderMaterial` -> Weighted OIT -> WebGPU tile renderer 的渐进路线。
  - `src/ply.js` 解析 `scale_0/1/2`、`rot_0..3` 和 `opacity`，生成前端 Gaussian scale / rotation / opacity 字段。
  - `src/PointCloudViewport.jsx` 使用 `ShaderMaterial`，通过 `gl_PointCoord` 计算椭圆 Gaussian kernel alpha。
  - `src/sampleScene.js` 为内置 demo 提供 scale / rotation fallback。
  - `scripts/audit-demo.mjs` 增加 `editRenderer="Gaussian Shader 编辑"` 断言。
- 验收:
  - 三个默认 Web demo 样例均能进入 `Gaussian Shader 编辑`。
  - 画布点选 object、隔离和删除预览仍工作。
  - 删除后回到 `自身颜色` 并显示剩余整体场景。
  - 定向 Playwright QA 无 shader compile/link console error。
- 验证:
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --url http://127.0.0.1:5192/`: passed，assets=3。
  - Targeted Playwright QA: passed，截图位于 `/tmp/objgauss-shader-edit-*.png`。
- 完成 commit: `8465e4a`.

### WEB-001: Make viewer interaction coherent

- 状态: done
- 类型: 标准 PR / 前端交互修复
- 目标: 修复 Web 查看器“工程验收能跑但交互模型断裂”的问题，使对象级查看/编辑路径更清楚可用。
- 范围外:
  - 不实现真实 splat shader 内对象删除；对象级编辑仍是点云编辑预览。
  - 不新增后端服务或提交 benchmark `/tmp` 报告产物。
  - 不把训练素材或 source-only assets 放回 Web 素材卡片。
- 实施:
  - 顶部 `旋转 / 框选 / 聚焦` 假工具替换为真实 `真实查看` / `对象编辑` 工作模式。
  - 无 `.splat` 的内置/PLY 场景默认对象编辑；带 `.splat` 样例加载后默认真实查看。
  - 对象选择会进入对象编辑模式，并在软点云 viewport 中用高亮层显示选中对象。
  - 点云编辑 viewport 支持点击画布选择最近 Gaussian 对应的 object，避免只能通过右侧列表选中。
  - 对象编辑 renderer 从硬点改为软圆形 sprite splat 近似，降低删除后自身颜色预览的颗粒感；仍明确不是完整 covariance-aware 3DGS shader。
  - 删除预览会退出 `只看所选` 隔离并切回 `自身颜色`，显示删除后的剩余整体场景。
  - 对象行拆成独立选择按钮和可见性按钮，去掉按钮内嵌 switch 的可访问性问题。
  - 素材库只展示可加载样例，并新增 Benchmark tab 显示 SEMANTIC-003 gates 与三场景指标。
  - 移动端布局改为 viewport 优先的纵向堆叠。
- 验收:
  - 初始内置场景显示对象编辑 / 点云编辑。
  - `ObjGauss v1 闭环样例` 加载后显示真实查看 / 真实 Splat。
  - 点击对象行后切入对象编辑并显示编辑预览 banner。
  - 在软点云编辑画布中点击可直接选中 object，并继续执行隔离/删除预览。
  - 隔离 object 后删除，显示删除后的剩余整体场景，并回到 `自身颜色`。
  - Benchmark tab 显示 smoke/candidate/paper pass 和 Lego/Fern/Chair 行。
  - Web 素材库不再显示 Poly Haven School Chair 1K 这类不能直接渲染的 source-only 卡片。
- 验证:
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
  - `npm run audit:demo -- --asset plush-v1-closure-local --url http://127.0.0.1:5190/`: passed。
  - `npm run audit:demo -- --asset plush-v1-closure-local --url http://127.0.0.1:5191/`: passed，包含 `canvasSelectedObject=1`、`editPixels=64530`、`visibleAfterDelete=230,581` 和 `renderModeAfterDelete="自身颜色"` 断言。
  - Targeted Playwright QA: passed，截图和状态日志位于 `/tmp/objgauss-web001-qa/`。
- 完成 commit: `b8b0fe3`.
- Follow-up commit: `cbf1b7b`.
- Follow-up commit: `0cccd1a`.
- Follow-up commit: `d7e9dbc`.

### SEMANTIC-003C: Add third Splatfacto-trained scene and close paper gate

- 状态: done
- 类型: 标准 PR / 训练实验
- 目标: 为 cross-scene paper gate 补齐第 3 个 Splatfacto-trained scene row 和第 3 个 held-out mask eval row。
- 范围外:
  - 不提交 `outputs/`、checkpoint、SAM checkpoint 或训练产物。
  - 不把当前 `scale_aware_cpu_splat_l1` 误称为完整 covariance-aware `gsplat` renderer。
  - 不把 Poly Haven Chair smoke 误称为真实相机采集场景；它是 CC0 mesh-derived NeRF-style render set。
- 实施:
  - 新增 `polyhaven-school-chair-nerf` 自动素材源，从 Poly Haven School Chair glTF 生成 16-frame NeRF-style RGBA orbit dataset。
  - 新增 `objgauss.mesh_nerf` 纯 Python/NumPy glTF rasterizer，写出 `train/*.png` 与 `transforms_train/val/test.json`。
  - 将 `chair-splatfacto-smoke` 加入 `docs/benchmarks/splatfacto-scenes.json`，并配置 SAM、train/held-out split、Object Field benchmark 参数和 prepare 命令。
  - 更新素材库、前端素材登记、benchmark runbook 和状态文档。
- 验收:
  - Chair asset pull 生成 `outputs/assets/training/polyhaven-school-chair-nerf/transforms_train.json`。
  - Chair 100-step Splatfacto smoke 导出 `outputs/training/polyhaven-chair-splatfacto-smoke/export-smoke-cuda/splat.ply`，50000 Gaussians。
  - Chair SAM smoke 生成 8 frames / 48 masks；scene split 为 train 6 frames / held-out 2 frames。
  - Chair scene row: ARI=0.614363、curve OES=0.757609、render=0.248716、held-out projection_loss=2.284750、held-out render=0.224084。
  - Scene suite summary 含 3 scenes：Lego safe-2000、LLFF Fern smoke、Poly Haven Chair smoke。
  - Cross-scene summary 含 9 rows：3 semantic smoke + 3 Splatfacto scene rows + 3 Lego variants。
  - Cross-scene stage gates 为 smoke=true、candidate=true、paper=true；failure report 显示 `Paper gate passed`。
- 验证:
  - `uv run objgauss assets pull polyhaven-school-chair-nerf`: passed。
  - `npm run train:splatfacto:smoke -- --run --asset-id polyhaven-school-chair-nerf ... --skip-benchmark`: passed。
  - `node scripts/benchmark-splatfacto-scenes.mjs --run --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth`: passed，scenes=3。
  - `node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-scenes --skip-variants`: passed，rows=9，stage gates smoke=true / candidate=true / paper=true。
  - `node scripts/benchmark-splatfacto-scenes.mjs --status`: `status=ready missing=0`。
  - `node scripts/benchmark-cross-scene.mjs --status`: `status=ready missing=0`。
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "asset_registry or assets_list or polyhaven_nerf or splatfacto_scene or cross_scene" -q`: 5 passed。
  - `uv run --extra dev pytest`: 41 passed。
  - `npm run build`: passed，仍有 Spark / Three bundle size warning。
- 完成 commit: `6bf95d2`。

### SEMANTIC-003A/003D/003E: Scale-aware renderer occlusion, stage gates, and failure reports

- 状态: done
- 类型: 标准 PR
- 目标: 将 SEMANTIC benchmark 从旧 point/depth render probe 推进到 scale-aware image-space occlusion contract，并补 stage gates 与 failure report。
- 范围外:
  - 不引入新的 `gsplat` runtime renderer。
  - 不伪造第三个真实 Splatfacto scene。
  - 不提交本地 `/tmp`、`outputs/`、SAM checkpoint 或训练产物。
- 实施:
  - `objgauss.render_probe.render_occlusion_delta` 升级为 `scale_aware_cpu_splat_l1`，使用 Gaussian scale / opacity rasterize CPU splat footprint，再比较 full-vs-object-removed RGBA delta。
  - render occlusion summary 增加 target delta、non-target delta 和 locality score。
  - `emergence-benchmark` 支持可选 `heldout_masks` / `heldout` 配置，能在最终 Object Field 上评估 held-out projection loss、supervision 和 render occlusion。
  - `emergence-benchmark` 输出 `failure-report.md`。
  - `benchmark-cross-scene.mjs` 输出 smoke / candidate / paper stage gates 与 cross-scene failure report。
- 验收:
  - semantic smoke acceptance 使用 `scale_aware_cpu_splat_l1` 通过。
  - Splatfacto scene suite 与 safe-2000 variant suite 均用新 renderer 重跑通过。
  - cross-scene summary rows=8，smoke/candidate gate 通过，paper gate 明确因第三 scene 和 held-out rows 缺失而失败。
- 验证:
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "emergence" -q`: 7 passed。
  - `npm run acceptance:semantic`: passed。
  - `node scripts/benchmark-splatfacto-scenes.mjs --run --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth`: passed。
  - `node scripts/benchmark-splatfacto-variants.mjs --run --skip-sam`: passed。
  - `node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-scenes --skip-variants`: passed，rows=8，stage gates smoke=true / candidate=true / paper=false。
- 完成 commit: `750f646`。

### BENCH-004: Real Splatfacto cross-scene suite with LLFF Fern

- 状态: done
- 类型: 标准 PR
- 目标: 将 cross-scene 表从 “Lego 单场景 + mask policy variants” 推进到真实 Splatfacto scene suite，新增 LLFF Fern 作为第二个真实 Splatfacto 场景。
- 范围外: 不提交 `outputs/`、`/tmp`、SAM checkpoint、Nerfstudio checkpoint 或训练产物；不声称 Fern smoke 是高质量 reconstruction；不替代 covariance-aware 3DGS renderer occlusion。
- 实施:
  - 新增 `nerf-llff-fern` 自动素材源，从 NeRF example zip 抽取 `nerf_llff_data/fern`，并从 COLMAP `sparse/0` 生成 `transforms_train.json`。
  - `objgauss masks from-nerf-sam` 支持 JPEG 输入和 `--max-image-size`，用于大图 CPU/GPU 资源安全 SAM smoke。
  - `scripts/train-splatfacto-smoke.mjs` 支持 `--asset-id`、`--data-parser colmap`、`--downscale-factor`、`--cuda-home`、`--max-jobs` 和 `--dataparser-transform`。
  - 新增 `scripts/apply-mask-dataparser-transform.mjs`，将 Nerfstudio `dataparser_transforms.json` 乘进 mask manifest camera transforms，使 COLMAP scene masks 与导出 PLY 坐标对齐。
  - 新增 `scripts/benchmark-splatfacto-scenes.mjs`、`npm run benchmark:splatfacto:scenes` 和 `docs/benchmarks/splatfacto-scenes.json`，定义 Lego safe-2000 与 Fern smoke 两个真实 Splatfacto scene。
  - `scripts/benchmark-cross-scene.mjs` 聚合 semantic smoke、Splatfacto scene suite 和 safe-2000 variant suite，cross-scene 表从 6 行扩展为 8 行。
- 验收:
  - Fern asset pull 生成 `outputs/assets/training/nerf-llff-fern/transforms_train.json`。
  - Fern 100-step Splatfacto smoke 导出 `outputs/training/nerf-fern-splatfacto-smoke/export-smoke-cuda/splat.ply`，10091 Gaussians。
  - Fern SAM smoke 使用 CPU + `max_image_size=768` 生成 4 frames / 24 masks。
  - dataparser transform 后，Fern Object Field register-output 监督 1247 Gaussians，projection loss `3.778366 -> 0.670971`。
  - scene suite summary 含 2 scenes：Lego safe-2000 与 Fern smoke。
  - cross-scene summary 含 8 rows：3 semantic smoke + 2 real Splatfacto scenes + 3 Lego variants。
- 验证:
  - `node scripts/benchmark-splatfacto-scenes.mjs --run --scene fern-splatfacto-smoke --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth`: passed。
  - `node scripts/benchmark-splatfacto-scenes.mjs --run --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth`: passed，scenes=2。
  - `node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-scenes --skip-variants`: passed，rows=8。
  - `node scripts/benchmark-splatfacto-scenes.mjs --status`: `status=ready missing=0`。
  - `node scripts/benchmark-cross-scene.mjs --status`: `status=ready missing=0`。
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "asset_registry or nerf_pull or fern_pull or splatfacto_scene or cross_scene or splatfacto_variant or splatfacto_balanced or splatfacto_smoke or nerf_sam" -q`: 10 passed。
  - `uv run --extra dev pytest`: 39 passed。
  - `npm run build`: 通过，仍有 Spark / Three bundle size warning。
- 完成 commit: `3536197`.

### BENCH-003: Cross-scene emergence benchmark table

- 状态: done
- 类型: 标准 PR
- 目标: 将 semantic smoke 多场景 suite 与 safe-2000 mask variant suite 聚合成同一张跨场景 / 跨变体实验表。
- 范围外: 不训练新场景；不新增外部数据源；不提交 `/tmp`、`outputs/`、SAM checkpoint 或训练产物；不把 point-splat render probe 升级为真实 gsplat renderer。
- 实施:
  - 新增 `scripts/benchmark-cross-scene.mjs`，支持 `--dry-run`、`--status`、`--run`、`--skip-semantic`、`--skip-variants` 和 `--refresh-sam`。
  - 新增 `npm run benchmark:cross-scene`。
  - 新增 `docs/benchmarks/cross-scene.md`，记录输入 suite、输出表格和解释边界。
  - 聚合 `semantic-smoke` 三个 scene 与 `splatfacto-safe2000-variants` 三个 variant，输出统一 `summary.json`、`summary.csv`、`summary.md` 和 `summary.html`。
- 验收:
  - 一条命令可重跑 semantic smoke suite、safe-2000 variant suite，并生成 6 行统一表。
  - summary 统一记录 suite、scene_id、variant_id、frames、masks、gaussians、slots、supervised_gaussians、object_id_counts、ARI、OES 和 render occlusion effect。
  - `--status` 可检查 semantic summary、variant summary 和 cross-scene 表是否齐全。
- 验证:
  - `npm run benchmark:cross-scene -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth`: passed。
  - `node scripts/benchmark-cross-scene.mjs --run`: passed，rows=6。
  - `node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-variants`: passed。
  - `node scripts/benchmark-cross-scene.mjs --status`: `status=ready missing=0`。
  - cross-scene summary: semantic smoke rows=3，safe-2000 variant rows=3；best render row 当前为 `lego-alpha-proxy/default`，safe-2000 内最佳仍为 `sam8f-slots4-balanced03`。
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "cross_scene or splatfacto_variant or splatfacto_balanced or splatfacto_smoke" -q`: 4 passed。
  - `uv run --extra dev pytest`: 36 passed。
  - `npm run build`: 通过，仍有 Spark / Three bundle size warning。
- 完成 commit: `05f40d0`.

### BENCH-002: Safe-2000 mask variant comparison suite

- 状态: done
- 类型: 标准 PR
- 目标: 将 safe-2000 balanced benchmark 扩展成同一 Splatfacto PLY 上的多 mask / slot policy 对比表，为后续多 scene 实验表格打基础。
- 范围外: 不重新训练 Splatfacto；不新增外部数据源；不提交 `outputs/`、SAM checkpoint、训练 checkpoint 或 benchmark 产物。
- 实施:
  - 新增 `scripts/benchmark-splatfacto-variants.mjs`，编排 `sam2f-slots8`、`sam8f-slots8-unfiltered`、`sam8f-slots4-balanced03` 三个变体。
  - 新增 `npm run benchmark:splatfacto:variants`。
  - 新增 `docs/benchmarks/splatfacto-variants.md`，记录变体定义、输入、输出和解释边界。
  - 变体 suite 复用 BENCH-001 的单变体 benchmark 脚本，输出 suite-level `summary.json`、`summary.csv`、`summary.md` 和 3-curve HTML report。
- 验收:
  - 一条命令可重跑三组 mask policy 的 register-output、emergence metrics、emergence curve 和 comparison report。
  - suite summary 记录每个变体的 frames、masks、slots、supervised_gaussians、object_id_counts、ARI、OES 和 render occlusion effect。
  - `--status` 可检查三组 manifest、per-variant summary 和 suite summary 是否齐全。
- 验证:
  - `npm run benchmark:splatfacto:variants -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth`: passed。
  - `node scripts/benchmark-splatfacto-variants.mjs --run --skip-sam`: passed。
  - `node scripts/benchmark-splatfacto-variants.mjs --status`: `status=ready missing=0`。
  - suite summary: `sam8f-slots4-balanced03` 在 ARI=0.468745、OES=0.693888、render_occlusion_effect_score=0.195308 三项中均为当前最佳。
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "splatfacto_variant or splatfacto_balanced or splatfacto_smoke" -q`: 3 passed。
  - `uv run --extra dev pytest`: 35 passed。
  - `npm run build`: 通过，仍有 Spark / Three bundle size warning。
- 完成 commit: `192cd3a`.

### BENCH-001: Stabilize safe-2000 balanced benchmark

- 状态: done
- 类型: 标准 PR
- 目标: 将 safe-2000 balanced 8-frame / 4-slot SAM candidate 纳入可复现 benchmark/runbook，减少手工命令和 ignored outputs 依赖。
- 范围外: 不重新训练 Splatfacto；不提交 checkpoint、SAM checkpoint 或大体积训练输出。
- 实施:
  - 新增 `scripts/benchmark-splatfacto-balanced.mjs`，支持 `--dry-run`、`--status`、`--run`、`--skip-sam` 和显式 `--publish`。
  - 新增 `npm run benchmark:splatfacto:balanced`。
  - 新增 `docs/benchmarks/splatfacto-balanced.md`，记录输入、固定参数、输出 contract、summary 字段和缺失输入处理。
  - 默认使用 `--no-public-copy`，避免 benchmark run 覆盖 `public/samples/nerf_lego_trained.*`。
- 验收:
  - `node scripts/benchmark-splatfacto-balanced.mjs --run` 重新生成 balanced SAM -> `training register-output` -> emergence metrics -> emergence curve -> HTML report -> summary。
  - `summary.json` 记录 frames=8、masks=27、object_id counts=126686/40747/34682/53679、ARI=0.468745、OES=0.693888、render_occlusion_effect_score=0.195308。
  - `node scripts/benchmark-splatfacto-balanced.mjs --status` 输出 `status=ready missing=0`；缺失项会打印对应 prepare 命令。
- 验证:
  - `npm run benchmark:splatfacto:balanced -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth`: passed。
  - `node scripts/benchmark-splatfacto-balanced.mjs --run`: passed。
  - `node scripts/benchmark-splatfacto-balanced.mjs --run --skip-sam`: passed。
  - `node scripts/benchmark-splatfacto-balanced.mjs --status`: `status=ready missing=0`。
  - `uv run --extra dev pytest tests/test_objgauss_mvp.py -k "splatfacto_balanced or splatfacto_smoke" -q`: 2 passed。
  - `uv run --extra dev pytest`: 34 passed。
  - `npm run build`: 通过，仍有 Spark / Three bundle size warning。
- 完成 commit: `b4d34da`.

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
- 完成 commit: `6d0a922`.

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
