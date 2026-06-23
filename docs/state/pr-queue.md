# ObjGauss PR 队列

> 最近更新: 2026-06-23

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
  - `RENDER-005A`: 在 WebGPU-capable 浏览器中重跑 first-frame runtime audit。
- 验收底线:
  - WebGPU 可用环境中暴露 `data-renderer="webgpu-tile"` 和 `data-object-filter="gpu-object-state-buffer"`。
  - 不支持 WebGPU 或初始化失败时明确 fallback 到当前 `Gaussian OIT 编辑`，不静默伪装成功。
  - 隔离 / 删除后 `visibleCount` 与 object-state 一致，并记录 `tileOverflowCount`。

## In Progress

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

## Done

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
