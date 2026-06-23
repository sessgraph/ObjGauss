# ObjGauss 当前状态总览

> 最近更新: 2026-06-24

## 当前阶段

MVP 原型可运行，已完成流程化基线提交，已接入真实 3DGS splat renderer，并具备可复现的 ObjGauss v1 闭环验收 demo。

## 阶段最终目标

当前阶段的最终目标不是先追求完整科研级训练质量，而是把 ObjGauss v1 的最小闭环变成可重复验收的工程事实：

```text
多视角数据 / 3DGS 场景
  -> Gaussian 场
  -> 每个 Gaussian 的 Object Field 概率
  -> 2D mask / 语义线索修正 object_logits
  -> 导出 object_id
  -> 前端可选择、隔离、删除对象
```

验收视角：一条命令能重新生成 Plush semantic、Plush v1 与 NeRF Lego 三个闭环样例，机器检查产物，再打开浏览器验证真实 splat 外观和对象级交互。

## 已完成能力

- Python CLI:
  - `objgauss convert-splat`
  - `objgauss cluster`
  - `objgauss colorize`
  - `objgauss filter`
  - `objgauss stats`
  - `objgauss assets list/pull`
  - `objgauss masks from-nerf-alpha/from-nerf-rgba-colors/from-nerf-sam`
  - `objgauss training register-output`
  - `objgauss demo v1-closure/verify-v1-closure/plush-semantic-closure/verify-plush-semantic-closure/lego-alpha-closure/verify-lego-alpha-closure/audit-v1-goal`
  - `objgauss object-field init/export/stats/emergence/emergence-curve/emergence-report/emergence-benchmark/inspect-nerf/vote-masks`
- 前端:
  - 中文 UI。
  - Spark / Three.js 真实 3DGS splat 预览。
  - Three.js Gaussian Shader 对象编辑 fallback。
  - 已拆分 `真实查看` / `对象编辑` 两个工作模式；对象操作会显式进入点云编辑预览，不再伪装成真实 splat 内直接编辑。
  - 原始颜色（编辑预览） / 对象色（编辑预览）切换；UI 不再把编辑预览文案伪装成真实 3DGS 重渲染。
  - 对象列表、点击 Gaussian OIT 画布选中对象、隔离、删除预览；选中对象在 shader 编辑模式下有高亮层，删除预览会退出隔离并切回原始颜色（编辑预览）显示剩余整体场景，且可一键清除编辑状态返回真实 Splat。
  - 对象编辑 renderer 已从 `PointsMaterial` 升级为 screen-space Gaussian kernel `ShaderMaterial`，消费 PLY `scale_0/1/2`、`rot_0..3` 和 `opacity` attributes，并通过 RGBA half-float accumulation / fullscreen resolve 实现 weighted blended OIT。
  - 对象过滤已进入 shader 路径：每个 Gaussian 上传 dense object index attribute，隐藏 / 隔离 / 删除通过 GPU object-state `DataTexture` 控制；WebGPU tile renderer 仍是后续任务。
  - RENDER-004 WebGPU tile renderer 已补 ADR 设计入口；RENDER-004A capability detection / renderer boundary、RENDER-004B buffer packing / binning smoke contract、RENDER-004C tile accumulation / resolve smoke contract、RENDER-004D object-state buffer smoke contract 与 RENDER-004E overflow gate / fallback hardening 已完成。
  - RENDER-005A 已落地 WebGPU device-backed first-frame skeleton：zero-overflow + WebGPU available 时会切到 `WebGPU Tile 编辑`，上传 tile resolve texture 并绘制 fullscreen triangle；当前本机 headless Chrome 无 WebGPU adapter，真实 runtime first-frame audit 仍 pending，Gaussian OIT 继续作为 fallback。
  - RENDER-005B 已落地 WebGPU storage-buffer upload contract：WebGPU route 会创建并写入 `webgpu-tile-storage-v1` buffers，覆盖 Gaussian geometry/color/object-state/tile counts/tile accumulation/resolve payload。
  - RENDER-005C 已落地 WebGPU storage-buffer resolve shader：WebGPU first-frame display path 会从 `tileResolvedRgba` storage buffer 读取并 fullscreen resolve。
  - RENDER-005D 已落地 WebGPU compute resolve shader：WebGPU route 会先 dispatch compute，把 `tileAccumulation` resolve 到 `tileResolvedRgba`，再由 storage-buffer fullscreen pass 显示。
  - RENDER-005E 已落地 WebGPU tile-center accumulation shader：WebGPU route 会先读取 `tileEntries`、Gaussian buffers 和 object-state storage，在 GPU compute 中写出 `tileAccumulation`。
  - RENDER-005F 已落地 WebGPU covariance-aware tile sampling：accumulation shader 消费 Gaussian scale / rotation，并在 tile 内 2x2 sample points 上近似椭圆高斯 footprint。
  - RENDER-005G 已落地 WebGPU viewport pixel output contract：WebGPU route 会把 tile accumulation resolve 到 `tileResolvedRgba`，再写入 viewport-sized `pixelResolvedRgba`，最后由 `webgpu-pixel-storage-resolve-v1` fullscreen shader 显示。
  - RENDER-005H 已落地 WebGPU per-pixel Gaussian accumulation：`webgpu-compute-pixel-accumulation-v1` 不再复制 `tileResolvedRgba`，而是每个像素读取所属 tile 的 Gaussian entries、object state、scale / rotation 和 color / opacity，直接计算椭圆高斯 weighted OIT 并写入 `pixelResolvedRgba`。
  - RENDER-005I 已落地 WebGPU compact tile entry list：tile entry storage 从 fixed-cap stride 推进到 `compact-offset-list`，新增 `tileOffsets` buffer；Plush 级大场景不再因为 fixed-cap tile overflow 被 capacity gate 阻塞，后续 WebGPU 可用时会进入真实 tile renderer 路径。
  - RENDER-005J 已落地 WebGPU storage/device-limit gate：WebGPU route 进入前会预测 runtime 11-buffer storage 规模，并用 `maxBufferSize` / `maxStorageBufferBindingSize` 阻断超限场景；当前 headless Chrome 仍无 WebGPU adapter，真实 runtime audit 继续 pending。
  - RENDER-005K 已落地强制 WebGPU runtime audit 入口：`npm run audit:webgpu-runtime` 会要求浏览器真实进入 `webgpu-tile`，常规 `audit:demo` 仍保留 fallback 验收。
  - RENDER-005L 已落地 WebGPU device-lost telemetry split：first-frame submission / accumulation / compute / pixel dispatch 与 `device.lost` 作为独立 runtime facts 暴露，强制 audit 可区分“已提交渲染命令”和“提交后 device lost”。
  - RENDER-005M 已落地 WebGPU requiredLimits + backend-loss diagnostics：capability detection 不再创建 probe device，runtime `requestDevice` 显式请求 `maxStorageBuffersPerShaderStage=9`，storage binding limit 会 blocked 于 `webgpu-binding-limit`，并新增 `uncapturederror` 与 queue `onSubmittedWorkDone()` telemetry；当前 blocker 已收敛为 headless unsafe WebGPU 的 queue/backend loss。
  - RENDER-005N 已落地 WebGPU runtime pass probes：`?webgpu-probe=accumulation-only|resolve-only|pixel-output-only` 和 `npm run audit:webgpu-probe` 可单独提交 runtime pass；当前诊断显示 accumulation-only 与 resolve-only queue done，pixel-output-only first frame 后 device lost。
  - RENDER-005O 已将 pixel-output probe 继续拆成 `pixel-compute-only`、`display-only` 和 `tiny-pixel-output`：当前 headless unsafe WebGPU 下 pixel compute / storage write 可 queue done，单独 fullscreen pixel-storage display pass 会 device lost，tiny 32px 合并路径仍 device lost。
  - RENDER-005P 已新增 texture-backed display 和 `clear-only` probes：sampled texture、buffer-to-texture copy display 以及无 draw 的 canvas clear pass 都会在当前 headless unsafe WebGPU 下 device lost；blocker 已从 shader/storage/texture 路径收敛为 canvas render pass / presentation backend loss。
  - RENDER-005Q 已完成 desktop WebGPU audit：headless diagnostic 仍分类为 presentation backend loss，但 headed desktop Chrome/WebGPU 下 `clear-only`、`texture-display-only` 和 `full` runtime audit 均通过；NeRF Lego proxy 已证明 WebGPU tile runtime 可以完成 first frame、compute dispatch、object-state 过滤、选择、隔离和删除预览。
  - RENDER-005R 已完成大场景 desktop WebGPU runtime audit：Plush semantic、Plush v1 与本机 safe-2000 Splatfacto sample 均以 255k-281k Gaussian 进入 `WebGPU Tile 编辑`，storage gate / compact tile list / full compute pipeline / object-state 交互均通过。
  - RENDER-005S 已完成 WebGPU runtime visual fidelity audit 第一轮：full runtime 内部 pixel output 从 `128x128` 提升到 `256x256`，并暴露 viewport / pixel-count telemetry；NeRF Lego proxy 与 Plush semantic 大场景均在 headed desktop Chrome/WebGPU 下通过 256px full runtime audit。
  - RENDER-005T-A 已完成 WebGPU pixel-storage bilinear resolve：full runtime display pass 不再用最近邻放大 `pixelResolvedRgba`，而是使用 bilinear storage sampling，并通过 audit 暴露 `resolveSource=webgpu-pixel-storage-resolve-v1:bilinear-storage`。
  - RENDER-005T-B 已完成 WebGPU aspect-fit runtime viewport：full runtime 会根据实际 viewer display size 计算 area-preserving internal viewport，projection bounds 改为 aspect-fit + 8% 留白，减少固定方形 viewport 和 x/z 独立拉伸导致的贴边 / 比例偏差。
  - RENDER-005T-C 已完成 WebGPU edit-camera perspective projection：WebGPU Tile 编辑预览现在按固定编辑相机在 CPU 端打包 screen-space center / depth / sigma，GPU accumulation / pixel resolve 和 canvas 点击命中不再按旧 x/z 正交 bounds 做二次投影，并通过 audit 暴露 `projection=edit-perspective-camera-v1:52`。
  - RENDER-005T-D 已完成 WebGPU front-weighted OIT depth contract：WebGPU Tile 编辑预览现在会记录 edit-camera depth range，并在 tile accumulation / per-pixel resolve 中使用 `front-weighted-oit-v1`，减少纯 weighted OIT 把前后层 Gaussian 直接混色的问题。
  - RENDER-005T-E 已完成 WebGPU camera-Jacobian screen covariance：前端 PLY parser 保留三轴 scale / quaternion，WebGPU Tile 按 edit-camera projection Jacobian 将 3D covariance 投影成 screen-space ellipse，并以 4:1 anisotropy clamp 降低低分辨率 tile preview 的针状 streak。
  - RENDER-005T-F 已完成 WebGPU adaptive runtime quality：默认 full runtime 从固定 256px 输出升级为按场景规模和显示比例自适应，小场景可到 `adaptive-high-512`，Plush 级大场景走 `adaptive-medium-384`，并通过 audit 暴露质量档和 pixel budget。
  - RENDER-005T-G 已完成 WebGPU source-color fidelity audit：前端 PLY parser 保留 RGB / SH DC / fallback 颜色来源，WebGPU Tile 暴露 `source-color-fidelity-v1`，browser audit 证明 Plush 与 Lego 删除预览后均回到 100% RGB 原始颜色而非对象调试色或 fallback 色。
  - RENDER-005T-H 已完成 WebGPU front-depth gated pixel resolve：per-pixel resolve 现在先找每像素最近有效 Gaussian contributor，再用 `front-depth-gated-oit-v1` 抑制后层 contributor 混入；这是比纯 front-weighted OIT 更强的遮挡近似，但仍不是完整 per-pixel sorted alpha 或 Spark 真实 `.splat` 重渲染。
  - RENDER-005T-I 已完成 Spark vs edit visual residual audit：browser audit 会采集 Spark canvas 与“对象编辑 / 原始颜色”canvas 的 coverage、luma、chroma 和 checksum，并输出 `spark-edit-visual-residual-v1`；NeRF Lego WebGPU full audit 当前显示编辑预览 coverage 是 Spark 的约 4.47x，说明下一步应优先校准 alpha / footprint coverage 或继续拆 view-dependent SH 差距。
  - RENDER-005T-J 已完成 WebGPU footprint coverage calibration：WebGPU Tile 编辑预览暴露 `footprint-weight-floor-calibrated-v1`，pixel resolve 使用 `0.004` 权重 floor，footprint scale 校准到 `2.2`；NeRF Lego Spark/edit coverage ratio 从 `4.469421` 降到 `3.271989`，但 luma / chroma 未同步改善，说明“自身颜色不像真实高斯”的剩余主因是排序 alpha / SH / Spark 合成路径差距，而不是 RGB 原色丢失。
  - RENDER-005T-K 已完成 WebGPU depth-binned alpha compositing：pixel resolve 升级为 `depth-binned-alpha-composite-v1`，每像素固定 8 个 depth bins 做 front-to-back alpha compositing；NeRF Lego luma / chroma delta 从 T-J 的 `0.207570 / 0.133965` 降到 `0.109000 / 0.087808`，Plush 大场景也通过 desktop WebGPU full audit，但 coverage ratio 未同步改善，说明后续需把 coverage 和 shading 分线治理。
  - RENDER-005T-L 已完成 WebGPU alpha presentation edge gate：fullscreen storage resolve 新增 `alpha-edge-gated-presentation-v1:0.035`，只在最终显示阶段压掉低 alpha halo，不改 compute resolve buffer；NeRF Lego coverage ratio 从 T-K 的 `3.856920` 小幅降到 `3.784251`，Plush 从 `6.680406` 降到 `6.448639`，说明 halo 有贡献但 coverage 主问题仍需 footprint / covariance / threshold sweep。
  - RENDER-005T-M 已完成 WebGPU coverage tuning sweep：WebGPU Tile 支持 runtime `webgpu-footprint-scale` / `webgpu-covariance-max-anisotropy`，新增 `npm run audit:webgpu-coverage-sweep`；Lego sweep 显示 coverage 可从 baseline `3.784251` 降到 tight `3.346752`，但 luma / chroma 从 `0.106079 / 0.086537` 恶化到 `0.142279 / 0.102668`，下一步需 Pareto scoring / multi-scene sweep。
  - RENDER-005T-N 已完成 WebGPU coverage Pareto multi-scene sweep：`npm run audit:webgpu-coverage-sweep` 支持 `--assets` 多场景、解析 `tileReferences`，并按 coverage / luma / chroma / tile reference cost 的 `0.35 / 0.25 / 0.25 / 0.15` 权重输出每场景和跨场景 variant score；Lego best Pareto 仍是 baseline，Plush best Pareto 是 compact，tight 只在 coverage / cost 上最好但 luma 代价明显，因此默认渲染参数暂不切到 tight。
  - RENDER-005T-O 已完成 WebGPU coverage report / threshold gate：`audit:webgpu-coverage-sweep` 支持 `--output-dir` 写 `summary.json` / `summary.md`，并支持 `--gate-variant` 与 mean / per-scene pareto、luma、chroma、tile-reference 阈值；新增 `npm run audit:webgpu-coverage-gate` 作为默认参数变更前的 baseline gate，当前 2-scene headed WebGPU gate 通过并写出 `/tmp/objgauss-webgpu-coverage-sweep-gate/summary.*`。
  - RENDER-005T-P 已完成 WebGPU runtime depth-bin tuning：WebGPU Tile 的 depth-binned alpha composite 不再把 8 bins 硬编码在 shader / smoke / audit 三处，新增 `runtime-depth-sort-tuning-v1` 和 URL / audit 参数 `webgpu-depth-bins`，运行时可在 4-16 bins 间调参；默认仍保持 8 bins，coverage gate 证明 baseline 未变化，12-bin headed WebGPU audit 证明 tuned shader 可真实进入 runtime。
  - 素材库卡片只展示当前 viewer 可直接加载/交互的本地 Gaussian 样例。
  - Web 内已有 Benchmark tab，展示 SEMANTIC-003 smoke / candidate / paper gates 和三场景 Splatfacto 指标。
  - 移动端已改为 viewport 优先的纵向堆叠布局。
  - `NeRF Lego 训练输出样例` 卡片已预留，外部训练产物登记到 `public/samples/nerf_lego_trained.*` 后可加载。
  - `npm run audit:demo` 可启动临时 Vite 服务并浏览器验收三个闭环样例。
- 素材:
  - `plush-3dgs-local` 可自动拉取。
  - Plush `.splat` 用于真实 renderer，`plush_objects.ply` 用于对象级编辑。
  - `polyhaven-school-chair-1k` 可自动拉取到 mesh Demo 输入目录。
  - `polyhaven-school-chair-nerf` 可从 Poly Haven School Chair glTF 离线渲染 NeRF-style RGBA orbit dataset，用作第三个 Splatfacto-trained benchmark scene row。
  - `nerf-synthetic-lego` 可自动拉取到训练素材目录。
  - `nerf-llff-fern` 可从 NeRF example zip 自动抽取 LLFF/COLMAP Fern，并生成 ObjGauss mask/vote 可用的 `transforms_train.json`。
  - ARKitScenes、ScanNet、OmniObject3D、Google Scanned Objects、Poly Haven、Mip-NeRF 360、Tanks and Temples 已登记为候选来源。
- Object Field:
  - 已有 `object_logits: (N, K)` 软分区文件格式。
  - 可从现有 Gaussian PLY warm start，并导出 hard `object_id` PLY 复用前端。
  - 可检查 NeRF-style `transforms_*.json` 训练素材完整性。
  - 可从 NeRF Synthetic RGBA alpha 通道生成真实图片 mask manifest。
  - 可从 NeRF Synthetic Lego RGBA 颜色生成多 slot 真实 2D mask manifest。
  - 可在本机提供 `segment-anything` 和 checkpoint 时生成 SAM automatic mask manifest，支持 JPEG 输入和 `--max-image-size` 资源安全降采样。
  - 可消费预计算 SAM / CLIP / 2D mask manifest，并投影投票到 Gaussian。
  - 可通过 projection loss 更新 Object Field logits。
  - 可输出 mask vote quality audit，检查监督覆盖率、每槽覆盖、冲突比例、target entropy 和观测权重。
  - 可输出 Object Emergence observability metrics，检查 assignment entropy、effective slots、空间紧致度、reference stability / ARI 和 partial OES。
  - 可输出 Object Emergence benchmark curves，跟踪 projection loss、entropy、effective slots、ARI、空间紧致度、mask-proxy occlusion delta 和 scale-aware CPU splat render occlusion delta 随 mask-vote training iteration 的变化。
  - 可将多个 emergence curve JSON 聚合为 HTML/SVG benchmark report artifact，用于横向比较多场景曲线。
  - 可从 benchmark manifest 一键重跑多场景 emergence curves、CSV、HTML report 和 summary，并执行阈值检查。
  - `emergence-benchmark` 支持可选 `heldout_masks`，可用同一训练参数生成最终 Object Field 后在 held-out mask manifest 上评估 projection loss、监督覆盖和 render occlusion effect。
  - `emergence-benchmark` 和 cross-scene 聚合会写 failure report，用于记录失败 checks 和 paper-readiness gap。
  - 可机器检查 mask guidance 是否实际改变 Object Field hard labels。
- 训练输出接入:
  - `objgauss training register-output` 可登记外部成熟 3DGS 训练器产出的 `.ply` / `.splat`。
  - 登记时可生成 viewer `.splat`、标准 Gaussian PLY、Object Field、mask 投票 summary 和 `object_id` PLY。
  - 带 mask 登记时，Object Field 初始场使用 Gaussian 几何 warm start，避免全零 logits 在稀疏 mask vote 下坍缩到少数对象槽。
  - 本机已验证 Nerfstudio Splatfacto 可读取 `nerf-synthetic-lego` 的 `blender-data` 格式，完成 100-step CUDA smoke 训练、导出 Gaussian PLY，并接入 Object Field / SAM mask voting。
  - `npm run train:splatfacto:smoke` 已固化为 TRAIN-003A smoke 生成入口，支持 `--dry-run`、`--status` 和 `--run`。
  - 本机已完成 NeRF Lego Splatfacto 500-step resource-safe candidate，导出 47168 个 Gaussian，并通过 `training register-output` 登记为本机 `NeRF Lego 训练输出样例` public sample；该产物在 ignored `outputs/` / `public/samples/`，不进入 git。
  - 本机已完成 NeRF Lego Splatfacto 2000-step resource-safe candidate，导出 255794 个 Gaussian；几何/渲染指标强于 safe-500，但 2-frame SAM supervision 下 object slots 仍不平衡，暂不作为最终语义样例结论。
  - `objgauss masks from-nerf-sam` 支持 `--max-area-fraction` 过滤过大的 SAM masks；safe-2000 当前最佳语义候选是 8-frame / 4-slot / `max_area_fraction=0.3`，已消除近空 object slots 并提升 render occlusion effect。
- Demo:
  - `objgauss demo v1-closure` 可生成当前 v1 闭环验收包。
  - `objgauss demo verify-v1-closure` 可重新读取产物并机器检查闭环证据。
  - `objgauss demo plush-semantic-closure` 可在真实 Plush `.splat` 上生成非 KMeans 的 2D color mask manifest、训练 Object Field，并导出保留原色的 `object_id` PLY。
  - `objgauss demo verify-plush-semantic-closure` 可检查真实 splat、2D color masks、Object Field、loss、`object_id` PLY、public assets 和前端素材注册。
  - `objgauss demo lego-alpha-closure` 可从 NeRF Lego 真实多视角 RGBA + pose 生成轻量 Gaussian proxy、2D color mask manifest 和 object-aware PLY。
  - `objgauss demo verify-lego-alpha-closure` 可检查 Lego proxy demo 的源图像、mask 文件、Object Field、loss、`object_id` PLY、public assets 和前端素材注册。
  - 前端素材库已有 `Plush 2D 语义 Mask 闭环样例`，加载后可查看真实 splat 外观并执行对象隔离/删除预览。
  - 前端素材库已有 `ObjGauss v1 闭环样例`，加载后可查看真实 splat 外观并执行对象隔离/删除预览。
  - 前端素材库已有 `NeRF Lego 闭环代理样例`，运行 demo 命令后可加载 proxy splat 和对象 PLY。
- 流程:
  - `docs/development-flow.md` 已建立。
  - `AGENTS.md` 和 `CLAUDE.md` 已指向统一流程。
  - `npm run acceptance:demo` 已固化为一键闭环总验收命令。
  - `npm run acceptance:semantic` 已固化为 SEMANTIC benchmark suite 验收命令。
  - `docs/training/splatfacto-smoke.md` 已记录 Splatfacto smoke 训练 / 导出 / SAM / Object Field 的 runbook 和输出 contract。
  - `npm run benchmark:splatfacto:balanced` 已固化为 safe-2000 balanced candidate 的一键本地 benchmark 入口，可重跑 balanced SAM、`training register-output`、emergence metrics、curve、report 和 summary。
  - `npm run benchmark:splatfacto:variants` 已固化为 safe-2000 同场景多 mask / slot policy 对比入口，可生成三变体 summary、CSV、Markdown 表格和 HTML 曲线报告。
  - `npm run benchmark:splatfacto:scenes` 已固化为 Splatfacto-trained scene suite，可比较 Lego safe-2000、LLFF Fern smoke 与 Poly Haven Chair smoke 三个 scene rows，并支持 train / held-out SAM manifest split。
  - `npm run benchmark:cross-scene` 已固化为跨场景 / 跨变体汇总入口，可聚合 semantic smoke suite、Splatfacto scene suite 和 safe-2000 variant suite 到同一张表，并输出 smoke / candidate / paper stage gates。
  - `npm run audit:webgpu-coverage-gate` 已固化为 WebGPU 编辑预览 coverage/luma/chroma/cost 的多场景 baseline gate，并输出可复查 summary report。
  - `objgauss demo audit-v1-goal --allow-incomplete` 已固化为阶段目标完成度审计命令。
  - baseline commit: `c8dcef7`.

## 最近验证

2026-06-24:

```bash
node --check src/webgpuDepthTuning.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5274 --probes full --webgpu-depth-bins 12
npm run audit:webgpu-coverage-gate -- --port 5275
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run audit:webgpu-coverage-gate -- --port 5270
npm run audit:webgpu-coverage-sweep -- --assets nerf-lego-alpha-closure-local,plush-semantic-closure-local --port 5268
node --check src/webgpuTileSmoke.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-coverage-sweep.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-coverage-sweep -- --port 5266
node --check src/webgpuTileResolveShader.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5264/ --no-server --probes full
node scripts/audit-webgpu-desktop.mjs --asset plush-semantic-closure-local --url http://127.0.0.1:5264/ --no-server --probes full
node --check src/webgpuTileSmoke.js
node --check src/webgpuTileComputeShader.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5263/ --no-server --probes full
node scripts/audit-webgpu-desktop.mjs --asset plush-semantic-closure-local --url http://127.0.0.1:5263/ --no-server --probes full
node --check src/webgpuTileSmoke.js
node --check src/webgpuTileComputeShader.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5262/ --no-server --probes full
node scripts/audit-webgpu-desktop.mjs --asset plush-semantic-closure-local --url http://127.0.0.1:5262/ --no-server --probes full
node --check scripts/audit-demo.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
node scripts/audit-demo.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5260/ --no-server
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5260/ --no-server --probes full
node scripts/audit-webgpu-desktop.mjs --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5260/ --no-server --probes pixel-compute-only --allow-device-lost-probes
node --check src/webgpuTileComputeShader.js
node --check src/webgpuTileSmoke.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5258 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5259 --probes full
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5245 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5246 --probes full
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5247 --probes full
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5252 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5253 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5254 --probes full
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5255 --probes full
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5256 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5257 --probes full
```

2026-06-23:

```bash
npm run build
npm run audit:demo -- --url http://127.0.0.1:5194/
npm run audit:demo -- --url http://127.0.0.1:5193/
npm run audit:demo -- --url http://127.0.0.1:5192/
npm run audit:demo -- --asset plush-v1-closure-local --url http://127.0.0.1:5191/
npm run audit:demo -- --asset plush-v1-closure-local --url http://127.0.0.1:5190/
uv run objgauss assets pull polyhaven-school-chair-nerf
npm run train:splatfacto:smoke -- --run --asset-id polyhaven-school-chair-nerf --dataset outputs/assets/training/polyhaven-school-chair-nerf --output-root outputs/training/polyhaven-chair-splatfacto-smoke --experiment chair-splatfacto-smoke --timestamp smoke-cuda --export-dir outputs/training/polyhaven-chair-splatfacto-smoke/export-smoke-cuda --object-field-dir outputs/training/polyhaven-chair-splatfacto-smoke/object-field-sam --sam-manifest outputs/masks/polyhaven-chair-sam-smoke/mask-manifest.json --data-parser blender-data --iterations 100 --steps-per-save 100 --vis tensorboard --cache-images cpu --camera-res-scale-factor 0.5 --cuda-home /tmp/objgauss-cuda13 --max-jobs 2 --device cuda --sam-max-frames 8 --sam-max-masks-per-frame 6 --sam-min-area 64 --sam-max-area-fraction 0.75 --slots 6 --object-iterations 80 --skip-benchmark
node scripts/benchmark-splatfacto-scenes.mjs --run --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth
node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-scenes --skip-variants
node scripts/benchmark-splatfacto-scenes.mjs --status
node scripts/benchmark-cross-scene.mjs --status
uv run objgauss assets pull nerf-llff-fern
npm run train:splatfacto:smoke -- --run --asset-id nerf-llff-fern --dataset outputs/assets/training/nerf-llff-fern --output-root outputs/training/nerf-fern-splatfacto-smoke --experiment fern-splatfacto-smoke --timestamp smoke-cuda --export-dir outputs/training/nerf-fern-splatfacto-smoke/export-smoke-cuda --object-field-dir outputs/training/nerf-fern-splatfacto-smoke/object-field-sam --sam-manifest outputs/masks/nerf-fern-sam-smoke/mask-manifest.json --dataparser-transform outputs/training/nerf-fern-splatfacto-smoke/fern-splatfacto-smoke/splatfacto/smoke-cuda/dataparser_transforms.json --data-parser colmap --downscale-factor 1 --images-path images --colmap-path sparse/0 --iterations 100 --steps-per-save 100 --vis tensorboard --cache-images cpu --camera-res-scale-factor 0.25 --cuda-home /tmp/objgauss-cuda13 --max-jobs 2 --device cpu --sam-max-frames 4 --sam-max-masks-per-frame 6 --sam-min-area 256 --sam-max-area-fraction 0.35 --sam-max-image-size 768 --slots 6 --object-iterations 80 --skip-benchmark
node scripts/benchmark-splatfacto-scenes.mjs --run --scene fern-splatfacto-smoke --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth
node scripts/benchmark-splatfacto-scenes.mjs --run --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth
node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-scenes --skip-variants
node scripts/benchmark-splatfacto-scenes.mjs --status
node scripts/benchmark-cross-scene.mjs --status
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "asset_registry or nerf_pull or fern_pull or splatfacto_scene or cross_scene or splatfacto_variant or splatfacto_balanced or splatfacto_smoke or nerf_sam" -q
npm run benchmark:cross-scene -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth
node scripts/benchmark-cross-scene.mjs --run
node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-variants
node scripts/benchmark-cross-scene.mjs --status
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "cross_scene or splatfacto_variant or splatfacto_balanced or splatfacto_smoke" -q
npm run benchmark:splatfacto:variants -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth
node scripts/benchmark-splatfacto-variants.mjs --run --skip-sam
node scripts/benchmark-splatfacto-variants.mjs --status
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "splatfacto_variant or splatfacto_balanced or splatfacto_smoke" -q
npm run benchmark:splatfacto:balanced -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth
node scripts/benchmark-splatfacto-balanced.mjs --run
node scripts/benchmark-splatfacto-balanced.mjs --run --skip-sam
node scripts/benchmark-splatfacto-balanced.mjs --status
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "splatfacto_balanced or splatfacto_smoke" -q
uv run --extra dev pytest
npm run build
uv run --with torch --with torchvision --with "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git" objgauss masks from-nerf-sam outputs/assets/training/nerf-synthetic-lego --output outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json --checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth --model-type vit_b --device cuda --split train --max-frames 8 --max-masks-per-frame 4 --min-area 64 --max-area-fraction 0.3
uv run objgauss training register-output outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --asset-id nerf-lego-splatfacto-safe-2000-sam8f-balanced03-slots4-local --output-dir outputs/assets/gaussians/nerf-lego-trained-safe-2000-sam8f-balanced03-slots4-public --dataset outputs/assets/training/nerf-synthetic-lego --masks outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json --slots 4 --public-name nerf_lego_trained --iterations 160 --learning-rate 1.0
uv run objgauss stats public/samples/nerf_lego_trained_objects.ply
uv run objgauss object-field emergence-curve outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --field outputs/assets/gaussians/nerf-lego-trained-safe-2000-sam8f-balanced03-slots4-warmstart/object_field_initial.npz --masks outputs/masks/nerf-lego-sam-8f-balanced03-slots4/mask-manifest.json --output /tmp/objgauss-lego-splatfacto-safe-2000-sam8f-balanced03-slots4-emergence-curve.json --csv-output /tmp/objgauss-lego-splatfacto-safe-2000-sam8f-balanced03-slots4-emergence-curve.csv --iterations 80 --learning-rate 1.0 --eval-every 20 --render-size 96
uv run --extra dev pytest tests/test_objgauss_mvp.py -k "nerf_sam" -q
uv run --extra dev pytest
npm run build
env CUDA_HOME=/tmp/objgauss-cuda13 PATH=/tmp/objgauss-cuda13/bin:$PATH LD_LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LD_LIBRARY_PATH LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LIBRARY_PATH MAX_JOBS=2 uv run --offline --with nerfstudio --with torch --with torchvision --with gsplat --with nvidia-cuda-nvcc==13.0.* --with nvidia-cuda-cccl==13.0.* --with nvidia-nvvm==13.0.* --with nvidia-cuda-crt==13.0.* ns-train splatfacto --max-num-iterations 2000 --steps-per-save 500 --output-dir outputs/training/nerf-lego-splatfacto-long --experiment-name lego-splatfacto-safe --timestamp safe-2000-cpu-cache-v1 --vis tensorboard --pipeline.datamanager.cache-images cpu --pipeline.datamanager.camera-res-scale-factor 0.5 blender-data --data outputs/assets/training/nerf-synthetic-lego
env TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_HOME=/tmp/objgauss-cuda13 PATH=/tmp/objgauss-cuda13/bin:$PATH LD_LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LD_LIBRARY_PATH LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LIBRARY_PATH MAX_JOBS=2 uv run --offline --with nerfstudio --with torch --with torchvision --with gsplat --with nvidia-cuda-nvcc==13.0.* --with nvidia-cuda-cccl==13.0.* --with nvidia-nvvm==13.0.* --with nvidia-cuda-crt==13.0.* ns-export gaussian-splat --load-config outputs/training/nerf-lego-splatfacto-long/lego-splatfacto-safe/splatfacto/safe-2000-cpu-cache-v1/config.yml --output-dir outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1
uv run objgauss training register-output outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --asset-id nerf-lego-splatfacto-safe-2000-local --output-dir outputs/assets/gaussians/nerf-lego-trained-safe-2000-cpu-cache-v1-warmstart --dataset outputs/assets/training/nerf-synthetic-lego --masks outputs/masks/nerf-lego-sam/mask-manifest.json --slots 8 --public-name nerf_lego_trained --iterations 160 --learning-rate 1.0
uv run objgauss object-field emergence outputs/assets/gaussians/nerf-lego-trained-safe-2000-cpu-cache-v1-warmstart/object_field_trained.npz --cloud outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --reference outputs/assets/gaussians/nerf-lego-trained-safe-2000-cpu-cache-v1-warmstart/object_field_initial.npz --output /tmp/objgauss-lego-splatfacto-safe-2000-emergence.json
uv run objgauss object-field emergence-curve outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply --field outputs/assets/gaussians/nerf-lego-trained-safe-2000-cpu-cache-v1-warmstart/object_field_initial.npz --masks outputs/masks/nerf-lego-sam/mask-manifest.json --output /tmp/objgauss-lego-splatfacto-safe-2000-emergence-curve.json --csv-output /tmp/objgauss-lego-splatfacto-safe-2000-emergence-curve.csv --iterations 80 --learning-rate 1.0 --eval-every 20 --render-size 96
uv run objgauss object-field emergence-report /tmp/objgauss-lego-splatfacto-safe-500-emergence-curve.json /tmp/objgauss-lego-splatfacto-safe-2000-emergence-curve.json --label safe-500 --label safe-2000 --output /tmp/objgauss-lego-splatfacto-safe-500-vs-2000-report.html --title "ObjGauss NeRF Lego Splatfacto Safe 500 vs 2000"
npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5186
npm run audit:demo -- --port 5187
uv run --extra dev pytest
npm run build
uv run objgauss training register-output outputs/training/nerf-lego-splatfacto-long/export-safe-500-cpu-cache-v2/splat.ply --asset-id nerf-lego-splatfacto-safe-500-local --output-dir outputs/assets/gaussians/nerf-lego-trained-safe-500-cpu-cache-v2-warmstart --dataset outputs/assets/training/nerf-synthetic-lego --masks outputs/masks/nerf-lego-sam/mask-manifest.json --slots 8 --public-name nerf_lego_trained --iterations 160 --learning-rate 1.0
uv run objgauss stats public/samples/nerf_lego_trained_objects.ply
npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5182
npm run audit:demo -- --port 5183
uv run --extra dev pytest
npm run build
node scripts/train-splatfacto-smoke.mjs --dry-run --sam-checkpoint /tmp/sam-vit-b.pth --skip-benchmark
node scripts/train-splatfacto-smoke.mjs --status
npm run train:splatfacto:smoke -- --dry-run --sam-checkpoint /tmp/sam-vit-b.pth --skip-benchmark
uv run --extra dev pytest
uv run objgauss object-field emergence outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_sam.npz --cloud outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply --reference outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz --output /tmp/objgauss-lego-splatfacto-emergence.json
uv run objgauss object-field emergence-curve outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply --field outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz --masks outputs/masks/nerf-lego-sam/mask-manifest.json --output /tmp/objgauss-lego-splatfacto-render-emergence-curve.json --csv-output /tmp/objgauss-lego-splatfacto-render-emergence-curve.csv --iterations 80 --learning-rate 1.0 --eval-every 20 --render-size 128
uv run objgauss object-field emergence-report /tmp/objgauss-benchmark-plush-semantic.json /tmp/objgauss-benchmark-lego-alpha.json /tmp/objgauss-benchmark-lego-splatfacto.json --label plush-semantic --label lego-alpha-proxy --label lego-splatfacto-smoke --output /tmp/objgauss-emergence-benchmark-report.html --title "ObjGauss Emergence Benchmark Smoke"
uv run objgauss object-field emergence-benchmark docs/benchmarks/semantic-smoke.json --output-dir /tmp/objgauss-semantic-smoke-suite --strict
npm run acceptance:semantic
npm run acceptance:demo
npm run build
npm run audit:demo -- --url http://127.0.0.1:5197/
uv run --extra dev pytest
npm run build
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run build
npm run audit:demo -- --url http://127.0.0.1:5199/
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run build
npm run audit:demo -- --url http://127.0.0.1:5201/
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run build
npm run audit:demo -- --url http://127.0.0.1:5202/
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run build
npm run audit:demo -- --url http://127.0.0.1:5203/
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:demo -- --url http://127.0.0.1:5204/
```

结果：

- RENDER-005A implementation progress: 新增 `src/WebGpuTileViewport.jsx`，在 WebGPU route 中创建 adapter/device/context/render pipeline，将 `tileResolvedRgba` 上传为 `rgba8unorm` texture，并用 fullscreen triangle 绘制第一帧；同时保留 CPU canvas pick fallback。`editRendererContract` 在 `webgpu-device-ready + tileCapacityGate=pass` 时切到 `rendererId="webgpu-tile"` 和 `objectFilter="gpu-object-state-buffer"`；overflow 或 capability failure 继续 fallback。
- RENDER-005A validation: `npm run audit:webgpu-tile-smoke` 通过，包含 simulated WebGPU available + roomy no-overflow contract 切到 `webgpu-tile`，overflow contract blocked 于 `tile-overflow`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5206/` 三样例通过，删除后均显示 `renderModeAfterDelete="原始颜色（编辑预览）"`；定向 Playwright QA 验证删除预览 banner 的 `真实 Splat` 动作会清除编辑状态并返回 Spark renderer。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以实际 WebGPU first frame 未执行。额外 Playwright probe 加 `--enable-unsafe-webgpu` / Vulkan flags 后 Chrome 在当前容器 SIGTRAP 退出，runtime WebGPU audit pending。
- RENDER-005B storage-buffer upload contract: 新增 `src/webgpuTileStorage.js`，定义 `webgpu-tile-storage-v1`，将 `positionRadius`、`colorOpacity`、`scaleRotation`、`objectIndices`、`objectState`、`tileCounts`、`tileAccumulation`、`tileResolvedRgba` 和可选 `tileEntries` 描述为 WebGPU storage buffers；`WebGpuTileViewport` 会在 first-frame / tileSmoke update path 中创建、写入并销毁 storage buffer bundle，并暴露 `data-webgpu-storage-*`。`audit-demo` 的 WebGPU route 现在要求 storage upload 成功，并要求隔离/删除改变 storage checksum。
- RENDER-005B validation: `npm run audit:webgpu-tile-smoke` 通过，fake device 验证 9 个 storage buffers，输出 `storage=86cb35c1:9`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5207/` 三样例通过。当前本机仍 fallback，`storage=null:null` 是预期，因为没有进入 WebGPU route。
- RENDER-005C storage-buffer resolve shader: 新增 `src/webgpuTileResolveShader.js`，定义 `webgpu-storage-resolve-v1` WGSL，fragment shader 直接读取 `tileResolvedRgba` storage buffer 和 16-byte `ResolveMeta` uniform；`WebGpuTileViewport` 不再为 first frame 创建 sampled resolve texture，而是绑定 storage buffer + uniform 绘制 fullscreen triangle。`audit-demo` 的 WebGPU route 现在要求 `data-webgpu-resolve-source="webgpu-storage-resolve-v1"`。
- RENDER-005C validation: `npm run audit:webgpu-tile-smoke` 通过，验证 shader 无 `textureSample` 依赖并输出 `resolveSource=webgpu-storage-resolve-v1`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5208/` 三样例通过。当前本机仍 fallback，`resolveSource=null` 是预期，因为没有进入 WebGPU route。
- RENDER-005D compute resolve shader: 新增 `src/webgpuTileComputeShader.js`，定义 `webgpu-compute-resolve-v1` WGSL compute shader，从 `tileAccumulation` 读取 weighted OIT accumulation 并写入 `tileResolvedRgba`；`WebGpuTileViewport` 会创建 compute pipeline，在 render pass 前 dispatch compute，并暴露 `data-webgpu-compute-*`。`audit-demo` 的 WebGPU route 现在要求 compute 已 dispatch。
- RENDER-005D validation: `npm run audit:webgpu-tile-smoke` 通过，验证 compute shader contract 并输出 `compute=webgpu-compute-resolve-v1:64`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5209/` 三样例通过。当前本机仍 fallback，`compute=null:null:0` 是预期，因为没有进入 WebGPU route。
- RENDER-005E tile-center accumulation shader: `src/webgpuTileComputeShader.js` 新增 `webgpu-compute-accumulation-v1` WGSL compute shader，从 `tileEntries`、`tileCounts`、Gaussian storage buffers 和 `objectState` 读取每个 tile 的 Gaussian list，并在 GPU compute 中写入 `tileAccumulation`；`WebGpuTileViewport` 会先 dispatch accumulation pass，再 dispatch resolve pass，最后由 storage-buffer fullscreen pass 显示。WebGPU route 现在要求 storage 中包含 `tileEntries`。
- RENDER-005E validation: `npm run audit:webgpu-tile-smoke` 通过，验证 accumulation shader contract 并输出 `accumulation=webgpu-compute-accumulation-v1:64`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5212/ --no-server` 三样例通过。当前本机仍 fallback，`accumulation=null:null:0` 是预期，因为没有进入 WebGPU route。
- RENDER-005F covariance-aware tile sampling: `src/webgpuTileComputeShader.js` 将 accumulation source 升级为 `webgpu-compute-covariance-accumulation-v1`，绑定 `scaleRotation` storage buffer，并使用 Gaussian scale / rotation 在 tile 内 2x2 sample points 上计算椭圆高斯 weighted OIT contribution；`src/webgpuTileSmoke.js` 的 CPU reference 同步切到 `tile-2x2-covariance-weighted-oit`。
- RENDER-005F validation: `npm run audit:webgpu-tile-smoke` 通过，验证 covariance accumulation shader contract 并输出 `accumulation=webgpu-compute-covariance-accumulation-v1:64`；`npm run build` 通过；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5213/ --no-server` 三样例通过。当前本机仍 fallback，`accumulation=null:null:0` 是预期，因为没有进入 WebGPU route。
- RENDER-005G viewport pixel output: `src/webgpuTileComputeShader.js` 新增 `webgpu-compute-pixel-resolve-v1`，WebGPU route 现在按 `covariance accumulation -> tile resolve -> pixel resolve -> pixel-storage fullscreen resolve` 执行；`src/webgpuTileStorage.js` 新增可选 `pixelResolvedRgba` storage buffer，runtime WebGPU tile route 会包含 10 个 storage buffers。
- RENDER-005G validation: `npm run audit:webgpu-tile-smoke` 通过，输出 `storage=5561b7fd:10 pixel=webgpu-compute-pixel-resolve-v1:16384 resolveSource=webgpu-pixel-storage-resolve-v1`；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5214/ --no-server` 三样例通过。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以 browser audit 是 fallback 验收，不宣称真实 WebGPU runtime 证据。
- RENDER-005H per-pixel Gaussian accumulation: `src/webgpuTileComputeShader.js` 将 pixel stage 升级为 `webgpu-compute-pixel-accumulation-v1`，pixel shader 直接读取 Gaussian storage buffers、tile entries、object-state 和 covariance scale / rotation，在每个像素计算 Gaussian kernel weighted OIT 后写入 `pixelResolvedRgba`；`src/webgpuTileSmoke.js` 的 Node smoke reference 同步计算 direct pixel Gaussian output，浏览器 runtime 只分配 GPU 写入用 pixel buffer，避免主线程 CPU 全帧 reference。
- RENDER-005H validation: `npm run audit:webgpu-tile-smoke` 通过，输出 `storage=243af027:10 pixel=webgpu-compute-pixel-accumulation-v1:16384 resolveSource=webgpu-pixel-storage-resolve-v1`；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5215/ --no-server` 三样例通过。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以 browser audit 是 fallback 验收，不宣称真实 WebGPU runtime 证据。
- RENDER-005I compact tile list: `src/webgpuTileSmoke.js` 默认使用 `compact-offset-list` capacity strategy，新增 per-tile `tileOffsets` prefix offsets 和 compact `tileEntries`；fixed-cap layout 仍保留为 audit 对照。`src/webgpuTileComputeShader.js` 的 tile accumulation / pixel accumulation shader 改为从 `tileOffsets[tileIndex]` 读取 entry base，不再假设 `tileIndex * maxEntriesPerTile` stride。
- RENDER-005I validation: `npm run audit:webgpu-tile-smoke` 通过，输出 `storage=de5eaf8f:11 capacity=pass pixel=webgpu-compute-pixel-accumulation-v1:16384`；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5216/ --no-server` 三样例通过，Plush semantic / Plush v1 均为 `tileCapacity="compact-offset-list":"ok":0`。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以 browser audit 是 fallback 验收，不宣称真实 WebGPU runtime 证据。
- RENDER-005J storage/device-limit gate: `src/webgpuTileStorage.js` 新增 WebGPU runtime 11-buffer storage estimate；`src/webgpuCapability.js` 在 target gate 中加入 `maxBufferSize` / `maxStorageBufferBindingSize` 检查，超限时 fallback 为 `webgpu-buffer-limit`；两个 viewport 和 `audit-demo` 暴露/检查 `data-webgpu-storage-limit-*` 与 estimated storage telemetry。
- RENDER-005J validation: `npm run audit:webgpu-tile-smoke` 通过，覆盖 compact pass、fixed overflow block 和模拟小 binding 的 `webgpu-buffer-limit` block；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5217/ --no-server` 三样例通过。当前 headless Chrome 仍为 `webgpu-adapter-unavailable`，所以 storage gate 为 `unknown:webgpu-capability`；Plush estimated max buffer 为 `tileEntries:42053252`，Lego estimated max buffer 为 `pixelResolvedRgba:16777216`。
- RENDER-005K runtime audit entry: `scripts/audit-demo.mjs` 新增 `--require-webgpu` 和 `--webgpu-flags none|unsafe|vulkan`；`package.json` 新增 `npm run audit:webgpu-runtime`。常规 fallback audit 默认不加 WebGPU flags；强制 runtime audit 要求 `data-renderer="webgpu-tile"`、target gate pass、无 fallback、first frame 经过 accumulation / compute / pixel / storage resolve。
- RENDER-005K validation: `npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:demo -- --url http://127.0.0.1:5218/ --no-server` 三样例通过；`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5218/ --no-server` 单样例通过。`npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5218/ --no-server` 在当前 headless Chrome + `--webgpu-flags unsafe` 下进入 WebGPU route，`accumulation=dispatched`、`compute=dispatched`、`pixel=dispatched`，但 first frame 失败为 `webgpu-device-lost-destroyed`；因此真实 WebGPU runtime audit 仍 pending。
- RENDER-005L device-lost telemetry split: `WebGpuTileViewport` 新增 `data-webgpu-device-lost-status/reason/message`，`device.lost` 不再覆盖 `data-webgpu-first-frame-status`；`audit-demo` 先验 first-frame accumulation / compute / pixel / storage resolve，再单独以 device-lost blocker 失败。
- RENDER-005L validation: `npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --url http://127.0.0.1:5221/ --no-server` 三样例通过；`npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5221/ --no-server` 在当前 headless unsafe WebGPU 下 expected failed，并明确报告 `WebGPU device was lost after first-frame submission: reason=webgpu-device-lost-destroyed`。
- RENDER-005M validation: `npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5222/ --no-server` 单样例通过，覆盖 Spark 真实查看、Gaussian OIT 编辑、画布选择、隔离、删除预览和 `原始颜色（编辑预览）`；`npm run audit:webgpu-runtime -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5222/ --no-server` 在当前 headless unsafe WebGPU 下 expected failed，失败收敛为 `deviceError=none:: queue=failed:webgpu-queue-submitted-work-failed:A valid external Instance reference no longer exists`。最小 localhost WebGPU 空提交可稳定完成，说明这不是普通 requestDevice / empty submit 失败。本轮 `npm run audit:demo -- --url http://127.0.0.1:5222/ --no-server` 全量 3-asset audit 在 Plush/Spark 大场景的 headless SwiftShader GPU process 上长时间满载，已中止，未作为 005M 验收证据。
- RENDER-005N runtime pass probes: 新增 `src/webgpuRuntimeProbe.js`，`WebGpuTileViewport` 支持 `full`、`accumulation-only`、`resolve-only`、`pixel-output-only` runtime probes；`scripts/audit-demo.mjs` 新增 `--webgpu-probe` 和 `--allow-webgpu-device-lost`，`package.json` 新增 `npm run audit:webgpu-probe`。
- RENDER-005N validation: `git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5223/ --no-server` 单样例通过；`accumulation-only` probe queue done、device active；`resolve-only` probe queue done、device active；`pixel-output-only` probe first frame rendered 后 device lost、queue failed；strict full runtime audit 仍 expected failed with `probe=full` and `A valid external Instance reference no longer exists`。
- RENDER-005O runtime probe split: 新增 `pixel-compute-only`、`display-only`、`tiny-pixel-output` runtime probes，并让 audit 在读取 telemetry 前等待 queue 进入 done/failed 或 device lost；`tiny-pixel-output` 将 WebGPU runtime viewport 降到 32px，用于排除纯 workload size 问题。
- RENDER-005O validation: `git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5224/ --no-server` 单样例通过。`pixel-compute-only` probe 为 queue done、device active、pixel workgroups=256；`display-only` probe 没有任何 compute dispatch，但 first frame 后 device lost、queue failed；`tiny-pixel-output` probe 为 32px viewport / pixel workgroups=16，仍 device lost、queue failed；strict full runtime audit 继续 expected failed with `probe=full` and `A valid external Instance reference no longer exists`。
- RENDER-005P texture/display probes: 新增 `src/webgpuTextureResolveShader.js`、`texture-display-only`、`texture-copy-display` 和 `clear-only` runtime probes；sampled texture display 使用 CPU 生成的 `rgba8unorm` texture，copy display 使用 pixel compute 后 `copyBufferToTexture` 到 `rgba32float` texture，clear-only 只提交 canvas render pass clear、不 draw。
- RENDER-005P validation: `git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；Browser plugin absent，使用 Playwright fallback + built `dist/` static server，`npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5225/ --no-server` 单样例通过。`texture-display-only` probe 为 all compute stages skipped、resolveSource=`webgpu-sampled-texture-resolve-v1`、device lost；`texture-copy-display` 为 pixel workgroups=256、resolveSource=`webgpu-buffer-copy-texture-resolve-v1`、device lost；`clear-only` 为 all compute stages skipped、resolveSource=`webgpu-clear-pass-v1`、device lost。clear-only no-draw fix 后，`npm run audit:webgpu-probe -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5226/ --no-server --webgpu-probe clear-only` 复跑通过，`accumulation/compute/pixel=skipped`。strict full runtime audit 仍 expected failed with `probe=full` and `A valid external Instance reference no longer exists`。
- RENDER-005Q desktop audit runner: `scripts/audit-demo.mjs` 新增 `--headed` / `--browser-channel` / `--executable-path`；`scripts/audit-webgpu-desktop.mjs` 和 `npm run audit:webgpu-desktop` 将 `clear-only`、`texture-display-only`、`full` 组合成一条 RENDER-005Q 桌面 runtime audit 命令，并在 `docs/rendering/webgpu-desktop-audit.md` 记录运行方式。
- RENDER-005Q validation: `git diff --check` 通过；`node --check scripts/audit-demo.mjs` 通过；`node --check scripts/audit-webgpu-desktop.mjs` 通过；`npm run audit:webgpu-desktop -- --headless --allow-failures --port 5232` 在当前 headless unsafe WebGPU 下完整收集三项 failure，suite classification=`desktop-webgpu-presentation-backend-loss`；preview server start path fix 后，`npm run audit:webgpu-desktop -- --headless --allow-failures --probes clear-only --port 5233` 复跑通过分类收集；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5234 --allow-failures` headed 通过；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5235` strict headed 通过，classification=`desktop-webgpu-runtime-passed`，full probe 中 accumulation / compute / pixel dispatched，queue done，device active，对象选择 / 隔离 / 删除通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed。
- RENDER-005R large-scene desktop runtime audit: `npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5236` 通过，281498 Gaussians、tileReferences=724881、maxTileOccupancy=38792、storage max buffer=`positionRadius:4503968`、visibleAfterIsolate=98770、visibleAfterDelete=182728；`npm run audit:webgpu-desktop -- --asset plush-v1-closure-local --port 5237` 通过，281498 Gaussians、tileReferences=724881、visibleAfterIsolate=85041、visibleAfterDelete=196457；`npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5238` 通过，255794 Gaussians、tileReferences=436816、visibleAfterIsolate=126686、visibleAfterDelete=129108。三者均为 `desktop-webgpu-runtime-passed`，`full` probe dispatch accumulation / compute / pixel stages，queue done，device active。
- RENDER-005S visual fidelity audit: WebGPU full runtime 默认内部 viewport 从 `128x128` 提升到 `256x256`；`App` 新增 `?webgpu-viewport-size=<n>` 配置入口，`audit-demo` 支持 `OBJGAUSS_WEBGPU_VIEWPORT_SIZE` / `--webgpu-viewport-size`，`audit-webgpu-desktop` 会透传该参数；`WebGpuTileViewport` 暴露 `data-webgpu-viewport-width/height` 和 `data-webgpu-pixel-count`。验证：`node --check scripts/audit-demo.mjs` 通过；`node --check scripts/audit-webgpu-desktop.mjs` 通过；`git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5241 --probes full` 通过，`webgpuViewport=256x256:65536`、firstFrame=65536、pixel workgroups=1024；`npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5242 --probes full` 通过，281498 Gaussians、tileReferences=1458084、maxTileOccupancy=21717、visibleAfterIsolate=98770、visibleAfterDelete=182728。
- RENDER-005T-A bilinear display resolve: `src/webgpuTileResolveShader.js` 的 `webgpu-pixel-storage-resolve-v1` 从最近邻 `floor()` storage read 改为 bilinear storage sampling，并新增 `WEBGPU_TILE_RESOLVE_FILTER="bilinear-storage"`；`WebGpuTileViewport` 暴露 `data-webgpu-resolve-filter`，`audit-demo` / `audit-webgpu-tile-smoke` 检查 full storage path 的 resolve filter。验证：`node --check scripts/audit-demo.mjs` 通过；`node --check scripts/audit-webgpu-desktop.mjs` 通过；`git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过，`resolveSource=webgpu-pixel-storage-resolve-v1:bilinear-storage`；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5243 --probes full` 通过，device active、queue done、对象交互通过。
- RENDER-005T-B aspect-fit viewport: `App` 在默认 full runtime 下按实际 viewer display size 计算 area-preserving internal viewport，保留 explicit square viewport override 和 tiny probe；`WebGpuTileViewport` 用 `ResizeObserver` 暴露 display size / viewport aspect mode；`webgpuTileSmoke` 的 bounds projection 改为 `aspect-fit-padding`，按 viewport aspect 扩展短轴并加 8% 留白；`audit-demo` 检查 display size、`display-aspect-area`、`aspect-fit-padding` 和 bounds aspect consistency。验证：`node --check scripts/audit-demo.mjs` 通过；`node --check scripts/audit-webgpu-desktop.mjs` 通过；`git diff --check` 通过；`npm run audit:webgpu-tile-smoke` 通过，Node smoke 覆盖 1:1 和 2:1 viewport bounds fit；`npm run build` 通过，仍有 Spark / Three bundle size warning；`uv run --extra dev pytest` 41 passed；`npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5244 --probes full` 通过，`webgpuViewport=256x256:65536:"display-aspect-area"`、`display=784x812`、`boundsFit="aspect-fit-padding":1/1`、device active、queue done。
- RENDER-004E overflow gate / fallback hardening: fixed-capacity tile smoke 现在输出 overflow tile count、overflow ratio、max excess、stored references、entry capacity/utilization、capacity mode/status/gate；WebGPU target gate 区分 `webgpu-capability`、`tile-overflow` 和 `renderer-not-implemented`。Browser audit 不再只检查 `tileOverflowCount`，而是验证 overflow 场景被 blocked，非 overflow 场景为 pass/ok。
- RENDER-004E validation: `npm run audit:webgpu-tile-smoke` 通过，内置 sample packed=5800、refs=157323、resolved=2301、overflow=40114、overflowTiles=1056、capacity=blocked；`uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5203/` 三样例通过。Plush semantic / Plush v1 为 `tileCapacity="overflow":169`，Lego 为 `tileCapacity="ok":0`，三者 targetGate 均为 `blocked:webgpu-capability`。
- RENDER-004D object-state buffer smoke: `src/webgpuTileSmoke.js` 现在输出 `webgpu-object-state-v1`，用 stride=4 的 `vec4u`-style buffer 编码 object flags、dense object index、Gaussian count 和 reserved slot；flags 覆盖 visible、selected、removed、isolated 和 enabled，并生成 visible/hidden/removed/selected/isolated object counts 与 checksum。`PointCloudViewport` 暴露 `data-webgpu-object-state-*`，浏览器 audit 会检查隔离 / 删除后的 checksum 和计数变化。
- RENDER-004D validation: `npm run audit:webgpu-tile-smoke` 通过，内置 sample packed=5800、tiles=2362/4096、refs=157323、resolved=2301、objectState=72aeff5e；`uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5202/` 三样例通过。Plush semantic objectState 362760d7 -> 637142bc，Plush v1 e1cdb2e4 -> b0b19f1f，Lego 7243475b -> 7ca4643c。
- RENDER-004C tile resolve smoke: `src/webgpuTileSmoke.js` 现在在 16x16 tile binning 后执行 deterministic tile-center weighted OIT accumulation / resolve，输出 resolved RGBA buffer、resolved tile count、resolve weight、alpha/luma mean 和 checksum；前端状态面板与 DOM contract 暴露 `webgpu-tile-resolve-v1` / `tile-center-weighted-oit`。
- RENDER-004C validation: `npm run audit:webgpu-tile-smoke` 通过，内置 sample packed=5800、tiles=2362/4096、refs=157323、resolved=2301、checksum=c8567887；`uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5201/` 三样例通过并检查 resolve contract。Plush semantic resolvedTiles=3051 / checksum=9feb3736，Plush v1 resolvedTiles=3051 / checksum=4e86df13，Lego resolvedTiles=3881 / checksum=2b4d3d8e。
- RENDER-004B tile smoke packing: 新增 `src/webgpuTileSmoke.js`，把当前 ObjGauss scene 打包成 future WebGPU storage-buffer layout，包括 `positionRadius`、`colorOpacity`、`scaleRotation`、`objectIndices`、`objectState`、`tileCounts` 和可选 `tileEntries` typed arrays；同时生成 deterministic 16x16 tile occupancy、tile references、max occupancy 和 overflow telemetry。
- RENDER-004B validation: `npm run audit:webgpu-tile-smoke` 通过，内置 sample packed=5800、tiles=2362/4096、refs=157323；`uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5199/` 三样例通过并检查 `tileSmokeLayout="webgpu-tile-smoke-v1"`、positive pack/bin counts 和 `objectFilterTarget="gpu-object-state-buffer"`。Plush 当前 telemetry: packed=281498、activeTiles=3119/4096、tileReferences=10513313、maxTileOccupancy=11026、tileOverflowCount=196038，说明 fixed-capacity smoke path 仍需要后续 prefix-sum / overflow hardening。
- RENDER-004A renderer boundary: 前端现在检测 `navigator.gpu` / adapter / device capability，状态面板显示目标 renderer、WebGPU 状态、fallback reason 和 tile overflow；Spark 真实查看暴露 `data-renderer="spark-splat"`，编辑 fallback 暴露 `data-renderer="gaussian-oit"`、`data-renderer-target="webgpu-tile"`、`data-renderer-fallback-reason`、`data-webgpu-status` 和 `data-tile-overflow-count`。
- RENDER-004A validation: `uv run --extra dev pytest` 41 passed；`npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5197/` 三样例通过，当前 headless Chrome 为 `webgpuStatus="unavailable"`、`fallbackReason="webgpu-adapter-unavailable"`、`tileOverflowCount=0`，并继续通过画布选中、隔离和删除预览。
- RENDER-003 object-state filtering: Gaussian OIT 编辑 renderer 现在保留全量 Gaussian geometry，使用 dense object index GPU attribute + `gpu-object-state-texture` 控制对象隐藏、隔离和删除；画布拾取会跳过当前 object-state 不可见对象。
- RENDER-004/005 design: `docs/adr/0005-webgpu-tile-renderer.md` 已定义 WebGPU tile renderer 的 staged delivery、data contract、tile binning、per-tile accumulation、object-state buffer、fallback contract 和验收标准；当前下一步是诊断 `webgpu-device-lost-destroyed`，或在真实桌面 WebGPU 浏览器中重跑 runtime audit。
- RENDER-003 validation: `npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5194/` 三样例通过并检查 `objectFilter="gpu-object-state-texture"`；targeted Playwright QA 保存到 `/tmp/objgauss-gpu-filter-*.png`，验证 `initialVisible=281498 -> isolatedVisible=48066 -> deletedVisible=233432`，且无 shader/framebuffer/texture console error。
- RENDER-002 Weighted OIT: 对象编辑 renderer 现在使用 RGBA half-float accumulation render target；RGB 累加 `sum(w*c)`，Alpha 累加 `sum(w)`，fullscreen resolve 后混回基础 grid / axes 场景。Phase 3 WebGPU tile renderer 尚未完成。
- RENDER-002 validation: `npm run build` 通过；`npm run audit:demo -- --url http://127.0.0.1:5193/` 三样例通过，分别检查 `editRenderer="Gaussian OIT 编辑"`、画布点选、隔离、删除后 `renderModeAfterDelete="自身颜色"`；targeted Playwright QA 保存到 `/tmp/objgauss-oit-edit-*.png`，断言真实 Splat -> Gaussian OIT 编辑 -> 画布选中 -> 删除预览全链路，过滤已知 Spark `Worker terminate` 噪声后无 shader/framebuffer/render target console error。
- WEB-002 renderer route: ADR 0004 已接受 B -> C 渐进路线；Phase 1 将对象编辑 renderer 从 `PointsMaterial` / soft sprite 过渡到 screen-space Gaussian kernel `ShaderMaterial`，并把 Gaussian scale / opacity / rotation 传入 GPU attributes。
- WEB-001 UX repair: 顶部假工具按钮已替换为真实 `真实查看` / `对象编辑` 模式；无 `.splat` 场景默认对象编辑，带 `.splat` 样例加载后默认真实查看；删除预览会退出 `只看所选` 隔离并切回 `原始颜色（编辑预览）`，显示删除后的剩余整体场景；对象编辑 banner 可直接清除编辑状态并返回 `真实 Splat`；素材库只显示 5 个可加载样例；Benchmark tab 显示 smoke/candidate/paper pass 和 Lego/Fern/Chair 三场景指标。
- SEMANTIC-003A: render occlusion probe 已从旧的 center/depth point-splat probe 升级为 `scale_aware_cpu_splat_l1`，使用 Gaussian `scale_0/1/2` 与 opacity rasterize 小 footprint，再做 full-vs-object-removed RGBA delta；仍明确不是 covariance-aware `gsplat` training renderer。
- SEMANTIC-003B: `objgauss object-field emergence-benchmark` manifest 支持 `heldout_masks` / `heldout` 配置，summary 可记录 held-out projection loss、supervised Gaussians 和 held-out render occlusion effect；Splatfacto scene suite 现在可从 source SAM manifest split 出 train / held-out manifests，并把 held-out projection loss 与 held-out render occlusion 写入 per-scene / cross-scene summary。
- SEMANTIC-003B current rows: Lego safe-2000 split 为 train 6 frames / held-out 2 frames，held-out supervised_gaussians=459，held-out projection_loss=2.301630，held-out render=0.197505；Fern smoke split 为 train 3 frames / held-out 1 frame，held-out supervised_gaussians=1011，held-out projection_loss=0.670722，held-out render=0.233851；Chair smoke split 为 train 6 frames / held-out 2 frames，held-out supervised_gaussians=6463，held-out projection_loss=2.284750，held-out render=0.224084。
- SEMANTIC-003C: 新增 Poly Haven School Chair NeRF render set 自动素材源，使用纯 Python/NumPy glTF rasterizer 生成 16-frame NeRF-style RGBA dataset；100-step Splatfacto smoke 导出 50000 Gaussians，SAM 生成 8 frames / 48 masks，register-output 监督 10499 Gaussians，projection loss `3.330907 -> 0.774314`。
- SEMANTIC-003D/E: cross-scene summary 新增 smoke / candidate / paper gates，并生成 `/tmp/objgauss-cross-scene-benchmark/failure-report.md`。当前 smoke、candidate 和 paper gate 均通过；paper gate 证据为 `real_splatfacto_scenes=3/3`、`heldout_eval_rows=3/3`、failure report 已写出。
- SEMANTIC-003A validation: `npm run acceptance:semantic` 通过，3 个 semantic smoke scenes 均使用 `scale_aware_cpu_splat_l1`；render effect 分别为 Plush semantic 0.242028、Lego alpha proxy 0.274398、Lego Splatfacto smoke 0.137784。
- BENCH refresh with scale-aware renderer and held-out split: `node scripts/benchmark-splatfacto-scenes.mjs --run --skip-sam --sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth` 通过；Lego safe-2000 render=0.229397 / held-out render=0.197505，Fern smoke render=0.235029 / held-out render=0.233851。
- Variant refresh with scale-aware renderer: `node scripts/benchmark-splatfacto-variants.mjs --run --skip-sam` 通过；safe-2000 最佳仍为 `sam8f-slots4-balanced03`，ARI=0.468745、curve OES=0.780806、render=0.221535。
- Cross-scene refresh: `node scripts/benchmark-cross-scene.mjs --run --skip-semantic --skip-scenes --skip-variants` 通过，rows=9，heldout_eval_rows=3；best render 为 `lego-alpha-proxy/default` 0.274398；stage gates 为 smoke=true、candidate=true、paper=true。

- SEMANTIC-003C scene result: `/tmp/objgauss-splatfacto-scene-suite/summary.json` 含 3 scenes。Lego safe-2000: ARI=0.469787、curve OES=0.784051、render=0.229397、held-out render=0.197505；Fern smoke: ARI=0.790636、curve OES=0.780132、render=0.235029、held-out render=0.233851；Chair smoke: ARI=0.614363、curve OES=0.757609、render=0.248716、held-out render=0.224084。
- SEMANTIC-003C cross-scene result: `/tmp/objgauss-cross-scene-benchmark/summary.json` 从 8 rows 扩展为 9 rows，新增 `splatfacto-scenes/chair-splatfacto-smoke/default`；failure report 显示 `Overall passed: true` 和 `Paper gate passed`。
- SEMANTIC-003C validation: focused tests 5 passed；full Python suite 41 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。

- BENCH-004 real Splatfacto scene suite: 新增 `nerf-llff-fern` 自动素材源、`scripts/benchmark-splatfacto-scenes.mjs`、`npm run benchmark:splatfacto:scenes`、`docs/benchmarks/splatfacto-scenes.json` 和 `docs/benchmarks/splatfacto-scenes.md`，将真实 Splatfacto scene comparison 从 Lego 单场景推进到 Lego safe-2000 + LLFF Fern smoke 两场景。
- BENCH-004 COLMAP handoff: Fern asset pull 从 NeRF example zip 抽取 `nerf_llff_data/fern`，解析 COLMAP `cameras.bin` / `images.bin` 生成 `transforms_train.json`；Nerfstudio COLMAP dataparser 的 `dataparser_transforms.json` 已通过 `scripts/apply-mask-dataparser-transform.mjs` 乘进 mask manifest，解决 raw COLMAP camera 与导出 PLY 坐标不一致的问题。
- BENCH-004 Fern smoke: 100-step Splatfacto smoke 导出 10091 Gaussians；SAM 使用 CPU + `max_image_size=768` 生成 4 frames / 24 masks；register-output 监督 1247 Gaussians，projection loss `3.778366 -> 0.670971`。
- BENCH-004 scene result: `/tmp/objgauss-splatfacto-scene-suite/summary.json` 含 2 scenes。Lego safe-2000: ARI=0.468745、OES=0.693888、curve OES=0.775560、render=0.195308；Fern smoke: ARI=0.783070、OES=0.824959、curve OES=0.772515、render=0.193574。
- BENCH-004 cross-scene result: `/tmp/objgauss-cross-scene-benchmark/summary.json` 从 6 rows 扩展为 8 rows，新增 `splatfacto-scenes/lego-splatfacto-safe-2000/default` 与 `splatfacto-scenes/fern-splatfacto-smoke/default`；全表 best render 仍为 `lego-alpha-proxy/default` 0.236530。
- BENCH-004 validation: scene suite `--status` 与 cross-scene `--status` 均为 `status=ready missing=0`；focused tests 10 passed；full Python suite 39 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- BENCH-003 cross-scene suite: 新增 `scripts/benchmark-cross-scene.mjs` 和 `npm run benchmark:cross-scene`，聚合 semantic smoke 三场景与 safe-2000 三 mask variants 到统一 summary / CSV / Markdown / HTML 表。
- BENCH-003 runbook: 新增 `docs/benchmarks/cross-scene.md`；semantic-smoke 和 splatfacto-variants runbooks 均已链接到 cross-scene 入口。
- BENCH-003 validation: `node scripts/benchmark-cross-scene.mjs --run` 重新跑 semantic smoke suite 和 safe-2000 variant suite，生成 `/tmp/objgauss-cross-scene-benchmark/summary.json`，rows=6；`--status` 输出 `status=ready missing=0`。
- BENCH-003 result: semantic rows 为 Plush semantic、Lego alpha proxy、Lego Splatfacto smoke；safe-2000 rows 为 `sam2f-slots8`、`sam8f-slots8-unfiltered`、`sam8f-slots4-balanced03`。全表 best render 当前为 `lego-alpha-proxy/default` 0.236530；safe-2000 内最佳仍为 `sam8f-slots4-balanced03`，ARI=0.468745、OES=0.775560、render=0.195308。
- BENCH-003 tests: focused script tests 4 passed；full Python suite 36 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- BENCH-002 variant suite: 新增 `scripts/benchmark-splatfacto-variants.mjs` 和 `npm run benchmark:splatfacto:variants`，编排 `sam2f-slots8`、`sam8f-slots8-unfiltered`、`sam8f-slots4-balanced03` 三个 safe-2000 mask policy 变体。
- BENCH-002 runbook: 新增 `docs/benchmarks/splatfacto-variants.md`；BENCH-001 runbook 已链接到 variant suite。
- BENCH-002 validation: `node scripts/benchmark-splatfacto-variants.mjs --run --skip-sam` 复用已有 SAM manifests，重新登记三组 Object Field、生成三条 emergence curve、三变体 HTML report、suite summary / CSV / Markdown；`--status` 输出 `status=ready missing=0`。
- BENCH-002 result: `sam8f-slots4-balanced03` 当前最好，ARI=0.468745、OES=0.693888、render_occlusion_effect_score=0.195308；`sam2f-slots8` 为 ARI=0.388430、OES=0.671132、render=0.123359；`sam8f-slots8-unfiltered` 虽有 frames=8、masks=44、supervised_gaussians=185949，但 ARI=0.113853、OES=0.531374、render=0.108884，证明更多 unfiltered SAM masks 会引入背景/slot 噪声。
- BENCH-002 tests: focused script tests 3 passed；full Python suite 35 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- BENCH-001 reproducible benchmark: 新增 `scripts/benchmark-splatfacto-balanced.mjs` 和 `npm run benchmark:splatfacto:balanced`，支持 `--dry-run`、`--status`、`--run`、`--skip-sam` 和显式 `--publish`；默认不会覆盖 `public/samples/`。
- BENCH-001 runbook: 新增 `docs/benchmarks/splatfacto-balanced.md`，记录 safe-2000 balanced 的输入、固定参数、输出 contract、summary 字段和缺失输入处理；`docs/benchmarks/semantic-smoke.md` 指向该本地 benchmark。
- BENCH-001 validation: full run 重新生成 balanced SAM manifest 并完成 register-output、single-point emergence、emergence curve、HTML report、object PLY stats 和 summary；复用 SAM 的 `--run --skip-sam` 也通过，`--status` 输出 `status=ready missing=0`。
- BENCH-001 summary: `/tmp/objgauss-splatfacto-balanced-benchmark/summary.json` 记录 frames=8、masks=27、mask_pixels=664780、object_id_counts=126686/40747/34682/53679、stability_ari=0.468745、object_emergence_score=0.693888、render_occlusion_effect_score=0.195308，summary_status=passed。
- BENCH-001 tests: focused script tests 2 passed；full Python suite 34 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- SEG-003 SAM filter: `objgauss masks from-nerf-sam` 新增 `--max-area-fraction`，默认 1.0 保持兼容；测试覆盖过大 SAM mask 过滤。
- SEG-003 multi-view finding: unfiltered 8-frame / 8-slot SAM 提升 coverage 到 185949 Gaussians，但 slot0/1 和背景 mask 主导，effective_slots=4.191789，stability_ari=0.113853，弱于 2-frame baseline。
- SEG-003 balanced candidate: 8-frame / 4-slot / `max_area_fraction=0.3` SAM manifest 生成 27 masks、664780 mask pixels；safe-2000 登记后 `supervised_gaussians=70025`，projection loss `2.782336 -> 0.044949`，object_id counts=126686/40747/34682/53679，effective_slots=3.509020，stability_ari=0.468745，partial OES=0.693888，render_occlusion_effect_score=0.195308。
- SEG-003 public sample: 当前本机 `public/samples/nerf_lego_trained.*` 已覆盖为 safe-2000 + balanced 8-frame SAM + 4-slot Object Field；`uv run objgauss stats public/samples/nerf_lego_trained_objects.ply` 通过。
- SEG-003 browser audit: Browser MCP 未暴露可用工具，使用 Playwright fallback；Vite dev server 因系统 inotify watcher 上限 `ENOSPC` 失败后，改用 `npm run preview -- --port 5188 --strictPort` 服务静态 `dist/`，`npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5188/` 通过，splatPixels=3256，editPixels=74388，隔离后可见 126686，删除预览为 1。
- SEG-003 validation: `uv run --extra dev pytest` 33 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- TRAIN-003C resource-safe 2000 candidate: Splatfacto 2000-step 在 `vis=tensorboard`、CPU image cache、0.5 camera scale 和 `MAX_JOBS=2` 下完成；TensorBoard final train loss=0.022640，train PSNR=25.625683，gaussian_count=255795，GPU memory=941.883MB，train total time=18.331932s。
- TRAIN-003C export: `outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1/splat.ply` 导出 255794 / 255795 个 Gaussian，PLY 大小约 61MB。
- TRAIN-003C registration: `training register-output` 使用同一 2-frame SAM manifest 和 8 slots 登记 safe-2000 PLY，`supervised_gaussians=85349`，projection loss `4.467615 -> 0.288167`，public local outputs 覆盖为 `public/samples/nerf_lego_trained.splat` 和 `public/samples/nerf_lego_trained_objects.ply`。
- TRAIN-003C Object Field distribution: safe-2000 object_id counts 为 84464/64455/111/14821/27910/23159/15867/25007，assignment_confidence=0.819222，effective_slots=5.996345，spatial_compactness_score=0.980746，stability_ari=0.388430，partial OES=0.671132。
- TRAIN-003C emergence curve: projection loss `4.467615 -> 0.302584`，final render_occlusion_effect_score=0.123359；与 safe-500 的 render occlusion 同量级，说明几何密度提升没有自动解决对象语义质量。
- TRAIN-003C browser audit: `npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5186` 通过，splatPixels=3256，editPixels=74388，隔离后可见 84464，删除预览为 1；截图 `/tmp/objgauss-audit-nerf-lego-trained-output-local.png`。
- UI regression audit: `npm run audit:demo -- --port 5187` 通过 Plush semantic、Plush v1、NeRF Lego proxy 三个默认闭环样例。
- SplatViewport 修复: 真实 splat 视口 fog 已随 bounding box 自适应，避免 denser / larger Splatfacto sample 在真实 splat 模式下被固定 fog 盖成背景。
- TRAIN-003C 判断: safe-2000 是更好的几何/渲染候选，但不是最终语义样例；下一步应扩展多视角 SAM / slot balancing，而不是盲目增加训练步数。
- TRAIN-003B resource-safe candidate: Nerfstudio Splatfacto 500-step 在 `vis=tensorboard`、CPU image cache、0.5 camera scale 和 `MAX_JOBS=2` 下完成；导出 `outputs/training/nerf-lego-splatfacto-long/export-safe-500-cpu-cache-v2/splat.ply`，47168 / 50000 Gaussian 通过 opacity filter。
- TRAIN-003B public sample registration: `training register-output` 使用 2-frame SAM manifest 和 8 slots 登记 safe-500 PLY，`supervised_gaussians=7676`，projection loss `3.047123 -> 0.321066`，public local outputs 为 `public/samples/nerf_lego_trained.splat` 和 `public/samples/nerf_lego_trained_objects.ply`。
- TRAIN-003B Object Field distribution: `nerf_lego_trained_objects.ply` 含 8 个 object_id，counts=9127/5528/5661/5815/6073/3923/5995/5046，避免了登记阶段 uniform init 造成的少槽坍缩。
- TRAIN-003B browser audit: `npm run audit:demo -- --asset nerf-lego-trained-output-local --port 5182` 通过，splatPixels=408，editPixels=86577，隔离后可见 9127，删除预览为 1。
- UI regression audit: `npm run audit:demo -- --port 5183` 通过 Plush semantic、Plush v1、NeRF Lego proxy 三个默认闭环样例。
- Validation: `uv run --extra dev pytest` 32 passed；`npm run build` 通过，仍有 Spark / Three bundle size warning。
- TRAIN-003A script smoke: dry-run 输出完整 Nerfstudio Splatfacto、`ns-export gaussian-splat`、SAM manifest、Object Field init / vote-masks 和 PLY stats 命令；`--status` 在本机检查 9 项输入/输出，`status=ready missing=0`。
- Python 测试: 32 passed。
- Object Emergence smoke: assignment_confidence=0.797826，effective_slots=7.323355，spatial_compactness_score=0.968811，stability_ari=0.642209，matched_label_agreement=0.825040，partial OES=0.772490。
- Object Emergence curve smoke: 5 points，projection_loss 4.384474 -> 0.308315，assignment_confidence 0.791077 -> 0.797826，effective_slots 7.994654 -> 7.323355，ari_to_initial 1.000000 -> 0.642209，spatial_compactness_score 0.979225 -> 0.968811，mask_proxy_occlusion_mean_delta_loss 1.428752 -> 1.927487；当前 scale-aware CPU splat probe 在 semantic smoke acceptance 中 Lego Splatfacto smoke render_occlusion_effect_score=0.137784。
- Object Emergence benchmark report smoke: Plush semantic、Lego alpha proxy、Lego Splatfacto smoke 三条本地曲线聚合为 `/tmp/objgauss-emergence-benchmark-report.html`，curves=3，charts=7；最终 render_occlusion_effect_score 分别为 0.227482、0.236530、0.124240。
- Object Emergence benchmark suite smoke: `docs/benchmarks/semantic-smoke.json` 严格模式通过，输出 `/tmp/objgauss-semantic-smoke-suite/summary.json` 和 `report.html`；3 scenes 全部 passed，projection loss 分别为 1.386294 -> 1.346402、1.386294 -> 0.235765、4.384474 -> 0.339695。
- Semantic benchmark acceptance: `npm run acceptance:semantic` 通过，输出 `acceptance_semantic_benchmark=passed`。
- Full demo acceptance: `npm run acceptance:demo` 通过，已在闭环 demo 生成、浏览器 audit 后执行 SEMANTIC benchmark suite，输出 `acceptance_demo=passed`。
- 前端构建: 通过，仍有 bundle size warning。

2026-06-22:

```bash
uv run --extra dev pytest
uv run objgauss demo audit-v1-goal
npm run build
npm run acceptance:demo
```

结果：

- Python 测试: 24 passed。
- 前端构建: 通过。
- 浏览器验证: 桌面 1440x920 与移动端 390x844 均渲染非空、无前端错误。
- ASSET-001: Poly Haven School Chair 实际拉取 5 个文件；NeRF Synthetic Lego 实际抽取 805 个文件。
- OBJFIELD-001: Plush PLY 可初始化 6-slot Object Field；NeRF Lego 检查 400 frames、缺图 0、无效 pose 0。
- SEG-001 / OBJFIELD-002: synthetic projection mask vote 可训练 Object Field，并输出 `object_id` PLY。
- DEMO-001: Plush v1 闭环 demo 生成 281498 个 Gaussian、6 个对象、3 个投影视角、18 个 masks；projection loss 1.791760 -> 1.201637；浏览器验证可加载 `ObjGauss v1 闭环样例` 并执行对象选择、隔离、删除预览。
- VERIFY-001: `objgauss demo verify-v1-closure` 通过，检查真实 splat、mask manifest、Object Field shape、loss 下降、`object_id` PLY、public copy 和前端素材注册。
- MASK-001: NeRF Lego 真实 RGBA 图片 alpha 生成 mask manifest，8 frames / 8 masks / 800x800 / 299242 foreground pixels。
- LEGO-001: `objgauss demo lego-alpha-closure --max-frames 12 --sample-stride 8 --iterations 120` 生成 5696 个 Gaussian proxy、4 个对象、12 个真实视角、48 个 2D color masks；projection loss 1.386294 -> 0.538856；浏览器验证可加载 `NeRF Lego 闭环代理样例` 并执行对象选择、隔离、删除预览。
- VERIFY-002: `objgauss demo verify-lego-alpha-closure` 通过，17 项检查全部通过，包括源图像和 mask 文件存在、Object Field shape、loss 下降、`object_id` PLY、public assets 和前端素材注册。
- UI-AUDIT-001: `npm run audit:demo` 通过，加载 `Plush 2D 语义 Mask 闭环样例`、`ObjGauss v1 闭环样例` 与 `NeRF Lego 闭环代理样例`，检查 splat / 点云编辑 canvas 非空，并执行对象选择、只看所选、预览删除。
- ACCEPT-001: `npm run acceptance:demo` 通过，重新生成并验证 Plush v1 closure、Plush semantic closure、NeRF Lego proxy closure，然后执行浏览器闭环验收；输出 `acceptance_demo=passed`。
- MASK-002: `objgauss masks from-nerf-rgba-colors` 在 NeRF Lego 真实 RGBA 上生成 8 frames / 32 masks / 4 slots；独立 `vote-masks` 消费该 manifest，3423 个 Gaussian 被监督，projection loss 1.386294 -> 0.390825，并输出 `object_id` PLY。
- TRAIN-002: `objgauss training register-output` 接入 Gaussian PLY smoke 通过，生成 viewer splat、Object Field 和 `object_id` PLY；使用真实 Lego color mask manifest 时 supervised_gaussians=4806，projection loss 1.386294 -> 0.375765。
- SEG-002A: `objgauss masks from-nerf-sam --help` 可用；SAM manifest 生成逻辑由 fake generator 测试覆盖，输出 `sam-automatic-mask-generator` manifest 和 boolean `.npy` masks。
- VERIFY-003: `npm run acceptance:demo` 已检查 `mask_guidance_changed_object_field`；Plush changed_gaussians=196457，Lego proxy changed_gaussians=4960，证明 mask supervision 实际改变 Object Field labels。
- SEMANTIC-001: `objgauss demo plush-semantic-closure` 在真实 Plush 3DGS 上生成 3 views / 12 masks / 4 objects；281498 个 Gaussian 全部被监督，104403 个 hard labels 被 2D mask guidance 改变，projection loss 1.386294 -> 1.345684。
- AUDIT-001: `objgauss demo audit-v1-goal` 严格模式通过，当前证据为 unified，completion_blockers=`-`。
- VERIFY-004: `objgauss object-field vote-masks` summary、闭环 demo manifest 和 verifier 已包含 mask vote quality audit；本地测试覆盖 per-slot coverage、conflict fraction、normalized target entropy 和 verifier 检查。
- SEG-002: 真实 SAM checkpoint 小场景验收通过；`from-nerf-sam` 在 NeRF Lego 2 帧上生成 8 个 SAM masks，`vote-masks` 监督 5567 / 5696 个 Gaussian，supervised_fraction=0.977353，vote_conflict_fraction=0.064308，projection loss 3.902681 -> 0.120758，并输出带 `object_id` 的 PLY。
- TRAIN-001: Nerfstudio Splatfacto smoke 训练通过。`ns-train splatfacto ... blender-data --data outputs/assets/training/nerf-synthetic-lego` 完成 100 iterations，checkpoint 为 `outputs/training/nerf-lego-splatfacto-smoke/lego-splatfacto-smoke/splatfacto/smoke-cuda/nerfstudio_models/step-000000099.ckpt`；`ns-export gaussian-splat` 导出 `outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply`，ObjGauss 读取为 50000 gaussians。
- TRAIN-001 环境结论: 当前 RTX 5060 Ti / PyTorch `2.12.1+cu130` / CUDA 13.0 环境需要为 `gsplat` JIT 显式加入 `nvidia-cuda-nvcc==13.0.*`、`nvidia-cuda-cccl==13.0.*`、`nvidia-nvvm==13.0.*`、`nvidia-cuda-crt==13.0.*`；未对齐时会出现 no `nvcc`、CUDA 13.3 header/compiler mismatch、PTX version mismatch 或 `-lcudart` 链接失败。导出本地可信 checkpoint 时需设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` 兼容 PyTorch 2.6+ 的 `torch.load` 默认行为。
- TRAIN-001 Object Field smoke: 对导出的 `splat.ply` 执行 8-slot init 和 SAM mask vote，`supervised_gaussians=8887 / 50000`，`supervised_fraction=0.177740`，`vote_conflict_fraction=0.268707`，projection loss `4.384474 -> 0.308315`，最终 `outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/lego_splatfacto_sam_objects.ply` 含 `object_id` 和 RGB 字段。
- SEMANTIC-002: `objgauss object-field emergence` 已提供 object emergence 观测指标。Synthetic 测试覆盖 assignment confidence、effective slots、spatial compactness、permutation-invariant ARI 和 partial OES；在 NeRF Lego Splatfacto smoke 上输出 assignment_confidence=0.797826，effective_slots=7.323355，spatial_compactness_score=0.968811，stability_ari=0.642209，matched_label_agreement=0.825040，partial OES=0.772490。
- SEMANTIC-003: `objgauss object-field emergence-curve` 已提供随 mask-vote training iteration 变化的 benchmark 曲线，输出 JSON 和 CSV。
- SEMANTIC-004: `emergence-curve` 已新增 scale-aware CPU splat render occlusion delta；默认从 mask manifest 的相机位姿做 CPU 重渲染 probe，输出 `render_occlusion_delta`、CSV render 列、target/non-target/locality 字段和曲线内 occlusion-effect OES component。当前 probe 仍不是 covariance-aware 3DGS / gsplat renderer。
- SEMANTIC-005: `objgauss object-field emergence-report` 已可将多个 curve JSON 聚合为 HTML/SVG 报告；本地 smoke 已覆盖 Plush semantic、Lego alpha proxy 和 Lego Splatfacto smoke 三个场景曲线。
- SEMANTIC-006: `objgauss object-field emergence-benchmark` 已可从 `docs/benchmarks/semantic-smoke.json` 一键重跑 3-scene semantic smoke suite，生成 per-scene curve JSON/CSV、summary JSON、HTML report，并在 `--strict` 下执行阈值检查。
- SEMANTIC-007: `npm run acceptance:semantic` 已作为独立 benchmark acceptance；`npm run acceptance:demo` 默认纳入 SEMANTIC benchmark suite，并提供 `--skip-semantic-benchmark` 保留 demo-only 验收。`docs/benchmarks/semantic-smoke.md` 记录缺失 `outputs/` 时的生成命令和 Splatfacto smoke 边界。
- TRAIN-003A: `npm run train:splatfacto:smoke` 已将 NeRF Lego Splatfacto 100-step smoke 的生成过程固化为 dry-run/status/run 三模式脚本；`docs/training/splatfacto-smoke.md` 记录 CUDA / `gsplat` 环境、SAM checkpoint、输出 contract 和验证命令。
- 已知提示: Vite 报 Spark / Three.js chunk 超过 500KB，不影响当前预览。

## 当前限制

- 对象聚类色、隐藏、隔离、删除预览当前仍通过 `Gaussian OIT 编辑` fallback 或 WebGPU tile route 完成，不是 Spark / gsplat 真实 renderer 内的对象级重渲染；`原始颜色（编辑预览）` 只使用 PLY RGB / SH DC 颜色和近似 screen-space Gaussian kernel。WebGPU full runtime 内部输出已从 128px 提到 256px，fullscreen display 已从最近邻放大改为 bilinear storage resolve，并已加入 display-aspect viewport / aspect-fit bounds / 8% 留白，因此颗粒、格子、贴边和非等比拉伸有所降低；剩余视觉差距主要表现为过度平滑、正交简化投影、真实 camera/depth/alpha-order 不匹配、近似 covariance / blending 和未接入 Spark 的真实对象级重渲染，而不是 Object Field 颜色本身。当前 headless unsafe WebGPU failure 已归类为 canvas render pass / presentation backend limitation；headed desktop Chrome/WebGPU 已通过 NeRF Lego proxy、Plush 和 safe-2000 Splatfacto 的 full WebGPU tile runtime audit。
- `plush-semantic-closure` 已证明真实 3DGS + 非 KMeans 2D color masks + Object Field + 前端对象编辑的统一闭环；但它仍是确定性颜色规则，不等价于 SAM / CLIP 实例语义分割。
- 当前 v1 闭环 demo 的 Plush mask manifest 由已有对象标签派生，用于回归验收；NeRF Lego alpha/color masks 已能从真实图片生成，但仍是确定性 alpha/颜色规则，不等价于 SAM / CLIP 实例语义分割。
- SAM 入口已用真实 checkpoint 跑通小场景 manifest 和 `vote-masks` 验收；仓库内还不运行 CLIP 模型，也未做跨视角 SAM slot 对齐或语义命名。
- Object Emergence Score 的单点 `emergence` CLI 仍是 partial OES；`emergence-curve` 在提供 cloud 和 mask manifest 时已覆盖 assignment / stability / spatial compactness / scale-aware CPU splat render occlusion。`emergence-benchmark` 当前是本地 smoke suite，依赖 ignored `outputs/` 产物；缺失输入时按 `docs/benchmarks/semantic-smoke.md` 与 `docs/benchmarks/splatfacto-scenes.md` 生成。本 suite 仍不是 CI 固定 public benchmark。gradient coherence 和 covariance-aware 3DGS renderer occlusion 仍未实现，不能据此单独宣称 object emergence 完成。
- 当前训练循环是 projection supervision，不是完整 3DGS render loss 联合训练。
- NeRF Lego 闭环代理样例仍是 posed RGBA 生成的轻量 Gaussian proxy；另有 Nerfstudio Splatfacto 100-step smoke 产物和 TRAIN-003A runbook/script 证明本机可复现真实 3DGS optimization PLY，但尚未作为前端公开样例固化。
- 外部训练输出接入命令已完成，本机已产出真实 NeRF Lego Splatfacto smoke PLY、500-step resource-safe public sample candidate 和 2000-step higher-quality geometry candidate；safe-2000 经过 8-frame balanced SAM 后已消除近空 object slots、提升 render occlusion effect，并通过当前 public sample 浏览器 audit。
- Poly Haven mesh Demo 还不能直接进入现有 3DGS viewer；当前已具备 mesh -> NeRF-style render set -> Splatfacto smoke 的 benchmark 链路，但不是公开前端 demo。
- 训练素材目录已接入 NeRF Lego、LLFF Fern 与 Poly Haven Chair NeRF render set；Fern 和 Chair 当前只是 100-step smoke，不代表高质量 reconstruction。

## 下一步主线

1. RENDER-005T-P: 继续拆 Spark vs edit 残差中的 sorted alpha、SH 颜色和真实 camera 对齐问题；默认 coverage 参数变更必须先通过 `audit:webgpu-coverage-gate`。
2. 将三场景 Splatfacto suite 从 smoke 推进到更高质量训练：统一训练步数、质量曲线、held-out view 指标和失败案例分析。
3. 后续 SEG: CLIP 语义命名、跨视角 SAM slot 对齐，以及与 color-mask / KMeans baseline 的质量对比。
4. 将 Poly Haven mesh -> NeRF-style render set -> Splatfacto smoke 链路升级为可审计的公开 demo 候选前，先补许可说明、质量阈值和浏览器验收。
5. 后续 renderer 优化: Spark 按需加载或拆包，降低首屏 bundle。
