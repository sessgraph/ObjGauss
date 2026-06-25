# ObjGauss 当前状态总览

> 最近更新: 2026-06-25

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
  - RENDER-005T-Q 已完成 WebGPU depth-bin sweep：新增 `npm run audit:webgpu-depth-sweep`，可对 `4/8/12/16` bins 跑同一套 headed desktop WebGPU full-runtime visual residual audit，并输出 `summary.json` / `summary.md`；Lego sweep 显示 8 bins 仍是 best Pareto，12 bins 只让 coverage ratio 从 `3.784251` 微降到 `3.784235` 但 chroma 变差，因此单纯提高 bin 数不是当前 Spark/edit 残差主因。
  - RENDER-005T-R 已完成 WebGPU runtime camera framing diagnostic：新增 `runtime-camera-tuning-v1` 和 URL / audit 参数 `webgpu-camera-mode=edit-fixed|spark-frame`，默认仍保持 `edit-fixed`；`spark-frame` 对齐 Spark viewer 的 58° FOV、scene-center target 和 `distance=maxDim*1.7` framing。Lego `spark-frame` headed audit 将 coverage ratio 从 `3.784251` 降到 `3.766657`、luma 从 `0.106079` 降到 `0.102396` 但 chroma 从 `0.086537` 略升到 `0.087290`；Plush `spark-frame` 大场景 audit 也通过并把 coverage ratio 降到 `4.713926`，但 luma/chroma 未同步改善。因此 camera alignment 是 coverage 残差项之一，不是“原始颜色编辑预览不像 Spark”的完整主因。
  - RENDER-005T-S 已完成 WebGPU front-top-k sorted-alpha diagnostic：`runtime-depth-sort-tuning-v1` 新增 URL / audit 参数 `webgpu-depth-alpha-mode=depth-binned|front-top-k`，默认仍保持 `depth-binned`；`front-top-k` 每像素保留最近 K 个 contributor 并前到后 alpha composite，K 复用 `webgpu-depth-bins`。Lego K=8 将 coverage ratio 从 `3.784251` 降到 `3.583371`，但 luma/chroma 恶化到 `0.208595/0.127958`；K=16 coverage 接近 baseline 但 luma/chroma 仍弱于 baseline；Plush K=8 大场景也通过但 luma/chroma 明显变差。因此 per-pixel sorted-alpha 诊断路径可运行，但当前不是默认候选。
  - RENDER-005T-T 已完成 WebGPU SH-rest presence audit：前端 PLY parser 记录 `f_rest_*` 系数数量与推断 SH degree，WebGPU Tile / Gaussian OIT fallback contract 暴露 `shRest=count/maxCoeffs/maxDegree`，browser audit 验证该 telemetry 合法且不改变默认 RGB/SH-DC 渲染路径；本机 `NeRF Lego 训练输出样例` headed WebGPU full audit 显示 `shRest=255794/45/3`，删除预览后仍回到 `colorAfterDelete=255794/0/0/0` RGB 原始色，证明 trained sample 存在完整 degree-3 view-dependent SH 但当前编辑预览尚未利用。
  - RENDER-005T-U 已完成 WebGPU SH-view color diagnostic：前端 PLY parser 在保留 RGB 原始色的同时保存 raw `f_dc`，并把 `f_rest_*` 系数打包为 typed array；WebGPU Tile 新增 URL / audit 参数 `webgpu-color-mode=source|sh-view`，默认仍是 `source`。`sh-view` 按 edit camera 方向评估 degree-3 SH，仅在“原始颜色（编辑预览）”路径生效，不影响对象调试色；headed WebGPU audit 证明本机 trained Lego 删除预览后 `shViewAfterDelete=255794`，luma/chroma residual 从 source 的 `0.090165/0.071164` 降到 `0.034507/0.055774`，但 coverage ratio 仍约 `31x`，说明“自身颜色不像高斯”的颜色主因已被确认并部分缓解，剩余颗粒/膨胀主要来自编辑 renderer 的 footprint / alpha / presentation 与 Spark 真实 splat 合成差距。
  - RENDER-005T-V 已完成 WebGPU SH-view coverage sweep：`audit:webgpu-coverage-sweep` 支持 `--webgpu-color-mode source|sh-view` 并在 report 中记录 `colorMode` / `shViewAfterDelete`，可以在 view-dependent color 生效后继续比较 footprint / anisotropy variants；trained Lego SH-view sweep 显示 baseline 仍是 best Pareto，tight 可将 coverage ratio 从 `31.205176` 降到 `23.164633` 并降低 tile refs，但 luma delta 从 `0.034507` 恶化到 `0.093626`。因此 footprint 收紧只能作为诊断，不能作为默认修复；下一步应优先看 presentation coverage threshold / alpha path 或 Spark object filter feasibility。
  - RENDER-005T-W 已完成 WebGPU alpha presentation floor diagnostic：`webgpu-alpha-presentation-floor` 可在 `0-0.2` 范围内 runtime tuning，默认仍保持 `0.035`；coverage sweep variants 支持 `id:footprint:maxAnisotropy:alphaFloor`。trained Lego + SH-view alpha sweep 显示 `0.1` floor 将 coverage ratio 从 `31.205176` 降到 `24.248059`，luma delta 从 `0.034507` 降到 `0.00276`，chroma 基本持平，因此 presentation threshold 是比 footprint tightening 更强的候选轴；但目前仍是单 scene 证据，不能默认切换，下一步需要多场景 alpha-floor gate 或 Spark renderer object filter feasibility。
  - RENDER-005T-X 已完成 WebGPU alpha presentation floor multi-scene candidate gate：新增 `npm run audit:webgpu-alpha-floor-sweep` 和 `npm run audit:webgpu-alpha-floor-candidate-gate`，在 NeRF Lego proxy + Plush semantic 上复现 `alpha10` 候选。`alpha10` 是 best mean Pareto (`0.965287`) 且同时降低两个 scene 的 coverage / luma，但 strict gate 失败：mean chroma norm=`1.178616`、Plush per-scene Pareto=`1.07908`、Plush chroma norm=`1.485213`。因此 alpha floor 仍是候选轴，默认 `0.035` 不变。
  - RENDER-005T-Y 已完成 Spark filtered edit feasibility implementation：`SplatViewport` 可从 object-aware PLY points 重建 filtered Spark `SplatMesh`；`App` 在 source/original object edit 状态下优先用 `Spark 过滤 Splat` 显示隔离/删除后的剩余场景，SH-view diagnostics 与对象色/点击选择仍走 WebGPU / Gaussian OIT 编辑路径；`audit-demo` 已能验证删除后 `postDelete="spark-splat":"spark-filtered-ply-reconstruct"`。
  - RENDER-005T-Z 已完成 Spark PLY reconstruction residual gate：新增 URL probe `spark-reconstruct-probe=1` 和 `npm run audit:spark-reconstruct-residual`，可直接比较 full `.splat` Spark 与 object-aware PLY reconstructed Spark 的 coverage / luma / chroma；Lego proxy 默认 gate 通过 (`coverageRatio=1.170841`, `lumaDelta=0.029762`, `chromaDelta=0.028407`)，Plush semantic 可选 multiscene 复查也通过 (`coverageRatio=1.303149`, `lumaDelta=0.049406`, `chromaDelta=0.002846`)，但 281k Gaussian 重建耗时明显，仍需 SH-rest preservation 与性能治理。
  - RENDER-ROUTE-003 已新增 WebGPU 100k-1M scale budget audit：`npm run audit:webgpu-scale-budget` 用 synthetic 100k / 300k / 1M `tileSmoke` 估算完整 11-buffer runtime storage，默认在 128 MiB 单 buffer / 256 MiB 总 storage 预算下通过；该 gate 已纳入 `acceptance:renderer-ci`，证明 C-path storage budget 形态，不证明真实 1M FPS。
  - RENDER-ROUTE-004 已新增 WebGPU object-state-filtered tile list mode：WebGPU runtime tile smoke 可保持全量 compact tile list，由 object-state buffer 控制隐藏 / 隔离 / 删除；`audit:webgpu-tile-smoke` 证明编辑前后 tileCounts / tileOffsets / tileEntries checksum 稳定，同时 objectState / resolve checksum 变化。
  - RENDER-ROUTE-005 已新增 WebGPU objectState-only incremental upload：当 tile/storage reuse signature 兼容时，WebGPU runtime 会复用既有 11-buffer storage bundle，只对 `objectState` buffer 执行 `queue.writeBuffer`，再重新 dispatch compute；静态输入变化时仍安全重建完整 bundle。
  - RENDER-ROUTE-006 已新增 WebGPU object-state update telemetry gate：WebGPU viewport 暴露 `data-webgpu-storage-update-mode` 与 `data-webgpu-storage-object-state-byte-size`；browser audit 在 WebGPU isolate / delete transition 后要求 `object-state-only` 增量更新。
  - RENDER-ROUTE-007 已新增 WebGPU edit cost budget audit：`npm run audit:webgpu-edit-cost-budget` 用 100k / 300k / 1M synthetic C-path profiles 检查 objectState-only edit upload、workgroups 和 tile-entry candidate scan 上界；该 gate 已纳入 `acceptance:renderer-ci`，证明编辑更新成本形态，不证明真实 FPS。
  - RENDER-ROUTE-008 已新增 WebGPU runtime timing telemetry gate：WebGPU viewport 暴露 storage update、queue submit 和 queue done 毫秒级 DOM telemetry；browser audit 在 WebGPU object isolate / delete transition 后等待 storage checksum 变化并检查 timing 合法。Isolate 要求 objectState-only；delete 允许 source/object color buffer 改变时的 full-upload fallback。
  - RENDER-ROUTE-009 已新增 WebGPU runtime performance smoke gate：`npm run audit:webgpu-runtime-performance` 复用 offscreen object-transition suite，聚合 Lego proxy 与 Plush 281k 场景的 browser storage update / submit / queue done timing，并按当前 smoke envelope 阻断明显回归。该 gate 仍不是 FPS 或 1M interactive SLA。
  - RENDER-ROUTE-010 已新增 WebGPU headed presentation performance smoke gate：`npm run audit:webgpu-presentation-performance` 在真实 headed browser 中强制 WebGPU full canvas presentation，默认覆盖 Lego proxy 与 Plush semantic 大场景，检查 first-frame pixels、device / queue 状态、pixel dispatch、storage timing、tile overflow 和截图产物。该 gate 证明当前 presentation path 在 smoke envelope 内，不证明 FPS 或 1M interactive SLA。
  - RENDER-ROUTE-011 已将 WebGPU presentation smoke 纳入 product renderer acceptance：`npm run acceptance:renderer-product` 会在 Spark commercial route 前显式验收 headed WebGPU presentation performance；CI profile 仍不跑 headed presentation，继续保持 fresh-clone/headless 友好。
  - RENDER-ROUTE-012 已新增 headed WebGPU presentation object-transition gate：`npm run audit:webgpu-presentation-transition` 强制 full canvas WebGPU C-path 执行选中、隔离、删除，检查 object-state checksum、storage timing、post-delete renderer 和截图；product renderer acceptance 已纳入该 gate。
  - RENDER-ROUTE-013 已新增 WebGPU C-path readiness audit 初版：`npm run audit:webgpu-cpath-readiness` 原始版本串起 build、1M scale budget、1M edit-cost budget 和 fixed-port headed presentation transition，输出 combined evidence / remaining gap report。初版本机通过：1M scale budget max/total=`122.07/173.24 MiB`，1M edit upload=`4 KiB`，headed browser 最大场景=`281498` Gaussians；当时报告仍保留 `browserRuntime1m=not-proven` 与 `fpsSla=not-proven`。
  - RENDER-ROUTE-014 已新增 headed WebGPU frame pacing smoke：`npm run audit:webgpu-frame-pacing` 强制 WebGPU Tile C-path，在 idle / isolate / delete 三段采样 `requestAnimationFrame` 间隔。当前本机覆盖 Lego proxy 与 Plush semantic，Plush `281498` Gaussians 通过，min approx FPS=`26.471`、max mean frame=`37.777ms`、max p95 frame=`16.8ms`、max long-frame ratio=`0.013`；该 gate 证明当前真实场景浏览器响应 smoke，不证明持续 FPS SLA 或真实 1M runtime。
  - RENDER-ROUTE-015 已新增 synthetic 1M headed browser runtime audit：`npm run audit:webgpu-synthetic-1m-runtime` 会在 `/tmp` 生成 binary PLY，通过真实文件上传控件加载 `1,000,000` Gaussians，并在 WebGPU Tile C-path 执行 first-frame、选中、隔离、删除和 rAF frame pacing 检查。当前本机完整 readiness 通过：tile refs=`1,709,862`、tile overflow=`0`、upload wall=`3150.075ms`、isolate object-state update=`513.9ms`、delete full-upload update=`551.5ms`、min approx FPS=`15.429`。`audit:webgpu-cpath-readiness` 已纳入该 gate；剩余 gap 是真实训练 1M scene runtime、持续 FPS SLA 和视觉质量。
  - RENDER-ROUTE-016 已新增 WebGPU sustained frame pacing baseline：`npm run audit:webgpu-sustained-frame-pacing` 会用 `120` rAF samples per phase 复跑当前真实场景 frame pacing 和 synthetic 1M upload/runtime。当前本机通过：真实场景最大 `281498` Gaussians，min approx FPS=`33.964`、max mean frame=`29.443ms`、max p95=`16.8ms`、long-frame ratio=`0.008`；synthetic 1M min approx FPS=`28.917`、max mean frame=`34.582ms`、max p95=`16.8ms`、long-frame ratio=`0.008`。这把 sustained frame pacing 从 smoke 推进为 baseline evidence；生产 FPS SLA 和真实训练 1M scene 仍未完成。
  - RENDER-ROUTE-017 已新增 real/trained PLY WebGPU runtime gate：`npm run audit:webgpu-ply-runtime -- --input-ply <path> --scene-kind trained --min-gaussians <n>` 可上传任意 object-aware PLY，通过真实文件输入进入 WebGPU Tile C-path，并检查 first-frame、对象选择、隔离、删除、tile overflow、更新模式和 rAF pacing。当前本机 trained Lego 样例通过：`255794` Gaussians、tile refs=`581933`、min approx FPS=`30.51`、max mean frame=`32.775ms`、upload wall=`1777.187ms`、isolate object-state update=`143.4ms`、delete object-state update=`142.8ms`。这证明真实/训练 PLY 验收入口可用；near-1M trained scene 仍需实际 PLY 产物后用 `--min-gaussians 1000000` 复跑。
  - RENDER-ROUTE-018 已将 trained PLY runtime 证据接入 WebGPU C-path readiness 聚合器：`npm run audit:webgpu-cpath-readiness -- --trained-ply <path> --trained-min-gaussians <n>` 现在会调用 `audit:webgpu-ply-runtime`，在同一份 readiness report 中输出 `trainedPlyRuntime` 和 `realTrainedBrowserRuntime1m`。当前本机可选路径用 `public/samples/nerf_lego_trained_objects.ply` / `--trained-min-gaussians 250000` 通过：`255794` Gaussians、tile refs=`581933`、min approx FPS=`26.215`、upload wall=`1738.952ms`、isolate update=`142.3ms`、delete update=`141.6ms`。结论仍是 `realTrainedBrowserRuntime1m=not-proven`；最终 gap 是拿到 near-1M trained object-aware PLY 后用 `--trained-min-gaussians 1000000` 复跑。
  - RENDER-ROUTE-019 已将 sustained frame-pacing baseline 证据接入 WebGPU C-path readiness 聚合器：`npm run audit:webgpu-cpath-readiness -- --include-sustained-frame-pacing` 现在会调用 `audit:webgpu-sustained-frame-pacing`，或通过 `--sustained-frame-pacing-summary <summary.json>` 读取已有 baseline，并在 readiness report 中输出 `sustainedFramePacing`。本次短采样聚合验证通过：real scenes min approx FPS=`5.405`、synthetic 1M min approx FPS=`4.138`。这把“当前真实场景 + synthetic 1M 长采样”纳入同一份 C-path readiness 判断；`fpsSla` 仍保持 `not-proven`，直到在目标硬件和真实训练 1M scene 上审定阈值。
  - RENDER-ROUTE-020 已将 trained PLY 接入 sustained frame-pacing baseline：`npm run audit:webgpu-sustained-frame-pacing -- --trained-ply <path> --trained-min-gaussians <n>` 现在会在 current real scenes 与 synthetic 1M 之外追加 `trainedPly` 长采样行；`audit:webgpu-cpath-readiness -- --include-sustained-frame-pacing --trained-ply <path>` 会把同一个 trained PLY 透传给 sustained baseline，并在 readiness terminal summary 中输出 `sustainedTrainedPly` / `sustainedTrainedMinApproxFps`。本次短采样 trained Lego PLY 通过：`255794` Gaussians、`trainedMinApproxFps=11.322`，readiness 同时保持 `realTrainedBrowserRuntime1m=not-proven` 与 `fpsSla=not-proven`。这把未来 near-1M trained scene 的“真实训练输入 + 长采样响应性”验收入口合并起来；当前仍未证明 near-1M trained scene 或生产 FPS SLA。
  - RENDER-ROUTE-021 已给 C-path readiness 增加 reviewed FPS SLA promotion contract：`fpsSla` 不再只是硬编码文字，而是只有在显式 `--fps-sla-reviewed`、提供 `--fps-sla-target-hardware`、真实 trained PLY 达到 1M browser runtime proof、sustained baseline 中 trained PLY 行通过、且 `--fps-sla-min-trained-approx-fps` 达标时才会变成 `passed`。当前 255k trained Lego summary 仍输出 `fpsSla=not-proven`，并列出 blockers：缺少 reviewed flag / target hardware、不是 real trained 1M proof、Gaussian 数不足 1M、trained sustained min FPS 低于默认 `24`。
  - RENDER-ROUTE-022 已新增严格 production SLA wrapper：`npm run audit:webgpu-cpath-production-sla -- --trained-ply <near-1m-trained-objects.ply> --target-hardware <label>` 会先读取 PLY header，拒绝低于 `1,000,000` Gaussians 的 trained PLY 和 summary shortcut，然后强制运行完整 C-path readiness、sustained trained PLY row 和 reviewed FPS SLA gate。当前 255k trained Lego dry-run 被正确挡在 preflight：`trainedGaussians=255794 < 1000000`、`readiness=not-run`、`fpsSla=not-run`。这把最终 SLA 证明入口固化了；真实 near-1M trained PLY 和目标硬件生产 FPS 仍是剩余 gap。
  - RENDER-005T-AA 已完成 Spark packed extract reconstruction route：filtered Spark 路径从 raw PLY `constructSplats` 推进到 base `PackedSplats` cache + visible-index `extractSplats`，浏览器 contract 暴露 `data-spark-reconstruct-source="packed-extract-v1"`、base / visible counts、build / extract timing 和 `data-spark-sh-rest-preserved="false"`；Lego delete preview 通过 `sparkPacked="packed-extract-v1":5696/3909:3.9/1.9` 验收。该路径仍不是原始 `.splat` 内部 object mask，native mask / SH-capable full-view baseline 仍是后续任务。
  - RENDER-005T-AB 已完成 Spark packed SH-rest preservation：filtered Spark PLY reconstruction 现在会把 scene-level `f_rest_*` 编码为 Spark `extra.sh1/sh2/sh3`，并在 visible-index extract 后保留 SH extra；SH-heavy route 暴露 `data-spark-reconstruct-source="packed-sh-extract-v1"` 和 `data-spark-sh-rest-preserved="true"`。本机 `NeRF Lego 训练输出样例` 删除预览通过：`sparkPacked="packed-sh-extract-v1":255794/129108:171.2/41.7`、`sparkShRest=255794:255794:"true":45:3`。剩余外观 gap 是 registered compact `.splat` viewer source 不携带 degree-3 SH rest，导致 trained full reconstruct visual residual 不能直接以 `.splat` 为完整 SH baseline。
  - RENDER-005T-AC 已完成 Spark PLY SH full-view source baseline：SH-heavy 场景在 `真实查看` source/original 且无对象编辑状态时自动使用 Spark PLY SH source，暴露 `data-object-filter="spark-ply-sh-source"` 和 `packed-sh-extract-v1`；`?spark-ply-source=off` 可回到 legacy compact `.splat` 诊断路径。本机 trained Lego same-source residual gate 通过：`fullSource="spark-ply-sh-source":"packed-sh-extract-v1":255794:255794:true:45:3`、`coverageRatio=1.170018`、`lumaDelta=0.058189`、`chromaDelta=0.007036`。trained browser interaction audit 通过，真实查看非背景像素提升到 `70188`，删除后继续保留 `sparkShRest=255794:255794:"true":45:3`。
  - RENDER-005T-AD 已完成 Spark display `PackedSplats` cache telemetry：filtered Spark route 新增 `visible-index-lru-v1` display cache，并在临时 `SplatMesh` dispose 前摘掉 cached packed 引用，避免回到同一 visible-index set 时重复 `extractSplats`；UI HUD 区分 `过滤重建` / `缓存过滤` / `PLY SH 源`，browser audit 会通过隐藏/恢复对象证明 cache hit。Lego proxy 与 trained Lego 删除预览均通过，trained route 保持 `packed-sh-extract-v1` 和完整 SH rest preservation。这仍不是原始 `.splat` 内部 native object mask，每个全新 visible set 仍会创建临时 `SplatMesh`。
  - RENDER-005T-AE 已完成 Spark filtered persistent `SplatMesh` update surface：filtered Spark session 现在会保留同一个 `SplatMesh`，在 isolate / delete / restore 等 object-state 变化时更新其 display `PackedSplats` source，并通过 `data-spark-mesh-update-mode="persistent-splatmesh-v1"`、mesh id、reuse 和 update count 暴露 browser contract。Lego proxy 与 trained Lego 删除预览均证明同一个 mesh 在 hide / restore 后复用并更新：`sparkMesh="persistent-splatmesh-v1":1:"true":4`。这一步减少交互重建感，但仍不是原始 `.splat` 内部 native object mask；每个全新 visible set 仍需要 display `PackedSplats` extract 或 cache hit。
  - RENDER-005T-AF 已完成 Spark object opacity mask over packed source：filtered Spark route 现在用 `object-opacity-texture-v1` mask texture + Spark Dyno `objectModifier` 控制每个 Gaussian opacity，object-state 变化不再做 display `PackedSplats.extractSplats(...)`，browser contract 要求 `data-object-filter="spark-object-opacity-mask"`、`data-spark-packed-extract-ms="0.000"` 和 `data-spark-display-cache-mode="disabled-by-native-mask-v1"`。Lego proxy 删除预览通过：`sparkObjectMask="object-opacity-texture-v1":"4096x2":3909/1787:4`；trained Lego 删除预览通过：`sparkObjectMask="object-opacity-texture-v1":"4096x63":129108/126686:2`、`sparkShRest=255794:255794:"true":45:3`。因此当前“原始颜色 / 自身颜色”在对象编辑后已是 Spark 高斯路径，剩余颗粒感主要来自隔离 / 删除后的 object_id 子集稀疏、边界 assignment 噪声，以及尚未实现 original compact `.splat` 内部 object mask。
  - RENDER-005T-AG 已完成 Spark object opacity mask visual delta guard：`audit-demo` 小场景 hide / restore stress 现在会截图 delete baseline、hide-one-object 和 restored Spark canvas，要求隐藏对象后 checksum 与 coverage/luma/chroma 发生实际变化，恢复后回到 baseline。Lego proxy 通过：`sparkMaskVisual="spark-object-mask-visual-delta-v1":"4a2ed0e8"/"be002ca4"/"4a2ed0e8":0.000752/0.014063/0.026019:0/0/0`。这把 Spark mask 从“DOM telemetry 正确”推进到“像素级可见变化已验收”；original compact `.splat` 与 object-aware PLY packed source 的 index mapping 仍是下一步。
  - RENDER-005T-AH 已完成 compact `.splat` / object-aware PLY index mapping audit：新增 `npm run audit:splat-index-mapping`，对 5 个 public/generated Gaussian 样例检查 count、逐 index position / scale delta、rounded-position multiset 和 object_id 范围；当前 Plush、Plush v1、Plush semantic、Lego proxy、trained Lego 全部 `maxPositionDelta=0`、`maxScaleDelta=0`、`positionMultisetCoverage=1`，证明这些样例可用 Gaussian index 作为外部 object mask key。该结论只覆盖 ObjGauss 生成/登记的 public samples，不等价于任意第三方 `.splat` 内部携带 object_id。
  - RENDER-005T-AI 已完成 Spark native compact `.splat` object mask prototype：新增 URL gate `?spark-native-mask=on`，source/original 对象编辑状态可直接加载原始 compact `.splat`，并用已验证 index mapping 的 object-aware PLY `object_id` 构建 `object-opacity-texture-v1`，通过 Spark `SplatMesh({ url, objectModifier })` 控制删除 / 隔离。Lego proxy browser audit 通过：`sparkMaskSource="native-splat"`、`sparkPacked="native-splat-source-v1":5696/3909:0/0`、`sparkMesh="persistent-splatmesh-v1":1:"true":4`；默认 packed-source 路径仍保持 `sparkMaskSource="ply-packed"`。该路径目前仍是诊断开关，尚未作为默认商业展示路线。
  - RENDER-005T-AJ 已完成 Spark native compact `.splat` object mask 多场景候选 gate：新增 `scripts/audit-spark-native-mask-gate.mjs` 和 `npm run audit:spark-native-mask-gate`，默认覆盖 Lego proxy + Plush semantic。Gate 验证 native source / object mask / persistent mesh contract；Lego pixel delta 由完整 `audit-demo` 覆盖。Plush 281k 大场景通过：`source="native-splat"`、`route="native-splat-source-v1"`、`visible=104403/281498`。该 gate 证明 native route 不是单场景偶然，但默认商业展示路线仍未切换。
  - RENDER-005T-AK 已完成 Spark native compact `.splat` mask 安全默认化：source/original 对象编辑状态下，无 SH-rest 样例默认走 native compact `.splat` source；SH-heavy 样例默认保留 PLY packed SH route，避免丢失 degree-3 SH 外观。Lego 默认 audit 通过：`sparkMaskSource="native-splat"`、`sparkPacked="native-splat-source-v1":5696/3909:0/0`、`sparkMaskVisual="spark-object-mask-visual-delta-v1"`；trained SH-heavy audit 通过：`sparkMaskSource="ply-packed"`、`sparkPacked="packed-sh-extract-v1":255794/129108:160.4/0`、`sparkShRest=255794:255794:"true":45:3`。诊断开关保留：`spark-object-source=packed` / `spark-native-mask=off` 强制 PLY packed，`spark-native-mask=on` 强制 native。
  - RENDER-005T-AL 已完成 Spark canvas object selection product path：source/original Spark filtered edit viewport 现在暴露 `screen-space-object-pick-v1`，点击 Spark 高斯画布会用 object-aware PLY Gaussian 投影选择最近可见 `object_id`，并通过 `data-spark-selection-mode` / `data-spark-selected-object` 验收。Lego 默认 audit 通过：删除后 `sparkCanvasSelectedObject=0`，同时保持 `sparkMaskSource="native-splat"`；trained SH-heavy audit 通过：删除后 `sparkCanvasSelectedObject=3`，同时保持 `sparkMaskSource="ply-packed"` 与完整 SH rest。
  - RENDER-005T-AM 已完成 Spark pick hit telemetry 与选中 marker：Spark canvas pick 现在暴露 `data-spark-pick-status/object/distance/candidate-objects/ambiguous` 和 `data-spark-selected-marker-visible`，画布内会显示非交互选中标记；`audit-demo` 要求 hit、选中对象匹配、距离在半径内、候选数大于 0、marker 可见。当前 Lego proxy 输出 `sparkPick="screen-space-object-pick-v1":"hit":"0":3.7:3:"true":"true"`，trained Lego 输出 `sparkPick="screen-space-object-pick-v1":"hit":"3":0.892:3:"true":"true"`；两者都有效命中但都 `ambiguous=true`。
  - RENDER-005T-AN 已完成 Spark pick 多点击 hit-rate / ambiguity report：新增 `npm run audit:spark-pick-report`，默认对 Lego proxy 的 Spark 删除预览执行 15 个画布点击点并输出 `/tmp/objgauss-spark-pick-report/summary.{json,md}`。当前 Lego proxy `14/15` hit、hit rate `0.933333`、ambiguous hit rate `0.928571`；本机 trained Lego 5-click 显式报告 `5/5` hit、ambiguous hit rate `1`，同时保持 `maskSource="ply-packed"` 与 `packed-sh-extract-v1`。结论是 Spark canvas pick 可用且 marker/selected 状态一致，但 screen-space object 邻近歧义很高，不能宣称 robust renderer-native picking。
  - RENDER-005T-AO 已完成 Spark pick object-support disambiguation：`screen-space-object-pick-v1` 不再只按最近 Gaussian 和第二 object 距离差判歧义，而是用 `object-support-score-v1` 聚合每个候选 object 的最近距离、局部支持占比和前景深度得分，并暴露 `data-spark-pick-strategy/score/score-margin/second-object/second-score`。默认 Lego proxy report 仍 `14/15` hit，但 ambiguity rate 从 `0.928571` 降到 `0.357143`；trained Lego 5-click 从 `1` 降到 `0.2`。`audit-demo` 两个样例的单次 Spark pick 均变为 `ambiguous=false`，同时保持 marker 可见和 selected object match。
  - RENDER-005T-AP 已完成 WebGPU offscreen readback probe：新增 `webgpu-probe=offscreen-readback` 和 `npm run audit:webgpu-offscreen-readback`，该 probe 只 dispatch pixel compute、把 `pixelResolvedRgba` copy 到 `MAP_READ` buffer 并暴露 `data-webgpu-readback-*`，不创建 canvas render pass。Lego proxy 本地 audit 通过：`firstFrame="readback":253952`、`queue="done"`、`deviceLost="active"`、`readback="mapped":"webgpu-compute-depth-binned-alpha-composite-v1":"897e852d":4063232:1015808/1015808:533740`，证明 headless/CI 可单独验证 WebGPU compute/storage/readback，而不把 presentation backend loss 误判为 renderer compute failure。
  - RENDER-005T-AQ 已完成 WebGPU offscreen readback 多场景 suite：新增 `scripts/audit-webgpu-offscreen-readback.mjs`，`npm run audit:webgpu-offscreen-readback` 现在默认覆盖 Lego proxy + Plush semantic，并写出 `/tmp/objgauss-webgpu-offscreen-readback/summary.{json,md}`。默认双场景 gate 通过：Lego `readback="mapped":"897e852d":4063232:1015808/1015808:533740`，Plush `readback="mapped":"0f87864a":2359296:589824/589824:254524`、`packedGaussians=281498`、`tileReferences=1190026`。
  - RENDER-005T-AR 已完成 WebGPU offscreen object-state transition gate：`audit-demo --webgpu-object-transition` 会在 `offscreen-readback` 下通过 `spark-filtered-edit=off` 留在 WebGPU Tile 编辑路径，执行选中、隔离和删除，并要求 object-state checksum 与 readback checksum 都三段变化。默认双场景 gate 通过：Lego readback `897e852d -> 3bd507d9 -> 916a5fc9`，Plush readback `0f87864a -> 0bdb3b09 -> 9660bc47`；对应 object-state 分别为 `7243475b -> f72fa1f4 -> 35652440` 和 `362760d7 -> fc48aab0 -> 637142bc`。
  - RENDER-005T-AS 已完成 WebGPU headless acceptance / presentation split：新增 `npm run acceptance:webgpu-headless`，一键执行 build、WebGPU tile smoke 和 offscreen object-transition suite；`docs/rendering/webgpu-headless-acceptance.md` 明确该命令证明 compute/storage/object-state/readback，不证明 canvas presentation，headed presentation 仍由 `npm run audit:webgpu-desktop` 覆盖，visual tuning 仍由 coverage gate 覆盖。
  - RENDER-005T-AT 已完成 renderer readiness matrix：`docs/rendering/renderer-readiness-matrix.md` 明确商业 Demo 默认继续走 Spark source/original route，WebGPU Tile 作为 C-path proof / diagnostics，Gaussian OIT 作为 fallback；同时把“原始颜色 / 自身颜色删除后仍可能颗粒”界定为 hard object mask、边界 assignment、未重优化补洞和 SH-heavy packed route 的质量边界，而不是颜色丢失 bug。
  - RENDER-005T-AU 已完成产品 renderer route UI contract：颜色下拉改为 `自身颜色` / `对象色诊断`，viewport 增加 route badge，状态面板显示 `展示路线`、`颜色用途`、`预览边界`；root DOM 暴露 route / color role / preview boundary attributes，并由 `audit-demo` 验证首屏 commercial Spark、对象色 diagnostic、删除后 `hard-object-mask-no-reoptimize`。
  - RENDER-005T-AV 已完成 SH-heavy Spark route-only audit：新增 `npm run audit:spark-trained-route`，默认用静态 preview 验证 `nerf-lego-trained-output-local` 的 `spark-ply-sh-source` 初始 route、删除后的 `spark-packed-sh-mask` / `packed-sh-extract-v1`、`object-opacity-texture-v1`、`hard-object-mask-no-reoptimize` 和完整 degree-3 SH rest preservation。
  - RENDER-005T-BD 已完成产品 hard-mask quality status：状态面板新增 `质量解释`，root DOM 暴露 `data-hard-mask-quality-*`，`audit-demo` 可验证并打印 hard-mask quality contract；`自身颜色` 现在只表示颜色来源，颗粒感来源由 `预览边界` 和 `质量解释` 区分为原始 Spark、高斯 hard mask 边界混合、重建残差或待审计。
  - RENDER-005T-BE 已完成 commercial demo readiness QA：新增 `npm run audit:commercial-demo-readiness`，可把 Spark route summary 与 hard-mask quality summary 合并成样例准入表，区分“商业展示路线可演示”“研究 / 诊断样例”“待 route QA”和 public-commercial license eligibility；当前本地报告显示 public-commercial candidate 为 0。
  - DEMO-004 已登记 Poly Haven Chair 商用展示样例：`polyhaven-chair-commercial-demo-local` 指向本地生成的 `polyhaven_chair_demo.splat` / `polyhaven_chair_demo_objects.ply`，发布命令为 `npm run publish:polyhaven-chair-demo`；当前本地 commercial readiness 报告显示 public-commercial candidate 为 1，但它仍是 hard-mask / no-reoptimize 删除预览。
  - DEMO-005A 已强化 `自身颜色` hard-mask 预览 UX contract：删除后 root DOM 暴露 `data-source-preview-result="hard-mask-no-inpaint"`，状态面板显示 `删除结果=源色 mask 预览`，避免把源色 hard-mask 误解为补洞后的完整高斯重渲染。
  - DEMO-005B 已新增 Spark object mask feather 诊断路径：`spark-object-mask-feather=on` 会保持 hidden object opacity 为 0，同时对靠近 hidden boundary 的 visible Gaussian 降低 opacity；默认仍关闭，并由 `npm run audit:spark-mask-feather` 单场景验收。
  - DEMO-005C 已新增 Spark object mask feather 多场景 sweep/report：`npm run audit:spark-mask-feather-sweep` 默认比较 Lego proxy 与 Plush semantic 的 hard mask / `feather55`，输出 route、opacity texture、coverage / luma / chroma、截图和 summary；当前结果显示 feather 可软化边界但未改善 coverage ratio，因此默认仍关闭。
  - DEMO-005D 已新增 Spark object mask feather UI toggle：对象编辑控制面板提供 `柔化删除边界` checkbox，默认关闭；开启后通过显式 `objectMaskFeathering` prop 驱动 Spark object opacity texture，root DOM 和状态面板暴露 enabled / opacity / radius，可由 `audit:spark-mask-feather-sweep -- --control ui` 验收。
  - DEMO-005E 已新增 Spark mask feather candidate gate：`npm run audit:spark-mask-feather-candidates` 默认覆盖 Lego proxy、Plush semantic 和 Poly Haven Chair commercial sample，比较 hard / feather55 / feather70 / feather55r035，并输出 promotion recommendation；当前三个 feather candidates 均为 `diagnostic-only`，默认 hard mask 不变。
  - DEMO-005F 已新增 object boundary cleanup candidate report：`npm run audit:object-boundary-cleanup` 默认覆盖 Lego proxy、Plush semantic 和 Poly Haven Chair commercial sample，在 hard-mask boundary diagnostic 中输出 `object-boundary-cleanup-candidate-v1`，定位值得做 assignment cleanup / remap review 的 object 子集；该报告只读，不自动改 PLY。
  - DEMO-005G 已新增 object boundary remap preview export：`npm run audit:object-boundary-remap-preview` 可生成保留原始 PLY 属性、仅 patch sampled `object_id` 的 `/tmp` preview PLY，用于下一步 browser residual gate；该实验不改默认样例、不证明视觉质量改善。
  - DEMO-005H 已新增 object boundary remap browser residual gate：`npm run audit:object-boundary-remap-residual` 会先生成 sampled remap preview PLY，再用 Playwright 上传原始 PLY 与 remap-preview PLY，强制同一 PLY-packed Spark object-mask route，删除 top remap candidate object 并比较 before/after canvas stats。Lego proxy gate 通过：target object=`2`，remap-preview 少隐藏 `49` 个 Gaussian，after-delete residual=`0.999216/0.004332/0.019990`，recommendation=`browser-evidence-only`，因此仍不默认替换样例。
  - DEMO-005I 已将 object boundary remap residual gate 扩展为默认三场景 promotion table：Lego proxy、Plush semantic、Poly Haven Chair 全部通过同一 PLY-packed Spark route browser gate，remap-preview 分别少隐藏 `49` / `2786` / `29` 个 Gaussian，after-delete residual max=`0.000784/0.004332/0.019990`；aggregate recommendation=`do-not-promote-default-hard-mask`，说明 remap preview 当前可作为证据，不作为默认样例替换。
  - DEMO-005J 已新增 top-N object boundary remap target sweep：`npm run audit:object-boundary-remap-target-sweep` 默认 `--target-count 2`，覆盖三场景 6 个高风险 target object。Top-2 gate 全部通过 route/residual 阈值，但只有 1/6 target 满足 promotion 条件；Lego target `3` 和 Plush target `0` 的 hidden delta 分别为 `+397` / `+4085`，证明 sampled remap 不能按全局默认启用，只能进入 target-level review / allowlist。
  - DEMO-005K 已新增 target-level object boundary remap decision policy：`npm run audit:object-boundary-remap-policy` 会在 top-N browser residual gate 后额外写出 `remap-decision-policy.json/md`，将 target 分为 `allowlist-candidate`、`deny-*` 和 `review-only`。Policy 默认动作仍是 `keep-hard-mask`，应用模式是 `manual-target-allowlist-only`，禁止把 sampled remap 作为全局默认；Lego smoke 中 target `3` 被归为 `deny-hidden-increase`，target `2` 为 `review-only`。
  - DEMO-005L 已让 remap export / QA 消费 decision policy：`npm run audit:object-boundary-remap-policy-export` 读取 `remap-decision-policy.json`，并且只有同时满足 policy `allowlist-candidate` 与命令行显式 `--allow-target asset_id:object_id` 的 target 才会 patch `object_id`。当前三场景 policy-export raw candidates=`10012`，实际 applied remaps=`0`，blocked=`10012`；显式传入 denied target `nerf-lego-alpha-closure-local:3` 仍 remapped=`0`，确认 risky/review-only target 保持 hard mask。
  - DEMO-005M 已新增 reviewed remap allowlist manifest 与正向 fixture gate：`docs/rendering/object-boundary-remap-reviewed-allowlist.json` 当前为空，`npm run audit:object-boundary-remap-policy-export` 同时要求 policy candidate 和 reviewed allowlist 命中，因此真实三场景仍 raw candidates=`10012`、applied=`0`、blocked=`10012`。`npm run audit:object-boundary-remap-reviewed-allowlist` 使用 `/tmp` synthetic policy + reviewed allowlist 证明正向路径可工作：Lego fixture target=`2`，applied=`402`，blocked=`741`。
  - DEMO-005N 已新增 reviewed allowlist 人工评审 runbook 与 manifest schema gate：`docs/rendering/object-boundary-remap-review-runbook.md` 定义进入 allowlist 前必须查看 fixed-port browser screenshots、hidden delta、non-target residual 和 owner approval；`npm run audit:object-boundary-remap-reviewed-allowlist-manifest` 会校验 committed allowlist，当前 `targets=0` 通过。Export 现在会拒绝缺少 reviewer / owner approval / allowlist-candidate evidence 的 approved target；真实三场景 policy export 仍 applied=`0`、blocked=`10012`。
  - DEMO-005O 已新增 Spark canvas hover-confirm selection UX：Spark source/original object edit 现在先在 hover 时显示候选 marker，再由 click 确认选中；root DOM 暴露 `data-spark-pick-interaction="hover-confirm-v1"`、hover pick object / marker 和 confirmed pick marker。`npm run audit:spark-pick-report -- --port 5395` 已验证 Lego proxy 6/6 hover hits、6/6 confirmed hits、markerHits=6/6；`audit:demo` 单样例也通过同一 fixed-port `5395` hover-confirm contract。
  - DEMO-005P 已新增 Spark native pick feasibility audit：`SplatViewport` 暴露 `spark-native-pick-feasibility-v1` telemetry，`npm run audit:spark-native-pick-feasibility -- --port 5395` 在 Lego proxy 删除预览下验证 Spark `SplatMesh.raycast` 可 hit，但 intersection payload 只有 `distance,object,point`，没有 splat index / object id。结论是 recommendation=`keep-screen-space-hover-confirm`，blocker=`raycast-intersection-missing-splat-index`；当前不能安全迁移到 renderer-native object picking。
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
  - `npm run train:splatfacto:near1m-candidate` 已固化为 TRAIN-003D/003E/003F/003G/003H/003I/003J/003K/003L/003M/003N near-1M candidate handoff：dry-run/status/run 三模式串起 10000-step Splatfacto candidate、balanced Object Field 登记和 `audit:webgpu-cpath-production-sla`，`--status-json` 可写出 `objgauss-near1m-candidate-status-v1` 机器可读 readiness report，并会在 dry-run、成功和已知失败路径留下当前 readiness、`launchReadiness`、`lastExit` 与 `lastFailure`。`npm run train:splatfacto:near1m-gpu-preflight` 可单独执行 `1GB` GPU memory reserve preflight，不启动 Splatfacto。`npm run train:splatfacto:near1m-background` 可 preflight / dry-run / status / guarded stop / 显式 confirmed 后台启动 near-1M 长训，并写 `/tmp` 下的 launcher manifest、status JSON、nested candidate status JSON 和 log；`--preflight` 会刷新 nested candidate status 并只在启动输入与 GPU preflight ready 时返回 0，`--status` 会汇总 nested candidate readiness、launch readiness、last exit 和 last failure kind，`--stop --confirm-stop` 会向记录的 detached process group 发送 `SIGTERM`。Pipeline 默认要求 exported PLY / object-aware PLY 均 `>=1000000` Gaussians；低于阈值会在登记或 SLA 前输出 `near1m_scale_gate=failed` 并停止。真正启动长训还必须显式传 `--confirm-long-run`，否则 `--run` 会输出 `near1m_long_run_guard=failed`；确认后的 CUDA 长训还会先用 `nvidia-smi` 执行默认 `1GB` GPU memory reserve preflight，失败时在启动 Splatfacto 前输出 `near1m_gpu_preflight=failed`。宿主机提权只读预检已通过：RTX 5060 Ti，free=`15186MiB`，reserve=`1024MiB`；当前 status JSON 显示 dataset、SAM checkpoint、balanced SAM manifest 已 present；near-1M train config / checkpoint / exported PLY / object-aware PLY / SLA summary 仍 missing。
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
  - `npm run acceptance:demo` 已固化为一键闭环总验收命令；browser audit 默认走 built preview 以避开 dev watcher 上限，Spark commercial route gate 可通过 `--include-spark-commercial-route` 显式纳入，但默认不要求本机 trained SH-heavy sample。
  - `npm run acceptance:semantic` 已固化为 SEMANTIC benchmark suite 验收命令。
  - `npm run acceptance:webgpu-headless` 已固化为 WebGPU CI/headless acceptance 命令；默认构建 viewer、跑 WebGPU tile smoke、再跑 Lego proxy + Plush semantic 的 offscreen object-transition suite。
  - `docs/training/splatfacto-smoke.md` 已记录 Splatfacto smoke 训练 / 导出 / SAM / Object Field 的 runbook 和输出 contract。
  - `npm run train:splatfacto:near1m-candidate` 已固化为 near-1M trained candidate 到 production SLA 的本地编排入口：只有 `--run --confirm-long-run` 才会启动长训，后续输出仍在 ignored `outputs/` 和 `/tmp`；默认 scale gate 会拒绝低于 1M Gaussians 的 exported / object-aware PLY。
  - `npm run benchmark:splatfacto:balanced` 已固化为 safe-2000 balanced candidate 的一键本地 benchmark 入口，可重跑 balanced SAM、`training register-output`、emergence metrics、curve、report 和 summary。
  - `npm run benchmark:splatfacto:variants` 已固化为 safe-2000 同场景多 mask / slot policy 对比入口，可生成三变体 summary、CSV、Markdown 表格和 HTML 曲线报告。
  - `npm run benchmark:splatfacto:scenes` 已固化为 Splatfacto-trained scene suite，可比较 Lego safe-2000、LLFF Fern smoke 与 Poly Haven Chair smoke 三个 scene rows，并支持 train / held-out SAM manifest split。
  - `npm run benchmark:cross-scene` 已固化为跨场景 / 跨变体汇总入口，可聚合 semantic smoke suite、Splatfacto scene suite 和 safe-2000 variant suite 到同一张表，并输出 smoke / candidate / paper stage gates。
  - `npm run audit:webgpu-coverage-gate` 已固化为 WebGPU 编辑预览 coverage/luma/chroma/cost 的多场景 baseline gate，并输出可复查 summary report。
  - `npm run audit:webgpu-alpha-floor-sweep` / `npm run audit:webgpu-alpha-floor-candidate-gate` 已固化为 alpha presentation floor 候选的多场景复现实验和 strict gate。
  - `npm run audit:webgpu-cpath-readiness` 已固化为 WebGPU C-path readiness 聚合审计：一条命令重跑 build、1M storage/edit budget、headed browser object transition、可选 `--trained-ply` runtime evidence、可选 `--include-sustained-frame-pacing` baseline evidence 和显式 reviewed FPS SLA promotion check，并写出哪些证据已完成、哪些 1M browser/FPS gap 仍未完成。
  - `npm run audit:webgpu-cpath-production-sla` 已固化为 WebGPU C-path production SLA 终局 wrapper：强制 real trained near-1M PLY preflight、目标硬件标签、完整 readiness 运行、sustained trained PLY row 和 reviewed FPS SLA，拒绝 summary shortcut 和低于 1M 的 trained PLY。
  - `npm run audit:webgpu-frame-pacing` 已固化为 headed browser C-path frame pacing smoke：默认覆盖 Lego proxy 与 Plush semantic，采样 idle / isolate / delete 三段 rAF interval，输出 summary report 和截图。
  - `npm run audit:webgpu-ply-runtime` 已固化为 real/trained object-aware PLY browser runtime gate：调用者提供 `--input-ply` 和 `--min-gaussians`，可直接用于后续 near-1M trained scene runtime 验收。
  - `npm run audit:webgpu-sustained-frame-pacing` 已固化为 headed browser C-path sustained frame-pacing baseline：默认覆盖当前真实场景和 synthetic 1M upload/runtime，各 phase 采样 `120` rAF interval，并可通过 `--trained-ply` 增加 real/trained object-aware PLY 长采样行。
  - `npm run audit:webgpu-offscreen-readback` 已固化为 WebGPU compute/storage/readback 多场景 object-transition suite；默认不依赖 canvas presentation 成功，覆盖 Lego proxy 与 Plush semantic，验证首帧、隔离、删除三段 readback checksum 和 object-state checksum，并输出 summary report，适合在 CI/headless 环境区分计算管线和 presentation backend loss。
  - `npm run audit:spark-reconstruct-residual` / `npm run audit:spark-reconstruct-residual-multiscene` 已固化为 Spark full `.splat` 与 PLY reconstructed Spark 的 visual residual gate。
  - `npm run audit:object-mask-boundary` 已固化为 hard object mask 质量诊断，可从 object-aware PLY 输出 deleted-subset coverage、unique coverage loss、shared boundary coverage 和 3D neighbor boundary risk，解释删除后 source/original 预览仍有颗粒感的可能来源。
  - `npm run audit:object-boundary-cleanup` 已固化为 hard-mask 边界清理候选报告，可输出 cleanup candidate Gaussian 估算量、dominant target object、priority score 和 recommendation，用于决定下一轮是否做 cleaned object_id / remap preview 实验。
  - `npm run audit:object-boundary-remap-preview` 已固化为 sampled cleaned `object_id` preview 导出入口，默认输出 `/tmp/objgauss-object-boundary-remap-preview/*.remap-preview.ply` 和 summary，用于后续对比删除后 visual residual / non-target damage。
  - `npm run audit:hard-mask-quality` 已固化为 hard-mask 质量解释链聚合器，可把 PLY boundary diagnostic、Spark route summary 和 browser visual residual summary 按 asset 对齐，区分 boundary mixing、coverage hole risk 与 browser/source residual 主导问题。
  - `npm run audit:commercial-demo-readiness` 已固化为商用展示 QA 准入报告，可复用 Spark route 与 hard-mask quality artifacts，列出 route tier、质量解释、素材许可范围、required copy 和截图路径，避免把研究/本地样例误标成 public commercial demo。
  - `npm run audit:splat-index-mapping` 已固化为 compact `.splat` 与 object-aware PLY 的 Gaussian index mapping gate，用于 native source / original `.splat` object mask 原型前置验收。
  - `npm run audit:spark-native-mask-gate` 已固化为 native compact `.splat` object mask 的 Lego + Plush 多场景默认 route contract gate。
  - `npm run audit:spark-trained-sample` 已固化为本机 trained SH-heavy sample availability preflight，可在浏览器 route gate 前检查 `nerf_lego_trained.*` public sample、`object_id`、degree-3 `f_rest_*` 和 object 数量。
  - `npm run audit:spark-trained-route` 已固化为本机 trained SH-heavy route-only browser gate，低成本验证 `spark-packed-sh-mask` / SH preservation / hard-object-mask boundary。
  - `npm run audit:spark-mask-feather-sweep` 已固化为 Spark object mask feather 多场景报告，可比较 hard mask 与 feather variants 对 opacity texture、coverage / luma / chroma 和截图的影响，并支持 `--control ui` 验证 Web 内显式 soft-boundary toggle。
  - `npm run audit:spark-mask-feather-candidates` 已固化为 soft-boundary promotion 前置 gate，覆盖三场景与多个 opacity / radius candidates；当前报告给出 `promotionCandidate=false`，说明 feather 仍只是诊断开关。
  - `npm run acceptance:renderer-ci` 已固化为 fresh-clone-safe renderer 默认 CI profile，覆盖 B -> C renderer route contract、build、WebGPU tile smoke、WebGPU 100k-1M scale budget、no-SH public sample index mapping 和 Spark native object mask gate，不要求本机 trained SH-heavy sample。
  - `npm run acceptance:renderer-product` 已固化为显式产品 / Demo route profile，会调用完整 `acceptance:spark-commercial-route`，包括 trained sample availability preflight 和 SH-heavy packed SH route。
  - `npm run acceptance:spark-commercial-route` 已固化为 Spark commercial route 总验收命令，一次覆盖 trained sample availability、no-SH native compact `.splat` object mask 与 SH-heavy packed SH object mask；该命令会写 `/tmp/objgauss-spark-commercial-route/summary.{json,md}`，证明 route contract，不证明删除后补洞或重优化。
  - `npm run audit:spark-pick-report` 已固化为 Spark canvas `screen-space-object-pick-v1` 的多点击 hit-rate / ambiguity report；默认跑 Lego proxy，小成本 gate，trained 大场景可用 `--assets nerf-lego-trained-output-local --max-clicks 5` 显式复查。当前 report 默认要求 ambiguity rate `<=0.5`，用于防止 pick 消歧回退。
  - `npm run audit:spark-native-pick-feasibility` 已固化为 Spark native ray/object metadata feasibility report；默认 fixed port `5395`，只在 URL probe 下执行 raycast sample，当前结论是 Spark raycast 可作为 depth probe，但不能返回 splat index / object id，因此不能替代 `hover-confirm-v1` screen-space object picking。
  - PORT-001 已将本地 browser audit / acceptance 默认端口收敛到 fixed `5395`：`audit:demo`、Spark audits、WebGPU browser audits、renderer / demo / Spark commercial / WebGPU headless acceptance 默认都走 `5395`。如端口占用，应停止占用 `5395` 的本地 preview/audit 进程后重跑，不再轮换随机端口；显式 override 参数仍保留用于特殊诊断。
  - PORT-002 已将裸跑 `npm run dev` 与 `npm run preview` 也固定到 `127.0.0.1:5395 --strictPort`；日常 Web 查看和 browser audit 使用同一端口，端口占用时先停止占用进程再重跑。
  - `npm run audit:renderer-route-contract` 已固化为 B -> C renderer 路线静态合约审计：检查 WebGL Gaussian OIT fallback、WebGPU tile terminal path、Spark commercial source route、browser audit telemetry 和 fixed `5395` 端口策略仍保持一致。
  - `docs/benchmarks/spark-filtered-edit.md` 已记录 Spark filtered edit preview 的 runtime contract、验证命令和剩余 gap。
  - `objgauss demo audit-v1-goal --allow-incomplete` 已固化为阶段目标完成度审计命令。
  - baseline commit: `c8dcef7`.

## 最近验证

2026-06-25:

```bash
node --check scripts/audit-webgpu-cpath-readiness.mjs
node --check scripts/audit-renderer-route-contract.mjs
npm run audit:webgpu-cpath-readiness -- --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-trained-ply --skip-synthetic-1m-runtime --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000
node --check scripts/audit-webgpu-cpath-readiness.mjs
node --check scripts/audit-renderer-route-contract.mjs
npm run audit:webgpu-sustained-frame-pacing -- --port 5395 --output-dir /tmp/objgauss-webgpu-sustained-frame-pacing-readiness-short --skip-build --frame-count 10 --min-real-approx-fps 1 --min-synthetic-approx-fps 1 --max-real-mean-frame-ms 300 --max-synthetic-mean-frame-ms 300 --max-p95-frame-ms 2500 --max-long-frame-ratio 1
npm run audit:webgpu-cpath-readiness -- --skip-run --skip-synthetic-1m-runtime --scale-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/scale-budget/summary.json --edit-cost-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/edit-cost-budget/summary.json --transition-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/presentation-transition/summary.json --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --trained-ply-runtime-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/trained-ply-runtime/summary.json --sustained-frame-pacing-summary /tmp/objgauss-webgpu-sustained-frame-pacing-readiness-short/summary.json --sustained-min-real-approx-fps 1 --sustained-min-synthetic-approx-fps 1 --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-sustained-summary
node --check scripts/audit-webgpu-sustained-frame-pacing.mjs
node --check scripts/audit-webgpu-cpath-readiness.mjs
node --check scripts/audit-renderer-route-contract.mjs
npm run audit:webgpu-sustained-frame-pacing -- --port 5395 --output-dir /tmp/objgauss-webgpu-sustained-frame-pacing-trained-ply-short --skip-build --frame-count 10 --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --min-real-approx-fps 1 --min-synthetic-approx-fps 1 --min-trained-approx-fps 1 --max-real-mean-frame-ms 300 --max-synthetic-mean-frame-ms 300 --max-trained-mean-frame-ms 300 --max-p95-frame-ms 2500 --max-long-frame-ratio 1
npm run audit:webgpu-cpath-readiness -- --skip-run --skip-synthetic-1m-runtime --scale-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/scale-budget/summary.json --edit-cost-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/edit-cost-budget/summary.json --transition-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/presentation-transition/summary.json --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --trained-ply-runtime-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/trained-ply-runtime/summary.json --sustained-frame-pacing-summary /tmp/objgauss-webgpu-sustained-frame-pacing-trained-ply-short/summary.json --sustained-min-real-approx-fps 1 --sustained-min-synthetic-approx-fps 1 --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-trained-sustained-summary
npm run audit:webgpu-cpath-readiness -- --skip-run --skip-synthetic-1m-runtime --scale-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/scale-budget/summary.json --edit-cost-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/edit-cost-budget/summary.json --transition-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/presentation-transition/summary.json --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --trained-ply-runtime-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/trained-ply-runtime/summary.json --sustained-frame-pacing-summary /tmp/objgauss-webgpu-sustained-frame-pacing-trained-ply-short/summary.json --sustained-min-real-approx-fps 1 --sustained-min-synthetic-approx-fps 1 --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-fps-sla-review
npm run audit:webgpu-cpath-readiness -- --allow-failures --skip-run --skip-synthetic-1m-runtime --scale-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/scale-budget/summary.json --edit-cost-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/edit-cost-budget/summary.json --transition-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/presentation-transition/summary.json --trained-ply public/samples/nerf_lego_trained_objects.ply --trained-min-gaussians 250000 --trained-ply-runtime-summary /tmp/objgauss-webgpu-cpath-readiness-trained-ply/trained-ply-runtime/summary.json --sustained-frame-pacing-summary /tmp/objgauss-webgpu-sustained-frame-pacing-trained-ply-short/summary.json --sustained-min-real-approx-fps 1 --sustained-min-synthetic-approx-fps 1 --fps-sla-reviewed --fps-sla-target-hardware local-rtx5060ti --fps-sla-min-trained-approx-fps 1 --port 5395 --output-dir /tmp/objgauss-webgpu-cpath-readiness-fps-sla-negative
node --check scripts/audit-webgpu-cpath-production-sla.mjs
node --check scripts/audit-renderer-route-contract.mjs
npm run audit:webgpu-cpath-production-sla -- --trained-ply public/samples/nerf_lego_trained_objects.ply --target-hardware local-rtx5060ti --dry-run --allow-failures --output-dir /tmp/objgauss-webgpu-cpath-production-sla-dry-run
npm run audit:webgpu-cpath-production-sla -- --trained-ply public/samples/nerf_lego_trained_objects.ply --target-hardware local-rtx5060ti --skip-run --dry-run --allow-failures --output-dir /tmp/objgauss-webgpu-cpath-production-sla-forbidden-dry-run
node --check scripts/train-splatfacto-near1m-candidate.mjs
node --check scripts/launch-splatfacto-near1m-background.mjs
# expected failed with exit 2: GPU preflight unavailable in sandbox, launchReadiness=not-ready
npm run train:splatfacto:near1m-background -- --preflight --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-background-preflight-not-ready
npm run train:splatfacto:near1m-background -- --preflight --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --skip-gpu-preflight --output-dir /tmp/objgauss-near1m-background-preflight-ready-smoke
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-background-preflight-ready-smoke
npm run train:splatfacto:near1m-candidate -- --dry-run --target-hardware local-rtx5060ti --skip-pull --status-json /tmp/objgauss-near1m-candidate-status-json-smoke/summary.json
npm run train:splatfacto:near1m-background -- --dry-run --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-background-status-json-smoke
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-background-status-json-smoke --candidate-status-json /tmp/objgauss-near1m-candidate-status-json-smoke/summary.json
# expected failed with exit 2: near-1M long training requires --confirm-long-run, but still writes status JSON
npm run train:splatfacto:near1m-candidate -- --run --skip-sla --target-hardware local-rtx5060ti --status-json /tmp/objgauss-near1m-candidate-run-guard-status/summary.json
# expected failed with exit 2: writes lastExit=failed and lastFailure=long-run-confirmation-required
npm run train:splatfacto:near1m-candidate -- --run --skip-sla --target-hardware local-rtx5060ti --status-json /tmp/objgauss-near1m-candidate-failure-status/summary.json
npm run train:splatfacto:near1m-candidate -- --dry-run --target-hardware local-rtx5060ti --skip-pull --status-json /tmp/objgauss-near1m-candidate-passed-status/summary.json
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-background-failure-status --candidate-status-json /tmp/objgauss-near1m-candidate-failure-status/summary.json
npm run train:splatfacto:near1m-candidate -- --dry-run --target-hardware local-rtx5060ti --skip-pull
npm run train:splatfacto:near1m-candidate -- --status
npm run train:splatfacto:near1m-candidate -- --status --status-json /tmp/objgauss-near1m-status/summary.json
npm run train:splatfacto:near1m-gpu-preflight -- --gpu-memory-reserve-gb 1 --status-json /tmp/objgauss-near1m-gpu-preflight-host/summary.json
npm run train:splatfacto:near1m-background -- --dry-run --target-hardware local-rtx5060ti --gpu-memory-reserve-gb 1 --output-dir /tmp/objgauss-near1m-background-smoke
npm run train:splatfacto:near1m-background -- --status --output-dir /tmp/objgauss-near1m-background-smoke
# expected failed with exit 2: stopping background near-1M training requires --confirm-stop
npm run train:splatfacto:near1m-background -- --stop --output-dir /tmp/objgauss-near1m-background-stop-guard
npm run train:splatfacto:near1m-background -- --stop --confirm-stop --output-dir /tmp/objgauss-near1m-background-stop-empty
# expected failed with exit 2: background near-1M training requires --confirm-long-run
npm run train:splatfacto:near1m-background -- --run --output-dir /tmp/objgauss-near1m-background-guard
# expected failed with exit 2: near-1M long training requires --confirm-long-run
npm run train:splatfacto:near1m-candidate -- --run --skip-sla --target-hardware local-rtx5060ti
# expected failed with exit 2 before training: nvidia-smi GPU reserve preflight unavailable/failed in this environment
npm run train:splatfacto:near1m-candidate -- --run --confirm-long-run --skip-sla --target-hardware local-rtx5060ti
# expected failed with exit 2: safe-2000 has 255794 Gaussians, below near-1M gate
npm run train:splatfacto:near1m-candidate -- --run --skip-train --skip-sla --skip-gpu-preflight --export-dir outputs/training/nerf-lego-splatfacto-long/export-safe-2000-cpu-cache-v1 --target-hardware local-rtx5060ti
npm run audit:renderer-route-contract
npm run build
uv run --extra dev pytest
git diff --check
```

2026-06-24:

```bash
node --check scripts/publish-polyhaven-chair-demo.mjs
npm run publish:polyhaven-chair-demo -- --output-dir /tmp/objgauss-polyhaven-chair-demo-publish
uv run objgauss stats public/samples/polyhaven_chair_demo_objects.ply
npm run audit:spark-trained-route -- --assets polyhaven-chair-commercial-demo-local --port 5367
npm run acceptance:spark-commercial-route -- --skip-build --skip-trained-sample-audit --trained-assets polyhaven-chair-commercial-demo-local --native-port 5368 --trained-port 5369 --output-dir /tmp/objgauss-spark-commercial-route-chair
npm run audit:object-mask-boundary -- --assets polyhaven-chair-commercial-demo-local --output-dir /tmp/objgauss-object-mask-boundary-chair
npm run audit:spark-reconstruct-residual -- --assets polyhaven-chair-commercial-demo-local --output-dir /tmp/objgauss-spark-reconstruct-residual-chair --port 5370 --allow-failures
npm run audit:hard-mask-quality -- --boundary-summary /tmp/objgauss-object-mask-boundary-chair/summary.json --route-summary /tmp/objgauss-spark-commercial-route-chair/summary.json --residual-summary /tmp/objgauss-spark-reconstruct-residual-chair/summary.json --output-dir /tmp/objgauss-hard-mask-quality-chair --require-route --require-residual
npm run audit:commercial-demo-readiness -- --output-dir /tmp/objgauss-commercial-demo-readiness-with-chair
node --check scripts/audit-demo.mjs
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5375
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5376 --spark-object-mask-feather --spark-object-mask-feather-opacity 0.55
npm run audit:spark-mask-feather
node --check scripts/audit-spark-mask-feather-sweep.mjs
npm run audit:spark-mask-feather-sweep -- --assets nerf-lego-alpha-closure-local --variants hard:off,feather55:0.55 --skip-build --port 5387 --output-dir /tmp/objgauss-spark-mask-feather-sweep-smoke
npm run audit:spark-mask-feather-sweep -- --skip-build --port 5388 --output-dir /tmp/objgauss-spark-mask-feather-sweep
npm run audit:spark-mask-feather-sweep -- --assets nerf-lego-alpha-closure-local --variants hard:off,feather55:0.55 --control ui --skip-build --skip-visual-stats --port 5389 --output-dir /tmp/objgauss-spark-mask-feather-ui-toggle
npm run audit:spark-mask-feather-sweep -- --assets nerf-lego-alpha-closure-local --variants hard:off,feather55:0.55 --control url --skip-build --skip-visual-stats --port 5390 --output-dir /tmp/objgauss-spark-mask-feather-url-regression
npm run audit:spark-mask-feather-candidates -- --skip-build --skip-visual-stats --port 5391 --output-dir /tmp/objgauss-spark-mask-feather-candidates-telemetry
npm run audit:spark-mask-feather-candidates -- --skip-build --port 5392 --output-dir /tmp/objgauss-spark-mask-feather-candidates
node --check scripts/audit-object-mask-boundary.mjs
npm run audit:object-boundary-cleanup
node --check scripts/export-object-boundary-remap-preview.mjs
npm run audit:object-boundary-remap-preview
uv run objgauss stats /tmp/objgauss-object-boundary-remap-preview/nerf-lego-alpha-closure-local.remap-preview.ply
node scripts/export-object-boundary-remap-preview.mjs --assets plush-semantic-closure-local --max-remap-samples 80000 --output-dir /tmp/objgauss-object-boundary-remap-preview-plush
uv run objgauss stats /tmp/objgauss-object-boundary-remap-preview-plush/plush-semantic-closure-local.remap-preview.ply
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5372
npm run build
uv run --extra dev pytest
git diff --check
node --check scripts/audit-commercial-demo-readiness.mjs
npm run audit:commercial-demo-readiness -- --output-dir /tmp/objgauss-commercial-demo-readiness
node --check scripts/audit-demo.mjs
npm run build
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --server-mode preview --port 5365
uv run --extra dev pytest
git diff --check
node --check scripts/audit-hard-mask-quality.mjs
npm run audit:hard-mask-quality -- --boundary-summary /tmp/objgauss-object-mask-boundary/summary.json --route-summary /tmp/objgauss-spark-commercial-route-availability/summary.json --residual-summary /tmp/objgauss-spark-reconstruct-residual-multiscene/summary.json,/tmp/objgauss-spark-reconstruct-residual-trained/summary.json --output-dir /tmp/objgauss-hard-mask-quality
npm run audit:hard-mask-quality -- --output-dir /tmp/objgauss-hard-mask-quality-default
npm run build
uv run --extra dev pytest
git diff --check
node --check scripts/audit-object-mask-boundary.mjs
npm run audit:object-mask-boundary -- --output-dir /tmp/objgauss-object-mask-boundary
npm run build
uv run --extra dev pytest
git diff --check
node --check scripts/acceptance-renderer-profile.mjs
npm run acceptance:renderer -- --profile ci --dry-run --output-dir /tmp/objgauss-renderer-profile-ci-dry-run
npm run acceptance:renderer-product -- --dry-run --output-dir /tmp/objgauss-renderer-profile-product-dry-run
npm run acceptance:renderer-ci -- --skip-native-route --output-dir /tmp/objgauss-renderer-profile-ci-nonbrowser
uv run --extra dev pytest
git diff --check
node --check scripts/acceptance-demo.mjs
node --check scripts/audit-demo.mjs
node --check scripts/acceptance-spark-commercial-route.mjs
npm run acceptance:demo -- --skip-semantic-benchmark --browser-audit-assets nerf-lego-alpha-closure-local --skip-browser-visual-residual --include-spark-commercial-route --spark-native-port 5351 --spark-trained-port 5352 --spark-route-output-dir /tmp/objgauss-acceptance-demo-spark-route --skip-spark-route-build
node -e "const r=require('/tmp/objgauss-acceptance-demo-spark-route/summary.json'); console.log(JSON.stringify({status:r.status, steps:r.steps.length, native:r.routes.native.length, trained:r.routes.trained.length}, null, 2))"
uv run --extra dev pytest
git diff --check
node --check scripts/acceptance-spark-commercial-route.mjs
npm run acceptance:spark-commercial-route -- --native-port 5349 --trained-port 5350 --output-dir /tmp/objgauss-spark-commercial-route-report
node -e "const r=require('/tmp/objgauss-spark-commercial-route-report/summary.json'); console.log(JSON.stringify({status:r.status, steps:r.steps.length, native:r.routes.native.length, trained:r.routes.trained.length}, null, 2))"
uv run --extra dev pytest
git diff --check
node --check scripts/acceptance-spark-commercial-route.mjs
npm run acceptance:spark-commercial-route -- --native-port 5347 --trained-port 5348
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
node --check scripts/audit-spark-trained-route.mjs
npm run build
npm run audit:spark-trained-route -- --port 5346
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --url http://127.0.0.1:5341/ --no-server
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
node --check scripts/acceptance-webgpu-headless.mjs
node --check scripts/audit-webgpu-offscreen-readback.mjs
node --check scripts/audit-webgpu-desktop.mjs
npm run audit:webgpu-tile-smoke
npm run build
npm run acceptance:webgpu-headless -- --port 5330 --output-dir /tmp/objgauss-webgpu-headless-acceptance
npm run audit:webgpu-offscreen-readback -- --assets nerf-lego-alpha-closure-local --port 5323 --output-dir /tmp/objgauss-webgpu-offscreen-readback-transition-single
npm run audit:webgpu-offscreen-readback -- --port 5324 --output-dir /tmp/objgauss-webgpu-offscreen-readback-transition
node --check scripts/audit-spark-pick-report.mjs
npm run build
npm run audit:spark-pick-report
npm run audit:spark-pick-report -- --assets nerf-lego-trained-output-local --max-clicks 5 --output-dir /tmp/objgauss-spark-pick-report-trained --port 5316
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5317/ --no-server
npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5317/ --no-server
node --check scripts/audit-demo.mjs
npm run build
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5314/ --no-server
npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5314/ --no-server
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
npm run audit:spark-reconstruct-residual
npm run audit:spark-native-mask-gate
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
npm run build
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5313/ --no-server
npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5313/ --no-server
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
npm run audit:spark-reconstruct-residual
npm run audit:spark-native-mask-gate
uv run --extra dev pytest
git diff --check
node --check scripts/audit-spark-native-mask-gate.mjs
node --check scripts/audit-demo.mjs
npm run build
npm run audit:spark-native-mask-gate
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --skip-visual-residual --url http://127.0.0.1:5312/ --no-server
npm run audit:demo -- --assets nerf-lego-trained-output-local --skip-visual-residual --url http://127.0.0.1:5312/ --no-server
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
npm run audit:spark-reconstruct-residual
uv run --extra dev pytest
git diff --check
node --check scripts/audit-spark-native-mask-gate.mjs
node --check scripts/audit-demo.mjs
npm run build
npm run audit:spark-native-mask-gate
npm run audit:demo -- --assets nerf-lego-alpha-closure-local --spark-native-mask --skip-visual-residual --port 5311
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
npm run build
npm run preview -- --host 127.0.0.1 --port 5302 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5302/ --no-server --spark-native-mask
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5302/ --no-server
npm run audit:splat-index-mapping
npm run audit:webgpu-tile-smoke
node --check scripts/audit-splat-index-mapping.mjs
npm run audit:splat-index-mapping
node --check scripts/audit-demo.mjs
npm run build
npm run preview -- --host 127.0.0.1 --port 5301 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5301/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5301/ --no-server
npm run audit:spark-reconstruct-residual
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
git diff --check
node --check src/sparkObjectMask.js
node --check scripts/audit-demo.mjs
npm run build
npm run preview -- --host 127.0.0.1 --port 5300 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5300/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5300/ --no-server
npm run audit:spark-reconstruct-residual
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
git diff --check
node --check scripts/audit-demo.mjs
npm run build
npm run preview -- --port 5299 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5299/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5299/ --no-server
npm run audit:spark-reconstruct-residual
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5298/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5298/ --no-server
npm run audit:spark-reconstruct-residual
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
node --check scripts/audit-demo.mjs
node --check scripts/audit-spark-reconstruct-residual.mjs
npm run build
npm run audit:spark-reconstruct-residual
npm run preview -- --port 5294 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5294/ --no-server
npm run audit:demo -- --asset nerf-lego-trained-output-local --url http://127.0.0.1:5294/ --no-server
node scripts/audit-spark-reconstruct-residual.mjs --assets nerf-lego-trained-output-local --output-dir /tmp/objgauss-spark-reconstruct-residual-trained-ac
node --check src/sparkPackedSh.js
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
uv run --extra dev pytest
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run build
uv run --extra dev pytest
npm run audit:webgpu-alpha-floor-candidate-gate -- --port 5292 --allow-failures
node --check scripts/audit-demo.mjs
npm run audit:webgpu-tile-smoke
npm run preview -- --port 5294 --strictPort
npm run audit:demo -- --asset nerf-lego-alpha-closure-local --url http://127.0.0.1:5294/ --no-server
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5295 --probes full
node --check scripts/lib/visual-stats.mjs
node --check scripts/audit-spark-reconstruct-residual.mjs
npm run audit:spark-reconstruct-residual
npm run audit:spark-reconstruct-residual-multiscene
node --check src/webgpuTileResolveShader.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-coverage-sweep -- --asset nerf-lego-trained-output-local --port 5290 --webgpu-color-mode sh-view --variants baseline:2.2:4:0.035,alpha05:2.2:4:0.05,alpha075:2.2:4:0.075,alpha10:2.2:4:0.1 --output-dir /tmp/objgauss-webgpu-alpha-floor-trained-sh-view
node --check scripts/audit-webgpu-coverage-sweep.mjs
npm run audit:webgpu-coverage-sweep -- --asset nerf-lego-trained-output-local --port 5289 --webgpu-color-mode sh-view --output-dir /tmp/objgauss-webgpu-coverage-trained-sh-view
node --check src/ply.js
node --check src/webgpuTileSmoke.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5287 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5288 --probes full --webgpu-color-mode sh-view
node --check src/ply.js
node --check src/webgpuTileSmoke.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-trained-output-local --port 5286 --probes full
node --check src/webgpuCameraTuning.js
node --check src/webgpuDepthTuning.js
node --check src/webgpuTileComputeShader.js
node --check src/webgpuTileSmoke.js
node --check src/webgpuCapability.js
node --check scripts/audit-demo.mjs
node --check scripts/audit-webgpu-desktop.mjs
node --check scripts/audit-webgpu-coverage-sweep.mjs
node --check scripts/audit-webgpu-depth-sweep.mjs
node --check scripts/audit-webgpu-tile-smoke.mjs
npm run audit:webgpu-tile-smoke
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5281 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5282 --probes full --webgpu-depth-alpha-mode front-top-k
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5283 --probes full --webgpu-depth-alpha-mode front-top-k --webgpu-depth-bins 16
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5284 --probes full --webgpu-depth-alpha-mode front-top-k
npm run build
uv run --extra dev pytest
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5278 --probes full
npm run audit:webgpu-desktop -- --asset nerf-lego-alpha-closure-local --port 5279 --probes full --webgpu-camera-mode spark-frame
npm run audit:webgpu-desktop -- --asset plush-semantic-closure-local --port 5280 --probes full --webgpu-camera-mode spark-frame
uv run --extra dev pytest
npm run build
node --check scripts/audit-webgpu-depth-sweep.mjs
npm run audit:webgpu-depth-sweep -- --asset nerf-lego-alpha-closure-local --bins 4,8,12,16 --port 5276 --output-dir /tmp/objgauss-webgpu-depth-sweep
npm run build
uv run --extra dev pytest
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

- 对象聚类色和部分诊断仍走 `Gaussian OIT 编辑` fallback 或 WebGPU tile route；`原始颜色（编辑预览）` 在 object edit active 后 no-SH 样例可走 Spark native compact `.splat` object mask，SH-heavy 样例保留 PLY packed route 以保存 degree-3 SH rest。剩余颗粒感主要来自 object_id 子集稀疏、边界 assignment 噪声、透明混合中被隐藏对象不再贡献，以及删除后没有补洞 / 重优化。WebGPU full runtime 内部输出、bilinear resolve、aspect-fit viewport、camera-Jacobian covariance、depth-binned alpha composite、Spark-frame camera diagnostic 和 front-top-k diagnostic 已把 coverage / sorting / color 残差拆成可审计项。当前 headless unsafe WebGPU failure 已归类为 canvas render pass / presentation backend limitation；headed desktop Chrome/WebGPU 已通过 NeRF Lego proxy、Plush 和 safe-2000 Splatfacto 的 full WebGPU tile runtime audit。
- Spark `SplatMesh.raycast` 当前可命中 splat depth，但 intersection 不暴露 splat index / object id，且不能证明 object opacity mask filter-aware；因此对象选择仍应使用已验收的 `hover-confirm-v1` screen-space pick，不能宣称已经是 renderer-native object picking。
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

1. RENDER-005T-AI: 基于已通过的 `.splat` / PLY index mapping，原型化 original source / native `.splat` object mask runtime，并复用 object-mask pixel-delta、Spark residual 和 index-mapping gates 验收；默认 coverage / depth / camera / alpha / color 参数变更必须先通过 `audit:webgpu-coverage-gate`，alpha floor 默认变更还必须先通过 `audit:webgpu-alpha-floor-candidate-gate`。
2. 将三场景 Splatfacto suite 从 smoke 推进到更高质量训练：统一训练步数、质量曲线、held-out view 指标和失败案例分析。
3. 后续 SEG: CLIP 语义命名、跨视角 SAM slot 对齐，以及与 color-mask / KMeans baseline 的质量对比。
4. 将 Poly Haven mesh -> NeRF-style render set -> Splatfacto smoke 链路升级为可审计的公开 demo 候选前，先补许可说明、质量阈值和浏览器验收。
5. 后续 renderer 优化: Spark 按需加载或拆包，降低首屏 bundle。
