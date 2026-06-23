# ObjGauss 当前状态总览

> 最近更新: 2026-06-23

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
  - Three.js 高斯中心 soft sprite 编辑 fallback。
  - 已拆分 `真实查看` / `对象编辑` 两个工作模式；对象操作会显式进入点云编辑预览，不再伪装成真实 splat 内直接编辑。
  - 自身颜色 / 对象聚类色切换。
  - 对象列表、点击软点云画布选中对象、隔离、删除预览；选中对象在软点云编辑模式下有高亮层，删除预览会退出隔离并切回自身颜色显示剩余整体场景。
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
  - `objgauss demo audit-v1-goal --allow-incomplete` 已固化为阶段目标完成度审计命令。
  - baseline commit: `c8dcef7`.

## 最近验证

2026-06-23:

```bash
npm run build
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
```

结果：

- WEB-001 UX repair: 顶部假工具按钮已替换为真实 `真实查看` / `对象编辑` 模式；无 `.splat` 场景默认对象编辑，带 `.splat` 样例加载后默认真实查看；对象可通过右侧列表或软点云画布选中，选择后会进入对象编辑并显示高亮层；对象编辑 renderer 从硬点改为软圆形 sprite splat 近似，但仍不是完整 covariance-aware 3DGS shader；删除预览会退出 `只看所选` 隔离并切回 `自身颜色`，显示删除后的剩余整体场景；素材库只显示 5 个可加载样例；Benchmark tab 显示 smoke/candidate/paper pass 和 Lego/Fern/Chair 三场景指标。
- WEB-001 validation: `npm run build` 通过；`npm run audit:demo -- --asset plush-v1-closure-local --url http://127.0.0.1:5191/` 通过并检查 `canvasSelectedObject=1`、`editPixels=64530`、`visibleAfterDelete=230,581`、`renderModeAfterDelete="自身颜色"`；targeted Playwright QA 保存到 `/tmp/objgauss-web001-qa/`、`/tmp/objgauss-canvas-select-*.png` 和 `/tmp/objgauss-audit-plush-v1-closure-local.png`，断言覆盖初始模式、真实 splat 加载、软点云画布选中、对象编辑预览、删除后自身颜色、Benchmark tab 和移动端截图；console 仅有 WebGL ReadPixels performance warning。
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

- 对象聚类色、隐藏、隔离、删除预览仍通过点云编辑 fallback 完成，不是对象级 splat shader。
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

1. 将三场景 Splatfacto suite 从 smoke 推进到更高质量训练：统一训练步数、质量曲线、held-out view 指标和失败案例分析。
2. 后续 SEG: CLIP 语义命名、跨视角 SAM slot 对齐，以及与 color-mask / KMeans baseline 的质量对比。
3. 将 Poly Haven mesh -> NeRF-style render set -> Splatfacto smoke 链路升级为可审计的公开 demo 候选前，先补许可说明、质量阈值和浏览器验收。
4. 后续 renderer 优化: Spark 按需加载或拆包，降低首屏 bundle。
