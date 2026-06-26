# ObjGauss 素材库

这个素材库先解决两个问题：

- 给 ObjGauss MVP 找稳定的测试输入。
- 把 scan / mesh / images / gaussian 统一登记成同一种资产记录。

## 当前已接入

| 资产 | 来源 | 本地文件 | 用途 | 许可备注 |
| --- | --- | --- | --- | --- |
| Plush 3DGS 示例 | https://huggingface.co/cakewalk/splat-data/blob/main/plush.splat | `public/samples/plush.splat` + `public/samples/plush_objects.ply` | 快速验证真实 splat 渲染、高斯点云加载、对象聚类、删除/隔离预览 | 上游说明来源许可混合，只作为本地测试素材 |
| ObjGauss v1 闭环样例 | Plush 3DGS 示例派生产物 | `public/samples/plush_v1_objects.ply` + `outputs/demos/v1-closure/` | 当前阶段闭环验收：真实 splat、Object Field、mask 投票、对象编辑 | 继承 Plush 来源限制，仅本地测试 |
| Plush 2D 语义 Mask 闭环样例 | Plush 3DGS 示例派生产物 | `public/samples/plush_semantic.splat` + `public/samples/plush_semantic_objects.ply` + `outputs/demos/plush-semantic-closure/` | 统一验收：真实 3DGS、非 KMeans 的 2D color masks、Object Field、对象编辑 | 继承 Plush 来源限制，仅本地测试；不是 SAM/CLIP 输出 |
| Poly Haven School Chair 1K | https://polyhaven.com/a/SchoolChair_01 | `outputs/assets/raw/polyhaven-school-chair-1k/` | 许可干净的单对象 Demo 输入，后续用于 mesh 多视角渲染和 3DGS 训练 | CC0；API 拉取仅按 Poly Haven API ToS 用于非商用/研究 |
| NeRF Synthetic Lego | https://github.com/bmild/nerf | `outputs/assets/training/nerf-synthetic-lego/` | ObjGauss v1 Object Field 的多视角训练烟测 | NeRF 官方示例数据，仅训练/研究使用 |
| NeRF Lego Alpha 前景/背景诊断基线 | NeRF Synthetic Lego + 本机外部 3DGS 训练输出 | `public/samples/nerf_lego_alpha_fgbg_bg005.splat` + `public/samples/nerf_lego_alpha_fgbg_bg005_objects.ply` | 页面对比 `background_confidence=0.05` 的 Level 1 foreground/background Object Field 训练结果 | NeRF 官方示例数据，仅训练/研究使用；诊断基线，不是展示默认，也不是 part-level 稳定分离结论 |
| NeRF LLFF Fern | https://github.com/bmild/nerf | `outputs/assets/training/nerf-llff-fern/` | Lego 之外的第二个真实多视角 Splatfacto/COLMAP benchmark scene | NeRF 官方示例数据，仅训练/研究使用 |
| Poly Haven School Chair NeRF render set | https://polyhaven.com/a/SchoolChair_01 | `outputs/assets/training/polyhaven-school-chair-nerf/` | 第三个 Splatfacto-trained benchmark scene，由 CC0 glTF mesh 离线渲染多视角 RGBA | CC0；API 拉取仅按 Poly Haven API ToS 用于非商用/研究 |
| Poly Haven Chair 商用展示样例 | https://polyhaven.com/a/SchoolChair_01 | `public/samples/polyhaven_chair_demo.splat` + `public/samples/polyhaven_chair_demo_objects.ply` | 许可干净、viewer 可直接加载和对象编辑的 Gaussian demo sample | CC0 派生训练输出；public sample 文件本地生成，不提交 git |

处理链路：

```text
plush.splat
  -> public/samples/plush.splat
  -> Spark 真实 3DGS renderer

plush.splat
  -> objgauss convert-splat
  -> plush.ply
  -> objgauss cluster --clusters 6
  -> public/samples/plush_objects.ply
  -> 点云编辑 fallback
```

当前前端默认优先用 `.splat` 进入真实 renderer；切换对象聚类色、隐藏、隔离或删除预览时，使用 PLY 内部 `red/green/blue` 与 `object_id` 进入点云编辑 fallback。

一键拉取当前样例：

```bash
objgauss assets list --pullable
objgauss assets pull plush-3dgs-local
```

默认输出：

```text
outputs/assets/raw/plush.splat
outputs/assets/converted/plush.ply
public/samples/plush.splat
public/samples/plush_objects.ply
```

如果远端文件或转换逻辑有更新，可以强制刷新：

```bash
objgauss assets pull plush-3dgs-local --force
```

闭环验收：

```bash
objgauss demo v1-closure
objgauss demo verify-v1-closure
objgauss demo plush-semantic-closure
objgauss demo verify-plush-semantic-closure
```

默认输出：

```text
outputs/demos/v1-closure/v1-closure-manifest.json
outputs/demos/v1-closure/mask-manifest.json
outputs/demos/v1-closure/object_field_trained.npz
outputs/demos/v1-closure/plush_v1_objects.ply
public/samples/plush_v1_objects.ply
public/samples/plush_semantic.splat
public/samples/plush_semantic_objects.ply
```

`plush-semantic-closure` 使用原始 `outputs/assets/converted/plush.ply`，
不读取旧 KMeans `object_id`。它把真实 Plush 3DGS 场景投影成 2D 视图，
按颜色语义生成 `red-subject`、`straw-frame`、`dark-detail`、
`other-surface` 四类 mask，再训练 Object Field 并导出带 `object_id`
的 PLY。这个样例用于证明“真实 3DGS 可被 2D mask 语义线索对象级寻址”，
但不声称替代 SAM / CLIP。

### Poly Haven School Chair 1K

处理链路：

```text
Poly Haven API: files/SchoolChair_01
  -> glTF 1K entrypoint + .bin + textures
  -> outputs/assets/raw/polyhaven-school-chair-1k/
  -> outputs/assets/converted/polyhaven-school-chair-1k/asset-manifest.json
```

一键拉取：

```bash
objgauss assets pull polyhaven-school-chair-1k
```

默认输出：

```text
outputs/assets/raw/polyhaven-school-chair-1k/SchoolChair_01_1k.gltf
outputs/assets/raw/polyhaven-school-chair-1k/SchoolChair_01.bin
outputs/assets/raw/polyhaven-school-chair-1k/textures/
outputs/assets/converted/polyhaven-school-chair-1k/asset-manifest.json
```

当前它是 mesh Demo 输入源，还不能直接进入现有 3DGS viewer。下一步转换链路是：

```text
glTF mesh
  -> 多视角离线渲染
  -> 3DGS 训练
  -> point_cloud.ply / .splat
  -> objgauss cluster
  -> public/samples/<demo>_objects.ply
```

### NeRF Synthetic Lego

处理链路：

```text
NeRF 官方 nerf_example_data.zip
  -> 只抽取 nerf_synthetic/lego
  -> outputs/assets/training/nerf-synthetic-lego/
  -> outputs/assets/converted/nerf-synthetic-lego/training-manifest.json
```

一键拉取：

```bash
objgauss assets pull nerf-synthetic-lego
```

默认输出：

```text
outputs/assets/raw/nerf_example_data.zip
outputs/assets/training/nerf-synthetic-lego/transforms_train.json
outputs/assets/training/nerf-synthetic-lego/transforms_test.json
outputs/assets/training/nerf-synthetic-lego/train/
outputs/assets/training/nerf-synthetic-lego/test/
outputs/assets/converted/nerf-synthetic-lego/training-manifest.json
```

闭环代理验收：

```bash
objgauss demo lego-alpha-closure
objgauss demo verify-lego-alpha-closure
```

独立生成真实 2D color mask manifest：

```bash
objgauss masks from-nerf-rgba-colors outputs/assets/training/nerf-synthetic-lego \
  --output outputs/masks/nerf-lego-rgba-colors/mask-manifest.json \
  --split train \
  --max-frames 8 \
  --alpha-threshold 16
```

登记外部 3DGS 训练器输出：

```bash
objgauss training register-output path/to/point_cloud.ply \
  --asset-id nerf-lego-trained-output-local \
  --output-dir outputs/assets/gaussians/nerf-lego-trained \
  --dataset outputs/assets/training/nerf-synthetic-lego \
  --masks outputs/masks/nerf-lego-rgba-colors/mask-manifest.json \
  --public-name nerf_lego_trained \
  --iterations 120 \
  --learning-rate 1.0
```

该命令只负责接入成熟训练器产物，不在 ObjGauss 内部训练 3DGS。
默认会生成 viewer `.splat`、Object Field、带 `object_id` 的 PLY 和
`training-output-manifest.json`。

Alpha foreground/background 本地页面预览：

```text
outputs/assets/gaussians/nerf-lego-alpha-fgbg-bg005-v2/gaussians.splat
  -> public/samples/nerf_lego_alpha_fgbg_bg005.splat

outputs/assets/gaussians/nerf-lego-alpha-fgbg-bg005-v2/object_aware_gaussians.ply
  -> public/samples/nerf_lego_alpha_fgbg_bg005_objects.ply
```

这组样例使用 `background_confidence=0.05` 的 alpha fgbg mask bundle：
168,653 Gaussians，foreground/background `object_id` counts 为 `35,579/133,074`。
它用于查看 Level 1 数据基线，不证明 part-level 对象稳定分离。

默认输出：

```text
outputs/demos/lego-alpha-closure/lego-alpha-closure-manifest.json
outputs/demos/lego-alpha-closure/mask-manifest.json
outputs/demos/lego-alpha-closure/object_field_trained.npz
outputs/demos/lego-alpha-closure/lego_proxy.splat
outputs/demos/lego-alpha-closure/lego_v1_objects.ply
public/samples/lego_alpha_proxy.splat
public/samples/lego_alpha_v1_objects.ply
```

该链路从 NeRF Lego 真实多视角 RGBA 和 pose 生成轻量 Gaussian proxy，
再用 2D color masks 投票更新 Object Field。它用于把 v1 闭环压到同一个
Lego 画面里验收，不等价于完整 3DGS 训练输出。

### NeRF LLFF Fern

处理链路：

```text
NeRF 官方 nerf_example_data.zip
  -> 抽取 nerf_llff_data/fern
  -> outputs/assets/training/nerf-llff-fern/
  -> 从 COLMAP sparse/0 生成 transforms_train.json
  -> Nerfstudio Splatfacto colmap dataparser
  -> Object Field / SAM benchmark
```

一键拉取：

```bash
objgauss assets pull nerf-llff-fern
```

默认输出：

```text
outputs/assets/training/nerf-llff-fern/images/
outputs/assets/training/nerf-llff-fern/sparse/0/
outputs/assets/training/nerf-llff-fern/transforms_train.json
outputs/assets/converted/nerf-llff-fern/training-manifest.json
```

Fern 不是前端 public sample；它用于 `docs/benchmarks/splatfacto-scenes.json`
里的第二个真实多视角 Splatfacto scene。训练和 benchmark 命令见
`docs/benchmarks/splatfacto-scenes.md`。

### Poly Haven School Chair NeRF Render Set

处理链路：

```text
Poly Haven School Chair glTF
  -> objgauss.mesh_nerf 纯 Python/NumPy 离线 rasterizer
  -> outputs/assets/training/polyhaven-school-chair-nerf/train/*.png
  -> outputs/assets/training/polyhaven-school-chair-nerf/transforms_train.json
  -> Nerfstudio Splatfacto blender-data dataparser
  -> Object Field / SAM benchmark
```

一键生成训练输入：

```bash
objgauss assets pull polyhaven-school-chair-nerf
```

默认输出：

```text
outputs/assets/training/polyhaven-school-chair-nerf/train/
outputs/assets/training/polyhaven-school-chair-nerf/transforms_train.json
outputs/assets/converted/polyhaven-school-chair-nerf/training-manifest.json
```

该数据集不是前端 public sample；它用于 `docs/benchmarks/splatfacto-scenes.json`
里的第三个 Splatfacto-trained scene row。当前 renderer 是确定性 mesh
rasterizer，用于生成可复现训练图像；它不是现实相机采集数据，也不替代
真实 3DGS / Spark renderer。

### Poly Haven Chair Commercial Demo Sample

处理链路：

```text
Poly Haven School Chair NeRF render set
  -> Nerfstudio Splatfacto smoke training
  -> Object Field / SAM mask vote
  -> object_aware_gaussians.ply + gaussians.splat
  -> public/samples/polyhaven_chair_demo_objects.ply
  -> public/samples/polyhaven_chair_demo.splat
```

发布到本地 viewer 样例目录：

```bash
npm run publish:polyhaven-chair-demo
```

默认输入：

```text
outputs/assets/gaussians/polyhaven-chair-splatfacto-smoke-sam8f-slots6-benchmark/gaussians.splat
outputs/assets/gaussians/polyhaven-chair-splatfacto-smoke-sam8f-slots6-benchmark/object_aware_gaussians.ply
```

默认输出：

```text
public/samples/polyhaven_chair_demo.splat
public/samples/polyhaven_chair_demo_objects.ply
/tmp/objgauss-polyhaven-chair-demo-publish/summary.json
/tmp/objgauss-polyhaven-chair-demo-publish/summary.md
```

该样例登记为 `polyhaven-chair-commercial-demo-local`，用于商业展示路线、
对象隔离和删除预览。`public/samples/*.splat` 与 `*.ply` 是本地生成
ignored 文件，不进入 git；fresh clone 需要先生成或复制训练输出，再运行
上面的 publish 命令。

浏览器闭环验收：

```bash
npm run audit:demo
```

该命令会启动临时 Vite 服务，分别加载 `ObjGauss v1 闭环样例` 和
`NeRF Lego 闭环代理样例`，检查真实 splat / 点云编辑 canvas 非空，并执行
对象选择、只看所选和预览删除。

完整本地验收：

```bash
npm run acceptance:demo
```

该命令会重新生成并验证 Plush v1 closure、重新生成并验证 NeRF Lego proxy
closure，然后执行浏览器闭环验收。

## 优先素材来源

| 优先级 | 来源 | 类型 | 适合用途 | 入口 |
| --- | --- | --- | --- | --- |
| P0 | ARKitScenes | 真实室内 scan | 手机 LiDAR 房间、家具对象化、真实用户输入形态 | https://github.com/apple/ARKitScenes |
| P0 | NeRF Synthetic Lego | 多视角合成图像 + pose | ObjGauss v1 Object Field 训练烟测 | https://github.com/bmild/nerf |
| P0 | NeRF LLFF Fern | 真实多视角图像 + COLMAP | 跨 Splatfacto scene benchmark | https://github.com/bmild/nerf |
| P0 | Poly Haven School Chair NeRF render set | CC0 mesh 派生多视角图像 + pose | 第三个 Splatfacto scene benchmark | https://polyhaven.com/a/SchoolChair_01 |
| P0 | OmniObject3D | 对象级 scan / mesh / point cloud | 单个真实扫描物体，高质量对象编辑实验 | https://omniobject3d.github.io/ |
| P0 | Poly Haven | CC0 mesh / texture / HDRI | 展示 demo、开源项目可复现素材 | https://polyhaven.com/models |
| P1 | ScanNet | 真实室内 scan + 语义/实例标注 | 场景到对象分组验证 | https://www.scan-net.org/ |
| P1 | Google Scanned Objects | 日用品 mesh / SDF / collider | 对象编辑、物理代理、机器人仿真 | https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/ |
| P1 | Mip-NeRF 360 | 多视角图像 + COLMAP | 训练 3DGS、渲染质量 benchmark | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ |
| P2 | Tanks and Temples | 真实场景重建 benchmark | 后期复杂场景评估 | https://www.tanksandtemples.org/ |

## 训练 / Demo 分层

素材进入本地后分成两条线处理：

| 层级 | 目录 | 作用 | 可进公开 demo |
| --- | --- | --- | --- |
| 原始素材 | `outputs/assets/raw/` | 下载后的源文件，保持原样，便于复现 | 否 |
| 转换中间产物 | `outputs/assets/converted/` | `.splat -> .ply`、mesh 采样点云、COLMAP 整理结果 | 否 |
| 训练素材 | `outputs/assets/training/<asset_id>/` | 多视角图像、相机位姿、COLMAP sparse、mesh、点云 | 通常否 |
| 训练输出 | `outputs/assets/gaussians/<asset_id>/` | 训练得到的 3DGS `point_cloud.ply` / checkpoint | 视许可而定 |
| Demo 样例 | `public/samples/` | 小型、已脱敏、许可明确或仅本地测试的 ObjGauss PLY | 可以，但必须先过许可检查 |

判断规则：

- **训练源**：ARKitScenes、ScanNet、OmniObject3D、Mip-NeRF 360 这类数据主要用于训练和评估，不默认进入公开 demo。
- **Demo 素材**：Poly Haven 这类许可干净的资产优先用于可公开展示。
- **v1 训练烟测**：NeRF Synthetic Lego 已自动抽取到训练目录，优先用于 Object Field 多视角一致性验证。
- **Demo 可用**：当前 `Plush 3DGS 示例` 只作为本地 demo 和管线烟测，不作为商用或公开发布素材。
- **训练输出**：即使模型是我们训练的，也要继承原始数据许可，不能自动视为可商用。

## 统一资产格式

素材库里的每个条目应能映射成：

```text
ObjGaussAsset
├── id
├── name
├── category
├── source_type: scan | mesh | images | gaussian
├── status: 已接入 | 候选 | 下载中 | 已转换
├── pipeline_stage: 来源登记 | 训练源 | Demo 素材 | Demo 可用
├── use_cases: list[str]
├── source_url
├── local_path: optional
├── splat_path: optional
├── mesh: optional
├── point_cloud: optional
├── gaussians: optional
├── images: optional
├── camera_poses: optional
├── semantic_label: optional
├── object_id: optional
├── bbox: optional
├── collider_proxy: optional
└── license
```

前端当前消费的是 `src/assetLibrary.js`，其中 `localPath` 存在的条目可以直接加载。
CLI 当前消费的是 `objgauss/assets.py`，其中 `pull_pipeline` 存在的条目可以自动拉取到本地。

## 转换规范

场景扫描：

```text
RGB-D / posed images / mesh
  -> COLMAP 或数据集自带 pose
  -> 训练 3DGS
  -> point_cloud.ply
  -> objgauss cluster
  -> ObjGauss PLY with object_id
```

对象 mesh：

```text
mesh + texture
  -> asset-manifest.json
  -> 采样 point cloud
  -> 可选：多视角渲染 + 训练 3DGS
  -> objgauss cluster / 手工对象 id
  -> ObjGauss PLY
```

NeRF / 3DGS 训练图像：

```text
posed images + transforms_*.json
  -> outputs/assets/training/<asset_id>/
  -> 训练 3DGS / Object Field
  -> point_cloud.ply / .splat
```

已有 splat：

```text
.splat
  -> objgauss convert-splat
  -> PLY
  -> objgauss cluster
  -> ObjGauss PLY
```

## 存储规则

- 小型、可直接打开的预览样例放在 `public/samples/`。
- 下载原始数据、转换中间产物放在 `outputs/assets/raw/` 和 `outputs/assets/converted/`。
- 训练集整理结果放在 `outputs/assets/training/<asset_id>/`。
- 训练出的高斯模型放在 `outputs/assets/gaussians/<asset_id>/`。
- 大型数据集不要提交到仓库。
- 每个外部来源必须记录 `source_url`、`license`、下载日期和转换命令。
- 许可不清楚的素材只能标为本地测试，不进入公开 demo 或发布包。

## 新增一个可自动拉取素材

1. 在 `objgauss/assets.py` 注册 `AssetSource`，至少填 `id`、`name`、`source_url`、`download_url`、`raw_file_name`、`output_file_name` 和 `pull_pipeline`；viewer 样例再填 `local_path` / `splat_path`。
2. 在 `src/assetLibrary.js` 增加对应前端卡片。
3. 如果是 `.splat`，优先复用 `splat-to-objgauss-ply` 管线。
4. 如果是 mesh / RGB-D / COLMAP 数据，先写转换脚本，再把管线挂到 `objgauss assets pull`。
5. 跑验证：

```bash
uv run --extra dev pytest
npm run build
```
