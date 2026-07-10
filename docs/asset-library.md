# ObjGauss 素材库

> **Development-stage assets / 开发阶段资产。** 本素材库记录研究原型、
> 训练 smoke、诊断样例和 handoff 资产。除非单独标注为 stable release 或
> commercial demo，否则这里的本地 `outputs/`、HF 资产和 public sample
> 说明都只代表开发阶段状态，文件布局、指标和模型产物可能继续变化。

这个素材库先解决两个问题：

- 给 ObjGauss MVP 找稳定的测试输入。
- 把 scan / mesh / images / gaussian 统一登记成同一种资产记录。

## 当前已接入

| 资产 | 来源 | 本地文件 | 用途 | 许可备注 |
| --- | --- | --- | --- | --- |
| Plush 3DGS 示例 | https://huggingface.co/cakewalk/splat-data/blob/main/plush.splat | `public/samples/plush.splat` + `public/samples/plush_objects.ply` | 快速验证真实 splat 渲染、高斯点云加载、对象聚类、删除/隔离预览 | 上游说明来源许可混合，只作为本地测试素材 |
| Nike 真实 3DGS 示例 | https://huggingface.co/cakewalk/splat-data/blob/main/nike.splat | `public/samples/nike.splat` + `public/samples/nike_objects.ply` | 小体积真实 Gaussian cloud demo，用于整理后的 viewer 入口和对象编辑 smoke | 上游说明来源许可混合，只作为本地测试素材 |
| ObjGauss v1 闭环样例 | Plush 3DGS 示例派生产物 | `public/samples/plush_v1_objects.ply` + `outputs/demos/v1-closure/` | 当前阶段闭环验收：真实 splat、Object Field、mask 投票、对象编辑 | 继承 Plush 来源限制，仅本地测试 |
| Plush 2D 语义 Mask 闭环样例 | Plush 3DGS 示例派生产物 | `public/samples/plush_semantic.splat` + `public/samples/plush_semantic_objects.ply` + `outputs/demos/plush-semantic-closure/` | 统一验收：真实 3DGS、非 KMeans 的 2D color masks、Object Field、对象编辑 | 继承 Plush 来源限制，仅本地测试；不是 SAM/CLIP 输出 |
| Poly Haven School Chair 1K | https://polyhaven.com/a/SchoolChair_01 | `outputs/assets/raw/polyhaven-school-chair-1k/` | 许可干净的单对象 Demo 输入，后续用于 mesh 多视角渲染和 3DGS 训练 | CC0；API 拉取仅按 Poly Haven API ToS 用于非商用/研究 |
| NeRF Synthetic Lego | https://github.com/bmild/nerf | `outputs/assets/training/nerf-synthetic-lego/` | ObjGauss v1 Object Field 的多视角训练烟测 | NeRF 官方示例数据，仅训练/研究使用 |
| NeRF Lego Alpha 前景/背景诊断基线 | NeRF Synthetic Lego + 本机外部 3DGS 训练输出 | `public/samples/nerf_lego_alpha_fgbg_bg005.splat` + `public/samples/nerf_lego_alpha_fgbg_bg005_objects.ply` | 页面对比 `background_confidence=0.05` 的 Level 1 foreground/background Object Field 训练结果 | NeRF 官方示例数据，仅训练/研究使用；诊断基线，不是展示默认，也不是 part-level 稳定分离结论 |
| NeRF LLFF Fern | https://github.com/bmild/nerf | `outputs/assets/training/nerf-llff-fern/` | Lego 之外的第二个真实多视角 Splatfacto/COLMAP benchmark scene | NeRF 官方示例数据，仅训练/研究使用 |
| Poly Haven School Chair NeRF render set | https://polyhaven.com/a/SchoolChair_01 | `outputs/assets/training/polyhaven-school-chair-nerf/` | 第三个 Splatfacto-trained benchmark scene，由 CC0 glTF mesh 离线渲染多视角 RGBA | CC0；API 拉取仅按 Poly Haven API ToS 用于非商用/研究 |
| Poly Haven School Chair dense NeRF render set | https://polyhaven.com/a/SchoolChair_01 | `outputs/assets/training/polyhaven-school-chair-nerf-dense/` | 更高密度训练输入：32-frame / 384px CC0 glTF orbit render，用于后续生成更好的 Splatfacto candidate | CC0；API 拉取仅按 Poly Haven API ToS 用于非商用/研究 |
| Poly Haven Chair 商用展示样例 | https://polyhaven.com/a/SchoolChair_01 | `public/samples/polyhaven_chair_demo.splat` + `public/samples/polyhaven_chair_demo_objects.ply` | 许可干净、viewer 可直接加载和对象编辑的 Gaussian demo sample | CC0 派生训练输出；public sample 文件本地生成，不提交 git |

## ObjectState 真实证据资产（仅本地）

| 资产 | 来源 | 本地文件 | 用途 | 许可 / 边界 |
| --- | --- | --- | --- | --- |
| BOP HOPE val Realsense | https://bop.felk.cvut.cz/datasets/ | `outputs/assets/raw/bop-hope/hope_val_realsense.zip`、`outputs/assets/raw/bop-hope/hope-val-realsense-subset/val/000001/`、`val/000002/` | public RGB-D pose replay；identity / prediction 负证据与 baseline | ledger 按 CC BY-SA 4.0 登记，重分发前仍需核对上游条款；无 action GT，不能进入 intervention pass |
| BOP LMO test BOP19 | https://bop.felk.cvut.cz/datasets/ | `outputs/assets/raw/bop-lmo/lmo_test_bop19.zip`、`outputs/assets/raw/bop-lmo/lmo-test-bop19-subset/test/000002/` | public RGB-D pose replay；identity / prediction 负证据与 baseline | ledger 按 CC BY-SA 4.0 登记，重分发前仍需核对上游条款；无 action GT，不能进入 intervention pass |
| RBO Articulated Objects local subset | https://doi.org/10.5281/zenodo.1036660 | `outputs/assets/raw/rbo-articulated-objects/`：官方 index、3 个 interaction archive 与 3 个 companion model archive | 筛选 RGB-D、MoCap link 6DoF、camera motion 与 100 Hz F/T wrench 同时存在的 interaction；2026-07-10 本地下载并通过大小、官方 MD5 与 tar 完整性校验 | CC BY 4.0；wrench 是 human/tool-applied measurement，不是 controller command；尚未验证遮挡或形成 gate row |
| RRC 2020 local subset | https://people.tuebingen.mpg.de/mpi-is-software/data/rrc2020/ | `outputs/assets/raw/rrc2020/`：官方 SQLite index、query helper 与 `7969.zip`、`8076.zip`、`9505.zip` | 筛选同一 robot、Phase 2 Level 4、持续抬升/位移/到达目标的真实控制 run；2026-07-10 本地下载并通过固定大小与 ZIP 完整性校验 | CC BY-NC-SA 4.0；pose 是视觉 tracker estimate，原生 action 是 9D robot control，不能直接冒充现行 3D action vector |

HOPE scene `000002` 的当前最小复跑输入只抽取前 3 帧：

```bash
unzip -o outputs/assets/raw/bop-hope/hope_val_realsense.zip \
  'val/000002/scene_*.json' \
  'val/000002/rgb/00000[0-2].png' \
  'val/000002/depth/00000[0-2].png' \
  -d outputs/assets/raw/bop-hope/hope-val-realsense-subset/

uv run objgauss object-state bop-rgbd-baseline-local-row-handoff \
  outputs/assets/raw/bop-hope/hope-val-realsense-subset/val/000002 \
  --output-root outputs/evidence/objectstate-bop-hope-public-000002-rgbd-baseline \
  --summary-output outputs/evidence/objectstate-bop-hope-public-000002-rgbd-baseline/bop-rgbd-baseline-local-row-summary.json \
  --sample-id bop-hope-val-scene-000002-rgbd-baseline \
  --dataset-id bop-hope \
  --object-category hope-objects \
  --scenario public-replay-pose-sequence \
  --license-text 'BOP HOPE CC-BY-SA-4.0; local research evidence only' \
  --max-frames 3 \
  --identity-policy pose_track_per_obj_id \
  --max-points-per-frame 10000 \
  --force
```

该结果是第三个 pose replay 的局部负证据，不是完整 controlled scene：identity
reviewability 为 fail，prediction 与 hold-last 同为零误差，且没有真实 action。
`outputs/assets/` 与 `outputs/evidence/` 继续 ignored，不提交 git。

### RBO / RRC 最小 acquisition 子集

本地下载与复核命令：

```bash
scripts/download-objectstate-evidence-subsets.sh --list
scripts/download-objectstate-evidence-subsets.sh --download --dataset all
scripts/download-objectstate-evidence-subsets.sh --verify-only --dataset all
```

脚本固定下载以下候选，不拉取 RBO 122.5 GB 全库或 RRC 全量：

| 数据集 | 候选 | 选择依据 | payload |
| --- | --- | --- | ---: |
| RBO | `cardboardbox22`、`tripod25`、`cabinet20`，以及三个 companion object model | 三条均为 `Camera Motion=1`、`Small Interaction=0`、F/T=1、空 warning comment；覆盖 natural / artificial / dark lighting | `868,060,043` bytes |
| RRC 2020 | Zarr jobs `7969`、`8076`、`9505` | 同属 `roboch1`、Phase 2 Level 4；`max_height_30 >= 0.08m`、`furthest_from_start_30 >= 0.12m`、`min_distance_to_goal_30 <= 0.01m` 且 reward 优于 baseline | `910,383,139` bytes |

总 payload 为 `1,778,443,182` bytes，约 1.66 GiB。脚本使用 `.part` 文件断点续传，
按 Zenodo 官方 MD5 校验 RBO，并按固定字节数与 ZIP 完整性校验 RRC。所有 payload
继续只写入 ignored `outputs/assets/raw/`，不解包到 `public/`。

2026-07-10 已完成 `--verify-only --dataset all`：六个 interaction/run archive 与三个
RBO companion model archive 均通过，且没有残留 `.part` 文件。该结果只证明下载完整，
不证明字段语义或 reality gate 合格。

这些条目仍只是 acquisition candidates。解包后必须独立检查 timestamp overlap、完整
link/object pose、实际 camera displacement、visible → occluded → visible、非零 wrench /
desired-applied action 与 pose transition 重叠。RBO wrench 需要 bias/gravity compensation
和坐标变换；RRC 9D control 不得静默降维或用未来 object pose delta 伪造成 3D action。

外部 controlled-interaction 候选：

- [H2O](https://taeinkwon.com/projects/h2o/)：同步多视角 RGB-D、interaction label、
  物体 6DoF、相机和手 pose。作者页现已链接
  [ETH Research Collection](https://doi.org/10.3929/ethz-b-000685070) 的匿名 Open Access /
  CC BY-NC 4.0 副本；最小 RGB-D 分卷约 13.57 GB。旧注册站的 academic-only /
  no-transfer 条款只适用于旧入口。当前未下载；原生 action 是语义标签，不是独立 3D
  control vector，不能直接进入 intervention gate。
- [HOI4D](https://hoi4d.github.io/)：RGB-D、category-level object pose、hand action，
  CC BY-NC 4.0；action annotation 只有类别与时间区间，没有独立 3D control vector。
  当前只有官方分卷入口，没有本地 raw sequence。

这些 evidence-only 数据集不进入 `src/assetLibrary.js` 或 `objgauss/assets.py` 的浏览器
Demo registry；只有许可、体积和发布用途单独确认后才允许同步到公开素材面。

## 现成 Gaussian 场景候选

> 本节只登记候选，不代表已经下载、训练、发布或进入 viewer 默认。所有大文件、
> 许可不清楚文件和第三方 `.splat` / `.ply` 默认只能放在 ignored `outputs/` 做本地审计。

| 候选 | 来源 | 体积 / 文件 | 优先级 | 适用用途 | 边界 |
| --- | --- | --- | --- | --- | --- |
| cakewalk room | https://huggingface.co/cakewalk/splat-data/tree/main | `room.splat`，约 51 MB；registry id `cakewalk-room-3dgs-local` | P0 | 现成室内 Gaussian scene；适合本地 object clustering、viewer load、cross-sample 静态场景扩展 | cakewalk README 明确来自多来源、多许可；只能本地测试，不作为 public demo / commercial demo |
| cakewalk train | https://huggingface.co/cakewalk/splat-data/tree/main | `train.splat`，约 32.8 MB；registry id `cakewalk-train-3dgs-local` | P0 | 小型现成 `.splat`；适合快速 pipeline smoke、source splat 加载和对象层生成实验 | 许可混合；无 RGB / pose / action GT，不能用于 State Variable Gate pass row |
| cakewalk truck | https://huggingface.co/cakewalk/splat-data/tree/main | `truck.splat`，约 81.3 MB；registry id `cakewalk-truck-3dgs-local` | P1 | 单一车辆主体，适合测试 object proposal / bbox picking / static segmentation 质量 | 许可混合；只能本地审计，不进入默认 viewer/export |
| cakewalk garden | https://huggingface.co/cakewalk/splat-data/tree/main | `garden.splat` 约 187 MB；registry id `cakewalk-garden-3dgs-local` | P1 | Mip-NeRF360 风格户外静态场景；适合 renderer / LOD 压力、object-field robustness 和 cross-sample 负例 | 文件较大且许可混合；不适合先做 Phase 1 identity / action evidence |
| cakewalk bicycle | https://huggingface.co/cakewalk/splat-data/tree/main | `bicycle.splat` 约 196 MB；registry id `cakewalk-bicycle-3dgs-local` | P1 | 复杂前景 / 背景静态场景；适合 bbox picking、object proposal 和 renderer pressure smoke | 文件较大且许可混合；只能本地审计，不进入默认 viewer/export |
| cakewalk stump | https://huggingface.co/cakewalk/splat-data/tree/main | `stump.splat` 约 159 MB；registry id `cakewalk-stump-3dgs-local` | P2 | 自然物体 / 背景混合场景，用于 object proposal 失败分析和 segmentation robustness 负例 | 文件较大且许可混合；不能替代真实 identity / pose / action GT |
| cakewalk treehill | https://huggingface.co/cakewalk/splat-data/tree/main | `treehill.splat` 约 121 MB；registry id `cakewalk-treehill-3dgs-local` | P2 | 大面积自然场景；适合 LOD / streaming pressure 和 no-object static negative evidence | 文件较大且许可混合；不能用于 State Variable Gate pass row |
| GraphDECO official results | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ | 官方页面提供 `Results - 7GB` | P1 | 高质量官方 3DGS 结果；适合静态 Gaussian scene benchmark 和 renderer / object-field robustness 审计 | 大文件；继承 Mip-NeRF360 / Tanks and Temples / Deep Blending 等原数据条款；不直接提供 action GT |
| GraphDECO official scenes | https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ | 官方页面提供 `Scenes - 650MB` | P2 | 若需要重新训练或对齐相机轨迹，可作为官方 scene 输入来源 | 是训练/评估源，不是 ready `.splat` 小样例；下载后仍需放 ignored `outputs/` |
| Inria Hierarchical 3DGS datasets | https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/ | 官方页面提供 `Datasets` 下载入口 | P2 | 大规模分层 3DGS / LOD / streaming 研究素材，用于后期 renderer route 压力和层级高斯路线评估 | 体量大；不是小型 ready viewer sample，也不提供 state-variable action / counterfactual GT |
| cakewalk Lumix Gaussian PLY | https://huggingface.co/cakewalk/splat-data/tree/main | `gs_FF3_lumix_4k 3.ply`，约 263 MB | P2 | 现成 Gaussian PLY；适合后续验证 PLY import / object clustering 是否能吃第三方大体量 Gaussian PLY | 当前 assets pull 自动化只稳定支持 `.splat -> ObjGauss PLY`；文件名含空格且体量较大，先记录候选，不进入 registry |
| GaussianSplats3D sample archive | https://projects.markkellogg.org/downloads/gaussian_splat_data.zip | demo sample archive，约 622 MB，格式需本地解包确认 | P2 | 可作为 Three.js / browser Gaussian renderer reference 场景来源，尤其适合对比 `.ksplat` / viewer performance 路线 | 当前 ObjGauss 没有 `.ksplat` import contract；只作格式适配候选，不作为可拉取 asset |
| Niantic SPZ samples | https://github.com/nianticlabs/spz/tree/main/samples | `.spz` sample files，单文件约十几到二十几 MB | P2 | 小体积 compressed Gaussian candidate，可用于后续评估 SPZ decoder / converter 是否值得接入 | 当前 ObjGauss 没有 `.spz` decoder；新增该格式属于单独标准 PR，不能直接进入 viewer / registry |

推荐顺序：

1. 先本地拉 `room.splat` 和 `train.splat` 做小型静态 cross-sample；它们比
   `garden` / `bicycle` 更适合快速验收。`room` / `train` 已有非默认 viewer
   catalog 入口；只有执行对应 `objgauss assets pull ...` 并生成本地 ignored
   `public/samples/*` 后才会在 viewer 中加载成功。
2. 若要评估真实大场景 renderer / LOD / streaming，优先拉 `garden.splat` 或
   `bicycle.splat`，再考虑 `stump` / `treehill`。
3. 若要做官方高质量结果审计，再考虑 GraphDECO results 或 Inria Hierarchical
   3DGS datasets；这些属于大体量本地审计，不适合作为首个小样例。
4. 若要探索新格式，先本地审计 cakewalk Gaussian PLY、GaussianSplats3D archive 或
   Niantic SPZ samples，再单独立项做 `.ply` / `.ksplat` / `.spz` import contract；
   不把未支持格式硬接进现有 `.splat` registry。
5. 若目标是证明 `ObjectState` 是真实状态变量，现成静态 Gaussian scene 只能提供
   reconstruction noise / segmentation negative evidence；仍必须另找带 timestamped
   physical identity、pose 和 action 的 controlled/public capture。

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

nike.splat
  -> outputs/assets/raw/nike.splat
  -> outputs/assets/converted/nike.ply
  -> public/samples/nike.splat
  -> public/samples/nike_objects.ply
  -> 整理后的本地真实 Gaussian demo
```

当前前端默认优先用 `.splat` 进入真实 renderer；切换对象聚类色、隐藏、隔离或删除预览时，使用 PLY 内部 `red/green/blue` 与 `object_id` 进入点云编辑 fallback。

一键拉取当前样例：

```bash
objgauss assets list --pullable
objgauss assets pull plush-3dgs-local
objgauss assets pull nike-3dgs-local
```

一键拉取现成 Gaussian 场景候选：

```bash
objgauss assets pull cakewalk-room-3dgs-local
objgauss assets pull cakewalk-train-3dgs-local
objgauss assets pull cakewalk-truck-3dgs-local
objgauss assets pull cakewalk-garden-3dgs-local
objgauss assets pull cakewalk-bicycle-3dgs-local
objgauss assets pull cakewalk-stump-3dgs-local
objgauss assets pull cakewalk-treehill-3dgs-local
```

默认输出：

```text
outputs/assets/raw/plush.splat
outputs/assets/converted/plush.ply
public/samples/plush.splat
public/samples/plush_objects.ply
outputs/assets/raw/nike.splat
outputs/assets/converted/nike.ply
public/samples/nike.splat
public/samples/nike_objects.ply
outputs/assets/raw/room.splat
outputs/assets/converted/room.ply
public/samples/room.splat
public/samples/room_objects.ply
outputs/assets/raw/train.splat
outputs/assets/converted/train.ply
public/samples/train.splat
public/samples/train_objects.ply
outputs/assets/raw/truck.splat
outputs/assets/converted/truck.ply
public/samples/truck.splat
public/samples/truck_objects.ply
outputs/assets/raw/garden.splat
outputs/assets/converted/garden.ply
public/samples/garden.splat
public/samples/garden_objects.ply
outputs/assets/raw/bicycle.splat
outputs/assets/converted/bicycle.ply
public/samples/bicycle.splat
public/samples/bicycle_objects.ply
outputs/assets/raw/stump.splat
outputs/assets/converted/stump.ply
public/samples/stump.splat
public/samples/stump_objects.ply
outputs/assets/raw/treehill.splat
outputs/assets/converted/treehill.ply
public/samples/treehill.splat
public/samples/treehill_objects.ply
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

near-1M trained candidate 的 Hugging Face development-stage release 记录：

```text
Dataset: https://huggingface.co/datasets/jianyong365/objgauss-nerf-lego-near1m
Model:   https://huggingface.co/jianyong365/objgauss-nerf-lego-near1m-model
State:   development-stage release only
Record:  docs/state/huggingface-release.md
```

这组 HF 资产用于研究复现和下载 handoff，不是 stable release。它继承
NeRF Synthetic Lego / upstream research dataset 的使用边界；公开页面必须保留
development-stage 声明。大型训练产物继续放在 HF / ignored `outputs/`，不要提交进 git。

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
objgauss assets pull polyhaven-school-chair-nerf-dense
```

默认输出：

```text
outputs/assets/training/polyhaven-school-chair-nerf/train/
outputs/assets/training/polyhaven-school-chair-nerf/transforms_train.json
outputs/assets/converted/polyhaven-school-chair-nerf/training-manifest.json
outputs/assets/training/polyhaven-school-chair-nerf-dense/train/
outputs/assets/training/polyhaven-school-chair-nerf-dense/transforms_train.json
outputs/assets/converted/polyhaven-school-chair-nerf-dense/training-manifest.json
```

该数据集不是前端 public sample；它用于 `docs/benchmarks/splatfacto-scenes.json`
里的第三个 Splatfacto-trained scene row。当前 renderer 是确定性 mesh
rasterizer，用于生成可复现训练图像；它不是现实相机采集数据，也不替代
真实 3DGS / Spark renderer。

`polyhaven-school-chair-nerf-dense` 使用同一 CC0 glTF 源，但把训练输入从
16-frame / 256px 提升到 32-frame / 384px。它只准备后续训练数据，不表示已经产生新的
Gaussian 模型，也不会提交 generated PNG / PLY。

本地构建记录（2026-07-07）：

- `uv run objgauss assets pull polyhaven-school-chair-nerf-dense`: passed。
- `training-manifest.json`: `frames=32`、`image_size=384`、`triangles=5072`、`files=35`。
- `inspect-nerf`: train / val / test 各 32 frames，总计 `frames=96`、
  `missing_images=0`、`invalid_transforms=0`。
- PNG alpha coverage: `min=0.182231`、`mean=0.286609`、`max=0.359138`。

Dense Splatfacto smoke（2026-07-07）：

```bash
npm run train:splatfacto:smoke -- --run \
  --asset-id polyhaven-school-chair-nerf-dense \
  --dataset outputs/assets/training/polyhaven-school-chair-nerf-dense \
  --output-root outputs/training/polyhaven-chair-dense-splatfacto-smoke \
  --experiment chair-dense-splatfacto-smoke \
  --timestamp smoke-cuda \
  --export-dir outputs/training/polyhaven-chair-dense-splatfacto-smoke/export-smoke-cuda \
  --object-field-dir outputs/training/polyhaven-chair-dense-splatfacto-smoke/object-field-sam \
  --sam-manifest outputs/masks/polyhaven-chair-dense-sam-smoke/mask-manifest.json \
  --data-parser blender-data \
  --iterations 100 \
  --steps-per-save 100 \
  --vis tensorboard \
  --cache-images cpu \
  --camera-res-scale-factor 0.5 \
  --cuda-home /tmp/objgauss-cuda13 \
  --max-jobs 2 \
  --device cuda \
  --sam-max-frames 8 \
  --sam-max-masks-per-frame 6 \
  --sam-min-area 64 \
  --sam-max-area-fraction 0.75 \
  --slots 6 \
  --object-iterations 80 \
  --skip-benchmark
```

输出仍是 ignored local training output，不提交 git：

```text
outputs/training/polyhaven-chair-dense-splatfacto-smoke/chair-dense-splatfacto-smoke/splatfacto/smoke-cuda/nerfstudio_models/step-000000099.ckpt
outputs/training/polyhaven-chair-dense-splatfacto-smoke/export-smoke-cuda/splat.ply
outputs/training/polyhaven-chair-dense-splatfacto-smoke/object-field-sam/polyhaven-school-chair-nerf-dense_splatfacto_sam_objects.ply
outputs/masks/polyhaven-chair-dense-sam-smoke/mask-manifest.json
```

对比旧 `polyhaven-chair-splatfacto-smoke`：dense 的 Splatfacto train loss / PSNR 与
Object emergence 指标更好，但 SAM vote 监督覆盖和 final vote loss 回退。因此 dense
candidate 先保留为训练候选，不推进为 viewer / export 默认策略。

Dense benchmark row 复验（2026-07-07）：

- `npm run benchmark:splatfacto:scenes -- --run --skip-sam`: passed，scene suite
  扩展为 4 rows。
- `chair-dense-splatfacto-smoke`: train / held-out split 为 `6 / 2` frames；
  ARI=`0.786356`、curve OES=`0.759438`、render=`0.185040`、held-out loss=`2.002325`、
  held-out render=`0.178836`。
- 对比旧 `chair-splatfacto-smoke`: dense 的 ARI / single-point OES 更高，但 render
  `0.185040 < 0.248716`、held-out render `0.178836 < 0.224084`，因此仍不 publish，
  也不设为 viewer / export 默认。

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
| P0 | Poly Haven School Chair dense NeRF render set | CC0 mesh 派生 32-frame / 384px 图像 + pose | 后续更高质量 Splatfacto candidate 训练输入 | https://polyhaven.com/a/SchoolChair_01 |
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
