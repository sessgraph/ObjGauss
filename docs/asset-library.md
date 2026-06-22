# ObjGauss 素材库

这个素材库先解决两个问题：

- 给 ObjGauss MVP 找稳定的测试输入。
- 把 scan / mesh / images / gaussian 统一登记成同一种资产记录。

## 当前已接入

| 资产 | 来源 | 本地文件 | 用途 | 许可备注 |
| --- | --- | --- | --- | --- |
| Plush 3DGS 示例 | https://huggingface.co/cakewalk/splat-data/blob/main/plush.splat | `public/samples/plush.splat` + `public/samples/plush_objects.ply` | 快速验证真实 splat 渲染、高斯点云加载、对象聚类、删除/隔离预览 | 上游说明来源许可混合，只作为本地测试素材 |

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

## 优先素材来源

| 优先级 | 来源 | 类型 | 适合用途 | 入口 |
| --- | --- | --- | --- | --- |
| P0 | ARKitScenes | 真实室内 scan | 手机 LiDAR 房间、家具对象化、真实用户输入形态 | https://github.com/apple/ARKitScenes |
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
  -> 采样 point cloud
  -> 可选：多视角渲染 + 训练 3DGS
  -> objgauss cluster / 手工对象 id
  -> ObjGauss PLY
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

1. 在 `objgauss/assets.py` 注册 `AssetSource`，至少填 `id`、`name`、`source_url`、`download_url`、`local_path`、`raw_file_name`、`output_file_name` 和 `pull_pipeline`。
2. 在 `src/assetLibrary.js` 增加对应前端卡片。
3. 如果是 `.splat`，优先复用 `splat-to-objgauss-ply` 管线。
4. 如果是 mesh / RGB-D / COLMAP 数据，先写转换脚本，再把管线挂到 `objgauss assets pull`。
5. 跑验证：

```bash
uv run --extra dev pytest
npm run build
```
