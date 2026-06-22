# ADR 0003: Asset Ingestion Strategy

> 状态: Proposed
> 日期: 2026-06-22

## 背景

ObjGauss 后续需要两类素材：

- **训练素材**: 用于训练 3DGS、验证对象分组、评估渲染质量。
- **Demo 素材**: 用于前端展示和可复现演示。

当前只有 `plush-3dgs-local` 自动拉取，来源许可混合，只适合作为本地测试和管线烟测。

## 决策

素材接入必须走统一 registry 和目录规范：

- CLI registry: `objgauss/assets.py`
- 前端 registry: `src/assetLibrary.js`
- 文档事实源: `docs/asset-library.md`

目录分层：

- `outputs/assets/raw/`
- `outputs/assets/converted/`
- `outputs/assets/training/<asset_id>/`
- `outputs/assets/gaussians/<asset_id>/`
- `public/samples/`

推荐接入顺序：

1. Demo: Poly Haven 小型 CC0 模型。
2. 对象训练: OmniObject3D 最小子集。
3. 室内训练: ARKitScenes 单房间子集。

## 验收标准

- 新素材记录来源、许可、下载方式、转换命令。
- 自动管线能输出可验证产物，或明确标记为 manual。
- Demo 样例能前端加载，训练素材能进入训练目录。
- 不提交大型数据集或训练输出。

## 风险

- 大数据集下载成本高，不适合一次拉全量。
- 研究数据许可可能限制公开演示。
- Mesh / RGB-D / COLMAP / 3DGS 输出格式差异大，转换管线需要分阶段实现。

## 后续任务

- `ASSET-001`: 建立 Demo/训练素材转换管线。
