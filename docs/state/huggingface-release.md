# ObjGauss Hugging Face 发布记录

> 最近更新: 2026-06-29
> 状态: development-stage release / 开发阶段发布
> Owner account: `jianyong365`

本文件记录 ObjGauss near-1M NeRF Lego 训练产物在 Hugging Face 上的
开发阶段托管状态。它是研究复现和 handoff 用发布，不是稳定 release；
资产、目录结构、指标、模型权重和文档在 stable release 前都可能变化。

## 仓库

- Dataset: `https://huggingface.co/datasets/jianyong365/objgauss-nerf-lego-near1m`
- Model: `https://huggingface.co/jianyong365/objgauss-nerf-lego-near1m-model`

两个仓库均按 public 仓库创建，用于开源研究和下载训练产物。README 顶部已标注
development-stage release：

- Dataset README commit: `9c202e5393c6dea70b1a077e5b70766205d83c87`
- Model README commit: `e4364f247644062bd22bbeceda974dd69a86f06d`
- Dataset current verified head: `295b13f8bac09bc302019ab6c9d238d11d2d6538`
- Model current verified head: `82b700392699852c62dca70ac4274dc722d82282`

## 本地源产物

near-1M tuned candidate 的本地源产物仍保留在 ignored `outputs/` 下，不进入 git：

```text
outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-candidate/
  gaussians.splat
  object_aware_gaussians.ply
  object_field_initial.npz
  object_field_trained.npz
  training-output-manifest.json
  mask-training-summary.json

outputs/training/nerf-lego-splatfacto-near1m-tuned/
  export-random1300k-v1/splat.ply
  lego-splatfacto-near1m-tuned/splatfacto/near1m-random1300k-v1/
    config.yml
    dataparser_transforms.json
    nerfstudio_models/step-000009999.ckpt
```

当前本地 candidate 规模：

- exported PLY: `4,503,634` Gaussians
- object-aware PLY: `4,503,634` Gaussians，约 `1.15GB`
- compact viewer `.splat`: 约 `144MB`
- Object Field metrics: assignment confidence `0.758583`，effective slots `3.364891`
- projection loss: `1.966522 -> 0.069276`

## HF 内容规划

Dataset repo 存放数据集和模型产物的可消费资产：

- `gaussians/gaussians.splat`
- `gaussians/object_aware_gaussians.ply`
- `object_field/object_field_initial.npz`
- `object_field/object_field_trained.npz`
- `masks/...`
- `manifests/training-output-manifest.json`
- `manifests/mask-training-summary.json`
- `metrics/*.json`
- `metrics/*.csv`
- `release-manifest.json`
- `checksums.sha256`

Model repo 存放 Nerfstudio / Splatfacto 可恢复训练权重和配置：

- `nerfstudio/config.yml`
- `nerfstudio/dataparser_transforms.json`
- `nerfstudio/nerfstudio_models/step-000009999.ckpt`
- `release-manifest.json`
- `checksums.sha256`

当前发布包有意不放：

- 重复的 exported `gaussians.ply`
- TensorBoard event logs
- 仓库内 git-tracked 大文件

## 已确认上传记录

本阶段已确认完成：

- Dataset / Model repo 创建。
- Dataset / Model README 更新为 development-stage release。
- Dataset / Model `release-manifest.json` 与 `checksums.sha256` 上传。
- Dataset masks、training manifest、mask training summary、metrics 上传。
- Dataset Object Field `.npz` 上传。
- Dataset compact viewer `.splat` 上传。
- Dataset object-aware PLY 上传并远端核对通过：
  - path: `gaussians/object_aware_gaussians.ply`
  - commit: `295b13f8bac09bc302019ab6c9d238d11d2d6538`
  - size: `1,148,428,347` bytes
  - LFS sha256: `b8a1f2d5c40c8cb5bfb1476565529aa31de018c1884ef0f58706fba4f3e0aecf`
- Model `config.yml` 与 `dataparser_transforms.json` 上传。
- Model Splatfacto checkpoint 上传并远端核对通过：
  - path: `nerfstudio/nerfstudio_models/step-000009999.ckpt`
  - commit: `82b700392699852c62dca70ac4274dc722d82282`
  - size: `3,200,373,037` bytes
  - LFS sha256: `f3558aadfb1d8d546232eb0f7e55e823a50a0011abcaf360879e5e91d84d36ce`

## 发布约束

- 许可证口径：NeRF Synthetic Lego / upstream research dataset 派生产物，仅按研究用途记录；HF card 使用 `license: other` 和 research-use-only 说明。
- 训练产物和 demo 资产分层：HF 承载大训练产物；`public/samples/` 只保留本地页面可加载的样例链接或 symlink，不把大文件提交进 git。
- 当前 HF 页面只能表述为 development-stage release，不能表述为 stable release、production ready 或 commercial demo。
- HF Dataset 中的全量 object-aware PLY 为 `4,503,634` Gaussians，已远端核对；它是开发阶段研究产物，不等于 production-interactive viewer asset。
- near-1M terminal proof 当前使用本地 ignored deterministic sampled1m derivative，通过 `npm run assets:sample-ply` 从全量 HF PLY 源产物派生；该 sampled1m PLY 尚未作为 HF stable asset 发布。

## 重传命令

如需重新上传 Dataset object-aware PLY：

```bash
uvx --with socksio hf upload \
  jianyong365/objgauss-nerf-lego-near1m \
  outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-candidate/object_aware_gaussians.ply \
  gaussians/object_aware_gaussians.ply \
  --repo-type dataset \
  --commit-message "Add object-aware near1m Gaussian PLY"
```

如需重新上传 Model checkpoint：

```bash
uvx --with socksio hf upload \
  jianyong365/objgauss-nerf-lego-near1m-model \
  outputs/training/nerf-lego-splatfacto-near1m-tuned/lego-splatfacto-near1m-tuned/splatfacto/near1m-random1300k-v1/nerfstudio_models/step-000009999.ckpt \
  nerfstudio/nerfstudio_models/step-000009999.ckpt \
  --repo-type model \
  --commit-message "Add near1m Splatfacto checkpoint"
```

上传前确认 HF 登录身份：

```bash
uvx --with socksio hf auth whoami
```

## 终局证据状态

当前事实：

- Full real trained object-aware PLY: 本地与 HF Dataset 均已形成，规模 `4,503,634` Gaussians。
- HF development-stage handoff: 已创建并记录，Dataset object-aware PLY 与 Model checkpoint 已远端核对。
- terminal proof: sampled1m derivative 已完成并通过 production SLA。

Sampled1m terminal proof 输入：

- path: `outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m/object_aware_gaussians.ply`
- manifest: `outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m/sample-manifest.json`
- source: `outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-candidate/object_aware_gaussians.ply`
- source count: `4,503,634`
- sampled count: `1,000,000`
- byte size: `255,001,677`
- sha256: `354440011354a80aeb23a357466e65ccb9c2f2ace07c3756590ab7e203271ea1`
- object counts: `0=450284`、`1=112243`、`2=360042`、`3=77431`

生成命令：

```bash
npm run assets:sample-ply -- \
  --input outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-candidate/object_aware_gaussians.ply \
  --output outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m/object_aware_gaussians.ply \
  --target-gaussians 1000000 \
  --manifest-output outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m/sample-manifest.json \
  --label near1m-random1300k-v1-deterministic-sampled1m \
  --overwrite
```

通过的终局审计：

- `npm run audit:webgpu-cpath-production-sla -- --trained-ply outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m/object_aware_gaussians.ply --target-hardware local-rtx5060ti --output-dir /tmp/objgauss-webgpu-cpath-production-sla-sampled1m-v2`
- report: `/tmp/objgauss-webgpu-cpath-production-sla-sampled1m-v2/summary.md`
- status: `passed`
- trainedGaussians: `1,000,000`
- real trained browser runtime: `passed`
- FPS SLA: `passed`
- sustained trained min approx FPS: `33.804`

配套 strict goal gate：

- `npm run audit:near1m-production-gap -- --require-ready --candidate-output-dir outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m --sla-output-dir /tmp/objgauss-webgpu-cpath-production-sla-sampled1m-v2 --export-dir outputs/training/nerf-lego-splatfacto-near1m-tuned/export-random1300k-v1 --output-root outputs/training/nerf-lego-splatfacto-near1m-tuned --experiment lego-splatfacto-near1m-tuned --timestamp near1m-random1300k-v1 --target-hardware local-rtx5060ti --output-dir /tmp/objgauss-near1m-production-gap-sampled1m`: `ready`
- `npm run audit:renderer-route-goal -- --require-production-ready --near1m-candidate-output-dir outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-sampled1m --near1m-sla-output-dir /tmp/objgauss-webgpu-cpath-production-sla-sampled1m-v2 --near1m-export-dir outputs/training/nerf-lego-splatfacto-near1m-tuned/export-random1300k-v1 --near1m-output-root outputs/training/nerf-lego-splatfacto-near1m-tuned --near1m-experiment lego-splatfacto-near1m-tuned --near1m-timestamp near1m-random1300k-v1 --near1m-target-hardware local-rtx5060ti --output-dir /tmp/objgauss-renderer-route-goal-production-ready-sampled1m-v2`: `ready`

边界和后续风险：全量 `4,503,634`-Gaussian PLY 的直接 browser runtime 审计未通过，
min approx FPS 为 `4.412`。因此 HF 全量 PLY 仍不能标记为 production-interactive；
后续需要 LOD、streaming、分块加载或全量性能优化。
