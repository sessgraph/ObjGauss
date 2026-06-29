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
- Model `config.yml` 与 `dataparser_transforms.json` 上传。

仍需远端核对或补传：

- Dataset `gaussians/object_aware_gaussians.ply` 大文件上传结果未写入项目事实源。
- Model `nerfstudio/nerfstudio_models/step-000009999.ckpt` 大文件上传结果未写入项目事实源。

如果后续确认这两个大文件已上传，应在本文件补上对应 HF commit hash。

## 发布约束

- 许可证口径：NeRF Synthetic Lego / upstream research dataset 派生产物，仅按研究用途记录；HF card 使用 `license: other` 和 research-use-only 说明。
- 训练产物和 demo 资产分层：HF 承载大训练产物；`public/samples/` 只保留本地页面可加载的样例链接或 symlink，不把大文件提交进 git。
- 当前 HF 页面只能表述为 development-stage release，不能表述为 stable release、production ready 或 commercial demo。
- near-1M trained object-aware PLY 已形成本地训练产物，但 `audit:webgpu-cpath-production-sla` 仍未通过，因此不能把 HF 发布当作 terminal proof。

## 待执行命令

如需补传 Dataset object-aware PLY：

```bash
uvx --with socksio hf upload \
  jianyong365/objgauss-nerf-lego-near1m \
  outputs/assets/gaussians/nerf-lego-trained-near1m-random1300k-v1-candidate/object_aware_gaussians.ply \
  gaussians/object_aware_gaussians.ply \
  --repo-type dataset \
  --commit-message "Add object-aware near1m Gaussian PLY"
```

如需补传 Model checkpoint：

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

- near-1M real trained object-aware PLY: 本地已形成，规模超过 `1,000,000` Gaussian。
- HF development-stage handoff: 已创建并记录，部分大文件仍需远端确认。
- terminal proof: 未完成。

剩余 blocker 是 `audit:webgpu-cpath-production-sla` 未通过。最近一次失败集中在
`scripts/audit-webgpu-presentation-transition.mjs` 的 `checkTiming` 读取
`mode` 时出现 `TypeError`；production SLA summary 存在但 status 不是 `passed`。
