# ADR 0006: Full 3DGS Training Renderer Path

> 状态: Accepted / dependency path selected
> 日期: 2026-07-02

## 背景

ObjGauss v1 kernel 现在已经形成非训练和训练 MVP 两条闭环：

- 非训练闭环: deterministic assignment / ObjectState pooling / stability diagnostics /
  ObjectState Debug OS。
- 训练 MVP: `frames -> assignment -> ObjectState -> CPU image point splat ->
  image_render_loss`，并能持久化
  `objgauss-trainable-kernel-model-artifact-v1` 供 viewer 调试。

当前训练 renderer 是 `cpu-image-point-splat-differentiable-v1`。它有明确的
`image_render_loss` 和 analytic assignment / decoder-color gradient path，但仍不是完整
3DGS training renderer：

- 不使用 3D covariance projection。
- 不使用 Gaussian scale / quaternion rotation / opacity 的真实 alpha compositing。
- 不支持 SH view-dependent color。
- 不作为大场景 GPU training path。

`objgauss.pipelines.renderer_loss` 也已经把当前 blocker 写明为
`full_3dgs_renderer_not_selected`。因此下一步需要先冻结训练 renderer 选择，而不是直接把
torch / CUDA / 大型 rasterizer 引入基础依赖。

## 决策

选择 `gsplat` 作为 ObjGauss v1 full differentiable 3DGS training renderer 的第一优先
实验路径，落在独立 optional dependency / adapter 后面，不进入基础安装依赖。

原因：

- `gsplat` 提供 PyTorch API 和 CUDA rasterization，输入覆盖完整 3DGS 字段：
  `means`、`quats`、`scales`、`opacities`、`colors`、`viewmats`、`Ks`、
  `width`、`height`。
- 它的 rasterization API 返回 rendered colors / alphas / meta，能自然承接当前
  `TrainingRendererLossResult` 的 `image_render_loss`、per-frame loss 和 telemetry。
- 它支持 batch cameras、SH colors、depth modes、sparse gradient 等训练相关能力。
- 项目为 Apache-2.0 license，比直接绑定原始 GraphDECO
  `diff-gaussian-rasterization` 更适合作为默认实验依赖路径。

GraphDECO `diff-gaussian-rasterization` 保留为兼容参考和结果对照，不作为 ObjGauss
默认 dependency path。

## Dependency Policy

基础包继续保持轻依赖：

```toml
dependencies = [
  "numpy>=1.23",
]
```

后续实现只能在显式 PR 中新增 optional extra，例如：

```toml
[project.optional-dependencies]
training-renderer = [
  "torch",
  "gsplat",
]
```

约束：

- 不把 `torch` / `gsplat` 放进默认 dependencies。
- 不要求普通 `uv run --extra dev pytest` 必须有 CUDA / gsplat。
- gsplat tests 必须使用 import guard；依赖或 CUDA 不可用时 skip，而不是失败。
- 训练输出、rendered images、checkpoints 和大型 artifacts 仍写入 ignored `outputs/`
  或 `/tmp`，不进 git。

## Training Renderer Contract

后续 adapter 必须复用当前 schema：

```text
objgauss-training-renderer-api-v1
```

新增 renderer identity：

```text
renderer_name = gsplat-rasterization-v1
gradient_path = torch-autograd-gsplat-rasterization-v1
```

输入 contract：

```text
GaussianState:
  means: float32[N,3]
  quats: float32[N,4]
  scales: float32[N,3]
  opacities: float32[N]
  colors: float32[N,3] or SH coefficients

CameraBatch:
  viewmats: float32[C,4,4]
  Ks: float32[C,3,3]
  width: int
  height: int

ObjectAssignment:
  A: float32[N,K]
  decoder_colors: float32[K,3]
```

第一版颜色路径保持与当前 trainable kernel 一致：

```text
per_gaussian_rgb = A @ decoder_colors
```

这保证 full renderer 首先验证 assignment / ObjectState training，而不是同时打开
Gaussian geometry optimization。

输出 contract：

```text
TrainingRendererLossResult:
  renderer_name
  gradient_path
  image_render_loss
  frame_losses
  differentiable_fields
  frozen_fields
  gradients telemetry
```

第一版 differentiable / frozen 边界：

```text
differentiable_fields:
  assignments
  decoder_colors

frozen_fields:
  means
  quats
  scales
  opacities
  cameras
```

后续只有在 `assignments + decoder_colors` 的 full renderer smoke 稳定后，才允许把
`means / quats / scales / opacities` 逐项加入 differentiable fields。

## Integration Boundary

训练 renderer 和 viewer renderer 必须隔离：

- `gsplat` 只服务 Python training loss producer。
- Three.js / Spark / WebGPU viewer 继续服务 ObjectState Debug OS 和 browser audit。
- trainable model artifact 可以记录 `renderer_api.renderer_name` 和
  `renderer_api.image_render_loss`，但 viewer 只消费已导出的 assignment /
  ObjectState / telemetry，不在浏览器内运行 gsplat。
- `renderer-loss-contract` 的升级 blocker 只由 training renderer evidence 关闭，
  不由前端渲染器能力关闭。

## 实施顺序

### TRAIN-GSPLAT-ADAPTER-001

新增可选 adapter 和 skipped smoke test：

- 新增 `objgauss/pipelines/gsplat_training_renderer.py`。
- import guard 检查 `torch`、`gsplat` 和 CUDA availability。
- 在 tiny fixture 上调用 `gsplat.rendering.rasterization(...)` 或当前官方等价 API。
- 输出 `objgauss-training-renderer-api-v1` summary。
- 不接入 optimizer，不改变 CLI 默认行为。

### TRAIN-GSPLAT-LOSS-001

把 full renderer 接入 trainable kernel 的可选 loss producer：

- 新增 CLI flag，例如 `--training-renderer point|gsplat`。
- `point` 保持默认，保证无 CUDA 环境仍可运行。
- `gsplat` 可用时要求 image targets / camera batch / Gaussian fields 完整。
- summary 必须能显示 `renderer_name=gsplat-rasterization-v1`。

### TRAIN-GSPLAT-MVP-001

跑第一个小规模 full renderer training MVP：

- 只训练 `A` 和 `decoder_colors`。
- 仍使用小样本或 `/tmp` training output。
- 验收 `image_render_loss` 下降、`L_object` 有效、artifact 可写出。
- viewer 继续通过 artifact 消费结果，不直接加载训练输出。

### TRAIN-GAUSSIAN-PARAMS-001

在前面稳定后再打开 Gaussian 参数训练：

- 逐项允许 `means`、`opacity`、`scale`、`rotation` 进入 differentiable fields。
- 每次只引入一个参数族。
- 每次都必须记录 geometry drift / opacity collapse / object assignment stability。

## 非目标

本 ADR 不做以下事情：

- 不直接修改 `pyproject.toml`。
- 不安装 `torch` / `gsplat`。
- 不引入 CUDA build step。
- 不替换 Three.js / Spark / WebGPU viewer renderer。
- 不提交真实训练输出或大模型。
- 不把 CPU point splat MVP 改名为 full 3DGS renderer。

## 验收

本 ADR 完成后，`TRAIN-FULL-3DGS-RENDERER-ADR-001` 的验收标准是：

- `full_3dgs_renderer_not_selected` blocker 有明确解除路径：`gsplat-rasterization-v1`。
- 明确 `torch` / `gsplat` 只能作为 optional extra 引入。
- 明确 viewer renderer 与 training renderer 隔离。
- 明确第一版 full renderer 只训练 assignment / decoder colors。
- 明确后续 PR 顺序和每一步边界。

## 参考

- `gsplat` rasterization API:
  <https://docs.gsplat.studio/main/apis/rasterization.html>
- `gsplat` GitHub:
  <https://github.com/nerfstudio-project/gsplat>
- GraphDECO `diff-gaussian-rasterization`:
  <https://github.com/graphdeco-inria/diff-gaussian-rasterization>
