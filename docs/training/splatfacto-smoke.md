# TRAIN-003A: Splatfacto Smoke Runbook

This runbook turns the local NeRF Lego Splatfacto smoke handoff into a
repeatable command sequence. It produces ignored local outputs only; do not
commit `outputs/`, checkpoints, TensorBoard logs, SAM checkpoints, or downloaded
training assets.

The smoke is intentionally short. It proves that this machine can train a real
Nerfstudio Splatfacto Gaussian PLY, export it, attach ObjGauss Object Field
slots with SAM masks, and feed the SEMANTIC benchmark suite. It is not a
quality reconstruction target and it is not the final public Lego sample.

## TRAIN-003B Resource-Safe Public Sample Candidate

TRAIN-003B used the same handoff, but ran a resource-safe 500 iteration Lego
Splatfacto candidate instead of the 100 iteration smoke:

```text
iterations=500
vis=tensorboard
cache_images=cpu
camera_res_scale_factor=0.5
MAX_JOBS=2
```

This keeps the run usable while reserving most of the GPU for the desktop. The
validated local output is:

```text
outputs/training/nerf-lego-splatfacto-long/lego-splatfacto-safe/splatfacto/safe-500-cpu-cache-v2/config.yml
outputs/training/nerf-lego-splatfacto-long/lego-splatfacto-safe/splatfacto/safe-500-cpu-cache-v2/nerfstudio_models/step-000000499.ckpt
outputs/training/nerf-lego-splatfacto-long/export-safe-500-cpu-cache-v2/splat.ply
outputs/assets/gaussians/nerf-lego-trained-safe-500-cpu-cache-v2-warmstart/training-output-manifest.json
public/samples/nerf_lego_trained.splat
public/samples/nerf_lego_trained_objects.ply
```

The public files are local ignored artifacts. Do not commit them unless the
license and size policy are explicitly changed.

Current TRAIN-003B validation:

```text
exported_gaussians=47168
slots=8
supervised_gaussians=7676
projection_loss=3.047123 -> 0.321066
browser_asset=nerf-lego-trained-output-local passed
```

This completes the "registered frontend training-output sample" handoff, but it
is still a safe candidate, not the final high-quality long training result.

## One Command

Preview the command sequence without running training:

```bash
npm run train:splatfacto:smoke -- --dry-run
```

Check whether the expected local outputs already exist:

```bash
npm run train:splatfacto:smoke -- --status
```

Run the full smoke pipeline:

```bash
SAM_CHECKPOINT=/home/ljy/models/sam/sam_vit_b_01ec64.pth \
npm run train:splatfacto:smoke -- --run
```

To regenerate only the training handoff and Object Field files without running
the SEMANTIC benchmark at the end:

```bash
SAM_CHECKPOINT=/home/ljy/models/sam/sam_vit_b_01ec64.pth \
npm run train:splatfacto:smoke -- --run --skip-benchmark
```

## Inputs

The wrapper pulls the NeRF Lego training asset if needed:

```text
outputs/assets/training/nerf-synthetic-lego/
```

SAM is optional for the repository but required for this smoke. The default
checkpoint path is:

```text
/home/ljy/models/sam/sam_vit_b_01ec64.pth
```

Override it with either `SAM_CHECKPOINT` or `--sam-checkpoint`:

```bash
npm run train:splatfacto:smoke -- \
  --dry-run \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth
```

## Outputs

The default run writes:

```text
outputs/training/nerf-lego-splatfacto-smoke/
outputs/training/nerf-lego-splatfacto-smoke/lego-splatfacto-smoke/splatfacto/smoke-cuda/config.yml
outputs/training/nerf-lego-splatfacto-smoke/lego-splatfacto-smoke/splatfacto/smoke-cuda/nerfstudio_models/step-000000099.ckpt
outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply
outputs/masks/nerf-lego-sam/mask-manifest.json
outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_initial.npz
outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/object_field_sam.npz
outputs/training/nerf-lego-splatfacto-smoke/object-field-sam/lego_splatfacto_sam_objects.ply
```

The final `splat.ply` is the SEMANTIC benchmark scene input. The Object Field
files are the corresponding benchmark field handoff.

## Defaults And Overrides

Common options:

```text
--iterations 100
--object-iterations 80
--device cuda
--slots 8
--sam-max-frames 2
--sam-max-masks-per-frame 8
--sam-min-area 64
--timestamp smoke-cuda
--checkpoint-step 000000099
--skip-benchmark
```

Path overrides:

```text
--dataset outputs/assets/training/nerf-synthetic-lego
--output-root outputs/training/nerf-lego-splatfacto-smoke
--experiment lego-splatfacto-smoke
--export-dir outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda
--object-field-dir outputs/training/nerf-lego-splatfacto-smoke/object-field-sam
--sam-manifest outputs/masks/nerf-lego-sam/mask-manifest.json
--sam-checkpoint /home/ljy/models/sam/sam_vit_b_01ec64.pth
```

## Environment Notes

The current verified local environment is:

```text
GPU: NVIDIA GeForce RTX 5060 Ti
Torch: 2.12.1+cu130
CUDA toolchain: 13.0 package set through uv --with
```

The wrapper includes the CUDA 13.0 packages needed by `gsplat` JIT on this
machine:

```text
nvidia-cuda-nvcc==13.0.*
nvidia-cuda-cccl==13.0.*
nvidia-nvvm==13.0.*
nvidia-cuda-crt==13.0.*
```

`ns-export gaussian-splat` is run with:

```text
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

This is needed for trusted local Nerfstudio checkpoints with newer PyTorch
`torch.load` defaults.

If `gsplat` reports missing `nvcc`, CUDA header/compiler mismatch, PTX version
mismatch, or `-lcudart` link errors, first re-run the dry-run command and verify
that the command includes the CUDA 13.0 packages above. If the package set is
present but linking still fails, check the local CUDA runtime library path before
changing ObjGauss code.

On this machine, `uv --with nvidia-cuda-nvcc` exposed the CUDA 13.0 toolkit but
did not provide an unversioned `libcudart.so`. A temporary wrapper directory in
`/tmp/objgauss-cuda13` was used for the TRAIN-003B run:

```text
CUDA_HOME=/tmp/objgauss-cuda13
PATH=/tmp/objgauss-cuda13/bin:$PATH
LD_LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LD_LIBRARY_PATH
LIBRARY_PATH=/tmp/objgauss-cuda13/lib:$LIBRARY_PATH
```

The wrapper only adds symlinks to the uv CUDA wheel files, including
`libcudart.so -> libcudart.so.13`; it should not modify the system CUDA install
or the uv cache.

## Verification

After a successful run:

```bash
npm run train:splatfacto:smoke -- --status
uv run objgauss stats outputs/training/nerf-lego-splatfacto-smoke/export-smoke-cuda/splat.ply
npm run acceptance:semantic
```

Expected smoke scale from the current local run:

```text
gaussians=50000
slots=8
checkpoint=step-000000099.ckpt
```

Use `training register-output` only after a longer training run is selected as a
public sample candidate. This smoke runbook does not publish the Splatfacto
output to `public/samples/`.
