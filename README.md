# ObjGauss MVP

ObjGauss is a minimal object-aware layer on top of a standard 3D Gaussian
Splatting export. This repository intentionally does not fork a renderer or
trainer. The MVP starts from a `gaussians.ply` file produced by an existing
3DGS implementation and adds:

- Gaussian feature extraction from position, color, and opacity.
- K-means object clustering.
- `object_id` attachment as a PLY vertex property.
- Object-colored PLY exports for inspection.
- Object removal and isolation exports.
- Soft Object Field initialization for object-slot experiments.

The goal is to validate whether Gaussian splats can be grouped into stable
object-level clusters before investing in semantic guidance or renderer
changes.

## Development workflow

AI coding sessions and human contributors should use the shared workflow in
`docs/development-flow.md`. Codex reads `AGENTS.md`; Claude Code reads
`CLAUDE.md`; both files point back to the same workflow to avoid duplicated
process rules.

Current project state lives in `docs/state/`. Start with
`docs/state/project-status.md` and `docs/state/pr-queue.md`.

## Install

Recommended:

```bash
uv sync --extra dev
```

Plain Python:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

`scikit-learn` is optional. If it is installed, the CLI uses it for k-means.
Otherwise ObjGauss uses a deterministic NumPy implementation.

## MVP workflow

1. Train a scene with a mature 3DGS implementation such as Inria 3DGS or
   gsplat.
2. Export the trained Gaussian cloud as `gaussians.ply`.
3. Cluster the Gaussian cloud:

```bash
objgauss cluster path/to/gaussians.ply -o outputs/scene_objects.ply --clusters 6 --colorize
```

4. Inspect cluster sizes:

```bash
objgauss stats outputs/scene_objects.ply
```

5. Run the object removal test:

```bash
objgauss filter outputs/scene_objects.ply -o outputs/scene_without_2.ply --ids 2 --mode remove
```

6. Optionally create a PLY that rewrites the 3DGS DC color channels to object
   colors for renderers that ignore generic `red`, `green`, and `blue`
   properties:

```bash
objgauss colorize outputs/scene_objects.ply -o outputs/scene_object_colors.ply --rewrite-sh
```

If you only have an antimatter15/cakewalk `.splat` sample, convert it first:

```bash
objgauss convert-splat path/to/model.splat -o outputs/model.ply
objgauss cluster outputs/model.ply -o outputs/model_objects.ply --clusters 6 --colorize
```

## Object Field v1-lite

Object Field keeps a soft `N x K` object-slot distribution for each Gaussian and
exports hard `object_id` labels only at the handoff boundary. The current
implementation is a NumPy warm start from the existing feature clustering. It is
not semantic segmentation yet.

Initialize a soft field and optionally export a viewer-compatible PLY:

```bash
objgauss object-field init public/samples/plush_objects.ply \
  --output outputs/plush_object_field.npz \
  --slots 6 \
  --ply-output outputs/plush_object_field.ply \
  --colorize
```

Inspect Object Field metrics:

```bash
objgauss object-field stats outputs/plush_object_field.npz
```

Check the NeRF Lego training smoke dataset before Object Field experiments:

```bash
objgauss object-field inspect-nerf outputs/assets/training/nerf-synthetic-lego
```

Build a real image-derived mask manifest from NeRF Synthetic RGBA alpha
channels:

```bash
objgauss masks from-nerf-alpha outputs/assets/training/nerf-synthetic-lego \
  --output outputs/masks/nerf-lego-alpha/mask-manifest.json \
  --split train \
  --max-frames 8
```

This creates boolean `.npy` masks next to the manifest and can be used as
foreground projection supervision for `vote-masks`. It is a deterministic mask
source for the Lego smoke dataset, not a replacement for SAM / CLIP instance
segmentation.

Build a multi-slot 2D color mask manifest from the same real NeRF Lego RGBA
images:

```bash
objgauss masks from-nerf-rgba-colors outputs/assets/training/nerf-synthetic-lego \
  --output outputs/masks/nerf-lego-rgba-colors/mask-manifest.json \
  --split train \
  --max-frames 8 \
  --alpha-threshold 16
```

This writes four Lego color slots (`yellow`, `red`, `dark`, `other`) in the
same manifest format consumed by `vote-masks`. The masks come from real 2D
training images and are useful for v1 closure validation; they are still a
deterministic color-rule source, not a SAM / CLIP model output.

Build a SAM automatic mask manifest when a local SAM environment and checkpoint
are available:

```bash
objgauss masks from-nerf-sam outputs/assets/training/nerf-synthetic-lego \
  --output outputs/masks/nerf-lego-sam/mask-manifest.json \
  --checkpoint path/to/sam_vit_b.pth \
  --model-type vit_b \
  --device cuda \
  --split train \
  --max-frames 8 \
  --max-masks-per-frame 8 \
  --min-area 64
```

SAM is optional: ObjGauss does not install model dependencies or download
weights by default. The command expects `segment-anything` and the chosen
checkpoint to already exist in the local environment, then writes the same mask
manifest format consumed by `vote-masks`.

Apply precomputed SAM / CLIP / 2D masks as projection supervision:

```bash
objgauss object-field vote-masks path/to/gaussians.ply \
  --field outputs/plush_object_field.npz \
  --masks path/to/masks.json \
  --output outputs/plush_object_field_masked.npz \
  --ply-output outputs/plush_object_field_masked.ply \
  --colorize
```

Minimal mask manifest:

```json
{
  "width": 100,
  "height": 100,
  "camera_angle_x": 1.5708,
  "frames": [
    {
      "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "masks": [
        {"slot": 0, "label": "object-a", "rect": [0, 0, 50, 100]},
        {"slot": 1, "label": "object-b", "mask_path": "frame_000_slot_1.npy"}
      ]
    }
  ]
}
```

`mask_path` points to a boolean `.npy` mask at image resolution. SAM / CLIP are
treated as upstream mask producers; this repo consumes their masks without
adding those model dependencies yet.

Build the current v1 closed-loop acceptance demo:

```bash
objgauss assets pull plush-3dgs-local
objgauss demo v1-closure
objgauss demo verify-v1-closure
```

This writes:

```text
outputs/demos/v1-closure/v1-closure-manifest.json
outputs/demos/v1-closure/object_field_trained.npz
outputs/demos/v1-closure/plush_v1_objects.ply
public/samples/plush_v1_objects.ply
```

`verify-v1-closure` re-reads the generated manifest, splat, mask manifest,
Object Field `.npz`, exported PLY, public viewer copy, and frontend asset
registration. It fails if any required v1 closure artifact is missing or if
projection supervision did not reduce loss.

Then open the viewer and load `ObjGauss v1 闭环样例` from the 素材库 panel.
The real splat renderer shows the original 3DGS appearance; switching to object
colors or using isolate/delete enters the object-editable point-cloud view.

Build a unified real-3DGS semantic closure demo from projected 2D color masks:

```bash
objgauss demo plush-semantic-closure
objgauss demo verify-plush-semantic-closure
```

This uses the real Plush `.splat` scene plus the raw converted Gaussian PLY,
projects the scene into 2D views, builds color-semantic masks
(`red-subject`, `straw-frame`, `dark-detail`, `other-surface`), trains Object
Field logits from those masks, and exports
`public/samples/plush_semantic_objects.ply` with `object_id`. It does not use
KMeans labels, SAM, or CLIP, and it preserves the original Gaussian colors in
the PLY.

Build a NeRF Lego proxy closure demo from real multi-view RGBA images and
2D color masks:

```bash
objgauss assets pull nerf-synthetic-lego
objgauss demo lego-alpha-closure
objgauss demo verify-lego-alpha-closure
```

This writes `outputs/demos/lego-alpha-closure/`, plus
`public/samples/lego_alpha_proxy.splat` and
`public/samples/lego_alpha_v1_objects.ply`. It is a lightweight Gaussian proxy
from posed RGBA images, not full 3DGS optimization, but it keeps the v1 closure
in one scene: NeRF images and camera poses create the Gaussian proxy, real 2D
color masks supervise Object Field logits, and the exported PLY can be loaded
from the `NeRF Lego 闭环代理样例` card.

Register an output produced by an external 3DGS trainer:

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

This command does not train 3DGS inside ObjGauss. It registers a trained
Gaussian PLY or `.splat`, writes a viewer `.splat`, runs Object Field mask
voting when `--masks` is supplied, and exports an object-aware PLY. With the
`--public-name nerf_lego_trained` path above, the frontend card
`NeRF Lego 训练输出样例` can load the registered output.

Run the browser audit for the closure cards:

```bash
npm run audit:demo
```

The audit starts a temporary Vite server, loads `Plush 2D 语义 Mask 闭环样例`,
`ObjGauss v1 闭环样例`, and `NeRF Lego 闭环代理样例`, checks that the splat and
point-edit canvases are non-empty, and exercises object selection, isolation,
and delete preview.

Run the full local acceptance loop:

```bash
npm run acceptance:demo
```

This rebuilds the Plush v1 closure demo, verifies it, rebuilds the Plush
semantic closure demo, verifies it, rebuilds the NeRF Lego proxy closure demo,
verifies it, and then runs the browser audit. Use
`npm run acceptance:demo -- --pull-assets` on a machine that still needs to
download the local Plush and NeRF Lego assets.

Audit the current evidence against the phase goal:

```bash
objgauss demo audit-v1-goal --allow-incomplete
```

The audit reports whether the current repository proves the full v1 goal. It is
strict by default; `--allow-incomplete` prints the same checks without failing.
The current unified proof is the Plush semantic closure demo. A future trained
NeRF Lego output can still be registered at
`outputs/assets/gaussians/nerf-lego-trained/training-output-manifest.json`,
but it is no longer the only accepted evidence for the phase goal.

## Asset library

素材库入口：

- Frontend data: `src/assetLibrary.js`
- Handoff docs: `docs/asset-library.md`
- Local preview sample: `public/samples/plush_objects.ply`

The viewer shows featured assets in the left-side 素材库 panel. Assets with
`localPath` can be loaded directly; candidate sources link to their upstream
dataset pages and should be downloaded outside the git repo.
Asset entries also distinguish training sources from demo-ready samples so
research datasets do not get mixed into public demos by accident.

Pull localizable samples and training inputs:

```bash
objgauss assets list --pullable
objgauss assets pull plush-3dgs-local
objgauss assets pull polyhaven-school-chair-1k
objgauss assets pull nerf-synthetic-lego
```

`polyhaven-school-chair-1k` is a CC0 mesh demo input. It is not rendered by the
current 3DGS viewer until the mesh -> multiview render -> 3DGS conversion chain
is added. `nerf-synthetic-lego` is the first multi-view training smoke dataset
for Object Field work.

## Notes

- The default feature vector is `[x, y, z, r, g, b, opacity]`.
- Position, color, and opacity groups are normalized before clustering.
- Standard 3DGS `f_dc_0`, `f_dc_1`, and `f_dc_2` fields are converted from
  spherical-harmonic DC values into approximate RGB features.
- Standard 3DGS raw opacity logits are sigmoid-activated before use.
- PLY IO supports scalar ASCII, binary little-endian, and binary big-endian
  vertex properties. It targets Gaussian PLY exports, not triangle meshes with
  list properties.

## Suggested experiment

Use one desktop scene with a cup, phone, and a few small objects. Success for
this MVP means:

- The 3DGS baseline reconstructs normally.
- The clustered PLY has an `object_id` for every Gaussian.
- Object colors are visually separable.
- Removing one `object_id` makes a coherent part of the scene disappear.
