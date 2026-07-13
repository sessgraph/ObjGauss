# ObjectState multi-object evidence contract

## Purpose

Phase M2 asks one falsifiable question:

> Does Model v0 group independent object instances in held-out multi-object
> scenes, or does it rely on color/location shortcuts and merge or split them?

The benchmark is diagnostic. It is allowed to fail.

## Dataset contract

One benchmark bundle contains at least six scenes and four instances per scene.
The canonical local stress set uses twelve scenes: eight train and four
held-out under a deterministic complete-`scene_id` split.

Each Gaussian row contains:

- renderer/model features: `x`, `y`, `z`, `red`, `green`, `blue`, `opacity`;
- dataset-only fields: `scene_id`, `gt_instance_id`, `shape_id`;
- no target-derived renderer color or predicted `object_id` in model input.

Every scene contains:

| instance | shape | appearance | purpose |
| --- | --- | --- | --- |
| 0 | cube | red | same-color instance A |
| 1 | cube | red | same-color instance B |
| 2 | cup | blue | distinct non-convex proxy |
| 3 | tool | gray | elongated composite proxy |

Layouts and rotations change by scene. The dataset manifest records whether a
scene includes contact and which instance is partially observed. The GT owner
is the procedural instance authoring step, not an RGB/mask classifier.

## Leakage and split gate

The benchmark is invalid unless all checks pass:

1. train and held-out `scene_id` sets are disjoint;
2. `gt_instance_id` is absent from the Model v0 feature order;
3. target provenance is `procedural_instance_authorship`, not color voting;
4. both same-color cubes occur in every evaluated scene;
5. at least one held-out scene contains contact and at least one contains
   partial observation;
6. metrics are recomputed from raw prediction labels and GT.

## Candidates

- `xyz_kmeans`: KMeans over normalized XYZ.
- `rgb_kmeans`: KMeans over normalized RGB; retained as a shortcut diagnostic.
- `connected_components_3d`: radius-neighbor components over XYZ, with small
  fragments assigned to their nearest non-small component.
- `objectstate_model_v0`: existing checkpoint inference over
  `[xyz,rgb,opacity]`.

All candidates are evaluated per held-out scene. Cluster addresses are not
assumed to match GT IDs.

## Metrics

Let the maximum-IoU bipartite match between predicted clusters and GT instances
be the canonical alignment.

- `ari`: adjusted Rand index.
- `hungarian_mean_iou`: mean matched IoU over GT instances; unmatched GT scores
  zero.
- `object_recall_iou_0_5` and `object_recall_iou_0_75`: fraction of GT instances
  whose matched IoU reaches the threshold.
- `object_count_error`: absolute difference between predicted and GT counts.
- `merge_rate`: predicted clusters materially covering more than one GT
  instance, normalized by predicted cluster count.
- `split_rate`: GT instances materially covered by more than one predicted
  cluster, normalized by GT count.

“Materially” means at least 10% of the relevant cluster or instance points.
Aggregate results report the arithmetic mean and retain every per-scene row.

## Output contract

The backend slice writes ignored local evidence:

```text
outputs/model-demo/<run_id>/
  dataset-manifest.json
  checkpoint.json
  training-summary.json
  benchmark.json
  model-manifest.json
  scenes/<scene_id>/raw.ply
  scenes/<scene_id>/raw.splat
  scenes/<scene_id>/prediction.ply
  scenes/<scene_id>/ground-truth.ply
```

`raw.ply` contains no `object_id`; prediction and GT each derive their own
renderer-facing `object_id` from prediction and `gt_instance_id` respectively.

The model manifest binds the selected held-out scene's raw splat/model input,
checkpoint, prediction, independent GT and aggregate benchmark evidence. The
Viewer exposes Raw/Prediction/GT as separate layers; switching layers isolates
that evidence model, and object controls act on the currently inspected layer.
The metrics panel must display a baseline-inferior verdict without changing the
rendered prediction or GT.
