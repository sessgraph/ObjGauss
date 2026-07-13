# ADR 0009: Multi-object instance evidence after the Lego wiring smoke

- Status: Accepted
- Date: 2026-07-13
- Owner decision: the NeRF Lego color-rule run remains a Phase M1 wiring smoke;
  it is not object-segmentation evidence. Add a falsifiable Phase M2 benchmark
  with independent instance ground truth before making an objectness claim.
- Extends: ADR 0008. Its model/checkpoint/browser contracts remain valid, but
  its Lego result is downgraded to engineering evidence.

## Context

The M1 Lego target has four deterministic color-rule slots (`yellow`, `red`,
`dark`, `other`) derived from the same RGB observations that enter Model v0.
The scene contains one physical Lego excavator. A whole-frame holdout prevents
row overlap, but does not break the function `RGB -> color-rule target`.
Therefore perfect ARI/IoU/Purity on that target proves checkpoint and Viewer
consistency, not multi-object instance discovery.

## Decision

1. Keep the Lego run as a reproducible wiring smoke for training, checkpoint
   roundtrip, browser inference, artifact binding and layer interaction.
2. Do not cite its assignment metrics as object-segmentation quality.
3. Add a deterministic local synthetic benchmark with four independently
   authored instances per scene: two same-color identical cubes, one cup and
   one tool. Layouts vary across scenes and include contact and partial
   observation/occlusion stress.
4. Ground truth is authored `gt_instance_id`, independent of RGB. Model input
   is restricted to the existing `[xyz,rgb,opacity]` contract and must not
   consume the GT field.
5. Split by complete `scene_id`, never by rows or views within the same scene.
6. Compare Model v0 with XYZ KMeans, RGB KMeans and 3D connected components.
7. Report permutation-invariant ARI, Hungarian-matched mean IoU, object-count
   error, merge/split rates and object recall at IoU 0.5/0.75 for every held-out
   scene and aggregate them without hiding failed scenes.
8. The Viewer slice must show Raw, Prediction and GT independently and
   preserve per-object selection/hide/show. Prediction and GT may not share one
   renderer-facing `object_id` source.

## Claim boundary

Phase M2 may claim only that the named model and baselines were evaluated on
independent multi-object instance labels under the recorded synthetic stress
conditions. A failed or baseline-inferior result is valid evidence and must
remain visible.

It does not establish real-world identity persistence, physical prediction,
intervention, causal state, cross-dataset generalization or a world model.
Synthetic instance evidence cannot pass the controlled-real Reality Gate.

## Consequences

- The project status must distinguish M1 engineering closure from M2
  segmentation evidence.
- Any future public dataset must pass the same no-color-derived-target and
  whole-scene-split checks before its segmentation metrics are cited.
- No new ML dependency is introduced; the benchmark uses existing NumPy,
  Gaussian, Model v0 and Viewer boundaries.
