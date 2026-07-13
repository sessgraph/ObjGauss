# ObjectState Model Diagnostic 001

Status: accepted implementation contract for pre-M3 diagnosis

## Question

`OBJECTSTATE-MODEL-DIAGNOSTIC-001` answers one bounded question:

> Why does the current `ObjectStateModelV0` assignment prototype fail to beat
> the recorded 3D connected-components baseline on M2?

This slice diagnoses the existing model. It does not introduce a larger model,
claim object discovery, or unlock physical capture.

## Controlling evidence

- The candidate is the same `ObjectStateModelV0` tanh encoder and assignment
  head used by M2.
- Instance truth is `gt_instance_id` from procedural object authorship. It is
  supervision/evaluation-only and is excluded from every inference feature.
- Train and test are split by complete `scene_id`; a physical `layout_id` may
  occur in more than one held-out view but must never cross the train boundary.
- Results are descriptive synthetic diagnostics. They do not pass the real
  identity, prediction, intervention, or Reality Gate contracts.

## Input ablation

All variants use the same model family, optimizer, split, iteration count and
seed set. Only the inference feature matrix changes. The four policies run on
both the original M2 distribution and the independent hard-case distribution:

| Policy | Features |
| --- | --- |
| `xyz_only` | XYZ; RGB/opacity blocks are constant |
| `rgb_only` | RGB; XYZ/opacity blocks are constant |
| `xyz_rgb` | XYZ + RGB; opacity is constant |
| `xyz_rgb_semantic` | XYZ + RGB + a class-semantic proxy |

The semantic proxy is a three-class procedural shape embedding shared by both
red cubes. It never contains `gt_instance_id`, does not distinguish the two
same-class cubes, and is an oracle-style upper-bound diagnostic rather than a
deployable teacher model. A gain can support testing semantic evidence next; it
cannot prove that a real semantic teacher will generalize.

Run at least three deterministic training seeds in the canonical experiment.
Report the mean, standard deviation and per-seed values rather than selecting a
favorable seed.

The original M2 cohort additionally runs `native_xyz_rgb_opacity` as a frozen
reproduction anchor. Seed 0 must preserve the recorded M2 comparison; the
three-seed mean is descriptive and is not allowed to replace that anchor.

## Hard-case observations

The held-out set contains these named cases:

1. `near_same_color_cubes`: two same-color, same-shape cubes at near/contact
   distance.
2. `cube_cup_contact`: a cube and cup touch, attacking pure connectivity.
3. `cube_behind_cup`: a depth-ordered, partially observed cube behind the cup.
4. `cross_view`: the same held-out layout under two camera transforms and
   visibility masks.

The first three cases have one observation each. `cross_view` has an anchor and
target observation so persistent slot identity can be evaluated. Every
observation retains all four physical instances.

## Metrics and error taxonomy

For each model seed, baseline, case and aggregate, report:

- Hungarian-matched mean IoU;
- adjusted Rand index (ARI);
- object recall at IoU 0.5 and 0.75;
- object-count error;
- merge count/rate;
- split count/rate.

Identity swap is not permutation-invariant per frame. For each cross-view pair,
derive the predicted-slot to physical-instance mapping on the anchor view, keep
that mapping fixed, and evaluate the dominant slot for each physical instance
in the target view. Report swap count/rate and unmapped identity count. Do not
rematch the target view before computing swaps.

Baselines are `xyz_kmeans`, `rgb_kmeans`, and
`connected_components_3d`. KMeans may use the known target object count only as
an explicitly recorded baseline upper bound. Connected components uses a fixed
radius recorded in the report.

## Leakage and validity gate

The diagnostic is invalid unless all checks pass:

- complete scene split and complete layout split between train and held-out;
- independently authored instance target;
- target and instance-derived features absent from model input;
- semantic proxy shared by the two cube identities;
- all required hard cases present;
- the cross-view pair shares one held-out layout and no training layout;
- all configured seeds ran, with no best-seed selection;
- all four instances remain observable in every held-out observation.

## Outputs

The canonical CLI writes one ignored output directory containing:

- `dataset-manifest.json`, containing the original M2 and hard-case contracts;
- per-policy/per-seed checkpoints and training summaries;
- `diagnostic-summary.json`;
- `ablation-matrix.csv`;
- `hard-case-matrix.csv`;
- `error-taxonomy.csv`;
- `report-artifact.json` and packaged `report.html` after report delivery.

The tracked code and tests are the reproducible producer. Generated diagnostics
remain under `outputs/` and are not committed.

## Decision rules

- If `xyz_only` is statistically indistinguishable from `xyz_rgb`, color is not
  a supported explanation for the aggregate result.
- If the connected-components advantage disappears or reverses on contact
  cases, its M2 lead is concentrated in spatial separation.
- If semantic proxy improves cup/tool metrics but same-cube swaps remain, class
  semantics are insufficient for instance identity.
- If no variant beats the best recorded baseline on aggregate, model
  superiority remains false and long-model escalation stays blocked.
- A model win in this synthetic diagnostic would justify the next controlled
  experiment only; it would not satisfy a real-data gate.

The slice is complete when the canonical run, machine-readable artifacts,
technical report and validators agree on the same conclusion, including a
negative conclusion.

Viewer Error View is deliberately outside this slice. It may consume these
artifacts in a later viewer task, but UI work cannot change or soften the
canonical diagnostic result.
