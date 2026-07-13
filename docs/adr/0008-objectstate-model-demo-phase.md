# ADR 0008: ObjectState model demo phase

- Status: Accepted
- Date: 2026-07-13
- Owner decision: stop further RBO adaptation and add a bounded Phase M / Phase 2.5
  model-demo track before returning to a self-built controlled capture.
- Supersedes: the single-priority and Viewer-freeze parts of ADR 0007; its
  correctness, evidence-separation and Reality Gate requirements remain active.

## Context

The RBO eligibility audit is complete negative evidence, not an unfinished
adapter: the audited camera-motion and force/torque routes produce zero scenes
that satisfy the strict visible-occluded-visible transition required by the
current controlled-data contract. More adapter work cannot create the missing
observation sequence.

At the same time, the repository already contains an assignment training loop,
checkpoint roundtrip, before/after object-color PLY output, ARI/IoU/Purity
metrics, a model artifact manifest, and an object-aware Viewer. The first Phase
M bootstrap connected them, but it also exposed a model gap: its checkpoint is
an `AssignmentSolverV2` center state trained and evaluated on the same points.
That is useful wiring evidence, not yet the requested Object Encoder +
Assignment Head model or held-out public-data evidence.

## Decision

Add a bounded Phase M / Phase 2.5 alongside, but separate from, the real-world
Reality Gate:

1. `OBJECTSTATE-PUBLIC-DATASET-TRAIN-001` runs the existing assignment contract
   on a license-recorded public research sample and emits a checkpoint plus a
   versioned run summary.
2. `OBJECTSTATE-DEMO-EVAL-001` emits `before.ply`, `after-object-color.ply` and
   `metrics.json` from that same run. ARI, mean-best-IoU and Purity must be
   recomputable from raw predictions and labels recorded by the run.
3. `OBJECTSTATE-VISUALIZATION-001` lets the existing Viewer load that result and
   distinguish the raw Gaussian, object-color result, object list and metrics.
4. Treat the existing assignment-solver demo as Phase M0 bootstrap. Phase M1
   adds ObjectState Model v0 with the explicit path
   `[xyz,rgb,opacity] -> Object Encoder -> Assignment Head -> A[N,K] -> ObjectState`.
   It must evaluate points from held-out `source_frame` values that were not used
   for gradient updates.
5. Approve the minimal Model v0 checkpoint/training contracts
   `objgauss-objectstate-model-v0-state-v1` and
   `objgauss-objectstate-model-v0-training-v1`. Continue using
   `objgauss-model-artifact-manifest-v1`; extend its role enum only with
   `model_input` and `objectstate_model` so the Viewer can recompute inference
   from Gaussian input + checkpoint.
6. Keep large checkpoints and training outputs under ignored `outputs/`.
   Publication under `public/` requires a separate size and license review.
7. After the demo chain is reproducible, return to a self-built controlled
   capture for identity persistence, prediction and action-conditioned
   transition. Do not resume RBO field mining without new source evidence that
   changes its eligibility result.

## Claim boundary

Phase M may demonstrate:

- a public-data Gaussian sample was consumed by a named run;
- the assignment model emitted `A[N,K]`, a checkpoint and object-color output;
- assignment metrics changed between the recorded before/after states;
- the Viewer displays artifacts and metrics from that same run.
- Model v0 held-out rows came only from `source_frame` values excluded from
  training, and the browser recomputed the recorded hard assignment from the
  checkpoint and browser input.

It must not be described as real identity persistence, causal state validation,
cross-scene generalization, a world model, or a license-clean public release.
Model v0 may be called same-scene held-out-frame generalization only when that
split is independently verified. A supervised target used for training must be
identified as teacher/GT evidence, not inference-time ObjectState.

## Consequences

- ADR 0007 no longer blocks this narrowly scoped Viewer/training integration or
  the two Model v0 contracts named above, but still blocks unrelated product
  expansion, schemas and false evidence.
- RBO is retained as negative dataset-feasibility evidence and download/audit
  tooling remains reproducible; it is no longer an active research route.
- Model demonstration and real-world validation have separate gates, artifacts
  and claims. Passing Phase M cannot satisfy the controlled-real exit criterion.
