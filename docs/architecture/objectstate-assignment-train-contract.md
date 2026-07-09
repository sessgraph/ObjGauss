# ObjectState Assignment Train Contract

> Status: current / OBJECTSTATE-ASSIGNMENT-TRAIN-CONTRACT-001
> Last updated: 2026-07-09
> Depends on:
> - `docs/architecture/objectstate-model-contract.md`
> - `docs/architecture/assignment-solver-v2-contract.md`
> - `docs/architecture/objectstate-state-variable-gate.md`

## Purpose

This contract defines the first short-training loop for ObjGauss assignment
learning:

```text
GaussianCloud -> training dataset -> AssignmentSolver -> A[N,K]
  -> ObjectStateProjection -> assignment metrics / gate handoff
```

The goal is to prove that the assignment path can learn object separation in a
bounded smoke experiment. It is not a long training plan, not a world model,
not a renderer optimization plan and not a production dataset contract.

## Stage Boundary

`OBJECTSTATE-ASSIGNMENT-TRAIN-CONTRACT-001` is a contract-only slice.

Allowed next implementation:

```text
OBJECTSTATE-ASSIGNMENT-TRAIN-001
```

Required property of the next implementation:

```text
single small dataset / fixture
short run under 10 minutes
loss decreases
assignment metrics improve
visualization artifact is generated
gate handoff does not regress
```

Long training is blocked until the smoke run proves those properties.

## Dataset Contract

### Dataset Schema

Future implementation should use a versioned dataset summary:

```text
schema = objgauss-objectstate-assignment-train-dataset-v1
```

The dataset consists of one or more training samples:

```text
AssignmentTrainSample = {
  sample_id: str
  source_kind: synthetic | public_replay | controlled_real
  gaussian_ref: str optional
  gaussian_count: int
  feature_dim: int
  slots: int
  gaussian_fields: list[str]
  evidence: AssignmentEvidenceBatch
  target_assignment: R[N,K]
  target_object_labels: R[N] optional
  split: train | val | test
  license: str
}
```

Rules:

- `target_assignment` is supervised training target, not inference-time state.
- `target_assignment` must be row-normalized, non-negative and have shape
  `N x K`.
- `target_object_labels` may be derived from `argmax(target_assignment)` for
  metrics and visualization.
- Synthetic and real / public replay rows must remain separated in summaries.
- Large Gaussian inputs, rendered images, checkpoints and visualizations stay
  under ignored `outputs/` unless explicitly reviewed for publication.

### Minimum Smoke Dataset

The first smoke can use:

- a tiny synthetic Gaussian fixture;
- an existing small object-aware Gaussian sample with trusted object labels;
- a sampled subset of a larger scene.

The smoke must record:

```text
sample_id
N
K
feature_dim
source_kind
target source
train / eval split
license / local-only status
```

## Model Contract

The public contract remains assignment-first:

```text
AssignmentEvidenceBatch -> logits/cost[N,K] -> softmax -> A[N,K]
```

The current implementation path should reuse `AssignmentSolverV2`:

```text
solver_family = cost-softmax-assignment-v2
```

A later MLP encoder can be added only if it preserves the same public handoff:

```text
Gaussian / AssignmentEvidence -> A[N,K] -> ObjectStateProjection
```

and lands with a separate architecture update, tests and checkpoint schema. The
first smoke should not introduce Transformer, Slot Attention, Sinkhorn / OT,
replay buffer, diffusion or dynamics.

## Loss Contract

The first train run must start with supervised assignment loss:

```text
L_train = lambda_assignment * L_assignment
```

Allowed MVP additions:

```text
L_train =
  lambda_assignment * L_assignment
+ lambda_compactness * L_compactness
+ lambda_separation  * L_separation
+ lambda_entropy     * L_entropy optional
+ lambda_balance     * L_balance optional
```

Mapping to current code:

- `L_assignment` maps to supervised CE / target assignment loss.
- `L_entropy` and `L_balance` map to existing assignment solver v2 helpers.
- `L_compactness` can be represented by cluster / within-slot spatial cost.
- `L_separation` should remain a metric or weak loss until it is separately
  tested; do not let it override assignment CE in the first smoke.

Forbidden shortcuts:

- using hard `object_id` equality as the only proof of identity;
- claiming training success from loss decrease without assignment metrics;
- using renderer loss to bypass assignment and identity gates;
- silently changing K during the smoke run.

## Checkpoint Contract

Future implementation should emit a versioned run summary:

```text
schema = objgauss-objectstate-assignment-train-run-v1
```

Required fields:

```text
AssignmentTrainRun = {
  schema: str
  dataset_schema: str
  model_family: str
  solver_state_schema: str
  initial_state: summary or ref
  final_state: summary or ref
  training_config: {
    iterations: int
    learning_rate: float
    loss_weights: dict
    seed: int
  }
  loss_curve: list[LossRecord]
  before_metrics: AssignmentMetrics
  after_metrics: AssignmentMetrics
  checkpoint_ref: str optional
  visualization_refs: list[str]
  gate_handoff: dict
  claim_policy: dict
}
```

Checkpoint files must record:

- `K` and feature dimension;
- solver family and solver state schema;
- cost / loss weights;
- dataset sample ids and target source;
- projection policy from `A[N,K]` to `ObjectStateProjection`;
- validation summaries used to decide whether to continue.

Large checkpoint payloads must not be committed to git.

## Evaluation Contract

A train smoke must compare before and after:

```text
loss
mean_best_iou
ari
purity
assignment_confidence
mean_normalized_entropy
effective_slots
ObjectStateProjection active_state_count
```

Minimum acceptance for a useful smoke:

- total loss decreases;
- assignment metrics improve or already start near-perfect with a stated
  reason;
- projection remains valid;
- no slot collapse is hidden;
- gate handoff reports pass / fail / blocked honestly.

Identity / prediction / causal gates remain separate:

- assignment smoke may hand off to identity gate;
- prediction requires future state candidates;
- causal requires action-conditioned candidates.

## Visualization Contract

Every smoke run must produce a local visualization artifact showing:

```text
Gaussian colored by argmax(A) before training
Gaussian colored by argmax(A) after training
```

Recommended local paths:

```text
outputs/assignment-smoke/<run_id>/assignment-before.ply
outputs/assignment-smoke/<run_id>/assignment-after.ply
outputs/assignment-smoke/<run_id>/summary.json
```

Visualization artifacts are local evidence and stay ignored by default. They
do not become public demo assets without separate license and size review.

## Long Training Gate

Do not start 10 minute or longer runs until a short smoke proves:

```text
1. loss decreases
2. IoU / ARI / purity improve
3. assignment visualization is coherent
4. ObjectStateProjection remains valid
5. gate handoff does not collapse
```

If any item fails, record negative evidence and fix the small loop before
scaling data, duration or model complexity.

## Non-Goals

This contract does not:

- implement a dataloader;
- run training;
- create checkpoints;
- introduce torch, CUDA, Transformer, Slot Attention, replay buffer, diffusion
  or dynamics;
- collect controlled real data;
- modify viewer defaults;
- claim ObjectState is a proven world-state variable.

## Next PRs

### OBJECTSTATE-ASSIGNMENT-TRAIN-001

Implement the bounded smoke training loop against this contract.

Required outputs:

- train dataset summary;
- train run summary;
- before / after assignment metrics;
- before / after assignment visualization refs;
- checkpoint or solver state roundtrip;
- clear blocked status for gates that lack required evidence.

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_assignment_train`.
- Dataset schema:
  `objgauss-objectstate-assignment-train-dataset-v1`.
- Run schema:
  `objgauss-objectstate-assignment-train-run-v1`.
- Entry points:
  - `objectstate_assignment_train_dataset_summary(...)`
  - `objectstate_assignment_train_smoke(...)`
- The smoke implementation uses existing `AssignmentSolverV2` and
  `train_assignment_solver_v2(...)`, not torch / CUDA / Transformer / Slot
  Attention.
- The run writes:
  - `summary.json`
  - `assignment-solver-v2-final-state.json`
  - `assignment-before.ply`
  - `assignment-after.ply`
- The run summary records before / after `mean_best_iou`, `ari`, `purity`,
  loss curve, checkpoint roundtrip, visualization refs and long-training gate.
- Iterations are bounded to `<= 600` in the smoke entry point.

### OBJECTSTATE-ASSIGNMENT-GENERALIZATION-001

Implemented v0.1 facts:

- Core module:
  `objgauss.core.objectstate_assignment_generalization`.
- Summary schema:
  `objgauss-objectstate-assignment-generalization-v1`.
- Entry point:
  `objectstate_assignment_generalization_summary(...)`.
- The audit accepts train and held-out test `GaussianCloud + target_assignment`
  samples with the same slot count.
- Training is performed only on train evidence through the existing
  `AssignmentSolverV2` / `train_assignment_solver_v2(...)` path.
- Evaluation runs before / after assignment MVP summaries on both train and
  held-out test samples, then reports `mean_best_iou`, `ari`, `purity`,
  metric deltas and train-test generalization gap.
- The summary writes `generalization-summary.json` and
  `assignment-generalization-final-state.json`, validates checkpoint roundtrip
  on the held-out sample and exposes a long-training gate.
- Iterations remain bounded to `<= 600`; the audit does not use GPU, renderer
  loss, Transformer, Slot Attention, replay buffer, diffusion or dynamics.
- Passing this audit does not claim identity gate, reality gate or world-model
  success; it only prevents train-only assignment memorization from being
  promoted silently.

### OBJECTSTATE-ASSIGNMENT-EVAL-001

Package assignment evaluation for reuse across synthetic smoke, public replay
and controlled real rows.

Required metrics:

- IoU;
- ARI;
- purity;
- assignment confidence;
- entropy / slot collapse diagnostics;
- identity gate handoff when candidate identity evidence exists.
