# ObjectState Model Contract

> Status: current / OBJECTSTATE-MODEL-CONTRACT-001
> Last updated: 2026-07-09
> Depends on:
> - `docs/architecture/objgauss-v1-kernel-contract.md`
> - `docs/architecture/assignment-solver-v2-contract.md`
> - `docs/architecture/objectstate-state-variable-gate.md`
> - `docs/dataset/controlled-reality-contract.md`

## Purpose

ObjGauss now has a state-variable gate, real evidence accounting and a
controlled reality dataset contract. This document freezes the model-facing
contract for the next model track:

```text
Gaussian / AssignmentEvidence -> ObjectState Model -> ObjectState -> Gate
```

The goal is not to claim a world model. The goal is to define the first
learnable `Gaussian -> ObjectState` interface that can be trained, exported,
audited and passed into existing identity / prediction / causal gates.

## Current Fact

The repository already has a trainable object emergence line:

```text
AssignmentEvidenceBatch
  -> AssignmentSolverV2
  -> A[N,K]
  -> ObjectStateProjection
  -> Gaussian decoder / renderer validation
```

Therefore `OBJECTSTATE-MODEL-CONTRACT-001` must not create a parallel model
truth. It standardizes the current assignment-first route and names the next
MVP boundary. Transformer encoders, Slot Attention, Sinkhorn / OT, replay
buffers, diffusion and dynamics models remain future proposals unless a later
ADR or PR explicitly introduces them.

## Model IO

### Input: Gaussian Tokens

The conceptual input is an unordered set of Gaussian tokens:

```text
G = {g_1, g_2, ..., g_N}
```

Each token may expose:

```text
GaussianToken = {
  position: R^3
  scale: R^3 optional
  rotation: R^4 optional
  color: R^3 or SH optional
  opacity: R^1 optional
  feature: R^D optional
  frame_index: int optional
  source: str
}
```

MVP implementation uses the existing `AssignmentEvidenceBatch` minimum:

```text
AssignmentEvidenceBatch = {
  positions: R[N,3]
  features: R[N,D]
  frame_index: int
  mask_votes: R[N,M] optional
  track_hints: R[N] optional
  target_assignment: R[N,K] optional
  source: str
}
```

Rules:

- Gaussian tokens are renderer / evidence primitives, not the reasoning unit.
- Inputs are permutation-sensitive only through explicit position / feature
  values; model code must not rely on file row order as identity.
- `target_assignment` is supervised training evidence, not inference-time
  state.
- Hard `object_id` can appear only as target, diagnostic or export address.

### Intermediate: Assignment Matrix

The model must expose one normalized assignment matrix:

```text
A in R[N,K]
```

Meaning:

```text
A[i,k] = probability / soft membership that Gaussian token i belongs to slot k
```

Rules:

- `A[N,K]` is the sole object assignment source.
- Hard labels are derived from `argmax(A)` plus matching / export policy.
- Hard labels cannot become a second source of truth beside `A`.
- Model diagnostics may expose logits, cost, entropy, slot mass and confidence,
  but downstream ObjectState pooling must consume `A`.

### Output: ObjectState

The model outputs object slots as ObjectState candidates:

```text
ObjectState = {
  object_id_embedding: R^D optional
  geometry_embedding: R^D optional
  appearance_embedding: R^D optional
  pose: {
    centroid: R^3
    bbox: R^6 optional
    rotation: R^4 optional
  }
  assignment_probability: R[N] or slot_prob summary
  uncertainty: float or R^D
  confidence: float
}
```

Current v1/v2 code may materialize this as `ObjectStateProjection` fields such
as centroid, bbox, feature, confidence and slot probability. That is the
authoritative MVP mapping until a later PR changes the artifact ABI.

Rules:

- `ObjectState` is the reasoning unit.
- Renderer-facing `object_id` is a derived address, not primary model state.
- A model artifact must preserve enough metadata to reproduce `A -> ObjectState`
  projection and gate handoff.

## MVP Model Family

The first model family remains assignment-first:

```text
Evidence[N] -> C[N,K] -> softmax(-C / temperature) -> A[N,K]
```

The current approved MVP is:

```text
solver_family = cost-softmax-assignment-v2
cost_terms = ["feature", "position", "slot_bias"]
balance_policy = "loss-only-v1"
temporal_policy = "disabled"
matching_policy = "disabled"
```

This is the immediate `OBJECTSTATE-MODEL-MVP-001` implementation target. A
Transformer encoder or Slot Attention module may later be introduced only if it
preserves the same public contract:

```text
Gaussian / AssignmentEvidence -> A[N,K] -> ObjectState
```

and lands with explicit tests, metrics and state handoff.

## Training Contract

The model MVP trains object assignment before dynamics:

```text
L_model =
  lambda_assign   * L_assignment
+ lambda_cluster  * L_cluster
+ lambda_entropy  * L_entropy
+ lambda_balance  * L_balance
+ lambda_identity * L_identity optional
+ lambda_temporal * L_temporal optional
```

MVP required losses:

- supervised assignment cross entropy when `target_assignment` is present;
- cluster / entropy / balance loss families from assignment solver v2.

Deferred losses:

- contrastive identity loss requires explicit identity fixtures or controlled
  real identity rows;
- temporal consistency requires timestamped state sequences;
- prediction and causal losses require transition / action evidence.

Forbidden shortcuts:

- optimizing hard `object_id` equality as identity truth;
- using renderer loss to bypass identity gate;
- treating decreasing training loss as state-variable proof.

## Gate Handoff

Every model candidate must be evaluated through existing gates:

```text
Model artifact
  -> identity predictions / ObjectState candidates
  -> identity gate
  -> prediction gate when transition candidates exist
  -> causal gate when action-conditioned candidates exist
```

Minimum MVP success standard:

```text
Gaussian -> Object assignment -> ObjectState -> gate evaluation
```

Initial MVP does not need to pass real causal evidence. It must produce a
reviewable model artifact and enough candidate output for synthetic identity /
prediction checks, while real controlled rows remain separated in the ledger.

## Artifact Requirements

A model artifact must record:

- schema and model family;
- source evidence schema and dataset / fixture id;
- slot count `K`;
- feature dimension and position dimension;
- solver state / checkpoint payload or reference;
- assignment temperature and cost / loss weights;
- projection policy from `A[N,K]` to ObjectState;
- evaluation summaries used for promotion decisions;
- claim policy and non-goals.

Large checkpoints and training outputs remain in ignored `outputs/` or external
handoff storage. They must not be committed to git.

## Non-Goals

This contract does not:

- introduce a new renderer;
- start controlled capture;
- create ground truth;
- train a new model by itself;
- add torch / CUDA / SAM / DINO / CoTracker as required dependencies;
- introduce replay buffer, diffusion or world dynamics;
- replace `AssignmentSolverV2` with Slot Attention;
- declare ObjectState a proven world-state variable.

## Next PRs

### OBJECTSTATE-ASSIGNMENT-MVP-001

Implemented v0.1 facts:

- Core module: `objgauss.core.objectstate_assignment_mvp`.
- Summary schema: `objgauss-objectstate-assignment-mvp-v1`.
- Entry point:
  `objectstate_assignment_mvp_summary(GaussianCloud, AssignmentSolverV2State, ...)`.
- The summary builds an `AssignmentEvidenceBatch` from Gaussian positions and
  extracted Gaussian features, runs `predict_assignment_solver_v2(...)`, then
  projects the normalized `A[N,K]` into `ObjectStateProjection`.
- If a target assignment is supplied, the summary reports hard assignment
  `mean_best_iou`, `ari` and `purity`.
- Claim policy keeps `A[N,K]` as the single assignment source and marks hard
  `object_id` as derived.
- Non-goals explicitly keep training, renderer loss, GPU, Slot Attention,
  Transformer, replay buffer, diffusion, dynamics and viewer defaults out of
  scope.

### OBJECTSTATE-MODEL-MVP-001

Implement the first model artifact around the current
`AssignmentSolverV2 -> A[N,K] -> ObjectStateProjection` route.

Required output:

- model artifact schema;
- training / inference summary;
- checkpoint or state roundtrip;
- synthetic gate handoff.

### OBJECTSTATE-MODEL-EVAL-001

Evaluate the model artifact through the existing state-variable gates.

Required metrics:

- identity IDF1 / retrieval / drift where applicable;
- prediction state-vs-history error when future candidates exist;
- clear blocked status for causal evidence if no real action candidates exist.

### OBJECTSTATE-MODEL-IDENTITY-GATE-001

Implemented v0.1 facts:

- Core module:
  `objgauss.core.objectstate_model_identity_gate`.
- Summary schema:
  `objgauss-objectstate-model-identity-gate-v1`.
- Entry point:
  `objectstate_model_identity_gate_summary(...)`.
- The gate applies one `AssignmentSolverV2State` to two Gaussian observations,
  projects both assignments into `ObjectStateProjection`, then evaluates
  physical identity labels without treating hard slot ids as identity truth.
- Matching is permutation-aware and dependency-free: retrieval uses
  ObjectState embeddings across frames, while `slot_swap_rate` is reported as a
  diagnostic instead of an automatic failure.
- Required metrics are emitted:
  `identity_retrieval_at_1`, `identity_margin`, `slot_swap_rate`,
  `objectstate_drift`, `assignment_consistency` and `occlusion_recovery`.
- Baselines are included in the same summary:
  `random_assignment`, `xyz_centroid`, `oracle_target_assignment` and
  `assignment_solver_v2`.
- The local artifact bundle includes `identity-summary.json`,
  `identity-matching.json`, `objectstate-retrieval.json`,
  `identity-pairwise-distances.csv`, `assignment-t0.ply` and
  `assignment-t1.ply`.
- This gate does not train a model, enable temporal / matching loss, add a
  Hungarian / scipy dependency, use renderer loss, or claim prediction /
  causal / reality gate success.

### OBJECTSTATE-MODEL-IDENTITY-BENCHMARK-001

Implemented v0.1 facts:

- Core module:
  `objgauss.core.objectstate_model_identity_benchmark`.
- Summary schema:
  `objgauss-objectstate-model-identity-benchmark-v1`.
- Entry point:
  `objectstate_model_identity_benchmark_summary(...)`.
- Runs the existing identity gate for explicit
  `ObjectStateModelIdentityBenchmarkScenario` rows across `viewpoint`,
  `dropout`, `occlusion`, `appearance` and `spatial` perturbations.
- Aggregates `random_assignment`, `xyz_centroid`, `oracle_target_assignment`
  and `assignment_solver_v2` into overall metrics, perturbation breakdowns,
  scenario artifact refs and a `long_training_gate`.
- `candidate_ready` requires solver retrieval > `xyz_centroid`, positive
  identity margin, occlusion recovery > random, bounded reported
  `slot_swap_rate`, and oracle target assignment as retrieval upper bound.
- `slot_swap_rate` is diagnostic, not required to be 0.
- This benchmark does not train, run identity ablation, add temporal loss,
  ingest real capture, mutate viewer defaults or claim prediction / causal /
  reality gate success.

### OBJECTSTATE-MODEL-IDENTITY-BENCHMARK-REPORT-001

Implemented v0.1 facts:

- Core module:
  `objgauss.core.objectstate_model_identity_benchmark_report`.
- Summary schema:
  `objgauss-objectstate-model-identity-benchmark-report-v1`.
- Entry point:
  `write_objectstate_model_identity_benchmark_report(...)`.
- The report generator builds a deterministic controlled synthetic difficulty
  ladder: `viewpoint`, `dropout`, `occlusion`, `appearance` and `spatial`,
  each at `easy`, `medium` and `hard`.
- It writes:
  `identity-benchmark-summary.json`, `identity-benchmark-report.md`,
  `identity-benchmark-breakdown.csv` and an
  `identity-benchmark-artifacts` directory.
- The first committed report is under
  `docs/benchmarks/objectstate-identity-benchmark/`, with per-scenario local
  artifacts written to `/tmp/objgauss-objectstate-identity-benchmark-artifacts`.
- The current deterministic evidence reports `candidate_ready` for a longer
  identity robustness smoke under the report's evidence policy, not for
  world-model training.
- This report remains controlled synthetic evidence only; it does not claim a
  real-data identity pass, identity ablation, temporal assignment, prediction /
  causal / reality gate success or world-model proof.

### OBJECTSTATE-MODEL-IDENTITY-ABLATION-001

Implemented v0.1 facts:

- Core module:
  `objgauss.core.objectstate_model_identity_ablation`.
- Summary schema:
  `objgauss-objectstate-model-identity-ablation-v1`.
- Entry point:
  `objectstate_model_identity_ablation_summary(...)`.
- The ablation reuses the benchmark report ladder exactly: `viewpoint`,
  `dropout`, `occlusion`, `appearance` and `spatial`, each at `easy`,
  `medium` and `hard`, for 15 scenarios / 60 identity pairs.
- Compared evidence policies are `xyz`, `rgb`, `xyz_rgb`,
  `xyz_rgb_opacity` and default optional `semantic`.
- Each policy runs through the existing identity benchmark, so every policy
  retains the same four baselines: `random_assignment`, `xyz_centroid`,
  `oracle_target_assignment` and `assignment_solver_v2`.
- Reported metrics remain identity metrics, not assignment IoU:
  `identity_retrieval_at_1`, `identity_margin`, `slot_swap_rate`,
  `objectstate_drift`, `assignment_consistency` and `occlusion_recovery`.
- The first deterministic controlled run reports
  `objectstate_model_identity_ablation_teacher_evidence_indicated`:
  `semantic` reaches retrieval@1 `1.000000` with positive margin, while native
  Gaussian policies do not pass the candidate gate
  (`rgb=0.783333`, `xyz=0.333333`, `xyz_rgb=0.283333`,
  `xyz_rgb_opacity=0.233333` retrieval@1).
- This means the previous benchmark report's `candidate_ready` result is
  currently explained by synthetic semantic / reference evidence, not by
  native `xyz/rgb/opacity` evidence alone.
- `next_stage_gate.long_training_allowed` remains false. The ablation may
  recommend a Teacher Evidence Layer contract, but it does not unlock long
  training, temporal assignment, real controlled identity pass, prediction /
  causal gates or world-model training.

### OBJECTSTATE-IDENTITY-GATE-POLICY-SCOPING-001

Implemented v0.1 facts:

- `objgauss.core.objectstate_model_identity_benchmark_summary(...)` now records
  an explicit `evidence_policy` object for each benchmark run.
- `long_training_gate` is no longer interpretable as global. It now carries:
  `candidate_ready_is_policy_scoped=true`, `scoped_to_policy` and a full
  `scope` payload matching `evidence_policy`.
- `write_objectstate_model_identity_benchmark_report(...)` marks the committed
  report's feature-backed reference as policy `semantic`, source
  `synthetic_report_feature_backed_reference`, with
  `native_gaussian_evidence_only=false` and `uses_semantic_evidence=true`.
- The committed benchmark summary and Markdown report under
  `docs/benchmarks/objectstate-identity-benchmark/` have been regenerated with
  this policy scope.
- Identity ablation variants now pass their own evidence policy into the
  benchmark, so native and semantic gates can be read separately.
- The ablation next-stage gate explicitly reports:
  `native_long_training_gate=blocked` and
  `semantic_long_training_gate=candidate_ready` for the current deterministic
  run.
- This is a scoping fix only. It does not define `TeacherEvidenceBatch`, run a
  leakage audit, introduce DINO / SAM / tracking dependencies, train a model,
  enable temporal loss, run controlled real capture or unlock world-model
  training.

### OBJECTSTATE-ASSIGNMENT-TRAIN-CONTRACT-001

Implemented v0.1 facts:

- Architecture spec:
  `docs/architecture/objectstate-assignment-train-contract.md`.
- Freezes the bounded smoke-training contract before any long run.
- Defines dataset summary, model handoff, loss families, checkpoint metadata,
  before / after assignment metrics, visualization refs and long-training gate.
- Keeps the first train implementation on the existing assignment-first public
  contract:

```text
AssignmentEvidenceBatch -> logits/cost[N,K] -> softmax -> A[N,K]
  -> ObjectStateProjection
```

- Requires loss decrease, assignment metric improvement, visualization and
  valid gate handoff before any 10 minute or longer training run.
