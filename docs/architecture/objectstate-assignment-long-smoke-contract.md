# ObjectState Assignment Long Smoke Contract

> Status: current / OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-CONTRACT-001
> Last updated: 2026-07-09
> Depends on:
> - `docs/architecture/objectstate-model-contract.md`
> - `docs/architecture/assignment-solver-v2-contract.md`
> - `docs/architecture/objectstate-teacher-evidence-contract.md`
> - `docs/architecture/objectstate-assignment-train-contract.md`

## Purpose

This contract freezes the first bounded semantic-policy long smoke before any
actual longer training run.

It exists because identity ablation showed that current identity readiness is
not supported by native `xyz/rgb/opacity` evidence alone. A long smoke is only
meaningful under the evidence policy that explains the identity candidate:

```text
policy = semantic
```

This contract does not run training. It defines the prerequisites, hard limits,
required artifacts and exit criteria for the later
`OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-001` implementation.

## Required Route

The only allowed route is evidence-conditioned assignment:

```text
GaussianCloud
    +
TeacherEvidenceBatch
    ↓
AssignmentSolverV2
    ↓
A[N,K]
    ↓
ObjectStateProjection
    ↓
before / after identity benchmark
```

Native Gaussian-only long training remains blocked. `xyz`, `rgb`,
`xyz_rgb` and `xyz_rgb_opacity` policies do not satisfy this contract.

## Preconditions

The long smoke implementation may start only when all are true:

- `OBJECTSTATE-TEACHER-EVIDENCE-CONTRACT-001` exists.
- `OBJECTSTATE-TEACHER-EVIDENCE-LEAKAGE-AUDIT-001` has passed.
- The leakage audit reports
  `semantic_teacher_evidence_training_allowed=true`.
- The teacher evidence source is inference-time, such as `dino_v2`, `clip`,
  `sam2`, `grounding_dino`, `tracking` or `teacher_fusion`.
- Provenance declares
  `train_test_semantic_source_split.direct_object_id_embedding_shared=false`.

The synthetic report one-hot semantic fixture is evaluation evidence only. It
does not clear this contract by default.

## Hard Training Limits

The future run must stay bounded:

```text
duration <= 10 minutes
fixed seed
policy = semantic
no dynamics
no diffusion
no replay buffer
no renderer loss
no temporal loss
checkpoint roundtrip required
before / after identity benchmark required
held-out generalization required
```

Large checkpoints and run outputs stay under ignored local output directories.
They must not be committed to git.

## Required Run Artifacts

`OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-001` must write or reference:

- before identity benchmark summary;
- after identity benchmark summary;
- held-out generalization summary;
- checkpoint ref;
- checkpoint roundtrip summary;
- loss curve;
- fixed seed and run config.

Loss decrease alone is not a success criterion.

## Success Criteria

The run succeeds only if all criteria are evaluated and pass:

- `held_out_identity_retrieval_at_1_not_decrease`:
  held-out `identity_retrieval_at_1` after training must not decrease.
- `identity_margin_improves`:
  held-out `identity_margin` must improve.
- `occlusion_recovery_not_decrease`:
  held-out `occlusion_recovery` must not decrease.
- `generalization_gap_not_expand`:
  train-vs-held-out generalization gap must not expand.
- `slot_swap_rate_interpretable`:
  `slot_swap_rate` must be finite, bounded and explicitly reported.
- `checkpoint_roundtrip`:
  restored checkpoint must reproduce the relevant assignment / identity summary.

If any criterion is missing, the run is blocked or reviewable, not pass.

## Machine Contract

Core module:

```text
objgauss.pipelines.objectstate_assignment_long_smoke_contract
```

Schemas:

```text
objgauss-objectstate-assignment-long-smoke-contract-v1
objgauss-objectstate-assignment-long-smoke-contract-summary-v1
```

Public API:

- `ObjectStateAssignmentLongSmokeContractThresholds`
- `objectstate_assignment_long_smoke_contract_summary(...)`
- `validate_objectstate_assignment_long_smoke_contract_summary(...)`

The contract summary is ready only when passed leakage audit evidence is
provided. Without that audit, the summary is valid but blocked.

## OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-001 Implementation

Implemented run module:

```text
objgauss.pipelines.objectstate_assignment_long_smoke
```

Run summary schema:

```text
objgauss-objectstate-assignment-long-smoke-v1
```

Public API:

- `objectstate_assignment_long_smoke_summary(...)`
- `validate_objectstate_assignment_long_smoke_summary(...)`

The first implementation is a bounded deterministic CPU synthetic smoke. It
does not run DINO / CLIP / SAM / tracking teachers. Instead, it consumes a
passed `OBJECTSTATE-TEACHER-EVIDENCE-LEAKAGE-AUDIT-001` summary as the gate
that permits semantic-policy training.

The implementation reuses the identity benchmark report ladder:

- train split: `easy` and `medium` scenarios;
- held-out split: `hard` scenarios;
- policy: `semantic`;
- training: supervised `AssignmentSolverV2` update on teacher semantic
  evidence only;
- artifacts: before / after benchmark summaries, held-out restored-checkpoint
  benchmark summary, final solver checkpoint JSON and run summary JSON.

The run status is `objectstate_assignment_long_smoke_pass` only when all
contract checks pass. `generalization_gap_not_expand` is evaluated on
retrieval, occlusion recovery and slot-swap stability; held-out
`identity_margin` is evaluated by its own required improvement check.

This implementation only authorizes the next contract slice:
`OBJECTSTATE-TEMPORAL-ASSIGNMENT-CONTRACT-001`. It does not unlock large
training, temporal loss, renderer loss, real-data pass claims or world-model
training.

## Non-Goals

This contract does not:

- run long smoke training;
- introduce GPU, torch or CUDA requirements;
- run renderer loss;
- add temporal / matching loss;
- introduce dynamics, diffusion or replay buffer;
- use native Gaussian-only evidence policy;
- ingest controlled real capture;
- change viewer or export defaults;
- claim identity / prediction / causal / reality gate pass;
- claim a world model.
