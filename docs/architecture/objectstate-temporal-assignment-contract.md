# ObjectState Temporal Assignment Contract

> Status: current / OBJECTSTATE-TEMPORAL-ASSIGNMENT-CONTRACT-001
> Last updated: 2026-07-09
> Depends on:
> - `docs/architecture/objectstate-model-contract.md`
> - `docs/architecture/assignment-solver-v2-contract.md`
> - `docs/architecture/objectstate-teacher-evidence-contract.md`
> - `docs/architecture/objectstate-assignment-long-smoke-contract.md`

## Purpose

This contract freezes the first temporal consistency boundary after the
semantic-policy assignment long smoke.

It exists because temporal assignment should not bind the wrong identity more
strongly. Temporal loss is allowed to become an implementation target only
after the semantic evidence route has passed the bounded long smoke and
explicitly allows the temporal contract.

This contract does not run temporal training and does not change
`AssignmentSolverV2Config.temporal_policy`, which remains disabled in the
current runtime.

## Required Route

The only allowed route remains evidence-conditioned assignment:

```text
Gaussian_t
    +
TeacherEvidenceBatch_t
    ↓
A_t[N,K]
    ↓
ObjectState_t

Gaussian_t+1
    +
TeacherEvidenceBatch_t+1
    ↓
A_t+1[N,K]
    ↓
ObjectState_t+1

    ↓

temporal consistency audit
```

Native Gaussian-only temporal training remains blocked. Hard `object_id` is
not a second source of truth; it can only be derived from `A[N,K]` or a
contracted slot-match manifest.

## Preconditions

The temporal assignment implementation may start only when all are true:

- `OBJECTSTATE-TEACHER-EVIDENCE-CONTRACT-001` exists.
- `OBJECTSTATE-TEACHER-EVIDENCE-LEAKAGE-AUDIT-001` has passed.
- `OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-001` has status
  `objectstate_assignment_long_smoke_pass`.
- The long-smoke summary uses `policy=semantic`.
- The long-smoke summary has
  `next_stage_gate.temporal_assignment_contract_allowed=true`.

Without a passed long-smoke summary, the contract summary is valid but blocked.

## Inputs

The future temporal implementation must consume paired frames:

- `gaussian_t`
- `teacher_evidence_t`
- `assignment_t`
- `objectstate_t`
- `gaussian_t_plus_1`
- `teacher_evidence_t_plus_1`
- `assignment_t_plus_1`
- `objectstate_t_plus_1`

Optional track hints may be used as evidence, but they are not ground-truth
identity. Physical identity labels remain evaluation-only.

## Allowed Loss Terms

The future implementation may introduce only these temporal loss families:

- `assignment_consistency`: matched Gaussian / track evidence should keep
  compatible assignment distributions across adjacent frames.
- `objectstate_embedding_consistency`: matched slots should keep nearby
  ObjectState embeddings across adjacent frames.
- `slot_transition_smoothness`: slot prototypes should move smoothly under a
  bounded frame interval.

Forbidden terms for the first temporal implementation:

- renderer loss;
- dynamics loss;
- diffusion loss;
- replay-buffer loss;
- oracle identity loss.

## Required Metrics

The implementation must report:

- `temporal_assignment_consistency`
- `identity_retrieval_at_1`
- `identity_margin`
- `slot_swap_rate`
- `occlusion_recovery`
- `track_fragmentation_rate`
- `checkpoint_roundtrip`

Loss decrease alone is not sufficient.

## Machine Contract

Core module:

```text
objgauss.pipelines.objectstate_temporal_assignment_contract
```

Schemas:

```text
objgauss-objectstate-temporal-assignment-contract-v1
objgauss-objectstate-temporal-assignment-contract-summary-v1
```

Public API:

- `ObjectStateTemporalAssignmentContractThresholds`
- `objectstate_temporal_assignment_contract_summary(...)`
- `validate_objectstate_temporal_assignment_contract_summary(...)`

The summary is ready only when a passed semantic assignment long-smoke summary
is provided. A ready summary points to
`OBJECTSTATE-TEMPORAL-ASSIGNMENT-001`.

## OBJECTSTATE-TEMPORAL-ASSIGNMENT-001 Implementation

Implemented run module:

```text
objgauss.pipelines.objectstate_temporal_assignment
```

Run summary schema:

```text
objgauss-objectstate-temporal-assignment-v1
```

Public API:

- `objectstate_temporal_assignment_summary(...)`
- `validate_objectstate_temporal_assignment_summary(...)`

The first implementation consumes:

- a passed `objgauss-objectstate-assignment-long-smoke-v1` summary;
- a ready
  `objgauss-objectstate-temporal-assignment-contract-summary-v1`;
- the long-smoke `AssignmentSolverV2` checkpoint.

It restores the solver checkpoint, reuses the 15-scenario identity benchmark
ladder and writes a temporal slot-match manifest. The manifest compares
derived assignment slots across frame pairs and reports temporal consistency
metrics. Physical identity labels are used only to evaluate consistency, not
as runtime input.

This implementation is a bounded temporal consistency smoke, not temporal
training. It keeps `AssignmentSolverV2Config.temporal_policy="disabled"` and
does not optimize temporal loss. A passing summary only allows the project to
return to controlled capture preparation.

## Non-Goals

This contract does not:

- run temporal assignment training;
- enable `AssignmentSolverV2Config.temporal_policy`;
- use renderer loss;
- introduce dynamics, diffusion or replay buffer;
- run teacher models or download weights;
- use GPU / torch / CUDA;
- ingest controlled real capture;
- change viewer or export defaults;
- claim temporal assignment implementation pass;
- claim real-data identity / prediction / causal / reality gate pass;
- claim a world model.
