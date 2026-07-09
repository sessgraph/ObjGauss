# ObjectState Teacher Evidence Contract

> Status: current / OBJECTSTATE-TEACHER-EVIDENCE-CONTRACT-001
> Last updated: 2026-07-09
> Depends on:
> - `docs/architecture/objectstate-model-contract.md`
> - `docs/architecture/assignment-solver-v2-contract.md`

## Purpose

`OBJECTSTATE-MODEL-IDENTITY-ABLATION-001` showed that the current identity
candidate signal is explained by semantic / reference evidence, not by native
`xyz/rgb/opacity` evidence alone. This document freezes the first Teacher
Evidence Layer contract before any long assignment or identity training.

The contract answers one narrow question:

```text
What semantic perception evidence is allowed to enter AssignmentSolverV2,
and under what provenance / leakage constraints?
```

Teacher evidence is perception evidence. It is not ground-truth identity, not
`target_assignment`, not physical object labels, and not proof that ObjectState
has learned an identity or world model.

## Route

The intended model route remains assignment-first:

```text
GaussianCloud
  + TeacherEvidenceBatch
  -> AssignmentEvidenceBatch
  -> AssignmentSolverV2
  -> A[N,K]
  -> ObjectStateProjection
```

`TeacherEvidenceBatch` can supply feature evidence to assignment, but it cannot
become a second assignment source of truth. Hard identity labels remain
evaluation or supervised target material only, never inference-time teacher
evidence.

## Batch Schema

Core schema:

```text
objgauss-objectstate-teacher-evidence-batch-v1
```

Required fields:

- `sample_id`: non-empty sample id.
- `gaussian_ids`: unique Gaussian token ids, length `N`.
- `evidence_policy`: policy name, default `semantic`.
- `feature_matrix`: finite `float32` matrix with shape `N x D`.
- `source`: one declared teacher evidence source.
- `confidence`: scalar or `N`-vector in `[0,1]`; defaults to `1.0`.
- `uncertainty`: scalar or `N`-vector in `[0,1]`; defaults to `0.0`.
- `provenance`: required metadata proving how the features were produced.
- `allowed_for_training`: whether the batch may enter training.
- `allowed_for_evaluation`: whether the batch may enter evaluation / audit.
- `leakage_risk`: declared risk level.

Summary artifacts intentionally do not inline feature values. They record
matrix shape, dtype, confidence / uncertainty statistics, provenance,
permissions and leakage policy.

## Sources

Allowed sources:

```text
dino_v2
clip
sam2
grounding_dino
tracking
teacher_fusion
synthetic_semantic
manual_fixture
```

Inference-time sources that can be considered for training after provenance and
risk checks:

```text
dino_v2
clip
sam2
grounding_dino
tracking
teacher_fusion
```

`synthetic_semantic` and `manual_fixture` are allowed for controlled evaluation,
fixture tests and contract development, but they are not training sources by
default. They must not be used to justify long training unless a later leakage
audit and Owner decision explicitly changes that status.

## Provenance

Every batch must include:

```text
producer
feature_space
input_refs
generation_method
```

Forbidden provenance keys, including nested keys:

```text
physical_identity
physical_identity_label
identity_label
target_assignment
oracle_object_id
gt_object_id
ground_truth_object_id
test_label
```

If any forbidden key appears, the batch is rejected. The intent is to prevent a
semantic feature from being a disguised answer key.

## Leakage Policy

Allowed leakage risk levels:

```text
none
low
medium
high
```

Training requires all of the following:

- `allowed_for_training=true`.
- `source` is an inference-time teacher source.
- `leakage_risk` is `none` or `low`.
- Provenance contains all required keys.
- Provenance contains no forbidden GT / identity keys.

Evaluation can include medium or high risk fixture evidence only when the risk
is explicit. Such evidence can diagnose behavior, but it cannot unlock long
training.

## Required Follow-Up Audits

`OBJECTSTATE-TEACHER-EVIDENCE-LEAKAGE-AUDIT-001` adds the first machine audit
for this contract.

Core audit schema:

```text
objgauss-objectstate-teacher-evidence-leakage-audit-v1
```

The audit covers:

- Semantic feature shuffle.
- Physical label ban.
- Random semantic baseline.
- Train / test semantic source split.

The deterministic report-ladder audit runs three identity benchmark variants:

```text
semantic_reference
semantic_feature_shuffle
random_semantic_baseline
```

The shuffle path must reduce retrieval and margin, proving the model is using
semantic evidence. The random semantic baseline must stay near random /
`xyz_centroid`, proving arbitrary embeddings are not enough. The physical label
ban rejects forbidden GT / identity provenance keys. The train / test split
check requires at least one training-allowed inference-time teacher batch with
`train_test_semantic_source_split.direct_object_id_embedding_shared=false`.

Important current interpretation:

- The existing synthetic report one-hot semantic fixture is still evaluation
  evidence only.
- By default, the leakage audit blocks training clearance for that fixture
  because it has no training-allowed inference-time source split.
- Passing the stressor checks does not by itself unlock long training; the
  teacher source split must also clear.

## Core API

Machine contract entry points:

- `TeacherEvidenceBatch`
- `TeacherEvidenceLeakageAuditThresholds`
- `validate_teacher_evidence_batch(...)`
- `teacher_evidence_batch_summary(...)`
- `validate_teacher_evidence_batch_summary(...)`
- `objectstate_teacher_evidence_contract_summary(...)`
- `validate_objectstate_teacher_evidence_contract_summary(...)`
- `objectstate_teacher_evidence_leakage_audit_summary(...)`
- `validate_objectstate_teacher_evidence_leakage_audit_summary(...)`

Schemas:

- `OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA`
- `OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SCHEMA`
- `OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SUMMARY_SCHEMA`
- `OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA`

## Non-Goals

This contract and audit do not:

- Download or run DINO, CLIP, SAM, GroundingDINO or tracking models.
- Define a teacher feature extraction pipeline.
- Clear synthetic one-hot semantic fixtures for training by default.
- Train `AssignmentSolverV2` or identity encoders.
- Run a long smoke.
- Introduce renderer loss, temporal loss or matching loss.
- Change viewer or export defaults.
- Claim real-data identity / prediction / causal / reality gate pass.
- Claim that ObjGauss has learned a world model.
