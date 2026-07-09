# ObjectState Controlled Capture Handoff

> Status: current / OBJECTSTATE-CONTROLLED-CAPTURE-001
> Last updated: 2026-07-09
> Depends on:
> - `docs/architecture/objectstate-model-contract.md`
> - `docs/architecture/objectstate-temporal-assignment-contract.md`
> - `docs/training/controlled-real-capture-runbook.md`

## Purpose

This handoff returns the ObjectState route from deterministic synthetic
temporal evidence to the existing controlled capture toolchain.

The preceding sequence established:

```text
TeacherEvidenceBatch
    ↓
AssignmentSolverV2
    ↓
ObjectStateProjection
    ↓
identity benchmark
    ↓
bounded semantic long smoke
    ↓
temporal consistency smoke
```

This document defines the next boundary:

```text
temporal assignment pass
    ↓
controlled capture environment preflight
    ↓
local controlled capture bundle readiness
    ↓
controlled identity / prediction / intervention handoff
```

It does not capture video, create GT, reconstruct Gaussians or claim reality
gate pass.

## Required Preconditions

The handoff requires:

- passed `objgauss-objectstate-temporal-assignment-v1` summary;
- `next_stage_gate.controlled_capture_allowed=true`;
- controlled capture environment preflight summary, when collection is being
  prepared;
- controlled capture bundle readiness summary, when existing real evidence is
  being handed off.

If only the temporal summary exists, the handoff is valid but blocked and tells
the operator to run capture host preflight.

## Machine Contract

Core module:

```text
objgauss.core.objectstate_controlled_capture_handoff
```

Schema:

```text
objgauss-objectstate-controlled-capture-handoff-v1
```

Public API:

- `objectstate_controlled_capture_handoff_summary(...)`
- `validate_objectstate_controlled_capture_handoff_summary(...)`

Status values:

- `objectstate_controlled_capture_handoff_blocked`
- `objectstate_controlled_capture_collection_ready`
- `objectstate_controlled_capture_handoff_ready`

`collection_ready` means the model route and capture environment are ready to
collect a bundle. It is not real-evidence pass. `handoff_ready` requires a
ready controlled capture bundle summary.

## Existing Routes

The handoff points to existing commands instead of duplicating capture logic:

```text
uv run objgauss object-state audit-controlled-capture-environment
uv run objgauss object-state init-controlled-capture-bundle outputs/captures/<sample_id>
uv run objgauss object-state audit-controlled-capture-bundle-readiness outputs/captures/<sample_id> --summary-output outputs/captures/<sample_id>/readiness-summary.json
uv run objgauss object-state controlled-identity-bundle-handoff <bundle-root> <objectstate-artifact.json>
```

Captured RGB, Gaussian reconstructions, annotations and candidate artifacts
remain local / ignored under `outputs/captures/` unless a separate asset /
release decision says otherwise.

## Non-Goals

This handoff does not:

- capture video;
- create ground truth;
- reconstruct Gaussians;
- run identity / prediction / intervention handoff;
- run reality gate or ledger package audit;
- train any model;
- use renderer loss, dynamics, diffusion or replay buffer;
- write public samples;
- change viewer or export defaults;
- claim real-data identity / prediction / causal / reality gate pass;
- claim a world model.
