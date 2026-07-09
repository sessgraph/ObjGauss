# Controlled Reality Dataset Contract

> Status: current
> Last updated: 2026-07-09

This document freezes `OBJECTSTATE-CONTROLLED-DATASET-CONTRACT-001`.
It defines the first real-data language that can feed ObjGauss Phase 2
Reality Evidence Foundation.

The contract does not replace the existing capture schema. The authoritative
machine schema remains `objgauss-objectstate-controlled-capture-manifest-v1`
in `objgauss.core.objectstate_controlled_capture`; this document names the
dataset concepts and invariants that a controlled real bundle must satisfy
before it can become useful ObjectState evidence.

## Goal

Build controlled real evidence that can support all three state-variable
questions:

```text
Identity:   does the same physical object persist across time?
Prediction: does S(t) connect to S(t+n)?
Causal:     does an action overlap a real object transition?
```

The output of this slice is a reviewable dataset contract, not a trained
model, public demo, viewer change, replay buffer, diffusion model, or claim
that the reality gate passes.

## Dataset Language

### Episode

An episode is represented by `sample` plus the ordered `frames` in the
controlled capture manifest.

Required fields are:

```yaml
episode_id: sample.sample_id
scene_id: sample.scenario
camera: sample.capture_device
objects: objects[]
actions: actions[]
frames: frames[]
```

Frame timestamps must be strictly increasing. Frames must carry RGB
observation references, and Phase 2 evidence rows should also carry per-frame
Gaussian evidence before handoff.

### Object Instance

An object instance is represented by `objects[]` and per-frame object
annotations inside `frames[].objects[]`.

Required fields are:

```yaml
object_id: stable physical object id
category: object category
geometry_reference: dimensions_m or artifact refs when available
initial_pose: first timestamped pose annotation when pose GT is available
```

`object_id` is physical identity ground truth in a controlled bundle. It is
not the same fact as renderer-facing `object_id` from public static artifacts.

### Action Event

An action event is represented by `actions[]`.

Required fields are:

```yaml
action_id: stable action id
actor: human, robot, scripted tool, or other declared actor
object_id: affected physical object
action_type: push_left, push_right, hold, or another declared type
vector: non-zero 3D vector for intervention rows
start_timestamp: action interval start
end_timestamp: action interval end
```

Action rows with no non-zero vector can be valid capture notes, but they are
not causal-ready evidence.

### State Transition

A state transition is derived from consecutive timestamped 6DoF pose
annotations for the same `object_id`.

Required fields are:

```yaml
object_id: stable physical object id
before_timestamp: source frame timestamp
after_timestamp: target frame timestamp
delta_position: target.position - source.position
delta_rotation: target.rotation - source.rotation accounting
```

The transition is a data contract concept. Evaluators may later materialize it
as `StateTransitionRow` in the real evidence bundle ledger.

## Invariants

### Identity Invariant

The same `object_id` must exist across timestamps. At minimum, the controlled
capture summary must report `identity_stage_ready=true`.

This makes identity rows eligible to become pass or fail evidence after a
candidate model supplies explicit identity predictions and metrics.

### Prediction Invariant

The bundle must contain:

```text
S(t) -> S(t+n)
```

In practice this means timestamped 6DoF pose tracks are present, so the
controlled capture summary reports `prediction_stage_ready=true`.

Prediction readiness is not a model result. A later evaluator must still
compare ObjectState prediction against a baseline such as history or Kalman.

### Causal Invariant

The bundle must contain:

```text
Action(t0,t1) overlaps StateTransition(t,t+n)
```

The action must have a non-zero vector and fit inside a referenced object's
consecutive pose transition interval. Otherwise the bundle cannot enter the
causal ledger as pass or fail evidence.

## Machine Audit

`objgauss.core.controlled_schema` exposes:

- `OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SCHEMA`
- `OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SUMMARY_SCHEMA`
- `objectstate_controlled_dataset_contract_summary(...)`
- `validate_objectstate_controlled_dataset_contract_summary(...)`

The summary reuses `objgauss-objectstate-controlled-capture-manifest-v1` and
reports:

- dataset language mapping for episode, object instances, action events and
  transition source;
- identity, prediction and causal invariant readiness;
- hard blockers for missing pose, missing action GT, zero vectors, or missing
  action-transition overlap;
- claim policy that this contract does not create GT, score a candidate,
  claim a reality gate pass, or claim a world model.

## Phase 2 Boundary

This contract is the entry point for real data foundation work:

```text
controlled capture bundle
        -> controlled dataset contract summary
        -> real evidence bundle / row ledger
        -> identity / prediction / intervention evaluators
```

Continue to keep synthetic, public replay, and controlled-real evidence
separate in the ledger. Do not use static RGB, masks, object detection, or
action labels alone as proof of ObjectState.
