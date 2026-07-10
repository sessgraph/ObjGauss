# ADR 0007: Research-first stabilization

- Status: Accepted
- Date: 2026-07-10
- Owner decision: freeze new schema / audit / handoff work for 1–2 weeks and
  make real ObjectState evidence the only research priority.

## Context

ObjGauss has a working Gaussian viewer, assignment kernel, training smokes,
controlled-data contracts, evaluators, and evidence ledgers. It also has more
orchestration and reporting surface than the current real evidence can justify:
the authoritative public replay ledger contains two samples and no identity or
intervention pass. Several gates still validate fixtures, caller declarations,
or pre-associated rows rather than independently measuring model output.

At the same time, the default viewer loads more assets than it displays and
offers controls whose Spark source behavior is only partially implemented.
Continuing to add schemas, audits, handoffs, or product features would increase
surface area without strengthening the central claim.

## Decision

From 2026-07-10 through 2026-07-24, the repository is in research-first
stabilization.

1. Do not add a new schema, audit command, handoff wrapper, renderer route, or
   viewer feature. A change is allowed only when it removes false evidence,
   fixes correctness/reproducibility, or is required to run the existing real
   experiment.
2. Treat the viewer as an evidence viewer. It must load on demand and only show
   controls that affect the rendered Gaussian source truthfully.
3. The next research result must come from at least three real controlled
   scenes with physical identity, timestamped 6DoF pose, occlusion/view change,
   and measured non-zero action. Existing tools and contracts must be reused.
4. Teacher, identity, prediction, intervention, and reality decisions must be
   recomputed from raw candidate output and ground truth. Caller-declared
   `pass`, provenance labels, or pre-associated GT rows are not sufficient.
5. `objgauss.core` is reserved for Gaussian, assignment, ObjectState, decoder,
   and metric primitives. Dataset adapters, filesystem workflows, reports, and
   orchestration move outward behind compatibility imports in later slices.
6. Current state files are snapshots. Append-only history is archived rather
   than copied into every active status update.

## Exit criteria

The stabilization window may end only when all of the following are true:

- the supervised CE clip gradient is mathematically consistent and bounded;
- ObjectState persistent identity, renderer address, and confidence obey the
  frozen ABI end to end;
- the viewer does not fetch hidden catalog entries and Spark-visible behavior
  matches the controls shown to users;
- required CI, license, and reproducible dependency instructions exist;
- gate decisions are independently computed from the evidence they claim to
  evaluate;
- at least three real controlled scenes have reviewable raw evidence and
  identity/prediction/intervention results;
- active status and queue files stay concise, with history in `docs/state/archive/`.

## Consequences

- Existing schemas and compatibility entry points remain available during the
  freeze, but no new peer schema is introduced.
- Negative real results are expected and count as progress; fixture pass does
  not unlock larger training.
- Viewer polish, new renderer paths, diffusion, replay buffers, and world-model
  expansion remain deferred.
