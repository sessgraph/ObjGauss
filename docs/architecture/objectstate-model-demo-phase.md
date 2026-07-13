# ObjectState Model Demo Phase

> Status: accepted / Phase M (Phase 2.5)
> Last updated: 2026-07-13
> Decision: `docs/adr/0008-objectstate-model-demo-phase.md`
> Reuses: `docs/architecture/objectstate-assignment-train-contract.md`

## Observable result

One command must turn one registered public research Gaussian sample into one
traceable local demo run:

```text
registered Gaussian + supervised assignment evidence
  -> Object Encoder -> Assignment Head -> A[N,K] -> ObjectStateProjection
  -> checkpoint + before/after PLY + metrics + model manifest
  -> existing Viewer
```

The Viewer must make the provenance boundary visible: raw input, learned
object-color output, object slots and run metrics belong to the same `run_id`.

Phase M has two evidence levels:

- M0: existing AssignmentSolverV2 same-sample wiring smoke;
- M1: ObjectState Model v0 with a train/held-out-frame split and browser
  checkpoint inference.

M0 does not satisfy M1 acceptance.

## ObjectState Model v0 contract

The v0 model is deliberately small and dependency-free:

```text
Gaussian [xyz, rgb, opacity]
  -> train-split feature standardization
  -> tanh Object Encoder
  -> linear Assignment Head
  -> softmax A[N,K]
  -> canonical ObjectStateProjection
```

Checkpoint schema:

```text
objgauss-objectstate-model-v0-state-v1
```

It records input feature order, train-only mean/std, encoder/head weights and
biases, hidden dimension, slot count, seed, step and loss weights. Browser and
Python inference must implement the same equations and output ordering.

Training schema:

```text
objgauss-objectstate-model-v0-training-v1
```

The first public-data split uses `source_frame`: deterministic frame ids are
assigned wholly to train or held-out, so no Gaussian row appears in both.
`object_id` is consumed only as train supervision and held-out evaluation GT.
The report must include frame ids, row counts, object counts and overlap checks.
This proves same-scene held-out-frame behavior, not cross-scene generalization.

The v0 objective is:

```text
L = assignment_weight * supervised_CE
  + compactness_weight * spatial_slot_variance
  + semantic_weight * rgb_slot_variance
```

Spatial and RGB consistency terms are computed from soft assignments and may
shape training without using held-out labels. Slot centroids are recomputed from
the current train assignment. Exact loss components and weights are recorded.

Model v0 acceptance requires total and supervised train loss to decrease,
finite non-collapsed assignments, checkpoint roundtrip equality and held-out
ARI/mean-best-IoU/Purity against a recorded untrained baseline. A result must
remain reviewable rather than pass when the held-out object coverage differs
from train or a frame/row overlap is detected.

## Canonical inputs and outputs

The input record uses the existing assignment-train dataset contract. It must
record `sample_id`, `source_kind`, Gaussian reference, target source, split and
license/use restriction. Deriving a supervised one-hot target from an existing
`object_id` field is allowed only when the field's provenance is recorded.

The run writes under `outputs/model-demo/<run_id>/` by default:

```text
checkpoint.json
before.ply
after-object-color.ply
metrics.json
model-manifest.json
```

These filenames are the command-level UX; their contents use the accepted M1
contracts:

- `checkpoint.json`: `objgauss-objectstate-model-v0-state-v1`;
- `metrics.json`: `objgauss-objectstate-model-v0-training-v1`;
- `model-manifest.json`: `objgauss-model-artifact-manifest-v1`;
- PLY object address: renderer-facing `object_id = argmax(A)`; a diagnostic
  `predicted_object_id` may mirror it but cannot be the only address field.

M1 adds browser-ready `model_input` and `objectstate_model` roles to the existing
model artifact manifest. `model_input` is a bounded demo PLY, not the default
full diagnostic route. `objectstate_model` is the Model v0 JSON checkpoint. The
Viewer must run checkpoint inference on `model_input`, compare its derived ids
with the precomputed `object_edit` artifact and surface a mismatch as an error.

No output may contain a caller-authored pass flag that overrides recomputed
metrics.

## Metrics and acceptance

The demo reports before and after for:

- supervised assignment loss;
- ARI;
- mean-best-IoU;
- Purity;
- assignment confidence, normalized entropy and effective slots;
- active ObjectState count.

Acceptance requires:

1. the command starts from the registered source and finishes without manually
   editing generated artifacts;
2. checkpoint reload reproduces the recorded final assignment;
3. total loss decreases, and metric movement is reported without hiding a
   near-perfect initial state or slot collapse;
4. `before.ply` and `after-object-color.ply` preserve Gaussian count and contain
   finite renderer-facing `object_id` values;
5. manifest hashes validate and resolve only artifacts from the same run;
6. the Viewer can select the run, switch raw/object-color evidence, filter
   objects and show the run's ARI/IoU/Purity rather than fixture metrics;
7. automated Python tests, production build and Viewer audit pass; a visible UI
   change additionally needs a real browser check before integration is called
   complete.
8. for M1, Viewer telemetry reports checkpoint schema, inference row count and
   precomputed-prediction agreement; the page must not silently fall back to
   the precomputed PLY when checkpoint inference fails.

## Work slices

### OBJECTSTATE-PUBLIC-DATASET-TRAIN-001

- Add a canonical CLI entry over the bounded Model v0 training function.
- Use one registered, small public research sample first.
- Emit the Model v0 checkpoint and canonical held-out-frame run summary.
- Keep training outputs ignored and record target/license provenance.

Canonical local command:

```bash
uv run objgauss training objectstate-model-demo public/samples/lego_alpha_v1_objects.ply \
  --output-dir outputs/model-demo/nerf-lego-phase-m1-model-v0 \
  --run-id nerf-lego-phase-m1-model-v0 \
  --license "NeRF official sample data; training/research use only" \
  --target-source "object_id from Lego alpha closure 2D color-mask voting; supervised train target and held-out evaluation GT" \
  --source-url https://github.com/bmild/nerf \
  --source-splat public/samples/lego_alpha_proxy.splat \
  --viewer-dir public/models/objectstate-model-demo-local \
  --hidden-dim 24 --heldout-stride 4 \
  --iterations 240 --learning-rate 0.08 --seed 0 --require-pass
```

When `--viewer-dir` is under `public/`, the CLI prints a same-origin route such
as:

```text
/?modelArtifactManifest=/models/objectstate-model-demo-local/model-manifest.json
```

The local package contains only files from the same validated run, rewrites all
manifest references to same-directory paths and verifies artifact/checkpoint
hashes before writing the manifest. The canonical local package directory is
gitignored; it is not a publication step.

The registered Lego files are local research assets and may be absent from a
clean checkout. The command must fail rather than synthesize a replacement.

### OBJECTSTATE-DEMO-EVAL-001

- Give the existing before/after visualizations stable demo filenames.
- Fix PLY renderer address compatibility (`object_id`).
- Produce a same-run manifest and validate hashes/references.
- Test metric recomputation and checkpoint roundtrip.

### OBJECTSTATE-VISUALIZATION-001

- Reuse the current catalog, object list, source switch and metric components.
- Add only the missing route from the existing manifest/run-summary roles.
- Do not create a second dashboard, renderer or metrics implementation.
- Label fixture, supervised train, held-out eval and real evidence distinctly.

For ad hoc ignored outputs, users may still select `model-manifest.json`,
`model-input.ply`, `checkpoint.json`, `after-object-color.ply`,
`quality-report.json`, `metrics.json` and the source `.splat` together through
the existing model-bundle import. The staged local package is the preferred
one-link path. The browser runs checkpoint inference, but does not run training.

## Explicit non-goals

- No heavy model dependency or architecture beyond the accepted bounded v0 MLP.
- No change to `ObjectState` ABI, hard/soft assignment truth ownership or
  Reality Gate thresholds.
- No claim that a supervised public sample validates temporal identity,
  prediction or intervention.
- No publication of research-only assets or checkpoints without Owner-approved
  license and size review.
