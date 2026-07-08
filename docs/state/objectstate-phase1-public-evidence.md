# ObjectState Phase 1 Public Evidence

> 状态: current
> 最近更新: 2026-07-08

本文件记录 ObjGauss Phase 1 controlled real / public route 的本地证据行。
大型 public dataset zip、RGB-D frames、PLY evidence seed、candidate artifacts 和
handoff outputs 留在 ignored `outputs/`，不提交进 git。

## BOP LMO Public RGB-D Baseline Row

- Evidence id: `OBJECTSTATE-BOP-LMO-PUBLIC-ROW-001`
- Source dataset: BOP Benchmark `bop-benchmark/lmo`
- Source URL:
  `https://huggingface.co/datasets/bop-benchmark/lmo`
- License: `cc-by-sa-4.0`
- Downloaded file:
  `outputs/assets/raw/bop-lmo/lmo_test_bop19.zip`
- Downloaded file size: `117550985` bytes
- SHA256:
  `42d7a15f317476ca3980ee7ec0344b691cbadc796835f0b14f72c89a1dcec421`
- Extracted local scene:
  `outputs/assets/raw/bop-lmo/lmo-test-bop19-subset/test/000002`
- Selected frames: `000003`, `000008`, `000017`
- Local evidence output:
  `outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/`

Command:

```bash
uv run objgauss object-state bop-rgbd-baseline-local-row-handoff \
  outputs/assets/raw/bop-lmo/lmo-test-bop19-subset/test/000002 \
  --output-root outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline \
  --sample-id bop-lmo-test-scene-000002-rgbd-baseline \
  --dataset-id bop-lmo \
  --object-category lmo_objects \
  --max-frames 3 \
  --max-points-per-frame 10000 \
  --overwrite-gaussian-evidence \
  --ply-format binary_little_endian \
  --summary-output outputs/evidence/objectstate-bop-lmo-public-000002-rgbd-baseline/bop-rgbd-baseline-local-row-summary.json \
  --force
```

Result:

- Summary schema:
  `objgauss-objectstate-bop-rgbd-baseline-local-row-handoff-v1`
- Summary status:
  `objectstate_bop_rgbd_baseline_local_row_handoff_incomplete`
- RGB-D evidence:
  `selected_frames=3`, `exported_frames=3`, `missing_depth_files=0`,
  `rgbd_total_vertices=30000`
- Baseline ObjectState artifact:
  `baseline_frames=3`, `baseline_states=3`,
  `baseline_total_gaussians=30000`
- Identity predictions: `24`
- Prediction candidates: `16`
- Prediction evidence package:
  `objectstate_controlled_prediction_evidence_package_reviewable`
- Phase 1 ledger maturity: `prediction_reviewable`
- Identity evidence package:
  `objectstate_controlled_identity_evidence_package_incomplete`

Metric results:

- Prediction eval:
  `objectstate_controlled_prediction_eval_fail`
- `state_ade=0.29266938346336563`
- `history_ade=0.19507330249123483`
- `prediction_gap_vs_history_model=0.0975960809721308`
- `error_ratio_vs_history_model=1.500304653305985`
- `prediction_count=16`
- Identity eval:
  `objectstate_controlled_identity_eval_fail`
- `idf1=1.0`
- `track_retrieval_recall_at_1=0.125`
- `fragmentation_rate=0.0`
- `swap_rate=0.0`
- `identity_collapse=true`
- `reconstruction_noise_evidence_present=false`

Interpretation:

This row is real public RGB-D + BOP 6D pose evidence, but it is not a pass row.
It is useful negative evidence for the current single-state Gaussian centroid
baseline:

- Prediction is reviewable and fails metric gates.
- Identity evaluation exposes identity collapse, low retrieval and missing
  reconstruction-noise robustness evidence.
- Identity package is not reviewable as a controlled identity scenario because
  the extracted BOP route does not provide explicit lighting-condition or camera
  pose condition metadata.
- No intervention / action evidence is claimed.

The row must not be used to claim ObjectState is a real-world state variable.
It only proves that the BOP public RGB-D route can produce a reviewable
prediction row and a clear identity negative result from actual public data.
