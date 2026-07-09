# ObjectState Identity Benchmark Report

- Schema: `objgauss-objectstate-model-identity-benchmark-report-v1`
- Sample: `objectstate-model-identity-benchmark-report-001`
- Status: `objectstate_model_identity_benchmark_report_candidate_ready`
- Evidence policy: `semantic`
- Scenarios: `15`
- Identity pairs: `60`

## Overall Ranking

| Rank | Baseline | Retrieval@1 | Margin | Drift | Occlusion | Slot Swap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `oracle_target_assignment` | 1.000000 | 1.046054 | 1.200000 | 1.000000 | 0.000000 |
| 2 | `assignment_solver_v2` | 1.000000 | 1.046047 | 1.200000 | 1.000000 | 0.000000 |
| 3 | `random_assignment` | 0.300000 | -0.912254 | 3.418822 | 0.300000 | 0.766667 |
| 4 | `xyz_centroid` | 0.250000 | -0.128240 | 2.141258 | 0.250000 | 0.233333 |

## Perturbation Breakdown

| Perturbation | Scenarios | Solver Retrieval@1 | XYZ Retrieval@1 | Solver - XYZ | Solver Occlusion |
| --- | ---: | ---: | ---: | ---: | ---: |
| `viewpoint` | 3 | 1.000000 | 0.250000 | 0.750000 | 1.000000 |
| `dropout` | 3 | 1.000000 | 0.250000 | 0.750000 | 1.000000 |
| `occlusion` | 3 | 1.000000 | 0.250000 | 0.750000 | 1.000000 |
| `appearance` | 3 | 1.000000 | 0.250000 | 0.750000 | 1.000000 |
| `spatial` | 3 | 1.000000 | 0.250000 | 0.750000 | 1.000000 |

## Difficulty Ladder

| Difficulty | Scenarios | Solver Retrieval@1 | Solver Margin | Solver Drift |
| --- | ---: | ---: | ---: | ---: |
| `easy` | 5 | 1.000000 | 1.221260 | 0.400000 |
| `medium` | 5 | 1.000000 | 0.986947 | 1.200000 |
| `hard` | 5 | 1.000000 | 0.929932 | 2.000000 |

## Long Training Gate

- Decision: `candidate_ready`
- Scoped to evidence policy: `semantic`
- Scope rule: `candidate_ready` is policy-scoped, not a global native Gaussian gate.
- Reasons: none

This only gates a longer identity robustness smoke under the stated evidence policy. It does not unlock world-model training.

## Interpretation Boundary

This report is deterministic controlled synthetic evidence. It does not use real controlled capture,
does not add temporal loss, and does not claim a real-data identity pass. Native Gaussian long training
must be justified by a separate policy-specific gate or ablation result.
