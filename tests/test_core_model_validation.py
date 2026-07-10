from __future__ import annotations

import numpy as np

from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
    train_assignment_solver_v2,
)
from objgauss.core.assignment_solver_v2 import assignment_solver_v2_checkpoint
from objgauss.evaluation.assignment_solver_v2_eval import (
    evaluate_assignment_solver_v2_stability,
)
from objgauss.pipelines.assignment_v2_renderer_validation import (
    evaluate_assignment_v2_renderer_joint,
)
from objgauss.pipelines.core_model_validation import (
    CORE_MODEL_TRAIN_VALIDATE_SCHEMA,
    CoreModelTrainValidateReport,
    core_model_train_validate_report,
    validate_core_model_train_validate_summary,
)
from objgauss.pipelines.renderer_loss import renderer_loss_boundary_report
from objgauss.pipelines.trainable_kernel import (
    TrainableKernelFrame,
    bind_image_targets_to_frames,
)
from objgauss.datasets.v2_stability_foundation import (
    ObservationModelConfig,
    make_synthetic_stability_scenario_fixture,
)


def test_core_model_train_validate_report_aggregates_training_eval_and_renderer_evidence():
    fixtures = _two_object_stability_suite()
    batches = tuple(observation.evidence for fixture in fixtures for observation in fixture.observations)
    result = train_assignment_solver_v2(
        batches,
        initial_state=_swapped_two_slot_state(feature_dim=6),
        iterations=120,
        learning_rate=0.4,
        cluster_weight=0.0,
        entropy_weight=0.0,
        balance_weight=0.0,
        supervised_weight=1.0,
    )
    stability = evaluate_assignment_solver_v2_stability(result, fixtures).as_dict()
    checkpoint = assignment_solver_v2_checkpoint(
        result,
        source="fixture://core-model-validation",
    )
    renderer = evaluate_assignment_v2_renderer_joint(
        _renderer_frames_from_fixture(fixtures[0]),
        checkpoint,
    ).as_dict()
    boundary = renderer_loss_boundary_report(renderer).as_dict()

    report = core_model_train_validate_report(
        assignment_training=result.as_dict(),
        stability_eval=stability,
        renderer_joint=renderer,
        renderer_boundary=boundary,
    )
    summary = report.as_dict()

    assert isinstance(report, CoreModelTrainValidateReport)
    assert summary["schema"] == CORE_MODEL_TRAIN_VALIDATE_SCHEMA
    assert summary["status"] == "core_model_train_validate_pass"
    assert all(summary["gates"].values())
    assert summary["evidence"]["stability_eval"]["after_status"] == (
        "synthetic_stability_suite_gate_pass"
    )
    assert summary["evidence"]["renderer_boundary"]["status"] == (
        "assignment_v2_renderer_joint_validation_ready"
    )
    assert summary["small_sample_smoke"]["status"] == "pass"
    assert summary["small_sample_smoke"]["real_public_sample"] is False
    assert summary["small_sample_smoke"]["promotion_requires_real_sample_repeat"] is True
    assert summary["checkpoint_roundtrip"] == {
        "assignment_stability_eval_pass": True,
        "renderer_joint_pass": True,
    }
    assert summary["non_goals"] == {
        "starts_long_gpu_training": False,
        "uses_rollout_model": False,
        "uses_replay_buffer": False,
        "uses_diffusion_world_model": False,
        "mutates_dynamic_k": False,
        "unfreezes_gaussian_geometry": False,
    }
    assert validate_core_model_train_validate_summary(summary) is summary


def _two_object_stability_suite():
    return tuple(
        make_synthetic_stability_scenario_fixture(
            scenario_kind=scenario_kind,
            object_count=2,
            feature_dim=6,
            seed=500 + index,
            observation_config=ObservationModelConfig(
                points_per_object=2,
                position_jitter=0.0,
                seed=510 + index,
            ),
        )
        for index, scenario_kind in enumerate(
            ("cross_view", "occlusion_recovery", "perturbation", "adversarial_swap")
        )
    )


def _renderer_frames_from_fixture(fixture):
    slot_rgb = np.asarray(
        [
            [0.92, 0.22, 0.16],
            [0.10, 0.70, 0.86],
        ],
        dtype=np.float32,
    )
    frames = []
    for observation in fixture.observations:
        target_assignment = observation.evidence.target_assignment
        target_rgb = slot_rgb[np.argmax(target_assignment, axis=1)]
        frames.append(
            TrainableKernelFrame(
                positions=observation.evidence.positions,
                features=observation.evidence.features,
                target_rgb=target_rgb,
                target_assignment=target_assignment,
            )
        )
    return bind_image_targets_to_frames(tuple(frames), width=8, height=8)


def _swapped_two_slot_state(*, feature_dim: int) -> AssignmentSolverV2State:
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(slots=2, feature_dim=feature_dim, temperature=0.7),
        feature_centers=np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        position_centers=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        slot_bias=np.zeros(2, dtype=np.float32),
        source="swapped_core_model_validation_fixture",
    )
