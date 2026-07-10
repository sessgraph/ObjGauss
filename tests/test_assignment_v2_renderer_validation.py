from __future__ import annotations

import numpy as np

from objgauss.core.assignment_evidence import assignment_evidence_sequence_from_trainable_frames
from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
    train_assignment_solver_v2,
)
from objgauss.core.assignment_solver_v2 import assignment_solver_v2_checkpoint
from objgauss.pipelines.assignment_v2_renderer_validation import (
    ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA,
    AssignmentV2RendererJointValidationReport,
    evaluate_assignment_v2_renderer_joint,
    validate_assignment_v2_renderer_joint_summary,
)
from objgauss.pipelines.renderer_loss import renderer_loss_boundary_report
from objgauss.pipelines.trainable_kernel import (
    TrainableKernelFrame,
    bind_image_targets_to_frames,
    make_trainable_kernel_mvp_fixture,
)


def test_assignment_v2_renderer_joint_validation_binds_checkpoint_to_renderer_loss():
    frames = _targeted_trainable_frames()
    batches = assignment_evidence_sequence_from_trainable_frames(frames)
    result = train_assignment_solver_v2(
        batches,
        initial_state=_swapped_state_from_frames(frames),
        iterations=100,
        learning_rate=0.35,
        cluster_weight=0.0,
        entropy_weight=0.0,
        balance_weight=0.0,
        supervised_weight=1.0,
    )
    checkpoint = assignment_solver_v2_checkpoint(
        result,
        source="fixture://assignment-v2-renderer-joint",
    )

    report = evaluate_assignment_v2_renderer_joint(frames, checkpoint)
    summary = report.as_dict()

    assert isinstance(report, AssignmentV2RendererJointValidationReport)
    assert summary["schema"] == ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA
    assert summary["status"] == "assignment_v2_renderer_joint_validation_pass"
    assert summary["checkpoint_schema"] == "objgauss-assignment-solver-v2-checkpoint"
    assert summary["loss_decreased"] is True
    assert summary["image_render_loss_decreased"] is True
    assert summary["object_loss_decreased"] is True
    assert summary["object_state_eval"]["status"] == "objectstate_eval_pass"
    assert summary["checkpoint_roundtrip"]["pass"] is True
    assert summary["identity_gate"]["renderer_loss_does_not_override_identity_gate"] is True
    assert "means" in summary["frozen_fields"]
    assert "dynamic_k" in summary["frozen_fields"]
    assert summary["non_goals"] == {
        "trains_optimizer": False,
        "unfreezes_gaussian_geometry": False,
        "mutates_dynamic_k": False,
        "uses_rollout_model": False,
        "uses_replay_buffer": False,
    }
    assert validate_assignment_v2_renderer_joint_summary(summary) is summary

    boundary = renderer_loss_boundary_report(summary).as_dict()
    assert boundary["status"] == "assignment_v2_renderer_joint_validation_ready"
    assert boundary["evidence"]["kind"] == "assignment_v2_renderer_joint_validation"
    assert boundary["decoder_handoff_contract"]["status"] == (
        "assignment_v2_renderer_joint_validation_ready"
    )
    assert boundary["evidence"]["loss_decreased"] is True
    assert boundary["evidence"]["image_render_loss_decreased"] is True


def _targeted_trainable_frames() -> tuple[TrainableKernelFrame, ...]:
    target = np.zeros((6, 2), dtype=np.float32)
    target[:3, 0] = 1.0
    target[3:, 1] = 1.0
    frames = tuple(
        TrainableKernelFrame(
            positions=frame.positions,
            features=frame.features,
            target_rgb=frame.target_rgb,
            target_assignment=target,
        )
        for frame in make_trainable_kernel_mvp_fixture()
    )
    return bind_image_targets_to_frames(frames, width=8, height=8)


def _swapped_state_from_frames(frames: tuple[TrainableKernelFrame, ...]) -> AssignmentSolverV2State:
    features = np.concatenate([frame.features for frame in frames], axis=0)
    positions = np.concatenate([frame.positions for frame in frames], axis=0)
    targets = np.concatenate([frame.target_assignment for frame in frames], axis=0)
    feature_centers = []
    position_centers = []
    for slot in range(targets.shape[1]):
        weights = targets[:, slot]
        feature_centers.append(np.average(features, axis=0, weights=weights))
        position_centers.append(np.average(positions, axis=0, weights=weights))
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=2,
            feature_dim=features.shape[1],
            temperature=0.6,
        ),
        feature_centers=np.asarray(feature_centers[::-1], dtype=np.float32),
        position_centers=np.asarray(position_centers[::-1], dtype=np.float32),
        slot_bias=np.zeros(2, dtype=np.float32),
        source="swapped_assignment_v2_renderer_joint_fixture",
    )
