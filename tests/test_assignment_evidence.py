from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.assignment_evidence import (
    ASSIGNMENT_EVIDENCE_BATCH_SCHEMA,
    AssignmentEvidenceBatch,
    assignment_evidence_from_object_emergence,
    assignment_evidence_from_trainable_frame,
    assignment_evidence_sequence_from_trainable_frames,
    validate_assignment_evidence_batch,
    validate_assignment_evidence_summary,
)
from objgauss.core.object_emergence_solver import ObjectEmergenceEvidence
from objgauss.pipelines.trainable_kernel import make_trainable_kernel_mvp_fixture


def test_assignment_evidence_from_trainable_frame_preserves_core_fields():
    frames = make_trainable_kernel_mvp_fixture()

    evidence = assignment_evidence_from_trainable_frame(frames[0], frame_index=3)
    payload = evidence.as_dict()

    assert isinstance(evidence, AssignmentEvidenceBatch)
    assert evidence.schema == ASSIGNMENT_EVIDENCE_BATCH_SCHEMA
    assert payload["kind"] == "assignment_evidence_batch"
    assert payload["frame_index"] == 3
    assert payload["evidence_count"] == frames[0].positions.shape[0]
    assert payload["feature_dim"] == frames[0].features.shape[1]
    assert payload["has_target_assignment"] is (frames[0].target_assignment is not None)
    assert payload["has_mask_votes"] is False
    assert validate_assignment_evidence_summary(payload) is True


def test_assignment_evidence_accepts_optional_mask_votes_and_track_hints():
    frames = make_trainable_kernel_mvp_fixture()
    mask_votes = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    track_hints = np.array([10, 10, 10, 20, 20, 20], dtype=np.int64)

    evidence = assignment_evidence_from_trainable_frame(
        frames[0],
        mask_votes=mask_votes,
        track_hints=track_hints,
    )
    payload = evidence.as_dict()

    assert payload["has_mask_votes"] is True
    assert payload["mask_vote_dim"] == 2
    assert payload["has_track_hints"] is True
    np.testing.assert_array_equal(evidence.track_hints, track_hints)
    np.testing.assert_allclose(evidence.mask_votes, mask_votes, atol=1e-6)


def test_assignment_evidence_sequence_sets_frame_indices():
    frames = make_trainable_kernel_mvp_fixture()

    evidence = assignment_evidence_sequence_from_trainable_frames(frames)

    assert [batch.frame_index for batch in evidence] == [0, 1]
    assert all(batch.source == "trainable_kernel_frame" for batch in evidence)


def test_assignment_evidence_from_object_emergence_preserves_source():
    frames = make_trainable_kernel_mvp_fixture()
    source = ObjectEmergenceEvidence(
        positions=frames[0].positions,
        features=frames[0].features,
        target_assignment=frames[0].target_assignment,
        frame_index=5,
        source="fixture_solver_evidence",
    )

    evidence = assignment_evidence_from_object_emergence(source)

    assert evidence.frame_index == 5
    assert evidence.source == "object_emergence_evidence:fixture_solver_evidence"
    np.testing.assert_allclose(evidence.features, frames[0].features, atol=1e-6)


def test_assignment_evidence_validation_rejects_mismatched_mask_votes():
    frames = make_trainable_kernel_mvp_fixture()
    bad_mask_votes = np.ones((frames[0].positions.shape[0] + 1, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="mask_votes rows must match positions"):
        assignment_evidence_from_trainable_frame(frames[0], mask_votes=bad_mask_votes)


def test_assignment_evidence_validation_rejects_mismatched_features():
    batch = AssignmentEvidenceBatch(
        positions=np.zeros((2, 3), dtype=np.float32),
        features=np.zeros((3, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="features rows must match positions"):
        validate_assignment_evidence_batch(batch)
