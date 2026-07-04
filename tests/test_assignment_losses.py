from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.assignment_losses import (
    ASSIGNMENT_LOSS_V2_SCHEMA,
    assignment_balance_loss_and_gradient,
    assignment_cluster_loss_and_gradient,
    assignment_entropy_loss_and_gradient,
    assignment_loss_v2_breakdown,
    supervised_assignment_loss_and_gradient,
    validate_assignment_loss_v2_summary,
)


def test_supervised_assignment_loss_matches_cross_entropy_contract():
    assignment = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    loss, gradients = supervised_assignment_loss_and_gradient([assignment], [target])

    assert loss == pytest.approx(-(np.log(0.8) + np.log(0.7)) / 2.0)
    np.testing.assert_allclose(
        gradients[0],
        np.array(
            [
                [-1.0 / 0.8 / 2.0, 0.0],
                [0.0, -1.0 / 0.7 / 2.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_assignment_entropy_and_balance_are_independently_computable():
    assignment = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=np.float32,
    )

    entropy_loss, entropy_gradients = assignment_entropy_loss_and_gradient([assignment])
    balance_loss, balance_gradients = assignment_balance_loss_and_gradient([assignment])

    expected_entropy = float(
        np.mean(-np.sum(assignment * np.log(assignment), axis=1) / np.log(2.0))
    )
    assert entropy_loss == pytest.approx(expected_entropy)
    assert entropy_gradients[0].shape == assignment.shape
    assert balance_loss == pytest.approx(0.0025)
    np.testing.assert_allclose(
        balance_gradients[0],
        np.array(
            [
                [0.025, -0.025],
                [0.025, -0.025],
            ],
            dtype=np.float32,
        ),
        atol=1e-6,
    )


def test_assignment_cluster_loss_uses_explicit_cost_matrix():
    assignment = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=np.float32,
    )
    costs = np.array(
        [
            [0.1, 1.0],
            [0.9, 0.2],
        ],
        dtype=np.float32,
    )

    loss, gradients = assignment_cluster_loss_and_gradient([assignment], [costs])

    assert loss == pytest.approx(0.345)
    np.testing.assert_allclose(gradients[0], costs / 2.0, atol=1e-6)


def test_assignment_loss_v2_breakdown_reports_disabled_temporal_and_matching():
    assignment = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=np.float32,
    )
    costs = np.array(
        [
            [0.1, 1.0],
            [0.9, 0.2],
        ],
        dtype=np.float32,
    )
    target = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    result = assignment_loss_v2_breakdown(
        [assignment],
        cluster_costs=[costs],
        target_assignments=[target],
        cluster_weight=2.0,
        entropy_weight=0.5,
        balance_weight=1.0,
        supervised_weight=1.0,
    )
    payload = result.as_dict()

    assert result.schema == ASSIGNMENT_LOSS_V2_SCHEMA
    assert payload["loss_family"] == "assignment_object_loss_v2"
    assert "cluster" in payload["enabled_terms"]
    assert "entropy" in payload["enabled_terms"]
    assert "balance" in payload["enabled_terms"]
    assert "supervised" in payload["enabled_terms"]
    assert "temporal" in payload["disabled_terms"]
    assert "matching" in payload["disabled_terms"]
    assert payload["gradient_shapes"] == [[2, 2]]
    assert payload["gradient_l2"][0] > 0.0
    assert validate_assignment_loss_v2_summary(payload) is True


def test_assignment_cluster_loss_requires_cost_shape_match():
    assignment = np.full((2, 2), 0.5, dtype=np.float32)
    costs = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="shape must match assignment"):
        assignment_cluster_loss_and_gradient([assignment], [costs])
