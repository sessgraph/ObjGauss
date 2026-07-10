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
    supervised_assignment_loss_and_logit_gradient,
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


def test_supervised_assignment_clip_plateau_has_zero_finite_gradient():
    assignment = np.array([[0.0, 1.0]], dtype=np.float32)
    target = np.array([[1.0, 0.0]], dtype=np.float32)

    loss, gradients = supervised_assignment_loss_and_gradient([assignment], [target])

    assert loss == pytest.approx(-np.log(1e-8))
    assert np.isfinite(gradients[0]).all()
    np.testing.assert_array_equal(gradients[0], np.zeros_like(assignment))


@pytest.mark.parametrize(
    ("target_probability", "expected_logit_gradient"),
    [
        (0.5e-8, np.array([[0.0, 0.0]], dtype=np.float32)),
        (2.0e-8, np.array([[-1.0, 1.0]], dtype=np.float32)),
    ],
)
def test_supervised_assignment_logit_gradient_is_bounded_across_clip_boundary(
    target_probability,
    expected_logit_gradient,
):
    assignment = np.array(
        [[target_probability, 1.0 - target_probability]],
        dtype=np.float32,
    )
    target = np.array([[1.0, 0.0]], dtype=np.float32)

    _loss, assignment_gradients = supervised_assignment_loss_and_gradient(
        [assignment],
        [target],
    )
    _loss, logit_gradients = supervised_assignment_loss_and_logit_gradient(
        [assignment],
        [target],
    )

    if target_probability > 1e-8:
        # dL/dA must retain its exact 1 / p magnitude.  The bounded training
        # quantity is the analytically fused softmax-logit VJP below.
        assert assignment_gradients[0][0, 0] == pytest.approx(
            -1.0 / target_probability,
            rel=1e-6,
        )
    else:
        np.testing.assert_array_equal(
            assignment_gradients[0],
            np.zeros_like(assignment),
        )
    assert np.max(np.abs(logit_gradients[0])) <= 1.0
    np.testing.assert_allclose(
        logit_gradients[0],
        expected_logit_gradient,
        atol=1e-6,
    )


@pytest.mark.parametrize("target_probability", [0.5e-8, 2.0e-8])
def test_supervised_assignment_logit_gradient_matches_finite_difference(
    target_probability,
):
    logits = np.array(
        [np.log(target_probability / (1.0 - target_probability)), 0.0],
        dtype=np.float64,
    )
    target = np.array([[1.0, 0.0]], dtype=np.float32)
    assignment = _softmax_row(logits)

    _loss, gradients = supervised_assignment_loss_and_logit_gradient(
        [assignment],
        [target],
    )
    finite_difference = _finite_difference_supervised_loss(logits, target)

    np.testing.assert_allclose(
        gradients[0][0],
        finite_difference,
        atol=2e-3,
    )


def test_supervised_assignment_averages_only_frames_with_targets():
    supervised = np.array([[0.8, 0.2]], dtype=np.float32)
    unsupervised = np.array([[0.4, 0.6]], dtype=np.float32)
    target = np.array([[1.0, 0.0]], dtype=np.float32)

    loss, gradients = supervised_assignment_loss_and_gradient(
        [supervised, unsupervised],
        [target, None],
    )

    assert loss == pytest.approx(-np.log(0.8))
    np.testing.assert_allclose(
        gradients[0],
        np.array([[-1.0 / 0.8, 0.0]], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_array_equal(gradients[1], np.zeros_like(unsupervised))


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
    expected_logit_gradient = assignment * (
        result.gradients[0]
        - np.sum(result.gradients[0] * assignment, axis=1, keepdims=True)
    )
    np.testing.assert_allclose(
        result.softmax_logit_gradients[0],
        expected_logit_gradient,
        atol=1e-6,
    )
    assert validate_assignment_loss_v2_summary(payload) is True


def test_assignment_loss_v2_exposes_bounded_softmax_logit_gradient():
    assignment = np.array([[2.0e-8, 1.0 - 2.0e-8]], dtype=np.float32)
    target = np.array([[1.0, 0.0]], dtype=np.float32)

    result = assignment_loss_v2_breakdown(
        [assignment],
        target_assignments=[target],
        supervised_weight=1.0,
    )

    assert result.gradients[0][0, 0] == pytest.approx(-5.0e7, rel=1e-6)
    assert np.max(np.abs(result.softmax_logit_gradients[0])) <= 1.0
    np.testing.assert_allclose(
        result.softmax_logit_gradients[0],
        np.array([[-1.0, 1.0]], dtype=np.float32),
        atol=1e-6,
    )


def test_assignment_cluster_loss_requires_cost_shape_match():
    assignment = np.full((2, 2), 0.5, dtype=np.float32)
    costs = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="shape must match assignment"):
        assignment_cluster_loss_and_gradient([assignment], [costs])


def _softmax_row(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
    return probabilities[None, :].astype(np.float32)


def _finite_difference_supervised_loss(
    logits: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float = 1e-3,
) -> np.ndarray:
    gradient = np.zeros_like(logits)
    for index in range(logits.size):
        plus = logits.copy()
        minus = logits.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss, _ = supervised_assignment_loss_and_logit_gradient(
            [_softmax_row(plus)],
            [target],
        )
        minus_loss, _ = supervised_assignment_loss_and_logit_gradient(
            [_softmax_row(minus)],
            [target],
        )
        gradient[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
    return gradient
