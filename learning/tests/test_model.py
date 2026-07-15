from __future__ import annotations

import math
import unittest

import torch

from objgauss_learning.model import (
    ROLLOUT_BOUNDARIES,
    MinimalObjectGNN,
    ModelInvalidError,
    canonicalize_quaternion_tensor,
    parameter_count,
    symmetry_quaternion_distance,
)


def state_batch() -> torch.Tensor:
    state = torch.zeros((1, 2, 13), dtype=torch.float32)
    state[0, 0, 0:3] = torch.tensor([0.0, 0.0, 0.02])
    state[0, 1, 0:3] = torch.tensor([0.4, 0.2, 0.02])
    state[..., 3] = 1.0
    return state


def target_mask() -> torch.Tensor:
    value = torch.zeros((1, 2, 1), dtype=torch.float32)
    value[0, 0, 0] = 1.0
    return value


class MinimalObjectGNNTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_learned_arms_have_identical_parameter_names_shapes_and_counts(self) -> None:
        free = MinimalObjectGNN(hidden_width=64, arm="action_free")
        conditioned = MinimalObjectGNN(hidden_width=64, arm="action_conditioned")
        self.assertEqual(parameter_count(free), parameter_count(conditioned))
        self.assertEqual(
            [(name, tuple(value.shape)) for name, value in free.state_dict().items()],
            [(name, tuple(value.shape)) for name, value in conditioned.state_dict().items()],
        )
        self.assertIn("action_free_mask_token", dict(free.named_parameters()))
        self.assertIn("action_free_mask_token", dict(conditioned.named_parameters()))
        for model in (free, conditioned):
            model.rollout(
                state_batch(),
                commanded_action=torch.tensor([[0.35, 0.0, 0.0, 0.1, 1.0]]),
                target_mask=target_mask(),
            ).sum().backward()
            gradient = model.action_free_mask_token.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())

    def test_action_free_masks_command_while_conditioned_injects_target_only(self) -> None:
        first_action = torch.tensor([[0.35, 0.0, 0.0, 0.1, 1.0]])
        second_action = torch.tensor([[-0.7, 0.0, 0.0, 0.1, 1.0]])
        free = MinimalObjectGNN(hidden_width=64, arm="action_free")
        conditioned = MinimalObjectGNN(hidden_width=64, arm="action_conditioned")
        conditioned.load_state_dict(free.state_dict())
        free_first = free.rollout(
            state_batch(), commanded_action=first_action, target_mask=target_mask()
        )
        free_second = free.rollout(
            state_batch(), commanded_action=second_action, target_mask=target_mask()
        )
        conditioned_first = conditioned.rollout(
            state_batch(), commanded_action=first_action, target_mask=target_mask()
        )
        conditioned_second = conditioned.rollout(
            state_batch(), commanded_action=second_action, target_mask=target_mask()
        )
        self.assertTrue(torch.equal(free_first, free_second))
        self.assertFalse(torch.equal(conditioned_first[:, :, 0], conditioned_second[:, :, 0]))
        self.assertTrue(torch.equal(conditioned_first[:, 0, 1], conditioned_second[:, 0, 1]))

    def test_rollout_reuses_one_transition_for_irregular_physical_intervals(self) -> None:
        model = MinimalObjectGNN(hidden_width=64, arm="action_conditioned")
        action_inputs = []

        def capture(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            action_inputs.append(args[0].detach().clone())

        handle = model.action_encoder[0].register_forward_pre_hook(capture)
        try:
            rollout = model.rollout(
                state_batch(),
                commanded_action=torch.tensor([[0.35, 0.0, 0.0, 0.1, 1.0]]),
                target_mask=target_mask(),
            )
        finally:
            handle.remove()
        self.assertEqual(rollout.shape, (1, 4, 2, 13))
        self.assertEqual(len(action_inputs), 4)
        self.assertEqual(action_inputs[0][0, 3:6].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(
            action_inputs[0][0, 6:9].tolist(),
            [0.10000000149011612, 1.0, 1.0],
        )
        for item in action_inputs[1:]:
            self.assertEqual(item[0].tolist(), [0.0] * 9)
        norms = torch.linalg.vector_norm(rollout[..., 3:7], dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-6))

    def test_rollout_rejects_extra_or_step_index_boundaries(self) -> None:
        model = MinimalObjectGNN(hidden_width=64, arm="action_free")
        with self.assertRaisesRegex(ModelInvalidError, "boundaries"):
            model.rollout(
                state_batch(),
                commanded_action=torch.zeros((1, 5)),
                target_mask=target_mask(),
                boundaries_s=(0.0, 1.0, 2.0, 3.0, 4.0),
            )
        self.assertEqual(ROLLOUT_BOUNDARIES, (0.0, 0.1, 0.2, 0.5, 1.1))

    def test_quaternion_normalization_and_finite_symmetry_loss(self) -> None:
        normalized = canonicalize_quaternion_tensor(
            torch.tensor([-2.0, 0.0, 0.0, 0.0])
        )
        self.assertTrue(torch.equal(normalized, torch.tensor([1.0, 0.0, 0.0, 0.0])))
        quarter_turn = math.sqrt(0.5)
        prediction = torch.tensor([quarter_turn, 0.0, 0.0, quarter_turn])
        target = torch.tensor([quarter_turn, 0.0, 0.0, -quarter_turn])
        symmetries = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        distance = symmetry_quaternion_distance(prediction, target, symmetries)
        self.assertTrue(torch.isfinite(distance))
        self.assertAlmostEqual(float(distance), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
