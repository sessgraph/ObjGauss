"""Minimal Object GNN and frozen four-interval PR-02C rollout semantics."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn


STATE_WIDTH = 13
ACTION_FEATURE_WIDTH = 9
ROLLOUT_BOUNDARIES = (0.0, 0.1, 0.2, 0.5, 1.1)
ROLLOUT_TIMES = ROLLOUT_BOUNDARIES[1:]
ARMS = ("action_free", "action_conditioned")


class ModelInvalidError(RuntimeError):
    """A model input, state transition, or frozen architecture is invalid."""


def canonicalize_quaternion_tensor(quaternion: Tensor) -> Tensor:
    if quaternion.shape[-1] != 4:
        raise ModelInvalidError("quaternion tensor must end with four wxyz values")
    if not torch.isfinite(quaternion).all():
        raise ModelInvalidError("quaternion tensor contains non-finite values")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= 1e-12):
        raise ModelInvalidError("quaternion tensor contains a zero-norm value")
    normalized = quaternion / norm
    nonzero = normalized.abs() > 1e-12
    first_index = nonzero.to(torch.int64).argmax(dim=-1, keepdim=True)
    first = normalized.gather(-1, first_index)
    has_nonzero = nonzero.any(dim=-1, keepdim=True)
    sign = torch.where(has_nonzero & (first < 0), -torch.ones_like(first), torch.ones_like(first))
    return normalized * sign


def quaternion_multiply_tensor(left: Tensor, right: Tensor) -> Tensor:
    if left.shape[-1] != 4 or right.shape[-1] != 4:
        raise ModelInvalidError("quaternion multiplication requires wxyz tensors")
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quaternion_inverse_tensor(quaternion: Tensor) -> Tensor:
    normalized = canonicalize_quaternion_tensor(quaternion)
    return torch.cat((normalized[..., :1], -normalized[..., 1:]), dim=-1)


def symmetry_quaternion_distance(
    prediction: Tensor, target: Tensor, symmetries_wxyz: Tensor
) -> Tensor:
    """Return the pilot-compatible minimum geodesic angle over explicit symmetries."""

    if prediction.shape[-1] != 4 or target.shape[-1] != 4:
        raise ModelInvalidError("orientation distance requires wxyz quaternions")
    if symmetries_wxyz.ndim != 2 or symmetries_wxyz.shape[-1] != 4:
        raise ModelInvalidError("explicit symmetry rotations must have shape [K, 4]")
    if symmetries_wxyz.shape[0] == 0:
        raise ModelInvalidError("orientation loss requires at least one explicit symmetry")
    prediction = canonicalize_quaternion_tensor(prediction)
    target = canonicalize_quaternion_tensor(target)
    symmetries = canonicalize_quaternion_tensor(symmetries_wxyz)
    equivalent = quaternion_multiply_tensor(target.unsqueeze(-2), symmetries)
    dot = torch.sum(prediction.unsqueeze(-2) * equivalent, dim=-1).abs()
    safe_dot = torch.clamp(dot, 0.0, 1.0 - 1e-7)
    angles = 2.0 * torch.acos(safe_dot)
    angles = torch.where(dot >= 1.0 - 1e-7, torch.zeros_like(angles), angles)
    return angles.min(dim=-1).values


def state_tensor(value: dict[str, Any], *, device: torch.device | str = "cpu") -> Tensor:
    expected = {
        "position_W_m",
        "quaternion_WO_wxyz",
        "linear_velocity_W_m_s",
        "angular_velocity_W_rad_s",
    }
    if set(value) - {"object_id"} != expected:
        raise ModelInvalidError("ObjectState field set drift")
    payload = [
        *value["position_W_m"],
        *value["quaternion_WO_wxyz"],
        *value["linear_velocity_W_m_s"],
        *value["angular_velocity_W_rad_s"],
    ]
    tensor = torch.tensor(payload, dtype=torch.float32, device=device)
    if tensor.shape != (STATE_WIDTH,) or not torch.isfinite(tensor).all():
        raise ModelInvalidError("ObjectState must contain 13 finite scalar values")
    tensor[3:7] = canonicalize_quaternion_tensor(tensor[3:7])
    return tensor


def state_payload(object_id: str, value: Tensor) -> dict[str, Any]:
    if value.shape != (STATE_WIDTH,) or not torch.isfinite(value).all():
        raise ModelInvalidError("predicted ObjectState is non-finite or has the wrong shape")
    quaternion = canonicalize_quaternion_tensor(value[3:7]).detach().cpu().tolist()
    vector = value.detach().cpu().tolist()
    return {
        "object_id": object_id,
        "position_W_m": vector[0:3],
        "quaternion_WO_wxyz": quaternion,
        "linear_velocity_W_m_s": vector[7:10],
        "angular_velocity_W_rad_s": vector[10:13],
    }


class MinimalObjectGNN(nn.Module):
    """One-round, mean-aggregated Object GNN shared by both learned arms."""

    def __init__(self, *, hidden_width: int, arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ModelInvalidError(f"unknown learned arm: {arm}")
        if hidden_width not in {64, 128}:
            raise ModelInvalidError("hidden width is outside the frozen grid")
        self.hidden_width = hidden_width
        self.arm = arm
        self.object_encoder = nn.Sequential(
            nn.Linear(STATE_WIDTH, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.pairwise_message = nn.Sequential(
            nn.Linear(hidden_width * 2 + 3, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_FEATURE_WIDTH, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.action_free_mask_token = nn.Parameter(torch.zeros(ACTION_FEATURE_WIDTH))
        self.shared_residual_head = nn.Sequential(
            nn.Linear(hidden_width * 3 + 1, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, STATE_WIDTH),
        )

    def transition(
        self,
        state: Tensor,
        *,
        interval_action: Tensor,
        target_mask: Tensor,
        delta_t: Tensor,
    ) -> Tensor:
        if state.ndim != 3 or state.shape[-1] != STATE_WIDTH:
            raise ModelInvalidError("state batch must have shape [B, N, 13]")
        batch, objects, _ = state.shape
        if objects < 1:
            raise ModelInvalidError("Object GNN requires at least one object")
        if interval_action.shape != (batch, ACTION_FEATURE_WIDTH):
            raise ModelInvalidError("interval action must have shape [B, 9]")
        if target_mask.shape != (batch, objects, 1):
            raise ModelInvalidError("target mask must have shape [B, N, 1]")
        if delta_t.shape != (batch, 1):
            raise ModelInvalidError("delta_t must have shape [B, 1]")
        if not all(
            torch.isfinite(value).all()
            for value in (state, interval_action, target_mask, delta_t)
        ):
            raise ModelInvalidError("transition input contains non-finite values")
        encoded = self.object_encoder(state)
        sender = encoded.unsqueeze(1).expand(batch, objects, objects, self.hidden_width)
        receiver = encoded.unsqueeze(2).expand(batch, objects, objects, self.hidden_width)
        positions = state[..., 0:3]
        relative = positions.unsqueeze(1) - positions.unsqueeze(2)
        messages = self.pairwise_message(torch.cat((sender, receiver, relative), dim=-1))
        if objects == 1:
            aggregate = torch.zeros_like(encoded)
        else:
            off_diagonal = (~torch.eye(objects, dtype=torch.bool, device=state.device)).view(
                1, objects, objects, 1
            )
            aggregate = (messages * off_diagonal).sum(dim=2) / float(objects - 1)
        action_input = self.action_free_mask_token.view(1, -1).expand(batch, -1)
        if self.arm == "action_conditioned":
            action_input = action_input + interval_action
        action_embedding = self.action_encoder(action_input).unsqueeze(1) * target_mask
        dt_feature = delta_t.unsqueeze(1).expand(batch, objects, 1)
        residual = self.shared_residual_head(
            torch.cat((encoded, aggregate, action_embedding, dt_feature), dim=-1)
        )
        next_state = state + residual * dt_feature
        next_state = torch.cat(
            (
                next_state[..., 0:3],
                canonicalize_quaternion_tensor(next_state[..., 3:7]),
                next_state[..., 7:13],
            ),
            dim=-1,
        )
        if not torch.isfinite(next_state).all():
            raise ModelInvalidError("transition produced a non-finite ObjectState")
        return next_state

    def rollout(
        self,
        initial_state: Tensor,
        *,
        commanded_action: Tensor,
        target_mask: Tensor,
        boundaries_s: tuple[float, ...] = ROLLOUT_BOUNDARIES,
    ) -> Tensor:
        if commanded_action.ndim != 2 or commanded_action.shape[-1] != 5:
            raise ModelInvalidError(
                "commanded action batch must contain vector, duration, and push flag"
            )
        if tuple(boundaries_s) != ROLLOUT_BOUNDARIES:
            raise ModelInvalidError("rollout boundaries differ from the four frozen intervals")
        state = initial_state
        outputs = []
        for start, finish in zip(boundaries_s[:-1], boundaries_s[1:], strict=True):
            delta = finish - start
            remaining = torch.clamp(commanded_action[:, 3] - start, min=0.0)
            active_duration = torch.clamp(remaining, max=delta)
            active_fraction = active_duration / delta
            application_point_object = torch.zeros_like(commanded_action[:, 0:3])
            active_push = commanded_action[:, 4:5] * (active_duration > 0).to(
                commanded_action.dtype
            ).unsqueeze(-1)
            interval_action = torch.cat(
                (
                    commanded_action[:, 0:3] * active_fraction.unsqueeze(-1),
                    application_point_object,
                    active_duration.unsqueeze(-1),
                    active_fraction.unsqueeze(-1),
                    active_push,
                ),
                dim=-1,
            )
            delta_t = torch.full(
                (initial_state.shape[0], 1),
                delta,
                dtype=initial_state.dtype,
                device=initial_state.device,
            )
            state = self.transition(
                state,
                interval_action=interval_action,
                target_mask=target_mask,
                delta_t=delta_t,
            )
            outputs.append(state)
        return torch.stack(outputs, dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def architecture_document(hidden_width: int) -> dict[str, Any]:
    return {
        "model_family": "minimal-object-gnn",
        "state_width": STATE_WIDTH,
        "action_feature_width": ACTION_FEATURE_WIDTH,
        "hidden_width": hidden_width,
        "object_encoder_layers": 2,
        "pairwise_message_layers": 2,
        "message_passing_rounds": 1,
        "aggregation": "mean",
        "shared_residual_head_layers": 2,
        "activation": "silu",
        "action_injection": "target-object-only",
        "action_application_point": "target-object-center-of-mass-zero-in-object-frame",
        "action_free_mask_token": "learned",
        "action_conditioned_input": "learned-mask-token-plus-interval-action",
        "rollout_boundaries_s": list(ROLLOUT_BOUNDARIES),
        "residual_scaled_by_delta_t": True,
    }


def assert_frozen_delta_t() -> None:
    observed = tuple(
        finish - start
        for start, finish in zip(ROLLOUT_BOUNDARIES[:-1], ROLLOUT_BOUNDARIES[1:], strict=True)
    )
    expected = (0.1, 0.1, 0.3, 0.6)
    if any(not math.isclose(left, right) for left, right in zip(observed, expected, strict=True)):
        raise ModelInvalidError("physical delta_t sequence drift")
