from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RENDERER_LOSS_BOUNDARY_SCHEMA = "objgauss-renderer-loss-boundary-v1"
POINT_RENDER_SMOKE_RENDERER = "cpu-point-rgb-smoke"
TARGET_IMAGE_RENDERER = "differentiable-gaussian-image-renderer"
TRAINABLE_KERNEL_SCHEMA = "objgauss-v1-trainable-kernel-mvp-v1"
OBJECT_EMERGENCE_SOLVER_TRAINING_SCHEMA = "objgauss-object-emergence-solver-training-v1"
OBJECT_EMERGENCE_SOLVER_CHECKPOINT_SCHEMA = "objgauss-object-emergence-solver-checkpoint-v1"
OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA = "objgauss-object-state-gaussian-decoder-training-v1"
SOLVER_DECODER_JOINT_TRAINING_SCHEMA = "objgauss-solver-decoder-joint-training-v1"
SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA = "objgauss-solver-decoder-joint-checkpoint-v1"
ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA = (
    "objgauss-assignment-v2-render-joint-validation-v1"
)
FULL_3DGS_RENDERERS = {"gsplat-rasterization-v1"}
OBJECT_EMERGENCE_SOLVER_EVIDENCE = {
    "object_emergence_solver_training",
    "object_emergence_solver_checkpoint",
}
OBJECT_STATE_DECODER_TRAINING_EVIDENCE = {"object_state_gaussian_decoder_training"}
SOLVER_DECODER_JOINT_EVIDENCE = {
    "solver_decoder_joint_training",
    "solver_decoder_joint_checkpoint",
}
ASSIGNMENT_V2_RENDER_JOINT_EVIDENCE = {
    "assignment_v2_renderer_joint_validation",
}

__all__ = (
    "RENDERER_LOSS_BOUNDARY_SCHEMA",
    "RendererLossBoundaryReport",
    "renderer_loss_boundary_report",
    "validate_renderer_loss_boundary_summary",
)


@dataclass(frozen=True)
class RendererLossBoundaryReport:
    schema: str
    status: str
    current_renderer: str
    target_renderer: str
    point_smoke_ready: bool
    input_frame_contract: dict[str, Any]
    render_target_contract: dict[str, Any]
    loss_telemetry_contract: dict[str, Any]
    decoder_handoff_contract: dict[str, Any]
    integration_contract: dict[str, Any]
    evidence: dict[str, Any]
    point_smoke_blockers: tuple[str, ...]
    upgrade_blockers: tuple[str, ...]
    next_steps: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": self.status,
            "current_renderer": self.current_renderer,
            "target_renderer": self.target_renderer,
            "point_smoke_ready": bool(self.point_smoke_ready),
            "input_frame_contract": self.input_frame_contract,
            "render_target_contract": self.render_target_contract,
            "loss_telemetry_contract": self.loss_telemetry_contract,
            "decoder_handoff_contract": self.decoder_handoff_contract,
            "integration_contract": self.integration_contract,
            "evidence": self.evidence,
            "point_smoke_blockers": list(self.point_smoke_blockers),
            "upgrade_blockers": list(self.upgrade_blockers),
            "next_steps": list(self.next_steps),
        }


def renderer_loss_boundary_report(
    kernel_summary: dict[str, Any] | None = None,
    *,
    target_renderer: str = TARGET_IMAGE_RENDERER,
) -> RendererLossBoundaryReport:
    evidence, point_blockers = _kernel_summary_evidence(kernel_summary)
    point_ready = bool(
        kernel_summary is not None
        and evidence.get("kind") == "trainable_kernel_summary"
        and not point_blockers
    )
    solver_ready = bool(
        kernel_summary is not None
        and evidence.get("kind") in OBJECT_EMERGENCE_SOLVER_EVIDENCE
        and evidence.get("solver_loss_decreased")
        and evidence.get("assignment_loss_decreased")
    )
    decoder_training_ready = bool(
        kernel_summary is not None
        and evidence.get("kind") in OBJECT_STATE_DECODER_TRAINING_EVIDENCE
        and evidence.get("loss_decreased")
        and evidence.get("image_render_loss_decreased")
    )
    joint_training_ready = bool(
        kernel_summary is not None
        and evidence.get("kind") in SOLVER_DECODER_JOINT_EVIDENCE
        and evidence.get("loss_decreased")
        and evidence.get("image_render_loss_decreased")
    )
    assignment_v2_renderer_ready = bool(
        kernel_summary is not None
        and evidence.get("kind") in ASSIGNMENT_V2_RENDER_JOINT_EVIDENCE
        and evidence.get("loss_decreased")
        and evidence.get("image_render_loss_decreased")
    )
    status = "point_render_smoke_ready" if point_ready else "contract_defined"
    if kernel_summary is not None and point_blockers:
        status = "point_render_smoke_blocked"
    if solver_ready:
        status = "object_emergence_solver_ready"
    if decoder_training_ready:
        status = "object_state_decoder_training_ready"
    if joint_training_ready:
        status = "solver_decoder_joint_training_ready"
    if assignment_v2_renderer_ready:
        status = "assignment_v2_renderer_joint_validation_ready"
    if point_ready and evidence.get("renderer_api_ready"):
        status = "renderer_api_ready"
    full_renderer_ready = (
        bool(evidence.get("renderer_api_ready"))
        and evidence.get("renderer_name") in FULL_3DGS_RENDERERS
    )
    if point_ready and full_renderer_ready:
        status = "full_3dgs_renderer_ready"
    if decoder_training_ready and full_renderer_ready:
        status = "full_3dgs_decoder_training_ready"
    if joint_training_ready and full_renderer_ready:
        status = "full_3dgs_solver_decoder_joint_training_ready"
    if assignment_v2_renderer_ready and full_renderer_ready:
        status = "full_3dgs_assignment_v2_renderer_joint_validation_ready"
    upgrade_blockers = []
    if not evidence.get("image_targets_bound"):
        upgrade_blockers.append("image_space_targets_not_bound")
    if evidence.get("renderer_api_ready"):
        if not full_renderer_ready:
            upgrade_blockers.append("full_3dgs_renderer_not_selected")
    else:
        upgrade_blockers.extend(
            [
                "differentiable_gaussian_renderer_not_selected",
                "renderer_gradient_path_not_defined",
            ]
        )
    if not evidence.get("image_target_visibility_policies"):
        upgrade_blockers.append("camera_visibility_policy_not_bound")
    if evidence.get("kind") in OBJECT_EMERGENCE_SOLVER_EVIDENCE:
        upgrade_blockers.extend(
            [
                "solver_checkpoint_not_bound_to_gaussian_decoder",
                "solver_checkpoint_not_bound_to_renderer_loss",
            ]
        )
    next_steps = (
        (
            "run small full 3DGS trainable renderer MVP",
            "persist renderer_api evidence into trainable model artifact",
            "keep viewer renderer as debug consumer, not the training renderer default",
        )
        if full_renderer_ready
        else
        (
            "run resume/load smoke from solver + decoder joint checkpoint",
            "scale solver + decoder joint training beyond smoke size",
            "keep geometry/opacity/camera frozen until assignment and color training are stable",
        )
        if evidence.get("kind") == "solver_decoder_joint_checkpoint"
        else
        (
            "promote v2 assignment checkpoint into solver-decoder joint training",
            "run a controlled gsplat smoke when host GPU is available",
            "keep identity hard gate as the promotion blocker before renderer loss",
        )
        if evidence.get("kind") in ASSIGNMENT_V2_RENDER_JOINT_EVIDENCE
        else
        (
            "select or implement full 3DGS image renderer behind the training renderer API",
            "wire image_render_loss into the trainable optimization objective",
            "keep viewer renderer as debug consumer, not the training renderer default",
        )
        if evidence.get("renderer_api_ready")
        else (
            "export Object Emergence Solver checkpoint with weights",
            "bind solver A[N,K] output to ObjectState -> Gaussian decoder",
            "connect renderer_api image_render_loss to solver/full renderer training loop",
            "run torch/gsplat/CUDA preflight before GPU training",
        )
        if evidence.get("kind") in OBJECT_EMERGENCE_SOLVER_EVIDENCE
        else (
            "promote decoder color training smoke to full 3DGS renderer",
            "bind solver checkpoint assignment generation into the decoder training command",
            "extend decoder trainable fields only after color-only loss is stable",
        )
        if evidence.get("kind") in OBJECT_STATE_DECODER_TRAINING_EVIDENCE
        else (
            "scale solver + decoder joint training beyond smoke size",
            "keep geometry/opacity/camera frozen until assignment and color training are stable",
            "add checkpoint export only after joint loss evidence is repeatable",
        )
        if evidence.get("kind") in SOLVER_DECODER_JOINT_EVIDENCE
        else (
            "bind trainable frames to camera/image targets",
            "define image-space renderer API and telemetry",
            "decide whether GPU/torch/differentiable rasterizer requires ADR",
            "keep viewer renderer as debug consumer, not the training renderer default",
        )
    )
    return RendererLossBoundaryReport(
        schema=RENDERER_LOSS_BOUNDARY_SCHEMA,
        status=status,
        current_renderer=evidence.get("renderer_name") or POINT_RENDER_SMOKE_RENDERER,
        target_renderer=target_renderer,
        point_smoke_ready=point_ready,
        input_frame_contract={
            "current": {
                "positions": "float32[N,3]",
                "features": "float32[N,D]",
                "target_rgb": "float32[N,3]",
                "target_assignment": "optional float32[N,K]",
            },
            "renderer_loss_upgrade_requires": {
                "camera": "per-frame intrinsics/extrinsics",
                "image_target": "float32[H,W,3] or equivalent image-space target",
                "gaussian_parameters": "renderer-native GaussianToken fields, not only object colors",
                "visibility": "defined alpha/depth/mask policy for supervised pixels",
            },
        },
        render_target_contract={
            "current": {
                "kind": "point_rgb_rows",
                "shape": "N x 3",
                "loss": "mean squared RGB reconstruction at Gaussian/evidence rows",
            },
            "target": {
                "kind": "image_space_render",
                "shape": "H x W x 3",
                "loss": "photometric image reconstruction plus object and temporal terms",
            },
            "non_goal": "Do not claim point_rgb_rows is a full 3DGS differentiable renderer.",
        },
        loss_telemetry_contract={
            "required_terms": ["render_loss", "object_loss", "temporal_loss", "total_loss"],
            "required_deltas": ["initial_loss", "final_loss", "loss_decreased"],
            "upgrade_terms": [
                "image_render_loss",
                "visibility_policy",
                "renderer_name",
                "renderer_gradient_path",
            ],
        },
        decoder_handoff_contract=_decoder_handoff_contract(evidence, target_renderer=target_renderer),
        integration_contract={
            "viewer_renderer_role": "debug visualization and browser audit only",
            "training_renderer_role": "separate loss producer behind a stable contract",
            "replacement_policy": "training renderer must not replace Three.js / viewer renderer by default",
            "artifact_policy": "large training outputs stay outside git and ignored outputs stay separated",
        },
        evidence=evidence,
        point_smoke_blockers=tuple(point_blockers),
        upgrade_blockers=tuple(upgrade_blockers),
        next_steps=next_steps,
    )


def validate_renderer_loss_boundary_summary(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        raise TypeError("renderer loss boundary payload must be a dict")
    if payload.get("schema") != RENDERER_LOSS_BOUNDARY_SCHEMA:
        raise ValueError(f"unsupported renderer loss boundary schema: {payload.get('schema')}")
    required = (
        "status",
        "current_renderer",
        "target_renderer",
        "input_frame_contract",
        "render_target_contract",
        "loss_telemetry_contract",
        "decoder_handoff_contract",
        "integration_contract",
        "upgrade_blockers",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"renderer loss boundary payload missing keys: {', '.join(missing)}")
    return True


def _kernel_summary_evidence(kernel_summary: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    if kernel_summary is None:
        return {"kind": "no_kernel_summary"}, ["missing_kernel_summary"]
    if not isinstance(kernel_summary, dict):
        raise TypeError("kernel_summary must be a dict")
    if kernel_summary.get("schema") == OBJECT_EMERGENCE_SOLVER_TRAINING_SCHEMA:
        return _solver_training_evidence(kernel_summary)
    if kernel_summary.get("schema") == OBJECT_EMERGENCE_SOLVER_CHECKPOINT_SCHEMA:
        return _solver_checkpoint_evidence(kernel_summary)
    if kernel_summary.get("schema") == OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA:
        return _decoder_training_evidence(kernel_summary)
    if kernel_summary.get("schema") in {
        SOLVER_DECODER_JOINT_TRAINING_SCHEMA,
        SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA,
        ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA,
    }:
        return _solver_decoder_joint_evidence(kernel_summary)
    if kernel_summary.get("schema") != TRAINABLE_KERNEL_SCHEMA:
        raise ValueError(f"unsupported trainable kernel schema: {kernel_summary.get('schema')}")

    initial = _loss_record(kernel_summary, "initial_loss")
    final = _loss_record(kernel_summary, "final_loss")
    loss_decreased = final["total_loss"] < initial["total_loss"]
    render_loss_decreased = final["render_loss"] < initial["render_loss"]
    blockers: list[str] = []
    if not loss_decreased:
        blockers.append("total_loss_not_decreased")
    if not render_loss_decreased:
        blockers.append("render_loss_not_decreased")
    sample = kernel_summary.get("sample") if isinstance(kernel_summary.get("sample"), dict) else {}
    image_target_contract = (
        kernel_summary.get("image_target_contract")
        if isinstance(kernel_summary.get("image_target_contract"), dict)
        else {}
    )
    renderer_api = (
        kernel_summary.get("renderer_api")
        if isinstance(kernel_summary.get("renderer_api"), dict)
        else {}
    )
    image_targets_bound = image_target_contract.get("status") == "image_targets_bound"
    renderer_api_ready = renderer_api.get("status") == "ready"
    evidence = {
        "kind": "trainable_kernel_summary",
        "schema": kernel_summary.get("schema"),
        "frame_count": _optional_int(kernel_summary.get("frame_count")),
        "slots": _optional_int(kernel_summary.get("slots")),
        "sampled_count": _optional_int(sample.get("sampled_count")),
        "source_count": _optional_int(sample.get("source_count")),
        "target_source": sample.get("target_source"),
        "image_targets_bound": image_targets_bound,
        "image_target_contract_schema": image_target_contract.get("schema"),
        "image_target_visibility_policies": image_target_contract.get("visibility_policies", []),
        "renderer_api_ready": renderer_api_ready,
        "renderer_api_schema": renderer_api.get("schema"),
        "renderer_name": renderer_api.get("renderer_name"),
        "renderer_gradient_path": renderer_api.get("gradient_path"),
        "image_render_loss": renderer_api.get("image_render_loss"),
        "initial_total_loss": initial["total_loss"],
        "final_total_loss": final["total_loss"],
        "initial_render_loss": initial["render_loss"],
        "final_render_loss": final["render_loss"],
        "initial_image_render_loss": initial["image_render_loss"],
        "final_image_render_loss": final["image_render_loss"],
        "loss_decreased": loss_decreased,
        "render_loss_decreased": render_loss_decreased,
    }
    return evidence, blockers


def _decoder_handoff_contract(evidence: dict[str, Any], *, target_renderer: str) -> dict[str, Any]:
    kind = evidence.get("kind")
    solver_ready = kind in OBJECT_EMERGENCE_SOLVER_EVIDENCE and bool(
        evidence.get("solver_loss_decreased") and evidence.get("assignment_loss_decreased")
    )
    decoder_training_ready = kind in OBJECT_STATE_DECODER_TRAINING_EVIDENCE and bool(
        evidence.get("loss_decreased") and evidence.get("image_render_loss_decreased")
    )
    joint_training_ready = kind in SOLVER_DECODER_JOINT_EVIDENCE and bool(
        evidence.get("loss_decreased") and evidence.get("image_render_loss_decreased")
    )
    assignment_v2_renderer_ready = kind in ASSIGNMENT_V2_RENDER_JOINT_EVIDENCE and bool(
        evidence.get("loss_decreased") and evidence.get("image_render_loss_decreased")
    )
    renderer_ready = bool(evidence.get("renderer_api_ready"))
    full_renderer_ready = renderer_ready and evidence.get("renderer_name") in FULL_3DGS_RENDERERS
    if full_renderer_ready and joint_training_ready:
        status = "full_renderer_solver_decoder_joint_training_ready"
    elif full_renderer_ready and assignment_v2_renderer_ready:
        status = "full_renderer_assignment_v2_renderer_joint_validation_ready"
    elif joint_training_ready:
        status = "solver_decoder_joint_training_ready"
    elif assignment_v2_renderer_ready:
        status = "assignment_v2_renderer_joint_validation_ready"
    elif full_renderer_ready and decoder_training_ready:
        status = "full_renderer_decoder_training_ready"
    elif decoder_training_ready:
        status = "decoder_training_ready"
    elif full_renderer_ready:
        status = "full_renderer_decoder_ready"
    elif renderer_ready:
        status = "renderer_api_decoder_smoke_ready"
    elif solver_ready:
        status = "solver_checkpoint_ready"
    else:
        status = "awaiting_solver_checkpoint"
    return {
        "schema": "objgauss-decoder-renderer-handoff-v1",
        "status": status,
        "source_evidence": kind,
        "target_renderer": target_renderer,
        "state_chain": [
            "solver_checkpoint",
            "PerceptionEvidence",
            "assignment A[N,K]",
            "ObjectStateProjection",
            "GaussianToken decode",
            "renderer_api image_render_loss",
        ],
        "object_state_input_contract": {
            "assignment": "float32[N,K] row-normalized",
            "evidence": "positions float32[N,3] + features float32[N,D]",
            "projection": "ObjectStateProjection with derived object_id, centroid, bbox, feature, confidence",
            "identity": "object_id remains derived from assignment/matching/export policy",
        },
        "gaussian_decoder_contract": {
            "function": "decode_gaussian(ObjectStateProjection, source_gaussian_artifact) -> GaussianArtifact",
            "token_fields": {
                "mu": "float32[3]",
                "covariance": "float32[6] or renderer-native covariance fields",
                "color": "float32[3] or SH",
                "opacity": "float32[1]",
                "object_id": "derived integer renderer address",
            },
            "ragged_children": "Gaussian children remain ragged per ObjectState; do not require dense R[B,T,K,N,D].",
        },
        "renderer_loss_binding_contract": {
            "target": "image-space photometric loss plus object and temporal terms",
            "requires": [
                "camera intrinsics/extrinsics",
                "image targets",
                "visibility/depth/alpha policy",
                "renderer-native Gaussian parameters",
                "gradient path from renderer loss back to decoder/solver trainable fields",
            ],
        },
        "ready_without_gpu": bool(
            solver_ready
            or renderer_ready
            or decoder_training_ready
            or joint_training_ready
            or assignment_v2_renderer_ready
        ),
        "starts_real_training": bool(decoder_training_ready or joint_training_ready),
        "remaining_before_full_training": [
            "promote v2 assignment checkpoint into solver-decoder joint training",
            "run controlled gsplat renderer smoke",
            "keep identity hard gate as promotion blocker before renderer loss",
        ]
        if assignment_v2_renderer_ready and not full_renderer_ready
        else [
            "switch decoder training smoke to full 3DGS renderer",
            "keep geometry/opacity/scale frozen until color-only optimization is stable",
        ]
        if decoder_training_ready and not full_renderer_ready
        else [
            "switch joint training smoke to full 3DGS renderer",
            "keep dynamic-K and Gaussian geometry updates gated until joint training is stable",
        ]
        if joint_training_ready and not full_renderer_ready
        else [
            "bind solver checkpoint output to Gaussian decoder parameters",
            "bind decoded Gaussian artifact to renderer_api loss producer",
            "pass torch/gsplat/CUDA/NVIDIA driver preflight",
        ]
        if not (
            full_renderer_ready
            or decoder_training_ready
            or joint_training_ready
            or assignment_v2_renderer_ready
        )
        else [],
    }


def _solver_training_evidence(summary: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    initial = _solver_loss_record(summary, "initial_loss")
    final = _solver_loss_record(summary, "final_loss")
    loss_decreased = final["total_loss"] < initial["total_loss"]
    assignment_loss_decreased = final["assignment_loss"] < initial["assignment_loss"]
    blockers: list[str] = ["point_render_smoke_not_present"]
    if not loss_decreased:
        blockers.append("solver_total_loss_not_decreased")
    if not assignment_loss_decreased:
        blockers.append("solver_assignment_loss_not_decreased")
    gpu_policy = summary.get("gpu_policy") if isinstance(summary.get("gpu_policy"), dict) else {}
    evidence = {
        "kind": "object_emergence_solver_training",
        "schema": summary.get("schema"),
        "iterations": _optional_int(summary.get("iterations")),
        "slots": _optional_int(summary.get("final_solver_state", {}).get("config", {}).get("slots"))
        if isinstance(summary.get("final_solver_state"), dict)
        else None,
        "sampled_count": _optional_int(summary.get("sampled_gaussians")),
        "source_count": _optional_int(summary.get("source_gaussians")),
        "target_source": summary.get("target_source"),
        "image_targets_bound": False,
        "image_target_contract_schema": None,
        "image_target_visibility_policies": [],
        "renderer_api_ready": False,
        "renderer_api_schema": None,
        "renderer_name": None,
        "renderer_gradient_path": None,
        "image_render_loss": None,
        "initial_total_loss": initial["total_loss"],
        "final_total_loss": final["total_loss"],
        "initial_assignment_loss": initial["assignment_loss"],
        "final_assignment_loss": final["assignment_loss"],
        "solver_loss_decreased": loss_decreased,
        "assignment_loss_decreased": assignment_loss_decreased,
        "gpu_used": bool(gpu_policy.get("uses_gpu", False)),
        "vram_reserve_gb": _optional_int(gpu_policy.get("vram_reserve_gb")),
    }
    return evidence, blockers


def _solver_checkpoint_evidence(checkpoint: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    training = checkpoint.get("training")
    if not isinstance(training, dict):
        raise ValueError("solver checkpoint missing training")
    initial = _solver_loss_record(training, "initial_loss")
    final = _solver_loss_record(training, "final_loss")
    loss_decreased = final["total_loss"] < initial["total_loss"]
    assignment_loss_decreased = final["assignment_loss"] < initial["assignment_loss"]
    blockers: list[str] = ["point_render_smoke_not_present"]
    if not loss_decreased:
        blockers.append("solver_total_loss_not_decreased")
    if not assignment_loss_decreased:
        blockers.append("solver_assignment_loss_not_decreased")
    source = checkpoint.get("source") if isinstance(checkpoint.get("source"), dict) else {}
    solver_state = checkpoint.get("solver_state") if isinstance(checkpoint.get("solver_state"), dict) else {}
    config = solver_state.get("config") if isinstance(solver_state.get("config"), dict) else {}
    gpu_policy = checkpoint.get("gpu_policy") if isinstance(checkpoint.get("gpu_policy"), dict) else {}
    evidence = {
        "kind": "object_emergence_solver_checkpoint",
        "schema": checkpoint.get("schema"),
        "training_schema": checkpoint.get("training_schema"),
        "iterations": _optional_int(training.get("iterations")),
        "slots": _optional_int(config.get("slots")),
        "feature_dim": _optional_int(config.get("feature_dim")),
        "sampled_count": _optional_int(source.get("sampled_gaussians")),
        "source_count": _optional_int(source.get("source_gaussians")),
        "target_source": source.get("target_source"),
        "image_targets_bound": False,
        "image_target_contract_schema": None,
        "image_target_visibility_policies": [],
        "renderer_api_ready": False,
        "renderer_api_schema": None,
        "renderer_name": None,
        "renderer_gradient_path": None,
        "image_render_loss": None,
        "initial_total_loss": initial["total_loss"],
        "final_total_loss": final["total_loss"],
        "initial_assignment_loss": initial["assignment_loss"],
        "final_assignment_loss": final["assignment_loss"],
        "solver_loss_decreased": loss_decreased,
        "assignment_loss_decreased": assignment_loss_decreased,
        "gpu_used": bool(gpu_policy.get("uses_gpu", False)),
        "vram_reserve_gb": _optional_int(gpu_policy.get("vram_reserve_gb")),
    }
    return evidence, blockers


def _decoder_training_evidence(summary: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    initial = _loss_record(summary, "initial_loss")
    final = _loss_record(summary, "final_loss")
    loss_decreased = final["total_loss"] < initial["total_loss"]
    image_render_loss_decreased = final["image_render_loss"] < initial["image_render_loss"]
    render_loss_decreased = final["render_loss"] < initial["render_loss"]
    blockers: list[str] = ["point_render_smoke_not_present"]
    if not loss_decreased:
        blockers.append("decoder_total_loss_not_decreased")
    if not image_render_loss_decreased:
        blockers.append("decoder_image_render_loss_not_decreased")
    sample = summary.get("sample") if isinstance(summary.get("sample"), dict) else {}
    image_target_contract = (
        summary.get("image_target_contract")
        if isinstance(summary.get("image_target_contract"), dict)
        else {}
    )
    renderer_api = (
        summary.get("renderer_api")
        if isinstance(summary.get("renderer_api"), dict)
        else {}
    )
    gpu_policy = summary.get("gpu_policy") if isinstance(summary.get("gpu_policy"), dict) else {}
    renderer_api_ready = renderer_api.get("status") == "ready"
    evidence = {
        "kind": "object_state_gaussian_decoder_training",
        "schema": summary.get("schema"),
        "decoder_schema": summary.get("decoder_schema"),
        "iterations": _optional_int(summary.get("iterations")),
        "frame_count": _optional_int(summary.get("frame_count")),
        "slots": _optional_int(summary.get("slots")),
        "sampled_count": _optional_int(sample.get("sampled_count")),
        "source_count": _optional_int(sample.get("source_count")),
        "target_source": summary.get("assignment_source") or sample.get("target_source"),
        "trained_fields": summary.get("trained_fields", []),
        "frozen_fields": summary.get("frozen_fields", []),
        "image_targets_bound": image_target_contract.get("status") == "image_targets_bound",
        "image_target_contract_schema": image_target_contract.get("schema"),
        "image_target_visibility_policies": image_target_contract.get("visibility_policies", []),
        "renderer_api_ready": renderer_api_ready,
        "renderer_api_schema": renderer_api.get("schema"),
        "renderer_name": renderer_api.get("renderer_name"),
        "renderer_gradient_path": renderer_api.get("gradient_path"),
        "image_render_loss": renderer_api.get("image_render_loss"),
        "initial_total_loss": initial["total_loss"],
        "final_total_loss": final["total_loss"],
        "initial_render_loss": initial["render_loss"],
        "final_render_loss": final["render_loss"],
        "initial_image_render_loss": initial["image_render_loss"],
        "final_image_render_loss": final["image_render_loss"],
        "loss_decreased": loss_decreased,
        "render_loss_decreased": render_loss_decreased,
        "image_render_loss_decreased": image_render_loss_decreased,
        "gpu_used": bool(gpu_policy.get("uses_gpu", False)),
        "vram_reserve_gb": _optional_int(gpu_policy.get("vram_reserve_gb")),
    }
    return evidence, blockers


def _solver_decoder_joint_evidence(summary: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    is_checkpoint = summary.get("schema") == SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA
    training = summary.get("training") if is_checkpoint else summary
    if not isinstance(training, dict):
        raise ValueError("joint checkpoint missing training")
    initial = _joint_loss_record(training, "initial_loss")
    final = _joint_loss_record(training, "final_loss")
    delta = _joint_run_delta(summary, initial=initial, final=final, use_run_loss=not is_checkpoint)
    loss_decreased = delta["loss_decreased"]
    image_render_loss_decreased = delta["image_render_loss_decreased"]
    object_loss_decreased = delta["object_loss_decreased"]
    blockers: list[str] = []
    if not loss_decreased:
        blockers.append("joint_total_loss_not_decreased")
    if not image_render_loss_decreased:
        blockers.append("joint_image_render_loss_not_decreased")
    sample = summary.get("sample") if isinstance(summary.get("sample"), dict) else {}
    source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
    image_target_contract = (
        summary.get("image_target_contract")
        if isinstance(summary.get("image_target_contract"), dict)
        else {}
    )
    renderer_api = (
        summary.get("renderer_api")
        if isinstance(summary.get("renderer_api"), dict)
        else {}
    )
    gpu_policy = summary.get("gpu_policy") if isinstance(summary.get("gpu_policy"), dict) else {}
    renderer_api_ready = renderer_api.get("status") == "ready"
    evidence = {
        "kind": summary.get("kind") or "solver_decoder_joint_training",
        "schema": summary.get("schema"),
        "decoder_schema": summary.get("decoder_schema"),
        "training_schema": summary.get("training_schema"),
        "iterations": _optional_int(training.get("iterations")),
        "frame_count": _optional_int(summary.get("frame_count")),
        "slots": _optional_int(summary.get("slots"))
        if summary.get("slots") is not None
        else _optional_int(summary.get("solver_state", {}).get("config", {}).get("slots"))
        if isinstance(summary.get("solver_state"), dict)
        else None,
        "sampled_count": _optional_int(sample.get("sampled_count") or source.get("sampled_gaussians")),
        "source_count": _optional_int(sample.get("source_count") or source.get("source_gaussians")),
        "target_source": summary.get("assignment_source")
        or source.get("assignment_source")
        or sample.get("target_source")
        or source.get("target_source"),
        "trained_fields": summary.get("trained_fields", []),
        "frozen_fields": summary.get("frozen_fields", []),
        "image_targets_bound": image_target_contract.get("status") == "image_targets_bound",
        "image_target_contract_schema": image_target_contract.get("schema"),
        "image_target_visibility_policies": image_target_contract.get("visibility_policies", []),
        "renderer_api_ready": renderer_api_ready,
        "renderer_api_schema": renderer_api.get("schema"),
        "renderer_name": renderer_api.get("renderer_name"),
        "renderer_gradient_path": renderer_api.get("gradient_path"),
        "image_render_loss": renderer_api.get("image_render_loss"),
        "loss_delta_source": delta["source"],
        "initial_total_loss": delta["initial_total_loss"],
        "final_total_loss": delta["final_total_loss"],
        "initial_image_render_loss": delta["initial_image_render_loss"],
        "final_image_render_loss": delta["final_image_render_loss"],
        "initial_object_loss": delta["initial_object_loss"],
        "final_object_loss": delta["final_object_loss"],
        "segment_initial_total_loss": initial["total_loss"],
        "segment_final_total_loss": final["total_loss"],
        "segment_initial_image_render_loss": initial["image_render_loss"],
        "segment_final_image_render_loss": final["image_render_loss"],
        "segment_initial_object_loss": initial["object_loss"],
        "segment_final_object_loss": final["object_loss"],
        "loss_decreased": loss_decreased,
        "image_render_loss_decreased": image_render_loss_decreased,
        "object_loss_decreased": object_loss_decreased,
        "gpu_used": bool(gpu_policy.get("uses_gpu", False)),
        "vram_reserve_gb": _optional_int(gpu_policy.get("vram_reserve_gb")),
    }
    return evidence, blockers


def _joint_run_delta(
    summary: dict[str, Any],
    *,
    initial: dict[str, float],
    final: dict[str, float],
    use_run_loss: bool,
) -> dict[str, Any]:
    run_loss = summary.get("run_loss") if use_run_loss else None
    if isinstance(run_loss, dict):
        required = (
            "initial_total_loss",
            "final_total_loss",
            "initial_image_render_loss",
            "final_image_render_loss",
            "initial_object_loss",
            "final_object_loss",
        )
        missing = [field for field in required if field not in run_loss]
        if missing:
            raise ValueError(f"run_loss missing fields: {', '.join(missing)}")
        initial_total = _finite_float(run_loss["initial_total_loss"], "run_loss.initial_total_loss")
        final_total = _finite_float(run_loss["final_total_loss"], "run_loss.final_total_loss")
        initial_image = _finite_float(
            run_loss["initial_image_render_loss"],
            "run_loss.initial_image_render_loss",
        )
        final_image = _finite_float(
            run_loss["final_image_render_loss"],
            "run_loss.final_image_render_loss",
        )
        initial_object = _finite_float(run_loss["initial_object_loss"], "run_loss.initial_object_loss")
        final_object = _finite_float(run_loss["final_object_loss"], "run_loss.final_object_loss")
        return {
            "source": "run_loss",
            "initial_total_loss": initial_total,
            "final_total_loss": final_total,
            "initial_image_render_loss": initial_image,
            "final_image_render_loss": final_image,
            "initial_object_loss": initial_object,
            "final_object_loss": final_object,
            "loss_decreased": final_total < initial_total,
            "image_render_loss_decreased": final_image < initial_image,
            "object_loss_decreased": final_object < initial_object,
        }
    return {
        "source": "segment_loss",
        "initial_total_loss": initial["total_loss"],
        "final_total_loss": final["total_loss"],
        "initial_image_render_loss": initial["image_render_loss"],
        "final_image_render_loss": final["image_render_loss"],
        "initial_object_loss": initial["object_loss"],
        "final_object_loss": final["object_loss"],
        "loss_decreased": final["total_loss"] < initial["total_loss"],
        "image_render_loss_decreased": final["image_render_loss"] < initial["image_render_loss"],
        "object_loss_decreased": final["object_loss"] < initial["object_loss"],
    }


def _loss_record(summary: dict[str, Any], key: str) -> dict[str, float]:
    value = summary.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    required = ("total_loss", "render_loss", "object_loss", "temporal_loss")
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"{key} missing loss fields: {', '.join(missing)}")
    record = {field: _finite_float(value[field], f"{key}.{field}") for field in required}
    record["image_render_loss"] = (
        _finite_float(value["image_render_loss"], f"{key}.image_render_loss")
        if "image_render_loss" in value
        else 0.0
    )
    return record


def _solver_loss_record(summary: dict[str, Any], key: str) -> dict[str, float]:
    value = summary.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    required = ("total_loss", "assignment_loss", "entropy_loss", "balance_loss", "temporal_loss")
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"{key} missing solver loss fields: {', '.join(missing)}")
    return {field: _finite_float(value[field], f"{key}.{field}") for field in required}


def _joint_loss_record(summary: dict[str, Any], key: str) -> dict[str, float]:
    value = summary.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    required = (
        "total_loss",
        "image_render_loss",
        "object_loss",
        "entropy_loss",
        "balance_loss",
        "temporal_loss",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"{key} missing joint loss fields: {', '.join(missing)}")
    return {field: _finite_float(value[field], f"{key}.{field}") for field in required}


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return number


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    number = int(value)
    return number
