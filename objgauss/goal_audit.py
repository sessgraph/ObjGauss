from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from objgauss.demo import verify_v1_closure_demo
from objgauss.lego_verify import verify_lego_alpha_closure_demo


@dataclass(frozen=True)
class GoalAuditCheck:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class GoalAuditResult:
    passed: bool
    checks: tuple[GoalAuditCheck, ...]
    summary: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "checks": [check.as_dict() for check in self.checks],
        }


def audit_v1_goal(
    *,
    v1_manifest: str | Path = "outputs/demos/v1-closure/v1-closure-manifest.json",
    lego_manifest: str | Path = "outputs/demos/lego-alpha-closure/lego-alpha-closure-manifest.json",
    trained_manifest: str | Path | None = "outputs/assets/gaussians/nerf-lego-trained/training-output-manifest.json",
    asset_library_path: str | Path = "src/assetLibrary.js",
) -> GoalAuditResult:
    checks: list[GoalAuditCheck] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(GoalAuditCheck(name=name, passed=bool(passed), detail=detail))

    v1_result = _optional_v1_verification(v1_manifest, asset_library_path=asset_library_path)
    lego_result = _optional_lego_verification(lego_manifest, asset_library_path=asset_library_path)
    trained = _load_optional_manifest(trained_manifest)

    add(
        "real_3dgs_scene_can_render",
        bool(v1_result and _check_passed(v1_result.checks, "real_3dgs_scene")),
        _verification_detail(v1_result, "real_3dgs_scene", missing=str(v1_manifest)),
    )
    add(
        "mask_guidance_influences_object_field",
        bool(
            (v1_result and _check_passed(v1_result.checks, "mask_guidance_changed_object_field"))
            or (lego_result and _check_passed(lego_result.checks, "mask_guidance_changed_object_field"))
            or _training_acceptance(trained, "mask_guidance_changed_object_field")
        ),
        _mask_guidance_detail(v1_result, lego_result, trained),
    )
    add(
        "object_id_export_available",
        bool(
            (v1_result and _check_passed(v1_result.checks, "viewer_ply_exports_object_id"))
            and (lego_result and _check_passed(lego_result.checks, "viewer_ply_exports_object_id"))
        ),
        _object_id_detail(v1_result, lego_result),
    )
    add(
        "frontend_object_interaction_registered",
        bool(
            (v1_result and _check_passed(v1_result.checks, "frontend_asset_registered"))
            and (lego_result and _check_passed(lego_result.checks, "frontend_asset_registered"))
        ),
        _frontend_detail(v1_result, lego_result),
    )
    add(
        "fixed_reproducible_commands_exist",
        Path("scripts/acceptance-demo.mjs").exists(),
        "npm run acceptance:demo -> scripts/acceptance-demo.mjs",
    )

    unified_passed, unified_detail = _unified_demo_check(trained)
    add("unified_real_3dgs_mask_demo_available", unified_passed, unified_detail)

    passed = all(check.passed for check in checks)
    return GoalAuditResult(
        passed=passed,
        checks=tuple(checks),
        summary={
            "v1_manifest": str(v1_manifest),
            "lego_manifest": str(lego_manifest),
            "trained_manifest": str(trained_manifest) if trained_manifest is not None else None,
            "current_evidence": (
                "split" if not unified_passed else "unified"
            ),
            "completion_blockers": [
                check.name for check in checks if not check.passed
            ],
        },
    )


def _optional_v1_verification(path: str | Path, *, asset_library_path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    return verify_v1_closure_demo(path, asset_library_path=asset_library_path)


def _optional_lego_verification(path: str | Path, *, asset_library_path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    return verify_lego_alpha_closure_demo(path, asset_library_path=asset_library_path)


def _load_optional_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _check_passed(checks: tuple[dict[str, object], ...], name: str) -> bool:
    return any(check.get("name") == name and bool(check.get("passed")) for check in checks)


def _check_detail(checks: tuple[dict[str, object], ...], name: str) -> str | None:
    for check in checks:
        if check.get("name") == name:
            return str(check.get("detail"))
    return None


def _verification_detail(result, check_name: str, *, missing: str) -> str:
    if result is None:
        return f"missing manifest: {missing}"
    detail = _check_detail(result.checks, check_name)
    return detail or f"{check_name} not found"


def _mask_guidance_detail(v1_result, lego_result, trained: dict[str, Any] | None) -> str:
    parts: list[str] = []
    if v1_result is not None:
        parts.append(
            "plush="
            + (_check_detail(v1_result.checks, "mask_guidance_changed_object_field") or "missing")
        )
    if lego_result is not None:
        parts.append(
            "lego_proxy="
            + (_check_detail(lego_result.checks, "mask_guidance_changed_object_field") or "missing")
        )
    if trained is not None:
        delta = trained.get("object_field_delta")
        if isinstance(delta, dict):
            parts.append(f"trained=changed_gaussians={delta.get('changed_gaussians')}")
    return "; ".join(parts) if parts else "no mask guidance evidence"


def _object_id_detail(v1_result, lego_result) -> str:
    return (
        "plush="
        + _verification_detail(v1_result, "viewer_ply_exports_object_id", missing="v1")
        + "; lego_proxy="
        + _verification_detail(lego_result, "viewer_ply_exports_object_id", missing="lego")
    )


def _frontend_detail(v1_result, lego_result) -> str:
    return (
        "plush="
        + _verification_detail(v1_result, "frontend_asset_registered", missing="v1")
        + "; lego_proxy="
        + _verification_detail(lego_result, "frontend_asset_registered", missing="lego")
    )


def _training_acceptance(manifest: dict[str, Any] | None, key: str) -> bool:
    if manifest is None:
        return False
    acceptance = manifest.get("acceptance")
    return isinstance(acceptance, dict) and bool(acceptance.get(key))


def _unified_demo_check(manifest: dict[str, Any] | None) -> tuple[bool, str]:
    if manifest is None:
        return (
            False,
            "missing registered trained Gaussian manifest; expected outputs/assets/gaussians/nerf-lego-trained/training-output-manifest.json",
        )
    gaussian_source_ok = manifest.get("gaussian_source") == "external_3dgs_training_output"
    object_ply = manifest.get("public_object_ply") or manifest.get("object_ply")
    splat = manifest.get("public_splat") or manifest.get("splat_path")
    object_ply_ok = isinstance(object_ply, str) and Path(object_ply).exists()
    splat_ok = isinstance(splat, str) and Path(splat).exists()
    mask_changed = _training_acceptance(manifest, "mask_guidance_changed_object_field")
    loss_decreased = _training_acceptance(manifest, "projection_loss_decreased")
    passed = gaussian_source_ok and object_ply_ok and splat_ok and mask_changed and loss_decreased
    return (
        passed,
        (
            f"gaussian_source={manifest.get('gaussian_source')} "
            f"splat={splat_ok} object_ply={object_ply_ok} "
            f"mask_changed={mask_changed} loss_decreased={loss_decreased}"
        ),
    )
