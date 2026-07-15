"""Behavior and failure tests for the PR-02C C0 runtime gate."""

from __future__ import annotations

import ast
import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock

from objgauss_learning import runtime


LEARNING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = LEARNING_ROOT.parent


class RuntimeTests(unittest.TestCase):
    def test_manifest_binds_frozen_runtime_lock_grid_and_policy(self) -> None:
        manifest = runtime.validate_manifest(
            repo_root=REPO_ROOT,
            manifest_path=LEARNING_ROOT / "runtime-manifest.json",
            lock_path=LEARNING_ROOT / "uv.lock",
            grid_path=REPO_ROOT
            / "contracts"
            / "fixtures"
            / "pr02b"
            / "hyperparameter-grid.json",
        )
        self.assertEqual(manifest["manifest_kind"], runtime.MANIFEST_KIND)
        self.assertEqual(manifest["resources"]["display_vram_reserve_bytes_min"], 1024**3)

    def test_learning_sources_do_not_import_simulator_packages(self) -> None:
        forbidden = set(runtime.FORBIDDEN_MODULES)
        violations: list[str] = []
        for path in sorted((LEARNING_ROOT / "src" / "objgauss_learning").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".", 1)[0] in forbidden:
                        violations.append(f"{path.name}:{name}")
        self.assertEqual(violations, [])

    def test_only_torch_is_a_production_dependency(self) -> None:
        text = (LEARNING_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"torch==2.13.0"', text)
        for forbidden in ("pyg", "lightning", "hydra", "mani-skill", "sapien"):
            self.assertNotIn(forbidden, text.lower())

    def test_runtime_versions_match_the_frozen_cuda_build(self) -> None:
        self.assertEqual(runtime.assert_versions(), runtime.REQUIRED_VERSIONS)

    def test_version_drift_is_blocked(self) -> None:
        with mock.patch.object(
            runtime,
            "observed_versions",
            return_value={**runtime.REQUIRED_VERSIONS, "torch_cuda": "12.8"},
        ):
            with self.assertRaisesRegex(runtime.RuntimeBlockedError, "version drift"):
                runtime.assert_versions()

    def test_forbidden_distribution_is_invalid(self) -> None:
        real_version = importlib.metadata.version

        def fake_version(name: str) -> str:
            if name == "sapien":
                return "3.0.3"
            return real_version(name)

        with mock.patch.object(importlib.metadata, "version", side_effect=fake_version):
            with self.assertRaisesRegex(runtime.RuntimeInvalidError, "isolation violated"):
                runtime.assert_isolation()

    def test_source_commit_and_paths_fail_closed(self) -> None:
        with self.assertRaises(runtime.RuntimeInvalidError):
            runtime.validate_source_commit("HEAD")
        with self.assertRaises(runtime.RuntimeInvalidError):
            runtime.require_relative_path(REPO_ROOT, "/tmp/report.json")
        with self.assertRaises(runtime.RuntimeInvalidError):
            runtime.require_relative_path(REPO_ROOT, "../report.json")
        with self.assertRaises(runtime.RuntimeInvalidError):
            runtime.require_relative_path(
                REPO_ROOT, "generated/other/report.json", prefix="generated/pr02c"
            )

    def test_clean_head_must_match_the_declared_source_commit(self) -> None:
        commit = "a" * 40
        with mock.patch.object(
            runtime.subprocess,
            "run",
            side_effect=[
                CompletedProcess([], 0, stdout="", stderr=""),
                CompletedProcess([], 0, stdout=f"{commit}\n", stderr=""),
            ],
        ):
            self.assertEqual(
                runtime.assert_clean_git_head(REPO_ROOT, commit),
                {"worktree_clean": True, "head_matches_source_commit": True},
            )
        with mock.patch.object(
            runtime.subprocess,
            "run",
            return_value=CompletedProcess([], 0, stdout=" M README.md\n", stderr=""),
        ):
            with self.assertRaisesRegex(runtime.RuntimeInvalidError, "clean git worktree"):
                runtime.assert_clean_git_head(REPO_ROOT, commit)

    def test_explicit_offline_mode_is_required(self) -> None:
        with mock.patch.dict(runtime.os.environ, {}, clear=True):
            with self.assertRaisesRegex(
                runtime.RuntimeInvalidError, "OBJGAUSS_LEARNING_OFFLINE"
            ):
                runtime.assert_offline_mode()
        with mock.patch.dict(
            runtime.os.environ, {"OBJGAUSS_LEARNING_OFFLINE": "1"}, clear=True
        ):
            self.assertEqual(
                runtime.assert_offline_mode(), {"explicit_offline_mode": True}
            )

    def test_source_tree_hash_is_stable_and_complete(self) -> None:
        first = runtime.package_tree_sha256(REPO_ROOT)
        second = runtime.package_tree_sha256(REPO_ROOT)
        self.assertEqual(first, second)
        self.assertRegex(first, r"^[a-f0-9]{64}$")

    def test_atomic_write_replaces_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            runtime.atomic_write(output, runtime.strict_json_bytes({"status": "supported"}))
            self.assertEqual(json.loads(output.read_text()), {"status": "supported"})
            self.assertEqual(list(output.parent.glob(".report.json.*")), [])

    def test_failure_reports_keep_claims_narrow(self) -> None:
        report = runtime.failure_report(
            status="blocked", reason_code="runtime_unavailable", message="no CUDA"
        )
        self.assertEqual(report["verdict"]["status"], "blocked")
        self.assertEqual(report["claim_boundary"]["supported_claim"], "none")


if __name__ == "__main__":
    unittest.main()
