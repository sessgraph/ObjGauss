from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from objgauss_sim.runtime import (
    REQUIRED_VERSIONS,
    assert_offline_asset_gate,
    assert_versions,
)


class RuntimeBoundaryTests(unittest.TestCase):
    def test_locked_versions_are_observed(self) -> None:
        self.assertEqual(assert_versions(), REQUIRED_VERSIONS)

    def test_asset_gate_accepts_only_empty_read_only_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            asset_dir = Path(temporary) / "no-assets"
            asset_dir.mkdir()
            asset_dir.chmod(0o555)
            with mock.patch.dict(
                os.environ,
                {
                    "OBJGAUSS_RUNTIME_OFFLINE": "1",
                    "MS_ASSET_DIR": str(asset_dir),
                },
                clear=True,
            ):
                observed = assert_offline_asset_gate()
        self.assertTrue(observed["empty"])
        self.assertEqual(observed["mode_octal"], "555")

    def test_asset_gate_rejects_writable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            asset_dir = Path(temporary) / "no-assets"
            asset_dir.mkdir()
            asset_dir.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
            with mock.patch.dict(
                os.environ,
                {
                    "OBJGAUSS_RUNTIME_OFFLINE": "1",
                    "MS_ASSET_DIR": str(asset_dir),
                },
                clear=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "must be read-only"):
                    assert_offline_asset_gate()

    def test_asset_gate_requires_explicit_offline_mode(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OBJGAUSS_RUNTIME_OFFLINE"):
                assert_offline_asset_gate()

    def test_network_guard_rejects_outbound_connection(self) -> None:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import socket; "
                    "from objgauss_sim.runtime import install_network_guard; "
                    "install_network_guard(); "
                    "socket.create_connection(('127.0.0.1', 9))"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(probe.returncode, 0)
        self.assertIn("network access is forbidden", probe.stderr)


if __name__ == "__main__":
    unittest.main()
