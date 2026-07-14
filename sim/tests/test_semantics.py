from __future__ import annotations

import math
import unittest

from objgauss_sim.canonical import canonical_json_bytes, digest
from objgauss_sim.primitive import SIBLINGS, SPEC, direction_checks
from objgauss_sim.smoke import compare_reports


class CanonicalSemanticsTests(unittest.TestCase):
    def test_dict_order_does_not_change_digest(self) -> None:
        self.assertEqual(digest({"b": 2, "a": 1}), digest({"a": 1, "b": 2}))

    def test_non_finite_values_are_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    canonical_json_bytes({"value": value})

    def test_five_branch_spec_and_changed_variable_are_frozen(self) -> None:
        self.assertEqual(len(SIBLINGS), 5)
        self.assertEqual(
            SPEC["changed_variable"], "/intervention/commanded_action"
        )
        self.assertEqual(
            [item["branch_id"] for item in SPEC["siblings"]], list(SIBLINGS)
        )

    def test_direction_sign_mutation_is_rejected(self) -> None:
        effects = {
            "hold_horizontal_drift_m": 0.0,
            "paired_displacement_m": {
                "hold": [0.0, 0.0, 0.0],
                "push_pos_x_weak": [-0.01, 0.0, 0.0],
                "push_pos_x_strong": [0.02, 0.0, 0.0],
                "push_neg_x_weak": [-0.01, 0.0, 0.0],
                "push_pos_y_weak": [0.0, 0.01, 0.0],
            },
        }
        self.assertFalse(direction_checks(effects)["push_pos_x_weak_direction_and_effect"])

    def test_repeat_requires_digest_order_and_valid_baseline(self) -> None:
        previous = {
            "evidence_sha256": "a" * 64,
            "execution_order": list(SIBLINGS),
            "verdict": "pending_repeat",
            "local_verdict": "supported",
            "runtime_telemetry": {"within_budget": True},
        }
        current = {
            "evidence_sha256": "a" * 64,
            "execution_order": list(reversed(SIBLINGS)),
        }
        comparison = compare_reports(previous, current)
        self.assertTrue(comparison["matches"])
        self.assertTrue(comparison["opposite_execution_orders"])
        self.assertTrue(comparison["previous_baseline_valid"])


if __name__ == "__main__":
    unittest.main()
