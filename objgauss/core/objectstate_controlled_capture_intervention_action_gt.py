"""Deprecated compatibility import for controlled action-GT readiness.

New code must import the canonical dataset module directly.  Explicit aliases
preserve object identity for callers migrating from the historical core path.
"""

from objgauss.datasets.objectstate_controlled_capture_intervention_action_gt import (
    objectstate_controlled_capture_intervention_action_gt_readiness,
    validate_objectstate_controlled_capture_intervention_action_gt_readiness,
)

__all__ = (
    "objectstate_controlled_capture_intervention_action_gt_readiness",
    "validate_objectstate_controlled_capture_intervention_action_gt_readiness",
)
