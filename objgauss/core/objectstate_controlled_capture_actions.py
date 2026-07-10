"""Deprecated compatibility import for controlled capture actions."""

from objgauss.datasets.objectstate_controlled_capture_actions import (
    OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_TEMPLATE_SCHEMA,
    finalize_objectstate_controlled_capture_actions,
    validate_objectstate_controlled_capture_action_finalize_summary,
    validate_objectstate_controlled_capture_action_template_summary,
    write_objectstate_controlled_capture_action_template,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_FINALIZE_SCHEMA",
    "OBJECTSTATE_CONTROLLED_CAPTURE_ACTION_TEMPLATE_SCHEMA",
    "finalize_objectstate_controlled_capture_actions",
    "validate_objectstate_controlled_capture_action_finalize_summary",
    "validate_objectstate_controlled_capture_action_template_summary",
    "write_objectstate_controlled_capture_action_template",
)
