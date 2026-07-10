"""Deprecated compatibility import for the identity bundle handoff."""

from objgauss.pipelines.objectstate_controlled_identity_bundle_handoff import (
    OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA,
    objectstate_controlled_identity_bundle_handoff,
    validate_objectstate_controlled_identity_bundle_handoff_summary,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA",
    "objectstate_controlled_identity_bundle_handoff",
    "validate_objectstate_controlled_identity_bundle_handoff_summary",
)
