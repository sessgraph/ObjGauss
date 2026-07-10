"""Deprecated compatibility import for the controlled capture contract.

New code must import the canonical dataset module directly.  Explicit aliases
preserve object identity for callers migrating from the historical core path.
"""

from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA,
    objectstate_controlled_capture_summary,
    objectstate_controlled_real_manifest_from_capture_manifest,
    read_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_summary,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA",
    "OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA",
    "objectstate_controlled_capture_summary",
    "objectstate_controlled_real_manifest_from_capture_manifest",
    "read_objectstate_controlled_capture_manifest",
    "validate_objectstate_controlled_capture_manifest",
    "validate_objectstate_controlled_capture_summary",
)
