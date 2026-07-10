"""Deprecated compatibility import for controlled capture bundle templates.

New code must import the canonical dataset module directly.  Explicit aliases
preserve object identity for callers migrating from the historical core path.
"""

from objgauss.datasets.objectstate_controlled_capture_template import (
    ACTIONS_CSV_HEADER,
    ANNOTATIONS_CSV_HEADER,
    FRAMES_CSV_HEADER,
    OBJECTS_CSV_HEADER,
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA,
    validate_objectstate_controlled_capture_bundle_template_summary,
    write_objectstate_controlled_capture_bundle_template,
)

__all__ = (
    "ACTIONS_CSV_HEADER",
    "ANNOTATIONS_CSV_HEADER",
    "FRAMES_CSV_HEADER",
    "OBJECTS_CSV_HEADER",
    "OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA",
    "validate_objectstate_controlled_capture_bundle_template_summary",
    "write_objectstate_controlled_capture_bundle_template",
)
