"""Deprecated compatibility import for controlled capture annotations."""

from objgauss.datasets.objectstate_controlled_capture_annotations import (
    OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_TEMPLATE_SCHEMA,
    finalize_objectstate_controlled_capture_annotations,
    validate_objectstate_controlled_capture_annotation_finalize_summary,
    validate_objectstate_controlled_capture_annotation_template_summary,
    write_objectstate_controlled_capture_annotation_template,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_FINALIZE_SCHEMA",
    "OBJECTSTATE_CONTROLLED_CAPTURE_ANNOTATION_TEMPLATE_SCHEMA",
    "finalize_objectstate_controlled_capture_annotations",
    "validate_objectstate_controlled_capture_annotation_finalize_summary",
    "validate_objectstate_controlled_capture_annotation_template_summary",
    "write_objectstate_controlled_capture_annotation_template",
)
