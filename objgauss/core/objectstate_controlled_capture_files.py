"""Deprecated compatibility import for controlled capture file auditing."""

from objgauss.datasets.objectstate_controlled_capture_files import (
    OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
    objectstate_controlled_capture_file_audit,
    objectstate_controlled_capture_missing_files_markdown,
    validate_objectstate_controlled_capture_file_audit_summary,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA",
    "objectstate_controlled_capture_file_audit",
    "objectstate_controlled_capture_missing_files_markdown",
    "validate_objectstate_controlled_capture_file_audit_summary",
)
