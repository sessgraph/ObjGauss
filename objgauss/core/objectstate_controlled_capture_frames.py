"""Deprecated compatibility import for controlled capture frame authoring."""

from objgauss.datasets.objectstate_controlled_capture_frames import (
    OBJECTSTATE_CONTROLLED_CAPTURE_FRAMES_SCHEMA,
    validate_objectstate_controlled_capture_frames_summary,
    write_objectstate_controlled_capture_frames,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_CAPTURE_FRAMES_SCHEMA",
    "validate_objectstate_controlled_capture_frames_summary",
    "write_objectstate_controlled_capture_frames",
)
