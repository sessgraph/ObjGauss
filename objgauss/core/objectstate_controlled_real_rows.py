"""Deprecated compatibility imports for controlled real manifests and rows.

The manifest contract is canonical in :mod:`objgauss.datasets`; row conversion
and gate summaries are canonical in :mod:`objgauss.evaluation`.
"""

from objgauss.datasets.objectstate_controlled_real_manifest import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    read_objectstate_controlled_real_manifest,
    validate_objectstate_controlled_real_manifest,
)
from objgauss.evaluation.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    evaluate_controlled_real_manifest_reality_gate,
    objectstate_controlled_real_rows_summary,
    objectstate_reality_rows_from_controlled_real_manifest,
    validate_objectstate_controlled_real_rows_summary,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA",
    "OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA",
    "evaluate_controlled_real_manifest_reality_gate",
    "objectstate_controlled_real_rows_summary",
    "objectstate_reality_rows_from_controlled_real_manifest",
    "read_objectstate_controlled_real_manifest",
    "validate_objectstate_controlled_real_manifest",
    "validate_objectstate_controlled_real_rows_summary",
)
