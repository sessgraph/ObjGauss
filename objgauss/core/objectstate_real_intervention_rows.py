"""Deprecated compatibility import for real intervention-row evaluation."""

from objgauss.evaluation.objectstate_real_intervention_rows import (
    OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA,
    objectstate_real_intervention_rows_from_bundle,
    objectstate_real_intervention_rows_summary,
    read_objectstate_real_intervention_rows_summary,
    validate_objectstate_real_intervention_rows_summary,
)

__all__ = (
    "OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA",
    "read_objectstate_real_intervention_rows_summary",
    "objectstate_real_intervention_rows_summary",
    "objectstate_real_intervention_rows_from_bundle",
    "validate_objectstate_real_intervention_rows_summary",
)
