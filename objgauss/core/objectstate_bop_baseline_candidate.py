"""Deprecated compatibility import for the BOP baseline candidate."""

from objgauss.pipelines.objectstate_bop_baseline_candidate import (
    OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA,
    validate_objectstate_bop_baseline_candidate_summary,
    write_objectstate_bop_gaussian_centroid_baseline_candidate,
)

__all__ = (
    "OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA",
    "write_objectstate_bop_gaussian_centroid_baseline_candidate",
    "validate_objectstate_bop_baseline_candidate_summary",
)
