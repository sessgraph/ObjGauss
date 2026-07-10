"""Deprecated compatibility import for the BOP candidate artifact template."""

from objgauss.pipelines.objectstate_bop_candidate_artifact_template import (
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA,
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA,
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA,
    finalize_objectstate_bop_candidate_artifact_template,
    validate_objectstate_bop_candidate_artifact_finalize_summary,
    validate_objectstate_bop_candidate_artifact_template,
    validate_objectstate_bop_candidate_artifact_template_summary,
    write_objectstate_bop_candidate_artifact_template,
)

__all__ = (
    "OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA",
    "OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA",
    "OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA",
    "write_objectstate_bop_candidate_artifact_template",
    "finalize_objectstate_bop_candidate_artifact_template",
    "validate_objectstate_bop_candidate_artifact_template",
    "validate_objectstate_bop_candidate_artifact_finalize_summary",
    "validate_objectstate_bop_candidate_artifact_template_summary",
)
