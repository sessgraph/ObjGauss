"""Deprecated compatibility import for controlled candidate templates."""

from objgauss.pipelines.objectstate_controlled_reality_candidate_template import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
    finalize_objectstate_controlled_prediction_candidate_template,
    finalize_objectstate_controlled_reality_candidate_templates,
    validate_objectstate_controlled_intervention_candidates_template,
    validate_objectstate_controlled_prediction_candidate_finalize_summary,
    validate_objectstate_controlled_prediction_candidates_template,
    validate_objectstate_controlled_reality_candidate_finalize_summary,
    validate_objectstate_controlled_reality_candidate_template_summary,
    write_objectstate_controlled_reality_candidate_templates,
    write_objectstate_controlled_reality_candidate_templates_from_manifest,
)

__all__ = (
    "OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA",
    "OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA",
    "OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA",
    "OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA",
    "OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA",
    "write_objectstate_controlled_reality_candidate_templates",
    "write_objectstate_controlled_reality_candidate_templates_from_manifest",
    "finalize_objectstate_controlled_reality_candidate_templates",
    "finalize_objectstate_controlled_prediction_candidate_template",
    "validate_objectstate_controlled_reality_candidate_template_summary",
    "validate_objectstate_controlled_reality_candidate_finalize_summary",
    "validate_objectstate_controlled_prediction_candidate_finalize_summary",
    "validate_objectstate_controlled_prediction_candidates_template",
    "validate_objectstate_controlled_intervention_candidates_template",
)
