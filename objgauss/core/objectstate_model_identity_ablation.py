"""Deprecated compatibility import for model identity ablation."""

from objgauss.evaluation.objectstate_model_identity_ablation import (
    DEFAULT_OBJECTSTATE_MODEL_IDENTITY_ABLATION_POLICIES,
    OBJECTSTATE_MODEL_IDENTITY_ABLATION_SCHEMA,
    objectstate_model_identity_ablation_summary,
    validate_objectstate_model_identity_ablation_summary,
)

__all__ = (
    "OBJECTSTATE_MODEL_IDENTITY_ABLATION_SCHEMA",
    "DEFAULT_OBJECTSTATE_MODEL_IDENTITY_ABLATION_POLICIES",
    "objectstate_model_identity_ablation_summary",
    "validate_objectstate_model_identity_ablation_summary",
)
