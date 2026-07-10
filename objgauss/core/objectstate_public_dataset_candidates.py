"""Deprecated compatibility import for public dataset candidates."""

from objgauss.pipelines.objectstate_public_dataset_candidates import (
    OBJECTSTATE_PUBLIC_DATASET_CANDIDATE_SCHEMA,
    OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA,
    OBJECTSTATE_PUBLIC_DATASET_COVERAGE_VALUES,
    OBJECTSTATE_PUBLIC_DATASET_GATE_KINDS,
    OBJECTSTATE_PUBLIC_DATASET_SOURCE_KINDS,
    OBJECTSTATE_PUBLIC_INTERACTION_ROUTE_AUDIT_SCHEMA,
    ObjectStatePublicDatasetCandidate,
    default_objectstate_public_dataset_candidates,
    objectstate_public_dataset_candidates_audit,
    objectstate_public_dataset_candidates_markdown,
    objectstate_public_interaction_route_audit,
    objectstate_public_interaction_route_markdown,
    validate_objectstate_public_dataset_candidate,
    validate_objectstate_public_dataset_candidates_audit,
    validate_objectstate_public_interaction_route_audit,
)

__all__ = (
    "OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA",
    "OBJECTSTATE_PUBLIC_DATASET_CANDIDATE_SCHEMA",
    "OBJECTSTATE_PUBLIC_INTERACTION_ROUTE_AUDIT_SCHEMA",
    "OBJECTSTATE_PUBLIC_DATASET_GATE_KINDS",
    "OBJECTSTATE_PUBLIC_DATASET_COVERAGE_VALUES",
    "OBJECTSTATE_PUBLIC_DATASET_SOURCE_KINDS",
    "ObjectStatePublicDatasetCandidate",
    "default_objectstate_public_dataset_candidates",
    "objectstate_public_dataset_candidates_audit",
    "objectstate_public_dataset_candidates_markdown",
    "objectstate_public_interaction_route_audit",
    "objectstate_public_interaction_route_markdown",
    "validate_objectstate_public_dataset_candidates_audit",
    "validate_objectstate_public_interaction_route_audit",
    "validate_objectstate_public_dataset_candidate",
)
