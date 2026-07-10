"""Deprecated compatibility import for the BOP identity route audit."""

from objgauss.pipelines.objectstate_bop_identity_route_audit import (
    OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA,
    objectstate_bop_identity_route_audit,
    validate_objectstate_bop_identity_route_audit_summary,
)

__all__ = (
    "OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA",
    "objectstate_bop_identity_route_audit",
    "validate_objectstate_bop_identity_route_audit_summary",
)
