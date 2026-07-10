"""Deprecated compatibility import for the BOP RGB-D Gaussian export."""

from objgauss.pipelines.objectstate_bop_rgbd_gaussian_export import (
    OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA,
    objectstate_bop_rgbd_gaussian_export,
    validate_objectstate_bop_rgbd_gaussian_export_summary,
)

__all__ = (
    "OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA",
    "objectstate_bop_rgbd_gaussian_export",
    "validate_objectstate_bop_rgbd_gaussian_export_summary",
)
