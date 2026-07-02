"""Stable core algorithm entry points for ObjGauss.

The namespace is intentionally lazy. Compatibility wrappers such as
`objgauss.gaussians` import specific core submodules during package
initialization, so eagerly importing every core domain here can create circular
imports while the migration is in progress.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "GaussianCloud": ("objgauss.core.gaussian", "GaussianCloud"),
    "ObjectField": ("objgauss.core.object_field", "ObjectField"),
    "align_mask_manifest_slots": ("objgauss.core.semantics", "align_mask_manifest_slots"),
    "append_or_replace_property": ("objgauss.core.io", "append_or_replace_property"),
    "assign_object_ids": ("objgauss.core.objects", "assign_object_ids"),
    "attach_hard_labels": ("objgauss.core.object_field", "attach_hard_labels"),
    "attach_object_aware_lod_metadata": ("objgauss.core.lod", "attach_object_aware_lod_metadata"),
    "attach_quantization_metadata": ("objgauss.core.quantization", "attach_quantization_metadata"),
    "build_chunk_index": ("objgauss.core.chunk_index", "build_chunk_index"),
    "cluster_features": ("objgauss.core.clustering", "cluster_features"),
    "compare_baseline_candidates": ("objgauss.core.evaluation", "compare_baseline_candidates"),
    "filter_objects": ("objgauss.core.objects", "filter_objects"),
    "initialize_object_field": ("objgauss.core.object_field", "initialize_object_field"),
    "object_emergence_metrics": ("objgauss.core.evaluation", "object_emergence_metrics"),
    "project_points": ("objgauss.core.projection", "project_points"),
    "read_ply": ("objgauss.core.io", "read_ply"),
    "read_splat": ("objgauss.core.io", "read_splat"),
    "score_mask_manifest_with_clip": ("objgauss.core.semantics", "score_mask_manifest_with_clip"),
    "validate_mask_manifest": ("objgauss.core.masks", "validate_mask_manifest"),
    "vote_masks_to_gaussians": ("objgauss.core.projection", "vote_masks_to_gaussians"),
    "write_ogc_payload": ("objgauss.core.ogc_payload", "write_ogc_payload"),
    "write_ply": ("objgauss.core.io", "write_ply"),
    "write_quantized_ogc_payload": ("objgauss.core.quantization", "write_quantized_ogc_payload"),
    "write_splat": ("objgauss.core.io", "write_splat"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
