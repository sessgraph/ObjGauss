"""Compatibility import for Object Field and former adjacent tooling."""

from objgauss.core.object_field import (
    ObjectField,
    ObjectFieldInit,
    ObjectFieldLabelDelta,
    ObjectFieldMetrics,
    attach_hard_labels,
    cloud_positions_for_metrics,
    field_from_labels,
    initialize_object_field,
    load_object_field,
    local_smoothness_loss,
    object_field_label_delta,
    object_field_metrics,
    save_object_field,
    softmax,
)
from objgauss.datasets.nerf_inspection import (
    NerfDatasetSummary,
    NerfSplitSummary,
    inspect_nerf_dataset,
)
from objgauss.pipelines.json_io import write_json

__all__ = (
    "NerfDatasetSummary",
    "NerfSplitSummary",
    "ObjectField",
    "ObjectFieldInit",
    "ObjectFieldLabelDelta",
    "ObjectFieldMetrics",
    "attach_hard_labels",
    "cloud_positions_for_metrics",
    "field_from_labels",
    "initialize_object_field",
    "inspect_nerf_dataset",
    "load_object_field",
    "local_smoothness_loss",
    "object_field_label_delta",
    "object_field_metrics",
    "save_object_field",
    "softmax",
    "write_json",
)
