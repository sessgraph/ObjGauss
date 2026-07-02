"""Compatibility wrapper for core projection and mask-voting algorithms."""

from objgauss.core.projection import (
    MaskTrainingResult,
    MaskVoteResult,
    Projection,
    depth_visibility_diagnostic,
    mask_vote_quality_audit,
    mask_vote_quality_check,
    mask_vote_targets,
    project_points,
    projection_loss,
    train_object_field_from_votes,
    training_summary,
    vote_masks_to_gaussians,
)

__all__ = [
    "MaskTrainingResult",
    "MaskVoteResult",
    "Projection",
    "depth_visibility_diagnostic",
    "mask_vote_quality_audit",
    "mask_vote_quality_check",
    "mask_vote_targets",
    "project_points",
    "projection_loss",
    "train_object_field_from_votes",
    "training_summary",
    "vote_masks_to_gaussians",
]
