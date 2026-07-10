"""Compatibility import for projection, mask-voting, and quality APIs."""

from objgauss.core.projection import (
    MaskVoteResult,
    Projection,
    mask_vote_quality_audit,
    mask_vote_targets,
    project_points,
    projection_loss,
)
from objgauss.evaluation.mask_vote_quality import mask_vote_quality_check
from objgauss.pipelines.mask_voting import (
    MaskTrainingResult,
    depth_visibility_diagnostic,
    train_object_field_from_votes,
    training_summary,
    vote_masks_to_gaussians,
)

__all__ = (
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
)
