"""Deprecated compatibility import for Gaussian decoder training."""

from objgauss.pipelines.gaussian_decoder_training import (
    OBJECT_STATE_GAUSSIAN_DECODER_STATE_SCHEMA,
    OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA,
    ObjectStateGaussianDecoderLoss,
    ObjectStateGaussianDecoderState,
    ObjectStateGaussianDecoderTrainingResult,
    initialize_object_state_gaussian_decoder,
    object_state_gaussian_decoder_state_from_dict,
    train_object_state_gaussian_decoder,
    validate_object_state_gaussian_decoder_state,
)

__all__ = (
    "OBJECT_STATE_GAUSSIAN_DECODER_STATE_SCHEMA",
    "OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA",
    "ObjectStateGaussianDecoderState",
    "ObjectStateGaussianDecoderLoss",
    "ObjectStateGaussianDecoderTrainingResult",
    "initialize_object_state_gaussian_decoder",
    "train_object_state_gaussian_decoder",
    "validate_object_state_gaussian_decoder_state",
    "object_state_gaussian_decoder_state_from_dict",
)
