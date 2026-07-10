"""Deprecated compatibility import for ObjectState identity encoder training."""

from objgauss.pipelines.objectstate_identity_encoder import (
    OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA,
    OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA,
    ObjectStateIdentityContrastiveLoss,
    ObjectStateIdentityEncoderConfig,
    ObjectStateIdentityEncoderState,
    ObjectStateIdentityEncoderTrainingResult,
    initialize_objectstate_identity_encoder_state,
    objectstate_identity_encoder_features,
    train_objectstate_identity_encoder,
    validate_objectstate_identity_encoder_config,
    validate_objectstate_identity_encoder_state,
    validate_objectstate_identity_encoder_training_summary,
)

__all__ = (
    "OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA",
    "OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA",
    "ObjectStateIdentityEncoderConfig",
    "ObjectStateIdentityEncoderState",
    "ObjectStateIdentityContrastiveLoss",
    "ObjectStateIdentityEncoderTrainingResult",
    "train_objectstate_identity_encoder",
    "initialize_objectstate_identity_encoder_state",
    "objectstate_identity_encoder_features",
    "validate_objectstate_identity_encoder_config",
    "validate_objectstate_identity_encoder_state",
    "validate_objectstate_identity_encoder_training_summary",
)
