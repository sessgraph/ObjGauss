"""Deprecated compatibility import for solver-decoder joint training."""

from objgauss.pipelines.solver_decoder_training import (
    SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA,
    SOLVER_DECODER_JOINT_TRAINING_SCHEMA,
    SolverDecoderJointLoss,
    SolverDecoderJointTrainingResult,
    solver_decoder_joint_checkpoint,
    solver_decoder_joint_states_from_dict,
    train_solver_decoder_joint,
    validate_solver_decoder_joint_checkpoint,
)

__all__ = (
    "SOLVER_DECODER_JOINT_TRAINING_SCHEMA",
    "SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA",
    "SolverDecoderJointLoss",
    "SolverDecoderJointTrainingResult",
    "train_solver_decoder_joint",
    "solver_decoder_joint_checkpoint",
    "validate_solver_decoder_joint_checkpoint",
    "solver_decoder_joint_states_from_dict",
)
