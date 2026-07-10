"""Deprecated compatibility import for ObjectState checkpoint evaluation."""

from objgauss.evaluation.object_state_eval import (
    OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA,
    evaluate_solver_decoder_object_states,
    validate_objectstate_checkpoint_eval,
)

__all__ = (
    "OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA",
    "evaluate_solver_decoder_object_states",
    "validate_objectstate_checkpoint_eval",
)
