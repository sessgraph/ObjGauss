"""Deprecated compatibility imports for controlled dataset contracts.

The canonical implementation lives in :mod:`objgauss.datasets.controlled_schema`.
"""

# Deprecated compatibility wrapper; new production code must import from datasets.
from objgauss.datasets.controlled_schema import (
    OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SCHEMA,
    OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SUMMARY_SCHEMA,
    objectstate_controlled_dataset_contract_summary,
    validate_objectstate_controlled_dataset_contract_summary,
)

__all__ = [
    "OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SCHEMA",
    "OBJECTSTATE_CONTROLLED_DATASET_CONTRACT_SUMMARY_SCHEMA",
    "objectstate_controlled_dataset_contract_summary",
    "validate_objectstate_controlled_dataset_contract_summary",
]
