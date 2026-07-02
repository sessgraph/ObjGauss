"""Training handoff and model-registration entry points."""

from objgauss.sample_bundle import SampleBundleResult, write_sample_bundle
from objgauss.training import TrainingOutputRegistration, register_training_output

__all__ = [
    "SampleBundleResult",
    "TrainingOutputRegistration",
    "register_training_output",
    "write_sample_bundle",
]
