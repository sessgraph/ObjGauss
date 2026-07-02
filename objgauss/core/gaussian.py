from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class GaussianCloud:
    """A structured vertex table loaded from a Gaussian PLY file."""

    vertices: np.ndarray
    comments: tuple[str, ...] = ()
    source_format: str = "binary_little_endian"

    def __post_init__(self) -> None:
        if self.vertices.dtype.names is None:
            raise TypeError("GaussianCloud vertices must be a structured NumPy array")

    @property
    def count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(self.vertices.dtype.names or ())

    def require_fields(self, names: Iterable[str]) -> None:
        missing = [name for name in names if name not in self.fields]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"PLY vertex data is missing required field(s): {joined}")

    def with_vertices(self, vertices: np.ndarray) -> "GaussianCloud":
        return replace(self, vertices=vertices)
