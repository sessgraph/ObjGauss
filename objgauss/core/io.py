"""Gaussian artifact IO entry points."""

from objgauss.core.io_ply import append_or_replace_property, read_ply, write_ply
from objgauss.core.io_splat import read_splat, write_splat

__all__ = [
    "append_or_replace_property",
    "read_ply",
    "read_splat",
    "write_ply",
    "write_splat",
]
