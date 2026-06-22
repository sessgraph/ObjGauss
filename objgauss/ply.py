from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from objgauss.gaussians import GaussianCloud


_PLY_TO_DTYPE = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}

_KIND_SIZE_TO_PLY = {
    ("i", 1): "char",
    ("u", 1): "uchar",
    ("i", 2): "short",
    ("u", 2): "ushort",
    ("i", 4): "int",
    ("u", 4): "uint",
    ("f", 4): "float",
    ("f", 8): "double",
}


@dataclass(frozen=True)
class _Property:
    name: str
    ply_type: str


@dataclass(frozen=True)
class _Header:
    fmt: str
    comments: tuple[str, ...]
    vertex_count: int
    vertex_properties: tuple[_Property, ...]


def read_ply(path: str | Path) -> GaussianCloud:
    """Read scalar vertex properties from a Gaussian PLY file."""

    path = Path(path)
    with path.open("rb") as file:
        header = _read_header(file)
        dtype = _vertex_dtype(header, byte_order=_byte_order(header.fmt))

        if header.fmt == "ascii":
            vertices = _read_ascii_vertices(file, dtype, header.vertex_count)
        elif header.fmt in {"binary_little_endian", "binary_big_endian"}:
            vertices = np.fromfile(file, dtype=dtype, count=header.vertex_count)
            vertices = vertices.astype(_native_dtype(dtype), copy=False)
        else:
            raise ValueError(f"Unsupported PLY format: {header.fmt}")

    return GaussianCloud(
        vertices=vertices,
        comments=header.comments,
        source_format=header.fmt,
    )


def write_ply(
    path: str | Path,
    cloud: GaussianCloud,
    *,
    fmt: str | None = None,
    comments: tuple[str, ...] | None = None,
) -> None:
    """Write a Gaussian cloud as a scalar-property vertex PLY file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = fmt or cloud.source_format
    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise ValueError(f"Unsupported PLY output format: {fmt}")

    comments = cloud.comments if comments is None else comments
    header = _format_header(fmt, comments, cloud.vertices)

    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        if fmt == "ascii":
            _write_ascii_vertices(file, cloud.vertices)
        else:
            dtype = _with_byte_order(cloud.vertices.dtype, _byte_order(fmt))
            cloud.vertices.astype(dtype, copy=False).tofile(file)


def append_or_replace_property(
    vertices: np.ndarray,
    name: str,
    values: np.ndarray,
    dtype: np.dtype | str,
) -> np.ndarray:
    """Return a structured array with one property appended or replaced."""

    if vertices.dtype.names is None:
        raise TypeError("vertices must be a structured NumPy array")

    dtype = np.dtype(dtype)
    values = np.asarray(values, dtype=dtype)
    if values.shape[0] != vertices.shape[0]:
        raise ValueError(
            f"property {name!r} has {values.shape[0]} values for "
            f"{vertices.shape[0]} vertices"
        )

    old_names = list(vertices.dtype.names)
    new_descr = [
        (field, dtype if field == name else vertices.dtype.fields[field][0])
        for field in old_names
    ]
    if name not in old_names:
        new_descr.append((name, dtype))

    output = np.empty(vertices.shape, dtype=np.dtype(new_descr))
    for field in output.dtype.names or ():
        output[field] = values if field == name else vertices[field]
    return output


def _read_header(file: BinaryIO) -> _Header:
    first = file.readline().decode("ascii").strip()
    if first != "ply":
        raise ValueError("Not a PLY file")

    fmt: str | None = None
    comments: list[str] = []
    current_element: str | None = None
    vertex_count: int | None = None
    vertex_properties: list[_Property] = []

    while True:
        raw = file.readline()
        if not raw:
            raise ValueError("Unexpected end of file before PLY end_header")
        line = raw.decode("ascii").strip()
        if line == "end_header":
            break
        if not line:
            continue
        parts = line.split()
        tag = parts[0]

        if tag == "format":
            if len(parts) < 3:
                raise ValueError("Malformed PLY format line")
            fmt = parts[1]
        elif tag == "comment":
            comments.append(line[len("comment") :].strip())
        elif tag == "element":
            if len(parts) != 3:
                raise ValueError(f"Malformed PLY element line: {line}")
            current_element = parts[1]
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif tag == "property" and current_element == "vertex":
            if len(parts) != 3:
                raise ValueError(
                    "ObjGauss PLY IO supports scalar vertex properties only"
                )
            ply_type, name = parts[1], parts[2]
            if ply_type not in _PLY_TO_DTYPE:
                raise ValueError(f"Unsupported PLY property type: {ply_type}")
            vertex_properties.append(_Property(name=name, ply_type=ply_type))

    if fmt is None:
        raise ValueError("PLY header does not declare a format")
    if vertex_count is None:
        raise ValueError("PLY header does not include a vertex element")
    if not vertex_properties:
        raise ValueError("PLY vertex element has no scalar properties")
    return _Header(fmt, tuple(comments), vertex_count, tuple(vertex_properties))


def _vertex_dtype(header: _Header, byte_order: str) -> np.dtype:
    fields = [
        (prop.name, np.dtype(byte_order + _PLY_TO_DTYPE[prop.ply_type]))
        for prop in header.vertex_properties
    ]
    return np.dtype(fields)


def _read_ascii_vertices(
    file: BinaryIO,
    dtype: np.dtype,
    vertex_count: int,
) -> np.ndarray:
    rows = []
    names = dtype.names or ()
    for row_index in range(vertex_count):
        line = file.readline().decode("ascii")
        if not line:
            raise ValueError(f"Missing vertex row {row_index}")
        parts = line.split()
        if len(parts) < len(names):
            raise ValueError(f"Vertex row {row_index} has too few values")
        rows.append(tuple(_parse_ascii_value(parts[i], dtype[name]) for i, name in enumerate(names)))
    return np.array(rows, dtype=_native_dtype(dtype))


def _parse_ascii_value(value: str, dtype: np.dtype) -> int | float:
    if dtype.kind in {"i", "u"}:
        return int(value)
    if dtype.kind == "f":
        return float(value)
    raise ValueError(f"Unsupported dtype for ASCII parse: {dtype}")


def _write_ascii_vertices(file: BinaryIO, vertices: np.ndarray) -> None:
    names = vertices.dtype.names or ()
    for row in vertices:
        values = [_format_ascii_value(row[name], vertices.dtype.fields[name][0]) for name in names]
        file.write((" ".join(values) + "\n").encode("ascii"))


def _format_ascii_value(value: np.generic, dtype: np.dtype) -> str:
    if dtype.kind in {"i", "u"}:
        return str(int(value))
    if dtype.kind == "f":
        return f"{float(value):.9g}"
    raise ValueError(f"Unsupported dtype for ASCII write: {dtype}")


def _format_header(fmt: str, comments: tuple[str, ...], vertices: np.ndarray) -> str:
    lines = ["ply", f"format {fmt} 1.0"]
    for comment in comments:
        lines.append(f"comment {comment}")
    lines.append(f"element vertex {vertices.shape[0]}")
    for name in vertices.dtype.names or ():
        lines.append(f"property {_ply_type(vertices.dtype.fields[name][0])} {name}")
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def _ply_type(dtype: np.dtype) -> str:
    base = np.dtype(dtype).newbyteorder("=")
    key = (base.kind, base.itemsize)
    if key not in _KIND_SIZE_TO_PLY:
        raise ValueError(f"Unsupported dtype for PLY output: {dtype}")
    return _KIND_SIZE_TO_PLY[key]


def _byte_order(fmt: str) -> str:
    if fmt == "binary_big_endian":
        return ">"
    return "<"


def _native_dtype(dtype: np.dtype) -> np.dtype:
    return dtype.newbyteorder("=")


def _with_byte_order(dtype: np.dtype, byte_order: str) -> np.dtype:
    fields = []
    for name in dtype.names or ():
        field_dtype = dtype.fields[name][0]
        if field_dtype.itemsize == 1:
            fields.append((name, field_dtype.newbyteorder("|")))
        else:
            fields.append((name, field_dtype.newbyteorder(byte_order)))
    return np.dtype(fields)
