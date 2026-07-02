"""Compatibility wrapper for minimal ObjGauss chunked payload writing."""

from objgauss.core.ogc_payload import (
    OGC_PAYLOAD_SCHEMA,
    OGC_RECORD_DTYPE,
    OGC_RECORD_FORMAT,
    OgcPayloadWriteResult,
    read_ogc_payload,
    records_from_cloud,
    write_ogc_payload,
)

__all__ = [
    "OGC_PAYLOAD_SCHEMA",
    "OGC_RECORD_DTYPE",
    "OGC_RECORD_FORMAT",
    "OgcPayloadWriteResult",
    "read_ogc_payload",
    "records_from_cloud",
    "write_ogc_payload",
]
