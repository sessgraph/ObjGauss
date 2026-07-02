"""Compatibility wrapper for OGC quantization metadata helpers."""

from objgauss.core.quantization import (
    DEQUANTIZED_RECORD_DTYPE,
    DEFAULT_QUANTIZATION_POLICY_ID,
    QUANTIZED_PAYLOAD_SCHEMA,
    QUANTIZED_RECORD_DTYPE,
    QUANTIZED_RECORD_FORMAT,
    QUANTIZATION_ESTIMATE_SCHEMA,
    QUANTIZATION_SCHEMA,
    QuantizedPayloadWriteResult,
    attach_quantization_metadata,
    dequantize_records,
    default_quantization_policy,
    estimate_quantized_payload_size,
    quantized_records_from_cloud,
    read_quantized_ogc_payload,
    write_quantized_ogc_payload,
)

__all__ = [
    "DEQUANTIZED_RECORD_DTYPE",
    "DEFAULT_QUANTIZATION_POLICY_ID",
    "QUANTIZED_PAYLOAD_SCHEMA",
    "QUANTIZED_RECORD_DTYPE",
    "QUANTIZED_RECORD_FORMAT",
    "QUANTIZATION_ESTIMATE_SCHEMA",
    "QUANTIZATION_SCHEMA",
    "QuantizedPayloadWriteResult",
    "attach_quantization_metadata",
    "dequantize_records",
    "default_quantization_policy",
    "estimate_quantized_payload_size",
    "quantized_records_from_cloud",
    "read_quantized_ogc_payload",
    "write_quantized_ogc_payload",
]
