import { colorForObject } from "./palette.js";

export const QUANTIZED_OGC_PAYLOAD_SCHEMA = "objgauss-ogc-quantized-payload-v0";
export const QUANTIZED_OGC_RECORD_FORMAT = "objgauss-ogc-quantized-record-v0";
export const QUANTIZED_OGC_RECORD_BYTE_SIZE = 10;

const DEFAULT_SCALE = 0.018;

export function decodeQuantizedOgcPayload(payloadBuffer, index, options = {}) {
  validateQuantizedOgcIndex(index);
  const chunks = selectChunks(index, options);
  const points = [];
  const decodedChunks = [];
  for (const chunk of chunks) {
    const result = decodeQuantizedOgcChunk(payloadBuffer, index, chunk, options);
    points.push(...result.points);
    decodedChunks.push(result.chunk);
  }
  return {
    points,
    chunks: decodedChunks,
    objects: Array.isArray(index.objects) ? index.objects : [],
    lod: index.lod ?? null,
    payload: index.payload,
    metadata: {
      schema: index.schema ?? "",
      sortKey: index.sort_key ?? "",
      gaussianCount: index.gaussian_count ?? points.length,
      objectCount: index.object_count ?? 0,
      decodedGaussians: points.length,
      decodedChunks: decodedChunks.length,
      recordFormat: index.payload.record_format,
    },
  };
}

export function quantizedOgcReadWindows(index, options = {}) {
  validateQuantizedOgcIndex(index);
  return selectChunks(index, options).map((chunk) => chunkReadWindow(chunk, options));
}

export function decodeQuantizedOgcPayloadWindows(payloadWindows, index, options = {}) {
  validateQuantizedOgcIndex(index);
  const windowsByChunk = new Map(
    payloadWindows.map((window) => [Number(window.chunkId), window]),
  );
  const chunks = selectChunks(index, options);
  const points = [];
  const decodedChunks = [];
  for (const chunk of chunks) {
    const window = chunkReadWindow(chunk, options);
    const payloadWindow = windowsByChunk.get(Number(chunk.chunk_id));
    if (!payloadWindow) {
      throw new Error(`OGC payload window for chunk ${chunk.chunk_id} is missing`);
    }
    const result = decodeQuantizedOgcChunk(payloadWindow.buffer, index, chunk, {
      ...options,
      payloadByteOffsetBase: Number(payloadWindow.byteOffset ?? window.byteOffset),
    });
    points.push(...result.points);
    decodedChunks.push(result.chunk);
  }
  return {
    points,
    chunks: decodedChunks,
    objects: Array.isArray(index.objects) ? index.objects : [],
    lod: index.lod ?? null,
    payload: index.payload,
    metadata: {
      schema: index.schema ?? "",
      sortKey: index.sort_key ?? "",
      gaussianCount: index.gaussian_count ?? points.length,
      objectCount: index.object_count ?? 0,
      decodedGaussians: points.length,
      decodedChunks: decodedChunks.length,
      decodedWindows: payloadWindows.length,
      recordFormat: index.payload.record_format,
    },
  };
}

export function decodeQuantizedOgcChunk(payloadBuffer, index, chunkOrId, options = {}) {
  validateQuantizedOgcIndex(index);
  const chunk = resolveChunk(index, chunkOrId);
  const window = chunkReadWindow(chunk, options);
  const byteOffsetBase = Number(options.payloadByteOffsetBase ?? 0);
  const localByteOffset = window.byteOffset - byteOffsetBase;
  validateChunkReadWindow({
    chunk,
    byteOffset: localByteOffset,
    byteLength: window.byteLength,
    recordCount: window.recordCount,
    payloadBuffer,
  });

  const view = new DataView(payloadBuffer);
  const points = [];
  const aabbMin = numericVec3(chunk.aabb_min, `chunk ${chunk.chunk_id} aabb_min`);
  const aabbMax = numericVec3(chunk.aabb_max, `chunk ${chunk.chunk_id} aabb_max`);
  const span = aabbMax.map((value, axis) => value - aabbMin[axis]);
  for (let row = 0; row < window.recordCount; row += 1) {
    const offset = localByteOffset + row * QUANTIZED_OGC_RECORD_BYTE_SIZE;
    const xq = view.getUint16(offset, true);
    const yq = view.getUint16(offset + 2, true);
    const zq = view.getUint16(offset + 4, true);
    const red = view.getUint8(offset + 6);
    const green = view.getUint8(offset + 7);
    const blue = view.getUint8(offset + 8);
    const opacity = view.getUint8(offset + 9) / 255;
    const objectId = Number(chunk.object_id);
    points.push({
      x: dequantizeUint16(xq, aabbMin[0], span[0]),
      y: dequantizeUint16(yq, aabbMin[1], span[1]),
      z: dequantizeUint16(zq, aabbMin[2], span[2]),
      opacity,
      scale: [DEFAULT_SCALE, DEFAULT_SCALE],
      scale3: [DEFAULT_SCALE, DEFAULT_SCALE, DEFAULT_SCALE],
      rotation: 0,
      rotationQuaternion: null,
      objectId,
      color: [red, green, blue],
      colorSource: "quantized-ogc-rgb",
      shDc: null,
      shRestCoefficientCount: 0,
      shDegree: 0,
      objectColor: colorForObject(objectId),
      chunkId: chunk.chunk_id,
      lodLevel: window.lodLevel,
    });
  }
  return {
    points,
    chunk: {
      chunkId: chunk.chunk_id,
      objectId: chunk.object_id,
      recordCount: window.recordCount,
      byteOffset: window.byteOffset,
      byteLength: window.byteLength,
      lod: chunk.lod ?? null,
      aabbMin,
      aabbMax,
    },
  };
}

export function validateQuantizedOgcIndex(index) {
  if (!index || typeof index !== "object") {
    throw new Error("OGC index must be an object");
  }
  const payload = index.payload;
  if (!payload || typeof payload !== "object") {
    throw new Error("OGC index payload metadata is required");
  }
  if (payload.schema !== QUANTIZED_OGC_PAYLOAD_SCHEMA) {
    throw new Error(`Unsupported OGC payload schema: ${payload.schema}`);
  }
  if (payload.record_format !== QUANTIZED_OGC_RECORD_FORMAT) {
    throw new Error(`Unsupported OGC record format: ${payload.record_format}`);
  }
  if (payload.record_byte_size !== QUANTIZED_OGC_RECORD_BYTE_SIZE) {
    throw new Error(`Unsupported OGC record byte size: ${payload.record_byte_size}`);
  }
  if (!Array.isArray(index.chunks) || index.chunks.length === 0) {
    throw new Error("OGC index chunks must be a non-empty array");
  }
  for (const chunk of index.chunks) {
    validateChunkMetadata(chunk);
  }
  return true;
}

function chunkReadWindow(chunk, options) {
  const level = resolveChunkLodLevel(chunk, options.lodLevel);
  const byteOffset = Number(level?.byte_offset ?? chunk.byte_offset);
  const byteLength = Number(level?.byte_length ?? chunk.byte_length);
  const recordCount = Number(level?.record_count ?? level?.gaussian_count ?? chunk.record_count);
  validateReadWindowShape({ chunk, byteOffset, byteLength, recordCount });
  return {
    chunkId: Number(chunk.chunk_id),
    objectId: Number(chunk.object_id),
    byteOffset,
    byteLength,
    byteEnd: byteOffset + Math.max(0, byteLength - 1),
    recordCount,
    lodLevel: level?.level ?? null,
  };
}

function selectChunks(index, options) {
  if (Array.isArray(options.chunkIds) && options.chunkIds.length > 0) {
    const requested = new Set(options.chunkIds.map(Number));
    return index.chunks.filter((chunk) => requested.has(Number(chunk.chunk_id)));
  }
  return index.chunks;
}

function resolveChunk(index, chunkOrId) {
  if (typeof chunkOrId === "object" && chunkOrId !== null) return chunkOrId;
  const chunkId = Number(chunkOrId);
  const chunk = index.chunks.find((entry) => Number(entry.chunk_id) === chunkId);
  if (!chunk) {
    throw new Error(`OGC chunk ${chunkOrId} not found`);
  }
  return chunk;
}

function resolveChunkLodLevel(chunk, lodLevel) {
  if (lodLevel === undefined || lodLevel === null) return null;
  const levels = Array.isArray(chunk.lod?.levels) ? chunk.lod.levels : [];
  const level = levels.find((entry) => Number(entry.level) === Number(lodLevel));
  if (!level) {
    throw new Error(`OGC chunk ${chunk.chunk_id} does not include LOD level ${lodLevel}`);
  }
  return level;
}

function validateChunkMetadata(chunk) {
  if (!Number.isInteger(chunk?.chunk_id) || chunk.chunk_id < 0) {
    throw new Error("OGC chunk_id must be a non-negative integer");
  }
  if (!Number.isInteger(chunk.object_id)) {
    throw new Error(`OGC chunk ${chunk.chunk_id} object_id must be an integer`);
  }
  if (!Number.isInteger(chunk.byte_offset) || chunk.byte_offset < 0) {
    throw new Error(`OGC chunk ${chunk.chunk_id} byte_offset must be a non-negative integer`);
  }
  if (!Number.isInteger(chunk.byte_length) || chunk.byte_length < 0) {
    throw new Error(`OGC chunk ${chunk.chunk_id} byte_length must be a non-negative integer`);
  }
  if (!Number.isInteger(chunk.record_count) || chunk.record_count < 0) {
    throw new Error(`OGC chunk ${chunk.chunk_id} record_count must be a non-negative integer`);
  }
  if (chunk.record_format !== QUANTIZED_OGC_RECORD_FORMAT) {
    throw new Error(`OGC chunk ${chunk.chunk_id} has unsupported record_format ${chunk.record_format}`);
  }
  numericVec3(chunk.aabb_min, `chunk ${chunk.chunk_id} aabb_min`);
  numericVec3(chunk.aabb_max, `chunk ${chunk.chunk_id} aabb_max`);
}

function validateReadWindowShape({ chunk, byteOffset, byteLength, recordCount }) {
  if (!Number.isInteger(byteOffset) || byteOffset < 0) {
    throw new Error(`OGC chunk ${chunk.chunk_id} read byte_offset is invalid`);
  }
  if (!Number.isInteger(byteLength) || byteLength < 0) {
    throw new Error(`OGC chunk ${chunk.chunk_id} read byte_length is invalid`);
  }
  if (!Number.isInteger(recordCount) || recordCount < 0) {
    throw new Error(`OGC chunk ${chunk.chunk_id} read record_count is invalid`);
  }
  if (byteLength !== recordCount * QUANTIZED_OGC_RECORD_BYTE_SIZE) {
    throw new Error(`OGC chunk ${chunk.chunk_id} byte_length does not match record_count`);
  }
}

function validateChunkReadWindow({ chunk, byteOffset, byteLength, recordCount, payloadBuffer }) {
  validateReadWindowShape({ chunk, byteOffset, byteLength, recordCount });
  if (byteOffset + byteLength > payloadBuffer.byteLength) {
    throw new Error(`OGC chunk ${chunk.chunk_id} read window exceeds payload byteLength`);
  }
}

function numericVec3(value, label) {
  if (!Array.isArray(value) || value.length !== 3) {
    throw new Error(`OGC ${label} must be a 3-number array`);
  }
  return value.map((entry) => {
    const number = Number(entry);
    if (!Number.isFinite(number)) {
      throw new Error(`OGC ${label} must contain finite numbers`);
    }
    return number;
  });
}

function dequantizeUint16(value, min, span) {
  return min + (Number(value) / 65535) * span;
}
