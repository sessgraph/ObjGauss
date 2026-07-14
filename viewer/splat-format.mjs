export const SPLAT_RECORD_BYTES = 32;
// Stage-0 deliberately caps local files at 1.5 million 32-byte records.
export const MAX_SPLAT_BYTES = 48_000_000;
export const MAX_SPLAT_DISPLAY_MULTIPLIER = 32;
export const MAX_SPLAT_SCALE = 10;
export const MAX_ABS_POSITION = 1_000_000;
export const SPLAT_FORMAT = "antimatter15-splat-v1";

function finiteNumber(value, label) {
  if (!Number.isFinite(value)) {
    throw new TypeError(`${label} must be finite`);
  }
  return value;
}

function asBytes(input) {
  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }

  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }

  throw new TypeError(".splat input must be an ArrayBuffer or ArrayBuffer view");
}

export function assertSplatByteLength(byteLength) {
  if (!Number.isSafeInteger(byteLength) || byteLength <= 0) {
    throw new RangeError(".splat input must contain at least one 32-byte record");
  }
  if (byteLength > MAX_SPLAT_BYTES) {
    throw new RangeError(`.splat input exceeds the ${MAX_SPLAT_BYTES}-byte hard limit`);
  }
  if (byteLength % SPLAT_RECORD_BYTES !== 0) {
    throw new RangeError(".splat byte length must be an exact multiple of 32");
  }
  return byteLength / SPLAT_RECORD_BYTES;
}

/**
 * Decode antimatter15 v1 quaternion bytes. The on-disk component order is
 * w, x, y, z and each byte represents (component * 128) + 128.
 */
export function decodeQuaternionBytes(wByte, xByte, yByte, zByte) {
  const components = [wByte, xByte, yByte, zByte].map((value, index) => {
    if (!Number.isInteger(value) || value < 0 || value > 255) {
      throw new RangeError(`quaternion byte ${index} must be an integer in [0, 255]`);
    }
    return (value - 128) / 128;
  });

  const norm = Math.hypot(...components);
  if (!Number.isFinite(norm) || norm === 0) {
    throw new RangeError("quaternion norm must be non-zero");
  }

  return Float32Array.from(components, (value) => value / norm);
}

/**
 * Return the symmetric covariance R diag(scale^2) R^T as
 * [xx, xy, xz, yy, yz, zz]. Quaternion order is w, x, y, z.
 */
export function covarianceFromScaleQuaternion(scale, quaternion) {
  if (scale == null || scale.length !== 3) {
    throw new TypeError("scale must contain exactly three components");
  }
  if (quaternion == null || quaternion.length !== 4) {
    throw new TypeError("quaternion must contain exactly four wxyz components");
  }

  const sx = finiteNumber(Number(scale[0]), "scale[0]");
  const sy = finiteNumber(Number(scale[1]), "scale[1]");
  const sz = finiteNumber(Number(scale[2]), "scale[2]");
  if (
    sx <= 0 || sy <= 0 || sz <= 0
    || sx > MAX_SPLAT_SCALE || sy > MAX_SPLAT_SCALE || sz > MAX_SPLAT_SCALE
  ) {
    throw new RangeError(`scale components must be greater than zero and at most ${MAX_SPLAT_SCALE}`);
  }
  if ([sx, sy, sz].some((value) => {
    const floatVariance = Math.fround(value * value);
    return !Number.isFinite(floatVariance) || floatVariance === 0;
  })) {
    throw new RangeError("scale components must produce finite non-zero float32 covariance");
  }

  let w = finiteNumber(Number(quaternion[0]), "quaternion[0]");
  let x = finiteNumber(Number(quaternion[1]), "quaternion[1]");
  let y = finiteNumber(Number(quaternion[2]), "quaternion[2]");
  let z = finiteNumber(Number(quaternion[3]), "quaternion[3]");
  const norm = Math.hypot(w, x, y, z);
  if (norm === 0) {
    throw new RangeError("quaternion norm must be non-zero");
  }
  w /= norm;
  x /= norm;
  y /= norm;
  z /= norm;

  const r00 = 1 - 2 * (y * y + z * z);
  const r01 = 2 * (x * y - z * w);
  const r02 = 2 * (x * z + y * w);
  const r10 = 2 * (x * y + z * w);
  const r11 = 1 - 2 * (x * x + z * z);
  const r12 = 2 * (y * z - x * w);
  const r20 = 2 * (x * z - y * w);
  const r21 = 2 * (y * z + x * w);
  const r22 = 1 - 2 * (x * x + y * y);

  const xx = sx * sx;
  const yy = sy * sy;
  const zz = sz * sz;
  const covariance = [
    r00 * r00 * xx + r01 * r01 * yy + r02 * r02 * zz,
    r00 * r10 * xx + r01 * r11 * yy + r02 * r12 * zz,
    r00 * r20 * xx + r01 * r21 * yy + r02 * r22 * zz,
    r10 * r10 * xx + r11 * r11 * yy + r12 * r12 * zz,
    r10 * r20 * xx + r11 * r21 * yy + r12 * r22 * zz,
    r20 * r20 * xx + r21 * r21 * yy + r22 * r22 * zz,
  ];

  if (!covariance.every(Number.isFinite)) {
    throw new RangeError("scale and quaternion produce a non-finite covariance");
  }
  const floatCovariance = Float32Array.from(covariance);
  if (!floatCovariance.every(Number.isFinite)) {
    throw new RangeError("scale and quaternion exceed the float32 covariance range");
  }
  return floatCovariance;
}

export function parseSplatV1(input) {
  const bytes = asBytes(input);
  const count = assertSplatByteLength(bytes.byteLength);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const colors = new Uint8Array(count * 4);
  const quaternions = new Float32Array(count * 4);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  let visibleCount = 0;

  for (let index = 0; index < count; index += 1) {
    const recordOffset = index * SPLAT_RECORD_BYTES;
    const vectorOffset = index * 3;
    for (let component = 0; component < 3; component += 1) {
      const position = view.getFloat32(recordOffset + component * 4, true);
      if (!Number.isFinite(position)) {
        throw new RangeError(`record ${index} position[${component}] must be finite`);
      }
      if (Math.abs(position) > MAX_ABS_POSITION) {
        throw new RangeError(
          `record ${index} position[${component}] exceeds the GPU-safe absolute limit ${MAX_ABS_POSITION}`,
        );
      }
      positions[vectorOffset + component] = position;
      min[component] = Math.min(min[component], position);
      max[component] = Math.max(max[component], position);

      const scale = view.getFloat32(recordOffset + 12 + component * 4, true);
      if (!Number.isFinite(scale) || scale <= 0 || scale > MAX_SPLAT_SCALE) {
        throw new RangeError(
          `record ${index} scale[${component}] must be finite, greater than zero, and at most ${MAX_SPLAT_SCALE}`,
        );
      }
      const floatVariance = Math.fround(scale * scale);
      if (!Number.isFinite(floatVariance) || floatVariance === 0) {
        throw new RangeError(
          `record ${index} scale[${component}] must produce finite non-zero float32 covariance`,
        );
      }
      scales[vectorOffset + component] = scale;
    }

    const colorOffset = index * 4;
    for (let component = 0; component < 4; component += 1) {
      colors[colorOffset + component] = view.getUint8(recordOffset + 24 + component);
    }
    if (colors[colorOffset + 3] > 0) {
      visibleCount += 1;
    }

    let quaternion;
    try {
      quaternion = decodeQuaternionBytes(
        view.getUint8(recordOffset + 28),
        view.getUint8(recordOffset + 29),
        view.getUint8(recordOffset + 30),
        view.getUint8(recordOffset + 31),
      );
    } catch (error) {
      throw new RangeError(`record ${index} has an invalid quaternion: ${error.message}`, { cause: error });
    }
    quaternions.set(quaternion, index * 4);
    const covariance = covarianceFromScaleQuaternion(
      scales.subarray(vectorOffset, vectorOffset + 3),
      quaternion,
    );
    if (!covariance.every((value) => Number.isFinite(
      Math.fround(value * MAX_SPLAT_DISPLAY_MULTIPLIER * MAX_SPLAT_DISPLAY_MULTIPLIER),
    ))) {
      throw new RangeError(
        `record ${index} covariance exceeds the GPU-safe range at ${MAX_SPLAT_DISPLAY_MULTIPLIER}× display scale`,
      );
    }
  }

  if (visibleCount === 0) {
    throw new RangeError(".splat input must contain at least one record with alpha greater than zero");
  }

  const center = min.map((value, index) => (value + max[index]) / 2);
  let radiusSquared = 0;
  for (let index = 0; index < count; index += 1) {
    const offset = index * 3;
    const dx = positions[offset] - center[0];
    const dy = positions[offset + 1] - center[1];
    const dz = positions[offset + 2] - center[2];
    radiusSquared = Math.max(radiusSquared, dx * dx + dy * dy + dz * dz);
  }

  return {
    format: SPLAT_FORMAT,
    byteLength: bytes.byteLength,
    count,
    visibleCount,
    positions,
    scales,
    colors,
    quaternions,
    bounds: {
      min: Float32Array.from(min),
      max: Float32Array.from(max),
      center: Float32Array.from(center),
      radius: Math.sqrt(radiusSquared),
    },
  };
}

export const parseSplat = parseSplatV1;
