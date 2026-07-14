const EPSILON = 1e-9;
const RIGID_TOLERANCE = 1e-8;

export class FrameMathError extends Error {
  constructor(code, message) {
    super(message);
    this.name = "FrameMathError";
    this.code = code;
  }
}

function fail(code, message) {
  throw new FrameMathError(code, message);
}

function assertFiniteArray(value, length, label) {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    fail("INVALID_SHAPE", `${label} must be an array`);
  }
  if (value.length !== length) {
    fail("INVALID_SHAPE", `${label} must contain ${length} values`);
  }
  if (![...value].every(Number.isFinite)) {
    fail("NON_FINITE", `${label} must contain only finite numbers`);
  }
}

export function quaternionNorm(quaternionWxyz) {
  assertFiniteArray(quaternionWxyz, 4, "quaternion wxyz");
  return Math.hypot(...quaternionWxyz);
}

export function normalizeQuaternionWxyz(quaternionWxyz) {
  const norm = quaternionNorm(quaternionWxyz);
  if (norm <= EPSILON) {
    fail("ZERO_QUATERNION", "quaternion norm must be greater than zero");
  }
  const normalized = [...quaternionWxyz].map((value) => value / norm);
  const firstNonZero = normalized.find((value) => Math.abs(value) > EPSILON) ?? 1;
  return firstNonZero < 0 ? normalized.map((value) => -value) : normalized;
}

export function quaternionToRotationMatrix(quaternionWxyz) {
  const [w, x, y, z] = normalizeQuaternionWxyz(quaternionWxyz);
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

export function rigidTransformFromQuaternionTranslation(quaternionWxyz, translation) {
  assertFiniteArray(translation, 3, "translation");
  const rotation = quaternionToRotationMatrix(quaternionWxyz);
  return [
    rotation[0], rotation[1], rotation[2], translation[0],
    rotation[3], rotation[4], rotation[5], translation[1],
    rotation[6], rotation[7], rotation[8], translation[2],
    0, 0, 0, 1,
  ];
}

function determinant3(matrix) {
  return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
    - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
    + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

export function assertRigidTransform(matrix, label = "transform") {
  assertFiniteArray(matrix, 16, label);
  if (Math.abs(matrix[12]) > RIGID_TOLERANCE
      || Math.abs(matrix[13]) > RIGID_TOLERANCE
      || Math.abs(matrix[14]) > RIGID_TOLERANCE
      || Math.abs(matrix[15] - 1) > RIGID_TOLERANCE) {
    fail("NON_RIGID_TRANSFORM", `${label} must end with [0, 0, 0, 1]`);
  }
  const rotation = [
    matrix[0], matrix[1], matrix[2],
    matrix[4], matrix[5], matrix[6],
    matrix[8], matrix[9], matrix[10],
  ];
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      const dot = rotation[row * 3] * rotation[column * 3]
        + rotation[row * 3 + 1] * rotation[column * 3 + 1]
        + rotation[row * 3 + 2] * rotation[column * 3 + 2];
      const expected = row === column ? 1 : 0;
      if (Math.abs(dot - expected) > RIGID_TOLERANCE) {
        fail("NON_RIGID_TRANSFORM", `${label} rotation must be orthonormal`);
      }
    }
  }
  if (Math.abs(determinant3(rotation) - 1) > RIGID_TOLERANCE) {
    fail("LEFT_HANDED_TRANSFORM", `${label} rotation determinant must equal +1`);
  }
  return matrix;
}

export function multiplyTransforms(T_AB, T_BC) {
  assertRigidTransform(T_AB, "T_AB");
  assertRigidTransform(T_BC, "T_BC");
  const result = new Array(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      for (let index = 0; index < 4; index += 1) {
        result[row * 4 + column] += T_AB[row * 4 + index] * T_BC[index * 4 + column];
      }
    }
  }
  return result;
}

export function invertRigidTransform(T_AB) {
  assertRigidTransform(T_AB, "T_AB");
  const tx = T_AB[3];
  const ty = T_AB[7];
  const tz = T_AB[11];
  const r00 = T_AB[0]; const r01 = T_AB[1]; const r02 = T_AB[2];
  const r10 = T_AB[4]; const r11 = T_AB[5]; const r12 = T_AB[6];
  const r20 = T_AB[8]; const r21 = T_AB[9]; const r22 = T_AB[10];
  return [
    r00, r10, r20, -(r00 * tx + r10 * ty + r20 * tz),
    r01, r11, r21, -(r01 * tx + r11 * ty + r21 * tz),
    r02, r12, r22, -(r02 * tx + r12 * ty + r22 * tz),
    0, 0, 0, 1,
  ];
}

export function transformPoint(T_AB, pointB) {
  assertRigidTransform(T_AB, "T_AB");
  assertFiniteArray(pointB, 3, "point_B");
  const [x, y, z] = pointB;
  return [
    T_AB[0] * x + T_AB[1] * y + T_AB[2] * z + T_AB[3],
    T_AB[4] * x + T_AB[5] * y + T_AB[6] * z + T_AB[7],
    T_AB[8] * x + T_AB[9] * y + T_AB[10] * z + T_AB[11],
  ];
}

export function assertIntrinsics(K, imageSizePx) {
  assertFiniteArray(K, 9, "K");
  assertFiniteArray(imageSizePx, 2, "image_size_px");
  if (!imageSizePx.every(Number.isInteger) || imageSizePx.some((value) => value <= 0)) {
    fail("INVALID_IMAGE_SIZE", "image_size_px must contain positive integers");
  }
  const determinant = determinant3(K);
  if (Math.abs(determinant) <= EPSILON) {
    fail("SINGULAR_INTRINSICS", "K must be invertible");
  }
  if (K[0] <= 0 || K[4] <= 0 || Math.abs(K[8] - 1) > RIGID_TOLERANCE) {
    fail("INVALID_INTRINSICS", "K must have positive focal lengths and K[2,2] = 1");
  }
}

export function projectWorldPoint({ T_WC, K, imageSizePx, pointW, requireInBounds = true }) {
  assertRigidTransform(T_WC, "T_WC");
  assertIntrinsics(K, imageSizePx);
  const pointC = transformPoint(invertRigidTransform(T_WC), pointW);
  if (pointC[2] <= EPSILON) {
    fail("BEHIND_CAMERA", "point must have positive OpenCV camera depth");
  }
  const u = (K[0] * pointC[0] + K[1] * pointC[1] + K[2] * pointC[2]) / pointC[2];
  const v = (K[3] * pointC[0] + K[4] * pointC[1] + K[5] * pointC[2]) / pointC[2];
  if (!Number.isFinite(u) || !Number.isFinite(v)) {
    fail("NON_FINITE_PROJECTION", "projected pixel must be finite");
  }
  if (requireInBounds && (u < 0 || v < 0 || u >= imageSizePx[0] || v >= imageSizePx[1])) {
    fail("OUT_OF_BOUNDS", "projected pixel must lie inside the image");
  }
  return { pixel: [u, v], depth_m: pointC[2], point_C_m: pointC };
}

function invertMatrix3(matrix) {
  const determinant = determinant3(matrix);
  if (Math.abs(determinant) <= EPSILON) {
    fail("SINGULAR_INTRINSICS", "K must be invertible");
  }
  return [
    (matrix[4] * matrix[8] - matrix[5] * matrix[7]) / determinant,
    (matrix[2] * matrix[7] - matrix[1] * matrix[8]) / determinant,
    (matrix[1] * matrix[5] - matrix[2] * matrix[4]) / determinant,
    (matrix[5] * matrix[6] - matrix[3] * matrix[8]) / determinant,
    (matrix[0] * matrix[8] - matrix[2] * matrix[6]) / determinant,
    (matrix[2] * matrix[3] - matrix[0] * matrix[5]) / determinant,
    (matrix[3] * matrix[7] - matrix[4] * matrix[6]) / determinant,
    (matrix[1] * matrix[6] - matrix[0] * matrix[7]) / determinant,
    (matrix[0] * matrix[4] - matrix[1] * matrix[3]) / determinant,
  ];
}

export function unprojectPixel({ T_WC, K, imageSizePx, pixel, depthM }) {
  assertRigidTransform(T_WC, "T_WC");
  assertIntrinsics(K, imageSizePx);
  assertFiniteArray(pixel, 2, "pixel");
  if (!Number.isFinite(depthM) || depthM <= EPSILON) {
    fail("INVALID_DEPTH", "depthM must be finite and greater than zero");
  }
  const inverseK = invertMatrix3(K);
  const homogeneous = [pixel[0] * depthM, pixel[1] * depthM, depthM];
  const pointC = [
    inverseK[0] * homogeneous[0] + inverseK[1] * homogeneous[1] + inverseK[2] * homogeneous[2],
    inverseK[3] * homogeneous[0] + inverseK[4] * homogeneous[1] + inverseK[5] * homogeneous[2],
    inverseK[6] * homogeneous[0] + inverseK[7] * homogeneous[1] + inverseK[8] * homogeneous[2],
  ];
  return transformPoint(T_WC, pointC);
}

export const FRAME_MATH_CONSTANTS = Object.freeze({
  convention: "T_AB · p_B = p_A",
  matrix_storage: "row-major",
  world: "right-handed,+Z-up,meter",
  camera: "+X-right,+Y-down,+Z-forward",
});
