const DEPTH_BIN_COUNT = 65_536;

function assertPositions(positions) {
  if (!(positions instanceof Float32Array) || positions.length === 0 || positions.length % 3 !== 0) {
    throw new TypeError("positions must be a non-empty Float32Array of xyz triples");
  }
}

function assertViewMatrix(viewMatrix) {
  if (viewMatrix == null || viewMatrix.length !== 16) {
    throw new TypeError("viewMatrix must contain exactly 16 values");
  }
  for (const value of viewMatrix) {
    if (!Number.isFinite(value)) {
      throw new TypeError("viewMatrix values must be finite");
    }
  }
}

export function cameraDepth(positionX, positionY, positionZ, viewMatrix) {
  return -(
    viewMatrix[2] * positionX
    + viewMatrix[6] * positionY
    + viewMatrix[10] * positionZ
    + viewMatrix[14]
  );
}

/**
 * Stable far-to-near ordering for premultiplied alpha blending.
 * Quantization keeps the work O(n) while preserving input order inside a bin.
 */
export function sortSplatIndices(positions, viewMatrix) {
  assertPositions(positions);
  assertViewMatrix(viewMatrix);

  const count = positions.length / 3;
  const depths = new Float32Array(count);
  let minDepth = Infinity;
  let maxDepth = -Infinity;

  for (let index = 0; index < count; index += 1) {
    const offset = index * 3;
    const depth = cameraDepth(
      positions[offset],
      positions[offset + 1],
      positions[offset + 2],
      viewMatrix,
    );
    depths[index] = depth;
    minDepth = Math.min(minDepth, depth);
    maxDepth = Math.max(maxDepth, depth);
  }

  const result = new Uint32Array(count);
  const range = maxDepth - minDepth;
  if (!Number.isFinite(range) || range <= 1e-8) {
    for (let index = 0; index < count; index += 1) {
      result[index] = index;
    }
    return result;
  }

  const bins = new Uint16Array(count);
  const counts = new Uint32Array(DEPTH_BIN_COUNT);
  const multiplier = (DEPTH_BIN_COUNT - 1) / range;
  for (let index = 0; index < count; index += 1) {
    const bin = Math.min(
      DEPTH_BIN_COUNT - 1,
      Math.max(0, Math.floor((depths[index] - minDepth) * multiplier)),
    );
    bins[index] = bin;
    counts[bin] += 1;
  }

  const offsets = new Uint32Array(DEPTH_BIN_COUNT);
  let cursor = 0;
  for (let bin = DEPTH_BIN_COUNT - 1; bin >= 0; bin -= 1) {
    offsets[bin] = cursor;
    cursor += counts[bin];
  }

  for (let index = 0; index < count; index += 1) {
    const bin = bins[index];
    result[offsets[bin]] = index;
    offsets[bin] += 1;
  }

  return result;
}
