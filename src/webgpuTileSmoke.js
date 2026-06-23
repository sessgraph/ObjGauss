export const WEBGPU_TILE_SMOKE_LAYOUT_VERSION = "webgpu-tile-smoke-v1";
export const WEBGPU_TILE_RESOLVE_VERSION = "webgpu-tile-resolve-v1";
export const WEBGPU_TILE_SIZE = 16;
export const WEBGPU_TILE_MAX_ENTRIES = 8192;
export const WEBGPU_TILE_VIEWPORT = Object.freeze({ width: 1024, height: 1024 });
const RESOLVE_ALPHA_SCALE = 0.18;
const RESOLVE_ALPHA_GAIN = 0.78;
const RESOLVE_KERNEL_CUTOFF = 13;

export function buildWebGpuTileSmoke({
  points,
  visibleIds,
  removedIds,
  isolatedId,
  renderMode,
  pointSize,
  viewportWidth = WEBGPU_TILE_VIEWPORT.width,
  viewportHeight = WEBGPU_TILE_VIEWPORT.height,
  tileSize = WEBGPU_TILE_SIZE,
  maxEntriesPerTile = WEBGPU_TILE_MAX_ENTRIES,
  includeTileEntries = false,
}) {
  const objectIndex = buildObjectIndex(points);
  const objectState = buildObjectState({
    objectIdsByIndex: objectIndex.objectIdsByIndex,
    visibleIds,
    removedIds,
    isolatedId,
  });
  const bounds = sceneBounds(points);
  const tileColumns = Math.max(1, Math.ceil(viewportWidth / tileSize));
  const tileRows = Math.max(1, Math.ceil(viewportHeight / tileSize));
  const tileCount = tileColumns * tileRows;
  const tileCounts = new Uint32Array(tileCount);
  const tileCenters = buildTileCenters({ tileColumns, tileRows, tileSize });
  const tileAccumulation = new Float32Array(tileCount * 4);
  const tileEntries = includeTileEntries
    ? new Uint32Array(tileCount * maxEntriesPerTile)
    : null;

  const positionRadius = new Float32Array(points.length * 4);
  const colorOpacity = new Float32Array(points.length * 4);
  const scaleRotation = new Float32Array(points.length * 4);
  const objectIndices = new Uint32Array(points.length);

  let visibleGaussians = 0;
  let binnedGaussians = 0;
  let tileReferenceCount = 0;
  let tileOverflowCount = 0;
  let maxTileOccupancy = 0;

  points.forEach((point, index) => {
    const objectDenseIndex = objectIndex.objectIndexById.get(point.objectId) ?? 0;
    const scale = pointScale(point);
    const radiusPixels = pointRadiusPixels({
      point,
      scale,
      bounds,
      viewportWidth,
      viewportHeight,
      pointSize,
    });
    packGaussian({
      point,
      index,
      objectDenseIndex,
      radiusPixels,
      scale,
      renderMode,
      positionRadius,
      colorOpacity,
      scaleRotation,
      objectIndices,
    });

    if (objectState[objectDenseIndex] !== 1) return;
    visibleGaussians += 1;

    const screen = projectPointToSmokeViewport({ point, bounds, viewportWidth, viewportHeight });
    const minTileX = clampInt(Math.floor((screen.x - radiusPixels) / tileSize), 0, tileColumns - 1);
    const maxTileX = clampInt(Math.floor((screen.x + radiusPixels) / tileSize), 0, tileColumns - 1);
    const minTileY = clampInt(Math.floor((screen.y - radiusPixels) / tileSize), 0, tileRows - 1);
    const maxTileY = clampInt(Math.floor((screen.y + radiusPixels) / tileSize), 0, tileRows - 1);

    binnedGaussians += 1;
    for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
      for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
        const tileIndex = tileY * tileColumns + tileX;
        const occupancy = tileCounts[tileIndex];
        if (tileEntries && occupancy < maxEntriesPerTile) {
          tileEntries[tileIndex * maxEntriesPerTile + occupancy] = index;
        } else if (occupancy >= maxEntriesPerTile) {
          tileOverflowCount += 1;
        }
        const nextOccupancy = occupancy + 1;
        tileCounts[tileIndex] = nextOccupancy;
        tileReferenceCount += 1;
        if (nextOccupancy > maxTileOccupancy) {
          maxTileOccupancy = nextOccupancy;
        }
        accumulateTileResolve({
          tileIndex,
          tileCenters,
          screen,
          radiusPixels,
          gaussianIndex: index,
          colorOpacity,
          tileAccumulation,
        });
      }
    }
  });

  const activeTileCount = countActiveTiles(tileCounts);
  const resolve = resolveTileAccumulation(tileAccumulation);
  return {
    layoutVersion: WEBGPU_TILE_SMOKE_LAYOUT_VERSION,
    resolveVersion: WEBGPU_TILE_RESOLVE_VERSION,
    resolveMode: "tile-center-weighted-oit",
    tileSize,
    viewportWidth,
    viewportHeight,
    tileColumns,
    tileRows,
    tileCount,
    maxEntriesPerTile,
    packedGaussians: points.length,
    visibleGaussians,
    binnedGaussians,
    activeTileCount,
    tileReferenceCount,
    tileOverflowCount,
    maxTileOccupancy,
    resolvedTileCount: resolve.resolvedTileCount,
    resolveWeightSum: resolve.resolveWeightSum,
    resolveAlphaMean: resolve.resolveAlphaMean,
    resolveLumaMean: resolve.resolveLumaMean,
    resolveChecksum: resolve.resolveChecksum,
    tileEntryCapacity: tileCount * maxEntriesPerTile,
    objectCount: objectIndex.objectIdsByIndex.length,
    buffers: {
      positionRadius,
      colorOpacity,
      scaleRotation,
      objectIndices,
      objectState,
      tileCounts,
      tileAccumulation,
      tileResolvedRgba: resolve.tileResolvedRgba,
      tileEntries,
    },
  };
}

function buildTileCenters({ tileColumns, tileRows, tileSize }) {
  const centersX = new Float32Array(tileColumns);
  const centersY = new Float32Array(tileRows);
  for (let tileX = 0; tileX < tileColumns; tileX += 1) {
    centersX[tileX] = tileX * tileSize + tileSize * 0.5;
  }
  for (let tileY = 0; tileY < tileRows; tileY += 1) {
    centersY[tileY] = tileY * tileSize + tileSize * 0.5;
  }
  return { centersX, centersY, tileColumns };
}

function accumulateTileResolve({
  tileIndex,
  tileCenters,
  screen,
  radiusPixels,
  gaussianIndex,
  colorOpacity,
  tileAccumulation,
}) {
  const tileX = tileIndex % tileCenters.tileColumns;
  const tileY = Math.floor(tileIndex / tileCenters.tileColumns);
  const dx = tileCenters.centersX[tileX] - screen.x;
  const dy = tileCenters.centersY[tileY] - screen.y;
  const sigma = Math.max(radiusPixels / 3, 0.0001);
  const d = (dx * dx + dy * dy) / (sigma * sigma);
  if (d > RESOLVE_KERNEL_CUTOFF) return;

  const gaussianOffset = gaussianIndex * 4;
  const weight =
    Math.exp(-0.5 * d) *
    colorOpacity[gaussianOffset + 3] *
    RESOLVE_ALPHA_GAIN;
  if (weight <= 0.0001) return;

  const tileOffset = tileIndex * 4;
  tileAccumulation[tileOffset] += colorOpacity[gaussianOffset] * weight;
  tileAccumulation[tileOffset + 1] += colorOpacity[gaussianOffset + 1] * weight;
  tileAccumulation[tileOffset + 2] += colorOpacity[gaussianOffset + 2] * weight;
  tileAccumulation[tileOffset + 3] += weight;
}

function resolveTileAccumulation(tileAccumulation) {
  const tileResolvedRgba = new Float32Array(tileAccumulation.length);
  let resolvedTileCount = 0;
  let resolveWeightSum = 0;
  let alphaSum = 0;
  let lumaSum = 0;
  let checksum = 2166136261;

  for (let tileOffset = 0; tileOffset < tileAccumulation.length; tileOffset += 4) {
    const weight = tileAccumulation[tileOffset + 3];
    if (weight <= 0.0001) continue;

    const red = tileAccumulation[tileOffset] / weight;
    const green = tileAccumulation[tileOffset + 1] / weight;
    const blue = tileAccumulation[tileOffset + 2] / weight;
    const alpha = clampNumber(1 - Math.exp(-weight * RESOLVE_ALPHA_SCALE), 0, 0.98);
    const luma = red * 0.2126 + green * 0.7152 + blue * 0.0722;

    tileResolvedRgba[tileOffset] = red;
    tileResolvedRgba[tileOffset + 1] = green;
    tileResolvedRgba[tileOffset + 2] = blue;
    tileResolvedRgba[tileOffset + 3] = alpha;
    resolvedTileCount += 1;
    resolveWeightSum += weight;
    alphaSum += alpha;
    lumaSum += luma;
    checksum = checksumValue(checksum, red, green, blue, alpha, weight);
  }

  return {
    tileResolvedRgba,
    resolvedTileCount,
    resolveWeightSum,
    resolveAlphaMean: resolvedTileCount > 0 ? alphaSum / resolvedTileCount : 0,
    resolveLumaMean: resolvedTileCount > 0 ? lumaSum / resolvedTileCount : 0,
    resolveChecksum: checksum.toString(16).padStart(8, "0"),
  };
}

function checksumValue(checksum, red, green, blue, alpha, weight) {
  const values = [
    Math.round(clampNumber(red, 0, 1) * 255),
    Math.round(clampNumber(green, 0, 1) * 255),
    Math.round(clampNumber(blue, 0, 1) * 255),
    Math.round(clampNumber(alpha, 0, 1) * 255),
    Math.round(clampNumber(weight, 0, 65535)),
  ];
  let next = checksum;
  for (const value of values) {
    next ^= value;
    next = Math.imul(next, 16777619) >>> 0;
  }
  return next;
}

function buildObjectIndex(points) {
  const ids = new Set();
  for (const point of points) {
    ids.add(point.objectId);
  }
  const objectIdsByIndex = [...ids].sort((left, right) => left - right);
  const objectIndexById = new Map(objectIdsByIndex.map((id, index) => [id, index]));
  return { objectIdsByIndex, objectIndexById };
}

function buildObjectState({ objectIdsByIndex, visibleIds, removedIds, isolatedId }) {
  const objectState = new Uint32Array(Math.max(objectIdsByIndex.length, 1));
  objectIdsByIndex.forEach((objectId, index) => {
    const visible =
      visibleIds.has(objectId) &&
      !removedIds.has(objectId) &&
      (isolatedId === null || objectId === isolatedId);
    objectState[index] = visible ? 1 : 0;
  });
  if (objectIdsByIndex.length === 0) objectState[0] = 1;
  return objectState;
}

function sceneBounds(points) {
  if (points.length === 0) {
    return {
      minX: -1,
      maxX: 1,
      minZ: -1,
      maxZ: 1,
      spanX: 2,
      spanZ: 2,
    };
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minZ = Math.min(minZ, point.z);
    maxZ = Math.max(maxZ, point.z);
  }

  const spanX = Math.max(maxX - minX, 0.0001);
  const spanZ = Math.max(maxZ - minZ, 0.0001);
  return { minX, maxX, minZ, maxZ, spanX, spanZ };
}

function packGaussian({
  point,
  index,
  objectDenseIndex,
  radiusPixels,
  scale,
  renderMode,
  positionRadius,
  colorOpacity,
  scaleRotation,
  objectIndices,
}) {
  const vec4Offset = index * 4;
  positionRadius[vec4Offset] = point.x;
  positionRadius[vec4Offset + 1] = point.z;
  positionRadius[vec4Offset + 2] = point.y;
  positionRadius[vec4Offset + 3] = radiusPixels;

  const rgb = renderMode === "original" ? point.color : point.objectColor;
  colorOpacity[vec4Offset] = rgb[0] / 255;
  colorOpacity[vec4Offset + 1] = rgb[1] / 255;
  colorOpacity[vec4Offset + 2] = rgb[2] / 255;
  colorOpacity[vec4Offset + 3] = clampNumber(point.opacity ?? 1, 0, 1);

  scaleRotation[vec4Offset] = scale[0];
  scaleRotation[vec4Offset + 1] = scale[1];
  scaleRotation[vec4Offset + 2] = Number.isFinite(point.rotation) ? point.rotation : 0;
  scaleRotation[vec4Offset + 3] = 0;
  objectIndices[index] = objectDenseIndex;
}

function projectPointToSmokeViewport({ point, bounds, viewportWidth, viewportHeight }) {
  return {
    x: ((point.x - bounds.minX) / bounds.spanX) * Math.max(1, viewportWidth - 1),
    y: (1 - (point.z - bounds.minZ) / bounds.spanZ) * Math.max(1, viewportHeight - 1),
  };
}

function pointRadiusPixels({ point, scale, bounds, viewportWidth, viewportHeight, pointSize }) {
  const maxScale = Math.max(scale[0], scale[1], pointSize, 0.0006);
  const pixelsPerWorldUnit = Math.min(
    viewportWidth / Math.max(bounds.spanX, 0.0001),
    viewportHeight / Math.max(bounds.spanZ, 0.0001),
  );
  return clampNumber(maxScale * pixelsPerWorldUnit * 4.8, 1.5, 96);
}

function pointScale(point) {
  if (Array.isArray(point.scale) && point.scale.length > 0) {
    const first = safeScale(point.scale[0]);
    const second = safeScale(point.scale[1] ?? first);
    return [first, second];
  }
  return [0.018, 0.018];
}

function safeScale(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0.018;
  return clampNumber(numeric, 0.0006, 0.35);
}

function countActiveTiles(tileCounts) {
  let active = 0;
  for (const count of tileCounts) {
    if (count > 0) active += 1;
  }
  return active;
}

function clampInt(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
