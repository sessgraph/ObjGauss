export const WEBGPU_TILE_SMOKE_LAYOUT_VERSION = "webgpu-tile-smoke-v1";
export const WEBGPU_TILE_RESOLVE_VERSION = "webgpu-tile-resolve-v1";
export const WEBGPU_OBJECT_STATE_LAYOUT_VERSION = "webgpu-object-state-v1";
export const WEBGPU_TILE_SIZE = 16;
export const WEBGPU_TILE_MAX_ENTRIES = 8192;
export const WEBGPU_TILE_VIEWPORT = Object.freeze({ width: 1024, height: 1024 });
export const WEBGPU_OBJECT_STATE_STRIDE_UINT32 = 4;
export const WEBGPU_TILE_ENTRY_LAYOUT_COMPACT = "compact-offset-list";
export const WEBGPU_TILE_ENTRY_LAYOUT_FIXED = "fixed-cap-smoke";
const OBJECT_STATE_VISIBLE = 1 << 0;
const OBJECT_STATE_SELECTED = 1 << 1;
const OBJECT_STATE_REMOVED = 1 << 2;
const OBJECT_STATE_ISOLATED = 1 << 3;
const OBJECT_STATE_ENABLED = 1 << 4;
const RESOLVE_ALPHA_SCALE = 0.18;
const RESOLVE_ALPHA_GAIN = 0.78;
const RESOLVE_KERNEL_CUTOFF = 13;
const VIEWPORT_FIT_PADDING_RATIO = 0.08;
const TILE_SAMPLE_OFFSETS = Object.freeze([
  [-0.25, -0.25],
  [0.25, -0.25],
  [-0.25, 0.25],
  [0.25, 0.25],
]);

export function buildWebGpuTileSmoke({
  points,
  visibleIds,
  removedIds,
  isolatedId,
  selectedId = null,
  renderMode,
  pointSize,
  viewportWidth = WEBGPU_TILE_VIEWPORT.width,
  viewportHeight = WEBGPU_TILE_VIEWPORT.height,
  tileSize = WEBGPU_TILE_SIZE,
  maxEntriesPerTile = WEBGPU_TILE_MAX_ENTRIES,
  tileEntryLayout = WEBGPU_TILE_ENTRY_LAYOUT_COMPACT,
  includeTileEntries = false,
  includePixelOutput = false,
  computePixelReference = includePixelOutput,
}) {
  const objectIndex = buildObjectIndex(points);
  const objectStateContract = buildObjectState({
    objectIdsByIndex: objectIndex.objectIdsByIndex,
    objectCountsByIndex: objectIndex.objectCountsByIndex,
    visibleIds,
    removedIds,
    isolatedId,
    selectedId,
  });
  const objectState = objectStateContract.buffer;
  const bounds = sceneBounds(points, viewportWidth, viewportHeight);
  const tileColumns = Math.max(1, Math.ceil(viewportWidth / tileSize));
  const tileRows = Math.max(1, Math.ceil(viewportHeight / tileSize));
  const tileCount = tileColumns * tileRows;
  const tileCounts = new Uint32Array(tileCount);
  const tileCenters = buildTileCenters({ tileColumns, tileRows, tileSize });
  const tileAccumulation = new Float32Array(tileCount * 4);
  const compactTileEntries = tileEntryLayout === WEBGPU_TILE_ENTRY_LAYOUT_COMPACT;
  let tileOffsets = includeTileEntries ? buildTileOffsets({ tileCounts, maxEntriesPerTile, compact: false }) : null;
  let tileEntries = includeTileEntries && !compactTileEntries
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

    if (!objectIsVisible(objectState, objectDenseIndex)) return;
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
        } else if (!compactTileEntries && occupancy >= maxEntriesPerTile) {
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
          scale,
          bounds,
          viewportWidth,
          viewportHeight,
          tileSize,
          gaussianIndex: index,
          colorOpacity,
          scaleRotation,
          tileAccumulation,
        });
      }
    }
  });

  if (includeTileEntries && compactTileEntries) {
    tileOffsets = buildTileOffsets({ tileCounts, maxEntriesPerTile, compact: true });
    tileEntries = new Uint32Array(tileReferenceCount);
    populateCompactTileEntries({
      points,
      objectIndex,
      objectState,
      bounds,
      viewportWidth,
      viewportHeight,
      tileSize,
      tileColumns,
      tileRows,
      pointSize,
      tileOffsets,
      tileEntries,
    });
  }

  const activeTileCount = countActiveTiles(tileCounts);
  const capacity = summarizeTileCapacity({
    tileCounts,
    tileReferenceCount,
    tileOverflowCount,
    tileCount,
    maxEntriesPerTile,
    tileEntryLayout,
  });
  const resolve = resolveTileAccumulation(tileAccumulation);
  const pixelResolve = includePixelOutput
    ? resolvePixelOutput({
        computePixelReference,
        positionRadius,
        colorOpacity,
        scaleRotation,
        objectIndices,
        objectState,
        tileCounts,
        tileEntries,
        tileOffsets,
        maxEntriesPerTile,
        tileEntryLayout,
        bounds,
        tileColumns,
        tileSize,
        viewportWidth,
        viewportHeight,
      })
    : null;
  return {
    layoutVersion: WEBGPU_TILE_SMOKE_LAYOUT_VERSION,
    resolveVersion: WEBGPU_TILE_RESOLVE_VERSION,
    resolveMode: "tile-2x2-covariance-weighted-oit",
    tileSize,
    viewportWidth,
    viewportHeight,
    pixelCount: viewportWidth * viewportHeight,
    pixelOutputMode: includePixelOutput ? "viewport-storage-rgba-direct-gaussian" : "not-allocated",
    pixelOutputIncluded: Boolean(pixelResolve),
    pixelReferenceIncluded: Boolean(pixelResolve?.computed),
    boundsMinX: bounds.minX,
    boundsMinZ: bounds.minZ,
    boundsSpanX: bounds.spanX,
    boundsSpanZ: bounds.spanZ,
    boundsFitMode: bounds.fitMode,
    boundsPaddingRatio: bounds.paddingRatio,
    boundsViewportAspect: bounds.viewportAspect,
    boundsWorldAspect: bounds.worldAspect,
    tileColumns,
    tileRows,
    tileCount,
    maxEntriesPerTile,
    tileEntryLayout,
    tileEntryOffsetCount: tileOffsets?.length ?? 0,
    packedGaussians: points.length,
    visibleGaussians,
    binnedGaussians,
    activeTileCount,
    tileReferenceCount,
    tileOverflowCount,
    tileOverflowTileCount: capacity.overflowTileCount,
    tileOverflowRatio: capacity.overflowRatio,
    tileOverflowMaxExcess: capacity.maxExcess,
    tileEntryStoredCount: capacity.storedReferenceCount,
    tileEntryCapacity: capacity.entryCapacity,
    tileEntryUtilization: capacity.entryUtilization,
    tileCapacityMode: capacity.mode,
    tileCapacityStatus: capacity.status,
    tileCapacityGate: capacity.gate,
    maxTileOccupancy,
    resolvedTileCount: resolve.resolvedTileCount,
    resolveWeightSum: resolve.resolveWeightSum,
    resolveAlphaMean: resolve.resolveAlphaMean,
    resolveLumaMean: resolve.resolveLumaMean,
    resolveChecksum: resolve.resolveChecksum,
    pixelResolvedCount: pixelResolve?.pixelResolvedCount ?? 0,
    pixelResolveChecksum: pixelResolve?.pixelResolveChecksum ?? "",
    objectCount: objectIndex.objectIdsByIndex.length,
    objectStateLayoutVersion: objectStateContract.layoutVersion,
    objectStateStrideUint32: objectStateContract.strideUint32,
    objectStateVisibleObjects: objectStateContract.visibleObjects,
    objectStateHiddenObjects: objectStateContract.hiddenObjects,
    objectStateRemovedObjects: objectStateContract.removedObjects,
    objectStateSelectedObjects: objectStateContract.selectedObjects,
    objectStateIsolatedObjects: objectStateContract.isolatedObjects,
    objectStateChecksum: objectStateContract.checksum,
    buffers: {
      positionRadius,
      colorOpacity,
      scaleRotation,
      objectIndices,
      objectState,
      objectIds: objectStateContract.objectIds,
      tileCounts,
      tileOffsets,
      tileAccumulation,
      tileResolvedRgba: resolve.tileResolvedRgba,
      pixelResolvedRgba: pixelResolve?.pixelResolvedRgba ?? null,
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

function buildTileOffsets({ tileCounts, maxEntriesPerTile, compact }) {
  const tileOffsets = new Uint32Array(tileCounts.length);
  if (!compact) {
    for (let tileIndex = 0; tileIndex < tileCounts.length; tileIndex += 1) {
      tileOffsets[tileIndex] = tileIndex * maxEntriesPerTile;
    }
    return tileOffsets;
  }

  let cursor = 0;
  for (let tileIndex = 0; tileIndex < tileCounts.length; tileIndex += 1) {
    tileOffsets[tileIndex] = cursor;
    cursor += tileCounts[tileIndex];
  }
  return tileOffsets;
}

function populateCompactTileEntries({
  points,
  objectIndex,
  objectState,
  bounds,
  viewportWidth,
  viewportHeight,
  tileSize,
  tileColumns,
  tileRows,
  pointSize,
  tileOffsets,
  tileEntries,
}) {
  const cursors = new Uint32Array(tileOffsets);
  points.forEach((point, index) => {
    const objectDenseIndex = objectIndex.objectIndexById.get(point.objectId) ?? 0;
    if (!objectIsVisible(objectState, objectDenseIndex)) return;
    const scale = pointScale(point);
    const radiusPixels = pointRadiusPixels({
      point,
      scale,
      bounds,
      viewportWidth,
      viewportHeight,
      pointSize,
    });
    const screen = projectPointToSmokeViewport({ point, bounds, viewportWidth, viewportHeight });
    const minTileX = clampInt(Math.floor((screen.x - radiusPixels) / tileSize), 0, tileColumns - 1);
    const maxTileX = clampInt(Math.floor((screen.x + radiusPixels) / tileSize), 0, tileColumns - 1);
    const minTileY = clampInt(Math.floor((screen.y - radiusPixels) / tileSize), 0, tileRows - 1);
    const maxTileY = clampInt(Math.floor((screen.y + radiusPixels) / tileSize), 0, tileRows - 1);

    for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
      for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
        const tileIndex = tileY * tileColumns + tileX;
        const cursor = cursors[tileIndex];
        tileEntries[cursor] = index;
        cursors[tileIndex] = cursor + 1;
      }
    }
  });
}

function accumulateTileResolve({
  tileIndex,
  tileCenters,
  screen,
  scale,
  bounds,
  viewportWidth,
  viewportHeight,
  tileSize,
  gaussianIndex,
  colorOpacity,
  scaleRotation,
  tileAccumulation,
}) {
  const tileX = tileIndex % tileCenters.tileColumns;
  const tileY = Math.floor(tileIndex / tileCenters.tileColumns);
  const gaussianOffset = gaussianIndex * 4;
  const scaleOffset = gaussianIndex * 4;
  const pixelsPerWorldUnit = Math.min(
    viewportWidth / Math.max(bounds.spanX, 0.0001),
    viewportHeight / Math.max(bounds.spanZ, 0.0001),
  );
  const sigmaX = Math.max((scale[0] * pixelsPerWorldUnit * 4.8) / 3, 0.0001);
  const sigmaY = Math.max((scale[1] * pixelsPerWorldUnit * 4.8) / 3, 0.0001);
  const rotation = scaleRotation[scaleOffset + 2];
  const cosine = Math.cos(rotation);
  const sine = Math.sin(rotation);
  let red = 0;
  let green = 0;
  let blue = 0;
  let alphaWeight = 0;

  for (const [sampleX, sampleY] of TILE_SAMPLE_OFFSETS) {
    const dx = tileCenters.centersX[tileX] + sampleX * tileSize - screen.x;
    const dy = tileCenters.centersY[tileY] + sampleY * tileSize - screen.y;
    const rotatedX = cosine * dx - sine * dy;
    const rotatedY = sine * dx + cosine * dy;
    const d =
      (rotatedX * rotatedX) / (sigmaX * sigmaX) +
      (rotatedY * rotatedY) / (sigmaY * sigmaY);
    if (d > RESOLVE_KERNEL_CUTOFF) continue;

    const weight =
      Math.exp(-0.5 * d) *
      colorOpacity[gaussianOffset + 3] *
      RESOLVE_ALPHA_GAIN *
      (1 / TILE_SAMPLE_OFFSETS.length);
    if (weight <= 0.0001) continue;
    red += colorOpacity[gaussianOffset] * weight;
    green += colorOpacity[gaussianOffset + 1] * weight;
    blue += colorOpacity[gaussianOffset + 2] * weight;
    alphaWeight += weight;
  }

  if (alphaWeight <= 0.0001) return;

  const tileOffset = tileIndex * 4;
  tileAccumulation[tileOffset] += red;
  tileAccumulation[tileOffset + 1] += green;
  tileAccumulation[tileOffset + 2] += blue;
  tileAccumulation[tileOffset + 3] += alphaWeight;
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

function resolvePixelOutput({
  computePixelReference,
  positionRadius,
  colorOpacity,
  scaleRotation,
  objectIndices,
  objectState,
  tileCounts,
  tileEntries,
  tileOffsets,
  maxEntriesPerTile,
  tileEntryLayout,
  bounds,
  tileColumns,
  tileSize,
  viewportWidth,
  viewportHeight,
}) {
  const pixelCount = viewportWidth * viewportHeight;
  const pixelResolvedRgba = new Float32Array(pixelCount * 4);
  if (!computePixelReference || !tileEntries || !tileOffsets) {
    return {
      pixelResolvedRgba,
      pixelResolvedCount: 0,
      pixelResolveChecksum: "",
      computed: false,
    };
  }

  const projection = buildPixelProjection({
    positionRadius,
    scaleRotation,
    bounds,
    viewportWidth,
    viewportHeight,
  });
  let pixelResolvedCount = 0;
  let checksum = 2166136261;

  for (let y = 0; y < viewportHeight; y += 1) {
    const tileY = Math.floor(y / tileSize);
    for (let x = 0; x < viewportWidth; x += 1) {
      const tileX = Math.min(Math.floor(x / tileSize), tileColumns - 1);
      const tileIndex = tileY * tileColumns + tileX;
      const storedCount =
        tileEntryLayout === WEBGPU_TILE_ENTRY_LAYOUT_COMPACT
          ? tileCounts[tileIndex]
          : Math.min(tileCounts[tileIndex], maxEntriesPerTile);
      const entryBase = tileOffsets[tileIndex];
      const pixelOffset = (y * viewportWidth + x) * 4;
      let accumulatedRed = 0;
      let accumulatedGreen = 0;
      let accumulatedBlue = 0;
      let accumulatedWeight = 0;

      for (let entryOffset = 0; entryOffset < storedCount; entryOffset += 1) {
        const gaussianIndex = tileEntries[entryBase + entryOffset];
        const objectDenseIndex = objectIndices[gaussianIndex];
        if (!objectIsVisible(objectState, objectDenseIndex)) continue;

        const dx = x + 0.5 - projection.screenX[gaussianIndex];
        const dy = y + 0.5 - projection.screenY[gaussianIndex];
        const cosine = projection.cosine[gaussianIndex];
        const sine = projection.sine[gaussianIndex];
        const rotatedX = cosine * dx - sine * dy;
        const rotatedY = sine * dx + cosine * dy;
        const d =
          (rotatedX * rotatedX) / projection.sigmaXSquared[gaussianIndex] +
          (rotatedY * rotatedY) / projection.sigmaYSquared[gaussianIndex];
        if (d > RESOLVE_KERNEL_CUTOFF) continue;

        const gaussianOffset = gaussianIndex * 4;
        const weight =
          Math.exp(-0.5 * d) *
          colorOpacity[gaussianOffset + 3] *
          RESOLVE_ALPHA_GAIN;
        if (weight <= 0.0001) continue;
        accumulatedRed += colorOpacity[gaussianOffset] * weight;
        accumulatedGreen += colorOpacity[gaussianOffset + 1] * weight;
        accumulatedBlue += colorOpacity[gaussianOffset + 2] * weight;
        accumulatedWeight += weight;
      }

      if (accumulatedWeight <= 0.0001) continue;

      const red = accumulatedRed / accumulatedWeight;
      const green = accumulatedGreen / accumulatedWeight;
      const blue = accumulatedBlue / accumulatedWeight;
      const alpha = clampNumber(1 - Math.exp(-accumulatedWeight * RESOLVE_ALPHA_SCALE), 0, 0.98);
      pixelResolvedRgba[pixelOffset] = red;
      pixelResolvedRgba[pixelOffset + 1] = green;
      pixelResolvedRgba[pixelOffset + 2] = blue;
      pixelResolvedRgba[pixelOffset + 3] = alpha;
      pixelResolvedCount += 1;
      checksum = checksumValue(checksum, red, green, blue, alpha, accumulatedWeight);
    }
  }

  return {
    pixelResolvedRgba,
    pixelResolvedCount,
    pixelResolveChecksum: checksum.toString(16).padStart(8, "0"),
    computed: true,
  };
}

function buildPixelProjection({
  positionRadius,
  scaleRotation,
  bounds,
  viewportWidth,
  viewportHeight,
}) {
  const gaussianCount = positionRadius.length / 4;
  const screenX = new Float32Array(gaussianCount);
  const screenY = new Float32Array(gaussianCount);
  const sigmaXSquared = new Float32Array(gaussianCount);
  const sigmaYSquared = new Float32Array(gaussianCount);
  const cosine = new Float32Array(gaussianCount);
  const sine = new Float32Array(gaussianCount);
  const pixelsPerWorldUnit = Math.min(
    viewportWidth / Math.max(bounds.spanX, 0.0001),
    viewportHeight / Math.max(bounds.spanZ, 0.0001),
  );

  for (let index = 0; index < gaussianCount; index += 1) {
    const offset = index * 4;
    screenX[index] =
      ((positionRadius[offset] - bounds.minX) / Math.max(bounds.spanX, 0.0001)) *
      Math.max(1, viewportWidth - 1);
    screenY[index] =
      (1 - (positionRadius[offset + 1] - bounds.minZ) / Math.max(bounds.spanZ, 0.0001)) *
      Math.max(1, viewportHeight - 1);
    const sigmaX = Math.max((scaleRotation[offset] * pixelsPerWorldUnit * 4.8) / 3, 0.0001);
    const sigmaY = Math.max((scaleRotation[offset + 1] * pixelsPerWorldUnit * 4.8) / 3, 0.0001);
    sigmaXSquared[index] = sigmaX * sigmaX;
    sigmaYSquared[index] = sigmaY * sigmaY;
    const rotation = scaleRotation[offset + 2];
    cosine[index] = Math.cos(rotation);
    sine[index] = Math.sin(rotation);
  }

  return { screenX, screenY, sigmaXSquared, sigmaYSquared, cosine, sine };
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
  const counts = new Map();
  for (const point of points) {
    counts.set(point.objectId, (counts.get(point.objectId) ?? 0) + 1);
  }
  const objectIdsByIndex = [...counts.keys()].sort((left, right) => left - right);
  const objectIndexById = new Map(objectIdsByIndex.map((id, index) => [id, index]));
  const objectCountsByIndex = objectIdsByIndex.map((id) => counts.get(id) ?? 0);
  return { objectIdsByIndex, objectCountsByIndex, objectIndexById };
}

function buildObjectState({
  objectIdsByIndex,
  objectCountsByIndex,
  visibleIds,
  removedIds,
  isolatedId,
  selectedId,
}) {
  const objectCount = objectIdsByIndex.length;
  const buffer = new Uint32Array(Math.max(objectCount, 1) * WEBGPU_OBJECT_STATE_STRIDE_UINT32);
  const objectIds = new Int32Array(objectCount);
  let visibleObjects = 0;
  let hiddenObjects = 0;
  let removedObjects = 0;
  let selectedObjects = 0;
  let isolatedObjects = 0;

  objectIdsByIndex.forEach((objectId, index) => {
    const enabled = visibleIds.has(objectId);
    const removed = removedIds.has(objectId);
    const selected = selectedId !== null && objectId === selectedId;
    const isolated = isolatedId !== null && objectId === isolatedId;
    const visible =
      enabled &&
      !removed &&
      (isolatedId === null || objectId === isolatedId);

    let flags = 0;
    if (visible) flags |= OBJECT_STATE_VISIBLE;
    if (selected) flags |= OBJECT_STATE_SELECTED;
    if (removed) flags |= OBJECT_STATE_REMOVED;
    if (isolated) flags |= OBJECT_STATE_ISOLATED;
    if (enabled) flags |= OBJECT_STATE_ENABLED;

    const offset = index * WEBGPU_OBJECT_STATE_STRIDE_UINT32;
    buffer[offset] = flags;
    buffer[offset + 1] = index;
    buffer[offset + 2] = objectCountsByIndex[index] ?? 0;
    buffer[offset + 3] = 0;
    objectIds[index] = objectId;

    if (visible) visibleObjects += 1;
    else hiddenObjects += 1;
    if (removed) removedObjects += 1;
    if (selected) selectedObjects += 1;
    if (isolated) isolatedObjects += 1;
  });

  if (objectCount === 0) {
    buffer[0] = OBJECT_STATE_VISIBLE | OBJECT_STATE_ENABLED;
  }

  return {
    layoutVersion: WEBGPU_OBJECT_STATE_LAYOUT_VERSION,
    strideUint32: WEBGPU_OBJECT_STATE_STRIDE_UINT32,
    buffer,
    objectIds,
    visibleObjects,
    hiddenObjects,
    removedObjects,
    selectedObjects,
    isolatedObjects,
    checksum: checksumObjectState(buffer, objectIds),
  };
}

function objectIsVisible(objectState, objectDenseIndex) {
  const offset = objectDenseIndex * WEBGPU_OBJECT_STATE_STRIDE_UINT32;
  return (objectState[offset] & OBJECT_STATE_VISIBLE) !== 0;
}

function checksumObjectState(buffer, objectIds) {
  let checksum = 2166136261;
  for (const value of buffer) {
    checksum = checksumUint32(checksum, value);
  }
  for (const value of objectIds) {
    checksum = checksumUint32(checksum, value >>> 0);
  }
  return checksum.toString(16).padStart(8, "0");
}

function checksumUint32(checksum, value) {
  let next = checksum;
  next ^= value & 0xff;
  next = Math.imul(next, 16777619) >>> 0;
  next ^= (value >>> 8) & 0xff;
  next = Math.imul(next, 16777619) >>> 0;
  next ^= (value >>> 16) & 0xff;
  next = Math.imul(next, 16777619) >>> 0;
  next ^= (value >>> 24) & 0xff;
  return Math.imul(next, 16777619) >>> 0;
}

function sceneBounds(points, viewportWidth = 1, viewportHeight = 1) {
  if (points.length === 0) {
    return {
      minX: -1,
      maxX: 1,
      minZ: -1,
      maxZ: 1,
      spanX: 2,
      spanZ: 2,
      fitMode: "empty-default",
      paddingRatio: 0,
      viewportAspect: 1,
      worldAspect: 1,
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

  return fitBoundsToViewport({
    minX,
    maxX,
    minZ,
    maxZ,
    viewportWidth,
    viewportHeight,
  });
}

function fitBoundsToViewport({ minX, maxX, minZ, maxZ, viewportWidth, viewportHeight }) {
  let spanX = Math.max(maxX - minX, 0.0001);
  let spanZ = Math.max(maxZ - minZ, 0.0001);
  const centerX = (minX + maxX) * 0.5;
  const centerZ = (minZ + maxZ) * 0.5;
  const viewportAspect = Math.min(
    4,
    Math.max(0.25, Math.max(1, viewportWidth) / Math.max(1, viewportHeight)),
  );
  const currentAspect = spanX / spanZ;
  if (currentAspect < viewportAspect) {
    spanX = spanZ * viewportAspect;
  } else {
    spanZ = spanX / viewportAspect;
  }
  spanX *= 1 + VIEWPORT_FIT_PADDING_RATIO * 2;
  spanZ *= 1 + VIEWPORT_FIT_PADDING_RATIO * 2;
  return {
    minX: centerX - spanX * 0.5,
    maxX: centerX + spanX * 0.5,
    minZ: centerZ - spanZ * 0.5,
    maxZ: centerZ + spanZ * 0.5,
    spanX,
    spanZ,
    fitMode: "aspect-fit-padding",
    paddingRatio: VIEWPORT_FIT_PADDING_RATIO,
    viewportAspect,
    worldAspect: spanX / spanZ,
  };
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

function summarizeTileCapacity({
  tileCounts,
  tileReferenceCount,
  tileOverflowCount,
  tileCount,
  maxEntriesPerTile,
  tileEntryLayout,
}) {
  if (tileEntryLayout === WEBGPU_TILE_ENTRY_LAYOUT_COMPACT) {
    return {
      mode: WEBGPU_TILE_ENTRY_LAYOUT_COMPACT,
      status: "ok",
      gate: "pass",
      overflowTileCount: 0,
      overflowRatio: 0,
      maxExcess: 0,
      storedReferenceCount: tileReferenceCount,
      entryCapacity: tileReferenceCount,
      entryUtilization: tileReferenceCount > 0 ? 1 : 0,
    };
  }

  let overflowTileCount = 0;
  let maxExcess = 0;
  for (const count of tileCounts) {
    const excess = Math.max(0, count - maxEntriesPerTile);
    if (excess > 0) {
      overflowTileCount += 1;
      maxExcess = Math.max(maxExcess, excess);
    }
  }

  const entryCapacity = tileCount * maxEntriesPerTile;
  const storedReferenceCount = Math.max(0, tileReferenceCount - tileOverflowCount);
  const overflowRatio =
    tileReferenceCount > 0 ? tileOverflowCount / tileReferenceCount : 0;
  const entryUtilization =
    entryCapacity > 0 ? Math.min(1, storedReferenceCount / entryCapacity) : 0;
  const hasOverflow = tileOverflowCount > 0 || overflowTileCount > 0;
  return {
    mode: WEBGPU_TILE_ENTRY_LAYOUT_FIXED,
    status: hasOverflow ? "overflow" : "ok",
    gate: hasOverflow ? "blocked" : "pass",
    overflowTileCount,
    overflowRatio,
    maxExcess,
    storedReferenceCount,
    entryCapacity,
    entryUtilization,
  };
}

function clampInt(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clampNumber(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
