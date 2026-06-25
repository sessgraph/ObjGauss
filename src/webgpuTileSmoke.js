import {
  WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED,
  WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K,
  normalizeWebGpuDepthSortTuning,
  WEBGPU_DEPTH_BIN_COUNT_DEFAULT,
  WEBGPU_DEPTH_SORT_TUNING_MODE,
} from "./webgpuDepthTuning.js";
import {
  normalizeWebGpuCameraTuning,
  WEBGPU_CAMERA_MODE_EDIT_FIXED,
  WEBGPU_CAMERA_MODE_SPARK_FRAME,
  WEBGPU_CAMERA_TUNING_MODE,
} from "./webgpuCameraTuning.js";

export const WEBGPU_TILE_SMOKE_LAYOUT_VERSION = "webgpu-tile-smoke-v1";
export const WEBGPU_TILE_RESOLVE_VERSION = "webgpu-tile-resolve-v1";
export const WEBGPU_OBJECT_STATE_LAYOUT_VERSION = "webgpu-object-state-v1";
export const WEBGPU_TILE_SIZE = 16;
export const WEBGPU_TILE_MAX_ENTRIES = 8192;
export const WEBGPU_TILE_VIEWPORT = Object.freeze({ width: 1024, height: 1024 });
export const WEBGPU_OBJECT_STATE_STRIDE_UINT32 = 4;
export const WEBGPU_TILE_ENTRY_LAYOUT_COMPACT = "compact-offset-list";
export const WEBGPU_TILE_ENTRY_LAYOUT_FIXED = "fixed-cap-smoke";
export const WEBGPU_TILE_LIST_MODE_VISIBLE = "visible-only";
export const WEBGPU_TILE_LIST_MODE_OBJECT_STATE = "object-state-filtered";
export const WEBGPU_TILE_PROJECTION_MODE = "edit-perspective-camera-v1";
export const WEBGPU_TILE_SPARK_FRAME_PROJECTION_MODE = "spark-framed-perspective-camera-v1";
export const WEBGPU_TILE_DEPTH_WEIGHT_MODE = "front-weighted-oit-v1";
export const WEBGPU_TILE_SCREEN_COVARIANCE_MODE = "camera-jacobian-covariance-v1";
export const WEBGPU_TILE_COLOR_FIDELITY_MODE = "source-color-fidelity-v1";
export const WEBGPU_COLOR_TUNING_MODE = "runtime-color-tuning-v1";
export const WEBGPU_COLOR_MODE_SOURCE = "source";
export const WEBGPU_COLOR_MODE_SH_VIEW = "sh-view";
export const WEBGPU_COLOR_MODE_DEFAULT = WEBGPU_COLOR_MODE_SOURCE;
export const WEBGPU_PIXEL_DEPTH_SORT_MODE_DEPTH_BINNED = "depth-binned-alpha-composite-v1";
export const WEBGPU_PIXEL_DEPTH_SORT_MODE_FRONT_TOP_K = "front-top-k-alpha-composite-v1";
export const WEBGPU_PIXEL_DEPTH_SORT_MODE = WEBGPU_PIXEL_DEPTH_SORT_MODE_DEPTH_BINNED;
export const WEBGPU_PIXEL_COVERAGE_MODE = "footprint-weight-floor-calibrated-v1";
const OBJECT_STATE_VISIBLE = 1 << 0;
const OBJECT_STATE_SELECTED = 1 << 1;
const OBJECT_STATE_REMOVED = 1 << 2;
const OBJECT_STATE_ISOLATED = 1 << 3;
const OBJECT_STATE_ENABLED = 1 << 4;
const RESOLVE_ALPHA_SCALE = 0.18;
const RESOLVE_ALPHA_GAIN = 0.78;
const PIXEL_COVERAGE_WEIGHT_FLOOR = 0.004;
const RESOLVE_KERNEL_CUTOFF = 13;
const VIEWPORT_FIT_PADDING_RATIO = 0.08;
const DEPTH_WEIGHT_STRENGTH = 1.45;
const DEPTH_WEIGHT_FLOOR = 0.22;
const FRONT_DEPTH_GATE_STRENGTH = 12;
const FRONT_DEPTH_GATE_FLOOR = 0.06;
export const WEBGPU_COVERAGE_TUNING_MODE = "runtime-coverage-tuning-v1";
export const WEBGPU_COVERAGE_TUNING_DEFAULTS = Object.freeze({
  footprintScale: 2.2,
  maxAnisotropy: 4,
});
const SCREEN_COVARIANCE_MAX_ANISOTROPY = WEBGPU_COVERAGE_TUNING_DEFAULTS.maxAnisotropy;
const EDIT_CAMERA_POSITION = Object.freeze([3.6, 2.8, 3.4]);
const EDIT_CAMERA_TARGET = Object.freeze([0, 0, 0.25]);
const EDIT_CAMERA_UP = Object.freeze([0, 1, 0]);
const EDIT_CAMERA_FOV_DEGREES = 52;
const SPARK_FRAME_CAMERA_FOV_DEGREES = 58;
const SPARK_FRAME_DISTANCE_MULTIPLIER = 1.7;
const SPARK_FRAME_HEIGHT_MULTIPLIER = 0.58;
const EDIT_SPLAT_SIZE_SCALE = WEBGPU_COVERAGE_TUNING_DEFAULTS.footprintScale;
const TILE_SAMPLE_OFFSETS = Object.freeze([
  [-0.25, -0.25],
  [0.25, -0.25],
  [-0.25, 0.25],
  [0.25, 0.25],
]);
const SH_C0 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = Object.freeze([
  1.0925484305920792,
  -1.0925484305920792,
  0.31539156525252005,
  -1.0925484305920792,
  0.5462742152960396,
]);
const SH_C3 = Object.freeze([
  -0.5900435899266435,
  2.890611442640554,
  -0.4570457994644658,
  0.3731763325901154,
  -0.4570457994644658,
  1.445305721320277,
  -0.5900435899266435,
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
  tileListMode = WEBGPU_TILE_LIST_MODE_VISIBLE,
  shRestCoefficients = null,
  shRestCoefficientCount = 0,
  includeTileEntries = false,
  includePixelOutput = false,
  computePixelReference = includePixelOutput,
  coverageTuning = null,
  depthSortTuning = null,
  cameraTuning = null,
  colorTuning = null,
}) {
  const resolvedCoverageTuning = normalizeWebGpuCoverageTuning(coverageTuning);
  const resolvedDepthSortTuning = normalizeWebGpuDepthSortTuning(depthSortTuning);
  const resolvedCameraTuning = normalizeWebGpuCameraTuning(cameraTuning);
  const resolvedColorTuning = normalizeWebGpuColorTuning(colorTuning);
  const resolvedTileListMode = normalizeWebGpuTileListMode(tileListMode);
  const binHiddenGaussians = resolvedTileListMode === WEBGPU_TILE_LIST_MODE_OBJECT_STATE;
  const pixelDepthBinCount = resolvedDepthSortTuning.pixelDepthBinCount;
  const pixelDepthAlphaMode = resolvedDepthSortTuning.pixelDepthAlphaMode;
  const pixelDepthSortMode = pixelDepthSortModeForAlphaMode(pixelDepthAlphaMode);
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
  const bounds = sceneBounds(points, viewportWidth, viewportHeight, resolvedCameraTuning);
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
  let screenCovarianceGaussians = 0;
  let screenCovarianceFallbackGaussians = 0;
  let screenCovarianceClampedGaussians = 0;
  let screenCovarianceSigmaSum = 0;
  let rgbColorGaussians = 0;
  let shDcColorGaussians = 0;
  let fallbackColorGaussians = 0;
  let objectColorGaussians = 0;
  let shRestGaussians = 0;
  let shRestCoefficientMax = 0;
  let shDegreeMax = 0;
  let shViewColorGaussians = 0;
  let opacitySum = 0;

  points.forEach((point, index) => {
    const objectDenseIndex = objectIndex.objectIndexById.get(point.objectId) ?? 0;
    const scale = pointScale(point);
    const screen = projectPointToSmokeViewport({ point, bounds, viewportWidth, viewportHeight });
    const screenCovariance = projectPointScreenCovariance({
      point,
      scale,
      screen,
      bounds,
      viewportWidth,
      viewportHeight,
      coverageTuning: resolvedCoverageTuning,
    });
    if (screenCovariance.mode === "full") {
      screenCovarianceGaussians += 1;
    } else {
      screenCovarianceFallbackGaussians += 1;
    }
    if (screenCovariance.clamped) {
      screenCovarianceClampedGaussians += 1;
    }
    screenCovarianceSigmaSum += screenCovariance.sigmaMajor;
    const colorSource = renderColorSource(point, renderMode);
    if (colorSource === "rgb") {
      rgbColorGaussians += 1;
    } else if (colorSource === "sh-dc") {
      shDcColorGaussians += 1;
    } else if (colorSource === "object-palette") {
      objectColorGaussians += 1;
    } else {
      fallbackColorGaussians += 1;
    }
    const pointShRestCount = pointShRestCoefficientCount(point);
    if (pointShRestCount > 0) {
      shRestGaussians += 1;
      shRestCoefficientMax = Math.max(shRestCoefficientMax, pointShRestCount);
      shDegreeMax = Math.max(shDegreeMax, pointShDegree(point, pointShRestCount));
    }
    const renderColor = pointRenderColor({
      point,
      index,
      renderMode,
      colorTuning: resolvedColorTuning,
      shRestCoefficients,
      shRestCoefficientCount,
      cameraProjection: bounds.cameraProjection,
    });
    if (renderColor.mode === WEBGPU_COLOR_MODE_SH_VIEW) {
      shViewColorGaussians += 1;
    }
    opacitySum += clampNumber(point.opacity ?? 1, 0, 1);
    const radiusPixels = pointRadiusPixels({
      screen,
      screenCovariance,
      pointSize,
      coverageTuning: resolvedCoverageTuning,
    });
    packGaussian({
      point,
      index,
      objectDenseIndex,
      radiusPixels,
      screen,
      screenCovariance,
      renderMode,
      rgb: renderColor.rgb,
      positionRadius,
      colorOpacity,
      scaleRotation,
      objectIndices,
    });

    const visible = objectIsVisible(objectState, objectDenseIndex);
    if (visible) visibleGaussians += 1;
    if (!visible && !binHiddenGaussians) return;
    if (!screenInfluencesViewport({ screen, radiusPixels, viewportWidth, viewportHeight })) return;

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
        if (visible) {
          accumulateTileResolve({
            tileIndex,
            tileCenters,
            screen,
            bounds,
            tileSize,
            gaussianIndex: index,
            colorOpacity,
            scaleRotation,
            tileAccumulation,
          });
        }
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
      coverageTuning: resolvedCoverageTuning,
      tileListMode: resolvedTileListMode,
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
        depthMin: bounds.depthMin,
        depthSpan: bounds.depthSpan,
        tileColumns,
        tileSize,
        viewportWidth,
        viewportHeight,
        pixelDepthBinCount,
        pixelDepthAlphaMode,
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
    projectionMode: bounds.projectionMode,
    projectionCameraTuningMode: bounds.cameraTuningMode,
    projectionCameraMode: bounds.cameraMode,
    projectionCameraFovDegrees: bounds.cameraFovDegrees,
    projectionCameraPosition: bounds.cameraPosition,
    projectionCameraTarget: bounds.cameraTarget,
    projectionCameraDistance: bounds.cameraDistance,
    projectionCameraFrameMaxDim: bounds.cameraFrameMaxDim,
    projectionDepthMin: bounds.depthMin,
    projectionDepthMax: bounds.depthMax,
    projectionDepthSpan: bounds.depthSpan,
    depthWeightMode: WEBGPU_TILE_DEPTH_WEIGHT_MODE,
    pixelDepthSortMode,
    pixelDepthTuningMode: WEBGPU_DEPTH_SORT_TUNING_MODE,
    pixelDepthAlphaMode,
    pixelDepthGateStrength: FRONT_DEPTH_GATE_STRENGTH,
    pixelDepthGateFloor: FRONT_DEPTH_GATE_FLOOR,
    pixelDepthBinCount,
    pixelCoverageMode: WEBGPU_PIXEL_COVERAGE_MODE,
    pixelCoverageWeightFloor: PIXEL_COVERAGE_WEIGHT_FLOOR,
    pixelCoverageFootprintScale: resolvedCoverageTuning.footprintScale,
    pixelCoverageTuningMode: WEBGPU_COVERAGE_TUNING_MODE,
    screenCovarianceMode: WEBGPU_TILE_SCREEN_COVARIANCE_MODE,
    colorFidelityMode: WEBGPU_TILE_COLOR_FIDELITY_MODE,
    colorTuningMode: WEBGPU_COLOR_TUNING_MODE,
    colorMode: resolvedColorTuning.colorMode,
    colorSourceRgbGaussians: rgbColorGaussians,
    colorSourceShDcGaussians: shDcColorGaussians,
    colorSourceFallbackGaussians: fallbackColorGaussians,
    colorSourceObjectGaussians: objectColorGaussians,
    colorShRestGaussians: shRestGaussians,
    colorShRestCoefficientMax: shRestCoefficientMax,
    colorShDegreeMax: shDegreeMax,
    colorShViewGaussians: shViewColorGaussians,
    colorOpacityMean: points.length > 0 ? opacitySum / points.length : 0,
    screenCovarianceGaussians,
    screenCovarianceFallbackGaussians,
    screenCovarianceClampedGaussians,
    screenCovarianceMaxAnisotropy: resolvedCoverageTuning.maxAnisotropy,
    screenCovarianceSigmaMean:
      points.length > 0 ? screenCovarianceSigmaSum / points.length : 0,
    tileColumns,
    tileRows,
    tileCount,
    maxEntriesPerTile,
    tileEntryLayout,
    tileListMode: resolvedTileListMode,
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

export function normalizeWebGpuCoverageTuning(tuning = null) {
  return {
    mode: WEBGPU_COVERAGE_TUNING_MODE,
    footprintScale: clampNumber(
      Number(tuning?.footprintScale ?? WEBGPU_COVERAGE_TUNING_DEFAULTS.footprintScale),
      1.2,
      4.8,
    ),
    maxAnisotropy: clampNumber(
      Number(tuning?.maxAnisotropy ?? WEBGPU_COVERAGE_TUNING_DEFAULTS.maxAnisotropy),
      1.5,
      8,
    ),
  };
}

export function normalizeWebGpuColorTuning(tuning = null) {
  const requested = String(tuning?.colorMode ?? WEBGPU_COLOR_MODE_DEFAULT);
  const colorMode =
    requested === WEBGPU_COLOR_MODE_SH_VIEW
      ? WEBGPU_COLOR_MODE_SH_VIEW
      : WEBGPU_COLOR_MODE_SOURCE;
  return {
    mode: WEBGPU_COLOR_TUNING_MODE,
    colorMode,
  };
}

export function normalizeWebGpuTileListMode(mode = WEBGPU_TILE_LIST_MODE_VISIBLE) {
  return String(mode) === WEBGPU_TILE_LIST_MODE_OBJECT_STATE
    ? WEBGPU_TILE_LIST_MODE_OBJECT_STATE
    : WEBGPU_TILE_LIST_MODE_VISIBLE;
}

export {
  normalizeWebGpuCameraTuning,
  WEBGPU_CAMERA_MODE_EDIT_FIXED,
  WEBGPU_CAMERA_MODE_SPARK_FRAME,
  WEBGPU_CAMERA_TUNING_MODE,
  WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED,
  WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K,
  normalizeWebGpuDepthSortTuning,
  WEBGPU_DEPTH_BIN_COUNT_DEFAULT,
  WEBGPU_DEPTH_SORT_TUNING_MODE,
};

export function buildWebGpuTileProjectionBounds(
  points,
  viewportWidth = WEBGPU_TILE_VIEWPORT.width,
  viewportHeight = WEBGPU_TILE_VIEWPORT.height,
  cameraTuning = null,
) {
  return sceneBounds(
    points,
    viewportWidth,
    viewportHeight,
    normalizeWebGpuCameraTuning(cameraTuning),
  );
}

export function projectPointToWebGpuTileViewport({
  point,
  bounds,
  viewportWidth = WEBGPU_TILE_VIEWPORT.width,
  viewportHeight = WEBGPU_TILE_VIEWPORT.height,
}) {
  return projectPointToSmokeViewport({ point, bounds, viewportWidth, viewportHeight });
}

function pixelDepthSortModeForAlphaMode(pixelDepthAlphaMode) {
  return pixelDepthAlphaMode === WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K
    ? WEBGPU_PIXEL_DEPTH_SORT_MODE_FRONT_TOP_K
    : WEBGPU_PIXEL_DEPTH_SORT_MODE_DEPTH_BINNED;
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
  coverageTuning,
  tileListMode,
  tileOffsets,
  tileEntries,
}) {
  const binHiddenGaussians = tileListMode === WEBGPU_TILE_LIST_MODE_OBJECT_STATE;
  const cursors = new Uint32Array(tileOffsets);
  points.forEach((point, index) => {
    const objectDenseIndex = objectIndex.objectIndexById.get(point.objectId) ?? 0;
    if (!binHiddenGaussians && !objectIsVisible(objectState, objectDenseIndex)) return;
    const scale = pointScale(point);
    const screen = projectPointToSmokeViewport({ point, bounds, viewportWidth, viewportHeight });
    const screenCovariance = projectPointScreenCovariance({
      point,
      scale,
      screen,
      bounds,
      viewportWidth,
      viewportHeight,
      coverageTuning,
    });
    const radiusPixels = pointRadiusPixels({
      screen,
      screenCovariance,
      pointSize,
      coverageTuning,
    });
    if (!screenInfluencesViewport({ screen, radiusPixels, viewportWidth, viewportHeight })) return;
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
  bounds,
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
  const rotation = scaleRotation[scaleOffset + 2];
  const sigmaX = Math.max(scaleRotation[scaleOffset], 0.0001);
  const sigmaY = Math.max(scaleRotation[scaleOffset + 1], 0.0001);
  const depthWeight = frontWeightedOitDepth(screen.depth, bounds.depthMin, bounds.depthSpan);
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
      depthWeight *
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
  depthMin,
  depthSpan,
  tileColumns,
  tileSize,
  viewportWidth,
  viewportHeight,
  pixelDepthBinCount = WEBGPU_DEPTH_BIN_COUNT_DEFAULT,
  pixelDepthAlphaMode = WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED,
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
    depthMin,
    depthSpan,
  });
  let pixelResolvedCount = 0;
  let checksum = 2166136261;
  const binRed = new Float32Array(pixelDepthBinCount);
  const binGreen = new Float32Array(pixelDepthBinCount);
  const binBlue = new Float32Array(pixelDepthBinCount);
  const binWeight = new Float32Array(pixelDepthBinCount);
  const topDepth = new Float32Array(pixelDepthBinCount);
  const topRed = new Float32Array(pixelDepthBinCount);
  const topGreen = new Float32Array(pixelDepthBinCount);
  const topBlue = new Float32Array(pixelDepthBinCount);
  const topWeight = new Float32Array(pixelDepthBinCount);
  const useFrontTopKAlpha = pixelDepthAlphaMode === WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K;

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
      let candidateCount = 0;
      binRed.fill(0);
      binGreen.fill(0);
      binBlue.fill(0);
      binWeight.fill(0);
      topDepth.fill(Number.POSITIVE_INFINITY);
      topRed.fill(0);
      topGreen.fill(0);
      topBlue.fill(0);
      topWeight.fill(0);

      for (let entryOffset = 0; entryOffset < storedCount; entryOffset += 1) {
        const gaussianIndex = tileEntries[entryBase + entryOffset];
        const objectDenseIndex = objectIndices[gaussianIndex];
        if (!objectIsVisible(objectState, objectDenseIndex)) continue;

        const d = pixelGaussianDistance({
          x,
          y,
          gaussianIndex,
          projection,
        });
        if (d > RESOLVE_KERNEL_CUTOFF) continue;

        const gaussianOffset = gaussianIndex * 4;
        const candidateWeight =
          Math.exp(-0.5 * d) * colorOpacity[gaussianOffset + 3] * RESOLVE_ALPHA_GAIN;
        if (candidateWeight <= PIXEL_COVERAGE_WEIGHT_FLOOR) continue;
        const normalizedDepth = clampNumber(
          (projection.depth[gaussianIndex] - depthMin) / Math.max(depthSpan, 0.0001),
          0,
          0.999999,
        );
        if (useFrontTopKAlpha) {
          let insertDepth = normalizedDepth;
          let insertRed = colorOpacity[gaussianOffset] * candidateWeight;
          let insertGreen = colorOpacity[gaussianOffset + 1] * candidateWeight;
          let insertBlue = colorOpacity[gaussianOffset + 2] * candidateWeight;
          let insertWeight = candidateWeight;
          for (let slot = 0; slot < pixelDepthBinCount; slot += 1) {
            if (insertDepth >= topDepth[slot]) continue;
            const previousDepth = topDepth[slot];
            const previousRed = topRed[slot];
            const previousGreen = topGreen[slot];
            const previousBlue = topBlue[slot];
            const previousWeight = topWeight[slot];
            topDepth[slot] = insertDepth;
            topRed[slot] = insertRed;
            topGreen[slot] = insertGreen;
            topBlue[slot] = insertBlue;
            topWeight[slot] = insertWeight;
            insertDepth = previousDepth;
            insertRed = previousRed;
            insertGreen = previousGreen;
            insertBlue = previousBlue;
            insertWeight = previousWeight;
          }
        } else {
          const depthBin = Math.min(
            pixelDepthBinCount - 1,
            Math.floor(normalizedDepth * pixelDepthBinCount),
          );
          binRed[depthBin] += colorOpacity[gaussianOffset] * candidateWeight;
          binGreen[depthBin] += colorOpacity[gaussianOffset + 1] * candidateWeight;
          binBlue[depthBin] += colorOpacity[gaussianOffset + 2] * candidateWeight;
          binWeight[depthBin] += candidateWeight;
        }
        candidateCount += 1;
      }

      if (candidateCount <= 0) continue;

      let outputRedPremultiplied = 0;
      let outputGreenPremultiplied = 0;
      let outputBluePremultiplied = 0;
      let outputAlpha = 0;
      let totalWeight = 0;
      if (useFrontTopKAlpha) {
        for (let slot = 0; slot < pixelDepthBinCount; slot += 1) {
          const weight = topWeight[slot];
          if (weight <= PIXEL_COVERAGE_WEIGHT_FLOOR) continue;
          const red = topRed[slot] / weight;
          const green = topGreen[slot] / weight;
          const blue = topBlue[slot] / weight;
          const alpha = clampNumber(1 - Math.exp(-weight * RESOLVE_ALPHA_SCALE), 0, 0.98);
          const visibility = 1 - outputAlpha;
          outputRedPremultiplied += visibility * red * alpha;
          outputGreenPremultiplied += visibility * green * alpha;
          outputBluePremultiplied += visibility * blue * alpha;
          outputAlpha += visibility * alpha;
          totalWeight += weight;
          if (outputAlpha >= 0.995) break;
        }
      } else {
        for (let depthBin = 0; depthBin < pixelDepthBinCount; depthBin += 1) {
          const weight = binWeight[depthBin];
          if (weight <= PIXEL_COVERAGE_WEIGHT_FLOOR) continue;
          const red = binRed[depthBin] / weight;
          const green = binGreen[depthBin] / weight;
          const blue = binBlue[depthBin] / weight;
          const alpha = clampNumber(1 - Math.exp(-weight * RESOLVE_ALPHA_SCALE), 0, 0.98);
          const visibility = 1 - outputAlpha;
          outputRedPremultiplied += visibility * red * alpha;
          outputGreenPremultiplied += visibility * green * alpha;
          outputBluePremultiplied += visibility * blue * alpha;
          outputAlpha += visibility * alpha;
          totalWeight += weight;
          if (outputAlpha >= 0.995) break;
        }
      }

      if (outputAlpha <= 0.0001 || totalWeight <= 0.0001) continue;

      const red = outputRedPremultiplied / Math.max(outputAlpha, 0.0001);
      const green = outputGreenPremultiplied / Math.max(outputAlpha, 0.0001);
      const blue = outputBluePremultiplied / Math.max(outputAlpha, 0.0001);
      const alpha = clampNumber(outputAlpha, 0, 0.98);
      pixelResolvedRgba[pixelOffset] = red;
      pixelResolvedRgba[pixelOffset + 1] = green;
      pixelResolvedRgba[pixelOffset + 2] = blue;
      pixelResolvedRgba[pixelOffset + 3] = alpha;
      pixelResolvedCount += 1;
      checksum = checksumValue(checksum, red, green, blue, alpha, totalWeight);
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
  depthMin,
  depthSpan,
}) {
  const gaussianCount = positionRadius.length / 4;
  const screenX = new Float32Array(gaussianCount);
  const screenY = new Float32Array(gaussianCount);
  const sigmaXSquared = new Float32Array(gaussianCount);
  const sigmaYSquared = new Float32Array(gaussianCount);
  const depth = new Float32Array(gaussianCount);
  const depthWeight = new Float32Array(gaussianCount);
  const cosine = new Float32Array(gaussianCount);
  const sine = new Float32Array(gaussianCount);

  for (let index = 0; index < gaussianCount; index += 1) {
    const offset = index * 4;
    screenX[index] = positionRadius[offset];
    screenY[index] = positionRadius[offset + 1];
    const sigmaX = Math.max(scaleRotation[offset], 0.0001);
    const sigmaY = Math.max(scaleRotation[offset + 1], 0.0001);
    sigmaXSquared[index] = sigmaX * sigmaX;
    sigmaYSquared[index] = sigmaY * sigmaY;
    depth[index] = positionRadius[offset + 2];
    depthWeight[index] = frontWeightedOitDepth(depth[index], depthMin, depthSpan);
    const rotation = scaleRotation[offset + 2];
    cosine[index] = Math.cos(rotation);
    sine[index] = Math.sin(rotation);
  }

  return { screenX, screenY, sigmaXSquared, sigmaYSquared, depth, depthWeight, cosine, sine };
}

function pixelGaussianDistance({ x, y, gaussianIndex, projection }) {
  const dx = x + 0.5 - projection.screenX[gaussianIndex];
  const dy = y + 0.5 - projection.screenY[gaussianIndex];
  const cosine = projection.cosine[gaussianIndex];
  const sine = projection.sine[gaussianIndex];
  const rotatedX = cosine * dx - sine * dy;
  const rotatedY = sine * dx + cosine * dy;
  return (
    (rotatedX * rotatedX) / projection.sigmaXSquared[gaussianIndex] +
    (rotatedY * rotatedY) / projection.sigmaYSquared[gaussianIndex]
  );
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

function sceneBounds(points, viewportWidth = 1, viewportHeight = 1, cameraTuning = null) {
  const resolvedCameraTuning = normalizeWebGpuCameraTuning(cameraTuning);
  const cameraFrame = cameraFrameForPoints(points, resolvedCameraTuning);
  const cameraProjection = editCameraProjection(viewportWidth, viewportHeight, cameraFrame);
  if (points.length === 0) {
    return {
      minX: 0,
      maxX: Math.max(1, viewportWidth),
      minZ: 0,
      maxZ: Math.max(1, viewportHeight),
      spanX: Math.max(1, viewportWidth),
      spanZ: Math.max(1, viewportHeight),
      fitMode: "empty-default",
      paddingRatio: 0,
      viewportAspect: cameraProjection.viewportAspect,
      worldAspect: cameraProjection.viewportAspect,
      projectionMode: cameraFrame.projectionMode,
      cameraTuningMode: WEBGPU_CAMERA_TUNING_MODE,
      cameraMode: cameraFrame.cameraMode,
      cameraFovDegrees: cameraFrame.fovDegrees,
      cameraPosition: cameraFrame.eye,
      cameraTarget: cameraFrame.target,
      cameraDistance: cameraFrame.distance,
      cameraFrameMaxDim: cameraFrame.frameMaxDim,
      depthMin: 0,
      depthMax: 1,
      depthSpan: 1,
      cameraProjection,
    };
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  let depthMin = Infinity;
  let depthMax = -Infinity;
  for (const point of points) {
    const screen = projectPointWithCamera(point, cameraProjection);
    minX = Math.min(minX, screen.x);
    maxX = Math.max(maxX, screen.x);
    minZ = Math.min(minZ, screen.y);
    maxZ = Math.max(maxZ, screen.y);
    depthMin = Math.min(depthMin, screen.depth);
    depthMax = Math.max(depthMax, screen.depth);
  }

  return fitBoundsToViewport({
    minX,
    maxX,
    minZ,
    maxZ,
    viewportWidth,
    viewportHeight,
    depthMin,
    depthMax,
    cameraProjection,
  });
}

function fitBoundsToViewport({
  minX,
  maxX,
  minZ,
  maxZ,
  viewportWidth,
  viewportHeight,
  depthMin,
  depthMax,
  cameraProjection,
}) {
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
    projectionMode: cameraProjection.projectionMode,
    cameraTuningMode: cameraProjection.cameraTuningMode,
    cameraMode: cameraProjection.cameraMode,
    cameraFovDegrees: cameraProjection.fovDegrees,
    cameraPosition: cameraProjection.eye,
    cameraTarget: cameraProjection.target,
    cameraDistance: cameraProjection.distance,
    cameraFrameMaxDim: cameraProjection.frameMaxDim,
    depthMin,
    depthMax,
    depthSpan: Math.max(depthMax - depthMin, 0.0001),
    cameraProjection,
  };
}

function cameraFrameForPoints(points, cameraTuning) {
  if (cameraTuning.cameraMode === WEBGPU_CAMERA_MODE_SPARK_FRAME && points.length > 0) {
    return sparkFrameCamera(points);
  }
  return {
    cameraMode: WEBGPU_CAMERA_MODE_EDIT_FIXED,
    projectionMode: WEBGPU_TILE_PROJECTION_MODE,
    eye: EDIT_CAMERA_POSITION,
    target: EDIT_CAMERA_TARGET,
    fovDegrees: EDIT_CAMERA_FOV_DEGREES,
    distance: Math.hypot(
      EDIT_CAMERA_POSITION[0] - EDIT_CAMERA_TARGET[0],
      EDIT_CAMERA_POSITION[1] - EDIT_CAMERA_TARGET[1],
      EDIT_CAMERA_POSITION[2] - EDIT_CAMERA_TARGET[2],
    ),
    frameMaxDim: 0,
  };
}

function sparkFrameCamera(points) {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (const point of points) {
    const world = pointRenderWorld(point);
    minX = Math.min(minX, world[0]);
    minY = Math.min(minY, world[1]);
    minZ = Math.min(minZ, world[2]);
    maxX = Math.max(maxX, world[0]);
    maxY = Math.max(maxY, world[1]);
    maxZ = Math.max(maxZ, world[2]);
  }
  const center = [
    (minX + maxX) * 0.5,
    (minY + maxY) * 0.5,
    (minZ + maxZ) * 0.5,
  ];
  const size = [
    Math.max(maxX - minX, 0),
    Math.max(maxY - minY, 0),
    Math.max(maxZ - minZ, 0),
  ];
  const frameMaxDim = Math.max(size[0], size[1], size[2], 0.5);
  const distance = frameMaxDim * SPARK_FRAME_DISTANCE_MULTIPLIER;
  return {
    cameraMode: WEBGPU_CAMERA_MODE_SPARK_FRAME,
    projectionMode: WEBGPU_TILE_SPARK_FRAME_PROJECTION_MODE,
    eye: [
      center[0] + distance,
      center[1] + distance * SPARK_FRAME_HEIGHT_MULTIPLIER,
      center[2] + distance,
    ],
    target: center,
    fovDegrees: SPARK_FRAME_CAMERA_FOV_DEGREES,
    distance,
    frameMaxDim,
  };
}

function editCameraProjection(viewportWidth, viewportHeight, cameraFrame) {
  const eye = cameraFrame.eye;
  const target = cameraFrame.target;
  const forward = normalize3(subtract3(target, eye));
  const right = normalize3(cross3(forward, EDIT_CAMERA_UP));
  const up = normalize3(cross3(right, forward));
  const viewportAspect = Math.min(
    4,
    Math.max(0.25, Math.max(1, viewportWidth) / Math.max(1, viewportHeight)),
  );
  return {
    eye,
    target,
    forward,
    right,
    up,
    viewportWidth: Math.max(1, viewportWidth),
    viewportHeight: Math.max(1, viewportHeight),
    viewportAspect,
    tanHalfFovY: Math.tan((cameraFrame.fovDegrees * Math.PI) / 360),
    projectionMode: cameraFrame.projectionMode,
    cameraTuningMode: WEBGPU_CAMERA_TUNING_MODE,
    cameraMode: cameraFrame.cameraMode,
    fovDegrees: cameraFrame.fovDegrees,
    distance: cameraFrame.distance,
    frameMaxDim: cameraFrame.frameMaxDim,
  };
}

function projectPointWithCamera(point, cameraProjection) {
  const world = pointRenderWorld(point);
  const cameraDelta = subtract3(world, cameraProjection.eye);
  const depth = Math.max(0.01, dot3(cameraDelta, cameraProjection.forward));
  const viewX = dot3(cameraDelta, cameraProjection.right);
  const viewY = dot3(cameraDelta, cameraProjection.up);
  const ndcX = viewX / (depth * cameraProjection.tanHalfFovY * cameraProjection.viewportAspect);
  const ndcY = viewY / (depth * cameraProjection.tanHalfFovY);
  const viewportWidth = cameraProjection.viewportWidth;
  const viewportHeight = cameraProjection.viewportHeight;
  return {
    x: (ndcX * 0.5 + 0.5) * Math.max(1, viewportWidth - 1),
    y: (0.5 - ndcY * 0.5) * Math.max(1, viewportHeight - 1),
    depth,
    viewX,
    viewY,
    pixelsPerWorldUnit: viewportHeight * 0.5 / (cameraProjection.tanHalfFovY * depth),
  };
}

function pointRenderWorld(point) {
  return [point.x, point.z, point.y];
}

function subtract3(left, right) {
  return [
    left[0] - right[0],
    left[1] - right[1],
    left[2] - right[2],
  ];
}

function dot3(left, right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function cross3(left, right) {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function normalize3(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function packGaussian({
  point,
  index,
  objectDenseIndex,
  radiusPixels,
  screen,
  screenCovariance,
  rgb,
  positionRadius,
  colorOpacity,
  scaleRotation,
  objectIndices,
}) {
  const vec4Offset = index * 4;
  positionRadius[vec4Offset] = screen.x;
  positionRadius[vec4Offset + 1] = screen.y;
  positionRadius[vec4Offset + 2] = screen.depth;
  positionRadius[vec4Offset + 3] = radiusPixels;

  colorOpacity[vec4Offset] = rgb[0] / 255;
  colorOpacity[vec4Offset + 1] = rgb[1] / 255;
  colorOpacity[vec4Offset + 2] = rgb[2] / 255;
  colorOpacity[vec4Offset + 3] = clampNumber(point.opacity ?? 1, 0, 1);

  scaleRotation[vec4Offset] = screenCovariance.sigmaMajor;
  scaleRotation[vec4Offset + 1] = screenCovariance.sigmaMinor;
  scaleRotation[vec4Offset + 2] = screenCovariance.rotation;
  scaleRotation[vec4Offset + 3] = 0;
  objectIndices[index] = objectDenseIndex;
}

function renderColorSource(point, renderMode) {
  if (renderMode !== "original") return "object-palette";
  const source = String(point.colorSource ?? "fallback");
  return ["rgb", "sh-dc", "fallback"].includes(source) ? source : "fallback";
}

function pointRenderColor({
  point,
  index,
  renderMode,
  colorTuning,
  shRestCoefficients,
  shRestCoefficientCount,
  cameraProjection,
}) {
  if (renderMode !== "original") {
    return { rgb: point.objectColor, mode: "object-palette" };
  }
  if (
    colorTuning?.colorMode === WEBGPU_COLOR_MODE_SH_VIEW &&
    pointCanUseShViewColor({
      point,
      index,
      shRestCoefficients,
      shRestCoefficientCount,
    })
  ) {
    return {
      rgb: shViewColor({
        point,
        index,
        shRestCoefficients,
        shRestCoefficientCount,
        cameraProjection,
      }),
      mode: WEBGPU_COLOR_MODE_SH_VIEW,
    };
  }
  return { rgb: point.color, mode: WEBGPU_COLOR_MODE_SOURCE };
}

function pointCanUseShViewColor({ point, index, shRestCoefficients, shRestCoefficientCount }) {
  if (!Array.isArray(point.shDc) || point.shDc.length < 3) return false;
  const coefficientCount = pointShRestCoefficientCount(point);
  if (coefficientCount <= 0 || shRestCoefficientCount <= 0) return false;
  if (!shRestCoefficients || typeof shRestCoefficients.length !== "number") return false;
  const offset = index * shRestCoefficientCount;
  return offset >= 0 && offset + Math.min(coefficientCount, shRestCoefficientCount) <= shRestCoefficients.length;
}

function shViewColor({ point, index, shRestCoefficients, shRestCoefficientCount, cameraProjection }) {
  const direction = normalize3(subtract3(pointRenderWorld(point), cameraProjection.eye));
  return [0, 1, 2].map((channel) =>
    shChannelToRgb(
      evaluateShChannel({
        dc: Number(point.shDc[channel] ?? 0),
        rest: shRestCoefficients,
        restOffset: index * shRestCoefficientCount,
        restCount: Math.min(pointShRestCoefficientCount(point), shRestCoefficientCount),
        channel,
        direction,
      }),
    ),
  );
}

function evaluateShChannel({ dc, rest, restOffset, restCount, channel, direction }) {
  const x = direction[0];
  const y = direction[1];
  const z = direction[2];
  let value = SH_C0 * dc;
  if (restCount >= 9) {
    value +=
      -SH_C1 * y * shRestCoefficient(rest, restOffset, restCount, 1, channel) +
      SH_C1 * z * shRestCoefficient(rest, restOffset, restCount, 2, channel) +
      -SH_C1 * x * shRestCoefficient(rest, restOffset, restCount, 3, channel);
  }
  if (restCount >= 24) {
    value +=
      SH_C2[0] * x * y * shRestCoefficient(rest, restOffset, restCount, 4, channel) +
      SH_C2[1] * y * z * shRestCoefficient(rest, restOffset, restCount, 5, channel) +
      SH_C2[2] * (2 * z * z - x * x - y * y) *
        shRestCoefficient(rest, restOffset, restCount, 6, channel) +
      SH_C2[3] * x * z * shRestCoefficient(rest, restOffset, restCount, 7, channel) +
      SH_C2[4] * (x * x - y * y) * shRestCoefficient(rest, restOffset, restCount, 8, channel);
  }
  if (restCount >= 45) {
    value +=
      SH_C3[0] * y * (3 * x * x - y * y) *
        shRestCoefficient(rest, restOffset, restCount, 9, channel) +
      SH_C3[1] * x * y * z * shRestCoefficient(rest, restOffset, restCount, 10, channel) +
      SH_C3[2] * y * (4 * z * z - x * x - y * y) *
        shRestCoefficient(rest, restOffset, restCount, 11, channel) +
      SH_C3[3] * z * (2 * z * z - 3 * x * x - 3 * y * y) *
        shRestCoefficient(rest, restOffset, restCount, 12, channel) +
      SH_C3[4] * x * (4 * z * z - x * x - y * y) *
        shRestCoefficient(rest, restOffset, restCount, 13, channel) +
      SH_C3[5] * z * (x * x - y * y) *
        shRestCoefficient(rest, restOffset, restCount, 14, channel) +
      SH_C3[6] * x * (x * x - 3 * y * y) *
        shRestCoefficient(rest, restOffset, restCount, 15, channel);
  }
  return value;
}

function shRestCoefficient(rest, restOffset, restCount, basisIndex, channel) {
  const coefficientIndex = (basisIndex - 1) * 3 + channel;
  if (coefficientIndex < 0 || coefficientIndex >= restCount) return 0;
  const value = Number(rest[restOffset + coefficientIndex] ?? 0);
  return Number.isFinite(value) ? value : 0;
}

function shChannelToRgb(value) {
  return Math.round(clampNumber(value + 0.5, 0, 1) * 255);
}

function pointShRestCoefficientCount(point) {
  const count = Number(point.shRestCoefficientCount ?? 0);
  if (!Number.isFinite(count) || count <= 0) return 0;
  return Math.trunc(count);
}

function pointShDegree(point, coefficientCount) {
  const explicit = Number(point.shDegree ?? 0);
  if (Number.isFinite(explicit) && explicit > 0) {
    return Math.trunc(explicit);
  }
  if (coefficientCount >= 45) return 3;
  if (coefficientCount >= 24) return 2;
  if (coefficientCount >= 9) return 1;
  return 0;
}

function projectPointToSmokeViewport({ point, bounds, viewportWidth, viewportHeight }) {
  const cameraScreen = projectPointWithCamera(point, bounds.cameraProjection);
  return {
    x: ((cameraScreen.x - bounds.minX) / bounds.spanX) * Math.max(1, viewportWidth - 1),
    y: ((cameraScreen.y - bounds.minZ) / bounds.spanZ) * Math.max(1, viewportHeight - 1),
    depth: cameraScreen.depth,
    viewX: cameraScreen.viewX,
    viewY: cameraScreen.viewY,
    pixelsPerWorldUnit:
      cameraScreen.pixelsPerWorldUnit *
      Math.min(
        viewportWidth / Math.max(bounds.spanX, 0.0001),
        viewportHeight / Math.max(bounds.spanZ, 0.0001),
      ),
  };
}

function projectPointScreenCovariance({
  point,
  scale,
  screen,
  bounds,
  viewportWidth,
  viewportHeight,
  coverageTuning,
}) {
  const scale3 = pointScale3(point);
  const rotationQuaternion = pointRotationQuaternion(point);
  if (!scale3 || !rotationQuaternion) {
    return legacyScreenCovariance({ point, scale, screen, coverageTuning });
  }

  const rows = screenJacobianRows({ screen, bounds, viewportWidth, viewportHeight });
  const axes = [
    mapRawVectorToRenderWorld(quaternionRotate(rotationQuaternion, [1, 0, 0])),
    mapRawVectorToRenderWorld(quaternionRotate(rotationQuaternion, [0, 1, 0])),
    mapRawVectorToRenderWorld(quaternionRotate(rotationQuaternion, [0, 0, 1])),
  ];
  let covarianceXx = 0;
  let covarianceXy = 0;
  let covarianceYy = 0;

  for (let axisIndex = 0; axisIndex < axes.length; axisIndex += 1) {
    const sigmaWorld = scale3[axisIndex] * coverageTuning.footprintScale / 3;
    const axis = axes[axisIndex];
    const projectedX = dot3(rows.x, axis) * sigmaWorld;
    const projectedY = dot3(rows.y, axis) * sigmaWorld;
    covarianceXx += projectedX * projectedX;
    covarianceXy += projectedX * projectedY;
    covarianceYy += projectedY * projectedY;
  }

  return ellipseFromCovariance(
    covarianceXx,
    covarianceXy,
    covarianceYy,
    "full",
    coverageTuning,
  );
}

function legacyScreenCovariance({ point, scale, screen, coverageTuning }) {
  return {
    sigmaMajor: Math.max((scale[0] * screen.pixelsPerWorldUnit * coverageTuning.footprintScale) / 3, 0.0001),
    sigmaMinor: Math.max((scale[1] * screen.pixelsPerWorldUnit * coverageTuning.footprintScale) / 3, 0.0001),
    rotation: Number.isFinite(point.rotation) ? point.rotation : 0,
    clamped: false,
    mode: "fallback",
  };
}

function screenJacobianRows({ screen, bounds, viewportWidth, viewportHeight }) {
  const camera = bounds.cameraProjection;
  const depth = Math.max(screen.depth, 0.0001);
  const depthSquared = depth * depth;
  const cameraPixelScaleX =
    Math.max(1, camera.viewportWidth - 1) * 0.5 /
    (camera.tanHalfFovY * camera.viewportAspect);
  const cameraPixelScaleY =
    Math.max(1, camera.viewportHeight - 1) * 0.5 /
    camera.tanHalfFovY;
  const fitScaleX = Math.max(1, viewportWidth - 1) / Math.max(bounds.spanX, 0.0001);
  const fitScaleY = Math.max(1, viewportHeight - 1) / Math.max(bounds.spanZ, 0.0001);
  const rowX = camera.right.map((value, index) =>
    fitScaleX * cameraPixelScaleX *
    (value / depth - (screen.viewX * camera.forward[index]) / depthSquared),
  );
  const rowY = camera.up.map((value, index) =>
    -fitScaleY * cameraPixelScaleY *
    (value / depth - (screen.viewY * camera.forward[index]) / depthSquared),
  );
  return { x: rowX, y: rowY };
}

function ellipseFromCovariance(covarianceXx, covarianceXy, covarianceYy, mode, coverageTuning) {
  const trace = (covarianceXx + covarianceYy) * 0.5;
  const spread = Math.hypot((covarianceXx - covarianceYy) * 0.5, covarianceXy);
  const lambdaMajor = Math.max(trace + spread, 0.000001);
  const lambdaMinor = Math.max(trace - spread, 0.000001);
  const sigmaMinor = Math.sqrt(lambdaMinor);
  const rawSigmaMajor = Math.sqrt(lambdaMajor);
  const sigmaMajor = Math.min(
    rawSigmaMajor,
    Math.max(sigmaMinor * coverageTuning.maxAnisotropy, sigmaMinor),
  );
  return {
    sigmaMajor,
    sigmaMinor,
    rotation: 0.5 * Math.atan2(2 * covarianceXy, covarianceXx - covarianceYy),
    clamped: sigmaMajor < rawSigmaMajor,
    mode,
  };
}

function mapRawVectorToRenderWorld(vector) {
  return [vector[0], vector[2], vector[1]];
}

function quaternionRotate(quaternion, vector) {
  const [w, x, y, z] = quaternion;
  const [vx, vy, vz] = vector;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

function frontWeightedOitDepth(depth, depthMin, depthSpan) {
  const normalizedDepth = clampNumber(
    (depth - depthMin) / Math.max(depthSpan, 0.0001),
    0,
    1,
  );
  return clampNumber(
    DEPTH_WEIGHT_FLOOR + (1 - DEPTH_WEIGHT_FLOOR) * Math.exp(-DEPTH_WEIGHT_STRENGTH * normalizedDepth),
    DEPTH_WEIGHT_FLOOR,
    1,
  );
}

function frontDepthGate(depth, nearestDepth, depthSpan) {
  const depthDelta = Math.max(depth - nearestDepth, 0);
  return clampNumber(
    FRONT_DEPTH_GATE_FLOOR +
      (1 - FRONT_DEPTH_GATE_FLOOR) *
        Math.exp(-FRONT_DEPTH_GATE_STRENGTH * depthDelta / Math.max(depthSpan, 0.0001)),
    FRONT_DEPTH_GATE_FLOOR,
    1,
  );
}

function pointRadiusPixels({ screen, screenCovariance, pointSize, coverageTuning }) {
  const pointSizeRadius = pointSize * screen.pixelsPerWorldUnit * coverageTuning.footprintScale;
  return clampNumber(Math.max(screenCovariance.sigmaMajor * 3, pointSizeRadius), 1.5, 96);
}

function screenInfluencesViewport({ screen, radiusPixels, viewportWidth, viewportHeight }) {
  return (
    screen.x + radiusPixels >= 0 &&
    screen.x - radiusPixels <= Math.max(1, viewportWidth - 1) &&
    screen.y + radiusPixels >= 0 &&
    screen.y - radiusPixels <= Math.max(1, viewportHeight - 1)
  );
}

function pointScale(point) {
  if (Array.isArray(point.scale) && point.scale.length > 0) {
    const first = safeScale(point.scale[0]);
    const second = safeScale(point.scale[1] ?? first);
    return [first, second];
  }
  return [0.018, 0.018];
}

function pointScale3(point) {
  if (!Array.isArray(point.scale3) || point.scale3.length < 3) return null;
  return [
    safeScale(point.scale3[0]),
    safeScale(point.scale3[1]),
    safeScale(point.scale3[2]),
  ];
}

function pointRotationQuaternion(point) {
  if (!Array.isArray(point.rotationQuaternion) || point.rotationQuaternion.length < 4) {
    return null;
  }
  const values = point.rotationQuaternion.map(Number);
  const length = Math.hypot(values[0], values[1], values[2], values[3]);
  if (!Number.isFinite(length) || length <= 0.0001) return null;
  return values.map((value) => value / length);
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
