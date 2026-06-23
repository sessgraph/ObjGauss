import { estimateWebGpuTileRuntimeStorage } from "./webgpuTileStorage.js";
import {
  WEBGPU_CAMERA_MODE_DEFAULT,
  WEBGPU_CAMERA_TUNING_MODE,
} from "./webgpuCameraTuning.js";
import {
  WEBGPU_DEPTH_BIN_COUNT_DEFAULT,
  WEBGPU_DEPTH_SORT_TUNING_MODE,
} from "./webgpuDepthTuning.js";
import {
  WEBGPU_PIXEL_COVERAGE_MODE,
  WEBGPU_TILE_DEPTH_WEIGHT_MODE,
  WEBGPU_PIXEL_DEPTH_SORT_MODE,
  WEBGPU_TILE_PROJECTION_MODE,
  WEBGPU_TILE_COLOR_FIDELITY_MODE,
  WEBGPU_TILE_SCREEN_COVARIANCE_MODE,
} from "./webgpuTileSmoke.js";

export const WEBGPU_TILE_RENDERER_ID = "webgpu-tile";
export const WEBGPU_TILE_RENDERER_LABEL = "WebGPU Tile 编辑";
export const WEBGPU_TILE_OBJECT_FILTER = "gpu-object-state-buffer";
export const GAUSSIAN_OIT_RENDERER_ID = "gaussian-oit";
export const GAUSSIAN_OIT_RENDERER_LABEL = "Gaussian OIT 编辑";
export const GAUSSIAN_OIT_OBJECT_FILTER = "gpu-object-state-texture";
export const WEBGPU_TILE_REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE = 9;

export const INITIAL_WEBGPU_CAPABILITY = Object.freeze({
  status: "pending",
  reason: "webgpu-capability-detecting",
  label: "检测中",
});
const EMPTY_TILE_SMOKE = Object.freeze({
  layoutVersion: "webgpu-tile-smoke-v1",
  tileSize: 16,
  viewportWidth: 1024,
  viewportHeight: 1024,
  boundsMinX: -1,
  boundsMinZ: -1,
  boundsSpanX: 2,
  boundsSpanZ: 2,
  boundsFitMode: "empty-default",
  boundsPaddingRatio: 0,
  boundsViewportAspect: 1,
  boundsWorldAspect: 1,
  projectionMode: WEBGPU_TILE_PROJECTION_MODE,
  projectionCameraTuningMode: WEBGPU_CAMERA_TUNING_MODE,
  projectionCameraMode: WEBGPU_CAMERA_MODE_DEFAULT,
  projectionCameraFovDegrees: 52,
  projectionCameraPosition: [3.6, 2.8, 3.4],
  projectionCameraTarget: [0, 0, 0.25],
  projectionCameraDistance: 5.542788,
  projectionCameraFrameMaxDim: 0,
  projectionDepthMin: 0,
  projectionDepthMax: 1,
  projectionDepthSpan: 1,
  depthWeightMode: WEBGPU_TILE_DEPTH_WEIGHT_MODE,
  pixelDepthSortMode: WEBGPU_PIXEL_DEPTH_SORT_MODE,
  pixelDepthTuningMode: WEBGPU_DEPTH_SORT_TUNING_MODE,
  pixelDepthGateStrength: 12,
  pixelDepthGateFloor: 0.06,
  pixelDepthBinCount: WEBGPU_DEPTH_BIN_COUNT_DEFAULT,
  pixelCoverageMode: WEBGPU_PIXEL_COVERAGE_MODE,
  pixelCoverageWeightFloor: 0.004,
  pixelCoverageFootprintScale: 2.2,
  pixelCoverageTuningMode: "runtime-coverage-tuning-v1",
  colorFidelityMode: WEBGPU_TILE_COLOR_FIDELITY_MODE,
  colorSourceRgbGaussians: 0,
  colorSourceShDcGaussians: 0,
  colorSourceFallbackGaussians: 0,
  colorSourceObjectGaussians: 0,
  colorOpacityMean: 0,
  screenCovarianceMode: WEBGPU_TILE_SCREEN_COVARIANCE_MODE,
  screenCovarianceGaussians: 0,
  screenCovarianceFallbackGaussians: 0,
  screenCovarianceClampedGaussians: 0,
  screenCovarianceMaxAnisotropy: 4,
  screenCovarianceSigmaMean: 0,
  packedGaussians: 0,
  binnedGaussians: 0,
  visibleGaussians: 0,
  tileCount: 0,
  activeTileCount: 0,
  tileReferenceCount: 0,
  tileOverflowCount: 0,
  tileOverflowTileCount: 0,
  tileOverflowRatio: 0,
  tileOverflowMaxExcess: 0,
  tileEntryStoredCount: 0,
  tileEntryCapacity: 0,
  tileEntryUtilization: 0,
  tileEntryLayout: "compact-offset-list",
  tileEntryOffsetCount: 0,
  tileCapacityMode: "compact-offset-list",
  tileCapacityStatus: "ok",
  tileCapacityGate: "pass",
  maxTileOccupancy: 0,
  resolveVersion: "webgpu-tile-resolve-v1",
  resolveMode: "tile-center-weighted-oit",
  resolvedTileCount: 0,
  resolveWeightSum: 0,
  resolveAlphaMean: 0,
  resolveLumaMean: 0,
  resolveChecksum: "00000000",
  objectCount: 0,
  objectStateLayoutVersion: "webgpu-object-state-v1",
  objectStateStrideUint32: 4,
  objectStateVisibleObjects: 0,
  objectStateHiddenObjects: 0,
  objectStateRemovedObjects: 0,
  objectStateSelectedObjects: 0,
  objectStateIsolatedObjects: 0,
  objectStateChecksum: "00000000",
});

export async function detectWebGpuCapability() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return {
      status: "unavailable",
      reason: "navigator-gpu-unavailable",
      label: "不可用",
    };
  }

  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch {
    return {
      status: "unavailable",
      reason: "webgpu-adapter-request-failed",
      label: "不可用",
    };
  }

  if (!adapter) {
    return {
      status: "unavailable",
      reason: "webgpu-adapter-unavailable",
      label: "不可用",
    };
  }

  const limits = adapter.limits;
  return {
    status: "available",
    reason: "webgpu-adapter-ready",
    label: "可用",
    maxBufferSize: limits?.maxBufferSize ?? null,
    maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize ?? null,
    maxStorageBuffersPerShaderStage: limits?.maxStorageBuffersPerShaderStage ?? null,
  };
}

export function editRendererContract(webGpuCapability, tileSmoke) {
  const smoke = tileSmoke ?? EMPTY_TILE_SMOKE;
  const storageEstimate = estimateWebGpuTileRuntimeStorage(smoke);
  const storageLimit = webGpuStorageLimitGate(webGpuCapability, storageEstimate);
  const targetGate = webGpuTileTargetGate(webGpuCapability, smoke, storageLimit);
  const useWebGpuTile = targetGate.gate === "pass";
  return {
    rendererId: useWebGpuTile ? WEBGPU_TILE_RENDERER_ID : GAUSSIAN_OIT_RENDERER_ID,
    rendererLabel: useWebGpuTile ? WEBGPU_TILE_RENDERER_LABEL : GAUSSIAN_OIT_RENDERER_LABEL,
    targetRendererId: WEBGPU_TILE_RENDERER_ID,
    targetRendererLabel: WEBGPU_TILE_RENDERER_LABEL,
    objectFilter: useWebGpuTile ? WEBGPU_TILE_OBJECT_FILTER : GAUSSIAN_OIT_OBJECT_FILTER,
    targetObjectFilter: WEBGPU_TILE_OBJECT_FILTER,
    webGpuStatus: webGpuCapability.status,
    webGpuLabel: webGpuCapability.label,
    fallbackReason: fallbackReason(webGpuCapability, targetGate),
    targetGate: targetGate.gate,
    targetGateReason: targetGate.reason,
    targetGateBlocker: targetGate.blocker,
    storageLimitGate: storageLimit.gate,
    storageLimitReason: storageLimit.reason,
    storageLimitBlocker: storageLimit.blocker,
    storageLimitMaxBufferSize: storageLimit.maxBufferSize,
    storageLimitMaxStorageBufferBindingSize: storageLimit.maxStorageBufferBindingSize,
    storageLimitMaxStorageBuffersPerShaderStage: storageLimit.maxStorageBuffersPerShaderStage,
    storageLimitRequiredStorageBuffersPerShaderStage:
      WEBGPU_TILE_REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
    storageLimitEffectiveMaxBufferByteLength: storageLimit.effectiveMaxBufferByteLength,
    storageEstimatedLayout: storageEstimate.layoutVersion,
    storageEstimatedBufferCount: storageEstimate.bufferCount,
    storageEstimatedByteSize: storageEstimate.totalByteLength,
    storageEstimatedMaxBufferByteSize: storageEstimate.maxBufferByteLength,
    storageEstimatedMaxBufferKey: storageEstimate.maxBufferKey,
    tileSmokeLayout: smoke.layoutVersion,
    tileSize: smoke.tileSize,
    viewportWidth: smoke.viewportWidth,
    viewportHeight: smoke.viewportHeight,
    boundsMinX: smoke.boundsMinX,
    boundsMinZ: smoke.boundsMinZ,
    boundsSpanX: smoke.boundsSpanX,
    boundsSpanZ: smoke.boundsSpanZ,
    boundsFitMode: smoke.boundsFitMode,
    boundsPaddingRatio: smoke.boundsPaddingRatio,
    boundsViewportAspect: smoke.boundsViewportAspect,
    boundsWorldAspect: smoke.boundsWorldAspect,
    projectionMode: smoke.projectionMode,
    projectionCameraTuningMode: smoke.projectionCameraTuningMode,
    projectionCameraMode: smoke.projectionCameraMode,
    projectionCameraFovDegrees: smoke.projectionCameraFovDegrees,
    projectionCameraPosition: smoke.projectionCameraPosition,
    projectionCameraTarget: smoke.projectionCameraTarget,
    projectionCameraDistance: smoke.projectionCameraDistance,
    projectionCameraFrameMaxDim: smoke.projectionCameraFrameMaxDim,
    projectionDepthMin: smoke.projectionDepthMin,
    projectionDepthMax: smoke.projectionDepthMax,
    projectionDepthSpan: smoke.projectionDepthSpan,
    depthWeightMode: smoke.depthWeightMode,
    pixelDepthSortMode: smoke.pixelDepthSortMode,
    pixelDepthTuningMode: smoke.pixelDepthTuningMode,
    pixelDepthGateStrength: smoke.pixelDepthGateStrength,
    pixelDepthGateFloor: smoke.pixelDepthGateFloor,
    pixelDepthBinCount: smoke.pixelDepthBinCount,
    pixelCoverageMode: smoke.pixelCoverageMode,
    pixelCoverageWeightFloor: smoke.pixelCoverageWeightFloor,
    pixelCoverageFootprintScale: smoke.pixelCoverageFootprintScale,
    pixelCoverageTuningMode: smoke.pixelCoverageTuningMode,
    colorFidelityMode: smoke.colorFidelityMode,
    colorSourceRgbGaussians: smoke.colorSourceRgbGaussians,
    colorSourceShDcGaussians: smoke.colorSourceShDcGaussians,
    colorSourceFallbackGaussians: smoke.colorSourceFallbackGaussians,
    colorSourceObjectGaussians: smoke.colorSourceObjectGaussians,
    colorOpacityMean: smoke.colorOpacityMean,
    screenCovarianceMode: smoke.screenCovarianceMode,
    screenCovarianceGaussians: smoke.screenCovarianceGaussians,
    screenCovarianceFallbackGaussians: smoke.screenCovarianceFallbackGaussians,
    screenCovarianceClampedGaussians: smoke.screenCovarianceClampedGaussians,
    screenCovarianceMaxAnisotropy: smoke.screenCovarianceMaxAnisotropy,
    screenCovarianceSigmaMean: smoke.screenCovarianceSigmaMean,
    packedGaussians: smoke.packedGaussians,
    binnedGaussians: smoke.binnedGaussians,
    visibleGaussians: smoke.visibleGaussians,
    tileCount: smoke.tileCount,
    activeTileCount: smoke.activeTileCount,
    tileReferenceCount: smoke.tileReferenceCount,
    tileOverflowCount: smoke.tileOverflowCount,
    tileOverflowTileCount: smoke.tileOverflowTileCount,
    tileOverflowRatio: smoke.tileOverflowRatio,
    tileOverflowMaxExcess: smoke.tileOverflowMaxExcess,
    tileEntryStoredCount: smoke.tileEntryStoredCount,
    tileEntryCapacity: smoke.tileEntryCapacity,
    tileEntryUtilization: smoke.tileEntryUtilization,
    tileEntryLayout: smoke.tileEntryLayout,
    tileEntryOffsetCount: smoke.tileEntryOffsetCount,
    tileCapacityMode: smoke.tileCapacityMode,
    tileCapacityStatus: smoke.tileCapacityStatus,
    tileCapacityGate: smoke.tileCapacityGate,
    maxTileOccupancy: smoke.maxTileOccupancy,
    resolveVersion: smoke.resolveVersion,
    resolveMode: smoke.resolveMode,
    resolvedTileCount: smoke.resolvedTileCount,
    resolveWeightSum: smoke.resolveWeightSum,
    resolveAlphaMean: smoke.resolveAlphaMean,
    resolveLumaMean: smoke.resolveLumaMean,
    resolveChecksum: smoke.resolveChecksum,
    objectCount: smoke.objectCount,
    objectStateLayoutVersion: smoke.objectStateLayoutVersion,
    objectStateStrideUint32: smoke.objectStateStrideUint32,
    objectStateVisibleObjects: smoke.objectStateVisibleObjects,
    objectStateHiddenObjects: smoke.objectStateHiddenObjects,
    objectStateRemovedObjects: smoke.objectStateRemovedObjects,
    objectStateSelectedObjects: smoke.objectStateSelectedObjects,
    objectStateIsolatedObjects: smoke.objectStateIsolatedObjects,
    objectStateChecksum: smoke.objectStateChecksum,
  };
}

function fallbackReason(webGpuCapability, targetGate) {
  if (targetGate.gate === "pass") return "";
  if (targetGate.blocker === "tile-overflow") return "webgpu-tile-overflow";
  if (targetGate.blocker === "webgpu-buffer-limit") return "webgpu-buffer-limit";
  if (targetGate.blocker === "webgpu-binding-limit") return "webgpu-binding-limit";
  if (targetGate.blocker === "webgpu-capability") return webGpuCapability.reason;
  if (webGpuCapability.status === "available") {
    return "webgpu-tile-renderer-not-implemented";
  }
  return webGpuCapability.reason;
}

function webGpuTileTargetGate(webGpuCapability, tileSmoke, storageLimit) {
  if (webGpuCapability.status !== "available") {
    return {
      gate: "blocked",
      reason: webGpuCapability.reason,
      blocker: "webgpu-capability",
    };
  }
  if (tileSmoke?.tileCapacityGate === "blocked") {
    return {
      gate: "blocked",
      reason: "webgpu-tile-overflow",
      blocker: "tile-overflow",
    };
  }
  if (storageLimit.gate === "blocked") {
    return {
      gate: "blocked",
      reason: storageLimit.reason,
      blocker: storageLimit.blocker,
    };
  }
  return {
    gate: "pass",
    reason: "webgpu-tile-first-frame-ready",
    blocker: "",
  };
}

function webGpuStorageLimitGate(webGpuCapability, storageEstimate) {
  const maxBufferSize = positiveLimit(webGpuCapability?.maxBufferSize);
  const maxStorageBufferBindingSize = positiveLimit(
    webGpuCapability?.maxStorageBufferBindingSize,
  );
  const maxStorageBuffersPerShaderStage = positiveLimit(
    webGpuCapability?.maxStorageBuffersPerShaderStage,
  );
  const effectiveMaxBufferByteLength = Math.min(
    maxBufferSize ?? Number.POSITIVE_INFINITY,
    maxStorageBufferBindingSize ?? Number.POSITIVE_INFINITY,
  );

  if (webGpuCapability.status !== "available") {
    return {
      gate: "unknown",
      reason: webGpuCapability.reason,
      blocker: "webgpu-capability",
      maxBufferSize,
      maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage,
      effectiveMaxBufferByteLength: finiteOrNull(effectiveMaxBufferByteLength),
    };
  }

  if (
    maxStorageBuffersPerShaderStage !== null &&
    maxStorageBuffersPerShaderStage < WEBGPU_TILE_REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE
  ) {
    return {
      gate: "blocked",
      reason: "webgpu-storage-buffer-bindings-too-many",
      blocker: "webgpu-binding-limit",
      maxBufferSize,
      maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage,
      effectiveMaxBufferByteLength: finiteOrNull(effectiveMaxBufferByteLength),
    };
  }

  if (!Number.isFinite(effectiveMaxBufferByteLength)) {
    return {
      gate: "pass",
      reason: "webgpu-storage-limits-unreported",
      blocker: "",
      maxBufferSize,
      maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage,
      effectiveMaxBufferByteLength: null,
    };
  }

  if (storageEstimate.maxBufferByteLength > effectiveMaxBufferByteLength) {
    return {
      gate: "blocked",
      reason: "webgpu-storage-buffer-too-large",
      blocker: "webgpu-buffer-limit",
      maxBufferSize,
      maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage,
      effectiveMaxBufferByteLength,
    };
  }

  return {
    gate: "pass",
    reason: "webgpu-storage-buffer-limits-pass",
    blocker: "",
    maxBufferSize,
    maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage,
    effectiveMaxBufferByteLength,
  };
}

function positiveLimit(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric <= 0) return null;
  return Math.floor(numeric);
}

function finiteOrNull(value) {
  return Number.isFinite(value) ? value : null;
}
