export const WEBGPU_TILE_RENDERER_ID = "webgpu-tile";
export const WEBGPU_TILE_RENDERER_LABEL = "WebGPU Tile 编辑";
export const GAUSSIAN_OIT_RENDERER_ID = "gaussian-oit";
export const GAUSSIAN_OIT_RENDERER_LABEL = "Gaussian OIT 编辑";
export const GAUSSIAN_OIT_OBJECT_FILTER = "gpu-object-state-texture";

export const INITIAL_WEBGPU_CAPABILITY = Object.freeze({
  status: "pending",
  reason: "webgpu-capability-detecting",
  label: "检测中",
});
const EMPTY_TILE_SMOKE = Object.freeze({
  layoutVersion: "webgpu-tile-smoke-v1",
  tileSize: 16,
  packedGaussians: 0,
  binnedGaussians: 0,
  visibleGaussians: 0,
  tileCount: 0,
  activeTileCount: 0,
  tileReferenceCount: 0,
  tileOverflowCount: 0,
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

  let device = null;
  try {
    device = await adapter.requestDevice();
  } catch {
    return {
      status: "unavailable",
      reason: "webgpu-device-request-failed",
      label: "不可用",
    };
  } finally {
    device?.destroy?.();
  }

  const limits = adapter.limits ?? device?.limits;
  return {
    status: "available",
    reason: "webgpu-device-ready",
    label: "可用",
    maxBufferSize: limits?.maxBufferSize ?? null,
    maxStorageBufferBindingSize: limits?.maxStorageBufferBindingSize ?? null,
  };
}

export function editRendererContract(webGpuCapability, tileSmoke) {
  const smoke = tileSmoke ?? EMPTY_TILE_SMOKE;
  return {
    rendererId: GAUSSIAN_OIT_RENDERER_ID,
    rendererLabel: GAUSSIAN_OIT_RENDERER_LABEL,
    targetRendererId: WEBGPU_TILE_RENDERER_ID,
    targetRendererLabel: WEBGPU_TILE_RENDERER_LABEL,
    objectFilter: GAUSSIAN_OIT_OBJECT_FILTER,
    targetObjectFilter: "gpu-object-state-buffer",
    webGpuStatus: webGpuCapability.status,
    webGpuLabel: webGpuCapability.label,
    fallbackReason: fallbackReason(webGpuCapability),
    tileSmokeLayout: smoke.layoutVersion,
    tileSize: smoke.tileSize,
    packedGaussians: smoke.packedGaussians,
    binnedGaussians: smoke.binnedGaussians,
    visibleGaussians: smoke.visibleGaussians,
    tileCount: smoke.tileCount,
    activeTileCount: smoke.activeTileCount,
    tileReferenceCount: smoke.tileReferenceCount,
    tileOverflowCount: smoke.tileOverflowCount,
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

function fallbackReason(webGpuCapability) {
  if (webGpuCapability.status === "available") {
    return "webgpu-tile-renderer-not-implemented";
  }
  return webGpuCapability.reason;
}
