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

export function editRendererContract(webGpuCapability) {
  return {
    rendererId: GAUSSIAN_OIT_RENDERER_ID,
    rendererLabel: GAUSSIAN_OIT_RENDERER_LABEL,
    targetRendererId: WEBGPU_TILE_RENDERER_ID,
    targetRendererLabel: WEBGPU_TILE_RENDERER_LABEL,
    objectFilter: GAUSSIAN_OIT_OBJECT_FILTER,
    webGpuStatus: webGpuCapability.status,
    webGpuLabel: webGpuCapability.label,
    fallbackReason: fallbackReason(webGpuCapability),
    tileOverflowCount: 0,
  };
}

function fallbackReason(webGpuCapability) {
  if (webGpuCapability.status === "available") {
    return "webgpu-tile-renderer-not-implemented";
  }
  return webGpuCapability.reason;
}
