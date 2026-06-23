export const WEBGPU_DEPTH_SORT_TUNING_MODE = "runtime-depth-sort-tuning-v1";
export const WEBGPU_DEPTH_BIN_COUNT_DEFAULT = 8;
export const WEBGPU_DEPTH_BIN_COUNT_MIN = 4;
export const WEBGPU_DEPTH_BIN_COUNT_MAX = 16;

export function normalizeWebGpuDepthSortTuning(tuning = null) {
  return {
    mode: WEBGPU_DEPTH_SORT_TUNING_MODE,
    pixelDepthBinCount: normalizeWebGpuPixelDepthBinCount(
      tuning?.pixelDepthBinCount ?? tuning?.depthBins,
    ),
  };
}

export function normalizeWebGpuPixelDepthBinCount(value = WEBGPU_DEPTH_BIN_COUNT_DEFAULT) {
  if (value === undefined || value === null || value === "") {
    return WEBGPU_DEPTH_BIN_COUNT_DEFAULT;
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return WEBGPU_DEPTH_BIN_COUNT_DEFAULT;
  return clampInt(Math.round(numeric), WEBGPU_DEPTH_BIN_COUNT_MIN, WEBGPU_DEPTH_BIN_COUNT_MAX);
}

function clampInt(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
