export const WEBGPU_DEPTH_SORT_TUNING_MODE = "runtime-depth-sort-tuning-v1";
export const WEBGPU_DEPTH_BIN_COUNT_DEFAULT = 8;
export const WEBGPU_DEPTH_BIN_COUNT_MIN = 4;
export const WEBGPU_DEPTH_BIN_COUNT_MAX = 16;
export const WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED = "depth-binned";
export const WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K = "front-top-k";
export const WEBGPU_DEPTH_ALPHA_MODE_DEFAULT = WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED;
export const WEBGPU_DEPTH_ALPHA_MODES = Object.freeze([
  WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED,
  WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K,
]);

export function normalizeWebGpuDepthSortTuning(tuning = null) {
  return {
    mode: WEBGPU_DEPTH_SORT_TUNING_MODE,
    pixelDepthAlphaMode: normalizeWebGpuDepthAlphaMode(
      tuning?.pixelDepthAlphaMode ?? tuning?.depthAlphaMode ?? tuning?.alphaMode,
    ),
    pixelDepthBinCount: normalizeWebGpuPixelDepthBinCount(
      tuning?.pixelDepthBinCount ?? tuning?.depthBins,
    ),
  };
}

export function normalizeWebGpuDepthAlphaMode(value = WEBGPU_DEPTH_ALPHA_MODE_DEFAULT) {
  if (value === undefined || value === null || value === "") {
    return WEBGPU_DEPTH_ALPHA_MODE_DEFAULT;
  }
  const text = String(value).trim();
  return WEBGPU_DEPTH_ALPHA_MODES.includes(text)
    ? text
    : WEBGPU_DEPTH_ALPHA_MODE_DEFAULT;
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
