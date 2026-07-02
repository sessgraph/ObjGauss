const DEFAULT_FEATHER_OPACITY = 0.62;

export function normalizeSparkObjectMaskFeathering(value = null) {
  if (!value || value === "off" || value.enabled === false) {
    return {
      enabled: false,
      radius: 0,
      opacity: 1,
    };
  }
  const radius = finiteNumber(value.radius, 0);
  const opacity = clampFinite(value.opacity, 0.05, 0.98, DEFAULT_FEATHER_OPACITY);
  return {
    enabled: true,
    radius,
    opacity,
  };
}

function finiteNumber(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function clampFinite(value, min, max, fallback) {
  const numeric = finiteNumber(value, fallback);
  return Math.min(Math.max(numeric, min), max);
}
