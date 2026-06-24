import { dyno } from "@sparkjsdev/spark";
import * as THREE from "three";

export const SPARK_OBJECT_MASK_MODE = "object-opacity-texture-v1";

const MASK_TEXTURE_WIDTH = 4096;
const MASK_OPACITY_FULL = 255;
const DEFAULT_FEATHER_OPACITY = 0.62;
const DEFAULT_FEATHER_RADIUS_RATIO = 0.008;
const DEFAULT_FEATHER_SCALE_MULTIPLIER = 2.0;
const MAX_NEIGHBOR_CANDIDATES_PER_CELL = 4096;

export function createSparkObjectMask(pointCount = 0) {
  const count = Math.max(1, Math.floor(Number(pointCount) || 0));
  const width = Math.min(MASK_TEXTURE_WIDTH, count);
  const height = Math.max(1, Math.ceil(count / width));
  const data = new Uint32Array(width * height);
  const texture = new THREE.DataTexture(data, width, height, THREE.RedIntegerFormat, THREE.UnsignedIntType);
  texture.magFilter = THREE.NearestFilter;
  texture.minFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;

  return {
    mode: SPARK_OBJECT_MASK_MODE,
    width,
    height,
    data,
    texture,
    modifier: createObjectOpacityModifier({ texture, width }),
    updates: 0,
    visibleGaussians: 0,
    hiddenGaussians: 0,
    featherMode: "off",
    featherRadius: 0,
    featherOpacity: 1,
    featheredGaussians: 0,
    opacityMean: 0,
    minOpacityScale: 0,
  };
}

export function updateSparkObjectMask(mask, {
  points,
  visibleIds,
  removedIds,
  isolatedId,
  feathering = null,
}) {
  if (!mask) return emptyMaskStats();
  const feather = normalizeSparkObjectMaskFeathering(feathering);
  const featherMap = feather.enabled
    ? buildFeatherMap({ points, visibleIds, removedIds, isolatedId, feather })
    : emptyFeatherMap();
  let visibleGaussians = 0;
  let hiddenGaussians = 0;
  let featheredGaussians = 0;
  let opacitySum = 0;
  let minOpacityScale = 1;
  for (let index = 0; index < mask.data.length; index += 1) {
    const point = points?.[index];
    const visible = Boolean(point) && pointVisible(point, visibleIds, removedIds, isolatedId);
    const featherScale = featherMap.scales?.[index] ?? 0;
    const opacityScale = visible
      ? featherScale > 0
        ? featherScale
        : MASK_OPACITY_FULL
      : 0;
    mask.data[index] = opacityScale;
    if (!point) continue;
    if (visible) {
      visibleGaussians += 1;
      opacitySum += opacityScale / MASK_OPACITY_FULL;
      minOpacityScale = Math.min(minOpacityScale, opacityScale / MASK_OPACITY_FULL);
      if (opacityScale > 0 && opacityScale < MASK_OPACITY_FULL) {
        featheredGaussians += 1;
      }
    } else {
      hiddenGaussians += 1;
    }
  }
  mask.texture.needsUpdate = true;
  mask.updates += 1;
  mask.visibleGaussians = visibleGaussians;
  mask.hiddenGaussians = hiddenGaussians;
  mask.featherMode = featherMap.mode;
  mask.featherRadius = featherMap.radius;
  mask.featherOpacity = featherMap.opacity;
  mask.featheredGaussians = featheredGaussians;
  mask.opacityMean = visibleGaussians > 0 ? opacitySum / visibleGaussians : 0;
  mask.minOpacityScale = visibleGaussians > 0 ? minOpacityScale : 0;
  return maskStats(mask);
}

export function disposeSparkObjectMask(mask) {
  mask?.texture?.dispose?.();
}

export function maskStats(mask) {
  return {
    objectMaskMode: mask?.mode ?? "none",
    objectMaskWidth: mask?.width ?? 0,
    objectMaskHeight: mask?.height ?? 0,
    objectMaskUpdates: mask?.updates ?? 0,
    objectMaskVisibleGaussians: mask?.visibleGaussians ?? 0,
    objectMaskHiddenGaussians: mask?.hiddenGaussians ?? 0,
    objectMaskFeatherMode: mask?.featherMode ?? "off",
    objectMaskFeatherRadius: mask?.featherRadius ?? 0,
    objectMaskFeatherOpacity: mask?.featherOpacity ?? 1,
    objectMaskFeatheredGaussians: mask?.featheredGaussians ?? 0,
    objectMaskOpacityMean: mask?.opacityMean ?? 0,
    objectMaskMinOpacityScale: mask?.minOpacityScale ?? 0,
  };
}

function createObjectOpacityModifier({ texture, width }) {
  const maskTexture = dyno.dynoUsampler2D(texture, "objGaussObjectMask");
  const maskWidth = dyno.dynoInt(width, "objGaussObjectMaskWidth");
  const zeroInt = dyno.dynoConst("int", 0);
  const zeroUint = dyno.dynoConst("uint", 0);
  const zeroFloat = dyno.dynoConst("float", 0);
  const fullOpacityScale = dyno.dynoFloat(MASK_OPACITY_FULL, "objGaussObjectMaskFullScale");

  return dyno.dynoBlock(
    { gsplat: dyno.Gsplat },
    { gsplat: dyno.Gsplat },
    ({ gsplat }) => {
      if (!gsplat) {
        throw new Error("No gsplat input");
      }
      const { index, opacity } = dyno.splitGsplat(gsplat).outputs;
      const coord = dyno.combine({
        vectorType: "ivec2",
        x: dyno.imod(index, maskWidth),
        y: dyno.div(index, maskWidth),
      });
      const maskValue = dyno.split(dyno.texelFetch(maskTexture, coord, zeroInt)).outputs.r;
      const maskOpacityScale = dyno.div(dyno.float(maskValue), fullOpacityScale);
      const maskedOpacity = dyno.select(
        dyno.greaterThan(maskValue, zeroUint),
        dyno.mul(opacity, maskOpacityScale),
        zeroFloat,
      );
      return {
        gsplat: dyno.combineGsplat({ gsplat, opacity: maskedOpacity }),
      };
    },
  );
}

function pointVisible(point, visibleIds, removedIds, isolatedId) {
  if (!point) return false;
  if (visibleIds && !visibleIds.has(point.objectId)) return false;
  if (removedIds?.has(point.objectId)) return false;
  if (isolatedId !== null && isolatedId !== undefined && point.objectId !== isolatedId) return false;
  return true;
}

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

function buildFeatherMap({ points, visibleIds, removedIds, isolatedId, feather }) {
  const count = points?.length ?? 0;
  if (count <= 0) return emptyFeatherMap();
  const radius = feather.radius > 0 ? feather.radius : autoFeatherRadius(points);
  if (radius <= 0) return emptyFeatherMap();
  const hiddenGrid = new Map();
  const visibleIndices = [];
  const cellSize = radius;

  for (let index = 0; index < count; index += 1) {
    const point = points[index];
    if (!point) continue;
    if (pointVisible(point, visibleIds, removedIds, isolatedId)) {
      visibleIndices.push(index);
      continue;
    }
    const key = cellKey(point, cellSize);
    const bucket = hiddenGrid.get(key);
    if (bucket) {
      if (bucket.length < MAX_NEIGHBOR_CANDIDATES_PER_CELL) bucket.push(index);
    } else {
      hiddenGrid.set(key, [index]);
    }
  }

  if (hiddenGrid.size === 0 || visibleIndices.length === 0) {
    return emptyFeatherMap({ radius, opacity: feather.opacity, mode: "off" });
  }

  const scales = new Uint32Array(count);
  const radiusSq = radius * radius;
  const minScale = Math.round(feather.opacity * MASK_OPACITY_FULL);
  let feathered = 0;

  for (const index of visibleIndices) {
    const point = points[index];
    const nearestSq = nearestHiddenDistanceSq({ point, points, hiddenGrid, cellSize, radiusSq });
    if (!Number.isFinite(nearestSq)) continue;
    const t = Math.min(Math.max(Math.sqrt(nearestSq) / radius, 0), 1);
    const scale = Math.round((feather.opacity + (1 - feather.opacity) * t) * MASK_OPACITY_FULL);
    const clampedScale = Math.min(Math.max(scale, minScale), MASK_OPACITY_FULL - 1);
    scales[index] = clampedScale;
    feathered += 1;
  }

  return {
    mode: feathered > 0 ? "spatial-neighbor-feather-v1" : "off",
    radius,
    opacity: feather.opacity,
    scales,
  };
}

function nearestHiddenDistanceSq({ point, points, hiddenGrid, cellSize, radiusSq }) {
  const cell = pointCell(point, cellSize);
  let nearestSq = Infinity;
  for (let dz = -1; dz <= 1; dz += 1) {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        const bucket = hiddenGrid.get(`${cell.x + dx}:${cell.y + dy}:${cell.z + dz}`);
        if (!bucket) continue;
        for (const hiddenIndex of bucket) {
          const hidden = points[hiddenIndex];
          const distanceSq =
            (Number(point.x) - Number(hidden.x)) ** 2 +
            (Number(point.y) - Number(hidden.y)) ** 2 +
            (Number(point.z) - Number(hidden.z)) ** 2;
          if (distanceSq <= radiusSq && distanceSq < nearestSq) nearestSq = distanceSq;
        }
      }
    }
  }
  return nearestSq;
}

function autoFeatherRadius(points) {
  const bounds = {
    min: [Infinity, Infinity, Infinity],
    max: [-Infinity, -Infinity, -Infinity],
  };
  const scales = [];
  for (const point of points ?? []) {
    const x = finiteNumber(point?.x, 0);
    const y = finiteNumber(point?.y, 0);
    const z = finiteNumber(point?.z, 0);
    bounds.min[0] = Math.min(bounds.min[0], x);
    bounds.min[1] = Math.min(bounds.min[1], y);
    bounds.min[2] = Math.min(bounds.min[2], z);
    bounds.max[0] = Math.max(bounds.max[0], x);
    bounds.max[1] = Math.max(bounds.max[1], y);
    bounds.max[2] = Math.max(bounds.max[2], z);
    if (Array.isArray(point?.scale3)) {
      const scale = Math.max(
        ...point.scale3.map((entry) => finiteNumber(entry, 0)).filter((entry) => entry > 0),
        0,
      );
      if (scale > 0) scales.push(scale);
    } else if (Array.isArray(point?.scale)) {
      const scale = Math.max(
        ...point.scale.map((entry) => finiteNumber(entry, 0)).filter((entry) => entry > 0),
        0,
      );
      if (scale > 0) scales.push(scale);
    }
  }
  const diagonal = Math.hypot(
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  );
  scales.sort((left, right) => left - right);
  const medianScale = scales[Math.floor(scales.length / 2)] ?? 0;
  return Math.max(
    Number.isFinite(diagonal) ? diagonal * DEFAULT_FEATHER_RADIUS_RATIO : 0,
    medianScale * DEFAULT_FEATHER_SCALE_MULTIPLIER,
  );
}

function cellKey(point, cellSize) {
  const cell = pointCell(point, cellSize);
  return `${cell.x}:${cell.y}:${cell.z}`;
}

function pointCell(point, cellSize) {
  const safeCellSize = Math.max(cellSize, 1e-6);
  return {
    x: Math.floor(finiteNumber(point?.x, 0) / safeCellSize),
    y: Math.floor(finiteNumber(point?.y, 0) / safeCellSize),
    z: Math.floor(finiteNumber(point?.z, 0) / safeCellSize),
  };
}

function emptyFeatherMap(overrides = {}) {
  return {
    mode: "off",
    radius: 0,
    opacity: 1,
    scales: null,
    ...overrides,
  };
}

function emptyMaskStats() {
  return {
    objectMaskMode: "none",
    objectMaskWidth: 0,
    objectMaskHeight: 0,
    objectMaskUpdates: 0,
    objectMaskVisibleGaussians: 0,
    objectMaskHiddenGaussians: 0,
    objectMaskFeatherMode: "off",
    objectMaskFeatherRadius: 0,
    objectMaskFeatherOpacity: 1,
    objectMaskFeatheredGaussians: 0,
    objectMaskOpacityMean: 0,
    objectMaskMinOpacityScale: 0,
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
