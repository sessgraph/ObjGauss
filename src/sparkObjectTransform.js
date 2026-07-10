import { dyno } from "@sparkjsdev/spark";
import * as THREE from "three";

export const SPARK_OBJECT_TRANSFORM_CONTRACT = "source-splat-object-translate-v1";
export const SPARK_OBJECT_TRANSFORM_MODE = "object-translate-texture-v1";

const OBJECT_INDEX_TEXTURE_WIDTH = 4096;
const OBJECT_TRANSFORM_TEXTURE_WIDTH = 512;

export function createSparkObjectTransform(points = []) {
  const pointCount = Array.isArray(points) ? points.length : 0;
  const safePointCount = Math.max(1, pointCount);
  const indexWidth = Math.min(OBJECT_INDEX_TEXTURE_WIDTH, safePointCount);
  const indexHeight = Math.max(1, Math.ceil(safePointCount / indexWidth));
  const objectIds = uniqueObjectIds(points);
  const slotByObjectId = new Map(objectIds.map((objectId, slot) => [String(objectId), slot]));
  const objectIndexData = new Uint32Array(indexWidth * indexHeight);
  const gaussianCountsBySlot = new Uint32Array(objectIds.length);

  points.forEach((point, index) => {
    const slot = slotByObjectId.get(String(normalizedObjectId(point))) ?? 0;
    objectIndexData[index] = slot;
    gaussianCountsBySlot[slot] += 1;
  });

  const objectIndexTexture = new THREE.DataTexture(
    objectIndexData,
    indexWidth,
    indexHeight,
    THREE.RedIntegerFormat,
    THREE.UnsignedIntType,
  );
  objectIndexTexture.magFilter = THREE.NearestFilter;
  objectIndexTexture.minFilter = THREE.NearestFilter;
  objectIndexTexture.wrapS = THREE.ClampToEdgeWrapping;
  objectIndexTexture.wrapT = THREE.ClampToEdgeWrapping;
  objectIndexTexture.unpackAlignment = 1;
  objectIndexTexture.needsUpdate = true;

  const safeObjectCount = Math.max(1, objectIds.length);
  const transformWidth = Math.min(OBJECT_TRANSFORM_TEXTURE_WIDTH, safeObjectCount);
  const transformHeight = Math.max(1, Math.ceil(safeObjectCount / transformWidth));
  const transformData = new Float32Array(transformWidth * transformHeight * 4);
  objectIds.forEach((_, slot) => {
    transformData[slot * 4 + 3] = 1;
  });
  const transformTexture = new THREE.DataTexture(
    transformData,
    transformWidth,
    transformHeight,
    THREE.RGBAFormat,
    THREE.FloatType,
  );
  transformTexture.magFilter = THREE.NearestFilter;
  transformTexture.minFilter = THREE.NearestFilter;
  transformTexture.wrapS = THREE.ClampToEdgeWrapping;
  transformTexture.wrapT = THREE.ClampToEdgeWrapping;
  transformTexture.unpackAlignment = 1;
  transformTexture.needsUpdate = true;

  const enabledUniform = dyno.dynoBool(false, "objGaussObjectTransformEnabled");
  const sourceCountUniform = dyno.dynoInt(pointCount, "objGaussObjectTransformSourceCount");

  const transform = {
    contract: SPARK_OBJECT_TRANSFORM_CONTRACT,
    mode: SPARK_OBJECT_TRANSFORM_MODE,
    pointCount,
    sourceCount: null,
    sourceCountMatches: false,
    objectIds,
    objectCount: objectIds.length,
    slotByObjectId,
    indexWidth,
    indexHeight,
    objectIndexData,
    gaussianCountsBySlot,
    objectIndexTexture,
    transformWidth,
    transformHeight,
    transformData,
    transformTexture,
    enabledUniform,
    sourceCountUniform,
    active: false,
    status: pointCount > 0 && objectIds.length > 0 ? "pending-source-count" : "missing-points",
    reason: pointCount > 0 && objectIds.length > 0 ? "waiting-for-splat-count" : "empty-object-index",
    updates: 0,
    transformedObjects: 0,
    hiddenObjects: 0,
    hiddenGaussians: 0,
    maxTranslate: 0,
    totalTranslate: 0,
  };
  transform.modifier = createObjectTransformModifier(transform);
  return transform;
}

export function configureSparkObjectTransform(transform, { sourceCount = null } = {}) {
  if (!transform) return sparkObjectTransformStats(transform);
  const count = Number(sourceCount);
  transform.sourceCount = Number.isFinite(count) ? Math.trunc(count) : null;
  transform.sourceCountMatches =
    Number.isInteger(transform.sourceCount) &&
    transform.sourceCount > 0 &&
    transform.sourceCount === transform.pointCount;
  if (!transform.pointCount || !transform.objectCount) {
    transform.status = "missing-points";
    transform.reason = "empty-object-index";
  } else if (!Number.isInteger(transform.sourceCount)) {
    transform.status = "pending-source-count";
    transform.reason = "waiting-for-splat-count";
  } else if (!transform.sourceCountMatches) {
    transform.status = "disabled-count-mismatch";
    transform.reason = `${transform.pointCount} object-aware points != ${transform.sourceCount} source splats`;
  } else {
    transform.status = "ready";
    transform.reason = null;
  }
  setUniformValue(transform.sourceCountUniform, transform.sourceCountMatches ? transform.pointCount : 0);
  setUniformValue(transform.enabledUniform, transform.sourceCountMatches && transform.active);
  return sparkObjectTransformStats(transform);
}

export function updateSparkObjectTransforms(
  transform,
  { objectGroups = [], sourceFrameScale = 1 } = {},
) {
  if (!transform) return sparkObjectTransformStats(transform);
  transform.transformData.fill(0);
  transform.objectIds.forEach((_, slot) => {
    transform.transformData[slot * 4 + 3] = 1;
  });
  const scale = Number.isFinite(sourceFrameScale) && sourceFrameScale > 0 ? sourceFrameScale : 1;
  let transformedObjects = 0;
  let hiddenObjects = 0;
  let hiddenGaussians = 0;
  let maxTranslate = 0;
  let totalTranslate = 0;

  for (const object of objectGroups ?? []) {
    const objectId = object?.userData?.objectId;
    const slot = transform.slotByObjectId.get(String(objectId));
    if (!Number.isInteger(slot)) continue;
    const initial = normalizedInitialPosition(object);
    const translateX = (Number(object.position.x) - initial[0]) / scale;
    const translateY = (Number(object.position.y) - initial[1]) / scale;
    const translateZ = (Number(object.position.z) - initial[2]) / scale;
    if (![translateX, translateY, translateZ].every(Number.isFinite)) continue;
    const offset = slot * 4;
    transform.transformData[offset] = translateX;
    transform.transformData[offset + 1] = translateY;
    transform.transformData[offset + 2] = translateZ;
    const visible = object.visible !== false;
    transform.transformData[offset + 3] = visible ? 1 : 0;
    if (!visible) {
      hiddenObjects += 1;
      hiddenGaussians += gaussianCountForSlot(transform, slot);
    }
    const magnitude = Math.hypot(translateX, translateY, translateZ);
    if (magnitude > 0.000001) {
      transformedObjects += 1;
      totalTranslate += magnitude;
      maxTranslate = Math.max(maxTranslate, magnitude);
    }
  }

  transform.transformTexture.needsUpdate = true;
  transform.updates += 1;
  transform.transformedObjects = transformedObjects;
  transform.hiddenObjects = hiddenObjects;
  transform.hiddenGaussians = hiddenGaussians;
  transform.maxTranslate = maxTranslate;
  transform.totalTranslate = totalTranslate;
  transform.active = transform.sourceCountMatches && (transformedObjects > 0 || hiddenObjects > 0);
  if (transform.sourceCountMatches) {
    transform.status = "ready";
    transform.reason = null;
  }
  setUniformValue(transform.enabledUniform, transform.active);
  return sparkObjectTransformStats(transform);
}

export function disposeSparkObjectTransform(transform) {
  transform?.objectIndexTexture?.dispose?.();
  transform?.transformTexture?.dispose?.();
}

export function sparkObjectTransformStats(transform) {
  return {
    contract: transform?.contract ?? SPARK_OBJECT_TRANSFORM_CONTRACT,
    mode: transform?.mode ?? "none",
    status: transform?.status ?? "missing",
    reason: transform?.reason ?? null,
    active: Boolean(transform?.active),
    pointCount: transform?.pointCount ?? 0,
    sourceCount: transform?.sourceCount ?? null,
    sourceCountMatches: Boolean(transform?.sourceCountMatches),
    objectCount: transform?.objectCount ?? 0,
    mappedGaussians: transform?.pointCount ?? 0,
    updates: transform?.updates ?? 0,
    transformedObjects: transform?.transformedObjects ?? 0,
    hiddenObjects: transform?.hiddenObjects ?? 0,
    hiddenGaussians: transform?.hiddenGaussians ?? 0,
    visibilitySynchronized: Boolean(transform?.sourceCountMatches),
    maxTranslate: round6(transform?.maxTranslate ?? 0),
    totalTranslate: round6(transform?.totalTranslate ?? 0),
  };
}

function createObjectTransformModifier(transform) {
  const objectIndexTexture = dyno.dynoUsampler2D(
    transform.objectIndexTexture,
    "objGaussObjectIndexTexture",
  );
  const objectTransformTexture = dyno.dynoSampler2D(
    transform.transformTexture,
    "objGaussObjectTransformTexture",
  );
  const indexWidth = dyno.dynoInt(transform.indexWidth, "objGaussObjectIndexWidth");
  const transformWidth = dyno.dynoInt(transform.transformWidth, "objGaussObjectTransformWidth");
  const zeroInt = dyno.dynoConst("int", 0);
  const zeroVec3 = dyno.dynoConst("vec3", new THREE.Vector3(0, 0, 0));

  return dyno.dynoBlock(
    { gsplat: dyno.Gsplat },
    { gsplat: dyno.Gsplat },
    ({ gsplat }) => {
      if (!gsplat) {
        throw new Error("No gsplat input");
      }
      const { index, center, opacity } = dyno.splitGsplat(gsplat).outputs;
      const inRange = dyno.lessThan(index, transform.sourceCountUniform);
      const safeIndex = dyno.select(inRange, index, zeroInt);
      const indexCoord = dyno.combine({
        vectorType: "ivec2",
        x: dyno.imod(safeIndex, indexWidth),
        y: dyno.div(safeIndex, indexWidth),
      });
      const slot = dyno.int(dyno.split(dyno.texelFetch(objectIndexTexture, indexCoord, zeroInt)).outputs.r);
      const transformCoord = dyno.combine({
        vectorType: "ivec2",
        x: dyno.imod(slot, transformWidth),
        y: dyno.div(slot, transformWidth),
      });
      const transformEntry = dyno.texelFetch(objectTransformTexture, transformCoord, zeroInt);
      const translate = dyno.vec3(transformEntry);
      const visibility = dyno.split(transformEntry).outputs.w;
      const transformedCenter = dyno.add(center, translate);
      const transformedOpacity = dyno.mul(opacity, visibility);
      const applyTransform = dyno.and(transform.enabledUniform, inRange);
      return {
        gsplat: dyno.combineGsplat({
          gsplat,
          center: dyno.select(applyTransform, transformedCenter, center ?? zeroVec3),
          opacity: dyno.select(applyTransform, transformedOpacity, opacity),
        }),
      };
    },
  );
}

function uniqueObjectIds(points) {
  return [...new Set((points ?? []).map(normalizedObjectId))].sort((left, right) => left - right);
}

function normalizedObjectId(point) {
  const value = Number(point?.objectId ?? point?.object_id ?? point?.label ?? 0);
  return Number.isFinite(value) ? Math.trunc(value) : 0;
}

function normalizedInitialPosition(object) {
  const raw = object?.userData?.sourceSplatInitialPosition;
  if (Array.isArray(raw) && raw.length >= 3) {
    return [
      finiteNumber(raw[0], 0),
      finiteNumber(raw[1], 0),
      finiteNumber(raw[2], 0),
    ];
  }
  return [
    finiteNumber(object?.position?.x, 0),
    finiteNumber(object?.position?.y, 0),
    finiteNumber(object?.position?.z, 0),
  ];
}

function gaussianCountForSlot(transform, slot) {
  return Number(transform?.gaussianCountsBySlot?.[slot] ?? 0);
}

function setUniformValue(uniform, value) {
  if (!uniform) return;
  uniform.value = value;
  if (uniform.uniform) uniform.uniform.value = value;
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function round6(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.round(number * 1e6) / 1e6 : 0;
}
