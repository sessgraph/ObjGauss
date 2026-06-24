import { dyno } from "@sparkjsdev/spark";
import * as THREE from "three";

export const SPARK_OBJECT_MASK_MODE = "object-opacity-texture-v1";

const MASK_TEXTURE_WIDTH = 4096;

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
  };
}

export function updateSparkObjectMask(mask, {
  points,
  visibleIds,
  removedIds,
  isolatedId,
}) {
  if (!mask) return emptyMaskStats();
  let visibleGaussians = 0;
  let hiddenGaussians = 0;
  for (let index = 0; index < mask.data.length; index += 1) {
    const point = points?.[index];
    const visible = Boolean(point) && pointVisible(point, visibleIds, removedIds, isolatedId);
    mask.data[index] = visible ? 1 : 0;
    if (!point) continue;
    if (visible) {
      visibleGaussians += 1;
    } else {
      hiddenGaussians += 1;
    }
  }
  mask.texture.needsUpdate = true;
  mask.updates += 1;
  mask.visibleGaussians = visibleGaussians;
  mask.hiddenGaussians = hiddenGaussians;
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
  };
}

function createObjectOpacityModifier({ texture, width }) {
  const maskTexture = dyno.dynoUsampler2D(texture, "objGaussObjectMask");
  const maskWidth = dyno.dynoInt(width, "objGaussObjectMaskWidth");
  const zeroInt = dyno.dynoConst("int", 0);
  const zeroUint = dyno.dynoConst("uint", 0);
  const zeroFloat = dyno.dynoConst("float", 0);

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
      const maskedOpacity = dyno.select(dyno.greaterThan(maskValue, zeroUint), opacity, zeroFloat);
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

function emptyMaskStats() {
  return {
    objectMaskMode: "none",
    objectMaskWidth: 0,
    objectMaskHeight: 0,
    objectMaskUpdates: 0,
    objectMaskVisibleGaussians: 0,
    objectMaskHiddenGaussians: 0,
  };
}
