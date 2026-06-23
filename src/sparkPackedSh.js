import { defines, utils } from "@sparkjsdev/spark";

export const SPARK_PACKED_EXTRACT_ROUTE = "packed-extract-v1";
export const SPARK_PACKED_SH_EXTRACT_ROUTE = "packed-sh-extract-v1";

const SH_C0 = 0.28209479177387814;
const SH_ENCODING = defines?.DEFAULT_SPLAT_ENCODING ?? {
  sh1Max: 1,
  sh2Max: 1,
  sh3Max: 1,
};

export function buildPackedShExtra({
  points,
  shRestCoefficients,
  shRestCoefficientCount,
}) {
  const coefficientCount = supportedShRestCoefficientCount(shRestCoefficientCount);
  const degree = shDegreeFromRestCoefficientCount(coefficientCount);
  const count = points?.length ?? 0;
  const canEncode =
    coefficientCount > 0 &&
    degree > 0 &&
    shRestCoefficients &&
    typeof shRestCoefficients.length === "number" &&
    typeof utils?.encodeSh1Rgb === "function";

  if (!canEncode) {
    return {
      extra: {},
      sourceGaussians: 0,
      preservedGaussians: 0,
      preserved: false,
      coefficientCount,
      degree,
      route: SPARK_PACKED_EXTRACT_ROUTE,
    };
  }

  const extra = {};
  if (coefficientCount >= 9) extra.sh1 = new Uint32Array(count * 2);
  if (coefficientCount >= 24) extra.sh2 = new Uint32Array(count * 4);
  if (coefficientCount >= 45) extra.sh3 = new Uint32Array(count * 4);

  const sh1 = new Float32Array(9);
  const sh2 = new Float32Array(15);
  const sh3 = new Float32Array(21);
  let sourceGaussians = 0;
  let preservedGaussians = 0;

  for (let index = 0; index < count; index += 1) {
    const point = points[index];
    const pointCoefficientCount = Math.min(
      pointShRestCoefficientCount(point),
      coefficientCount,
    );
    const offset = index * shRestCoefficientCount;
    if (
      pointCoefficientCount < 9 ||
      offset < 0 ||
      offset + coefficientCount > shRestCoefficients.length
    ) {
      continue;
    }

    sourceGaussians += 1;
    const basisCount = coefficientCount / 3;
    fillShRgb(sh1, shRestCoefficients, offset, basisCount, 0, 3);
    utils.encodeSh1Rgb(extra.sh1, index, sh1, SH_ENCODING);
    if (coefficientCount >= 24 && pointCoefficientCount >= 24) {
      fillShRgb(sh2, shRestCoefficients, offset, basisCount, 3, 5);
      utils.encodeSh2Rgb(extra.sh2, index, sh2, SH_ENCODING);
    }
    if (coefficientCount >= 45 && pointCoefficientCount >= 45) {
      fillShRgb(sh3, shRestCoefficients, offset, basisCount, 8, 7);
      utils.encodeSh3Rgb(extra.sh3, index, sh3, SH_ENCODING);
    }
    preservedGaussians += 1;
  }

  const preserved = sourceGaussians > 0 && preservedGaussians === sourceGaussians;
  return {
    extra: preserved ? extra : {},
    sourceGaussians,
    preservedGaussians: preserved ? preservedGaussians : 0,
    preserved,
    coefficientCount: preserved ? coefficientCount : 0,
    degree: preserved ? degree : 0,
    route: preserved ? SPARK_PACKED_SH_EXTRACT_ROUTE : SPARK_PACKED_EXTRACT_ROUTE,
  };
}

export function extractPackedShExtra(extra, indices) {
  const extracted = {};
  if (extra?.sh1) extracted.sh1 = extractWords(extra.sh1, indices, 2);
  if (extra?.sh2) extracted.sh2 = extractWords(extra.sh2, indices, 4);
  if (extra?.sh3) extracted.sh3 = extractWords(extra.sh3, indices, 4);
  return extracted;
}

export function shDcRgb01(point) {
  if (!Array.isArray(point?.shDc) || point.shDc.length < 3) return null;
  return [0, 1, 2].map((channel) =>
    clampFinite(Number(point.shDc[channel]) * SH_C0 + 0.5, 0, 1, 0.5),
  );
}

function fillShRgb(target, rest, restOffset, basisCount, basisStart, basisLength) {
  let output = 0;
  for (let basis = 0; basis < basisLength; basis += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const index = restOffset + basisStart + basis + channel * basisCount;
      const value = Number(rest[index] ?? 0);
      target[output] = Number.isFinite(value) ? value : 0;
      output += 1;
    }
  }
}

function extractWords(source, indices, wordsPerSplat) {
  const extracted = new Uint32Array((indices?.length ?? 0) * wordsPerSplat);
  for (let outputIndex = 0; outputIndex < (indices?.length ?? 0); outputIndex += 1) {
    const sourceIndex = Number(indices[outputIndex] ?? -1);
    if (!Number.isInteger(sourceIndex) || sourceIndex < 0) continue;
    const sourceOffset = sourceIndex * wordsPerSplat;
    const outputOffset = outputIndex * wordsPerSplat;
    extracted.set(source.subarray(sourceOffset, sourceOffset + wordsPerSplat), outputOffset);
  }
  return extracted;
}

function supportedShRestCoefficientCount(count) {
  const numeric = Number(count);
  if (!Number.isFinite(numeric) || numeric < 9) return 0;
  const integer = Math.trunc(numeric);
  if (integer >= 45) return 45;
  if (integer >= 24) return 24;
  return 9;
}

function shDegreeFromRestCoefficientCount(count) {
  if (count >= 45) return 3;
  if (count >= 24) return 2;
  if (count >= 9) return 1;
  return 0;
}

function pointShRestCoefficientCount(point) {
  const count = Number(point?.shRestCoefficientCount ?? 0);
  if (!Number.isFinite(count) || count <= 0) return 0;
  return Math.trunc(count);
}

function clampFinite(value, min, max, fallback) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(Math.max(numeric, min), max);
}
