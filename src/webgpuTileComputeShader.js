export const WEBGPU_TILE_COMPUTE_SOURCE = "webgpu-compute-resolve-v1";
export const WEBGPU_TILE_COMPUTE_WORKGROUP_SIZE = 64;
export const WEBGPU_TILE_ACCUMULATION_SOURCE = "webgpu-compute-covariance-accumulation-v1";
export const WEBGPU_TILE_ACCUMULATION_WORKGROUP_SIZE = 64;
export const WEBGPU_PIXEL_RESOLVE_SOURCE = "webgpu-compute-depth-binned-alpha-composite-v1";
export const WEBGPU_PIXEL_RESOLVE_WORKGROUP_SIZE = 64;

export const WEBGPU_TILE_ACCUMULATION_SHADER = `
const OBJECT_STATE_VISIBLE = 1u;
const RESOLVE_ALPHA_GAIN = 0.78;
const RESOLVE_KERNEL_CUTOFF = 13.0;
const SAMPLE_WEIGHT = 0.25;
const DEPTH_WEIGHT_STRENGTH = 1.45;
const DEPTH_WEIGHT_FLOOR = 0.22;

struct AccumulationMeta {
  tileCount: f32,
  maxEntriesPerTile: f32,
  tileColumns: f32,
  tileSize: f32,
  viewportWidth: f32,
  viewportHeight: f32,
  boundsMinX: f32,
  boundsMinZ: f32,
  boundsSpanX: f32,
  boundsSpanZ: f32,
  depthMin: f32,
  depthSpan: f32,
};

@group(0) @binding(0) var<storage, read> positionRadius: array<vec4f>;
@group(0) @binding(1) var<storage, read> colorOpacity: array<vec4f>;
@group(0) @binding(2) var<storage, read> objectIndices: array<u32>;
@group(0) @binding(3) var<storage, read> objectState: array<vec4u>;
@group(0) @binding(4) var<storage, read> tileCounts: array<u32>;
@group(0) @binding(5) var<storage, read> tileEntries: array<u32>;
@group(0) @binding(6) var<storage, read_write> tileAccumulation: array<vec4f>;
@group(0) @binding(7) var<uniform> accumulationMeta: AccumulationMeta;
@group(0) @binding(8) var<storage, read> scaleRotation: array<vec4f>;
@group(0) @binding(9) var<storage, read> tileOffsets: array<u32>;

@compute @workgroup_size(${WEBGPU_TILE_ACCUMULATION_WORKGROUP_SIZE})
fn accumulationMain(@builtin(global_invocation_id) globalId: vec3u) {
  let tileIndex = globalId.x;
  let tileCount = u32(accumulationMeta.tileCount);
  if (tileIndex >= tileCount) {
    return;
  }

  let tileColumns = max(1u, u32(accumulationMeta.tileColumns));
  let tileX = tileIndex % tileColumns;
  let tileY = tileIndex / tileColumns;
  let tileCenter = vec2f(
    (f32(tileX) + 0.5) * accumulationMeta.tileSize,
    (f32(tileY) + 0.5) * accumulationMeta.tileSize
  );
  let storedCount = tileCounts[tileIndex];
  let entryBase = tileOffsets[tileIndex];
  var accumulation = vec4f(0.0);

  for (var entryOffset = 0u; entryOffset < storedCount; entryOffset = entryOffset + 1u) {
    let gaussianIndex = tileEntries[entryBase + entryOffset];
    let objectIndex = objectIndices[gaussianIndex];
    if ((objectState[objectIndex].x & OBJECT_STATE_VISIBLE) == 0u) {
      continue;
    }

    let centerRadius = positionRadius[gaussianIndex];
    let screen = centerRadius.xy;
    let normalizedDepth = clamp(
      (centerRadius.z - accumulationMeta.depthMin) / max(accumulationMeta.depthSpan, 0.0001),
      0.0,
      1.0
    );
    let frontWeight = clamp(
      DEPTH_WEIGHT_FLOOR + (1.0 - DEPTH_WEIGHT_FLOOR) * exp(-DEPTH_WEIGHT_STRENGTH * normalizedDepth),
      DEPTH_WEIGHT_FLOOR,
      1.0
    );
    let gaussianScale = scaleRotation[gaussianIndex];
    let sigma = max(gaussianScale.xy, vec2f(0.0001));
    let cosine = cos(gaussianScale.z);
    let sine = sin(gaussianScale.z);
    let color = colorOpacity[gaussianIndex];

    for (var sampleIndex = 0u; sampleIndex < 4u; sampleIndex = sampleIndex + 1u) {
      let sampleOffset = vec2f(
        (f32(sampleIndex & 1u) - 0.5) * accumulationMeta.tileSize * 0.5,
        (f32((sampleIndex >> 1u) & 1u) - 0.5) * accumulationMeta.tileSize * 0.5
      );
      let delta = tileCenter + sampleOffset - screen;
      let rotated = vec2f(
        cosine * delta.x - sine * delta.y,
        sine * delta.x + cosine * delta.y
      );
      let normalized = rotated / sigma;
      let d = dot(normalized, normalized);
      if (d > RESOLVE_KERNEL_CUTOFF) {
        continue;
      }

      let weight = exp(-0.5 * d) * color.a * RESOLVE_ALPHA_GAIN * frontWeight * SAMPLE_WEIGHT;
      if (weight <= 0.0001) {
        continue;
      }
      accumulation = accumulation + vec4f(color.rgb * weight, weight);
    }
  }

  tileAccumulation[tileIndex] = accumulation;
}
`;

export const WEBGPU_TILE_COMPUTE_SHADER = `
const RESOLVE_ALPHA_SCALE = 0.18;

struct ComputeMeta {
  tileCount: u32,
  reserved0: u32,
  reserved1: u32,
  reserved2: u32,
};

@group(0) @binding(0) var<storage, read> tileAccumulation: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> tileResolvedRgba: array<vec4f>;
@group(0) @binding(2) var<uniform> computeMeta: ComputeMeta;

@compute @workgroup_size(${WEBGPU_TILE_COMPUTE_WORKGROUP_SIZE})
fn computeMain(@builtin(global_invocation_id) globalId: vec3u) {
  let tileIndex = globalId.x;
  if (tileIndex >= computeMeta.tileCount) {
    return;
  }

  let accumulation = tileAccumulation[tileIndex];
  let weight = accumulation.a;
  if (weight <= 0.0001) {
    tileResolvedRgba[tileIndex] = vec4f(0.0);
    return;
  }

  let color = accumulation.rgb / max(weight, 0.0001);
  let alpha = clamp(1.0 - exp(-weight * RESOLVE_ALPHA_SCALE), 0.0, 0.98);
  tileResolvedRgba[tileIndex] = vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), alpha);
}
`;

export const WEBGPU_PIXEL_RESOLVE_SHADER = `
const OBJECT_STATE_VISIBLE = 1u;
const RESOLVE_ALPHA_SCALE = 0.18;
const RESOLVE_ALPHA_GAIN = 0.78;
const PIXEL_COVERAGE_WEIGHT_FLOOR = 0.004;
const RESOLVE_KERNEL_CUTOFF = 13.0;
const DEPTH_WEIGHT_STRENGTH = 1.45;
const DEPTH_WEIGHT_FLOOR = 0.22;
const FRONT_DEPTH_GATE_STRENGTH = 12.0;
const FRONT_DEPTH_GATE_FLOOR = 0.06;
const PIXEL_DEPTH_BIN_COUNT = 8;

struct PixelResolveMeta {
  pixelCount: f32,
  viewportWidth: f32,
  viewportHeight: f32,
  tileSize: f32,
  tileColumns: f32,
  maxEntriesPerTile: f32,
  boundsMinX: f32,
  boundsMinZ: f32,
  boundsSpanX: f32,
  boundsSpanZ: f32,
  depthMin: f32,
  depthSpan: f32,
};

@group(0) @binding(0) var<storage, read> positionRadius: array<vec4f>;
@group(0) @binding(1) var<storage, read> colorOpacity: array<vec4f>;
@group(0) @binding(2) var<storage, read> objectIndices: array<u32>;
@group(0) @binding(3) var<storage, read> objectState: array<vec4u>;
@group(0) @binding(4) var<storage, read> tileCounts: array<u32>;
@group(0) @binding(5) var<storage, read> tileEntries: array<u32>;
@group(0) @binding(6) var<storage, read_write> pixelResolvedRgba: array<vec4f>;
@group(0) @binding(7) var<uniform> pixelResolveMeta: PixelResolveMeta;
@group(0) @binding(8) var<storage, read> scaleRotation: array<vec4f>;
@group(0) @binding(9) var<storage, read> tileOffsets: array<u32>;

@compute @workgroup_size(${WEBGPU_PIXEL_RESOLVE_WORKGROUP_SIZE})
fn pixelResolveMain(@builtin(global_invocation_id) globalId: vec3u) {
  let pixelIndex = globalId.x;
  if (pixelIndex >= u32(pixelResolveMeta.pixelCount)) {
    return;
  }

  let viewportWidth = max(u32(pixelResolveMeta.viewportWidth), 1u);
  let tileSize = max(u32(pixelResolveMeta.tileSize), 1u);
  let tileColumns = max(u32(pixelResolveMeta.tileColumns), 1u);
  let pixelX = pixelIndex % viewportWidth;
  let pixelY = pixelIndex / viewportWidth;
  let tileX = min(pixelX / tileSize, tileColumns - 1u);
  let tileY = pixelY / tileSize;
  let tileIndex = tileY * tileColumns + tileX;
  let storedCount = tileCounts[tileIndex];
  let entryBase = tileOffsets[tileIndex];
  let pixelCenter = vec2f(f32(pixelX) + 0.5, f32(pixelY) + 0.5);
  var binAccumulation: array<vec4f, PIXEL_DEPTH_BIN_COUNT>;
  var candidateCount = 0u;

  for (var binIndex = 0u; binIndex < u32(PIXEL_DEPTH_BIN_COUNT); binIndex = binIndex + 1u) {
    binAccumulation[binIndex] = vec4f(0.0);
  }

  for (var entryOffset = 0u; entryOffset < storedCount; entryOffset = entryOffset + 1u) {
    let gaussianIndex = tileEntries[entryBase + entryOffset];
    let objectIndex = objectIndices[gaussianIndex];
    if ((objectState[objectIndex].x & OBJECT_STATE_VISIBLE) == 0u) {
      continue;
    }

    let centerRadius = positionRadius[gaussianIndex];
    let screen = centerRadius.xy;
    let gaussianScale = scaleRotation[gaussianIndex];
    let sigma = max(gaussianScale.xy, vec2f(0.0001));
    let cosine = cos(gaussianScale.z);
    let sine = sin(gaussianScale.z);
    let delta = pixelCenter - screen;
    let rotated = vec2f(
      cosine * delta.x - sine * delta.y,
      sine * delta.x + cosine * delta.y
    );
    let normalized = rotated / sigma;
    let d = dot(normalized, normalized);
    if (d > RESOLVE_KERNEL_CUTOFF) {
      continue;
    }

    let color = colorOpacity[gaussianIndex];
    let candidateWeight = exp(-0.5 * d) * color.a * RESOLVE_ALPHA_GAIN;
    if (candidateWeight <= PIXEL_COVERAGE_WEIGHT_FLOOR) {
      continue;
    }
    let normalizedDepth = clamp(
      (centerRadius.z - pixelResolveMeta.depthMin) / max(pixelResolveMeta.depthSpan, 0.0001),
      0.0,
      0.999999
    );
    let depthBin = min(
      u32(floor(normalizedDepth * f32(PIXEL_DEPTH_BIN_COUNT))),
      u32(PIXEL_DEPTH_BIN_COUNT - 1)
    );
    binAccumulation[depthBin] = binAccumulation[depthBin] + vec4f(color.rgb * candidateWeight, candidateWeight);
    candidateCount = candidateCount + 1u;
  }

  if (candidateCount == 0u) {
    pixelResolvedRgba[pixelIndex] = vec4f(0.0);
    return;
  }

  var outputRgbPremultiplied = vec3f(0.0);
  var outputAlpha = 0.0;
  var totalWeight = 0.0;
  for (var binIndex = 0u; binIndex < u32(PIXEL_DEPTH_BIN_COUNT); binIndex = binIndex + 1u) {
    let bin = binAccumulation[binIndex];
    let weight = bin.a;
    if (weight <= PIXEL_COVERAGE_WEIGHT_FLOOR) {
      continue;
    }
    let color = bin.rgb / max(weight, 0.0001);
    let alpha = clamp(1.0 - exp(-weight * RESOLVE_ALPHA_SCALE), 0.0, 0.98);
    let visibility = 1.0 - outputAlpha;
    outputRgbPremultiplied = outputRgbPremultiplied + visibility * color * alpha;
    outputAlpha = outputAlpha + visibility * alpha;
    totalWeight = totalWeight + weight;
    if (outputAlpha >= 0.995) {
      break;
    }
  }

  if (outputAlpha <= 0.0001 || totalWeight <= 0.0001) {
    pixelResolvedRgba[pixelIndex] = vec4f(0.0);
    return;
  }

  let color = outputRgbPremultiplied / max(outputAlpha, 0.0001);
  pixelResolvedRgba[pixelIndex] = vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), clamp(outputAlpha, 0.0, 0.98));
}
`;

export function createWebGpuComputeMeta(tileSmoke) {
  return new Uint32Array([
    Math.max(1, tileSmoke?.tileCount ?? 1),
    0,
    0,
    0,
  ]);
}

export function webGpuComputeWorkgroups(tileSmoke) {
  const tileCount = Math.max(1, tileSmoke?.tileCount ?? 1);
  return Math.ceil(tileCount / WEBGPU_TILE_COMPUTE_WORKGROUP_SIZE);
}

export function createWebGpuAccumulationMeta(tileSmoke) {
  return new Float32Array([
    Math.max(1, tileSmoke?.tileCount ?? 1),
    Math.max(1, tileSmoke?.maxEntriesPerTile ?? 1),
    Math.max(1, tileSmoke?.tileColumns ?? 1),
    Math.max(1, tileSmoke?.tileSize ?? 1),
    Math.max(1, tileSmoke?.viewportWidth ?? 1),
    Math.max(1, tileSmoke?.viewportHeight ?? 1),
    Number.isFinite(tileSmoke?.boundsMinX) ? tileSmoke.boundsMinX : -1,
    Number.isFinite(tileSmoke?.boundsMinZ) ? tileSmoke.boundsMinZ : -1,
    Math.max(0.0001, tileSmoke?.boundsSpanX ?? 2),
    Math.max(0.0001, tileSmoke?.boundsSpanZ ?? 2),
    Number.isFinite(tileSmoke?.projectionDepthMin) ? tileSmoke.projectionDepthMin : 0,
    Math.max(0.0001, tileSmoke?.projectionDepthSpan ?? 1),
  ]);
}

export function webGpuAccumulationWorkgroups(tileSmoke) {
  const tileCount = Math.max(1, tileSmoke?.tileCount ?? 1);
  return Math.ceil(tileCount / WEBGPU_TILE_ACCUMULATION_WORKGROUP_SIZE);
}

export function createWebGpuPixelResolveMeta(tileSmoke) {
  const pixelCount = Math.max(
    1,
    tileSmoke?.pixelCount ??
      Math.max(1, tileSmoke?.viewportWidth ?? 1) * Math.max(1, tileSmoke?.viewportHeight ?? 1),
  );
  return new Float32Array([
    pixelCount,
    Math.max(1, tileSmoke?.viewportWidth ?? 1),
    Math.max(1, tileSmoke?.viewportHeight ?? 1),
    Math.max(1, tileSmoke?.tileSize ?? 1),
    Math.max(1, tileSmoke?.tileColumns ?? 1),
    Math.max(1, tileSmoke?.maxEntriesPerTile ?? 1),
    Number.isFinite(tileSmoke?.boundsMinX) ? tileSmoke.boundsMinX : -1,
    Number.isFinite(tileSmoke?.boundsMinZ) ? tileSmoke.boundsMinZ : -1,
    Math.max(0.0001, tileSmoke?.boundsSpanX ?? 2),
    Math.max(0.0001, tileSmoke?.boundsSpanZ ?? 2),
    Number.isFinite(tileSmoke?.projectionDepthMin) ? tileSmoke.projectionDepthMin : 0,
    Math.max(0.0001, tileSmoke?.projectionDepthSpan ?? 1),
  ]);
}

export function webGpuPixelResolveWorkgroups(tileSmoke) {
  const pixelCount = Math.max(
    1,
    tileSmoke?.pixelCount ??
      Math.max(1, tileSmoke?.viewportWidth ?? 1) * Math.max(1, tileSmoke?.viewportHeight ?? 1),
  );
  return Math.ceil(pixelCount / WEBGPU_PIXEL_RESOLVE_WORKGROUP_SIZE);
}
