export const WEBGPU_TILE_COMPUTE_SOURCE = "webgpu-compute-resolve-v1";
export const WEBGPU_TILE_COMPUTE_WORKGROUP_SIZE = 64;
export const WEBGPU_TILE_ACCUMULATION_SOURCE = "webgpu-compute-accumulation-v1";
export const WEBGPU_TILE_ACCUMULATION_WORKGROUP_SIZE = 64;

export const WEBGPU_TILE_ACCUMULATION_SHADER = `
const OBJECT_STATE_VISIBLE = 1u;
const RESOLVE_ALPHA_GAIN = 0.78;
const RESOLVE_KERNEL_CUTOFF = 13.0;

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
  reserved0: f32,
  reserved1: f32,
};

@group(0) @binding(0) var<storage, read> positionRadius: array<vec4f>;
@group(0) @binding(1) var<storage, read> colorOpacity: array<vec4f>;
@group(0) @binding(2) var<storage, read> objectIndices: array<u32>;
@group(0) @binding(3) var<storage, read> objectState: array<vec4u>;
@group(0) @binding(4) var<storage, read> tileCounts: array<u32>;
@group(0) @binding(5) var<storage, read> tileEntries: array<u32>;
@group(0) @binding(6) var<storage, read_write> tileAccumulation: array<vec4f>;
@group(0) @binding(7) var<uniform> accumulationMeta: AccumulationMeta;

@compute @workgroup_size(${WEBGPU_TILE_ACCUMULATION_WORKGROUP_SIZE})
fn accumulationMain(@builtin(global_invocation_id) globalId: vec3u) {
  let tileIndex = globalId.x;
  let tileCount = u32(accumulationMeta.tileCount);
  if (tileIndex >= tileCount) {
    return;
  }

  let maxEntriesPerTile = u32(accumulationMeta.maxEntriesPerTile);
  let tileColumns = max(1u, u32(accumulationMeta.tileColumns));
  let tileX = tileIndex % tileColumns;
  let tileY = tileIndex / tileColumns;
  let tileCenter = vec2f(
    (f32(tileX) + 0.5) * accumulationMeta.tileSize,
    (f32(tileY) + 0.5) * accumulationMeta.tileSize
  );
  let storedCount = min(tileCounts[tileIndex], maxEntriesPerTile);
  let entryBase = tileIndex * maxEntriesPerTile;
  var accumulation = vec4f(0.0);

  for (var entryOffset = 0u; entryOffset < storedCount; entryOffset = entryOffset + 1u) {
    let gaussianIndex = tileEntries[entryBase + entryOffset];
    let objectIndex = objectIndices[gaussianIndex];
    if ((objectState[objectIndex].x & OBJECT_STATE_VISIBLE) == 0u) {
      continue;
    }

    let centerRadius = positionRadius[gaussianIndex];
    let screen = vec2f(
      ((centerRadius.x - accumulationMeta.boundsMinX) / max(accumulationMeta.boundsSpanX, 0.0001)) *
        max(1.0, accumulationMeta.viewportWidth - 1.0),
      (1.0 - ((centerRadius.y - accumulationMeta.boundsMinZ) / max(accumulationMeta.boundsSpanZ, 0.0001))) *
        max(1.0, accumulationMeta.viewportHeight - 1.0)
    );
    let sigma = max(centerRadius.w / 3.0, 0.0001);
    let delta = tileCenter - screen;
    let d = dot(delta, delta) / (sigma * sigma);
    if (d > RESOLVE_KERNEL_CUTOFF) {
      continue;
    }

    let color = colorOpacity[gaussianIndex];
    let weight = exp(-0.5 * d) * color.a * RESOLVE_ALPHA_GAIN;
    if (weight <= 0.0001) {
      continue;
    }
    accumulation = accumulation + vec4f(color.rgb * weight, weight);
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
    0,
    0,
  ]);
}

export function webGpuAccumulationWorkgroups(tileSmoke) {
  const tileCount = Math.max(1, tileSmoke?.tileCount ?? 1);
  return Math.ceil(tileCount / WEBGPU_TILE_ACCUMULATION_WORKGROUP_SIZE);
}
