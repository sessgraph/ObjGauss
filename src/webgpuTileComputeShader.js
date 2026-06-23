export const WEBGPU_TILE_COMPUTE_SOURCE = "webgpu-compute-resolve-v1";
export const WEBGPU_TILE_COMPUTE_WORKGROUP_SIZE = 64;

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
