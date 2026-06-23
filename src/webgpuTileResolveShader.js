export const WEBGPU_TILE_RESOLVE_SOURCE = "webgpu-pixel-storage-resolve-v1";

export const WEBGPU_TILE_RESOLVE_SHADER = `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

struct ResolveMeta {
  viewportWidth: u32,
  viewportHeight: u32,
  reserved0: u32,
  reserved1: u32,
};

@group(0) @binding(0) var<storage, read> pixelResolvedRgba: array<vec4f>;
@group(0) @binding(1) var<uniform> resolveMeta: ResolveMeta;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  let position = positions[vertexIndex];
  var output: VertexOutput;
  output.position = vec4f(position, 0.0, 1.0);
  output.uv = position * 0.5 + vec2f(0.5);
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let safeWidth = max(resolveMeta.viewportWidth, 1u);
  let safeHeight = max(resolveMeta.viewportHeight, 1u);
  let uv = clamp(input.uv, vec2f(0.0), vec2f(0.999999));
  let pixelX = min(u32(floor(uv.x * f32(safeWidth))), safeWidth - 1u);
  let pixelY = min(u32(floor(uv.y * f32(safeHeight))), safeHeight - 1u);
  let pixelIndex = pixelY * safeWidth + pixelX;
  let pixel = pixelResolvedRgba[pixelIndex];
  let background = vec3f(0.0627, 0.0745, 0.0863);
  let alpha = clamp(pixel.a, 0.0, 0.98);
  let rgb = background * (1.0 - alpha) + clamp(pixel.rgb, vec3f(0.0), vec3f(1.0)) * alpha;
  return vec4f(rgb, 1.0);
}
`;

export function createWebGpuResolveMeta(tileSmoke) {
  return new Uint32Array([
    Math.max(1, tileSmoke?.viewportWidth ?? 1),
    Math.max(1, tileSmoke?.viewportHeight ?? 1),
    0,
    0,
  ]);
}
