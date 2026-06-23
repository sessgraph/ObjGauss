export const WEBGPU_SAMPLED_TEXTURE_RESOLVE_SOURCE = "webgpu-sampled-texture-resolve-v1";
export const WEBGPU_FLOAT_TEXTURE_COPY_RESOLVE_SOURCE = "webgpu-buffer-copy-texture-resolve-v1";

const FULLSCREEN_VERTEX_SHADER = `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

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
`;

export const WEBGPU_SAMPLED_TEXTURE_RESOLVE_SHADER = `
${FULLSCREEN_VERTEX_SHADER}

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var sourceSampler: sampler;

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let uv = clamp(input.uv, vec2f(0.0), vec2f(0.999999));
  let color = textureSampleLevel(sourceTexture, sourceSampler, uv, 0.0);
  return vec4f(clamp(color.rgb, vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

export const WEBGPU_FLOAT_TEXTURE_LOAD_RESOLVE_SHADER = `
${FULLSCREEN_VERTEX_SHADER}

struct ResolveMeta {
  viewportWidth: u32,
  viewportHeight: u32,
  reserved0: u32,
  reserved1: u32,
};

@group(0) @binding(0) var sourceTexture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> resolveMeta: ResolveMeta;

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let safeWidth = max(resolveMeta.viewportWidth, 1u);
  let safeHeight = max(resolveMeta.viewportHeight, 1u);
  let uv = clamp(input.uv, vec2f(0.0), vec2f(0.999999));
  let pixelX = min(u32(floor(uv.x * f32(safeWidth))), safeWidth - 1u);
  let pixelY = min(u32(floor(uv.y * f32(safeHeight))), safeHeight - 1u);
  let pixel = textureLoad(sourceTexture, vec2i(i32(pixelX), i32(pixelY)), 0);
  let background = vec3f(0.0627, 0.0745, 0.0863);
  let alpha = clamp(pixel.a, 0.0, 0.98);
  let rgb = background * (1.0 - alpha) + clamp(pixel.rgb, vec3f(0.0), vec3f(1.0)) * alpha;
  return vec4f(rgb, 1.0);
}
`;
