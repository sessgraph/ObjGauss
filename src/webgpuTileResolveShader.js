export const WEBGPU_TILE_RESOLVE_SOURCE = "webgpu-pixel-storage-resolve-v1";
export const WEBGPU_TILE_RESOLVE_FILTER = "bilinear-storage";
export const WEBGPU_TILE_ALPHA_PRESENTATION_MODE = "alpha-edge-gated-presentation-v1";
export const WEBGPU_TILE_ALPHA_PRESENTATION_TUNING_MODE = "runtime-alpha-presentation-tuning-v1";
export const WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR = 0.035;
export const WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR_MIN = 0;
export const WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR_MAX = 0.2;

export const WEBGPU_TILE_RESOLVE_SHADER = createWebGpuTileResolveShader();

export function createWebGpuTileResolveShader(tuning = null) {
  const resolved = normalizeWebGpuAlphaPresentationTuning(tuning);
  return `
const ALPHA_PRESENTATION_FLOOR = ${resolved.alphaPresentationFloor.toFixed(6)};

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

fn samplePixel(pixelX: u32, pixelY: u32, safeWidth: u32) -> vec4f {
  return pixelResolvedRgba[pixelY * safeWidth + pixelX];
}

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
  let pixelPosition = uv * vec2f(f32(safeWidth - 1u), f32(safeHeight - 1u));
  let baseX = min(u32(floor(pixelPosition.x)), safeWidth - 1u);
  let baseY = min(u32(floor(pixelPosition.y)), safeHeight - 1u);
  let nextX = min(baseX + 1u, safeWidth - 1u);
  let nextY = min(baseY + 1u, safeHeight - 1u);
  let blend = fract(pixelPosition);
  let top = mix(
    samplePixel(baseX, baseY, safeWidth),
    samplePixel(nextX, baseY, safeWidth),
    blend.x
  );
  let bottom = mix(
    samplePixel(baseX, nextY, safeWidth),
    samplePixel(nextX, nextY, safeWidth),
    blend.x
  );
  let pixel = mix(top, bottom, blend.y);
  let background = vec3f(0.0627, 0.0745, 0.0863);
  var alpha = clamp(pixel.a, 0.0, 0.98);
  if (alpha < ALPHA_PRESENTATION_FLOOR) {
    alpha = 0.0;
  }
  let rgb = background * (1.0 - alpha) + clamp(pixel.rgb, vec3f(0.0), vec3f(1.0)) * alpha;
  return vec4f(rgb, 1.0);
}
`;
}

export function normalizeWebGpuAlphaPresentationTuning(tuning = null) {
  const requested = Number(tuning?.alphaPresentationFloor ?? WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR);
  const alphaPresentationFloor = Number.isFinite(requested)
    ? clampNumber(
        requested,
        WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR_MIN,
        WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR_MAX,
      )
    : WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR;
  return {
    mode: WEBGPU_TILE_ALPHA_PRESENTATION_TUNING_MODE,
    alphaPresentationFloor,
  };
}

export function createWebGpuResolveMeta(tileSmoke) {
  return new Uint32Array([
    Math.max(1, tileSmoke?.viewportWidth ?? 1),
    Math.max(1, tileSmoke?.viewportHeight ?? 1),
    0,
    0,
  ]);
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
