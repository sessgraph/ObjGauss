import { MAX_SPLAT_DISPLAY_MULTIPLIER } from "./splat-format.mjs";
import { sortSplatIndices } from "./splat-sort.mjs";

const FIRST_FRAME_TIMEOUT_MS = 30_000;
const MAX_RENDER_DIMENSION = 4_096;

const VERTEX_SHADER = `#version 300 es
precision highp float;
precision highp int;

layout(location = 0) in vec2 aCorner;
layout(location = 1) in uint aIndex;

uniform sampler2D uSplatData;
uniform int uTextureWidth;
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec2 uViewport;
uniform float uFocalX;
uniform float uFocalY;
uniform float uNear;
uniform float uScaleMultiplier;

out vec2 vLocal;
out vec4 vColor;

vec4 readSplat(uint index, int offset) {
  int address = int(index) * 4 + offset;
  return texelFetch(uSplatData, ivec2(address % uTextureWidth, address / uTextureWidth), 0);
}

void main() {
  vec4 positionAlpha = readSplat(aIndex, 0);
  vec4 color = readSplat(aIndex, 1);
  vec4 covarianceA = readSplat(aIndex, 2);
  vec4 covarianceB = readSplat(aIndex, 3);
  vec4 centerView = uView * vec4(positionAlpha.xyz, 1.0);
  float depth = -centerView.z;
  vec4 clip = uProjection * centerView;
  vLocal = aCorner * 3.0;
  vColor = vec4(color.rgb, positionAlpha.w);

  if (depth <= uNear || positionAlpha.w <= 0.0
      || any(isnan(centerView)) || any(isinf(centerView))
      || any(isnan(clip)) || any(isinf(clip))
      || abs(clip.x) > abs(clip.w) * 4.0
      || abs(clip.y) > abs(clip.w) * 4.0) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  mat3 covarianceWorld = mat3(
    covarianceA.x, covarianceA.y, covarianceA.z,
    covarianceA.y, covarianceA.w, covarianceB.x,
    covarianceA.z, covarianceB.x, covarianceB.y
  );
  mat3 viewRotation = mat3(uView);
  mat3 covarianceCamera = viewRotation * covarianceWorld * transpose(viewRotation);
  covarianceCamera *= uScaleMultiplier * uScaleMultiplier;

  float depthSquared = depth * depth;
  // centerView.z is negative in the OpenGL camera convention, so the
  // derivative with respect to camera-z has a positive sign here.
  vec3 jacobianX = vec3(uFocalX / depth, 0.0, uFocalX * centerView.x / depthSquared);
  vec3 jacobianY = vec3(0.0, uFocalY / depth, uFocalY * centerView.y / depthSquared);
  float covarianceXX = dot(jacobianX, covarianceCamera * jacobianX) + 0.30;
  float covarianceXY = dot(jacobianX, covarianceCamera * jacobianY);
  float covarianceYY = dot(jacobianY, covarianceCamera * jacobianY) + 0.30;

  float trace = covarianceXX + covarianceYY;
  float delta = covarianceXX - covarianceYY;
  float doubledXY = 2.0 * covarianceXY;
  float discriminantScale = max(abs(delta), abs(doubledXY));
  float discriminant = discriminantScale == 0.0
    ? 0.0
    : discriminantScale * sqrt(
      (delta / discriminantScale) * (delta / discriminantScale)
      + (doubledXY / discriminantScale) * (doubledXY / discriminantScale)
    );
  float lambdaMajor = max(0.10, 0.5 * (trace + discriminant));
  float lambdaMinor = max(0.10, 0.5 * (trace - discriminant));

  vec2 majorDirection;
  if (abs(covarianceXY) > 1e-5) {
    majorDirection = normalize(vec2(covarianceXY, lambdaMajor - covarianceXX));
  } else {
    majorDirection = covarianceXX >= covarianceYY ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  }
  vec2 minorDirection = vec2(-majorDirection.y, majorDirection.x);
  float majorRadius = min(sqrt(lambdaMajor), 512.0);
  float minorRadius = min(sqrt(lambdaMinor), 512.0);
  vec2 offsetPixels = majorDirection * vLocal.x * majorRadius
    + minorDirection * vLocal.y * minorRadius;

  clip.xy += (2.0 * offsetPixels / uViewport) * clip.w;
  gl_Position = clip;
}
`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 vLocal;
in vec4 vColor;
out vec4 fragmentColor;

void main() {
  float radiusSquared = dot(vLocal, vLocal);
  if (radiusSquared > 9.0) {
    discard;
  }
  float alpha = vColor.a * exp(-0.5 * radiusSquared);
  if (alpha < (1.0 / 255.0)) {
    discard;
  }
  fragmentColor = vec4(vColor.rgb * alpha, alpha);
}
`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  if (shader === null) {
    throw new Error("WebGL2 could not allocate a shader");
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) || "unknown shader error";
    gl.deleteShader(shader);
    throw new Error(`WebGL2 shader compilation failed: ${log}`);
  }
  return shader;
}

function createProgram(gl, vertexSource = VERTEX_SHADER, fragmentSource = FRAGMENT_SHADER) {
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const program = gl.createProgram();
  if (program === null) {
    throw new Error("WebGL2 could not allocate a program");
  }
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) || "unknown link error";
    gl.deleteProgram(program);
    throw new Error(`WebGL2 program linking failed: ${log}`);
  }
  return program;
}

export function perspectiveMatrix(fieldOfViewRadians, aspect, near, far) {
  if (![fieldOfViewRadians, aspect, near, far].every(Number.isFinite)
    || fieldOfViewRadians <= 0
    || aspect <= 0
    || near <= 0
    || far <= near) {
    throw new RangeError("perspective parameters must define a finite positive frustum");
  }
  const focal = 1 / Math.tan(fieldOfViewRadians / 2);
  const rangeInverse = 1 / (near - far);
  return Float32Array.from([
    focal / aspect, 0, 0, 0,
    0, focal, 0, 0,
    0, 0, (far + near) * rangeInverse, -1,
    0, 0, 2 * far * near * rangeInverse, 0,
  ]);
}

export function lookAtMatrix(eye, target, up = [0, 1, 0]) {
  if (eye?.length !== 3 || target?.length !== 3 || up?.length !== 3) {
    throw new TypeError("eye, target, and up must be xyz triples");
  }
  let zx = eye[0] - target[0];
  let zy = eye[1] - target[1];
  let zz = eye[2] - target[2];
  let length = Math.hypot(zx, zy, zz);
  if (!Number.isFinite(length) || length <= 1e-8) {
    throw new RangeError("eye and target must be distinct finite points");
  }
  zx /= length;
  zy /= length;
  zz /= length;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  length = Math.hypot(xx, xy, xz);
  if (!Number.isFinite(length) || length <= 1e-8) {
    throw new RangeError("up cannot be parallel to the view direction");
  }
  xx /= length;
  xy /= length;
  xz /= length;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  return Float32Array.from([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
    -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
    -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
    1,
  ]);
}

export function centerPositionsForRendering(positions, center) {
  if (!(positions instanceof Float32Array) || positions.length === 0 || positions.length % 3 !== 0) {
    throw new TypeError("positions must be a non-empty Float32Array of xyz triples");
  }
  if (center == null || center.length !== 3 || !Array.from(center).every(Number.isFinite)) {
    throw new TypeError("center must contain three finite components");
  }
  const centered = new Float32Array(positions.length);
  for (let index = 0; index < positions.length; index += 3) {
    centered[index] = positions[index] - center[0];
    centered[index + 1] = positions[index + 1] - center[1];
    centered[index + 2] = positions[index + 2] - center[2];
  }
  if (!centered.every(Number.isFinite)) {
    throw new RangeError("centered render positions must remain finite float32 values");
  }
  return centered;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function uniform(gl, program, name) {
  const location = gl.getUniformLocation(program, name);
  if (location === null) {
    throw new Error(`WebGL2 uniform ${name} is unavailable`);
  }
  return location;
}

export class SplatRenderer {
  constructor(canvas, { onFatal = () => {}, onStats = () => {} } = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError("SplatRenderer requires a canvas element");
    }
    this.canvas = canvas;
    this.onFatal = onFatal;
    this.onStats = onStats;
    this.gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      depth: false,
      powerPreference: "high-performance",
      preserveDrawingBuffer: true,
    });
    if (this.gl === null) {
      throw new Error("当前浏览器或 GPU 不支持 WebGL2，不能渲染 3D Gaussian");
    }

    this.program = createProgram(this.gl);
    this.uniforms = {
      splatData: uniform(this.gl, this.program, "uSplatData"),
      textureWidth: uniform(this.gl, this.program, "uTextureWidth"),
      view: uniform(this.gl, this.program, "uView"),
      projection: uniform(this.gl, this.program, "uProjection"),
      viewport: uniform(this.gl, this.program, "uViewport"),
      focalX: uniform(this.gl, this.program, "uFocalX"),
      focalY: uniform(this.gl, this.program, "uFocalY"),
      near: uniform(this.gl, this.program, "uNear"),
      scaleMultiplier: uniform(this.gl, this.program, "uScaleMultiplier"),
    };
    this._createGeometry();

    this.texture = null;
    this.textureWidth = 0;
    this.totalCount = 0;
    this.drawCount = 0;
    this.loaded = false;
    this.blocked = false;
    this.generation = 0;
    this.requestId = 0;
    this.sortPending = false;
    this.sortDirty = false;
    this.lastSortRequestedAt = 0;
    this.sortMs = 0;
    this.sortTimeoutId = null;
    this.scaleMultiplier = 1;
    this.autoRotate = true;
    this.fieldOfView = Math.PI / 4;
    this.target = Float32Array.from([0, 0, 0]);
    this.homeTarget = Float32Array.from(this.target);
    this.boundsMin = Float32Array.from([-1, -1, -1]);
    this.boundsMax = Float32Array.from([1, 1, 1]);
    this.boundsRadius = 1;
    this.camera = { yaw: 0.72, pitch: 0.32, distance: 3 };
    this.cameraMode = "fit";
    this.viewMatrix = new Float32Array(16);
    this.projectionMatrix = new Float32Array(16);
    this.near = 0.01;
    this.pointers = new Map();
    this.lastPinchDistance = null;
    this.lastFrameAt = performance.now();
    this.lastStatsAt = 0;
    this.fps = 0;
    this.firstSort = null;
    this.firstFrame = null;
    this.firstFrameTimeoutId = null;

    this.worker = new Worker(new URL("./splat-sort-worker.mjs", import.meta.url), { type: "module" });
    this.worker.addEventListener("message", (event) => this._handleWorkerMessage(event));
    this.worker.addEventListener("error", (event) => {
      this._fatal(`深度排序 Worker 失败：${event.message || "unknown error"}`);
    });

    this._bindInteraction();
    this.resizeObserver = new ResizeObserver(() => this._resize());
    this.resizeObserver.observe(canvas);
    canvas.addEventListener("webglcontextlost", (event) => {
      event.preventDefault();
      this._fatal("WebGL2 context 已丢失；请刷新页面重新加载");
    });

    this.gl.disable(this.gl.DEPTH_TEST);
    this.gl.disable(this.gl.CULL_FACE);
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.ONE, this.gl.ONE_MINUS_SRC_ALPHA);
    this.animationFrame = requestAnimationFrame((time) => this._frame(time));
  }

  _createGeometry() {
    const gl = this.gl;
    this.vao = gl.createVertexArray();
    this.quadBuffer = gl.createBuffer();
    this.indexBuffer = gl.createBuffer();
    if (this.vao === null || this.quadBuffer === null || this.indexBuffer === null) {
      throw new Error("WebGL2 could not allocate renderer buffers");
    }
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      Float32Array.from([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, 4, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.bindVertexArray(null);

  }

  _packTexture(parsed) {
    const gl = this.gl;
    const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    const textureWidth = Math.min(2_048, maxTextureSize);
    const textureRows = Math.ceil((parsed.count * 4) / textureWidth);
    if (textureRows > maxTextureSize) {
      throw new RangeError(
        `${parsed.count.toLocaleString()} splats exceed this GPU's texture capacity`,
      );
    }

    const packed = new Float32Array(textureWidth * textureRows * 4);
    for (let index = 0; index < parsed.count; index += 1) {
      const positionOffset = index * 3;
      const colorOffset = index * 4;
      const packedOffset = index * 16;
      const scaleX2 = parsed.scales[positionOffset] ** 2;
      const scaleY2 = parsed.scales[positionOffset + 1] ** 2;
      const scaleZ2 = parsed.scales[positionOffset + 2] ** 2;
      const w = parsed.quaternions[colorOffset];
      const x = parsed.quaternions[colorOffset + 1];
      const y = parsed.quaternions[colorOffset + 2];
      const z = parsed.quaternions[colorOffset + 3];
      const r00 = 1 - 2 * (y * y + z * z);
      const r01 = 2 * (x * y - z * w);
      const r02 = 2 * (x * z + y * w);
      const r10 = 2 * (x * y + z * w);
      const r11 = 1 - 2 * (x * x + z * z);
      const r12 = 2 * (y * z - x * w);
      const r20 = 2 * (x * z - y * w);
      const r21 = 2 * (y * z + x * w);
      const r22 = 1 - 2 * (x * x + y * y);
      packed[packedOffset] = parsed.positions[positionOffset];
      packed[packedOffset + 1] = parsed.positions[positionOffset + 1];
      packed[packedOffset + 2] = parsed.positions[positionOffset + 2];
      packed[packedOffset + 3] = parsed.colors[colorOffset + 3] / 255;
      packed[packedOffset + 4] = parsed.colors[colorOffset] / 255;
      packed[packedOffset + 5] = parsed.colors[colorOffset + 1] / 255;
      packed[packedOffset + 6] = parsed.colors[colorOffset + 2] / 255;
      const covariance = [
        r00 * r00 * scaleX2 + r01 * r01 * scaleY2 + r02 * r02 * scaleZ2,
        r00 * r10 * scaleX2 + r01 * r11 * scaleY2 + r02 * r12 * scaleZ2,
        r00 * r20 * scaleX2 + r01 * r21 * scaleY2 + r02 * r22 * scaleZ2,
        r10 * r10 * scaleX2 + r11 * r11 * scaleY2 + r12 * r12 * scaleZ2,
        r10 * r20 * scaleX2 + r11 * r21 * scaleY2 + r12 * r22 * scaleZ2,
        r20 * r20 * scaleX2 + r21 * r21 * scaleY2 + r22 * r22 * scaleZ2,
      ];
      if (!covariance.every((value) => Number.isFinite(
        Math.fround(value * MAX_SPLAT_DISPLAY_MULTIPLIER * MAX_SPLAT_DISPLAY_MULTIPLIER),
      ))) {
        throw new RangeError(
          `record ${index} covariance exceeds the GPU-safe range at ${MAX_SPLAT_DISPLAY_MULTIPLIER}× display scale`,
        );
      }
      packed.set(covariance, packedOffset + 8);
    }

    const texture = gl.createTexture();
    if (texture === null) {
      throw new Error("WebGL2 could not allocate the Gaussian texture");
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      textureWidth,
      textureRows,
      0,
      gl.RGBA,
      gl.FLOAT,
      packed,
    );
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      gl.deleteTexture(texture);
      throw new Error(`WebGL2 Gaussian texture upload failed (error ${error})`);
    }
    return { texture, textureWidth };
  }

  load(parsed, { cameraMode = "fit" } = {}) {
    this.clear("replaced");
    this.blocked = false;
    const generation = this.generation;
    const renderOrigin = Float32Array.from(parsed.bounds.center);
    const centeredPositions = centerPositionsForRendering(parsed.positions, renderOrigin);
    const centeredBounds = {
      min: Float32Array.from(parsed.bounds.min, (value, index) => value - renderOrigin[index]),
      max: Float32Array.from(parsed.bounds.max, (value, index) => value - renderOrigin[index]),
      center: Float32Array.from([0, 0, 0]),
      radius: parsed.bounds.radius,
    };
    const renderData = { ...parsed, positions: centeredPositions, bounds: centeredBounds };
    let textureResult;
    try {
      textureResult = this._packTexture(renderData);
    } catch (error) {
      this._fatal(error instanceof Error ? error.message : String(error));
      return Promise.reject(error);
    }

    this.texture = textureResult.texture;
    this.textureWidth = textureResult.textureWidth;
    this.totalCount = parsed.count;
    this.target = Float32Array.from(centeredBounds.center);
    this.boundsMin = Float32Array.from(centeredBounds.min);
    this.boundsMax = Float32Array.from(centeredBounds.max);
    this.boundsRadius = Math.max(parsed.bounds.radius, 0.01);
    this.cameraMode = cameraMode;
    if (cameraMode === "immersive") {
      this.target[1] = centeredBounds.min[1] + 2;
    }
    this.homeTarget = Float32Array.from(this.target);
    this.resetCamera();

    const positionCopy = centeredPositions.slice();
    try {
      this.worker.postMessage(
        { type: "init", generation, positions: positionCopy.buffer },
        [positionCopy.buffer],
      );
    } catch (error) {
      const message = `无法初始化深度排序：${error instanceof Error ? error.message : String(error)}`;
      this._fatal(message);
      return Promise.reject(new Error(message));
    }

    this.loaded = true;
    this._updateMatrices();
    this.sortDirty = true;
    const firstFrameReady = new Promise((resolve, reject) => {
      this.firstFrame = { generation, resolve, reject };
    });
    this._clearFirstFrameTimeout();
    this.firstFrameTimeoutId = setTimeout(() => {
      if (this.firstFrame?.generation === generation) {
        this._fatal("首个 WebGL2 Gaussian frame 超过 30 秒未完成；渲染已按 fail-closed 阻断");
      }
    }, FIRST_FRAME_TIMEOUT_MS);
    if (parsed.count <= 150_000) {
      const startedAt = performance.now();
      try {
        const indices = sortSplatIndices(centeredPositions, this.viewMatrix);
        this._uploadSortedIndices(indices, performance.now() - startedAt);
      } catch (error) {
        this._fatal(error instanceof Error ? error.message : String(error));
        return Promise.reject(error);
      }
      this.sortDirty = false;
      this.lastSortRequestedAt = performance.now();
      return firstFrameReady;
    }
    const ready = new Promise((resolve, reject) => {
      this.firstSort = { generation, resolve, reject };
    });
    this._requestSort(true);
    return Promise.all([ready, firstFrameReady]).then(() => undefined);
  }

  clear(reason = "cleared") {
    this.generation += 1;
    this.loaded = false;
    this.drawCount = 0;
    this.totalCount = 0;
    this.sortPending = false;
    this.sortDirty = false;
    this._clearSortTimeout();
    this._clearFirstFrameTimeout();
    if (this.texture !== null) {
      this.gl.deleteTexture(this.texture);
      this.texture = null;
    }
    if (this.firstSort !== null) {
      this.firstSort.reject(new Error(`Gaussian load ${reason}`));
      this.firstSort = null;
    }
    if (this.firstFrame !== null) {
      this.firstFrame.reject(new Error(`Gaussian frame ${reason}`));
      this.firstFrame = null;
    }
  }

  _fatal(message) {
    this.generation += 1;
    this.blocked = true;
    this.loaded = false;
    this.drawCount = 0;
    this.totalCount = 0;
    this.sortPending = false;
    this._clearSortTimeout();
    this._clearFirstFrameTimeout();
    if (this.texture !== null) {
      this.gl.deleteTexture(this.texture);
      this.texture = null;
    }
    if (this.firstSort !== null) {
      this.firstSort.reject(new Error(message));
      this.firstSort = null;
    }
    if (this.firstFrame !== null) {
      this.firstFrame.reject(new Error(message));
      this.firstFrame = null;
    }
    this.onFatal(message);
  }

  _handleWorkerMessage(event) {
    const message = event.data;
    if (message.generation !== this.generation) {
      return;
    }
    if (message.type === "error") {
      this._fatal(`深度排序失败：${message.message}`);
      return;
    }
    if (message.type !== "sorted") {
      return;
    }
    this._clearSortTimeout();

    const indices = new Uint32Array(message.indices);
    if (indices.length !== this.totalCount) {
      this._fatal("深度排序返回了错误数量的 splat 索引");
      return;
    }
    try {
      this._uploadSortedIndices(indices, message.sortMs);
    } catch (error) {
      this._fatal(error instanceof Error ? error.message : String(error));
      return;
    }
    this.sortPending = false;
    if (this.firstSort?.generation === this.generation) {
      this.firstSort.resolve();
      this.firstSort = null;
    }
    if (this.sortDirty) {
      this._requestSort(false);
    }
  }

  _uploadSortedIndices(indices, sortMs) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.indexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, indices, gl.DYNAMIC_DRAW);
    if (gl.getError() !== gl.NO_ERROR) {
      throw new Error("WebGL2 无法上传排序后的 splat 索引");
    }
    this.drawCount = indices.length;
    this.sortMs = sortMs;
  }

  _requestSort(force) {
    if (!this.loaded || this.sortPending) {
      return;
    }
    const now = performance.now();
    if (!force && now - this.lastSortRequestedAt < 90) {
      return;
    }
    this._updateMatrices();
    this.sortDirty = false;
    this.sortPending = true;
    this.lastSortRequestedAt = now;
    this.requestId += 1;
    const generation = this.generation;
    const requestId = this.requestId;
    try {
      this.worker.postMessage({
        type: "sort",
        generation,
        requestId,
        viewMatrix: Array.from(this.viewMatrix),
      });
    } catch (error) {
      this.sortPending = false;
      this._fatal(`无法请求深度排序：${error instanceof Error ? error.message : String(error)}`);
      return;
    }
    this._clearSortTimeout();
    this.sortTimeoutId = setTimeout(() => {
      if (this.generation === generation && this.sortPending && this.requestId === requestId) {
        this._fatal("深度排序超过 30 秒未返回；渲染已按 fail-closed 阻断");
      }
    }, 30_000);
  }

  _clearSortTimeout() {
    if (this.sortTimeoutId !== null) {
      clearTimeout(this.sortTimeoutId);
      this.sortTimeoutId = null;
    }
  }

  _clearFirstFrameTimeout() {
    if (this.firstFrameTimeoutId !== null) {
      clearTimeout(this.firstFrameTimeoutId);
      this.firstFrameTimeoutId = null;
    }
  }

  _cameraChanged() {
    this.sortDirty = true;
    this._requestSort(false);
  }

  resetCamera() {
    this.target.set(this.homeTarget);
    if (this.cameraMode === "immersive") {
      this.camera.yaw = Math.PI;
      this.camera.pitch = 0.07;
      this.fieldOfView = 55 * Math.PI / 180;
      const narrowScreen = this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight) < 0.75;
      this.camera.distance = this.boundsRadius * (narrowScreen ? 0.42 : 0.5);
      this._cameraChanged();
      return;
    }
    this.camera.yaw = 0.72;
    this.camera.pitch = 0.32;
    this.fieldOfView = Math.PI / 4;
    const aspect = Math.max(0.1, this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight));
    const horizontalFieldOfView = 2 * Math.atan(Math.tan(this.fieldOfView / 2) * aspect);
    const fitFieldOfView = Math.min(this.fieldOfView, horizontalFieldOfView);
    this.camera.distance = this.boundsRadius / Math.sin(fitFieldOfView / 2) * 1.16;
    this._cameraChanged();
  }

  setScaleMultiplier(value) {
    if (!Number.isFinite(value) || value <= 0 || value > MAX_SPLAT_DISPLAY_MULTIPLIER) {
      throw new RangeError(
        `Gaussian scale multiplier must be in (0, ${MAX_SPLAT_DISPLAY_MULTIPLIER}]`,
      );
    }
    this.scaleMultiplier = value;
  }

  setAutoRotate(enabled) {
    this.autoRotate = Boolean(enabled);
  }

  _bindInteraction() {
    const canvas = this.canvas;
    canvas.addEventListener("pointerdown", (event) => {
      canvas.focus({ preventScroll: true });
      canvas.setPointerCapture(event.pointerId);
      this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      this.lastPinchDistance = this._pinchDistance();
    });
    canvas.addEventListener("pointermove", (event) => {
      const previous = this.pointers.get(event.pointerId);
      if (previous === undefined) {
        return;
      }
      this.pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      if (this.pointers.size === 1) {
        const deltaX = event.clientX - previous.x;
        const deltaY = event.clientY - previous.y;
        if (event.shiftKey || (event.buttons & 2) !== 0) {
          this._panByPixels(deltaX, deltaY);
        } else {
          this.camera.yaw -= deltaX * 0.006;
          this.camera.pitch = clamp(this.camera.pitch + deltaY * 0.005, -1.42, 1.42);
        }
      } else if (this.pointers.size === 2) {
        const distance = this._pinchDistance();
        if (distance !== null && this.lastPinchDistance !== null && distance > 0) {
          this.camera.distance *= this.lastPinchDistance / distance;
          this._clampDistance();
        }
        this.lastPinchDistance = distance;
      }
      this._cameraChanged();
    });
    const releasePointer = (event) => {
      this.pointers.delete(event.pointerId);
      this.lastPinchDistance = this._pinchDistance();
    };
    canvas.addEventListener("pointerup", releasePointer);
    canvas.addEventListener("pointercancel", releasePointer);
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.camera.distance *= Math.exp(event.deltaY * 0.0012);
      this._clampDistance();
      this._cameraChanged();
    }, { passive: false });
    canvas.addEventListener("dblclick", () => this.resetCamera());
    canvas.addEventListener("keydown", (event) => {
      const rotations = {
        ArrowLeft: [-0.08, 0],
        ArrowRight: [0.08, 0],
        ArrowUp: [0, -0.06],
        ArrowDown: [0, 0.06],
      };
      if (event.key.toLowerCase() === "r") {
        this.resetCamera();
        return;
      }
      if (["+", "="].includes(event.key)) {
        event.preventDefault();
        this.camera.distance *= 0.9;
        this._clampDistance();
        this._cameraChanged();
        return;
      }
      if (["-", "_"].includes(event.key)) {
        event.preventDefault();
        this.camera.distance *= 1.1;
        this._clampDistance();
        this._cameraChanged();
        return;
      }
      const rotation = rotations[event.key];
      if (rotation !== undefined) {
        event.preventDefault();
        this.camera.yaw += rotation[0];
        this.camera.pitch = clamp(this.camera.pitch + rotation[1], -1.42, 1.42);
        this._cameraChanged();
        return;
      }
      const movement = {
        w: [0, 1],
        s: [0, -1],
        a: [-1, 0],
        d: [1, 0],
      }[event.key.toLowerCase()];
      if (movement !== undefined) {
        event.preventDefault();
        this._moveTarget(movement[0], movement[1], this.boundsRadius * 0.04);
        this._cameraChanged();
      }
    });
  }

  _moveTarget(rightAmount, forwardAmount, distance) {
    const rightX = Math.cos(this.camera.yaw);
    const rightZ = -Math.sin(this.camera.yaw);
    const forwardX = -Math.sin(this.camera.yaw);
    const forwardZ = -Math.cos(this.camera.yaw);
    this.target[0] = clamp(
      this.target[0] + distance * (rightAmount * rightX + forwardAmount * forwardX),
      this.boundsMin[0],
      this.boundsMax[0],
    );
    this.target[2] = clamp(
      this.target[2] + distance * (rightAmount * rightZ + forwardAmount * forwardZ),
      this.boundsMin[2],
      this.boundsMax[2],
    );
  }

  _panByPixels(deltaX, deltaY) {
    const viewportHeight = Math.max(1, this.canvas.clientHeight);
    const worldPerPixel = 2 * this.camera.distance * Math.tan(this.fieldOfView / 2) / viewportHeight;
    this._moveTarget(-deltaX, deltaY, worldPerPixel);
  }

  _pinchDistance() {
    if (this.pointers.size !== 2) {
      return null;
    }
    const [first, second] = [...this.pointers.values()];
    return Math.hypot(first.x - second.x, first.y - second.y);
  }

  _clampDistance() {
    this.camera.distance = clamp(
      this.camera.distance,
      this.boundsRadius * 0.35,
      this.boundsRadius * 20,
    );
  }

  _resize() {
    const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.min(
      MAX_RENDER_DIMENSION,
      Math.max(1, Math.round(this.canvas.clientWidth * ratio)),
    );
    const height = Math.min(
      MAX_RENDER_DIMENSION,
      Math.max(1, Math.round(this.canvas.clientHeight * ratio)),
    );
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.gl.viewport(0, 0, width, height);
    }
  }

  _updateMatrices() {
    const cosinePitch = Math.cos(this.camera.pitch);
    const eye = [
      this.target[0] + this.camera.distance * cosinePitch * Math.sin(this.camera.yaw),
      this.target[1] + this.camera.distance * Math.sin(this.camera.pitch),
      this.target[2] + this.camera.distance * cosinePitch * Math.cos(this.camera.yaw),
    ];
    this.viewMatrix = lookAtMatrix(eye, this.target);
    this.near = Math.max(0.01, this.camera.distance - this.boundsRadius * 1.8);
    const far = Math.max(this.near + 1, this.camera.distance + this.boundsRadius * 3.5);
    this.projectionMatrix = perspectiveMatrix(
      this.fieldOfView,
      this.canvas.width / Math.max(1, this.canvas.height),
      this.near,
      far,
    );
  }

  _frame(time) {
    const delta = Math.max(0.001, time - this.lastFrameAt);
    this.lastFrameAt = time;
    const instantaneousFps = Math.min(120, 1_000 / delta);
    this.fps = this.fps === 0 ? instantaneousFps : this.fps * 0.9 + instantaneousFps * 0.1;

    if (this.loaded && this.autoRotate && this.pointers.size === 0) {
      this.camera.yaw += delta * 0.000025;
      this.sortDirty = true;
    }
    if (this.sortDirty) {
      this._requestSort(false);
    }

    this._resize();
    this._updateMatrices();
    const gl = this.gl;
    gl.clearColor(0.018, 0.027, 0.025, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (this.loaded && this.texture !== null && this.drawCount > 0) {
      gl.useProgram(this.program);
      gl.bindVertexArray(this.vao);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      gl.uniform1i(this.uniforms.splatData, 0);
      gl.uniform1i(this.uniforms.textureWidth, this.textureWidth);
      gl.uniformMatrix4fv(this.uniforms.view, false, this.viewMatrix);
      gl.uniformMatrix4fv(this.uniforms.projection, false, this.projectionMatrix);
      gl.uniform2f(this.uniforms.viewport, this.canvas.width, this.canvas.height);
      const focalY = this.canvas.height / (2 * Math.tan(this.fieldOfView / 2));
      const focalX = focalY;
      gl.uniform1f(this.uniforms.focalX, focalX);
      gl.uniform1f(this.uniforms.focalY, focalY);
      gl.uniform1f(this.uniforms.near, this.near);
      gl.uniform1f(this.uniforms.scaleMultiplier, this.scaleMultiplier);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.drawCount);
      if (this.firstFrame?.generation === this.generation) {
        const error = gl.getError();
        if (error !== gl.NO_ERROR) {
          this._fatal(`首个 WebGL2 Gaussian draw 失败 (error ${error})`);
        } else {
          const { resolve } = this.firstFrame;
          this.firstFrame = null;
          this._clearFirstFrameTimeout();
          resolve();
        }
      }
      gl.bindVertexArray(null);
    }

    if (time - this.lastStatsAt >= 250) {
      this.lastStatsAt = time;
      this.onStats({
        fps: this.fps,
        rendered: this.drawCount,
        total: this.totalCount,
        sortMs: this.sortMs,
      });
    }
    this.animationFrame = requestAnimationFrame((nextTime) => this._frame(nextTime));
  }
}
