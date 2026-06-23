import { useEffect, useMemo, useRef, useState } from "react";

const BACKGROUND_RGB = [16, 19, 22];
const FULLSCREEN_SHADER = `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@group(0) @binding(0) var tileSampler: sampler;
@group(0) @binding(1) var tileTexture: texture_2d<f32>;

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
  let uv = clamp(input.uv, vec2f(0.0), vec2f(1.0));
  let tile = textureSample(tileTexture, tileSampler, uv);
  let background = vec3f(0.0627, 0.0745, 0.0863);
  let rgb = background * (1.0 - tile.a) + tile.rgb * tile.a;
  return vec4f(rgb, 1.0);
}
`;

export default function WebGpuTileViewport({
  points,
  visibleIds,
  removedIds,
  isolatedId,
  tileSmoke,
  rendererContract,
  onSelectObject,
  renderModeLabel,
}) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const runtimeRef = useRef(null);
  const [frame, setFrame] = useState({
    status: "pending",
    reason: "webgpu-runtime-initializing",
    checksum: "",
    pixels: 0,
  });
  const visibleCount = useMemo(
    () =>
      points.filter(
        (point) =>
          visibleIds.has(point.objectId) &&
          !removedIds.has(point.objectId) &&
          (isolatedId === null || point.objectId === isolatedId),
      ).length,
    [isolatedId, points, removedIds, visibleIds],
  );

  useEffect(() => {
    let cancelled = false;
    const canvas = canvasRef.current;
    if (!canvas || typeof navigator === "undefined" || !navigator.gpu) {
      setFrame({
        status: "failed",
        reason: "navigator-gpu-unavailable",
        checksum: "",
        pixels: 0,
      });
      return undefined;
    }

    async function init() {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("webgpu-adapter-unavailable");
        const device = await adapter.requestDevice();
        if (cancelled) {
          device.destroy();
          return;
        }

        const context = canvas.getContext("webgpu");
        if (!context) throw new Error("webgpu-context-unavailable");
        const format = navigator.gpu.getPreferredCanvasFormat();
        const module = device.createShaderModule({ code: FULLSCREEN_SHADER });
        const pipeline = device.createRenderPipeline({
          layout: "auto",
          vertex: { module, entryPoint: "vertexMain" },
          fragment: {
            module,
            entryPoint: "fragmentMain",
            targets: [{ format }],
          },
          primitive: { topology: "triangle-list" },
        });
        const sampler = device.createSampler({
          magFilter: "nearest",
          minFilter: "nearest",
        });

        runtimeRef.current = { device, context, format, pipeline, sampler };
        device.lost.then((info) => {
          if (cancelled) return;
          setFrame({
            status: "failed",
            reason: `webgpu-device-lost-${info.reason || "unknown"}`,
            checksum: "",
            pixels: 0,
          });
        });
        renderFrame({
          runtime: runtimeRef.current,
          canvas,
          tileSmoke,
          setFrame,
        });
      } catch (error) {
        if (cancelled) return;
        setFrame({
          status: "failed",
          reason: error?.message || "webgpu-first-frame-failed",
          checksum: "",
          pixels: 0,
        });
      }
    }

    init();
    return () => {
      cancelled = true;
      runtimeRef.current?.device?.destroy();
      runtimeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const runtime = runtimeRef.current;
    const canvas = canvasRef.current;
    if (!runtime || !canvas) return;
    renderFrame({ runtime, canvas, tileSmoke, setFrame });
  }, [tileSmoke]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !onSelectObject) return undefined;

    const pointerUp = (event) => {
      const objectId = pickObjectFromTileFrame({
        event,
        canvas,
        points,
        visibleIds,
        removedIds,
        isolatedId,
      });
      if (objectId !== null) onSelectObject(objectId);
    };

    canvas.addEventListener("pointerup", pointerUp);
    return () => {
      canvas.removeEventListener("pointerup", pointerUp);
    };
  }, [isolatedId, onSelectObject, points, removedIds, visibleIds]);

  return (
    <div
      className="viewport"
      data-renderer={rendererContract?.rendererId ?? "webgpu-tile"}
      data-renderer-target={rendererContract?.targetRendererId ?? "webgpu-tile"}
      data-renderer-fallback-reason={rendererContract?.fallbackReason ?? ""}
      data-webgpu-target-gate={rendererContract?.targetGate ?? ""}
      data-webgpu-target-gate-reason={rendererContract?.targetGateReason ?? ""}
      data-webgpu-target-gate-blocker={rendererContract?.targetGateBlocker ?? ""}
      data-webgpu-status={rendererContract?.webGpuStatus ?? "available"}
      data-object-filter={rendererContract?.objectFilter ?? "gpu-object-state-buffer"}
      data-webgpu-object-filter-target={rendererContract?.targetObjectFilter ?? "gpu-object-state-buffer"}
      data-webgpu-pack-layout={rendererContract?.tileSmokeLayout ?? ""}
      data-webgpu-packed-gaussians={rendererContract?.packedGaussians ?? 0}
      data-webgpu-visible-gaussians={rendererContract?.visibleGaussians ?? 0}
      data-webgpu-binned-gaussians={rendererContract?.binnedGaussians ?? 0}
      data-webgpu-tile-size={rendererContract?.tileSize ?? 0}
      data-webgpu-tile-count={rendererContract?.tileCount ?? 0}
      data-webgpu-active-tile-count={rendererContract?.activeTileCount ?? 0}
      data-webgpu-tile-reference-count={rendererContract?.tileReferenceCount ?? 0}
      data-tile-overflow-count={rendererContract?.tileOverflowCount ?? 0}
      data-webgpu-tile-overflow-tile-count={rendererContract?.tileOverflowTileCount ?? 0}
      data-webgpu-tile-overflow-ratio={rendererContract?.tileOverflowRatio ?? 0}
      data-webgpu-tile-overflow-max-excess={rendererContract?.tileOverflowMaxExcess ?? 0}
      data-webgpu-tile-entry-stored-count={rendererContract?.tileEntryStoredCount ?? 0}
      data-webgpu-tile-entry-capacity={rendererContract?.tileEntryCapacity ?? 0}
      data-webgpu-tile-entry-utilization={rendererContract?.tileEntryUtilization ?? 0}
      data-webgpu-tile-capacity-mode={rendererContract?.tileCapacityMode ?? ""}
      data-webgpu-tile-capacity-status={rendererContract?.tileCapacityStatus ?? ""}
      data-webgpu-tile-capacity-gate={rendererContract?.tileCapacityGate ?? ""}
      data-webgpu-max-tile-occupancy={rendererContract?.maxTileOccupancy ?? 0}
      data-webgpu-resolve-layout={rendererContract?.resolveVersion ?? ""}
      data-webgpu-resolve-mode={rendererContract?.resolveMode ?? ""}
      data-webgpu-resolved-tile-count={rendererContract?.resolvedTileCount ?? 0}
      data-webgpu-resolve-weight-sum={rendererContract?.resolveWeightSum ?? 0}
      data-webgpu-resolve-alpha-mean={rendererContract?.resolveAlphaMean ?? 0}
      data-webgpu-resolve-luma-mean={rendererContract?.resolveLumaMean ?? 0}
      data-webgpu-resolve-checksum={rendererContract?.resolveChecksum ?? ""}
      data-webgpu-object-state-layout={rendererContract?.objectStateLayoutVersion ?? ""}
      data-webgpu-object-state-stride={rendererContract?.objectStateStrideUint32 ?? 0}
      data-webgpu-object-state-visible-objects={rendererContract?.objectStateVisibleObjects ?? 0}
      data-webgpu-object-state-hidden-objects={rendererContract?.objectStateHiddenObjects ?? 0}
      data-webgpu-object-state-removed-objects={rendererContract?.objectStateRemovedObjects ?? 0}
      data-webgpu-object-state-selected-objects={rendererContract?.objectStateSelectedObjects ?? 0}
      data-webgpu-object-state-isolated-objects={rendererContract?.objectStateIsolatedObjects ?? 0}
      data-webgpu-object-state-checksum={rendererContract?.objectStateChecksum ?? ""}
      data-webgpu-first-frame-status={frame.status}
      data-webgpu-first-frame-reason={frame.reason}
      data-webgpu-first-frame-checksum={frame.checksum}
      data-webgpu-first-frame-pixels={frame.pixels}
      data-visible-count={visibleCount}
      ref={containerRef}
    >
      <canvas ref={canvasRef} />
      <div className="viewportHud">
        <div>
          <span className="hudLabel">可见</span>
          <strong>{visibleCount.toLocaleString()}</strong>
        </div>
        <div>
          <span className="hudLabel">模式</span>
          <strong>{renderModeLabel}</strong>
        </div>
        <div>
          <span className="hudLabel">WebGPU</span>
          <strong>{frame.status}</strong>
        </div>
      </div>
    </div>
  );
}

function renderFrame({ runtime, canvas, tileSmoke, setFrame }) {
  const { device, context, format, pipeline, sampler } = runtime;
  resizeCanvasToDisplaySize(canvas);
  context.configure({ device, format, alphaMode: "opaque" });

  const texture = createTileTexture(device, tileSmoke);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: texture.createView() },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        clearValue: {
          r: BACKGROUND_RGB[0] / 255,
          g: BACKGROUND_RGB[1] / 255,
          b: BACKGROUND_RGB[2] / 255,
          a: 1,
        },
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(3);
  pass.end();
  device.queue.submit([encoder.finish()]);
  texture.destroy();
  setFrame({
    status: "rendered",
    reason: "webgpu-tile-first-frame-rendered",
    checksum: tileSmoke.resolveChecksum,
    pixels: tileSmoke.resolvedTileCount,
  });
}

function createTileTexture(device, tileSmoke) {
  const width = Math.max(1, tileSmoke.tileColumns);
  const height = Math.max(1, tileSmoke.tileRows);
  const data = new Uint8Array(width * height * 4);
  const source = tileSmoke.buffers.tileResolvedRgba;
  for (let index = 0; index < data.length; index += 4) {
    data[index] = toByte(source[index]);
    data[index + 1] = toByte(source[index + 1]);
    data[index + 2] = toByte(source[index + 2]);
    data[index + 3] = toByte(source[index + 3]);
  }
  const texture = device.createTexture({
    size: [width, height, 1],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.writeTexture(
    { texture },
    data,
    { bytesPerRow: width * 4, rowsPerImage: height },
    { width, height, depthOrArrayLayers: 1 },
  );
  return texture;
}

function pickObjectFromTileFrame({
  event,
  canvas,
  points,
  visibleIds,
  removedIds,
  isolatedId,
}) {
  const bounds = sceneBounds(points);
  const rect = canvas.getBoundingClientRect();
  const targetX = event.clientX - rect.left;
  const targetY = event.clientY - rect.top;
  const threshold = Math.max(32, Math.min(rect.width, rect.height) * 0.06);
  let bestDistance = threshold * threshold;
  let bestObjectId = null;

  for (const point of points) {
    if (
      !visibleIds.has(point.objectId) ||
      removedIds.has(point.objectId) ||
      (isolatedId !== null && point.objectId !== isolatedId)
    ) {
      continue;
    }
    const x = ((point.x - bounds.minX) / bounds.spanX) * rect.width;
    const y = (1 - (point.z - bounds.minZ) / bounds.spanZ) * rect.height;
    const distance = (x - targetX) ** 2 + (y - targetY) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestObjectId = point.objectId;
    }
  }
  return bestObjectId;
}

function resizeCanvasToDisplaySize(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function sceneBounds(points) {
  if (points.length === 0) {
    return { minX: -1, minZ: -1, spanX: 2, spanZ: 2 };
  }

  let minX = Infinity;
  let maxX = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
  for (const point of points) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minZ = Math.min(minZ, point.z);
    maxZ = Math.max(maxZ, point.z);
  }

  return {
    minX,
    minZ,
    spanX: Math.max(maxX - minX, 0.0001),
    spanZ: Math.max(maxZ - minZ, 0.0001),
  };
}

function toByte(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  return Math.round(Math.min(1, Math.max(0, numeric)) * 255);
}
