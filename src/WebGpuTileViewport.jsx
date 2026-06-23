import { useEffect, useMemo, useRef, useState } from "react";

import {
  createWebGpuAccumulationMeta,
  createWebGpuComputeMeta,
  webGpuAccumulationWorkgroups,
  webGpuComputeWorkgroups,
  WEBGPU_TILE_ACCUMULATION_SHADER,
  WEBGPU_TILE_ACCUMULATION_SOURCE,
  WEBGPU_TILE_COMPUTE_SHADER,
  WEBGPU_TILE_COMPUTE_SOURCE,
} from "./webgpuTileComputeShader.js";
import { createWebGpuTileStorageBuffers } from "./webgpuTileStorage.js";
import {
  createWebGpuResolveMeta,
  WEBGPU_TILE_RESOLVE_SHADER,
  WEBGPU_TILE_RESOLVE_SOURCE,
} from "./webgpuTileResolveShader.js";

const BACKGROUND_RGB = [16, 19, 22];

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
    source: "",
  });
  const [compute, setCompute] = useState({
    status: "pending",
    reason: "webgpu-compute-pending",
    source: "",
    workgroups: 0,
  });
  const [accumulation, setAccumulation] = useState({
    status: "pending",
    reason: "webgpu-accumulation-pending",
    source: "",
    workgroups: 0,
  });
  const [storage, setStorage] = useState({
    status: "pending",
    reason: "webgpu-storage-pending",
    layoutVersion: "",
    checksum: "",
    bufferCount: 0,
    byteLength: 0,
    tileEntriesIncluded: false,
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
        source: "",
      });
      setStorage({
        status: "failed",
        reason: "navigator-gpu-unavailable",
        layoutVersion: "",
        checksum: "",
        bufferCount: 0,
        byteLength: 0,
        tileEntriesIncluded: false,
      });
      setCompute({
        status: "failed",
        reason: "navigator-gpu-unavailable",
        source: "",
        workgroups: 0,
      });
      setAccumulation({
        status: "failed",
        reason: "navigator-gpu-unavailable",
        source: "",
        workgroups: 0,
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
        const resolveModule = device.createShaderModule({ code: WEBGPU_TILE_RESOLVE_SHADER });
        const resolveComputeModule = device.createShaderModule({ code: WEBGPU_TILE_COMPUTE_SHADER });
        const accumulationModule = device.createShaderModule({ code: WEBGPU_TILE_ACCUMULATION_SHADER });
        const accumulationPipeline = device.createComputePipeline({
          layout: "auto",
          compute: { module: accumulationModule, entryPoint: "accumulationMain" },
        });
        const computePipeline = device.createComputePipeline({
          layout: "auto",
          compute: { module: resolveComputeModule, entryPoint: "computeMain" },
        });
        const pipeline = device.createRenderPipeline({
          layout: "auto",
          vertex: { module: resolveModule, entryPoint: "vertexMain" },
          fragment: {
            module: resolveModule,
            entryPoint: "fragmentMain",
            targets: [{ format }],
          },
          primitive: { topology: "triangle-list" },
        });
        runtimeRef.current = {
          device,
          context,
          format,
          pipeline,
          accumulationPipeline,
          computePipeline,
        };
        device.lost.then((info) => {
          if (cancelled) return;
          setFrame({
            status: "failed",
            reason: `webgpu-device-lost-${info.reason || "unknown"}`,
            checksum: "",
            pixels: 0,
            source: "",
          });
        });
        renderFrame({
          runtime: runtimeRef.current,
          canvas,
          tileSmoke,
          setFrame,
          setStorage,
          setCompute,
          setAccumulation,
        });
      } catch (error) {
        if (cancelled) return;
        setFrame({
          status: "failed",
          reason: error?.message || "webgpu-first-frame-failed",
          checksum: "",
          pixels: 0,
          source: "",
        });
        setStorage({
          status: "failed",
          reason: error?.message || "webgpu-storage-unavailable",
          layoutVersion: "",
          checksum: "",
          bufferCount: 0,
          byteLength: 0,
          tileEntriesIncluded: false,
        });
        setCompute({
          status: "failed",
          reason: error?.message || "webgpu-compute-unavailable",
          source: "",
          workgroups: 0,
        });
        setAccumulation({
          status: "failed",
          reason: error?.message || "webgpu-accumulation-unavailable",
          source: "",
          workgroups: 0,
        });
      }
    }

    init();
    return () => {
      cancelled = true;
      destroyTransientBuffers(runtimeRef.current);
      runtimeRef.current?.storageBundle?.destroy?.();
      runtimeRef.current?.device?.destroy();
      runtimeRef.current = null;
    };
  }, []);

  useEffect(() => {
    const runtime = runtimeRef.current;
    const canvas = canvasRef.current;
    if (!runtime || !canvas) return;
    renderFrame({ runtime, canvas, tileSmoke, setFrame, setStorage, setCompute, setAccumulation });
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
      data-webgpu-resolve-source={frame.source}
      data-webgpu-accumulation-source={accumulation.source}
      data-webgpu-accumulation-status={accumulation.status}
      data-webgpu-accumulation-reason={accumulation.reason}
      data-webgpu-accumulation-workgroups={accumulation.workgroups}
      data-webgpu-compute-source={compute.source}
      data-webgpu-compute-status={compute.status}
      data-webgpu-compute-reason={compute.reason}
      data-webgpu-compute-workgroups={compute.workgroups}
      data-webgpu-storage-layout={storage.layoutVersion}
      data-webgpu-storage-status={storage.status}
      data-webgpu-storage-reason={storage.reason}
      data-webgpu-storage-buffer-count={storage.bufferCount}
      data-webgpu-storage-byte-size={storage.byteLength}
      data-webgpu-storage-checksum={storage.checksum}
      data-webgpu-storage-tile-entries={storage.tileEntriesIncluded ? "true" : "false"}
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

function renderFrame({
  runtime,
  canvas,
  tileSmoke,
  setFrame,
  setStorage,
  setCompute,
  setAccumulation,
}) {
  const { device, context, format, pipeline, accumulationPipeline, computePipeline } = runtime;
  resizeCanvasToDisplaySize(canvas);
  context.configure({ device, format, alphaMode: "opaque" });
  destroyTransientBuffers(runtime);

  let storageBundle = null;
  try {
    storageBundle = createWebGpuTileStorageBuffers(device, tileSmoke);
    runtime.storageBundle?.destroy?.();
    runtime.storageBundle = storageBundle;
    setStorage({
      status: "uploaded",
      reason: "webgpu-storage-uploaded",
      layoutVersion: storageBundle.layoutVersion,
      checksum: storageBundle.checksum,
      bufferCount: storageBundle.bufferCount,
      byteLength: storageBundle.totalByteLength,
      tileEntriesIncluded: storageBundle.tileEntriesIncluded,
    });
  } catch (error) {
    setStorage({
      status: "failed",
      reason: error?.message || "webgpu-storage-upload-failed",
      layoutVersion: "",
      checksum: "",
      bufferCount: 0,
      byteLength: 0,
      tileEntriesIncluded: false,
    });
    setFrame({
      status: "failed",
      reason: error?.message || "webgpu-storage-upload-failed",
      checksum: "",
      pixels: 0,
      source: "",
    });
    setCompute({
      status: "failed",
      reason: error?.message || "webgpu-storage-upload-failed",
      source: "",
      workgroups: 0,
    });
    setAccumulation({
      status: "failed",
      reason: error?.message || "webgpu-storage-upload-failed",
      source: "",
      workgroups: 0,
    });
    return;
  }

  try {
    const accumulationMetaBuffer = createAccumulationMetaBuffer(device, tileSmoke);
    const computeMetaBuffer = createComputeMetaBuffer(device, tileSmoke);
    const resolveMetaBuffer = createResolveMetaBuffer(device, tileSmoke);
    runtime.transientBuffers = [accumulationMetaBuffer, computeMetaBuffer, resolveMetaBuffer];
    const accumulationBindGroup = device.createBindGroup({
      layout: accumulationPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: storageBundle.getBuffer("positionRadius").buffer } },
        { binding: 1, resource: { buffer: storageBundle.getBuffer("colorOpacity").buffer } },
        { binding: 2, resource: { buffer: storageBundle.getBuffer("objectIndices").buffer } },
        { binding: 3, resource: { buffer: storageBundle.getBuffer("objectState").buffer } },
        { binding: 4, resource: { buffer: storageBundle.getBuffer("tileCounts").buffer } },
        { binding: 5, resource: { buffer: storageBundle.getBuffer("tileEntries").buffer } },
        { binding: 6, resource: { buffer: storageBundle.getBuffer("tileAccumulation").buffer } },
        { binding: 7, resource: { buffer: accumulationMetaBuffer } },
      ],
    });
    const computeBindGroup = device.createBindGroup({
      layout: computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: storageBundle.getBuffer("tileAccumulation").buffer } },
        { binding: 1, resource: { buffer: storageBundle.getBuffer("tileResolvedRgba").buffer } },
        { binding: 2, resource: { buffer: computeMetaBuffer } },
      ],
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: storageBundle.getBuffer("tileResolvedRgba").buffer } },
        { binding: 1, resource: { buffer: resolveMetaBuffer } },
      ],
    });
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    const accumulationWorkgroups = webGpuAccumulationWorkgroups(tileSmoke);
    const workgroups = webGpuComputeWorkgroups(tileSmoke);
    computePass.setPipeline(accumulationPipeline);
    computePass.setBindGroup(0, accumulationBindGroup);
    computePass.dispatchWorkgroups(accumulationWorkgroups);
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    computePass.dispatchWorkgroups(workgroups);
    computePass.end();
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
    setAccumulation({
      status: "dispatched",
      reason: "webgpu-tile-accumulation-dispatched",
      source: WEBGPU_TILE_ACCUMULATION_SOURCE,
      workgroups: accumulationWorkgroups,
    });
    setCompute({
      status: "dispatched",
      reason: "webgpu-compute-resolve-dispatched",
      source: WEBGPU_TILE_COMPUTE_SOURCE,
      workgroups,
    });
    setFrame({
      status: "rendered",
      reason: "webgpu-storage-resolve-rendered",
      checksum: tileSmoke.resolveChecksum,
      pixels: tileSmoke.resolvedTileCount,
      source: WEBGPU_TILE_RESOLVE_SOURCE,
    });
  } catch (error) {
    setAccumulation({
      status: "failed",
      reason: error?.message || "webgpu-tile-accumulation-failed",
      source: "",
      workgroups: 0,
    });
    setCompute({
      status: "failed",
      reason: error?.message || "webgpu-compute-resolve-failed",
      source: "",
      workgroups: 0,
    });
    setFrame({
      status: "failed",
      reason: error?.message || "webgpu-compute-resolve-failed",
      checksum: "",
      pixels: 0,
      source: "",
    });
  }
}

function createAccumulationMetaBuffer(device, tileSmoke) {
  const data = createWebGpuAccumulationMeta(tileSmoke);
  const usage = globalThis.GPUBufferUsage ?? {
    COPY_DST: 0x0008,
    UNIFORM: 0x0040,
  };
  const buffer = device.createBuffer({
    label: "objgauss-accumulation-meta",
    size: data.byteLength,
    usage: usage.UNIFORM | usage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function createComputeMetaBuffer(device, tileSmoke) {
  const data = createWebGpuComputeMeta(tileSmoke);
  const usage = globalThis.GPUBufferUsage ?? {
    COPY_DST: 0x0008,
    UNIFORM: 0x0040,
  };
  const buffer = device.createBuffer({
    label: "objgauss-compute-meta",
    size: data.byteLength,
    usage: usage.UNIFORM | usage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function createResolveMetaBuffer(device, tileSmoke) {
  const data = createWebGpuResolveMeta(tileSmoke);
  const usage = globalThis.GPUBufferUsage ?? {
    COPY_DST: 0x0008,
    UNIFORM: 0x0040,
  };
  const buffer = device.createBuffer({
    label: "objgauss-resolve-meta",
    size: data.byteLength,
    usage: usage.UNIFORM | usage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function destroyTransientBuffers(runtime) {
  if (!runtime?.transientBuffers) return;
  for (const buffer of runtime.transientBuffers) {
    buffer?.destroy?.();
  }
  runtime.transientBuffers = [];
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
