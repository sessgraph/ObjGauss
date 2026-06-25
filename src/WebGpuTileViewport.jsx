import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import {
  createWebGpuAccumulationMeta,
  createWebGpuComputeMeta,
  createWebGpuPixelResolveMeta,
  createWebGpuPixelResolveShader,
  webGpuAccumulationWorkgroups,
  webGpuComputeWorkgroups,
  webGpuPixelResolveWorkgroups,
  WEBGPU_PIXEL_RESOLVE_SOURCE,
  WEBGPU_TILE_ACCUMULATION_SHADER,
  WEBGPU_TILE_ACCUMULATION_SOURCE,
  WEBGPU_TILE_COMPUTE_SHADER,
  WEBGPU_TILE_COMPUTE_SOURCE,
} from "./webgpuTileComputeShader.js";
import { WEBGPU_TILE_REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE } from "./webgpuCapability.js";
import {
  canReuseWebGpuTileStorageBuffers,
  createWebGpuTileStorageBuffers,
  updateWebGpuTileObjectStateBuffer,
} from "./webgpuTileStorage.js";
import {
  createWebGpuResolveMeta,
  createWebGpuTileResolveShader,
  normalizeWebGpuAlphaPresentationTuning,
  WEBGPU_TILE_ALPHA_PRESENTATION_MODE,
  WEBGPU_TILE_ALPHA_PRESENTATION_TUNING_MODE,
  WEBGPU_TILE_RESOLVE_FILTER,
  WEBGPU_TILE_RESOLVE_SOURCE,
} from "./webgpuTileResolveShader.js";
import {
  WEBGPU_FLOAT_TEXTURE_COPY_RESOLVE_SOURCE,
  WEBGPU_FLOAT_TEXTURE_LOAD_RESOLVE_SHADER,
  WEBGPU_SAMPLED_TEXTURE_RESOLVE_SHADER,
  WEBGPU_SAMPLED_TEXTURE_RESOLVE_SOURCE,
} from "./webgpuTextureResolveShader.js";
import {
  buildWebGpuTileProjectionBounds,
  projectPointToWebGpuTileViewport,
  WEBGPU_TILE_LIST_MODE_OBJECT_STATE,
} from "./webgpuTileSmoke.js";
import {
  normalizeWebGpuRuntimeProbe,
  WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY,
  WEBGPU_RUNTIME_PROBE_CLEAR_ONLY,
  WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY,
  WEBGPU_RUNTIME_PROBE_FULL,
  WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK,
  WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY,
  WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY,
  WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY,
  WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY,
  WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY,
  WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT,
} from "./webgpuRuntimeProbe.js";

const BACKGROUND_RGB = [16, 19, 22];
const WEBGPU_DEVICE_INIT_DELAY_MS = 500;

export default function WebGpuTileViewport({
  points,
  visibleIds,
  removedIds,
  isolatedId,
  tileSmoke,
  rendererContract,
  onSelectObject,
  renderModeLabel,
  runtimeViewportAspectMode = "default-square",
  runtimeViewportQuality = "unknown",
  runtimeViewportPixelBudget = 0,
  onDisplaySizeChange,
}) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const runtimeRef = useRef(null);
  const runtimeProbe = useMemo(readWebGpuRuntimeProbe, []);
  const alphaPresentationTuning = useMemo(readWebGpuAlphaPresentationTuning, []);
  const [frame, setFrame] = useState({
    status: "pending",
    reason: "webgpu-runtime-initializing",
    checksum: "",
    pixels: 0,
    source: "",
    filter: "",
  });
  const [readback, setReadback] = useState({
    status: "skipped",
    reason: "webgpu-readback-not-requested",
    source: "",
    checksum: "",
    byteLength: 0,
    floatCount: 0,
    finiteFloats: 0,
    nonzeroFloats: 0,
  });
  const [deviceLost, setDeviceLost] = useState({
    status: "active",
    reason: "webgpu-device-active",
    message: "",
  });
  const [deviceError, setDeviceError] = useState({
    status: "none",
    type: "",
    message: "",
  });
  const [queue, setQueue] = useState({
    status: "pending",
    reason: "webgpu-queue-pending",
    message: "",
  });
  const [compute, setCompute] = useState({
    status: "pending",
    reason: "webgpu-compute-pending",
    source: "",
    workgroups: 0,
  });
  const [pixel, setPixel] = useState({
    status: "pending",
    reason: "webgpu-pixel-accumulation-pending",
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
    tileOffsetsIncluded: false,
    pixelOutputIncluded: false,
  });
  const [displaySize, setDisplaySize] = useState({ width: 0, height: 0 });
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

  useLayoutEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;
    let animationFrame = 0;
    const reportSize = () => {
      const next = {
        width: Math.max(1, Math.floor(container.clientWidth)),
        height: Math.max(1, Math.floor(container.clientHeight)),
      };
      setDisplaySize((current) =>
        current.width === next.width && current.height === next.height ? current : next,
      );
      onDisplaySizeChange?.(next);
    };
    reportSize();
    if (typeof ResizeObserver !== "undefined") {
      const observer = new ResizeObserver(() => {
        cancelAnimationFrame(animationFrame);
        animationFrame = requestAnimationFrame(reportSize);
      });
      observer.observe(container);
      return () => {
        cancelAnimationFrame(animationFrame);
        observer.disconnect();
      };
    }
    window.addEventListener("resize", reportSize);
    return () => {
      cancelAnimationFrame(animationFrame);
      window.removeEventListener("resize", reportSize);
    };
  }, [onDisplaySizeChange]);

  useEffect(() => {
    let cancelled = false;
    let ownedRuntime = null;
    const canvas = canvasRef.current;
    setDeviceLost({
      status: "active",
      reason: "webgpu-device-active",
      message: "",
    });
    setDeviceError({
      status: "none",
      type: "",
      message: "",
    });
    setQueue({
      status: "pending",
      reason: "webgpu-queue-pending",
      message: "",
    });
    if (!canvas || typeof navigator === "undefined" || !navigator.gpu) {
      setFrame({
        status: "failed",
        reason: "navigator-gpu-unavailable",
        checksum: "",
        pixels: 0,
        source: "",
        filter: "",
      });
      setStorage({
        status: "failed",
        reason: "navigator-gpu-unavailable",
        layoutVersion: "",
        checksum: "",
        bufferCount: 0,
        byteLength: 0,
        tileEntriesIncluded: false,
        tileOffsetsIncluded: false,
        pixelOutputIncluded: false,
      });
      setCompute({
        status: "failed",
        reason: "navigator-gpu-unavailable",
        source: "",
        workgroups: 0,
      });
      setPixel({
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
      setReadback({
        status: "failed",
        reason: "navigator-gpu-unavailable",
        source: "",
        checksum: "",
        byteLength: 0,
        floatCount: 0,
        finiteFloats: 0,
        nonzeroFloats: 0,
      });
      setDeviceLost({
        status: "unavailable",
        reason: "navigator-gpu-unavailable",
        message: "",
      });
      setDeviceError({
        status: "unavailable",
        type: "",
        message: "navigator-gpu-unavailable",
      });
      setQueue({
        status: "unavailable",
        reason: "navigator-gpu-unavailable",
        message: "",
      });
      return undefined;
    }

    async function init() {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("webgpu-adapter-unavailable");
        const device = await requestWebGpuTileDevice(adapter);
        if (cancelled) {
          return;
        }
        device.addEventListener("uncapturederror", (event) => {
          if (cancelled) return;
          const error = event.error;
          setDeviceError({
            status: "error",
            type: error?.constructor?.name || error?.name || "GPUError",
            message: error?.message || String(error || ""),
          });
        });

        const needsPresentation = runtimeProbeNeedsPresentation(runtimeProbe);
        const context = needsPresentation ? canvas.getContext("webgpu") : null;
        if (needsPresentation && !context) throw new Error("webgpu-context-unavailable");
        const format = needsPresentation ? navigator.gpu.getPreferredCanvasFormat() : "";
        const resolveModule = needsPresentation
          ? device.createShaderModule({
              code: createWebGpuTileResolveShader(alphaPresentationTuning),
            })
          : null;
        const sampledTextureModule = needsPresentation
          ? device.createShaderModule({ code: WEBGPU_SAMPLED_TEXTURE_RESOLVE_SHADER })
          : null;
        const floatTextureModule = needsPresentation
          ? device.createShaderModule({ code: WEBGPU_FLOAT_TEXTURE_LOAD_RESOLVE_SHADER })
          : null;
        const resolveComputeModule = device.createShaderModule({ code: WEBGPU_TILE_COMPUTE_SHADER });
        const pixelResolveModule = device.createShaderModule({
          code: createWebGpuPixelResolveShader({
            pixelDepthBinCount: tileSmoke?.pixelDepthBinCount,
            pixelDepthAlphaMode: tileSmoke?.pixelDepthAlphaMode,
          }),
        });
        const accumulationModule = device.createShaderModule({ code: WEBGPU_TILE_ACCUMULATION_SHADER });
        const accumulationPipeline = device.createComputePipeline({
          layout: "auto",
          compute: { module: accumulationModule, entryPoint: "accumulationMain" },
        });
        const computePipeline = device.createComputePipeline({
          layout: "auto",
          compute: { module: resolveComputeModule, entryPoint: "computeMain" },
        });
        const pixelComputePipeline = device.createComputePipeline({
          layout: "auto",
          compute: { module: pixelResolveModule, entryPoint: "pixelResolveMain" },
        });
        const pipeline = needsPresentation
          ? device.createRenderPipeline({
              layout: "auto",
              vertex: { module: resolveModule, entryPoint: "vertexMain" },
              fragment: {
                module: resolveModule,
                entryPoint: "fragmentMain",
                targets: [{ format }],
              },
              primitive: { topology: "triangle-list" },
            })
          : null;
        const sampledTexturePipeline = needsPresentation
          ? device.createRenderPipeline({
              layout: "auto",
              vertex: { module: sampledTextureModule, entryPoint: "vertexMain" },
              fragment: {
                module: sampledTextureModule,
                entryPoint: "fragmentMain",
                targets: [{ format }],
              },
              primitive: { topology: "triangle-list" },
            })
          : null;
        const floatTexturePipeline = needsPresentation
          ? device.createRenderPipeline({
              layout: "auto",
              vertex: { module: floatTextureModule, entryPoint: "vertexMain" },
              fragment: {
                module: floatTextureModule,
                entryPoint: "fragmentMain",
                targets: [{ format }],
              },
              primitive: { topology: "triangle-list" },
            })
          : null;
        const runtime = {
          device,
          context,
          format,
          pipeline,
          sampledTexturePipeline,
          floatTexturePipeline,
          accumulationPipeline,
          computePipeline,
          pixelComputePipeline,
        };
        ownedRuntime = runtime;
        runtimeRef.current = runtime;
        device.lost.then((info) => {
          if (cancelled) return;
          setDeviceLost({
            status: "lost",
            reason: `webgpu-device-lost-${info.reason || "unknown"}`,
            message: info.message || "",
          });
        });
        renderFrame({
          runtime,
          canvas,
          tileSmoke,
          runtimeProbe,
          setFrame,
          setStorage,
          setCompute,
          setPixel,
          setAccumulation,
          setReadback,
          setQueue,
        });
      } catch (error) {
        if (cancelled) return;
        setFrame({
          status: "failed",
          reason: error?.message || "webgpu-first-frame-failed",
          checksum: "",
          pixels: 0,
          source: "",
          filter: "",
        });
        setStorage({
          status: "failed",
          reason: error?.message || "webgpu-storage-unavailable",
          layoutVersion: "",
          checksum: "",
          bufferCount: 0,
          byteLength: 0,
          tileEntriesIncluded: false,
          tileOffsetsIncluded: false,
          pixelOutputIncluded: false,
        });
        setCompute({
          status: "failed",
          reason: error?.message || "webgpu-compute-unavailable",
          source: "",
          workgroups: 0,
        });
        setPixel({
          status: "failed",
          reason: error?.message || "webgpu-pixel-accumulation-unavailable",
          source: "",
          workgroups: 0,
        });
        setAccumulation({
          status: "failed",
          reason: error?.message || "webgpu-accumulation-unavailable",
          source: "",
          workgroups: 0,
        });
        setReadback({
          status: "failed",
          reason: error?.message || "webgpu-readback-unavailable",
          source: "",
          checksum: "",
          byteLength: 0,
          floatCount: 0,
          finiteFloats: 0,
          nonzeroFloats: 0,
        });
        setDeviceLost({
          status: "unavailable",
          reason: error?.message || "webgpu-device-unavailable",
          message: "",
        });
        setDeviceError({
          status: "unavailable",
          type: error?.constructor?.name || error?.name || "Error",
          message: error?.message || "webgpu-device-unavailable",
        });
        setQueue({
          status: "failed",
          reason: error?.message || "webgpu-queue-unavailable",
          message: "",
        });
      }
    }

    const initTimer =
      typeof globalThis.setTimeout === "function"
        ? globalThis.setTimeout(init, WEBGPU_DEVICE_INIT_DELAY_MS)
        : null;
    if (initTimer === null) init();
    return () => {
      cancelled = true;
      if (initTimer !== null && typeof globalThis.clearTimeout === "function") {
        globalThis.clearTimeout(initTimer);
      }
      destroyTransientBuffers(ownedRuntime);
      ownedRuntime?.storageBundle?.destroy?.();
      if (runtimeRef.current === ownedRuntime) {
        runtimeRef.current = null;
      }
    };
  }, [alphaPresentationTuning, runtimeProbe]);

  useEffect(() => {
    const runtime = runtimeRef.current;
    const canvas = canvasRef.current;
    if (!runtime || !canvas) return;
    renderFrame({
      runtime,
      canvas,
      tileSmoke,
      runtimeProbe,
      setFrame,
      setStorage,
      setCompute,
      setPixel,
      setAccumulation,
      setReadback,
      setQueue,
    });
  }, [runtimeProbe, tileSmoke]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !onSelectObject) return undefined;

    const pointerUp = (event) => {
      const objectId = pickObjectFromTileFrame({
        event,
        canvas,
        points,
        tileSmoke,
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
  }, [isolatedId, onSelectObject, points, removedIds, tileSmoke, visibleIds]);

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
      data-webgpu-storage-limit-gate={rendererContract?.storageLimitGate ?? ""}
      data-webgpu-storage-limit-reason={rendererContract?.storageLimitReason ?? ""}
      data-webgpu-storage-limit-blocker={rendererContract?.storageLimitBlocker ?? ""}
      data-webgpu-storage-limit-max-buffer-size={rendererContract?.storageLimitMaxBufferSize ?? ""}
      data-webgpu-storage-limit-max-binding-size={rendererContract?.storageLimitMaxStorageBufferBindingSize ?? ""}
      data-webgpu-storage-limit-max-storage-buffers-per-stage={rendererContract?.storageLimitMaxStorageBuffersPerShaderStage ?? ""}
      data-webgpu-storage-limit-required-storage-buffers-per-stage={rendererContract?.storageLimitRequiredStorageBuffersPerShaderStage ?? ""}
      data-webgpu-storage-limit-effective-max-buffer-size={rendererContract?.storageLimitEffectiveMaxBufferByteLength ?? ""}
      data-webgpu-storage-estimated-layout={rendererContract?.storageEstimatedLayout ?? ""}
      data-webgpu-storage-estimated-buffer-count={rendererContract?.storageEstimatedBufferCount ?? 0}
      data-webgpu-storage-estimated-byte-size={rendererContract?.storageEstimatedByteSize ?? 0}
      data-webgpu-storage-estimated-max-buffer-byte-size={rendererContract?.storageEstimatedMaxBufferByteSize ?? 0}
      data-webgpu-storage-estimated-max-buffer-key={rendererContract?.storageEstimatedMaxBufferKey ?? ""}
      data-object-filter={rendererContract?.objectFilter ?? "gpu-object-state-buffer"}
      data-webgpu-object-filter-target={rendererContract?.targetObjectFilter ?? "gpu-object-state-buffer"}
      data-webgpu-pack-layout={rendererContract?.tileSmokeLayout ?? ""}
      data-webgpu-viewport-width={rendererContract?.viewportWidth ?? tileSmoke?.viewportWidth ?? 0}
      data-webgpu-viewport-height={rendererContract?.viewportHeight ?? tileSmoke?.viewportHeight ?? 0}
      data-webgpu-pixel-count={tileSmoke?.pixelCount ?? 0}
      data-webgpu-viewport-aspect-mode={runtimeViewportAspectMode}
      data-webgpu-viewport-quality={runtimeViewportQuality}
      data-webgpu-viewport-pixel-budget={runtimeViewportPixelBudget}
      data-webgpu-display-width={displaySize.width}
      data-webgpu-display-height={displaySize.height}
      data-webgpu-bounds-fit-mode={rendererContract?.boundsFitMode ?? ""}
      data-webgpu-bounds-padding-ratio={rendererContract?.boundsPaddingRatio ?? 0}
      data-webgpu-bounds-viewport-aspect={rendererContract?.boundsViewportAspect ?? 0}
      data-webgpu-bounds-world-aspect={rendererContract?.boundsWorldAspect ?? 0}
      data-webgpu-projection-mode={rendererContract?.projectionMode ?? ""}
      data-webgpu-projection-camera-tuning-mode={rendererContract?.projectionCameraTuningMode ?? ""}
      data-webgpu-projection-camera-mode={rendererContract?.projectionCameraMode ?? ""}
      data-webgpu-projection-camera-fov={rendererContract?.projectionCameraFovDegrees ?? 0}
      data-webgpu-projection-camera-position={vectorAttribute(rendererContract?.projectionCameraPosition)}
      data-webgpu-projection-camera-target={vectorAttribute(rendererContract?.projectionCameraTarget)}
      data-webgpu-projection-camera-distance={rendererContract?.projectionCameraDistance ?? 0}
      data-webgpu-projection-camera-frame-max-dim={rendererContract?.projectionCameraFrameMaxDim ?? 0}
      data-webgpu-depth-weight-mode={rendererContract?.depthWeightMode ?? ""}
      data-webgpu-pixel-depth-sort-mode={rendererContract?.pixelDepthSortMode ?? ""}
      data-webgpu-pixel-depth-tuning-mode={rendererContract?.pixelDepthTuningMode ?? ""}
      data-webgpu-pixel-depth-alpha-mode={rendererContract?.pixelDepthAlphaMode ?? ""}
      data-webgpu-pixel-depth-gate-strength={rendererContract?.pixelDepthGateStrength ?? 0}
      data-webgpu-pixel-depth-gate-floor={rendererContract?.pixelDepthGateFloor ?? 0}
      data-webgpu-pixel-depth-bin-count={rendererContract?.pixelDepthBinCount ?? 0}
      data-webgpu-pixel-coverage-mode={rendererContract?.pixelCoverageMode ?? ""}
      data-webgpu-pixel-coverage-tuning-mode={rendererContract?.pixelCoverageTuningMode ?? ""}
      data-webgpu-pixel-coverage-weight-floor={rendererContract?.pixelCoverageWeightFloor ?? 0}
      data-webgpu-pixel-coverage-footprint-scale={rendererContract?.pixelCoverageFootprintScale ?? 0}
      data-webgpu-projection-depth-min={rendererContract?.projectionDepthMin ?? 0}
      data-webgpu-projection-depth-max={rendererContract?.projectionDepthMax ?? 0}
      data-webgpu-projection-depth-span={rendererContract?.projectionDepthSpan ?? 0}
      data-webgpu-color-fidelity-mode={rendererContract?.colorFidelityMode ?? ""}
      data-webgpu-color-tuning-mode={rendererContract?.colorTuningMode ?? ""}
      data-webgpu-color-mode={rendererContract?.colorMode ?? ""}
      data-webgpu-color-source-rgb-gaussians={rendererContract?.colorSourceRgbGaussians ?? 0}
      data-webgpu-color-source-sh-dc-gaussians={rendererContract?.colorSourceShDcGaussians ?? 0}
      data-webgpu-color-source-fallback-gaussians={rendererContract?.colorSourceFallbackGaussians ?? 0}
      data-webgpu-color-source-object-gaussians={rendererContract?.colorSourceObjectGaussians ?? 0}
      data-webgpu-color-sh-rest-gaussians={rendererContract?.colorShRestGaussians ?? 0}
      data-webgpu-color-sh-rest-coefficient-max={rendererContract?.colorShRestCoefficientMax ?? 0}
      data-webgpu-color-sh-degree-max={rendererContract?.colorShDegreeMax ?? 0}
      data-webgpu-color-sh-view-gaussians={rendererContract?.colorShViewGaussians ?? 0}
      data-webgpu-color-opacity-mean={rendererContract?.colorOpacityMean ?? 0}
      data-webgpu-screen-covariance-mode={rendererContract?.screenCovarianceMode ?? ""}
      data-webgpu-screen-covariance-gaussians={rendererContract?.screenCovarianceGaussians ?? 0}
      data-webgpu-screen-covariance-fallback-gaussians={rendererContract?.screenCovarianceFallbackGaussians ?? 0}
      data-webgpu-screen-covariance-clamped-gaussians={rendererContract?.screenCovarianceClampedGaussians ?? 0}
      data-webgpu-screen-covariance-max-anisotropy={rendererContract?.screenCovarianceMaxAnisotropy ?? 0}
      data-webgpu-screen-covariance-sigma-mean={rendererContract?.screenCovarianceSigmaMean ?? 0}
      data-webgpu-packed-gaussians={rendererContract?.packedGaussians ?? 0}
      data-webgpu-visible-gaussians={rendererContract?.visibleGaussians ?? 0}
      data-webgpu-binned-gaussians={rendererContract?.binnedGaussians ?? 0}
      data-webgpu-tile-list-mode={rendererContract?.tileListMode ?? tileSmoke?.tileListMode ?? ""}
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
      data-webgpu-tile-entry-layout={rendererContract?.tileEntryLayout ?? ""}
      data-webgpu-tile-entry-offset-count={rendererContract?.tileEntryOffsetCount ?? 0}
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
      data-webgpu-alpha-presentation-mode={WEBGPU_TILE_ALPHA_PRESENTATION_MODE}
      data-webgpu-alpha-presentation-tuning-mode={WEBGPU_TILE_ALPHA_PRESENTATION_TUNING_MODE}
      data-webgpu-alpha-presentation-floor={alphaPresentationTuning.alphaPresentationFloor}
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
      data-webgpu-resolve-filter={frame.filter}
      data-webgpu-runtime-probe={runtimeProbe}
      data-webgpu-device-lost-status={deviceLost.status}
      data-webgpu-device-lost-reason={deviceLost.reason}
      data-webgpu-device-lost-message={deviceLost.message}
      data-webgpu-device-error-status={deviceError.status}
      data-webgpu-device-error-type={deviceError.type}
      data-webgpu-device-error-message={deviceError.message}
      data-webgpu-queue-status={queue.status}
      data-webgpu-queue-reason={queue.reason}
      data-webgpu-queue-message={queue.message}
      data-webgpu-accumulation-source={accumulation.source}
      data-webgpu-accumulation-status={accumulation.status}
      data-webgpu-accumulation-reason={accumulation.reason}
      data-webgpu-accumulation-workgroups={accumulation.workgroups}
      data-webgpu-compute-source={compute.source}
      data-webgpu-compute-status={compute.status}
      data-webgpu-compute-reason={compute.reason}
      data-webgpu-compute-workgroups={compute.workgroups}
      data-webgpu-pixel-source={pixel.source}
      data-webgpu-pixel-status={pixel.status}
      data-webgpu-pixel-reason={pixel.reason}
      data-webgpu-pixel-workgroups={pixel.workgroups}
      data-webgpu-readback-status={readback.status}
      data-webgpu-readback-reason={readback.reason}
      data-webgpu-readback-source={readback.source}
      data-webgpu-readback-checksum={readback.checksum}
      data-webgpu-readback-byte-size={readback.byteLength}
      data-webgpu-readback-float-count={readback.floatCount}
      data-webgpu-readback-finite-floats={readback.finiteFloats}
      data-webgpu-readback-nonzero-floats={readback.nonzeroFloats}
      data-webgpu-storage-layout={storage.layoutVersion}
      data-webgpu-storage-status={storage.status}
      data-webgpu-storage-reason={storage.reason}
      data-webgpu-storage-buffer-count={storage.bufferCount}
      data-webgpu-storage-byte-size={storage.byteLength}
      data-webgpu-storage-checksum={storage.checksum}
      data-webgpu-storage-tile-entries={storage.tileEntriesIncluded ? "true" : "false"}
      data-webgpu-storage-tile-offsets={storage.tileOffsetsIncluded ? "true" : "false"}
      data-webgpu-storage-pixel-output={storage.pixelOutputIncluded ? "true" : "false"}
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

function vectorAttribute(value) {
  if (!Array.isArray(value)) return "";
  return value
    .map((entry) => Number(entry))
    .map((entry) => (Number.isFinite(entry) ? entry.toFixed(6) : "0.000000"))
    .join(",");
}

function readWebGpuRuntimeProbe() {
  if (typeof window === "undefined") return WEBGPU_RUNTIME_PROBE_FULL;
  return normalizeWebGpuRuntimeProbe(
    new URLSearchParams(window.location.search).get("webgpu-probe"),
  );
}

function readWebGpuAlphaPresentationTuning() {
  if (typeof window === "undefined") {
    return normalizeWebGpuAlphaPresentationTuning();
  }
  const params = new URLSearchParams(window.location.search);
  return normalizeWebGpuAlphaPresentationTuning({
    alphaPresentationFloor: params.get("webgpu-alpha-presentation-floor"),
  });
}

function runtimeProbeNeedsPresentation(probe) {
  const normalized = normalizeWebGpuRuntimeProbe(probe);
  return ![
    WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY,
    WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY,
    WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY,
    WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK,
  ].includes(normalized);
}

async function requestWebGpuTileDevice(adapter) {
  const supportedStorageBuffersPerStage =
    adapter.limits?.maxStorageBuffersPerShaderStage ?? 0;
  if (
    supportedStorageBuffersPerStage <
    WEBGPU_TILE_REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE
  ) {
    throw new Error("webgpu-storage-buffer-bindings-too-many");
  }
  return adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage:
        WEBGPU_TILE_REQUIRED_STORAGE_BUFFERS_PER_SHADER_STAGE,
    },
  });
}

function renderFrame({
  runtime,
  canvas,
  tileSmoke,
  runtimeProbe,
  setFrame,
  setStorage,
  setCompute,
  setPixel,
  setAccumulation,
  setReadback,
  setQueue,
}) {
  const {
    device,
    context,
    format,
    pipeline,
    sampledTexturePipeline,
    floatTexturePipeline,
    accumulationPipeline,
    computePipeline,
    pixelComputePipeline,
  } = runtime;
  const probe = normalizeWebGpuRuntimeProbe(runtimeProbe);
  const runAccumulation =
    probe === WEBGPU_RUNTIME_PROBE_FULL ||
    probe === WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY;
  const runResolve =
    probe === WEBGPU_RUNTIME_PROBE_FULL ||
    probe === WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY;
  const runPixel =
    probe === WEBGPU_RUNTIME_PROBE_FULL ||
    probe === WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY ||
    probe === WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY ||
    probe === WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK ||
    probe === WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY ||
    probe === WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT;
  const runReadback = probe === WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK;
  const runStorageDisplay =
    probe === WEBGPU_RUNTIME_PROBE_FULL ||
    probe === WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY ||
    probe === WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY ||
    probe === WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT;
  const runSampledTextureDisplay =
    probe === WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY;
  const runFloatTextureCopyDisplay =
    probe === WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY;
  const runClearOnly =
    probe === WEBGPU_RUNTIME_PROBE_CLEAR_ONLY;
  const runDisplay =
    runStorageDisplay || runSampledTextureDisplay || runFloatTextureCopyDisplay || runClearOnly;
  if (runDisplay) {
    if (!context) throw new Error("webgpu-context-unavailable");
    resizeCanvasToDisplaySize(canvas);
    context.configure({ device, format, alphaMode: "opaque" });
  }
  destroyTransientBuffers(runtime);
  setReadback(
    runReadback
      ? {
          status: "pending",
          reason: "webgpu-offscreen-readback-pending",
          source: WEBGPU_PIXEL_RESOLVE_SOURCE,
          checksum: "",
          byteLength: 0,
          floatCount: 0,
          finiteFloats: 0,
          nonzeroFloats: 0,
        }
      : {
          status: "skipped",
          reason: "webgpu-readback-not-requested",
          source: "",
          checksum: "",
          byteLength: 0,
          floatCount: 0,
          finiteFloats: 0,
          nonzeroFloats: 0,
        },
  );

  let storageBundle = null;
  try {
    const canReuseStorage =
      tileSmoke?.tileListMode === WEBGPU_TILE_LIST_MODE_OBJECT_STATE &&
      (runAccumulation || runPixel) &&
      canReuseWebGpuTileStorageBuffers(runtime.storageBundle, tileSmoke);
    if (canReuseStorage) {
      storageBundle = runtime.storageBundle;
      const description = updateWebGpuTileObjectStateBuffer(device, storageBundle, tileSmoke);
      setStorage({
        status: "object-state-updated",
        reason: "webgpu-object-state-buffer-updated",
        layoutVersion: description.layoutVersion,
        checksum: description.checksum,
        bufferCount: description.bufferCount,
        byteLength: description.totalByteLength,
        tileEntriesIncluded: description.tileEntriesIncluded,
        tileOffsetsIncluded: description.tileOffsetsIncluded,
        pixelOutputIncluded: description.pixelOutputIncluded,
      });
    } else {
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
        tileOffsetsIncluded: storageBundle.tileOffsetsIncluded,
        pixelOutputIncluded: storageBundle.pixelOutputIncluded,
      });
    }
  } catch (error) {
    setStorage({
      status: "failed",
      reason: error?.message || "webgpu-storage-upload-failed",
      layoutVersion: "",
      checksum: "",
      bufferCount: 0,
      byteLength: 0,
      tileEntriesIncluded: false,
      tileOffsetsIncluded: false,
      pixelOutputIncluded: false,
    });
    setFrame({
      status: "failed",
      reason: error?.message || "webgpu-storage-upload-failed",
      checksum: "",
      pixels: 0,
      source: "",
      filter: "",
    });
    setCompute({
      status: "failed",
      reason: error?.message || "webgpu-storage-upload-failed",
      source: "",
      workgroups: 0,
    });
    setPixel({
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
    setReadback({
      status: "failed",
      reason: error?.message || "webgpu-storage-upload-failed",
      source: "",
      checksum: "",
      byteLength: 0,
      floatCount: 0,
      finiteFloats: 0,
      nonzeroFloats: 0,
    });
    return;
  }

  try {
    const transientBuffers = [];
    const accumulationMetaBuffer = runAccumulation
      ? createAccumulationMetaBuffer(device, tileSmoke)
      : null;
    const computeMetaBuffer = runResolve ? createComputeMetaBuffer(device, tileSmoke) : null;
    const pixelMetaBuffer = runPixel ? createPixelMetaBuffer(device, tileSmoke) : null;
    const resolveMetaBuffer =
      runStorageDisplay || runFloatTextureCopyDisplay
        ? createResolveMetaBuffer(device, tileSmoke)
        : null;
    for (const buffer of [
      accumulationMetaBuffer,
      computeMetaBuffer,
      pixelMetaBuffer,
      resolveMetaBuffer,
    ]) {
      if (buffer) transientBuffers.push(buffer);
    }
    runtime.transientBuffers = transientBuffers;
    const accumulationBindGroup = runAccumulation
      ? device.createBindGroup({
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
            { binding: 8, resource: { buffer: storageBundle.getBuffer("scaleRotation").buffer } },
            { binding: 9, resource: { buffer: storageBundle.getBuffer("tileOffsets").buffer } },
          ],
        })
      : null;
    const computeBindGroup = runResolve
      ? device.createBindGroup({
          layout: computePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: storageBundle.getBuffer("tileAccumulation").buffer } },
            { binding: 1, resource: { buffer: storageBundle.getBuffer("tileResolvedRgba").buffer } },
            { binding: 2, resource: { buffer: computeMetaBuffer } },
          ],
        })
      : null;
    const pixelBindGroup = runPixel
      ? device.createBindGroup({
          layout: pixelComputePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: storageBundle.getBuffer("positionRadius").buffer } },
            { binding: 1, resource: { buffer: storageBundle.getBuffer("colorOpacity").buffer } },
            { binding: 2, resource: { buffer: storageBundle.getBuffer("objectIndices").buffer } },
            { binding: 3, resource: { buffer: storageBundle.getBuffer("objectState").buffer } },
            { binding: 4, resource: { buffer: storageBundle.getBuffer("tileCounts").buffer } },
            { binding: 5, resource: { buffer: storageBundle.getBuffer("tileEntries").buffer } },
            { binding: 6, resource: { buffer: storageBundle.getBuffer("pixelResolvedRgba").buffer } },
            { binding: 7, resource: { buffer: pixelMetaBuffer } },
            { binding: 8, resource: { buffer: storageBundle.getBuffer("scaleRotation").buffer } },
            { binding: 9, resource: { buffer: storageBundle.getBuffer("tileOffsets").buffer } },
          ],
        })
      : null;
    const bindGroup = runStorageDisplay
      ? device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: storageBundle.getBuffer("pixelResolvedRgba").buffer } },
            { binding: 1, resource: { buffer: resolveMetaBuffer } },
          ],
        })
      : null;
    const sampledTexture = runSampledTextureDisplay
      ? createSampledResolveTexture(device, tileSmoke)
      : null;
    const sampledTextureSampler = runSampledTextureDisplay
      ? device.createSampler({
          magFilter: "nearest",
          minFilter: "nearest",
          addressModeU: "clamp-to-edge",
          addressModeV: "clamp-to-edge",
        })
      : null;
    const sampledTextureBindGroup = runSampledTextureDisplay
      ? device.createBindGroup({
          layout: sampledTexturePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: sampledTexture.createView() },
            { binding: 1, resource: sampledTextureSampler },
          ],
        })
      : null;
    const floatTexture = runFloatTextureCopyDisplay
      ? createFloatResolveTexture(device, tileSmoke)
      : null;
    const floatTextureBindGroup = runFloatTextureCopyDisplay
      ? device.createBindGroup({
          layout: floatTexturePipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: floatTexture.createView() },
            { binding: 1, resource: { buffer: resolveMetaBuffer } },
          ],
        })
      : null;
    const readbackBuffer = runReadback
      ? createReadbackBuffer(device, storageBundle.getBuffer("pixelResolvedRgba").allocatedByteLength)
      : null;
    runtime.transientTextures = [sampledTexture, floatTexture].filter(Boolean);
    const encoder = device.createCommandEncoder();
    const accumulationWorkgroups = webGpuAccumulationWorkgroups(tileSmoke);
    const workgroups = webGpuComputeWorkgroups(tileSmoke);
    const pixelWorkgroups = webGpuPixelResolveWorkgroups(tileSmoke);
    if (runAccumulation || runResolve || runPixel) {
      const computePass = encoder.beginComputePass();
      if (runAccumulation) {
        computePass.setPipeline(accumulationPipeline);
        computePass.setBindGroup(0, accumulationBindGroup);
        computePass.dispatchWorkgroups(accumulationWorkgroups);
      }
      if (runResolve) {
        computePass.setPipeline(computePipeline);
        computePass.setBindGroup(0, computeBindGroup);
        computePass.dispatchWorkgroups(workgroups);
      }
      if (runPixel) {
        computePass.setPipeline(pixelComputePipeline);
        computePass.setBindGroup(0, pixelBindGroup);
        computePass.dispatchWorkgroups(pixelWorkgroups);
      }
      computePass.end();
    }
    if (runFloatTextureCopyDisplay) {
      encoder.copyBufferToTexture(
        {
          buffer: storageBundle.getBuffer("pixelResolvedRgba").buffer,
          bytesPerRow: Math.max(1, tileSmoke.viewportWidth) * 16,
          rowsPerImage: Math.max(1, tileSmoke.viewportHeight),
        },
        { texture: floatTexture },
        {
          width: Math.max(1, tileSmoke.viewportWidth),
          height: Math.max(1, tileSmoke.viewportHeight),
          depthOrArrayLayers: 1,
        },
      );
    }
    if (runReadback) {
      const pixelBuffer = storageBundle.getBuffer("pixelResolvedRgba");
      encoder.copyBufferToBuffer(
        pixelBuffer.buffer,
        0,
        readbackBuffer,
        0,
        pixelBuffer.allocatedByteLength,
      );
    }
    if (runDisplay) {
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
      if (runClearOnly) {
        // Deliberately no draw call: this isolates canvas render pass submission.
      } else if (runSampledTextureDisplay) {
        pass.setPipeline(sampledTexturePipeline);
        pass.setBindGroup(0, sampledTextureBindGroup);
        pass.draw(3);
      } else if (runFloatTextureCopyDisplay) {
        pass.setPipeline(floatTexturePipeline);
        pass.setBindGroup(0, floatTextureBindGroup);
        pass.draw(3);
      } else {
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(3);
      }
      pass.end();
    }
    device.queue.submit([encoder.finish()]);
    setQueue({
      status: "submitted",
      reason: "webgpu-queue-submitted",
      message: "",
    });
    device.queue.onSubmittedWorkDone().then(
      async () => {
        setQueue({
          status: "done",
          reason: "webgpu-queue-submitted-work-done",
          message: "",
        });
        if (runReadback) {
          try {
            const result = await readbackBufferTelemetry({
              buffer: readbackBuffer,
              byteLength: storageBundle.getBuffer("pixelResolvedRgba").byteLength,
            });
            setReadback({
              status: "mapped",
              reason: "webgpu-offscreen-readback-mapped",
              source: WEBGPU_PIXEL_RESOLVE_SOURCE,
              ...result,
            });
            setFrame({
              status: "readback",
              reason: "webgpu-offscreen-readback-probe-mapped",
              checksum: result.checksum,
              pixels: Math.floor(result.floatCount / 4),
              source: WEBGPU_PIXEL_RESOLVE_SOURCE,
              filter: "offscreen-map-read",
            });
          } catch (error) {
            setReadback({
              status: "failed",
              reason: error?.message || "webgpu-offscreen-readback-failed",
              source: WEBGPU_PIXEL_RESOLVE_SOURCE,
              checksum: "",
              byteLength: 0,
              floatCount: 0,
              finiteFloats: 0,
              nonzeroFloats: 0,
            });
            setFrame({
              status: "failed",
              reason: error?.message || "webgpu-offscreen-readback-failed",
              checksum: "",
              pixels: 0,
              source: WEBGPU_PIXEL_RESOLVE_SOURCE,
              filter: "offscreen-map-read",
            });
          }
        }
      },
      (error) =>
        setQueue({
          status: "failed",
          reason: "webgpu-queue-submitted-work-failed",
          message: error?.message || String(error || ""),
        }),
    );
    setAccumulation(stageTelemetry({
      active: runAccumulation,
      activeReason: "webgpu-tile-accumulation-dispatched",
      skippedReason: "webgpu-runtime-probe-skipped",
      source: WEBGPU_TILE_ACCUMULATION_SOURCE,
      workgroups: accumulationWorkgroups,
    }));
    setCompute(stageTelemetry({
      active: runResolve,
      activeReason: "webgpu-compute-resolve-dispatched",
      skippedReason: "webgpu-runtime-probe-skipped",
      source: WEBGPU_TILE_COMPUTE_SOURCE,
      workgroups,
    }));
    setPixel(stageTelemetry({
      active: runPixel,
      activeReason: "webgpu-pixel-accumulation-dispatched",
      skippedReason: "webgpu-runtime-probe-skipped",
      source: WEBGPU_PIXEL_RESOLVE_SOURCE,
      workgroups: pixelWorkgroups,
    }));
    if (!runReadback) {
      setFrame(frameTelemetry({ probe, tileSmoke, runDisplay }));
    }
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
    setPixel({
      status: "failed",
      reason: error?.message || "webgpu-pixel-accumulation-failed",
      source: "",
      workgroups: 0,
    });
    setFrame({
      status: "failed",
      reason: error?.message || "webgpu-pixel-accumulation-failed",
      checksum: "",
      pixels: 0,
      source: "",
      filter: "",
    });
    setReadback({
      status: "failed",
      reason: error?.message || "webgpu-offscreen-readback-failed",
      source: runReadback ? WEBGPU_PIXEL_RESOLVE_SOURCE : "",
      checksum: "",
      byteLength: 0,
      floatCount: 0,
      finiteFloats: 0,
      nonzeroFloats: 0,
    });
    setQueue({
      status: "failed",
      reason: error?.message || "webgpu-queue-submit-failed",
      message: "",
    });
  }
}

function stageTelemetry({
  active,
  activeReason,
  skippedReason,
  source,
  workgroups,
}) {
  return active
    ? {
        status: "dispatched",
        reason: activeReason,
        source,
        workgroups,
      }
    : {
        status: "skipped",
        reason: skippedReason,
        source: "",
        workgroups: 0,
      };
}

function frameTelemetry({ probe, tileSmoke, runDisplay }) {
  if (runDisplay) {
    const textureDisplay = textureDisplayFrameTelemetry(probe, tileSmoke);
    return {
      status: "rendered",
      reason: renderedFrameReason(probe),
      checksum: textureDisplay.checksum,
      pixels: textureDisplay.pixels,
      source: textureDisplay.source,
      filter: textureDisplay.filter,
    };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY) {
    return {
      status: "probed",
      reason: "webgpu-accumulation-only-probe-submitted",
      checksum: tileSmoke.resolveChecksum,
      pixels: tileSmoke.resolvedTileCount,
      source: WEBGPU_TILE_ACCUMULATION_SOURCE,
      filter: "",
    };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY) {
    return {
      status: "probed",
      reason: "webgpu-resolve-only-probe-submitted",
      checksum: tileSmoke.resolveChecksum,
      pixels: tileSmoke.resolvedTileCount,
      source: WEBGPU_TILE_COMPUTE_SOURCE,
      filter: "",
    };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY) {
    return {
      status: "probed",
      reason: "webgpu-pixel-compute-only-probe-submitted",
      checksum: tileSmoke.pixelResolveChecksum || tileSmoke.resolveChecksum,
      pixels: tileSmoke.pixelResolvedCount || tileSmoke.pixelCount || tileSmoke.resolvedTileCount,
      source: WEBGPU_PIXEL_RESOLVE_SOURCE,
      filter: "",
    };
  }
  return {
    status: "probed",
    reason: "webgpu-runtime-probe-submitted",
    checksum: tileSmoke.resolveChecksum,
    pixels: tileSmoke.resolvedTileCount,
    source: "",
    filter: "",
  };
}

function renderedFrameReason(probe) {
  if (probe === WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY) {
    return "webgpu-pixel-output-only-probe-rendered";
  }
  if (probe === WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY) {
    return "webgpu-display-only-probe-rendered";
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT) {
    return "webgpu-tiny-pixel-output-probe-rendered";
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY) {
    return "webgpu-texture-display-only-probe-rendered";
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY) {
    return "webgpu-texture-copy-display-probe-rendered";
  }
  if (probe === WEBGPU_RUNTIME_PROBE_CLEAR_ONLY) {
    return "webgpu-clear-only-probe-rendered";
  }
  return "webgpu-pixel-storage-resolve-rendered";
}

function textureDisplayFrameTelemetry(probe, tileSmoke) {
  if (probe === WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY) {
    return {
      checksum: tileSmoke.resolveChecksum,
      pixels: tileSmoke.pixelCount || tileSmoke.resolvedTileCount,
      source: WEBGPU_SAMPLED_TEXTURE_RESOLVE_SOURCE,
      filter: "nearest-sampled-texture",
    };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY) {
    return {
      checksum: tileSmoke.pixelResolveChecksum || tileSmoke.resolveChecksum,
      pixels: tileSmoke.pixelResolvedCount || tileSmoke.pixelCount || tileSmoke.resolvedTileCount,
      source: WEBGPU_FLOAT_TEXTURE_COPY_RESOLVE_SOURCE,
      filter: "nearest-texture-load",
    };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_CLEAR_ONLY) {
    return {
      checksum: tileSmoke.resolveChecksum,
      pixels: tileSmoke.pixelCount || tileSmoke.resolvedTileCount,
      source: "webgpu-clear-pass-v1",
      filter: "clear-pass",
    };
  }
  return {
    checksum: tileSmoke.pixelResolveChecksum || tileSmoke.resolveChecksum,
    pixels: tileSmoke.pixelResolvedCount || tileSmoke.pixelCount || tileSmoke.resolvedTileCount,
    source: WEBGPU_TILE_RESOLVE_SOURCE,
    filter: WEBGPU_TILE_RESOLVE_FILTER,
  };
}

function createReadbackBuffer(device, byteLength) {
  const usage = globalThis.GPUBufferUsage ?? {
    COPY_DST: 0x0008,
    MAP_READ: 0x0001,
  };
  return device.createBuffer({
    label: "objgauss-offscreen-pixel-readback",
    size: Math.max(4, Math.ceil((Number(byteLength) || 0) / 4) * 4),
    usage: usage.COPY_DST | usage.MAP_READ,
  });
}

async function readbackBufferTelemetry({ buffer, byteLength }) {
  const readByteLength = Math.max(4, Math.floor(Number(byteLength) || 0));
  const mapMode = globalThis.GPUMapMode ?? { READ: 0x0001 };
  await buffer.mapAsync(mapMode.READ, 0, readByteLength);
  const mapped = buffer.getMappedRange(0, readByteLength);
  const bytes = new Uint8Array(mapped);
  const floats = new Float32Array(
    mapped,
    0,
    Math.floor(readByteLength / Float32Array.BYTES_PER_ELEMENT),
  );
  const checksum = checksumBytes(bytes);
  const floatCount = floats.length;
  let finiteFloats = 0;
  let nonzeroFloats = 0;
  for (const value of floats) {
    if (Number.isFinite(value)) finiteFloats += 1;
    if (value !== 0) nonzeroFloats += 1;
  }
  buffer.unmap();
  buffer.destroy?.();
  return {
    checksum,
    byteLength: readByteLength,
    floatCount,
    finiteFloats,
    nonzeroFloats,
  };
}

function checksumBytes(bytes) {
  let hash = 2166136261;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
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

function createPixelMetaBuffer(device, tileSmoke) {
  const data = createWebGpuPixelResolveMeta(tileSmoke);
  const usage = globalThis.GPUBufferUsage ?? {
    COPY_DST: 0x0008,
    UNIFORM: 0x0040,
  };
  const buffer = device.createBuffer({
    label: "objgauss-pixel-resolve-meta",
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

function createSampledResolveTexture(device, tileSmoke) {
  const width = Math.max(1, tileSmoke?.viewportWidth ?? 1);
  const height = Math.max(1, tileSmoke?.viewportHeight ?? 1);
  const usage = textureUsage();
  const texture = device.createTexture({
    label: "objgauss-sampled-resolve-texture",
    size: { width, height, depthOrArrayLayers: 1 },
    format: "rgba8unorm",
    usage: usage.TEXTURE_BINDING | usage.COPY_DST,
  });
  device.queue.writeTexture(
    { texture },
    createSampledResolveTextureData(tileSmoke, width, height),
    {
      bytesPerRow: width * 4,
      rowsPerImage: height,
    },
    { width, height, depthOrArrayLayers: 1 },
  );
  return texture;
}

function createFloatResolveTexture(device, tileSmoke) {
  const usage = textureUsage();
  return device.createTexture({
    label: "objgauss-float-copy-resolve-texture",
    size: {
      width: Math.max(1, tileSmoke?.viewportWidth ?? 1),
      height: Math.max(1, tileSmoke?.viewportHeight ?? 1),
      depthOrArrayLayers: 1,
    },
    format: "rgba32float",
    usage: usage.TEXTURE_BINDING | usage.COPY_DST,
  });
}

function createSampledResolveTextureData(tileSmoke, width, height) {
  const data = new Uint8Array(width * height * 4);
  const tileResolved = tileSmoke?.buffers?.tileResolvedRgba;
  const tileColumns = Math.max(1, tileSmoke?.tileColumns ?? 1);
  const tileSize = Math.max(1, tileSmoke?.tileSize ?? 1);
  for (let y = 0; y < height; y += 1) {
    const tileY = Math.floor(y / tileSize);
    for (let x = 0; x < width; x += 1) {
      const tileX = Math.min(Math.floor(x / tileSize), tileColumns - 1);
      const tileIndex = tileY * tileColumns + tileX;
      const tileOffset = tileIndex * 4;
      const pixelOffset = (y * width + x) * 4;
      const alpha = clampNumber(tileResolved?.[tileOffset + 3] ?? 0, 0, 0.98);
      const red = clampNumber(tileResolved?.[tileOffset] ?? 0, 0, 1);
      const green = clampNumber(tileResolved?.[tileOffset + 1] ?? 0, 0, 1);
      const blue = clampNumber(tileResolved?.[tileOffset + 2] ?? 0, 0, 1);
      data[pixelOffset] = Math.round((BACKGROUND_RGB[0] * (1 - alpha)) + red * alpha * 255);
      data[pixelOffset + 1] = Math.round((BACKGROUND_RGB[1] * (1 - alpha)) + green * alpha * 255);
      data[pixelOffset + 2] = Math.round((BACKGROUND_RGB[2] * (1 - alpha)) + blue * alpha * 255);
      data[pixelOffset + 3] = 255;
    }
  }
  return data;
}

function textureUsage() {
  return globalThis.GPUTextureUsage ?? {
    COPY_DST: 0x02,
    TEXTURE_BINDING: 0x04,
  };
}

function destroyTransientBuffers(runtime) {
  if (!runtime) return;
  for (const buffer of runtime?.transientBuffers ?? []) {
    buffer?.destroy?.();
  }
  for (const texture of runtime?.transientTextures ?? []) {
    texture?.destroy?.();
  }
  runtime.transientBuffers = [];
  runtime.transientTextures = [];
}

function clampNumber(value, min, max) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return min;
  return Math.min(max, Math.max(min, numeric));
}

function pickObjectFromTileFrame({
  event,
  canvas,
  points,
  tileSmoke,
  visibleIds,
  removedIds,
  isolatedId,
}) {
  const rect = canvas.getBoundingClientRect();
  const viewportWidth = Math.max(1, tileSmoke?.viewportWidth ?? canvas.width ?? rect.width);
  const viewportHeight = Math.max(1, tileSmoke?.viewportHeight ?? canvas.height ?? rect.height);
  const bounds = buildWebGpuTileProjectionBounds(points, viewportWidth, viewportHeight);
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
    const screen = projectPointToWebGpuTileViewport({ point, bounds, viewportWidth, viewportHeight });
    const x = (screen.x / Math.max(1, viewportWidth - 1)) * rect.width;
    const y = (screen.y / Math.max(1, viewportHeight - 1)) * rect.height;
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
