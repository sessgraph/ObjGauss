import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";
import {
  canvasVisualStats,
  compareVisualStats,
  roundMetric,
  validateCanvasVisualStats,
} from "./lib/visual-stats.mjs";
import {
  normalizeWebGpuDepthAlphaMode,
  normalizeWebGpuPixelDepthBinCount,
  WEBGPU_DEPTH_ALPHA_MODE_DEFAULT,
  WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED,
  WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K,
  WEBGPU_DEPTH_SORT_TUNING_MODE,
} from "../src/webgpuDepthTuning.js";
import {
  normalizeWebGpuCameraTuning,
  WEBGPU_CAMERA_MODE_EDIT_FIXED,
  WEBGPU_CAMERA_MODE_SPARK_FRAME,
  WEBGPU_CAMERA_TUNING_MODE,
} from "../src/webgpuCameraTuning.js";
import { WEBGPU_PIXEL_RESOLVE_SOURCE } from "../src/webgpuTileComputeShader.js";
import {
  WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR,
  WEBGPU_TILE_ALPHA_PRESENTATION_MODE,
  WEBGPU_TILE_ALPHA_PRESENTATION_TUNING_MODE,
  normalizeWebGpuAlphaPresentationTuning,
} from "../src/webgpuTileResolveShader.js";
import {
  normalizeWebGpuColorTuning,
  WEBGPU_COLOR_MODE_SOURCE,
  WEBGPU_COLOR_MODE_SH_VIEW,
  WEBGPU_COLOR_TUNING_MODE,
} from "../src/webgpuTileSmoke.js";
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
} from "../src/webgpuRuntimeProbe.js";

const DEFAULT_PORT = 5395;
const DEFAULT_ASSETS = [
  {
    id: "plush-semantic-closure-local",
    name: "Plush 2D 语义 Mask 闭环样例",
    fileName: "plush_semantic_objects.ply",
  },
  {
    id: "plush-v1-closure-local",
    name: "ObjGauss v1 闭环样例",
    fileName: "plush_v1_objects.ply",
  },
  {
    id: "nerf-lego-alpha-closure-local",
    name: "NeRF Lego 闭环代理样例",
    fileName: "lego_alpha_v1_objects.ply",
  },
];
const KNOWN_ASSETS = [
  ...DEFAULT_ASSETS,
  {
    id: "nerf-lego-trained-output-local",
    name: "NeRF Lego 训练输出样例",
    fileName: "nerf_lego_trained_objects.ply",
  },
  {
    id: "polyhaven-chair-commercial-demo-local",
    name: "Poly Haven Chair 商用展示样例",
    fileName: "polyhaven_chair_demo_objects.ply",
  },
];
const HARD_MASK_QUALITY_EXPECTATIONS = {
  "nerf-lego-alpha-closure-local": "boundary-mixing-dominant",
  "plush-semantic-closure-local": "boundary-mixing-dominant",
  "nerf-lego-trained-output-local": "browser-residual-dominant",
  "polyhaven-chair-commercial-demo-local": "boundary-mixing-dominant",
};
const DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE = 320;
const VISUAL_RESIDUAL_MODE = "spark-edit-visual-residual-v1";
const VISUAL_RESIDUAL_SKIPPED_MODE = "skipped-by-native-mask-gate-v1";
const SPARK_RECONSTRUCT_SOURCE = "packed-extract-v1";
const SPARK_RECONSTRUCT_SH_SOURCE = "packed-sh-extract-v1";
const SPARK_NATIVE_RECONSTRUCT_SOURCE = "native-splat-source-v1";
const SPARK_DISPLAY_CACHE_DISABLED = "disabled-by-native-mask-v1";
const SPARK_OBJECT_FILTER_MASK = "spark-object-opacity-mask";
const SPARK_OBJECT_MASK_MODE = "object-opacity-texture-v1";
const SPARK_MESH_UPDATE_MODE = "persistent-splatmesh-v1";
const SPARK_OBJECT_MASK_VISUAL_DELTA_MODE = "spark-object-mask-visual-delta-v1";
const SPARK_PICK_MODE = "screen-space-object-pick-v1";
const SPARK_PICK_INTERACTION_MODE = "hover-confirm-v1";
const SPARK_RESTORE_STRESS_MAX_GAUSSIANS = 100_000;
const SPARK_OBJECT_MASK_MIN_VISUAL_DELTA = 0.0005;
const SPARK_OBJECT_MASK_MAX_RESTORE_DELTA = 0.002;

const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = args.url ?? `http://127.0.0.1:${port}/`;
const assets = selectAssets(args);
const serverMode = normalizeServerMode(args.serverMode ?? args["server-mode"] ?? "dev");
const auditOptions = {
  requireWebGpu: Boolean(args.requireWebgpu ?? args["require-webgpu"]),
  webGpuFlags: String(args.webgpuFlags ?? args["webgpu-flags"] ?? process.env.OBJGAUSS_WEBGPU_FLAGS ?? "none"),
  headed: flagEnabled(args.headed ?? args.headful ?? process.env.OBJGAUSS_PLAYWRIGHT_HEADED),
  browserChannel: optionalString(args.browserChannel ?? args["browser-channel"] ?? process.env.PLAYWRIGHT_BROWSER_CHANNEL),
  executablePath: optionalString(args.executablePath ?? args["executable-path"] ?? process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE),
  slowMo: Number(args.slowMo ?? args["slow-mo"] ?? 0),
  webGpuProbe: normalizeWebGpuRuntimeProbe(
    args.webgpuProbe ?? args["webgpu-probe"] ?? process.env.OBJGAUSS_WEBGPU_PROBE,
  ),
  webGpuViewportSize: optionalPositiveInteger(
    args.webgpuViewportSize ??
      args["webgpu-viewport-size"] ??
      process.env.OBJGAUSS_WEBGPU_VIEWPORT_SIZE,
  ),
  webGpuFootprintScale: optionalFiniteNumber(
    args.webgpuFootprintScale ??
      args["webgpu-footprint-scale"] ??
      process.env.OBJGAUSS_WEBGPU_FOOTPRINT_SCALE,
  ),
  webGpuCovarianceMaxAnisotropy: optionalFiniteNumber(
    args.webgpuCovarianceMaxAnisotropy ??
      args["webgpu-covariance-max-anisotropy"] ??
      process.env.OBJGAUSS_WEBGPU_COVARIANCE_MAX_ANISOTROPY,
  ),
  webGpuDepthBins: optionalFiniteNumber(
    args.webGpuDepthBins ??
      args["webgpu-depth-bins"] ??
      process.env.OBJGAUSS_WEBGPU_DEPTH_BINS,
  ),
  webGpuDepthAlphaMode: normalizeWebGpuDepthAlphaMode(
    args.webGpuDepthAlphaMode ??
      args["webgpu-depth-alpha-mode"] ??
      process.env.OBJGAUSS_WEBGPU_DEPTH_ALPHA_MODE,
  ),
  webGpuCameraTuning: normalizeWebGpuCameraTuning({
    cameraMode:
      args.webGpuCameraMode ??
      args["webgpu-camera-mode"] ??
      process.env.OBJGAUSS_WEBGPU_CAMERA_MODE,
  }),
  webGpuColorTuning: normalizeWebGpuColorTuning({
    colorMode:
      args.webGpuColorMode ??
      args["webgpu-color-mode"] ??
      process.env.OBJGAUSS_WEBGPU_COLOR_MODE,
  }),
  webGpuAlphaPresentationTuning: normalizeWebGpuAlphaPresentationTuning({
    alphaPresentationFloor:
      args.webGpuAlphaPresentationFloor ??
      args["webgpu-alpha-presentation-floor"] ??
      process.env.OBJGAUSS_WEBGPU_ALPHA_PRESENTATION_FLOOR,
  }),
  sparkNativeMask: flagEnabled(
    args.sparkNativeMask ??
      args["spark-native-mask"] ??
      process.env.OBJGAUSS_SPARK_NATIVE_MASK,
  ),
  sparkObjectMaskFeather: {
    enabled: flagEnabled(
      args.sparkObjectMaskFeather ??
        args["spark-object-mask-feather"] ??
        process.env.OBJGAUSS_SPARK_OBJECT_MASK_FEATHER,
    ),
    radius: optionalFiniteNumber(
      args.sparkObjectMaskFeatherRadius ??
        args["spark-object-mask-feather-radius"] ??
        process.env.OBJGAUSS_SPARK_OBJECT_MASK_FEATHER_RADIUS,
    ),
    opacity: optionalFiniteNumber(
      args.sparkObjectMaskFeatherOpacity ??
        args["spark-object-mask-feather-opacity"] ??
        process.env.OBJGAUSS_SPARK_OBJECT_MASK_FEATHER_OPACITY,
    ),
  },
  skipVisualResidual: flagEnabled(
    args.skipVisualResidual ??
      args["skip-visual-residual"] ??
      process.env.OBJGAUSS_SKIP_VISUAL_RESIDUAL,
  ),
  allowWebGpuDeviceLost: Boolean(
    args.allowWebgpuDeviceLost ?? args["allow-webgpu-device-lost"],
  ),
  webGpuObjectTransition: flagEnabled(
    args.webGpuObjectTransition ??
      args["webgpu-object-transition"] ??
      process.env.OBJGAUSS_WEBGPU_OBJECT_TRANSITION,
  ),
  webGpuPresentationOnly: flagEnabled(
    args.webGpuPresentationOnly ??
      args["webgpu-presentation-only"] ??
      process.env.OBJGAUSS_WEBGPU_PRESENTATION_ONLY,
  ),
};

if (
  auditOptions.webGpuObjectTransition &&
  ![
    WEBGPU_RUNTIME_PROBE_FULL,
    WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK,
  ].includes(auditOptions.webGpuProbe)
) {
  throw new Error("--webgpu-object-transition is only supported with full or offscreen-readback WebGPU probes");
}

const server = args.url || args.noServer ? null : startServer(port, serverMode);
try {
  await waitForApp(baseUrl);
  const results = await runAudit(baseUrl, assets, auditOptions);
  for (const result of results) {
    console.log(
        `asset=${result.assetId} title=${JSON.stringify(result.title)} ` +
        `splatPixels=${result.splatPixels} splatRendererId=${JSON.stringify(result.splatRendererId)} ` +
        `visualResidual=${JSON.stringify(result.visualResidualMode)}:${result.sparkVisualCoverage}/${result.editOriginalVisualCoverage}:${result.sparkEditCoverageRatio}:${result.sparkEditLumaDelta}:${result.sparkEditChromaDelta} ` +
        `editRenderer=${JSON.stringify(result.editRenderer)} ` +
        `editRendererId=${JSON.stringify(result.editRendererId)} ` +
        `route=${JSON.stringify(result.initialRouteId)}:${JSON.stringify(result.initialRouteKind)}:${JSON.stringify(result.initialColorModeRole)}:${JSON.stringify(result.initialSourcePreviewBoundary)}:${JSON.stringify(result.initialHardMaskQuality)} ` +
        `objectColorRoute=${JSON.stringify(result.objectColorRouteId)}:${JSON.stringify(result.objectColorRouteKind)}:${JSON.stringify(result.objectColorModeRole)}:${JSON.stringify(result.objectColorSourcePreviewBoundary)}:${JSON.stringify(result.objectColorHardMaskQuality)} ` +
        `runtimeProbe=${JSON.stringify(result.webGpuRuntimeProbe)} ` +
        `firstFrame=${JSON.stringify(result.webGpuFirstFrameStatus)}:${result.webGpuFirstFramePixels} ` +
        `webgpuViewport=${result.webGpuViewportWidth}x${result.webGpuViewportHeight}:${result.webGpuPixelCount}:${JSON.stringify(result.webGpuViewportAspectMode)}:${JSON.stringify(result.webGpuViewportQuality)}:${result.webGpuViewportPixelBudget} ` +
        `display=${result.webGpuDisplayWidth}x${result.webGpuDisplayHeight} boundsFit=${JSON.stringify(result.webGpuBoundsFitMode)}:${result.webGpuBoundsWorldAspect}/${result.webGpuBoundsViewportAspect} ` +
        `projection=${JSON.stringify(result.webGpuProjectionMode)}:${JSON.stringify(result.webGpuProjectionCameraTuningMode)}:${JSON.stringify(result.webGpuProjectionCameraMode)}:${result.webGpuProjectionCameraFov}:${result.webGpuProjectionCameraDistance} ` +
        `depthWeight=${JSON.stringify(result.webGpuDepthWeightMode)}:${result.webGpuProjectionDepthMin}/${result.webGpuProjectionDepthMax}/${result.webGpuProjectionDepthSpan} ` +
        `pixelDepthSort=${JSON.stringify(result.webGpuPixelDepthSortMode)}:${JSON.stringify(result.webGpuPixelDepthTuningMode)}:${JSON.stringify(result.webGpuPixelDepthAlphaMode)}:${result.webGpuPixelDepthGateStrength}/${result.webGpuPixelDepthGateFloor}:${result.webGpuPixelDepthBinCount} ` +
        `pixelCoverage=${JSON.stringify(result.webGpuPixelCoverageMode)}:${JSON.stringify(result.webGpuPixelCoverageTuningMode)}:${result.webGpuPixelCoverageWeightFloor}:${result.webGpuPixelCoverageFootprintScale} ` +
        `colorTuning=${JSON.stringify(result.webGpuColorTuningMode)}:${JSON.stringify(result.webGpuColorMode)}:${result.webGpuColorShViewGaussians} ` +
        `colorFidelity=${JSON.stringify(result.webGpuColorFidelityMode)}:${result.webGpuColorSourceRgbGaussians}/${result.webGpuColorSourceShDcGaussians}/${result.webGpuColorSourceFallbackGaussians}/${result.webGpuColorSourceObjectGaussians}:${result.webGpuColorOpacityMean} ` +
        `shRest=${result.webGpuColorShRestGaussians}/${result.webGpuColorShRestCoefficientMax}/${result.webGpuColorShDegreeMax} ` +
        `colorAfterDelete=${result.webGpuColorSourceRgbGaussiansAfterDelete}/${result.webGpuColorSourceShDcGaussiansAfterDelete}/${result.webGpuColorSourceFallbackGaussiansAfterDelete}/${result.webGpuColorSourceObjectGaussiansAfterDelete} ` +
        `shViewAfterDelete=${result.webGpuColorShViewGaussiansAfterDelete} ` +
        `screenCovariance=${JSON.stringify(result.webGpuScreenCovarianceMode)}:${result.webGpuScreenCovarianceGaussians}/${result.webGpuScreenCovarianceFallbackGaussians}/${result.webGpuScreenCovarianceClampedGaussians}:${result.webGpuScreenCovarianceMaxAnisotropy}:${result.webGpuScreenCovarianceSigmaMean} ` +
        `deviceLost=${JSON.stringify(result.webGpuDeviceLostStatus)}:${JSON.stringify(result.webGpuDeviceLostReason)} ` +
        `deviceError=${JSON.stringify(result.webGpuDeviceErrorStatus)}:${JSON.stringify(result.webGpuDeviceErrorType)} ` +
        `queue=${JSON.stringify(result.webGpuQueueStatus)}:${JSON.stringify(result.webGpuQueueReason)} ` +
        `accumulation=${JSON.stringify(result.webGpuAccumulationStatus)}:${JSON.stringify(result.webGpuAccumulationSource)}:${result.webGpuAccumulationWorkgroups} ` +
        `compute=${JSON.stringify(result.webGpuComputeStatus)}:${JSON.stringify(result.webGpuComputeSource)}:${result.webGpuComputeWorkgroups} ` +
        `pixel=${JSON.stringify(result.webGpuPixelStatus)}:${JSON.stringify(result.webGpuPixelSource)}:${result.webGpuPixelWorkgroups} ` +
        `readback=${JSON.stringify(result.webGpuReadbackStatus ?? "")}:${JSON.stringify(result.webGpuReadbackSource ?? "")}:${JSON.stringify(result.webGpuReadbackChecksum ?? "")}:${result.webGpuReadbackByteSize ?? 0}:${result.webGpuReadbackFiniteFloats ?? 0}/${result.webGpuReadbackFloatCount ?? 0}:${result.webGpuReadbackNonzeroFloats ?? 0} ` +
        `readbackAfterIsolate=${JSON.stringify(result.webGpuReadbackStatusAfterIsolate ?? "")}:${JSON.stringify(result.webGpuReadbackSourceAfterIsolate ?? "")}:${JSON.stringify(result.webGpuReadbackChecksumAfterIsolate ?? "")}:${result.webGpuReadbackByteSizeAfterIsolate ?? 0}:${result.webGpuReadbackFiniteFloatsAfterIsolate ?? 0}/${result.webGpuReadbackFloatCountAfterIsolate ?? 0}:${result.webGpuReadbackNonzeroFloatsAfterIsolate ?? 0} ` +
        `readbackAfterDelete=${JSON.stringify(result.webGpuReadbackStatusAfterDelete ?? "")}:${JSON.stringify(result.webGpuReadbackSourceAfterDelete ?? "")}:${JSON.stringify(result.webGpuReadbackChecksumAfterDelete ?? "")}:${result.webGpuReadbackByteSizeAfterDelete ?? 0}:${result.webGpuReadbackFiniteFloatsAfterDelete ?? 0}/${result.webGpuReadbackFloatCountAfterDelete ?? 0}:${result.webGpuReadbackNonzeroFloatsAfterDelete ?? 0} ` +
        `resolveSource=${JSON.stringify(result.webGpuResolveSource)}:${JSON.stringify(result.webGpuResolveFilter)}:${JSON.stringify(result.webGpuAlphaPresentationMode)}:${result.webGpuAlphaPresentationFloor} ` +
        `storage=${JSON.stringify(result.webGpuStorageStatus)}:${JSON.stringify(result.webGpuStorageChecksum)} ` +
        `storageTiming=${JSON.stringify(result.webGpuStorageStatus)}:${JSON.stringify(result.webGpuStorageUpdateMode)}:${result.webGpuStorageUpdateMs}:${result.webGpuFrameSubmitMs}:${result.webGpuQueueDoneMs}:${result.webGpuStorageObjectStateByteSize} ` +
        `storageTimingAfterIsolate=${JSON.stringify(result.webGpuStorageStatusAfterIsolate ?? "")}:${JSON.stringify(result.webGpuStorageUpdateModeAfterIsolate ?? "")}:${result.webGpuStorageUpdateMsAfterIsolate ?? 0}:${result.webGpuFrameSubmitMsAfterIsolate ?? 0}:${result.webGpuQueueDoneMsAfterIsolate ?? 0}:${result.webGpuStorageObjectStateByteSizeAfterIsolate ?? 0} ` +
        `storageTimingAfterDelete=${JSON.stringify(result.webGpuStorageStatusAfterDelete ?? "")}:${JSON.stringify(result.webGpuStorageUpdateModeAfterDelete ?? "")}:${result.webGpuStorageUpdateMsAfterDelete ?? 0}:${result.webGpuFrameSubmitMsAfterDelete ?? 0}:${result.webGpuQueueDoneMsAfterDelete ?? 0}:${result.webGpuStorageObjectStateByteSizeAfterDelete ?? 0} ` +
        `storageLimit=${JSON.stringify(result.storageLimitGate)}:${JSON.stringify(result.storageLimitBlocker)}:${JSON.stringify(result.storageEstimatedMaxBufferKey)}:${result.storageEstimatedMaxBufferByteSize}:${result.storageLimitRequiredStorageBuffersPerStage}/${result.storageLimitMaxStorageBuffersPerStage} ` +
        `rendererTarget=${JSON.stringify(result.rendererTarget)} ` +
        `targetGate=${JSON.stringify(result.targetGate)}:${JSON.stringify(result.targetGateBlocker)} ` +
        `webgpuStatus=${JSON.stringify(result.webgpuStatus)} ` +
        `fallbackReason=${JSON.stringify(result.fallbackReason)} ` +
        `tileSmokeLayout=${JSON.stringify(result.tileSmokeLayout)} ` +
        `packedGaussians=${result.packedGaussians} ` +
        `binnedGaussians=${result.binnedGaussians} ` +
        `activeTiles=${result.activeTileCount}/${result.tileCount} ` +
        `tileReferences=${result.tileReferenceCount} ` +
        `maxTileOccupancy=${result.maxTileOccupancy} ` +
        `tileCapacity=${JSON.stringify(result.tileCapacityMode)}:${JSON.stringify(result.tileCapacityStatus)}:${result.tileOverflowTileCount} ` +
        `tileEntryLayout=${JSON.stringify(result.tileEntryLayout)}:${result.tileEntryOffsetCount} ` +
        `resolveLayout=${JSON.stringify(result.resolveLayout)} ` +
        `resolvedTiles=${result.resolvedTileCount} ` +
        `resolveChecksum=${JSON.stringify(result.resolveChecksum)} ` +
        `objectState=${JSON.stringify(result.objectStateLayout)}:${JSON.stringify(result.objectStateChecksum)} ` +
        `objectStateAfterIsolate=${JSON.stringify(result.objectStateChecksumAfterIsolate)} ` +
        `objectStateAfterDelete=${JSON.stringify(result.objectStateChecksumAfterDelete)} ` +
        `tileOverflowCount=${result.tileOverflowCount} ` +
        `objectFilter=${JSON.stringify(result.objectFilter)} ` +
        `objectFilterTarget=${JSON.stringify(result.objectFilterTarget)} ` +
        `editPixels=${result.editPixels} ` +
        `canvasSelectedObject=${result.canvasSelectedObject} ` +
        `sparkCanvasSelectedObject=${result.sparkCanvasSelectedObjectAfterDelete ?? "not-run"} ` +
        `sparkPick=${JSON.stringify(result.sparkPickModeAfterDelete ?? "")}:${JSON.stringify(result.sparkPickInteractionAfterDelete ?? "")}:${JSON.stringify(result.sparkPickStatusAfterDelete ?? "")}:${JSON.stringify(result.sparkPickObjectAfterDelete ?? "")}:${result.sparkPickDistancePxAfterDelete ?? 0}:${result.sparkPickCandidateObjectsAfterDelete ?? 0}:${JSON.stringify(result.sparkPickAmbiguousAfterDelete ?? "")}:${JSON.stringify(result.sparkPickHoverStatusAfterDelete ?? "")}:${JSON.stringify(result.sparkPickHoverObjectAfterDelete ?? "")}:${JSON.stringify(result.sparkPickHoverMarkerVisibleAfterDelete ?? "")}:${JSON.stringify(result.sparkSelectedMarkerVisibleAfterDelete ?? "")} ` +
        `visibleAfterIsolate=${result.visibleAfterIsolate} ` +
        `visibleAfterDelete=${result.visibleAfterDelete} ` +
        `renderModeAfterDelete=${JSON.stringify(result.renderModeAfterDelete)} ` +
        `postDeleteRoute=${JSON.stringify(result.routeAfterDeleteId)}:${JSON.stringify(result.routeAfterDeleteKind)}:${JSON.stringify(result.colorModeRoleAfterDelete)}:${JSON.stringify(result.sourcePreviewBoundaryAfterDelete)}:${JSON.stringify(result.sourcePreviewResultAfterDelete)}:${JSON.stringify(result.hardMaskQualityAfterDelete)} ` +
        `hardMaskQuality=${JSON.stringify(result.hardMaskQualityAfterDelete)}:${JSON.stringify(result.hardMaskQualitySourceAfterDelete)}:${result.hardMaskGapScoreAfterDelete}:${result.hardMaskResidualCoverageRatioAfterDelete}:${JSON.stringify(result.hardMaskDeletedObjectAfterDelete)} ` +
        `deletedObjects=${result.deletedObjects} ` +
        `postDelete=${JSON.stringify(result.postDeleteRendererId ?? "")}:${JSON.stringify(result.postDeleteObjectFilter ?? "")}:${result.sparkFilteredGaussiansAfterDelete ?? "unknown"} ` +
        `sparkMaskSource=${JSON.stringify(result.sparkMaskSourceAfterDelete ?? "")} ` +
        `sparkPacked=${JSON.stringify(result.sparkReconstructSourceAfterDelete ?? "")}:${result.sparkPackedBaseGaussiansAfterDelete ?? 0}/${result.sparkPackedVisibleIndicesAfterDelete ?? 0}:${result.sparkPackedBaseBuildMsAfterDelete ?? 0}/${result.sparkPackedExtractMsAfterDelete ?? 0} ` +
        `sparkDisplayCache=${JSON.stringify(result.sparkDisplayCacheModeAfterDelete ?? "")}:${JSON.stringify(result.sparkDisplayCacheHitAfterDelete ?? "")}:${result.sparkDisplayCacheSizeAfterDelete ?? 0}:${result.sparkDisplayCacheHitsAfterDelete ?? 0}/${result.sparkDisplayCacheMissesAfterDelete ?? 0}/${result.sparkDisplayCacheEvictionsAfterDelete ?? 0} ` +
        `sparkObjectMask=${JSON.stringify(result.sparkObjectMaskModeAfterDelete ?? "")}:${JSON.stringify(result.sparkObjectMaskSizeAfterDelete ?? "")}:${result.sparkObjectMaskVisibleGaussiansAfterDelete ?? 0}/${result.sparkObjectMaskHiddenGaussiansAfterDelete ?? 0}:${result.sparkObjectMaskUpdatesAfterDelete ?? 0} ` +
        `sparkObjectMaskFeather=${JSON.stringify(result.sparkObjectMaskFeatherModeAfterDelete ?? "")}:${result.sparkObjectMaskFeatheredGaussiansAfterDelete ?? 0}:${result.sparkObjectMaskFeatherRadiusAfterDelete ?? 0}:${result.sparkObjectMaskFeatherOpacityAfterDelete ?? 0}:${result.sparkObjectMaskOpacityMeanAfterDelete ?? 0}/${result.sparkObjectMaskMinOpacityAfterDelete ?? 0} ` +
        `sparkMaskVisual=${JSON.stringify(result.sparkObjectMaskVisualMode ?? "")}:${JSON.stringify(result.sparkObjectMaskVisualBeforeChecksum ?? "")}/${JSON.stringify(result.sparkObjectMaskVisualHiddenChecksum ?? "")}/${JSON.stringify(result.sparkObjectMaskVisualRestoredChecksum ?? "")}:${result.sparkObjectMaskVisualCoverageDelta ?? 0}/${result.sparkObjectMaskVisualLumaDelta ?? 0}/${result.sparkObjectMaskVisualChromaDelta ?? 0}:${result.sparkObjectMaskVisualRestoreCoverageDelta ?? 0}/${result.sparkObjectMaskVisualRestoreLumaDelta ?? 0}/${result.sparkObjectMaskVisualRestoreChromaDelta ?? 0} ` +
        `sparkMesh=${JSON.stringify(result.sparkMeshUpdateModeAfterDelete ?? "")}:${result.sparkMeshIdAfterDelete ?? 0}:${JSON.stringify(result.sparkMeshReusedAfterDelete ?? "")}:${result.sparkMeshUpdatesAfterDelete ?? 0} ` +
        `sparkShRest=${result.sparkShRestSourceGaussiansAfterDelete ?? 0}:${result.sparkShRestPreservedGaussiansAfterDelete ?? 0}:${JSON.stringify(result.sparkShRestPreservedAfterDelete ?? "")}:${result.sparkShRestCoefficientCountAfterDelete ?? 0}:${result.sparkShDegreeAfterDelete ?? 0} ` +
        `screenshot=${result.screenshotPath}`,
    );
  }
  console.log(
    `browser_audit=passed assets=${results.length} url=${baseUrl} ` +
      `requireWebGpu=${auditOptions.requireWebGpu} webGpuFlags=${JSON.stringify(auditOptions.webGpuFlags)} ` +
      `webGpuProbe=${JSON.stringify(auditOptions.webGpuProbe)} webGpuViewportSize=${auditOptions.webGpuViewportSize ?? "default"} ` +
      `webGpuObjectTransition=${auditOptions.webGpuObjectTransition} ` +
      `webGpuPresentationOnly=${auditOptions.webGpuPresentationOnly} ` +
      `sparkNativeMask=${auditOptions.sparkNativeMask} ` +
      `sparkObjectMaskFeather=${auditOptions.sparkObjectMaskFeather.enabled}:${auditOptions.sparkObjectMaskFeather.radius ?? "auto"}:${auditOptions.sparkObjectMaskFeather.opacity ?? "default"} ` +
      `skipVisualResidual=${auditOptions.skipVisualResidual} ` +
      `allowWebGpuDeviceLost=${auditOptions.allowWebGpuDeviceLost}`,
  );
} finally {
  if (server) {
    stopDevServer(server);
  }
}

async function runAudit(url, assetsToCheck, options) {
  const browser = await chromium.launch(launchOptions(options));
  const page = await browser.newPage({ viewport: { width: 1440, height: 920 } });
  const auditUrl = urlWithWebGpuOptions(url, options);
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  const results = [];
  try {
    for (const asset of assetsToCheck) {
      console.log(`browser_audit_asset_start asset=${JSON.stringify(asset.id)}`);
      await page.goto(auditUrl, { waitUntil: "networkidle" });
      const title = await page.title();
      if (title !== "ObjGauss 查看器") {
        throw new Error(`unexpected page title: ${title}`);
      }
      await expectText(page, "素材库");
      await expectNoFrameworkOverlay(page);
      const card = page.locator("article.assetCard").filter({ hasText: asset.name }).first();
      await card.getByRole("button", { name: "加载" }).click();
      await page.waitForFunction(
        (fileName) => document.body.innerText.includes(fileName),
        asset.fileName,
        { timeout: 15000 },
      );
      await page.waitForTimeout(1800);
      const splatPixels = await waitForNonBackgroundPixels(page);
      if (splatPixels <= 0) {
        throw new Error(`${asset.id} splat canvas appears blank: ${splatPixels}`);
      }
      const splatRendererId = await page.locator(".splatViewport").first().getAttribute("data-renderer");
      if (splatRendererId !== "spark-splat") {
        throw new Error(`${asset.id} did not expose Spark renderer id: ${splatRendererId}`);
      }
      const appShell = page.locator(".appShell").first();
      const initialRouteId = await appShell.getAttribute("data-renderer-route");
      const initialRouteKind = await appShell.getAttribute("data-renderer-route-kind");
      const initialColorModeRole = await appShell.getAttribute("data-color-mode-role");
      const initialSourcePreviewBoundary = await appShell.getAttribute("data-source-preview-boundary");
      const initialHardMaskQuality = await appShell.getAttribute("data-hard-mask-quality-interpretation");
      const initialHardMaskQualitySource = await appShell.getAttribute("data-hard-mask-quality-source");
      if (
        initialRouteKind !== "commercial" ||
        initialColorModeRole !== "source-color" ||
        initialSourcePreviewBoundary !== "source-splat" ||
        initialHardMaskQuality !== "source-splat" ||
        initialHardMaskQualitySource !== "route-state"
      ) {
        throw new Error(
          `${asset.id} initial route did not expose commercial source splat contract: route=${initialRouteId}:${initialRouteKind} color=${initialColorModeRole} boundary=${initialSourcePreviewBoundary} quality=${initialHardMaskQuality}:${initialHardMaskQualitySource}`,
        );
      }
      const screenshotOptions = { timeoutMs: 60000, usePageClip: true };
      const sparkVisualStats = options.skipVisualResidual
        ? null
        : await canvasVisualStats(page, ".splatViewport canvas", screenshotOptions);
      if (sparkVisualStats) {
        validateCanvasVisualStats(asset.id, "Spark", sparkVisualStats);
      }

      await page.locator(".modeTabs").getByRole("button", { name: "对象编辑" }).click();
      await waitForEditViewportReady(page);
      const editOriginalVisualStats = options.skipVisualResidual
        ? null
        : await canvasVisualStats(page, ".viewport canvas", screenshotOptions);
      if (editOriginalVisualStats) {
        validateCanvasVisualStats(asset.id, "edit original", editOriginalVisualStats);
      }
      const visualResidual =
        sparkVisualStats && editOriginalVisualStats
          ? compareVisualStats(sparkVisualStats, editOriginalVisualStats)
          : null;
      if (visualResidual) {
        validateVisualResidual(asset.id, visualResidual);
      }

      await page.getByLabel("渲染模式").selectOption("clustered");
      const objectColorRouteId = await appShell.getAttribute("data-renderer-route");
      const objectColorRouteKind = await appShell.getAttribute("data-renderer-route-kind");
      const objectColorModeRole = await appShell.getAttribute("data-color-mode-role");
      const objectColorSourcePreviewBoundary = await appShell.getAttribute("data-source-preview-boundary");
      const objectColorHardMaskQuality = await appShell.getAttribute("data-hard-mask-quality-interpretation");
      if (
        objectColorRouteId !== "diagnostic-object-color" ||
        objectColorRouteKind !== "diagnostic" ||
        objectColorModeRole !== "diagnostic-object-color" ||
        objectColorSourcePreviewBoundary !== "diagnostic-object-color" ||
        objectColorHardMaskQuality !== "diagnostic-object-color"
      ) {
        throw new Error(
          `${asset.id} object color mode did not expose diagnostic route: route=${objectColorRouteId}:${objectColorRouteKind} color=${objectColorModeRole} boundary=${objectColorSourcePreviewBoundary} quality=${objectColorHardMaskQuality}`,
        );
      }
      const editRenderer = await labeledValue(page, "渲染器");
      if (!["Gaussian OIT 编辑", "WebGPU Tile 编辑"].includes(editRenderer)) {
        throw new Error(`${asset.id} did not enter a known edit renderer: ${editRenderer}`);
      }
      await page.waitForFunction(() => {
        const viewport = document.querySelector(".viewport");
        return viewport?.getAttribute("data-webgpu-status") !== "pending";
      }, undefined, { timeout: 15000 });
      const viewport = page.locator(".viewport").first();
      const editRendererId = await viewport.getAttribute("data-renderer");
      if (!["gaussian-oit", "webgpu-tile"].includes(editRendererId ?? "")) {
        throw new Error(`${asset.id} did not expose a known edit renderer id: ${editRendererId}`);
      }
      if (editRendererId === "webgpu-tile") {
        await page.waitForFunction(() => {
          const activeViewport = document.querySelector(".viewport");
          return activeViewport?.getAttribute("data-webgpu-first-frame-status") !== "pending";
        }, undefined, { timeout: 15000 });
        await waitForWebGpuQueueTelemetry(page);
      }
      const rendererTarget = await viewport.getAttribute("data-renderer-target");
      if (rendererTarget !== "webgpu-tile") {
        throw new Error(`${asset.id} did not expose WebGPU tile target: ${rendererTarget}`);
      }
      const targetGate = await viewport.getAttribute("data-webgpu-target-gate");
      const targetGateReason = await viewport.getAttribute("data-webgpu-target-gate-reason");
      const targetGateBlocker = await viewport.getAttribute("data-webgpu-target-gate-blocker");
      const webgpuStatus = await viewport.getAttribute("data-webgpu-status");
      const storageLimitGate = await viewport.getAttribute("data-webgpu-storage-limit-gate");
      const storageLimitReason = await viewport.getAttribute("data-webgpu-storage-limit-reason");
      const storageLimitBlocker = await viewport.getAttribute("data-webgpu-storage-limit-blocker");
      const storageLimitMaxBufferSize = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-max-buffer-size") ?? "0");
      const storageLimitMaxBindingSize = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-max-binding-size") ?? "0");
      const storageLimitMaxStorageBuffersPerStage = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-max-storage-buffers-per-stage") ?? "0");
      const storageLimitRequiredStorageBuffersPerStage = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-required-storage-buffers-per-stage") ?? "0");
      const storageLimitEffectiveMaxBufferSize = numericValue(await viewport.getAttribute("data-webgpu-storage-limit-effective-max-buffer-size") ?? "0");
      const storageEstimatedLayout = await viewport.getAttribute("data-webgpu-storage-estimated-layout");
      const storageEstimatedBufferCount = numericValue(await viewport.getAttribute("data-webgpu-storage-estimated-buffer-count") ?? "0");
      const storageEstimatedByteSize = numericValue(await viewport.getAttribute("data-webgpu-storage-estimated-byte-size") ?? "0");
      const storageEstimatedMaxBufferByteSize = numericValue(await viewport.getAttribute("data-webgpu-storage-estimated-max-buffer-byte-size") ?? "0");
      const storageEstimatedMaxBufferKey = await viewport.getAttribute("data-webgpu-storage-estimated-max-buffer-key");
      if (!["available", "unavailable"].includes(webgpuStatus)) {
        throw new Error(`${asset.id} WebGPU capability status was not resolved: ${webgpuStatus}`);
      }
      const fallbackReason = await viewport.getAttribute("data-renderer-fallback-reason");
      if (editRendererId !== "webgpu-tile" && !fallbackReason) {
        throw new Error(`${asset.id} did not expose a renderer fallback reason`);
      }
      if (!["blocked", "pass"].includes(targetGate ?? "") || !targetGateReason) {
        throw new Error(
          `${asset.id} did not expose hardened WebGPU target gate: gate=${targetGate} reason=${targetGateReason} blocker=${targetGateBlocker}`,
        );
      }
      if (editRendererId === "webgpu-tile" && (targetGate !== "pass" || targetGateBlocker)) {
        throw new Error(
          `${asset.id} entered WebGPU renderer without a pass gate: gate=${targetGate} blocker=${targetGateBlocker}`,
        );
      }
      if (editRendererId !== "webgpu-tile" && (targetGate !== "blocked" || !targetGateBlocker)) {
        throw new Error(
          `${asset.id} fallback renderer did not expose blocked gate: gate=${targetGate} blocker=${targetGateBlocker}`,
        );
      }
      if (webgpuStatus !== "available" && targetGateBlocker !== "webgpu-capability") {
        throw new Error(
          `${asset.id} WebGPU unavailable but target gate was not capability-blocked: status=${webgpuStatus} blocker=${targetGateBlocker}`,
        );
      }
      if (options.requireWebGpu) {
        if (
          webgpuStatus !== "available" ||
          editRendererId !== "webgpu-tile" ||
          targetGate !== "pass" ||
          targetGateBlocker ||
          fallbackReason
        ) {
          throw new Error(
            `${asset.id} did not satisfy required WebGPU runtime route: status=${webgpuStatus} renderer=${editRendererId} gate=${targetGate} blocker=${targetGateBlocker} fallback=${fallbackReason} flags=${options.webGpuFlags}`,
          );
        }
      }
      if (
        !["unknown", "pass", "blocked"].includes(storageLimitGate ?? "") ||
        !storageLimitReason ||
        storageEstimatedLayout !== "webgpu-tile-storage-v1" ||
        storageEstimatedBufferCount < 11 ||
        storageEstimatedByteSize <= 0 ||
        storageEstimatedMaxBufferByteSize <= 0 ||
        !storageEstimatedMaxBufferKey ||
        storageLimitRequiredStorageBuffersPerStage <= 0
      ) {
        throw new Error(
          `${asset.id} invalid WebGPU storage limit telemetry: gate=${storageLimitGate} reason=${storageLimitReason} blocker=${storageLimitBlocker} layout=${storageEstimatedLayout} buffers=${storageEstimatedBufferCount} bytes=${storageEstimatedByteSize} max=${storageEstimatedMaxBufferKey}:${storageEstimatedMaxBufferByteSize} storageBuffersPerStage=${storageLimitRequiredStorageBuffersPerStage}/${storageLimitMaxStorageBuffersPerStage}`,
        );
      }
      if (webgpuStatus !== "available") {
        if (storageLimitGate !== "unknown" || storageLimitBlocker !== "webgpu-capability") {
          throw new Error(
            `${asset.id} WebGPU unavailable but storage limit gate was not capability-unknown: gate=${storageLimitGate} blocker=${storageLimitBlocker}`,
          );
        }
      } else if (storageLimitGate === "blocked") {
        if (
          targetGate !== "blocked" ||
          !["webgpu-buffer-limit", "webgpu-binding-limit"].includes(targetGateBlocker ?? "") ||
          fallbackReason !== targetGateBlocker
        ) {
          throw new Error(
            `${asset.id} storage limit blocked but target gate/fallback did not expose buffer-limit: gate=${targetGate} blocker=${targetGateBlocker} fallback=${fallbackReason}`,
          );
        }
      } else if (storageLimitGate !== "pass") {
        throw new Error(`${asset.id} unexpected WebGPU storage limit gate for available device: ${storageLimitGate}`);
      } else if (
        storageLimitMaxStorageBuffersPerStage > 0 &&
        storageLimitMaxStorageBuffersPerStage < storageLimitRequiredStorageBuffersPerStage
      ) {
        throw new Error(
          `${asset.id} WebGPU available but storage buffers per stage are below renderer requirement: ${storageLimitRequiredStorageBuffersPerStage}/${storageLimitMaxStorageBuffersPerStage}`,
        );
      }
      const tileSmokeLayout = await viewport.getAttribute("data-webgpu-pack-layout");
      if (tileSmokeLayout !== "webgpu-tile-smoke-v1") {
        throw new Error(`${asset.id} did not expose WebGPU tile smoke layout: ${tileSmokeLayout}`);
      }
      const webGpuViewportWidth = numericValue(await viewport.getAttribute("data-webgpu-viewport-width") ?? "0");
      const webGpuViewportHeight = numericValue(await viewport.getAttribute("data-webgpu-viewport-height") ?? "0");
      const webGpuPixelCount = numericValue(await viewport.getAttribute("data-webgpu-pixel-count") ?? "0");
      const webGpuViewportAspectMode = await viewport.getAttribute("data-webgpu-viewport-aspect-mode");
      const webGpuViewportQuality = await viewport.getAttribute("data-webgpu-viewport-quality");
      const webGpuViewportPixelBudget = numericValue(await viewport.getAttribute("data-webgpu-viewport-pixel-budget") ?? "0");
      const webGpuDisplayWidth = numericValue(await viewport.getAttribute("data-webgpu-display-width") ?? "0");
      const webGpuDisplayHeight = numericValue(await viewport.getAttribute("data-webgpu-display-height") ?? "0");
      const webGpuBoundsFitMode = await viewport.getAttribute("data-webgpu-bounds-fit-mode");
      const webGpuBoundsPaddingRatio = Number(await viewport.getAttribute("data-webgpu-bounds-padding-ratio") ?? "0");
      const webGpuBoundsViewportAspect = Number(await viewport.getAttribute("data-webgpu-bounds-viewport-aspect") ?? "0");
      const webGpuBoundsWorldAspect = Number(await viewport.getAttribute("data-webgpu-bounds-world-aspect") ?? "0");
      const webGpuProjectionMode = await viewport.getAttribute("data-webgpu-projection-mode");
      const webGpuProjectionCameraTuningMode = await viewport.getAttribute("data-webgpu-projection-camera-tuning-mode");
      const webGpuProjectionCameraMode = await viewport.getAttribute("data-webgpu-projection-camera-mode");
      const webGpuProjectionCameraFov = Number(await viewport.getAttribute("data-webgpu-projection-camera-fov") ?? "0");
      const webGpuProjectionCameraPosition = await viewport.getAttribute("data-webgpu-projection-camera-position");
      const webGpuProjectionCameraTarget = await viewport.getAttribute("data-webgpu-projection-camera-target");
      const webGpuProjectionCameraDistance = Number(await viewport.getAttribute("data-webgpu-projection-camera-distance") ?? "0");
      const webGpuProjectionCameraFrameMaxDim = Number(await viewport.getAttribute("data-webgpu-projection-camera-frame-max-dim") ?? "0");
      const webGpuDepthWeightMode = await viewport.getAttribute("data-webgpu-depth-weight-mode");
      const webGpuPixelDepthSortMode = await viewport.getAttribute("data-webgpu-pixel-depth-sort-mode");
      const webGpuPixelDepthTuningMode = await viewport.getAttribute("data-webgpu-pixel-depth-tuning-mode");
      const webGpuPixelDepthAlphaMode = await viewport.getAttribute("data-webgpu-pixel-depth-alpha-mode");
      const webGpuPixelDepthGateStrength = Number(await viewport.getAttribute("data-webgpu-pixel-depth-gate-strength") ?? "0");
      const webGpuPixelDepthGateFloor = Number(await viewport.getAttribute("data-webgpu-pixel-depth-gate-floor") ?? "0");
      const webGpuPixelDepthBinCount = Number(await viewport.getAttribute("data-webgpu-pixel-depth-bin-count") ?? "0");
      const webGpuPixelCoverageMode = await viewport.getAttribute("data-webgpu-pixel-coverage-mode");
      const webGpuPixelCoverageTuningMode = await viewport.getAttribute("data-webgpu-pixel-coverage-tuning-mode");
      const webGpuPixelCoverageWeightFloor = Number(await viewport.getAttribute("data-webgpu-pixel-coverage-weight-floor") ?? "0");
      const webGpuPixelCoverageFootprintScale = Number(await viewport.getAttribute("data-webgpu-pixel-coverage-footprint-scale") ?? "0");
      const webGpuProjectionDepthMin = Number(await viewport.getAttribute("data-webgpu-projection-depth-min") ?? "0");
      const webGpuProjectionDepthMax = Number(await viewport.getAttribute("data-webgpu-projection-depth-max") ?? "0");
      const webGpuProjectionDepthSpan = Number(await viewport.getAttribute("data-webgpu-projection-depth-span") ?? "0");
      const webGpuColorFidelityMode = await viewport.getAttribute("data-webgpu-color-fidelity-mode");
      const webGpuColorTuningMode = await viewport.getAttribute("data-webgpu-color-tuning-mode");
      const webGpuColorMode = await viewport.getAttribute("data-webgpu-color-mode");
      const webGpuColorSourceRgbGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-source-rgb-gaussians") ?? "0");
      const webGpuColorSourceShDcGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-source-sh-dc-gaussians") ?? "0");
      const webGpuColorSourceFallbackGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-source-fallback-gaussians") ?? "0");
      const webGpuColorSourceObjectGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-source-object-gaussians") ?? "0");
      const webGpuColorShRestGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-sh-rest-gaussians") ?? "0");
      const webGpuColorShRestCoefficientMax = numericValue(await viewport.getAttribute("data-webgpu-color-sh-rest-coefficient-max") ?? "0");
      const webGpuColorShDegreeMax = numericValue(await viewport.getAttribute("data-webgpu-color-sh-degree-max") ?? "0");
      const webGpuColorShViewGaussians = numericValue(await viewport.getAttribute("data-webgpu-color-sh-view-gaussians") ?? "0");
      const webGpuColorOpacityMean = Number(await viewport.getAttribute("data-webgpu-color-opacity-mean") ?? "0");
      const webGpuScreenCovarianceMode = await viewport.getAttribute("data-webgpu-screen-covariance-mode");
      const webGpuScreenCovarianceGaussians = numericValue(await viewport.getAttribute("data-webgpu-screen-covariance-gaussians") ?? "0");
      const webGpuScreenCovarianceFallbackGaussians = numericValue(await viewport.getAttribute("data-webgpu-screen-covariance-fallback-gaussians") ?? "0");
      const webGpuScreenCovarianceClampedGaussians = numericValue(await viewport.getAttribute("data-webgpu-screen-covariance-clamped-gaussians") ?? "0");
      const webGpuScreenCovarianceMaxAnisotropy = Number(await viewport.getAttribute("data-webgpu-screen-covariance-max-anisotropy") ?? "0");
      const webGpuScreenCovarianceSigmaMean = Number(await viewport.getAttribute("data-webgpu-screen-covariance-sigma-mean") ?? "0");
      const packedGaussians = numericValue(await viewport.getAttribute("data-webgpu-packed-gaussians") ?? "0");
      const binnedGaussians = numericValue(await viewport.getAttribute("data-webgpu-binned-gaussians") ?? "0");
      const visibleGaussians = numericValue(await viewport.getAttribute("data-webgpu-visible-gaussians") ?? "0");
      const tileListMode = await viewport.getAttribute("data-webgpu-tile-list-mode");
      const tileSize = numericValue(await viewport.getAttribute("data-webgpu-tile-size") ?? "0");
      const tileCount = numericValue(await viewport.getAttribute("data-webgpu-tile-count") ?? "0");
      const activeTileCount = numericValue(await viewport.getAttribute("data-webgpu-active-tile-count") ?? "0");
      const tileReferenceCount = numericValue(await viewport.getAttribute("data-webgpu-tile-reference-count") ?? "0");
      const tileOverflowCount = numericValue(await viewport.getAttribute("data-tile-overflow-count") ?? "0");
      const tileOverflowTileCount = numericValue(await viewport.getAttribute("data-webgpu-tile-overflow-tile-count") ?? "0");
      const tileOverflowRatio = Number(await viewport.getAttribute("data-webgpu-tile-overflow-ratio") ?? "0");
      const tileOverflowMaxExcess = numericValue(await viewport.getAttribute("data-webgpu-tile-overflow-max-excess") ?? "0");
      const tileEntryStoredCount = numericValue(await viewport.getAttribute("data-webgpu-tile-entry-stored-count") ?? "0");
      const tileEntryCapacity = numericValue(await viewport.getAttribute("data-webgpu-tile-entry-capacity") ?? "0");
      const tileEntryUtilization = Number(await viewport.getAttribute("data-webgpu-tile-entry-utilization") ?? "0");
      const tileEntryLayout = await viewport.getAttribute("data-webgpu-tile-entry-layout");
      const tileEntryOffsetCount = numericValue(await viewport.getAttribute("data-webgpu-tile-entry-offset-count") ?? "0");
      const tileCapacityMode = await viewport.getAttribute("data-webgpu-tile-capacity-mode");
      const tileCapacityStatus = await viewport.getAttribute("data-webgpu-tile-capacity-status");
      const tileCapacityGate = await viewport.getAttribute("data-webgpu-tile-capacity-gate");
      const maxTileOccupancy = numericValue(await viewport.getAttribute("data-webgpu-max-tile-occupancy") ?? "0");
      const resolveLayout = await viewport.getAttribute("data-webgpu-resolve-layout");
      const resolveMode = await viewport.getAttribute("data-webgpu-resolve-mode");
      const resolvedTileCount = numericValue(await viewport.getAttribute("data-webgpu-resolved-tile-count") ?? "0");
      const resolveWeightSum = Number(await viewport.getAttribute("data-webgpu-resolve-weight-sum") ?? "0");
      const resolveAlphaMean = Number(await viewport.getAttribute("data-webgpu-resolve-alpha-mean") ?? "0");
      const resolveLumaMean = Number(await viewport.getAttribute("data-webgpu-resolve-luma-mean") ?? "0");
      const resolveChecksum = await viewport.getAttribute("data-webgpu-resolve-checksum");
      if (tileSize !== 16) {
        throw new Error(`${asset.id} unexpected WebGPU tile size: ${tileSize}`);
      }
      if (editRendererId === "webgpu-tile") {
        if (tileListMode !== "object-state-filtered") {
          throw new Error(`${asset.id} WebGPU tile list mode was not object-state-filtered: ${tileListMode}`);
        }
        if (webGpuDisplayWidth <= 0 || webGpuDisplayHeight <= 0) {
          throw new Error(
            `${asset.id} WebGPU display size was not measured: ${webGpuDisplayWidth}x${webGpuDisplayHeight}`,
          );
        }
        if (options.webGpuViewportSize) {
          if (
            webGpuViewportWidth !== options.webGpuViewportSize ||
            webGpuViewportHeight !== options.webGpuViewportSize ||
            webGpuViewportAspectMode !== "explicit-square"
          ) {
            throw new Error(
              `${asset.id} WebGPU viewport did not honor requested size: requested=${options.webGpuViewportSize} actual=${webGpuViewportWidth}x${webGpuViewportHeight} mode=${webGpuViewportAspectMode}`,
            );
          }
        } else if (
          options.webGpuProbe === WEBGPU_RUNTIME_PROBE_FULL &&
          webGpuPixelCount <
            DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE *
              DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE
        ) {
          throw new Error(
            `${asset.id} WebGPU full runtime pixel budget is below visual audit floor: ${webGpuViewportWidth}x${webGpuViewportHeight}:${webGpuPixelCount}`,
          );
        }
        if (
          !options.webGpuViewportSize &&
          options.webGpuProbe === WEBGPU_RUNTIME_PROBE_FULL &&
          webGpuViewportAspectMode !== "display-aspect-adaptive"
        ) {
          throw new Error(
            `${asset.id} WebGPU full runtime viewport did not use adaptive display aspect matching: ${webGpuViewportAspectMode}`,
          );
        }
        if (
          !options.webGpuViewportSize &&
          options.webGpuProbe === WEBGPU_RUNTIME_PROBE_FULL &&
          (!String(webGpuViewportQuality ?? "").startsWith("adaptive-") ||
            webGpuViewportPixelBudget <
              DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE *
                DEFAULT_WEBGPU_VISUAL_AUDIT_MIN_VIEWPORT_SIZE)
        ) {
          throw new Error(
            `${asset.id} WebGPU full runtime did not expose an adaptive quality budget: quality=${webGpuViewportQuality} budget=${webGpuViewportPixelBudget}`,
          );
        }
        if (webGpuPixelCount !== webGpuViewportWidth * webGpuViewportHeight) {
          throw new Error(
            `${asset.id} WebGPU pixel count does not match viewport: ${webGpuPixelCount} vs ${webGpuViewportWidth}x${webGpuViewportHeight}`,
          );
        }
        if (
          webGpuBoundsFitMode !== "aspect-fit-padding" ||
          webGpuBoundsPaddingRatio < 0.079 ||
          Math.abs(webGpuBoundsWorldAspect - webGpuBoundsViewportAspect) > 0.02
        ) {
          throw new Error(
            `${asset.id} WebGPU bounds were not aspect-fit with padding: mode=${webGpuBoundsFitMode} padding=${webGpuBoundsPaddingRatio} worldAspect=${webGpuBoundsWorldAspect} viewportAspect=${webGpuBoundsViewportAspect}`,
          );
        }
        const expectedCameraMode = options.webGpuCameraTuning.cameraMode;
        const expectedProjection =
          expectedCameraMode === WEBGPU_CAMERA_MODE_SPARK_FRAME
            ? {
                mode: "spark-framed-perspective-camera-v1",
                fov: 58,
              }
            : {
                mode: "edit-perspective-camera-v1",
                fov: 52,
              };
        if (
          webGpuProjectionMode !== expectedProjection.mode ||
          webGpuProjectionCameraTuningMode !== WEBGPU_CAMERA_TUNING_MODE ||
          webGpuProjectionCameraMode !== expectedCameraMode ||
          Math.abs(webGpuProjectionCameraFov - expectedProjection.fov) > 0.001 ||
          webGpuProjectionCameraDistance <= 0 ||
          !validVectorAttribute(webGpuProjectionCameraPosition) ||
          !validVectorAttribute(webGpuProjectionCameraTarget)
        ) {
          throw new Error(
            `${asset.id} WebGPU projection did not use requested camera contract: expected=${expectedProjection.mode}/${expectedCameraMode}/${expectedProjection.fov} actual=${webGpuProjectionMode}/${webGpuProjectionCameraTuningMode}/${webGpuProjectionCameraMode}/${webGpuProjectionCameraFov} distance=${webGpuProjectionCameraDistance} position=${webGpuProjectionCameraPosition} target=${webGpuProjectionCameraTarget}`,
          );
        }
        if (
          expectedCameraMode === WEBGPU_CAMERA_MODE_SPARK_FRAME &&
          webGpuProjectionCameraFrameMaxDim <= 0
        ) {
          throw new Error(
            `${asset.id} WebGPU spark-frame camera did not expose a positive frame max dimension: ${webGpuProjectionCameraFrameMaxDim}`,
          );
        }
        if (
          ![
            "depth-binned-alpha-composite-v1",
            "front-top-k-alpha-composite-v1",
          ].includes(webGpuPixelDepthSortMode ?? "") ||
          webGpuDepthWeightMode !== "front-weighted-oit-v1" ||
          webGpuPixelDepthTuningMode !== WEBGPU_DEPTH_SORT_TUNING_MODE ||
          webGpuPixelDepthAlphaMode !== options.webGpuDepthAlphaMode ||
          (webGpuPixelDepthAlphaMode === WEBGPU_DEPTH_ALPHA_MODE_DEPTH_BINNED &&
            webGpuPixelDepthSortMode !== "depth-binned-alpha-composite-v1") ||
          (webGpuPixelDepthAlphaMode === WEBGPU_DEPTH_ALPHA_MODE_FRONT_TOP_K &&
            webGpuPixelDepthSortMode !== "front-top-k-alpha-composite-v1") ||
          webGpuPixelDepthGateStrength <= 1 ||
          webGpuPixelDepthGateFloor <= 0 ||
          webGpuPixelDepthGateFloor >= 1 ||
          webGpuPixelDepthBinCount !== normalizeWebGpuPixelDepthBinCount(webGpuPixelDepthBinCount) ||
          webGpuPixelCoverageMode !== "footprint-weight-floor-calibrated-v1" ||
          webGpuPixelCoverageTuningMode !== "runtime-coverage-tuning-v1" ||
          webGpuPixelCoverageWeightFloor < 0.003 ||
          webGpuPixelCoverageWeightFloor > 0.005 ||
          webGpuPixelCoverageFootprintScale < 1.2 ||
          webGpuPixelCoverageFootprintScale > 4.8 ||
          webGpuProjectionDepthSpan <= 0 ||
          webGpuProjectionDepthMax <= webGpuProjectionDepthMin
        ) {
          throw new Error(
            `${asset.id} WebGPU depth/coverage weighting did not expose a valid alpha contract: mode=${webGpuDepthWeightMode} pixelSort=${webGpuPixelDepthSortMode}:${webGpuPixelDepthTuningMode}:${webGpuPixelDepthAlphaMode} strength=${webGpuPixelDepthGateStrength} floor=${webGpuPixelDepthGateFloor} bins=${webGpuPixelDepthBinCount} coverage=${webGpuPixelCoverageMode}:${webGpuPixelCoverageTuningMode}:${webGpuPixelCoverageWeightFloor}:${webGpuPixelCoverageFootprintScale} min=${webGpuProjectionDepthMin} max=${webGpuProjectionDepthMax} span=${webGpuProjectionDepthSpan}`,
          );
        }
        if (
          Number.isFinite(options.webGpuDepthBins) &&
          webGpuPixelDepthBinCount !== normalizeWebGpuPixelDepthBinCount(options.webGpuDepthBins)
        ) {
          throw new Error(
            `${asset.id} WebGPU depth bins did not match requested tuning: requested=${options.webGpuDepthBins} actual=${webGpuPixelDepthBinCount}`,
          );
        }
        if (
          Number.isFinite(options.webGpuFootprintScale) &&
          Math.abs(webGpuPixelCoverageFootprintScale - clampNumber(options.webGpuFootprintScale, 1.2, 4.8)) > 0.000001
        ) {
          throw new Error(
            `${asset.id} WebGPU footprint scale did not match requested tuning: requested=${options.webGpuFootprintScale} actual=${webGpuPixelCoverageFootprintScale}`,
          );
        }
        if (
          webGpuColorTuningMode !== WEBGPU_COLOR_TUNING_MODE ||
          webGpuColorMode !== options.webGpuColorTuning.colorMode ||
          ![WEBGPU_COLOR_MODE_SOURCE, WEBGPU_COLOR_MODE_SH_VIEW].includes(webGpuColorMode ?? "") ||
          webGpuColorShViewGaussians < 0 ||
          webGpuColorShViewGaussians > packedGaussians ||
          (webGpuColorMode === WEBGPU_COLOR_MODE_SOURCE && webGpuColorShViewGaussians !== 0) ||
          (webGpuColorMode === WEBGPU_COLOR_MODE_SH_VIEW &&
            webGpuColorSourceObjectGaussians === 0 &&
            webGpuColorShRestGaussians > 0 &&
            webGpuColorShViewGaussians <= 0)
        ) {
          throw new Error(
            `${asset.id} WebGPU color tuning contract is invalid: requested=${options.webGpuColorTuning.colorMode} actual=${webGpuColorTuningMode}:${webGpuColorMode}:${webGpuColorShViewGaussians} shRest=${webGpuColorShRestGaussians}`,
          );
        }
        if (
          webGpuColorFidelityMode !== "source-color-fidelity-v1" ||
          webGpuColorSourceRgbGaussians +
            webGpuColorSourceShDcGaussians +
            webGpuColorSourceFallbackGaussians +
            webGpuColorSourceObjectGaussians !==
            packedGaussians ||
          webGpuColorShRestGaussians < 0 ||
          webGpuColorShRestGaussians > packedGaussians ||
          webGpuColorShRestCoefficientMax < 0 ||
          webGpuColorShDegreeMax < 0 ||
          webGpuColorShDegreeMax > 3 ||
          (webGpuColorShRestGaussians === 0 &&
            (webGpuColorShRestCoefficientMax !== 0 || webGpuColorShDegreeMax !== 0)) ||
          (webGpuColorShRestGaussians > 0 && webGpuColorShRestCoefficientMax <= 0) ||
          webGpuColorOpacityMean <= 0 ||
          webGpuColorOpacityMean > 1
        ) {
          throw new Error(
            `${asset.id} WebGPU color fidelity contract is invalid: mode=${webGpuColorFidelityMode} rgb=${webGpuColorSourceRgbGaussians} shDc=${webGpuColorSourceShDcGaussians} fallback=${webGpuColorSourceFallbackGaussians} object=${webGpuColorSourceObjectGaussians} shRest=${webGpuColorShRestGaussians}/${webGpuColorShRestCoefficientMax}/${webGpuColorShDegreeMax} packed=${packedGaussians} opacityMean=${webGpuColorOpacityMean}`,
          );
        }
        if (
          webGpuColorSourceObjectGaussians > 0 &&
          (webGpuColorSourceObjectGaussians !== packedGaussians ||
            webGpuColorSourceRgbGaussians !== 0 ||
            webGpuColorSourceShDcGaussians !== 0 ||
            webGpuColorSourceFallbackGaussians !== 0)
        ) {
          throw new Error(
            `${asset.id} WebGPU object-color mode mixed source colors unexpectedly: rgb=${webGpuColorSourceRgbGaussians} shDc=${webGpuColorSourceShDcGaussians} fallback=${webGpuColorSourceFallbackGaussians} object=${webGpuColorSourceObjectGaussians} packed=${packedGaussians}`,
          );
        }
        if (
          webGpuScreenCovarianceMode !== "camera-jacobian-covariance-v1" ||
          webGpuScreenCovarianceGaussians + webGpuScreenCovarianceFallbackGaussians !== packedGaussians ||
          webGpuScreenCovarianceClampedGaussians < 0 ||
          webGpuScreenCovarianceMaxAnisotropy < 1 ||
          webGpuScreenCovarianceSigmaMean <= 0
        ) {
          throw new Error(
            `${asset.id} WebGPU screen covariance contract is invalid: mode=${webGpuScreenCovarianceMode} full=${webGpuScreenCovarianceGaussians} fallback=${webGpuScreenCovarianceFallbackGaussians} clamped=${webGpuScreenCovarianceClampedGaussians} maxAnisotropy=${webGpuScreenCovarianceMaxAnisotropy} packed=${packedGaussians} sigmaMean=${webGpuScreenCovarianceSigmaMean}`,
          );
        }
        if (
          Number.isFinite(options.webGpuCovarianceMaxAnisotropy) &&
          Math.abs(webGpuScreenCovarianceMaxAnisotropy - clampNumber(options.webGpuCovarianceMaxAnisotropy, 1.5, 8)) > 0.000001
        ) {
          throw new Error(
            `${asset.id} WebGPU covariance max anisotropy did not match requested tuning: requested=${options.webGpuCovarianceMaxAnisotropy} actual=${webGpuScreenCovarianceMaxAnisotropy}`,
          );
        }
      }
      if (packedGaussians <= 0 || binnedGaussians <= 0 || visibleGaussians <= 0) {
        throw new Error(
          `${asset.id} did not expose positive WebGPU pack/bin counts: packed=${packedGaussians} visible=${visibleGaussians} binned=${binnedGaussians}`,
        );
      }
      if (tileCount <= 0 || activeTileCount <= 0 || tileReferenceCount < binnedGaussians) {
        throw new Error(
          `${asset.id} invalid WebGPU tile occupancy: tiles=${activeTileCount}/${tileCount} refs=${tileReferenceCount} binned=${binnedGaussians}`,
        );
      }
      if (maxTileOccupancy <= 0) {
        throw new Error(`${asset.id} did not expose positive max tile occupancy: ${maxTileOccupancy}`);
      }
      if (
        !["compact-offset-list", "fixed-cap-smoke"].includes(tileCapacityMode ?? "") ||
        !["compact-offset-list", "fixed-cap-smoke"].includes(tileEntryLayout ?? "") ||
        !["ok", "overflow"].includes(tileCapacityStatus ?? "") ||
        !["pass", "blocked"].includes(tileCapacityGate ?? "") ||
        tileEntryStoredCount <= 0 ||
        tileEntryCapacity <= 0 ||
        (tileEntryLayout === "compact-offset-list" && tileEntryStoredCount !== tileEntryCapacity) ||
        (tileEntryLayout === "compact-offset-list" && tileEntryOffsetCount !== 0 && tileEntryOffsetCount !== tileCount) ||
        tileEntryUtilization <= 0 ||
        tileEntryUtilization > 1
      ) {
        throw new Error(
          `${asset.id} invalid tile capacity telemetry: mode=${tileCapacityMode} layout=${tileEntryLayout} offsets=${tileEntryOffsetCount} status=${tileCapacityStatus} gate=${tileCapacityGate} stored=${tileEntryStoredCount} capacity=${tileEntryCapacity} utilization=${tileEntryUtilization}`,
        );
      }
      if (tileOverflowCount > 0) {
        if (
          tileCapacityStatus !== "overflow" ||
          tileCapacityGate !== "blocked" ||
          tileOverflowTileCount <= 0 ||
          tileOverflowRatio <= 0 ||
          tileOverflowMaxExcess <= 0
        ) {
          throw new Error(
            `${asset.id} overflow telemetry did not block tile capacity: overflow=${tileOverflowCount} overflowTiles=${tileOverflowTileCount} ratio=${tileOverflowRatio} maxExcess=${tileOverflowMaxExcess} status=${tileCapacityStatus} gate=${tileCapacityGate}`,
          );
        }
        if (webgpuStatus === "available" && targetGateBlocker !== "tile-overflow") {
          throw new Error(`${asset.id} WebGPU available with overflow but target gate blocker was ${targetGateBlocker}`);
        }
      } else if (
        tileCapacityStatus !== "ok" ||
        tileCapacityGate !== "pass" ||
        tileOverflowTileCount !== 0 ||
        tileOverflowRatio !== 0 ||
        tileOverflowMaxExcess !== 0
      ) {
        throw new Error(
          `${asset.id} non-overflow telemetry was not pass/ok: overflowTiles=${tileOverflowTileCount} ratio=${tileOverflowRatio} maxExcess=${tileOverflowMaxExcess} status=${tileCapacityStatus} gate=${tileCapacityGate}`,
        );
      }
      if (resolveLayout !== "webgpu-tile-resolve-v1" || resolveMode !== "tile-2x2-covariance-weighted-oit") {
        throw new Error(
          `${asset.id} did not expose WebGPU tile resolve contract: layout=${resolveLayout} mode=${resolveMode}`,
        );
      }
      if (
        resolvedTileCount <= 0 ||
        resolveWeightSum <= 0 ||
        resolveAlphaMean <= 0 ||
        resolveLumaMean <= 0 ||
        !/^[0-9a-f]{8}$/.test(resolveChecksum ?? "")
      ) {
        throw new Error(
          `${asset.id} invalid WebGPU tile resolve telemetry: resolved=${resolvedTileCount} weight=${resolveWeightSum} alpha=${resolveAlphaMean} luma=${resolveLumaMean} checksum=${resolveChecksum}`,
        );
      }
      const objectFilter = await viewport.getAttribute("data-object-filter");
      const expectedObjectFilter =
        editRendererId === "webgpu-tile" ? "gpu-object-state-buffer" : "gpu-object-state-texture";
      if (objectFilter !== expectedObjectFilter) {
        throw new Error(`${asset.id} did not expose expected object filtering: ${objectFilter}`);
      }
      const objectFilterTarget = await viewport.getAttribute("data-webgpu-object-filter-target");
      if (objectFilterTarget !== "gpu-object-state-buffer") {
        throw new Error(`${asset.id} did not expose WebGPU object-state buffer target: ${objectFilterTarget}`);
      }
      const objectStateLayout = await viewport.getAttribute("data-webgpu-object-state-layout");
      const objectStateStride = numericValue(await viewport.getAttribute("data-webgpu-object-state-stride") ?? "0");
      const objectStateVisibleObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-visible-objects") ?? "0");
      const objectStateHiddenObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-hidden-objects") ?? "0");
      const objectStateRemovedObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-removed-objects") ?? "0");
      const objectStateSelectedObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-selected-objects") ?? "0");
      const objectStateIsolatedObjects = numericValue(await viewport.getAttribute("data-webgpu-object-state-isolated-objects") ?? "0");
      const objectStateChecksum = await viewport.getAttribute("data-webgpu-object-state-checksum");
      if (
        objectStateLayout !== "webgpu-object-state-v1" ||
        objectStateStride !== 4 ||
        objectStateVisibleObjects <= 0 ||
        objectStateHiddenObjects !== 0 ||
        objectStateRemovedObjects !== 0 ||
        objectStateSelectedObjects !== 0 ||
        objectStateIsolatedObjects !== 0 ||
        !/^[0-9a-f]{8}$/.test(objectStateChecksum ?? "")
      ) {
        throw new Error(
          `${asset.id} invalid initial WebGPU object-state buffer: layout=${objectStateLayout} stride=${objectStateStride} visible=${objectStateVisibleObjects} hidden=${objectStateHiddenObjects} removed=${objectStateRemovedObjects} selected=${objectStateSelectedObjects} isolated=${objectStateIsolatedObjects} checksum=${objectStateChecksum}`,
        );
      }
      const webGpuFirstFrameStatus = await viewport.getAttribute("data-webgpu-first-frame-status");
      const webGpuFirstFrameReason = await viewport.getAttribute("data-webgpu-first-frame-reason");
      const webGpuFirstFramePixels = numericValue(await viewport.getAttribute("data-webgpu-first-frame-pixels") ?? "0");
      const webGpuFirstFrameChecksum = await viewport.getAttribute("data-webgpu-first-frame-checksum");
      const webGpuResolveSource = await viewport.getAttribute("data-webgpu-resolve-source");
      const webGpuResolveFilter = await viewport.getAttribute("data-webgpu-resolve-filter");
      const webGpuAlphaPresentationMode = await viewport.getAttribute("data-webgpu-alpha-presentation-mode");
      const webGpuAlphaPresentationTuningMode = await viewport.getAttribute("data-webgpu-alpha-presentation-tuning-mode");
      const webGpuAlphaPresentationFloor = Number(await viewport.getAttribute("data-webgpu-alpha-presentation-floor") ?? "0");
      const webGpuRuntimeProbe = await viewport.getAttribute("data-webgpu-runtime-probe");
      const webGpuDeviceLostStatus = await viewport.getAttribute("data-webgpu-device-lost-status");
      const webGpuDeviceLostReason = await viewport.getAttribute("data-webgpu-device-lost-reason");
      const webGpuDeviceLostMessage = await viewport.getAttribute("data-webgpu-device-lost-message");
      const webGpuDeviceErrorStatus = await viewport.getAttribute("data-webgpu-device-error-status");
      const webGpuDeviceErrorType = await viewport.getAttribute("data-webgpu-device-error-type");
      const webGpuDeviceErrorMessage = await viewport.getAttribute("data-webgpu-device-error-message");
      const webGpuQueueStatus = await viewport.getAttribute("data-webgpu-queue-status");
      const webGpuQueueReason = await viewport.getAttribute("data-webgpu-queue-reason");
      const webGpuQueueMessage = await viewport.getAttribute("data-webgpu-queue-message");
      const webGpuFrameSubmitMs = finiteNumericValue(await viewport.getAttribute("data-webgpu-frame-submit-ms") ?? "NaN");
      const webGpuQueueDoneMs = finiteNumericValue(await viewport.getAttribute("data-webgpu-queue-done-ms") ?? "NaN");
      const webGpuAccumulationSource = await viewport.getAttribute("data-webgpu-accumulation-source");
      const webGpuAccumulationStatus = await viewport.getAttribute("data-webgpu-accumulation-status");
      const webGpuAccumulationReason = await viewport.getAttribute("data-webgpu-accumulation-reason");
      const webGpuAccumulationWorkgroups = numericValue(await viewport.getAttribute("data-webgpu-accumulation-workgroups") ?? "0");
      const webGpuComputeSource = await viewport.getAttribute("data-webgpu-compute-source");
      const webGpuComputeStatus = await viewport.getAttribute("data-webgpu-compute-status");
      const webGpuComputeReason = await viewport.getAttribute("data-webgpu-compute-reason");
      const webGpuComputeWorkgroups = numericValue(await viewport.getAttribute("data-webgpu-compute-workgroups") ?? "0");
      const webGpuPixelSource = await viewport.getAttribute("data-webgpu-pixel-source");
      const webGpuPixelStatus = await viewport.getAttribute("data-webgpu-pixel-status");
      const webGpuPixelReason = await viewport.getAttribute("data-webgpu-pixel-reason");
      const webGpuPixelWorkgroups = numericValue(await viewport.getAttribute("data-webgpu-pixel-workgroups") ?? "0");
      const webGpuReadbackStatus = await viewport.getAttribute("data-webgpu-readback-status");
      const webGpuReadbackReason = await viewport.getAttribute("data-webgpu-readback-reason");
      const webGpuReadbackSource = await viewport.getAttribute("data-webgpu-readback-source");
      const webGpuReadbackChecksum = await viewport.getAttribute("data-webgpu-readback-checksum");
      const webGpuReadbackByteSize = numericValue(await viewport.getAttribute("data-webgpu-readback-byte-size") ?? "0");
      const webGpuReadbackFloatCount = numericValue(await viewport.getAttribute("data-webgpu-readback-float-count") ?? "0");
      const webGpuReadbackFiniteFloats = numericValue(await viewport.getAttribute("data-webgpu-readback-finite-floats") ?? "0");
      const webGpuReadbackNonzeroFloats = numericValue(await viewport.getAttribute("data-webgpu-readback-nonzero-floats") ?? "0");
      const webGpuStorageLayout = await viewport.getAttribute("data-webgpu-storage-layout");
      const webGpuStorageStatus = await viewport.getAttribute("data-webgpu-storage-status");
      const webGpuStorageReason = await viewport.getAttribute("data-webgpu-storage-reason");
      const webGpuStorageUpdateMode = await viewport.getAttribute("data-webgpu-storage-update-mode");
      const webGpuStorageUpdateMs = finiteNumericValue(await viewport.getAttribute("data-webgpu-storage-update-ms") ?? "NaN");
      const webGpuStorageBufferCount = numericValue(await viewport.getAttribute("data-webgpu-storage-buffer-count") ?? "0");
      const webGpuStorageByteSize = numericValue(await viewport.getAttribute("data-webgpu-storage-byte-size") ?? "0");
      const webGpuStorageObjectStateByteSize = numericValue(await viewport.getAttribute("data-webgpu-storage-object-state-byte-size") ?? "0");
      const webGpuStorageChecksum = await viewport.getAttribute("data-webgpu-storage-checksum");
      const webGpuStorageTileEntries = await viewport.getAttribute("data-webgpu-storage-tile-entries");
      const webGpuStorageTileOffsets = await viewport.getAttribute("data-webgpu-storage-tile-offsets");
      const webGpuStoragePixelOutput = await viewport.getAttribute("data-webgpu-storage-pixel-output");
      if (editRendererId === "webgpu-tile") {
        validateWebGpuRuntimeProbe({
          assetId: asset.id,
          expectedProbe: options.webGpuProbe,
          actualProbe: webGpuRuntimeProbe,
          webGpuAccumulationSource,
          webGpuAccumulationStatus,
          webGpuAccumulationReason,
          webGpuAccumulationWorkgroups,
          webGpuComputeSource,
          webGpuComputeStatus,
          webGpuComputeReason,
          webGpuComputeWorkgroups,
          webGpuPixelSource,
          webGpuPixelStatus,
          webGpuPixelReason,
          webGpuPixelWorkgroups,
          webGpuFirstFrameStatus,
          webGpuFirstFrameReason,
          webGpuFirstFramePixels,
          webGpuFirstFrameChecksum,
          webGpuResolveSource,
          webGpuResolveFilter,
          webGpuReadbackStatus,
          webGpuReadbackReason,
          webGpuReadbackSource,
          webGpuReadbackChecksum,
          webGpuReadbackByteSize,
          webGpuReadbackFloatCount,
          webGpuReadbackFiniteFloats,
          webGpuReadbackNonzeroFloats,
          webGpuAlphaPresentationMode,
          webGpuAlphaPresentationTuningMode,
          webGpuAlphaPresentationFloor,
          expectedAlphaPresentationFloor:
            options.webGpuAlphaPresentationTuning.alphaPresentationFloor,
        });
        if (webGpuDeviceLostStatus === "lost" && !options.allowWebGpuDeviceLost) {
          throw new Error(
            `${asset.id} WebGPU device was lost after first-frame submission: probe=${webGpuRuntimeProbe} reason=${webGpuDeviceLostReason} message=${webGpuDeviceLostMessage} deviceError=${webGpuDeviceErrorStatus}:${webGpuDeviceErrorType}:${webGpuDeviceErrorMessage} queue=${webGpuQueueStatus}:${webGpuQueueReason}:${webGpuQueueMessage}`,
          );
        }
        if (
          webGpuStorageLayout !== "webgpu-tile-storage-v1" ||
          !["uploaded", "object-state-updated"].includes(webGpuStorageStatus ?? "") ||
          !["full-upload", "object-state-only"].includes(webGpuStorageUpdateMode ?? "") ||
          !Number.isFinite(webGpuStorageUpdateMs) ||
          webGpuStorageUpdateMs < 0 ||
          !Number.isFinite(webGpuFrameSubmitMs) ||
          webGpuFrameSubmitMs < 0 ||
          ((webGpuQueueStatus === "done" || webGpuQueueStatus === "failed") &&
            (!Number.isFinite(webGpuQueueDoneMs) || webGpuQueueDoneMs < 0)) ||
          webGpuStorageBufferCount < 11 ||
          webGpuStorageByteSize <= 0 ||
          webGpuStorageObjectStateByteSize <= 0 ||
          webGpuStorageTileEntries !== "true" ||
          webGpuStorageTileOffsets !== "true" ||
          webGpuStoragePixelOutput !== "true" ||
          !/^[0-9a-f]{8}$/.test(webGpuStorageChecksum ?? "")
        ) {
          throw new Error(
            `${asset.id} WebGPU storage buffers were not uploaded with tile entries, offsets, pixel output, and timing: layout=${webGpuStorageLayout} status=${webGpuStorageStatus} reason=${webGpuStorageReason} updateMode=${webGpuStorageUpdateMode} updateMs=${webGpuStorageUpdateMs} submitMs=${webGpuFrameSubmitMs} queueDoneMs=${webGpuQueueDoneMs} queue=${webGpuQueueStatus}:${webGpuQueueReason} buffers=${webGpuStorageBufferCount} bytes=${webGpuStorageByteSize} objectStateBytes=${webGpuStorageObjectStateByteSize} tileEntries=${webGpuStorageTileEntries} tileOffsets=${webGpuStorageTileOffsets} pixelOutput=${webGpuStoragePixelOutput} checksum=${webGpuStorageChecksum}`,
          );
        }
        if (
          (options.webGpuProbe !== WEBGPU_RUNTIME_PROBE_FULL &&
            !options.webGpuObjectTransition) ||
          options.webGpuPresentationOnly
        ) {
          const screenshotPath = options.webGpuPresentationOnly
            ? `/tmp/objgauss-audit-${asset.id}-webgpu-presentation.png`
            : `/tmp/objgauss-audit-${asset.id}-${options.webGpuProbe}.png`;
          await page.screenshot({ path: screenshotPath, fullPage: false });
          results.push({
            assetId: asset.id,
            title,
            splatPixels,
            splatRendererId,
            ...visualResidualResultFields(sparkVisualStats, editOriginalVisualStats, visualResidual),
            editPixels: webGpuFirstFramePixels,
            editRenderer,
            editRendererId,
            webGpuFirstFrameStatus,
            webGpuFirstFramePixels,
            webGpuFirstFrameChecksum,
            webGpuResolveSource,
            webGpuResolveFilter,
            webGpuAlphaPresentationMode,
            webGpuAlphaPresentationTuningMode,
            webGpuAlphaPresentationFloor,
            webGpuRuntimeProbe,
            webGpuViewportWidth,
            webGpuViewportHeight,
            webGpuPixelCount,
            webGpuViewportAspectMode,
            webGpuViewportQuality,
            webGpuViewportPixelBudget,
            webGpuDisplayWidth,
            webGpuDisplayHeight,
            webGpuBoundsFitMode,
            webGpuBoundsPaddingRatio,
            webGpuBoundsViewportAspect,
            webGpuBoundsWorldAspect,
            webGpuProjectionMode,
            webGpuProjectionCameraTuningMode,
            webGpuProjectionCameraMode,
            webGpuProjectionCameraFov,
            webGpuProjectionCameraPosition,
            webGpuProjectionCameraTarget,
            webGpuProjectionCameraDistance,
            webGpuProjectionCameraFrameMaxDim,
            webGpuDepthWeightMode,
            webGpuPixelDepthSortMode,
            webGpuPixelDepthTuningMode,
            webGpuPixelDepthAlphaMode,
            webGpuPixelDepthGateStrength,
            webGpuPixelDepthGateFloor,
            webGpuPixelDepthBinCount,
            webGpuPixelCoverageMode,
            webGpuPixelCoverageTuningMode,
            webGpuPixelCoverageWeightFloor,
            webGpuPixelCoverageFootprintScale,
            webGpuProjectionDepthMin,
            webGpuProjectionDepthMax,
            webGpuProjectionDepthSpan,
            webGpuColorFidelityMode,
            webGpuColorSourceRgbGaussians,
            webGpuColorSourceShDcGaussians,
            webGpuColorSourceFallbackGaussians,
            webGpuColorSourceObjectGaussians,
            webGpuColorTuningMode,
            webGpuColorMode,
            webGpuColorShRestGaussians,
            webGpuColorShRestCoefficientMax,
            webGpuColorShDegreeMax,
            webGpuColorShViewGaussians,
            webGpuColorOpacityMean,
            webGpuColorSourceRgbGaussiansAfterDelete: "probe-skipped",
            webGpuColorSourceShDcGaussiansAfterDelete: "probe-skipped",
            webGpuColorSourceFallbackGaussiansAfterDelete: "probe-skipped",
            webGpuColorSourceObjectGaussiansAfterDelete: "probe-skipped",
            webGpuColorShViewGaussiansAfterDelete: "probe-skipped",
            webGpuScreenCovarianceMode,
            webGpuScreenCovarianceGaussians,
            webGpuScreenCovarianceFallbackGaussians,
            webGpuScreenCovarianceClampedGaussians,
            webGpuScreenCovarianceMaxAnisotropy,
            webGpuScreenCovarianceSigmaMean,
            webGpuDeviceLostStatus,
            webGpuDeviceLostReason,
            webGpuDeviceLostMessage,
            webGpuDeviceErrorStatus,
            webGpuDeviceErrorType,
            webGpuDeviceErrorMessage,
            webGpuQueueStatus,
            webGpuQueueReason,
            webGpuQueueMessage,
            webGpuAccumulationSource,
            webGpuAccumulationStatus,
            webGpuAccumulationWorkgroups,
            webGpuComputeSource,
            webGpuComputeStatus,
            webGpuComputeWorkgroups,
            webGpuPixelSource,
            webGpuPixelStatus,
            webGpuPixelWorkgroups,
            webGpuReadbackStatus,
            webGpuReadbackReason,
            webGpuReadbackSource,
            webGpuReadbackChecksum,
            webGpuReadbackByteSize,
            webGpuReadbackFloatCount,
            webGpuReadbackFiniteFloats,
            webGpuReadbackNonzeroFloats,
            webGpuReadbackStatusAfterIsolate: "probe-skipped",
            webGpuReadbackSourceAfterIsolate: "",
            webGpuReadbackChecksumAfterIsolate: "",
            webGpuReadbackByteSizeAfterIsolate: 0,
            webGpuReadbackFloatCountAfterIsolate: 0,
            webGpuReadbackFiniteFloatsAfterIsolate: 0,
            webGpuReadbackNonzeroFloatsAfterIsolate: 0,
            webGpuReadbackStatusAfterDelete: "probe-skipped",
            webGpuReadbackSourceAfterDelete: "",
            webGpuReadbackChecksumAfterDelete: "",
            webGpuReadbackByteSizeAfterDelete: 0,
            webGpuReadbackFloatCountAfterDelete: 0,
            webGpuReadbackFiniteFloatsAfterDelete: 0,
            webGpuReadbackNonzeroFloatsAfterDelete: 0,
            webGpuStorageLayout,
            webGpuStorageStatus,
            webGpuStorageBufferCount,
            webGpuStorageByteSize,
            webGpuStorageTileEntries,
            webGpuStorageTileOffsets,
            webGpuStoragePixelOutput,
            webGpuStorageChecksum,
            webGpuStorageUpdateMode,
            webGpuStorageUpdateMs,
            webGpuFrameSubmitMs,
            webGpuQueueDoneMs,
            webGpuStorageObjectStateByteSize,
            webGpuStorageChecksumAfterIsolate: webGpuStorageChecksum,
            webGpuStorageChecksumAfterDelete: webGpuStorageChecksum,
            storageLimitGate,
            storageLimitReason,
            storageLimitBlocker,
            storageLimitMaxBufferSize,
            storageLimitMaxBindingSize,
            storageLimitMaxStorageBuffersPerStage,
            storageLimitRequiredStorageBuffersPerStage,
            storageLimitEffectiveMaxBufferSize,
            storageEstimatedLayout,
            storageEstimatedBufferCount,
            storageEstimatedByteSize,
            storageEstimatedMaxBufferByteSize,
            storageEstimatedMaxBufferKey,
            rendererTarget,
            targetGate,
            targetGateReason,
            targetGateBlocker,
            webgpuStatus,
            fallbackReason,
            tileSmokeLayout,
            packedGaussians,
            visibleGaussians,
            binnedGaussians,
            tileSize,
            tileCount,
            activeTileCount,
            tileReferenceCount,
            maxTileOccupancy,
            tileOverflowTileCount,
            tileOverflowRatio,
            tileOverflowMaxExcess,
            tileEntryStoredCount,
            tileEntryCapacity,
            tileEntryUtilization,
            tileEntryLayout,
            tileEntryOffsetCount,
            tileCapacityMode,
            tileCapacityStatus,
            tileCapacityGate,
            resolveLayout,
            resolveMode,
            resolvedTileCount,
            resolveWeightSum,
            resolveAlphaMean,
            resolveLumaMean,
            resolveChecksum,
            tileOverflowCount,
            objectFilter,
            objectFilterTarget,
            objectStateLayout,
            objectStateStride,
            objectStateVisibleObjects,
            objectStateHiddenObjects,
            objectStateRemovedObjects,
            objectStateSelectedObjects,
            objectStateIsolatedObjects,
            objectStateChecksum,
            objectStateChecksumAfterIsolate: objectStateChecksum,
            objectStateChecksumAfterDelete: objectStateChecksum,
            canvasSelectedObject: "probe-skipped",
            sparkCanvasSelectedObjectAfterDelete: "probe-skipped",
            ...emptySparkPickResultFields("probe-skipped"),
            visibleAfterIsolate: "probe-skipped",
            visibleAfterDelete: "probe-skipped",
            renderModeAfterDelete: "probe-skipped",
            deletedObjects: "probe-skipped",
            postDeleteRendererId: "probe-skipped",
            postDeleteObjectFilter: "probe-skipped",
            sparkFilteredGaussiansAfterDelete: "probe-skipped",
            screenshotPath,
          });
          continue;
        }
      }
      const editPixels =
        editRendererId === "webgpu-tile"
          ? webGpuFirstFramePixels
          : await waitForNonBackgroundPixels(page);
      if (editPixels <= 0) {
        throw new Error(`${asset.id} point-edit canvas appears blank: ${editPixels}`);
      }
      const canvasSelectedObject = await selectObjectFromCanvas(page, asset.id);
      await page.getByRole("button", { name: "只看所选" }).click();
      await page.waitForTimeout(300);
      if (editRendererId === "webgpu-tile") {
        await waitForWebGpuStorageUpdate(page, {
          assetId: asset.id,
          label: "isolate",
          previousChecksum: webGpuStorageChecksum,
        });
      }
      const visibleAfterIsolate = await labeledValue(page, "可见");
      const objectStateChecksumAfterIsolate = await viewport.getAttribute("data-webgpu-object-state-checksum");
      const objectStateVisibleAfterIsolate = numericValue(await viewport.getAttribute("data-webgpu-object-state-visible-objects") ?? "0");
      const objectStateHiddenAfterIsolate = numericValue(await viewport.getAttribute("data-webgpu-object-state-hidden-objects") ?? "0");
      const objectStateSelectedAfterIsolate = numericValue(await viewport.getAttribute("data-webgpu-object-state-selected-objects") ?? "0");
      const objectStateIsolatedAfterIsolate = numericValue(await viewport.getAttribute("data-webgpu-object-state-isolated-objects") ?? "0");
      const webGpuStorageStatusAfterIsolate = await viewport.getAttribute("data-webgpu-storage-status");
      const webGpuStorageUpdateModeAfterIsolate = await viewport.getAttribute("data-webgpu-storage-update-mode");
      const webGpuStorageUpdateMsAfterIsolate = finiteNumericValue(await viewport.getAttribute("data-webgpu-storage-update-ms") ?? "NaN");
      const webGpuFrameSubmitMsAfterIsolate = finiteNumericValue(await viewport.getAttribute("data-webgpu-frame-submit-ms") ?? "NaN");
      const webGpuQueueDoneMsAfterIsolate = finiteNumericValue(await viewport.getAttribute("data-webgpu-queue-done-ms") ?? "NaN");
      const webGpuStorageObjectStateByteSizeAfterIsolate = numericValue(await viewport.getAttribute("data-webgpu-storage-object-state-byte-size") ?? "0");
      const webGpuStorageChecksumAfterIsolate = await viewport.getAttribute("data-webgpu-storage-checksum");
      const webGpuReadbackAfterIsolate =
        editRendererId === "webgpu-tile" &&
        options.webGpuProbe === WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK
          ? await waitForWebGpuReadbackTransition(page, {
              assetId: asset.id,
              label: "isolate",
              previousChecksum: webGpuReadbackChecksum,
            })
          : emptyWebGpuReadbackTelemetry();
      if (
        objectStateChecksumAfterIsolate === objectStateChecksum ||
        objectStateVisibleAfterIsolate !== 1 ||
        objectStateHiddenAfterIsolate <= 0 ||
        objectStateSelectedAfterIsolate !== 1 ||
        objectStateIsolatedAfterIsolate !== 1
      ) {
        throw new Error(
          `${asset.id} isolate did not update WebGPU object-state buffer: checksum=${objectStateChecksumAfterIsolate} visible=${objectStateVisibleAfterIsolate} hidden=${objectStateHiddenAfterIsolate} selected=${objectStateSelectedAfterIsolate} isolated=${objectStateIsolatedAfterIsolate}`,
        );
      }
      if (editRendererId === "webgpu-tile" && webGpuStorageChecksumAfterIsolate === webGpuStorageChecksum) {
        throw new Error(`${asset.id} isolate did not update WebGPU storage checksum`);
      }
      if (
        editRendererId === "webgpu-tile" &&
        (webGpuStorageStatusAfterIsolate !== "object-state-updated" ||
          webGpuStorageUpdateModeAfterIsolate !== "object-state-only" ||
          !Number.isFinite(webGpuStorageUpdateMsAfterIsolate) ||
          webGpuStorageUpdateMsAfterIsolate < 0 ||
          !Number.isFinite(webGpuFrameSubmitMsAfterIsolate) ||
          webGpuFrameSubmitMsAfterIsolate < 0 ||
          !Number.isFinite(webGpuQueueDoneMsAfterIsolate) ||
          webGpuQueueDoneMsAfterIsolate < 0 ||
          webGpuStorageObjectStateByteSizeAfterIsolate <= 0)
      ) {
        throw new Error(
          `${asset.id} isolate did not use WebGPU objectState-only storage update with timing: status=${webGpuStorageStatusAfterIsolate} updateMode=${webGpuStorageUpdateModeAfterIsolate} updateMs=${webGpuStorageUpdateMsAfterIsolate} submitMs=${webGpuFrameSubmitMsAfterIsolate} queueDoneMs=${webGpuQueueDoneMsAfterIsolate} objectStateBytes=${webGpuStorageObjectStateByteSizeAfterIsolate}`,
        );
      }
      if (
        options.webGpuObjectTransition &&
        options.webGpuProbe === WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK &&
        editRendererId === "webgpu-tile" &&
        webGpuReadbackAfterIsolate.checksum === webGpuReadbackChecksum
      ) {
        throw new Error(
          `${asset.id} isolate did not update WebGPU offscreen readback checksum: ${webGpuReadbackChecksum}`,
        );
      }
      await page.getByRole("button", { name: "预览删除" }).click();
      await page.waitForTimeout(300);
      await page.waitForFunction(() => {
        const activeViewport = document.querySelector(".viewport");
        if (!activeViewport) return false;
        if (activeViewport.getAttribute("data-renderer") !== "spark-splat") return true;
        return activeViewport.getAttribute("data-spark-filter-status") === "ready";
      }, undefined, { timeout: 15000 });
      if (editRendererId === "webgpu-tile") {
        const activeRendererAfterDelete = await page.locator(".viewport").first().getAttribute("data-renderer");
        if (activeRendererAfterDelete === "webgpu-tile") {
          await waitForWebGpuStorageUpdate(page, {
            assetId: asset.id,
            label: "delete",
            previousChecksum: webGpuStorageChecksumAfterIsolate,
            allowFullUpload: true,
          });
        }
      }
      const deletedObjects = await labeledValue(page, "已删除对象");
      const visibleAfterDelete = await labeledValue(page, "可见");
      const renderModeAfterDelete = await labeledValue(page, "模式");
      const routeAfterDeleteId = await appShell.getAttribute("data-renderer-route");
      const routeAfterDeleteKind = await appShell.getAttribute("data-renderer-route-kind");
      const colorModeRoleAfterDelete = await appShell.getAttribute("data-color-mode-role");
      const sourcePreviewBoundaryAfterDelete = await appShell.getAttribute("data-source-preview-boundary");
      const sourcePreviewResultAfterDelete = await appShell.getAttribute("data-source-preview-result");
      const hardMaskQualityAfterDelete = await appShell.getAttribute("data-hard-mask-quality-interpretation");
      const hardMaskQualitySourceAfterDelete = await appShell.getAttribute("data-hard-mask-quality-source");
      const hardMaskGapScoreAfterDelete = finiteNumericValue(
        await appShell.getAttribute("data-hard-mask-gap-score") ?? "0",
      );
      const hardMaskResidualCoverageRatioAfterDelete = finiteNumericValue(
        await appShell.getAttribute("data-hard-mask-residual-coverage-ratio") ?? "0",
      );
      const hardMaskDeletedObjectAfterDelete = await appShell.getAttribute("data-hard-mask-deleted-object");
      const postDeleteViewport = page.locator(".viewport").first();
      const postDeleteRendererId = await postDeleteViewport.getAttribute("data-renderer");
      const postDeleteObjectFilter = await postDeleteViewport.getAttribute("data-object-filter");
      const sparkFilteredAfterDelete =
        postDeleteRendererId === "spark-splat" &&
        postDeleteObjectFilter === SPARK_OBJECT_FILTER_MASK;
      const sparkMaskSourceAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-mask-source")
        : "";
      const sparkVisibleGaussiansAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-visible-gaussians") ?? "0")
        : 0;
      const sparkRemovedObjectsAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-removed-objects") ?? "0")
        : 0;
      const sparkColorModeAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-color-mode")
        : "";
      const sparkColorSourceGaussiansAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-color-source-gaussians") ?? "0")
        : 0;
      const sparkColorObjectGaussiansAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-color-object-gaussians") ?? "0")
        : 0;
      const sparkReconstructSourceAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-reconstruct-source")
        : "";
      const sparkPackedBaseGaussiansAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-packed-base-gaussians") ?? "0")
        : 0;
      const sparkPackedVisibleIndicesAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-packed-visible-indices") ?? "0")
        : 0;
      const sparkPackedBaseBuildMsAfterDelete = sparkFilteredAfterDelete
        ? finiteNumericValue(await postDeleteViewport.getAttribute("data-spark-packed-base-build-ms") ?? "0")
        : 0;
      const sparkPackedExtractMsAfterDelete = sparkFilteredAfterDelete
        ? finiteNumericValue(await postDeleteViewport.getAttribute("data-spark-packed-extract-ms") ?? "0")
        : 0;
      let sparkDisplayCacheModeAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-display-cache-mode")
        : "";
      let sparkDisplayCacheKeyAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-display-cache-key")
        : "";
      let sparkDisplayCacheHitAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-display-cache-hit")
        : "";
      let sparkDisplayCacheSizeAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-display-cache-size") ?? "0")
        : 0;
      let sparkDisplayCacheHitsAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-display-cache-hits") ?? "0")
        : 0;
      let sparkDisplayCacheMissesAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-display-cache-misses") ?? "0")
        : 0;
      let sparkDisplayCacheEvictionsAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-display-cache-evictions") ?? "0")
        : 0;
      let sparkObjectMaskModeAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-object-mask-mode")
        : "";
      let sparkObjectMaskSizeAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-object-mask-size")
        : "";
      let sparkObjectMaskUpdatesAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-object-mask-updates") ?? "0")
        : 0;
      let sparkObjectMaskVisibleGaussiansAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-object-mask-visible-gaussians") ?? "0")
        : 0;
      let sparkObjectMaskHiddenGaussiansAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-object-mask-hidden-gaussians") ?? "0")
        : 0;
      let sparkObjectMaskFeatherModeAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-object-mask-feather-mode")
        : "";
      let sparkObjectMaskFeatherRadiusAfterDelete = sparkFilteredAfterDelete
        ? finiteNumericValue(await postDeleteViewport.getAttribute("data-spark-object-mask-feather-radius") ?? "0")
        : 0;
      let sparkObjectMaskFeatherOpacityAfterDelete = sparkFilteredAfterDelete
        ? finiteNumericValue(await postDeleteViewport.getAttribute("data-spark-object-mask-feather-opacity") ?? "0")
        : 0;
      let sparkObjectMaskFeatheredGaussiansAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-object-mask-feathered-gaussians") ?? "0")
        : 0;
      let sparkObjectMaskOpacityMeanAfterDelete = sparkFilteredAfterDelete
        ? finiteNumericValue(await postDeleteViewport.getAttribute("data-spark-object-mask-opacity-mean") ?? "0")
        : 0;
      let sparkObjectMaskMinOpacityAfterDelete = sparkFilteredAfterDelete
        ? finiteNumericValue(await postDeleteViewport.getAttribute("data-spark-object-mask-min-opacity") ?? "0")
        : 0;
      let sparkMeshUpdateModeAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-mesh-update-mode")
        : "";
      let sparkMeshIdAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-mesh-id") ?? "0")
        : 0;
      let sparkMeshReusedAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-mesh-reused")
        : "";
      let sparkMeshUpdatesAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-mesh-updates") ?? "0")
        : 0;
      let sparkCanvasSelectedObjectAfterDelete = "not-run";
      let sparkPickStatsAfterDelete = emptySparkPickStats();
      let sparkObjectMaskVisualDelta = emptySparkObjectMaskVisualDelta();
      const sparkShRestSourceGaussiansAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-sh-rest-source-gaussians") ?? "0")
        : 0;
      const sparkShRestPreservedGaussiansAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-sh-rest-preserved-gaussians") ?? "0")
        : 0;
      const sparkShRestPreservedAfterDelete = sparkFilteredAfterDelete
        ? await postDeleteViewport.getAttribute("data-spark-sh-rest-preserved")
        : "";
      const sparkShRestCoefficientCountAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-sh-rest-coefficients") ?? "0")
        : 0;
      const sparkShDegreeAfterDelete = sparkFilteredAfterDelete
        ? numericValue(await postDeleteViewport.getAttribute("data-spark-sh-degree") ?? "0")
        : 0;
      const sparkExpectedMaskSourceAfterDelete = options.sparkNativeMask
        ? "native-splat"
        : sparkShRestSourceGaussiansAfterDelete > 0
        ? "ply-packed"
        : "native-splat";
      const sparkExpectedReconstructSourceAfterDelete =
        sparkExpectedMaskSourceAfterDelete === "native-splat"
          ? SPARK_NATIVE_RECONSTRUCT_SOURCE
          : sparkShRestSourceGaussiansAfterDelete > 0
          ? SPARK_RECONSTRUCT_SH_SOURCE
          : SPARK_RECONSTRUCT_SOURCE;
      const sparkExpectedShRestPreservedAfterDelete =
        sparkShRestSourceGaussiansAfterDelete > 0 ? "true" : "false";
      const objectStateChecksumAfterDelete = sparkFilteredAfterDelete
        ? SPARK_OBJECT_FILTER_MASK
        : await viewport.getAttribute("data-webgpu-object-state-checksum");
      const objectStateVisibleAfterDelete = sparkFilteredAfterDelete
        ? objectStateVisibleObjects - sparkRemovedObjectsAfterDelete
        : numericValue(await viewport.getAttribute("data-webgpu-object-state-visible-objects") ?? "0");
      const objectStateRemovedAfterDelete = sparkFilteredAfterDelete
        ? sparkRemovedObjectsAfterDelete
        : numericValue(await viewport.getAttribute("data-webgpu-object-state-removed-objects") ?? "0");
      const objectStateIsolatedAfterDelete = sparkFilteredAfterDelete
        ? 0
        : numericValue(await viewport.getAttribute("data-webgpu-object-state-isolated-objects") ?? "0");
      const webGpuStorageStatusAfterDelete = sparkFilteredAfterDelete
        ? "spark-filtered"
        : await viewport.getAttribute("data-webgpu-storage-status");
      const webGpuStorageUpdateModeAfterDelete = sparkFilteredAfterDelete
        ? "spark-filtered"
        : await viewport.getAttribute("data-webgpu-storage-update-mode");
      const webGpuStorageUpdateMsAfterDelete = sparkFilteredAfterDelete
        ? 0
        : finiteNumericValue(await viewport.getAttribute("data-webgpu-storage-update-ms") ?? "NaN");
      const webGpuFrameSubmitMsAfterDelete = sparkFilteredAfterDelete
        ? 0
        : finiteNumericValue(await viewport.getAttribute("data-webgpu-frame-submit-ms") ?? "NaN");
      const webGpuQueueDoneMsAfterDelete = sparkFilteredAfterDelete
        ? 0
        : finiteNumericValue(await viewport.getAttribute("data-webgpu-queue-done-ms") ?? "NaN");
      const webGpuStorageObjectStateByteSizeAfterDelete = sparkFilteredAfterDelete
        ? 0
        : numericValue(await viewport.getAttribute("data-webgpu-storage-object-state-byte-size") ?? "0");
      const webGpuStorageChecksumAfterDelete = sparkFilteredAfterDelete
        ? SPARK_OBJECT_FILTER_MASK
        : await viewport.getAttribute("data-webgpu-storage-checksum");
      const webGpuReadbackAfterDelete =
        editRendererId === "webgpu-tile" &&
        options.webGpuProbe === WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK &&
        !sparkFilteredAfterDelete
          ? await waitForWebGpuReadbackTransition(page, {
              assetId: asset.id,
              label: "delete",
              previousChecksum: webGpuReadbackAfterIsolate.checksum,
            })
          : emptyWebGpuReadbackTelemetry();
      const webGpuColorSourceRgbGaussiansAfterDelete = sparkFilteredAfterDelete
        ? sparkColorSourceGaussiansAfterDelete
        : numericValue(await viewport.getAttribute("data-webgpu-color-source-rgb-gaussians") ?? "0");
      const webGpuColorSourceShDcGaussiansAfterDelete = sparkFilteredAfterDelete
        ? 0
        : numericValue(await viewport.getAttribute("data-webgpu-color-source-sh-dc-gaussians") ?? "0");
      const webGpuColorSourceFallbackGaussiansAfterDelete = sparkFilteredAfterDelete
        ? 0
        : numericValue(await viewport.getAttribute("data-webgpu-color-source-fallback-gaussians") ?? "0");
      const webGpuColorSourceObjectGaussiansAfterDelete = sparkFilteredAfterDelete
        ? sparkColorObjectGaussiansAfterDelete
        : numericValue(await viewport.getAttribute("data-webgpu-color-source-object-gaussians") ?? "0");
      const webGpuColorShViewGaussiansAfterDelete = sparkFilteredAfterDelete
        ? 0
        : numericValue(await viewport.getAttribute("data-webgpu-color-sh-view-gaussians") ?? "0");
      if (deletedObjects !== "1") {
        throw new Error(`${asset.id} delete preview did not update: ${deletedObjects}`);
      }
      if (numericValue(visibleAfterDelete) <= 0) {
        throw new Error(`${asset.id} delete preview did not show remaining scene`);
      }
      if (renderModeAfterDelete !== "原始颜色（编辑预览）") {
        throw new Error(`${asset.id} delete preview did not restore edit-preview original colors`);
      }
      if (
        colorModeRoleAfterDelete !== "source-color" ||
        sourcePreviewBoundaryAfterDelete !== "hard-object-mask-no-reoptimize" ||
        sourcePreviewResultAfterDelete !== "hard-mask-no-inpaint"
      ) {
        throw new Error(
          `${asset.id} delete preview did not expose source-color hard-mask boundary: route=${routeAfterDeleteId}:${routeAfterDeleteKind} color=${colorModeRoleAfterDelete} boundary=${sourcePreviewBoundaryAfterDelete} result=${sourcePreviewResultAfterDelete}`,
        );
      }
      const expectedHardMaskQuality =
        HARD_MASK_QUALITY_EXPECTATIONS[asset.id] ?? "hard-mask-quality-unmeasured";
      const expectedHardMaskQualitySource =
        HARD_MASK_QUALITY_EXPECTATIONS[asset.id] ? "hard-mask-quality-chain-v1" : "route-state";
      if (
        hardMaskQualityAfterDelete !== expectedHardMaskQuality ||
        hardMaskQualitySourceAfterDelete !== expectedHardMaskQualitySource
      ) {
        throw new Error(
          `${asset.id} delete preview hard-mask quality contract failed: quality=${hardMaskQualityAfterDelete}:${hardMaskQualitySourceAfterDelete} expected=${expectedHardMaskQuality}:${expectedHardMaskQualitySource}`,
        );
      }
      if (
        hardMaskQualitySourceAfterDelete === "hard-mask-quality-chain-v1" &&
        (hardMaskGapScoreAfterDelete <= 0 || hardMaskResidualCoverageRatioAfterDelete <= 0)
      ) {
        throw new Error(
          `${asset.id} report-backed hard-mask quality did not expose positive metrics: gap=${hardMaskGapScoreAfterDelete} coverage=${hardMaskResidualCoverageRatioAfterDelete}`,
        );
      }
      if (sparkFilteredAfterDelete && routeAfterDeleteKind !== "commercial") {
        throw new Error(
          `${asset.id} Spark filtered delete preview did not expose commercial route: route=${routeAfterDeleteId}:${routeAfterDeleteKind}`,
        );
      }
      if (!sparkFilteredAfterDelete && postDeleteRendererId === "webgpu-tile" && routeAfterDeleteId !== "webgpu-c-path-diagnostic") {
        throw new Error(
          `${asset.id} WebGPU delete preview did not expose C-path diagnostic route: route=${routeAfterDeleteId}:${routeAfterDeleteKind}`,
        );
      }
      if (
        sparkFilteredAfterDelete &&
        (sparkVisibleGaussiansAfterDelete <= 0 ||
          sparkRemovedObjectsAfterDelete !== 1 ||
          sparkMaskSourceAfterDelete !== sparkExpectedMaskSourceAfterDelete ||
          sparkColorModeAfterDelete !== "original" ||
          sparkColorSourceGaussiansAfterDelete <= 0 ||
          sparkColorObjectGaussiansAfterDelete !== 0 ||
          sparkReconstructSourceAfterDelete !== sparkExpectedReconstructSourceAfterDelete ||
          sparkPackedBaseGaussiansAfterDelete <= 0 ||
          sparkPackedVisibleIndicesAfterDelete !== sparkVisibleGaussiansAfterDelete ||
          !Number.isFinite(sparkPackedBaseBuildMsAfterDelete) ||
          !Number.isFinite(sparkPackedExtractMsAfterDelete) ||
          sparkPackedExtractMsAfterDelete !== 0 ||
          sparkDisplayCacheModeAfterDelete !== SPARK_DISPLAY_CACHE_DISABLED ||
          sparkDisplayCacheKeyAfterDelete !== "" ||
          sparkDisplayCacheHitAfterDelete !== "false" ||
          sparkDisplayCacheSizeAfterDelete !== 0 ||
          sparkDisplayCacheHitsAfterDelete !== 0 ||
          sparkDisplayCacheMissesAfterDelete !== 0 ||
          sparkDisplayCacheEvictionsAfterDelete !== 0 ||
          sparkObjectMaskModeAfterDelete !== SPARK_OBJECT_MASK_MODE ||
          !sparkObjectMaskSizeAfterDelete ||
          sparkObjectMaskVisibleGaussiansAfterDelete !== sparkVisibleGaussiansAfterDelete ||
          sparkObjectMaskHiddenGaussiansAfterDelete !== sparkPackedBaseGaussiansAfterDelete - sparkVisibleGaussiansAfterDelete ||
          sparkObjectMaskUpdatesAfterDelete <= 0 ||
          (options.sparkObjectMaskFeather.enabled
            ? sparkObjectMaskFeatherModeAfterDelete !== "spatial-neighbor-feather-v1" ||
              sparkObjectMaskFeatheredGaussiansAfterDelete <= 0 ||
              sparkObjectMaskFeatherRadiusAfterDelete <= 0 ||
              sparkObjectMaskFeatherOpacityAfterDelete <= 0 ||
              sparkObjectMaskOpacityMeanAfterDelete <= 0 ||
              sparkObjectMaskOpacityMeanAfterDelete >= 1 ||
              sparkObjectMaskMinOpacityAfterDelete <= 0 ||
              sparkObjectMaskMinOpacityAfterDelete >= 1
            : sparkObjectMaskFeatherModeAfterDelete !== "off") ||
          sparkMeshUpdateModeAfterDelete !== SPARK_MESH_UPDATE_MODE ||
          sparkMeshIdAfterDelete <= 0 ||
          sparkMeshUpdatesAfterDelete <= 0 ||
          sparkShRestPreservedAfterDelete !== sparkExpectedShRestPreservedAfterDelete ||
          (sparkShRestSourceGaussiansAfterDelete > 0 &&
            (sparkShRestPreservedGaussiansAfterDelete !== sparkShRestSourceGaussiansAfterDelete ||
              sparkShRestCoefficientCountAfterDelete <= 0 ||
              sparkShDegreeAfterDelete <= 0)))
      ) {
        throw new Error(
          `${asset.id} Spark filtered delete preview contract failed: visible=${sparkVisibleGaussiansAfterDelete} removed=${sparkRemovedObjectsAfterDelete} source=${sparkMaskSourceAfterDelete}/${sparkExpectedMaskSourceAfterDelete} color=${sparkColorModeAfterDelete}:${sparkColorSourceGaussiansAfterDelete}/${sparkColorObjectGaussiansAfterDelete} route=${sparkReconstructSourceAfterDelete}/${sparkExpectedReconstructSourceAfterDelete} packed=${sparkPackedBaseGaussiansAfterDelete}/${sparkPackedVisibleIndicesAfterDelete}:${sparkPackedBaseBuildMsAfterDelete}/${sparkPackedExtractMsAfterDelete} cache=${sparkDisplayCacheModeAfterDelete}:${sparkDisplayCacheKeyAfterDelete}:${sparkDisplayCacheHitAfterDelete}:${sparkDisplayCacheSizeAfterDelete}:${sparkDisplayCacheHitsAfterDelete}/${sparkDisplayCacheMissesAfterDelete}/${sparkDisplayCacheEvictionsAfterDelete} objectMask=${sparkObjectMaskModeAfterDelete}:${sparkObjectMaskSizeAfterDelete}:${sparkObjectMaskVisibleGaussiansAfterDelete}/${sparkObjectMaskHiddenGaussiansAfterDelete}:${sparkObjectMaskUpdatesAfterDelete} feather=${sparkObjectMaskFeatherModeAfterDelete}:${sparkObjectMaskFeatheredGaussiansAfterDelete}:${sparkObjectMaskFeatherRadiusAfterDelete}:${sparkObjectMaskFeatherOpacityAfterDelete}:${sparkObjectMaskOpacityMeanAfterDelete}/${sparkObjectMaskMinOpacityAfterDelete} mesh=${sparkMeshUpdateModeAfterDelete}:${sparkMeshIdAfterDelete}:${sparkMeshReusedAfterDelete}:${sparkMeshUpdatesAfterDelete} shRest=${sparkShRestSourceGaussiansAfterDelete}:${sparkShRestPreservedGaussiansAfterDelete}:${sparkShRestPreservedAfterDelete}:${sparkShRestCoefficientCountAfterDelete}:${sparkShDegreeAfterDelete}`,
        );
      }
      if (sparkFilteredAfterDelete) {
        const selectedBeforeSparkCanvasPick = await selectedObjectValue(page);
        sparkCanvasSelectedObjectAfterDelete = await selectObjectFromCanvas(page, asset.id, {
          previousSelected: selectedBeforeSparkCanvasPick,
          requireSparkPick: true,
        });
        sparkPickStatsAfterDelete = await readSparkPickStats(page);
        const sparkSelectedObject = await page
          .locator(".viewport")
          .first()
          .getAttribute("data-spark-selected-object");
        const selectedRemovedRows = await page.locator(".objectRow.selected.removed").count();
        if (
          sparkCanvasSelectedObjectAfterDelete === selectedBeforeSparkCanvasPick ||
          String(sparkCanvasSelectedObjectAfterDelete) !== String(sparkSelectedObject ?? "") ||
          selectedRemovedRows > 0 ||
          sparkPickStatsAfterDelete.mode !== SPARK_PICK_MODE ||
          sparkPickStatsAfterDelete.interaction !== SPARK_PICK_INTERACTION_MODE ||
          sparkPickStatsAfterDelete.status !== "hit" ||
          String(sparkPickStatsAfterDelete.object) !== String(sparkCanvasSelectedObjectAfterDelete) ||
          sparkPickStatsAfterDelete.hoverStatus !== "hit" ||
          String(sparkPickStatsAfterDelete.hoverObject) !== String(sparkCanvasSelectedObjectAfterDelete) ||
          sparkPickStatsAfterDelete.hoverMarkerVisible !== "true" ||
          sparkPickStatsAfterDelete.distancePx > sparkPickStatsAfterDelete.radiusPx ||
          sparkPickStatsAfterDelete.candidateObjects <= 0 ||
          sparkPickStatsAfterDelete.markerVisible !== "true"
        ) {
          throw new Error(
            `${asset.id} Spark canvas selection did not produce a visible hit: before=${selectedBeforeSparkCanvasPick} after=${sparkCanvasSelectedObjectAfterDelete} attr=${sparkSelectedObject} removedRows=${selectedRemovedRows} pick=${JSON.stringify(sparkPickStatsAfterDelete)}`,
          );
        }
      }
      if (
        sparkFilteredAfterDelete &&
        sparkPackedBaseGaussiansAfterDelete <= SPARK_RESTORE_STRESS_MAX_GAUSSIANS
      ) {
        const maskVisualBefore = await canvasVisualStats(page, ".splatViewport canvas", screenshotOptions);
        validateCanvasVisualStats(asset.id, "Spark object mask before hide", maskVisualBefore);
        const objectMaskUpdatesBeforeRestore = sparkObjectMaskUpdatesAfterDelete;
        const meshIdBeforeRestore = sparkMeshIdAfterDelete;
        const meshUpdatesBeforeRestore = sparkMeshUpdatesAfterDelete;
        const toggleButton = page.locator(".objectRow:not(.removed) .eyeButton").first();
        if ((await toggleButton.count()) <= 0) {
          throw new Error(`${asset.id} Spark object mask visual delta could not find a restorable object`);
        }

        await toggleButton.click();
        await waitForSparkViewportReady(page);
        const maskVisualHidden = await canvasVisualStats(page, ".splatViewport canvas", screenshotOptions);
        validateCanvasVisualStats(asset.id, "Spark object mask hidden", maskVisualHidden);

        await toggleButton.click();
        await waitForSparkViewportReady(page);
        const maskVisualRestored = await canvasVisualStats(page, ".splatViewport canvas", screenshotOptions);
        validateCanvasVisualStats(asset.id, "Spark object mask restored", maskVisualRestored);
        sparkObjectMaskVisualDelta = buildSparkObjectMaskVisualDelta({
          before: maskVisualBefore,
          hidden: maskVisualHidden,
          restored: maskVisualRestored,
        });
        validateSparkObjectMaskVisualDelta(asset.id, sparkObjectMaskVisualDelta);

        const restoredViewport = page.locator(".viewport").first();
        sparkDisplayCacheModeAfterDelete = await restoredViewport.getAttribute("data-spark-display-cache-mode");
        sparkDisplayCacheKeyAfterDelete = await restoredViewport.getAttribute("data-spark-display-cache-key");
        sparkDisplayCacheHitAfterDelete = await restoredViewport.getAttribute("data-spark-display-cache-hit");
        sparkDisplayCacheSizeAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-display-cache-size") ?? "0");
        sparkDisplayCacheHitsAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-display-cache-hits") ?? "0");
        sparkDisplayCacheMissesAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-display-cache-misses") ?? "0");
        sparkDisplayCacheEvictionsAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-display-cache-evictions") ?? "0");
        sparkObjectMaskModeAfterDelete = await restoredViewport.getAttribute("data-spark-object-mask-mode");
        sparkObjectMaskSizeAfterDelete = await restoredViewport.getAttribute("data-spark-object-mask-size");
        sparkObjectMaskUpdatesAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-object-mask-updates") ?? "0");
        sparkObjectMaskVisibleGaussiansAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-object-mask-visible-gaussians") ?? "0");
        sparkObjectMaskHiddenGaussiansAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-object-mask-hidden-gaussians") ?? "0");
        sparkObjectMaskFeatherModeAfterDelete = await restoredViewport.getAttribute("data-spark-object-mask-feather-mode");
        sparkObjectMaskFeatherRadiusAfterDelete = finiteNumericValue(await restoredViewport.getAttribute("data-spark-object-mask-feather-radius") ?? "0");
        sparkObjectMaskFeatherOpacityAfterDelete = finiteNumericValue(await restoredViewport.getAttribute("data-spark-object-mask-feather-opacity") ?? "0");
        sparkObjectMaskFeatheredGaussiansAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-object-mask-feathered-gaussians") ?? "0");
        sparkObjectMaskOpacityMeanAfterDelete = finiteNumericValue(await restoredViewport.getAttribute("data-spark-object-mask-opacity-mean") ?? "0");
        sparkObjectMaskMinOpacityAfterDelete = finiteNumericValue(await restoredViewport.getAttribute("data-spark-object-mask-min-opacity") ?? "0");
        sparkMeshUpdateModeAfterDelete = await restoredViewport.getAttribute("data-spark-mesh-update-mode");
        sparkMeshIdAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-mesh-id") ?? "0");
        sparkMeshReusedAfterDelete = await restoredViewport.getAttribute("data-spark-mesh-reused");
        sparkMeshUpdatesAfterDelete = numericValue(await restoredViewport.getAttribute("data-spark-mesh-updates") ?? "0");
        if (
          sparkDisplayCacheModeAfterDelete !== SPARK_DISPLAY_CACHE_DISABLED ||
          sparkDisplayCacheHitAfterDelete !== "false" ||
          sparkDisplayCacheSizeAfterDelete !== 0 ||
          sparkDisplayCacheHitsAfterDelete !== 0 ||
          sparkDisplayCacheMissesAfterDelete !== 0 ||
          sparkDisplayCacheEvictionsAfterDelete !== 0 ||
          sparkObjectMaskModeAfterDelete !== SPARK_OBJECT_MASK_MODE ||
          sparkObjectMaskUpdatesAfterDelete <= objectMaskUpdatesBeforeRestore ||
          sparkObjectMaskVisibleGaussiansAfterDelete !== sparkVisibleGaussiansAfterDelete ||
          sparkObjectMaskHiddenGaussiansAfterDelete !== sparkPackedBaseGaussiansAfterDelete - sparkVisibleGaussiansAfterDelete ||
          (options.sparkObjectMaskFeather.enabled
            ? sparkObjectMaskFeatherModeAfterDelete !== "spatial-neighbor-feather-v1" ||
              sparkObjectMaskFeatheredGaussiansAfterDelete <= 0 ||
              sparkObjectMaskOpacityMeanAfterDelete <= 0 ||
              sparkObjectMaskOpacityMeanAfterDelete >= 1
            : sparkObjectMaskFeatherModeAfterDelete !== "off")
        ) {
          throw new Error(
            `${asset.id} Spark object mask did not update after restoring visible set: cache=${sparkDisplayCacheModeAfterDelete}:${sparkDisplayCacheHitAfterDelete}:${sparkDisplayCacheSizeAfterDelete}:${sparkDisplayCacheHitsAfterDelete}/${sparkDisplayCacheMissesAfterDelete}/${sparkDisplayCacheEvictionsAfterDelete} objectMask=${sparkObjectMaskModeAfterDelete}:${sparkObjectMaskSizeAfterDelete}:${sparkObjectMaskVisibleGaussiansAfterDelete}/${sparkObjectMaskHiddenGaussiansAfterDelete}:${sparkObjectMaskUpdatesAfterDelete}/${objectMaskUpdatesBeforeRestore} feather=${sparkObjectMaskFeatherModeAfterDelete}:${sparkObjectMaskFeatheredGaussiansAfterDelete}:${sparkObjectMaskOpacityMeanAfterDelete}/${sparkObjectMaskMinOpacityAfterDelete}`,
          );
        }
        if (
          sparkMeshUpdateModeAfterDelete !== SPARK_MESH_UPDATE_MODE ||
          sparkMeshIdAfterDelete !== meshIdBeforeRestore ||
          sparkMeshReusedAfterDelete !== "true"
        ) {
          throw new Error(
            `${asset.id} Spark mesh was not persistently reused after visible-set update: mode=${sparkMeshUpdateModeAfterDelete} mesh=${sparkMeshIdAfterDelete}/${meshIdBeforeRestore} reused=${sparkMeshReusedAfterDelete} updates=${sparkMeshUpdatesAfterDelete}/${meshUpdatesBeforeRestore}`,
          );
        }
      }
      if (
        webGpuColorSourceRgbGaussiansAfterDelete +
          webGpuColorSourceShDcGaussiansAfterDelete <=
          0 ||
        webGpuColorSourceFallbackGaussiansAfterDelete !== 0 ||
        webGpuColorSourceObjectGaussiansAfterDelete !== 0
      ) {
        throw new Error(
          `${asset.id} delete preview did not return to source original colors: rgb=${webGpuColorSourceRgbGaussiansAfterDelete} shDc=${webGpuColorSourceShDcGaussiansAfterDelete} fallback=${webGpuColorSourceFallbackGaussiansAfterDelete} object=${webGpuColorSourceObjectGaussiansAfterDelete}`,
        );
      }
      if (
        webGpuColorMode === WEBGPU_COLOR_MODE_SOURCE &&
        webGpuColorShViewGaussiansAfterDelete !== 0
      ) {
        throw new Error(
          `${asset.id} source color mode unexpectedly used SH-view after delete: shView=${webGpuColorShViewGaussiansAfterDelete}`,
        );
      }
      if (
        webGpuColorMode === WEBGPU_COLOR_MODE_SH_VIEW &&
        webGpuColorShRestGaussians > 0 &&
        webGpuColorShViewGaussiansAfterDelete <= 0
      ) {
        throw new Error(
          `${asset.id} SH-view color mode did not affect source colors after delete: shView=${webGpuColorShViewGaussiansAfterDelete} shRest=${webGpuColorShRestGaussians}`,
        );
      }
      if (!sparkFilteredAfterDelete) {
        if (
          objectStateChecksumAfterDelete === objectStateChecksumAfterIsolate ||
          objectStateVisibleAfterDelete !== objectStateVisibleObjects - 1 ||
          objectStateRemovedAfterDelete !== 1 ||
          objectStateIsolatedAfterDelete !== 0
        ) {
          throw new Error(
            `${asset.id} delete did not update WebGPU object-state buffer: checksum=${objectStateChecksumAfterDelete} visible=${objectStateVisibleAfterDelete} removed=${objectStateRemovedAfterDelete} isolated=${objectStateIsolatedAfterDelete}`,
          );
        }
      }
      if (
        editRendererId === "webgpu-tile" &&
        !sparkFilteredAfterDelete &&
        webGpuStorageChecksumAfterDelete === webGpuStorageChecksumAfterIsolate
      ) {
        throw new Error(`${asset.id} delete did not update WebGPU storage checksum`);
      }
      if (
        editRendererId === "webgpu-tile" &&
        !sparkFilteredAfterDelete &&
        (!(
          (webGpuStorageStatusAfterDelete === "object-state-updated" &&
            webGpuStorageUpdateModeAfterDelete === "object-state-only") ||
          (webGpuStorageStatusAfterDelete === "uploaded" &&
            webGpuStorageUpdateModeAfterDelete === "full-upload")
        ) ||
          !Number.isFinite(webGpuStorageUpdateMsAfterDelete) ||
          webGpuStorageUpdateMsAfterDelete < 0 ||
          !Number.isFinite(webGpuFrameSubmitMsAfterDelete) ||
          webGpuFrameSubmitMsAfterDelete < 0 ||
          webGpuStorageObjectStateByteSizeAfterDelete <= 0)
      ) {
        throw new Error(
          `${asset.id} delete did not use WebGPU storage update fallback with timing: status=${webGpuStorageStatusAfterDelete} updateMode=${webGpuStorageUpdateModeAfterDelete} updateMs=${webGpuStorageUpdateMsAfterDelete} submitMs=${webGpuFrameSubmitMsAfterDelete} queueDoneMs=${webGpuQueueDoneMsAfterDelete} objectStateBytes=${webGpuStorageObjectStateByteSizeAfterDelete}`,
        );
      }
      if (
        options.webGpuObjectTransition &&
        options.webGpuProbe === WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK &&
        editRendererId === "webgpu-tile" &&
        !sparkFilteredAfterDelete &&
        webGpuReadbackAfterDelete.checksum === webGpuReadbackAfterIsolate.checksum
      ) {
        throw new Error(
          `${asset.id} delete did not update WebGPU offscreen readback checksum: ${webGpuReadbackAfterDelete.checksum}`,
        );
      }
      const screenshotPath = `/tmp/objgauss-audit-${asset.id}.png`;
      await page.screenshot({ path: screenshotPath, fullPage: false });
      results.push({
        assetId: asset.id,
        title,
        splatPixels,
        splatRendererId,
        initialRouteId,
        initialRouteKind,
        initialColorModeRole,
        initialSourcePreviewBoundary,
        initialHardMaskQuality,
        initialHardMaskQualitySource,
        ...visualResidualResultFields(sparkVisualStats, editOriginalVisualStats, visualResidual),
        editPixels,
        editRenderer,
        editRendererId,
        objectColorRouteId,
        objectColorRouteKind,
        objectColorModeRole,
        objectColorSourcePreviewBoundary,
        objectColorHardMaskQuality,
        webGpuFirstFrameStatus,
        webGpuFirstFramePixels,
        webGpuFirstFrameChecksum,
        webGpuResolveSource,
        webGpuResolveFilter,
        webGpuAlphaPresentationMode,
        webGpuAlphaPresentationTuningMode,
        webGpuAlphaPresentationFloor,
        webGpuRuntimeProbe,
        webGpuViewportWidth,
        webGpuViewportHeight,
        webGpuPixelCount,
        webGpuViewportAspectMode,
        webGpuViewportQuality,
        webGpuViewportPixelBudget,
        webGpuDisplayWidth,
        webGpuDisplayHeight,
        webGpuBoundsFitMode,
        webGpuBoundsPaddingRatio,
        webGpuBoundsViewportAspect,
        webGpuBoundsWorldAspect,
        webGpuProjectionMode,
        webGpuProjectionCameraTuningMode,
        webGpuProjectionCameraMode,
        webGpuProjectionCameraFov,
        webGpuProjectionCameraPosition,
        webGpuProjectionCameraTarget,
        webGpuProjectionCameraDistance,
        webGpuProjectionCameraFrameMaxDim,
        webGpuDepthWeightMode,
        webGpuPixelDepthSortMode,
        webGpuPixelDepthTuningMode,
        webGpuPixelDepthAlphaMode,
        webGpuPixelDepthGateStrength,
        webGpuPixelDepthGateFloor,
        webGpuPixelDepthBinCount,
        webGpuPixelCoverageMode,
        webGpuPixelCoverageTuningMode,
        webGpuPixelCoverageWeightFloor,
        webGpuPixelCoverageFootprintScale,
        webGpuProjectionDepthMin,
        webGpuProjectionDepthMax,
        webGpuProjectionDepthSpan,
        webGpuColorFidelityMode,
        webGpuColorSourceRgbGaussians,
        webGpuColorSourceShDcGaussians,
        webGpuColorSourceFallbackGaussians,
        webGpuColorSourceObjectGaussians,
        webGpuColorTuningMode,
        webGpuColorMode,
        webGpuColorShRestGaussians,
        webGpuColorShRestCoefficientMax,
        webGpuColorShDegreeMax,
        webGpuColorShViewGaussians,
        webGpuColorOpacityMean,
        webGpuColorSourceRgbGaussiansAfterDelete,
        webGpuColorSourceShDcGaussiansAfterDelete,
        webGpuColorSourceFallbackGaussiansAfterDelete,
        webGpuColorSourceObjectGaussiansAfterDelete,
        webGpuColorShViewGaussiansAfterDelete,
        webGpuScreenCovarianceMode,
        webGpuScreenCovarianceGaussians,
        webGpuScreenCovarianceFallbackGaussians,
        webGpuScreenCovarianceClampedGaussians,
        webGpuScreenCovarianceMaxAnisotropy,
        webGpuScreenCovarianceSigmaMean,
        webGpuDeviceLostStatus,
        webGpuDeviceLostReason,
        webGpuDeviceLostMessage,
        webGpuDeviceErrorStatus,
        webGpuDeviceErrorType,
        webGpuDeviceErrorMessage,
        webGpuQueueStatus,
        webGpuQueueReason,
        webGpuQueueMessage,
        webGpuAccumulationSource,
        webGpuAccumulationStatus,
        webGpuAccumulationWorkgroups,
        webGpuComputeSource,
        webGpuComputeStatus,
        webGpuComputeWorkgroups,
        webGpuPixelSource,
        webGpuPixelStatus,
        webGpuPixelWorkgroups,
        webGpuReadbackStatus,
        webGpuReadbackReason,
        webGpuReadbackSource,
        webGpuReadbackChecksum,
        webGpuReadbackByteSize,
        webGpuReadbackFloatCount,
        webGpuReadbackFiniteFloats,
        webGpuReadbackNonzeroFloats,
        webGpuReadbackStatusAfterIsolate: webGpuReadbackAfterIsolate.status,
        webGpuReadbackSourceAfterIsolate: webGpuReadbackAfterIsolate.source,
        webGpuReadbackChecksumAfterIsolate: webGpuReadbackAfterIsolate.checksum,
        webGpuReadbackByteSizeAfterIsolate: webGpuReadbackAfterIsolate.byteSize,
        webGpuReadbackFloatCountAfterIsolate: webGpuReadbackAfterIsolate.floatCount,
        webGpuReadbackFiniteFloatsAfterIsolate: webGpuReadbackAfterIsolate.finiteFloats,
        webGpuReadbackNonzeroFloatsAfterIsolate: webGpuReadbackAfterIsolate.nonzeroFloats,
        webGpuReadbackStatusAfterDelete: webGpuReadbackAfterDelete.status,
        webGpuReadbackSourceAfterDelete: webGpuReadbackAfterDelete.source,
        webGpuReadbackChecksumAfterDelete: webGpuReadbackAfterDelete.checksum,
        webGpuReadbackByteSizeAfterDelete: webGpuReadbackAfterDelete.byteSize,
        webGpuReadbackFloatCountAfterDelete: webGpuReadbackAfterDelete.floatCount,
        webGpuReadbackFiniteFloatsAfterDelete: webGpuReadbackAfterDelete.finiteFloats,
        webGpuReadbackNonzeroFloatsAfterDelete: webGpuReadbackAfterDelete.nonzeroFloats,
        webGpuStorageLayout,
        webGpuStorageStatus,
        webGpuStorageBufferCount,
        webGpuStorageByteSize,
        webGpuStorageTileEntries,
        webGpuStorageTileOffsets,
        webGpuStoragePixelOutput,
        webGpuStorageChecksum,
        webGpuStorageUpdateMode,
        webGpuStorageUpdateMs,
        webGpuFrameSubmitMs,
        webGpuQueueDoneMs,
        webGpuStorageObjectStateByteSize,
        webGpuStorageStatusAfterIsolate,
        webGpuStorageUpdateModeAfterIsolate,
        webGpuStorageUpdateMsAfterIsolate,
        webGpuFrameSubmitMsAfterIsolate,
        webGpuQueueDoneMsAfterIsolate,
        webGpuStorageObjectStateByteSizeAfterIsolate,
        webGpuStorageStatusAfterDelete,
        webGpuStorageUpdateModeAfterDelete,
        webGpuStorageUpdateMsAfterDelete,
        webGpuFrameSubmitMsAfterDelete,
        webGpuQueueDoneMsAfterDelete,
        webGpuStorageObjectStateByteSizeAfterDelete,
        webGpuStorageChecksumAfterIsolate,
        webGpuStorageChecksumAfterDelete,
        storageLimitGate,
        storageLimitReason,
        storageLimitBlocker,
        storageLimitMaxBufferSize,
        storageLimitMaxBindingSize,
        storageLimitMaxStorageBuffersPerStage,
        storageLimitRequiredStorageBuffersPerStage,
        storageLimitEffectiveMaxBufferSize,
        storageEstimatedLayout,
        storageEstimatedBufferCount,
        storageEstimatedByteSize,
        storageEstimatedMaxBufferByteSize,
        storageEstimatedMaxBufferKey,
        rendererTarget,
        targetGate,
        targetGateReason,
        targetGateBlocker,
        webgpuStatus,
        fallbackReason,
        tileSmokeLayout,
        packedGaussians,
        visibleGaussians,
        binnedGaussians,
        tileSize,
        tileCount,
        activeTileCount,
        tileReferenceCount,
        maxTileOccupancy,
        tileOverflowTileCount,
        tileOverflowRatio,
        tileOverflowMaxExcess,
        tileEntryStoredCount,
        tileEntryCapacity,
        tileEntryUtilization,
        tileEntryLayout,
        tileEntryOffsetCount,
        tileCapacityMode,
        tileCapacityStatus,
        tileCapacityGate,
        resolveLayout,
        resolveMode,
        resolvedTileCount,
        resolveWeightSum,
        resolveAlphaMean,
        resolveLumaMean,
        resolveChecksum,
        tileOverflowCount,
        objectFilter,
        objectFilterTarget,
        objectStateLayout,
        objectStateStride,
        objectStateVisibleObjects,
        objectStateHiddenObjects,
        objectStateRemovedObjects,
        objectStateSelectedObjects,
        objectStateIsolatedObjects,
        objectStateChecksum,
        objectStateChecksumAfterIsolate,
        objectStateChecksumAfterDelete,
        canvasSelectedObject,
        sparkCanvasSelectedObjectAfterDelete,
        ...sparkPickResultFields(sparkPickStatsAfterDelete),
        visibleAfterIsolate,
        visibleAfterDelete,
        renderModeAfterDelete,
        routeAfterDeleteId,
        routeAfterDeleteKind,
        colorModeRoleAfterDelete,
        sourcePreviewBoundaryAfterDelete,
        sourcePreviewResultAfterDelete,
        hardMaskQualityAfterDelete,
        hardMaskQualitySourceAfterDelete,
        hardMaskGapScoreAfterDelete,
        hardMaskResidualCoverageRatioAfterDelete,
        hardMaskDeletedObjectAfterDelete,
        deletedObjects,
        postDeleteRendererId,
        postDeleteObjectFilter,
        sparkMaskSourceAfterDelete,
        sparkFilteredGaussiansAfterDelete: sparkVisibleGaussiansAfterDelete,
        sparkReconstructSourceAfterDelete,
        sparkPackedBaseGaussiansAfterDelete,
        sparkPackedVisibleIndicesAfterDelete,
        sparkPackedBaseBuildMsAfterDelete,
        sparkPackedExtractMsAfterDelete,
        sparkDisplayCacheModeAfterDelete,
        sparkDisplayCacheKeyAfterDelete,
        sparkDisplayCacheHitAfterDelete,
        sparkDisplayCacheSizeAfterDelete,
        sparkDisplayCacheHitsAfterDelete,
        sparkDisplayCacheMissesAfterDelete,
        sparkDisplayCacheEvictionsAfterDelete,
        sparkObjectMaskModeAfterDelete,
        sparkObjectMaskSizeAfterDelete,
        sparkObjectMaskUpdatesAfterDelete,
        sparkObjectMaskVisibleGaussiansAfterDelete,
        sparkObjectMaskHiddenGaussiansAfterDelete,
        sparkObjectMaskFeatherModeAfterDelete,
        sparkObjectMaskFeatherRadiusAfterDelete,
        sparkObjectMaskFeatherOpacityAfterDelete,
        sparkObjectMaskFeatheredGaussiansAfterDelete,
        sparkObjectMaskOpacityMeanAfterDelete,
        sparkObjectMaskMinOpacityAfterDelete,
        sparkObjectMaskVisualMode: sparkObjectMaskVisualDelta.mode,
        sparkObjectMaskVisualBeforeChecksum: sparkObjectMaskVisualDelta.beforeChecksum,
        sparkObjectMaskVisualHiddenChecksum: sparkObjectMaskVisualDelta.hiddenChecksum,
        sparkObjectMaskVisualRestoredChecksum: sparkObjectMaskVisualDelta.restoredChecksum,
        sparkObjectMaskVisualCoverageDelta: sparkObjectMaskVisualDelta.coverageDelta,
        sparkObjectMaskVisualLumaDelta: sparkObjectMaskVisualDelta.lumaDelta,
        sparkObjectMaskVisualChromaDelta: sparkObjectMaskVisualDelta.chromaDelta,
        sparkObjectMaskVisualRestoreCoverageDelta: sparkObjectMaskVisualDelta.restoreCoverageDelta,
        sparkObjectMaskVisualRestoreLumaDelta: sparkObjectMaskVisualDelta.restoreLumaDelta,
        sparkObjectMaskVisualRestoreChromaDelta: sparkObjectMaskVisualDelta.restoreChromaDelta,
        sparkMeshUpdateModeAfterDelete,
        sparkMeshIdAfterDelete,
        sparkMeshReusedAfterDelete,
        sparkMeshUpdatesAfterDelete,
        sparkShRestSourceGaussiansAfterDelete,
        sparkShRestPreservedGaussiansAfterDelete,
        sparkShRestPreservedAfterDelete,
        sparkShRestCoefficientCountAfterDelete,
        sparkShDegreeAfterDelete,
        screenshotPath,
      });
    }
    const relevantIssues = consoleIssues.filter(
      (issue) =>
        !issue.includes("THREE.WebGLRenderer") &&
        !issue.includes("GPU stall due to ReadPixels") &&
        !issue.includes("No available adapters.") &&
        !(options.allowWebGpuDeviceLost && issue.includes("CONTEXT_LOST_WEBGL")),
    );
    if (relevantIssues.length > 0) {
      throw new Error(`browser console issues:\n${relevantIssues.join("\n")}`);
    }
    return results;
  } finally {
    await closeBrowserWithTimeout(browser);
  }
}

async function closeBrowserWithTimeout(browser, timeoutMs = 5000) {
  const closePromise = browser.close().then(
    () => true,
    () => true,
  );
  const closed = await Promise.race([
    closePromise,
    sleep(timeoutMs).then(() => false),
  ]);
  if (closed) return;

  const child = typeof browser.process === "function" ? browser.process() : null;
  child?.kill?.("SIGTERM");
  await Promise.race([
    closePromise,
    sleep(1000).then(() => false),
  ]);
}

function validateWebGpuRuntimeProbe({
  assetId,
  expectedProbe,
  actualProbe,
  webGpuAccumulationSource,
  webGpuAccumulationStatus,
  webGpuAccumulationReason,
  webGpuAccumulationWorkgroups,
  webGpuComputeSource,
  webGpuComputeStatus,
  webGpuComputeReason,
  webGpuComputeWorkgroups,
  webGpuPixelSource,
  webGpuPixelStatus,
  webGpuPixelReason,
  webGpuPixelWorkgroups,
  webGpuFirstFrameStatus,
  webGpuFirstFrameReason,
  webGpuFirstFramePixels,
  webGpuFirstFrameChecksum,
  webGpuResolveSource,
  webGpuResolveFilter,
  webGpuReadbackStatus,
  webGpuReadbackReason,
  webGpuReadbackSource,
  webGpuReadbackChecksum,
  webGpuReadbackByteSize,
  webGpuReadbackFloatCount,
  webGpuReadbackFiniteFloats,
  webGpuReadbackNonzeroFloats,
  webGpuAlphaPresentationMode,
  webGpuAlphaPresentationTuningMode,
  webGpuAlphaPresentationFloor,
  expectedAlphaPresentationFloor,
}) {
  if (actualProbe !== expectedProbe) {
    throw new Error(`${assetId} WebGPU runtime probe mismatch: expected=${expectedProbe} actual=${actualProbe}`);
  }
  if (
    webGpuResolveSource === "webgpu-pixel-storage-resolve-v1" &&
    (
      webGpuAlphaPresentationMode !== WEBGPU_TILE_ALPHA_PRESENTATION_MODE ||
      webGpuAlphaPresentationTuningMode !== WEBGPU_TILE_ALPHA_PRESENTATION_TUNING_MODE ||
      Math.abs(
        webGpuAlphaPresentationFloor -
          (Number.isFinite(expectedAlphaPresentationFloor)
            ? expectedAlphaPresentationFloor
            : WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR),
      ) > 0.000001
    )
  ) {
    throw new Error(
      `${assetId} WebGPU storage resolve did not expose alpha presentation gate: mode=${webGpuAlphaPresentationMode} tuning=${webGpuAlphaPresentationTuningMode} floor=${webGpuAlphaPresentationFloor} expected=${expectedAlphaPresentationFloor}`,
    );
  }

  const stageState = {
    accumulation: {
      status: webGpuAccumulationStatus,
      source: webGpuAccumulationSource,
      workgroups: webGpuAccumulationWorkgroups,
      expectedSource: "webgpu-compute-covariance-accumulation-v1",
      reason: webGpuAccumulationReason,
    },
    compute: {
      status: webGpuComputeStatus,
      source: webGpuComputeSource,
      workgroups: webGpuComputeWorkgroups,
      expectedSource: "webgpu-compute-resolve-v1",
      reason: webGpuComputeReason,
    },
    pixel: {
      status: webGpuPixelStatus,
      source: webGpuPixelSource,
      workgroups: webGpuPixelWorkgroups,
      expectedSource: WEBGPU_PIXEL_RESOLVE_SOURCE,
      reason: webGpuPixelReason,
    },
  };

  const expected = expectedProbeStages(expectedProbe);
  for (const [stage, state] of Object.entries(stageState)) {
    if (expected.dispatched.has(stage)) {
      if (
        state.status !== "dispatched" ||
        state.source !== state.expectedSource ||
        state.workgroups <= 0
      ) {
        throw new Error(
          `${assetId} WebGPU ${expectedProbe} did not dispatch ${stage}: ${state.status}:${state.source}:${state.workgroups}:${state.reason}`,
        );
      }
    } else if (state.status !== "skipped" || state.source || state.workgroups !== 0) {
      throw new Error(
        `${assetId} WebGPU ${expectedProbe} did not skip ${stage}: ${state.status}:${state.source}:${state.workgroups}:${state.reason}`,
      );
    }
  }

  if (
    expectedProbe === WEBGPU_RUNTIME_PROBE_FULL ||
    expectedProbe === WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY ||
    expectedProbe === WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY ||
    expectedProbe === WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT
  ) {
    validateRenderedProbeFrame({
      assetId,
      expectedProbe,
      webGpuFirstFrameStatus,
      webGpuFirstFrameReason,
      webGpuFirstFramePixels,
      webGpuFirstFrameChecksum,
      webGpuResolveSource,
      webGpuResolveFilter,
      expectedSource: "webgpu-pixel-storage-resolve-v1",
      expectedFilter: "bilinear-storage",
    });
  } else if (expectedProbe === WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY) {
    validateRenderedProbeFrame({
      assetId,
      expectedProbe,
      webGpuFirstFrameStatus,
      webGpuFirstFrameReason,
      webGpuFirstFramePixels,
      webGpuFirstFrameChecksum,
      webGpuResolveSource,
      webGpuResolveFilter,
      expectedSource: "webgpu-sampled-texture-resolve-v1",
      expectedFilter: "nearest-sampled-texture",
    });
  } else if (expectedProbe === WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY) {
    validateRenderedProbeFrame({
      assetId,
      expectedProbe,
      webGpuFirstFrameStatus,
      webGpuFirstFrameReason,
      webGpuFirstFramePixels,
      webGpuFirstFrameChecksum,
      webGpuResolveSource,
      webGpuResolveFilter,
      expectedSource: "webgpu-buffer-copy-texture-resolve-v1",
      expectedFilter: "nearest-texture-load",
    });
  } else if (expectedProbe === WEBGPU_RUNTIME_PROBE_CLEAR_ONLY) {
    validateRenderedProbeFrame({
      assetId,
      expectedProbe,
      webGpuFirstFrameStatus,
      webGpuFirstFrameReason,
      webGpuFirstFramePixels,
      webGpuFirstFrameChecksum,
      webGpuResolveSource,
      webGpuResolveFilter,
      expectedSource: "webgpu-clear-pass-v1",
      expectedFilter: "clear-pass",
    });
  } else if (expectedProbe === WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY) {
    if (
      webGpuFirstFrameStatus !== "probed" ||
      webGpuFirstFramePixels <= 0 ||
      !/^[0-9a-f]{8}$/.test(webGpuFirstFrameChecksum ?? "") ||
      webGpuResolveSource !== WEBGPU_PIXEL_RESOLVE_SOURCE
    ) {
      throw new Error(
        `${assetId} WebGPU pixel-compute-only probe did not submit pixel compute: frame=${webGpuFirstFrameStatus}:${webGpuFirstFrameReason} pixels=${webGpuFirstFramePixels} checksum=${webGpuFirstFrameChecksum} source=${webGpuResolveSource}`,
      );
    }
  } else if (expectedProbe === WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK) {
    if (
      webGpuFirstFrameStatus !== "readback" ||
      webGpuFirstFramePixels <= 0 ||
      !/^[0-9a-f]{8}$/.test(webGpuFirstFrameChecksum ?? "") ||
      webGpuResolveSource !== WEBGPU_PIXEL_RESOLVE_SOURCE ||
      webGpuResolveFilter !== "offscreen-map-read" ||
      webGpuReadbackStatus !== "mapped" ||
      webGpuReadbackSource !== WEBGPU_PIXEL_RESOLVE_SOURCE ||
      webGpuReadbackChecksum !== webGpuFirstFrameChecksum ||
      webGpuReadbackByteSize <= 0 ||
      webGpuReadbackFloatCount <= 0 ||
      webGpuReadbackFiniteFloats !== webGpuReadbackFloatCount ||
      webGpuReadbackNonzeroFloats <= 0
    ) {
      throw new Error(
        `${assetId} WebGPU offscreen-readback probe did not map GPU pixel output: frame=${webGpuFirstFrameStatus}:${webGpuFirstFrameReason} pixels=${webGpuFirstFramePixels} checksum=${webGpuFirstFrameChecksum} source=${webGpuResolveSource}:${webGpuResolveFilter} readback=${webGpuReadbackStatus}:${webGpuReadbackReason}:${webGpuReadbackSource}:${webGpuReadbackChecksum}:${webGpuReadbackByteSize}:${webGpuReadbackFloatCount}:${webGpuReadbackFiniteFloats}:${webGpuReadbackNonzeroFloats}`,
      );
    }
  } else if (
    webGpuFirstFrameStatus !== "probed" ||
    webGpuFirstFramePixels <= 0 ||
    !/^[0-9a-f]{8}$/.test(webGpuFirstFrameChecksum ?? "")
  ) {
    throw new Error(
      `${assetId} WebGPU ${expectedProbe} did not expose a submitted probe frame fact: frame=${webGpuFirstFrameStatus}:${webGpuFirstFrameReason} pixels=${webGpuFirstFramePixels} checksum=${webGpuFirstFrameChecksum} source=${webGpuResolveSource}`,
    );
  }
}

function validateRenderedProbeFrame({
  assetId,
  expectedProbe,
  webGpuFirstFrameStatus,
  webGpuFirstFrameReason,
  webGpuFirstFramePixels,
  webGpuFirstFrameChecksum,
  webGpuResolveSource,
  webGpuResolveFilter,
  expectedSource,
  expectedFilter,
}) {
  if (
    webGpuFirstFrameStatus !== "rendered" ||
    webGpuFirstFramePixels <= 0 ||
    !/^[0-9a-f]{8}$/.test(webGpuFirstFrameChecksum ?? "") ||
    webGpuResolveSource !== expectedSource
  ) {
    throw new Error(
      `${assetId} WebGPU ${expectedProbe} route did not render through ${expectedSource}: frame=${webGpuFirstFrameStatus}:${webGpuFirstFrameReason} pixels=${webGpuFirstFramePixels} checksum=${webGpuFirstFrameChecksum} source=${webGpuResolveSource}`,
    );
  }
  if (webGpuResolveFilter !== expectedFilter) {
    throw new Error(
      `${assetId} WebGPU ${expectedProbe} resolve filter mismatch: expected=${expectedFilter} actual=${webGpuResolveFilter}`,
    );
  }
}

function expectedProbeStages(probe) {
  if (probe === WEBGPU_RUNTIME_PROBE_ACCUMULATION_ONLY) {
    return { dispatched: new Set(["accumulation"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_RESOLVE_ONLY) {
    return { dispatched: new Set(["compute"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_PIXEL_OUTPUT_ONLY) {
    return { dispatched: new Set(["pixel"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_PIXEL_COMPUTE_ONLY) {
    return { dispatched: new Set(["pixel"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_OFFSCREEN_READBACK) {
    return { dispatched: new Set(["pixel"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_DISPLAY_ONLY) {
    return { dispatched: new Set() };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT) {
    return { dispatched: new Set(["pixel"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TEXTURE_DISPLAY_ONLY) {
    return { dispatched: new Set() };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_TEXTURE_COPY_DISPLAY) {
    return { dispatched: new Set(["pixel"]) };
  }
  if (probe === WEBGPU_RUNTIME_PROBE_CLEAR_ONLY) {
    return { dispatched: new Set() };
  }
  return { dispatched: new Set(["accumulation", "compute", "pixel"]) };
}

function launchOptions(options = {}) {
  const executablePath = options.executablePath ?? firstExisting([
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
  ]);
  const args = ["--no-sandbox", ...webGpuLaunchArgs(options.webGpuFlags)];
  const launch = {
    args,
    headless: !options.headed,
  };
  if (Number.isFinite(options.slowMo) && options.slowMo > 0) {
    launch.slowMo = options.slowMo;
  }
  if (options.browserChannel) {
    launch.channel = options.browserChannel;
    return launch;
  }
  if (executablePath) {
    launch.executablePath = executablePath;
  }
  return launch;
}

function urlWithWebGpuOptions(url, options) {
  if (
    options.webGpuProbe === WEBGPU_RUNTIME_PROBE_FULL &&
    !options.webGpuViewportSize &&
    !Number.isFinite(options.webGpuFootprintScale) &&
    !Number.isFinite(options.webGpuCovarianceMaxAnisotropy) &&
    !Number.isFinite(options.webGpuDepthBins) &&
    options.webGpuDepthAlphaMode === WEBGPU_DEPTH_ALPHA_MODE_DEFAULT &&
    options.webGpuCameraTuning.cameraMode === WEBGPU_CAMERA_MODE_EDIT_FIXED &&
    options.webGpuColorTuning.colorMode === WEBGPU_COLOR_MODE_SOURCE &&
    !options.sparkNativeMask &&
    !options.sparkObjectMaskFeather.enabled &&
    Math.abs(
      options.webGpuAlphaPresentationTuning.alphaPresentationFloor -
        WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR,
    ) <= 0.000001
    && !options.webGpuObjectTransition
  ) {
    return url;
  }
  const parsed = new URL(url);
  if (options.requireWebGpu || options.webGpuProbe !== WEBGPU_RUNTIME_PROBE_FULL) {
    parsed.searchParams.set("webgpu-probe", options.webGpuProbe);
  }
  if (options.webGpuViewportSize) {
    parsed.searchParams.set("webgpu-viewport-size", String(options.webGpuViewportSize));
  }
  if (Number.isFinite(options.webGpuFootprintScale)) {
    parsed.searchParams.set("webgpu-footprint-scale", String(options.webGpuFootprintScale));
  }
  if (Number.isFinite(options.webGpuCovarianceMaxAnisotropy)) {
    parsed.searchParams.set("webgpu-covariance-max-anisotropy", String(options.webGpuCovarianceMaxAnisotropy));
  }
  if (Number.isFinite(options.webGpuDepthBins)) {
    parsed.searchParams.set("webgpu-depth-bins", String(options.webGpuDepthBins));
  }
  if (options.webGpuDepthAlphaMode !== WEBGPU_DEPTH_ALPHA_MODE_DEFAULT) {
    parsed.searchParams.set("webgpu-depth-alpha-mode", options.webGpuDepthAlphaMode);
  }
  if (options.webGpuCameraTuning.cameraMode !== WEBGPU_CAMERA_MODE_EDIT_FIXED) {
    parsed.searchParams.set("webgpu-camera-mode", options.webGpuCameraTuning.cameraMode);
  }
  if (options.webGpuColorTuning.colorMode !== WEBGPU_COLOR_MODE_SOURCE) {
    parsed.searchParams.set("webgpu-color-mode", options.webGpuColorTuning.colorMode);
  }
  if (options.sparkNativeMask) {
    parsed.searchParams.set("spark-native-mask", "on");
  }
  if (options.sparkObjectMaskFeather.enabled) {
    parsed.searchParams.set("spark-object-mask-feather", "on");
    if (Number.isFinite(options.sparkObjectMaskFeather.radius)) {
      parsed.searchParams.set("spark-object-mask-feather-radius", String(options.sparkObjectMaskFeather.radius));
    }
    if (Number.isFinite(options.sparkObjectMaskFeather.opacity)) {
      parsed.searchParams.set("spark-object-mask-feather-opacity", String(options.sparkObjectMaskFeather.opacity));
    }
  }
  if (options.webGpuObjectTransition) {
    parsed.searchParams.set("spark-filtered-edit", "off");
  }
  if (
    Math.abs(
      options.webGpuAlphaPresentationTuning.alphaPresentationFloor -
        WEBGPU_TILE_ALPHA_PRESENTATION_FLOOR,
    ) > 0.000001
  ) {
    parsed.searchParams.set(
      "webgpu-alpha-presentation-floor",
      String(options.webGpuAlphaPresentationTuning.alphaPresentationFloor),
    );
  }
  return parsed.toString();
}

function webGpuLaunchArgs(mode) {
  if (mode === "none" || mode === "false" || mode === "") return [];
  if (mode === "unsafe") {
    return ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"];
  }
  if (mode === "vulkan") {
    return [
      "--enable-unsafe-webgpu",
      "--ignore-gpu-blocklist",
      "--enable-features=Vulkan,DefaultANGLEVulkan,WebGPUDeveloperFeatures",
      "--use-angle=vulkan",
    ];
  }
  throw new Error(`unsupported WebGPU launch flags mode: ${mode}`);
}

function firstExisting(paths) {
  return paths.find((path) => existsSync(path));
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === false || value === undefined || value === null) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function optionalString(value) {
  if (value === undefined || value === null || value === true || value === false) return undefined;
  const text = String(value).trim();
  return text ? text : undefined;
}

function optionalPositiveInteger(value) {
  if (value === undefined || value === null || value === true || value === false) return undefined;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return undefined;
  return Math.round(parsed);
}

function optionalFiniteNumber(value) {
  if (value === undefined || value === null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function selectAssets(parsedArgs) {
  const assetList = parsedArgs.assets ?? parsedArgs.asset;
  if (!assetList) return DEFAULT_ASSETS;
  if (assetList === true) {
    throw new Error("--assets requires a comma-separated asset id list");
  }
  const requested = String(assetList)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
  if (requested.length === 0) {
    throw new Error("--assets requires at least one asset id");
  }
  const byId = new Map(KNOWN_ASSETS.map((asset) => [asset.id, asset]));
  const unknown = requested.filter((id) => !byId.has(id));
  if (unknown.length > 0) {
    throw new Error(`unknown asset id(s): ${unknown.join(",")}`);
  }
  return requested.map((id) => byId.get(id));
}

function clampNumber(value, min, max) {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function validVectorAttribute(value) {
  const parts = String(value ?? "")
    .split(",")
    .map((entry) => Number(entry));
  return parts.length === 3 && parts.every((entry) => Number.isFinite(entry));
}

function normalizeServerMode(value) {
  const mode = String(value ?? "dev").trim().toLowerCase();
  if (mode === "dev" || mode === "preview") return mode;
  throw new Error(`unknown --server-mode: ${value}`);
}

function startServer(port, mode) {
  const script = mode === "preview" ? "preview" : "dev";
  const child = spawn(
    "npm",
    ["run", script, "--", "--port", String(port), "--strictPort"],
    { detached: true, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stdout.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr.on("data", (chunk) => process.stderr.write(chunk));
  return child;
}

function stopDevServer(child) {
  if (!child.pid) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    child.kill("SIGTERM");
  }
}

async function waitForApp(url) {
  const deadline = Date.now() + 30000;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  throw new Error(`app did not become ready at ${url}: ${lastError?.message ?? "timeout"}`);
}

async function expectText(page, text) {
  await page.getByText(text).first().waitFor({ timeout: 15000 });
}

async function expectNoFrameworkOverlay(page) {
  const bodyText = await page.locator("body").innerText();
  const overlaySignals = ["Vite Error", "Internal server error", "Failed to resolve import"];
  for (const signal of overlaySignals) {
    if (bodyText.includes(signal)) {
      throw new Error(`framework overlay detected: ${signal}`);
    }
  }
}

async function nonBackgroundPixels(page) {
  const webGpuPixels = await page.locator(".viewport").first().evaluate((viewport) => {
    if (viewport?.getAttribute("data-renderer") !== "webgpu-tile") return 0;
    return Number(viewport.getAttribute("data-webgpu-first-frame-pixels") ?? "0");
  }).catch(() => 0);
  if (webGpuPixels > 0) return webGpuPixels;

  return page.locator("canvas").evaluateAll((canvases) => {
    let maxCount = -1;
    for (const canvas of canvases) {
      const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
      if (!gl) continue;
      const width = canvas.width;
      const height = canvas.height;
      const pixels = new Uint8Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      let count = 0;
      for (let index = 0; index < pixels.length; index += 4) {
        if (pixels[index] !== 16 || pixels[index + 1] !== 19 || pixels[index + 2] !== 22) {
          count += 1;
        }
      }
      maxCount = Math.max(maxCount, count);
    }
    return maxCount;
  });
}

async function waitForNonBackgroundPixels(page, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  let pixels = 0;
  while (Date.now() < deadline) {
    pixels = await nonBackgroundPixels(page);
    if (pixels > 0) {
      return pixels;
    }
    await page.waitForTimeout(250);
  }
  return pixels;
}

async function waitForWebGpuQueueTelemetry(page, timeoutMs = 8000) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    const queueStatus = viewport?.getAttribute("data-webgpu-queue-status");
    const deviceLostStatus = viewport?.getAttribute("data-webgpu-device-lost-status");
    return (
      queueStatus === "done" ||
      queueStatus === "failed" ||
      queueStatus === "unavailable" ||
      deviceLostStatus === "lost"
    );
  }, undefined, { timeout: timeoutMs }).catch(() => {});
}

async function waitForWebGpuStorageUpdate(
  page,
  { assetId, label, previousChecksum, allowFullUpload = false },
  timeoutMs = 20000,
) {
  await page.waitForFunction(
    ({ previous, allowFullUpload: allowFull }) => {
      const viewport = document.querySelector(".viewport");
      if (viewport?.getAttribute("data-renderer") !== "webgpu-tile") return false;
      const storageStatus = viewport.getAttribute("data-webgpu-storage-status");
      const storageUpdateMode = viewport.getAttribute("data-webgpu-storage-update-mode");
      const storageChecksum = viewport.getAttribute("data-webgpu-storage-checksum") ?? "";
      const storageUpdateMs = Number(viewport.getAttribute("data-webgpu-storage-update-ms") ?? "NaN");
      const submitMs = Number(viewport.getAttribute("data-webgpu-frame-submit-ms") ?? "NaN");
      const queueDoneMs = Number(viewport.getAttribute("data-webgpu-queue-done-ms") ?? "NaN");
      const queueStatus = viewport.getAttribute("data-webgpu-queue-status");
      const deviceLostStatus = viewport.getAttribute("data-webgpu-device-lost-status");
      const objectStateOnly =
        storageStatus === "object-state-updated" &&
        storageUpdateMode === "object-state-only";
      const fullUpload =
        allowFull &&
        storageStatus === "uploaded" &&
        storageUpdateMode === "full-upload";
      const queueReady =
        queueStatus === "submitted" ||
        (queueStatus === "done" && Number.isFinite(queueDoneMs) && queueDoneMs >= 0) ||
        queueStatus === "failed" ||
        deviceLostStatus === "lost";
      return (
        (objectStateOnly || fullUpload) &&
        /^[0-9a-f]{8}$/.test(storageChecksum) &&
        storageChecksum !== previous &&
        Number.isFinite(storageUpdateMs) &&
        storageUpdateMs >= 0 &&
        Number.isFinite(submitMs) &&
        submitMs >= 0 &&
        queueReady
      );
    },
    { previous: previousChecksum, allowFullUpload },
    { timeout: timeoutMs },
  ).catch((error) => {
    throw new Error(
      `${assetId} WebGPU ${label} storage update timing did not settle: ${error.message}`,
    );
  });
}

async function waitForWebGpuReadbackTransition(
  page,
  { assetId, label, previousChecksum },
  timeoutMs = 15000,
) {
  await page.waitForFunction(
    ({ previous }) => {
      const viewport = document.querySelector(".viewport");
      if (viewport?.getAttribute("data-renderer") !== "webgpu-tile") return false;
      const queueStatus = viewport.getAttribute("data-webgpu-queue-status");
      const readbackStatus = viewport.getAttribute("data-webgpu-readback-status");
      const checksum = viewport.getAttribute("data-webgpu-readback-checksum") ?? "";
      const finiteFloats = Number(viewport.getAttribute("data-webgpu-readback-finite-floats") ?? "0");
      const floatCount = Number(viewport.getAttribute("data-webgpu-readback-float-count") ?? "0");
      return (
        queueStatus === "done" &&
        readbackStatus === "mapped" &&
        /^[0-9a-f]{8}$/.test(checksum) &&
        checksum !== previous &&
        floatCount > 0 &&
        finiteFloats === floatCount
      );
    },
    { previous: previousChecksum },
    { timeout: timeoutMs },
  );
  const telemetry = await readWebGpuReadbackTelemetry(page);
  if (
    telemetry.status !== "mapped" ||
    telemetry.source !== WEBGPU_PIXEL_RESOLVE_SOURCE ||
    !/^[0-9a-f]{8}$/.test(telemetry.checksum) ||
    telemetry.checksum === previousChecksum ||
    telemetry.byteSize <= 0 ||
    telemetry.floatCount <= 0 ||
    telemetry.finiteFloats !== telemetry.floatCount ||
    telemetry.nonzeroFloats <= 0
  ) {
    throw new Error(
      `${assetId} invalid WebGPU readback after ${label}: ` +
        `${telemetry.status}:${telemetry.source}:${telemetry.checksum}:` +
        `${telemetry.byteSize}:${telemetry.finiteFloats}/${telemetry.floatCount}:` +
        `${telemetry.nonzeroFloats} previous=${previousChecksum}`,
    );
  }
  return telemetry;
}

async function readWebGpuReadbackTelemetry(page) {
  const viewport = page.locator(".viewport").first();
  return {
    status: (await viewport.getAttribute("data-webgpu-readback-status")) ?? "",
    source: (await viewport.getAttribute("data-webgpu-readback-source")) ?? "",
    checksum: (await viewport.getAttribute("data-webgpu-readback-checksum")) ?? "",
    byteSize: numericValue(await viewport.getAttribute("data-webgpu-readback-byte-size") ?? "0"),
    floatCount: numericValue(await viewport.getAttribute("data-webgpu-readback-float-count") ?? "0"),
    finiteFloats: numericValue(await viewport.getAttribute("data-webgpu-readback-finite-floats") ?? "0"),
    nonzeroFloats: numericValue(await viewport.getAttribute("data-webgpu-readback-nonzero-floats") ?? "0"),
  };
}

function emptyWebGpuReadbackTelemetry() {
  return {
    status: "probe-skipped",
    source: "",
    checksum: "",
    byteSize: 0,
    floatCount: 0,
    finiteFloats: 0,
    nonzeroFloats: 0,
  };
}

async function waitForEditViewportReady(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    const renderer = viewport?.getAttribute("data-renderer");
    const webGpuStatus = viewport?.getAttribute("data-webgpu-status");
    return (
      (renderer === "gaussian-oit" || renderer === "webgpu-tile") &&
      webGpuStatus !== "pending"
    );
  }, undefined, { timeout: 15000 });
  const rendererId = await page.locator(".viewport").first().getAttribute("data-renderer");
  if (rendererId === "webgpu-tile") {
    await page.waitForFunction(() => {
      const viewport = document.querySelector(".viewport");
      return viewport?.getAttribute("data-webgpu-first-frame-status") !== "pending";
    }, undefined, { timeout: 15000 });
    await waitForWebGpuQueueTelemetry(page);
  }
  await waitForNonBackgroundPixels(page);
}

async function waitForSparkViewportReady(page) {
  await page.waitForFunction(() => {
    const viewport = document.querySelector(".viewport");
    return (
      viewport?.getAttribute("data-renderer") === "spark-splat" &&
      viewport?.getAttribute("data-spark-filter-status") === "ready"
    );
  }, undefined, { timeout: 15000 });
}

function validateVisualResidual(assetId, residual) {
  if (
    !Number.isFinite(residual.coverageRatio) ||
    residual.coverageRatio <= 0 ||
    !Number.isFinite(residual.lumaDelta) ||
    !Number.isFinite(residual.chromaDelta)
  ) {
    throw new Error(`${assetId} visual residual is invalid: ${JSON.stringify(residual)}`);
  }
}

function emptySparkObjectMaskVisualDelta() {
  return {
    mode: "not-run",
    beforeChecksum: "",
    hiddenChecksum: "",
    restoredChecksum: "",
    coverageDelta: 0,
    lumaDelta: 0,
    chromaDelta: 0,
    restoreCoverageDelta: 0,
    restoreLumaDelta: 0,
    restoreChromaDelta: 0,
  };
}

function buildSparkObjectMaskVisualDelta({ before, hidden, restored }) {
  return {
    mode: SPARK_OBJECT_MASK_VISUAL_DELTA_MODE,
    beforeChecksum: before.checksum,
    hiddenChecksum: hidden.checksum,
    restoredChecksum: restored.checksum,
    coverageDelta: roundMetric(Math.abs(hidden.coverage - before.coverage)),
    lumaDelta: roundMetric(Math.abs(hidden.lumaMean - before.lumaMean)),
    chromaDelta: roundMetric(Math.abs(hidden.chromaMean - before.chromaMean)),
    restoreCoverageDelta: roundMetric(Math.abs(restored.coverage - before.coverage)),
    restoreLumaDelta: roundMetric(Math.abs(restored.lumaMean - before.lumaMean)),
    restoreChromaDelta: roundMetric(Math.abs(restored.chromaMean - before.chromaMean)),
  };
}

function validateSparkObjectMaskVisualDelta(assetId, delta) {
  const hideChanged =
    delta.beforeChecksum !== delta.hiddenChecksum &&
    (delta.coverageDelta >= SPARK_OBJECT_MASK_MIN_VISUAL_DELTA ||
      delta.lumaDelta >= SPARK_OBJECT_MASK_MIN_VISUAL_DELTA ||
      delta.chromaDelta >= SPARK_OBJECT_MASK_MIN_VISUAL_DELTA);
  if (!hideChanged) {
    throw new Error(
      `${assetId} Spark object mask did not produce a visible hide delta: checksums=${delta.beforeChecksum}/${delta.hiddenChecksum} delta=${delta.coverageDelta}/${delta.lumaDelta}/${delta.chromaDelta}`,
    );
  }

  const restored =
    delta.beforeChecksum === delta.restoredChecksum ||
    (delta.restoreCoverageDelta <= SPARK_OBJECT_MASK_MAX_RESTORE_DELTA &&
      delta.restoreLumaDelta <= SPARK_OBJECT_MASK_MAX_RESTORE_DELTA &&
      delta.restoreChromaDelta <= SPARK_OBJECT_MASK_MAX_RESTORE_DELTA);
  if (!restored) {
    throw new Error(
      `${assetId} Spark object mask did not visually restore the deleted-state baseline: checksums=${delta.beforeChecksum}/${delta.restoredChecksum} restoreDelta=${delta.restoreCoverageDelta}/${delta.restoreLumaDelta}/${delta.restoreChromaDelta}`,
    );
  }
}

function visualResidualResultFields(sparkStats, editStats, residual) {
  if (!sparkStats || !editStats || !residual) {
    return {
      visualResidualMode: VISUAL_RESIDUAL_SKIPPED_MODE,
      sparkVisualWidth: 0,
      sparkVisualHeight: 0,
      sparkVisualPixels: 0,
      sparkVisualNonBackgroundPixels: 0,
      sparkVisualCoverage: 0,
      sparkVisualLumaMean: 0,
      sparkVisualChromaMean: 0,
      sparkVisualChecksum: "",
      editOriginalVisualWidth: 0,
      editOriginalVisualHeight: 0,
      editOriginalVisualPixels: 0,
      editOriginalVisualNonBackgroundPixels: 0,
      editOriginalVisualCoverage: 0,
      editOriginalVisualLumaMean: 0,
      editOriginalVisualChromaMean: 0,
      editOriginalVisualChecksum: "",
      sparkEditCoverageRatio: 0,
      sparkEditLumaDelta: 0,
      sparkEditChromaDelta: 0,
    };
  }
  return {
    visualResidualMode: VISUAL_RESIDUAL_MODE,
    sparkVisualWidth: sparkStats.width,
    sparkVisualHeight: sparkStats.height,
    sparkVisualPixels: sparkStats.pixels,
    sparkVisualNonBackgroundPixels: sparkStats.nonBackgroundPixels,
    sparkVisualCoverage: sparkStats.coverage,
    sparkVisualLumaMean: sparkStats.lumaMean,
    sparkVisualChromaMean: sparkStats.chromaMean,
    sparkVisualChecksum: sparkStats.checksum,
    editOriginalVisualWidth: editStats.width,
    editOriginalVisualHeight: editStats.height,
    editOriginalVisualPixels: editStats.pixels,
    editOriginalVisualNonBackgroundPixels: editStats.nonBackgroundPixels,
    editOriginalVisualCoverage: editStats.coverage,
    editOriginalVisualLumaMean: editStats.lumaMean,
    editOriginalVisualChromaMean: editStats.chromaMean,
    editOriginalVisualChecksum: editStats.checksum,
    sparkEditCoverageRatio: residual.coverageRatio,
    sparkEditLumaDelta: residual.lumaDelta,
    sparkEditChromaDelta: residual.chromaDelta,
  };
}

async function labeledValue(page, label) {
  const value = await page.locator(".metric, .stateRow").evaluateAll((rows, targetLabel) => {
    const row = rows.find(
      (candidate) => candidate.querySelector("span")?.textContent?.trim() === targetLabel,
    );
    return row?.querySelector("strong")?.textContent?.trim() ?? "";
  }, label);
  if (!value) {
    throw new Error(`missing labeled value: ${label}`);
  }
  return value;
}

function numericValue(value) {
  return Number(value.replace(/[^\d]/g, ""));
}

function finiteNumericValue(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : Number.NaN;
}

async function selectObjectFromCanvas(
  page,
  assetId,
  { previousSelected = "无", requireSparkPick = false } = {},
) {
  const canvas = page.locator(".viewport canvas").first();
  const box = await canvas.boundingBox();
  if (!box) {
    throw new Error(`${assetId} point-edit canvas is missing`);
  }

  const clickPoints = [
    [0.5, 0.5],
    [0.45, 0.48],
    [0.55, 0.48],
    [0.4, 0.55],
    [0.6, 0.55],
    [0.5, 0.4],
    [0.35, 0.48],
    [0.65, 0.48],
  ];
  for (const [xRatio, yRatio] of clickPoints) {
    const x = box.x + box.width * xRatio;
    const y = box.y + box.height * yRatio;
    await page.mouse.move(x, y);
    await page.waitForTimeout(180);
    await page.mouse.click(x, y);
    await page.waitForTimeout(250);
    const selectedObject = await selectedObjectValue(page);
    if (selectedObject !== "无" && selectedObject !== previousSelected) {
      if (requireSparkPick) {
        const pick = await readSparkPickStats(page);
        if (
          pick.mode !== SPARK_PICK_MODE ||
          pick.interaction !== SPARK_PICK_INTERACTION_MODE ||
          pick.status !== "hit" ||
          String(pick.object) !== String(selectedObject) ||
          pick.hoverStatus !== "hit" ||
          String(pick.hoverObject) !== String(selectedObject) ||
          pick.hoverMarkerVisible !== "true" ||
          pick.distancePx > pick.radiusPx ||
          pick.candidateObjects <= 0 ||
          pick.markerVisible !== "true"
        ) {
          continue;
        }
      }
      return selectedObject;
    }
  }
  throw new Error(`${assetId} canvas selection did not choose an object`);
}

async function selectedObjectValue(page) {
  const status = await page.locator(".statusBar").innerText();
  const match = status.match(/所选：([^\n]+)/);
  return match?.[1] ?? "无";
}

async function readSparkPickStats(page) {
  return page.locator(".viewport").first().evaluate((viewport) => {
    const numericAttr = (name) => {
      const numeric = Number(viewport.getAttribute(name) ?? "0");
      return Number.isFinite(numeric) ? numeric : 0;
    };
    return {
      mode: viewport.getAttribute("data-spark-selection-mode") ?? "",
      interaction: viewport.getAttribute("data-spark-pick-interaction") ?? "",
      status: viewport.getAttribute("data-spark-pick-status") ?? "",
      object: viewport.getAttribute("data-spark-pick-object") ?? "",
      distancePx: numericAttr("data-spark-pick-distance-px"),
      candidateObjects: numericAttr("data-spark-pick-candidate-objects"),
      ambiguous: viewport.getAttribute("data-spark-pick-ambiguous") ?? "",
      radiusPx: numericAttr("data-spark-pick-radius-px"),
      hoverStatus: viewport.getAttribute("data-spark-hover-pick-status") ?? "",
      hoverObject: viewport.getAttribute("data-spark-hover-pick-object") ?? "",
      hoverMarkerVisible: viewport.getAttribute("data-spark-hover-marker-visible") ?? "",
      markerVisible: viewport.getAttribute("data-spark-selected-marker-visible") ?? "",
    };
  });
}

function emptySparkPickStats() {
  return {
    mode: "",
    interaction: "",
    status: "not-run",
    object: "",
    distancePx: 0,
    candidateObjects: 0,
    ambiguous: "",
    radiusPx: 0,
    hoverStatus: "",
    hoverObject: "",
    hoverMarkerVisible: "",
    markerVisible: "",
  };
}

function sparkPickResultFields(stats) {
  return {
    sparkPickModeAfterDelete: stats.mode,
    sparkPickInteractionAfterDelete: stats.interaction,
    sparkPickStatusAfterDelete: stats.status,
    sparkPickObjectAfterDelete: stats.object,
    sparkPickDistancePxAfterDelete: stats.distancePx,
    sparkPickCandidateObjectsAfterDelete: stats.candidateObjects,
    sparkPickAmbiguousAfterDelete: stats.ambiguous,
    sparkPickRadiusPxAfterDelete: stats.radiusPx,
    sparkPickHoverStatusAfterDelete: stats.hoverStatus,
    sparkPickHoverObjectAfterDelete: stats.hoverObject,
    sparkPickHoverMarkerVisibleAfterDelete: stats.hoverMarkerVisible,
    sparkSelectedMarkerVisibleAfterDelete: stats.markerVisible,
  };
}

function emptySparkPickResultFields(status = "not-run") {
  return sparkPickResultFields({
    ...emptySparkPickStats(),
    status,
  });
}

function parseArgs(rawArgs) {
  const parsed = {};
  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = rawArgs[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}
