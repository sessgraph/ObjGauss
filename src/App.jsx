import {
  BarChart3,
  Database,
  Eye,
  EyeOff,
  FileUp,
  Layers3,
  LoaderCircle,
  RefreshCw,
  Rotate3D,
  Scissors,
  Trash2,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { ASSET_LIBRARY, featuredAssets } from "./assetLibrary.js";
import { parsePly, parsePlyFile } from "./ply.js";
import PointCloudViewport from "./PointCloudViewport.jsx";
import { rgbToCss } from "./palette.js";
import { createSampleScene } from "./sampleScene.js";
import SplatViewport from "./SplatViewport.jsx";
import { normalizeSparkObjectMaskFeathering } from "./sparkObjectMask.js";
import WebGpuTileViewport from "./WebGpuTileViewport.jsx";
import {
  detectWebGpuCapability,
  editRendererContract,
  INITIAL_WEBGPU_CAPABILITY,
} from "./webgpuCapability.js";
import {
  normalizeWebGpuRuntimeProbe,
  WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT,
  WEBGPU_RUNTIME_PROBE_TINY_VIEWPORT_SIZE,
} from "./webgpuRuntimeProbe.js";
import {
  buildWebGpuTileSmoke,
  normalizeWebGpuCameraTuning,
  normalizeWebGpuColorTuning,
  normalizeWebGpuCoverageTuning,
  normalizeWebGpuDepthSortTuning,
} from "./webgpuTileSmoke.js";

const FEATURED_ASSETS = featuredAssets();
const LOCAL_SAMPLE_ASSET = ASSET_LIBRARY.find((asset) => asset.id === "plush-3dgs-local");
const BENCHMARK_GATES = [
  { label: "Smoke", value: "pass" },
  { label: "Candidate", value: "pass" },
  { label: "Paper", value: "pass" },
];
const BENCHMARK_SCENES = [
  {
    id: "lego-safe-2000",
    label: "Lego safe-2000",
    ari: 0.469787,
    oes: 0.784051,
    render: 0.229397,
    heldout: 0.197505,
  },
  {
    id: "fern-smoke",
    label: "Fern smoke",
    ari: 0.790636,
    oes: 0.780132,
    render: 0.235029,
    heldout: 0.233851,
  },
  {
    id: "chair-smoke",
    label: "Chair smoke",
    ari: 0.614363,
    oes: 0.757609,
    render: 0.248716,
    heldout: 0.224084,
  },
];
const WEBGPU_RUNTIME_VIEWPORT_SIZE = 256;
const WEBGPU_RUNTIME_MIN_VIEWPORT_SIZE = 64;
const WEBGPU_RUNTIME_MAX_VIEWPORT_SIZE = 512;
const WEBGPU_RUNTIME_VIEWPORT_TILE_SIZE = 16;
const WEBGPU_RUNTIME_HIGH_MAX_GAUSSIANS = 50_000;
const WEBGPU_RUNTIME_MEDIUM_MAX_GAUSSIANS = 300_000;
const WEBGPU_RUNTIME_HIGH_VIEWPORT_SIZE = 512;
const WEBGPU_RUNTIME_MEDIUM_VIEWPORT_SIZE = 384;
const WEBGPU_RUNTIME_SAFE_VIEWPORT_SIZE = 320;
const UI_SPARK_OBJECT_MASK_FEATHER_OPACITY = 0.55;
const HARD_MASK_QUALITY_BY_ASSET = {
  "nerf-lego-alpha-closure-local": {
    interpretation: "boundary-mixing-dominant",
    label: "边界混合主导",
    source: "hard-mask-quality-chain-v1",
    deletedObjectId: 0,
    hardMaskGapScore: 0.524659,
    residualCoverageRatio: 1.170841,
  },
  "plush-semantic-closure-local": {
    interpretation: "boundary-mixing-dominant",
    label: "边界混合主导",
    source: "hard-mask-quality-chain-v1",
    deletedObjectId: 0,
    hardMaskGapScore: 0.513937,
    residualCoverageRatio: 1.303149,
  },
  "nerf-lego-trained-output-local": {
    interpretation: "browser-residual-dominant",
    label: "重建残差主导",
    source: "hard-mask-quality-chain-v1",
    deletedObjectId: 0,
    hardMaskGapScore: 0.377656,
    residualCoverageRatio: 15.599172,
  },
  "polyhaven-chair-commercial-demo-local": {
    interpretation: "boundary-mixing-dominant",
    label: "边界混合主导",
    source: "hard-mask-quality-chain-v1",
    deletedObjectId: 0,
    hardMaskGapScore: 0.29813,
    residualCoverageRatio: 1.075414,
  },
};

export default function App() {
  const [scene, setScene] = useState(() => createSampleScene());
  const [viewMode, setViewMode] = useState("edit");
  const [sideTab, setSideTab] = useState("samples");
  const [renderMode, setRenderMode] = useState("original");
  const [pointSize, setPointSize] = useState(0.018);
  const [showGrid, setShowGrid] = useState(true);
  const [showAxes, setShowAxes] = useState(true);
  const [visibleIds, setVisibleIds] = useState(() => allIds(createSampleScene().points));
  const [removedIds, setRemovedIds] = useState(() => new Set());
  const [selectedId, setSelectedId] = useState(null);
  const [isolatedId, setIsolatedId] = useState(null);
  const [webGpuCapability, setWebGpuCapability] = useState(INITIAL_WEBGPU_CAPABILITY);
  const [sparkObjectMaskFeathering, setSparkObjectMaskFeathering] = useState(
    readInitialSparkObjectMaskFeathering,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    detectWebGpuCapability().then((capability) => {
      if (!cancelled) setWebGpuCapability(capability);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const summary = useMemo(() => summarize(scene.points), [scene.points]);
  const renderModeText = renderModeLabel(renderMode);
  const webGpuCoverageTuning = useMemo(readWebGpuCoverageTuning, []);
  const webGpuDepthSortTuning = useMemo(readWebGpuDepthSortTuning, []);
  const webGpuCameraTuning = useMemo(readWebGpuCameraTuning, []);
  const webGpuColorTuning = useMemo(readWebGpuColorTuning, []);
  const sparkReconstructProbe = useMemo(readSparkReconstructProbe, []);
  const sparkPlySourceMode = useMemo(readSparkPlySourceMode, []);
  const sparkNativeMaskMode = useMemo(readSparkNativeMaskMode, []);
  const sparkFilteredEditEnabled = useMemo(readSparkFilteredEditEnabled, []);
  const webGpuTileSmoke = useMemo(
    () =>
      buildWebGpuTileSmoke({
        points: scene.points,
        shRestCoefficients: scene.shRestCoefficients,
        shRestCoefficientCount: scene.shRestCoefficientCount,
        visibleIds,
        removedIds,
        isolatedId,
        selectedId,
        renderMode,
        pointSize,
        coverageTuning: webGpuCoverageTuning,
        depthSortTuning: webGpuDepthSortTuning,
        cameraTuning: webGpuCameraTuning,
        colorTuning: webGpuColorTuning,
      }),
    [
      scene.points,
      scene.shRestCoefficients,
      scene.shRestCoefficientCount,
      visibleIds,
      removedIds,
      isolatedId,
      selectedId,
      renderMode,
      pointSize,
      webGpuCoverageTuning,
      webGpuDepthSortTuning,
      webGpuCameraTuning,
      webGpuColorTuning,
    ],
  );
  const editRenderer = useMemo(
    () => editRendererContract(webGpuCapability, webGpuTileSmoke),
    [webGpuCapability, webGpuTileSmoke],
  );
  const sceneObjectIds = useMemo(() => allIds(scene.points), [scene.points]);
  const hasSplatRenderer = Boolean(scene.splatSource);
  const objectEditActive = useMemo(
    () =>
      removedIds.size > 0 ||
      isolatedId !== null ||
      sceneObjectIds.size !== visibleIds.size ||
      [...sceneObjectIds].some((id) => !visibleIds.has(id)),
    [isolatedId, removedIds, sceneObjectIds, visibleIds],
  );
  const canUseSplatRenderer = hasSplatRenderer && renderMode === "original" && !objectEditActive;
  const canUseSparkPlySourceRenderer =
    hasSplatRenderer &&
    renderMode === "original" &&
    !objectEditActive &&
    webGpuColorTuning.colorMode === "source" &&
    sparkPlySourceMode !== "off" &&
    (sparkPlySourceMode === "force" || sceneHasShRestSource(scene));
  const canUseSparkFilteredRenderer =
    hasSplatRenderer &&
    renderMode === "original" &&
    (objectEditActive || sparkReconstructProbe) &&
    sparkFilteredEditEnabled &&
    webGpuColorTuning.colorMode === "source";
  const canUseSparkNativeMaskSource = hasSplatRenderer && !sceneHasShRestSource(scene);
  const useSparkPlySourceRenderer = viewMode === "view" && canUseSparkPlySourceRenderer;
  const useSplatRenderer = viewMode === "view" && canUseSplatRenderer && !useSparkPlySourceRenderer;
  const useSparkFilteredRenderer = viewMode === "edit" && canUseSparkFilteredRenderer;
  const useSparkNativeMaskRenderer =
    useSparkFilteredRenderer &&
    objectEditActive &&
    (sparkNativeMaskMode === "force" ||
      (sparkNativeMaskMode === "auto" && canUseSparkNativeMaskSource));
  const useSparkPointRenderer = useSparkPlySourceRenderer || useSparkFilteredRenderer;
  const waitForEditRenderer =
    !useSplatRenderer && !useSparkPointRenderer && webGpuCapability.status === "pending";
  const useWebGpuTileRenderer =
    !useSplatRenderer && !useSparkPointRenderer && editRenderer.rendererId === "webgpu-tile";
  const webGpuRuntimeProbe = useMemo(readWebGpuRuntimeProbe, []);
  const webGpuRuntimeViewportRequest = useMemo(readWebGpuRuntimeViewportRequest, []);
  const [webGpuRuntimeDisplaySize, setWebGpuRuntimeDisplaySize] = useState({ width: 0, height: 0 });
  const updateWebGpuRuntimeDisplaySize = useCallback((nextSize) => {
    const width = Math.max(0, Math.round(Number(nextSize?.width) || 0));
    const height = Math.max(0, Math.round(Number(nextSize?.height) || 0));
    setWebGpuRuntimeDisplaySize((current) =>
      current.width === width && current.height === height ? current : { width, height },
    );
  }, []);
  const webGpuRuntimeViewport = useMemo(
    () =>
      buildWebGpuRuntimeViewport({
        probe: webGpuRuntimeProbe,
        request: webGpuRuntimeViewportRequest,
        displaySize: webGpuRuntimeDisplaySize,
        gaussianCount: scene.points.length,
      }),
    [scene.points.length, webGpuRuntimeDisplaySize, webGpuRuntimeProbe, webGpuRuntimeViewportRequest],
  );
  const webGpuRuntimeTileSmoke = useMemo(() => {
    if (!useWebGpuTileRenderer) return webGpuTileSmoke;
    return buildWebGpuTileSmoke({
      points: scene.points,
      shRestCoefficients: scene.shRestCoefficients,
      shRestCoefficientCount: scene.shRestCoefficientCount,
      visibleIds,
      removedIds,
      isolatedId,
      selectedId,
      renderMode,
      pointSize,
      viewportWidth: webGpuRuntimeViewport.width,
      viewportHeight: webGpuRuntimeViewport.height,
      includeTileEntries: true,
      includePixelOutput: true,
      computePixelReference: false,
      maxEntriesPerTile: Math.max(1, webGpuTileSmoke.maxTileOccupancy),
      coverageTuning: webGpuCoverageTuning,
      depthSortTuning: webGpuDepthSortTuning,
      cameraTuning: webGpuCameraTuning,
      colorTuning: webGpuColorTuning,
    });
  }, [
    scene.points,
    scene.shRestCoefficients,
    scene.shRestCoefficientCount,
    visibleIds,
    removedIds,
    isolatedId,
    selectedId,
    renderMode,
    pointSize,
    useWebGpuTileRenderer,
    webGpuRuntimeViewport,
    webGpuTileSmoke,
    webGpuCoverageTuning,
    webGpuDepthSortTuning,
    webGpuCameraTuning,
    webGpuColorTuning,
  ]);
  const activeEditRenderer = useMemo(
    () =>
      useWebGpuTileRenderer
        ? editRendererContract(webGpuCapability, webGpuRuntimeTileSmoke)
        : editRenderer,
    [editRenderer, useWebGpuTileRenderer, webGpuCapability, webGpuRuntimeTileSmoke],
  );
  const activeRendererText = useSplatRenderer
    ? "真实 Splat"
    : useSparkPlySourceRenderer
      ? "Spark PLY SH 源"
    : useSparkFilteredRenderer
      ? objectEditActive
        ? useSparkNativeMaskRenderer
          ? "Spark 原生 Splat 过滤"
          : "Spark 过滤 Splat"
        : "Spark PLY 重建"
      : activeEditRenderer.rendererLabel;
  const modeText = viewMode === "view" ? "真实查看" : "对象编辑";
  const sparkObjectMaskFeatheringLabel = sparkObjectMaskFeathering.enabled
    ? `柔化 ${sparkObjectMaskFeathering.opacity.toFixed(2)}`
    : "关闭";
  const rendererRoute = useMemo(
    () =>
      rendererRouteContract({
        renderMode,
        objectEditActive,
        useSplatRenderer,
        useSparkPlySourceRenderer,
        useSparkFilteredRenderer,
        useSparkNativeMaskRenderer,
        useWebGpuTileRenderer,
      }),
    [
      renderMode,
      objectEditActive,
      useSplatRenderer,
      useSparkPlySourceRenderer,
      useSparkFilteredRenderer,
      useSparkNativeMaskRenderer,
      useWebGpuTileRenderer,
    ],
  );
  const hardMaskQuality = useMemo(
    () => hardMaskQualityContract(scene, rendererRoute),
    [scene, rendererRoute],
  );
  const visibleCount = useMemo(
    () =>
      scene.points.filter(
        (point) =>
          visibleIds.has(point.objectId) &&
          !removedIds.has(point.objectId) &&
          (isolatedId === null || point.objectId === isolatedId),
      ).length,
    [scene.points, visibleIds, removedIds, isolatedId],
  );

  const applyScene = (next) => {
    setScene(next);
    const ids = allIds(next.points);
    setVisibleIds(ids);
    setRemovedIds(new Set());
    setSelectedId(null);
    setIsolatedId(null);
    setViewMode(next.splatSource ? "view" : "edit");
    setRenderMode("original");
  };

  const loadFile = async (file) => {
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const next = await parsePlyFile(file);
      applyScene(next);
    } catch (loadError) {
      setError(loadError.message || "PLY 加载失败");
    } finally {
      setBusy(false);
    }
  };

  const loadAsset = async (asset) => {
    if (!asset?.localPath) return;
    setBusy(true);
    setError("");
    try {
      const response = await fetch(asset.localPath);
      if (!response.ok) {
        throw new Error(`示例 PLY 加载失败 (${response.status})`);
      }
      const cloud = parsePly(await response.arrayBuffer());
      applyScene({
        assetId: asset.id,
        name: asset.fileName ?? asset.name,
        points: cloud.points,
        shRestCoefficients: cloud.shRestCoefficients,
        shRestCoefficientCount: cloud.shRestCoefficientCount,
        shDegree: cloud.shDegree,
        splatSource: {
          url: asset.splatPath ?? asset.localPath,
          fileName: asset.fileName ?? asset.name,
        },
      });
    } catch (loadError) {
      setError(loadError.message || "示例 PLY 加载失败");
    } finally {
      setBusy(false);
    }
  };

  const loadSample = () => loadAsset(LOCAL_SAMPLE_ASSET);

  const resetDemo = () => {
    const next = createSampleScene();
    setScene(next);
    setVisibleIds(allIds(next.points));
    setRemovedIds(new Set());
    setSelectedId(null);
    setIsolatedId(null);
    setViewMode("edit");
    setRenderMode("original");
    setError("");
  };

  const enterViewMode = () => {
    if (!hasSplatRenderer) return;
    setVisibleIds(new Set(sceneObjectIds));
    setRemovedIds(new Set());
    setIsolatedId(null);
    setRenderMode("original");
    setViewMode("view");
  };

  const enterEditMode = () => {
    setViewMode("edit");
  };

  const setEditRenderMode = (mode) => {
    setRenderMode(mode);
    if (mode === "clustered") {
      setViewMode("edit");
    }
  };

  const toggleSparkObjectMaskFeathering = useCallback((enabled) => {
    setSparkObjectMaskFeathering((current) =>
      normalizeSparkObjectMaskFeathering({
        ...current,
        enabled,
        opacity: current.opacity > 0 && current.opacity < 1
          ? current.opacity
          : UI_SPARK_OBJECT_MASK_FEATHER_OPACITY,
      }),
    );
  }, []);

  const selectObject = (id) => {
    setSelectedId(id);
    setViewMode("edit");
  };

  const toggleVisible = (id) => {
    setViewMode("edit");
    setVisibleIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const removeSelected = () => {
    if (selectedId === null) return;
    setViewMode("edit");
    setRenderMode("original");
    setIsolatedId(null);
    setRemovedIds((current) => new Set([...current, selectedId]));
  };

  const restoreRemoved = () => {
    setRemovedIds(new Set());
  };

  const isolateSelected = () => {
    if (selectedId === null) return;
    setViewMode("edit");
    setIsolatedId(selectedId);
  };

  return (
    <main
      className="appShell"
      data-renderer-route={rendererRoute.id}
      data-renderer-route-kind={rendererRoute.kind}
      data-color-mode-role={rendererRoute.colorModeRole}
      data-source-preview-boundary={rendererRoute.sourceBoundary}
      data-source-preview-result={rendererRoute.sourceResult}
      data-preview-quality={rendererRoute.qualityId}
      data-hard-mask-quality-interpretation={hardMaskQuality.interpretation}
      data-hard-mask-quality-source={hardMaskQuality.source}
      data-hard-mask-gap-score={hardMaskQuality.hardMaskGapScore ?? ""}
      data-hard-mask-residual-coverage-ratio={hardMaskQuality.residualCoverageRatio ?? ""}
      data-hard-mask-deleted-object={hardMaskQuality.deletedObjectId ?? ""}
      data-spark-object-mask-feather-control="ui-v1"
      data-spark-object-mask-feather-enabled={String(sparkObjectMaskFeathering.enabled)}
      data-spark-object-mask-feather-opacity={sparkObjectMaskFeathering.opacity}
      data-spark-object-mask-feather-radius={sparkObjectMaskFeathering.radius}
    >
      <header className="topbar">
        <div className="brand">
          <div className="brandMark">
            <Layers3 size={19} />
          </div>
          <div>
            <h1>ObjGauss 查看器</h1>
            <p>对象级高斯点云预览</p>
          </div>
        </div>

        <nav className="modeTabs" aria-label="工作模式">
          <button
            className={`modeTab ${viewMode === "view" ? "active" : ""}`}
            type="button"
            onClick={enterViewMode}
            disabled={!hasSplatRenderer}
            title={objectEditActive ? "清除编辑预览并查看真实 splat 外观" : "查看真实 splat 外观"}
          >
            <Rotate3D size={17} />
            <span>真实查看</span>
          </button>
          <button
            className={`modeTab ${viewMode === "edit" ? "active" : ""}`}
            type="button"
            onClick={enterEditMode}
            title="对象操作使用点云编辑预览"
          >
            <Scissors size={17} />
            <span>对象编辑</span>
          </button>
        </nav>

        <div className="topActions">
          <select
            value={renderMode}
            onChange={(event) => setEditRenderMode(event.target.value)}
            aria-label="渲染模式"
          >
            <option value="original">自身颜色</option>
            <option value="clustered">对象色诊断</option>
          </select>
          <button className="iconButton" type="button" onClick={resetDemo} title="重置演示">
            <RefreshCw size={17} />
          </button>
        </div>
      </header>

      <section className="workspace">
        <aside className="leftPanel">
          <section className="panelSection">
            <h2>数据</h2>
            <label className="loadButton">
              {busy ? <LoaderCircle className="spin" size={18} /> : <FileUp size={18} />}
              <span>加载 PLY</span>
              <input
                type="file"
                accept=".ply"
                onChange={(event) => loadFile(event.target.files?.[0])}
              />
            </label>
            <button
              className="secondaryButton sampleButton"
              type="button"
              disabled={busy}
              onClick={loadSample}
            >
              <Database size={16} />
              加载示例 3DGS
            </button>
            <div className="fileBox">
              <span>{scene.name}</span>
              <small>{scene.points.length.toLocaleString()} 个高斯点</small>
            </div>
            {error && <div className="errorBox">{error}</div>}
          </section>

          <section className="panelSection">
            <h2>场景</h2>
            <ControlRow label="工作模式">
              <div className="modeToggle" role="group" aria-label="工作模式">
                <button
                  className={viewMode === "view" ? "active" : ""}
                  type="button"
                  onClick={enterViewMode}
                  disabled={!hasSplatRenderer}
                  title={objectEditActive ? "清除编辑预览并查看真实 splat 外观" : "查看真实 splat 外观"}
                >
                  真实查看
                </button>
                <button
                  className={viewMode === "edit" ? "active" : ""}
                  type="button"
                  onClick={enterEditMode}
                >
                  对象编辑
                </button>
              </div>
            </ControlRow>
            <ControlRow label="颜色模式">
              <select value={renderMode} onChange={(event) => setEditRenderMode(event.target.value)}>
                <option value="original">自身颜色</option>
                <option value="clustered">对象色诊断</option>
              </select>
            </ControlRow>
            <ControlRow label="点大小">
              <input
                type="range"
                min="0.006"
                max="0.05"
                step="0.002"
                value={pointSize}
                onChange={(event) => setPointSize(Number(event.target.value))}
              />
              <span className="value">{pointSize.toFixed(3)}</span>
            </ControlRow>
            <label className="checkRow">
              <input
                type="checkbox"
                checked={showGrid}
                onChange={(event) => setShowGrid(event.target.checked)}
              />
              显示网格
            </label>
            <label className="checkRow">
              <input
                type="checkbox"
                checked={showAxes}
                onChange={(event) => setShowAxes(event.target.checked)}
              />
              显示坐标轴
            </label>
            <label className="checkRow" title="诊断开关：对 Spark 对象 mask 边界使用可审计 feather">
              <input
                type="checkbox"
                checked={sparkObjectMaskFeathering.enabled}
                disabled={!hasSplatRenderer}
                onChange={(event) => toggleSparkObjectMaskFeathering(event.target.checked)}
              />
              柔化删除边界
            </label>
          </section>

          <section className="panelSection assetLibraryPanel">
            <div className="sectionTitleRow">
              <h2>素材库</h2>
              <span>{FEATURED_ASSETS.length} 个可加载样例</span>
            </div>
            <div className="panelTabs" role="tablist" aria-label="素材库视图">
              <button
                className={sideTab === "samples" ? "active" : ""}
                type="button"
                role="tab"
                aria-selected={sideTab === "samples"}
                onClick={() => setSideTab("samples")}
              >
                可打开样例
              </button>
              <button
                className={sideTab === "benchmark" ? "active" : ""}
                type="button"
                role="tab"
                aria-selected={sideTab === "benchmark"}
                onClick={() => setSideTab("benchmark")}
              >
                Benchmark
              </button>
            </div>
            {sideTab === "samples" ? (
              <div className="assetCards">
                {FEATURED_ASSETS.map((asset) => (
                  <article className="assetCard" key={asset.id}>
                    <div className="assetMeta">
                      <span>{asset.category}</span>
                      <span>{asset.pipelineStage}</span>
                      <span>{asset.priority}</span>
                    </div>
                    <strong>{asset.name}</strong>
                    <p>{asset.bestFor}</p>
                    <div className="assetUseCases" aria-label={`${asset.name} 用途`}>
                      {asset.useCases.map((useCase) => (
                        <span key={useCase}>{useCase}</span>
                      ))}
                    </div>
                    <div className="assetTags" aria-label={`${asset.name} 格式`}>
                      {asset.formats.slice(0, 3).map((format) => (
                        <span key={format}>{format}</span>
                      ))}
                    </div>
                    <div className="assetFooter">
                      <span className="assetStatus ready">{asset.status}</span>
                      <button
                        className="assetActionButton"
                        type="button"
                        disabled={busy}
                        onClick={() => loadAsset(asset)}
                      >
                        加载
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <BenchmarkPanel />
            )}
          </section>

          <section className="panelSection">
            <h2>视图</h2>
            <button className="secondaryButton" type="button" onClick={resetDemo}>
              <RefreshCw size={16} />
              重置演示
            </button>
            <div className="hintText">
              路线：{rendererRoute.label} / {rendererRoute.qualityLabel}。
            </div>
          </section>
        </aside>

        <section className="viewerStage" aria-label="3D 视图">
          <div
            className={`routeBadge ${rendererRoute.tone}`}
            title={rendererRoute.title}
            aria-hidden="true"
          >
            <span>{rendererRoute.label}</span>
            <strong>{rendererRoute.qualityLabel}</strong>
          </div>
          {!useSplatRenderer && !useSparkPlySourceRenderer && (
            <div className="viewportBanner">
              <strong>{rendererRoute.bannerTitle}</strong>
              <span>{renderModeText} / {activeRendererText}</span>
              {hasSplatRenderer && objectEditActive ? (
                <button className="bannerAction" type="button" onClick={enterViewMode}>
                  <Rotate3D size={15} />
                  真实 Splat
                </button>
              ) : null}
            </div>
          )}
          {useSplatRenderer || useSparkPointRenderer ? (
            <SplatViewport
              source={scene.splatSource}
              points={useSparkPointRenderer ? scene.points : null}
              shRestCoefficients={useSparkPointRenderer ? scene.shRestCoefficients : null}
              shRestCoefficientCount={useSparkPointRenderer ? scene.shRestCoefficientCount : 0}
              visibleIds={useSparkFilteredRenderer ? visibleIds : null}
              removedIds={useSparkFilteredRenderer ? removedIds : null}
              isolatedId={useSparkFilteredRenderer ? isolatedId : null}
              renderMode={renderMode}
              filtered={useSparkPointRenderer}
              reconstructRole={useSparkPlySourceRenderer ? "source" : "filter"}
              filterSource={useSparkNativeMaskRenderer ? "native-splat" : "ply-packed"}
              objectMaskFeathering={sparkObjectMaskFeathering}
              showGrid={showGrid}
              showAxes={showAxes}
              pointCount={useSparkFilteredRenderer ? visibleCount : scene.points.length}
              rendererLabel={activeRendererText}
              selectedId={useSparkFilteredRenderer ? selectedId : null}
              onSelectObject={useSparkFilteredRenderer ? selectObject : null}
            />
          ) : waitForEditRenderer ? (
            <RendererPendingViewport
              rendererContract={activeEditRenderer}
              visibleCount={visibleCount}
              renderModeLabel={renderModeText}
            />
          ) : useWebGpuTileRenderer ? (
            <WebGpuTileViewport
              points={scene.points}
              visibleIds={visibleIds}
              removedIds={removedIds}
              isolatedId={isolatedId}
              tileSmoke={webGpuRuntimeTileSmoke}
              rendererContract={activeEditRenderer}
              onSelectObject={selectObject}
              renderModeLabel={renderModeText}
              runtimeViewportAspectMode={webGpuRuntimeViewport.aspectMode}
              runtimeViewportQuality={webGpuRuntimeViewport.quality}
              runtimeViewportPixelBudget={webGpuRuntimeViewport.pixelBudget}
              onDisplaySizeChange={updateWebGpuRuntimeDisplaySize}
            />
          ) : (
            <PointCloudViewport
              points={scene.points}
              visibleIds={visibleIds}
              removedIds={removedIds}
              renderMode={renderMode}
              pointSize={pointSize}
              showGrid={showGrid}
              showAxes={showAxes}
              isolatedId={isolatedId}
              selectedId={selectedId}
              onSelectObject={selectObject}
              renderModeLabel={renderModeText}
              rendererContract={activeEditRenderer}
            />
          )}
        </section>

        <aside className="rightPanel">
          <section className="panelSection inspectorHead">
            <h2>对象检查器</h2>
            <div className="metricGrid">
              <Metric label="对象" value={summary.length} />
              <Metric label="高斯点" value={scene.points.length.toLocaleString()} />
              <Metric label="可见" value={visibleCount.toLocaleString()} />
            </div>
          </section>

          <section className="objectList" aria-label="对象列表">
            <div className="objectListHead">
              <span>对象 ID</span>
              <span>对象色</span>
              <span>点数</span>
              <span>可见</span>
            </div>
            {summary.map((item) => (
              <div
                key={item.id}
                className={`objectRow ${selectedId === item.id ? "selected" : ""} ${
                  removedIds.has(item.id) ? "removed" : ""
                }`}
              >
                <button
                  className="objectSelectButton"
                  type="button"
                  aria-pressed={selectedId === item.id}
                  onClick={() => selectObject(item.id)}
                >
                  <span className="idCell">{item.id}</span>
                  <span
                    className="swatch"
                    style={{ backgroundColor: rgbToCss(item.color) }}
                    aria-hidden="true"
                  />
                  <span>{item.count.toLocaleString()}</span>
                </button>
                <button
                  className="eyeButton"
                  type="button"
                  aria-pressed={visibleIds.has(item.id)}
                  aria-label={`${visibleIds.has(item.id) ? "隐藏" : "显示"}对象 ${item.id}`}
                  onClick={() => toggleVisible(item.id)}
                >
                  {visibleIds.has(item.id) ? <Eye size={16} /> : <EyeOff size={16} />}
                </button>
              </div>
            ))}
          </section>

          <section className="panelSection actionPanel">
            <h2>对象操作</h2>
            <button
              className="accentButton"
              type="button"
              disabled={selectedId === null}
              onClick={isolateSelected}
            >
              <Scissors size={16} />
              只看所选
            </button>
            <button className="secondaryButton" type="button" onClick={() => setIsolatedId(null)}>
              <Eye size={16} />
              取消隔离
            </button>
            <button
              className="dangerButton"
              type="button"
              disabled={selectedId === null || removedIds.has(selectedId)}
              onClick={removeSelected}
            >
              <Trash2 size={16} />
              预览删除
            </button>
          </section>

          <section className="panelSection statePanel">
            <h2>渲染状态</h2>
            <StateRow label="工作模式" value={modeText} />
            <StateRow label="展示路线" value={rendererRoute.label} />
            <StateRow label="渲染器" value={activeRendererText} />
            <StateRow label="颜色用途" value={rendererRoute.colorRoleLabel} />
            <StateRow label="预览边界" value={rendererRoute.sourceBoundaryLabel} />
            <StateRow label="删除结果" value={rendererRoute.sourceResultLabel} />
            <StateRow label="边界柔化" value={sparkObjectMaskFeatheringLabel} />
            <StateRow label="质量解释" value={hardMaskQuality.label} />
            <StateRow label="目标渲染器" value={activeEditRenderer.targetRendererLabel} />
            <StateRow label="目标状态" value={`${activeEditRenderer.targetGate} / ${activeEditRenderer.targetGateReason}`} />
            <StateRow
              label="存储门禁"
              value={`${activeEditRenderer.storageLimitGate} / ${activeEditRenderer.storageEstimatedMaxBufferKey || "none"} ${formatBytes(activeEditRenderer.storageEstimatedMaxBufferByteSize)}`}
            />
            <StateRow label="WebGPU" value={activeEditRenderer.webGpuLabel} />
            <StateRow label="回退原因" value={activeEditRenderer.fallbackReason} />
            <StateRow
              label="WebGPU pack"
              value={`${activeEditRenderer.packedGaussians.toLocaleString()} / ${activeEditRenderer.objectCount}`}
            />
            <StateRow
              label="Tile bins"
              value={`${activeEditRenderer.activeTileCount.toLocaleString()} / ${activeEditRenderer.tileCount.toLocaleString()}`}
            />
            <StateRow
              label="Tile capacity"
              value={`${activeEditRenderer.tileCapacityStatus} / ${activeEditRenderer.tileOverflowTileCount.toLocaleString()} tiles`}
            />
            <StateRow
              label="Tile resolve"
              value={`${activeEditRenderer.resolvedTileCount.toLocaleString()} / ${activeEditRenderer.resolveChecksum}`}
            />
            <StateRow
              label="Object state"
              value={`${activeEditRenderer.objectStateVisibleObjects.toLocaleString()} / ${activeEditRenderer.objectStateChecksum}`}
            />
            <StateRow label="Tile overflow" value={activeEditRenderer.tileOverflowCount} />
            <StateRow label="模式" value={renderModeText} />
            <StateRow label="所选对象" value={selectedId ?? "无"} />
            <StateRow label="已删除对象" value={removedIds.size} />
            <StateRow label="已删除点数" value={removedPointCount(summary, removedIds)} />
            <button className="secondaryButton" type="button" onClick={restoreRemoved}>
              清空删除预览
            </button>
          </section>
        </aside>
      </section>

      <footer className="statusBar">
        <span>状态：{busy ? "加载中" : error ? "错误" : "就绪"}</span>
        <span>模式：{modeText}</span>
        <span>路线：{rendererRoute.label}</span>
        <span>渲染器：{activeRendererText}</span>
        <span>WebGPU：{activeEditRenderer.webGpuLabel}</span>
        <span>高斯点：{scene.points.length.toLocaleString()}</span>
        <span>可见：{visibleCount.toLocaleString()}</span>
        <span>所选：{selectedId ?? "无"}</span>
      </footer>
    </main>
  );
}

function readInitialSparkObjectMaskFeathering() {
  if (typeof window === "undefined") {
    return normalizeSparkObjectMaskFeathering({
      enabled: false,
      opacity: UI_SPARK_OBJECT_MASK_FEATHER_OPACITY,
    });
  }
  const params = new URLSearchParams(window.location.search);
  const mode = String(params.get("spark-object-mask-feather") ?? "off").toLowerCase();
  const enabled = ["1", "true", "yes", "on"].includes(mode);
  return normalizeSparkObjectMaskFeathering({
    enabled,
    radius: Number(params.get("spark-object-mask-feather-radius") ?? 0),
    opacity: Number(
      params.get("spark-object-mask-feather-opacity") ?? UI_SPARK_OBJECT_MASK_FEATHER_OPACITY,
    ),
  });
}

function RendererPendingViewport({ rendererContract, visibleCount, renderModeLabel }) {
  return (
    <div
      className="viewport"
      data-renderer="renderer-pending"
      data-renderer-target={rendererContract?.targetRendererId ?? "webgpu-tile"}
      data-renderer-fallback-reason=""
      data-webgpu-target-gate={rendererContract?.targetGate ?? "blocked"}
      data-webgpu-target-gate-reason={rendererContract?.targetGateReason ?? "webgpu-capability-detecting"}
      data-webgpu-target-gate-blocker={rendererContract?.targetGateBlocker ?? "webgpu-capability"}
      data-webgpu-status="pending"
      data-visible-count={visibleCount}
    >
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
          <strong>pending</strong>
        </div>
      </div>
    </div>
  );
}

function BenchmarkPanel() {
  return (
    <div className="benchmarkPanel">
      <div className="benchmarkGates" aria-label="SEMANTIC benchmark gates">
        {BENCHMARK_GATES.map((gate) => (
          <div className="benchmarkGate" key={gate.label}>
            <span>{gate.label}</span>
            <strong>{gate.value}</strong>
          </div>
        ))}
      </div>
      <div className="benchmarkStatLine">
        <BarChart3 size={15} />
        <span>Cross-scene rows: 9</span>
      </div>
      <div className="benchmarkTable" aria-label="Splatfacto scene benchmark">
        <div className="benchmarkTableHead">
          <span>Scene</span>
          <span>ARI</span>
          <span>Render</span>
          <span>Held</span>
        </div>
        {BENCHMARK_SCENES.map((scene) => (
          <div className="benchmarkRow" key={scene.id}>
            <strong>{scene.label}</strong>
            <span>{scene.ari.toFixed(3)}</span>
            <span>{scene.render.toFixed(3)}</span>
            <span>{scene.heldout.toFixed(3)}</span>
          </div>
        ))}
      </div>
      <div className="benchmarkFootnote">SEMANTIC-003 / paper gate passed</div>
    </div>
  );
}

function ControlRow({ label, children }) {
  return (
    <label className="controlRow">
      <span>{label}</span>
      <div>{children}</div>
    </label>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function StateRow({ label, value }) {
  return (
    <div className="stateRow">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function allIds(points) {
  return new Set(points.map((point) => point.objectId));
}

function renderModeLabel(mode) {
  if (mode === "original") return "原始颜色（编辑预览）";
  return "对象色（编辑预览）";
}

function rendererRouteContract({
  renderMode,
  objectEditActive,
  useSplatRenderer,
  useSparkPlySourceRenderer,
  useSparkFilteredRenderer,
  useSparkNativeMaskRenderer,
  useWebGpuTileRenderer,
}) {
  const colorModeRole =
    renderMode === "clustered" ? "diagnostic-object-color" : "source-color";
  const colorRoleLabel = renderMode === "clustered" ? "对象色诊断" : "自身颜色";
  const sourceBoundary =
    renderMode === "clustered"
      ? "diagnostic-object-color"
      : objectEditActive
        ? "hard-object-mask-no-reoptimize"
        : "source-splat";
  const sourceBoundaryLabel =
    sourceBoundary === "hard-object-mask-no-reoptimize"
      ? "硬 mask，无补洞"
      : sourceBoundary === "source-splat"
        ? "原始 Splat"
        : "调试色";
  const sourceResult =
    sourceBoundary === "hard-object-mask-no-reoptimize"
      ? "hard-mask-no-inpaint"
      : sourceBoundary === "source-splat"
        ? "source-splat"
        : "diagnostic-preview";
  const sourceResultLabel =
    sourceResult === "hard-mask-no-inpaint"
      ? "源色 mask 预览"
      : sourceResult === "source-splat"
        ? "完整源 Splat"
        : "诊断预览";

  const base = {
    colorModeRole,
    colorRoleLabel,
    sourceBoundary,
    sourceBoundaryLabel,
    sourceResult,
    sourceResultLabel,
  };

  if (renderMode === "clustered") {
    return {
      ...base,
      id: "diagnostic-object-color",
      kind: "diagnostic",
      label: "对象色诊断",
      qualityId: "debug-object-color",
      qualityLabel: "调试分组",
      bannerTitle: "对象色诊断",
      tone: "diagnostic",
      title: "诊断路线：用于检查 object_id 分组，不代表商用展示外观",
    };
  }

  if (useSplatRenderer) {
    return {
      ...base,
      id: "spark-original-view",
      kind: "commercial",
      label: "商用 Spark",
      qualityId: "source-splat",
      qualityLabel: "原始 Splat",
      bannerTitle: "Spark 源色",
      tone: "commercial",
      title: "商业展示默认路线：Spark 原始 Splat",
    };
  }

  if (useSparkPlySourceRenderer) {
    return {
      ...base,
      id: "spark-ply-sh-source",
      kind: "commercial",
      label: "商用 Spark SH",
      qualityId: "ply-sh-source",
      qualityLabel: "PLY SH 源",
      bannerTitle: "Spark SH 源",
      tone: "commercial",
      title: "商业展示路线：保留 SH-heavy 本地训练输出的 SH 系数",
    };
  }

  if (useSparkFilteredRenderer) {
    return {
      ...base,
      id: useSparkNativeMaskRenderer ? "spark-native-mask" : "spark-packed-sh-mask",
      kind: "commercial",
      label: "商用 Spark",
      qualityId: useSparkNativeMaskRenderer ? "native-splat-mask" : "packed-sh-mask",
      qualityLabel: useSparkNativeMaskRenderer ? "原生 Splat mask" : "SH packed mask",
      bannerTitle: "Spark 源色编辑",
      tone: "commercial",
      title: "商业展示路线：自身颜色 + hard object mask；删除后不补洞、不重优化",
    };
  }

  if (useWebGpuTileRenderer) {
    return {
      ...base,
      id: "webgpu-c-path-diagnostic",
      kind: "diagnostic",
      label: "WebGPU C-path",
      qualityId: "tile-diagnostic-preview",
      qualityLabel: "诊断预览",
      bannerTitle: "WebGPU 诊断预览",
      tone: "diagnostic",
      title: "C-path 诊断路线：验证 tile renderer，不是当前商用默认外观",
    };
  }

  return {
    ...base,
    id: "gaussian-oit-fallback",
    kind: "fallback",
    label: "Fallback",
    qualityId: "gaussian-oit-preview",
    qualityLabel: "Gaussian OIT",
    bannerTitle: "Fallback 预览",
    tone: "fallback",
    title: "兼容回退路线：近似编辑预览",
  };
}

function hardMaskQualityContract(scene, rendererRoute) {
  if (rendererRoute.sourceBoundary === "source-splat") {
    return {
      interpretation: "source-splat",
      label: "原始 Spark 高斯",
      source: "route-state",
    };
  }
  if (rendererRoute.sourceBoundary === "diagnostic-object-color") {
    return {
      interpretation: "diagnostic-object-color",
      label: "对象色诊断",
      source: "route-state",
    };
  }

  const reportBacked = HARD_MASK_QUALITY_BY_ASSET[scene?.assetId];
  if (reportBacked) return reportBacked;

  return {
    interpretation: "hard-mask-quality-unmeasured",
    label: "硬 mask 待审计",
    source: "route-state",
  };
}

function readWebGpuRuntimeProbe() {
  if (typeof window === "undefined") return "full";
  return normalizeWebGpuRuntimeProbe(
    new URLSearchParams(window.location.search).get("webgpu-probe"),
  );
}

function readWebGpuRuntimeViewportRequest() {
  if (typeof window === "undefined") {
    return { size: WEBGPU_RUNTIME_VIEWPORT_SIZE, explicit: false };
  }
  const value = Number(
    new URLSearchParams(window.location.search).get("webgpu-viewport-size"),
  );
  if (!Number.isFinite(value) || value <= 0) {
    return { size: WEBGPU_RUNTIME_VIEWPORT_SIZE, explicit: false };
  }
  return {
    size: Math.min(
      WEBGPU_RUNTIME_MAX_VIEWPORT_SIZE,
      Math.max(WEBGPU_RUNTIME_MIN_VIEWPORT_SIZE, Math.round(value)),
    ),
    explicit: true,
  };
}

function readWebGpuCoverageTuning() {
  if (typeof window === "undefined") {
    return normalizeWebGpuCoverageTuning();
  }
  const params = new URLSearchParams(window.location.search);
  return normalizeWebGpuCoverageTuning({
    footprintScale: params.get("webgpu-footprint-scale"),
    maxAnisotropy: params.get("webgpu-covariance-max-anisotropy"),
  });
}

function readWebGpuDepthSortTuning() {
  if (typeof window === "undefined") {
    return normalizeWebGpuDepthSortTuning();
  }
  const params = new URLSearchParams(window.location.search);
  return normalizeWebGpuDepthSortTuning({
    depthBins: params.get("webgpu-depth-bins"),
    depthAlphaMode: params.get("webgpu-depth-alpha-mode"),
  });
}

function readWebGpuCameraTuning() {
  if (typeof window === "undefined") {
    return normalizeWebGpuCameraTuning();
  }
  const params = new URLSearchParams(window.location.search);
  return normalizeWebGpuCameraTuning({
    cameraMode: params.get("webgpu-camera-mode"),
  });
}

function readWebGpuColorTuning() {
  if (typeof window === "undefined") {
    return normalizeWebGpuColorTuning();
  }
  const params = new URLSearchParams(window.location.search);
  return normalizeWebGpuColorTuning({
    colorMode: params.get("webgpu-color-mode"),
  });
}

function readSparkReconstructProbe() {
  if (typeof window === "undefined") return false;
  const value = new URLSearchParams(window.location.search).get("spark-reconstruct-probe");
  return ["1", "true", "yes", "on"].includes(String(value ?? "").toLowerCase());
}

function readSparkPlySourceMode() {
  if (typeof window === "undefined") return "auto";
  const value = String(
    new URLSearchParams(window.location.search).get("spark-ply-source") ?? "auto",
  ).toLowerCase();
  if (["0", "false", "no", "off"].includes(value)) return "off";
  if (["1", "true", "yes", "on", "force"].includes(value)) return "force";
  return "auto";
}

function readSparkNativeMaskMode() {
  if (typeof window === "undefined") return "auto";
  const params = new URLSearchParams(window.location.search);
  const value = String(
    params.get("spark-native-mask") ?? params.get("spark-object-source") ?? "auto",
  ).toLowerCase();
  if (["0", "false", "no", "off", "packed", "ply", "ply-packed"].includes(value)) {
    return "off";
  }
  if (["1", "true", "yes", "on", "native", "force"].includes(value)) return "force";
  return "auto";
}

function readSparkFilteredEditEnabled() {
  if (typeof window === "undefined") return true;
  const value = String(
    new URLSearchParams(window.location.search).get("spark-filtered-edit") ?? "auto",
  ).toLowerCase();
  return !["0", "false", "no", "off"].includes(value);
}

function sceneHasShRestSource(scene) {
  const coefficientCount = Number(scene?.shRestCoefficientCount ?? 0);
  if (!Number.isFinite(coefficientCount) || coefficientCount < 9) return false;
  const gaussianCount = scene?.points?.length ?? 0;
  const shRestCoefficients = scene?.shRestCoefficients;
  return Boolean(
    shRestCoefficients &&
      typeof shRestCoefficients.length === "number" &&
      shRestCoefficients.length >= gaussianCount * coefficientCount,
  );
}

function buildWebGpuRuntimeViewport({ probe, request, displaySize, gaussianCount }) {
  if (probe === WEBGPU_RUNTIME_PROBE_TINY_PIXEL_OUTPUT) {
    return {
      width: WEBGPU_RUNTIME_PROBE_TINY_VIEWPORT_SIZE,
      height: WEBGPU_RUNTIME_PROBE_TINY_VIEWPORT_SIZE,
      aspectMode: "tiny-square",
      quality: "diagnostic-tiny",
      pixelBudget: WEBGPU_RUNTIME_PROBE_TINY_VIEWPORT_SIZE ** 2,
    };
  }
  if (request.explicit) {
    return {
      width: request.size,
      height: request.size,
      aspectMode: "explicit-square",
      quality: "explicit-square",
      pixelBudget: request.size ** 2,
    };
  }
  const quality = webGpuRuntimeQuality(gaussianCount);
  const displayWidth = Number(displaySize?.width) || 0;
  const displayHeight = Number(displaySize?.height) || 0;
  if (displayWidth <= 0 || displayHeight <= 0) {
    return {
      width: quality.size,
      height: quality.size,
      aspectMode: "adaptive-square-pending-display",
      quality: quality.label,
      pixelBudget: quality.pixelBudget,
    };
  }
  const aspect = Math.min(4, Math.max(0.25, displayWidth / displayHeight));
  const displayArea = Math.max(
    WEBGPU_RUNTIME_MIN_VIEWPORT_SIZE ** 2,
    displayWidth * displayHeight,
  );
  const area = Math.min(quality.pixelBudget, displayArea);
  return {
    width: clampViewportSize(roundToViewportTile(Math.sqrt(area * aspect))),
    height: clampViewportSize(roundToViewportTile(Math.sqrt(area / aspect))),
    aspectMode: "display-aspect-adaptive",
    quality: quality.label,
    pixelBudget: quality.pixelBudget,
  };
}

function webGpuRuntimeQuality(gaussianCount) {
  const count = Math.max(0, Number(gaussianCount) || 0);
  if (count <= WEBGPU_RUNTIME_HIGH_MAX_GAUSSIANS) {
    return {
      label: "adaptive-high-512",
      size: WEBGPU_RUNTIME_HIGH_VIEWPORT_SIZE,
      pixelBudget: WEBGPU_RUNTIME_HIGH_VIEWPORT_SIZE ** 2,
    };
  }
  if (count <= WEBGPU_RUNTIME_MEDIUM_MAX_GAUSSIANS) {
    return {
      label: "adaptive-medium-384",
      size: WEBGPU_RUNTIME_MEDIUM_VIEWPORT_SIZE,
      pixelBudget: WEBGPU_RUNTIME_MEDIUM_VIEWPORT_SIZE ** 2,
    };
  }
  return {
    label: "adaptive-safe-320",
    size: WEBGPU_RUNTIME_SAFE_VIEWPORT_SIZE,
    pixelBudget: WEBGPU_RUNTIME_SAFE_VIEWPORT_SIZE ** 2,
  };
}

function roundToViewportTile(value) {
  return Math.max(
    WEBGPU_RUNTIME_VIEWPORT_TILE_SIZE,
    Math.round(value / WEBGPU_RUNTIME_VIEWPORT_TILE_SIZE) * WEBGPU_RUNTIME_VIEWPORT_TILE_SIZE,
  );
}

function clampViewportSize(value) {
  return Math.min(
    WEBGPU_RUNTIME_MAX_VIEWPORT_SIZE,
    Math.max(WEBGPU_RUNTIME_MIN_VIEWPORT_SIZE, value),
  );
}

function formatBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${Math.round(bytes)} B`;
}

function summarize(points) {
  const counts = new Map();
  const colors = new Map();
  for (const point of points) {
    counts.set(point.objectId, (counts.get(point.objectId) ?? 0) + 1);
    colors.set(point.objectId, point.objectColor);
  }
  return [...counts.entries()]
    .sort(([a], [b]) => a - b)
    .map(([id, count]) => ({ id, count, color: colors.get(id) }));
}

function removedPointCount(summary, removedIds) {
  return summary
    .filter((item) => removedIds.has(item.id))
    .reduce((total, item) => total + item.count, 0)
    .toLocaleString();
}
