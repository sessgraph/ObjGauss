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
import { useMemo, useState } from "react";

import { ASSET_LIBRARY, featuredAssets } from "./assetLibrary.js";
import { parsePly, parsePlyFile } from "./ply.js";
import PointCloudViewport from "./PointCloudViewport.jsx";
import { rgbToCss } from "./palette.js";
import { createSampleScene } from "./sampleScene.js";
import SplatViewport from "./SplatViewport.jsx";

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
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const summary = useMemo(() => summarize(scene.points), [scene.points]);
  const renderModeText = renderModeLabel(renderMode);
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
  const useSplatRenderer = viewMode === "view" && canUseSplatRenderer;
  const activeRendererText = useSplatRenderer ? "真实 Splat" : "软点云编辑";
  const modeText = viewMode === "view" ? "真实查看" : "对象编辑";
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
        name: asset.fileName ?? asset.name,
        points: cloud.points,
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
    <main className="appShell">
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
            title="查看真实 splat 外观"
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
            <option value="clustered">对象聚类色</option>
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
                <option value="clustered">对象聚类色</option>
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
              当前：{modeText} / {activeRendererText}。对象操作会进入点云编辑预览。
            </div>
          </section>
        </aside>

        <section className="viewerStage" aria-label="3D 视图">
          {!useSplatRenderer && (
            <div className="viewportBanner">
              <strong>对象编辑预览</strong>
              <span>点击软点云或右侧列表选择对象；隔离、隐藏、删除在这里预览。</span>
            </div>
          )}
          {useSplatRenderer ? (
            <SplatViewport
              source={scene.splatSource}
              showGrid={showGrid}
              showAxes={showAxes}
              pointCount={scene.points.length}
              rendererLabel={activeRendererText}
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
            <StateRow label="渲染器" value={activeRendererText} />
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
        <span>渲染器：{activeRendererText}</span>
        <span>高斯点：{scene.points.length.toLocaleString()}</span>
        <span>可见：{visibleCount.toLocaleString()}</span>
        <span>所选：{selectedId ?? "无"}</span>
      </footer>
    </main>
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
  if (mode === "original") return "自身颜色";
  return "对象聚类色";
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
