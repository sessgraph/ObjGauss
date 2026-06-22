import {
  BoxSelect,
  Database,
  Eye,
  EyeOff,
  ExternalLink,
  FileUp,
  Focus,
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

export default function App() {
  const [scene, setScene] = useState(() => createSampleScene());
  const [rendererKind, setRendererKind] = useState("splat");
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
  const objectEditActive = useMemo(
    () =>
      removedIds.size > 0 ||
      isolatedId !== null ||
      sceneObjectIds.size !== visibleIds.size ||
      [...sceneObjectIds].some((id) => !visibleIds.has(id)),
    [isolatedId, removedIds, sceneObjectIds, visibleIds],
  );
  const canUseSplatRenderer = Boolean(scene.splatSource) && renderMode === "original" && !objectEditActive;
  const useSplatRenderer = rendererKind === "splat" && canUseSplatRenderer;
  const activeRendererText = useSplatRenderer ? "真实 Splat" : "点云编辑";
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
    setSelectedId(ids.values().next().value ?? null);
    setIsolatedId(null);
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
    setRenderMode("original");
    setError("");
  };

  const toggleVisible = (id) => {
    setVisibleIds((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const removeSelected = () => {
    if (selectedId === null) return;
    setRemovedIds((current) => new Set([...current, selectedId]));
    if (isolatedId === selectedId) setIsolatedId(null);
  };

  const restoreRemoved = () => {
    setRemovedIds(new Set());
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

        <nav className="toolStrip" aria-label="查看器工具">
          <button className="toolButton active" type="button" title="旋转视图">
            <Rotate3D size={17} />
            <span>旋转</span>
          </button>
          <button className="toolButton" type="button" title="框选对象">
            <BoxSelect size={17} />
            <span>框选</span>
          </button>
          <button className="toolButton" type="button" title="聚焦所选">
            <Focus size={17} />
            <span>聚焦</span>
          </button>
        </nav>

        <div className="topActions">
          <select
            value={rendererKind}
            onChange={(event) => setRendererKind(event.target.value)}
            aria-label="渲染器"
          >
            <option value="splat">真实 Splat</option>
            <option value="points">点云编辑</option>
          </select>
          <select
            value={renderMode}
            onChange={(event) => setRenderMode(event.target.value)}
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
            <ControlRow label="渲染器">
              <select
                value={rendererKind}
                onChange={(event) => setRendererKind(event.target.value)}
              >
                <option value="splat">真实 Splat</option>
                <option value="points">点云编辑</option>
              </select>
            </ControlRow>
            <ControlRow label="颜色模式">
              <select value={renderMode} onChange={(event) => setRenderMode(event.target.value)}>
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
              <span>{ASSET_LIBRARY.length} 个来源</span>
            </div>
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
                    <span className={asset.localPath ? "assetStatus ready" : "assetStatus"}>
                      {asset.status}
                    </span>
                    {asset.localPath ? (
                      <button
                        className="assetActionButton"
                        type="button"
                        disabled={busy}
                        onClick={() => loadAsset(asset)}
                      >
                        加载
                      </button>
                    ) : (
                      <a
                        className="assetLink"
                        href={asset.sourceUrl}
                        target="_blank"
                        rel="noreferrer"
                      >
                        来源
                        <ExternalLink size={13} />
                      </a>
                    )}
                  </div>
                </article>
              ))}
            </div>
          </section>

          <section className="panelSection">
            <h2>视图</h2>
            <button className="secondaryButton" type="button" onClick={resetDemo}>
              <RefreshCw size={16} />
              重置演示
            </button>
            <div className="hintText">
              鼠标拖拽旋转，滚轮缩放。默认显示高斯自身颜色，也可以切换到对象聚类色查看分割结果。
            </div>
          </section>
        </aside>

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
            renderModeLabel={renderModeText}
          />
        )}

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
              <button
                key={item.id}
                className={`objectRow ${selectedId === item.id ? "selected" : ""} ${
                  removedIds.has(item.id) ? "removed" : ""
                }`}
                type="button"
                onClick={() => setSelectedId(item.id)}
              >
                <span className="idCell">{item.id}</span>
                <span
                  className="swatch"
                  style={{ backgroundColor: rgbToCss(item.color) }}
                  aria-hidden="true"
                />
                <span>{item.count.toLocaleString()}</span>
                <span
                  className="eyeButton"
                  role="switch"
                  aria-checked={visibleIds.has(item.id)}
                  onClick={(event) => {
                    event.stopPropagation();
                    toggleVisible(item.id);
                  }}
                >
                  {visibleIds.has(item.id) ? <Eye size={16} /> : <EyeOff size={16} />}
                </span>
              </button>
            ))}
          </section>

          <section className="panelSection actionPanel">
            <h2>对象操作</h2>
            <button
              className="accentButton"
              type="button"
              disabled={selectedId === null}
              onClick={() => setIsolatedId(selectedId)}
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
              disabled={selectedId === null}
              onClick={removeSelected}
            >
              <Trash2 size={16} />
              预览删除
            </button>
          </section>

          <section className="panelSection statePanel">
            <h2>渲染状态</h2>
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
        <span>渲染器：{activeRendererText}</span>
        <span>高斯点：{scene.points.length.toLocaleString()}</span>
        <span>可见：{visibleCount.toLocaleString()}</span>
        <span>所选：{selectedId ?? "无"}</span>
      </footer>
    </main>
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
