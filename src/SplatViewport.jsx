import { useEffect, useMemo, useRef, useState } from "react";
import { PackedSplats, SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  buildPackedShExtra,
  extractPackedShExtra,
  shDcRgb01,
} from "./sparkPackedSh.js";

const SPARK_DISPLAY_PACKED_CACHE_LIMIT = 4;
const SPARK_DISPLAY_CACHE_MODE = "visible-index-lru-v1";

export default function SplatViewport({
  source,
  points = null,
  shRestCoefficients = null,
  shRestCoefficientCount = 0,
  visibleIds = null,
  removedIds = null,
  isolatedId = null,
  renderMode = "original",
  filtered = false,
  showGrid,
  showAxes,
  pointCount,
  rendererLabel,
  reconstructRole = "filter",
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const gridRef = useRef(null);
  const axesRef = useRef(null);
  const [status, setStatus] = useState("加载中");
  const [packedStats, setPackedStats] = useState(() => emptyPackedStats());

  const sourceKey = useMemo(() => {
    if (filtered) {
      const visibleKey = visibleIds ? [...visibleIds].sort((left, right) => left - right).join(",") : "";
      const removedKey = removedIds ? [...removedIds].sort((left, right) => left - right).join(",") : "";
      return [
        "filtered",
        points?.length ?? 0,
        reconstructRole,
        visibleKey,
        removedKey,
        isolatedId ?? "all",
        renderMode,
      ].join(":");
    }
    if (source?.url) return source.url;
    if (source?.fileName) return `${source.fileName}:${source.fileBytes?.byteLength ?? 0}`;
    return "none";
  }, [filtered, isolatedId, points, reconstructRole, removedIds, renderMode, source, visibleIds]);

  const filteredStats = useMemo(
    () =>
      filtered
        ? buildFilteredSplatStats({
            points,
            visibleIds,
            removedIds,
            isolatedId,
            renderMode,
            reconstructRole,
          })
        : {
            mode: "none",
            visibleGaussians: 0,
            filteredGaussians: 0,
            hiddenObjects: 0,
            removedObjects: 0,
            isolatedObject: "",
            colorSourceGaussians: 0,
            objectColorGaussians: 0,
          },
    [filtered, isolatedId, points, reconstructRole, removedIds, renderMode, visibleIds],
  );
  const objectFilter = filtered
    ? sparkObjectFilter({ filteredStats, packedStats, reconstructRole })
    : "none";
  const sparkFilterMode = filtered
    ? reconstructRole === "source"
      ? "ply-source"
      : "ply-reconstruct"
    : "none";

  const packedCache = useMemo(() => {
    if (!filtered) return null;
    return buildPackedSplatCache({
      points,
      renderMode,
      shRestCoefficients,
      shRestCoefficientCount,
    });
  }, [filtered, points, renderMode, shRestCoefficients, shRestCoefficientCount]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101316);
    scene.fog = new THREE.Fog(0x101316, 8, 18);

    const camera = new THREE.PerspectiveCamera(58, 1, 0.01, 1000);
    camera.position.set(3.6, 2.6, 3.6);

    const renderer = new THREE.WebGLRenderer({
      antialias: false,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x101316, 1);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);

    const spark = new SparkRenderer({
      renderer,
      sortRadial: false,
      maxStdDev: Math.sqrt(8),
    });
    scene.add(spark);

    const grid = new THREE.GridHelper(6, 24, 0x42505c, 0x24303a);
    grid.position.y = -0.92;
    scene.add(grid);

    const axes = new THREE.AxesHelper(1.05);
    axes.position.set(2.45, -0.88, -2.4);
    scene.add(axes);

    sceneRef.current = scene;
    cameraRef.current = camera;
    controlsRef.current = controls;
    gridRef.current = grid;
    axesRef.current = axes;

    const resize = () => {
      const width = Math.max(container.clientWidth, 1);
      const height = Math.max(container.clientHeight, 1);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    return () => {
      renderer.setAnimationLoop(null);
      observer.disconnect();
      controls.dispose();
      spark.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!scene || !camera || !controls || (!source && !filtered)) return undefined;

    let disposed = false;
    setStatus(filtered ? "构建中" : "加载中");
    setPackedStats(filtered ? pendingPackedStats(packedCache) : emptyPackedStats());

    let displayPackedSplats = null;
    if (filtered && packedCache?.packedSplats) {
      const visibleIndices = visiblePointIndices({
        points,
        visibleIds,
        removedIds,
        isolatedId,
      });
      const displayResult = getCachedDisplayPackedSplats({
        packedCache,
        visibleIndices,
      });
      displayPackedSplats = displayResult.packedSplats;
      setPackedStats({
        route: packedCache.route,
        baseGaussians: packedCache.baseGaussians,
        visibleIndices: visibleIndices.length,
        baseBuildMs: packedCache.baseBuildMs,
        extractMs: displayResult.extractMs,
        shRestSourceGaussians: packedCache.shRestSourceGaussians,
        shRestPreservedGaussians: packedCache.shRestPreservedGaussians,
        shRestPreserved: packedCache.shRestPreserved,
        shRestCoefficientCount: packedCache.shRestCoefficientCount,
        shDegree: packedCache.shDegree,
        displayCacheMode: displayResult.cacheMode,
        displayCacheKey: displayResult.cacheKey,
        displayCacheHit: displayResult.cacheHit,
        displayCacheSize: displayResult.cacheSize,
        displayCacheHits: displayResult.cacheHits,
        displayCacheMisses: displayResult.cacheMisses,
        displayCacheEvictions: displayResult.cacheEvictions,
      });
    }

    const splat = filtered
      ? new SplatMesh({
          packedSplats: displayPackedSplats,
          onLoad: () => {
            if (!disposed) setStatus("就绪");
          },
        })
      : new SplatMesh({
          url: source.url,
          fileBytes: source.fileBytes,
          fileName: source.fileName,
          onProgress: (event) => {
            if (!event.lengthComputable || event.total === 0) return;
            const percent = Math.round((event.loaded / event.total) * 100);
            setStatus(`${percent}%`);
          },
          onLoad: () => {
            if (!disposed) setStatus("就绪");
          },
        });

    scene.add(splat);
    splat.initialized
      .then(() => {
        if (disposed) return;
        setStatus("就绪");
        frameSplat(splat, camera, controls, scene);
      })
      .catch((error) => {
        if (disposed) return;
        console.error(error);
        setStatus("加载失败");
      });

    return () => {
      disposed = true;
      scene.remove(splat);
      disposeSplatMesh(splat, filtered ? displayPackedSplats : null);
    };
  }, [filtered, isolatedId, packedCache, points, removedIds, renderMode, source, sourceKey, visibleIds]);

  useEffect(() => {
    return () => {
      disposeDisplayPackedCache(packedCache?.displayCache);
      packedCache?.packedSplats?.dispose();
    };
  }, [packedCache]);

  useEffect(() => {
    if (gridRef.current) gridRef.current.visible = showGrid;
  }, [showGrid]);

  useEffect(() => {
    if (axesRef.current) axesRef.current.visible = showAxes;
  }, [showAxes]);

  return (
    <div
      className="viewport splatViewport"
      data-renderer="spark-splat"
      data-object-filter={objectFilter}
      data-spark-filter-mode={sparkFilterMode}
      data-spark-filter-status={status === "就绪" ? "ready" : "pending"}
      data-spark-visible-gaussians={filteredStats.visibleGaussians}
      data-spark-filtered-gaussians={filteredStats.filteredGaussians}
      data-spark-hidden-objects={filteredStats.hiddenObjects}
      data-spark-removed-objects={filteredStats.removedObjects}
      data-spark-isolated-object={filteredStats.isolatedObject}
      data-spark-color-mode={renderMode}
      data-spark-color-source-gaussians={filteredStats.colorSourceGaussians}
      data-spark-color-object-gaussians={filteredStats.objectColorGaussians}
      data-spark-reconstruct-source={packedStats.route}
      data-spark-packed-base-gaussians={packedStats.baseGaussians}
      data-spark-packed-visible-indices={packedStats.visibleIndices}
      data-spark-packed-base-build-ms={formatMillis(packedStats.baseBuildMs)}
      data-spark-packed-extract-ms={formatMillis(packedStats.extractMs)}
      data-spark-display-cache-mode={packedStats.displayCacheMode}
      data-spark-display-cache-key={packedStats.displayCacheKey}
      data-spark-display-cache-hit={String(packedStats.displayCacheHit)}
      data-spark-display-cache-size={packedStats.displayCacheSize}
      data-spark-display-cache-hits={packedStats.displayCacheHits}
      data-spark-display-cache-misses={packedStats.displayCacheMisses}
      data-spark-display-cache-evictions={packedStats.displayCacheEvictions}
      data-spark-sh-rest-source-gaussians={packedStats.shRestSourceGaussians}
      data-spark-sh-rest-preserved-gaussians={packedStats.shRestPreservedGaussians}
      data-spark-sh-rest-preserved={String(packedStats.shRestPreserved)}
      data-spark-sh-rest-coefficients={packedStats.shRestCoefficientCount}
      data-spark-sh-degree={packedStats.shDegree}
      ref={containerRef}
    >
      <div className="viewportHud">
        <div>
          <span className="hudLabel">SPLAT</span>
          <strong>{pointCount.toLocaleString()}</strong>
        </div>
        <div>
          <span className="hudLabel">渲染器</span>
          <strong>{rendererLabel}</strong>
        </div>
        {filtered ? (
          <div>
            <span className="hudLabel">路径</span>
            <strong>{sparkPathLabel({ objectFilter, packedStats })}</strong>
          </div>
        ) : null}
        <div>
          <span className="hudLabel">状态</span>
          <strong>{status}</strong>
        </div>
      </div>
      <div className="axisLegend">
        <span className="axis x">X</span>
        <span className="axis y">Y</span>
        <span className="axis z">Z</span>
      </div>
    </div>
  );
}

function buildFilteredSplatStats({
  points,
  visibleIds,
  removedIds,
  isolatedId,
  renderMode,
  reconstructRole,
}) {
  const allObjectIds = new Set();
  const hiddenObjectIds = new Set();
  let visibleGaussians = 0;
  for (const point of points ?? []) {
    allObjectIds.add(point.objectId);
    if (pointVisible(point, visibleIds, removedIds, isolatedId)) {
      visibleGaussians += 1;
    } else {
      hiddenObjectIds.add(point.objectId);
    }
  }
  return {
    mode: "ply-reconstruct",
    objectFilter:
      reconstructRole === "source"
        ? "spark-ply-source"
        : hiddenObjectIds.size > 0
          ? "spark-filtered-ply-reconstruct"
          : "spark-ply-reconstruct",
    visibleGaussians,
    filteredGaussians: Math.max(0, (points?.length ?? 0) - visibleGaussians),
    hiddenObjects: hiddenObjectIds.size,
    removedObjects: removedIds?.size ?? 0,
    isolatedObject: isolatedId === null || isolatedId === undefined ? "" : String(isolatedId),
    colorSourceGaussians: renderMode === "original" ? visibleGaussians : 0,
    objectColorGaussians: renderMode === "original" ? 0 : visibleGaussians,
    objectCount: allObjectIds.size,
  };
}

function sparkObjectFilter({ filteredStats, packedStats, reconstructRole }) {
  if (reconstructRole !== "source") return filteredStats.objectFilter;
  return packedStats.shRestPreserved ? "spark-ply-sh-source" : "spark-ply-source";
}

function buildPackedSplatCache({
  points,
  renderMode,
  shRestCoefficients,
  shRestCoefficientCount,
}) {
  const startedAt = performance.now();
  const packedSplats = new PackedSplats({ maxSplats: Math.max(1, points?.length ?? 0) });
  const shRest =
    renderMode === "original"
      ? buildPackedShExtra({
          points,
          shRestCoefficients,
          shRestCoefficientCount,
        })
      : buildPackedShExtra({
          points: [],
          shRestCoefficients: null,
          shRestCoefficientCount: 0,
        });
  for (const point of points ?? []) {
    const scale = pointScale3(point);
    const quaternion = pointQuaternion(point);
    const color = pointColor(point, renderMode, { preferShDc: shRest.preserved });
    packedSplats.pushSplat(
      new THREE.Vector3(Number(point.x) || 0, Number(point.y) || 0, Number(point.z) || 0),
      new THREE.Vector3(scale[0], scale[1], scale[2]),
      quaternion,
      pointOpacity(point),
      color,
    );
  }
  packedSplats.extra = shRest.extra;
  return {
    packedSplats,
    route: shRest.route,
    baseGaussians: packedSplats.getNumSplats(),
    baseBuildMs: performance.now() - startedAt,
    displayCache: createDisplayPackedCache(),
    shRestSourceGaussians: shRest.sourceGaussians,
    shRestPreservedGaussians: shRest.preservedGaussians,
    shRestPreserved: shRest.preserved,
    shRestCoefficientCount: shRest.coefficientCount,
    shDegree: shRest.degree,
  };
}

function createDisplayPackedCache() {
  return {
    entries: new Map(),
    hits: 0,
    misses: 0,
    evictions: 0,
  };
}

function getCachedDisplayPackedSplats({
  packedCache,
  visibleIndices,
}) {
  const cache = packedCache.displayCache;
  const cacheKey = visibleIndicesCacheKey(visibleIndices);
  const cached = cache.entries.get(cacheKey);
  if (cached) {
    cache.entries.delete(cacheKey);
    cache.entries.set(cacheKey, cached);
    cache.hits += 1;
    return {
      packedSplats: cached.packedSplats,
      cacheMode: SPARK_DISPLAY_CACHE_MODE,
      cacheKey,
      cacheHit: true,
      cacheSize: cache.entries.size,
      cacheHits: cache.hits,
      cacheMisses: cache.misses,
      cacheEvictions: cache.evictions,
      extractMs: 0,
    };
  }

  const extractStartedAt = performance.now();
  const packedSplats = extractPackedSplats({
    packedSplats: packedCache.packedSplats,
    visibleIndices,
    preserveExtra: packedCache.shRestPreserved,
  });
  const extractMs = performance.now() - extractStartedAt;
  cache.entries.set(cacheKey, { packedSplats });
  cache.misses += 1;
  evictDisplayPackedCacheEntries(cache);
  return {
    packedSplats,
    cacheMode: SPARK_DISPLAY_CACHE_MODE,
    cacheKey,
    cacheHit: false,
    cacheSize: cache.entries.size,
    cacheHits: cache.hits,
    cacheMisses: cache.misses,
    cacheEvictions: cache.evictions,
    extractMs,
  };
}

function evictDisplayPackedCacheEntries(cache) {
  while (cache.entries.size > SPARK_DISPLAY_PACKED_CACHE_LIMIT) {
    const oldestKey = cache.entries.keys().next().value;
    const oldest = cache.entries.get(oldestKey);
    oldest?.packedSplats?.dispose();
    cache.entries.delete(oldestKey);
    cache.evictions += 1;
  }
}

function disposeDisplayPackedCache(cache) {
  for (const entry of cache?.entries?.values?.() ?? []) {
    entry?.packedSplats?.dispose();
  }
  cache?.entries?.clear?.();
}

function visibleIndicesCacheKey(visibleIndices) {
  const length = visibleIndices?.length ?? 0;
  let hash = 2166136261;
  for (let index = 0; index < length; index += 1) {
    hash ^= Number(visibleIndices[index] ?? 0);
    hash = Math.imul(hash, 16777619);
  }
  const first = length > 0 ? visibleIndices[0] : -1;
  const last = length > 0 ? visibleIndices[length - 1] : -1;
  return `${length}:${first}:${last}:${hash >>> 0}`;
}

function extractPackedSplats({ packedSplats, visibleIndices, preserveExtra }) {
  const extracted = packedSplats.extractSplats(visibleIndices, false);
  if (preserveExtra) {
    extracted.extra = extractPackedShExtra(packedSplats.extra, visibleIndices);
  }
  return extracted;
}

function disposeSplatMesh(splat, preservedPackedSplats) {
  if (preservedPackedSplats && splat?.packedSplats === preservedPackedSplats) {
    splat.splats = undefined;
    splat.packedSplats = undefined;
  }
  splat.dispose();
}

function visiblePointIndices({
  points,
  visibleIds,
  removedIds,
  isolatedId,
}) {
  const indices = [];
  for (let index = 0; index < (points?.length ?? 0); index += 1) {
    if (pointVisible(points[index], visibleIds, removedIds, isolatedId)) {
      indices.push(index);
    }
  }
  return new Uint32Array(indices);
}

function pointVisible(point, visibleIds, removedIds, isolatedId) {
  if (!point) return false;
  if (visibleIds && !visibleIds.has(point.objectId)) return false;
  if (removedIds?.has(point.objectId)) return false;
  if (isolatedId !== null && isolatedId !== undefined && point.objectId !== isolatedId) return false;
  return true;
}

function pointScale3(point) {
  if (Array.isArray(point?.scale3) && point.scale3.length >= 3) {
    return point.scale3.map((value) => clampFinite(value, 0.0006, 0.35, 0.018));
  }
  if (Array.isArray(point?.scale) && point.scale.length >= 2) {
    return [
      clampFinite(point.scale[0], 0.0006, 0.35, 0.018),
      clampFinite(point.scale[1], 0.0006, 0.35, 0.018),
      clampFinite(point.scale[1], 0.0006, 0.35, 0.018),
    ];
  }
  return [0.018, 0.018, 0.018];
}

function pointQuaternion(point) {
  if (Array.isArray(point?.rotationQuaternion) && point.rotationQuaternion.length >= 4) {
    const [w, x, y, z] = point.rotationQuaternion.map(Number);
    const length = Math.hypot(w, x, y, z);
    if (Number.isFinite(length) && length > 0.0001) {
      return new THREE.Quaternion(x / length, y / length, z / length, w / length);
    }
  }
  return new THREE.Quaternion();
}

function pointColor(point, renderMode, { preferShDc = false } = {}) {
  const shDc = renderMode === "original" && preferShDc ? shDcRgb01(point) : null;
  if (shDc) {
    return new THREE.Color(shDc[0], shDc[1], shDc[2]);
  }
  const rgb = renderMode === "original" ? point?.color : point?.objectColor;
  const values = Array.isArray(rgb) && rgb.length >= 3 ? rgb : [198, 207, 217];
  return new THREE.Color(
    clampFinite(values[0] / 255, 0, 1, 0.78),
    clampFinite(values[1] / 255, 0, 1, 0.81),
    clampFinite(values[2] / 255, 0, 1, 0.85),
  );
}

function pointOpacity(point) {
  return clampFinite(point?.opacity, 0, 1, 1);
}

function clampFinite(value, min, max, fallback) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(Math.max(numeric, min), max);
}

function emptyPackedStats() {
  return {
    route: "none",
    baseGaussians: 0,
    visibleIndices: 0,
    baseBuildMs: 0,
    extractMs: 0,
    shRestSourceGaussians: 0,
    shRestPreservedGaussians: 0,
    shRestPreserved: false,
    shRestCoefficientCount: 0,
    shDegree: 0,
    displayCacheMode: "none",
    displayCacheKey: "",
    displayCacheHit: false,
    displayCacheSize: 0,
    displayCacheHits: 0,
    displayCacheMisses: 0,
    displayCacheEvictions: 0,
  };
}

function pendingPackedStats(cache) {
  return {
    route: cache ? cache.route : "none",
    baseGaussians: cache?.baseGaussians ?? 0,
    visibleIndices: 0,
    baseBuildMs: cache?.baseBuildMs ?? 0,
    extractMs: 0,
    shRestSourceGaussians: cache?.shRestSourceGaussians ?? 0,
    shRestPreservedGaussians: cache?.shRestPreservedGaussians ?? 0,
    shRestPreserved: cache?.shRestPreserved ?? false,
    shRestCoefficientCount: cache?.shRestCoefficientCount ?? 0,
    shDegree: cache?.shDegree ?? 0,
    displayCacheMode: cache ? SPARK_DISPLAY_CACHE_MODE : "none",
    displayCacheKey: "",
    displayCacheHit: false,
    displayCacheSize: cache?.displayCache?.entries?.size ?? 0,
    displayCacheHits: cache?.displayCache?.hits ?? 0,
    displayCacheMisses: cache?.displayCache?.misses ?? 0,
    displayCacheEvictions: cache?.displayCache?.evictions ?? 0,
  };
}

function sparkPathLabel({ objectFilter, packedStats }) {
  if (objectFilter === "spark-ply-sh-source") return "PLY SH 源";
  if (objectFilter === "spark-ply-source") return "PLY 源";
  if (objectFilter === "spark-filtered-ply-reconstruct") {
    return packedStats.displayCacheHit ? "缓存过滤" : "过滤重建";
  }
  if (objectFilter === "spark-ply-reconstruct") return "PLY 重建";
  return "原生 Splat";
}

function formatMillis(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "0.000";
}

function frameSplat(splat, camera, controls, scene) {
  const box = splat.getBoundingBox(true);
  if (!box || box.isEmpty()) return;

  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);

  const maxDim = Math.max(size.x, size.y, size.z, 0.5);
  const distance = maxDim * 1.7;
  camera.position.set(center.x + distance, center.y + distance * 0.58, center.z + distance);
  camera.near = Math.max(maxDim / 120, 0.001);
  camera.far = maxDim * 120;
  camera.updateProjectionMatrix();
  if (scene?.fog) {
    scene.fog.near = Math.max(distance * 1.4, maxDim * 2);
    scene.fog.far = Math.max(distance * 4, maxDim * 10);
  }
  controls.target.copy(center);
  controls.update();
}
