import { useEffect, useMemo, useRef, useState } from "react";
import { PackedSplats, SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  buildPackedShExtra,
  shDcRgb01,
} from "./sparkPackedSh.js";
import {
  createSparkObjectMask,
  disposeSparkObjectMask,
  maskStats,
  SPARK_OBJECT_MASK_MODE,
  updateSparkObjectMask,
} from "./sparkObjectMask.js";

const SPARK_MESH_UPDATE_MODE = "persistent-splatmesh-v1";
const SPARK_OBJECT_FILTER_MASK = "spark-object-opacity-mask";
const SPARK_NATIVE_SPLAT_SOURCE = "native-splat-source-v1";
const SPARK_FILTER_SOURCE_NATIVE = "native-splat";
const SPARK_FILTER_SOURCE_PACKED = "ply-packed";
const SPARK_SELECTION_MODE = "screen-space-object-pick-v1";
const SPARK_PICK_DRAG_PX = 5;
const SPARK_PICK_MAX_RADIUS_PX = 28;
const SPARK_PICK_STRATEGY = "object-support-score-v1";
const SPARK_PICK_SCORE_MARGIN = 0.08;
const SPARK_PICK_SUPPORT_SIGMA_PX = SPARK_PICK_MAX_RADIUS_PX * 0.45;

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
  selectedId = null,
  onSelectObject = null,
  reconstructRole = "filter",
  filterSource = SPARK_FILTER_SOURCE_PACKED,
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const gridRef = useRef(null);
  const axesRef = useRef(null);
  const filteredSplatRef = useRef(null);
  const filteredObjectMaskRef = useRef(null);
  const filteredMeshStateRef = useRef(createFilteredMeshState());
  const pickStartRef = useRef(null);
  const [status, setStatus] = useState("加载中");
  const [packedStats, setPackedStats] = useState(() => emptyPackedStats());
  const [pickStats, setPickStats] = useState(() => emptyPickStats());

  const sourceKey = useMemo(() => {
    if (source?.url) return source.url;
    if (source?.fileName) return `${source.fileName}:${source.fileBytes?.byteLength ?? 0}`;
    return "none";
  }, [source]);
  const useNativeSplatMask =
    filtered &&
    reconstructRole === "filter" &&
    filterSource === SPARK_FILTER_SOURCE_NATIVE &&
    Boolean(source);

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
    ? useNativeSplatMask
      ? "native-splat-mask"
      : reconstructRole === "source"
      ? "ply-source"
      : "ply-reconstruct"
    : "none";
  const sparkMaskSource = filtered
    ? useNativeSplatMask
      ? SPARK_FILTER_SOURCE_NATIVE
      : SPARK_FILTER_SOURCE_PACKED
    : "none";
  const sparkSelectionMode =
    filtered && points?.length > 0 && onSelectObject ? SPARK_SELECTION_MODE : "none";

  const packedCache = useMemo(() => {
    if (!filtered || useNativeSplatMask) return null;
    return buildPackedSplatCache({
      points,
      renderMode,
      shRestCoefficients,
      shRestCoefficientCount,
    });
  }, [filtered, points, renderMode, shRestCoefficients, shRestCoefficientCount, useNativeSplatMask]);

  const nativeMaskCache = useMemo(() => {
    if (!useNativeSplatMask) return null;
    return buildNativeSplatMaskCache({ points });
  }, [points, useNativeSplatMask]);

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
    if (!scene || !camera || !controls || filtered || !source) return undefined;

    let disposed = false;
    setStatus("加载中");
    setPackedStats(emptyPackedStats());

    const splat = new SplatMesh({
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
      disposeSplatMesh(splat);
    };
  }, [filtered, source, sourceKey]);

  useEffect(() => {
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (
      !scene ||
      !camera ||
      !controls ||
      !filtered ||
      useNativeSplatMask ||
      !packedCache?.packedSplats
    ) {
      return undefined;
    }

    let disposed = false;
    setStatus("构建中");
    setPackedStats(pendingPackedStats(packedCache));

    const objectMaskStats = updateSparkObjectMask(packedCache.objectMask, {
      points,
      visibleIds,
      removedIds,
      isolatedId,
    });
    const meshState = createFilteredMeshState();
    filteredMeshStateRef.current = meshState;
    const splat = new SplatMesh({
      packedSplats: packedCache.packedSplats,
      objectModifier: packedCache.objectMask.modifier,
      onLoad: () => {
        if (!disposed) setStatus("就绪");
      },
    });
    filteredSplatRef.current = splat;
    filteredObjectMaskRef.current = packedCache.objectMask;
    meshState.meshId += 1;
    meshState.reused = false;
    meshState.updates = 1;
    updatePackedStatsFromMaskResult({ packedCache, objectMaskStats, meshState, setPackedStats });

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
      disposeSplatMesh(splat, packedCache.packedSplats);
      if (filteredSplatRef.current === splat) {
        filteredSplatRef.current = null;
        filteredObjectMaskRef.current = null;
      }
    };
  }, [filtered, packedCache, useNativeSplatMask]);

  useEffect(() => {
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!scene || !camera || !controls || !useNativeSplatMask || !source || !nativeMaskCache) {
      return undefined;
    }

    let disposed = false;
    setStatus("加载中");
    setPackedStats(pendingPackedStats(nativeMaskCache));

    const objectMaskStats = updateSparkObjectMask(nativeMaskCache.objectMask, {
      points,
      visibleIds,
      removedIds,
      isolatedId,
    });
    const meshState = createFilteredMeshState();
    filteredMeshStateRef.current = meshState;
    const splat = new SplatMesh({
      url: source.url,
      fileBytes: source.fileBytes,
      fileName: source.fileName,
      objectModifier: nativeMaskCache.objectMask.modifier,
      onProgress: (event) => {
        if (!event.lengthComputable || event.total === 0) return;
        const percent = Math.round((event.loaded / event.total) * 100);
        setStatus(`${percent}%`);
      },
      onLoad: () => {
        if (!disposed) setStatus("就绪");
      },
    });
    filteredSplatRef.current = splat;
    filteredObjectMaskRef.current = nativeMaskCache.objectMask;
    meshState.meshId += 1;
    meshState.reused = false;
    meshState.updates = 1;
    updatePackedStatsFromMaskResult({
      packedCache: nativeMaskCache,
      objectMaskStats,
      meshState,
      setPackedStats,
    });

    scene.add(splat);
    splat.initialized
      .then(() => {
        if (disposed) return;
        const sourceCount = splat.splats?.getNumSplats?.() ?? nativeMaskCache.baseGaussians;
        setStatus(sourceCount === (points?.length ?? 0) ? "就绪" : "索引不匹配");
        updatePackedStatsFromMaskResult({
          packedCache: withSourceCount(nativeMaskCache, sourceCount),
          objectMaskStats,
          meshState,
          setPackedStats,
        });
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
      disposeSplatMesh(splat);
      if (filteredSplatRef.current === splat) {
        filteredSplatRef.current = null;
        filteredObjectMaskRef.current = null;
      }
    };
  }, [
    nativeMaskCache,
    source,
    sourceKey,
    useNativeSplatMask,
  ]);

  useEffect(() => {
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const splat = filteredSplatRef.current;
    if (
      !scene ||
      !camera ||
      !controls ||
      !filtered ||
      useNativeSplatMask ||
      !packedCache?.packedSplats ||
      !splat
    ) return;

    setStatus("构建中");
    const objectMaskStats = updateSparkObjectMask(packedCache.objectMask, {
      points,
      visibleIds,
      removedIds,
      isolatedId,
    });
    const meshState = filteredMeshStateRef.current ?? createFilteredMeshState();
    meshState.reused = filteredObjectMaskRef.current === packedCache.objectMask;
    meshState.updates += 1;
    splat.needsUpdate = true;
    filteredMeshStateRef.current = meshState;
    updatePackedStatsFromMaskResult({ packedCache, objectMaskStats, meshState, setPackedStats });
    splat.initialized
      .then(() => {
        setStatus("就绪");
        frameSplat(splat, camera, controls, scene);
      })
      .catch((error) => {
        console.error(error);
        setStatus("加载失败");
      });
  }, [filtered, isolatedId, packedCache, points, removedIds, useNativeSplatMask, visibleIds]);

  useEffect(() => {
    const scene = sceneRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const splat = filteredSplatRef.current;
    if (!scene || !camera || !controls || !useNativeSplatMask || !nativeMaskCache || !splat) return;

    setStatus("构建中");
    const objectMaskStats = updateSparkObjectMask(nativeMaskCache.objectMask, {
      points,
      visibleIds,
      removedIds,
      isolatedId,
    });
    const meshState = filteredMeshStateRef.current ?? createFilteredMeshState();
    meshState.reused = filteredObjectMaskRef.current === nativeMaskCache.objectMask;
    meshState.updates += 1;
    splat.needsUpdate = true;
    filteredMeshStateRef.current = meshState;
    const sourceCount = splat.splats?.getNumSplats?.() ?? nativeMaskCache.baseGaussians;
    updatePackedStatsFromMaskResult({
      packedCache: withSourceCount(nativeMaskCache, sourceCount),
      objectMaskStats,
      meshState,
      setPackedStats,
    });
    splat.initialized
      .then(() => {
        const initializedSourceCount = splat.splats?.getNumSplats?.() ?? sourceCount;
        setStatus(initializedSourceCount === (points?.length ?? 0) ? "就绪" : "索引不匹配");
        frameSplat(splat, camera, controls, scene);
      })
      .catch((error) => {
        console.error(error);
        setStatus("加载失败");
      });
  }, [isolatedId, nativeMaskCache, points, removedIds, useNativeSplatMask, visibleIds]);

  useEffect(() => {
    return () => {
      disposeSparkObjectMask(packedCache?.objectMask);
      packedCache?.packedSplats?.dispose();
    };
  }, [packedCache]);

  useEffect(() => {
    return () => {
      disposeSparkObjectMask(nativeMaskCache?.objectMask);
    };
  }, [nativeMaskCache]);

  useEffect(() => {
    if (gridRef.current) gridRef.current.visible = showGrid;
  }, [showGrid]);

  useEffect(() => {
    if (axesRef.current) axesRef.current.visible = showAxes;
  }, [showAxes]);

  useEffect(() => {
    const camera = cameraRef.current;
    const canvas = containerRef.current?.querySelector("canvas");
    if (!camera || !canvas || !filtered || !onSelectObject || !points?.length) return undefined;

    const pointerDown = (event) => {
      if (event.button !== 0) return;
      pickStartRef.current = { x: event.clientX, y: event.clientY };
    };

    const pointerUp = (event) => {
      const start = pickStartRef.current;
      pickStartRef.current = null;
      if (!start) return;
      const dragDistance = Math.hypot(event.clientX - start.x, event.clientY - start.y);
      if (dragDistance > SPARK_PICK_DRAG_PX) return;

      const pick = pickSparkObjectFromPointer({
        event,
        canvas,
        camera,
        points,
        visibleIds,
        removedIds,
        isolatedId,
      });
      setPickStats(pick);
      if (pick.objectId !== null) onSelectObject(pick.objectId);
    };

    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointerup", pointerUp);
    return () => {
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointerup", pointerUp);
    };
  }, [filtered, isolatedId, onSelectObject, points, removedIds, visibleIds]);

  return (
    <div
      className="viewport splatViewport"
      data-renderer="spark-splat"
      data-object-filter={objectFilter}
      data-spark-filter-mode={sparkFilterMode}
      data-spark-mask-source={sparkMaskSource}
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
      data-spark-object-mask-mode={packedStats.objectMaskMode}
      data-spark-object-mask-size={`${packedStats.objectMaskWidth}x${packedStats.objectMaskHeight}`}
      data-spark-object-mask-updates={packedStats.objectMaskUpdates}
      data-spark-object-mask-visible-gaussians={packedStats.objectMaskVisibleGaussians}
      data-spark-object-mask-hidden-gaussians={packedStats.objectMaskHiddenGaussians}
      data-spark-mesh-update-mode={packedStats.meshUpdateMode}
      data-spark-mesh-id={packedStats.meshId}
      data-spark-mesh-reused={String(packedStats.meshReused)}
      data-spark-mesh-updates={packedStats.meshUpdates}
      data-spark-sh-rest-source-gaussians={packedStats.shRestSourceGaussians}
      data-spark-sh-rest-preserved-gaussians={packedStats.shRestPreservedGaussians}
      data-spark-sh-rest-preserved={String(packedStats.shRestPreserved)}
      data-spark-sh-rest-coefficients={packedStats.shRestCoefficientCount}
      data-spark-sh-degree={packedStats.shDegree}
      data-spark-selection-mode={sparkSelectionMode}
      data-spark-selected-object={selectedId ?? ""}
      data-spark-pick-status={pickStats.status}
      data-spark-pick-strategy={pickStats.strategy}
      data-spark-pick-object={pickStats.objectId ?? ""}
      data-spark-pick-distance-px={formatPickMetric(pickStats.distancePx)}
      data-spark-pick-candidate-objects={pickStats.candidateObjects}
      data-spark-pick-ambiguous={String(pickStats.ambiguous)}
      data-spark-pick-radius-px={SPARK_PICK_MAX_RADIUS_PX}
      data-spark-pick-score={formatPickMetric(pickStats.score)}
      data-spark-pick-score-margin={formatPickMetric(pickStats.scoreMargin)}
      data-spark-pick-second-object={pickStats.secondObjectId ?? ""}
      data-spark-pick-second-score={formatPickMetric(pickStats.secondScore)}
      data-spark-selected-marker-visible={String(
        pickStats.status === "hit" && pickStats.objectId === selectedId,
      )}
      ref={containerRef}
    >
      {pickStats.status === "hit" && pickStats.objectId === selectedId ? (
        <div
          className="sparkSelectionMarker"
          style={{ left: pickStats.screenX, top: pickStats.screenY }}
          aria-hidden="true"
        >
          <span>{pickStats.objectId}</span>
        </div>
      ) : null}
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
        {filtered ? (
          <div>
            <span className="hudLabel">所选</span>
            <strong>{selectedId ?? "无"}</strong>
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
          ? SPARK_OBJECT_FILTER_MASK
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
    objectMask: createSparkObjectMask(points?.length ?? 0),
    route: shRest.route,
    baseGaussians: packedSplats.getNumSplats(),
    baseBuildMs: performance.now() - startedAt,
    shRestSourceGaussians: shRest.sourceGaussians,
    shRestPreservedGaussians: shRest.preservedGaussians,
    shRestPreserved: shRest.preserved,
    shRestCoefficientCount: shRest.coefficientCount,
    shDegree: shRest.degree,
  };
}

function buildNativeSplatMaskCache({ points }) {
  return {
    objectMask: createSparkObjectMask(points?.length ?? 0),
    route: SPARK_NATIVE_SPLAT_SOURCE,
    baseGaussians: points?.length ?? 0,
    baseBuildMs: 0,
    shRestSourceGaussians: 0,
    shRestPreservedGaussians: 0,
    shRestPreserved: false,
    shRestCoefficientCount: 0,
    shDegree: 0,
  };
}

function withSourceCount(cache, baseGaussians) {
  return {
    ...cache,
    baseGaussians,
  };
}

function createFilteredMeshState() {
  return {
    meshId: 0,
    reused: false,
    updates: 0,
  };
}

function updatePackedStatsFromMaskResult({
  packedCache,
  objectMaskStats,
  meshState,
  setPackedStats,
}) {
  setPackedStats({
    route: packedCache.route,
    baseGaussians: packedCache.baseGaussians,
    visibleIndices: objectMaskStats.objectMaskVisibleGaussians,
    baseBuildMs: packedCache.baseBuildMs,
    extractMs: 0,
    shRestSourceGaussians: packedCache.shRestSourceGaussians,
    shRestPreservedGaussians: packedCache.shRestPreservedGaussians,
    shRestPreserved: packedCache.shRestPreserved,
    shRestCoefficientCount: packedCache.shRestCoefficientCount,
    shDegree: packedCache.shDegree,
    displayCacheMode: "disabled-by-native-mask-v1",
    displayCacheKey: "",
    displayCacheHit: false,
    displayCacheSize: 0,
    displayCacheHits: 0,
    displayCacheMisses: 0,
    displayCacheEvictions: 0,
    ...objectMaskStats,
    meshUpdateMode: SPARK_MESH_UPDATE_MODE,
    meshId: meshState.meshId,
    meshReused: meshState.reused,
    meshUpdates: meshState.updates,
  });
}

function disposeSplatMesh(splat, preservedPackedSplats) {
  if (preservedPackedSplats && splat?.packedSplats === preservedPackedSplats) {
    splat.splats = undefined;
    splat.packedSplats = undefined;
  }
  splat.dispose();
}

function pointVisible(point, visibleIds, removedIds, isolatedId) {
  if (!point) return false;
  if (visibleIds && !visibleIds.has(point.objectId)) return false;
  if (removedIds?.has(point.objectId)) return false;
  if (isolatedId !== null && isolatedId !== undefined && point.objectId !== isolatedId) return false;
  return true;
}

function pickSparkObjectFromPointer({
  event,
  canvas,
  camera,
  points,
  visibleIds,
  removedIds,
  isolatedId,
}) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return missedPickStats("invalid-viewport");
  const targetX = event.clientX - rect.left;
  const targetY = event.clientY - rect.top;
  const maxDistanceSq = SPARK_PICK_MAX_RADIUS_PX * SPARK_PICK_MAX_RADIUS_PX;
  const supportSigmaSq = SPARK_PICK_SUPPORT_SIGMA_PX * SPARK_PICK_SUPPORT_SIGMA_PX;
  const projected = new THREE.Vector3();
  const candidatesByObject = new Map();
  let totalSupport = 0;

  for (const point of points ?? []) {
    if (!pointVisible(point, visibleIds, removedIds, isolatedId)) continue;
    projected.set(
      Number(point.x) || 0,
      Number(point.y) || 0,
      Number(point.z) || 0,
    );
    projected.project(camera);
    if (
      !Number.isFinite(projected.x) ||
      !Number.isFinite(projected.y) ||
      !Number.isFinite(projected.z) ||
      projected.z < -1 ||
      projected.z > 1
    ) {
      continue;
    }

    const screenX = (projected.x * 0.5 + 0.5) * rect.width;
    const screenY = (-projected.y * 0.5 + 0.5) * rect.height;
    const distanceSq = (screenX - targetX) ** 2 + (screenY - targetY) ** 2;
    if (distanceSq > maxDistanceSq) continue;

    const support = Math.exp(-0.5 * (distanceSq / supportSigmaSq));
    totalSupport += support;
    const current = candidatesByObject.get(point.objectId) ?? {
      objectId: point.objectId,
      bestDistanceSq: Infinity,
      bestDepth: Infinity,
      support: 0,
      samples: 0,
    };
    current.support += support;
    current.samples += 1;
    if (
      distanceSq < current.bestDistanceSq ||
      (distanceSq === current.bestDistanceSq && projected.z < current.bestDepth)
    ) {
      current.bestDistanceSq = distanceSq;
      current.bestDepth = projected.z;
    }
    candidatesByObject.set(point.objectId, current);
  }

  if (candidatesByObject.size === 0) {
    return {
      ...missedPickStats("miss"),
      screenX: targetX,
      screenY: targetY,
      candidateObjects: 0,
    };
  }

  const candidateEntries = [...candidatesByObject.values()];
  const minDepth = Math.min(...candidateEntries.map((candidate) => candidate.bestDepth));
  const maxDepth = Math.max(...candidateEntries.map((candidate) => candidate.bestDepth));
  const depthSpan = Math.max(maxDepth - minDepth, 0);
  const scoredCandidates = candidateEntries
    .map((candidate) => {
      const bestDistancePx = Math.sqrt(candidate.bestDistanceSq);
      const distanceScore = clampFinite(
        1 - bestDistancePx / SPARK_PICK_MAX_RADIUS_PX,
        0,
        1,
        0,
      );
      const supportShare =
        totalSupport > 0 ? clampFinite(candidate.support / totalSupport, 0, 1, 0) : 0;
      const depthScore =
        depthSpan > 0
          ? clampFinite(1 - (candidate.bestDepth - minDepth) / depthSpan, 0, 1, 0)
          : 1;
      return {
        ...candidate,
        bestDistancePx,
        score: 0.58 * distanceScore + 0.32 * supportShare + 0.1 * depthScore,
      };
    })
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      if (a.bestDistanceSq !== b.bestDistanceSq) return a.bestDistanceSq - b.bestDistanceSq;
      return a.bestDepth - b.bestDepth;
    });

  const best = scoredCandidates[0];
  const second = scoredCandidates[1] ?? null;
  const scoreMargin = second ? best.score - second.score : 1;
  const ambiguous = Boolean(second && scoreMargin <= SPARK_PICK_SCORE_MARGIN);

  return {
    status: "hit",
    strategy: SPARK_PICK_STRATEGY,
    objectId: best.objectId,
    screenX: targetX,
    screenY: targetY,
    distancePx: best.bestDistancePx,
    candidateObjects: candidatesByObject.size,
    ambiguous,
    score: best.score,
    scoreMargin,
    secondObjectId: second?.objectId ?? null,
    secondScore: second?.score ?? 0,
  };
}

function emptyPickStats() {
  return missedPickStats("idle");
}

function missedPickStats(status) {
  return {
    status,
    strategy: SPARK_PICK_STRATEGY,
    objectId: null,
    screenX: 0,
    screenY: 0,
    distancePx: 0,
    candidateObjects: 0,
    ambiguous: false,
    score: 0,
    scoreMargin: 0,
    secondObjectId: null,
    secondScore: 0,
  };
}

function formatPickMetric(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "0.000";
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
    objectMaskMode: "none",
    objectMaskWidth: 0,
    objectMaskHeight: 0,
    objectMaskUpdates: 0,
    objectMaskVisibleGaussians: 0,
    objectMaskHiddenGaussians: 0,
    meshUpdateMode: "none",
    meshId: 0,
    meshReused: false,
    meshUpdates: 0,
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
    displayCacheMode: cache ? "disabled-by-native-mask-v1" : "none",
    displayCacheKey: "",
    displayCacheHit: false,
    displayCacheSize: 0,
    displayCacheHits: 0,
    displayCacheMisses: 0,
    displayCacheEvictions: 0,
    ...maskStats(cache?.objectMask),
    meshUpdateMode: cache ? SPARK_MESH_UPDATE_MODE : "none",
    meshId: 0,
    meshReused: false,
    meshUpdates: 0,
  };
}

function sparkPathLabel({ objectFilter, packedStats }) {
  if (objectFilter === "spark-ply-sh-source") return "PLY SH 源";
  if (objectFilter === "spark-ply-source") return "PLY 源";
  if (
    objectFilter === SPARK_OBJECT_FILTER_MASK &&
    packedStats.route === SPARK_NATIVE_SPLAT_SOURCE
  ) {
    return "原生过滤";
  }
  if (objectFilter === SPARK_OBJECT_FILTER_MASK) {
    return packedStats.objectMaskMode === SPARK_OBJECT_MASK_MODE ? "Shader 过滤" : "过滤重建";
  }
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
