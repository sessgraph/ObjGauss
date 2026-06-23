import { useEffect, useMemo, useRef, useState } from "react";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export default function SplatViewport({
  source,
  points = null,
  visibleIds = null,
  removedIds = null,
  isolatedId = null,
  renderMode = "original",
  filtered = false,
  showGrid,
  showAxes,
  pointCount,
  rendererLabel,
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const gridRef = useRef(null);
  const axesRef = useRef(null);
  const [status, setStatus] = useState("加载中");

  const sourceKey = useMemo(() => {
    if (filtered) {
      const visibleKey = visibleIds ? [...visibleIds].sort((left, right) => left - right).join(",") : "";
      const removedKey = removedIds ? [...removedIds].sort((left, right) => left - right).join(",") : "";
      return [
        "filtered",
        points?.length ?? 0,
        visibleKey,
        removedKey,
        isolatedId ?? "all",
        renderMode,
      ].join(":");
    }
    if (source?.url) return source.url;
    if (source?.fileName) return `${source.fileName}:${source.fileBytes?.byteLength ?? 0}`;
    return "none";
  }, [filtered, isolatedId, points, removedIds, renderMode, source, visibleIds]);

  const filteredStats = useMemo(
    () =>
      filtered
        ? buildFilteredSplatStats({
            points,
            visibleIds,
            removedIds,
            isolatedId,
            renderMode,
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
    [filtered, isolatedId, points, removedIds, renderMode, visibleIds],
  );
  const objectFilter = filtered ? filteredStats.objectFilter : "none";

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

    const splat = filtered
      ? new SplatMesh({
          maxSplats: Math.max(1, filteredStats.visibleGaussians),
          constructSplats: (splats) => {
            pushFilteredSplats({
              splats,
              points,
              visibleIds,
              removedIds,
              isolatedId,
              renderMode,
            });
          },
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
      splat.dispose();
    };
  }, [filtered, filteredStats.visibleGaussians, isolatedId, points, removedIds, renderMode, source, sourceKey, visibleIds]);

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
      data-spark-filter-mode={filtered ? "ply-reconstruct" : "none"}
      data-spark-filter-status={status === "就绪" ? "ready" : "pending"}
      data-spark-visible-gaussians={filteredStats.visibleGaussians}
      data-spark-filtered-gaussians={filteredStats.filteredGaussians}
      data-spark-hidden-objects={filteredStats.hiddenObjects}
      data-spark-removed-objects={filteredStats.removedObjects}
      data-spark-isolated-object={filteredStats.isolatedObject}
      data-spark-color-mode={renderMode}
      data-spark-color-source-gaussians={filteredStats.colorSourceGaussians}
      data-spark-color-object-gaussians={filteredStats.objectColorGaussians}
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
    objectFilter: hiddenObjectIds.size > 0 ? "spark-filtered-ply-reconstruct" : "spark-ply-reconstruct",
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

function pushFilteredSplats({
  splats,
  points,
  visibleIds,
  removedIds,
  isolatedId,
  renderMode,
}) {
  for (const point of points ?? []) {
    if (!pointVisible(point, visibleIds, removedIds, isolatedId)) continue;
    const scale = pointScale3(point);
    const quaternion = pointQuaternion(point);
    const color = pointColor(point, renderMode);
    splats.pushSplat(
      new THREE.Vector3(Number(point.x) || 0, Number(point.y) || 0, Number(point.z) || 0),
      new THREE.Vector3(scale[0], scale[1], scale[2]),
      quaternion,
      pointOpacity(point),
      color,
    );
  }
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

function pointColor(point, renderMode) {
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
