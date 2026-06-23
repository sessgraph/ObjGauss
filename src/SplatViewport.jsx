import { useEffect, useMemo, useRef, useState } from "react";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export default function SplatViewport({
  source,
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
    if (source?.url) return source.url;
    if (source?.fileName) return `${source.fileName}:${source.fileBytes?.byteLength ?? 0}`;
    return "none";
  }, [source]);

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
    if (!scene || !camera || !controls || !source) return undefined;

    let disposed = false;
    setStatus("加载中");

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
      splat.dispose();
    };
  }, [source, sourceKey]);

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
      data-object-filter="none"
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
