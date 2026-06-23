import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const EDIT_SPLAT_SIZE_SCALE = 4.4;
const SELECTED_SPLAT_SIZE_SCALE = 6.2;
let softPointTexture = null;

export default function PointCloudViewport({
  points,
  visibleIds,
  removedIds,
  renderMode,
  pointSize,
  showGrid,
  showAxes,
  isolatedId,
  selectedId,
  onSelectObject,
  renderModeLabel,
}) {
  const containerRef = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const pointsObjectRef = useRef(null);
  const selectedObjectRef = useRef(null);
  const gridRef = useRef(null);
  const axesRef = useRef(null);
  const pickStartRef = useRef(null);

  const buffers = useMemo(
    () =>
      buildBuffers({
        points,
        visibleIds,
        removedIds,
        renderMode,
        isolatedId,
        selectedId,
      }),
    [points, visibleIds, removedIds, renderMode, isolatedId, selectedId],
  );

  useEffect(() => {
    const container = containerRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101316);
    scene.fog = new THREE.Fog(0x101316, 6, 14);

    const camera = new THREE.PerspectiveCamera(52, 1, 0.01, 1000);
    camera.position.set(3.6, 2.8, 3.4);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0.25);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 1.3);
    scene.add(hemi);

    const grid = new THREE.GridHelper(6, 24, 0x42505c, 0x24303a);
    grid.position.y = -0.92;
    scene.add(grid);

    const axes = new THREE.AxesHelper(1.05);
    axes.position.set(2.45, -0.88, -2.4);
    scene.add(axes);

    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;
    controlsRef.current = controls;
    gridRef.current = grid;
    axesRef.current = axes;

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    let animationId = 0;
    const render = () => {
      animationId = window.requestAnimationFrame(render);
      controls.update();
      renderer.render(scene, camera);
    };
    render();

    return () => {
      window.cancelAnimationFrame(animationId);
      observer.disconnect();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;

    if (pointsObjectRef.current) {
      scene.remove(pointsObjectRef.current);
      pointsObjectRef.current.geometry.dispose();
      pointsObjectRef.current.material.dispose();
      pointsObjectRef.current = null;
    }
    if (selectedObjectRef.current) {
      scene.remove(selectedObjectRef.current);
      selectedObjectRef.current.geometry.dispose();
      selectedObjectRef.current.material.dispose();
      selectedObjectRef.current = null;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(buffers.positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(buffers.colors, 3));
    geometry.computeBoundingSphere();
    const splatTexture = getSoftPointTexture();

    const material = new THREE.PointsMaterial({
      size: pointSize * EDIT_SPLAT_SIZE_SCALE,
      map: splatTexture,
      vertexColors: true,
      transparent: true,
      opacity: 0.58,
      alphaTest: 0.006,
      sizeAttenuation: true,
      depthWrite: false,
    });
    const cloud = new THREE.Points(geometry, material);
    scene.add(cloud);
    pointsObjectRef.current = cloud;

    if (buffers.selectedPositions.length > 0) {
      const selectedGeometry = new THREE.BufferGeometry();
      selectedGeometry.setAttribute(
        "position",
        new THREE.BufferAttribute(buffers.selectedPositions, 3),
      );
      const selectedMaterial = new THREE.PointsMaterial({
        size: pointSize * SELECTED_SPLAT_SIZE_SCALE,
        map: splatTexture,
        color: 0xfff0a8,
        transparent: true,
        opacity: 0.6,
        alphaTest: 0.006,
        sizeAttenuation: true,
        depthWrite: false,
        depthTest: false,
      });
      const selectedCloud = new THREE.Points(selectedGeometry, selectedMaterial);
      scene.add(selectedCloud);
      selectedObjectRef.current = selectedCloud;
    }

    if (buffers.positions.length > 0) {
      frameGeometry(geometry, cameraRef.current, controlsRef.current, scene);
    }
  }, [buffers, pointSize]);

  useEffect(() => {
    if (pointsObjectRef.current) {
      pointsObjectRef.current.material.size = pointSize * EDIT_SPLAT_SIZE_SCALE;
    }
    if (selectedObjectRef.current) {
      selectedObjectRef.current.material.size = pointSize * SELECTED_SPLAT_SIZE_SCALE;
    }
  }, [pointSize]);

  useEffect(() => {
    if (gridRef.current) gridRef.current.visible = showGrid;
  }, [showGrid]);

  useEffect(() => {
    if (axesRef.current) axesRef.current.visible = showAxes;
  }, [showAxes]);

  useEffect(() => {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    if (!renderer || !camera || !onSelectObject) return;

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    raycaster.params.Points.threshold = Math.max(pointSize * 2.8, 0.018);

    const pointerDown = (event) => {
      pickStartRef.current = { x: event.clientX, y: event.clientY };
    };

    const pointerUp = (event) => {
      const start = pickStartRef.current;
      pickStartRef.current = null;
      if (!start) return;

      const dragDistance = Math.hypot(event.clientX - start.x, event.clientY - start.y);
      if (dragDistance > 5) return;

      const cloud = pointsObjectRef.current;
      if (!cloud || buffers.objectIds.length === 0) return;

      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);

      const hit = raycaster.intersectObject(cloud, false)[0];
      if (!hit || typeof hit.index !== "number") return;

      const objectId = buffers.objectIds[hit.index];
      if (objectId !== undefined) {
        onSelectObject(objectId);
      }
    };

    const canvas = renderer.domElement;
    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointerup", pointerUp);
    return () => {
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointerup", pointerUp);
    };
  }, [buffers.objectIds, onSelectObject, pointSize]);

  return (
    <div className="viewport" ref={containerRef}>
      <div className="viewportHud">
        <div>
          <span className="hudLabel">可见</span>
          <strong>{buffers.visibleCount.toLocaleString()}</strong>
        </div>
        <div>
          <span className="hudLabel">模式</span>
          <strong>{renderModeLabel}</strong>
        </div>
        <div>
          <span className="hudLabel">所选</span>
          <strong>{selectedId ?? "无"}</strong>
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

function buildBuffers({
  points,
  visibleIds,
  removedIds,
  renderMode,
  isolatedId,
  selectedId,
}) {
  const selected = points.filter((point) => {
    if (removedIds.has(point.objectId)) return false;
    if (isolatedId !== null && point.objectId !== isolatedId) return false;
    return visibleIds.has(point.objectId);
  });
  const positions = new Float32Array(selected.length * 3);
  const colors = new Float32Array(selected.length * 3);
  const objectIds = new Int32Array(selected.length);
  const selectedHighlight = selected.filter((point) => point.objectId === selectedId);
  const selectedPositions = new Float32Array(selectedHighlight.length * 3);

  selected.forEach((point, index) => {
    const offset = index * 3;
    positions[offset] = point.x;
    positions[offset + 1] = point.z;
    positions[offset + 2] = point.y;
    const rgb = renderMode === "original" ? point.color : point.objectColor;
    colors[offset] = rgb[0] / 255;
    colors[offset + 1] = rgb[1] / 255;
    colors[offset + 2] = rgb[2] / 255;
    objectIds[index] = point.objectId;
  });
  selectedHighlight.forEach((point, index) => {
    const offset = index * 3;
    selectedPositions[offset] = point.x;
    selectedPositions[offset + 1] = point.z;
    selectedPositions[offset + 2] = point.y;
  });

  return {
    positions,
    colors,
    objectIds,
    selectedPositions,
    visibleCount: selected.length,
  };
}

function getSoftPointTexture() {
  if (softPointTexture) return softPointTexture;

  const size = 64;
  const center = size / 2;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d");
  if (!context) return null;
  const gradient = context.createRadialGradient(center, center, 0, center, center, center);
  gradient.addColorStop(0, "rgba(255, 255, 255, 0.98)");
  gradient.addColorStop(0.42, "rgba(255, 255, 255, 0.68)");
  gradient.addColorStop(0.78, "rgba(255, 255, 255, 0.16)");
  gradient.addColorStop(1, "rgba(255, 255, 255, 0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, size, size);

  softPointTexture = new THREE.CanvasTexture(canvas);
  softPointTexture.minFilter = THREE.LinearFilter;
  softPointTexture.magFilter = THREE.LinearFilter;
  softPointTexture.wrapS = THREE.ClampToEdgeWrapping;
  softPointTexture.wrapT = THREE.ClampToEdgeWrapping;
  return softPointTexture;
}

function frameGeometry(geometry, camera, controls, scene) {
  geometry.computeBoundingBox();
  const box = geometry.boundingBox;
  if (!box || !camera || !controls) return;

  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const maxDim = Math.max(size.x, size.y, size.z, 0.5);
  const distance = maxDim * 1.55;
  camera.position.set(center.x + distance, center.y + distance * 0.68, center.z + distance);
  camera.near = Math.max(maxDim / 100, 0.001);
  camera.far = maxDim * 100;
  camera.updateProjectionMatrix();
  if (scene?.fog) {
    scene.fog.near = Math.max(distance * 1.4, maxDim * 2);
    scene.fog.far = Math.max(distance * 4, maxDim * 10);
  }
  controls.target.copy(center);
  controls.update();
}
