import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const DEFAULT_POINT_SIZE = 0.018;
const EDIT_SPLAT_SIZE_SCALE = 4.8;
const SELECTED_SPLAT_SIZE_SCALE = 7.0;
const EDIT_VERTEX_SHADER = `
attribute vec3 color;
attribute vec2 gaussianScale;
attribute float gaussianOpacity;
attribute float gaussianRotation;

uniform vec2 uViewport;
uniform float uSizeScale;
uniform float uMinPointSize;
uniform float uMaxPointSize;
uniform vec3 uSolidColor;
uniform float uSolidColorMix;

varying vec3 vColor;
varying float vOpacity;
varying vec2 vAxisRatio;
varying float vRotation;

void main() {
  vec2 safeScale = max(gaussianScale, vec2(0.0006));
  float majorScale = max(safeScale.x, safeScale.y);
  vAxisRatio = clamp(safeScale / majorScale, vec2(0.18), vec2(1.0));
  vRotation = gaussianRotation;
  vOpacity = clamp(gaussianOpacity, 0.0, 1.0);
  vColor = mix(color, uSolidColor, uSolidColorMix);

  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  float viewZ = max(0.001, -mvPosition.z);
  float pixelsPerWorldUnit = projectionMatrix[1][1] * uViewport.y * 0.5 / viewZ;
  float radiusPixels = clamp(
    majorScale * pixelsPerWorldUnit * uSizeScale,
    uMinPointSize,
    uMaxPointSize
  );

  gl_PointSize = radiusPixels * 2.0;
  gl_Position = projectionMatrix * mvPosition;
}
`;

const EDIT_FRAGMENT_SHADER = `
precision highp float;

uniform float uSigmaExtent;
uniform float uKernelCutoff;
uniform float uAlphaScale;

varying vec3 vColor;
varying float vOpacity;
varying vec2 vAxisRatio;
varying float vRotation;

void main() {
  vec2 centered = gl_PointCoord * 2.0 - 1.0;
  float cosine = cos(vRotation);
  float sine = sin(vRotation);
  vec2 rotated = vec2(
    cosine * centered.x - sine * centered.y,
    sine * centered.x + cosine * centered.y
  );
  vec2 normalized = rotated * uSigmaExtent / vAxisRatio;
  float d = dot(normalized, normalized);
  if (d > uKernelCutoff) {
    discard;
  }

  float alpha = exp(-0.5 * d) * vOpacity * uAlphaScale;
  if (alpha < 0.004) {
    discard;
  }
  gl_FragColor = vec4(vColor, alpha);
}
`;

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
      updateMaterialViewport(pointsObjectRef.current?.material, width, height);
      updateMaterialViewport(selectedObjectRef.current?.material, width, height);
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

    const geometry = createEditGeometry({
      positions: buffers.positions,
      colors: buffers.colors,
      scales: buffers.scales,
      opacities: buffers.opacities,
      rotations: buffers.rotations,
    });
    geometry.computeBoundingSphere();
    const viewport = rendererViewport(rendererRef.current);
    const material = createGaussianSplatMaterial({
      pointSize,
      sizeScale: EDIT_SPLAT_SIZE_SCALE,
      alphaScale: 0.78,
      solidColorMix: 0,
      depthTest: true,
      viewport,
    });
    const cloud = new THREE.Points(geometry, material);
    scene.add(cloud);
    pointsObjectRef.current = cloud;

    if (buffers.selectedPositions.length > 0) {
      const selectedGeometry = createEditGeometry({
        positions: buffers.selectedPositions,
        colors: buffers.selectedColors,
        scales: buffers.selectedScales,
        opacities: buffers.selectedOpacities,
        rotations: buffers.selectedRotations,
      });
      const selectedMaterial = createGaussianSplatMaterial({
        pointSize,
        sizeScale: SELECTED_SPLAT_SIZE_SCALE,
        alphaScale: 0.62,
        solidColorMix: 1,
        depthTest: false,
        viewport,
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
      updateMaterialPointSize(pointsObjectRef.current.material, pointSize, EDIT_SPLAT_SIZE_SCALE);
    }
    if (selectedObjectRef.current) {
      updateMaterialPointSize(
        selectedObjectRef.current.material,
        pointSize,
        SELECTED_SPLAT_SIZE_SCALE,
      );
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
  const scales = new Float32Array(selected.length * 2);
  const opacities = new Float32Array(selected.length);
  const rotations = new Float32Array(selected.length);
  const objectIds = new Int32Array(selected.length);
  const selectedHighlight = selected.filter((point) => point.objectId === selectedId);
  const selectedPositions = new Float32Array(selectedHighlight.length * 3);
  const selectedColors = new Float32Array(selectedHighlight.length * 3);
  const selectedScales = new Float32Array(selectedHighlight.length * 2);
  const selectedOpacities = new Float32Array(selectedHighlight.length);
  const selectedRotations = new Float32Array(selectedHighlight.length);

  selected.forEach((point, index) => {
    const offset = index * 3;
    const scaleOffset = index * 2;
    positions[offset] = point.x;
    positions[offset + 1] = point.z;
    positions[offset + 2] = point.y;
    const rgb = renderMode === "original" ? point.color : point.objectColor;
    colors[offset] = rgb[0] / 255;
    colors[offset + 1] = rgb[1] / 255;
    colors[offset + 2] = rgb[2] / 255;
    const scale = pointScale(point);
    scales[scaleOffset] = scale[0];
    scales[scaleOffset + 1] = scale[1];
    opacities[index] = pointOpacity(point);
    rotations[index] = pointRotation(point);
    objectIds[index] = point.objectId;
  });
  selectedHighlight.forEach((point, index) => {
    const offset = index * 3;
    const scaleOffset = index * 2;
    selectedPositions[offset] = point.x;
    selectedPositions[offset + 1] = point.z;
    selectedPositions[offset + 2] = point.y;
    selectedColors[offset] = 1;
    selectedColors[offset + 1] = 0.94;
    selectedColors[offset + 2] = 0.66;
    const scale = pointScale(point);
    selectedScales[scaleOffset] = scale[0];
    selectedScales[scaleOffset + 1] = scale[1];
    selectedOpacities[index] = pointOpacity(point);
    selectedRotations[index] = pointRotation(point);
  });

  return {
    positions,
    colors,
    scales,
    opacities,
    rotations,
    objectIds,
    selectedPositions,
    selectedColors,
    selectedScales,
    selectedOpacities,
    selectedRotations,
    visibleCount: selected.length,
  };
}

function createEditGeometry({ positions, colors, scales, opacities, rotations }) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("gaussianScale", new THREE.BufferAttribute(scales, 2));
  geometry.setAttribute("gaussianOpacity", new THREE.BufferAttribute(opacities, 1));
  geometry.setAttribute("gaussianRotation", new THREE.BufferAttribute(rotations, 1));
  return geometry;
}

function createGaussianSplatMaterial({
  pointSize,
  sizeScale,
  alphaScale,
  solidColorMix,
  depthTest,
  viewport,
}) {
  return new THREE.ShaderMaterial({
    vertexShader: EDIT_VERTEX_SHADER,
    fragmentShader: EDIT_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest,
    blending: THREE.NormalBlending,
    uniforms: {
      uViewport: { value: viewport.clone() },
      uSizeScale: { value: pointSizeScale(pointSize, sizeScale) },
      uMinPointSize: { value: 1.5 },
      uMaxPointSize: { value: 96 },
      uSigmaExtent: { value: 3.0 },
      uKernelCutoff: { value: 13.0 },
      uAlphaScale: { value: alphaScale },
      uSolidColor: { value: new THREE.Color(0xfff0a8) },
      uSolidColorMix: { value: solidColorMix },
    },
  });
}

function rendererViewport(renderer) {
  const size = new THREE.Vector2(1, 1);
  renderer?.getSize(size);
  return size;
}

function updateMaterialViewport(material, width, height) {
  if (!material?.uniforms?.uViewport) return;
  material.uniforms.uViewport.value.set(Math.max(width, 1), Math.max(height, 1));
}

function updateMaterialPointSize(material, pointSize, sizeScale) {
  if (!material?.uniforms?.uSizeScale) return;
  material.uniforms.uSizeScale.value = pointSizeScale(pointSize, sizeScale);
}

function pointSizeScale(pointSize, sizeScale) {
  return Math.max(0.2, (pointSize / DEFAULT_POINT_SIZE) * sizeScale);
}

function pointScale(point) {
  if (Array.isArray(point.scale) && point.scale.length >= 2) {
    return [
      clampNumber(point.scale[0], 0.0006, 0.35, DEFAULT_POINT_SIZE),
      clampNumber(point.scale[1], 0.0006, 0.35, DEFAULT_POINT_SIZE),
    ];
  }
  return [DEFAULT_POINT_SIZE, DEFAULT_POINT_SIZE];
}

function pointOpacity(point) {
  return clampNumber(point.opacity, 0, 1, 1);
}

function pointRotation(point) {
  return Number.isFinite(point.rotation) ? point.rotation : 0;
}

function clampNumber(value, min, max, fallback) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(max, Math.max(min, numeric));
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
