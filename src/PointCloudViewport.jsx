import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const DEFAULT_POINT_SIZE = 0.018;
const EDIT_SPLAT_SIZE_SCALE = 4.8;
const SELECTED_SPLAT_SIZE_SCALE = 7.0;
const OIT_CLEAR_COLOR = new THREE.Color(0x000000);
const SCENE_CLEAR_COLOR = new THREE.Color(0x101316);
const EDIT_VERTEX_SHADER = `
attribute vec3 color;
attribute vec2 gaussianScale;
attribute float gaussianOpacity;
attribute float gaussianRotation;
attribute float gaussianObjectIndex;

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
varying float vObjectIndex;

void main() {
  vec2 safeScale = max(gaussianScale, vec2(0.0006));
  float majorScale = max(safeScale.x, safeScale.y);
  vAxisRatio = clamp(safeScale / majorScale, vec2(0.18), vec2(1.0));
  vRotation = gaussianRotation;
  vObjectIndex = gaussianObjectIndex;
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
uniform sampler2D uObjectState;
uniform float uObjectStateWidth;

varying vec3 vColor;
varying float vOpacity;
varying vec2 vAxisRatio;
varying float vRotation;
varying float vObjectIndex;

void main() {
  float stateU = (vObjectIndex + 0.5) / max(uObjectStateWidth, 1.0);
  float objectVisible = texture2D(uObjectState, vec2(stateU, 0.5)).r;
  if (objectVisible < 0.5) {
    discard;
  }

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

  float weight = exp(-0.5 * d) * vOpacity * uAlphaScale;
  if (weight < 0.004) {
    discard;
  }
  gl_FragColor = vec4(vColor * weight, weight);
}
`;

const RESOLVE_VERTEX_SHADER = `
varying vec2 vUv;

void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const RESOLVE_FRAGMENT_SHADER = `
precision highp float;

uniform sampler2D uAccumulation;
uniform float uOpacityScale;
varying vec2 vUv;

void main() {
  vec4 accumulation = texture2D(uAccumulation, vUv);
  float weight = accumulation.a;
  if (weight <= 0.0001) {
    discard;
  }

  vec3 color = accumulation.rgb / max(weight, 0.0001);
  float alpha = clamp(1.0 - exp(-weight * uOpacityScale), 0.0, 0.98);
  gl_FragColor = vec4(color, alpha);
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
  rendererContract,
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
  const accumulationTargetRef = useRef(null);
  const resolveSceneRef = useRef(null);
  const resolveCameraRef = useRef(null);
  const resolveMaterialRef = useRef(null);
  const objectStateTextureRef = useRef(null);

  const buffers = useMemo(
    () =>
      buildBuffers({
        points,
        renderMode,
        selectedId,
      }),
    [points, renderMode, selectedId],
  );
  const objectFilter = useMemo(
    () =>
      buildObjectFilter({
        objectIdsByIndex: buffers.objectIdsByIndex,
        objectCountsByIndex: buffers.objectCountsByIndex,
        visibleIds,
        removedIds,
        isolatedId,
      }),
    [buffers.objectCountsByIndex, buffers.objectIdsByIndex, visibleIds, removedIds, isolatedId],
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
    renderer.autoClear = false;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    const accumulationTarget = createAccumulationTarget(renderer);
    const { resolveScene, resolveCamera, resolveMaterial } =
      createResolvePass(accumulationTarget);

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
    accumulationTargetRef.current = accumulationTarget;
    resolveSceneRef.current = resolveScene;
    resolveCameraRef.current = resolveCamera;
    resolveMaterialRef.current = resolveMaterial;

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
      const drawingSize = rendererDrawingSize(renderer);
      accumulationTarget.setSize(drawingSize.x, drawingSize.y);
      updateResolveTexture(resolveMaterial, accumulationTarget);
      updateMaterialViewport(pointsObjectRef.current?.material, drawingSize.x, drawingSize.y);
      updateMaterialViewport(selectedObjectRef.current?.material, drawingSize.x, drawingSize.y);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(container);
    resize();

    let animationId = 0;
    const render = () => {
      animationId = window.requestAnimationFrame(render);
      controls.update();
      renderWeightedOitFrame({
        renderer,
        scene,
        camera,
        accumulationTarget,
        resolveScene,
        resolveCamera,
        clouds: [pointsObjectRef.current, selectedObjectRef.current],
        baseObjects: [gridRef.current, axesRef.current],
      });
    };
    render();

    return () => {
      window.cancelAnimationFrame(animationId);
      observer.disconnect();
      controls.dispose();
      accumulationTarget.dispose();
      disposeResolvePass(resolveScene);
      objectStateTextureRef.current?.dispose();
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
    objectStateTextureRef.current?.dispose();
    objectStateTextureRef.current = createObjectStateTexture(objectFilter.states);

    const geometry = createEditGeometry({
      positions: buffers.positions,
      colors: buffers.colors,
      scales: buffers.scales,
      opacities: buffers.opacities,
      rotations: buffers.rotations,
      objectIndices: buffers.objectIndices,
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
      objectStateTexture: objectStateTextureRef.current,
      objectStateWidth: objectFilter.objectCount,
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
        objectIndices: buffers.selectedObjectIndices,
      });
      const selectedMaterial = createGaussianSplatMaterial({
        pointSize,
        sizeScale: SELECTED_SPLAT_SIZE_SCALE,
        alphaScale: 0.62,
        solidColorMix: 1,
        depthTest: false,
        viewport,
        objectStateTexture: objectStateTextureRef.current,
        objectStateWidth: objectFilter.objectCount,
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
    updateObjectStateTexture(objectStateTextureRef.current, objectFilter.states);
    updateMaterialObjectState(pointsObjectRef.current?.material, objectFilter.objectCount);
    updateMaterialObjectState(selectedObjectRef.current?.material, objectFilter.objectCount);
  }, [objectFilter]);

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

      const hits = raycaster.intersectObject(cloud, false);
      for (const hit of hits) {
        if (typeof hit.index !== "number") continue;
        const objectId = buffers.objectIds[hit.index];
        if (objectId === undefined || !objectFilter.visibleObjectIds.has(objectId)) continue;
        onSelectObject(objectId);
        return;
      }
    };

    const canvas = renderer.domElement;
    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointerup", pointerUp);
    return () => {
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointerup", pointerUp);
    };
  }, [buffers.objectIds, objectFilter.visibleObjectIds, onSelectObject, pointSize]);

  return (
    <div
      className="viewport"
      data-renderer={rendererContract?.rendererId ?? "gaussian-oit"}
      data-renderer-target={rendererContract?.targetRendererId ?? "webgpu-tile"}
      data-renderer-fallback-reason={rendererContract?.fallbackReason ?? ""}
      data-webgpu-target-gate={rendererContract?.targetGate ?? ""}
      data-webgpu-target-gate-reason={rendererContract?.targetGateReason ?? ""}
      data-webgpu-target-gate-blocker={rendererContract?.targetGateBlocker ?? ""}
      data-webgpu-status={rendererContract?.webGpuStatus ?? "unknown"}
      data-webgpu-storage-limit-gate={rendererContract?.storageLimitGate ?? ""}
      data-webgpu-storage-limit-reason={rendererContract?.storageLimitReason ?? ""}
      data-webgpu-storage-limit-blocker={rendererContract?.storageLimitBlocker ?? ""}
      data-webgpu-storage-limit-max-buffer-size={rendererContract?.storageLimitMaxBufferSize ?? ""}
      data-webgpu-storage-limit-max-binding-size={rendererContract?.storageLimitMaxStorageBufferBindingSize ?? ""}
      data-webgpu-storage-limit-max-storage-buffers-per-stage={rendererContract?.storageLimitMaxStorageBuffersPerShaderStage ?? ""}
      data-webgpu-storage-limit-required-storage-buffers-per-stage={rendererContract?.storageLimitRequiredStorageBuffersPerShaderStage ?? ""}
      data-webgpu-storage-limit-effective-max-buffer-size={rendererContract?.storageLimitEffectiveMaxBufferByteLength ?? ""}
      data-webgpu-storage-estimated-layout={rendererContract?.storageEstimatedLayout ?? ""}
      data-webgpu-storage-estimated-buffer-count={rendererContract?.storageEstimatedBufferCount ?? 0}
      data-webgpu-storage-estimated-byte-size={rendererContract?.storageEstimatedByteSize ?? 0}
      data-webgpu-storage-estimated-max-buffer-byte-size={rendererContract?.storageEstimatedMaxBufferByteSize ?? 0}
      data-webgpu-storage-estimated-max-buffer-key={rendererContract?.storageEstimatedMaxBufferKey ?? ""}
      data-object-filter={rendererContract?.objectFilter ?? "gpu-object-state-texture"}
      data-webgpu-object-filter-target={rendererContract?.targetObjectFilter ?? "gpu-object-state-buffer"}
      data-webgpu-pack-layout={rendererContract?.tileSmokeLayout ?? ""}
      data-webgpu-packed-gaussians={rendererContract?.packedGaussians ?? 0}
      data-webgpu-visible-gaussians={rendererContract?.visibleGaussians ?? 0}
      data-webgpu-binned-gaussians={rendererContract?.binnedGaussians ?? 0}
      data-webgpu-tile-size={rendererContract?.tileSize ?? 0}
      data-webgpu-tile-count={rendererContract?.tileCount ?? 0}
      data-webgpu-active-tile-count={rendererContract?.activeTileCount ?? 0}
      data-webgpu-tile-reference-count={rendererContract?.tileReferenceCount ?? 0}
      data-tile-overflow-count={rendererContract?.tileOverflowCount ?? 0}
      data-webgpu-tile-overflow-tile-count={rendererContract?.tileOverflowTileCount ?? 0}
      data-webgpu-tile-overflow-ratio={rendererContract?.tileOverflowRatio ?? 0}
      data-webgpu-tile-overflow-max-excess={rendererContract?.tileOverflowMaxExcess ?? 0}
      data-webgpu-tile-entry-stored-count={rendererContract?.tileEntryStoredCount ?? 0}
      data-webgpu-tile-entry-capacity={rendererContract?.tileEntryCapacity ?? 0}
      data-webgpu-tile-entry-utilization={rendererContract?.tileEntryUtilization ?? 0}
      data-webgpu-tile-entry-layout={rendererContract?.tileEntryLayout ?? ""}
      data-webgpu-tile-entry-offset-count={rendererContract?.tileEntryOffsetCount ?? 0}
      data-webgpu-tile-capacity-mode={rendererContract?.tileCapacityMode ?? ""}
      data-webgpu-tile-capacity-status={rendererContract?.tileCapacityStatus ?? ""}
      data-webgpu-tile-capacity-gate={rendererContract?.tileCapacityGate ?? ""}
      data-webgpu-max-tile-occupancy={rendererContract?.maxTileOccupancy ?? 0}
      data-webgpu-resolve-layout={rendererContract?.resolveVersion ?? ""}
      data-webgpu-resolve-mode={rendererContract?.resolveMode ?? ""}
      data-webgpu-resolved-tile-count={rendererContract?.resolvedTileCount ?? 0}
      data-webgpu-resolve-weight-sum={rendererContract?.resolveWeightSum ?? 0}
      data-webgpu-resolve-alpha-mean={rendererContract?.resolveAlphaMean ?? 0}
      data-webgpu-resolve-luma-mean={rendererContract?.resolveLumaMean ?? 0}
      data-webgpu-resolve-checksum={rendererContract?.resolveChecksum ?? ""}
      data-webgpu-pixel-coverage-mode={rendererContract?.pixelCoverageMode ?? ""}
      data-webgpu-pixel-coverage-weight-floor={rendererContract?.pixelCoverageWeightFloor ?? 0}
      data-webgpu-pixel-coverage-footprint-scale={rendererContract?.pixelCoverageFootprintScale ?? 0}
      data-webgpu-color-fidelity-mode={rendererContract?.colorFidelityMode ?? ""}
      data-webgpu-color-source-rgb-gaussians={rendererContract?.colorSourceRgbGaussians ?? 0}
      data-webgpu-color-source-sh-dc-gaussians={rendererContract?.colorSourceShDcGaussians ?? 0}
      data-webgpu-color-source-fallback-gaussians={rendererContract?.colorSourceFallbackGaussians ?? 0}
      data-webgpu-color-source-object-gaussians={rendererContract?.colorSourceObjectGaussians ?? 0}
      data-webgpu-color-opacity-mean={rendererContract?.colorOpacityMean ?? 0}
      data-webgpu-object-state-layout={rendererContract?.objectStateLayoutVersion ?? ""}
      data-webgpu-object-state-stride={rendererContract?.objectStateStrideUint32 ?? 0}
      data-webgpu-object-state-visible-objects={rendererContract?.objectStateVisibleObjects ?? 0}
      data-webgpu-object-state-hidden-objects={rendererContract?.objectStateHiddenObjects ?? 0}
      data-webgpu-object-state-removed-objects={rendererContract?.objectStateRemovedObjects ?? 0}
      data-webgpu-object-state-selected-objects={rendererContract?.objectStateSelectedObjects ?? 0}
      data-webgpu-object-state-isolated-objects={rendererContract?.objectStateIsolatedObjects ?? 0}
      data-webgpu-object-state-checksum={rendererContract?.objectStateChecksum ?? ""}
      data-visible-count={objectFilter.visibleCount}
      ref={containerRef}
    >
      <div className="viewportHud">
        <div>
          <span className="hudLabel">可见</span>
          <strong>{objectFilter.visibleCount.toLocaleString()}</strong>
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
  renderMode,
  selectedId,
}) {
  const { objectIndexById, objectIdsByIndex, objectCountsByIndex } = objectIndex(points);
  const selected = points;
  const positions = new Float32Array(selected.length * 3);
  const colors = new Float32Array(selected.length * 3);
  const scales = new Float32Array(selected.length * 2);
  const opacities = new Float32Array(selected.length);
  const rotations = new Float32Array(selected.length);
  const objectIndices = new Float32Array(selected.length);
  const objectIds = new Int32Array(selected.length);
  const selectedHighlight =
    selectedId === null ? [] : selected.filter((point) => point.objectId === selectedId);
  const selectedPositions = new Float32Array(selectedHighlight.length * 3);
  const selectedColors = new Float32Array(selectedHighlight.length * 3);
  const selectedScales = new Float32Array(selectedHighlight.length * 2);
  const selectedOpacities = new Float32Array(selectedHighlight.length);
  const selectedRotations = new Float32Array(selectedHighlight.length);
  const selectedObjectIndices = new Float32Array(selectedHighlight.length);

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
    objectIndices[index] = objectIndexById.get(point.objectId) ?? 0;
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
    selectedObjectIndices[index] = objectIndexById.get(point.objectId) ?? 0;
  });

  return {
    positions,
    colors,
    scales,
    opacities,
    rotations,
    objectIndices,
    objectIds,
    objectIdsByIndex,
    objectCountsByIndex,
    selectedPositions,
    selectedColors,
    selectedScales,
    selectedOpacities,
    selectedRotations,
    selectedObjectIndices,
  };
}

function objectIndex(points) {
  const counts = new Map();
  for (const point of points) {
    counts.set(point.objectId, (counts.get(point.objectId) ?? 0) + 1);
  }
  const objectIdsByIndex = [...counts.keys()].sort((left, right) => left - right);
  const objectCountsByIndex = objectIdsByIndex.map((id) => counts.get(id) ?? 0);
  const objectIndexById = new Map(objectIdsByIndex.map((id, index) => [id, index]));
  return { objectIndexById, objectIdsByIndex, objectCountsByIndex };
}

function buildObjectFilter({
  objectIdsByIndex,
  objectCountsByIndex,
  visibleIds,
  removedIds,
  isolatedId,
}) {
  const objectCount = Math.max(objectIdsByIndex.length, 1);
  const states = new Uint8Array(objectCount * 4);
  const visibleObjectIds = new Set();
  let visibleCount = 0;

  objectIdsByIndex.forEach((objectId, index) => {
    const visible =
      visibleIds.has(objectId) &&
      !removedIds.has(objectId) &&
      (isolatedId === null || objectId === isolatedId);
    const offset = index * 4;
    states[offset] = visible ? 255 : 0;
    states[offset + 3] = 255;
    if (visible) {
      visibleObjectIds.add(objectId);
      visibleCount += objectCountsByIndex[index] ?? 0;
    }
  });

  if (objectIdsByIndex.length === 0) {
    states[0] = 255;
    states[3] = 255;
  }

  return {
    objectCount,
    states,
    visibleObjectIds,
    visibleCount,
  };
}

function createEditGeometry({
  positions,
  colors,
  scales,
  opacities,
  rotations,
  objectIndices,
}) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute("gaussianScale", new THREE.BufferAttribute(scales, 2));
  geometry.setAttribute("gaussianOpacity", new THREE.BufferAttribute(opacities, 1));
  geometry.setAttribute("gaussianRotation", new THREE.BufferAttribute(rotations, 1));
  geometry.setAttribute("gaussianObjectIndex", new THREE.BufferAttribute(objectIndices, 1));
  return geometry;
}

function createAccumulationTarget(renderer) {
  const size = rendererDrawingSize(renderer);
  const target = new THREE.WebGLRenderTarget(size.x, size.y, {
    format: THREE.RGBAFormat,
    type: THREE.HalfFloatType,
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });
  target.texture.name = "ObjGauss weighted OIT accumulation";
  return target;
}

function createObjectStateTexture(states) {
  const texture = new THREE.DataTexture(
    states.slice(),
    Math.max(states.length / 4, 1),
    1,
    THREE.RGBAFormat,
  );
  texture.name = "ObjGauss object visibility state";
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

function updateObjectStateTexture(texture, states) {
  if (!texture) return;
  if (texture.image.data.length !== states.length) return;
  texture.image.data.set(states);
  texture.needsUpdate = true;
}

function createResolvePass(accumulationTarget) {
  const resolveScene = new THREE.Scene();
  const resolveCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const resolveMaterial = new THREE.ShaderMaterial({
    vertexShader: RESOLVE_VERTEX_SHADER,
    fragmentShader: RESOLVE_FRAGMENT_SHADER,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.NormalBlending,
    uniforms: {
      uAccumulation: { value: accumulationTarget.texture },
      uOpacityScale: { value: 0.18 },
    },
  });
  const resolveQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), resolveMaterial);
  resolveQuad.frustumCulled = false;
  resolveScene.add(resolveQuad);
  return { resolveScene, resolveCamera, resolveMaterial };
}

function disposeResolvePass(resolveScene) {
  for (const child of resolveScene.children) {
    child.geometry?.dispose();
    child.material?.dispose();
  }
}

function renderWeightedOitFrame({
  renderer,
  scene,
  camera,
  accumulationTarget,
  resolveScene,
  resolveCamera,
  clouds,
  baseObjects,
}) {
  const activeClouds = clouds.filter(Boolean);
  const visibleBaseObjects = baseObjects.filter(Boolean);

  const cloudVisibility = setVisibility(activeClouds, false);
  renderer.setRenderTarget(null);
  renderer.setClearColor(SCENE_CLEAR_COLOR, 1);
  renderer.clear(true, true, true);
  renderer.render(scene, camera);

  const previousBackground = scene.background;
  scene.background = null;
  const baseVisibility = setVisibility(visibleBaseObjects, false);
  setVisibility(activeClouds, true);
  renderer.setRenderTarget(accumulationTarget);
  renderer.setClearColor(OIT_CLEAR_COLOR, 0);
  renderer.clear(true, true, true);
  renderer.render(scene, camera);

  scene.background = previousBackground;
  restoreVisibility(visibleBaseObjects, baseVisibility);
  restoreVisibility(activeClouds, cloudVisibility);
  renderer.setRenderTarget(null);
  renderer.render(resolveScene, resolveCamera);
}

function setVisibility(objects, visible) {
  const previous = [];
  for (const object of objects) {
    previous.push(object.visible);
    object.visible = visible;
  }
  return previous;
}

function restoreVisibility(objects, previous) {
  objects.forEach((object, index) => {
    object.visible = previous[index] ?? object.visible;
  });
}

function createGaussianSplatMaterial({
  pointSize,
  sizeScale,
  alphaScale,
  solidColorMix,
  depthTest,
  viewport,
  objectStateTexture,
  objectStateWidth,
}) {
  return new THREE.ShaderMaterial({
    vertexShader: EDIT_VERTEX_SHADER,
    fragmentShader: EDIT_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest,
    blending: THREE.CustomBlending,
    blendEquation: THREE.AddEquation,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneFactor,
    blendEquationAlpha: THREE.AddEquation,
    blendSrcAlpha: THREE.OneFactor,
    blendDstAlpha: THREE.OneFactor,
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
      uObjectState: { value: objectStateTexture },
      uObjectStateWidth: { value: Math.max(objectStateWidth, 1) },
    },
  });
}

function rendererViewport(renderer) {
  return rendererDrawingSize(renderer);
}

function rendererDrawingSize(renderer) {
  const size = new THREE.Vector2(1, 1);
  renderer?.getDrawingBufferSize(size);
  size.x = Math.max(size.x, 1);
  size.y = Math.max(size.y, 1);
  return size;
}

function updateResolveTexture(material, accumulationTarget) {
  if (!material?.uniforms?.uAccumulation) return;
  material.uniforms.uAccumulation.value = accumulationTarget.texture;
}

function updateMaterialViewport(material, width, height) {
  if (!material?.uniforms?.uViewport) return;
  material.uniforms.uViewport.value.set(Math.max(width, 1), Math.max(height, 1));
}

function updateMaterialPointSize(material, pointSize, sizeScale) {
  if (!material?.uniforms?.uSizeScale) return;
  material.uniforms.uSizeScale.value = pointSizeScale(pointSize, sizeScale);
}

function updateMaterialObjectState(material, objectStateWidth) {
  if (!material?.uniforms?.uObjectStateWidth) return;
  material.uniforms.uObjectStateWidth.value = Math.max(objectStateWidth, 1);
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
