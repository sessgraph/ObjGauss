import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { DragControls } from "three/examples/jsm/controls/DragControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { MODEL_CATALOG, catalogSummary } from "./modelCatalog.js";
import { parsePly } from "./ply.js";

const INITIAL_CAMERA = {
  position: [0, 4.7, 8.4],
  target: [0, 0.95, 0],
};

export default function App() {
  const worldApi = useRef(null);
  const loadStarted = useRef(false);
  const [worldReady, setWorldReady] = useState(false);
  const [selectedId, setSelectedId] = useState(MODEL_CATALOG[0]?.id ?? "");
  const [models, setModels] = useState(() => initialModelStates());
  const summary = useMemo(() => catalogSummary(MODEL_CATALOG), []);
  const loadedCount = useMemo(
    () => Object.values(models).filter((model) => model.status === "loaded").length,
    [models],
  );
  const selected = models[selectedId] ?? Object.values(models)[0];

  const patchModel = useCallback((id, patch) => {
    setModels((current) => ({
      ...current,
      [id]: {
        ...current[id],
        ...patch,
      },
    }));
  }, []);

  const selectModel = useCallback((id) => {
    setSelectedId(id);
    worldApi.current?.focusModel(id);
  }, []);

  const handleWorldReady = useCallback((api) => {
    worldApi.current = api;
    setWorldReady(true);
  }, []);

  useEffect(() => {
    if (!worldReady || loadStarted.current) return;
    loadStarted.current = true;
    let cancelled = false;

    async function loadModels() {
      for (const model of MODEL_CATALOG) {
        if (cancelled) return;
        if (model.loadMode !== "eager") {
          const rendered = worldApi.current?.upsertModel(model, null);
          patchModel(model.id, {
            status: "compressed",
            message: "compressed chunks",
            displayCount: rendered?.displayCount ?? 0,
            objectCount: model.objectCount,
            corePoint: rendered?.corePoint ?? [0, 0, 0],
          });
          continue;
        }

        const startedAt = performance.now();
        patchModel(model.id, { status: "loading", message: "loading" });
        try {
          const response = await fetch(model.sourcePath);
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          const cloud = parsePly(await response.arrayBuffer());
          const rendered = worldApi.current?.upsertModel(model, cloud.points);
          if (cancelled) return;
          patchModel(model.id, {
            status: "loaded",
            message: "ready",
            gaussianCount: cloud.points.length,
            displayCount: rendered?.displayCount ?? 0,
            objectCount: rendered?.objectCount ?? model.objectCount,
            corePoint: rendered?.corePoint ?? null,
            loadMs: Math.round(performance.now() - startedAt),
          });
        } catch (error) {
          if (cancelled) return;
          patchModel(model.id, {
            status: "error",
            message: error?.message ?? "load failed",
          });
        }
      }
    }

    void loadModels();
    return () => {
      cancelled = true;
    };
  }, [patchModel, worldReady]);

  return (
    <main
      className="worldShell"
      data-app-mode="vr-three-world"
      data-three-renderer="enabled"
      data-sidebars="none"
      data-frosted-ui="enabled"
      data-model-count={MODEL_CATALOG.length}
      data-loaded-count={loadedCount}
      data-selected-model={selected?.id ?? ""}
      data-compression-layout="per-object-corepoint-chunks"
    >
      <ThreeWorld
        models={MODEL_CATALOG}
        selectedId={selectedId}
        onReady={handleWorldReady}
        onSelectModel={setSelectedId}
        onModelMoved={(id, position) => patchModel(id, { galleryPosition: position })}
      />

      <div className="glassHud topHud">
        <div className="brandBlock">
          <div className="brandMark">OG</div>
          <div>
            <h1>ObjGauss</h1>
            <span>VR 3D Gaussian World</span>
          </div>
        </div>
        <div className="metricStrip">
          <Metric label="模型" value={summary.modelCount} />
          <Metric label="已加载" value={loadedCount} />
          <Metric label="压缩原型" value={summary.compressedReadyCount} />
        </div>
        <button className="glassButton" type="button" onClick={() => worldApi.current?.resetCamera()}>
          重置视角
        </button>
      </div>

      <div className="glassHud objectDock" aria-label="模型入口">
        {Object.values(models).map((model) => (
          <button
            className={`modelPill ${selectedId === model.id ? "selected" : ""}`}
            type="button"
            key={model.id}
            data-model-row-id={model.id}
            data-model-load-state={model.status}
            onClick={() => selectModel(model.id)}
          >
            <span className="modelAccent" style={{ background: model.accent }} />
            <span>{model.label}</span>
            <small>{model.status}</small>
          </button>
        ))}
      </div>

      {selected ? (
        <section className="glassHud floatingInspector" data-floating-inspector="true">
          <div className="inspectorHead">
            <span className="modelAccent large" style={{ background: selected.accent }} />
            <div>
              <h2>{selected.name}</h2>
              <span>{selected.kind}</span>
            </div>
          </div>
          <dl className="metaGrid">
            <Meta label="加载状态" value={selected.message ?? selected.status} />
            <Meta label="点数" value={formatNumber(selected.gaussianCount)} />
            <Meta label="展示点" value={formatNumber(selected.displayCount)} />
            <Meta label="对象" value={formatNumber(selected.objectCount)} />
            <Meta label="核心点" value={formatVec(selected.corePoint)} />
            <Meta label="加载耗时" value={selected.loadMs ? `${selected.loadMs} ms` : "-"} />
            <Meta label="压缩布局" value={selected.compression?.layout ?? "-"} />
            <Meta label="分块根" value={selected.compression?.chunkRoot ?? "-"} />
          </dl>
        </section>
      ) : null}

      <div className="glassHud bottomStatus">
        <span>拖动展品即可移动对象</span>
        <span>核心点为每个处理后 Gaussian 对象的加载锚点</span>
        <span>后端负责单对象压缩块与按需加载</span>
      </div>
    </main>
  );
}

function ThreeWorld({ models, selectedId, onReady, onSelectModel, onModelMoved }) {
  const mountRef = useRef(null);
  const apiRef = useRef(null);
  const selectedRef = useRef(selectedId);
  const callbacksRef = useRef({ onSelectModel, onModelMoved });

  useEffect(() => {
    selectedRef.current = selectedId;
    apiRef.current?.setSelected(selectedId);
  }, [selectedId]);

  useEffect(() => {
    callbacksRef.current = { onSelectModel, onModelMoved };
  }, [onSelectModel, onModelMoved]);

  useEffect(() => {
    if (!mountRef.current) return undefined;

    const mount = mountRef.current;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#070b11");
    scene.fog = new THREE.Fog("#070b11", 10, 30);

    const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 100);
    camera.position.fromArray(INITIAL_CAMERA.position);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.domElement.dataset.threeWorldCanvas = "true";
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.fromArray(INITIAL_CAMERA.target);

    buildWorldShell(scene);

    const modelObjects = new Map();
    let dragControls = null;
    let animationFrame = 0;

    const publishAuditHandle = () => {
      window.__OBJGAUSS_WORLD__ = {
        renderer: "three.js",
        ui: "frosted-glass-in-world",
        sidebars: false,
        modelCount: models.length,
        draggableCount: modelObjects.size,
        selectedId: selectedRef.current,
        modelPositions: [...modelObjects.values()].map((object) => ({
          id: object.userData.modelId,
          position: object.position.toArray().map(round3),
        })),
      };
    };

    const rebuildDragControls = () => {
      dragControls?.dispose();
      dragControls = new DragControls([...modelObjects.values()], camera, renderer.domElement);
      dragControls.transformGroup = true;
      dragControls.addEventListener("dragstart", (event) => {
        controls.enabled = false;
        selectObject(event.object.userData.modelId);
      });
      dragControls.addEventListener("drag", (event) => {
        event.object.position.y = 0;
      });
      dragControls.addEventListener("dragend", (event) => {
        controls.enabled = true;
        callbacksRef.current.onModelMoved?.(event.object.userData.modelId, [
          round3(event.object.position.x),
          round3(event.object.position.y),
          round3(event.object.position.z),
        ]);
        publishAuditHandle();
      });
      publishAuditHandle();
    };

    const disposeObject = (object) => {
      object.traverse((child) => {
        child.geometry?.dispose?.();
        if (Array.isArray(child.material)) {
          child.material.forEach((material) => material.dispose?.());
        } else {
          child.material?.dispose?.();
        }
      });
    };

    const upsertModel = (model, points = null) => {
      const previous = modelObjects.get(model.id);
      if (previous) {
        scene.remove(previous);
        disposeObject(previous);
      }
      const result = points?.length ? createPointCloudGroup(model, points) : createCompressedModelGroup(model);
      scene.add(result.group);
      modelObjects.set(model.id, result.group);
      rebuildDragControls();
      api.setSelected(selectedRef.current);
      return result.summary;
    };

    const selectObject = (id) => {
      if (!id) return;
      selectedRef.current = id;
      api.setSelected(id);
      callbacksRef.current.onSelectModel?.(id);
      publishAuditHandle();
    };

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const onPointerDown = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const intersections = raycaster.intersectObjects([...modelObjects.values()], true);
      const hit = intersections[0]?.object;
      const modelId = nearestModelId(hit);
      selectObject(modelId);
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);

    const api = {
      upsertModel,
      focusModel(id) {
        const object = modelObjects.get(id);
        if (!object) return;
        selectedRef.current = id;
        api.setSelected(id);
        controls.target.copy(object.position.clone().add(new THREE.Vector3(0, 0.9, 0)));
      },
      resetCamera() {
        camera.position.fromArray(INITIAL_CAMERA.position);
        controls.target.fromArray(INITIAL_CAMERA.target);
      },
      setSelected(id) {
        for (const [modelId, object] of modelObjects) {
          const selected = modelId === id;
          object.userData.selected = selected;
          object.traverse((child) => {
            if (child.userData.role === "selection-ring" || child.userData.role === "core-glow") {
              child.visible = selected;
            }
          });
        }
        publishAuditHandle();
      },
    };

    const resize = () => {
      const width = mount.clientWidth || 1;
      const height = mount.clientHeight || 1;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };
    const animate = () => {
      controls.update();
      renderer.render(scene, camera);
      animationFrame = requestAnimationFrame(animate);
    };

    models.forEach((model) => upsertModel(model));
    resize();
    animate();
    window.addEventListener("resize", resize);
    apiRef.current = api;
    onReady(api);

    return () => {
      cancelAnimationFrame(animationFrame);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("resize", resize);
      dragControls?.dispose();
      controls.dispose();
      modelObjects.forEach(disposeObject);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
      if (window.__OBJGAUSS_WORLD__?.renderer === "three.js") {
        delete window.__OBJGAUSS_WORLD__;
      }
    };
  }, [models, onReady]);

  return <div className="threeWorldMount" ref={mountRef} />;
}

function buildWorldShell(scene) {
  const ambient = new THREE.AmbientLight("#7e91a1", 1.1);
  const key = new THREE.DirectionalLight("#e9fbff", 2.2);
  key.position.set(4, 8, 5);
  const rim = new THREE.DirectionalLight("#36d7ff", 0.95);
  rim.position.set(-7, 4, -4);
  scene.add(ambient, key, rim);

  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(8.6, 96),
    new THREE.MeshBasicMaterial({
      color: "#0c141b",
      transparent: true,
      opacity: 0.98,
      side: THREE.DoubleSide,
    }),
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = -0.01;
  scene.add(floor);

  const grid = new THREE.GridHelper(15, 30, "#2a5f6a", "#152932");
  grid.material.transparent = true;
  grid.material.opacity = 0.58;
  scene.add(grid);

  const halo = new THREE.Mesh(
    new THREE.RingGeometry(4.4, 4.45, 128),
    new THREE.MeshBasicMaterial({
      color: "#1bc7d6",
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
    }),
  );
  halo.rotation.x = -Math.PI / 2;
  halo.position.y = 0.02;
  scene.add(halo);
}

function createPointCloudGroup(model, points) {
  const sampled = samplePoints(points, model.maxDisplayPoints ?? 32000);
  const bounds = pointBounds(sampled);
  const center = bounds.center;
  const span = Math.max(bounds.size.x, bounds.size.y, bounds.size.z, 0.001);
  const scale = 1.45 / span;
  const positions = new Float32Array(sampled.length * 3);
  const colors = new Float32Array(sampled.length * 3);
  const fallback = new THREE.Color(model.accent);
  let minY = Infinity;

  sampled.forEach((point, index) => {
    const x = (Number(point.x) - center.x) * scale;
    const y = (Number(point.y) - center.y) * scale;
    const z = (Number(point.z) - center.z) * scale;
    positions[index * 3] = x;
    positions[index * 3 + 1] = y;
    positions[index * 3 + 2] = z;
    minY = Math.min(minY, y);
    const color = Array.isArray(point.color) ? point.color : null;
    colors[index * 3] = color ? color[0] / 255 : fallback.r;
    colors[index * 3 + 1] = color ? color[1] / 255 : fallback.g;
    colors[index * 3 + 2] = color ? color[2] / 255 : fallback.b;
  });

  for (let index = 1; index < positions.length; index += 3) {
    positions[index] -= minY;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();

  const material = new THREE.PointsMaterial({
    size: 0.026,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: 0.92,
    depthWrite: false,
  });

  const group = baseModelGroup(model);
  const cloud = new THREE.Points(geometry, material);
  cloud.userData.role = "gaussian-cloud";
  group.add(cloud);
  group.add(corePointMesh(-minY, model.accent));
  group.add(coreGlow(-minY, model.accent));
  group.add(selectionRing(model.accent));

  const objectIds = new Set(sampled.map((point) => Number(point.objectId ?? 0)));
  return {
    group,
    summary: {
      displayCount: sampled.length,
      objectCount: objectIds.size,
      corePoint: [round3(center.x), round3(center.y), round3(center.z)],
    },
  };
}

function createCompressedModelGroup(model) {
  const group = baseModelGroup(model);
  const points = syntheticGaussianShell(model.id, 1800);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(points.positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(points.colors, 3));
  const material = new THREE.PointsMaterial({
    size: 0.038,
    sizeAttenuation: true,
    vertexColors: true,
    transparent: true,
    opacity: 0.48,
    depthWrite: false,
  });
  group.add(new THREE.Points(geometry, material));
  group.add(corePointMesh(0.72, model.accent));
  group.add(coreGlow(0.72, model.accent));
  group.add(selectionRing(model.accent));
  return {
    group,
    summary: {
      displayCount: points.positions.length / 3,
      objectCount: model.objectCount,
      corePoint: [0, 0, 0],
    },
  };
}

function baseModelGroup(model) {
  const group = new THREE.Group();
  group.name = model.name;
  group.position.set(...model.galleryPosition);
  group.userData = { modelId: model.id, draggable: true };
  return group;
}

function corePointMesh(y, accent) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.06, 20, 20),
    new THREE.MeshBasicMaterial({ color: accent }),
  );
  mesh.position.set(0, y, 0);
  mesh.userData.role = "core-point";
  return mesh;
}

function coreGlow(y, accent) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.15, 24, 24),
    new THREE.MeshBasicMaterial({
      color: accent,
      transparent: true,
      opacity: 0.18,
    }),
  );
  mesh.position.set(0, y, 0);
  mesh.visible = false;
  mesh.userData.role = "core-glow";
  return mesh;
}

function selectionRing(accent) {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.86, 0.9, 80),
    new THREE.MeshBasicMaterial({
      color: accent,
      transparent: true,
      opacity: 0.88,
      side: THREE.DoubleSide,
    }),
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.018;
  ring.visible = false;
  ring.userData.role = "selection-ring";
  return ring;
}

function nearestModelId(object) {
  let cursor = object;
  while (cursor) {
    if (cursor.userData?.modelId) return cursor.userData.modelId;
    cursor = cursor.parent;
  }
  return null;
}

function samplePoints(points, maxPoints) {
  if (points.length <= maxPoints) return points;
  const stride = Math.ceil(points.length / maxPoints);
  const sampled = [];
  for (let index = 0; index < points.length && sampled.length < maxPoints; index += stride) {
    sampled.push(points[index]);
  }
  return sampled;
}

function pointBounds(points) {
  const min = new THREE.Vector3(Infinity, Infinity, Infinity);
  const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
  for (const point of points) {
    min.x = Math.min(min.x, Number(point.x) || 0);
    min.y = Math.min(min.y, Number(point.y) || 0);
    min.z = Math.min(min.z, Number(point.z) || 0);
    max.x = Math.max(max.x, Number(point.x) || 0);
    max.y = Math.max(max.y, Number(point.y) || 0);
    max.z = Math.max(max.z, Number(point.z) || 0);
  }
  const center = min.clone().add(max).multiplyScalar(0.5);
  const size = max.clone().sub(min);
  return { min, max, center, size };
}

function syntheticGaussianShell(seedText, count) {
  const seed = [...seedText].reduce((total, char) => total + char.charCodeAt(0), 0);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const colorA = new THREE.Color("#1fd1d9");
  const colorB = new THREE.Color("#8e6cff");
  for (let index = 0; index < count; index += 1) {
    const t = index / count;
    const angle = t * Math.PI * 9 + seed * 0.021;
    const radius = 0.35 + 0.5 * Math.sin(t * Math.PI);
    positions[index * 3] = Math.cos(angle) * radius;
    positions[index * 3 + 1] = 0.18 + t * 1.15 + Math.sin(angle * 1.7) * 0.11;
    positions[index * 3 + 2] = Math.sin(angle) * radius * 0.78;
    const mix = (Math.sin(angle) + 1) / 2;
    const color = colorA.clone().lerp(colorB, mix);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  }
  return { positions, colors };
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Meta({ label, value }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value || "-"}</dd>
    </>
  );
}

function initialModelStates() {
  return Object.fromEntries(
    MODEL_CATALOG.map((model) => [
      model.id,
      {
        ...model,
        status: "queued",
        message: "queued",
        gaussianCount: 0,
        displayCount: 0,
        corePoint: null,
        loadMs: 0,
      },
    ]),
  );
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number.toLocaleString() : "-";
}

function formatVec(value) {
  return Array.isArray(value) ? value.map((entry) => Number(entry).toFixed(3)).join(", ") : "-";
}

function round3(value) {
  return Math.round(Number(value) * 1000) / 1000;
}
