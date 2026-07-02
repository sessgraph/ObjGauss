import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { DragControls } from "three/examples/jsm/controls/DragControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { MODEL_CATALOG, catalogSummary } from "./modelCatalog.js";
import { colorForObject, rgbToCss } from "./palette.js";
import { parsePly } from "./ply.js";

const INITIAL_CAMERA = {
  position: [0, 5.4, 10.8],
  target: [0, 1.15, 0],
};

export default function App() {
  const worldApi = useRef(null);
  const loadStarted = useRef(false);
  const [worldReady, setWorldReady] = useState(false);
  const [selection, setSelection] = useState(() => ({
    modelId: MODEL_CATALOG[0]?.id ?? "",
    objectId: null,
    selectionId: MODEL_CATALOG[0]?.id ?? "",
  }));
  const [models, setModels] = useState(() => initialModelStates());
  const summary = useMemo(() => catalogSummary(MODEL_CATALOG), []);
  const loadedCount = useMemo(
    () => Object.values(models).filter((model) => ["loaded", "compressed"].includes(model.status)).length,
    [models],
  );
  const objectCount = useMemo(
    () =>
      Object.values(models).reduce(
        (total, model) => total + (model.objects?.length || Number(model.objectCount) || 0),
        0,
      ),
    [models],
  );
  const selectedId = selection.modelId;
  const selected = models[selectedId] ?? Object.values(models)[0];
  const selectedObject =
    selected?.objects?.find((object) => String(object.objectId) === String(selection.objectId)) ?? null;

  const patchModel = useCallback((id, patch) => {
    setModels((current) => {
      const previous = current[id];
      if (!previous) return current;
      const nextPatch = typeof patch === "function" ? patch(previous) : patch;
      if (!nextPatch) return current;
      return {
        ...current,
        [id]: {
          ...previous,
          ...nextPatch,
        },
      };
    });
  }, []);

  const selectModel = useCallback((id) => {
    setSelection({ modelId: id, objectId: null, selectionId: id });
    worldApi.current?.focusModel(id);
  }, []);

  const selectObject = useCallback((target) => {
    if (!target?.modelId) return;
    setSelection({
      modelId: target.modelId,
      objectId: target.objectId,
      selectionId: target.selectionId,
    });
  }, []);

  const handleObjectMoved = useCallback(
    (target, position) => {
      if (!target?.modelId) return;
      patchModel(target.modelId, (current) => ({
        objects: (current.objects ?? []).map((object) =>
          String(object.objectId) === String(target.objectId)
            ? { ...object, galleryPosition: position }
            : object,
        ),
      }));
    },
    [patchModel],
  );

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
            objects: rendered?.objects ?? [],
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
            objects: rendered?.objects ?? [],
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
      data-object-count={objectCount}
      data-selected-object={selection.objectId ?? ""}
      data-selected-target={selection.selectionId ?? selected?.id ?? ""}
      data-compression-layout="per-object-corepoint-chunks"
    >
      <ThreeWorld
        models={MODEL_CATALOG}
        selectedTargetId={selection.selectionId || selectedId}
        onReady={handleWorldReady}
        onSelectObject={selectObject}
        onObjectMoved={handleObjectMoved}
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
          <Metric label="可交互" value={loadedCount} />
          <Metric label="对象" value={objectCount} />
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
              <span>
                {selectedObject ? `Object ${selectedObject.objectId} / ${selected.kind}` : selected.kind}
              </span>
            </div>
          </div>
          <dl className="metaGrid">
            <Meta label="加载状态" value={selected.message ?? selected.status} />
            <Meta label="点数" value={formatNumber(selected.gaussianCount)} />
            <Meta label={selectedObject ? "对象展示点" : "展示点"} value={formatNumber(selectedObject?.displayCount ?? selected.displayCount)} />
            <Meta label="对象" value={formatNumber(selected.objectCount)} />
            <Meta label="对象 ID" value={selectedObject ? String(selectedObject.objectId) : "-"} />
            <Meta label="核心点" value={formatVec(selectedObject?.corePoint ?? selected.corePoint)} />
            <Meta label="对象位置" value={formatVec(selectedObject?.galleryPosition)} />
            <Meta label="加载耗时" value={selected.loadMs ? `${selected.loadMs} ms` : "-"} />
            <Meta label="压缩布局" value={selected.compression?.layout ?? "-"} />
            <Meta label="分块路径" value={selectedObject?.chunkPath ?? selected.compression?.chunkRoot ?? "-"} />
          </dl>
        </section>
      ) : null}

      <div className="glassHud bottomStatus">
        <span>拖动任意对象即可独立移动</span>
        <span>核心点为每个处理后 Gaussian 对象的加载锚点</span>
        <span>后端负责单对象压缩块与按需加载</span>
      </div>
    </main>
  );
}

function ThreeWorld({ models, selectedTargetId, onReady, onSelectObject, onObjectMoved }) {
  const mountRef = useRef(null);
  const apiRef = useRef(null);
  const selectedRef = useRef(selectedTargetId);
  const callbacksRef = useRef({ onSelectObject, onObjectMoved });

  useEffect(() => {
    selectedRef.current = selectedTargetId;
    apiRef.current?.setSelected(selectedTargetId);
  }, [selectedTargetId]);

  useEffect(() => {
    callbacksRef.current = { onSelectObject, onObjectMoved };
  }, [onSelectObject, onObjectMoved]);

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

    const modelRoots = new Map();
    const draggableObjects = new Map();
    let dragControls = null;
    let animationFrame = 0;

    const publishAuditHandle = () => {
      const selectedObject = draggableObjects.get(selectedRef.current);
      const selectedModelId =
        selectedObject?.userData.modelId ??
        (modelRoots.has(selectedRef.current) ? selectedRef.current : null);
      window.__OBJGAUSS_WORLD__ = {
        renderer: "three.js",
        ui: "frosted-glass-in-world",
        sidebars: false,
        modelCount: models.length,
        objectCount: draggableObjects.size,
        draggableCount: draggableObjects.size,
        draggableObjectCount: draggableObjects.size,
        selectedId: selectedRef.current,
        selectedModelId,
        selectedObjectId: selectedObject?.userData.objectId ?? null,
        modelPositions: [...modelRoots.values()].map((object) => ({
          id: object.userData.modelId,
          position: object.position.toArray().map(round3),
        })),
        objectSelections: [...draggableObjects.values()].map((object) => {
          const worldPosition = new THREE.Vector3();
          object.getWorldPosition(worldPosition);
          return {
            selectionId: object.userData.selectionId,
            modelId: object.userData.modelId,
            objectId: object.userData.objectId,
            position: worldPosition.toArray().map(round3),
          };
        }),
        selectObjectForAudit(selectionId = null) {
          const object =
            (selectionId ? draggableObjects.get(selectionId) : null) ??
            [...draggableObjects.values()][0];
          if (!object) return false;
          selectObjectGroup(object);
          return true;
        },
      };
    };

    const rebuildDragControls = () => {
      dragControls?.dispose();
      dragControls = new DragControls([...draggableObjects.values()], camera, renderer.domElement);
      dragControls.transformGroup = true;
      dragControls.addEventListener("dragstart", (event) => {
        controls.enabled = false;
        selectObjectGroup(event.object);
      });
      dragControls.addEventListener("drag", (event) => {
        event.object.position.y = 0;
      });
      dragControls.addEventListener("dragend", (event) => {
        controls.enabled = true;
        callbacksRef.current.onObjectMoved?.(objectTarget(event.object), [
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
      const previous = modelRoots.get(model.id);
      if (previous) {
        scene.remove(previous);
        disposeObject(previous);
      }
      for (const [selectionId, object] of draggableObjects) {
        if (object.userData.modelId === model.id) draggableObjects.delete(selectionId);
      }
      const result = points?.length ? createPointCloudGroup(model, points) : createCompressedModelGroup(model);
      scene.add(result.group);
      modelRoots.set(model.id, result.group);
      result.objectGroups.forEach((object) => {
        draggableObjects.set(object.userData.selectionId, object);
      });
      rebuildDragControls();
      api.setSelected(selectedRef.current);
      return result.summary;
    };

    const selectObjectGroup = (object) => {
      const target = objectTarget(object);
      if (!target?.selectionId) return;
      selectedRef.current = target.selectionId;
      api.setSelected(target.selectionId);
      callbacksRef.current.onSelectObject?.(target);
      publishAuditHandle();
    };

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const onPointerDown = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const intersections = raycaster.intersectObjects([...draggableObjects.values()], true);
      const hit = intersections[0]?.object;
      const object = nearestObjectGroup(hit);
      if (object) selectObjectGroup(object);
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);

    const api = {
      upsertModel,
      focusModel(id) {
        const object = modelRoots.get(id);
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
        for (const object of draggableObjects.values()) {
          const selected = object.userData.selectionId === id || object.userData.modelId === id;
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
      modelRoots.forEach(disposeObject);
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
    new THREE.CircleGeometry(12.6, 128),
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

  const grid = new THREE.GridHelper(22, 44, "#2a5f6a", "#152932");
  grid.material.transparent = true;
  grid.material.opacity = 0.58;
  scene.add(grid);

  const halo = new THREE.Mesh(
    new THREE.RingGeometry(6.2, 6.25, 128),
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
  const scale = (model.displayScale ?? 2.35) / span;
  const normalizedByObject = new Map();
  let minY = Infinity;

  sampled.forEach((point) => {
    const objectId = normalizedObjectId(point);
    const x = (Number(point.x) - center.x) * scale;
    const y = (Number(point.y) - center.y) * scale;
    const z = (Number(point.z) - center.z) * scale;
    minY = Math.min(minY, y);
    if (!normalizedByObject.has(objectId)) normalizedByObject.set(objectId, []);
    normalizedByObject.get(objectId).push({ point, x, y, z });
  });

  const group = baseModelGroup(model);
  const objectGroups = [];
  const objects = [];

  for (const [objectId, entries] of [...normalizedByObject.entries()].sort(sortObjectEntries)) {
    entries.forEach((entry) => {
      entry.y -= minY;
    });
    const normalizedBounds = pointBounds(entries);
    const originalBounds = pointBounds(entries.map((entry) => entry.point));
    const objectBoost = objectDisplayBoost(normalizedBounds, model);
    const accent = objectAccent(objectId, model.accent);
    const objectGroup = baseObjectGroup(model, objectId, {
      x: normalizedBounds.center.x,
      y: 0,
      z: normalizedBounds.center.z,
    });
    const positions = new Float32Array(entries.length * 3);
    const colors = new Float32Array(entries.length * 3);
    const fallback = new THREE.Color(accent);

    entries.forEach((entry, index) => {
      positions[index * 3] = (entry.x - objectGroup.position.x) * objectBoost;
      positions[index * 3 + 1] = (entry.y - normalizedBounds.min.y) * objectBoost;
      positions[index * 3 + 2] = (entry.z - objectGroup.position.z) * objectBoost;
      const color = Array.isArray(entry.point.color) ? entry.point.color : null;
      colors[index * 3] = color ? color[0] / 255 : fallback.r;
      colors[index * 3 + 1] = color ? color[1] / 255 : fallback.g;
      colors[index * 3 + 2] = color ? color[2] / 255 : fallback.b;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();

    const material = new THREE.PointsMaterial({
      size: model.pointSize ?? 0.033,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 0.94,
      depthWrite: false,
    });

    const cloud = new THREE.Points(geometry, material);
    cloud.userData.role = "gaussian-cloud";
    objectGroup.add(cloud);
    objectGroup.add(corePointMesh((normalizedBounds.center.y - normalizedBounds.min.y) * objectBoost, accent));
    objectGroup.add(coreGlow((normalizedBounds.center.y - normalizedBounds.min.y) * objectBoost, accent));
    objectGroup.add(selectionRing(accent, ringRadiusForBounds(normalizedBounds, objectBoost)));
    group.add(objectGroup);
    objectGroups.push(objectGroup);
    objects.push({
      objectId,
      selectionId: selectionIdForObject(model.id, objectId),
      displayCount: entries.length,
      corePoint: [
        round3(originalBounds.center.x),
        round3(originalBounds.center.y),
        round3(originalBounds.center.z),
      ],
      galleryPosition: objectGroup.position.toArray().map(round3),
      chunkPath: objectChunkPath(model, objectId),
      accent,
    });
  }

  return {
    group,
    objectGroups,
    summary: {
      displayCount: sampled.length,
      objectCount: objects.length,
      corePoint: [round3(center.x), round3(center.y), round3(center.z)],
      objects,
    },
  };
}

function createCompressedModelGroup(model) {
  const group = baseModelGroup(model);
  const objectCount = Math.max(1, Number(model.objectCount) || 1);
  const objectGroups = [];
  const objects = [];
  let displayCount = 0;
  for (let index = 0; index < objectCount; index += 1) {
    const objectId = index;
    const accent = objectAccent(objectId, model.accent);
    const position = compressedObjectPosition(index, objectCount, model.displayScale ?? 2.1);
    const objectGroup = baseObjectGroup(model, objectId, position);
    const points = syntheticGaussianShell(`${model.id}-${objectId}`, model.placeholderPointsPerObject ?? 760, accent);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(points.positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(points.colors, 3));
    geometry.computeBoundingSphere();
    const material = new THREE.PointsMaterial({
      size: model.pointSize ?? 0.04,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 0.58,
      depthWrite: false,
    });
    const cloud = new THREE.Points(geometry, material);
    cloud.userData.role = "gaussian-cloud";
    objectGroup.add(cloud);
    objectGroup.add(corePointMesh(0.72, accent));
    objectGroup.add(coreGlow(0.72, accent));
    objectGroup.add(selectionRing(accent, 0.54));
    group.add(objectGroup);
    objectGroups.push(objectGroup);
    displayCount += points.positions.length / 3;
    objects.push({
      objectId,
      selectionId: selectionIdForObject(model.id, objectId),
      displayCount: points.positions.length / 3,
      corePoint: [0, 0, 0],
      galleryPosition: objectGroup.position.toArray().map(round3),
      chunkPath: objectChunkPath(model, objectId),
      accent,
    });
  }
  return {
    group,
    objectGroups,
    summary: {
      displayCount,
      objectCount: objects.length,
      corePoint: [0, 0, 0],
      objects,
    },
  };
}

function baseModelGroup(model) {
  const group = new THREE.Group();
  group.name = model.name;
  group.position.set(...model.galleryPosition);
  group.userData = { modelId: model.id, draggable: false };
  return group;
}

function baseObjectGroup(model, objectId, position) {
  const group = new THREE.Group();
  group.name = `${model.name} / object ${objectId}`;
  group.position.set(position.x, position.y ?? 0, position.z);
  group.userData = {
    modelId: model.id,
    objectId,
    selectionId: selectionIdForObject(model.id, objectId),
    draggable: true,
  };
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

function selectionRing(accent, radius = 0.86) {
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(radius, radius + 0.045, 80),
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

function nearestObjectGroup(object) {
  let cursor = object;
  while (cursor) {
    if (cursor.userData?.selectionId) return cursor;
    cursor = cursor.parent;
  }
  return null;
}

function objectTarget(object) {
  if (!object?.userData?.selectionId) return null;
  return {
    modelId: object.userData.modelId,
    objectId: object.userData.objectId,
    selectionId: object.userData.selectionId,
  };
}

function samplePoints(points, maxPoints) {
  if (points.length <= maxPoints) return points;
  const byObject = new Map();
  points.forEach((point) => {
    const objectId = normalizedObjectId(point);
    if (!byObject.has(objectId)) byObject.set(objectId, []);
    byObject.get(objectId).push(point);
  });
  const sampled = [];
  const total = points.length;
  const entries = [...byObject.entries()].sort(sortObjectEntries);
  for (const [, objectPoints] of entries) {
    const quota = Math.max(1, Math.floor((objectPoints.length / total) * maxPoints));
    const stride = Math.max(1, Math.ceil(objectPoints.length / quota));
    for (let index = 0; index < objectPoints.length && sampled.length < maxPoints; index += stride) {
      sampled.push(objectPoints[index]);
    }
  }
  let cursor = 0;
  while (sampled.length < maxPoints && sampled.length < points.length) {
    sampled.push(points[cursor]);
    cursor = (cursor + Math.ceil(points.length / maxPoints)) % points.length;
  }
  return sampled;
}

function normalizedObjectId(point) {
  const objectId = Number(point?.objectId ?? 0);
  return Number.isFinite(objectId) ? Math.trunc(objectId) : 0;
}

function sortObjectEntries(left, right) {
  return Number(left[0]) - Number(right[0]);
}

function selectionIdForObject(modelId, objectId) {
  return `${modelId}::object-${objectId}`;
}

function objectChunkPath(model, objectId) {
  const root = model.compression?.chunkRoot ?? `/models/${model.id}/objects/`;
  return `${root.replace(/\/?$/, "/")}${objectId}/`;
}

function objectAccent(objectId, fallback) {
  const rgb = colorForObject(objectId);
  return Array.isArray(rgb) ? rgbToCss(rgb) : fallback;
}

function objectDisplayBoost(bounds, model) {
  const span = Math.max(bounds.size.x, bounds.size.y, bounds.size.z, 0.001);
  const targetSpan = model.minObjectDisplaySpan ?? 1.02;
  const maxBoost = model.maxObjectDisplayBoost ?? 4.2;
  return Math.max(1, Math.min(maxBoost, targetSpan / span));
}

function ringRadiusForBounds(bounds, boost = 1) {
  const horizontal = Math.max(bounds.size.x, bounds.size.z) * boost;
  return Math.max(0.42, Math.min(1.36, horizontal * 0.58 + 0.2));
}

function compressedObjectPosition(index, count, displayScale) {
  if (count === 1) return { x: 0, y: 0, z: 0 };
  const columns = Math.ceil(Math.sqrt(count));
  const row = Math.floor(index / columns);
  const column = index % columns;
  const rows = Math.ceil(count / columns);
  const spacing = Math.max(0.56, Math.min(0.82, displayScale / Math.max(columns, rows)));
  return {
    x: (column - (columns - 1) / 2) * spacing,
    y: 0,
    z: (row - (rows - 1) / 2) * spacing,
  };
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

function syntheticGaussianShell(seedText, count, accent) {
  const seed = [...seedText].reduce((total, char) => total + char.charCodeAt(0), 0);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const colorA = new THREE.Color(accent);
  const colorB = new THREE.Color("#dffaff");
  for (let index = 0; index < count; index += 1) {
    const t = index / count;
    const angle = t * Math.PI * 9 + seed * 0.021;
    const radius = 0.35 + 0.5 * Math.sin(t * Math.PI);
    positions[index * 3] = Math.cos(angle) * radius * 1.08;
    positions[index * 3 + 1] = 0.18 + t * 1.24 + Math.sin(angle * 1.7) * 0.11;
    positions[index * 3 + 2] = Math.sin(angle) * radius * 0.86;
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
  const displayValue = value === null || value === undefined || value === "" ? "-" : value;
  return (
    <>
      <dt>{label}</dt>
      <dd>{displayValue}</dd>
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
        objects: [],
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
