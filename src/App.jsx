import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { DragControls } from "three/examples/jsm/controls/DragControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { MODEL_CATALOG, catalogSummary } from "./modelCatalog.js";
import { browserReadyArtifact } from "./modelArtifactManifest.js";
import { decodeQuantizedOgcPayload } from "./ogcDecoder.js";
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
  const [debugMode, setDebugMode] = useState(true);
  const [hoveredTarget, setHoveredTarget] = useState(null);
  const [debugProbe, setDebugProbe] = useState(null);
  const [hiddenObjects, setHiddenObjects] = useState(() => new Set());
  const summary = useMemo(() => catalogSummary(MODEL_CATALOG), []);
  const loadedCount = useMemo(
    () => Object.values(models).filter((model) => ["loaded", "compressed"].includes(model.status)).length,
    [models],
  );
  const ogcLoadedCount = useMemo(
    () =>
      Object.values(models).filter(
        (model) => model.status === "loaded" && model.delivery?.source === "quantized-ogc",
      ).length,
    [models],
  );
  const trainableArtifactLoadedCount = useMemo(
    () =>
      Object.values(models).filter(
        (model) => model.status === "loaded" && model.delivery?.source === "trainable-kernel-model-artifact",
      ).length,
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
  const selectedObjectKey = selectedObject?.selectionId ?? "";
  const selectedAssignmentSource =
    debugProbe?.source ?? selectedObject?.objectState?.source ?? selected?.delivery?.source ?? "";
  const hiddenCount = hiddenObjects.size;

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

  const selectObject = useCallback((target, probe = null) => {
    if (!target?.modelId) return;
    setSelection({
      modelId: target.modelId,
      objectId: target.objectId,
      selectionId: target.selectionId,
    });
    setDebugProbe(probe);
  }, []);

  const handleHoverObject = useCallback((target) => {
    setHoveredTarget(target ?? null);
  }, []);

  const toggleDebugMode = useCallback(() => {
    setDebugMode((enabled) => {
      const next = !enabled;
      worldApi.current?.setDebugMode(next);
      return next;
    });
  }, []);

  const toggleObjectVisibility = useCallback((object) => {
    if (!object?.selectionId) return;
    setHiddenObjects((current) => {
      const next = new Set(current);
      const willHide = !next.has(object.selectionId);
      if (willHide) {
        next.add(object.selectionId);
      } else {
        next.delete(object.selectionId);
      }
      worldApi.current?.setObjectVisibility(object.selectionId, !willHide);
      return next;
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
        if (model.loadMode === "ogc-chunked") {
          const startedAt = performance.now();
          patchModel(model.id, { status: "loading", message: "loading ogc chunks" });
          try {
            const { artifact, decoded } = await loadOgcModel(model);
            const rendered = worldApi.current?.upsertModel(model, decoded.points);
            if (cancelled) return;
            patchModel(model.id, {
              status: "loaded",
              message: "ogc chunks",
              gaussianCount: decoded.points.length,
              displayCount: rendered?.displayCount ?? 0,
              objectCount: rendered?.objectCount ?? decoded.metadata.objectCount ?? model.objectCount,
              corePoint: rendered?.corePoint ?? null,
              objects: rendered?.objects ?? [],
              loadMs: Math.round(performance.now() - startedAt),
              delivery: {
                source: "quantized-ogc",
                role: artifact.role,
                decodedChunks: decoded.metadata.decodedChunks,
                decodedGaussians: decoded.metadata.decodedGaussians,
                recordFormat: decoded.metadata.recordFormat,
                lodLevel: model.ogc?.lodLevel ?? "full",
              },
            });
          } catch (error) {
            if (cancelled) return;
            patchModel(model.id, {
              status: "error",
              message: error?.message ?? "ogc load failed",
            });
          }
          continue;
        }
        if (model.loadMode === "trainable-artifact") {
          const startedAt = performance.now();
          patchModel(model.id, { status: "loading", message: "loading trained artifact" });
          try {
            const rendered = worldApi.current?.upsertModel(model, null);
            if (cancelled) return;
            patchModel(model.id, {
              status: "loaded",
              message: "trained artifact",
              gaussianCount: rendered?.gaussianCount ?? rendered?.displayCount ?? 0,
              displayCount: rendered?.displayCount ?? 0,
              objectCount: rendered?.objectCount ?? model.objectCount,
              corePoint: rendered?.corePoint ?? null,
              objects: rendered?.objects ?? [],
              loadMs: Math.round(performance.now() - startedAt),
              delivery: {
                source: "trainable-kernel-model-artifact",
                schema: model.trainableArtifact?.schema,
                rendererName: model.trainableArtifact?.renderer_api?.renderer_name,
                imageRenderLoss: model.trainableArtifact?.renderer_api?.image_render_loss,
                gradientPath: model.trainableArtifact?.renderer_api?.gradient_path,
              },
            });
          } catch (error) {
            if (cancelled) return;
            patchModel(model.id, {
              status: "error",
              message: error?.message ?? "trainable artifact load failed",
            });
          }
          continue;
        }
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
      data-debug-os="object-state"
      data-assignment-debug={debugMode ? "enabled" : "disabled"}
      data-selected-gaussian={debugProbe?.gaussianIndex ?? ""}
      data-hidden-objects={hiddenCount}
      data-ogc-loaded-count={ogcLoadedCount}
      data-trainable-artifact-loaded-count={trainableArtifactLoadedCount}
      data-assignment-source={selectedAssignmentSource}
    >
      <ThreeWorld
        models={MODEL_CATALOG}
        selectedTargetId={selection.selectionId || selectedId}
        debugMode={debugMode}
        hiddenSelectionIds={hiddenObjects}
        onReady={handleWorldReady}
        onSelectObject={selectObject}
        onHoverObject={handleHoverObject}
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
        <div className="topActions">
          <button
            className={`glassButton ${debugMode ? "active" : ""}`}
            type="button"
            data-assignment-debug-toggle="true"
            onClick={toggleDebugMode}
          >
            A[N,K]
          </button>
          <button className="glassButton" type="button" onClick={() => worldApi.current?.resetCamera()}>
            重置视角
          </button>
        </div>
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
            <Meta label="交付源" value={selected.delivery?.source ?? "-"} />
            <Meta label="OGC chunks" value={selected.delivery?.decodedChunks ?? "-"} />
            <Meta label="assignment" value={selectedAssignmentSource} />
            <Meta label="renderer loss" value={formatLoss(selected.delivery?.imageRenderLoss)} />
          </dl>
        </section>
      ) : null}

      <DebugPanel
        selected={selected}
        selectedObject={selectedObject}
        selectedObjectKey={selectedObjectKey}
        hoveredTarget={hoveredTarget}
        debugProbe={debugProbe}
        debugMode={debugMode}
        hiddenObjects={hiddenObjects}
        onToggleObjectVisibility={toggleObjectVisibility}
      />

      <div className="glassHud bottomStatus">
        <span>Phase 1 Debug OS: A[N,K] / ObjectState / Gaussian probe</span>
        <span>点击 Gaussian 查看 assignment vector</span>
        <span>对象开关用于验证 cluster 是否独立</span>
      </div>
    </main>
  );
}

async function loadOgcModel(model) {
  const artifact = browserReadyArtifact(model, "compressed_chunked");
  if (!artifact) {
    throw new Error("missing browser-ready compressed_chunked artifact");
  }
  const index = await loadOgcIndex(artifact);
  const payload = await loadOgcPayload(artifact, index);
  return {
    artifact,
    decoded: decodeQuantizedOgcPayload(payload, index, {
      chunkIds: model.ogc?.chunkIds,
      lodLevel: model.ogc?.lodLevel,
    }),
  };
}

async function loadOgcIndex(artifact) {
  if (artifact.inlineIndex) return artifact.inlineIndex;
  const indexPath = artifact.indexPath ?? artifact.chunk_index?.path;
  if (!indexPath || isInlineRoute(indexPath)) {
    throw new Error("missing fetchable OGC chunk index");
  }
  const response = await fetch(indexPath);
  if (!response.ok) throw new Error(`OGC index HTTP ${response.status}`);
  return response.json();
}

async function loadOgcPayload(artifact, index) {
  if (artifact.payloadBase64) return base64ToArrayBuffer(artifact.payloadBase64);
  const payloadPath = artifact.payloadPath ?? artifact.path ?? index?.payload?.path;
  if (!payloadPath || isInlineRoute(payloadPath)) {
    throw new Error("missing fetchable OGC payload");
  }
  const response = await fetch(payloadPath);
  if (!response.ok) throw new Error(`OGC payload HTTP ${response.status}`);
  return response.arrayBuffer();
}

function base64ToArrayBuffer(value) {
  const binary = atob(String(value));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes.buffer;
}

function isInlineRoute(path) {
  return String(path).startsWith("inline://");
}

function ThreeWorld({
  models,
  selectedTargetId,
  debugMode,
  hiddenSelectionIds,
  onReady,
  onSelectObject,
  onHoverObject,
  onObjectMoved,
}) {
  const mountRef = useRef(null);
  const apiRef = useRef(null);
  const selectedRef = useRef(selectedTargetId);
  const debugRef = useRef(debugMode);
  const hiddenRef = useRef(hiddenSelectionIds);
  const callbacksRef = useRef({ onSelectObject, onHoverObject, onObjectMoved });

  useEffect(() => {
    selectedRef.current = selectedTargetId;
    apiRef.current?.setSelected(selectedTargetId);
  }, [selectedTargetId]);

  useEffect(() => {
    debugRef.current = debugMode;
    apiRef.current?.setDebugMode(debugMode);
  }, [debugMode]);

  useEffect(() => {
    hiddenRef.current = hiddenSelectionIds;
    apiRef.current?.setHiddenObjects(hiddenSelectionIds);
  }, [hiddenSelectionIds]);

  useEffect(() => {
    callbacksRef.current = { onSelectObject, onHoverObject, onObjectMoved };
  }, [onSelectObject, onHoverObject, onObjectMoved]);

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
    let hoveredObject = null;
    let dragControls = null;
    let animationFrame = 0;

    const publishAuditHandle = () => {
      const selectedObject = draggableObjects.get(selectedRef.current);
      const selectedModelId =
        selectedObject?.userData.modelId ??
        (modelRoots.has(selectedRef.current) ? selectedRef.current : null);
      const selectedModel = selectedModelId ? modelRoots.get(selectedModelId) : null;
      const selectedAssignmentSource =
        selectedObject?.userData.objectState?.source ??
        selectedModel?.userData.assignmentSource ??
        "derived_from_object_id";
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
        hoveredId: hoveredObject?.userData.selectionId ?? null,
        debugMode: debugRef.current,
        debugProtocol: "object-state-debug-os-v1",
        assignmentSource: selectedAssignmentSource,
        trainableArtifactLoadedCount: [...modelRoots.values()].filter(
          (object) => object.userData?.artifactSchema === "objgauss-trainable-kernel-model-artifact-v1",
        ).length,
        visibleObjectCount: [...draggableObjects.values()].filter((object) => object.visible).length,
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
            visible: object.visible,
            confidence: object.userData.objectState?.confidence ?? null,
            entropy: object.userData.objectState?.assignmentEntropy ?? null,
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
        selectGaussianForAudit(selectionId = null, gaussianIndex = 0) {
          const object =
            (selectionId ? draggableObjects.get(selectionId) : null) ??
            [...draggableObjects.values()].find((entry) => entry.visible);
          if (!object) return false;
          const cloud = firstGaussianCloud(object);
          const probe = cloud?.userData.gaussianDebug?.[gaussianIndex];
          selectObjectGroup(object, probe ? { gaussian: probe } : null);
          return Boolean(probe);
        },
        toggleObjectVisibilityForAudit(selectionId = null) {
          const object =
            (selectionId ? draggableObjects.get(selectionId) : null) ??
            [...draggableObjects.values()][0];
          if (!object) return false;
          api.setObjectVisibility(object.userData.selectionId, !object.visible);
          publishAuditHandle();
          return object.visible;
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
      const result =
        model.loadMode === "trainable-artifact"
          ? createTrainableArtifactGroup(model)
          : points?.length
            ? createPointCloudGroup(model, points)
            : createCompressedModelGroup(model);
      scene.add(result.group);
      modelRoots.set(model.id, result.group);
      result.objectGroups.forEach((object) => {
        draggableObjects.set(object.userData.selectionId, object);
      });
      rebuildDragControls();
      api.setSelected(selectedRef.current);
      api.setDebugMode(debugRef.current);
      api.setHiddenObjects(hiddenRef.current);
      return result.summary;
    };

    const selectObjectGroup = (object, probe = null) => {
      const target = objectTarget(object);
      if (!target?.selectionId) return;
      selectedRef.current = target.selectionId;
      api.setSelected(target.selectionId);
      callbacksRef.current.onSelectObject?.(target, probe?.gaussian ?? null);
      publishAuditHandle();
    };

    const raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.12;
    const pointer = new THREE.Vector2();
    const pointerTarget = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const intersections = raycaster.intersectObjects([...draggableObjects.values()], true);
      const hit = intersections[0] ?? null;
      return {
        hit,
        object: nearestObjectGroup(hit?.object),
      };
    };
    const onPointerDown = (event) => {
      const target = pointerTarget(event);
      const probe = gaussianProbeFromIntersection(target.hit);
      if (target.object) selectObjectGroup(target.object, probe ? { gaussian: probe } : null);
    };
    const onPointerMove = (event) => {
      const target = pointerTarget(event);
      const nextHover = target.object ?? null;
      if (nextHover === hoveredObject) return;
      hoveredObject = nextHover;
      api.setHover(hoveredObject?.userData.selectionId ?? null);
      callbacksRef.current.onHoverObject?.(hoveredObject ? objectTarget(hoveredObject) : null);
      publishAuditHandle();
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);

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
      setDebugMode(enabled) {
        debugRef.current = Boolean(enabled);
        for (const object of draggableObjects.values()) {
          object.traverse((child) => {
            if (child.userData.role === "gaussian-cloud") {
              const color = enabled ? child.userData.assignmentColor : child.userData.originalColor;
              if (color) {
                child.geometry.setAttribute("color", color);
                child.geometry.attributes.color.needsUpdate = true;
              }
            }
            if (child.userData.role === "object-state-bbox") {
              child.visible = Boolean(enabled);
            }
          });
        }
        publishAuditHandle();
      },
      setHover(selectionId) {
        for (const object of draggableObjects.values()) {
          const hovered = object.userData.selectionId === selectionId;
          object.userData.hovered = hovered;
          applyObjectVisualState(object, {
            selected: object.userData.selected,
            hovered,
            debug: debugRef.current,
          });
        }
      },
      setObjectVisibility(selectionId, visible) {
        const object = draggableObjects.get(selectionId);
        if (!object) return;
        object.visible = Boolean(visible);
        publishAuditHandle();
      },
      setHiddenObjects(hiddenIds) {
        const hidden = new Set(hiddenIds ?? []);
        for (const object of draggableObjects.values()) {
          object.visible = !hidden.has(object.userData.selectionId);
        }
        publishAuditHandle();
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
          applyObjectVisualState(object, {
            selected,
            hovered: object.userData.hovered,
            debug: debugRef.current,
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
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
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

function DebugPanel({
  selected,
  selectedObject,
  selectedObjectKey,
  hoveredTarget,
  debugProbe,
  debugMode,
  hiddenObjects,
  onToggleObjectVisibility,
}) {
  if (!selected) return null;
  const objects = selected.objects ?? [];
  const activeState = selectedObject?.objectState ?? objects[0]?.objectState ?? null;
  const assignment = debugProbe?.assignment ?? selectedObject?.assignment ?? activeState?.assignment ?? [];
  const probeEntropy = debugProbe?.entropy ?? activeState?.assignmentEntropy ?? 0;
  const probeConfidence = debugProbe?.confidence ?? activeState?.confidence ?? 0;
  const rendererLoss = selected?.delivery?.imageRenderLoss;
  return (
    <section
      className="glassHud debugPanel"
      data-object-debug-panel="true"
      data-debug-mode={debugMode ? "assignment" : "appearance"}
      data-probe-source={debugProbe?.source ?? activeState?.source ?? "none"}
    >
      <div className="debugHeader">
        <div>
          <h2>ObjectState Debug</h2>
          <span>{debugMode ? "assignment projection" : "appearance view"}</span>
        </div>
        <strong>{selectedObject ? `#${selectedObject.objectId}` : selected.label}</strong>
      </div>

      <div className="debugMetrics">
        <Metric label="conf" value={formatRatio(probeConfidence)} />
        <Metric label="entropy" value={formatRatio(probeEntropy)} />
        <Metric label="mass" value={formatNumber(activeState?.slotMass)} />
        <Metric label="img loss" value={formatLoss(rendererLoss)} />
      </div>

      <AssignmentHeatmap assignment={assignment} selectedObject={selectedObject} debugProbe={debugProbe} />

      <dl className="debugStateGrid">
        <Meta label="source" value={debugProbe?.source ?? activeState?.source} />
        <Meta label="renderer" value={selected.delivery?.rendererName} />
        <Meta label="gaussian n" value={debugProbe?.gaussianIndex ?? "-"} />
        <Meta label="centroid" value={formatVec(activeState?.centroid)} />
        <Meta label="bbox" value={formatBox(activeState?.bbox)} />
        <Meta label="hover" value={hoveredTarget ? `${hoveredTarget.modelId} #${hoveredTarget.objectId}` : "-"} />
        <Meta label="hidden" value={hiddenObjects.size} />
      </dl>

      <div className="objectStateList" data-object-toggle-list="true">
        {objects.map((object) => {
          const hidden = hiddenObjects.has(object.selectionId);
          const selectedRow = object.selectionId === selectedObjectKey;
          return (
            <button
              key={object.selectionId}
              type="button"
              className={`objectStateRow ${selectedRow ? "selected" : ""} ${hidden ? "hidden" : ""}`}
              data-object-toggle={object.selectionId}
              data-object-visible={hidden ? "false" : "true"}
              onClick={() => onToggleObjectVisibility(object)}
            >
              <span className="modelAccent" style={{ background: object.accent }} />
              <span>#{object.objectId}</span>
              <small>{formatRatio(object.objectState?.confidence)}</small>
              <i>{hidden ? "off" : "on"}</i>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function AssignmentHeatmap({ assignment, selectedObject, debugProbe }) {
  const rows = assignment?.length ? assignment : [];
  return (
    <div
      className="assignmentHeatmap"
      data-assignment-heatmap="true"
      data-assignment-source={debugProbe?.source ?? selectedObject?.objectState?.source ?? "none"}
      data-assignment-slots={rows.length}
    >
      {rows.map((slot) => {
        const width = `${Math.max(2, Math.min(100, Number(slot.probability) * 100))}%`;
        const color = objectAccent(slot.objectId, "#9eeaf2");
        return (
          <div className="assignmentRow" key={`${slot.slot}-${slot.objectId}`}>
            <span>k{slot.slot}</span>
            <div className="assignmentTrack">
              <b style={{ width, background: color }} />
            </div>
            <strong>{formatRatio(slot.probability)}</strong>
          </div>
        );
      })}
    </div>
  );
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
  const objectIds = [...normalizedByObject.keys()].sort((left, right) => Number(left) - Number(right));

  for (const [objectId, entries] of [...normalizedByObject.entries()].sort(sortObjectEntries)) {
    entries.forEach((entry) => {
      entry.y -= minY;
    });
    const normalizedBounds = pointBounds(entries);
    const originalBounds = pointBounds(entries.map((entry) => entry.point));
    const objectBoost = objectDisplayBoost(normalizedBounds, model);
    const accent = objectAccent(objectId, model.accent);
    const assignment = assignmentVectorForObject(objectId, objectIds, model.assignmentConfidence ?? 0.94);
    const assignmentEntropy = normalizedEntropy(assignment.map((slot) => slot.probability));
    const assignmentConfidence = Math.max(...assignment.map((slot) => slot.probability));
    const objectGroup = baseObjectGroup(model, objectId, {
      x: normalizedBounds.center.x,
      y: 0,
      z: normalizedBounds.center.z,
    });
    const positions = new Float32Array(entries.length * 3);
    const originalColors = new Float32Array(entries.length * 3);
    const assignmentColors = new Float32Array(entries.length * 3);
    const fallback = new THREE.Color(accent);
    const debugColor = fallback.clone().lerp(new THREE.Color("#f1fdff"), assignmentEntropy * 0.34);

    entries.forEach((entry, index) => {
      positions[index * 3] = (entry.x - objectGroup.position.x) * objectBoost;
      positions[index * 3 + 1] = (entry.y - normalizedBounds.min.y) * objectBoost;
      positions[index * 3 + 2] = (entry.z - objectGroup.position.z) * objectBoost;
      const color = Array.isArray(entry.point.color) ? entry.point.color : null;
      originalColors[index * 3] = color ? color[0] / 255 : fallback.r;
      originalColors[index * 3 + 1] = color ? color[1] / 255 : fallback.g;
      originalColors[index * 3 + 2] = color ? color[2] / 255 : fallback.b;
      assignmentColors[index * 3] = debugColor.r;
      assignmentColors[index * 3 + 1] = debugColor.g;
      assignmentColors[index * 3 + 2] = debugColor.b;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const originalColorAttr = new THREE.BufferAttribute(originalColors, 3);
    const assignmentColorAttr = new THREE.BufferAttribute(assignmentColors, 3);
    geometry.setAttribute("color", assignmentColorAttr);
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

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
    cloud.userData.originalColor = originalColorAttr;
    cloud.userData.assignmentColor = assignmentColorAttr;
    cloud.userData.gaussianDebug = entries.map((entry, index) => ({
      protocol: "object-state-debug-os-v1",
      source: "derived_from_object_id",
      gaussianIndex: index,
      objectId,
      slot: objectIds.indexOf(objectId),
      confidence: round3(assignmentConfidence),
      entropy: round3(assignmentEntropy),
      assignment,
      position: [round3(entry.point.x), round3(entry.point.y), round3(entry.point.z)],
      opacity: round3(entry.point.opacity ?? 0),
    }));
    const objectState = objectStateSummary({
      objectId,
      assignment,
      assignmentEntropy,
      assignmentConfidence,
      slotMass: entries.length,
      totalMass: sampled.length,
      bounds: originalBounds,
      centroid: originalBounds.center,
      displayBounds: geometry.boundingBox,
      source: "derived_from_object_id",
    });
    objectGroup.userData.objectState = objectState;
    objectGroup.add(cloud);
    objectGroup.add(objectStateWireBox(geometry.boundingBox, accent));
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
      bbox: objectState.bbox,
      objectState,
      assignment,
      assignmentEntropy: objectState.assignmentEntropy,
      assignmentConfidence: objectState.confidence,
      slotMass: objectState.slotMass,
      massFraction: objectState.massFraction,
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
  const objectIds = Array.from({ length: objectCount }, (_item, index) => index);
  const objectGroups = [];
  const objects = [];
  let displayCount = 0;
  for (let index = 0; index < objectCount; index += 1) {
    const objectId = index;
    const accent = objectAccent(objectId, model.accent);
    const assignment = assignmentVectorForObject(objectId, objectIds, model.assignmentConfidence ?? 0.9);
    const assignmentEntropy = normalizedEntropy(assignment.map((slot) => slot.probability));
    const assignmentConfidence = Math.max(...assignment.map((slot) => slot.probability));
    const position = compressedObjectPosition(index, objectCount, model.displayScale ?? 2.1);
    const objectGroup = baseObjectGroup(model, objectId, position);
    const points = syntheticGaussianShell(`${model.id}-${objectId}`, model.placeholderPointsPerObject ?? 760, accent);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(points.positions, 3));
    const originalColorAttr = new THREE.BufferAttribute(points.colors, 3);
    const assignmentColorAttr = new THREE.BufferAttribute(points.assignmentColors, 3);
    geometry.setAttribute("color", assignmentColorAttr);
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();
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
    cloud.userData.originalColor = originalColorAttr;
    cloud.userData.assignmentColor = assignmentColorAttr;
    cloud.userData.gaussianDebug = Array.from({ length: points.positions.length / 3 }, (_item, gaussianIndex) => ({
      protocol: "object-state-debug-os-v1",
      source: "compressed_placeholder_assignment",
      gaussianIndex,
      objectId,
      slot: index,
      confidence: round3(assignmentConfidence),
      entropy: round3(assignmentEntropy),
      assignment,
      position: [
        round3(points.positions[gaussianIndex * 3]),
        round3(points.positions[gaussianIndex * 3 + 1]),
        round3(points.positions[gaussianIndex * 3 + 2]),
      ],
      opacity: 0.58,
    }));
    const bounds = geometry.boundingBox;
    const objectState = objectStateSummary({
      objectId,
      assignment,
      assignmentEntropy,
      assignmentConfidence,
      slotMass: points.positions.length / 3,
      totalMass: objectCount * (model.placeholderPointsPerObject ?? 760),
      bounds,
      centroid: bounds.getCenter(new THREE.Vector3()),
      displayBounds: bounds,
      source: "compressed_placeholder_assignment",
    });
    objectGroup.userData.objectState = objectState;
    objectGroup.add(cloud);
    objectGroup.add(objectStateWireBox(bounds, accent));
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
      bbox: objectState.bbox,
      objectState,
      assignment,
      assignmentEntropy: objectState.assignmentEntropy,
      assignmentConfidence: objectState.confidence,
      slotMass: objectState.slotMass,
      massFraction: objectState.massFraction,
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

function createTrainableArtifactGroup(model) {
  const artifact = model.trainableArtifact;
  if (artifact?.schema !== "objgauss-trainable-kernel-model-artifact-v1") {
    throw new Error("missing trainable kernel model artifact fixture");
  }
  const frame = artifact.object_states?.[model.trainableFrameIndex ?? 0] ?? artifact.object_states?.[0];
  const assignmentFrame = artifact.assignments?.[frame?.frame_index ?? 0] ?? artifact.assignments?.[0];
  const matrix = Array.isArray(assignmentFrame?.matrix) ? assignmentFrame.matrix : [];
  const states = Array.isArray(frame?.states) ? frame.states : [];
  if (!states.length || !matrix.length) {
    throw new Error("trainable artifact fixture needs states and assignment matrix");
  }

  const group = baseModelGroup(model);
  group.userData.assignmentSource = "trainable_kernel_model_artifact";
  group.userData.artifactSchema = artifact.schema;
  const objectGroups = [];
  const objects = [];
  const objectIds = states.map((state) => state.id);
  const allCorners = states.flatMap((state) => bboxCorners(state.bbox));
  const globalBounds = pointBounds(allCorners);
  const span = Math.max(globalBounds.size.x, globalBounds.size.y, globalBounds.size.z, 0.001);
  const scale = (model.displayScale ?? 1.9) / span;
  const decoderColors = artifact.learned_parameters?.decoder_colors ?? [];
  let displayCount = 0;

  states.forEach((state, slot) => {
    const objectId = state.id;
    const accent = objectAccent(objectId, model.accent);
    const rows = matrix
      .map((row, rowIndex) => ({ row, rowIndex, dominantSlot: dominantIndex(row) }))
      .filter((entry) => entry.dominantSlot === slot);
    const stateRows = rows.length ? rows : [{ row: oneHot(slot, states.length), rowIndex: slot, dominantSlot: slot }];
    const box = boxFromBbox(state.bbox);
    const centroid = vectorFromArray(state.centroid, box.getCenter(new THREE.Vector3()));
    const normalizedCenter = normalizeArtifactVector(centroid, globalBounds.center, scale);
    const objectGroup = baseObjectGroup(model, objectId, {
      x: normalizedCenter.x,
      y: 0,
      z: normalizedCenter.z,
    });
    const positions = new Float32Array(stateRows.length * 3);
    const originalColors = new Float32Array(stateRows.length * 3);
    const assignmentColors = new Float32Array(stateRows.length * 3);
    const fallback = new THREE.Color(accent);
    const debugColor = fallback.clone().lerp(new THREE.Color("#f1fdff"), Number(state.normalized_assignment_entropy ?? 0) * 0.34);
    const assignment = averageAssignmentVector(stateRows.map((entry) => entry.row), objectIds);

    stateRows.forEach((entry, index) => {
      const point = artifactPointInBox(box, index, stateRows.length);
      const normalized = normalizeArtifactVector(point, globalBounds.center, scale);
      positions[index * 3] = normalized.x - objectGroup.position.x;
      positions[index * 3 + 1] = normalized.y - globalBounds.min.y * scale;
      positions[index * 3 + 2] = normalized.z - objectGroup.position.z;
      const learnedColor = decoderColors[entry.dominantSlot] ?? null;
      originalColors[index * 3] = Number(learnedColor?.[0]) || fallback.r;
      originalColors[index * 3 + 1] = Number(learnedColor?.[1]) || fallback.g;
      originalColors[index * 3 + 2] = Number(learnedColor?.[2]) || fallback.b;
      assignmentColors[index * 3] = debugColor.r;
      assignmentColors[index * 3 + 1] = debugColor.g;
      assignmentColors[index * 3 + 2] = debugColor.b;
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const originalColorAttr = new THREE.BufferAttribute(originalColors, 3);
    const assignmentColorAttr = new THREE.BufferAttribute(assignmentColors, 3);
    geometry.setAttribute("color", assignmentColorAttr);
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    const material = new THREE.PointsMaterial({
      size: model.pointSize ?? 0.08,
      sizeAttenuation: true,
      vertexColors: true,
      transparent: true,
      opacity: 0.96,
      depthWrite: false,
    });
    const cloud = new THREE.Points(geometry, material);
    cloud.userData.role = "gaussian-cloud";
    cloud.userData.originalColor = originalColorAttr;
    cloud.userData.assignmentColor = assignmentColorAttr;
    cloud.userData.gaussianDebug = stateRows.map((entry, index) => {
      const vector = assignmentVectorFromProbabilities(entry.row, objectIds);
      return {
        protocol: "object-state-debug-os-v1",
        source: "trainable_kernel_model_artifact",
        gaussianIndex: entry.rowIndex,
        objectId,
        slot,
        confidence: round3(Math.max(...entry.row.map((value) => Number(value) || 0))),
        entropy: round3(normalizedEntropy(entry.row.map((value) => Number(value) || 0))),
        assignment: vector,
        position: Array.from(positions.slice(index * 3, index * 3 + 3)).map(round3),
        opacity: 0.96,
      };
    });

    const objectState = {
      schema: "objgauss-object-state-debug-v1",
      source: "trainable_kernel_model_artifact",
      objectId,
      slot,
      slotMass: round3(state.slot_mass),
      massFraction: round3(state.mass_fraction),
      confidence: round3(state.confidence),
      assignmentEntropy: round3(state.normalized_assignment_entropy ?? state.assignment_entropy ?? 0),
      centroid: (state.centroid ?? []).map(round3),
      bbox: (state.bbox ?? []).map(round3),
      status: state.status ?? "trained_artifact_slot",
      assignment,
    };
    objectGroup.userData.objectState = objectState;
    objectGroup.add(cloud);
    objectGroup.add(objectStateWireBox(geometry.boundingBox, accent));
    objectGroup.add(corePointMesh(Math.max(0.42, geometry.boundingBox.getCenter(new THREE.Vector3()).y), accent));
    objectGroup.add(coreGlow(Math.max(0.42, geometry.boundingBox.getCenter(new THREE.Vector3()).y), accent));
    objectGroup.add(selectionRing(accent, ringRadiusForBounds(geometryBoundsInfo(geometry.boundingBox), 1)));
    group.add(objectGroup);
    objectGroups.push(objectGroup);
    displayCount += stateRows.length;
    objects.push({
      objectId,
      selectionId: selectionIdForObject(model.id, objectId),
      displayCount: stateRows.length,
      corePoint: objectState.centroid,
      bbox: objectState.bbox,
      objectState,
      assignment,
      assignmentEntropy: objectState.assignmentEntropy,
      assignmentConfidence: objectState.confidence,
      slotMass: objectState.slotMass,
      massFraction: objectState.massFraction,
      galleryPosition: objectGroup.position.toArray().map(round3),
      chunkPath: objectChunkPath(model, objectId),
      accent,
    });
  });

  return {
    group,
    objectGroups,
    summary: {
      displayCount,
      gaussianCount: matrix.length,
      objectCount: objects.length,
      corePoint: states[0]?.centroid?.map(round3) ?? [0, 0, 0],
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

function objectStateWireBox(bounds, accent) {
  const box = bounds?.isBox3 ? bounds.clone() : null;
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  if (box) {
    box.getSize(size);
    box.getCenter(center);
  } else {
    size.set(0.8, 0.8, 0.8);
    center.set(0, 0.4, 0);
  }
  size.x = Math.max(size.x, 0.08);
  size.y = Math.max(size.y, 0.08);
  size.z = Math.max(size.z, 0.08);
  const geometry = new THREE.BoxGeometry(size.x, size.y, size.z);
  const edges = new THREE.EdgesGeometry(geometry);
  geometry.dispose();
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({
      color: accent,
      transparent: true,
      opacity: 0.42,
    }),
  );
  line.position.copy(center);
  line.userData.role = "object-state-bbox";
  return line;
}

function nearestObjectGroup(object) {
  let cursor = object;
  while (cursor) {
    if (cursor.userData?.selectionId) return cursor;
    cursor = cursor.parent;
  }
  return null;
}

function firstGaussianCloud(object) {
  let result = null;
  object?.traverse?.((child) => {
    if (!result && child.userData?.role === "gaussian-cloud") result = child;
  });
  return result;
}

function gaussianProbeFromIntersection(intersection) {
  if (!intersection?.object?.userData || intersection.index === undefined) return null;
  if (intersection.object.userData.role !== "gaussian-cloud") return null;
  const probe = intersection.object.userData.gaussianDebug?.[intersection.index];
  return probe ?? null;
}

function applyObjectVisualState(object, { selected = false, hovered = false, debug = true } = {}) {
  const selectedOrHovered = Boolean(selected || hovered);
  object.traverse((child) => {
    if (child.userData.role === "gaussian-cloud") {
      child.material.opacity = selectedOrHovered ? 1 : debug ? 0.82 : 0.64;
      child.material.size = selectedOrHovered ? child.material.size * 1.0 : child.material.size;
    }
    if (child.userData.role === "object-state-bbox") {
      child.material.opacity = selected ? 0.82 : hovered ? 0.64 : 0.34;
    }
  });
}

function objectTarget(object) {
  if (!object?.userData?.selectionId) return null;
  return {
    modelId: object.userData.modelId,
    objectId: object.userData.objectId,
    selectionId: object.userData.selectionId,
  };
}

function assignmentVectorForObject(objectId, objectIds, confidence = 0.94) {
  const ids = objectIds.length ? objectIds : [objectId];
  const targetIndex = ids.findIndex((id) => String(id) === String(objectId));
  const cleanConfidence = Math.max(0, Math.min(1, Number(confidence) || 1));
  const rest = ids.length <= 1 ? 0 : (1 - cleanConfidence) / (ids.length - 1);
  return ids.map((id, slot) => ({
    slot,
    objectId: id,
    probability: round3(slot === targetIndex ? cleanConfidence : rest),
  }));
}

function normalizedEntropy(probabilities) {
  const values = probabilities.filter((value) => Number(value) > 0);
  if (probabilities.length <= 1 || values.length === 0) return 0;
  const entropy = -values.reduce((total, value) => total + value * Math.log(value), 0);
  return Math.max(0, Math.min(1, entropy / Math.log(probabilities.length)));
}

function objectStateSummary({
  objectId,
  assignment,
  assignmentEntropy,
  assignmentConfidence,
  slotMass,
  totalMass,
  bounds,
  centroid,
  source,
}) {
  const box = bounds?.isBox3
    ? bounds
    : new THREE.Box3(bounds?.min ?? new THREE.Vector3(), bounds?.max ?? new THREE.Vector3());
  const center = centroid?.isVector3 ? centroid : box.getCenter(new THREE.Vector3());
  const min = box.min ?? new THREE.Vector3();
  const max = box.max ?? new THREE.Vector3();
  const mass = Number(slotMass) || 0;
  const total = Math.max(1, Number(totalMass) || mass || 1);
  return {
    schema: "objgauss-object-state-debug-v1",
    source,
    objectId,
    slot: assignment.find((entry) => String(entry.objectId) === String(objectId))?.slot ?? 0,
    slotMass: mass,
    massFraction: round3(mass / total),
    confidence: round3(assignmentConfidence),
    assignmentEntropy: round3(assignmentEntropy),
    centroid: [round3(center.x), round3(center.y), round3(center.z)],
    bbox: [
      round3(min.x),
      round3(min.y),
      round3(min.z),
      round3(max.x),
      round3(max.y),
      round3(max.z),
    ],
    status: assignmentEntropy > 0.55 ? "mixed_slot" : "stable_debug_projection",
    assignment,
  };
}

function bboxCorners(bbox) {
  const box = boxFromBbox(bbox);
  return [
    { x: box.min.x, y: box.min.y, z: box.min.z },
    { x: box.min.x, y: box.min.y, z: box.max.z },
    { x: box.min.x, y: box.max.y, z: box.min.z },
    { x: box.max.x, y: box.min.y, z: box.min.z },
    { x: box.max.x, y: box.max.y, z: box.max.z },
  ];
}

function boxFromBbox(bbox) {
  const values = Array.isArray(bbox) && bbox.length >= 6 ? bbox.map(Number) : [-0.5, 0, -0.5, 0.5, 1, 0.5];
  return new THREE.Box3(
    new THREE.Vector3(values[0] || 0, values[1] || 0, values[2] || 0),
    new THREE.Vector3(values[3] || 0, values[4] || 0, values[5] || 0),
  );
}

function vectorFromArray(value, fallback = new THREE.Vector3()) {
  if (!Array.isArray(value) || value.length < 3) return fallback.clone();
  return new THREE.Vector3(Number(value[0]) || 0, Number(value[1]) || 0, Number(value[2]) || 0);
}

function normalizeArtifactVector(vector, center, scale) {
  return new THREE.Vector3(
    (vector.x - center.x) * scale,
    (vector.y - center.y) * scale,
    (vector.z - center.z) * scale,
  );
}

function artifactPointInBox(box, index, count) {
  const t = count <= 1 ? 0.5 : index / (count - 1);
  const wave = Math.sin((index + 1) * 1.73) * 0.18;
  return new THREE.Vector3(
    THREE.MathUtils.lerp(box.min.x, box.max.x, 0.24 + 0.52 * t),
    THREE.MathUtils.lerp(box.min.y, box.max.y, 0.34 + 0.22 * ((index + 1) % 2)),
    THREE.MathUtils.lerp(box.min.z, box.max.z, 0.5 + wave),
  );
}

function averageAssignmentVector(rows, objectIds) {
  const width = Math.max(objectIds.length, ...rows.map((row) => row.length));
  const totals = Array.from({ length: width }, () => 0);
  rows.forEach((row) => {
    row.forEach((value, index) => {
      totals[index] += Number(value) || 0;
    });
  });
  const divisor = Math.max(1, rows.length);
  return assignmentVectorFromProbabilities(totals.map((value) => value / divisor), objectIds);
}

function assignmentVectorFromProbabilities(probabilities, objectIds) {
  return probabilities.map((value, slot) => ({
    slot,
    objectId: objectIds[slot] ?? slot,
    probability: round3(Number(value) || 0),
  }));
}

function dominantIndex(values) {
  let bestIndex = 0;
  let bestValue = -Infinity;
  values.forEach((value, index) => {
    const number = Number(value) || 0;
    if (number > bestValue) {
      bestValue = number;
      bestIndex = index;
    }
  });
  return bestIndex;
}

function oneHot(index, width) {
  return Array.from({ length: width }, (_item, slot) => (slot === index ? 1 : 0));
}

function geometryBoundsInfo(box) {
  const size = new THREE.Vector3();
  box?.getSize?.(size);
  return {
    size,
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
  const assignmentColors = new Float32Array(count * 3);
  const colorA = new THREE.Color(accent);
  const colorB = new THREE.Color("#dffaff");
  const debugColor = colorA.clone().lerp(colorB, 0.18);
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
    assignmentColors[index * 3] = debugColor.r;
    assignmentColors[index * 3 + 1] = debugColor.g;
    assignmentColors[index * 3 + 2] = debugColor.b;
  }
  return { positions, colors, assignmentColors };
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

function formatBox(value) {
  if (!Array.isArray(value) || value.length !== 6) return "-";
  return `${value.slice(0, 3).map((entry) => Number(entry).toFixed(2)).join(", ")} / ${value
    .slice(3)
    .map((entry) => Number(entry).toFixed(2))
    .join(", ")}`;
}

function formatRatio(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "-";
}

function formatLoss(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(6) : "-";
}

function round3(value) {
  return Math.round(Number(value) * 1000) / 1000;
}
