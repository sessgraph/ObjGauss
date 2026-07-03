import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { DragControls } from "three/examples/jsm/controls/DragControls.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import { catalogSummary, defaultModelIdForCatalog, modelCatalogFromSearch } from "./modelCatalog.js";
import { MODEL_ARTIFACT_MANIFEST_SCHEMA, browserReadyArtifact } from "./modelArtifactManifest.js";
import {
  decodeQuantizedOgcPayload,
  decodeQuantizedOgcPayloadWindows,
  quantizedOgcReadWindows,
} from "./ogcDecoder.js";
import { colorForObject, rgbToCss } from "./palette.js";
import { parsePly } from "./ply.js";

const INITIAL_CAMERA = {
  position: [0, 5.4, 10.8],
  target: [0, 1.15, 0],
};

const DEBUG_LENSES = ["assignment", "confidence", "entropy"];
const DEBUG_EVENT_LIMIT = 12;

export default function App() {
  const modelCatalog = useMemo(
    () => modelCatalogFromSearch(typeof window === "undefined" ? "" : window.location.search),
    [],
  );
  const initialModelId = useMemo(() => defaultModelIdForCatalog(modelCatalog), [modelCatalog]);
  const worldApi = useRef(null);
  const loadStarted = useRef(false);
  const artifactInputRef = useRef(null);
  const ogcInputRef = useRef(null);
  const [worldReady, setWorldReady] = useState(false);
  const [selection, setSelection] = useState(() => ({
    modelId: initialModelId,
    objectId: null,
    selectionId: initialModelId,
  }));
  const [models, setModels] = useState(() => initialModelStates(modelCatalog));
  const [debugMode, setDebugMode] = useState(true);
  const [debugLens, setDebugLens] = useState("assignment");
  const [debugEvents, setDebugEvents] = useState(() => []);
  const debugEventSeq = useRef(0);
  const [hoveredTarget, setHoveredTarget] = useState(null);
  const [debugProbe, setDebugProbe] = useState(null);
  const [hiddenObjects, setHiddenObjects] = useState(() => new Set());
  const [artifactImport, setArtifactImport] = useState(() => ({
    status: "idle",
    modelId: "",
    fileName: "",
    error: "",
  }));
  const [ogcImport, setOgcImport] = useState(() => ({
    status: "idle",
    modelId: "",
    fileName: "",
    error: "",
  }));
  const modelList = useMemo(() => Object.values(models), [models]);
  const summary = useMemo(() => catalogSummary(modelList), [modelList]);
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
      modelList.reduce(
        (total, model) => total + (model.objects?.length || Number(model.objectCount) || 0),
        0,
      ),
    [modelList],
  );
  const selectedId = selection.modelId;
  const selected = models[selectedId] ?? Object.values(models)[0];
  const selectedObject =
    selected?.objects?.find((object) => String(object.objectId) === String(selection.objectId)) ?? null;
  const selectedObjectKey = selectedObject?.selectionId ?? "";
  const selectedStability = useMemo(
    () => summarizeObjectStability(selected?.objects ?? []),
    [selected?.objects],
  );
  const selectedAssignmentSource =
    debugProbe?.source ?? selectedObject?.objectState?.source ?? selected?.delivery?.source ?? "";
  const hiddenCount = hiddenObjects.size;
  const selectedOgcChunkScope =
    selected?.delivery?.source === "quantized-ogc" ? formatChunkScope(selected.delivery?.chunkIds) : "";
  const selectedOgcAvailableChunks =
    selected?.delivery?.source === "quantized-ogc" ? formatChunkScope(selected.delivery?.availableChunkIds) : "";
  const selectedTrainingEvidence =
    selected?.delivery?.source === "trainable-kernel-model-artifact"
      ? trainableEvidenceSummary(selected.trainableArtifact)
      : null;
  const selectedDebugSnapshot = objectStateDebugSnapshot({
    selected,
    selectedObject,
    selection,
    debugMode,
    debugLens,
    debugProbe,
    hiddenCount,
    stability: selectedStability,
    assignmentSource: selectedAssignmentSource,
    trainingEvidence: selectedTrainingEvidence,
    debugEvents,
  });

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    window.__OBJGAUSS_DEBUG_SNAPSHOT__ = selectedDebugSnapshot;
    return () => {
      if (window.__OBJGAUSS_DEBUG_SNAPSHOT__ === selectedDebugSnapshot) {
        delete window.__OBJGAUSS_DEBUG_SNAPSHOT__;
      }
    };
  }, [selectedDebugSnapshot]);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    window.__OBJGAUSS_DEBUG_EVENTS__ = debugEvents;
    return () => {
      if (window.__OBJGAUSS_DEBUG_EVENTS__ === debugEvents) {
        delete window.__OBJGAUSS_DEBUG_EVENTS__;
      }
    };
  }, [debugEvents]);

  const recordDebugEvent = useCallback((type, detail = {}) => {
    const seq = debugEventSeq.current + 1;
    debugEventSeq.current = seq;
    const event = debugEventFromDetail(type, detail, seq);
    setDebugEvents((current) => [event, ...current].slice(0, DEBUG_EVENT_LIMIT));
    return event;
  }, []);

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

  const upsertTrainableArtifactModel = useCallback((model, artifact, options = {}) => {
    const hydratedModel = { ...model, trainableArtifact: artifact, trainableFrameIndex: 0 };
    const startedAt = Number(options.startedAt);
    const rendered = worldApi.current?.upsertModel(hydratedModel, null);
    const loadMs = Number.isFinite(startedAt) ? Math.round(performance.now() - startedAt) : 0;
    const nextModel = {
      ...hydratedModel,
      status: "loaded",
      message: options.message ?? "trained artifact json",
      gaussianCount: rendered?.gaussianCount ?? rendered?.displayCount ?? 0,
      displayCount: rendered?.displayCount ?? 0,
      objectCount: rendered?.objectCount ?? model.objectCount,
      corePoint: rendered?.corePoint ?? null,
      objects: rendered?.objects ?? [],
      loadMs,
      delivery: {
        source: "trainable-kernel-model-artifact",
        loadRoute: options.loadRoute ?? (model.trainableArtifactPath ? "fetch-json" : "inline"),
        artifactPath: options.artifactPath ?? model.trainableArtifactPath ?? "inline://trainable-artifact",
        frameIndex: rendered?.trainableFrameIndex ?? 0,
        frameCount: rendered?.trainableFrameCount ?? artifact.object_states?.length ?? 0,
        schema: artifact.schema,
        rendererName: artifact.renderer_api?.renderer_name,
        imageRenderLoss: artifact.renderer_api?.image_render_loss,
        gradientPath: artifact.renderer_api?.gradient_path,
      },
    };
    setModels((current) => ({
      ...current,
      [model.id]: {
        ...(current[model.id] ?? {}),
        ...nextModel,
      },
    }));
    return nextModel;
  }, []);

  const upsertDecodedOgcModel = useCallback((model, decoded, index, delivery, artifact, options = {}) => {
    const rendered = worldApi.current?.upsertModel(model, decoded.points);
    const startedAt = Number(options.startedAt);
    const nextModel = {
      ...model,
      status: "loaded",
      message: options.message ?? "ogc chunks",
      gaussianCount: decoded.points.length,
      displayCount: rendered?.displayCount ?? 0,
      objectCount: rendered?.objectCount ?? decoded.metadata.objectCount ?? model.objectCount,
      corePoint: rendered?.corePoint ?? null,
      objects: rendered?.objects ?? [],
      loadMs: Number.isFinite(startedAt) ? Math.round(performance.now() - startedAt) : 0,
      delivery: {
        source: "quantized-ogc",
        role: artifact.role,
        decodedChunks: decoded.metadata.decodedChunks,
        decodedGaussians: decoded.metadata.decodedGaussians,
        recordFormat: decoded.metadata.recordFormat,
        lodLevel: model.ogc?.lodLevel ?? "full",
        lodLevels: availableOgcLodLevels(index),
        chunkIds: Array.isArray(model.ogc?.chunkIds) ? model.ogc.chunkIds : [],
        availableChunkIds: availableOgcChunkIds(index),
        loadRoute: delivery.loadRoute,
        indexPath: artifact.indexPath ?? artifact.chunk_index?.path ?? "",
        payloadPath: artifact.payloadPath ?? artifact.path ?? index?.payload?.path ?? "",
        fetchedBytes: delivery.fetchedBytes,
        requestedBytes: delivery.requestedBytes,
        decodedWindows: delivery.decodedWindows,
      },
    };
    setModels((current) => ({
      ...current,
      [model.id]: {
        ...(current[model.id] ?? {}),
        ...nextModel,
      },
    }));
    return nextModel;
  }, []);

  const selectModel = useCallback(
    (id) => {
      setSelection({ modelId: id, objectId: null, selectionId: id });
      recordDebugEvent("select-model", { modelId: id, selectionId: id, source: "model-dock" });
      worldApi.current?.focusModel(id);
    },
    [recordDebugEvent],
  );

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
    const next = !debugMode;
    setDebugMode(next);
    recordDebugEvent("debug-toggle", {
      enabled: next,
      lens: next ? debugLens : "appearance",
      source: "top-hud",
    });
    worldApi.current?.setDebugMode(next);
  }, [debugLens, debugMode, recordDebugEvent]);

  const selectDebugLens = useCallback((lens) => {
    const next = normalizeDebugLens(lens);
    setDebugLens(next);
    setDebugMode(true);
    recordDebugEvent("debug-lens", { lens: next, enabled: true, source: "debug-panel" });
    worldApi.current?.setDebugMode(true);
    worldApi.current?.setDebugLens(next);
  }, [recordDebugEvent]);

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

  const importTrainableArtifactFile = useCallback(
    async (event) => {
      const file = event.target.files?.[0] ?? null;
      event.target.value = "";
      if (!file) return;
      const model = trainableLocalArtifactModel(file.name);
      const startedAt = performance.now();
      setArtifactImport({ status: "loading", modelId: model.id, fileName: file.name, error: "" });
      try {
        const artifact = validateTrainableArtifact(JSON.parse(await file.text()));
        const imported = upsertTrainableArtifactModel(model, artifact, {
          startedAt,
          loadRoute: "local-file",
          artifactPath: `local://${file.name}`,
          message: "local artifact json",
        });
        setDebugProbe(null);
        setHoveredTarget(null);
        setHiddenObjects((current) => {
          const next = new Set(
            [...current].filter((selectionId) => !String(selectionId).startsWith(`${model.id}::`)),
          );
          worldApi.current?.setHiddenObjects(next);
          return next;
        });
        setSelection({ modelId: model.id, objectId: null, selectionId: model.id });
        worldApi.current?.focusModel(model.id);
        setArtifactImport({ status: "loaded", modelId: model.id, fileName: file.name, error: "" });
        recordDebugEvent("import-artifact", {
          modelId: model.id,
          selectionId: model.id,
          fileName: file.name,
          source: imported.delivery.loadRoute,
        });
      } catch (error) {
        setArtifactImport({
          status: "error",
          modelId: model.id,
          fileName: file.name,
          error: error?.message ?? "artifact import failed",
        });
        recordDebugEvent("import-artifact-error", {
          modelId: model.id,
          selectionId: model.id,
          fileName: file.name,
          source: "local-file",
        });
      }
    },
    [recordDebugEvent, upsertTrainableArtifactModel],
  );

  const importOgcArtifactFiles = useCallback(
    async (event) => {
      const files = Array.from(event.target.files ?? []);
      event.target.value = "";
      if (!files.length) return;
      const fileName = files.map((file) => file.name).join(" + ");
      const modelId = "ogc-local-artifact";
      const startedAt = performance.now();
      setOgcImport({ status: "loading", modelId, fileName, error: "" });
      try {
        const { indexFile, payloadFile } = ogcFilePair(files);
        const index = JSON.parse(await indexFile.text());
        const payloadBuffer = await payloadFile.arrayBuffer();
        const model = ogcLocalArtifactModel({
          index,
          payloadBuffer,
          indexFileName: indexFile.name,
          payloadFileName: payloadFile.name,
        });
        const { artifact, decoded, index: loadedIndex, delivery } = await loadOgcModel(model);
        const imported = upsertDecodedOgcModel(model, decoded, loadedIndex, delivery, artifact, {
          startedAt,
          message: "local ogc files",
        });
        setDebugProbe(null);
        setHoveredTarget(null);
        setHiddenObjects((current) => {
          const next = new Set(
            [...current].filter((selectionId) => !String(selectionId).startsWith(`${model.id}::`)),
          );
          worldApi.current?.setHiddenObjects(next);
          return next;
        });
        setSelection({ modelId: model.id, objectId: null, selectionId: model.id });
        worldApi.current?.focusModel(model.id);
        setOgcImport({ status: "loaded", modelId: model.id, fileName, error: "" });
        recordDebugEvent("import-ogc", {
          modelId: model.id,
          selectionId: model.id,
          fileName,
          source: imported.delivery.loadRoute,
        });
      } catch (error) {
        setOgcImport({
          status: "error",
          modelId,
          fileName,
          error: error?.message ?? "ogc import failed",
        });
        recordDebugEvent("import-ogc-error", {
          modelId,
          selectionId: modelId,
          fileName,
          source: "local-file",
        });
      }
    },
    [recordDebugEvent, upsertDecodedOgcModel],
  );

  const selectTrainableFrame = useCallback(
    (frameIndex) => {
      const current = models[selectedId];
      const artifact = current?.trainableArtifact;
      const frames = Array.isArray(artifact?.object_states) ? artifact.object_states : [];
      if (current?.loadMode !== "trainable-artifact" || !frames.length) return;
      const nextFrameIndex = Math.max(0, Math.min(frames.length - 1, Number(frameIndex) || 0));
      const nextModel = { ...current, trainableFrameIndex: nextFrameIndex };
      const rendered = worldApi.current?.upsertModel(nextModel, null);
      setDebugProbe(null);
      recordDebugEvent("frame-select", {
        modelId: current.id,
        selectionId: current.id,
        frameIndex: nextFrameIndex,
        source: "debug-panel",
      });
      patchModel(current.id, (latest) => ({
        trainableFrameIndex: nextFrameIndex,
        gaussianCount: rendered?.gaussianCount ?? rendered?.displayCount ?? latest.gaussianCount ?? 0,
        displayCount: rendered?.displayCount ?? latest.displayCount ?? 0,
        objectCount: rendered?.objectCount ?? latest.objectCount,
        corePoint: rendered?.corePoint ?? latest.corePoint ?? null,
        objects: rendered?.objects ?? latest.objects ?? [],
        delivery: {
          ...(latest.delivery ?? {}),
          frameIndex: nextFrameIndex,
          frameCount: frames.length,
        },
      }));
    },
    [models, patchModel, recordDebugEvent, selectedId],
  );

  const reloadOgcModel = useCallback(
    async (current, nextOgc, messages) => {
      if (current?.loadMode !== "ogc-chunked") return;
      const nextModel = { ...current, ogc: nextOgc };
      const startedAt = performance.now();
      patchModel(current.id, {
        ogc: nextOgc,
        status: "loading",
        message: messages.loading,
      });
      try {
        const { artifact, decoded, index, delivery } = await loadOgcModel(nextModel);
        setDebugProbe(null);
        setHiddenObjects((currentHidden) => {
          const nextHidden = new Set(
            [...currentHidden].filter((selectionId) => !String(selectionId).startsWith(`${current.id}::`)),
          );
          worldApi.current?.setHiddenObjects(nextHidden);
          return nextHidden;
        });
        upsertDecodedOgcModel(nextModel, decoded, index, delivery, artifact, {
          startedAt,
          message: messages.loaded,
        });
      } catch (error) {
        patchModel(current.id, {
          ogc: nextOgc,
          status: "error",
          message: error?.message ?? messages.error,
        });
      }
    },
    [patchModel, upsertDecodedOgcModel],
  );

  const selectOgcLod = useCallback(
    async (lodLevel) => {
      const current = models[selectedId];
      if (current?.loadMode !== "ogc-chunked") return;
      const nextLodLevel = Math.max(0, Math.min(16, Number(lodLevel) || 0));
      recordDebugEvent("ogc-lod", {
        modelId: current.id,
        selectionId: current.id,
        lodLevel: nextLodLevel,
        source: "debug-panel",
      });
      await reloadOgcModel(
        current,
        {
          ...(current.ogc ?? {}),
          lodLevel: nextLodLevel,
        },
        {
          loading: `loading ogc lod ${nextLodLevel}`,
          loaded: `ogc lod ${nextLodLevel}`,
          error: "ogc lod load failed",
        },
      );
    },
    [models, recordDebugEvent, reloadOgcModel, selectedId],
  );

  const selectOgcChunks = useCallback(
    async (chunkIds) => {
      const current = models[selectedId];
      if (current?.loadMode !== "ogc-chunked") return;
      const nextChunkIds = Array.isArray(chunkIds) && chunkIds.length ? chunkIds : undefined;
      const label = nextChunkIds?.length ? `chunk ${nextChunkIds.join(",")}` : "all chunks";
      recordDebugEvent("ogc-chunks", {
        modelId: current.id,
        selectionId: current.id,
        chunkScope: nextChunkIds?.length ? nextChunkIds.join(",") : "all",
        source: "debug-panel",
      });
      await reloadOgcModel(
        current,
        {
          ...(current.ogc ?? {}),
          chunkIds: nextChunkIds,
        },
        {
          loading: `loading ogc ${label}`,
          loaded: `ogc ${label}`,
          error: "ogc chunk load failed",
        },
      );
    },
    [models, recordDebugEvent, reloadOgcModel, selectedId],
  );

  const handleObjectMoved = useCallback(
    (target, position) => {
      if (!target?.modelId) return;
      recordDebugEvent("move-object", {
        modelId: target.modelId,
        objectId: target.objectId,
        selectionId: target.selectionId,
        position,
        source: "drag",
      });
      patchModel(target.modelId, (current) => ({
        objects: (current.objects ?? []).map((object) =>
          String(object.objectId) === String(target.objectId)
            ? { ...object, galleryPosition: position }
            : object,
        ),
      }));
    },
    [patchModel, recordDebugEvent],
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
      for (const model of modelCatalog) {
        if (cancelled) return;
        if (model.loadMode === "ogc-chunked") {
          const startedAt = performance.now();
          patchModel(model.id, { status: "loading", message: "loading ogc chunks" });
          try {
            const { artifact, decoded, index, delivery } = await loadOgcModel(model);
            if (cancelled) return;
            upsertDecodedOgcModel(model, decoded, index, delivery, artifact, {
              startedAt,
              message: "ogc chunks",
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
            const artifact = await loadTrainableArtifact(model);
            if (cancelled) return;
            upsertTrainableArtifactModel(model, artifact, {
              startedAt,
              loadRoute: model.trainableArtifactPath ? "fetch-json" : "inline",
              artifactPath: model.trainableArtifactPath ?? "inline://trainable-artifact",
              message: "trained artifact json",
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
  }, [modelCatalog, patchModel, upsertDecodedOgcModel, upsertTrainableArtifactModel, worldReady]);

  return (
    <main
      className="worldShell"
      data-app-mode="vr-three-world"
      data-three-renderer="enabled"
      data-sidebars="none"
      data-frosted-ui="enabled"
      data-model-count={modelList.length}
      data-catalog-model-count={modelCatalog.length}
      data-loaded-count={loadedCount}
      data-selected-model={selected?.id ?? ""}
      data-object-count={objectCount}
      data-selected-object={selection.objectId ?? ""}
      data-selected-target={selection.selectionId ?? selected?.id ?? ""}
      data-compression-layout="per-object-corepoint-chunks"
      data-debug-os="object-state"
      data-debug-snapshot-schema={selectedDebugSnapshot.schema}
      data-debug-snapshot-model={selectedDebugSnapshot.model.id}
      data-debug-snapshot-object={selectedDebugSnapshot.selection.objectId ?? ""}
      data-debug-snapshot-assignment-slots={selectedDebugSnapshot.assignment.slotCount}
      data-debug-snapshot-stability={selectedDebugSnapshot.stability.status}
      data-debug-snapshot-training-status={selectedDebugSnapshot.training?.status ?? ""}
      data-debug-event-count={debugEvents.length}
      data-debug-event-last={debugEvents[0]?.type ?? ""}
      data-debug-event-schema={debugEvents[0]?.schema ?? ""}
      data-assignment-debug={debugMode ? "enabled" : "disabled"}
      data-debug-lens={debugMode ? debugLens : "appearance"}
      data-selected-gaussian={debugProbe?.gaussianIndex ?? ""}
      data-hovered-target={hoveredTarget?.selectionId ?? ""}
      data-hovered-model={hoveredTarget?.modelId ?? ""}
      data-hovered-object={hoveredTarget?.objectId ?? ""}
      data-hovered-gaussians={hoveredTarget?.gaussianCount ?? ""}
      data-hidden-objects={hiddenCount}
      data-ogc-loaded-count={ogcLoadedCount}
      data-ogc-artifact-load-route={selected?.delivery?.source === "quantized-ogc" ? selected?.delivery?.loadRoute ?? "" : ""}
      data-ogc-artifact-index-path={selected?.delivery?.source === "quantized-ogc" ? selected?.delivery?.indexPath ?? "" : ""}
      data-ogc-artifact-payload-path={selected?.delivery?.source === "quantized-ogc" ? selected?.delivery?.payloadPath ?? "" : ""}
      data-ogc-artifact-lod-level={selected?.delivery?.source === "quantized-ogc" ? selected?.delivery?.lodLevel ?? "" : ""}
      data-ogc-artifact-fetched-bytes={selected?.delivery?.source === "quantized-ogc" ? selected?.delivery?.fetchedBytes ?? "" : ""}
      data-ogc-artifact-requested-bytes={selected?.delivery?.source === "quantized-ogc" ? selected?.delivery?.requestedBytes ?? "" : ""}
      data-ogc-artifact-decoded-windows={selected?.delivery?.source === "quantized-ogc" ? selected?.delivery?.decodedWindows ?? "" : ""}
      data-ogc-artifact-chunk-scope={selectedOgcChunkScope}
      data-ogc-artifact-available-chunks={selectedOgcAvailableChunks}
      data-trainable-artifact-loaded-count={trainableArtifactLoadedCount}
      data-assignment-source={selectedAssignmentSource}
      data-stability-dashboard="enabled"
      data-stability-status={selectedStability.status}
      data-stability-slot-utilization={selectedStability.slotUtilization}
      data-stability-mean-entropy={selectedStability.meanEntropy}
      data-stability-mixed-slots={selectedStability.mixedSlots}
      data-stability-low-confidence={selectedStability.lowConfidenceSlots}
      data-stability-purity-available={selectedStability.purityAvailable ? "true" : "false"}
      data-stability-mean-purity={selectedStability.meanPurity ?? ""}
      data-stability-temporal-available={selectedStability.temporalAvailable ? "true" : "false"}
      data-stability-mean-temporal-drift={selectedStability.meanTemporalDrift ?? ""}
      data-stability-spatial-available={selectedStability.spatialAvailable ? "true" : "false"}
      data-stability-mean-spatial-compactness={selectedStability.meanSpatialCompactness ?? ""}
      data-stability-jitter-available={selectedStability.jitterAvailable ? "true" : "false"}
      data-stability-mean-assignment-jitter={selectedStability.meanAssignmentJitter ?? ""}
      data-stability-bbox-available={selectedStability.bboxAvailable ? "true" : "false"}
      data-stability-mean-bbox-stability={selectedStability.meanBboxStability ?? ""}
      data-trainable-artifact-load-route={selected?.delivery?.loadRoute ?? ""}
      data-trainable-artifact-path={selected?.delivery?.artifactPath ?? ""}
      data-trainable-artifact-frame-index={selected?.delivery?.frameIndex ?? ""}
      data-trainable-artifact-frame-count={selected?.delivery?.frameCount ?? ""}
      data-trainable-training-status={selectedTrainingEvidence?.status ?? ""}
      data-trainable-training-iterations={selectedTrainingEvidence?.iterations ?? ""}
      data-trainable-training-final-total-loss={selectedTrainingEvidence?.finalTotalLoss ?? ""}
      data-trainable-training-loss-delta={selectedTrainingEvidence?.totalLossDelta ?? ""}
      data-trainable-training-final-image-loss={selectedTrainingEvidence?.finalImageLoss ?? ""}
      data-trainable-training-image-loss-delta={selectedTrainingEvidence?.imageLossDelta ?? ""}
      data-trainable-training-image-loss-decreased={selectedTrainingEvidence?.imageLossDecreased ? "true" : ""}
      data-trainable-import-status={artifactImport.status}
      data-trainable-import-model={artifactImport.modelId}
      data-trainable-import-file={artifactImport.fileName}
      data-trainable-import-error={artifactImport.error}
      data-ogc-import-status={ogcImport.status}
      data-ogc-import-model={ogcImport.modelId}
      data-ogc-import-file={ogcImport.fileName}
      data-ogc-import-error={ogcImport.error}
    >
      <ThreeWorld
        models={modelCatalog}
        selectedTargetId={selection.selectionId || selectedId}
        debugMode={debugMode}
        debugLens={debugLens}
        hiddenSelectionIds={hiddenObjects}
        onReady={handleWorldReady}
        onSelectObject={selectObject}
        onHoverObject={handleHoverObject}
        onObjectMoved={handleObjectMoved}
        onDebugEvent={recordDebugEvent}
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
          <input
            ref={artifactInputRef}
            className="fileInputHidden"
            type="file"
            accept="application/json,.json"
            data-trainable-artifact-file-input="true"
            onChange={importTrainableArtifactFile}
          />
          <input
            ref={ogcInputRef}
            className="fileInputHidden"
            type="file"
            accept="application/json,.json,.ogc"
            multiple
            data-ogc-artifact-file-input="true"
            onChange={importOgcArtifactFiles}
          />
          <button
            className={`glassButton ${artifactImport.status === "loaded" ? "active" : ""}`}
            type="button"
            data-trainable-artifact-import-button="true"
            data-import-status={artifactImport.status}
            onClick={() => artifactInputRef.current?.click()}
          >
            {artifactImport.status === "loading"
              ? "导入中"
              : artifactImport.status === "error"
                ? "导入失败"
                : "导入训练"}
          </button>
          <button
            className={`glassButton ${ogcImport.status === "loaded" ? "active" : ""}`}
            type="button"
            data-ogc-artifact-import-button="true"
            data-import-status={ogcImport.status}
            onClick={() => ogcInputRef.current?.click()}
          >
            {ogcImport.status === "loading"
              ? "OGC中"
              : ogcImport.status === "error"
                ? "OGC失败"
                : "导入OGC"}
          </button>
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
        {modelList.map((model) => (
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
            <Meta label="artifact" value={selected.delivery?.artifactPath ?? "-"} />
            <Meta label="frame" value={formatFrame(selected.delivery?.frameIndex, selected.delivery?.frameCount)} />
            <Meta label="train loss" value={formatLoss(selectedTrainingEvidence?.finalTotalLoss)} />
            <Meta label="loss delta" value={formatSignedLoss(selectedTrainingEvidence?.totalLossDelta)} />
            <Meta label="OGC chunks" value={selected.delivery?.decodedChunks ?? "-"} />
            <Meta label="OGC route" value={selected.delivery?.loadRoute ?? "-"} />
            <Meta label="OGC bytes" value={formatByteWindow(selected.delivery?.fetchedBytes, selected.delivery?.requestedBytes)} />
            <Meta
              label="OGC scope"
              value={selected.delivery?.source === "quantized-ogc" ? formatChunkScope(selected.delivery?.chunkIds) : "-"}
            />
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
        debugLens={debugLens}
        debugEvents={debugEvents}
        debugSnapshot={selectedDebugSnapshot}
        hiddenObjects={hiddenObjects}
        stability={selectedStability}
        onToggleObjectVisibility={toggleObjectVisibility}
        onSelectDebugLens={selectDebugLens}
        onSelectTrainableFrame={selectTrainableFrame}
        onSelectOgcLod={selectOgcLod}
        onSelectOgcChunks={selectOgcChunks}
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
  const options = {
    chunkIds: model.ogc?.chunkIds,
    lodLevel: model.ogc?.lodLevel,
  };
  const payload = await loadOgcPayload(artifact, index, options);
  return {
    artifact,
    index,
    delivery: payload.delivery,
    decoded: payload.windows
      ? decodeQuantizedOgcPayloadWindows(payload.windows, index, options)
      : decodeQuantizedOgcPayload(payload.buffer, index, options),
  };
}

async function loadTrainableArtifact(model) {
  if (model.trainableArtifactPath) {
    const response = await fetch(model.trainableArtifactPath);
    if (!response.ok) throw new Error(`trainable artifact HTTP ${response.status}`);
    return validateTrainableArtifact(await response.json());
  }
  return validateTrainableArtifact(model.trainableArtifact);
}

function validateTrainableArtifact(artifact) {
  if (artifact?.schema !== "objgauss-trainable-kernel-model-artifact-v1") {
    throw new Error("unsupported trainable kernel model artifact schema");
  }
  if (artifact.kind !== "trainable_kernel_mvp_model") {
    throw new Error("unsupported trainable kernel model artifact kind");
  }
  if (!Array.isArray(artifact.assignments) || !artifact.assignments.length) {
    throw new Error("trainable artifact missing assignments");
  }
  if (!Array.isArray(artifact.object_states) || !artifact.object_states.length) {
    throw new Error("trainable artifact missing object states");
  }
  return artifact;
}

function trainableLocalArtifactModel(fileName) {
  return {
    id: "trainable-local-artifact",
    name: "Local trainable artifact",
    label: "Local Artifact",
    loadMode: "trainable-artifact",
    kind: "trainable-kernel-model-artifact",
    stage: "local-debug-artifact",
    objectCount: 0,
    galleryPosition: [5.45, 0, 4.48],
    accent: "#7ff1d6",
    displayScale: 1.92,
    pointSize: 0.082,
    maxDisplayPoints: 256,
    trainableArtifactName: fileName,
    compression: {
      layout: "trainable-kernel-artifact-json",
      status: "local-debug-artifact",
      chunkRoot: "/models/local-trainable-artifact/objects/",
    },
  };
}

function ogcFilePair(files) {
  const indexFile = files.find((file) => /\.index\.json$/i.test(file.name)) ??
    files.find((file) => /\.json$/i.test(file.name));
  const payloadFile = files.find((file) => /\.ogc$/i.test(file.name));
  if (!indexFile || !payloadFile) {
    throw new Error("select one .index.json and one .ogc file");
  }
  return { indexFile, payloadFile };
}

function ogcLocalArtifactModel({ index, payloadBuffer, indexFileName, payloadFileName }) {
  const payloadPath = localFileRoute(payloadFileName);
  const indexPath = localFileRoute(indexFileName);
  const artifact = {
    role: "compressed_chunked",
    path: payloadPath,
    payloadPath,
    indexPath,
    format: ".ogc",
    delivery_tier: "browser_edit",
    browser_ready: true,
    gaussian_count: ogcPositiveInteger(index?.gaussian_count),
    object_count: ogcPositiveInteger(index?.object_count),
    byte_size: payloadBuffer.byteLength,
    sha256: index?.payload?.sha256,
    chunk_index: {
      schema: index?.schema,
      path: indexPath,
      chunk_count: Array.isArray(index?.chunks) ? index.chunks.length : undefined,
      sort_key: index?.sort_key,
      chunk_size_target: index?.chunk_size_target,
    },
    compression: index?.compression,
    lod: index?.lod,
    object_id_coverage: index?.object_id_coverage,
    inlineIndex: index,
    payloadBuffer,
  };
  return {
    id: "ogc-local-artifact",
    name: "Local OGC artifact",
    label: "Local OGC",
    loadMode: "ogc-chunked",
    kind: "compressed-chunked-ogc",
    stage: "local-debug-artifact",
    objectCount: ogcPositiveInteger(index?.object_count) ?? 0,
    galleryPosition: [2.85, 0, 4.56],
    accent: "#5df2df",
    displayScale: 1.88,
    pointSize: 0.058,
    maxDisplayPoints: 1200,
    license: "local-file-debug",
    ogc: { lodLevel: 0 },
    compression: {
      layout: index?.compression?.layout ?? "object-aware-chunked-local-quantized",
      status: "local-debug-artifact",
      chunkRoot: indexPath,
    },
    modelArtifactManifest: {
      schema: MODEL_ARTIFACT_MANIFEST_SCHEMA,
      manifest_id: "ogc-local-artifact-manifest",
      asset_id: "ogc-local-artifact",
      name: "Local OGC artifact",
      stage: "local-debug-artifact",
      source: {
        type: "local_file_pair",
        index_path: indexPath,
        payload_path: payloadPath,
      },
      license: "local-file-debug",
      counts: {
        gaussians: ogcPositiveInteger(index?.gaussian_count),
        objects: ogcPositiveInteger(index?.object_count),
      },
      artifacts: [artifact],
      quality_evidence: [],
      limitations: [
        "Local OGC artifacts are browser-session debug inputs and are not copied into public assets.",
      ],
      created_from: {
        index_file: indexFileName,
        payload_file: payloadFileName,
      },
    },
  };
}

function localFileRoute(fileName) {
  return `local://${fileName}`;
}

function ogcPositiveInteger(value) {
  const number = Number(value);
  return Number.isInteger(number) && number >= 0 ? number : undefined;
}

function availableOgcLodLevels(index) {
  const levels = Array.isArray(index?.lod?.levels) ? index.lod.levels : [];
  const ids = levels
    .map((level) => Number(level?.level))
    .filter((level) => Number.isInteger(level) && level >= 0);
  return ids.length ? [...new Set(ids)].sort((left, right) => left - right) : [0];
}

function availableOgcChunkIds(index) {
  const chunks = Array.isArray(index?.chunks) ? index.chunks : [];
  const ids = chunks
    .map((chunk) => Number(chunk?.chunk_id))
    .filter((chunkId) => Number.isInteger(chunkId) && chunkId >= 0);
  return [...new Set(ids)].sort((left, right) => left - right);
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

async function loadOgcPayload(artifact, index, options) {
  if (artifact.payloadBuffer) {
    const buffer = artifact.payloadBuffer;
    const readWindows = quantizedOgcReadWindows(index, options);
    return {
      buffer,
      delivery: {
        loadRoute: "local-file",
        fetchedBytes: buffer.byteLength,
        requestedBytes: readWindows.length
          ? readWindows.reduce((total, window) => total + window.byteLength, 0)
          : buffer.byteLength,
        decodedWindows: readWindows.length,
      },
    };
  }
  if (artifact.payloadBase64) {
    const buffer = base64ToArrayBuffer(artifact.payloadBase64);
    return {
      buffer,
      delivery: {
        loadRoute: "inline-ogc",
        fetchedBytes: buffer.byteLength,
        requestedBytes: buffer.byteLength,
        decodedWindows: 0,
      },
    };
  }
  const payloadPath = artifact.payloadPath ?? artifact.path ?? index?.payload?.path;
  if (!payloadPath || isInlineRoute(payloadPath)) {
    throw new Error("missing fetchable OGC payload");
  }
  const rangeResult = await tryLoadOgcPayloadWindows(payloadPath, index, options);
  if (rangeResult) return rangeResult;
  const response = await fetch(payloadPath);
  if (!response.ok) throw new Error(`OGC payload HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  return {
    buffer,
    delivery: {
      loadRoute: "fetch-ogc",
      fetchedBytes: buffer.byteLength,
      requestedBytes: buffer.byteLength,
      decodedWindows: 0,
    },
  };
}

async function tryLoadOgcPayloadWindows(payloadPath, index, options) {
  const readWindows = quantizedOgcReadWindows(index, options);
  if (!readWindows.length) return null;
  try {
    const windows = await Promise.all(
      readWindows.map((window) => fetchOgcPayloadWindow(payloadPath, window)),
    );
    return {
      windows,
      delivery: {
        loadRoute: "range-ogc",
        fetchedBytes: windows.reduce((total, window) => total + window.buffer.byteLength, 0),
        requestedBytes: readWindows.reduce((total, window) => total + window.byteLength, 0),
        decodedWindows: windows.length,
      },
    };
  } catch (error) {
    console.warn(`OGC range payload load failed; falling back to full payload fetch: ${error.message}`);
    return null;
  }
}

async function fetchOgcPayloadWindow(payloadPath, window) {
  const response = await fetch(payloadPath, {
    headers: {
      Range: `bytes=${window.byteOffset}-${window.byteEnd}`,
    },
  });
  if (response.status !== 206) {
    throw new Error(`OGC payload range ${window.chunkId} HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== window.byteLength) {
    throw new Error(`OGC payload range ${window.chunkId} returned ${buffer.byteLength}/${window.byteLength} bytes`);
  }
  return {
    chunkId: window.chunkId,
    byteOffset: window.byteOffset,
    byteLength: window.byteLength,
    buffer,
  };
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
  debugLens,
  hiddenSelectionIds,
  onReady,
  onSelectObject,
  onHoverObject,
  onObjectMoved,
  onDebugEvent,
}) {
  const mountRef = useRef(null);
  const apiRef = useRef(null);
  const selectedRef = useRef(selectedTargetId);
  const debugRef = useRef(debugMode);
  const debugLensRef = useRef(normalizeDebugLens(debugLens));
  const hiddenRef = useRef(hiddenSelectionIds);
  const callbacksRef = useRef({ onSelectObject, onHoverObject, onObjectMoved, onDebugEvent });

  useEffect(() => {
    selectedRef.current = selectedTargetId;
    apiRef.current?.setSelected(selectedTargetId);
  }, [selectedTargetId]);

  useEffect(() => {
    debugRef.current = debugMode;
    apiRef.current?.setDebugMode(debugMode);
  }, [debugMode]);

  useEffect(() => {
    const next = normalizeDebugLens(debugLens);
    debugLensRef.current = next;
    apiRef.current?.setDebugLens(next);
  }, [debugLens]);

  useEffect(() => {
    hiddenRef.current = hiddenSelectionIds;
    apiRef.current?.setHiddenObjects(hiddenSelectionIds);
  }, [hiddenSelectionIds]);

  useEffect(() => {
    callbacksRef.current = { onSelectObject, onHoverObject, onObjectMoved, onDebugEvent };
  }, [onDebugEvent, onHoverObject, onObjectMoved, onSelectObject]);

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
      const selectedStability = summarizeObjectStability(
        [...draggableObjects.values()].filter((object) => object.userData.modelId === selectedModelId),
      );
      const selectedAssignmentSource =
        selectedObject?.userData.objectState?.source ??
        selectedModel?.userData.assignmentSource ??
        "derived_from_object_id";
      const hoveredTarget = objectTarget(hoveredObject);
      window.__OBJGAUSS_WORLD__ = {
        renderer: "three.js",
        ui: "frosted-glass-in-world",
        sidebars: false,
        modelCount: modelRoots.size,
        objectCount: draggableObjects.size,
        draggableCount: draggableObjects.size,
        draggableObjectCount: draggableObjects.size,
        selectedId: selectedRef.current,
        selectedModelId,
        selectedObjectId: selectedObject?.userData.objectId ?? null,
        hoveredId: hoveredObject?.userData.selectionId ?? null,
        hoveredModelId: hoveredTarget?.modelId ?? null,
        hoveredObjectId: hoveredTarget?.objectId ?? null,
        hoveredGaussianCount: hoveredTarget?.gaussianCount ?? 0,
        hoveredAssignmentSource: hoveredTarget?.assignmentSource ?? null,
        debugMode: debugRef.current,
        debugLens: debugRef.current ? debugLensRef.current : "appearance",
        debugProtocol: "object-state-debug-os-v1",
        assignmentSource: selectedAssignmentSource,
        stabilitySummary: selectedStability,
        selectedTrainableFrameIndex: selectedModel?.userData?.trainableFrameIndex ?? null,
        selectedTrainableFrameCount: selectedModel?.userData?.trainableFrameCount ?? null,
        trainableArtifactLoadedCount: [...modelRoots.values()].filter(
          (object) => object.userData?.artifactSchema === "objgauss-trainable-kernel-model-artifact-v1",
        ).length,
        visibleObjectCount: [...draggableObjects.values()].filter((object) => object.visible).length,
        lensOpacitySamples: [...draggableObjects.values()].map((object) => {
          const cloud = firstGaussianCloud(object);
          return {
            selectionId: object.userData.selectionId,
            modelId: object.userData.modelId,
            objectId: object.userData.objectId,
            activeLens: cloud?.userData?.activeColorLens ?? null,
            opacityLens: cloud?.userData?.activeOpacityLens ?? null,
            opacity: round3(cloud?.material?.opacity ?? 0),
          };
        }),
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
            gaussianCount: objectGaussianCount(object),
            confidence: object.userData.objectState?.confidence ?? null,
            entropy: object.userData.objectState?.assignmentEntropy ?? null,
            spatialCompactness: object.userData.objectState?.spatialCompactness ?? null,
            assignmentJitter: object.userData.objectState?.assignmentJitter ?? null,
            bboxStability: object.userData.objectState?.bboxStability ?? null,
            frameIndex: object.userData.objectState?.frameIndex ?? null,
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
        hoverObjectForAudit(selectionId = null) {
          const object =
            (selectionId ? draggableObjects.get(selectionId) : null) ??
            [...draggableObjects.values()].find((entry) => entry.visible);
          if (!object) return { ok: false, selectionId: null, gaussianCount: 0 };
          const target = setHoveredObjectGroup(object);
          return {
            ok: Boolean(target?.selectionId),
            selectionId: target?.selectionId ?? null,
            modelId: target?.modelId ?? null,
            objectId: target?.objectId ?? null,
            gaussianCount: target?.gaussianCount ?? 0,
            assignmentSource: target?.assignmentSource ?? null,
          };
        },
        clearHoverForAudit() {
          setHoveredObjectGroup(null);
          return true;
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
      api.setDebugLens(debugLensRef.current);
      api.setHiddenObjects(hiddenRef.current);
      return result.summary;
    };

    const selectObjectGroup = (object, probe = null) => {
      const target = objectTarget(object);
      if (!target?.selectionId) return;
      selectedRef.current = target.selectionId;
      api.setSelected(target.selectionId);
      const gaussian = probe?.gaussian ?? null;
      callbacksRef.current.onDebugEvent?.(gaussian ? "gaussian-probe" : "select-object", {
        ...target,
        gaussianIndex: gaussian?.gaussianIndex ?? null,
        lens: debugRef.current ? debugLensRef.current : "appearance",
        source: gaussian?.source ?? target.assignmentSource ?? "world",
      });
      callbacksRef.current.onSelectObject?.(target, probe?.gaussian ?? null);
      publishAuditHandle();
    };

    const setHoveredObjectGroup = (object) => {
      const nextHover = object?.visible === false ? null : object;
      if (nextHover === hoveredObject) {
        publishAuditHandle();
        return objectTarget(hoveredObject);
      }
      hoveredObject = nextHover;
      api.setHover(hoveredObject?.userData.selectionId ?? null);
      const target = objectTarget(hoveredObject);
      if (target?.selectionId) {
        callbacksRef.current.onDebugEvent?.("hover-object", {
          ...target,
          lens: debugRef.current ? debugLensRef.current : "appearance",
          source: target.assignmentSource ?? "world",
        });
      }
      callbacksRef.current.onHoverObject?.(target);
      publishAuditHandle();
      return target;
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
      setHoveredObjectGroup(nextHover);
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
        refreshDebugLens();
        publishAuditHandle();
      },
      setDebugLens(lens) {
        debugLensRef.current = normalizeDebugLens(lens);
        refreshDebugLens();
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
            lens: debugLensRef.current,
          });
        }
      },
      setObjectVisibility(selectionId, visible) {
        const object = draggableObjects.get(selectionId);
        if (!object) return;
        object.visible = Boolean(visible);
        callbacksRef.current.onDebugEvent?.("toggle-visibility", {
          ...objectTarget(object),
          visible: object.visible,
          lens: debugRef.current ? debugLensRef.current : "appearance",
          source: "object-visibility",
        });
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
            lens: debugLensRef.current,
          });
        }
        publishAuditHandle();
      },
    };

    const refreshDebugLens = () => {
      const lens = debugLensRef.current;
      const debugEnabled = debugRef.current;
      for (const object of draggableObjects.values()) {
        object.traverse((child) => {
          if (child.userData.role === "gaussian-cloud") {
            const color = colorAttributeForDebugLens(child, lens, debugEnabled);
            if (color) {
              child.geometry.setAttribute("color", color);
              child.geometry.attributes.color.needsUpdate = true;
            }
          }
          if (child.userData.role === "object-state-bbox") {
            child.visible = Boolean(debugEnabled);
          }
        });
        applyObjectVisualState(object, {
          selected: object.userData.selected,
          hovered: object.userData.hovered,
          debug: debugEnabled,
          lens,
        });
      }
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

    models
      .filter((model) => model.loadMode !== "trainable-artifact")
      .forEach((model) => upsertModel(model));
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
  debugLens,
  debugEvents,
  debugSnapshot,
  hiddenObjects,
  stability,
  onToggleObjectVisibility,
  onSelectDebugLens,
  onSelectTrainableFrame,
  onSelectOgcLod,
  onSelectOgcChunks,
}) {
  if (!selected) return null;
  const objects = selected.objects ?? [];
  const activeState = selectedObject?.objectState ?? objects[0]?.objectState ?? null;
  const assignment = debugProbe?.assignment ?? selectedObject?.assignment ?? activeState?.assignment ?? [];
  const probeEntropy = debugProbe?.entropy ?? activeState?.assignmentEntropy ?? 0;
  const probeConfidence = debugProbe?.confidence ?? activeState?.confidence ?? 0;
  const rendererLoss = selected?.delivery?.imageRenderLoss;
  const frameCount = selected.delivery?.frameCount ?? selected.trainableArtifact?.object_states?.length ?? 0;
  const selectedFrameIndex = Number(selected.delivery?.frameIndex ?? selected.trainableFrameIndex ?? 0) || 0;
  const ogcLodLevels = selected.loadMode === "ogc-chunked" && Array.isArray(selected.delivery?.lodLevels)
    ? selected.delivery.lodLevels
    : [];
  const selectedOgcLod = Number(selected.delivery?.lodLevel ?? selected.ogc?.lodLevel ?? 0) || 0;
  const ogcChunkIds = selected.loadMode === "ogc-chunked" && Array.isArray(selected.delivery?.availableChunkIds)
    ? selected.delivery.availableChunkIds
    : [];
  const selectedOgcChunks = Array.isArray(selected.delivery?.chunkIds) ? selected.delivery.chunkIds : [];
  const selectedOgcChunkScope = formatChunkScope(selectedOgcChunks);
  return (
    <section
      className="glassHud debugPanel"
      data-object-debug-panel="true"
      data-debug-mode={debugMode ? "assignment" : "appearance"}
      data-debug-lens={debugMode ? debugLens : "appearance"}
      data-probe-source={debugProbe?.source ?? activeState?.source ?? "none"}
      data-trainable-frame-index={selectedFrameIndex}
      data-trainable-frame-count={frameCount}
      data-ogc-lod-index={selectedOgcLod}
      data-ogc-lod-count={ogcLodLevels.length}
      data-ogc-chunk-scope={selectedOgcChunkScope}
      data-ogc-chunk-count={ogcChunkIds.length}
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

      <div
        className="trainableFrameSelector debugLensSelector"
        data-debug-lens-selector="true"
        data-selected-lens={debugMode ? debugLens : "appearance"}
        data-debug-enabled={debugMode ? "true" : "false"}
      >
        <span>lens</span>
        {DEBUG_LENSES.map((lens) => (
          <button
            key={lens}
            type="button"
            className={debugMode && debugLens === lens ? "active" : ""}
            data-debug-lens-button={lens}
            data-active={debugMode && debugLens === lens ? "true" : "false"}
            onClick={() => onSelectDebugLens?.(lens)}
          >
            {debugLensLabel(lens)}
          </button>
        ))}
      </div>

      {frameCount > 1 ? (
        <div
          className="trainableFrameSelector"
          data-trainable-frame-selector="true"
          data-selected-frame={selectedFrameIndex}
          data-frame-count={frameCount}
        >
          <span>frame</span>
          {Array.from({ length: frameCount }, (_, index) => (
            <button
              key={index}
              type="button"
              className={index === selectedFrameIndex ? "active" : ""}
              data-trainable-frame-button={index}
              data-active={index === selectedFrameIndex ? "true" : "false"}
              onClick={() => onSelectTrainableFrame?.(index)}
            >
              f{index}
            </button>
          ))}
        </div>
      ) : null}

      {ogcLodLevels.length > 1 ? (
        <div
          className="trainableFrameSelector"
          data-ogc-lod-selector="true"
          data-selected-lod={selectedOgcLod}
          data-lod-count={ogcLodLevels.length}
        >
          <span>lod</span>
          {ogcLodLevels.map((level) => (
            <button
              key={level}
              type="button"
              className={level === selectedOgcLod ? "active" : ""}
              data-ogc-lod-button={level}
              data-active={level === selectedOgcLod ? "true" : "false"}
              onClick={() => onSelectOgcLod?.(level)}
            >
              L{level}
            </button>
          ))}
        </div>
      ) : null}

      {ogcChunkIds.length > 1 ? (
        <div
          className="trainableFrameSelector"
          data-ogc-chunk-selector="true"
          data-selected-chunks={selectedOgcChunkScope}
          data-chunk-count={ogcChunkIds.length}
        >
          <span>chunk</span>
          <button
            type="button"
            className={selectedOgcChunkScope === "all" ? "active" : ""}
            data-ogc-chunk-button="all"
            data-active={selectedOgcChunkScope === "all" ? "true" : "false"}
            onClick={() => onSelectOgcChunks?.([])}
          >
            all
          </button>
          {ogcChunkIds.map((chunkId) => (
            <button
              key={chunkId}
              type="button"
              className={selectedOgcChunks.length === 1 && selectedOgcChunks[0] === chunkId ? "active" : ""}
              data-ogc-chunk-button={chunkId}
              data-active={selectedOgcChunks.length === 1 && selectedOgcChunks[0] === chunkId ? "true" : "false"}
              onClick={() => onSelectOgcChunks?.([chunkId])}
            >
              c{chunkId}
            </button>
          ))}
        </div>
      ) : null}

      <AssignmentHeatmap assignment={assignment} selectedObject={selectedObject} debugProbe={debugProbe} />
      <DebugSnapshotPanel snapshot={debugSnapshot} />
      <DebugEventTracePanel events={debugEvents} />
      <StabilityDashboard stability={stability} />
      <TrainingEvidencePanel artifact={selected.trainableArtifact} />

      <dl className="debugStateGrid">
        <Meta label="source" value={debugProbe?.source ?? activeState?.source} />
        <Meta label="renderer" value={selected.delivery?.rendererName} />
        <Meta label="gaussian n" value={debugProbe?.gaussianIndex ?? "-"} />
        <Meta label="centroid" value={formatVec(activeState?.centroid)} />
        <Meta label="bbox" value={formatBox(activeState?.bbox)} />
        <Meta
          label="hover"
          value={
            hoveredTarget
              ? `${hoveredTarget.modelId} #${hoveredTarget.objectId} / ${formatNumber(hoveredTarget.gaussianCount)}G`
              : "-"
          }
        />
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

function DebugEventTracePanel({ events }) {
  const recent = Array.isArray(events) ? events.slice(0, 4) : [];
  if (!recent.length) return null;
  return (
    <div
      className="stabilityDashboard debugTracePanel"
      data-debug-event-trace="true"
      data-debug-event-count={events.length}
      data-debug-event-last={recent[0]?.type ?? ""}
      data-debug-event-schema={recent[0]?.schema ?? ""}
    >
      <div className="stabilityHead">
        <span>Trace</span>
        <strong>{recent[0]?.type ?? "-"}</strong>
      </div>
      <div className="debugEventRows">
        {recent.map((event) => (
          <div
            className="debugEventRow"
            key={event.seq}
            data-debug-event-row="true"
            data-debug-event-type={event.type}
          >
            <span>{event.type}</span>
            <small>{debugEventDetailLabel(event)}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function DebugSnapshotPanel({ snapshot }) {
  if (!snapshot) return null;
  return (
    <div
      className="stabilityDashboard debugSnapshotPanel"
      data-debug-snapshot-panel="true"
      data-debug-snapshot-schema={snapshot.schema}
      data-debug-snapshot-model={snapshot.model.id}
      data-debug-snapshot-object={snapshot.selection.objectId ?? ""}
      data-debug-snapshot-gaussian={snapshot.selection.gaussianIndex ?? ""}
      data-debug-snapshot-lens={snapshot.debug.lens}
      data-debug-snapshot-source={snapshot.assignment.source}
      data-debug-snapshot-slots={snapshot.assignment.slotCount}
      data-debug-snapshot-stability={snapshot.stability.status}
      data-debug-snapshot-training-status={snapshot.training?.status ?? ""}
    >
      <div className="stabilityHead">
        <span>Protocol</span>
        <strong>snapshot-v1</strong>
      </div>
      <dl className="stabilityMeta snapshotMeta">
        <Meta label="model" value={snapshot.model.id} />
        <Meta label="object" value={snapshot.selection.objectId ?? "-"} />
        <Meta label="lens" value={snapshot.debug.lens} />
        <Meta label="slots" value={formatCount(snapshot.assignment.slotCount)} />
        <Meta label="source" value={snapshot.assignment.source} />
        <Meta label="state" value={snapshot.stability.status} />
      </dl>
    </div>
  );
}

function TrainingEvidencePanel({ artifact }) {
  const summary = trainableEvidenceSummary(artifact);
  if (!summary) return null;
  return (
    <div
      className="stabilityDashboard trainingEvidence"
      data-training-evidence="true"
      data-training-status={summary.status}
      data-training-schema={summary.schema}
      data-training-renderer={summary.rendererName}
      data-training-gradient-path={summary.gradientPath}
      data-training-iterations={summary.iterations}
      data-training-initial-total-loss={summary.initialTotalLoss ?? ""}
      data-training-final-total-loss={summary.finalTotalLoss ?? ""}
      data-training-loss-delta={summary.totalLossDelta ?? ""}
      data-training-initial-image-loss={summary.initialImageLoss ?? ""}
      data-training-final-image-loss={summary.finalImageLoss ?? ""}
      data-training-image-loss-delta={summary.imageLossDelta ?? ""}
      data-training-image-loss-decreased={summary.imageLossDecreased ? "true" : "false"}
      data-training-final-render-loss={summary.finalRenderLoss ?? ""}
      data-training-final-object-loss={summary.finalObjectLoss ?? ""}
      data-training-final-temporal-loss={summary.finalTemporalLoss ?? ""}
    >
      <div className="stabilityHead">
        <span>Training</span>
        <strong>{summary.status}</strong>
      </div>
      <div className="stabilityGrid trainingGrid">
        <Metric label="total" value={formatLoss(summary.finalTotalLoss)} />
        <Metric label="image" value={formatLoss(summary.finalImageLoss)} />
        <Metric label="object" value={formatLoss(summary.finalObjectLoss)} />
        <Metric label="temp" value={formatLoss(summary.finalTemporalLoss)} />
      </div>
      <dl className="stabilityMeta trainingMeta">
        <Meta label="delta" value={formatSignedLoss(summary.totalLossDelta)} />
        <Meta label="iter" value={formatCount(summary.iterations)} />
        <Meta label="renderer" value={summary.rendererName} />
        <Meta label="grad" value={summary.gradientPath} />
      </dl>
    </div>
  );
}

function StabilityDashboard({ stability }) {
  const summary = stability ?? summarizeObjectStability([]);
  return (
    <div
      className="stabilityDashboard"
      data-stability-dashboard="true"
      data-stability-status={summary.status}
      data-slot-utilization={summary.slotUtilization}
      data-mean-entropy={summary.meanEntropy}
      data-mixed-slots={summary.mixedSlots}
      data-low-confidence-slots={summary.lowConfidenceSlots}
      data-purity-available={summary.purityAvailable ? "true" : "false"}
      data-mean-purity={summary.meanPurity ?? ""}
      data-temporal-available={summary.temporalAvailable ? "true" : "false"}
      data-mean-temporal-drift={summary.meanTemporalDrift ?? ""}
      data-spatial-available={summary.spatialAvailable ? "true" : "false"}
      data-mean-spatial-compactness={summary.meanSpatialCompactness ?? ""}
      data-jitter-available={summary.jitterAvailable ? "true" : "false"}
      data-mean-assignment-jitter={summary.meanAssignmentJitter ?? ""}
      data-bbox-available={summary.bboxAvailable ? "true" : "false"}
      data-mean-bbox-stability={summary.meanBboxStability ?? ""}
    >
      <div className="stabilityHead">
        <span>Stability</span>
        <strong>{summary.status}</strong>
      </div>
      <div className="stabilityGrid">
        <Metric label="slot util" value={formatRatio(summary.slotUtilization)} />
        <Metric label="mean H" value={formatRatio(summary.meanEntropy)} />
        <Metric label="mixed" value={formatCount(summary.mixedSlots)} />
        <Metric label="low conf" value={formatCount(summary.lowConfidenceSlots)} />
      </div>
      <div className="stabilityBars">
        <StabilityBar label="confidence" value={summary.meanConfidence} invert={false} />
        <StabilityBar label="entropy" value={summary.meanEntropy} invert />
        <StabilityBar label="compact" value={summary.meanSpatialCompactness} available={summary.spatialAvailable} />
        <StabilityBar label="jitter" value={summary.meanAssignmentJitter} available={summary.jitterAvailable} invert />
      </div>
      <dl className="stabilityMeta">
        <Meta label="purity" value={summary.purityAvailable ? formatRatio(summary.meanPurity) : "n/a"} />
        <Meta label="spatial" value={summary.spatialAvailable ? formatRatio(summary.meanSpatialCompactness) : "n/a"} />
        <Meta label="drift" value={summary.temporalAvailable ? formatRatio(summary.meanTemporalDrift) : "n/a"} />
        <Meta label="jitter" value={summary.jitterAvailable ? formatRatio(summary.meanAssignmentJitter) : "n/a"} />
        <Meta label="bbox" value={summary.bboxAvailable ? formatRatio(summary.meanBboxStability) : "n/a"} />
      </dl>
    </div>
  );
}

function StabilityBar({ label, value, invert = false, available = true }) {
  const number = Number(value);
  const hasValue = Boolean(available && Number.isFinite(number));
  const clean = hasValue ? Math.max(0, Math.min(1, number)) : 0;
  const width = hasValue ? `${Math.max(3, clean * 100)}%` : "0%";
  return (
    <div className={`stabilityBar ${hasValue ? "" : "missing"}`}>
      <span>{label}</span>
      <div className={invert ? "stabilityTrack invert" : "stabilityTrack"}>
        <b style={{ width }} />
      </div>
      <strong>{hasValue ? formatRatio(clean) : "n/a"}</strong>
    </div>
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
    const confidenceColors = new Float32Array(entries.length * 3);
    const entropyColors = new Float32Array(entries.length * 3);
    const fallback = new THREE.Color(accent);
    const debugColor = fallback.clone().lerp(new THREE.Color("#f1fdff"), assignmentEntropy * 0.34);
    const confidenceColor = debugConfidenceColor(assignmentConfidence);
    const entropyColor = debugEntropyColor(assignmentEntropy);

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
      writeColor(confidenceColors, index, confidenceColor);
      writeColor(entropyColors, index, entropyColor);
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const originalColorAttr = new THREE.BufferAttribute(originalColors, 3);
    const assignmentColorAttr = new THREE.BufferAttribute(assignmentColors, 3);
    const confidenceColorAttr = new THREE.BufferAttribute(confidenceColors, 3);
    const entropyColorAttr = new THREE.BufferAttribute(entropyColors, 3);
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
    cloud.userData.confidenceColor = confidenceColorAttr;
    cloud.userData.entropyColor = entropyColorAttr;
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
      spatialCompactness: spatialCompactnessForGeometry(geometry),
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
      spatialCompactness: objectState.spatialCompactness,
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
    const confidenceColorAttr = uniformColorAttribute(
      points.positions.length / 3,
      debugConfidenceColor(assignmentConfidence),
    );
    const entropyColorAttr = uniformColorAttribute(
      points.positions.length / 3,
      debugEntropyColor(assignmentEntropy),
    );
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
    cloud.userData.confidenceColor = confidenceColorAttr;
    cloud.userData.entropyColor = entropyColorAttr;
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
      spatialCompactness: spatialCompactnessForGeometry(geometry),
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
      spatialCompactness: objectState.spatialCompactness,
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
  const frameIndex = Number(frame?.frame_index ?? 0) || 0;
  const derivedIds = Array.isArray(frame?.derived_object_ids) ? frame.derived_object_ids : [];
  if (!states.length || !matrix.length) {
    throw new Error("trainable artifact fixture needs states and assignment matrix");
  }

  const group = baseModelGroup(model);
  group.userData.assignmentSource = "trainable_kernel_model_artifact";
  group.userData.artifactSchema = artifact.schema;
  group.userData.trainableFrameIndex = frameIndex;
  group.userData.trainableFrameCount = artifact.object_states?.length ?? 0;
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
    const confidenceColors = new Float32Array(stateRows.length * 3);
    const entropyColors = new Float32Array(stateRows.length * 3);
    const fallback = new THREE.Color(accent);
    const debugColor = fallback.clone().lerp(new THREE.Color("#f1fdff"), Number(state.normalized_assignment_entropy ?? 0) * 0.34);
    const assignment = averageAssignmentVector(stateRows.map((entry) => entry.row), objectIds);
    const purity = objectPurityForRows(stateRows, derivedIds);
    const temporalDrift = objectTemporalDrift(artifact, frameIndex, objectId, state.centroid);
    const assignmentJitter = objectAssignmentJitter(artifact, frameIndex, stateRows);
    const bboxStability = objectBboxStability(artifact, frameIndex, objectId, state.bbox);

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
      const rowValues = entry.row.map((value) => Number(value) || 0);
      writeColor(confidenceColors, index, debugConfidenceColor(Math.max(...rowValues)));
      writeColor(entropyColors, index, debugEntropyColor(normalizedEntropy(rowValues)));
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const originalColorAttr = new THREE.BufferAttribute(originalColors, 3);
    const assignmentColorAttr = new THREE.BufferAttribute(assignmentColors, 3);
    const confidenceColorAttr = new THREE.BufferAttribute(confidenceColors, 3);
    const entropyColorAttr = new THREE.BufferAttribute(entropyColors, 3);
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
    cloud.userData.confidenceColor = confidenceColorAttr;
    cloud.userData.entropyColor = entropyColorAttr;
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
      objectPurity: purity.value,
      purityLabel: purity.label,
      temporalDrift,
      spatialCompactness: spatialCompactnessForGeometry(geometry),
      assignmentJitter,
      bboxStability,
      frameIndex,
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
      objectPurity: objectState.objectPurity,
      temporalDrift: objectState.temporalDrift,
      spatialCompactness: objectState.spatialCompactness,
      assignmentJitter: objectState.assignmentJitter,
      bboxStability: objectState.bboxStability,
      frameIndex,
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
      trainableFrameIndex: frameIndex,
      trainableFrameCount: artifact.object_states?.length ?? 0,
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

function objectGaussianCount(object) {
  const cloud = firstGaussianCloud(object);
  return cloud?.userData?.gaussianDebug?.length ?? cloud?.geometry?.attributes?.position?.count ?? 0;
}

function gaussianProbeFromIntersection(intersection) {
  if (!intersection?.object?.userData || intersection.index === undefined) return null;
  if (intersection.object.userData.role !== "gaussian-cloud") return null;
  const probe = intersection.object.userData.gaussianDebug?.[intersection.index];
  return probe ?? null;
}

function colorAttributeForDebugLens(cloud, lens, debugEnabled = true) {
  if (!debugEnabled) {
    cloud.userData.activeColorLens = "appearance";
    return cloud.userData.originalColor;
  }
  const normalized = normalizeDebugLens(lens);
  cloud.userData.activeColorLens = normalized;
  if (normalized === "confidence") return cloud.userData.confidenceColor ?? cloud.userData.assignmentColor;
  if (normalized === "entropy") return cloud.userData.entropyColor ?? cloud.userData.assignmentColor;
  return cloud.userData.assignmentColor;
}

function applyObjectVisualState(object, { selected = false, hovered = false, debug = true, lens = "assignment" } = {}) {
  const selectedOrHovered = Boolean(selected || hovered);
  const normalizedLens = normalizeDebugLens(lens);
  object.traverse((child) => {
    if (child.userData.role === "gaussian-cloud") {
      const baseSize = child.userData.basePointSize ?? child.material.size;
      child.userData.basePointSize = baseSize;
      const lensOpacity = opacityForDebugLens(object.userData.objectState, normalizedLens);
      child.userData.activeOpacityLens = debug ? normalizedLens : "appearance";
      child.material.opacity = selected ? 1 : hovered ? Math.max(0.82, lensOpacity) : debug ? lensOpacity : 0.5;
      child.material.size = selected ? baseSize * 1.22 : hovered ? baseSize * 1.16 : baseSize;
      child.material.needsUpdate = true;
    }
    if (child.userData.role === "object-state-bbox") {
      child.material.opacity = selected ? 0.86 : hovered ? 0.72 : 0.34;
    }
    if (child.userData.role === "selection-ring" || child.userData.role === "core-glow") {
      child.visible = selectedOrHovered;
    }
  });
}

function opacityForDebugLens(objectState, lens) {
  if (lens === "confidence") {
    return round3(0.32 + 0.58 * clamp01(objectState?.confidence ?? 0.5));
  }
  if (lens === "entropy") {
    return round3(0.32 + 0.58 * clamp01(objectState?.assignmentEntropy ?? 0.5));
  }
  return 0.62;
}

function objectTarget(object) {
  if (!object?.userData?.selectionId) return null;
  return {
    modelId: object.userData.modelId,
    objectId: object.userData.objectId,
    selectionId: object.userData.selectionId,
    gaussianCount: objectGaussianCount(object),
    assignmentSource: object.userData.objectState?.source ?? null,
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

function normalizeDebugLens(lens) {
  const value = String(lens ?? "assignment");
  return DEBUG_LENSES.includes(value) ? value : "assignment";
}

function debugLensLabel(lens) {
  if (lens === "confidence") return "conf";
  if (lens === "entropy") return "H";
  return "assign";
}

function debugConfidenceColor(confidence) {
  return new THREE.Color("#ff4d6d").lerp(new THREE.Color("#5df2df"), clamp01(confidence));
}

function debugEntropyColor(entropy) {
  return new THREE.Color("#53d8da").lerp(new THREE.Color("#ff8a5c"), clamp01(entropy));
}

function writeColor(target, index, color) {
  target[index * 3] = color.r;
  target[index * 3 + 1] = color.g;
  target[index * 3 + 2] = color.b;
}

function uniformColorAttribute(count, color) {
  const safeCount = Math.max(0, Math.trunc(Number(count) || 0));
  const values = new Float32Array(safeCount * 3);
  for (let index = 0; index < safeCount; index += 1) {
    writeColor(values, index, color);
  }
  return new THREE.BufferAttribute(values, 3);
}

function clamp01(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0;
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
  spatialCompactness,
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
    spatialCompactness: optionalRound3(spatialCompactness),
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

function summarizeObjectStability(objectsOrStates = []) {
  const states = objectsOrStates
    .map((entry) => entry?.objectState ?? entry?.userData?.objectState ?? entry)
    .filter((state) => state && typeof state === "object");
  const totalSlots = states.length;
  const activeStates = states.filter((state) => state.status !== "inactive" && Number(state.slotMass ?? 0) > 0);
  const count = Math.max(1, states.length);
  const entropies = states.map((state) => finiteOrZero(state.assignmentEntropy));
  const confidences = states.map((state) => finiteOrZero(state.confidence));
  const purities = states
    .map((state) => firstFinite(state.objectPurity, state.purity, state.maskPurity))
    .filter(Number.isFinite);
  const drifts = states
    .map((state) => firstFinite(state.temporalDrift, state.drift, state.centroidDrift))
    .filter(Number.isFinite);
  const compactness = states
    .map((state) => firstFinite(state.spatialCompactness, state.compactness, state.spatialContinuity))
    .filter(Number.isFinite);
  const jitters = states
    .map((state) => firstFinite(state.assignmentJitter, state.jitter, state.assignmentDrift))
    .filter(Number.isFinite);
  const bboxStabilities = states
    .map((state) => firstFinite(state.bboxStability, state.bboxIoU, state.bboxConvergence))
    .filter(Number.isFinite);
  const mixedSlots = states.filter((state) => {
    const entropy = finiteOrZero(state.assignmentEntropy);
    return entropy >= 0.55 || String(state.status ?? "").includes("mixed");
  }).length;
  const lowConfidenceSlots = states.filter((state) => finiteOrZero(state.confidence) < 0.7).length;
  const meanEntropy = average(entropies);
  const meanConfidence = average(confidences);
  const slotUtilization = totalSlots === 0 ? 0 : activeStates.length / totalSlots;
  const status =
    totalSlots === 0
      ? "empty"
      : mixedSlots > 0
        ? "mixed"
        : lowConfidenceSlots > 0
          ? "low-confidence"
          : "stable";
  return {
    schema: "objgauss-stability-dashboard-v1",
    status,
    slotCount: totalSlots,
    activeSlots: activeStates.length,
    slotUtilization: round3(slotUtilization),
    meanEntropy: round3(meanEntropy),
    maxEntropy: round3(Math.max(0, ...entropies)),
    meanConfidence: round3(meanConfidence),
    minConfidence: round3(confidences.length ? Math.min(...confidences) : 0),
    mixedSlots,
    lowConfidenceSlots,
    purityAvailable: purities.length > 0,
    meanPurity: purities.length ? round3(average(purities)) : null,
    temporalAvailable: drifts.length > 0,
    meanTemporalDrift: drifts.length ? round3(average(drifts)) : null,
    spatialAvailable: compactness.length > 0,
    meanSpatialCompactness: compactness.length ? round3(average(compactness)) : null,
    jitterAvailable: jitters.length > 0,
    meanAssignmentJitter: jitters.length ? round3(average(jitters)) : null,
    bboxAvailable: bboxStabilities.length > 0,
    meanBboxStability: bboxStabilities.length ? round3(average(bboxStabilities)) : null,
  };
}

function firstFinite(...values) {
  for (const value of values) {
    if (value === null || value === undefined || value === "") continue;
    const number = Number(value);
    if (Number.isFinite(number)) return number;
  }
  return NaN;
}

function finiteOrZero(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function average(values) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function optionalRound3(value) {
  const number = Number(value);
  return Number.isFinite(number) ? round3(number) : null;
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
  if (!Array.isArray(value) || value.length < 3) return fallback?.clone?.() ?? null;
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

function objectPurityForRows(rows, derivedIds = []) {
  const counts = new Map();
  rows.forEach((entry) => {
    const label = derivedIds[entry.rowIndex];
    if (label === null || label === undefined) return;
    const key = String(label);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  });
  const total = [...counts.values()].reduce((sum, value) => sum + value, 0);
  if (!total) return { value: null, label: null };
  let bestLabel = null;
  let bestCount = 0;
  for (const [label, count] of counts) {
    if (count > bestCount) {
      bestLabel = label;
      bestCount = count;
    }
  }
  return {
    value: round3(bestCount / total),
    label: bestLabel,
  };
}

function objectTemporalDrift(artifact, frameIndex, objectId, centroid) {
  const frames = Array.isArray(artifact?.object_states) ? artifact.object_states : [];
  const current = vectorFromArray(centroid, null);
  if (!current) return null;
  const candidateFrames = frames
    .filter((frame) => Number(frame?.frame_index) !== frameIndex)
    .map((frame) => ({
      frame,
      distance: Math.abs(Number(frame?.frame_index ?? 0) - frameIndex),
    }))
    .filter((entry) => Number.isFinite(entry.distance))
    .sort((left, right) => left.distance - right.distance);
  for (const entry of candidateFrames) {
    const peer = entry.frame?.states?.find((state) => String(state.id) === String(objectId));
    const peerCentroid = vectorFromArray(peer?.centroid, null);
    if (peerCentroid) return round3(current.distanceTo(peerCentroid));
  }
  return null;
}

function objectAssignmentJitter(artifact, frameIndex, rows) {
  const frames = Array.isArray(artifact?.assignments) ? artifact.assignments : [];
  const candidateFrames = frames
    .filter((frame) => Number(frame?.frame_index) !== frameIndex && Array.isArray(frame?.matrix))
    .map((frame) => ({
      frame,
      distance: Math.abs(Number(frame?.frame_index ?? 0) - frameIndex),
    }))
    .filter((entry) => Number.isFinite(entry.distance))
    .sort((left, right) => left.distance - right.distance);
  const peer = candidateFrames[0]?.frame;
  if (!peer) return null;
  const deltas = [];
  rows.forEach((entry) => {
    const current = Array.isArray(entry.row) ? entry.row : [];
    const next = Array.isArray(peer.matrix?.[entry.rowIndex]) ? peer.matrix[entry.rowIndex] : [];
    const width = Math.max(current.length, next.length);
    if (!width) return;
    let l1 = 0;
    for (let index = 0; index < width; index += 1) {
      l1 += Math.abs((Number(current[index]) || 0) - (Number(next[index]) || 0));
    }
    deltas.push(Math.min(1, l1 * 0.5));
  });
  return deltas.length ? round3(average(deltas)) : null;
}

function objectBboxStability(artifact, frameIndex, objectId, bbox) {
  const frames = Array.isArray(artifact?.object_states) ? artifact.object_states : [];
  const currentBox = validBoxFromBbox(bbox);
  if (!currentBox) return null;
  const candidateFrames = frames
    .filter((frame) => Number(frame?.frame_index) !== frameIndex)
    .map((frame) => ({
      frame,
      distance: Math.abs(Number(frame?.frame_index ?? 0) - frameIndex),
    }))
    .filter((entry) => Number.isFinite(entry.distance))
    .sort((left, right) => left.distance - right.distance);
  for (const entry of candidateFrames) {
    const peer = entry.frame?.states?.find((state) => String(state.id) === String(objectId));
    const peerBox = validBoxFromBbox(peer?.bbox);
    if (peerBox) return round3(bboxIoU(currentBox, peerBox));
  }
  return null;
}

function bboxIoU(left, right) {
  const intersection = left.clone().intersect(right);
  if (intersection.isEmpty()) return 0;
  const intersectionVolume = boxVolume(intersection);
  const unionVolume = boxVolume(left) + boxVolume(right) - intersectionVolume;
  return unionVolume > 0 ? Math.max(0, Math.min(1, intersectionVolume / unionVolume)) : 0;
}

function validBoxFromBbox(bbox) {
  if (!Array.isArray(bbox) || bbox.length < 6) return null;
  const values = bbox.slice(0, 6).map(Number);
  if (!values.every(Number.isFinite)) return null;
  const box = new THREE.Box3(
    new THREE.Vector3(values[0], values[1], values[2]),
    new THREE.Vector3(values[3], values[4], values[5]),
  );
  return box.isEmpty() ? null : box;
}

function boxVolume(box) {
  const size = box.getSize(new THREE.Vector3());
  return Math.max(0, size.x) * Math.max(0, size.y) * Math.max(0, size.z);
}

function spatialCompactnessForGeometry(geometry) {
  const position = geometry?.attributes?.position;
  const box = geometry?.boundingBox?.isBox3 ? geometry.boundingBox : null;
  if (!position || !position.count || !box) return null;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const halfDiagonal = Math.max(size.length() / 2, 1e-6);
  const point = new THREE.Vector3();
  let distance = 0;
  for (let index = 0; index < position.count; index += 1) {
    point.fromBufferAttribute(position, index);
    distance += point.distanceTo(center);
  }
  const normalizedSpread = distance / (position.count * halfDiagonal);
  return round3(Math.max(0, Math.min(1, 1 / (1 + normalizedSpread))));
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

function initialModelStates(models = []) {
  return Object.fromEntries(
    models.map((model) => [
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

function objectStateDebugSnapshot({
  selected,
  selectedObject,
  selection,
  debugMode,
  debugLens,
  debugProbe,
  hiddenCount,
  stability,
  assignmentSource,
  trainingEvidence,
  debugEvents,
}) {
  const objects = selected?.objects ?? [];
  const activeObject = selectedObject ?? objects[0] ?? null;
  const activeState = activeObject?.objectState ?? null;
  const assignment = debugProbe?.assignment ?? activeObject?.assignment ?? activeState?.assignment ?? [];
  return {
    schema: "objgauss-object-state-debug-snapshot-v1",
    protocol: "object-state-debug-os-v1",
    model: {
      id: selected?.id ?? "",
      kind: selected?.kind ?? "",
      loadMode: selected?.loadMode ?? "",
      status: selected?.status ?? "",
      deliverySource: selected?.delivery?.source ?? "",
    },
    selection: {
      selectionId: selection?.selectionId ?? selected?.id ?? "",
      objectId: activeObject?.objectId ?? selection?.objectId ?? null,
      gaussianIndex: debugProbe?.gaussianIndex ?? null,
      hiddenObjectCount: hiddenCount,
    },
    debug: {
      enabled: Boolean(debugMode),
      lens: debugMode ? normalizeDebugLens(debugLens) : "appearance",
      probeSource: debugProbe?.source ?? activeState?.source ?? "none",
    },
    assignment: {
      source: assignmentSource || activeState?.source || selected?.delivery?.source || "none",
      slotCount: Array.isArray(assignment) ? assignment.length : 0,
      confidence: finiteNumber(debugProbe?.confidence ?? activeState?.confidence),
      entropy: finiteNumber(debugProbe?.entropy ?? activeState?.assignmentEntropy),
      vector: compactAssignmentVector(assignment),
    },
    objectState: {
      objectId: activeState?.objectId ?? activeObject?.objectId ?? null,
      status: activeState?.status ?? "",
      confidence: finiteNumber(activeState?.confidence),
      entropy: finiteNumber(activeState?.assignmentEntropy),
      slotMass: finiteNumber(activeState?.slotMass),
      massFraction: finiteNumber(activeState?.massFraction),
      centroid: cleanNumberArray(activeState?.centroid),
      bbox: cleanNumberArray(activeState?.bbox),
    },
    stability: {
      status: stability?.status ?? "unknown",
      slotUtilization: finiteNumber(stability?.slotUtilization),
      meanEntropy: finiteNumber(stability?.meanEntropy),
      mixedSlots: finiteNumber(stability?.mixedSlots),
      meanPurity: finiteNumber(stability?.meanPurity),
      meanTemporalDrift: finiteNumber(stability?.meanTemporalDrift),
      meanSpatialCompactness: finiteNumber(stability?.meanSpatialCompactness),
      meanAssignmentJitter: finiteNumber(stability?.meanAssignmentJitter),
      meanBboxStability: finiteNumber(stability?.meanBboxStability),
    },
    training: trainingEvidence
      ? {
          status: trainingEvidence.status,
          iterations: trainingEvidence.iterations,
          finalTotalLoss: trainingEvidence.finalTotalLoss,
          totalLossDelta: trainingEvidence.totalLossDelta,
          finalImageLoss: trainingEvidence.finalImageLoss,
          imageLossDelta: trainingEvidence.imageLossDelta,
          rendererName: trainingEvidence.rendererName,
          gradientPath: trainingEvidence.gradientPath,
        }
      : null,
    delivery: {
      loadRoute: selected?.delivery?.loadRoute ?? "",
      frameIndex: selected?.delivery?.frameIndex ?? null,
      frameCount: selected?.delivery?.frameCount ?? null,
      lodLevel: selected?.delivery?.lodLevel ?? null,
      chunkIds: Array.isArray(selected?.delivery?.chunkIds) ? selected.delivery.chunkIds : [],
    },
    events: compactDebugEvents(debugEvents),
  };
}

function debugEventFromDetail(type, detail = {}, seq = 0) {
  const clean = detail && typeof detail === "object" ? detail : {};
  return {
    schema: "objgauss-debug-event-v1",
    seq,
    type: String(type || "debug-event"),
    modelId: cleanString(clean.modelId),
    objectId: cleanNullable(clean.objectId),
    selectionId: cleanString(clean.selectionId),
    gaussianIndex: cleanNullable(clean.gaussianIndex),
    lens: cleanString(clean.lens),
    frameIndex: cleanNullable(clean.frameIndex),
    lodLevel: cleanNullable(clean.lodLevel),
    chunkScope: cleanString(clean.chunkScope),
    fileName: cleanString(clean.fileName),
    visible: typeof clean.visible === "boolean" ? clean.visible : null,
    source: cleanString(clean.source),
    position: cleanNumberArray(clean.position),
  };
}

function compactDebugEvents(events) {
  if (!Array.isArray(events)) return [];
  return events.slice(0, DEBUG_EVENT_LIMIT).map((event) => ({
    schema: event?.schema ?? "objgauss-debug-event-v1",
    seq: finiteNumber(event?.seq),
    type: String(event?.type ?? "debug-event"),
    modelId: cleanString(event?.modelId),
    objectId: cleanNullable(event?.objectId),
    selectionId: cleanString(event?.selectionId),
    gaussianIndex: cleanNullable(event?.gaussianIndex),
    lens: cleanString(event?.lens),
    frameIndex: cleanNullable(event?.frameIndex),
    lodLevel: cleanNullable(event?.lodLevel),
    chunkScope: cleanString(event?.chunkScope),
    fileName: cleanString(event?.fileName),
    visible: typeof event?.visible === "boolean" ? event.visible : null,
    source: cleanString(event?.source),
    position: cleanNumberArray(event?.position),
  }));
}

function debugEventDetailLabel(event) {
  if (!event) return "-";
  if (event.type === "debug-lens") return event.lens || "-";
  if (event.type === "frame-select") return event.frameIndex === null ? "frame -" : `f${event.frameIndex}`;
  if (event.type === "gaussian-probe") return event.gaussianIndex === null ? "G -" : `G${event.gaussianIndex}`;
  if (event.type === "toggle-visibility") return event.visible ? "visible" : "hidden";
  if (event.type === "ogc-lod") return event.lodLevel === null ? "LOD -" : `L${event.lodLevel}`;
  if (event.type === "ogc-chunks") return event.chunkScope || "all";
  if (
    event.type === "import-artifact" ||
    event.type === "import-artifact-error" ||
    event.type === "import-ogc" ||
    event.type === "import-ogc-error"
  ) return event.fileName || "local";
  if (event.objectId !== null && event.objectId !== undefined) return `#${event.objectId}`;
  return event.modelId || event.source || "-";
}

function compactAssignmentVector(assignment) {
  if (!Array.isArray(assignment)) return [];
  return assignment.slice(0, 16).map((slot) => ({
    slot: Number.isFinite(Number(slot?.slot)) ? Number(slot.slot) : null,
    objectId: slot?.objectId ?? null,
    probability: finiteNumber(slot?.probability),
  }));
}

function cleanNumberArray(value) {
  return Array.isArray(value)
    ? value.map((entry) => finiteNumber(entry)).filter((entry) => entry !== null)
    : [];
}

function cleanString(value) {
  return value === null || value === undefined ? "" : String(value);
}

function cleanNullable(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : String(value);
}

function trainableEvidenceSummary(artifact) {
  if (artifact?.schema !== "objgauss-trainable-kernel-model-artifact-v1") return null;
  const training = artifact.training;
  if (!training || typeof training !== "object") return null;
  const initial = training.initial_loss ?? {};
  const final = training.final_loss ?? {};
  const initialTotalLoss = finiteNumber(initial.total_loss);
  const finalTotalLoss = finiteNumber(final.total_loss);
  const initialImageLoss = finiteNumber(initial.image_render_loss);
  const finalImageLoss = finiteNumber(final.image_render_loss ?? artifact.renderer_api?.image_render_loss);
  const totalLossDelta =
    initialTotalLoss !== null && finalTotalLoss !== null ? round6(initialTotalLoss - finalTotalLoss) : null;
  const imageLossDelta =
    initialImageLoss !== null && finalImageLoss !== null ? round6(initialImageLoss - finalImageLoss) : null;
  const imageLossDecreased = Boolean(training.image_render_loss_decreased ?? (imageLossDelta !== null && imageLossDelta > 0));
  const totalLossDecreased = totalLossDelta !== null && totalLossDelta > 0;
  return {
    schema: training.schema ?? "",
    status: imageLossDecreased || totalLossDecreased ? "loss_down" : "loss_flat",
    iterations: finiteNumber(training.iterations),
    rendererName: artifact.renderer_api?.renderer_name ?? "-",
    gradientPath: artifact.renderer_api?.gradient_path ?? "-",
    imageLossDecreased,
    initialTotalLoss,
    finalTotalLoss,
    totalLossDelta,
    initialImageLoss,
    finalImageLoss,
    imageLossDelta,
    finalRenderLoss: finiteNumber(final.render_loss),
    finalObjectLoss: finiteNumber(final.object_loss),
    finalTemporalLoss: finiteNumber(final.temporal_loss),
  };
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number.toLocaleString() : "-";
}

function formatCount(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  return Number.isFinite(number) ? Math.trunc(number).toLocaleString() : "-";
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
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "-";
}

function formatLoss(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(6) : "-";
}

function formatSignedLoss(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number >= 0 ? "+" : ""}${number.toFixed(6)}`;
}

function formatFrame(index, count) {
  const frameIndex = Number(index);
  const frameCount = Number(count);
  if (!Number.isFinite(frameIndex) || !Number.isFinite(frameCount) || frameCount <= 0) return "-";
  return `${Math.trunc(frameIndex)} / ${Math.trunc(frameCount)}`;
}

function formatByteWindow(fetched, requested) {
  const fetchedBytes = Number(fetched);
  const requestedBytes = Number(requested);
  if (!Number.isFinite(fetchedBytes) || !Number.isFinite(requestedBytes)) return "-";
  return `${Math.trunc(fetchedBytes)} / ${Math.trunc(requestedBytes)}`;
}

function formatChunkScope(chunkIds) {
  return Array.isArray(chunkIds) && chunkIds.length ? chunkIds.join(",") : "all";
}

function round3(value) {
  return Math.round(Number(value) * 1000) / 1000;
}

function round6(value) {
  return Math.round(Number(value) * 1000000) / 1000000;
}
