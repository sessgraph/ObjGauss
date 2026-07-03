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

const OGC_CHUNK_INDEX_SCHEMA = "objgauss-chunk-index-v1";
const OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA = "objgauss-object-state-stability-benchmark-v1";
const DEBUG_LENSES = ["assignment", "confidence", "entropy", "opacity"];
const OBJECT_OVERLAY_MODES = ["full", "bbox", "centroid", "off"];
const DEBUG_EVENT_LIMIT = 12;
const HOVER_DIM_OPACITY = 0.18;

export default function App() {
  const modelCatalog = useMemo(
    () => modelCatalogFromSearch(typeof window === "undefined" ? "" : window.location.search),
    [],
  );
  const initialModelId = useMemo(() => defaultModelIdForCatalog(modelCatalog), [modelCatalog]);
  const worldApi = useRef(null);
  const loadStarted = useRef(false);
  const artifactInputRef = useRef(null);
  const modelBundleInputRef = useRef(null);
  const ogcInputRef = useRef(null);
  const debugSessionInputRef = useRef(null);
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
  const [benchmarkCaseName, setBenchmarkCaseName] = useState("");
  const [objectOverlayMode, setObjectOverlayMode] = useState("full");
  const [artifactImport, setArtifactImport] = useState(() => ({
    status: "idle",
    modelId: "",
    fileName: "",
    error: "",
  }));
  const [modelImport, setModelImport] = useState(() => ({
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
  const [snapshotExport, setSnapshotExport] = useState(() => ({
    status: "idle",
    fileName: "",
    schema: "",
    error: "",
  }));
  const [sessionExport, setSessionExport] = useState(() => ({
    status: "idle",
    fileName: "",
    schema: "",
    error: "",
  }));
  const [sessionImport, setSessionImport] = useState(() => ({
    status: "idle",
    fileName: "",
    schema: "",
    error: "",
  }));
  const [debugSessionArchive, setDebugSessionArchive] = useState(null);
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
  const selectedAssignmentProbe = assignmentProbeSummary(
    debugProbe?.assignment ?? selectedObject?.assignment ?? selectedObject?.objectState?.assignment ?? [],
    debugProbe,
  );
  const selectedContinuity = objectContinuitySummary(selectedObject ?? selected?.objects?.[0] ?? null);
  const selectedTemporal = objectTemporalSummary(selectedObject ?? selected?.objects?.[0] ?? null);
  const selectedExplainability = objectExplainabilitySummary({
    object: selectedObject ?? selected?.objects?.[0] ?? null,
    assignmentProbe: selectedAssignmentProbe,
    continuity: selectedContinuity,
    temporal: selectedTemporal,
  });
  const hoveredAssignmentProbe = assignmentProbeSummary(hoveredTarget?.assignment ?? [], {
    confidence: hoveredTarget?.confidence,
    entropy: hoveredTarget?.entropy,
    source: hoveredTarget?.assignmentSource,
  });
  const hoveredContinuity = objectContinuitySummary(hoveredTarget);
  const hoveredTemporal = objectTemporalSummary(hoveredTarget);
  const hoveredExplainability = objectExplainabilitySummary({
    object: hoveredTarget,
    assignmentProbe: hoveredAssignmentProbe,
    continuity: hoveredContinuity,
    temporal: hoveredTemporal,
  });
  const selectedObjectOverlayMode = normalizeObjectOverlayMode(objectOverlayMode);
  const hiddenCount = hiddenObjects.size;
  const objectVisibility = useMemo(
    () => objectVisibilitySummary(modelList, hiddenObjects),
    [modelList, hiddenObjects],
  );
  const selectedOgcChunkScope =
    selected?.delivery?.source === "quantized-ogc" ? formatChunkScope(selected.delivery?.chunkIds) : "";
  const selectedOgcAvailableChunks =
    selected?.delivery?.source === "quantized-ogc" ? formatChunkScope(selected.delivery?.availableChunkIds) : "";
  const selectedTrainingEvidence =
    selected?.delivery?.source === "trainable-kernel-model-artifact"
      ? trainableEvidenceSummary(selected.trainableArtifact)
      : null;
  const selectedQualityReport = qualityReportSummary(selected?.qualityReport);
  const selectedObjectStateBenchmark = objectStateBenchmarkSummary(selected?.objectStateBenchmark);
  const selectedBenchmarkCase = activeObjectStateBenchmarkCase(selectedObjectStateBenchmark, benchmarkCaseName);
  const selectedDebugSnapshot = objectStateDebugSnapshot({
    selected,
    selectedObject,
    selection,
    debugMode,
    debugLens,
    objectOverlayMode: selectedObjectOverlayMode,
    debugProbe,
    hoveredTarget,
    hoverAssignmentProbe: hoveredAssignmentProbe,
    objectContinuity: selectedContinuity,
    hoverContinuity: hoveredContinuity,
    objectTemporal: selectedTemporal,
    hoverTemporal: hoveredTemporal,
    objectExplainability: selectedExplainability,
    hoverExplainability: hoveredExplainability,
    hiddenCount,
    objectVisibility,
    stability: selectedStability,
    assignmentSource: selectedAssignmentSource,
    assignmentProbe: selectedAssignmentProbe,
    trainingEvidence: selectedTrainingEvidence,
    qualityReport: selectedQualityReport,
    objectStateBenchmark: selectedObjectStateBenchmark,
    objectStateBenchmarkCase: selectedBenchmarkCase,
    debugEvents,
  });
  const selectedDebugSession = objectStateDebugSession({
    snapshot: selectedDebugSnapshot,
    models: modelList,
    debugEvents,
  });
  const debugSessionDiff = debugSessionSnapshotDiff(selectedDebugSnapshot, debugSessionArchive?.snapshot);

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

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    window.__OBJGAUSS_DEBUG_SESSION__ = selectedDebugSession;
    return () => {
      if (window.__OBJGAUSS_DEBUG_SESSION__ === selectedDebugSession) {
        delete window.__OBJGAUSS_DEBUG_SESSION__;
      }
    };
  }, [selectedDebugSession]);

  useEffect(() => {
    if (typeof window === "undefined") return undefined;
    if (debugSessionArchive) {
      window.__OBJGAUSS_IMPORTED_DEBUG_SESSION__ = debugSessionArchive;
    } else {
      delete window.__OBJGAUSS_IMPORTED_DEBUG_SESSION__;
    }
    return () => {
      if (window.__OBJGAUSS_IMPORTED_DEBUG_SESSION__ === debugSessionArchive) {
        delete window.__OBJGAUSS_IMPORTED_DEBUG_SESSION__;
      }
    };
  }, [debugSessionArchive]);

  const recordDebugEvent = useCallback((type, detail = {}) => {
    const seq = debugEventSeq.current + 1;
    debugEventSeq.current = seq;
    const event = debugEventFromDetail(type, detail, seq);
    setDebugEvents((current) => [event, ...current].slice(0, DEBUG_EVENT_LIMIT));
    return event;
  }, []);

  const exportDebugSnapshot = useCallback(() => {
    if (!selectedDebugSnapshot) {
      setSnapshotExport({
        status: "error",
        fileName: "",
        schema: "",
        error: "debug snapshot unavailable",
      });
      return;
    }
    const fileName = debugSnapshotExportFileName(selectedDebugSnapshot);
    try {
      const exportedSnapshot = {
        ...selectedDebugSnapshot,
        export: {
          schema: "objgauss-debug-snapshot-export-v1",
          fileName,
          generatedBy: "objgauss-world-viewer",
          generatedAt: new Date().toISOString(),
        },
      };
      const text = `${JSON.stringify(exportedSnapshot, null, 2)}\n`;
      if (typeof window !== "undefined") {
        window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SNAPSHOT__ = exportedSnapshot;
        window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SNAPSHOT_TEXT__ = text;
      }
      if (typeof document !== "undefined" && typeof URL !== "undefined" && typeof Blob !== "undefined") {
        const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
        const link = document.createElement("a");
        link.href = url;
        link.download = fileName;
        link.rel = "noopener";
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
      setSnapshotExport({
        status: "exported",
        fileName,
        schema: exportedSnapshot.schema,
        error: "",
      });
      recordDebugEvent("export-snapshot", {
        modelId: selectedDebugSnapshot.model.id,
        objectId: selectedDebugSnapshot.selection.objectId,
        selectionId: selectedDebugSnapshot.selection.selectionId,
        gaussianIndex: selectedDebugSnapshot.selection.gaussianIndex,
        fileName,
        source: "debug-panel",
      });
    } catch (error) {
      const message = error?.message ?? "debug snapshot export failed";
      setSnapshotExport({
        status: "error",
        fileName,
        schema: selectedDebugSnapshot.schema ?? "",
        error: message,
      });
      recordDebugEvent("export-snapshot-error", {
        modelId: selectedDebugSnapshot.model.id,
        objectId: selectedDebugSnapshot.selection.objectId,
        selectionId: selectedDebugSnapshot.selection.selectionId,
        gaussianIndex: selectedDebugSnapshot.selection.gaussianIndex,
        fileName,
        source: "debug-panel",
      });
    }
  }, [recordDebugEvent, selectedDebugSnapshot]);

  const exportDebugSession = useCallback(() => {
    if (!selectedDebugSession) {
      setSessionExport({
        status: "error",
        fileName: "",
        schema: "",
        error: "debug session unavailable",
      });
      return;
    }
    const fileName = debugSessionExportFileName(selectedDebugSession);
    try {
      const exportedSession = {
        ...selectedDebugSession,
        export: {
          schema: "objgauss-debug-session-export-v1",
          fileName,
          generatedBy: "objgauss-world-viewer",
          generatedAt: new Date().toISOString(),
        },
      };
      const text = `${JSON.stringify(exportedSession, null, 2)}\n`;
      if (typeof window !== "undefined") {
        window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SESSION__ = exportedSession;
        window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SESSION_TEXT__ = text;
      }
      if (typeof document !== "undefined" && typeof URL !== "undefined" && typeof Blob !== "undefined") {
        const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
        const link = document.createElement("a");
        link.href = url;
        link.download = fileName;
        link.rel = "noopener";
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
      setSessionExport({
        status: "exported",
        fileName,
        schema: exportedSession.schema,
        error: "",
      });
      recordDebugEvent("export-session", {
        modelId: selectedDebugSession.snapshot.model.id,
        objectId: selectedDebugSession.snapshot.selection.objectId,
        selectionId: selectedDebugSession.snapshot.selection.selectionId,
        gaussianIndex: selectedDebugSession.snapshot.selection.gaussianIndex,
        fileName,
        source: "debug-panel",
      });
    } catch (error) {
      const message = error?.message ?? "debug session export failed";
      setSessionExport({
        status: "error",
        fileName,
        schema: selectedDebugSession.schema ?? "",
        error: message,
      });
      recordDebugEvent("export-session-error", {
        modelId: selectedDebugSession.snapshot.model.id,
        objectId: selectedDebugSession.snapshot.selection.objectId,
        selectionId: selectedDebugSession.snapshot.selection.selectionId,
        gaussianIndex: selectedDebugSession.snapshot.selection.gaussianIndex,
        fileName,
        source: "debug-panel",
      });
    }
  }, [recordDebugEvent, selectedDebugSession]);

  const importDebugSessionFile = useCallback(
    async (event) => {
      const file = event.target.files?.[0] ?? null;
      event.target.value = "";
      if (!file) return;
      setSessionImport({ status: "loading", fileName: file.name, schema: "", error: "" });
      try {
        const archive = validateDebugSessionArchive(JSON.parse(await file.text()), file.name);
        setDebugSessionArchive(archive);
        setSessionImport({
          status: "loaded",
          fileName: file.name,
          schema: archive.schema,
          error: "",
        });
        recordDebugEvent("import-session", {
          modelId: archive.snapshot.model.id,
          objectId: archive.snapshot.selection.objectId,
          selectionId: archive.snapshot.selection.selectionId,
          gaussianIndex: archive.snapshot.selection.gaussianIndex,
          fileName: file.name,
          source: "debug-panel",
        });
      } catch (error) {
        const message = error?.message ?? "debug session import failed";
        setDebugSessionArchive(null);
        setSessionImport({
          status: "error",
          fileName: file.name,
          schema: "",
          error: message,
        });
        recordDebugEvent("import-session-error", {
          fileName: file.name,
          source: "debug-panel",
        });
      }
    },
    [recordDebugEvent],
  );

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
      setDebugProbe(null);
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

  const selectObjectOverlayMode = useCallback((mode) => {
    const next = normalizeObjectOverlayMode(mode);
    setObjectOverlayMode(next);
    recordDebugEvent("object-overlay", { lens: debugLens, source: next });
    worldApi.current?.setObjectOverlayMode(next);
  }, [debugLens, recordDebugEvent]);

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

  const importModelArtifactBundleFiles = useCallback(
    async (event) => {
      const files = Array.from(event.target.files ?? []);
      event.target.value = "";
      if (!files.length) return;
      const fileName = files.map((file) => file.name).join(" + ");
      const parentId = "model-local-manifest";
      setModelImport({ status: "loading", modelId: parentId, fileName, error: "" });
      try {
        const { manifest, parentModel, children, qualityReport, objectStateBenchmark } =
          await localModelArtifactBundleModelsFromFiles(files);
        if (!children.length) {
          throw new Error("local model manifest has no Debug OS browser routes");
        }
        setModels((current) => {
          const parentState = initialModelStates([parentModel])[parentModel.id];
          return {
            ...current,
            ...initialModelStates(children),
            [parentModel.id]: {
              ...parentState,
              ...(current[parentModel.id] ?? {}),
              ...parentModel,
              status: "loaded",
              message: `${children.length} local debug routes`,
              modelArtifactManifest: manifest,
              qualityReport,
              objectStateBenchmark,
              delivery: {
                source: "local-model-artifact-manifest",
                loadRoute: "local-file",
                artifactPath: parentModel.manifestPath,
                childModelIds: children.map((child) => child.id),
              },
            },
          };
        });
        recordDebugEvent("import-model-manifest", {
          modelId: parentModel.id,
          selectionId: parentModel.id,
          fileName,
          childModelIds: children.map((child) => child.id),
          source: "local-file",
        });

        let selectedChild = null;
        for (const child of children) {
          if (child.loadMode === "trainable-artifact") {
            const startedAt = performance.now();
            const artifact = await loadTrainableArtifact(child);
            const imported = upsertTrainableArtifactModel(child, artifact, {
              startedAt,
              loadRoute: "local-manifest-file",
              artifactPath: child.trainableArtifactRoute ?? "local://trainable-kernel-artifact.json",
              message: "local manifest trainable artifact",
            });
            selectedChild = selectedChild ?? imported;
            continue;
          }
          if (child.loadMode === "ogc-chunked") {
            const startedAt = performance.now();
            const { artifact, decoded, index: loadedIndex, delivery } = await loadOgcModel(child);
            const imported = upsertDecodedOgcModel(child, decoded, loadedIndex, delivery, artifact, {
              startedAt,
              message: "local manifest ogc files",
            });
            selectedChild = selectedChild ?? imported;
          }
        }

        const importedIds = [parentModel.id, ...children.map((child) => child.id)];
        setDebugProbe(null);
        setHoveredTarget(null);
        setHiddenObjects((current) => {
          const next = new Set(
            [...current].filter((selectionId) =>
              importedIds.every((modelId) => !String(selectionId).startsWith(`${modelId}::`)),
            ),
          );
          worldApi.current?.setHiddenObjects(next);
          return next;
        });
        if (selectedChild) {
          setSelection({ modelId: selectedChild.id, objectId: null, selectionId: selectedChild.id });
          worldApi.current?.focusModel(selectedChild.id);
        }
        setModelImport({ status: "loaded", modelId: parentModel.id, fileName, error: "" });
      } catch (error) {
        setModelImport({
          status: "error",
          modelId: parentId,
          fileName,
          error: error?.message ?? "model manifest import failed",
        });
        recordDebugEvent("import-model-manifest-error", {
          modelId: parentId,
          selectionId: parentId,
          fileName,
          source: "local-file",
        });
      }
    },
    [recordDebugEvent, upsertDecodedOgcModel, upsertTrainableArtifactModel],
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
        const model = await ogcLocalArtifactModelFromFiles(files);
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
        if (model.loadMode === "model-artifact-manifest") {
          const startedAt = performance.now();
          patchModel(model.id, { status: "loading", message: "loading model manifest" });
          try {
            const { manifest, children, qualityReport, objectStateBenchmark } =
              await loadModelArtifactManifestModels(model);
            if (!children.length) {
              throw new Error("model artifact manifest has no Debug OS browser routes");
            }
            if (cancelled) return;
            setModels((current) => ({
              ...current,
              ...initialModelStates(children),
              [model.id]: {
                ...(current[model.id] ?? model),
                ...model,
                status: "loaded",
                message: `${children.length} debug routes`,
                modelArtifactManifest: manifest,
                qualityReport,
                objectStateBenchmark,
                loadMs: Math.round(performance.now() - startedAt),
                delivery: {
                  source: "model-artifact-manifest",
                  loadRoute: "fetch-json",
                  artifactPath: model.manifestPath,
                  childModelIds: children.map((child) => child.id),
                },
              },
            }));
            recordDebugEvent("manifest-load", {
              modelId: model.id,
              manifestPath: model.manifestPath,
              childModelIds: children.map((child) => child.id),
            });
            let selectedChild = null;
            for (const child of children) {
              if (cancelled) return;
              if (child.loadMode === "trainable-artifact") {
                const childStartedAt = performance.now();
                patchModel(child.id, { status: "loading", message: "loading manifest trainable artifact" });
                const artifact = await loadTrainableArtifact(child);
                const imported = upsertTrainableArtifactModel(child, artifact, {
                  startedAt: childStartedAt,
                  loadRoute: "model-manifest-json",
                  artifactPath: child.trainableArtifactPath,
                  message: "manifest trainable artifact",
                });
                selectedChild = selectedChild ?? imported;
                continue;
              }
              if (child.loadMode === "ogc-chunked") {
                const childStartedAt = performance.now();
                patchModel(child.id, { status: "loading", message: "loading manifest ogc chunks" });
                const { artifact, decoded, index, delivery } = await loadOgcModel(child);
                const imported = upsertDecodedOgcModel(child, decoded, index, delivery, artifact, {
                  startedAt: childStartedAt,
                  message: "manifest ogc chunks",
                });
                selectedChild = selectedChild ?? imported;
              }
            }
            if (selectedChild) {
              setSelection({
                modelId: selectedChild.id,
                objectId: null,
                selectionId: selectedChild.id,
              });
            }
          } catch (error) {
            if (cancelled) return;
            patchModel(model.id, {
              status: "error",
              message: error?.message ?? "model artifact manifest load failed",
            });
          }
          continue;
        }
        if (model.loadMode === "ogc-manifest") {
          const startedAt = performance.now();
          patchModel(model.id, { status: "loading", message: "loading ogc manifest" });
          try {
            const resolvedModel = await loadOgcManifestModel(model);
            const { artifact, decoded, index, delivery } = await loadOgcModel(resolvedModel);
            if (cancelled) return;
            upsertDecodedOgcModel(resolvedModel, decoded, index, delivery, artifact, {
              startedAt,
              message: "ogc manifest",
            });
          } catch (error) {
            if (cancelled) return;
            patchModel(model.id, {
              status: "error",
              message: error?.message ?? "ogc manifest load failed",
            });
          }
          continue;
        }
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
  }, [modelCatalog, patchModel, recordDebugEvent, upsertDecodedOgcModel, upsertTrainableArtifactModel, worldReady]);

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
      data-debug-snapshot-assignment-probe-status={selectedDebugSnapshot.assignment.probe?.status ?? ""}
      data-debug-snapshot-stability={selectedDebugSnapshot.stability.status}
      data-debug-snapshot-training-status={selectedDebugSnapshot.training?.status ?? ""}
      data-debug-snapshot-export-status={snapshotExport.status}
      data-debug-snapshot-export-file={snapshotExport.fileName}
      data-debug-snapshot-export-schema={snapshotExport.schema}
      data-debug-snapshot-export-error={snapshotExport.error}
      data-debug-session-schema={selectedDebugSession.schema}
      data-debug-session-model-count={selectedDebugSession.models.length}
      data-debug-session-event-count={selectedDebugSession.events.length}
      data-debug-session-export-status={sessionExport.status}
      data-debug-session-export-file={sessionExport.fileName}
      data-debug-session-export-schema={sessionExport.schema}
      data-debug-session-export-error={sessionExport.error}
      data-debug-session-import-status={sessionImport.status}
      data-debug-session-import-file={sessionImport.fileName}
      data-debug-session-import-schema={sessionImport.schema}
      data-debug-session-import-error={sessionImport.error}
      data-debug-session-archive-schema={debugSessionArchive?.schema ?? ""}
      data-debug-session-archive-model={debugSessionArchive?.snapshot?.model?.id ?? ""}
      data-debug-session-archive-quality={debugSessionArchive?.snapshot?.quality?.status ?? ""}
      data-debug-session-archive-event-count={debugSessionArchive?.events?.length ?? ""}
      data-debug-session-archive-model-count={debugSessionArchive?.models?.length ?? ""}
      data-debug-session-diff-status={debugSessionDiff?.status ?? ""}
      data-debug-session-diff-model-match={debugSessionDiff?.modelMatch ? "true" : debugSessionDiff ? "false" : ""}
      data-debug-session-diff-source-match={debugSessionDiff?.sourceMatch ? "true" : debugSessionDiff ? "false" : ""}
      data-debug-session-diff-quality-match={debugSessionDiff?.qualityMatch ? "true" : debugSessionDiff ? "false" : ""}
      data-debug-session-diff-training-match={debugSessionDiff?.trainingMatch ? "true" : debugSessionDiff ? "false" : ""}
      data-debug-session-diff-slot-delta={debugSessionDiff?.slotDelta ?? ""}
      data-debug-session-diff-entropy-delta={debugSessionDiff?.entropyDelta ?? ""}
      data-debug-session-diff-event-delta={debugSessionDiff?.eventDelta ?? ""}
      data-debug-session-diff-field-count={debugSessionDiff?.changedFields?.length ?? ""}
      data-debug-session-diff-fields={debugSessionDiff?.changedFieldNames ?? ""}
      data-debug-event-count={debugEvents.length}
      data-debug-event-last={debugEvents[0]?.type ?? ""}
      data-debug-event-schema={debugEvents[0]?.schema ?? ""}
      data-assignment-debug={debugMode ? "enabled" : "disabled"}
      data-debug-lens={debugMode ? debugLens : "appearance"}
      data-assignment-probe-status={selectedAssignmentProbe.status}
      data-assignment-probe-top-slot={selectedAssignmentProbe.topSlot ?? ""}
      data-assignment-probe-top-probability={selectedAssignmentProbe.topProbability ?? ""}
      data-assignment-probe-second-probability={selectedAssignmentProbe.secondProbability ?? ""}
      data-assignment-probe-margin={selectedAssignmentProbe.margin ?? ""}
      data-assignment-probe-ambiguous={selectedAssignmentProbe.ambiguous ? "true" : "false"}
      data-assignment-probe-collapse-risk={selectedAssignmentProbe.collapseRisk ? "true" : "false"}
      data-object-continuity-status={selectedContinuity.status}
      data-object-continuity-spatial-compactness={selectedContinuity.spatialCompactness ?? ""}
      data-object-continuity-bbox-diagonal={selectedContinuity.bboxDiagonal ?? ""}
      data-object-continuity-density={selectedContinuity.gaussianDensity ?? ""}
      data-object-continuity-centroid-contained={selectedContinuity.centroidContained ? "true" : "false"}
      data-object-temporal-status={selectedTemporal.status}
      data-object-temporal-drift={selectedTemporal.temporalDrift ?? ""}
      data-object-assignment-jitter={selectedTemporal.assignmentJitter ?? ""}
      data-object-bbox-stability={selectedTemporal.bboxStability ?? ""}
      data-object-temporal-stable={selectedTemporal.stable ? "true" : "false"}
      data-object-explainability-status={selectedExplainability.status}
      data-object-explainable={selectedExplainability.explainable ? "true" : "false"}
      data-object-explainability-score={selectedExplainability.score ?? ""}
      data-object-explainability-reasons={selectedExplainability.reasonNames}
      data-object-overlay-mode={selectedObjectOverlayMode}
      data-object-overlay-bbox-visible={debugMode && objectOverlayShows(selectedObjectOverlayMode, "bbox") ? "true" : "false"}
      data-object-overlay-centroid-visible={debugMode && objectOverlayShows(selectedObjectOverlayMode, "centroid") ? "true" : "false"}
      data-selected-gaussian={debugProbe?.gaussianIndex ?? ""}
      data-hovered-target={hoveredTarget?.selectionId ?? ""}
      data-hovered-model={hoveredTarget?.modelId ?? ""}
      data-hovered-object={hoveredTarget?.objectId ?? ""}
      data-hovered-gaussians={hoveredTarget?.gaussianCount ?? ""}
      data-hover-highlight={hoveredTarget?.selectionId ? "enabled" : "disabled"}
      data-hover-highlight-object={hoveredTarget?.selectionId ?? ""}
      data-hover-highlight-gaussians={hoveredTarget?.gaussianCount ?? ""}
      data-hover-assignment-source={hoveredTarget?.assignmentSource ?? ""}
      data-hover-assignment-slots={hoveredAssignmentProbe.slotCount}
      data-hover-assignment-confidence={hoveredAssignmentProbe.confidence ?? ""}
      data-hover-assignment-entropy={hoveredAssignmentProbe.entropy ?? ""}
      data-hover-assignment-probe-status={hoveredAssignmentProbe.status}
      data-hover-assignment-probe-margin={hoveredAssignmentProbe.margin ?? ""}
      data-hover-assignment-top-slot={hoveredAssignmentProbe.topSlot ?? ""}
      data-hover-assignment-ambiguous={hoveredAssignmentProbe.ambiguous ? "true" : "false"}
      data-hover-assignment-collapse-risk={hoveredAssignmentProbe.collapseRisk ? "true" : "false"}
      data-hover-continuity-status={hoveredContinuity.status}
      data-hover-continuity-spatial-compactness={hoveredContinuity.spatialCompactness ?? ""}
      data-hover-continuity-bbox-diagonal={hoveredContinuity.bboxDiagonal ?? ""}
      data-hover-continuity-centroid-contained={hoveredContinuity.centroidContained ? "true" : "false"}
      data-hover-temporal-status={hoveredTemporal.status}
      data-hover-temporal-drift={hoveredTemporal.temporalDrift ?? ""}
      data-hover-assignment-jitter={hoveredTemporal.assignmentJitter ?? ""}
      data-hover-bbox-stability={hoveredTemporal.bboxStability ?? ""}
      data-hover-temporal-stable={hoveredTemporal.stable ? "true" : "false"}
      data-hover-explainability-status={hoveredExplainability.status}
      data-hover-explainable={hoveredExplainability.explainable ? "true" : "false"}
      data-hover-explainability-score={hoveredExplainability.score ?? ""}
      data-hover-explainability-reasons={hoveredExplainability.reasonNames}
      data-hidden-objects={hiddenCount}
      data-object-visibility-contract="enabled"
      data-visible-objects={objectVisibility.visibleObjectCount}
      data-visible-gaussians={objectVisibility.visibleGaussianCount}
      data-hidden-gaussians={objectVisibility.hiddenGaussianCount}
      data-hidden-object-ids={objectVisibility.hiddenSelectionIds.join(",")}
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
      data-quality-report-status={selectedQualityReport?.status ?? ""}
      data-quality-report-schema={selectedQualityReport?.schema ?? ""}
      data-quality-report-assignment-entropy={selectedQualityReport?.assignmentEntropy ?? ""}
      data-quality-report-object-purity={selectedQualityReport?.objectPurity ?? ""}
      data-quality-report-temporal-drift={selectedQualityReport?.temporalDrift ?? ""}
      data-quality-report-assignment-jitter={selectedQualityReport?.assignmentJitter ?? ""}
      data-quality-report-gate-count={selectedQualityReport?.gateCount ?? ""}
      data-object-state-benchmark-status={selectedObjectStateBenchmark?.status ?? ""}
      data-object-state-benchmark-schema={selectedObjectStateBenchmark?.schema ?? ""}
      data-object-state-benchmark-case-count={selectedObjectStateBenchmark?.caseCount ?? ""}
      data-object-state-benchmark-warn-count={selectedObjectStateBenchmark?.warnCount ?? ""}
      data-object-state-benchmark-observed-warn-count={selectedObjectStateBenchmark?.observedWarnCount ?? ""}
      data-object-state-benchmark-failure-mode-count={selectedObjectStateBenchmark?.failureModeCount ?? ""}
      data-object-state-benchmark-active-case={selectedBenchmarkCase?.name ?? ""}
      data-object-state-benchmark-active-status={selectedBenchmarkCase?.status ?? ""}
      data-object-state-benchmark-active-observed-status={selectedBenchmarkCase?.observedStatus ?? ""}
      data-object-state-benchmark-active-failure-modes={selectedBenchmarkCase?.failureModeNames ?? ""}
      data-object-state-benchmark-active-diagnostics={selectedBenchmarkCase?.diagnosticNames ?? ""}
      data-object-state-benchmark-active-assignment-confidence={selectedBenchmarkCase?.assignmentConfidence ?? ""}
      data-object-state-benchmark-active-entropy={selectedBenchmarkCase?.meanEntropy ?? ""}
      data-object-state-benchmark-active-purity={selectedBenchmarkCase?.objectPurity ?? ""}
      data-object-state-benchmark-active-temporal-drift={selectedBenchmarkCase?.meanTemporalDrift ?? ""}
      data-object-state-benchmark-active-dynamic-proposals={selectedBenchmarkCase?.dynamicProposalCount ?? ""}
      data-trainable-import-status={artifactImport.status}
      data-trainable-import-model={artifactImport.modelId}
      data-trainable-import-file={artifactImport.fileName}
      data-trainable-import-error={artifactImport.error}
      data-model-manifest-import-status={modelImport.status}
      data-model-manifest-import-model={modelImport.modelId}
      data-model-manifest-import-file={modelImport.fileName}
      data-model-manifest-import-error={modelImport.error}
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
        objectOverlayMode={selectedObjectOverlayMode}
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
          <input
            ref={modelBundleInputRef}
            className="fileInputHidden"
            type="file"
            accept="application/json,.json,.ogc"
            multiple
            data-model-artifact-file-input="true"
            onChange={importModelArtifactBundleFiles}
          />
          <input
            ref={debugSessionInputRef}
            className="fileInputHidden"
            type="file"
            accept="application/json,.json"
            data-debug-session-file-input="true"
            onChange={importDebugSessionFile}
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
            className={`glassButton ${modelImport.status === "loaded" ? "active" : ""}`}
            type="button"
            data-model-artifact-import-button="true"
            data-import-status={modelImport.status}
            onClick={() => modelBundleInputRef.current?.click()}
          >
            {modelImport.status === "loading"
              ? "模型中"
              : modelImport.status === "error"
                ? "模型失败"
                : "导入模型"}
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
        hoverAssignmentProbe={hoveredAssignmentProbe}
        objectContinuity={selectedContinuity}
        hoverContinuity={hoveredContinuity}
        objectTemporal={selectedTemporal}
        hoverTemporal={hoveredTemporal}
        objectExplainability={selectedExplainability}
        hoverExplainability={hoveredExplainability}
        debugProbe={debugProbe}
        assignmentProbe={selectedAssignmentProbe}
        debugMode={debugMode}
        debugLens={debugLens}
        objectOverlayMode={selectedObjectOverlayMode}
        debugEvents={debugEvents}
        debugSnapshot={selectedDebugSnapshot}
        snapshotExport={snapshotExport}
        sessionExport={sessionExport}
        sessionImport={sessionImport}
        debugSessionArchive={debugSessionArchive}
        debugSessionDiff={debugSessionDiff}
        hiddenObjects={hiddenObjects}
        objectVisibility={objectVisibility}
        stability={selectedStability}
        qualityReport={selectedQualityReport}
        objectStateBenchmark={selectedObjectStateBenchmark}
        benchmarkCase={selectedBenchmarkCase}
        onSelectBenchmarkCase={setBenchmarkCaseName}
        onToggleObjectVisibility={toggleObjectVisibility}
        onSelectDebugLens={selectDebugLens}
        onSelectObjectOverlayMode={selectObjectOverlayMode}
        onSelectTrainableFrame={selectTrainableFrame}
        onSelectOgcLod={selectOgcLod}
        onSelectOgcChunks={selectOgcChunks}
        onExportDebugSnapshot={exportDebugSnapshot}
        onExportDebugSession={exportDebugSession}
        onImportDebugSession={() => debugSessionInputRef.current?.click()}
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

async function loadModelArtifactManifestModels(model) {
  const manifestPath = model.manifestPath;
  if (!manifestPath) {
    throw new Error("missing model artifact manifest path");
  }
  const response = await fetch(manifestPath);
  if (!response.ok) throw new Error(`model artifact manifest HTTP ${response.status}`);
  const manifest = await response.json();
  if (manifest?.schema !== MODEL_ARTIFACT_MANIFEST_SCHEMA) {
    throw new Error("unsupported model artifact manifest schema");
  }
  const qualityReportArtifact = browserReadyArtifact({ modelArtifactManifest: manifest }, "quality_report");
  const qualityReport = qualityReportArtifact
    ? await loadQualityReportArtifact(qualityReportArtifact, manifestPath)
    : null;
  const objectStateBenchmarkArtifact = browserReadyArtifact({ modelArtifactManifest: manifest }, "object_state_benchmark");
  const objectStateBenchmark = objectStateBenchmarkArtifact
    ? await loadObjectStateBenchmarkArtifact(objectStateBenchmarkArtifact, manifestPath)
    : null;
  const children = [];
  if (browserReadyArtifact({ modelArtifactManifest: manifest }, "trainable_kernel")) {
    children.push(trainableManifestModelFromManifest(model, manifest, manifestPath));
  }
  if (browserReadyArtifact({ modelArtifactManifest: manifest }, "compressed_chunked")) {
    children.push(ogcUrlManifestModelFromManifest(ogcManifestChildModel(model), manifest, manifestPath));
  }
  return {
    manifest,
    children: children.map((child) => attachDebugEvidence(child, { qualityReport, objectStateBenchmark })),
    qualityReport,
    objectStateBenchmark,
  };
}

async function loadOgcManifestModel(model) {
  const manifestPath = model.ogc?.manifestPath;
  if (!manifestPath) {
    throw new Error("missing OGC model artifact manifest path");
  }
  const response = await fetch(manifestPath);
  if (!response.ok) throw new Error(`OGC manifest HTTP ${response.status}`);
  const manifest = await response.json();
  return ogcUrlManifestModelFromManifest(model, manifest, manifestPath);
}

async function loadTrainableArtifact(model) {
  if (model.trainableArtifactPath) {
    const response = await fetch(model.trainableArtifactPath);
    if (!response.ok) throw new Error(`trainable artifact HTTP ${response.status}`);
    return validateTrainableArtifact(await response.json());
  }
  return validateTrainableArtifact(model.trainableArtifact);
}

async function loadQualityReportArtifact(artifact, manifestPath) {
  const reportPath = resolveSameOriginManifestRoute(artifact.reportPath ?? artifact.path, manifestPath);
  const response = await fetch(reportPath);
  if (!response.ok) throw new Error(`quality report HTTP ${response.status}`);
  return validateQualityReport(await response.json(), reportPath);
}

async function loadObjectStateBenchmarkArtifact(artifact, manifestPath) {
  const reportPath = resolveSameOriginManifestRoute(artifact.reportPath ?? artifact.path, manifestPath);
  const response = await fetch(reportPath);
  if (!response.ok) throw new Error(`object state benchmark HTTP ${response.status}`);
  return validateObjectStateBenchmark(await response.json(), reportPath);
}

function attachQualityReport(model, qualityReport) {
  if (!qualityReport) return model;
  return {
    ...model,
    qualityReport,
  };
}

function attachObjectStateBenchmark(model, objectStateBenchmark) {
  if (!objectStateBenchmark) return model;
  return {
    ...model,
    objectStateBenchmark,
  };
}

function attachDebugEvidence(model, { qualityReport, objectStateBenchmark } = {}) {
  return attachObjectStateBenchmark(attachQualityReport(model, qualityReport), objectStateBenchmark);
}

function trainableManifestModelFromManifest(model, manifest, manifestPath) {
  const artifact = browserReadyArtifact({ modelArtifactManifest: manifest }, "trainable_kernel");
  if (!artifact) {
    throw new Error("model manifest missing browser-ready trainable_kernel artifact");
  }
  const artifactPath = resolveSameOriginManifestRoute(artifact.artifactPath ?? artifact.path, manifestPath);
  const resolvedArtifact = {
    ...artifact,
    path: artifactPath,
    artifactPath,
  };
  return {
    id: "model-manifest-trainable-artifact",
    name: manifest.name ? `${manifest.name} trainable` : "Manifest trainable artifact",
    label: "Manifest Train",
    loadMode: "trainable-artifact",
    kind: "trainable-kernel-model-artifact",
    stage: "algorithm-handoff-artifact",
    objectCount: ogcPositiveInteger(manifest.counts?.objects ?? artifact.object_count) ?? model.objectCount ?? 0,
    galleryPosition: [-3.95, 0, 5.08],
    accent: "#f7df63",
    displayScale: model.displayScale ?? 1.92,
    pointSize: model.pointSize ?? 0.082,
    maxDisplayPoints: 256,
    license: manifest.license ?? model.license ?? "url-debug-artifact",
    trainableArtifactPath: artifactPath,
    compression: {
      layout: "trainable-kernel-artifact-json",
      status: "model-manifest-debug-artifact",
      chunkRoot: "/models/model-manifest-trainable/objects/",
    },
    modelArtifactManifest: replaceManifestArtifact(manifest, artifact, resolvedArtifact),
  };
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

function validateQualityReport(report, path = "") {
  if (report?.schema !== "objgauss-object-state-quality-report-v1") {
    throw new Error("unsupported ObjectState quality report schema");
  }
  if (typeof report.metrics !== "object" || report.metrics === null) {
    throw new Error("quality report missing metrics");
  }
  return {
    ...report,
    path,
  };
}

function validateObjectStateBenchmark(report, path = "") {
  if (report?.schema !== OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA) {
    throw new Error("unsupported ObjectState benchmark schema");
  }
  if (!Array.isArray(report.cases) || !report.cases.length) {
    throw new Error("ObjectState benchmark missing cases");
  }
  if (typeof report.aggregate !== "object" || report.aggregate === null) {
    throw new Error("ObjectState benchmark missing aggregate");
  }
  return {
    ...report,
    path,
  };
}

function ogcManifestChildModel(model) {
  return {
    id: "model-manifest-ogc-artifact",
    name: "Manifest OGC artifact",
    label: "Manifest OGC",
    loadMode: "ogc-manifest",
    kind: "compressed-chunked-ogc-manifest",
    stage: "algorithm-handoff-artifact",
    objectCount: model.objectCount ?? 0,
    galleryPosition: [3.85, 0, 5.02],
    accent: "#58f2c2",
    displayScale: model.displayScale ?? 1.72,
    pointSize: 0.07,
    maxDisplayPoints: 2400,
    license: model.license,
    ogc: {
      ...(model.ogc ?? {}),
      manifestPath: model.manifestPath,
    },
    compression: {
      layout: "object-aware-quantized-ogc-manifest",
      status: "model-manifest-debug-artifact",
      chunkRoot: "/models/model-manifest-ogc/objects/",
    },
  };
}

function ogcUrlManifestModelFromManifest(model, manifest, manifestPath) {
  if (manifest?.schema !== MODEL_ARTIFACT_MANIFEST_SCHEMA) {
    throw new Error("unsupported model artifact manifest schema");
  }
  const artifact = browserReadyArtifact({ modelArtifactManifest: manifest }, "compressed_chunked");
  if (!artifact) {
    throw new Error("OGC manifest missing browser-ready compressed_chunked artifact");
  }
  const payloadPath = resolveSameOriginManifestRoute(artifact.payloadPath ?? artifact.path, manifestPath);
  const indexPath = resolveSameOriginManifestRoute(artifact.indexPath ?? artifact.chunk_index?.path, manifestPath);
  const resolvedArtifact = {
    ...artifact,
    path: payloadPath,
    payloadPath,
    indexPath,
    chunk_index: {
      ...(artifact.chunk_index ?? {}),
      path: indexPath,
    },
  };
  return {
    ...model,
    name: manifest.name ?? model.name,
    label: model.label,
    loadMode: "ogc-chunked",
    kind: "compressed-chunked-ogc",
    objectCount: ogcPositiveInteger(manifest.counts?.objects ?? artifact.object_count) ?? model.objectCount,
    license: manifest.license ?? model.license ?? "url-debug-artifact",
    ogc: {
      ...(model.ogc ?? {}),
      indexPath,
      payloadPath,
    },
    modelArtifactManifest: replaceManifestArtifact(manifest, artifact, resolvedArtifact),
  };
}

function replaceManifestArtifact(manifest, sourceArtifact, replacementArtifact) {
  let replaced = false;
  const artifacts = Array.isArray(manifest?.artifacts)
    ? manifest.artifacts.map((entry) => {
        if (replaced || entry !== sourceArtifact) return entry;
        replaced = true;
        return replacementArtifact;
      })
    : [replacementArtifact];
  return {
    ...manifest,
    artifacts: replaced ? artifacts : [replacementArtifact, ...artifacts],
  };
}

function resolveSameOriginManifestRoute(route, manifestPath) {
  const value = String(route ?? "");
  if (!value || isInlineRoute(value) || value.startsWith("local://")) {
    throw new Error("OGC manifest artifact route must be same-origin fetchable");
  }
  const origin = typeof window === "undefined" ? "http://127.0.0.1" : window.location.origin;
  const resolved = new URL(value, new URL(manifestPath, origin));
  if (resolved.origin !== origin || resolved.search || resolved.hash) {
    throw new Error("OGC manifest artifact route must stay on the same origin");
  }
  return resolved.pathname;
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

async function localModelArtifactBundleModelsFromFiles(files) {
  const entries = Array.from(files ?? []);
  const jsonEntries = await Promise.all(
    entries
      .filter((file) => /\.json$/i.test(file.name))
      .map(async (file) => ({ file, json: JSON.parse(await file.text()) })),
  );
  const manifestEntry = jsonEntries.find((entry) => {
    if (entry.json?.schema !== MODEL_ARTIFACT_MANIFEST_SCHEMA) return false;
    return (
      browserReadyArtifact({ modelArtifactManifest: entry.json }, "trainable_kernel") ||
      browserReadyArtifact({ modelArtifactManifest: entry.json }, "compressed_chunked")
    );
  });
  if (!manifestEntry) {
    throw new Error("select one model artifact manifest with trainable_kernel or compressed_chunked artifacts");
  }

  const manifest = manifestEntry.json;
  const parentModel = localModelArtifactParentModel(manifest, manifestEntry.file.name);
  const children = [];
  const qualityReportArtifact = browserReadyArtifact({ modelArtifactManifest: manifest }, "quality_report");
  const qualityReportEntry = qualityReportArtifact
    ? findQualityReportEntry(jsonEntries, qualityReportArtifact.reportPath ?? qualityReportArtifact.path)
    : null;
  const qualityReport = qualityReportEntry
    ? validateQualityReport(qualityReportEntry.json, localFileRoute(qualityReportEntry.file.name))
    : null;
  const objectStateBenchmarkArtifact = browserReadyArtifact({ modelArtifactManifest: manifest }, "object_state_benchmark");
  const objectStateBenchmarkEntry = objectStateBenchmarkArtifact
    ? findObjectStateBenchmarkEntry(jsonEntries, objectStateBenchmarkArtifact.reportPath ?? objectStateBenchmarkArtifact.path)
    : null;
  const objectStateBenchmark = objectStateBenchmarkEntry
    ? validateObjectStateBenchmark(objectStateBenchmarkEntry.json, localFileRoute(objectStateBenchmarkEntry.file.name))
    : null;
  const trainableArtifact = browserReadyArtifact({ modelArtifactManifest: manifest }, "trainable_kernel");
  if (trainableArtifact) {
    const trainableEntry = findTrainableArtifactEntry(
      jsonEntries,
      trainableArtifact.artifactPath ?? trainableArtifact.path,
    );
    children.push(
      attachDebugEvidence(
        trainableLocalManifestModelFromManifest({
          manifest,
          artifact: trainableArtifact,
          trainableArtifact: trainableEntry.json,
          artifactFileName: trainableEntry.file.name,
        }),
        { qualityReport, objectStateBenchmark },
      ),
    );
  }

  const ogcArtifact = browserReadyArtifact({ modelArtifactManifest: manifest }, "compressed_chunked");
  if (ogcArtifact) {
    const indexEntry = findOgcIndexEntry(jsonEntries, ogcArtifact.indexPath ?? ogcArtifact.chunk_index?.path);
    const payloadFile = findOgcPayloadFile(entries, ogcArtifact.payloadPath ?? ogcArtifact.path);
    const ogcModel = ogcLocalArtifactModelFromManifest({
      manifest,
      artifact: ogcArtifact,
      index: indexEntry.json,
      payloadBuffer: await payloadFile.arrayBuffer(),
      indexFileName: indexEntry.file.name,
      payloadFileName: payloadFile.name,
    });
    children.push(
      attachDebugEvidence(
        {
          ...ogcModel,
          id: "model-local-manifest-ogc-artifact",
          name: manifest.name ? `Local ${manifest.name} OGC` : "Local manifest OGC artifact",
          label: "Local OGC",
          stage: "local-model-manifest-debug-artifact",
          galleryPosition: [3.85, 0, 5.02],
          compression: {
            ...(ogcModel.compression ?? {}),
            status: "local-model-manifest-debug-artifact",
          },
        },
        { qualityReport, objectStateBenchmark },
      ),
    );
  }
  return {
    manifest,
    parentModel: attachDebugEvidence(parentModel, { qualityReport, objectStateBenchmark }),
    children,
    qualityReport,
    objectStateBenchmark,
  };
}

function localModelArtifactParentModel(manifest, fileName) {
  return {
    id: "model-local-manifest",
    name: manifest?.name ? `Local ${manifest.name}` : "Local model manifest",
    label: "Local Manifest",
    loadMode: "local-model-artifact-manifest",
    kind: "algorithm-model-artifact-manifest",
    stage: "local-model-manifest-debug-artifact",
    objectCount: ogcPositiveInteger(manifest?.counts?.objects) ?? 0,
    galleryPosition: [-0.9, 0, 5.18],
    accent: "#d7f45a",
    displayScale: 1.62,
    pointSize: 0.068,
    maxDisplayPoints: 2400,
    license: manifest?.license ?? "local-file-debug",
    manifestPath: localFileRoute(fileName),
    compression: {
      layout: "model-artifact-manifest-handoff",
      status: "local-debug-artifact",
      chunkRoot: "/models/local-model-artifact-manifest/objects/",
    },
  };
}

function trainableLocalManifestModelFromManifest({
  manifest,
  artifact,
  trainableArtifact,
  artifactFileName,
}) {
  const artifactPath = localFileRoute(artifactFileName);
  const localizedArtifact = {
    ...artifact,
    path: artifactPath,
    artifactPath,
    byte_size: artifact.byte_size,
  };
  return {
    id: "model-local-manifest-trainable-artifact",
    name: manifest.name ? `Local ${manifest.name} trainable` : "Local manifest trainable artifact",
    label: "Local Train",
    loadMode: "trainable-artifact",
    kind: "trainable-kernel-model-artifact",
    stage: "local-model-manifest-debug-artifact",
    objectCount: ogcPositiveInteger(manifest.counts?.objects ?? artifact.object_count) ?? 0,
    galleryPosition: [-3.95, 0, 5.08],
    accent: "#f7df63",
    displayScale: 1.92,
    pointSize: 0.082,
    maxDisplayPoints: 256,
    license: manifest.license ?? "local-file-debug",
    trainableArtifact: validateTrainableArtifact(trainableArtifact),
    trainableArtifactRoute: artifactPath,
    compression: {
      layout: "trainable-kernel-artifact-json",
      status: "local-model-manifest-debug-artifact",
      chunkRoot: "/models/local-model-manifest-trainable/objects/",
    },
    modelArtifactManifest: localizeTrainableManifest(manifest, artifact, localizedArtifact, {
      artifactPath,
      artifactFileName,
    }),
  };
}

function localizeTrainableManifest(manifest, sourceArtifact, localizedArtifact, files) {
  const replaced = replaceManifestArtifact(manifest, sourceArtifact, localizedArtifact);
  return {
    ...replaced,
    source: {
      ...(replaced?.source ?? {}),
      local_trainable_import: {
        type: "local_trainable_kernel_artifact",
        artifact_path: files.artifactPath,
      },
    },
    created_from: {
      ...(replaced?.created_from ?? {}),
      local_trainable_file: files.artifactFileName,
    },
  };
}

async function ogcLocalArtifactModelFromFiles(files) {
  const entries = Array.from(files ?? []);
  const jsonEntries = await Promise.all(
    entries
      .filter((file) => /\.json$/i.test(file.name))
      .map(async (file) => ({ file, json: JSON.parse(await file.text()) })),
  );
  const manifestEntry = jsonEntries.find((entry) => entry.json?.schema === MODEL_ARTIFACT_MANIFEST_SCHEMA);
  if (manifestEntry) {
    const artifact = browserReadyArtifact({ modelArtifactManifest: manifestEntry.json }, "compressed_chunked");
    if (!artifact) {
      throw new Error("local model manifest missing browser-ready compressed_chunked artifact");
    }
    const indexEntry = findOgcIndexEntry(jsonEntries, artifact.indexPath ?? artifact.chunk_index?.path);
    const payloadFile = findOgcPayloadFile(entries, artifact.payloadPath ?? artifact.path);
    return ogcLocalArtifactModelFromManifest({
      manifest: manifestEntry.json,
      artifact,
      index: indexEntry.json,
      payloadBuffer: await payloadFile.arrayBuffer(),
      indexFileName: indexEntry.file.name,
      payloadFileName: payloadFile.name,
    });
  }

  const indexEntry = findOgcIndexEntry(jsonEntries);
  const payloadFile = findOgcPayloadFile(entries);
  return ogcLocalArtifactModel({
    index: indexEntry.json,
    payloadBuffer: await payloadFile.arrayBuffer(),
    indexFileName: indexEntry.file.name,
    payloadFileName: payloadFile.name,
  });
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

function ogcLocalArtifactModelFromManifest({
  manifest,
  artifact,
  index,
  payloadBuffer,
  indexFileName,
  payloadFileName,
}) {
  const payloadPath = localFileRoute(payloadFileName);
  const indexPath = localFileRoute(indexFileName);
  const localizedArtifact = {
    ...artifact,
    path: payloadPath,
    payloadPath,
    indexPath,
    byte_size: payloadBuffer.byteLength,
    gaussian_count: ogcPositiveInteger(artifact.gaussian_count ?? index?.gaussian_count),
    object_count: ogcPositiveInteger(artifact.object_count ?? index?.object_count),
    sha256: artifact.sha256 ?? index?.payload?.sha256,
    chunk_index: {
      ...(artifact.chunk_index ?? {}),
      schema: artifact.chunk_index?.schema ?? index?.schema,
      path: indexPath,
      chunk_count: artifact.chunk_index?.chunk_count ?? (Array.isArray(index?.chunks) ? index.chunks.length : undefined),
      sort_key: artifact.chunk_index?.sort_key ?? index?.sort_key,
      chunk_size_target: artifact.chunk_index?.chunk_size_target ?? index?.chunk_size_target,
    },
    compression: artifact.compression ?? index?.compression,
    lod: artifact.lod ?? index?.lod,
    object_id_coverage: artifact.object_id_coverage ?? index?.object_id_coverage,
    inlineIndex: index,
    payloadBuffer,
  };
  return {
    id: "ogc-local-artifact",
    name: manifest.name ? `Local ${manifest.name}` : "Local OGC artifact",
    label: "Local OGC",
    loadMode: "ogc-chunked",
    kind: "compressed-chunked-ogc",
    stage: "local-manifest-debug-artifact",
    objectCount: ogcPositiveInteger(index?.object_count ?? manifest.counts?.objects) ?? 0,
    galleryPosition: [2.85, 0, 4.56],
    accent: "#5df2df",
    displayScale: 1.88,
    pointSize: 0.058,
    maxDisplayPoints: 1200,
    license: manifest.license ?? "local-file-debug",
    ogc: { lodLevel: 0 },
    compression: {
      layout: localizedArtifact.compression?.layout ?? "object-aware-chunked-local-quantized",
      status: "local-manifest-debug-artifact",
      chunkRoot: indexPath,
    },
    modelArtifactManifest: localizeOgcManifest(manifest, artifact, localizedArtifact, {
      indexPath,
      payloadPath,
      indexFileName,
      payloadFileName,
    }),
  };
}

function localizeOgcManifest(manifest, sourceArtifact, localizedArtifact, files) {
  let replaced = false;
  const artifacts = Array.isArray(manifest?.artifacts)
    ? manifest.artifacts.map((entry) => {
        if (replaced || entry !== sourceArtifact) return entry;
        replaced = true;
        return localizedArtifact;
      })
    : [localizedArtifact];
  return {
    ...manifest,
    source: {
      ...(manifest?.source ?? {}),
      local_import: {
        type: "local_model_artifact_manifest_package",
        index_path: files.indexPath,
        payload_path: files.payloadPath,
      },
    },
    artifacts: replaced ? artifacts : [localizedArtifact, ...artifacts],
    created_from: {
      ...(manifest?.created_from ?? {}),
      local_index_file: files.indexFileName,
      local_payload_file: files.payloadFileName,
    },
  };
}

function findOgcIndexEntry(jsonEntries, expectedRoute = "") {
  const expectedName = fileNameFromRoute(expectedRoute);
  const candidates = jsonEntries.filter((entry) => isOgcChunkIndex(entry.json));
  const matched = expectedName
    ? candidates.find((entry) => entry.file.name === expectedName)
    : candidates.find((entry) => /\.index\.json$/i.test(entry.file.name)) ?? candidates[0];
  if (!matched) {
    throw new Error(expectedName
      ? `select OGC chunk index file ${expectedName}`
      : "select one OGC .index.json file");
  }
  return matched;
}

function findTrainableArtifactEntry(jsonEntries, expectedRoute = "") {
  const expectedName = fileNameFromRoute(expectedRoute);
  const candidates = jsonEntries.filter(
    (entry) => entry.json?.schema === "objgauss-trainable-kernel-model-artifact-v1",
  );
  const matched = expectedName
    ? candidates.find((entry) => entry.file.name === expectedName) ?? candidates[0]
    : candidates[0];
  if (!matched) {
    throw new Error(expectedName
      ? `select trainable kernel artifact file ${expectedName}`
      : "select one trainable kernel artifact JSON file");
  }
  return matched;
}

function findQualityReportEntry(jsonEntries, expectedRoute = "") {
  const expectedName = fileNameFromRoute(expectedRoute);
  const candidates = jsonEntries.filter(
    (entry) => entry.json?.schema === "objgauss-object-state-quality-report-v1",
  );
  const matched = expectedName
    ? candidates.find((entry) => entry.file.name === expectedName) ?? candidates[0]
    : candidates[0];
  if (!matched) {
    throw new Error(expectedName
      ? `select ObjectState quality report file ${expectedName}`
      : "select one ObjectState quality report JSON file");
  }
  return matched;
}

function findObjectStateBenchmarkEntry(jsonEntries, expectedRoute = "") {
  const expectedName = fileNameFromRoute(expectedRoute);
  const candidates = jsonEntries.filter(
    (entry) => entry.json?.schema === OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA,
  );
  const matched = expectedName
    ? candidates.find((entry) => entry.file.name === expectedName) ?? candidates[0]
    : candidates[0];
  if (!matched) {
    throw new Error(expectedName
      ? `select ObjectState benchmark report file ${expectedName}`
      : "select one ObjectState benchmark report JSON file");
  }
  return matched;
}

function findOgcPayloadFile(files, expectedRoute = "") {
  const expectedName = fileNameFromRoute(expectedRoute);
  const payloadFiles = Array.from(files ?? []).filter((file) => /\.ogc$/i.test(file.name));
  const matched = expectedName
    ? payloadFiles.find((file) => file.name === expectedName)
    : payloadFiles[0];
  if (!matched) {
    throw new Error(expectedName ? `select OGC payload file ${expectedName}` : "select one .ogc file");
  }
  return matched;
}

function isOgcChunkIndex(value) {
  return value?.schema === OGC_CHUNK_INDEX_SCHEMA || (
    Array.isArray(value?.chunks) &&
    typeof value?.payload === "object" &&
    value.payload !== null
  );
}

function fileNameFromRoute(route) {
  const clean = String(route ?? "").split("?")[0].split("#")[0];
  return clean.split(/[\\/]/).filter(Boolean).pop() ?? "";
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
  objectOverlayMode,
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
  const overlayModeRef = useRef(normalizeObjectOverlayMode(objectOverlayMode));
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
    const next = normalizeObjectOverlayMode(objectOverlayMode);
    overlayModeRef.current = next;
    apiRef.current?.setObjectOverlayMode(next);
  }, [objectOverlayMode]);

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
    let selectedGaussianProbe = null;
    let selectedGaussianProbeSelectionId = null;
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
      const selectedAssignmentProbe = assignmentProbeSummary(
        selectedGaussianProbe?.assignment ?? selectedObject?.userData.objectState?.assignment ?? [],
        selectedGaussianProbe,
      );
      const selectedContinuity = objectContinuitySummary(selectedObject ? objectTarget(selectedObject) : null);
      const selectedTemporal = objectTemporalSummary(selectedObject ? objectTarget(selectedObject) : null);
      const selectedExplainability = objectExplainabilitySummary({
        object: selectedObject ? objectTarget(selectedObject) : null,
        assignmentProbe: selectedAssignmentProbe,
        continuity: selectedContinuity,
        temporal: selectedTemporal,
      });
      const hoveredTarget = objectTarget(hoveredObject);
      const hoveredAssignmentProbe = assignmentProbeSummary(hoveredTarget?.assignment ?? [], {
        confidence: hoveredTarget?.confidence,
        entropy: hoveredTarget?.entropy,
        source: hoveredTarget?.assignmentSource,
      });
      const hoveredContinuity = objectContinuitySummary(hoveredTarget);
      const hoveredTemporal = objectTemporalSummary(hoveredTarget);
      const hoveredExplainability = objectExplainabilitySummary({
        object: hoveredTarget,
        assignmentProbe: hoveredAssignmentProbe,
        continuity: hoveredContinuity,
        temporal: hoveredTemporal,
      });
      const hoverHighlightSamples = [...draggableObjects.values()].map((object) => {
        const cloud = firstGaussianCloud(object);
        return {
          selectionId: object.userData.selectionId,
          modelId: object.userData.modelId,
          objectId: object.userData.objectId,
          visible: object.visible,
          selected: Boolean(object.userData.selected),
          hovered: Boolean(object.userData.hovered),
          hoverHighlighted: Boolean(cloud?.userData?.hoverHighlighted),
          hoverDimmed: Boolean(cloud?.userData?.hoverDimmed),
          hoverMode: cloud?.userData?.hoverHighlightMode ?? "off",
          gaussianCount: objectGaussianCount(object),
          opacity: round3(cloud?.material?.opacity ?? 0),
          pointSize: round3(cloud?.material?.size ?? 0),
        };
      });
      const highlightedHoverSamples = hoverHighlightSamples.filter((sample) => sample.hoverHighlighted);
      const dimmedHoverSamples = hoverHighlightSamples.filter((sample) => sample.hoverDimmed);
      const objectVisibilitySamples = [...draggableObjects.values()].map((object) => ({
        selectionId: object.userData.selectionId,
        modelId: object.userData.modelId,
        objectId: object.userData.objectId,
        visible: Boolean(object.visible),
        gaussianCount: objectGaussianCount(object),
      }));
      const visibleVisibilitySamples = objectVisibilitySamples.filter((sample) => sample.visible);
      const hiddenVisibilitySamples = objectVisibilitySamples.filter((sample) => !sample.visible);
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
        hoveredAssignment: hoveredTarget?.assignment ?? [],
        hoveredAssignmentConfidence: hoveredAssignmentProbe.confidence,
        hoveredAssignmentEntropy: hoveredAssignmentProbe.entropy,
        hoveredAssignmentProbe,
        hoveredAssignmentProbeStatus: hoveredAssignmentProbe.status,
        hoveredAssignmentProbeMargin: hoveredAssignmentProbe.margin,
        hoveredAssignmentTopSlot: hoveredAssignmentProbe.topSlot,
        hoveredAssignmentAmbiguous: hoveredAssignmentProbe.ambiguous,
        hoveredAssignmentCollapseRisk: hoveredAssignmentProbe.collapseRisk,
        hoveredContinuity,
        hoveredContinuityStatus: hoveredContinuity.status,
        hoveredContinuityBboxDiagonal: hoveredContinuity.bboxDiagonal,
        hoveredContinuitySpatialCompactness: hoveredContinuity.spatialCompactness,
        hoveredContinuityCentroidContained: hoveredContinuity.centroidContained,
        hoveredTemporal,
        hoveredTemporalStatus: hoveredTemporal.status,
        hoveredTemporalDrift: hoveredTemporal.temporalDrift,
        hoveredAssignmentJitter: hoveredTemporal.assignmentJitter,
        hoveredBboxStability: hoveredTemporal.bboxStability,
        hoveredTemporalStable: hoveredTemporal.stable,
        hoveredExplainability,
        hoveredExplainabilityStatus: hoveredExplainability.status,
        hoveredExplainable: hoveredExplainability.explainable,
        hoveredExplainabilityScore: hoveredExplainability.score,
        hoveredExplainabilityReasons: hoveredExplainability.reasonNames,
        hoverHighlightActive: Boolean(hoveredTarget?.selectionId),
        hoverHighlightedObjectCount: highlightedHoverSamples.length,
        hoverHighlightedGaussianCount: highlightedHoverSamples.reduce(
          (total, sample) => total + (Number(sample.gaussianCount) || 0),
          0,
        ),
        hoverDimmedObjectCount: dimmedHoverSamples.length,
        hoverDimmedGaussianCount: dimmedHoverSamples.reduce(
          (total, sample) => total + (Number(sample.gaussianCount) || 0),
          0,
        ),
        hoverHighlightSamples,
        debugMode: debugRef.current,
        debugLens: debugRef.current ? debugLensRef.current : "appearance",
        objectOverlayMode: overlayModeRef.current,
        objectOverlayBboxVisible: debugRef.current && objectOverlayShows(overlayModeRef.current, "bbox"),
        objectOverlayCentroidVisible: debugRef.current && objectOverlayShows(overlayModeRef.current, "centroid"),
        debugProtocol: "object-state-debug-os-v1",
        assignmentSource: selectedAssignmentSource,
        assignmentProbe: selectedAssignmentProbe,
        assignmentProbeStatus: selectedAssignmentProbe.status,
        assignmentProbeTopSlot: selectedAssignmentProbe.topSlot,
        assignmentProbeTopProbability: selectedAssignmentProbe.topProbability,
        assignmentProbeSecondProbability: selectedAssignmentProbe.secondProbability,
        assignmentProbeMargin: selectedAssignmentProbe.margin,
        assignmentProbeAmbiguous: selectedAssignmentProbe.ambiguous,
        assignmentProbeCollapseRisk: selectedAssignmentProbe.collapseRisk,
        objectContinuity: selectedContinuity,
        objectContinuityStatus: selectedContinuity.status,
        objectContinuityBboxDiagonal: selectedContinuity.bboxDiagonal,
        objectContinuitySpatialCompactness: selectedContinuity.spatialCompactness,
        objectContinuityCentroidContained: selectedContinuity.centroidContained,
        objectTemporal: selectedTemporal,
        objectTemporalStatus: selectedTemporal.status,
        objectTemporalDrift: selectedTemporal.temporalDrift,
        objectAssignmentJitter: selectedTemporal.assignmentJitter,
        objectBboxStability: selectedTemporal.bboxStability,
        objectTemporalStable: selectedTemporal.stable,
        objectExplainability: selectedExplainability,
        objectExplainabilityStatus: selectedExplainability.status,
        objectExplainable: selectedExplainability.explainable,
        objectExplainabilityScore: selectedExplainability.score,
        objectExplainabilityReasons: selectedExplainability.reasonNames,
        stabilitySummary: selectedStability,
        selectedTrainableFrameIndex: selectedModel?.userData?.trainableFrameIndex ?? null,
        selectedTrainableFrameCount: selectedModel?.userData?.trainableFrameCount ?? null,
        trainableArtifactLoadedCount: [...modelRoots.values()].filter(
          (object) => object.userData?.artifactSchema === "objgauss-trainable-kernel-model-artifact-v1",
        ).length,
        visibleObjectCount: visibleVisibilitySamples.length,
        hiddenObjectCount: hiddenVisibilitySamples.length,
        visibleGaussianCount: visibleVisibilitySamples.reduce(
          (total, sample) => total + (Number(sample.gaussianCount) || 0),
          0,
        ),
        hiddenGaussianCount: hiddenVisibilitySamples.reduce(
          (total, sample) => total + (Number(sample.gaussianCount) || 0),
          0,
        ),
        objectVisibilitySamples,
        hiddenObjectIds: hiddenVisibilitySamples.map((sample) => sample.selectionId),
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
        objectOverlaySamples: [...draggableObjects.values()].map((object) => ({
          selectionId: object.userData.selectionId,
          modelId: object.userData.modelId,
          objectId: object.userData.objectId,
          bboxVisible: objectChildVisible(object, "object-state-bbox"),
          centroidVisible: objectChildVisible(object, "core-point"),
        })),
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
      api.setObjectOverlayMode(overlayModeRef.current);
      api.setHiddenObjects(hiddenRef.current);
      return result.summary;
    };

    const selectObjectGroup = (object, probe = null) => {
      const target = objectTarget(object);
      if (!target?.selectionId) return;
      selectedRef.current = target.selectionId;
      const gaussian = probe?.gaussian ?? null;
      selectedGaussianProbe = gaussian;
      selectedGaussianProbeSelectionId = gaussian ? target.selectionId : null;
      api.setSelected(target.selectionId);
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
        refreshObjectOverlay();
        publishAuditHandle();
      },
      setDebugLens(lens) {
        debugLensRef.current = normalizeDebugLens(lens);
        refreshDebugLens();
        publishAuditHandle();
      },
      setObjectOverlayMode(mode) {
        overlayModeRef.current = normalizeObjectOverlayMode(mode);
        refreshObjectOverlay();
        publishAuditHandle();
      },
      setHover(selectionId) {
        const hoverFocus = Boolean(selectionId);
        for (const object of draggableObjects.values()) {
          const hovered = object.userData.selectionId === selectionId;
          object.userData.hovered = hovered;
          applyObjectVisualState(object, {
            selected: object.userData.selected,
            hovered,
            debug: debugRef.current,
            lens: debugLensRef.current,
            overlayMode: overlayModeRef.current,
            hoverFocus,
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
        if (selectedGaussianProbeSelectionId && selectedGaussianProbeSelectionId !== id) {
          selectedGaussianProbe = null;
          selectedGaussianProbeSelectionId = null;
        }
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
            overlayMode: overlayModeRef.current,
            hoverFocus: Boolean(hoveredObject?.userData?.selectionId),
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
        });
        applyObjectVisualState(object, {
          selected: object.userData.selected,
          hovered: object.userData.hovered,
          debug: debugEnabled,
          lens,
          overlayMode: overlayModeRef.current,
          hoverFocus: Boolean(hoveredObject?.userData?.selectionId),
        });
      }
    };

    const refreshObjectOverlay = () => {
      for (const object of draggableObjects.values()) {
        applyObjectVisualState(object, {
          selected: object.userData.selected,
          hovered: object.userData.hovered,
          debug: debugRef.current,
          lens: debugLensRef.current,
          overlayMode: overlayModeRef.current,
          hoverFocus: Boolean(hoveredObject?.userData?.selectionId),
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
  hoverAssignmentProbe,
  objectContinuity,
  hoverContinuity,
  objectTemporal,
  hoverTemporal,
  objectExplainability,
  hoverExplainability,
  debugProbe,
  assignmentProbe,
  debugMode,
  debugLens,
  objectOverlayMode,
  debugEvents,
  debugSnapshot,
  snapshotExport,
  sessionExport,
  sessionImport,
  debugSessionArchive,
  debugSessionDiff,
  hiddenObjects,
  objectVisibility,
  stability,
  qualityReport,
  objectStateBenchmark,
  benchmarkCase,
  onToggleObjectVisibility,
  onSelectDebugLens,
  onSelectObjectOverlayMode,
  onSelectTrainableFrame,
  onSelectOgcLod,
  onSelectOgcChunks,
  onSelectBenchmarkCase,
  onExportDebugSnapshot,
  onExportDebugSession,
  onImportDebugSession,
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
      data-object-overlay-mode={objectOverlayMode}
      data-object-overlay-bbox-visible={debugMode && objectOverlayShows(objectOverlayMode, "bbox") ? "true" : "false"}
      data-object-overlay-centroid-visible={debugMode && objectOverlayShows(objectOverlayMode, "centroid") ? "true" : "false"}
      data-probe-source={debugProbe?.source ?? activeState?.source ?? "none"}
      data-assignment-probe-status={assignmentProbe?.status ?? "none"}
      data-assignment-probe-top-slot={assignmentProbe?.topSlot ?? ""}
      data-assignment-probe-top-probability={assignmentProbe?.topProbability ?? ""}
      data-assignment-probe-second-probability={assignmentProbe?.secondProbability ?? ""}
      data-assignment-probe-margin={assignmentProbe?.margin ?? ""}
      data-assignment-probe-ambiguous={assignmentProbe?.ambiguous ? "true" : "false"}
      data-assignment-probe-collapse-risk={assignmentProbe?.collapseRisk ? "true" : "false"}
      data-object-continuity-status={objectContinuity?.status ?? "none"}
      data-object-continuity-spatial-compactness={objectContinuity?.spatialCompactness ?? ""}
      data-object-continuity-bbox-diagonal={objectContinuity?.bboxDiagonal ?? ""}
      data-object-continuity-density={objectContinuity?.gaussianDensity ?? ""}
      data-object-continuity-centroid-contained={objectContinuity?.centroidContained ? "true" : "false"}
      data-object-temporal-status={objectTemporal?.status ?? "none"}
      data-object-temporal-drift={objectTemporal?.temporalDrift ?? ""}
      data-object-assignment-jitter={objectTemporal?.assignmentJitter ?? ""}
      data-object-bbox-stability={objectTemporal?.bboxStability ?? ""}
      data-object-temporal-stable={objectTemporal?.stable ? "true" : "false"}
      data-object-explainability-status={objectExplainability?.status ?? "none"}
      data-object-explainable={objectExplainability?.explainable ? "true" : "false"}
      data-object-explainability-score={objectExplainability?.score ?? ""}
      data-object-explainability-reasons={objectExplainability?.reasonNames ?? ""}
      data-trainable-frame-index={selectedFrameIndex}
      data-trainable-frame-count={frameCount}
      data-ogc-lod-index={selectedOgcLod}
      data-ogc-lod-count={ogcLodLevels.length}
      data-ogc-chunk-scope={selectedOgcChunkScope}
      data-ogc-chunk-count={ogcChunkIds.length}
      data-hover-highlight={hoveredTarget?.selectionId ? "enabled" : "disabled"}
      data-hover-highlight-object={hoveredTarget?.selectionId ?? ""}
      data-hover-highlight-gaussians={hoveredTarget?.gaussianCount ?? ""}
      data-hover-assignment-source={hoveredTarget?.assignmentSource ?? ""}
      data-hover-assignment-slots={hoverAssignmentProbe?.slotCount ?? 0}
      data-hover-assignment-confidence={hoverAssignmentProbe?.confidence ?? ""}
      data-hover-assignment-entropy={hoverAssignmentProbe?.entropy ?? ""}
      data-hover-assignment-probe-status={hoverAssignmentProbe?.status ?? "none"}
      data-hover-assignment-probe-margin={hoverAssignmentProbe?.margin ?? ""}
      data-hover-assignment-top-slot={hoverAssignmentProbe?.topSlot ?? ""}
      data-hover-assignment-ambiguous={hoverAssignmentProbe?.ambiguous ? "true" : "false"}
      data-hover-assignment-collapse-risk={hoverAssignmentProbe?.collapseRisk ? "true" : "false"}
      data-hover-continuity-status={hoverContinuity?.status ?? "none"}
      data-hover-continuity-spatial-compactness={hoverContinuity?.spatialCompactness ?? ""}
      data-hover-continuity-bbox-diagonal={hoverContinuity?.bboxDiagonal ?? ""}
      data-hover-continuity-centroid-contained={hoverContinuity?.centroidContained ? "true" : "false"}
      data-hover-temporal-status={hoverTemporal?.status ?? "none"}
      data-hover-temporal-drift={hoverTemporal?.temporalDrift ?? ""}
      data-hover-assignment-jitter={hoverTemporal?.assignmentJitter ?? ""}
      data-hover-bbox-stability={hoverTemporal?.bboxStability ?? ""}
      data-hover-temporal-stable={hoverTemporal?.stable ? "true" : "false"}
      data-hover-explainability-status={hoverExplainability?.status ?? "none"}
      data-hover-explainable={hoverExplainability?.explainable ? "true" : "false"}
      data-hover-explainability-score={hoverExplainability?.score ?? ""}
      data-hover-explainability-reasons={hoverExplainability?.reasonNames ?? ""}
      data-object-visibility-contract="enabled"
      data-visible-objects={objectVisibility?.visibleObjectCount ?? 0}
      data-visible-gaussians={objectVisibility?.visibleGaussianCount ?? 0}
      data-hidden-objects={objectVisibility?.hiddenObjectCount ?? hiddenObjects.size}
      data-hidden-gaussians={objectVisibility?.hiddenGaussianCount ?? 0}
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

      <div
        className="trainableFrameSelector objectOverlaySelector"
        data-object-overlay-selector="true"
        data-selected-overlay={objectOverlayMode}
        data-overlay-debug-enabled={debugMode ? "true" : "false"}
      >
        <span>overlay</span>
        {OBJECT_OVERLAY_MODES.map((mode) => (
          <button
            key={mode}
            type="button"
            className={objectOverlayMode === mode ? "active" : ""}
            data-object-overlay-button={mode}
            data-active={objectOverlayMode === mode ? "true" : "false"}
            onClick={() => onSelectObjectOverlayMode?.(mode)}
          >
            {objectOverlayLabel(mode)}
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

      <AssignmentHeatmap
        assignment={assignment}
        selectedObject={selectedObject}
        debugProbe={debugProbe}
        assignmentProbe={assignmentProbe}
      />
      <ObjectStateVerdictPanel
        objectExplainability={objectExplainability}
        hoverExplainability={hoverExplainability}
      />
      <DebugSnapshotPanel
        snapshot={debugSnapshot}
        snapshotExport={snapshotExport}
        sessionExport={sessionExport}
        sessionImport={sessionImport}
        onExportDebugSnapshot={onExportDebugSnapshot}
        onExportDebugSession={onExportDebugSession}
        onImportDebugSession={onImportDebugSession}
      />
      <DebugSessionArchivePanel
        archive={debugSessionArchive}
        sessionImport={sessionImport}
        diff={debugSessionDiff}
      />
      <DebugEventTracePanel events={debugEvents} />
      <StabilityDashboard stability={stability} />
      <QualityReportPanel report={qualityReport} />
      <ObjectStateBenchmarkPanel
        benchmark={objectStateBenchmark}
        activeCase={benchmarkCase}
        onSelectCase={onSelectBenchmarkCase}
      />
      <TrainingEvidencePanel artifact={selected.trainableArtifact} />

      <dl className="debugStateGrid">
        <Meta label="source" value={debugProbe?.source ?? activeState?.source} />
        <Meta label="renderer" value={selected.delivery?.rendererName} />
        <Meta label="gaussian n" value={debugProbe?.gaussianIndex ?? "-"} />
        <Meta label="centroid" value={formatVec(activeState?.centroid)} />
        <Meta label="bbox" value={formatBox(activeState?.bbox)} />
        <Meta label="spatial" value={objectContinuity?.status ?? "-"} />
        <Meta label="diag" value={formatRatio(objectContinuity?.bboxDiagonal)} />
        <Meta label="motion" value={objectTemporal?.status ?? "-"} />
        <Meta label="jitter" value={formatRatio(objectTemporal?.assignmentJitter)} />
        <Meta label="explain" value={objectExplainability?.status ?? "-"} />
        <Meta
          label="hover"
          value={
            hoveredTarget
              ? `${hoveredTarget.modelId} #${hoveredTarget.objectId} / ${formatNumber(hoveredTarget.gaussianCount)}G`
              : "-"
          }
        />
        <Meta label="hover focus" value={hoveredTarget?.selectionId ? "enabled" : "-"} />
        <Meta label="hover A" value={hoverAssignmentProbe?.status !== "none" ? hoverAssignmentProbe?.status : "-"} />
        <Meta label="hover H" value={formatRatio(hoverAssignmentProbe?.entropy)} />
        <Meta label="hover spatial" value={hoverContinuity?.status !== "none" ? hoverContinuity?.status : "-"} />
        <Meta label="hover motion" value={hoverTemporal?.status !== "none" ? hoverTemporal?.status : "-"} />
        <Meta
          label="hover explain"
          value={hoverExplainability?.status !== "none" ? hoverExplainability?.status : "-"}
        />
        <Meta label="hidden" value={hiddenObjects.size} />
        <Meta label="hidden G" value={formatCount(objectVisibility?.hiddenGaussianCount)} />
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
              data-object-gaussians={objectGaussianCountForSummary(object)}
              data-object-hidden-gaussians={hidden ? objectGaussianCountForSummary(object) : 0}
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

function ObjectStateVerdictPanel({ objectExplainability, hoverExplainability }) {
  const selected = objectExplainability ?? compactObjectExplainability(null);
  const hover = hoverExplainability ?? compactObjectExplainability(null);
  const selectedReasons = verdictReasonRows(selected);
  const hoverReasons = verdictReasonRows(hover);
  const hoverActive = hover?.status && hover.status !== "none";
  return (
    <div
      className={`stabilityDashboard objectStateVerdict ${selected?.explainable ? "pass" : "warn"}`}
      data-object-verdict-panel="true"
      data-object-verdict-status={selected?.status ?? "none"}
      data-object-verdict-explainable={selected?.explainable ? "true" : "false"}
      data-object-verdict-score={selected?.score ?? ""}
      data-object-verdict-reason-count={selected?.reasons?.length ?? 0}
      data-object-verdict-reasons={selected?.reasonNames ?? ""}
      data-object-verdict-clear={selected?.explainable ? "true" : "false"}
      data-object-verdict-assignment-confidence={selected?.assignmentConfidence ?? ""}
      data-object-verdict-assignment-margin={selected?.assignmentMargin ?? ""}
      data-object-verdict-assignment-entropy={selected?.assignmentEntropy ?? ""}
      data-object-verdict-continuity-status={selected?.continuityStatus ?? ""}
      data-object-verdict-temporal-status={selected?.temporalStatus ?? ""}
      data-hover-verdict-status={hover?.status ?? "none"}
      data-hover-verdict-explainable={hover?.explainable ? "true" : "false"}
      data-hover-verdict-score={hover?.score ?? ""}
      data-hover-verdict-reason-count={hover?.reasons?.length ?? 0}
      data-hover-verdict-reasons={hover?.reasonNames ?? ""}
      data-hover-verdict-clear={hover?.explainable ? "true" : "false"}
      data-hover-verdict-assignment-confidence={hover?.assignmentConfidence ?? ""}
      data-hover-verdict-assignment-margin={hover?.assignmentMargin ?? ""}
      data-hover-verdict-assignment-entropy={hover?.assignmentEntropy ?? ""}
      data-hover-verdict-continuity-status={hover?.continuityStatus ?? ""}
      data-hover-verdict-temporal-status={hover?.temporalStatus ?? ""}
    >
      <div className="stabilityHead">
        <span>Verdict</span>
        <strong>{selected?.status ?? "-"}</strong>
      </div>
      <div className="stabilityGrid verdictGrid">
        <Metric label="score" value={formatRatio(selected?.score)} />
        <Metric label="margin" value={formatRatio(selected?.assignmentMargin)} />
        <Metric label="spatial" value={selected?.continuityStatus || "-"} />
        <Metric label="motion" value={selected?.temporalStatus || "-"} />
      </div>
      <dl className="stabilityMeta trainingMeta">
        <Meta label="A conf" value={formatRatio(selected?.assignmentConfidence)} />
        <Meta label="A entropy" value={formatRatio(selected?.assignmentEntropy)} />
        <Meta label="reasons" value={selected?.reasonNames || "clear"} />
      </dl>
      <div
        className="qualityGateRows objectVerdictRows"
        data-object-verdict-reasons-list="true"
        data-object-verdict-row-count={selectedReasons.length}
      >
        {selectedReasons.map((reason) => (
          <div
            className={`qualityGateRow ${reason.status}`}
            key={reason.name}
            data-object-verdict-reason-row="true"
            data-object-verdict-reason-name={reason.name}
            data-object-verdict-reason-status={reason.status}
          >
            <span>{reason.name}</span>
            <small>{reason.value}</small>
            <strong>{reason.status}</strong>
          </div>
        ))}
      </div>
      {hoverActive ? (
        <>
          <dl className="stabilityMeta trainingMeta hoverVerdictMeta" data-hover-verdict-meta="true">
            <Meta label="hover" value={hover.status} />
            <Meta label="score" value={formatRatio(hover.score)} />
            <Meta label="reasons" value={hover.reasonNames || "clear"} />
          </dl>
          <div
            className="qualityGateRows objectVerdictRows"
            data-hover-verdict-reasons-list="true"
            data-hover-verdict-row-count={hoverReasons.length}
          >
            {hoverReasons.map((reason) => (
              <div
                className={`qualityGateRow ${reason.status}`}
                key={`hover-${reason.name}`}
                data-hover-verdict-reason-row="true"
                data-hover-verdict-reason-name={reason.name}
                data-hover-verdict-reason-status={reason.status}
              >
                <span>{reason.name}</span>
                <small>{reason.value}</small>
                <strong>{reason.status}</strong>
              </div>
            ))}
          </div>
        </>
      ) : null}
    </div>
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

function DebugSnapshotPanel({
  snapshot,
  snapshotExport,
  sessionExport,
  sessionImport,
  onExportDebugSnapshot,
  onExportDebugSession,
  onImportDebugSession,
}) {
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
      data-debug-snapshot-overlay-mode={snapshot.debug.overlayMode}
      data-debug-snapshot-source={snapshot.assignment.source}
      data-debug-snapshot-slots={snapshot.assignment.slotCount}
      data-debug-snapshot-assignment-probe-status={snapshot.assignment.probe?.status ?? ""}
      data-debug-snapshot-assignment-probe-top-slot={snapshot.assignment.probe?.topSlot ?? ""}
      data-debug-snapshot-assignment-probe-margin={snapshot.assignment.probe?.margin ?? ""}
      data-debug-snapshot-assignment-probe-ambiguous={snapshot.assignment.probe?.ambiguous ? "true" : "false"}
      data-debug-snapshot-assignment-probe-collapse-risk={snapshot.assignment.probe?.collapseRisk ? "true" : "false"}
      data-debug-snapshot-hidden-objects={snapshot.visibility?.hiddenObjectCount ?? ""}
      data-debug-snapshot-hidden-gaussians={snapshot.visibility?.hiddenGaussianCount ?? ""}
      data-debug-snapshot-continuity-status={snapshot.continuity?.status ?? ""}
      data-debug-snapshot-continuity-bbox-diagonal={snapshot.continuity?.bboxDiagonal ?? ""}
      data-debug-snapshot-continuity-spatial-compactness={snapshot.continuity?.spatialCompactness ?? ""}
      data-debug-snapshot-continuity-centroid-contained={snapshot.continuity?.centroidContained ? "true" : "false"}
      data-debug-snapshot-temporal-status={snapshot.temporal?.status ?? ""}
      data-debug-snapshot-temporal-drift={snapshot.temporal?.temporalDrift ?? ""}
      data-debug-snapshot-assignment-jitter={snapshot.temporal?.assignmentJitter ?? ""}
      data-debug-snapshot-bbox-stability={snapshot.temporal?.bboxStability ?? ""}
      data-debug-snapshot-temporal-stable={snapshot.temporal?.stable ? "true" : "false"}
      data-debug-snapshot-explainability-status={snapshot.explainability?.status ?? ""}
      data-debug-snapshot-explainable={snapshot.explainability?.explainable ? "true" : "false"}
      data-debug-snapshot-explainability-score={snapshot.explainability?.score ?? ""}
      data-debug-snapshot-explainability-reasons={snapshot.explainability?.reasonNames ?? ""}
      data-debug-snapshot-hover-object={snapshot.hover?.selectionId ?? ""}
      data-debug-snapshot-hover-assignment-status={snapshot.hover?.probe?.status ?? ""}
      data-debug-snapshot-hover-assignment-margin={snapshot.hover?.probe?.margin ?? ""}
      data-debug-snapshot-hover-continuity-status={snapshot.hover?.continuity?.status ?? ""}
      data-debug-snapshot-hover-continuity-bbox-diagonal={snapshot.hover?.continuity?.bboxDiagonal ?? ""}
      data-debug-snapshot-hover-continuity-centroid-contained={snapshot.hover?.continuity?.centroidContained ? "true" : "false"}
      data-debug-snapshot-hover-temporal-status={snapshot.hover?.temporal?.status ?? ""}
      data-debug-snapshot-hover-temporal-drift={snapshot.hover?.temporal?.temporalDrift ?? ""}
      data-debug-snapshot-hover-assignment-jitter={snapshot.hover?.temporal?.assignmentJitter ?? ""}
      data-debug-snapshot-hover-temporal-stable={snapshot.hover?.temporal?.stable ? "true" : "false"}
      data-debug-snapshot-hover-explainability-status={snapshot.hover?.explainability?.status ?? ""}
      data-debug-snapshot-hover-explainable={snapshot.hover?.explainability?.explainable ? "true" : "false"}
      data-debug-snapshot-hover-explainability-score={snapshot.hover?.explainability?.score ?? ""}
      data-debug-snapshot-stability={snapshot.stability.status}
      data-debug-snapshot-training-status={snapshot.training?.status ?? ""}
      data-debug-snapshot-quality-status={snapshot.quality?.status ?? ""}
      data-debug-snapshot-export-status={snapshotExport?.status ?? "idle"}
      data-debug-snapshot-export-file={snapshotExport?.fileName ?? ""}
      data-debug-snapshot-export-schema={snapshotExport?.schema ?? ""}
      data-debug-session-export-status={sessionExport?.status ?? "idle"}
      data-debug-session-export-file={sessionExport?.fileName ?? ""}
      data-debug-session-export-schema={sessionExport?.schema ?? ""}
      data-debug-session-import-status={sessionImport?.status ?? "idle"}
      data-debug-session-import-file={sessionImport?.fileName ?? ""}
      data-debug-session-import-schema={sessionImport?.schema ?? ""}
    >
      <div className="stabilityHead">
        <span>Protocol</span>
        <div className="snapshotActions">
          <strong>snapshot-v1</strong>
          <button
            type="button"
            data-debug-snapshot-export-button="true"
            data-export-status={snapshotExport?.status ?? "idle"}
            onClick={() => onExportDebugSnapshot?.()}
          >
            JSON
          </button>
          <button
            type="button"
            data-debug-session-export-button="true"
            data-export-status={sessionExport?.status ?? "idle"}
            onClick={() => onExportDebugSession?.()}
          >
            SESSION
          </button>
          <button
            type="button"
            data-debug-session-import-button="true"
            data-import-status={sessionImport?.status ?? "idle"}
            onClick={() => onImportDebugSession?.()}
          >
            LOAD
          </button>
        </div>
      </div>
      <dl className="stabilityMeta snapshotMeta">
        <Meta label="model" value={snapshot.model.id} />
        <Meta label="object" value={snapshot.selection.objectId ?? "-"} />
        <Meta label="lens" value={snapshot.debug.lens} />
        <Meta label="overlay" value={snapshot.debug.overlayMode} />
        <Meta label="slots" value={formatCount(snapshot.assignment.slotCount)} />
        <Meta label="source" value={snapshot.assignment.source} />
        <Meta label="probe" value={snapshot.assignment.probe?.status ?? "-"} />
        <Meta label="margin" value={formatRatio(snapshot.assignment.probe?.margin)} />
        <Meta label="hidden G" value={formatCount(snapshot.visibility?.hiddenGaussianCount)} />
        <Meta label="spatial" value={snapshot.continuity?.status ?? "-"} />
        <Meta label="diag" value={formatRatio(snapshot.continuity?.bboxDiagonal)} />
        <Meta label="motion" value={snapshot.temporal?.status ?? "-"} />
        <Meta label="explain" value={snapshot.explainability?.status ?? "-"} />
        <Meta label="hover A" value={snapshot.hover?.probe?.status ?? "-"} />
        <Meta label="hover spatial" value={snapshot.hover?.continuity?.status ?? "-"} />
        <Meta label="hover motion" value={snapshot.hover?.temporal?.status ?? "-"} />
        <Meta label="hover explain" value={snapshot.hover?.explainability?.status ?? "-"} />
        <Meta label="state" value={snapshot.stability.status} />
        <Meta label="export" value={snapshotExport?.fileName || snapshotExport?.status || "idle"} />
        <Meta label="session" value={sessionExport?.fileName || sessionExport?.status || "idle"} />
        <Meta label="archive" value={sessionImport?.fileName || sessionImport?.status || "idle"} />
      </dl>
    </div>
  );
}

function DebugSessionArchivePanel({ archive, sessionImport, diff }) {
  if (!archive && !["loading", "error"].includes(sessionImport?.status)) return null;
  const recent = Array.isArray(archive?.events) ? archive.events.slice(0, 3) : [];
  return (
    <div
      className="stabilityDashboard debugSessionArchivePanel"
      data-debug-session-archive="true"
      data-debug-session-archive-status={sessionImport?.status ?? "idle"}
      data-debug-session-archive-file={sessionImport?.fileName ?? ""}
      data-debug-session-archive-schema={archive?.schema ?? ""}
      data-debug-session-archive-model={archive?.snapshot?.model?.id ?? ""}
      data-debug-session-archive-quality={archive?.snapshot?.quality?.status ?? ""}
      data-debug-session-archive-events={archive?.events?.length ?? ""}
      data-debug-session-archive-models={archive?.models?.length ?? ""}
      data-debug-session-archive-error={sessionImport?.error ?? ""}
      data-debug-session-diff-status={diff?.status ?? ""}
      data-debug-session-diff-model-match={diff?.modelMatch ? "true" : diff ? "false" : ""}
      data-debug-session-diff-source-match={diff?.sourceMatch ? "true" : diff ? "false" : ""}
      data-debug-session-diff-quality-match={diff?.qualityMatch ? "true" : diff ? "false" : ""}
      data-debug-session-diff-training-match={diff?.trainingMatch ? "true" : diff ? "false" : ""}
      data-debug-session-diff-slot-delta={diff?.slotDelta ?? ""}
      data-debug-session-diff-entropy-delta={diff?.entropyDelta ?? ""}
      data-debug-session-diff-event-delta={diff?.eventDelta ?? ""}
      data-debug-session-diff-field-count={diff?.changedFields?.length ?? ""}
      data-debug-session-diff-fields={diff?.changedFieldNames ?? ""}
    >
      <div className="stabilityHead">
        <span>Archive</span>
        <strong>{archive?.snapshot?.model?.id ?? sessionImport?.status ?? "-"}</strong>
      </div>
      <dl className="stabilityMeta snapshotMeta">
        <Meta label="file" value={sessionImport?.fileName || "-"} />
        <Meta label="schema" value={archive?.schema || sessionImport?.schema || "-"} />
        <Meta label="models" value={formatCount(archive?.models?.length)} />
        <Meta label="events" value={formatCount(archive?.events?.length)} />
        <Meta label="quality" value={archive?.snapshot?.quality?.status ?? "-"} />
        <Meta label="error" value={sessionImport?.error || "-"} />
      </dl>
      {diff ? (
        <dl className="stabilityMeta snapshotMeta debugSessionDiffMeta" data-debug-session-diff="true">
          <Meta label="diff" value={diff.status} />
          <Meta label="model" value={diff.modelMatch ? "match" : "changed"} />
          <Meta label="source" value={diff.sourceMatch ? "match" : "changed"} />
          <Meta label="quality" value={diff.qualityMatch ? "match" : "changed"} />
          <Meta label="d slots" value={formatSignedCount(diff.slotDelta)} />
          <Meta label="d H" value={formatSignedRatio(diff.entropyDelta)} />
          <Meta label="d conf" value={formatSignedRatio(diff.confidenceDelta)} />
          <Meta label="d events" value={formatSignedCount(diff.eventDelta)} />
          <Meta label="fields" value={diff.changedFieldNames || "-"} />
        </dl>
      ) : null}
      {recent.length ? (
        <div className="debugEventRows">
          {recent.map((event, index) => (
            <div
              className="debugEventRow"
              key={`${event.seq}-${event.type}-${index}`}
              data-debug-session-archive-event-row="true"
              data-debug-event-type={event.type}
            >
              <span>{event.type}</span>
              <small>{debugEventDetailLabel(event)}</small>
            </div>
          ))}
        </div>
      ) : null}
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

function QualityReportPanel({ report }) {
  const summary = report ?? null;
  if (!summary) return null;
  return (
    <div
      className="stabilityDashboard qualityReport"
      data-quality-report="true"
      data-quality-report-status={summary.status}
      data-quality-report-schema={summary.schema}
      data-quality-report-assignment-entropy={summary.assignmentEntropy ?? ""}
      data-quality-report-slot-utilization={summary.slotUtilization ?? ""}
      data-quality-report-object-purity={summary.objectPurity ?? ""}
      data-quality-report-temporal-drift={summary.temporalDrift ?? ""}
      data-quality-report-assignment-jitter={summary.assignmentJitter ?? ""}
      data-quality-report-bbox-stability={summary.bboxStability ?? ""}
      data-quality-report-gate-count={summary.gateCount}
      data-quality-report-failing-gates={summary.failingGates}
      data-quality-report-failing-gate-names={summary.failingGateNames}
    >
      <div className="stabilityHead">
        <span>Quality</span>
        <strong>{summary.status}</strong>
      </div>
      <div className="stabilityGrid trainingGrid">
        <Metric label="H" value={formatRatio(summary.assignmentEntropy)} />
        <Metric label="purity" value={formatRatio(summary.objectPurity)} />
        <Metric label="drift" value={formatRatio(summary.temporalDrift)} />
        <Metric label="jitter" value={formatRatio(summary.assignmentJitter)} />
      </div>
      <dl className="stabilityMeta trainingMeta">
        <Meta label="schema" value={summary.schema} />
        <Meta label="gates" value={`${formatCount(summary.passingGates)} / ${formatCount(summary.gateCount)}`} />
        <Meta label="slot" value={formatRatio(summary.slotUtilization)} />
        <Meta label="bbox" value={formatRatio(summary.bboxStability)} />
      </dl>
      {summary.gates.length ? (
        <div
          className="qualityGateRows"
          data-quality-gates="true"
          data-quality-gate-count={summary.gates.length}
          data-quality-failing-gate-names={summary.failingGateNames}
        >
          {summary.gates.map((gate) => (
            <div
              className={`qualityGateRow ${gate.status}`}
              key={gate.name}
              data-quality-gate-row="true"
              data-quality-gate-name={gate.name}
              data-quality-gate-status={gate.status}
              data-quality-gate-value={gate.value ?? ""}
              data-quality-gate-threshold={gate.threshold ?? ""}
            >
              <span>{gate.name}</span>
              <small>{formatGateValue(gate)}</small>
              <strong>{gate.status}</strong>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ObjectStateBenchmarkPanel({ benchmark, activeCase, onSelectCase }) {
  const summary = benchmark ?? null;
  if (!summary) return null;
  const selectedCase = activeCase ?? activeObjectStateBenchmarkCase(summary, "");
  return (
    <div
      className="stabilityDashboard objectStateBenchmark"
      data-object-state-benchmark="true"
      data-object-state-benchmark-status={summary.status}
      data-object-state-benchmark-schema={summary.schema}
      data-object-state-benchmark-case-count={summary.caseCount}
      data-object-state-benchmark-warn-count={summary.warnCount}
      data-object-state-benchmark-observed-warn-count={summary.observedWarnCount}
      data-object-state-benchmark-failure-mode-count={summary.failureModeCount}
      data-object-state-benchmark-first-case={summary.cases[0]?.name ?? ""}
      data-object-state-benchmark-active-case={selectedCase?.name ?? ""}
      data-object-state-benchmark-active-status={selectedCase?.status ?? ""}
      data-object-state-benchmark-active-observed-status={selectedCase?.observedStatus ?? ""}
      data-object-state-benchmark-active-failure-modes={selectedCase?.failureModeNames ?? ""}
      data-object-state-benchmark-active-diagnostics={selectedCase?.diagnosticNames ?? ""}
      data-object-state-benchmark-active-assignment-confidence={selectedCase?.assignmentConfidence ?? ""}
      data-object-state-benchmark-active-entropy={selectedCase?.meanEntropy ?? ""}
      data-object-state-benchmark-active-purity={selectedCase?.objectPurity ?? ""}
      data-object-state-benchmark-active-temporal-drift={selectedCase?.meanTemporalDrift ?? ""}
      data-object-state-benchmark-active-dynamic-proposals={selectedCase?.dynamicProposalCount ?? ""}
    >
      <div className="stabilityHead">
        <span>Benchmark</span>
        <strong>{summary.status}</strong>
      </div>
      <div className="stabilityGrid trainingGrid">
        <Metric label="cases" value={formatCount(summary.caseCount)} />
        <Metric label="warn" value={formatCount(summary.warnCount)} />
        <Metric label="observed" value={formatCount(summary.observedWarnCount)} />
        <Metric label="modes" value={formatCount(summary.failureModeCount)} />
      </div>
      <dl className="stabilityMeta trainingMeta">
        <Meta label="schema" value={summary.schema} />
        <Meta label="report" value={summary.reportId} />
        <Meta label="coverage" value={formatCount(summary.failureModeCount)} />
        <Meta label="path" value={summary.path || "-"} />
      </dl>
      {selectedCase ? (
        <>
          <div
            className="stabilityGrid trainingGrid benchmarkCaseGrid"
            data-object-state-benchmark-active-metrics="true"
          >
            <Metric label="conf" value={formatRatio(selectedCase.assignmentConfidence)} />
            <Metric label="H" value={formatRatio(selectedCase.meanEntropy)} />
            <Metric label="purity" value={formatRatio(selectedCase.objectPurity)} />
            <Metric label="drift" value={formatRatio(selectedCase.meanTemporalDrift)} />
          </div>
          <dl className="stabilityMeta trainingMeta benchmarkCaseMeta">
            <Meta label="case" value={selectedCase.name} />
            <Meta label="diag" value={selectedCase.diagnosticNames || "-"} />
            <Meta label="modes" value={selectedCase.failureModeNames || "-"} />
            <Meta label="dynK" value={formatCount(selectedCase.dynamicProposalCount)} />
          </dl>
        </>
      ) : null}
      {summary.cases.length ? (
        <div
          className="qualityGateRows"
          data-object-state-benchmark-cases="true"
          data-object-state-benchmark-case-row-count={summary.cases.length}
        >
          {summary.cases.map((testCase) => (
            <button
              type="button"
              className={`qualityGateRow caseButton ${testCase.status} ${selectedCase?.name === testCase.name ? "selected" : ""}`}
              key={testCase.name}
              data-object-state-benchmark-case-row="true"
              data-object-state-benchmark-case-name={testCase.name}
              data-object-state-benchmark-case-status={testCase.status}
              data-object-state-benchmark-case-observed-status={testCase.observedStatus}
              data-object-state-benchmark-case-selected={selectedCase?.name === testCase.name ? "true" : "false"}
              data-object-state-benchmark-case-failure-modes={testCase.failureModeNames}
              data-object-state-benchmark-case-diagnostics={testCase.diagnosticNames}
              onClick={() => onSelectCase?.(testCase.name)}
            >
              <span>{testCase.name}</span>
              <small>{testCase.observedStatus}</small>
              <strong>{testCase.failureModeCount ? formatCount(testCase.failureModeCount) : testCase.status}</strong>
            </button>
          ))}
        </div>
      ) : null}
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

function AssignmentHeatmap({ assignment, selectedObject, debugProbe, assignmentProbe }) {
  const rows = assignment?.length ? assignment : [];
  return (
    <div
      className="assignmentHeatmap"
      data-assignment-heatmap="true"
      data-assignment-source={debugProbe?.source ?? selectedObject?.objectState?.source ?? "none"}
      data-assignment-slots={rows.length}
      data-assignment-probe-status={assignmentProbe?.status ?? "none"}
      data-assignment-probe-top-slot={assignmentProbe?.topSlot ?? ""}
      data-assignment-probe-top-object={assignmentProbe?.topObjectId ?? ""}
      data-assignment-probe-top-probability={assignmentProbe?.topProbability ?? ""}
      data-assignment-probe-second-probability={assignmentProbe?.secondProbability ?? ""}
      data-assignment-probe-margin={assignmentProbe?.margin ?? ""}
      data-assignment-probe-ambiguous={assignmentProbe?.ambiguous ? "true" : "false"}
      data-assignment-probe-collapse-risk={assignmentProbe?.collapseRisk ? "true" : "false"}
    >
      {assignmentProbe?.slotCount ? (
        <div
          className="assignmentProbeMeta"
          data-assignment-probe="true"
          data-assignment-probe-status={assignmentProbe.status}
        >
          <span>{assignmentProbe.status}</span>
          <strong>k{assignmentProbe.topSlot ?? "-"}</strong>
          <small>{formatRatio(assignmentProbe.topProbability)}</small>
          <small>{formatRatio(assignmentProbe.margin)}</small>
        </div>
      ) : null}
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
    const opacityColors = new Float32Array(entries.length * 3);
    const fallback = new THREE.Color(accent);
    const debugColor = fallback.clone().lerp(new THREE.Color("#f1fdff"), assignmentEntropy * 0.34);
    const confidenceColor = debugConfidenceColor(assignmentConfidence);
    const entropyColor = debugEntropyColor(assignmentEntropy);
    const opacityValues = [];

    entries.forEach((entry, index) => {
      const opacityValue = normalizedGaussianOpacity(entry.point.opacity, 0.94);
      opacityValues.push(opacityValue);
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
      writeColor(opacityColors, index, debugOpacityColor(opacityValue));
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const originalColorAttr = new THREE.BufferAttribute(originalColors, 3);
    const assignmentColorAttr = new THREE.BufferAttribute(assignmentColors, 3);
    const confidenceColorAttr = new THREE.BufferAttribute(confidenceColors, 3);
    const entropyColorAttr = new THREE.BufferAttribute(entropyColors, 3);
    const opacityColorAttr = new THREE.BufferAttribute(opacityColors, 3);
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
    cloud.userData.opacityColor = opacityColorAttr;
    cloud.userData.opacityMean = round3(average(opacityValues));
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
      opacity: round3(opacityValues[index] ?? 0),
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
      gaussianOpacityMean: cloud.userData.opacityMean,
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
    const opacityColorAttr = uniformColorAttribute(points.positions.length / 3, debugOpacityColor(0.58));
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
    cloud.userData.opacityColor = opacityColorAttr;
    cloud.userData.opacityMean = 0.58;
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
      gaussianOpacityMean: cloud.userData.opacityMean,
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
    const opacityColors = new Float32Array(stateRows.length * 3);
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
      writeColor(opacityColors, index, debugOpacityColor(0.96));
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const originalColorAttr = new THREE.BufferAttribute(originalColors, 3);
    const assignmentColorAttr = new THREE.BufferAttribute(assignmentColors, 3);
    const confidenceColorAttr = new THREE.BufferAttribute(confidenceColors, 3);
    const entropyColorAttr = new THREE.BufferAttribute(entropyColors, 3);
    const opacityColorAttr = new THREE.BufferAttribute(opacityColors, 3);
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
    cloud.userData.opacityColor = opacityColorAttr;
    cloud.userData.opacityMean = 0.96;
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
      gaussianOpacityMean: cloud.userData.opacityMean,
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

function objectChildVisible(object, role) {
  let visible = false;
  object?.traverse?.((child) => {
    if (child.userData?.role === role && child.visible !== false) visible = true;
  });
  return visible;
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
  if (normalized === "opacity") return cloud.userData.opacityColor ?? cloud.userData.assignmentColor;
  return cloud.userData.assignmentColor;
}

function applyObjectVisualState(
  object,
  {
    selected = false,
    hovered = false,
    debug = true,
    lens = "assignment",
    overlayMode = "full",
    hoverFocus = false,
  } = {},
) {
  const selectedOrHovered = Boolean(selected || hovered);
  const normalizedLens = normalizeDebugLens(lens);
  const normalizedOverlay = normalizeObjectOverlayMode(overlayMode);
  const showBbox = Boolean(debug && objectOverlayShows(normalizedOverlay, "bbox"));
  const showCentroid = Boolean(debug && objectOverlayShows(normalizedOverlay, "centroid"));
  const dimmedByHover = Boolean(hoverFocus && !selected && !hovered);
  object.traverse((child) => {
    if (child.userData.role === "gaussian-cloud") {
      const baseSize = child.userData.basePointSize ?? child.material.size;
      child.userData.basePointSize = baseSize;
      const lensOpacity = opacityForDebugLens(object.userData.objectState, normalizedLens);
      child.userData.activeOpacityLens = debug ? normalizedLens : "appearance";
      const baseOpacity = selected ? 1 : hovered ? Math.max(0.86, lensOpacity) : debug ? lensOpacity : 0.5;
      child.material.opacity = dimmedByHover ? Math.min(baseOpacity, HOVER_DIM_OPACITY) : baseOpacity;
      child.material.size = selected ? baseSize * 1.22 : hovered ? baseSize * 1.2 : dimmedByHover ? baseSize * 0.92 : baseSize;
      child.userData.hoverHighlighted = Boolean(hovered);
      child.userData.hoverDimmed = dimmedByHover;
      child.userData.hoverHighlightMode = hoverFocus
        ? hovered
          ? "highlighted"
          : selected
            ? "selected"
            : "dimmed"
        : "off";
      child.material.needsUpdate = true;
    }
    if (child.userData.role === "object-state-bbox") {
      child.visible = showBbox;
      child.material.opacity = selected ? 0.86 : hovered ? 0.72 : 0.34;
    }
    if (child.userData.role === "core-point") {
      child.visible = showCentroid;
    }
    if (child.userData.role === "core-glow") {
      child.visible = selectedOrHovered && showCentroid;
    }
    if (child.userData.role === "selection-ring") {
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
  if (lens === "opacity") {
    return round3(0.32 + 0.58 * clamp01(objectState?.gaussianOpacityMean ?? 0.5));
  }
  return 0.62;
}

function objectTarget(object) {
  if (!object?.userData?.selectionId) return null;
  const objectState = object.userData.objectState ?? {};
  const assignment = Array.isArray(objectState.assignment) ? objectState.assignment : [];
  return {
    modelId: object.userData.modelId,
    objectId: object.userData.objectId,
    selectionId: object.userData.selectionId,
    gaussianCount: objectGaussianCount(object),
    assignmentSource: objectState.source ?? null,
    assignment: compactAssignmentVector(assignment),
    confidence: finiteNumber(objectState.confidence),
    entropy: finiteNumber(objectState.assignmentEntropy),
    temporalDrift: finiteNumber(objectState.temporalDrift),
    assignmentJitter: finiteNumber(objectState.assignmentJitter),
    spatialCompactness: finiteNumber(objectState.spatialCompactness),
    bboxStability: finiteNumber(objectState.bboxStability),
    status: objectState.status ?? "",
    centroid: cleanNumberArray(objectState.centroid),
    bbox: cleanNumberArray(objectState.bbox),
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

function normalizeObjectOverlayMode(mode) {
  const value = String(mode ?? "full");
  return OBJECT_OVERLAY_MODES.includes(value) ? value : "full";
}

function objectOverlayShows(mode, target) {
  const normalized = normalizeObjectOverlayMode(mode);
  if (normalized === "off") return false;
  if (normalized === "full") return true;
  return normalized === target;
}

function objectOverlayLabel(mode) {
  if (mode === "bbox") return "bbox";
  if (mode === "centroid") return "center";
  if (mode === "off") return "off";
  return "full";
}

function debugLensLabel(lens) {
  if (lens === "confidence") return "conf";
  if (lens === "entropy") return "H";
  if (lens === "opacity") return "opac";
  return "assign";
}

function debugConfidenceColor(confidence) {
  return new THREE.Color("#ff4d6d").lerp(new THREE.Color("#5df2df"), clamp01(confidence));
}

function debugEntropyColor(entropy) {
  return new THREE.Color("#53d8da").lerp(new THREE.Color("#ff8a5c"), clamp01(entropy));
}

function debugOpacityColor(opacity) {
  return new THREE.Color("#33485d").lerp(new THREE.Color("#eefbff"), clamp01(opacity));
}

function normalizedGaussianOpacity(value, fallback = 0.62) {
  const number = Number(value);
  if (!Number.isFinite(number)) return clamp01(fallback);
  if (number > 1) return clamp01(number / 255);
  return clamp01(number);
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
  gaussianOpacityMean,
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
    gaussianOpacityMean: optionalRound3(gaussianOpacityMean),
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

function objectGaussianCountForSummary(object) {
  const candidates = [
    object?.displayCount,
    object?.gaussianCount,
    object?.objectState?.slotMass,
    object?.objectState?.gaussianCount,
  ];
  const count = candidates.map((value) => Number(value)).find((value) => Number.isFinite(value) && value >= 0);
  return count ? Math.trunc(count) : 0;
}

function objectVisibilitySummary(models = [], hiddenObjects = new Set()) {
  const hidden = hiddenObjects instanceof Set ? hiddenObjects : new Set(hiddenObjects ?? []);
  const samples = [];
  for (const model of models ?? []) {
    for (const object of model?.objects ?? []) {
      const selectionId = object?.selectionId ?? selectionIdForObject(model?.id ?? "", object?.objectId ?? "");
      const gaussianCount = objectGaussianCountForSummary(object);
      const visible = !hidden.has(selectionId);
      samples.push({
        selectionId,
        modelId: model?.id ?? "",
        objectId: object?.objectId ?? null,
        visible,
        gaussianCount,
      });
    }
  }
  const visibleSamples = samples.filter((sample) => sample.visible);
  const hiddenSamples = samples.filter((sample) => !sample.visible);
  return {
    schema: "objgauss-object-visibility-summary-v1",
    objectCount: samples.length,
    visibleObjectCount: visibleSamples.length,
    hiddenObjectCount: hiddenSamples.length,
    visibleGaussianCount: visibleSamples.reduce((total, sample) => total + sample.gaussianCount, 0),
    hiddenGaussianCount: hiddenSamples.reduce((total, sample) => total + sample.gaussianCount, 0),
    hiddenSelectionIds: hiddenSamples.map((sample) => sample.selectionId),
    samples: samples.slice(0, 64),
  };
}

function objectContinuitySummary(object) {
  if (!object) {
    return {
      schema: "objgauss-object-continuity-summary-v1",
      status: "none",
      spatialCompactness: null,
      bboxDiagonal: null,
      gaussianDensity: null,
      centroidContained: false,
      bboxValid: false,
      gaussianCount: 0,
      centroid: [],
      bbox: [],
    };
  }
  const state = object?.objectState ?? object;
  const bbox = cleanNumberArray(state?.bbox);
  const centroid = cleanNumberArray(state?.centroid);
  const spatialCompactness = finiteNumber(state?.spatialCompactness ?? object?.spatialCompactness);
  const gaussianCount = Math.trunc(
    finiteNumber(object?.gaussianCount) ??
      finiteNumber(state?.gaussianCount) ??
      finiteNumber(state?.slotMass) ??
      objectGaussianCountForSummary(object),
  );
  const bboxValid = Boolean(
    bbox.length >= 6 &&
      bbox.slice(0, 6).every((value) => Number.isFinite(Number(value))) &&
      bbox[3] >= bbox[0] &&
      bbox[4] >= bbox[1] &&
      bbox[5] >= bbox[2],
  );
  const dx = bboxValid ? Math.max(0, bbox[3] - bbox[0]) : 0;
  const dy = bboxValid ? Math.max(0, bbox[4] - bbox[1]) : 0;
  const dz = bboxValid ? Math.max(0, bbox[5] - bbox[2]) : 0;
  const bboxDiagonal = bboxValid ? round6(Math.sqrt(dx * dx + dy * dy + dz * dz)) : null;
  const volume = bboxValid ? dx * dy * dz : 0;
  const gaussianDensity = volume > 0 && gaussianCount > 0 ? round6(gaussianCount / volume) : null;
  const centroidContained = Boolean(
    bboxValid &&
      centroid.length >= 3 &&
      centroid[0] >= bbox[0] &&
      centroid[0] <= bbox[3] &&
      centroid[1] >= bbox[1] &&
      centroid[1] <= bbox[4] &&
      centroid[2] >= bbox[2] &&
      centroid[2] <= bbox[5],
  );
  const status = !bboxValid
    ? "invalid-bbox"
    : gaussianCount <= 0
      ? "empty"
      : !centroidContained
        ? "centroid-outside"
        : spatialCompactness !== null && spatialCompactness < 0.35
          ? "fragmented"
          : bboxDiagonal !== null && bboxDiagonal <= 0
            ? "degenerate"
            : "continuous";
  return {
    schema: "objgauss-object-continuity-summary-v1",
    status,
    spatialCompactness,
    bboxDiagonal,
    gaussianDensity,
    centroidContained,
    bboxValid,
    gaussianCount,
    centroid,
    bbox,
  };
}

function compactObjectContinuity(summary) {
  if (!summary || typeof summary !== "object") return null;
  return {
    schema: "objgauss-object-continuity-summary-v1",
    status: cleanString(summary.status || "none"),
    spatialCompactness: finiteNumber(summary.spatialCompactness),
    bboxDiagonal: finiteNumber(summary.bboxDiagonal),
    gaussianDensity: finiteNumber(summary.gaussianDensity),
    centroidContained: Boolean(summary.centroidContained),
    bboxValid: Boolean(summary.bboxValid),
    gaussianCount: finiteNumber(summary.gaussianCount),
    centroid: cleanNumberArray(summary.centroid),
    bbox: cleanNumberArray(summary.bbox),
  };
}

function objectTemporalSummary(object) {
  if (!object) {
    return {
      schema: "objgauss-object-temporal-summary-v1",
      status: "none",
      temporalDrift: null,
      assignmentJitter: null,
      bboxStability: null,
      temporalAvailable: false,
      jitterAvailable: false,
      bboxAvailable: false,
      stable: false,
      thresholds: objectTemporalThresholds(),
    };
  }
  const state = object?.objectState ?? object;
  const temporalDrift = finiteNumber(state?.temporalDrift ?? object?.temporalDrift ?? state?.drift ?? state?.centroidDrift);
  const assignmentJitter = finiteNumber(
    state?.assignmentJitter ?? object?.assignmentJitter ?? state?.jitter ?? state?.assignmentDrift,
  );
  const bboxStability = finiteNumber(
    state?.bboxStability ?? object?.bboxStability ?? state?.bboxIoU ?? state?.bboxConvergence,
  );
  const thresholds = objectTemporalThresholds();
  const temporalAvailable = temporalDrift !== null;
  const jitterAvailable = assignmentJitter !== null;
  const bboxAvailable = bboxStability !== null;
  const status =
    !temporalAvailable && !jitterAvailable && !bboxAvailable
      ? "unavailable"
      : jitterAvailable && assignmentJitter > thresholds.assignmentJitter
        ? "assignment-jitter"
        : temporalAvailable && temporalDrift > thresholds.temporalDrift
          ? "temporal-drift"
          : bboxAvailable && bboxStability < thresholds.bboxStability
            ? "bbox-unstable"
            : "stable";
  return {
    schema: "objgauss-object-temporal-summary-v1",
    status,
    temporalDrift,
    assignmentJitter,
    bboxStability,
    temporalAvailable,
    jitterAvailable,
    bboxAvailable,
    stable: status === "stable",
    thresholds,
  };
}

function compactObjectTemporal(summary) {
  if (!summary || typeof summary !== "object") return null;
  return {
    schema: "objgauss-object-temporal-summary-v1",
    status: cleanString(summary.status || "none"),
    temporalDrift: finiteNumber(summary.temporalDrift),
    assignmentJitter: finiteNumber(summary.assignmentJitter),
    bboxStability: finiteNumber(summary.bboxStability),
    temporalAvailable: Boolean(summary.temporalAvailable),
    jitterAvailable: Boolean(summary.jitterAvailable),
    bboxAvailable: Boolean(summary.bboxAvailable),
    stable: Boolean(summary.stable),
    thresholds: objectTemporalThresholds(summary.thresholds),
  };
}

function objectTemporalThresholds(overrides = {}) {
  return {
    temporalDrift: finiteNumber(overrides.temporalDrift) ?? 0.08,
    assignmentJitter: finiteNumber(overrides.assignmentJitter) ?? 0.08,
    bboxStability: finiteNumber(overrides.bboxStability) ?? 0.5,
  };
}

function objectExplainabilitySummary({ object, assignmentProbe, continuity, temporal } = {}) {
  if (!object) {
    return {
      schema: "objgauss-object-explainability-summary-v1",
      status: "none",
      explainable: false,
      score: null,
      reasonNames: "",
      reasons: [],
      assignmentConfidence: null,
      assignmentMargin: null,
      assignmentEntropy: null,
      continuityStatus: continuity?.status ?? "none",
      temporalStatus: temporal?.status ?? "none",
    };
  }
  const state = object?.objectState ?? object;
  const confidence = finiteNumber(assignmentProbe?.confidence ?? state?.confidence ?? object?.confidence);
  const margin = finiteNumber(assignmentProbe?.margin);
  const entropy = finiteNumber(assignmentProbe?.entropy ?? state?.assignmentEntropy ?? object?.entropy);
  const continuityStatus = cleanString(continuity?.status ?? "none");
  const temporalStatus = cleanString(temporal?.status ?? "none");
  const reasons = [];
  if (assignmentProbe?.collapseRisk) reasons.push("assignment_collapse_risk");
  if (assignmentProbe?.ambiguous) reasons.push("assignment_ambiguous");
  if (confidence !== null && confidence < 0.7) reasons.push("low_assignment_confidence");
  if (margin !== null && margin < 0.45) reasons.push("low_assignment_margin");
  if (continuityStatus && !["continuous", "none"].includes(continuityStatus)) {
    reasons.push(`spatial_${continuityStatus}`);
  }
  if (continuity && continuity.centroidContained === false && continuityStatus !== "none") {
    reasons.push("centroid_outside_bbox");
  }
  if (temporalStatus && !["stable", "unavailable", "none"].includes(temporalStatus)) {
    reasons.push(`temporal_${temporalStatus}`);
  }
  const uniqueReasons = [...new Set(reasons)];
  const explainable = uniqueReasons.length === 0;
  const status = explainable
    ? temporalStatus === "unavailable"
      ? "explainable-static"
      : "explainable"
    : uniqueReasons[0];
  const scoreParts = [
    confidence,
    margin,
    entropy !== null ? 1 - Math.max(0, Math.min(1, entropy)) : null,
    continuityStatus === "continuous" ? 1 : continuityStatus === "none" ? null : 0,
    temporal?.stable ? 1 : temporalStatus === "unavailable" || temporalStatus === "none" ? null : 0,
  ].filter((value) => Number.isFinite(value));
  const score = scoreParts.length ? round3(average(scoreParts)) : null;
  return {
    schema: "objgauss-object-explainability-summary-v1",
    status,
    explainable,
    score,
    reasonNames: uniqueReasons.join(","),
    reasons: uniqueReasons,
    assignmentConfidence: confidence,
    assignmentMargin: margin,
    assignmentEntropy: entropy,
    continuityStatus,
    temporalStatus,
  };
}

function compactObjectExplainability(summary) {
  if (!summary || typeof summary !== "object") return null;
  const reasons = cleanStringList(summary.reasons).slice(0, 8);
  const reasonNames = reasons.length ? reasons.join(",") : cleanString(summary.reasonNames);
  return {
    schema: "objgauss-object-explainability-summary-v1",
    status: cleanString(summary.status || "none"),
    explainable: Boolean(summary.explainable),
    score: finiteNumber(summary.score),
    reasonNames,
    reasons,
    assignmentConfidence: finiteNumber(summary.assignmentConfidence),
    assignmentMargin: finiteNumber(summary.assignmentMargin),
    assignmentEntropy: finiteNumber(summary.assignmentEntropy),
    continuityStatus: cleanString(summary.continuityStatus),
    temporalStatus: cleanString(summary.temporalStatus),
  };
}

function verdictReasonRows(summary) {
  if (!summary || typeof summary !== "object" || cleanString(summary.status) === "none") {
    return [{ name: "no_object", value: "none", status: "warn" }];
  }
  const reasons = cleanStringList(summary.reasons).slice(0, 8);
  if (!reasons.length) {
    return [{ name: "clear", value: formatRatio(summary.score), status: "pass" }];
  }
  return reasons.map((reason) => ({
    name: reason,
    value: cleanString(summary.status),
    status: "warn",
  }));
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
  objectOverlayMode,
  debugProbe,
  hoveredTarget,
  hoverAssignmentProbe,
  objectContinuity,
  hoverContinuity,
  objectTemporal,
  hoverTemporal,
  objectExplainability,
  hoverExplainability,
  hiddenCount,
  objectVisibility,
  stability,
  assignmentSource,
  assignmentProbe,
  trainingEvidence,
  qualityReport,
  objectStateBenchmark,
  objectStateBenchmarkCase,
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
      overlayMode: normalizeObjectOverlayMode(objectOverlayMode),
      overlayBboxVisible: Boolean(debugMode && objectOverlayShows(objectOverlayMode, "bbox")),
      overlayCentroidVisible: Boolean(debugMode && objectOverlayShows(objectOverlayMode, "centroid")),
      probeSource: debugProbe?.source ?? activeState?.source ?? "none",
    },
    assignment: {
      source: assignmentSource || activeState?.source || selected?.delivery?.source || "none",
      slotCount: Array.isArray(assignment) ? assignment.length : 0,
      confidence: finiteNumber(debugProbe?.confidence ?? activeState?.confidence),
      entropy: finiteNumber(debugProbe?.entropy ?? activeState?.assignmentEntropy),
      vector: compactAssignmentVector(assignment),
      probe: compactAssignmentProbe(assignmentProbe),
    },
    hover: hoveredTarget
      ? {
          modelId: cleanString(hoveredTarget.modelId),
          objectId: cleanNullable(hoveredTarget.objectId),
          selectionId: cleanString(hoveredTarget.selectionId),
          gaussianCount: finiteNumber(hoveredTarget.gaussianCount),
          source: cleanString(hoveredTarget.assignmentSource),
          confidence: finiteNumber(hoveredTarget.confidence),
          entropy: finiteNumber(hoveredTarget.entropy),
          status: cleanString(hoveredTarget.status),
          assignment: compactAssignmentVector(hoveredTarget.assignment),
          probe: compactAssignmentProbe(hoverAssignmentProbe),
          continuity: compactObjectContinuity(hoverContinuity),
          temporal: compactObjectTemporal(hoverTemporal),
          explainability: compactObjectExplainability(hoverExplainability),
        }
      : null,
    visibility: {
      hiddenObjectCount: finiteNumber(objectVisibility?.hiddenObjectCount ?? hiddenCount),
      visibleObjectCount: finiteNumber(objectVisibility?.visibleObjectCount),
      hiddenGaussianCount: finiteNumber(objectVisibility?.hiddenGaussianCount),
      visibleGaussianCount: finiteNumber(objectVisibility?.visibleGaussianCount),
      hiddenSelectionIds: Array.isArray(objectVisibility?.hiddenSelectionIds)
        ? objectVisibility.hiddenSelectionIds.slice(0, 16)
        : [],
    },
    continuity: compactObjectContinuity(objectContinuity),
    temporal: compactObjectTemporal(objectTemporal),
    explainability: compactObjectExplainability(objectExplainability),
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
    quality: qualityReport
      ? {
          schema: qualityReport.schema,
          status: qualityReport.status,
          assignmentEntropy: qualityReport.assignmentEntropy,
          objectPurity: qualityReport.objectPurity,
          temporalDrift: qualityReport.temporalDrift,
          assignmentJitter: qualityReport.assignmentJitter,
          gateCount: qualityReport.gateCount,
          failingGates: qualityReport.failingGates,
          failingGateNames: qualityReport.failingGateNames,
          gates: compactQualityGates(qualityReport.gates),
        }
      : null,
    benchmark: objectStateBenchmark
      ? {
          schema: objectStateBenchmark.schema,
          status: objectStateBenchmark.status,
          caseCount: objectStateBenchmark.caseCount,
          warnCount: objectStateBenchmark.warnCount,
          observedWarnCount: objectStateBenchmark.observedWarnCount,
          failureModeCount: objectStateBenchmark.failureModeCount,
          activeCase: objectStateBenchmarkCase
            ? {
                name: objectStateBenchmarkCase.name,
                status: objectStateBenchmarkCase.status,
                observedStatus: objectStateBenchmarkCase.observedStatus,
                failureModeNames: objectStateBenchmarkCase.failureModeNames,
                diagnosticNames: objectStateBenchmarkCase.diagnosticNames,
                assignmentConfidence: objectStateBenchmarkCase.assignmentConfidence,
                meanEntropy: objectStateBenchmarkCase.meanEntropy,
                objectPurity: objectStateBenchmarkCase.objectPurity,
                meanTemporalDrift: objectStateBenchmarkCase.meanTemporalDrift,
                dynamicProposalCount: objectStateBenchmarkCase.dynamicProposalCount,
              }
            : null,
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

function objectStateDebugSession({ snapshot, models, debugEvents }) {
  const compactModels = Array.isArray(models) ? models.map(compactDebugModel) : [];
  const loadedModels = compactModels.filter((model) => model.status === "loaded" || model.status === "compressed");
  const trainableModels = compactModels.filter((model) => model.deliverySource === "trainable-kernel-model-artifact");
  const ogcModels = compactModels.filter((model) => model.deliverySource === "quantized-ogc");
  return {
    schema: "objgauss-object-state-debug-session-v1",
    protocol: "object-state-debug-os-v1",
    snapshot,
    summary: {
      modelCount: compactModels.length,
      loadedModelCount: loadedModels.length,
      trainableArtifactCount: trainableModels.length,
      ogcArtifactCount: ogcModels.length,
      eventCount: Array.isArray(debugEvents) ? Math.min(debugEvents.length, DEBUG_EVENT_LIMIT) : 0,
    },
    models: compactModels,
    events: compactDebugEvents(debugEvents),
    exportPolicy: {
      scope: "browser-local-download-only",
      repositoryWrite: "none",
      trainingOutputs: "not_committed",
      payloadPolicy: "summaries_only",
    },
  };
}

function validateDebugSessionArchive(session, path = "") {
  if (session?.schema !== "objgauss-object-state-debug-session-v1") {
    throw new Error("unsupported ObjectState debug session schema");
  }
  if (session.protocol !== "object-state-debug-os-v1") {
    throw new Error("unsupported ObjectState debug session protocol");
  }
  const snapshot = session.snapshot;
  if (snapshot?.schema !== "objgauss-object-state-debug-snapshot-v1") {
    throw new Error("debug session missing ObjectState snapshot");
  }
  const models = Array.isArray(session.models) ? session.models.map(compactDebugModel) : [];
  const events = compactDebugEvents(Array.isArray(session.events) ? session.events : snapshot.events);
  return {
    schema: session.schema,
    protocol: session.protocol,
    path,
    snapshot: {
      schema: snapshot.schema,
      protocol: snapshot.protocol ?? session.protocol,
      model: {
        id: cleanString(snapshot.model?.id),
        kind: cleanString(snapshot.model?.kind),
        loadMode: cleanString(snapshot.model?.loadMode),
        status: cleanString(snapshot.model?.status),
        deliverySource: cleanString(snapshot.model?.deliverySource),
      },
      selection: {
        selectionId: cleanString(snapshot.selection?.selectionId),
        objectId: cleanNullable(snapshot.selection?.objectId),
        gaussianIndex: cleanNullable(snapshot.selection?.gaussianIndex),
        hiddenObjectCount: finiteNumber(snapshot.selection?.hiddenObjectCount),
      },
      debug: {
        enabled: Boolean(snapshot.debug?.enabled),
        lens: normalizeDebugLens(snapshot.debug?.lens),
        overlayMode: normalizeObjectOverlayMode(snapshot.debug?.overlayMode),
        overlayBboxVisible: Boolean(snapshot.debug?.overlayBboxVisible),
        overlayCentroidVisible: Boolean(snapshot.debug?.overlayCentroidVisible),
        probeSource: cleanString(snapshot.debug?.probeSource),
      },
      assignment: {
        source: cleanString(snapshot.assignment?.source),
        slotCount: finiteNumber(snapshot.assignment?.slotCount),
        confidence: finiteNumber(snapshot.assignment?.confidence),
        entropy: finiteNumber(snapshot.assignment?.entropy),
        vector: compactAssignmentVector(snapshot.assignment?.vector),
        probe: compactAssignmentProbe(snapshot.assignment?.probe),
      },
      hover: snapshot.hover
        ? {
            modelId: cleanString(snapshot.hover.modelId),
            objectId: cleanNullable(snapshot.hover.objectId),
            selectionId: cleanString(snapshot.hover.selectionId),
            gaussianCount: finiteNumber(snapshot.hover.gaussianCount),
            source: cleanString(snapshot.hover.source),
            confidence: finiteNumber(snapshot.hover.confidence),
            entropy: finiteNumber(snapshot.hover.entropy),
            status: cleanString(snapshot.hover.status),
            assignment: compactAssignmentVector(snapshot.hover.assignment),
            probe: compactAssignmentProbe(snapshot.hover.probe),
            continuity: compactObjectContinuity(snapshot.hover.continuity),
            temporal: compactObjectTemporal(snapshot.hover.temporal),
            explainability: compactObjectExplainability(snapshot.hover.explainability),
          }
        : null,
      visibility: {
        hiddenObjectCount: finiteNumber(snapshot.visibility?.hiddenObjectCount),
        visibleObjectCount: finiteNumber(snapshot.visibility?.visibleObjectCount),
        hiddenGaussianCount: finiteNumber(snapshot.visibility?.hiddenGaussianCount),
        visibleGaussianCount: finiteNumber(snapshot.visibility?.visibleGaussianCount),
        hiddenSelectionIds: cleanStringList(snapshot.visibility?.hiddenSelectionIds).slice(0, 16),
      },
      continuity: compactObjectContinuity(snapshot.continuity),
      temporal: compactObjectTemporal(snapshot.temporal),
      explainability: compactObjectExplainability(snapshot.explainability),
      stability: {
        status: cleanString(snapshot.stability?.status),
        slotUtilization: finiteNumber(snapshot.stability?.slotUtilization),
        meanEntropy: finiteNumber(snapshot.stability?.meanEntropy),
        mixedSlots: finiteNumber(snapshot.stability?.mixedSlots),
      },
      training: snapshot.training
        ? {
            status: cleanString(snapshot.training.status),
            iterations: finiteNumber(snapshot.training.iterations),
            finalTotalLoss: finiteNumber(snapshot.training.finalTotalLoss),
            totalLossDelta: finiteNumber(snapshot.training.totalLossDelta),
            finalImageLoss: finiteNumber(snapshot.training.finalImageLoss),
            imageLossDelta: finiteNumber(snapshot.training.imageLossDelta),
            rendererName: cleanString(snapshot.training.rendererName),
            gradientPath: cleanString(snapshot.training.gradientPath),
          }
        : null,
      quality: snapshot.quality
        ? {
            schema: cleanString(snapshot.quality.schema),
            status: cleanString(snapshot.quality.status),
            assignmentEntropy: finiteNumber(snapshot.quality.assignmentEntropy),
            objectPurity: finiteNumber(snapshot.quality.objectPurity),
            temporalDrift: finiteNumber(snapshot.quality.temporalDrift),
            assignmentJitter: finiteNumber(snapshot.quality.assignmentJitter),
            gateCount: finiteNumber(snapshot.quality.gateCount),
            failingGates: finiteNumber(snapshot.quality.failingGates),
            failingGateNames: Array.isArray(snapshot.quality.failingGateNames)
              ? snapshot.quality.failingGateNames.map(cleanString).filter(Boolean)
              : [],
            gates: compactQualityGates(snapshot.quality.gates),
          }
        : null,
      benchmark: snapshot.benchmark
        ? {
            schema: cleanString(snapshot.benchmark.schema),
            status: cleanString(snapshot.benchmark.status),
            caseCount: finiteNumber(snapshot.benchmark.caseCount),
            warnCount: finiteNumber(snapshot.benchmark.warnCount),
            observedWarnCount: finiteNumber(snapshot.benchmark.observedWarnCount),
            failureModeCount: finiteNumber(snapshot.benchmark.failureModeCount),
            activeCase: snapshot.benchmark.activeCase
              ? {
                  name: cleanString(snapshot.benchmark.activeCase.name),
                  status: cleanString(snapshot.benchmark.activeCase.status),
                  observedStatus: cleanString(snapshot.benchmark.activeCase.observedStatus),
                  failureModeNames: cleanString(snapshot.benchmark.activeCase.failureModeNames),
                  diagnosticNames: cleanString(snapshot.benchmark.activeCase.diagnosticNames),
                  assignmentConfidence: finiteNumber(snapshot.benchmark.activeCase.assignmentConfidence),
                  meanEntropy: finiteNumber(snapshot.benchmark.activeCase.meanEntropy),
                  objectPurity: finiteNumber(snapshot.benchmark.activeCase.objectPurity),
                  meanTemporalDrift: finiteNumber(snapshot.benchmark.activeCase.meanTemporalDrift),
                  dynamicProposalCount: finiteNumber(snapshot.benchmark.activeCase.dynamicProposalCount),
                }
              : null,
          }
        : null,
      delivery: {
        loadRoute: cleanString(snapshot.delivery?.loadRoute),
        frameIndex: cleanNullable(snapshot.delivery?.frameIndex),
        frameCount: cleanNullable(snapshot.delivery?.frameCount),
        lodLevel: cleanNullable(snapshot.delivery?.lodLevel),
        chunkIds: Array.isArray(snapshot.delivery?.chunkIds) ? snapshot.delivery.chunkIds.slice(0, 16) : [],
      },
      events,
    },
    summary: {
      modelCount: finiteNumber(session.summary?.modelCount) ?? models.length,
      loadedModelCount: finiteNumber(session.summary?.loadedModelCount),
      trainableArtifactCount: finiteNumber(session.summary?.trainableArtifactCount),
      ogcArtifactCount: finiteNumber(session.summary?.ogcArtifactCount),
      eventCount: finiteNumber(session.summary?.eventCount) ?? events.length,
    },
    models,
    events,
    exportPolicy: {
      scope: cleanString(session.exportPolicy?.scope),
      repositoryWrite: cleanString(session.exportPolicy?.repositoryWrite),
      trainingOutputs: cleanString(session.exportPolicy?.trainingOutputs),
      payloadPolicy: cleanString(session.exportPolicy?.payloadPolicy),
    },
  };
}

function debugSessionSnapshotDiff(liveSnapshot, archiveSnapshot) {
  if (!liveSnapshot || !archiveSnapshot) return null;
  const modelMatch = cleanString(liveSnapshot.model?.id) === cleanString(archiveSnapshot.model?.id);
  const objectMatch = String(liveSnapshot.selection?.objectId ?? "") === String(archiveSnapshot.selection?.objectId ?? "");
  const sourceMatch = cleanString(liveSnapshot.assignment?.source) === cleanString(archiveSnapshot.assignment?.source);
  const qualityMatch = cleanString(liveSnapshot.quality?.status) === cleanString(archiveSnapshot.quality?.status);
  const trainingMatch = cleanString(liveSnapshot.training?.status) === cleanString(archiveSnapshot.training?.status);
  const stabilityMatch = cleanString(liveSnapshot.stability?.status) === cleanString(archiveSnapshot.stability?.status);
  const temporalMatch = cleanString(liveSnapshot.temporal?.status) === cleanString(archiveSnapshot.temporal?.status);
  const explainabilityMatch =
    cleanString(liveSnapshot.explainability?.status) === cleanString(archiveSnapshot.explainability?.status);
  const deliveryMatch = cleanString(liveSnapshot.delivery?.loadRoute) === cleanString(archiveSnapshot.delivery?.loadRoute);
  const probeStatusMatch =
    cleanString(liveSnapshot.assignment?.probe?.status) === cleanString(archiveSnapshot.assignment?.probe?.status);
  const slotDelta = numericDelta(liveSnapshot.assignment?.slotCount, archiveSnapshot.assignment?.slotCount);
  const entropyDelta = numericDelta(liveSnapshot.assignment?.entropy, archiveSnapshot.assignment?.entropy);
  const confidenceDelta = numericDelta(liveSnapshot.assignment?.confidence, archiveSnapshot.assignment?.confidence);
  const probeMarginDelta = numericDelta(
    liveSnapshot.assignment?.probe?.margin,
    archiveSnapshot.assignment?.probe?.margin,
  );
  const qualityEntropyDelta = numericDelta(
    liveSnapshot.quality?.assignmentEntropy,
    archiveSnapshot.quality?.assignmentEntropy,
  );
  const eventDelta = numericDelta(
    Array.isArray(liveSnapshot.events) ? liveSnapshot.events.length : 0,
    Array.isArray(archiveSnapshot.events) ? archiveSnapshot.events.length : 0,
  );
  const changedFields = [
    !modelMatch ? "model" : "",
    !objectMatch ? "object" : "",
    !sourceMatch ? "source" : "",
    !qualityMatch ? "quality" : "",
    !trainingMatch ? "training" : "",
    !stabilityMatch ? "stability" : "",
    !temporalMatch ? "temporal" : "",
    !explainabilityMatch ? "explainability" : "",
    !deliveryMatch ? "delivery" : "",
    !probeStatusMatch ? "probe_status" : "",
    deltaChanged(slotDelta) ? "slots" : "",
    deltaChanged(entropyDelta) ? "entropy" : "",
    deltaChanged(confidenceDelta) ? "confidence" : "",
    deltaChanged(probeMarginDelta) ? "probe_margin" : "",
    deltaChanged(qualityEntropyDelta) ? "quality_entropy" : "",
  ].filter(Boolean);
  return {
    status: changedFields.length ? "changed" : "match",
    modelMatch,
    objectMatch,
    sourceMatch,
    qualityMatch,
    trainingMatch,
    stabilityMatch,
    temporalMatch,
    explainabilityMatch,
    deliveryMatch,
    probeStatusMatch,
    slotDelta,
    entropyDelta,
    confidenceDelta,
    probeMarginDelta,
    qualityEntropyDelta,
    eventDelta,
    changedFields,
    changedFieldNames: changedFields.join(","),
  };
}

function numericDelta(liveValue, archiveValue) {
  const live = finiteNumber(liveValue);
  const archive = finiteNumber(archiveValue);
  if (live === null && archive === null) return 0;
  if (live === null || archive === null) return null;
  return live - archive;
}

function deltaChanged(delta) {
  return delta === null || Math.abs(Number(delta)) > 0.0005;
}

function compactDebugModel(model) {
  return {
    id: cleanString(model?.id),
    label: cleanString(model?.label),
    name: cleanString(model?.name),
    kind: cleanString(model?.kind),
    loadMode: cleanString(model?.loadMode),
    status: cleanString(model?.status),
    gaussianCount: finiteNumber(model?.gaussianCount),
    objectCount: finiteNumber(model?.objectCount),
    deliverySource: cleanString(model?.delivery?.source),
    loadRoute: cleanString(model?.delivery?.loadRoute),
    artifactPath: cleanString(model?.delivery?.artifactPath),
    indexPath: cleanString(model?.delivery?.indexPath),
    payloadPath: cleanString(model?.delivery?.payloadPath),
    frameIndex: cleanNullable(model?.delivery?.frameIndex),
    frameCount: cleanNullable(model?.delivery?.frameCount),
    lodLevel: cleanNullable(model?.delivery?.lodLevel),
    chunkIds: Array.isArray(model?.delivery?.chunkIds) ? model.delivery.chunkIds.slice(0, 16) : [],
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
  if (event.type === "object-overlay") return event.source || "-";
  if (event.type === "frame-select") return event.frameIndex === null ? "frame -" : `f${event.frameIndex}`;
  if (event.type === "gaussian-probe") return event.gaussianIndex === null ? "G -" : `G${event.gaussianIndex}`;
  if (event.type === "toggle-visibility") return event.visible ? "visible" : "hidden";
  if (event.type === "ogc-lod") return event.lodLevel === null ? "LOD -" : `L${event.lodLevel}`;
  if (event.type === "ogc-chunks") return event.chunkScope || "all";
  if (
    event.type === "import-artifact" ||
    event.type === "import-artifact-error" ||
    event.type === "import-model-manifest" ||
    event.type === "import-model-manifest-error" ||
    event.type === "import-ogc" ||
    event.type === "import-ogc-error" ||
    event.type === "import-session" ||
    event.type === "import-session-error" ||
    event.type === "export-snapshot" ||
    event.type === "export-snapshot-error" ||
    event.type === "export-session" ||
    event.type === "export-session-error"
  ) return event.fileName || "local";
  if (event.objectId !== null && event.objectId !== undefined) return `#${event.objectId}`;
  return event.modelId || event.source || "-";
}

function debugSnapshotExportFileName(snapshot) {
  const model = sanitizeFileSegment(snapshot?.model?.id || "model");
  const object = sanitizeFileSegment(snapshot?.selection?.objectId ?? "scene");
  const gaussian = snapshot?.selection?.gaussianIndex === null || snapshot?.selection?.gaussianIndex === undefined
    ? "g-none"
    : `g-${sanitizeFileSegment(snapshot.selection.gaussianIndex)}`;
  return `objgauss-debug-snapshot-${model}-object-${object}-${gaussian}.json`;
}

function debugSessionExportFileName(session) {
  const model = sanitizeFileSegment(session?.snapshot?.model?.id || "model");
  const object = sanitizeFileSegment(session?.snapshot?.selection?.objectId ?? "scene");
  const events = sanitizeFileSegment(session?.summary?.eventCount ?? 0);
  return `objgauss-debug-session-${model}-object-${object}-events-${events}.json`;
}

function sanitizeFileSegment(value) {
  const segment = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return segment || "unknown";
}

function compactAssignmentVector(assignment) {
  if (!Array.isArray(assignment)) return [];
  return assignment.slice(0, 16).map((slot) => ({
    slot: Number.isFinite(Number(slot?.slot)) ? Number(slot.slot) : null,
    objectId: slot?.objectId ?? null,
    probability: finiteNumber(slot?.probability),
  }));
}

function assignmentProbeSummary(assignment, debugProbe = null) {
  const rows = Array.isArray(assignment)
    ? assignment
        .map((slot, index) => ({
          slot: Number.isFinite(Number(slot?.slot)) ? Number(slot.slot) : index,
          objectId: cleanNullable(slot?.objectId),
          probability: Math.max(0, Math.min(1, Number(slot?.probability) || 0)),
        }))
        .filter((slot) => Number.isFinite(slot.probability))
    : [];
  const sorted = [...rows].sort((left, right) => right.probability - left.probability);
  const top = sorted[0] ?? null;
  const second = sorted[1] ?? null;
  const topProbability = top ? round3(top.probability) : null;
  const secondProbability = second ? round3(second.probability) : null;
  const margin =
    top && second
      ? round3(top.probability - second.probability)
      : top
        ? round3(top.probability)
        : null;
  const entropy = finiteNumber(debugProbe?.entropy) ?? round3(normalizedEntropy(rows.map((slot) => slot.probability)));
  const confidence = finiteNumber(debugProbe?.confidence) ?? topProbability;
  const ambiguous = Boolean(
    rows.length > 1 &&
      ((margin !== null && margin < 0.2) || (entropy > 0.72 && (margin === null || margin < 0.35))),
  );
  const collapseRisk = Boolean(rows.length > 1 && topProbability !== null && topProbability >= 0.98 && entropy <= 0.05);
  const status = !rows.length
    ? "none"
    : collapseRisk
      ? "collapse-risk"
      : ambiguous
        ? "ambiguous"
        : topProbability !== null && topProbability >= 0.7 && margin !== null && margin >= 0.35
          ? "confident"
          : "soft";
  return {
    status,
    slotCount: rows.length,
    topSlot: top?.slot ?? null,
    topObjectId: top?.objectId ?? null,
    topProbability,
    secondProbability,
    margin,
    entropy: finiteNumber(entropy),
    confidence: finiteNumber(confidence),
    ambiguous,
    collapseRisk,
    gaussianIndex: cleanNullable(debugProbe?.gaussianIndex),
  };
}

function compactAssignmentProbe(probe) {
  if (!probe || typeof probe !== "object") return null;
  return {
    status: cleanString(probe.status || "none"),
    slotCount: finiteNumber(probe.slotCount),
    topSlot: cleanNullable(probe.topSlot),
    topObjectId: cleanNullable(probe.topObjectId),
    topProbability: finiteNumber(probe.topProbability),
    secondProbability: finiteNumber(probe.secondProbability),
    margin: finiteNumber(probe.margin),
    entropy: finiteNumber(probe.entropy),
    confidence: finiteNumber(probe.confidence),
    ambiguous: Boolean(probe.ambiguous),
    collapseRisk: Boolean(probe.collapseRisk),
    gaussianIndex: cleanNullable(probe.gaussianIndex),
  };
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

function qualityReportSummary(report) {
  if (report?.schema !== "objgauss-object-state-quality-report-v1") return null;
  const metrics = report.metrics ?? {};
  const gates = compactQualityGates(report.gates);
  const failingGates = gates.filter((gate) => gate?.status && gate.status !== "pass").length;
  const passingGates = gates.filter((gate) => gate?.status === "pass").length;
  const failingGateNames = gates
    .filter((gate) => gate.status && gate.status !== "pass")
    .map((gate) => gate.name)
    .join(",");
  return {
    schema: report.schema,
    status: report.status ?? (failingGates ? "warn" : "pass"),
    assignmentEntropy: finiteNumber(metrics.assignment_entropy ?? metrics.mean_entropy),
    slotUtilization: finiteNumber(metrics.slot_utilization),
    objectPurity: finiteNumber(metrics.object_purity ?? metrics.mean_purity),
    temporalDrift: finiteNumber(metrics.temporal_drift ?? metrics.mean_temporal_drift),
    assignmentJitter: finiteNumber(metrics.assignment_jitter ?? metrics.mean_assignment_jitter),
    bboxStability: finiteNumber(metrics.bbox_stability ?? metrics.mean_bbox_stability),
    spatialCompactness: finiteNumber(metrics.spatial_compactness ?? metrics.mean_spatial_compactness),
    gateCount: gates.length,
    passingGates,
    failingGates,
    failingGateNames,
    gates,
    path: report.path ?? "",
  };
}

function objectStateBenchmarkSummary(report) {
  if (report?.schema !== OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA) return null;
  const aggregate = report.aggregate ?? {};
  const failureModes = Array.isArray(aggregate.failure_mode_coverage)
    ? aggregate.failure_mode_coverage.filter(Boolean)
    : [];
  const cases = compactObjectStateBenchmarkCases(report.cases);
  return {
    schema: report.schema,
    reportId: cleanString(report.report_id || "object-state-benchmark"),
    status: cleanString(report.status || "unknown"),
    caseCount: finiteNumber(aggregate.case_count ?? cases.length) ?? cases.length,
    warnCount: finiteNumber(aggregate.warn_count) ?? 0,
    observedWarnCount: finiteNumber(aggregate.observed_warn_count) ?? 0,
    failureModeCount: failureModes.length,
    failureModes: failureModes.slice(0, 16),
    cases,
    path: report.path ?? "",
  };
}

function activeObjectStateBenchmarkCase(summary, selectedName) {
  const cases = Array.isArray(summary?.cases) ? summary.cases : [];
  if (!cases.length) return null;
  const requested = cleanString(selectedName);
  return (
    (requested ? cases.find((testCase) => testCase.name === requested) : null) ??
    cases.find((testCase) => testCase.observedStatus && testCase.observedStatus !== "pass") ??
    cases[0]
  );
}

function compactObjectStateBenchmarkCases(cases) {
  if (!Array.isArray(cases)) return [];
  return cases.slice(0, 8).map((testCase, index) => {
    const metrics = testCase?.metrics ?? {};
    const failureModes = cleanStringList(testCase?.failure_modes);
    const stabilityDiagnostics = cleanStringList(testCase?.stability?.diagnostics);
    const temporalDiagnostics = cleanStringList(testCase?.temporal?.diagnostics);
    const dynamicProposalKinds = cleanStringList(testCase?.dynamic_k?.proposal_kinds);
    const diagnostics = [...stabilityDiagnostics, ...temporalDiagnostics, ...dynamicProposalKinds];
    const uniqueDiagnostics = Array.from(new Set(diagnostics));
    return {
      name: cleanString(testCase?.name || `case_${index}`),
      status: cleanString(testCase?.status || "unknown"),
      observedStatus: cleanString(testCase?.observed_status || "unknown"),
      assignmentConfidence: finiteNumber(metrics.assignment_confidence),
      meanEntropy: finiteNumber(metrics.mean_normalized_entropy),
      objectPurity: finiteNumber(metrics.object_purity),
      effectiveSlots: finiteNumber(metrics.effective_slots),
      rawAssignmentJitter: finiteNumber(metrics.raw_assignment_jitter),
      meanTemporalDrift: finiteNumber(metrics.mean_temporal_drift),
      maxTemporalDrift: finiteNumber(metrics.max_temporal_drift),
      bboxDiagonalMean: finiteNumber(metrics.bbox_diagonal_mean),
      dynamicProposalCount: finiteNumber(testCase?.dynamic_k?.proposal_count) ?? 0,
      failureModes,
      failureModeCount: failureModes.length,
      failureModeNames: formatNameList(failureModes),
      stabilityDiagnostics,
      temporalDiagnostics,
      dynamicProposalKinds,
      diagnostics: uniqueDiagnostics,
      diagnosticNames: formatNameList(uniqueDiagnostics),
    };
  });
}

function cleanStringList(value) {
  return Array.isArray(value)
    ? value.map((entry) => cleanString(entry)).filter(Boolean)
    : [];
}

function formatNameList(value) {
  return cleanStringList(value).join(",");
}

function compactQualityGates(gates) {
  if (!Array.isArray(gates)) return [];
  return gates.slice(0, 8).map((gate, index) => ({
    name: cleanString(gate?.name || `gate_${index}`),
    status: cleanString(gate?.status || "unknown"),
    value: finiteNumber(gate?.value),
    threshold: finiteNumber(gate?.threshold),
  }));
}

function formatGateValue(gate) {
  const value = gate?.value === null || gate?.value === undefined ? "-" : formatRatio(gate.value);
  const threshold = gate?.threshold === null || gate?.threshold === undefined ? "-" : formatRatio(gate.threshold);
  return `${value} / ${threshold}`;
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

function formatSignedRatio(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number >= 0 ? "+" : ""}${number.toFixed(3)}`;
}

function formatSignedCount(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number >= 0 ? "+" : ""}${Math.trunc(number)}`;
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
