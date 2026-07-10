import { spawn } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";

import { chromium } from "playwright";

const DEFAULT_PORT = 5395;
const args = parseArgs(process.argv.slice(2));
const port = Number(args.port ?? DEFAULT_PORT);
const baseUrl = String(args.url ?? `http://127.0.0.1:${port}/`);
const server = args.url || args.noServer ? null : startServer(port);

try {
  await waitForApp(baseUrl);
  const viewerTruthOnly = Boolean(args["viewer-truth"]);
  const summary = viewerTruthOnly ? await auditViewerTruth(baseUrl) : await auditWorld(baseUrl);
  if (viewerTruthOnly) {
    console.log(
      [
        "world_viewer_truth=passed",
        `initialHiddenRequests=${summary.initialHiddenRequestCount}`,
        `lazyModel=${JSON.stringify(summary.lazyModelId)}`,
        `lazyRequest=${summary.lazyRequestObserved}`,
        `transformModes=${summary.transformModeCount}`,
        `sparkHiddenObjects=${summary.sparkHiddenObjects}`,
        `sparkHiddenGaussians=${summary.sparkHiddenGaussians}`,
        `cliHandoff=${summary.cliHandoffStatus}`,
        `screenshot=${summary.screenshotPath}`,
      ].join(" "),
    );
  } else {
    console.log(
    [
      "world_viewer=passed",
      `models=${summary.modelCount}`,
      `objects=${summary.objectCount}`,
      `draggableObjects=${summary.draggableObjectCount}`,
      `selectedModel=${JSON.stringify(summary.selectedModelId)}`,
      `selectedObject=${JSON.stringify(summary.selectedObjectId)}`,
      `debugOs=${summary.debugOs}`,
      `debugSnapshot=${summary.debugSnapshotSchema}`,
      `debugLens=${summary.debugLens}`,
      `objectOverlay=${summary.objectOverlayMode}`,
      `debugEvents=${summary.debugEventCount}`,
      `ogcLoaded=${summary.ogcLoadedCount}`,
      `trainableArtifacts=${summary.trainableArtifactLoadedCount}`,
      `trainableRoute=${summary.trainableArtifactLoadRoute}`,
      `trainableArtifact=${JSON.stringify(summary.trainableArtifactPath)}`,
      `trainableFrame=${summary.trainableFrameIndex}/${summary.trainableFrameCount}`,
      `trainLoss=${summary.trainableTrainingFinalLoss}`,
      `trainLossDelta=${summary.trainableTrainingLossDelta}`,
      `trainImageLoss=${summary.trainableTrainingFinalImageLoss}`,
      `solverLoss=${summary.solverTrainingFinalLoss}`,
      `solverLossDelta=${summary.solverTrainingLossDelta}`,
      `solverGpu=${summary.solverTrainingGpuUsed}`,
      `solverVramReserveGb=${summary.solverTrainingVramReserveGb}`,
      `urlArtifact=${summary.urlArtifactStatus}`,
      `urlOgc=${summary.urlOgcStatus}`,
      `urlOgcManifest=${summary.urlOgcManifestStatus}`,
      `algorithmManifest=${summary.algorithmManifestStatus}`,
      `debugSnapshotExport=${summary.debugSnapshotExportStatus}`,
      `debugSessionExport=${summary.debugSessionExportStatus}`,
      `debugSessionImport=${summary.debugSessionImportStatus}`,
      `debugSessionDiff=${summary.debugSessionDiffStatus}`,
      `debugSessionDrift=${summary.debugSessionDriftStatus}`,
      `localModelManifest=${summary.localModelManifestStatus}`,
      `localTrainableManifest=${summary.localTrainableManifestStatus}`,
      `qualityReport=${summary.qualityReportStatus}`,
      `objectStateBenchmark=${summary.objectStateBenchmarkStatus}`,
      `localArtifact=${summary.localArtifactStatus}`,
      `localOgc=${summary.localOgcStatus}`,
      `localOgcManifest=${summary.localOgcManifestStatus}`,
      `assignmentSlots=${summary.assignmentSlots}`,
      `assignmentSource=${summary.assignmentSource}`,
      `stability=${summary.stabilityStatus}`,
      `slotUtil=${summary.slotUtilization}`,
      `mixedSlots=${summary.mixedSlots}`,
      `objectContinuity=${summary.objectContinuityStatus}`,
      `objectMove=${summary.objectMoveStatus}`,
      `sourceSplatMotion=${summary.sourceSplatObjectMotionStatus}`,
      `objectTransform=${summary.objectTransformStatus}`,
      `objectTransformState=${summary.objectTransformStateStatus}`,
      `objectPick=${summary.objectPickStatus}`,
      `trainingStage=${summary.trainingStageStatus}`,
      `gaussianDisplay=${summary.gaussianDisplayMode}`,
      `sourceSplat=${summary.sourceSplatStatus}`,
      `sourceSplatReady=${summary.sourceSplatReadyCount}`,
      `fullGaussianSources=${summary.fullGaussianSourceCount}`,
      `objectLayers=${summary.objectLayerLoadedCount}/${summary.objectLayerRegisteredCount}`,
      `stageModels=${summary.stageVisibleModelCount}/${summary.modelCount}`,
      `stageProcessed=${summary.stageProcessedModelCount}`,
      `objectFragmentation=${summary.objectFragmentationStatus}`,
      `objectTemporal=${summary.objectTemporalStatus}`,
      `objectExplainability=${summary.objectExplainabilityStatus}`,
      `objectVerdict=${summary.objectVerdictStatus}`,
      `purity=${summary.meanPurity}`,
      `temporalDrift=${summary.meanTemporalDrift}`,
      `compactness=${summary.meanSpatialCompactness}`,
      `assignmentJitter=${summary.meanAssignmentJitter}`,
      `bboxStability=${summary.meanBboxStability}`,
      `hoverContinuity=${summary.hoveredContinuityStatus}`,
      `hoverTemporal=${summary.hoveredTemporalStatus}`,
      `hoverExplainability=${summary.hoveredExplainabilityStatus}`,
      `hoverVerdict=${summary.hoverVerdictStatus}`,
      `hoveredObject=${JSON.stringify(summary.hoveredObjectId)}`,
      `hoveredGaussians=${summary.hoveredGaussianCount}`,
      `gaussianProbe=${summary.probePanelStatus}`,
      `gaussianProbeSource=${summary.probePanelSource}`,
      `gaussianProbeMargin=${summary.probePanelMargin}`,
      `assignmentTimeline=${summary.assignmentTimelineStatus}`,
      `assignmentTimelineJitter=${summary.assignmentTimelineJitter}`,
      `selectedGaussian=${JSON.stringify(summary.selectedGaussian)}`,
      `sidebars=${summary.sidebars}`,
      `screenshot=${summary.screenshotPath}`,
      `mobileScreenshot=${summary.mobileScreenshotPath}`,
    ].join(" "),
    );
  }
} finally {
  if (server) stopServer(server);
}

async function auditViewerTruth(url) {
  const browser = await chromium.launch(launchOptions());
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const requests = [];
  const pageErrors = [];
  page.on("request", (request) => requests.push(new URL(request.url()).pathname));
  page.on("pageerror", (error) => pageErrors.push(error.message));

  const hiddenCatalogPaths = [
    "/samples/plush_v1_objects.ply",
    "/samples/room_objects.ply",
    "/samples/room.splat",
    "/samples/train_objects.ply",
    "/samples/train.splat",
    "/models/trainable-mvp-debug/model-artifact.json",
  ];

  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const world = window.__OBJGAUSS_WORLD__;
      return (
        Number(shell?.getAttribute("data-loaded-count") ?? 0) > 0 &&
        Number(world?.modelCount ?? 0) > 0 &&
        Array.isArray(world?.sourceSplatObjectMotionSamples)
      );
    }, undefined, { timeout: 30000 });
    await page.waitForTimeout(750);

    const initialHiddenRequests = requests.filter((path) => hiddenCatalogPaths.includes(path));
    if (initialHiddenRequests.length > 0) {
      throw new Error(`hidden catalog assets fetched before selection: ${initialHiddenRequests.join(", ")}`);
    }

    const sourceTarget = await page.waitForFunction(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const sample = (world?.sourceSplatObjectMotionSamples ?? []).find(
        (entry) =>
          entry.status === "ready" &&
          entry.countMatches === true &&
          Number(entry.mappedGaussians) > 0,
      );
      const object = (world?.objectSelections ?? []).find(
        (entry) => entry.modelId === sample?.modelId && entry.visible,
      );
      if (!sample || !object) return null;
      return { sample, object };
    }, undefined, { timeout: 30000 }).then((handle) => handle.jsonValue());

    const selected = await page.evaluate((selectionId) => {
      return window.__OBJGAUSS_WORLD__?.selectObjectForAudit?.(selectionId) ?? false;
    }, sourceTarget.object.selectionId);
    if (!selected) throw new Error("could not select Spark-bound object for truth audit");

    const transformTruth = await page.waitForFunction(() => {
      const layer = document.querySelector("[data-object-interaction-layer='true']");
      const buttons = layer?.querySelectorAll("[data-object-transform-mode-button]") ?? [];
      const truth = layer?.querySelector("[data-object-transform-truth='translation-only']");
      if (
        layer?.getAttribute("data-object-transform-capabilities") !== "translate-xz" ||
        layer?.getAttribute("data-source-splat-translation-ready") !== "true" ||
        buttons.length !== 1 ||
        buttons[0]?.getAttribute("data-object-transform-mode-button") !== "translate" ||
        !truth?.textContent?.includes("旋转与缩放未开放")
      ) {
        return null;
      }
      return { modeCount: buttons.length, text: truth.textContent };
    }, undefined, { timeout: 15000 }).then((handle) => handle.jsonValue());

    const hidden = await page.evaluate((selectionId) => {
      return window.__OBJGAUSS_WORLD__?.toggleObjectVisibilityForAudit?.(selectionId) ?? true;
    }, sourceTarget.object.selectionId);
    if (hidden !== false) throw new Error("Spark-bound object did not enter hidden state");

    await page.waitForTimeout(250);
    const sparkVisibility = await page.evaluate((modelId) => {
      return (window.__OBJGAUSS_WORLD__?.sourceSplatObjectMotionSamples ?? []).find(
        (entry) => entry.modelId === modelId,
      ) ?? null;
    }, sourceTarget.sample.modelId);
    if (
      !sparkVisibility ||
      sparkVisibility.visibilitySynchronized !== true ||
      sparkVisibility.visibilityMode !== "per-object-opacity" ||
      Number(sparkVisibility.hiddenObjects) < 1 ||
      Number(sparkVisibility.hiddenGaussians) < 1 ||
      sparkVisibility.sourceLayerVisible !== true
    ) {
      throw new Error(`Spark visibility did not follow object hide: ${JSON.stringify(sparkVisibility)}`);
    }

    await page.evaluate((selectionId) => {
      window.__OBJGAUSS_WORLD__?.toggleObjectVisibilityForAudit?.(selectionId);
    }, sourceTarget.object.selectionId);

    await clickSelector(page, "[data-model-process-button='trainable-mvp-debug']");
    await page.waitForTimeout(1500);
    const lazyState = await page.evaluate(() => {
      const shell = document.querySelector(".worldShell");
      const row = document.querySelector("[data-model-version-row-id='trainable-mvp-debug']");
      return {
        selectedModel: shell?.getAttribute("data-selected-model"),
        rowStatus: row?.getAttribute("data-model-object-layer-status"),
        processStatus: row?.getAttribute("data-model-process-status"),
        worldSelectedModel: window.__OBJGAUSS_WORLD__?.selectedModelId ?? null,
        stageModelIds: window.__OBJGAUSS_WORLD__?.stageModelIds ?? [],
      };
    });
    const lazyRequestObserved = requests.includes("/models/trainable-mvp-debug/model-artifact.json");
    if (
      lazyState.selectedModel !== "trainable-mvp-debug" ||
      !lazyState.stageModelIds.includes("trainable-mvp-debug") ||
      !["processing", "object-layer-loaded"].includes(lazyState.processStatus) ||
      !lazyRequestObserved
    ) {
      throw new Error(`selected hidden model did not load on demand: ${JSON.stringify({ lazyState, lazyRequestObserved, pageErrors })}`);
    }

    await clickSelector(page, "[data-model-process-button='ogc-debug']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const rows = [...document.querySelectorAll("[data-model-version-row-id='ogc-debug']")];
      return (
        shell?.getAttribute("data-selected-model") === "ogc-debug" &&
        rows.some((row) => row.getAttribute("data-model-object-layer-status") === "loaded") &&
        window.__OBJGAUSS_WORLD__?.selectedModelId === "ogc-debug"
      );
    }, undefined, { timeout: 15000 });

    await clickSelector(page, "[data-model-process-button='lego-alpha-raw-source']");
    const cliHandoff = await page.waitForFunction(() => {
      const row = document.querySelector("[data-model-version-row-id='lego-alpha-raw-source']");
      const button = row?.querySelector("[data-model-process-button='lego-alpha-raw-source']");
      const handoff = document.querySelector("[data-object-process-handoff='true']");
      if (
        row?.getAttribute("data-model-process-status") !== "raw-gaussian" ||
        button?.getAttribute("data-model-process-action") !== "cli-handoff" ||
        button?.textContent?.trim() !== "CLI 命令" ||
        handoff?.getAttribute("data-object-process-execution") !== "cli-only" ||
        !handoff?.textContent?.includes("浏览器不会运行对象模型")
      ) {
        return null;
      }
      return { status: handoff.getAttribute("data-object-process-handoff-status") };
    }, undefined, { timeout: 15000 }).then((handle) => handle.jsonValue());

    const screenshotPath = "/tmp/objgauss-viewer-truth.png";
    await page.addStyleTag({ content: "canvas { visibility: hidden !important; }" });
    await page.screenshot({ path: screenshotPath, fullPage: false, animations: "disabled", timeout: 15000 });
    if (pageErrors.length > 0) {
      throw new Error(`browser page errors: ${pageErrors.join(" | ")}`);
    }
    return {
      initialHiddenRequestCount: initialHiddenRequests.length,
      lazyModelId: "ogc-debug",
      lazyRequestObserved,
      transformModeCount: transformTruth.modeCount,
      sparkHiddenObjects: sparkVisibility.hiddenObjects,
      sparkHiddenGaussians: sparkVisibility.hiddenGaussians,
      cliHandoffStatus: cliHandoff.status,
      screenshotPath,
    };
  } finally {
    await closeBrowserWithTimeout(browser);
  }
}

async function auditWorld(url) {
  const browser = await chromium.launch(launchOptions());
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const consoleIssues = [];
  page.on("console", (message) => {
    if (["error", "warning"].includes(message.type())) {
      consoleIssues.push(`${message.type()}: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => consoleIssues.push(`pageerror: ${error.message}`));

  try {
    await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
    const app = page.locator(".worldShell");
    await app.waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const world = window.__OBJGAUSS_WORLD__;
      return world?.draggableObjectCount > world?.modelCount && world?.objectSelections?.length > world?.modelCount;
    }, undefined, {
      timeout: 15000,
    });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      return (
        shell?.getAttribute("data-sidebars") === "none" &&
        shell?.getAttribute("data-frosted-ui") === "enabled" &&
        shell?.getAttribute("data-app-mode") === "vr-three-world" &&
        shell?.getAttribute("data-debug-os") === "object-state"
      );
    }, undefined, { timeout: 15000 });

    const sidebars = await page.locator("aside, .leftRail, .rightRail").count();
    if (sidebars !== 0) {
      throw new Error(`sidebars should not exist in VR world shell, found ${sidebars}`);
    }
    const canvas = page.locator("canvas[data-three-world-canvas='true']");
    await canvas.waitFor({ timeout: 15000 });
    await page.locator(".glassHud.floatingInspector").waitFor({ timeout: 15000 });
    await page.locator("[data-object-debug-panel='true']").waitFor({ timeout: 15000 });
    await page.locator("[data-assignment-heatmap='true']").waitFor({ state: "attached", timeout: 15000 });
    await page.locator("[data-training-stage-panel='true']").waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-training-stage-panel='true']");
      const world = window.__OBJGAUSS_WORLD__;
      const rowCount = panel?.querySelectorAll("[data-model-version-row-id]").length ?? 0;
      return (
        shell?.getAttribute("data-training-stage") === "model-version-processing-v1" &&
        panel?.getAttribute("data-training-stage-schema") === "model-version-processing-v1" &&
        rowCount >= 7 &&
        Number(shell?.getAttribute("data-stage-visible-model-count") ?? 0) >= Number(world?.visibleModelCount ?? 0) &&
        Number(panel?.getAttribute("data-stage-processed-models") ?? 0) > 0
      );
    }, undefined, { timeout: 15000 });
    const sourceSplatStage = await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const world = window.__OBJGAUSS_WORLD__;
      const samples = world?.sourceSplatSamples ?? [];
      const selectedSample = samples.find((sample) => sample.modelId === world?.selectedModelId);
      if (
        shell?.getAttribute("data-source-splat-active") !== "true" ||
        shell?.getAttribute("data-source-splat-stage-contract") !== "spark-source-splat-stage-v1" ||
        shell?.getAttribute("data-stage-display-contract") !== "spark-source-splat-stage-v1" ||
        shell?.getAttribute("data-gaussian-display-mode") !== "完整 splat" ||
        panel?.getAttribute("data-stage-display-label") !== "完整 splat" ||
        world?.sourceSplatStageContract !== "spark-source-splat-stage-v1" ||
        world?.sourceSplatActive !== true ||
        !selectedSample ||
        !["loading", "ready"].includes(selectedSample.status)
      ) {
        return null;
      }
      return {
        contract: world.sourceSplatStageContract,
        status: selectedSample.status,
        modelId: selectedSample.modelId,
        path: selectedSample.path,
        readyCount: world.sourceSplatReadyCount ?? 0,
        mode: shell.getAttribute("data-gaussian-display-mode"),
      };
    }, undefined, { timeout: 30000 }).then((handle) => handle.jsonValue());
    const pills = await page.locator(".modelPill").count();
    if (pills < 7) {
      throw new Error(`expected at least 7 model pills, found ${pills}`);
    }
    const sourceSplatMotionSelection = await page.waitForFunction(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const sample = (world?.sourceSplatObjectMotionSamples ?? []).find(
        (entry) =>
          entry.status === "ready" &&
          entry.countMatches === true &&
          Number(entry.mappedGaussians) > 0,
      );
      const selection = (world?.objectSelections ?? []).find(
        (entry) => entry.modelId === sample?.modelId && entry.visible,
      );
      if (!sample || !selection || typeof world?.selectObjectForAudit !== "function") {
        return null;
      }
      const selected = world.selectObjectForAudit(selection.selectionId);
      if (!selected) return null;
      return {
        selectionId: selection.selectionId,
        modelId: selection.modelId,
        objectId: selection.objectId,
        before: selection.position,
        sample,
      };
    }, undefined, { timeout: 30000 }).then((handle) => handle.jsonValue());
    const sourceSplatObjectProjection = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return world?.sourceSplatObjectMotionProjectionForAudit?.(selectionId, [0.42, 0, 0]) ?? null;
    }, sourceSplatMotionSelection.selectionId);
    if (
      !sourceSplatObjectProjection ||
      sourceSplatObjectProjection.ok !== true ||
      sourceSplatObjectProjection.motion?.contract !== "source-splat-object-translate-v1" ||
      sourceSplatObjectProjection.motion?.status !== "ready" ||
      sourceSplatObjectProjection.motion?.active !== true ||
      sourceSplatObjectProjection.motion?.transformedObjects < 1 ||
      sourceSplatObjectProjection.motion?.maxTranslate <= 0 ||
      sourceSplatObjectProjection.motion?.selectedTranslateMagnitude <= 0 ||
      sourceSplatObjectProjection.motion?.sourceCount !== sourceSplatObjectProjection.motion?.pointCount ||
      sourceSplatObjectProjection.selectedScreenDelta <= 1 ||
      sourceSplatObjectProjection.peerScreenDelta > 0.5 ||
      sourceSplatObjectProjection.peerWorldDelta !== 0
    ) {
      throw new Error(`expected source splat projected motion with stable peer: ${JSON.stringify(sourceSplatObjectProjection)}`);
    }
    const sourceSplatMotionUi = await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const metric = document.querySelector("[data-source-splat-motion-metric='true']");
      if (
        shell?.getAttribute("data-source-splat-motion-status") !== "ready" ||
        shell?.getAttribute("data-source-splat-motion-active") !== "true" ||
        shell?.getAttribute("data-source-splat-motion-count-matches") !== "true" ||
        panel?.getAttribute("data-source-splat-motion-status") !== "ready" ||
        metric?.querySelector("strong")?.textContent?.trim() !== "已随动"
      ) {
        return null;
      }
      return {
        shellStatus: shell.getAttribute("data-source-splat-motion-status"),
        panelStatus: panel.getAttribute("data-source-splat-motion-status"),
        metricLabel: metric.querySelector("strong")?.textContent?.trim() ?? null,
      };
    }, undefined, { timeout: 15000 }).then((handle) => handle.jsonValue());
    Object.assign(sourceSplatObjectProjection, sourceSplatMotionUi);
    const sourceSplatObjectMotion = sourceSplatObjectProjection.motion;
    const ogcPill = page.locator(".modelPill[data-model-row-id='ogc-debug']");
    await ogcPill.waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const pill = document.querySelector(".modelPill[data-model-row-id='ogc-debug']");
      return pill?.getAttribute("data-model-load-state") === "loaded" &&
        Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0) > 0;
    }, undefined, { timeout: 15000 });
    await clickModelPill(page, "ogc-debug");
    await page.waitForFunction(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      return world?.selectedModelId === "ogc-debug" && world?.selectedModelId === shell?.getAttribute("data-selected-model");
    }, undefined, { timeout: 15000 });
    const objectSelection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const selection = world?.objectSelections?.find(
        (entry) => entry.modelId === "ogc-debug",
      );
      const objectCount = world?.objectSelections?.filter((entry) => entry.modelId === "ogc-debug").length ?? 0;
      return {
        ok: world?.selectObjectForAudit?.(selection?.selectionId) ?? false,
        selectionId: selection?.selectionId ?? null,
        objectCount,
      };
    });
    if (!objectSelection.ok) {
      throw new Error("expected audit handle to select an OGC per-object render target");
    }
    if (objectSelection.objectCount < 2) {
      throw new Error(`expected OGC model to expose at least 2 object render targets, found ${objectSelection.objectCount}`);
    }
    await page.waitForFunction((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      return (
        world?.selectedId === selectionId &&
        shell?.getAttribute("data-selected-target") === selectionId &&
        shell?.getAttribute("data-selected-object") !== ""
      );
    }, objectSelection.selectionId, { timeout: 15000 });
    const objectTransform = await page.waitForFunction((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      const worldPlane = document.querySelector("[data-world-plane-panel='true']");
      const interactionLayer = document.querySelector("[data-object-interaction-layer='true']");
      const systemDrawer = document.querySelector("[data-system-drawer='true']");
      const actionButtons = document.querySelectorAll("[data-object-transform-actions='true'] [data-object-transform-button]");
      const modeButtons = document.querySelectorAll("[data-object-transform-mode-selector='true'] [data-object-transform-mode-button]");
      const gaussianDisplayMode = shell?.getAttribute("data-gaussian-display-mode") ?? "";
      const topMetricLabels = [...document.querySelectorAll(".metricStrip .metric span")]
        .map((node) => node.textContent?.trim())
        .filter(Boolean);
      if (
        world?.objectTransformContract !== "three-transform-controls-v1" ||
        world?.objectTransformEngine !== "object-transform-state-v1" ||
        world?.objectPickingContract !== "projected-object-bbox-picker-v2" ||
        world?.objectPickingDecoupledFromRenderer !== true ||
        typeof world?.pickObjectForAudit !== "function" ||
        world?.transformGizmoAttached !== true ||
        world?.transformGizmoObject !== selectionId ||
        worldPlane?.getAttribute("data-world-plane-panel") !== "true" ||
        interactionLayer?.getAttribute("data-object-transform-primary") !== "three-transform-controls-v1" ||
        interactionLayer?.getAttribute("data-object-move-contract") !== "object-group-position-v1" ||
        interactionLayer?.getAttribute("data-object-move-selected") !== selectionId ||
        shell?.getAttribute("data-render-surface-contract") !== "three-world-point-preview-v1" ||
        (!gaussianDisplayMode.endsWith("点预览") && gaussianDisplayMode !== "完整 splat") ||
        worldPlane?.closest("[data-object-debug-panel='true']")?.getAttribute("data-render-surface-contract") !== "three-world-point-preview-v1" ||
        worldPlane?.closest("[data-object-debug-panel='true']")?.getAttribute("data-object-layer-status") !== "loaded" ||
        actionButtons.length !== 3 ||
        modeButtons.length !== 3 ||
        !systemDrawer ||
        document.querySelectorAll(".bottomStatus").length !== 0 ||
        topMetricLabels.join("|") !== "Three.js|高斯展示|展示版本|对象层"
      ) {
        return null;
      }
      return {
        contract: world.objectTransformContract,
        engine: world.objectTransformEngine,
        pickingContract: world.objectPickingContract,
        pickingDecoupledFromRenderer: world.objectPickingDecoupledFromRenderer,
        gizmoObject: world.transformGizmoObject,
        mode: world.objectTransformMode,
        renderSurfaceContract: shell.getAttribute("data-render-surface-contract"),
        gaussianDisplayMode,
        sourceSplatActive: shell.getAttribute("data-source-splat-active"),
        objectLayerStatus: worldPlane.closest("[data-object-debug-panel='true']").getAttribute("data-object-layer-status"),
        interactionLayer: interactionLayer.getAttribute("data-object-transform-primary"),
        actionButtonCount: actionButtons.length,
        modeButtonCount: modeButtons.length,
        topMetricLabels,
      };
    }, objectSelection.selectionId, { timeout: 15000 }).then((handle) => handle.jsonValue());
    const objectPick = await page.evaluate((selectionId) => {
      const result = window.__OBJGAUSS_WORLD__?.pickObjectForAudit?.(selectionId) ?? null;
      const world = window.__OBJGAUSS_WORLD__;
      return {
        result,
        contract: world?.objectPickingContract ?? null,
        decoupledFromRenderer: world?.objectPickingDecoupledFromRenderer ?? null,
        last: world?.objectPickingLast ?? null,
      };
    }, objectSelection.selectionId);
    if (
      objectPick.contract !== "projected-object-bbox-picker-v2" ||
      objectPick.decoupledFromRenderer !== true ||
      objectPick.result?.ok !== true ||
      objectPick.result?.pick?.contract !== "projected-object-bbox-picker-v2" ||
      objectPick.result?.pick?.decoupledFromRenderer !== true ||
      objectPick.result?.pick?.selectionId !== objectSelection.selectionId ||
      objectPick.result?.pick?.hitTest !== "projected-object-bbox" ||
      !objectPick.result?.pick?.bounds ||
      !(Number(objectPick.result?.pick?.candidateCount) > 0)
    ) {
      throw new Error(`expected bbox object picking to select ObjectState target: ${JSON.stringify(objectPick)}`);
    }
    const objectPickOutside = await page.evaluate((selectionId) => {
      const result = window.__OBJGAUSS_WORLD__?.pickOutsideObjectForAudit?.(selectionId) ?? null;
      const world = window.__OBJGAUSS_WORLD__;
      return {
        result,
        contract: world?.objectPickingContract ?? null,
        last: world?.objectPickingLast ?? null,
      };
    }, objectSelection.selectionId);
    if (
      objectPickOutside.contract !== "projected-object-bbox-picker-v2" ||
      objectPickOutside.result?.ok !== true ||
      objectPickOutside.result?.pick?.contract !== "projected-object-bbox-picker-v2" ||
      objectPickOutside.result?.pick?.hit !== false ||
      objectPickOutside.result?.pick?.hitTest !== "projected-object-bbox"
    ) {
      throw new Error(`expected bbox object picking to miss outside ObjectState frame: ${JSON.stringify(objectPickOutside)}`);
    }
    const beforeMove = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      const target = world?.objectSelections?.find((entry) => entry.selectionId === selectionId);
      return {
        position: target?.position ?? null,
        contract: world?.objectMoveContract ?? null,
        hasMoveAudit: typeof world?.moveObjectForAudit === "function",
      };
    }, objectSelection.selectionId);
    if (
      !Array.isArray(beforeMove.position) ||
      beforeMove.contract !== "object-group-position-v1" ||
      !beforeMove.hasMoveAudit
    ) {
      throw new Error(`expected movable object audit contract: ${JSON.stringify(beforeMove)}`);
    }
    await page.locator("[data-object-move-panel='true']").waitFor({ timeout: 15000 });
    await clickSelector(page, "[data-object-move-button='+x']");
    const objectMove = await page.waitForFunction((input) => {
      const world = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const movePanel = document.querySelector("[data-object-move-panel='true']");
      const target = world?.objectSelections?.find((entry) => entry.selectionId === input.selectionId);
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const moveEvent = events.find((event) =>
        event.type === "move-object" &&
        event.selectionId === input.selectionId &&
        event.source === "debug-panel"
      );
      if (
        !target ||
        !Array.isArray(target.position) ||
        target.position[0] <= input.before[0] + 0.3 ||
        target.position[1] !== 0 ||
        shell?.getAttribute("data-object-move-contract") !== "object-group-position-v1" ||
        panel?.getAttribute("data-object-move-contract") !== "object-group-position-v1" ||
        panel?.getAttribute("data-object-move-selected") !== input.selectionId ||
        movePanel?.getAttribute("data-object-move-selected") !== input.selectionId ||
        !moveEvent
      ) {
        return null;
      }
      return {
        selectionId: input.selectionId,
        before: input.before,
        after: target.position,
        source: moveEvent.source ?? null,
      };
    }, {
      selectionId: objectSelection.selectionId,
      before: beforeMove.position,
    }, { timeout: 15000 }).then((handle) => handle.jsonValue());
    const objectTransformState = await page.evaluate((input) => {
      const currentWorld = () => window.__OBJGAUSS_WORLD__;
      const closePosition = (left, right) =>
        Array.isArray(left) &&
        Array.isArray(right) &&
        left.every((value, index) => Math.abs(Number(value) - Number(right[index])) < 0.001);
      const afterMove = currentWorld()?.objectSelections?.find((entry) => entry.selectionId === input.selectionId)?.position ?? null;
      const undo = currentWorld()?.undoObjectTransformForAudit?.();
      const afterUndo = currentWorld()?.objectSelections?.find((entry) => entry.selectionId === input.selectionId)?.position ?? null;
      const undoWorld = currentWorld();
      const redo = undoWorld?.redoObjectTransformForAudit?.();
      const afterRedo = currentWorld()?.objectSelections?.find((entry) => entry.selectionId === input.selectionId)?.position ?? null;
      const snapOn = currentWorld()?.setTransformSnapForAudit?.(true);
      const snapEnabled = currentWorld()?.transformSnapEnabled;
      const snapStep = currentWorld()?.transformSnapStep;
      const rotateMode = currentWorld()?.setTransformModeForAudit?.("rotate");
      const scaleMode = currentWorld()?.setTransformModeForAudit?.("scale");
      const translateMode = currentWorld()?.setTransformModeForAudit?.("translate");
      const snapOff = currentWorld()?.setTransformSnapForAudit?.(false);
      const begin = currentWorld()?.beginTransformForAudit?.(input.selectionId);
      const nudged = currentWorld()?.nudgeActiveTransformForAudit?.([0.51, 0, 0]);
      const cancel = currentWorld()?.cancelTransformForAudit?.("audit");
      const afterCancel = currentWorld()?.objectSelections?.find((entry) => entry.selectionId === input.selectionId)?.position ?? null;
      const finalWorld = currentWorld();
      const actionButtons = [...document.querySelectorAll("[data-object-transform-actions='true'] [data-object-transform-button]")]
        .map((node) => node.getAttribute("data-object-transform-button"));
      return {
        afterMove,
        undo,
        afterUndo,
        redo,
        afterRedo,
        snapOn,
        snapEnabled,
        snapStep,
        rotateMode,
        scaleMode,
        translateMode,
        snapOff,
        begin,
        nudged,
        cancel,
        afterCancel,
        historyDepth: finalWorld?.transformHistoryDepth ?? null,
        redoDepth: finalWorld?.transformRedoDepth ?? null,
        canUndo: finalWorld?.transformCanUndo ?? null,
        canRedo: finalWorld?.transformCanRedo ?? null,
        lastEvent: finalWorld?.transformLastEvent ?? null,
        actionButtons,
        ok:
          currentWorld()?.objectTransformEngine === "object-transform-state-v1" &&
          closePosition(afterUndo, input.before) &&
          closePosition(afterRedo, afterMove) &&
          closePosition(afterCancel, afterRedo) &&
          undo?.ok === true &&
          redo?.ok === true &&
          snapOn === true &&
          snapEnabled === true &&
          snapStep === 0.25 &&
          rotateMode === "rotate" &&
          scaleMode === "scale" &&
          translateMode === "translate" &&
          snapOff === false &&
          begin?.ok === true &&
          nudged?.ok === true &&
          cancel?.ok === true &&
          finalWorld?.transformHistoryDepth >= 1 &&
          finalWorld?.transformRedoDepth === 0 &&
          actionButtons.join("|") === "undo|redo|cancel",
      };
    }, {
      selectionId: objectSelection.selectionId,
      before: beforeMove.position,
    });
    if (!objectTransformState.ok) {
      throw new Error(`expected object transform state graph to support undo/redo/snap/cancel: ${JSON.stringify(objectTransformState)}`);
    }
    const gaussianSelection = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        protocol: world?.debugProtocol ?? null,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, objectSelection.selectionId);
    if (!gaussianSelection.ok) {
      throw new Error("expected audit handle to select a Gaussian assignment probe");
    }
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      return (
        shell?.getAttribute("data-selected-gaussian") !== "" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) > 0
      );
    }, undefined, { timeout: 15000 });
    if (gaussianSelection.protocol !== "object-state-debug-os-v1") {
      throw new Error(`unexpected debug protocol: ${gaussianSelection.protocol}`);
    }
    if (gaussianSelection.assignmentSource !== "derived_from_object_id") {
      throw new Error(`unexpected assignment source: ${gaussianSelection.assignmentSource}`);
    }
    const trainablePill = page.locator(".modelPill[data-model-row-id='trainable-mvp-debug']");
    await trainablePill.waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const pill = document.querySelector(".modelPill[data-model-row-id='trainable-mvp-debug']");
      return pill?.getAttribute("data-model-load-state") === "loaded" &&
        Number(shell?.getAttribute("data-trainable-artifact-loaded-count") ?? 0) > 0;
    }, undefined, { timeout: 15000 });
    await clickModelPill(page, "trainable-mvp-debug");
    await page.waitForFunction(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      return world?.selectedModelId === "trainable-mvp-debug" &&
        world?.selectedModelId === shell?.getAttribute("data-selected-model") &&
        shell?.getAttribute("data-trainable-artifact-load-route") === "fetch-json" &&
        shell?.getAttribute("data-trainable-artifact-path") === "/models/trainable-mvp-debug/model-artifact.json" &&
        shell?.getAttribute("data-trainable-artifact-frame-index") === "0" &&
        shell?.getAttribute("data-trainable-artifact-frame-count") === "2";
    }, undefined, { timeout: 15000 });
    const trainableSelection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const selection = world?.objectSelections?.find(
        (entry) => entry.modelId === "trainable-mvp-debug",
      );
      const objectCount = world?.objectSelections?.filter((entry) => entry.modelId === "trainable-mvp-debug").length ?? 0;
      return {
        ok: world?.selectObjectForAudit?.(selection?.selectionId) ?? false,
        selectionId: selection?.selectionId ?? null,
        objectCount,
      };
    });
    if (!trainableSelection.ok) {
      throw new Error("expected audit handle to select a trainable artifact object");
    }
    if (trainableSelection.objectCount < 2) {
      throw new Error(`expected trainable artifact to expose at least 2 object render targets, found ${trainableSelection.objectCount}`);
    }
    const trainableGaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: window.__OBJGAUSS_WORLD__?.assignmentSource ?? null,
        trainableArtifactLoadedCount: window.__OBJGAUSS_WORLD__?.trainableArtifactLoadedCount ?? 0,
      };
    }, trainableSelection.selectionId);
    if (!trainableGaussian.ok) {
      throw new Error("expected audit handle to select a trainable artifact Gaussian probe");
    }
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const probePanel = document.querySelector("[data-gaussian-probe-panel='true']");
      const timelinePanel = document.querySelector("[data-assignment-timeline-panel='true']");
      const fragmentPanel = document.querySelector("[data-object-fragmentation-panel='true']");
      const verdict = document.querySelector("[data-object-verdict-panel='true']");
      const stability = document.querySelector("[data-stability-dashboard='true']");
      const training = document.querySelector("[data-training-evidence='true']");
      const solver = document.querySelector("[data-solver-training-evidence='true']");
      const snapshot = window.__OBJGAUSS_DEBUG_SNAPSHOT__;
      const world = window.__OBJGAUSS_WORLD__;
      return (
        shell?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        heatmap?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2 &&
        shell?.getAttribute("data-assignment-probe-status") === "confident" &&
        panel?.getAttribute("data-assignment-probe-status") === "confident" &&
        heatmap?.getAttribute("data-assignment-probe-status") === "confident" &&
        probePanel?.getAttribute("data-gaussian-probe-source") === "trainable_kernel_model_artifact" &&
        probePanel?.getAttribute("data-gaussian-probe-index") === "0" &&
        probePanel?.getAttribute("data-gaussian-probe-status") === "confident" &&
        probePanel?.getAttribute("data-gaussian-probe-top-slot") === shell?.getAttribute("data-assignment-probe-top-slot") &&
        probePanel?.getAttribute("data-gaussian-probe-ambiguous") === "false" &&
        probePanel?.getAttribute("data-gaussian-probe-collapse-risk") === "false" &&
        Number(probePanel?.getAttribute("data-gaussian-probe-margin") ?? 0) > 0.55 &&
        Number(probePanel?.getAttribute("data-gaussian-probe-confidence") ?? 0) > 0.7 &&
        Number(probePanel?.getAttribute("data-gaussian-probe-entropy") ?? 0) > 0 &&
        Number(probePanel?.getAttribute("data-gaussian-probe-opacity") ?? 0) > 0 &&
        probePanel?.getAttribute("data-gaussian-probe-position") !== "-" &&
        shell?.getAttribute("data-assignment-timeline-status") === "stable" &&
        panel?.getAttribute("data-assignment-timeline-status") === "stable" &&
        timelinePanel?.getAttribute("data-assignment-timeline-status") === "stable" &&
        timelinePanel?.getAttribute("data-assignment-timeline-frame-count") === "2" &&
        timelinePanel?.getAttribute("data-assignment-timeline-current-frame") === "0" &&
        timelinePanel?.getAttribute("data-assignment-timeline-gaussian-index") === "0" &&
        Number(timelinePanel?.getAttribute("data-assignment-timeline-jitter") ?? 0) > 0 &&
        Number(timelinePanel?.getAttribute("data-assignment-timeline-jitter") ?? 1) < 0.05 &&
        timelinePanel?.querySelectorAll("[data-assignment-timeline-row='true']").length === 2 &&
        timelinePanel?.querySelector("[data-assignment-timeline-row-current='true']")?.getAttribute("data-assignment-timeline-row-frame") === "0" &&
        snapshot?.assignment?.probe?.status === "confident" &&
        snapshot?.assignment?.timeline?.status === "stable" &&
        snapshot?.assignment?.timeline?.frameCount === 2 &&
        snapshot?.assignment?.timeline?.gaussianIndex === 0 &&
        snapshot?.assignment?.timeline?.meanDelta > 0 &&
        snapshot?.assignment?.timeline?.meanDelta < 0.05 &&
        world?.assignmentProbeStatus === "confident" &&
        shell?.getAttribute("data-object-continuity-status") === "continuous" &&
        panel?.getAttribute("data-object-continuity-status") === "continuous" &&
        snapshot?.continuity?.status === "continuous" &&
        world?.objectContinuityStatus === "continuous" &&
        Number(shell?.getAttribute("data-object-continuity-bbox-diagonal") ?? 0) > 0 &&
        Number(panel?.getAttribute("data-object-continuity-bbox-diagonal") ?? 0) > 0 &&
        Number(snapshot?.continuity?.bboxDiagonal ?? 0) > 0 &&
        Number(world?.objectContinuityBboxDiagonal ?? 0) > 0 &&
        shell?.getAttribute("data-object-continuity-centroid-contained") === "true" &&
        panel?.getAttribute("data-object-continuity-centroid-contained") === "true" &&
        fragmentPanel?.getAttribute("data-object-fragmentation-status") === "continuous" &&
        fragmentPanel?.getAttribute("data-object-fragmentation-object") === "0" &&
        Number(fragmentPanel?.getAttribute("data-object-fragmentation-gaussians") ?? 0) > 0 &&
        Number(fragmentPanel?.getAttribute("data-object-fragmentation-compactness") ?? 0) > 0 &&
        Number(fragmentPanel?.getAttribute("data-object-fragmentation-bbox-diagonal") ?? 0) > 0 &&
        Number(fragmentPanel?.getAttribute("data-object-fragmentation-density") ?? 0) > 0 &&
        fragmentPanel?.getAttribute("data-object-fragmentation-centroid-contained") === "true" &&
        fragmentPanel?.getAttribute("data-object-fragmentation-bbox-valid") === "true" &&
        fragmentPanel?.getAttribute("data-object-fragmentation-fragmented") === "false" &&
        fragmentPanel?.getAttribute("data-object-fragmentation-centroid-outside") === "false" &&
        fragmentPanel?.getAttribute("data-object-fragmentation-degenerate") === "false" &&
        fragmentPanel?.querySelectorAll("[data-object-fragmentation-flags='true'] [class*='qualityGateRow']").length === 3 &&
        snapshot?.continuity?.centroidContained === true &&
        world?.objectContinuityCentroidContained === true &&
        shell?.getAttribute("data-object-temporal-status") === "stable" &&
        panel?.getAttribute("data-object-temporal-status") === "stable" &&
        snapshot?.temporal?.status === "stable" &&
        world?.objectTemporalStatus === "stable" &&
        shell?.getAttribute("data-object-temporal-stable") === "true" &&
        panel?.getAttribute("data-object-temporal-stable") === "true" &&
        snapshot?.temporal?.stable === true &&
        world?.objectTemporalStable === true &&
        Number(shell?.getAttribute("data-object-temporal-drift") ?? 0) > 0 &&
        Number(shell?.getAttribute("data-object-temporal-drift") ?? 1) < 0.08 &&
        Number(shell?.getAttribute("data-object-assignment-jitter") ?? 0) > 0 &&
        Number(shell?.getAttribute("data-object-assignment-jitter") ?? 1) < 0.08 &&
        Number(shell?.getAttribute("data-object-bbox-stability") ?? 0) > 0.5 &&
        Number(panel?.getAttribute("data-object-temporal-drift") ?? 0) > 0 &&
        Number(panel?.getAttribute("data-object-assignment-jitter") ?? 0) > 0 &&
        Number(panel?.getAttribute("data-object-bbox-stability") ?? 0) > 0.5 &&
        snapshot?.temporal?.temporalDrift > 0 &&
        snapshot?.temporal?.assignmentJitter > 0 &&
        snapshot?.temporal?.bboxStability > 0.5 &&
        world?.objectTemporalDrift > 0 &&
        world?.objectAssignmentJitter > 0 &&
        world?.objectBboxStability > 0.5 &&
        shell?.getAttribute("data-object-explainability-status") === "explainable" &&
        panel?.getAttribute("data-object-explainability-status") === "explainable" &&
        snapshot?.explainability?.status === "explainable" &&
        world?.objectExplainabilityStatus === "explainable" &&
        shell?.getAttribute("data-object-explainable") === "true" &&
        panel?.getAttribute("data-object-explainable") === "true" &&
        snapshot?.explainability?.explainable === true &&
        world?.objectExplainable === true &&
        Number(shell?.getAttribute("data-object-explainability-score") ?? 0) > 0.6 &&
        Number(panel?.getAttribute("data-object-explainability-score") ?? 0) > 0.6 &&
        Number(snapshot?.explainability?.score ?? 0) > 0.6 &&
        Number(world?.objectExplainabilityScore ?? 0) > 0.6 &&
        shell?.getAttribute("data-object-explainability-reasons") === "" &&
        panel?.getAttribute("data-object-explainability-reasons") === "" &&
        verdict?.getAttribute("data-object-verdict-status") === "explainable" &&
        verdict?.getAttribute("data-object-verdict-explainable") === "true" &&
        Number(verdict?.getAttribute("data-object-verdict-score") ?? 0) > 0.6 &&
        verdict?.getAttribute("data-object-verdict-reason-count") === "0" &&
        verdict?.getAttribute("data-object-verdict-clear") === "true" &&
        verdict?.getAttribute("data-object-verdict-continuity-status") === "continuous" &&
        verdict?.getAttribute("data-object-verdict-temporal-status") === "stable" &&
        verdict?.querySelector("[data-object-verdict-reason-row='true']")?.getAttribute("data-object-verdict-reason-name") === "clear" &&
        verdict?.querySelector("[data-object-verdict-reason-row='true']")?.getAttribute("data-object-verdict-reason-status") === "pass" &&
        snapshot?.explainability?.reasonNames === "" &&
        world?.objectExplainabilityReasons === "" &&
        Number(shell?.getAttribute("data-assignment-probe-margin") ?? 0) > 0.55 &&
        Number(heatmap?.getAttribute("data-assignment-probe-margin") ?? 0) > 0.55 &&
        snapshot?.assignment?.probe?.margin > 0.55 &&
        world?.assignmentProbeMargin > 0.55 &&
        stability?.getAttribute("data-stability-status") === shell?.getAttribute("data-stability-status") &&
        training?.getAttribute("data-training-status") === "loss_down" &&
        training?.getAttribute("data-training-renderer") === "cpu-image-point-splat-differentiable-v1" &&
        training?.getAttribute("data-training-image-loss-decreased") === "true" &&
        Number(training?.getAttribute("data-training-final-total-loss") ?? 0) > 0 &&
        Number(training?.getAttribute("data-training-loss-delta") ?? 0) > 0 &&
        Number(training?.getAttribute("data-training-final-image-loss") ?? 0) > 0 &&
        solver?.getAttribute("data-solver-training-status") === "loss_down" &&
        solver?.getAttribute("data-solver-training-schema") === "objgauss-object-emergence-solver-training-v1" &&
        solver?.getAttribute("data-solver-training-loss-decreased") === "true" &&
        solver?.getAttribute("data-solver-training-assignment-loss-decreased") === "true" &&
        solver?.getAttribute("data-solver-training-gpu-used") === "false" &&
        solver?.getAttribute("data-solver-training-vram-reserve-gb") === "1" &&
        shell?.getAttribute("data-solver-training-status") === "loss_down" &&
        shell?.getAttribute("data-solver-training-gpu-used") === "false" &&
        shell?.getAttribute("data-solver-training-vram-reserve-gb") === "1" &&
        panel?.getAttribute("data-solver-training-status") === "loss_down" &&
        panel?.getAttribute("data-solver-training-gpu-used") === "false" &&
        panel?.getAttribute("data-solver-training-vram-reserve-gb") === "1" &&
        snapshot?.solverTraining?.status === "loss_down" &&
        snapshot?.solverTraining?.gpuUsed === false &&
        snapshot?.solverTraining?.vramReserveGb === 1 &&
        Number(solver?.getAttribute("data-solver-training-final-total-loss") ?? 0) > 0 &&
        Number(solver?.getAttribute("data-solver-training-loss-delta") ?? 0) > 0 &&
        Number(solver?.getAttribute("data-solver-training-final-assignment-loss") ?? 0) > 0 &&
        Number(solver?.getAttribute("data-solver-training-assignment-loss-delta") ?? 0) > 0 &&
        Number(world?.solverTrainingLossDelta ?? 0) > 0 &&
        world?.solverTrainingGpuUsed === false &&
        world?.solverTrainingVramReserveGb === 1 &&
        Number(stability?.getAttribute("data-slot-utilization") ?? 0) > 0 &&
        stability?.getAttribute("data-purity-available") === "true" &&
        stability?.getAttribute("data-temporal-available") === "true" &&
        stability?.getAttribute("data-spatial-available") === "true" &&
        stability?.getAttribute("data-jitter-available") === "true" &&
        stability?.getAttribute("data-bbox-available") === "true" &&
        Number(stability?.getAttribute("data-mean-purity") ?? 0) > 0 &&
        Number(stability?.getAttribute("data-mean-temporal-drift") ?? 0) > 0 &&
        Number(stability?.getAttribute("data-mean-spatial-compactness") ?? 0) > 0 &&
        Number(stability?.getAttribute("data-mean-assignment-jitter") ?? -1) >= 0 &&
        Number(stability?.getAttribute("data-mean-bbox-stability") ?? 0) > 0
      );
    }, undefined, { timeout: 15000 });
    if (trainableGaussian.assignmentSource !== "trainable_kernel_model_artifact") {
      throw new Error(`unexpected trainable assignment source: ${trainableGaussian.assignmentSource}`);
    }
    if (trainableGaussian.trainableArtifactLoadedCount < 1) {
      throw new Error("expected trainable artifact load count to be visible in audit handle");
    }
    await clickSelector(page, "[data-trainable-frame-button='1']");
    const frameSwitch = await page.waitForFunction(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const selector = document.querySelector("[data-trainable-frame-selector='true']");
      const frameObject = world?.objectSelections?.find(
        (entry) => entry.modelId === "trainable-mvp-debug" && entry.frameIndex === 1,
      );
      return {
        ok:
          shell?.getAttribute("data-trainable-artifact-frame-index") === "1" &&
          shell?.getAttribute("data-trainable-artifact-frame-count") === "2" &&
          panel?.getAttribute("data-trainable-frame-index") === "1" &&
          selector?.getAttribute("data-selected-frame") === "1" &&
          world?.selectedTrainableFrameIndex === 1 &&
          world?.selectedTrainableFrameCount === 2 &&
          Boolean(frameObject),
        selectionId: frameObject?.selectionId ?? null,
        frameIndex: world?.selectedTrainableFrameIndex ?? null,
        frameCount: world?.selectedTrainableFrameCount ?? null,
      };
    }, undefined, { timeout: 15000 }).then((handle) => handle.jsonValue());
    if (!frameSwitch.ok) {
      throw new Error(`expected trainable artifact frame switch to frame 1: ${JSON.stringify(frameSwitch)}`);
    }
    const frameGaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: window.__OBJGAUSS_WORLD__?.assignmentSource ?? null,
        assignmentProbeStatus: window.__OBJGAUSS_WORLD__?.assignmentProbeStatus ?? null,
        assignmentProbeMargin: window.__OBJGAUSS_WORLD__?.assignmentProbeMargin ?? null,
      };
    }, frameSwitch.selectionId);
    if (
      !frameGaussian.ok ||
      frameGaussian.assignmentSource !== "trainable_kernel_model_artifact" ||
      frameGaussian.assignmentProbeStatus !== "confident" ||
      !(Number(frameGaussian.assignmentProbeMargin) > 0.55)
    ) {
      throw new Error(`expected frame 1 Gaussian probe to remain trainable artifact sourced: ${JSON.stringify(frameGaussian)}`);
    }
    await clickSelector(page, "[data-debug-lens-button='confidence']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const selector = document.querySelector("[data-debug-lens-selector='true']");
      const world = window.__OBJGAUSS_WORLD__;
      const samples = world?.lensOpacitySamples ?? [];
      return (
        shell?.getAttribute("data-debug-lens") === "confidence" &&
        panel?.getAttribute("data-debug-lens") === "confidence" &&
        selector?.getAttribute("data-selected-lens") === "confidence" &&
        world?.debugLens === "confidence" &&
        samples.some((sample) =>
          sample.modelId === "trainable-mvp-debug" &&
          sample.activeLens === "confidence" &&
          sample.opacityLens === "confidence" &&
          sample.opacity > 0.32 &&
          sample.opacity <= 1
        )
      );
    }, undefined, { timeout: 15000 });
    await clickSelector(page, "[data-debug-lens-button='opacity']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const selector = document.querySelector("[data-debug-lens-selector='true']");
      const world = window.__OBJGAUSS_WORLD__;
      const samples = world?.lensOpacitySamples ?? [];
      return (
        shell?.getAttribute("data-debug-lens") === "opacity" &&
        panel?.getAttribute("data-debug-lens") === "opacity" &&
        selector?.getAttribute("data-selected-lens") === "opacity" &&
        selector?.querySelector("[data-debug-lens-button='opacity']")?.getAttribute("data-active") === "true" &&
        world?.debugLens === "opacity" &&
        samples.some((sample) =>
          sample.modelId === "trainable-mvp-debug" &&
          sample.activeLens === "opacity" &&
          sample.opacityLens === "opacity" &&
          sample.opacity > 0.85 &&
          sample.opacity <= 1
        )
      );
    }, undefined, { timeout: 15000 });
    await clickSelector(page, "[data-debug-lens-button='entropy']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const selector = document.querySelector("[data-debug-lens-selector='true']");
      const world = window.__OBJGAUSS_WORLD__;
      const samples = world?.lensOpacitySamples ?? [];
      return (
        shell?.getAttribute("data-debug-lens") === "entropy" &&
        panel?.getAttribute("data-debug-lens") === "entropy" &&
        selector?.getAttribute("data-selected-lens") === "entropy" &&
        world?.debugLens === "entropy" &&
        samples.some((sample) =>
          sample.modelId === "trainable-mvp-debug" &&
          sample.activeLens === "entropy" &&
          sample.opacityLens === "entropy" &&
          sample.opacity > 0.32 &&
          sample.opacity <= 1
        )
      );
    }, undefined, { timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const selector = document.querySelector("[data-object-overlay-selector='true']");
      const world = window.__OBJGAUSS_WORLD__;
      const samples = world?.objectOverlaySamples ?? [];
      return (
        shell?.getAttribute("data-object-overlay-mode") === "full" &&
        shell?.getAttribute("data-object-overlay-bbox-visible") === "true" &&
        shell?.getAttribute("data-object-overlay-centroid-visible") === "true" &&
        panel?.getAttribute("data-object-overlay-mode") === "full" &&
        selector?.getAttribute("data-selected-overlay") === "full" &&
        world?.objectOverlayMode === "full" &&
        world?.objectOverlayBboxVisible === true &&
        world?.objectOverlayCentroidVisible === true &&
        samples.some((sample) =>
          sample.modelId === "trainable-mvp-debug" &&
          sample.bboxVisible === true &&
          sample.centroidVisible === true
        )
      );
    }, undefined, { timeout: 15000 });
    await clickSelector(page, "[data-object-overlay-button='bbox']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const world = window.__OBJGAUSS_WORLD__;
      const samples = world?.objectOverlaySamples ?? [];
      return (
        shell?.getAttribute("data-object-overlay-mode") === "bbox" &&
        shell?.getAttribute("data-object-overlay-bbox-visible") === "true" &&
        shell?.getAttribute("data-object-overlay-centroid-visible") === "false" &&
        panel?.getAttribute("data-object-overlay-mode") === "bbox" &&
        world?.objectOverlayMode === "bbox" &&
        world?.objectOverlayBboxVisible === true &&
        world?.objectOverlayCentroidVisible === false &&
        samples.some((sample) =>
          sample.modelId === "trainable-mvp-debug" &&
          sample.bboxVisible === true &&
          sample.centroidVisible === false
        )
      );
    }, undefined, { timeout: 15000 });
    await clickSelector(page, "[data-object-overlay-button='centroid']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const world = window.__OBJGAUSS_WORLD__;
      const samples = world?.objectOverlaySamples ?? [];
      return (
        shell?.getAttribute("data-object-overlay-mode") === "centroid" &&
        shell?.getAttribute("data-object-overlay-bbox-visible") === "false" &&
        shell?.getAttribute("data-object-overlay-centroid-visible") === "true" &&
        panel?.getAttribute("data-object-overlay-mode") === "centroid" &&
        world?.objectOverlayMode === "centroid" &&
        world?.objectOverlayBboxVisible === false &&
        world?.objectOverlayCentroidVisible === true &&
        samples.some((sample) =>
          sample.modelId === "trainable-mvp-debug" &&
          sample.bboxVisible === false &&
          sample.centroidVisible === true
        )
      );
    }, undefined, { timeout: 15000 });
    await clickSelector(page, "[data-object-overlay-button='off']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const world = window.__OBJGAUSS_WORLD__;
      const samples = world?.objectOverlaySamples ?? [];
      return (
        shell?.getAttribute("data-object-overlay-mode") === "off" &&
        shell?.getAttribute("data-object-overlay-bbox-visible") === "false" &&
        shell?.getAttribute("data-object-overlay-centroid-visible") === "false" &&
        panel?.getAttribute("data-object-overlay-mode") === "off" &&
        world?.objectOverlayMode === "off" &&
        world?.objectOverlayBboxVisible === false &&
        world?.objectOverlayCentroidVisible === false &&
        samples.some((sample) =>
          sample.modelId === "trainable-mvp-debug" &&
          sample.bboxVisible === false &&
          sample.centroidVisible === false
        )
      );
    }, undefined, { timeout: 15000 });
    await clickSelector(page, "[data-object-overlay-button='full']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const world = window.__OBJGAUSS_WORLD__;
      return (
        shell?.getAttribute("data-object-overlay-mode") === "full" &&
        shell?.getAttribute("data-object-overlay-bbox-visible") === "true" &&
        shell?.getAttribute("data-object-overlay-centroid-visible") === "true" &&
        panel?.getAttribute("data-object-overlay-mode") === "full" &&
        world?.objectOverlayMode === "full" &&
        world?.objectOverlayBboxVisible === true &&
        world?.objectOverlayCentroidVisible === true
      );
    }, undefined, { timeout: 15000 });
    const hoverSelection = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      const result = world?.hoverObjectForAudit?.(selectionId) ?? null;
      return {
        result,
        hoveredId: window.__OBJGAUSS_WORLD__?.hoveredId ?? null,
        hoveredGaussianCount: window.__OBJGAUSS_WORLD__?.hoveredGaussianCount ?? 0,
        hoveredAssignmentSource: window.__OBJGAUSS_WORLD__?.hoveredAssignmentSource ?? null,
        hoveredAssignmentSlotCount: window.__OBJGAUSS_WORLD__?.hoveredAssignment?.length ?? 0,
        hoveredAssignmentProbeStatus: window.__OBJGAUSS_WORLD__?.hoveredAssignmentProbeStatus ?? null,
        hoveredAssignmentProbeMargin: window.__OBJGAUSS_WORLD__?.hoveredAssignmentProbeMargin ?? null,
        hoveredAssignmentConfidence: window.__OBJGAUSS_WORLD__?.hoveredAssignmentConfidence ?? null,
        hoveredAssignmentEntropy: window.__OBJGAUSS_WORLD__?.hoveredAssignmentEntropy ?? null,
        hoveredAssignmentTopSlot: window.__OBJGAUSS_WORLD__?.hoveredAssignmentTopSlot ?? null,
        hoveredContinuityStatus: window.__OBJGAUSS_WORLD__?.hoveredContinuityStatus ?? null,
        hoveredContinuityBboxDiagonal: window.__OBJGAUSS_WORLD__?.hoveredContinuityBboxDiagonal ?? null,
        hoveredContinuityCentroidContained: window.__OBJGAUSS_WORLD__?.hoveredContinuityCentroidContained ?? false,
        hoveredTemporalStatus: window.__OBJGAUSS_WORLD__?.hoveredTemporalStatus ?? null,
        hoveredTemporalDrift: window.__OBJGAUSS_WORLD__?.hoveredTemporalDrift ?? null,
        hoveredAssignmentJitter: window.__OBJGAUSS_WORLD__?.hoveredAssignmentJitter ?? null,
        hoveredBboxStability: window.__OBJGAUSS_WORLD__?.hoveredBboxStability ?? null,
        hoveredTemporalStable: window.__OBJGAUSS_WORLD__?.hoveredTemporalStable ?? false,
        hoveredExplainabilityStatus: window.__OBJGAUSS_WORLD__?.hoveredExplainabilityStatus ?? null,
        hoveredExplainable: window.__OBJGAUSS_WORLD__?.hoveredExplainable ?? false,
        hoveredExplainabilityScore: window.__OBJGAUSS_WORLD__?.hoveredExplainabilityScore ?? null,
        hoveredExplainabilityReasons: window.__OBJGAUSS_WORLD__?.hoveredExplainabilityReasons ?? null,
        hoverHighlightActive: window.__OBJGAUSS_WORLD__?.hoverHighlightActive ?? false,
        hoverHighlightedObjectCount: window.__OBJGAUSS_WORLD__?.hoverHighlightedObjectCount ?? 0,
        hoverHighlightedGaussianCount: window.__OBJGAUSS_WORLD__?.hoverHighlightedGaussianCount ?? 0,
        hoverDimmedObjectCount: window.__OBJGAUSS_WORLD__?.hoverDimmedObjectCount ?? 0,
        hoverDimmedGaussianCount: window.__OBJGAUSS_WORLD__?.hoverDimmedGaussianCount ?? 0,
        highlightedSamples: (window.__OBJGAUSS_WORLD__?.hoverHighlightSamples ?? []).filter(
          (sample) => sample.hoverHighlighted,
        ),
        dimmedSamples: (window.__OBJGAUSS_WORLD__?.hoverHighlightSamples ?? []).filter(
          (sample) => sample.hoverDimmed,
        ),
      };
    }, trainableSelection.selectionId);
    if (!hoverSelection.result?.ok) {
      throw new Error(`expected audit handle to hover a trainable ObjectState target: ${JSON.stringify(hoverSelection)}`);
    }
    if (hoverSelection.result.selectionId !== trainableSelection.selectionId) {
      throw new Error(`hovered wrong ObjectState target: ${JSON.stringify(hoverSelection)}`);
    }
    if (!(hoverSelection.result.gaussianCount > 0)) {
      throw new Error(`expected hover target to expose assigned Gaussians: ${JSON.stringify(hoverSelection)}`);
    }
    if (hoverSelection.hoveredAssignmentSource !== "trainable_kernel_model_artifact") {
      throw new Error(`unexpected hover assignment source: ${hoverSelection.hoveredAssignmentSource}`);
    }
    if (
      hoverSelection.hoveredAssignmentSlotCount !== 2 ||
      hoverSelection.hoveredAssignmentProbeStatus !== "confident" ||
      !(Number(hoverSelection.hoveredAssignmentProbeMargin) > 0.45) ||
      !(Number(hoverSelection.hoveredAssignmentConfidence) > 0.7) ||
      !(Number(hoverSelection.hoveredAssignmentEntropy) > 0) ||
      hoverSelection.hoveredAssignmentTopSlot !== 0
    ) {
      throw new Error(`expected hover assignment preview for trainable ObjectState: ${JSON.stringify(hoverSelection)}`);
    }
    if (
      hoverSelection.hoveredContinuityStatus !== "continuous" ||
      !(Number(hoverSelection.hoveredContinuityBboxDiagonal) > 0) ||
      hoverSelection.hoveredContinuityCentroidContained !== true
    ) {
      throw new Error(`expected hover continuity diagnostic for trainable ObjectState: ${JSON.stringify(hoverSelection)}`);
    }
    if (
      hoverSelection.hoveredTemporalStatus !== "stable" ||
      hoverSelection.hoveredTemporalStable !== true ||
      !(Number(hoverSelection.hoveredTemporalDrift) > 0) ||
      !(Number(hoverSelection.hoveredTemporalDrift) < 0.08) ||
      !(Number(hoverSelection.hoveredAssignmentJitter) > 0) ||
      !(Number(hoverSelection.hoveredAssignmentJitter) < 0.08) ||
      !(Number(hoverSelection.hoveredBboxStability) > 0.5)
    ) {
      throw new Error(`expected hover temporal stability diagnostic for trainable ObjectState: ${JSON.stringify(hoverSelection)}`);
    }
    if (
      hoverSelection.hoveredExplainabilityStatus !== "explainable" ||
      hoverSelection.hoveredExplainable !== true ||
      !(Number(hoverSelection.hoveredExplainabilityScore) > 0.6) ||
      hoverSelection.hoveredExplainabilityReasons !== ""
    ) {
      throw new Error(`expected hover explainability diagnostic for trainable ObjectState: ${JSON.stringify(hoverSelection)}`);
    }
    if (
      hoverSelection.hoverHighlightActive !== true ||
      hoverSelection.hoverHighlightedObjectCount !== 1 ||
      hoverSelection.hoverHighlightedGaussianCount !== hoverSelection.hoveredGaussianCount ||
      hoverSelection.hoverDimmedObjectCount <= 0 ||
      hoverSelection.hoverDimmedGaussianCount <= 0 ||
      !hoverSelection.highlightedSamples.every((sample) => sample.hoverMode === "highlighted" && sample.opacity >= 0.86) ||
      !hoverSelection.dimmedSamples.some((sample) => sample.hoverMode === "dimmed" && sample.opacity <= 0.18)
    ) {
      throw new Error(`expected hover to highlight assigned Gaussian cluster and dim others: ${JSON.stringify(hoverSelection)}`);
    }
    await page.waitForFunction((selectionId) => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const verdict = document.querySelector("[data-object-verdict-panel='true']");
      const hoverHeatmap = document.querySelector("[data-hover-assignment-heatmap='true']");
      const snapshotPanel = document.querySelector("[data-debug-snapshot-panel='true']");
      const snapshot = window.__OBJGAUSS_DEBUG_SNAPSHOT__;
      const world = window.__OBJGAUSS_WORLD__;
      return (
        world?.hoveredId === selectionId &&
        world?.hoveredAssignmentProbeStatus === "confident" &&
        world?.hoveredAssignment?.length === 2 &&
        world?.objectContinuityStatus === "continuous" &&
        world?.hoveredContinuityStatus === "continuous" &&
        Number(world?.objectContinuityBboxDiagonal ?? 0) > 0 &&
        Number(world?.hoveredContinuityBboxDiagonal ?? 0) > 0 &&
        world?.objectContinuityCentroidContained === true &&
        world?.hoveredContinuityCentroidContained === true &&
        world?.objectTemporalStatus === "stable" &&
        world?.hoveredTemporalStatus === "stable" &&
        world?.objectTemporalStable === true &&
        world?.hoveredTemporalStable === true &&
        Number(world?.objectTemporalDrift ?? 0) > 0 &&
        Number(world?.objectAssignmentJitter ?? 0) > 0 &&
        Number(world?.objectBboxStability ?? 0) > 0.5 &&
        Number(world?.hoveredTemporalDrift ?? 0) > 0 &&
        Number(world?.hoveredAssignmentJitter ?? 0) > 0 &&
        Number(world?.hoveredBboxStability ?? 0) > 0.5 &&
        world?.objectExplainabilityStatus === "explainable" &&
        world?.hoveredExplainabilityStatus === "explainable" &&
        world?.objectExplainable === true &&
        world?.hoveredExplainable === true &&
        Number(world?.objectExplainabilityScore ?? 0) > 0.6 &&
        Number(world?.hoveredExplainabilityScore ?? 0) > 0.6 &&
        world?.objectExplainabilityReasons === "" &&
        world?.hoveredExplainabilityReasons === "" &&
        world?.hoverHighlightActive === true &&
        world?.hoverHighlightedObjectCount === 1 &&
        world?.hoverHighlightedGaussianCount === world?.hoveredGaussianCount &&
        world?.hoverDimmedObjectCount > 0 &&
        shell?.getAttribute("data-object-continuity-status") === "continuous" &&
        shell?.getAttribute("data-object-continuity-centroid-contained") === "true" &&
        Number(shell?.getAttribute("data-object-continuity-bbox-diagonal") ?? 0) > 0 &&
        shell?.getAttribute("data-object-temporal-status") === "stable" &&
        shell?.getAttribute("data-object-temporal-stable") === "true" &&
        Number(shell?.getAttribute("data-object-temporal-drift") ?? 0) > 0 &&
        Number(shell?.getAttribute("data-object-assignment-jitter") ?? 0) > 0 &&
        Number(shell?.getAttribute("data-object-bbox-stability") ?? 0) > 0.5 &&
        shell?.getAttribute("data-object-explainability-status") === "explainable" &&
        shell?.getAttribute("data-object-explainable") === "true" &&
        Number(shell?.getAttribute("data-object-explainability-score") ?? 0) > 0.6 &&
        shell?.getAttribute("data-object-explainability-reasons") === "" &&
        shell?.getAttribute("data-hovered-target") === selectionId &&
        shell?.getAttribute("data-hovered-model") === "trainable-mvp-debug" &&
        Number(shell?.getAttribute("data-hovered-gaussians") ?? 0) > 0 &&
        shell?.getAttribute("data-hover-highlight") === "enabled" &&
        shell?.getAttribute("data-hover-highlight-object") === selectionId &&
        Number(shell?.getAttribute("data-hover-highlight-gaussians") ?? 0) === world?.hoveredGaussianCount &&
        shell?.getAttribute("data-hover-assignment-source") === "trainable_kernel_model_artifact" &&
        shell?.getAttribute("data-hover-assignment-slots") === "2" &&
        shell?.getAttribute("data-hover-assignment-probe-status") === "confident" &&
        Number(shell?.getAttribute("data-hover-assignment-probe-margin") ?? 0) > 0.45 &&
        shell?.getAttribute("data-hover-continuity-status") === "continuous" &&
        shell?.getAttribute("data-hover-continuity-centroid-contained") === "true" &&
        Number(shell?.getAttribute("data-hover-continuity-bbox-diagonal") ?? 0) > 0 &&
        shell?.getAttribute("data-hover-temporal-status") === "stable" &&
        shell?.getAttribute("data-hover-temporal-stable") === "true" &&
        Number(shell?.getAttribute("data-hover-temporal-drift") ?? 0) > 0 &&
        Number(shell?.getAttribute("data-hover-assignment-jitter") ?? 0) > 0 &&
        Number(shell?.getAttribute("data-hover-bbox-stability") ?? 0) > 0.5 &&
        shell?.getAttribute("data-hover-explainability-status") === "explainable" &&
        shell?.getAttribute("data-hover-explainable") === "true" &&
        Number(shell?.getAttribute("data-hover-explainability-score") ?? 0) > 0.6 &&
        shell?.getAttribute("data-hover-explainability-reasons") === "" &&
        panel?.getAttribute("data-object-continuity-status") === "continuous" &&
        panel?.getAttribute("data-object-continuity-centroid-contained") === "true" &&
        Number(panel?.getAttribute("data-object-continuity-bbox-diagonal") ?? 0) > 0 &&
        panel?.getAttribute("data-object-temporal-status") === "stable" &&
        panel?.getAttribute("data-object-temporal-stable") === "true" &&
        Number(panel?.getAttribute("data-object-temporal-drift") ?? 0) > 0 &&
        Number(panel?.getAttribute("data-object-assignment-jitter") ?? 0) > 0 &&
        Number(panel?.getAttribute("data-object-bbox-stability") ?? 0) > 0.5 &&
        panel?.getAttribute("data-object-explainability-status") === "explainable" &&
        panel?.getAttribute("data-object-explainable") === "true" &&
        Number(panel?.getAttribute("data-object-explainability-score") ?? 0) > 0.6 &&
        panel?.getAttribute("data-object-explainability-reasons") === "" &&
        panel?.getAttribute("data-hover-highlight") === "enabled" &&
        panel?.getAttribute("data-hover-highlight-object") === selectionId &&
        panel?.getAttribute("data-hover-assignment-source") === "trainable_kernel_model_artifact" &&
        panel?.getAttribute("data-hover-assignment-probe-status") === "confident" &&
        panel?.getAttribute("data-hover-continuity-status") === "continuous" &&
        panel?.getAttribute("data-hover-continuity-centroid-contained") === "true" &&
        Number(panel?.getAttribute("data-hover-continuity-bbox-diagonal") ?? 0) > 0 &&
        panel?.getAttribute("data-hover-temporal-status") === "stable" &&
        panel?.getAttribute("data-hover-temporal-stable") === "true" &&
        Number(panel?.getAttribute("data-hover-temporal-drift") ?? 0) > 0 &&
        Number(panel?.getAttribute("data-hover-assignment-jitter") ?? 0) > 0 &&
        Number(panel?.getAttribute("data-hover-bbox-stability") ?? 0) > 0.5 &&
        panel?.getAttribute("data-hover-explainability-status") === "explainable" &&
        panel?.getAttribute("data-hover-explainable") === "true" &&
        Number(panel?.getAttribute("data-hover-explainability-score") ?? 0) > 0.6 &&
        panel?.getAttribute("data-hover-explainability-reasons") === "" &&
        hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-target") === selectionId &&
        hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-model") === "trainable-mvp-debug" &&
        hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-source") === "trainable_kernel_model_artifact" &&
        hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-slots") === "2" &&
        hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-status") === "confident" &&
        Number(hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-margin") ?? 0) > 0.45 &&
        hoverHeatmap?.querySelectorAll("[data-hover-assignment-row='true']").length === 2 &&
        hoverHeatmap?.querySelector("[data-hover-assignment-row-slot='0']") !== null &&
        verdict?.getAttribute("data-object-verdict-status") === "explainable" &&
        verdict?.getAttribute("data-object-verdict-explainable") === "true" &&
        Number(verdict?.getAttribute("data-object-verdict-score") ?? 0) > 0.6 &&
        verdict?.getAttribute("data-object-verdict-reason-count") === "0" &&
        verdict?.getAttribute("data-object-verdict-clear") === "true" &&
        verdict?.getAttribute("data-hover-verdict-status") === "explainable" &&
        verdict?.getAttribute("data-hover-verdict-explainable") === "true" &&
        Number(verdict?.getAttribute("data-hover-verdict-score") ?? 0) > 0.6 &&
        verdict?.getAttribute("data-hover-verdict-reason-count") === "0" &&
        verdict?.getAttribute("data-hover-verdict-clear") === "true" &&
        verdict?.getAttribute("data-hover-verdict-continuity-status") === "continuous" &&
        verdict?.getAttribute("data-hover-verdict-temporal-status") === "stable" &&
        verdict?.querySelector("[data-object-verdict-reason-row='true']")?.getAttribute("data-object-verdict-reason-name") === "clear" &&
        verdict?.querySelector("[data-hover-verdict-reason-row='true']")?.getAttribute("data-hover-verdict-reason-name") === "clear" &&
        snapshot?.continuity?.status === "continuous" &&
        snapshot?.continuity?.centroidContained === true &&
        Number(snapshot?.continuity?.bboxDiagonal ?? 0) > 0 &&
        snapshot?.temporal?.status === "stable" &&
        snapshot?.temporal?.stable === true &&
        Number(snapshot?.temporal?.temporalDrift ?? 0) > 0 &&
        Number(snapshot?.temporal?.assignmentJitter ?? 0) > 0 &&
        Number(snapshot?.temporal?.bboxStability ?? 0) > 0.5 &&
        snapshot?.explainability?.status === "explainable" &&
        snapshot?.explainability?.explainable === true &&
        Number(snapshot?.explainability?.score ?? 0) > 0.6 &&
        snapshot?.explainability?.reasonNames === "" &&
        snapshot?.hover?.selectionId === selectionId &&
        snapshot?.hover?.probe?.status === "confident" &&
        snapshot?.hover?.continuity?.status === "continuous" &&
        snapshot?.hover?.continuity?.centroidContained === true &&
        Number(snapshot?.hover?.continuity?.bboxDiagonal ?? 0) > 0 &&
        snapshot?.hover?.temporal?.status === "stable" &&
        snapshot?.hover?.temporal?.stable === true &&
        Number(snapshot?.hover?.temporal?.temporalDrift ?? 0) > 0 &&
        Number(snapshot?.hover?.temporal?.assignmentJitter ?? 0) > 0 &&
        Number(snapshot?.hover?.temporal?.bboxStability ?? 0) > 0.5 &&
        snapshot?.hover?.explainability?.status === "explainable" &&
        snapshot?.hover?.explainability?.explainable === true &&
        Number(snapshot?.hover?.explainability?.score ?? 0) > 0.6 &&
        snapshot?.hover?.explainability?.reasonNames === "" &&
        snapshotPanel?.getAttribute("data-debug-snapshot-hover-object") === selectionId &&
        snapshotPanel?.getAttribute("data-debug-snapshot-hover-assignment-status") === "confident" &&
        snapshotPanel?.getAttribute("data-debug-snapshot-continuity-status") === "continuous" &&
        snapshotPanel?.getAttribute("data-debug-snapshot-continuity-centroid-contained") === "true" &&
        Number(snapshotPanel?.getAttribute("data-debug-snapshot-continuity-bbox-diagonal") ?? 0) > 0 &&
        snapshotPanel?.getAttribute("data-debug-snapshot-temporal-status") === "stable" &&
        snapshotPanel?.getAttribute("data-debug-snapshot-temporal-stable") === "true" &&
        Number(snapshotPanel?.getAttribute("data-debug-snapshot-temporal-drift") ?? 0) > 0 &&
        Number(snapshotPanel?.getAttribute("data-debug-snapshot-assignment-jitter") ?? 0) > 0 &&
        Number(snapshotPanel?.getAttribute("data-debug-snapshot-bbox-stability") ?? 0) > 0.5 &&
        snapshotPanel?.getAttribute("data-debug-snapshot-explainability-status") === "explainable" &&
        snapshotPanel?.getAttribute("data-debug-snapshot-explainable") === "true" &&
        Number(snapshotPanel?.getAttribute("data-debug-snapshot-explainability-score") ?? 0) > 0.6 &&
        snapshotPanel?.getAttribute("data-debug-snapshot-explainability-reasons") === "" &&
        snapshotPanel?.getAttribute("data-debug-snapshot-hover-continuity-status") === "continuous" &&
        snapshotPanel?.getAttribute("data-debug-snapshot-hover-continuity-centroid-contained") === "true" &&
        Number(snapshotPanel?.getAttribute("data-debug-snapshot-hover-continuity-bbox-diagonal") ?? 0) > 0 &&
        snapshotPanel?.getAttribute("data-debug-snapshot-hover-temporal-status") === "stable" &&
        snapshotPanel?.getAttribute("data-debug-snapshot-hover-temporal-stable") === "true" &&
        Number(snapshotPanel?.getAttribute("data-debug-snapshot-hover-temporal-drift") ?? 0) > 0 &&
        Number(snapshotPanel?.getAttribute("data-debug-snapshot-hover-assignment-jitter") ?? 0) > 0 &&
        snapshotPanel?.getAttribute("data-debug-snapshot-hover-explainability-status") === "explainable" &&
        snapshotPanel?.getAttribute("data-debug-snapshot-hover-explainable") === "true" &&
        Number(snapshotPanel?.getAttribute("data-debug-snapshot-hover-explainability-score") ?? 0) > 0.6
      );
    }, trainableSelection.selectionId, { timeout: 15000 });
    const toggleTarget = await page.evaluate((selectedId) => {
      const world = window.__OBJGAUSS_WORLD__;
      const target = world?.objectSelections?.find(
        (entry) => entry.modelId === "trainable-mvp-debug" && entry.selectionId !== selectedId && entry.visible,
      );
      return {
        selectionId: target?.selectionId ?? null,
        gaussianCount: target?.gaussianCount ?? 0,
        beforeVisibleObjects: world?.visibleObjectCount ?? null,
        beforeVisibleGaussians: world?.visibleGaussianCount ?? null,
        beforeHiddenObjects: world?.hiddenObjectCount ?? 0,
        beforeHiddenGaussians: world?.hiddenGaussianCount ?? 0,
      };
    }, trainableSelection.selectionId);
    if (!toggleTarget.selectionId || !(toggleTarget.gaussianCount > 0)) {
      throw new Error(`expected a secondary trainable object toggle target: ${JSON.stringify(toggleTarget)}`);
    }
    await page.evaluate((selectionId) => {
      const button = document.querySelector(`[data-object-toggle="${CSS.escape(selectionId)}"]`);
      button?.click();
    }, toggleTarget.selectionId);
    const visibilityToggle = await page.waitForFunction((target) => {
      const world = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const row = document.querySelector(`[data-object-toggle="${CSS.escape(target.selectionId)}"]`);
      const snapshot = window.__OBJGAUSS_DEBUG_SNAPSHOT__;
      const hiddenGaussians = Number(shell?.getAttribute("data-hidden-gaussians") ?? 0);
      const visibleGaussians = Number(shell?.getAttribute("data-visible-gaussians") ?? 0);
      const expectedHiddenObjects = Number(target.beforeHiddenObjects ?? 0) + 1;
      const expectedHiddenGaussians = Number(target.beforeHiddenGaussians ?? 0) + Number(target.gaussianCount ?? 0);
      const expectedVisibleObjects = Number(target.beforeVisibleObjects ?? 0) - 1;
      const expectedVisibleGaussians = Number(target.beforeVisibleGaussians ?? 0) - Number(target.gaussianCount ?? 0);
      if (
        world?.visibleObjectCount !== expectedVisibleObjects ||
        world?.hiddenObjectCount !== expectedHiddenObjects ||
        world?.hiddenGaussianCount !== expectedHiddenGaussians ||
        world?.visibleGaussianCount !== expectedVisibleGaussians ||
        shell?.getAttribute("data-object-visibility-contract") !== "enabled" ||
        shell?.getAttribute("data-hidden-objects") !== String(expectedHiddenObjects) ||
        hiddenGaussians !== expectedHiddenGaussians ||
        visibleGaussians !== expectedVisibleGaussians ||
        panel?.getAttribute("data-object-visibility-contract") !== "enabled" ||
        panel?.getAttribute("data-hidden-objects") !== String(expectedHiddenObjects) ||
        Number(panel?.getAttribute("data-hidden-gaussians") ?? 0) !== expectedHiddenGaussians ||
        row?.getAttribute("data-object-visible") !== "false" ||
        Number(row?.getAttribute("data-object-hidden-gaussians") ?? 0) !== Number(target.gaussianCount ?? 0) ||
        snapshot?.visibility?.hiddenObjectCount !== expectedHiddenObjects ||
        snapshot?.visibility?.hiddenGaussianCount !== expectedHiddenGaussians
      ) {
        return null;
      }
      return {
        hiddenObjects: expectedHiddenObjects,
        hiddenGaussians: expectedHiddenGaussians,
        visibleObjects: expectedVisibleObjects,
        visibleGaussians: expectedVisibleGaussians,
      };
    }, toggleTarget, { timeout: 15000 }).then((handle) => handle.jsonValue());
    await page.waitForFunction(() => {
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const types = new Set(events.map((event) => event.type));
      const shell = document.querySelector(".worldShell");
      const trace = document.querySelector("[data-debug-event-trace='true']");
      const snapshot = window.__OBJGAUSS_DEBUG_SNAPSHOT__;
      return (
        events.length > 0 &&
        types.has("gaussian-probe") &&
        types.has("debug-lens") &&
        types.has("frame-select") &&
        types.has("hover-object") &&
        types.has("toggle-visibility") &&
        shell?.getAttribute("data-debug-event-count") === String(events.length) &&
        trace?.getAttribute("data-debug-event-count") === String(events.length) &&
        snapshot?.events?.length === events.length
      );
    }, undefined, { timeout: 15000 });

    const relevantIssues = consoleIssues.filter(
      (issue) =>
        !issue.includes("THREE.WebGLRenderer") &&
        !issue.includes("Multiple instances of Three.js") &&
        !issue.includes("GPU stall due to ReadPixels") &&
        !issue.includes("WebGL warning"),
    );
    if (relevantIssues.length > 0) {
      throw new Error(`browser console issues:\n${relevantIssues.join("\n")}`);
    }

    const screenshotPath = "/tmp/objgauss-world-viewer.png";
    await page.screenshot({ path: screenshotPath, fullPage: false });
    const mobileScreenshotPath = await auditMobileWorld(browser, url);
    const urlArtifact = await auditUrlTrainableArtifact(browser, url);
    const urlOgc = await auditUrlOgcArtifact(browser, url);
    const urlOgcManifest = await auditUrlOgcManifestArtifact(browser, url);
    const algorithmManifest = await auditAlgorithmManifestBundle(browser, url);
    const localModelManifest = await auditLocalModelManifestBundleImport(browser, url);
    const localTrainableManifest = await auditLocalTrainableManifestPackageImport(browser, url);
    const localArtifact = await auditLocalTrainableArtifactImport(browser, url);
    const localOgc = await auditLocalOgcArtifactImport(browser, url);
    const localOgcManifest = await auditLocalOgcManifestPackageImport(browser, url);
    const world = await page.evaluate(() => {
      const handle = window.__OBJGAUSS_WORLD__;
      const snapshot = window.__OBJGAUSS_DEBUG_SNAPSHOT__;
      const shell = document.querySelector(".worldShell");
      const stability = document.querySelector("[data-stability-dashboard='true']");
      const training = document.querySelector("[data-training-evidence='true']");
      const solver = document.querySelector("[data-solver-training-evidence='true']");
      const debugPanel = document.querySelector("[data-object-debug-panel='true']");
      const probePanel = document.querySelector("[data-gaussian-probe-panel='true']");
      const timelinePanel = document.querySelector("[data-assignment-timeline-panel='true']");
      const fragmentPanel = document.querySelector("[data-object-fragmentation-panel='true']");
      const verdict = document.querySelector("[data-object-verdict-panel='true']");
      const hoverHeatmap = document.querySelector("[data-hover-assignment-heatmap='true']");
      const snapshotPanel = document.querySelector("[data-debug-snapshot-panel='true']");
      const tracePanel = document.querySelector("[data-debug-event-trace='true']");
      const trainingStagePanel = document.querySelector("[data-training-stage-panel='true']");
      const worldPlane = document.querySelector("[data-world-plane-panel='true']");
      const interactionLayer = document.querySelector("[data-object-interaction-layer='true']");
      const systemDrawer = document.querySelector("[data-system-drawer='true']");
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      return {
        modelCount: handle.modelCount,
        stageVisibleModelCount: handle.visibleModelCount,
        stageHiddenModelCount: handle.hiddenModelCount,
        stageModelIds: handle.stageModelIds,
        shellTrainingStage: shell?.getAttribute("data-training-stage") ?? null,
        shellStageVisibleModelCount: Number(shell?.getAttribute("data-stage-visible-model-count") ?? 0),
        shellStageProcessedModelCount: Number(shell?.getAttribute("data-stage-processed-model-count") ?? 0),
        shellStagePendingModelCount: Number(shell?.getAttribute("data-stage-pending-model-count") ?? 0),
        shellRenderSurfaceContract: shell?.getAttribute("data-render-surface-contract") ?? null,
        shellGaussianDisplayMode: shell?.getAttribute("data-gaussian-display-mode") ?? null,
        shellStageDisplayContract: shell?.getAttribute("data-stage-display-contract") ?? null,
        shellSourceSplatActive: shell?.getAttribute("data-source-splat-active") ?? null,
        shellSourceSplatStatus: shell?.getAttribute("data-source-splat-status") ?? null,
        shellSourceSplatPath: shell?.getAttribute("data-source-splat-path") ?? null,
        shellFullGaussianSourceCount: Number(shell?.getAttribute("data-full-gaussian-source-count") ?? 0),
        shellObjectLayerRegisteredCount: Number(shell?.getAttribute("data-object-layer-registered-count") ?? 0),
        shellObjectLayerLoadedCount: Number(shell?.getAttribute("data-object-layer-loaded-count") ?? 0),
        panelTrainingStageSchema: trainingStagePanel?.getAttribute("data-training-stage-schema") ?? null,
        panelStageVisibleModelCount: Number(trainingStagePanel?.getAttribute("data-stage-visible-models") ?? 0),
        panelStageRenderSurfaceContract: trainingStagePanel?.getAttribute("data-stage-render-surface-contract") ?? null,
        panelStageDisplayContract: trainingStagePanel?.getAttribute("data-stage-display-contract") ?? null,
        panelStageSourceSplatModels: Number(trainingStagePanel?.getAttribute("data-stage-source-splat-models") ?? 0),
        panelStageFullGaussianSources: Number(trainingStagePanel?.getAttribute("data-stage-full-gaussian-sources") ?? 0),
        panelStagePointPreviewModels: Number(trainingStagePanel?.getAttribute("data-stage-point-preview-models") ?? 0),
        panelStageObjectLayerRegistered: Number(trainingStagePanel?.getAttribute("data-stage-object-layer-registered") ?? 0),
        panelStageObjectLayerLoaded: Number(trainingStagePanel?.getAttribute("data-stage-object-layer-loaded") ?? 0),
        panelStageProcessedModelCount: Number(trainingStagePanel?.getAttribute("data-stage-processed-models") ?? 0),
        panelStagePendingModelCount: Number(trainingStagePanel?.getAttribute("data-stage-pending-models") ?? 0),
        panelModelVersionRows: trainingStagePanel?.querySelectorAll("[data-model-version-row-id]").length ?? 0,
        panelLayerBadgeRows: trainingStagePanel?.querySelectorAll("[data-model-layer-badges='true']").length ?? 0,
        worldPlanePresent: Boolean(worldPlane),
        systemDrawerPresent: Boolean(systemDrawer),
        objectTransformContract: handle.objectTransformContract ?? null,
        objectTransformEngine: handle.objectTransformEngine ?? null,
        transformGizmoAttached: Boolean(handle.transformGizmoAttached),
        transformGizmoObject: handle.transformGizmoObject ?? null,
        transformHistoryDepth: handle.transformHistoryDepth ?? null,
        transformRedoDepth: handle.transformRedoDepth ?? null,
        transformCanUndo: handle.transformCanUndo ?? null,
        transformCanRedo: handle.transformCanRedo ?? null,
        transformSnapStep: handle.transformSnapStep ?? null,
        objectTransformMode: handle.objectTransformMode ?? null,
        objectInteractionLayerContract: interactionLayer?.getAttribute("data-object-transform-primary") ?? null,
        objectInteractionLayerSelected: interactionLayer?.getAttribute("data-object-move-selected") ?? null,
        bottomStatusCount: document.querySelectorAll(".bottomStatus").length,
        objectCount: handle.objectCount,
        draggableObjectCount: handle.draggableObjectCount,
        selectedId: handle.selectedId,
        selectedModelId: handle.selectedModelId,
        selectedObjectId: handle.selectedObjectId,
        debugMode: handle.debugMode,
        debugLens: handle.debugLens,
        objectOverlayMode: handle.objectOverlayMode,
        objectOverlayBboxVisible: handle.objectOverlayBboxVisible,
        objectOverlayCentroidVisible: handle.objectOverlayCentroidVisible,
        debugProtocol: handle.debugProtocol,
        debugSnapshotSchema: snapshot?.schema ?? null,
        debugSnapshotProtocol: snapshot?.protocol ?? null,
        debugSnapshotModel: snapshot?.model?.id ?? null,
        debugSnapshotObject: snapshot?.selection?.objectId ?? null,
        debugSnapshotLens: snapshot?.debug?.lens ?? null,
        debugSnapshotOverlayMode: snapshot?.debug?.overlayMode ?? null,
        debugSnapshotAssignmentSource: snapshot?.assignment?.source ?? null,
        debugSnapshotAssignmentSlots: Number(snapshot?.assignment?.slotCount ?? 0),
        debugSnapshotAssignmentProbeStatus: snapshot?.assignment?.probe?.status ?? null,
        debugSnapshotAssignmentProbeMargin: snapshot?.assignment?.probe?.margin ?? null,
        debugSnapshotAssignmentTimelineStatus: snapshot?.assignment?.timeline?.status ?? null,
        debugSnapshotAssignmentTimelineJitter: snapshot?.assignment?.timeline?.meanDelta ?? null,
        debugSnapshotAssignmentTimelineFrameCount: snapshot?.assignment?.timeline?.frameCount ?? null,
        debugSnapshotContinuityStatus: snapshot?.continuity?.status ?? null,
        debugSnapshotContinuityBboxDiagonal: snapshot?.continuity?.bboxDiagonal ?? null,
        debugSnapshotContinuityCentroidContained: snapshot?.continuity?.centroidContained ?? false,
        debugSnapshotTemporalStatus: snapshot?.temporal?.status ?? null,
        debugSnapshotTemporalDrift: snapshot?.temporal?.temporalDrift ?? null,
        debugSnapshotAssignmentJitter: snapshot?.temporal?.assignmentJitter ?? null,
        debugSnapshotBboxStability: snapshot?.temporal?.bboxStability ?? null,
        debugSnapshotTemporalStable: snapshot?.temporal?.stable ?? false,
        debugSnapshotExplainabilityStatus: snapshot?.explainability?.status ?? null,
        debugSnapshotExplainable: snapshot?.explainability?.explainable ?? false,
        debugSnapshotExplainabilityScore: snapshot?.explainability?.score ?? null,
        debugSnapshotExplainabilityReasons: snapshot?.explainability?.reasonNames ?? null,
        debugSnapshotTrainingStatus: snapshot?.training?.status ?? null,
        debugSnapshotEventCount: Array.isArray(snapshot?.events) ? snapshot.events.length : 0,
        debugSnapshotEventTypes: Array.isArray(snapshot?.events) ? snapshot.events.map((event) => event.type) : [],
        shellDebugSnapshotSchema: shell?.getAttribute("data-debug-snapshot-schema") ?? null,
        shellDebugSnapshotModel: shell?.getAttribute("data-debug-snapshot-model") ?? null,
        shellDebugSnapshotSlots: Number(shell?.getAttribute("data-debug-snapshot-assignment-slots") ?? 0),
        shellDebugSnapshotAssignmentProbeStatus: shell?.getAttribute("data-debug-snapshot-assignment-probe-status") ?? null,
        shellDebugSnapshotStability: shell?.getAttribute("data-debug-snapshot-stability") ?? null,
        debugEventCount: events.length,
        debugEventLast: events[0]?.type ?? null,
        debugEventTypes: events.map((event) => event.type),
        shellDebugEventCount: Number(shell?.getAttribute("data-debug-event-count") ?? 0),
        shellDebugEventLast: shell?.getAttribute("data-debug-event-last") ?? null,
        panelDebugEventCount: Number(tracePanel?.getAttribute("data-debug-event-count") ?? 0),
        panelDebugEventLast: tracePanel?.getAttribute("data-debug-event-last") ?? null,
        panelDebugEventSchema: tracePanel?.getAttribute("data-debug-event-schema") ?? null,
        panelDebugSnapshotSchema: snapshotPanel?.getAttribute("data-debug-snapshot-schema") ?? null,
        panelDebugSnapshotModel: snapshotPanel?.getAttribute("data-debug-snapshot-model") ?? null,
        panelDebugSnapshotLens: snapshotPanel?.getAttribute("data-debug-snapshot-lens") ?? null,
        panelDebugSnapshotSlots: Number(snapshotPanel?.getAttribute("data-debug-snapshot-slots") ?? 0),
        panelDebugSnapshotAssignmentProbeStatus: snapshotPanel?.getAttribute("data-debug-snapshot-assignment-probe-status") ?? null,
        panelDebugSnapshotAssignmentTimelineStatus: snapshotPanel?.getAttribute("data-debug-snapshot-assignment-timeline-status") ?? null,
        panelDebugSnapshotAssignmentTimelineJitter: Number(snapshotPanel?.getAttribute("data-debug-snapshot-assignment-timeline-jitter") ?? 0),
        panelDebugSnapshotContinuityStatus: snapshotPanel?.getAttribute("data-debug-snapshot-continuity-status") ?? null,
        panelDebugSnapshotContinuityBboxDiagonal: Number(snapshotPanel?.getAttribute("data-debug-snapshot-continuity-bbox-diagonal") ?? 0),
        panelDebugSnapshotContinuityCentroidContained: snapshotPanel?.getAttribute("data-debug-snapshot-continuity-centroid-contained") ?? null,
        panelDebugSnapshotTemporalStatus: snapshotPanel?.getAttribute("data-debug-snapshot-temporal-status") ?? null,
        panelDebugSnapshotTemporalDrift: Number(snapshotPanel?.getAttribute("data-debug-snapshot-temporal-drift") ?? 0),
        panelDebugSnapshotAssignmentJitter: Number(snapshotPanel?.getAttribute("data-debug-snapshot-assignment-jitter") ?? 0),
        panelDebugSnapshotBboxStability: Number(snapshotPanel?.getAttribute("data-debug-snapshot-bbox-stability") ?? 0),
        panelDebugSnapshotTemporalStable: snapshotPanel?.getAttribute("data-debug-snapshot-temporal-stable") ?? null,
        panelDebugSnapshotExplainabilityStatus: snapshotPanel?.getAttribute("data-debug-snapshot-explainability-status") ?? null,
        panelDebugSnapshotExplainable: snapshotPanel?.getAttribute("data-debug-snapshot-explainable") ?? null,
        panelDebugSnapshotExplainabilityScore: Number(snapshotPanel?.getAttribute("data-debug-snapshot-explainability-score") ?? 0),
        panelDebugSnapshotExplainabilityReasons: snapshotPanel?.getAttribute("data-debug-snapshot-explainability-reasons") ?? null,
        assignmentSource: handle.assignmentSource,
        assignmentProbeStatus: handle.assignmentProbeStatus ?? null,
        assignmentProbeMargin: handle.assignmentProbeMargin ?? null,
        shellAssignmentProbeStatus: shell?.getAttribute("data-assignment-probe-status") ?? null,
        shellAssignmentProbeMargin: Number(shell?.getAttribute("data-assignment-probe-margin") ?? 0),
        shellAssignmentTimelineStatus: shell?.getAttribute("data-assignment-timeline-status") ?? null,
        shellAssignmentTimelineFrameCount: Number(shell?.getAttribute("data-assignment-timeline-frame-count") ?? 0),
        shellAssignmentTimelineJitter: Number(shell?.getAttribute("data-assignment-timeline-jitter") ?? 0),
        panelAssignmentTimelineStatus: debugPanel?.getAttribute("data-assignment-timeline-status") ?? null,
        panelAssignmentTimelineFrameCount: Number(debugPanel?.getAttribute("data-assignment-timeline-frame-count") ?? 0),
        timelinePanelStatus: timelinePanel?.getAttribute("data-assignment-timeline-status") ?? null,
        timelinePanelFrameCount: Number(timelinePanel?.getAttribute("data-assignment-timeline-frame-count") ?? 0),
        timelinePanelCurrentFrame: timelinePanel?.getAttribute("data-assignment-timeline-current-frame") ?? null,
        timelinePanelGaussianIndex: timelinePanel?.getAttribute("data-assignment-timeline-gaussian-index") ?? null,
        timelinePanelJitter: Number(timelinePanel?.getAttribute("data-assignment-timeline-jitter") ?? 0),
        timelinePanelMaxDelta: Number(timelinePanel?.getAttribute("data-assignment-timeline-max-delta") ?? 0),
        timelinePanelRows: timelinePanel?.querySelectorAll("[data-assignment-timeline-row='true']").length ?? 0,
        probePanelStatus: probePanel?.getAttribute("data-gaussian-probe-status") ?? null,
        probePanelSource: probePanel?.getAttribute("data-gaussian-probe-source") ?? null,
        probePanelIndex: probePanel?.getAttribute("data-gaussian-probe-index") ?? null,
        probePanelTopSlot: probePanel?.getAttribute("data-gaussian-probe-top-slot") ?? null,
        probePanelMargin: Number(probePanel?.getAttribute("data-gaussian-probe-margin") ?? 0),
        probePanelConfidence: Number(probePanel?.getAttribute("data-gaussian-probe-confidence") ?? 0),
        probePanelEntropy: Number(probePanel?.getAttribute("data-gaussian-probe-entropy") ?? 0),
        probePanelAmbiguous: probePanel?.getAttribute("data-gaussian-probe-ambiguous") ?? null,
        probePanelCollapseRisk: probePanel?.getAttribute("data-gaussian-probe-collapse-risk") ?? null,
        probePanelOpacity: Number(probePanel?.getAttribute("data-gaussian-probe-opacity") ?? 0),
        objectContinuityStatus: handle.objectContinuityStatus ?? null,
        objectContinuityBboxDiagonal: handle.objectContinuityBboxDiagonal ?? null,
        objectContinuityCentroidContained: handle.objectContinuityCentroidContained ?? false,
        objectTemporalStatus: handle.objectTemporalStatus ?? null,
        objectTemporalDrift: handle.objectTemporalDrift ?? null,
        objectAssignmentJitter: handle.objectAssignmentJitter ?? null,
        objectBboxStability: handle.objectBboxStability ?? null,
        objectTemporalStable: handle.objectTemporalStable ?? false,
        objectExplainabilityStatus: handle.objectExplainabilityStatus ?? null,
        objectExplainable: handle.objectExplainable ?? false,
        objectExplainabilityScore: handle.objectExplainabilityScore ?? null,
        objectExplainabilityReasons: handle.objectExplainabilityReasons ?? null,
        shellObjectContinuityStatus: shell?.getAttribute("data-object-continuity-status") ?? null,
        shellObjectContinuityBboxDiagonal: Number(shell?.getAttribute("data-object-continuity-bbox-diagonal") ?? 0),
        shellObjectContinuityCentroidContained: shell?.getAttribute("data-object-continuity-centroid-contained") ?? null,
        shellObjectTemporalStatus: shell?.getAttribute("data-object-temporal-status") ?? null,
        shellObjectTemporalDrift: Number(shell?.getAttribute("data-object-temporal-drift") ?? 0),
        shellObjectAssignmentJitter: Number(shell?.getAttribute("data-object-assignment-jitter") ?? 0),
        shellObjectBboxStability: Number(shell?.getAttribute("data-object-bbox-stability") ?? 0),
        shellObjectTemporalStable: shell?.getAttribute("data-object-temporal-stable") ?? null,
        shellObjectExplainabilityStatus: shell?.getAttribute("data-object-explainability-status") ?? null,
        shellObjectExplainable: shell?.getAttribute("data-object-explainable") ?? null,
        shellObjectExplainabilityScore: Number(shell?.getAttribute("data-object-explainability-score") ?? 0),
        shellObjectExplainabilityReasons: shell?.getAttribute("data-object-explainability-reasons") ?? null,
        panelObjectContinuityStatus: debugPanel?.getAttribute("data-object-continuity-status") ?? null,
        panelObjectContinuityBboxDiagonal: Number(debugPanel?.getAttribute("data-object-continuity-bbox-diagonal") ?? 0),
        panelObjectContinuityCentroidContained: debugPanel?.getAttribute("data-object-continuity-centroid-contained") ?? null,
        fragmentPanelStatus: fragmentPanel?.getAttribute("data-object-fragmentation-status") ?? null,
        fragmentPanelObject: fragmentPanel?.getAttribute("data-object-fragmentation-object") ?? null,
        fragmentPanelGaussians: Number(fragmentPanel?.getAttribute("data-object-fragmentation-gaussians") ?? 0),
        fragmentPanelCompactness: Number(fragmentPanel?.getAttribute("data-object-fragmentation-compactness") ?? 0),
        fragmentPanelBboxDiagonal: Number(fragmentPanel?.getAttribute("data-object-fragmentation-bbox-diagonal") ?? 0),
        fragmentPanelDensity: Number(fragmentPanel?.getAttribute("data-object-fragmentation-density") ?? 0),
        fragmentPanelCentroidContained: fragmentPanel?.getAttribute("data-object-fragmentation-centroid-contained") ?? null,
        fragmentPanelBboxValid: fragmentPanel?.getAttribute("data-object-fragmentation-bbox-valid") ?? null,
        fragmentPanelFragmented: fragmentPanel?.getAttribute("data-object-fragmentation-fragmented") ?? null,
        fragmentPanelCentroidOutside: fragmentPanel?.getAttribute("data-object-fragmentation-centroid-outside") ?? null,
        fragmentPanelDegenerate: fragmentPanel?.getAttribute("data-object-fragmentation-degenerate") ?? null,
        fragmentPanelFlagRows: fragmentPanel?.querySelectorAll("[data-object-fragmentation-flags='true'] .qualityGateRow").length ?? 0,
        panelObjectTemporalStatus: debugPanel?.getAttribute("data-object-temporal-status") ?? null,
        panelObjectTemporalDrift: Number(debugPanel?.getAttribute("data-object-temporal-drift") ?? 0),
        panelObjectAssignmentJitter: Number(debugPanel?.getAttribute("data-object-assignment-jitter") ?? 0),
        panelObjectBboxStability: Number(debugPanel?.getAttribute("data-object-bbox-stability") ?? 0),
        panelObjectTemporalStable: debugPanel?.getAttribute("data-object-temporal-stable") ?? null,
        panelObjectExplainabilityStatus: debugPanel?.getAttribute("data-object-explainability-status") ?? null,
        panelObjectExplainable: debugPanel?.getAttribute("data-object-explainable") ?? null,
        panelObjectExplainabilityScore: Number(debugPanel?.getAttribute("data-object-explainability-score") ?? 0),
        panelObjectExplainabilityReasons: debugPanel?.getAttribute("data-object-explainability-reasons") ?? null,
        objectVerdictStatus: verdict?.getAttribute("data-object-verdict-status") ?? null,
        objectVerdictExplainable: verdict?.getAttribute("data-object-verdict-explainable") ?? null,
        objectVerdictScore: Number(verdict?.getAttribute("data-object-verdict-score") ?? 0),
        objectVerdictReasonCount: Number(verdict?.getAttribute("data-object-verdict-reason-count") ?? -1),
        objectVerdictReasons: verdict?.getAttribute("data-object-verdict-reasons") ?? null,
        objectVerdictClear: verdict?.getAttribute("data-object-verdict-clear") ?? null,
        objectVerdictContinuityStatus: verdict?.getAttribute("data-object-verdict-continuity-status") ?? null,
        objectVerdictTemporalStatus: verdict?.getAttribute("data-object-verdict-temporal-status") ?? null,
        objectVerdictReasonName: verdict?.querySelector("[data-object-verdict-reason-row='true']")?.getAttribute("data-object-verdict-reason-name") ?? null,
        objectVerdictReasonStatus: verdict?.querySelector("[data-object-verdict-reason-row='true']")?.getAttribute("data-object-verdict-reason-status") ?? null,
        stabilityStatus: handle.stabilitySummary?.status ?? null,
        slotUtilization: handle.stabilitySummary?.slotUtilization ?? null,
        mixedSlots: handle.stabilitySummary?.mixedSlots ?? null,
        meanPurity: handle.stabilitySummary?.meanPurity ?? null,
        meanTemporalDrift: handle.stabilitySummary?.meanTemporalDrift ?? null,
        meanSpatialCompactness: handle.stabilitySummary?.meanSpatialCompactness ?? null,
        meanAssignmentJitter: handle.stabilitySummary?.meanAssignmentJitter ?? null,
        meanBboxStability: handle.stabilitySummary?.meanBboxStability ?? null,
        shellMeanPurity: Number(shell?.getAttribute("data-stability-mean-purity") ?? 0),
        shellMeanTemporalDrift: Number(shell?.getAttribute("data-stability-mean-temporal-drift") ?? 0),
        shellMeanSpatialCompactness: Number(shell?.getAttribute("data-stability-mean-spatial-compactness") ?? 0),
        shellMeanAssignmentJitter: Number(shell?.getAttribute("data-stability-mean-assignment-jitter") ?? -1),
        shellMeanBboxStability: Number(shell?.getAttribute("data-stability-mean-bbox-stability") ?? 0),
        dashboardMeanPurity: Number(stability?.getAttribute("data-mean-purity") ?? 0),
        dashboardMeanTemporalDrift: Number(stability?.getAttribute("data-mean-temporal-drift") ?? 0),
        dashboardMeanSpatialCompactness: Number(stability?.getAttribute("data-mean-spatial-compactness") ?? 0),
        dashboardMeanAssignmentJitter: Number(stability?.getAttribute("data-mean-assignment-jitter") ?? -1),
        dashboardMeanBboxStability: Number(stability?.getAttribute("data-mean-bbox-stability") ?? 0),
        hoveredId: handle.hoveredId,
        hoveredModelId: handle.hoveredModelId,
        hoveredObjectId: handle.hoveredObjectId,
        hoveredGaussianCount: handle.hoveredGaussianCount,
        hoveredAssignmentSource: handle.hoveredAssignmentSource ?? null,
        hoveredAssignmentSlotCount: (handle.hoveredAssignment ?? []).length,
        hoveredAssignmentProbeStatus: handle.hoveredAssignmentProbeStatus ?? null,
        hoveredAssignmentProbeMargin: handle.hoveredAssignmentProbeMargin ?? null,
        hoveredAssignmentConfidence: handle.hoveredAssignmentConfidence ?? null,
        hoveredAssignmentEntropy: handle.hoveredAssignmentEntropy ?? null,
        hoveredAssignmentTopSlot: handle.hoveredAssignmentTopSlot ?? null,
        hoveredContinuityStatus: handle.hoveredContinuityStatus ?? null,
        hoveredContinuityBboxDiagonal: handle.hoveredContinuityBboxDiagonal ?? null,
        hoveredContinuityCentroidContained: handle.hoveredContinuityCentroidContained ?? false,
        hoveredTemporalStatus: handle.hoveredTemporalStatus ?? null,
        hoveredTemporalDrift: handle.hoveredTemporalDrift ?? null,
        hoveredAssignmentJitter: handle.hoveredAssignmentJitter ?? null,
        hoveredBboxStability: handle.hoveredBboxStability ?? null,
        hoveredTemporalStable: handle.hoveredTemporalStable ?? false,
        hoveredExplainabilityStatus: handle.hoveredExplainabilityStatus ?? null,
        hoveredExplainable: handle.hoveredExplainable ?? false,
        hoveredExplainabilityScore: handle.hoveredExplainabilityScore ?? null,
        hoveredExplainabilityReasons: handle.hoveredExplainabilityReasons ?? null,
        hoverHighlightActive: handle.hoverHighlightActive ?? false,
        hoverHighlightedObjectCount: handle.hoverHighlightedObjectCount ?? 0,
        hoverHighlightedGaussianCount: handle.hoverHighlightedGaussianCount ?? 0,
        hoverDimmedObjectCount: handle.hoverDimmedObjectCount ?? 0,
        hoverDimmedGaussianCount: handle.hoverDimmedGaussianCount ?? 0,
        hoverHighlightSampleCount: (handle.hoverHighlightSamples ?? []).length,
        hoverHighlightedSamples: (handle.hoverHighlightSamples ?? []).filter((sample) => sample.hoverHighlighted),
        hoverDimmedSamples: (handle.hoverHighlightSamples ?? []).filter((sample) => sample.hoverDimmed),
        shellHoveredTarget: shell?.getAttribute("data-hovered-target") ?? null,
        shellHoveredGaussians: Number(shell?.getAttribute("data-hovered-gaussians") ?? 0),
        shellHoverHighlight: shell?.getAttribute("data-hover-highlight") ?? null,
        shellHoverHighlightObject: shell?.getAttribute("data-hover-highlight-object") ?? null,
        shellHoverHighlightGaussians: Number(shell?.getAttribute("data-hover-highlight-gaussians") ?? 0),
        shellHoverAssignmentSource: shell?.getAttribute("data-hover-assignment-source") ?? null,
        shellHoverAssignmentSlots: Number(shell?.getAttribute("data-hover-assignment-slots") ?? 0),
        shellHoverAssignmentStatus: shell?.getAttribute("data-hover-assignment-probe-status") ?? null,
        shellHoverAssignmentMargin: Number(shell?.getAttribute("data-hover-assignment-probe-margin") ?? 0),
        shellHoverContinuityStatus: shell?.getAttribute("data-hover-continuity-status") ?? null,
        shellHoverContinuityBboxDiagonal: Number(shell?.getAttribute("data-hover-continuity-bbox-diagonal") ?? 0),
        shellHoverContinuityCentroidContained: shell?.getAttribute("data-hover-continuity-centroid-contained") ?? null,
        shellHoverTemporalStatus: shell?.getAttribute("data-hover-temporal-status") ?? null,
        shellHoverTemporalDrift: Number(shell?.getAttribute("data-hover-temporal-drift") ?? 0),
        shellHoverAssignmentJitter: Number(shell?.getAttribute("data-hover-assignment-jitter") ?? 0),
        shellHoverBboxStability: Number(shell?.getAttribute("data-hover-bbox-stability") ?? 0),
        shellHoverTemporalStable: shell?.getAttribute("data-hover-temporal-stable") ?? null,
        shellHoverExplainabilityStatus: shell?.getAttribute("data-hover-explainability-status") ?? null,
        shellHoverExplainable: shell?.getAttribute("data-hover-explainable") ?? null,
        shellHoverExplainabilityScore: Number(shell?.getAttribute("data-hover-explainability-score") ?? 0),
        shellHoverExplainabilityReasons: shell?.getAttribute("data-hover-explainability-reasons") ?? null,
        panelHoverAssignmentSource: debugPanel?.getAttribute("data-hover-assignment-source") ?? null,
        panelHoverAssignmentStatus: debugPanel?.getAttribute("data-hover-assignment-probe-status") ?? null,
        panelHoverContinuityStatus: debugPanel?.getAttribute("data-hover-continuity-status") ?? null,
        panelHoverContinuityBboxDiagonal: Number(debugPanel?.getAttribute("data-hover-continuity-bbox-diagonal") ?? 0),
        panelHoverContinuityCentroidContained: debugPanel?.getAttribute("data-hover-continuity-centroid-contained") ?? null,
        panelHoverTemporalStatus: debugPanel?.getAttribute("data-hover-temporal-status") ?? null,
        panelHoverTemporalDrift: Number(debugPanel?.getAttribute("data-hover-temporal-drift") ?? 0),
        panelHoverAssignmentJitter: Number(debugPanel?.getAttribute("data-hover-assignment-jitter") ?? 0),
        panelHoverBboxStability: Number(debugPanel?.getAttribute("data-hover-bbox-stability") ?? 0),
        panelHoverTemporalStable: debugPanel?.getAttribute("data-hover-temporal-stable") ?? null,
        panelHoverExplainabilityStatus: debugPanel?.getAttribute("data-hover-explainability-status") ?? null,
        panelHoverExplainable: debugPanel?.getAttribute("data-hover-explainable") ?? null,
        panelHoverExplainabilityScore: Number(debugPanel?.getAttribute("data-hover-explainability-score") ?? 0),
        panelHoverExplainabilityReasons: debugPanel?.getAttribute("data-hover-explainability-reasons") ?? null,
        hoverHeatmapTarget: hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-target") ?? null,
        hoverHeatmapModel: hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-model") ?? null,
        hoverHeatmapSource: hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-source") ?? null,
        hoverHeatmapSlots: Number(hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-slots") ?? 0),
        hoverHeatmapStatus: hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-status") ?? null,
        hoverHeatmapTopSlot: Number(hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-top-slot") ?? -1),
        hoverHeatmapMargin: Number(hoverHeatmap?.getAttribute("data-hover-assignment-heatmap-margin") ?? 0),
        hoverHeatmapRowCount: hoverHeatmap?.querySelectorAll("[data-hover-assignment-row='true']").length ?? 0,
        hoverHeatmapTopRowProbability: Number(
          hoverHeatmap?.querySelector("[data-hover-assignment-row-slot='0']")?.getAttribute("data-hover-assignment-row-probability") ?? 0,
        ),
        hoverVerdictStatus: verdict?.getAttribute("data-hover-verdict-status") ?? null,
        hoverVerdictExplainable: verdict?.getAttribute("data-hover-verdict-explainable") ?? null,
        hoverVerdictScore: Number(verdict?.getAttribute("data-hover-verdict-score") ?? 0),
        hoverVerdictReasonCount: Number(verdict?.getAttribute("data-hover-verdict-reason-count") ?? -1),
        hoverVerdictReasons: verdict?.getAttribute("data-hover-verdict-reasons") ?? null,
        hoverVerdictClear: verdict?.getAttribute("data-hover-verdict-clear") ?? null,
        hoverVerdictContinuityStatus: verdict?.getAttribute("data-hover-verdict-continuity-status") ?? null,
        hoverVerdictTemporalStatus: verdict?.getAttribute("data-hover-verdict-temporal-status") ?? null,
        hoverVerdictReasonName: verdict?.querySelector("[data-hover-verdict-reason-row='true']")?.getAttribute("data-hover-verdict-reason-name") ?? null,
        hoverVerdictReasonStatus: verdict?.querySelector("[data-hover-verdict-reason-row='true']")?.getAttribute("data-hover-verdict-reason-status") ?? null,
        snapshotHoverObject: snapshot?.hover?.selectionId ?? null,
        snapshotHoverAssignmentStatus: snapshot?.hover?.probe?.status ?? null,
        snapshotHoverContinuityStatus: snapshot?.hover?.continuity?.status ?? null,
        snapshotHoverContinuityBboxDiagonal: snapshot?.hover?.continuity?.bboxDiagonal ?? null,
        snapshotHoverContinuityCentroidContained: snapshot?.hover?.continuity?.centroidContained ?? false,
        snapshotHoverTemporalStatus: snapshot?.hover?.temporal?.status ?? null,
        snapshotHoverTemporalDrift: snapshot?.hover?.temporal?.temporalDrift ?? null,
        snapshotHoverAssignmentJitter: snapshot?.hover?.temporal?.assignmentJitter ?? null,
        snapshotHoverBboxStability: snapshot?.hover?.temporal?.bboxStability ?? null,
        snapshotHoverTemporalStable: snapshot?.hover?.temporal?.stable ?? false,
        snapshotHoverExplainabilityStatus: snapshot?.hover?.explainability?.status ?? null,
        snapshotHoverExplainable: snapshot?.hover?.explainability?.explainable ?? false,
        snapshotHoverExplainabilityScore: snapshot?.hover?.explainability?.score ?? null,
        snapshotHoverExplainabilityReasons: snapshot?.hover?.explainability?.reasonNames ?? null,
        panelSnapshotHoverObject: snapshotPanel?.getAttribute("data-debug-snapshot-hover-object") ?? null,
        panelSnapshotHoverAssignmentStatus: snapshotPanel?.getAttribute("data-debug-snapshot-hover-assignment-status") ?? null,
        panelSnapshotHoverContinuityStatus: snapshotPanel?.getAttribute("data-debug-snapshot-hover-continuity-status") ?? null,
        panelSnapshotHoverContinuityBboxDiagonal: Number(snapshotPanel?.getAttribute("data-debug-snapshot-hover-continuity-bbox-diagonal") ?? 0),
        panelSnapshotHoverContinuityCentroidContained: snapshotPanel?.getAttribute("data-debug-snapshot-hover-continuity-centroid-contained") ?? null,
        panelSnapshotHoverTemporalStatus: snapshotPanel?.getAttribute("data-debug-snapshot-hover-temporal-status") ?? null,
        panelSnapshotHoverTemporalDrift: Number(snapshotPanel?.getAttribute("data-debug-snapshot-hover-temporal-drift") ?? 0),
        panelSnapshotHoverAssignmentJitter: Number(snapshotPanel?.getAttribute("data-debug-snapshot-hover-assignment-jitter") ?? 0),
        panelSnapshotHoverTemporalStable: snapshotPanel?.getAttribute("data-debug-snapshot-hover-temporal-stable") ?? null,
        panelSnapshotHoverExplainabilityStatus: snapshotPanel?.getAttribute("data-debug-snapshot-hover-explainability-status") ?? null,
        panelSnapshotHoverExplainable: snapshotPanel?.getAttribute("data-debug-snapshot-hover-explainable") ?? null,
        panelSnapshotHoverExplainabilityScore: Number(snapshotPanel?.getAttribute("data-debug-snapshot-hover-explainability-score") ?? 0),
        worldVisibleObjectCount: handle.visibleObjectCount ?? 0,
        worldHiddenObjectCount: handle.hiddenObjectCount ?? 0,
        worldVisibleGaussianCount: handle.visibleGaussianCount ?? 0,
        worldHiddenGaussianCount: handle.hiddenGaussianCount ?? 0,
        worldHiddenObjectIds: handle.hiddenObjectIds ?? [],
        objectVisibilitySampleCount: (handle.objectVisibilitySamples ?? []).length,
        shellVisibilityContract: shell?.getAttribute("data-object-visibility-contract") ?? null,
        shellHiddenObjects: Number(shell?.getAttribute("data-hidden-objects") ?? 0),
        shellVisibleObjects: Number(shell?.getAttribute("data-visible-objects") ?? 0),
        shellVisibleGaussians: Number(shell?.getAttribute("data-visible-gaussians") ?? 0),
        shellHiddenGaussians: Number(shell?.getAttribute("data-hidden-gaussians") ?? 0),
        snapshotHiddenObjects: snapshot?.visibility?.hiddenObjectCount ?? null,
        snapshotHiddenGaussians: snapshot?.visibility?.hiddenGaussianCount ?? null,
        panelVisibilityContract: debugPanel?.getAttribute("data-object-visibility-contract") ?? null,
        panelHiddenObjects: Number(debugPanel?.getAttribute("data-hidden-objects") ?? 0),
        panelHiddenGaussians: Number(debugPanel?.getAttribute("data-hidden-gaussians") ?? 0),
        dashboardStatus: stability?.getAttribute("data-stability-status") ?? null,
        trainableArtifactLoadRoute: shell?.getAttribute("data-trainable-artifact-load-route") ?? null,
        trainableArtifactPath: shell?.getAttribute("data-trainable-artifact-path") ?? null,
        trainableFrameIndex: Number(shell?.getAttribute("data-trainable-artifact-frame-index") ?? -1),
        trainableFrameCount: Number(shell?.getAttribute("data-trainable-artifact-frame-count") ?? -1),
        trainableTrainingStatus: shell?.getAttribute("data-trainable-training-status") ?? null,
        trainableTrainingIterations: Number(shell?.getAttribute("data-trainable-training-iterations") ?? 0),
        trainableTrainingFinalLoss: Number(shell?.getAttribute("data-trainable-training-final-total-loss") ?? 0),
        trainableTrainingLossDelta: Number(shell?.getAttribute("data-trainable-training-loss-delta") ?? 0),
        trainableTrainingFinalImageLoss: Number(shell?.getAttribute("data-trainable-training-final-image-loss") ?? 0),
        trainableTrainingImageLossDelta: Number(shell?.getAttribute("data-trainable-training-image-loss-delta") ?? 0),
        trainableTrainingImageLossDecreased: shell?.getAttribute("data-trainable-training-image-loss-decreased") ?? null,
        panelTrainingStatus: training?.getAttribute("data-training-status") ?? null,
        panelTrainingRenderer: training?.getAttribute("data-training-renderer") ?? null,
        panelTrainingFinalLoss: Number(training?.getAttribute("data-training-final-total-loss") ?? 0),
        panelTrainingLossDelta: Number(training?.getAttribute("data-training-loss-delta") ?? 0),
        solverTrainingStatus: shell?.getAttribute("data-solver-training-status") ?? null,
        solverTrainingSchema: shell?.getAttribute("data-solver-training-schema") ?? null,
        solverTrainingFinalLoss: Number(shell?.getAttribute("data-solver-training-final-total-loss") ?? 0),
        solverTrainingLossDelta: Number(shell?.getAttribute("data-solver-training-loss-delta") ?? 0),
        solverTrainingFinalAssignmentLoss: Number(shell?.getAttribute("data-solver-training-final-assignment-loss") ?? 0),
        solverTrainingAssignmentLossDelta: Number(shell?.getAttribute("data-solver-training-assignment-loss-delta") ?? 0),
        solverTrainingGpuUsed: shell?.getAttribute("data-solver-training-gpu-used") ?? null,
        solverTrainingVramReserveGb: Number(shell?.getAttribute("data-solver-training-vram-reserve-gb") ?? -1),
        panelSolverTrainingStatus: solver?.getAttribute("data-solver-training-status") ?? null,
        panelSolverTrainingFinalLoss: Number(solver?.getAttribute("data-solver-training-final-total-loss") ?? 0),
        panelSolverTrainingLossDelta: Number(solver?.getAttribute("data-solver-training-loss-delta") ?? 0),
        panelSolverTrainingGpuUsed: solver?.getAttribute("data-solver-training-gpu-used") ?? null,
        panelSolverTrainingVramReserveGb: Number(solver?.getAttribute("data-solver-training-vram-reserve-gb") ?? -1),
        snapshotSolverTrainingStatus: snapshot?.solverTraining?.status ?? null,
        snapshotSolverTrainingGpuUsed: snapshot?.solverTraining?.gpuUsed ?? null,
        snapshotSolverTrainingVramReserveGb: snapshot?.solverTraining?.vramReserveGb ?? null,
        worldSolverTrainingStatus: handle.solverTrainingStatus ?? null,
        worldSolverTrainingLossDelta: handle.solverTrainingLossDelta ?? null,
        worldSolverTrainingGpuUsed: handle.solverTrainingGpuUsed ?? null,
        worldSolverTrainingVramReserveGb: handle.solverTrainingVramReserveGb ?? null,
        entropyLensSamples: (handle.lensOpacitySamples ?? []).filter(
          (sample) => sample.modelId === "trainable-mvp-debug" && sample.activeLens === "entropy",
        ),
        worldTrainableFrameIndex: handle.selectedTrainableFrameIndex ?? null,
        worldTrainableFrameCount: handle.selectedTrainableFrameCount ?? null,
        trainableArtifactLoadedCount: handle.trainableArtifactLoadedCount,
        ogcLoadedCount: Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0),
      };
    });
    if (world.stabilityStatus !== world.dashboardStatus) {
      throw new Error(`stability dashboard mismatch: ${JSON.stringify(world)}`);
    }
    if (
      world.objectOverlayMode !== "full" ||
      world.objectOverlayBboxVisible !== true ||
      world.objectOverlayCentroidVisible !== true ||
      world.debugSnapshotOverlayMode !== "full"
    ) {
      throw new Error(`expected full ObjectState overlay context: ${JSON.stringify(world)}`);
    }
    if (!(Number(world.slotUtilization) > 0)) {
      throw new Error(`expected positive slot utilization, got ${world.slotUtilization}`);
    }
    if (
      world.trainableArtifactLoadRoute !== "fetch-json" ||
      world.trainableArtifactPath !== "/models/trainable-mvp-debug/model-artifact.json"
    ) {
      throw new Error(`expected fetched trainable artifact route: ${JSON.stringify(world)}`);
    }
    if (!(
      world.trainableFrameIndex === 1 &&
      world.trainableFrameCount === 2 &&
      world.worldTrainableFrameIndex === 1 &&
      world.worldTrainableFrameCount === 2
    )) {
      throw new Error(`expected trainable artifact frame 1 telemetry: ${JSON.stringify(world)}`);
    }
    if (!(
      world.trainableTrainingStatus === "loss_down" &&
      world.panelTrainingStatus === "loss_down" &&
      world.panelTrainingRenderer === "cpu-image-point-splat-differentiable-v1" &&
      world.trainableTrainingIterations === 18 &&
      world.trainableTrainingFinalLoss > 0 &&
      world.trainableTrainingLossDelta > 0 &&
      world.trainableTrainingFinalImageLoss > 0 &&
      world.trainableTrainingImageLossDelta > 0 &&
      world.trainableTrainingImageLossDecreased === "true" &&
      world.panelTrainingFinalLoss === world.trainableTrainingFinalLoss &&
      world.panelTrainingLossDelta === world.trainableTrainingLossDelta
    )) {
      throw new Error(`expected trainable loss evidence telemetry: ${JSON.stringify(world)}`);
    }
    if (!(
      world.solverTrainingStatus === "loss_down" &&
      world.solverTrainingSchema === "objgauss-object-emergence-solver-training-v1" &&
      world.solverTrainingFinalLoss > 0 &&
      world.solverTrainingLossDelta > 0 &&
      world.solverTrainingFinalAssignmentLoss > 0 &&
      world.solverTrainingAssignmentLossDelta > 0 &&
      world.solverTrainingGpuUsed === "false" &&
      world.solverTrainingVramReserveGb === 1 &&
      world.panelSolverTrainingStatus === "loss_down" &&
      world.panelSolverTrainingFinalLoss === world.solverTrainingFinalLoss &&
      world.panelSolverTrainingLossDelta === world.solverTrainingLossDelta &&
      world.panelSolverTrainingGpuUsed === "false" &&
      world.panelSolverTrainingVramReserveGb === 1 &&
      world.snapshotSolverTrainingStatus === "loss_down" &&
      world.snapshotSolverTrainingGpuUsed === false &&
      world.snapshotSolverTrainingVramReserveGb === 1 &&
      world.worldSolverTrainingStatus === "loss_down" &&
      world.worldSolverTrainingLossDelta > 0 &&
      world.worldSolverTrainingGpuUsed === false &&
      world.worldSolverTrainingVramReserveGb === 1
    )) {
      throw new Error(`expected Object Emergence Solver loss evidence telemetry: ${JSON.stringify(world)}`);
    }
    if (!(
      world.debugLens === "entropy" &&
      world.entropyLensSamples.length >= 2 &&
      world.entropyLensSamples.every((sample) => sample.opacityLens === "entropy") &&
      world.entropyLensSamples.some((sample) => sample.opacity > 0.32 && sample.opacity <= 1)
    )) {
      throw new Error(`expected entropy debug lens telemetry: ${JSON.stringify(world)}`);
    }
    if (!(
      world.debugSnapshotSchema === "objgauss-object-state-debug-snapshot-v1" &&
      world.debugSnapshotProtocol === "object-state-debug-os-v1" &&
      world.debugSnapshotModel === "trainable-mvp-debug" &&
      world.debugSnapshotLens === "entropy" &&
      world.debugSnapshotAssignmentSource === "trainable_kernel_model_artifact" &&
      world.debugSnapshotAssignmentSlots === 2 &&
      world.debugSnapshotAssignmentProbeStatus === "confident" &&
      world.debugSnapshotAssignmentProbeMargin > 0.55 &&
      world.objectContinuityStatus === "continuous" &&
      world.objectContinuityBboxDiagonal > 0 &&
      world.objectContinuityCentroidContained === true &&
      world.shellObjectContinuityStatus === world.objectContinuityStatus &&
      world.shellObjectContinuityBboxDiagonal > 0 &&
      world.shellObjectContinuityCentroidContained === "true" &&
      world.panelObjectContinuityStatus === world.objectContinuityStatus &&
      world.panelObjectContinuityBboxDiagonal > 0 &&
      world.panelObjectContinuityCentroidContained === "true" &&
      world.fragmentPanelStatus === world.objectContinuityStatus &&
      world.fragmentPanelObject === "0" &&
      world.fragmentPanelGaussians > 0 &&
      world.fragmentPanelCompactness > 0 &&
      world.fragmentPanelBboxDiagonal > 0 &&
      world.fragmentPanelDensity > 0 &&
      world.fragmentPanelCentroidContained === "true" &&
      world.fragmentPanelBboxValid === "true" &&
      world.fragmentPanelFragmented === "false" &&
      world.fragmentPanelCentroidOutside === "false" &&
      world.fragmentPanelDegenerate === "false" &&
      world.fragmentPanelFlagRows === 3 &&
      world.debugSnapshotContinuityStatus === world.objectContinuityStatus &&
      world.debugSnapshotContinuityBboxDiagonal > 0 &&
      world.debugSnapshotContinuityCentroidContained === true &&
      world.panelDebugSnapshotContinuityStatus === world.objectContinuityStatus &&
      world.panelDebugSnapshotContinuityBboxDiagonal > 0 &&
      world.panelDebugSnapshotContinuityCentroidContained === "true" &&
      world.objectTemporalStatus === "stable" &&
      world.objectTemporalDrift > 0 &&
      world.objectTemporalDrift < 0.08 &&
      world.objectAssignmentJitter > 0 &&
      world.objectAssignmentJitter < 0.08 &&
      world.objectBboxStability > 0.5 &&
      world.objectTemporalStable === true &&
      world.shellObjectTemporalStatus === world.objectTemporalStatus &&
      world.shellObjectTemporalDrift > 0 &&
      world.shellObjectAssignmentJitter > 0 &&
      world.shellObjectBboxStability > 0.5 &&
      world.shellObjectTemporalStable === "true" &&
      world.panelObjectTemporalStatus === world.objectTemporalStatus &&
      world.panelObjectTemporalDrift > 0 &&
      world.panelObjectAssignmentJitter > 0 &&
      world.panelObjectBboxStability > 0.5 &&
      world.panelObjectTemporalStable === "true" &&
      world.debugSnapshotTemporalStatus === world.objectTemporalStatus &&
      world.debugSnapshotTemporalDrift > 0 &&
      world.debugSnapshotAssignmentJitter > 0 &&
      world.debugSnapshotBboxStability > 0.5 &&
      world.debugSnapshotTemporalStable === true &&
      world.panelDebugSnapshotTemporalStatus === world.objectTemporalStatus &&
      world.panelDebugSnapshotTemporalDrift > 0 &&
      world.panelDebugSnapshotAssignmentJitter > 0 &&
      world.panelDebugSnapshotBboxStability > 0.5 &&
      world.panelDebugSnapshotTemporalStable === "true" &&
      world.objectExplainabilityStatus === "explainable" &&
      world.objectExplainable === true &&
      world.objectExplainabilityScore > 0.6 &&
      world.objectExplainabilityReasons === "" &&
      world.shellObjectExplainabilityStatus === world.objectExplainabilityStatus &&
      world.shellObjectExplainable === "true" &&
      world.shellObjectExplainabilityScore > 0.6 &&
      world.shellObjectExplainabilityReasons === "" &&
      world.panelObjectExplainabilityStatus === world.objectExplainabilityStatus &&
      world.panelObjectExplainable === "true" &&
      world.panelObjectExplainabilityScore > 0.6 &&
      world.panelObjectExplainabilityReasons === "" &&
      world.debugSnapshotExplainabilityStatus === world.objectExplainabilityStatus &&
      world.debugSnapshotExplainable === true &&
      world.debugSnapshotExplainabilityScore > 0.6 &&
      world.debugSnapshotExplainabilityReasons === "" &&
      world.panelDebugSnapshotExplainabilityStatus === world.objectExplainabilityStatus &&
      world.panelDebugSnapshotExplainable === "true" &&
      world.panelDebugSnapshotExplainabilityScore > 0.6 &&
      world.panelDebugSnapshotExplainabilityReasons === "" &&
      world.objectVerdictStatus === world.objectExplainabilityStatus &&
      world.objectVerdictExplainable === "true" &&
      world.objectVerdictScore > 0.6 &&
      world.objectVerdictReasonCount === 0 &&
      world.objectVerdictReasons === "" &&
      world.objectVerdictClear === "true" &&
      world.objectVerdictContinuityStatus === "continuous" &&
      world.objectVerdictTemporalStatus === "stable" &&
      world.objectVerdictReasonName === "clear" &&
      world.objectVerdictReasonStatus === "pass" &&
      world.debugSnapshotTrainingStatus === "loss_down" &&
      world.shellDebugSnapshotSchema === world.debugSnapshotSchema &&
      world.shellDebugSnapshotModel === world.debugSnapshotModel &&
      world.shellDebugSnapshotSlots === world.debugSnapshotAssignmentSlots &&
      world.shellDebugSnapshotAssignmentProbeStatus === world.debugSnapshotAssignmentProbeStatus &&
      world.shellDebugSnapshotStability === world.stabilityStatus &&
      world.panelDebugSnapshotSchema === world.debugSnapshotSchema &&
      world.panelDebugSnapshotModel === world.debugSnapshotModel &&
      world.panelDebugSnapshotLens === world.debugSnapshotLens &&
      world.panelDebugSnapshotSlots === world.debugSnapshotAssignmentSlots &&
      world.panelDebugSnapshotAssignmentProbeStatus === world.debugSnapshotAssignmentProbeStatus &&
      world.debugSnapshotAssignmentTimelineStatus === "stable" &&
      world.debugSnapshotAssignmentTimelineFrameCount === 2 &&
      world.debugSnapshotAssignmentTimelineJitter > 0 &&
      world.debugSnapshotAssignmentTimelineJitter < 0.05 &&
      world.panelDebugSnapshotAssignmentTimelineStatus === world.debugSnapshotAssignmentTimelineStatus &&
      world.panelDebugSnapshotAssignmentTimelineJitter > 0 &&
      world.panelDebugSnapshotAssignmentTimelineJitter < 0.05 &&
      world.assignmentProbeStatus === "confident" &&
      world.assignmentProbeMargin > 0.55 &&
      world.shellAssignmentProbeStatus === "confident" &&
      world.shellAssignmentProbeMargin > 0.55 &&
      world.shellAssignmentTimelineStatus === "stable" &&
      world.shellAssignmentTimelineFrameCount === 2 &&
      world.shellAssignmentTimelineJitter > 0 &&
      world.shellAssignmentTimelineJitter < 0.05 &&
      world.panelAssignmentTimelineStatus === world.shellAssignmentTimelineStatus &&
      world.panelAssignmentTimelineFrameCount === 2 &&
      world.timelinePanelStatus === "stable" &&
      world.timelinePanelFrameCount === 2 &&
      world.timelinePanelCurrentFrame === "1" &&
      world.timelinePanelGaussianIndex === "0" &&
      world.timelinePanelJitter > 0 &&
      world.timelinePanelJitter < 0.05 &&
      world.timelinePanelMaxDelta > 0 &&
      world.timelinePanelMaxDelta < 0.05 &&
      world.timelinePanelRows === 2 &&
      world.probePanelStatus === "confident" &&
      world.probePanelSource === "trainable_kernel_model_artifact" &&
      world.probePanelIndex === "0" &&
      world.probePanelMargin > 0.55 &&
      world.probePanelConfidence > 0.7 &&
      world.probePanelEntropy > 0 &&
      world.probePanelAmbiguous === "false" &&
      world.probePanelCollapseRisk === "false" &&
      world.probePanelOpacity > 0
    )) {
      throw new Error(`expected stable ObjectState debug snapshot protocol: ${JSON.stringify(world)}`);
    }
    if (!(
      world.debugEventCount > 0 &&
      world.debugEventCount === world.shellDebugEventCount &&
      world.debugEventCount === world.panelDebugEventCount &&
      world.debugEventCount === world.debugSnapshotEventCount &&
      world.debugEventLast === world.shellDebugEventLast &&
      world.debugEventLast === world.panelDebugEventLast &&
      world.panelDebugEventSchema === "objgauss-debug-event-v1" &&
      world.debugEventTypes.includes("gaussian-probe") &&
      world.debugEventTypes.includes("debug-lens") &&
      world.debugEventTypes.includes("object-overlay") &&
      world.debugEventTypes.includes("frame-select") &&
      world.debugEventTypes.includes("hover-object") &&
      world.debugEventTypes.includes("toggle-visibility") &&
      world.debugSnapshotEventTypes.includes("toggle-visibility")
    )) {
      throw new Error(`expected ObjectState debug event trace protocol: ${JSON.stringify(world)}`);
    }
    if (!(Number(world.meanPurity) > 0 && Number(world.dashboardMeanPurity) > 0 && Number(world.shellMeanPurity) > 0)) {
      throw new Error(`expected object purity metric to be available: ${JSON.stringify(world)}`);
    }
    if (!(
      Number(world.meanTemporalDrift) > 0 &&
      Number(world.dashboardMeanTemporalDrift) > 0 &&
      Number(world.shellMeanTemporalDrift) > 0
    )) {
      throw new Error(`expected temporal drift metric to be available: ${JSON.stringify(world)}`);
    }
    if (!(
      Number(world.meanSpatialCompactness) > 0 &&
      Number(world.dashboardMeanSpatialCompactness) > 0 &&
      Number(world.shellMeanSpatialCompactness) > 0
    )) {
      throw new Error(`expected spatial compactness metric to be available: ${JSON.stringify(world)}`);
    }
    if (!(
      Number(world.meanAssignmentJitter) >= 0 &&
      Number(world.dashboardMeanAssignmentJitter) >= 0 &&
      Number(world.shellMeanAssignmentJitter) >= 0
    )) {
      throw new Error(`expected assignment jitter metric to be available: ${JSON.stringify(world)}`);
    }
    if (!(
      Number(world.meanBboxStability) > 0 &&
      Number(world.dashboardMeanBboxStability) > 0 &&
      Number(world.shellMeanBboxStability) > 0
    )) {
      throw new Error(`expected bbox stability metric to be available: ${JSON.stringify(world)}`);
    }
    if (world.hoveredId !== world.shellHoveredTarget) {
      throw new Error(`hover audit mismatch: ${JSON.stringify(world)}`);
    }
    if (!(Number(world.hoveredGaussianCount) > 0 && Number(world.shellHoveredGaussians) > 0)) {
      throw new Error(`expected hovered ObjectState target to expose assigned Gaussians: ${JSON.stringify(world)}`);
    }
    if (!(
      world.hoveredAssignmentSource === "trainable_kernel_model_artifact" &&
      world.hoveredAssignmentSlotCount === 2 &&
      world.hoveredAssignmentProbeStatus === "confident" &&
      world.hoveredAssignmentProbeMargin > 0.45 &&
      world.hoveredAssignmentConfidence > 0.7 &&
      world.hoveredAssignmentEntropy > 0 &&
      world.hoveredAssignmentTopSlot === 0 &&
      world.shellHoverAssignmentSource === world.hoveredAssignmentSource &&
      world.shellHoverAssignmentSlots === world.hoveredAssignmentSlotCount &&
      world.shellHoverAssignmentStatus === world.hoveredAssignmentProbeStatus &&
      world.shellHoverAssignmentMargin > 0.45 &&
      world.panelHoverAssignmentSource === world.hoveredAssignmentSource &&
      world.panelHoverAssignmentStatus === world.hoveredAssignmentProbeStatus &&
      world.snapshotHoverObject === world.hoveredId &&
      world.snapshotHoverAssignmentStatus === world.hoveredAssignmentProbeStatus &&
      world.panelSnapshotHoverObject === world.hoveredId &&
      world.panelSnapshotHoverAssignmentStatus === world.hoveredAssignmentProbeStatus &&
      world.hoveredContinuityStatus === "continuous" &&
      world.hoveredContinuityBboxDiagonal > 0 &&
      world.hoveredContinuityCentroidContained === true &&
      world.shellHoverContinuityStatus === world.hoveredContinuityStatus &&
      world.shellHoverContinuityBboxDiagonal > 0 &&
      world.shellHoverContinuityCentroidContained === "true" &&
      world.panelHoverContinuityStatus === world.hoveredContinuityStatus &&
      world.panelHoverContinuityBboxDiagonal > 0 &&
      world.panelHoverContinuityCentroidContained === "true" &&
      world.snapshotHoverContinuityStatus === world.hoveredContinuityStatus &&
      world.snapshotHoverContinuityBboxDiagonal > 0 &&
      world.snapshotHoverContinuityCentroidContained === true &&
      world.panelSnapshotHoverContinuityStatus === world.hoveredContinuityStatus &&
      world.panelSnapshotHoverContinuityBboxDiagonal > 0 &&
      world.panelSnapshotHoverContinuityCentroidContained === "true" &&
      world.hoveredTemporalStatus === "stable" &&
      world.hoveredTemporalDrift > 0 &&
      world.hoveredTemporalDrift < 0.08 &&
      world.hoveredAssignmentJitter > 0 &&
      world.hoveredAssignmentJitter < 0.08 &&
      world.hoveredBboxStability > 0.5 &&
      world.hoveredTemporalStable === true &&
      world.shellHoverTemporalStatus === world.hoveredTemporalStatus &&
      world.shellHoverTemporalDrift > 0 &&
      world.shellHoverAssignmentJitter > 0 &&
      world.shellHoverBboxStability > 0.5 &&
      world.shellHoverTemporalStable === "true" &&
      world.panelHoverTemporalStatus === world.hoveredTemporalStatus &&
      world.panelHoverTemporalDrift > 0 &&
      world.panelHoverAssignmentJitter > 0 &&
      world.panelHoverBboxStability > 0.5 &&
      world.panelHoverTemporalStable === "true" &&
      world.snapshotHoverTemporalStatus === world.hoveredTemporalStatus &&
      world.snapshotHoverTemporalDrift > 0 &&
      world.snapshotHoverAssignmentJitter > 0 &&
      world.snapshotHoverBboxStability > 0.5 &&
      world.snapshotHoverTemporalStable === true &&
      world.panelSnapshotHoverTemporalStatus === world.hoveredTemporalStatus &&
      world.panelSnapshotHoverTemporalDrift > 0 &&
      world.panelSnapshotHoverAssignmentJitter > 0 &&
      world.panelSnapshotHoverTemporalStable === "true" &&
      world.hoveredExplainabilityStatus === "explainable" &&
      world.hoveredExplainable === true &&
      world.hoveredExplainabilityScore > 0.6 &&
      world.hoveredExplainabilityReasons === "" &&
      world.shellHoverExplainabilityStatus === world.hoveredExplainabilityStatus &&
      world.shellHoverExplainable === "true" &&
      world.shellHoverExplainabilityScore > 0.6 &&
      world.shellHoverExplainabilityReasons === "" &&
      world.panelHoverExplainabilityStatus === world.hoveredExplainabilityStatus &&
      world.panelHoverExplainable === "true" &&
      world.panelHoverExplainabilityScore > 0.6 &&
      world.panelHoverExplainabilityReasons === "" &&
      world.hoverHeatmapTarget === world.hoveredId &&
      world.hoverHeatmapModel === world.hoveredModelId &&
      world.hoverHeatmapSource === world.hoveredAssignmentSource &&
      world.hoverHeatmapSlots === world.hoveredAssignmentSlotCount &&
      world.hoverHeatmapStatus === world.hoveredAssignmentProbeStatus &&
      world.hoverHeatmapTopSlot === world.hoveredAssignmentTopSlot &&
      world.hoverHeatmapMargin > 0.45 &&
      world.hoverHeatmapRowCount === world.hoveredAssignmentSlotCount &&
      world.hoverHeatmapTopRowProbability > 0.7 &&
      world.snapshotHoverExplainabilityStatus === world.hoveredExplainabilityStatus &&
      world.snapshotHoverExplainable === true &&
      world.snapshotHoverExplainabilityScore > 0.6 &&
      world.snapshotHoverExplainabilityReasons === "" &&
      world.panelSnapshotHoverExplainabilityStatus === world.hoveredExplainabilityStatus &&
      world.panelSnapshotHoverExplainable === "true" &&
      world.panelSnapshotHoverExplainabilityScore > 0.6 &&
      world.hoverVerdictStatus === world.hoveredExplainabilityStatus &&
      world.hoverVerdictExplainable === "true" &&
      world.hoverVerdictScore > 0.6 &&
      world.hoverVerdictReasonCount === 0 &&
      world.hoverVerdictReasons === "" &&
      world.hoverVerdictClear === "true" &&
      world.hoverVerdictContinuityStatus === "continuous" &&
      world.hoverVerdictTemporalStatus === "stable" &&
      world.hoverVerdictReasonName === "clear" &&
      world.hoverVerdictReasonStatus === "pass"
    )) {
      throw new Error(`expected hover to expose ObjectState assignment preview: ${JSON.stringify(world)}`);
    }
    if (!(
      world.hoverHighlightActive === true &&
      world.shellHoverHighlight === "enabled" &&
      world.shellHoverHighlightObject === world.hoveredId &&
      world.hoverHighlightedObjectCount === 1 &&
      world.hoverHighlightedGaussianCount === world.hoveredGaussianCount &&
      world.shellHoverHighlightGaussians === world.hoveredGaussianCount &&
      world.hoverDimmedObjectCount > 0 &&
      world.hoverDimmedGaussianCount > 0 &&
      world.hoverHighlightSampleCount >= world.objectCount &&
      world.hoverHighlightedSamples.every((sample) => sample.hoverMode === "highlighted" && sample.opacity >= 0.86) &&
      world.hoverDimmedSamples.some((sample) => sample.hoverMode === "dimmed" && sample.opacity <= 0.18)
    )) {
      throw new Error(`expected hover highlight to isolate assigned Gaussian cluster: ${JSON.stringify(world)}`);
    }
    if (!(
      world.shellVisibilityContract === "enabled" &&
      world.panelVisibilityContract === "enabled" &&
      world.worldHiddenObjectCount === world.shellHiddenObjects &&
      world.worldHiddenObjectCount === world.panelHiddenObjects &&
      world.worldHiddenObjectCount === world.snapshotHiddenObjects &&
      world.worldHiddenGaussianCount === world.shellHiddenGaussians &&
      world.worldHiddenGaussianCount === world.panelHiddenGaussians &&
      world.worldHiddenGaussianCount === world.snapshotHiddenGaussians &&
      world.worldVisibleObjectCount === world.shellVisibleObjects &&
      world.worldVisibleGaussianCount === world.shellVisibleGaussians &&
      world.worldHiddenObjectCount > 0 &&
      world.worldHiddenGaussianCount > 0 &&
      world.worldHiddenObjectIds.length === world.worldHiddenObjectCount &&
      world.objectVisibilitySampleCount >= world.objectCount
    )) {
      throw new Error(`expected object toggle to expose Gaussian visibility contract: ${JSON.stringify(world)}`);
    }
    const assignmentSlots = await page
      .locator("[data-assignment-heatmap='true']")
      .evaluate((node) => Number(node.getAttribute("data-assignment-slots") ?? 0));
    const selectedGaussian = await page
      .locator(".worldShell")
      .evaluate((node) => node.getAttribute("data-selected-gaussian"));
    return {
      modelCount: world.modelCount,
      objectCount: world.objectCount,
      draggableObjectCount: world.draggableObjectCount,
      selectedModelId: world.selectedModelId,
      selectedObjectId: world.selectedObjectId,
      debugOs: world.debugProtocol,
      debugSnapshotSchema: world.debugSnapshotSchema,
      debugLens: world.debugLens,
      objectOverlayMode: world.objectOverlayMode,
      debugEventCount: world.debugEventCount,
      ogcLoadedCount: world.ogcLoadedCount,
      trainableArtifactLoadedCount: world.trainableArtifactLoadedCount,
      trainableArtifactLoadRoute: world.trainableArtifactLoadRoute,
      trainableArtifactPath: world.trainableArtifactPath,
      trainableFrameIndex: world.trainableFrameIndex,
      trainableFrameCount: world.trainableFrameCount,
      trainableTrainingFinalLoss: world.trainableTrainingFinalLoss,
      trainableTrainingLossDelta: world.trainableTrainingLossDelta,
      trainableTrainingFinalImageLoss: world.trainableTrainingFinalImageLoss,
      solverTrainingFinalLoss: world.solverTrainingFinalLoss,
      solverTrainingLossDelta: world.solverTrainingLossDelta,
      solverTrainingGpuUsed: world.solverTrainingGpuUsed,
      solverTrainingVramReserveGb: world.solverTrainingVramReserveGb,
      urlArtifactStatus: urlArtifact.status,
      urlOgcStatus: urlOgc.status,
      urlOgcManifestStatus: urlOgcManifest.status,
      algorithmManifestStatus: algorithmManifest.status,
      debugSnapshotExportStatus: algorithmManifest.snapshotExportStatus,
      debugSessionExportStatus: algorithmManifest.sessionExportStatus,
      debugSessionImportStatus: algorithmManifest.sessionImportStatus,
      debugSessionDiffStatus: algorithmManifest.sessionDiffStatus,
      debugSessionDriftStatus: algorithmManifest.sessionDriftStatus,
      localModelManifestStatus: localModelManifest.status,
      localTrainableManifestStatus: localTrainableManifest.status,
      qualityReportStatus: algorithmManifest.qualityReportStatus,
      objectStateBenchmarkStatus: algorithmManifest.objectStateBenchmarkStatus,
      localArtifactStatus: localArtifact.status,
      localOgcStatus: localOgc.status,
      localOgcManifestStatus: localOgcManifest.status,
      assignmentSource: world.assignmentSource,
      stabilityStatus: world.stabilityStatus,
      slotUtilization: world.slotUtilization,
      mixedSlots: world.mixedSlots,
      objectContinuityStatus: world.objectContinuityStatus,
      objectMoveStatus: objectMove.source === "debug-panel" ? "passed" : "failed",
      sourceSplatObjectMotionStatus:
        sourceSplatObjectMotion.status === "ready" &&
        sourceSplatObjectMotion.active === true &&
        sourceSplatObjectMotion.maxTranslate > 0 &&
        sourceSplatObjectProjection.ok === true
          ? "passed"
          : "failed",
      objectTransformStatus:
        objectTransform.contract === "three-transform-controls-v1" &&
        objectTransform.engine === "object-transform-state-v1" &&
        objectTransform.pickingContract === "projected-object-bbox-picker-v2" &&
        objectTransform.pickingDecoupledFromRenderer === true &&
        objectTransform.renderSurfaceContract === "three-world-point-preview-v1" &&
        (objectTransform.gaussianDisplayMode.endsWith("点预览") ||
          objectTransform.gaussianDisplayMode === "完整 splat") &&
        objectTransform.objectLayerStatus === "loaded" &&
        objectTransform.gizmoObject === objectSelection.selectionId &&
        objectTransform.interactionLayer === "three-transform-controls-v1" &&
        objectTransform.actionButtonCount === 3 &&
        objectTransform.modeButtonCount === 3
          ? "passed"
          : "failed",
      objectTransformStateStatus: objectTransformState.ok ? "passed" : "failed",
      objectPickStatus: objectPick.result?.ok === true ? "passed" : "failed",
      objectMoveBefore: objectMove.before,
      objectMoveAfter: objectMove.after,
      sourceSplatObjectMotion,
      sourceSplatObjectProjection,
      trainingStageStatus:
        world.shellTrainingStage === "model-version-processing-v1" &&
        world.panelTrainingStageSchema === "model-version-processing-v1" &&
        world.shellRenderSurfaceContract === "three-world-point-preview-v1" &&
        sourceSplatStage.contract === "spark-source-splat-stage-v1" &&
        sourceSplatStage.mode === "完整 splat" &&
        world.shellStageDisplayContract === "spark-source-splat-stage-v1" &&
        world.shellSourceSplatActive === "true" &&
        world.panelStageRenderSurfaceContract === "three-world-point-preview-v1" &&
        world.panelStageDisplayContract === "spark-source-splat-stage-v1" &&
        world.panelStageSourceSplatModels >= 1 &&
        world.shellFullGaussianSourceCount >= 1 &&
        world.panelStageFullGaussianSources >= 1 &&
        world.shellObjectLayerRegisteredCount >= 1 &&
        world.panelStageObjectLayerRegistered >= 1 &&
        world.shellObjectLayerLoadedCount >= 1 &&
        world.panelStageObjectLayerLoaded >= 1 &&
        world.panelLayerBadgeRows >= world.panelModelVersionRows &&
        world.panelModelVersionRows >= 7
          ? "passed"
          : "failed",
      gaussianDisplayMode: world.shellGaussianDisplayMode,
      sourceSplatStatus: sourceSplatStage.status,
      sourceSplatReadyCount: sourceSplatStage.readyCount,
      fullGaussianSourceCount: world.shellFullGaussianSourceCount,
      objectLayerRegisteredCount: world.shellObjectLayerRegisteredCount,
      objectLayerLoadedCount: world.shellObjectLayerLoadedCount,
      stageVisibleModelCount: world.stageVisibleModelCount,
      stageProcessedModelCount: world.panelStageProcessedModelCount,
      objectFragmentationStatus: world.fragmentPanelStatus,
      objectTemporalStatus: world.objectTemporalStatus,
      objectExplainabilityStatus: world.objectExplainabilityStatus,
      objectVerdictStatus: world.objectVerdictStatus,
      meanPurity: world.meanPurity,
      meanTemporalDrift: world.meanTemporalDrift,
      meanSpatialCompactness: world.meanSpatialCompactness,
      meanAssignmentJitter: world.meanAssignmentJitter,
      meanBboxStability: world.meanBboxStability,
      hoveredContinuityStatus: world.hoveredContinuityStatus,
      hoveredTemporalStatus: world.hoveredTemporalStatus,
      hoveredExplainabilityStatus: world.hoveredExplainabilityStatus,
      hoverVerdictStatus: world.hoverVerdictStatus,
      hoveredObjectId: world.hoveredObjectId,
      hoveredGaussianCount: world.hoveredGaussianCount,
      probePanelStatus: world.probePanelStatus,
      probePanelSource: world.probePanelSource,
      probePanelMargin: world.probePanelMargin,
      assignmentTimelineStatus: world.timelinePanelStatus,
      assignmentTimelineJitter: world.timelinePanelJitter,
      assignmentSlots,
      selectedGaussian,
      sidebars,
      screenshotPath,
      mobileScreenshotPath,
    };
  } finally {
    await closeBrowserWithTimeout(browser);
  }
}

async function auditMobileWorld(browser, url) {
  const page = await browser.newPage({
    viewport: { width: 390, height: 844 },
    isMobile: true,
  });
  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.locator("[data-object-debug-panel='true']").waitFor({ timeout: 15000 });
    await page.locator("[data-stability-dashboard='true']").waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const pill = document.querySelector(".modelPill[data-model-row-id='trainable-mvp-debug']");
      return pill?.getAttribute("data-model-load-state") === "loaded";
    }, undefined, { timeout: 15000 });
    await clickModelPill(page, "trainable-mvp-debug");
    await clickSelector(page, "[data-trainable-frame-button='1']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const dashboard = document.querySelector("[data-stability-dashboard='true']");
      const frameSelector = document.querySelector("[data-trainable-frame-selector='true']");
      return shell?.getAttribute("data-stability-dashboard") === "enabled" &&
        dashboard?.getAttribute("data-stability-status") !== "" &&
        shell?.getAttribute("data-selected-model") === "trainable-mvp-debug" &&
        shell?.getAttribute("data-trainable-artifact-frame-index") === "1" &&
        frameSelector?.getAttribute("data-selected-frame") === "1";
    }, undefined, { timeout: 15000 });
    const screenshotPath = "/tmp/objgauss-world-viewer-mobile.png";
    await page.screenshot({ path: screenshotPath, fullPage: false });
    return screenshotPath;
  } finally {
    await page.close();
  }
}

async function auditUrlTrainableArtifact(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  try {
    const artifactUrl = new URL(url);
    artifactUrl.searchParams.set("trainableArtifact", "/models/trainable-mvp-debug/model-artifact.json");
    await page.goto(String(artifactUrl), { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const pill = document.querySelector(".modelPill[data-model-row-id='trainable-url-artifact']");
      return (
        pill?.getAttribute("data-model-load-state") === "loaded" &&
        shell?.getAttribute("data-model-count") === "8" &&
        shell?.getAttribute("data-selected-model") === "trainable-url-artifact" &&
        shell?.getAttribute("data-trainable-artifact-load-route") === "fetch-json" &&
        shell?.getAttribute("data-trainable-artifact-path") === "/models/trainable-mvp-debug/model-artifact.json" &&
        shell?.getAttribute("data-trainable-training-status") === "loss_down" &&
        shell?.getAttribute("data-trainable-training-image-loss-decreased") === "true" &&
        Number(shell?.getAttribute("data-trainable-training-final-total-loss") ?? 0) > 0 &&
        Number(shell?.getAttribute("data-trainable-training-loss-delta") ?? 0) > 0 &&
        Number(shell?.getAttribute("data-trainable-artifact-loaded-count") ?? 0) >= 2
      );
    }, undefined, { timeout: 15000 });
    const selection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const target = world?.objectSelections?.find((entry) => entry.modelId === "trainable-url-artifact");
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
      };
    });
    if (!selection.ok || selection.modelId !== "trainable-url-artifact") {
      throw new Error(`expected URL artifact object selection: ${JSON.stringify(selection)}`);
    }
    await clickSelector(page, "[data-trainable-frame-button='1']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const frameSelector = document.querySelector("[data-trainable-frame-selector='true']");
      const training = document.querySelector("[data-training-evidence='true']");
      return (
        shell?.getAttribute("data-selected-model") === "trainable-url-artifact" &&
        shell?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        shell?.getAttribute("data-trainable-artifact-frame-index") === "1" &&
        frameSelector?.getAttribute("data-selected-frame") === "1" &&
        training?.getAttribute("data-training-status") === "loss_down" &&
        training?.getAttribute("data-training-renderer") === "cpu-image-point-splat-differentiable-v1" &&
        Number(training?.getAttribute("data-training-final-image-loss") ?? 0) > 0 &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2
      );
    }, undefined, { timeout: 15000 });
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-url-artifact.png", fullPage: false });
    return { status: "fetch-json" };
  } finally {
    await page.close();
  }
}

async function auditLocalTrainableArtifactImport(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.locator("[data-trainable-artifact-file-input='true']").setInputFiles(
      "public/models/trainable-mvp-debug/model-artifact.json",
    );
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const pill = document.querySelector(".modelPill[data-model-row-id='trainable-local-artifact']");
      const button = document.querySelector("[data-trainable-artifact-import-button='true']");
      const training = document.querySelector("[data-training-evidence='true']");
      return (
        pill?.getAttribute("data-model-load-state") === "loaded" &&
        button?.getAttribute("data-import-status") === "loaded" &&
        shell?.getAttribute("data-model-count") === "8" &&
        shell?.getAttribute("data-catalog-model-count") === "7" &&
        shell?.getAttribute("data-selected-model") === "trainable-local-artifact" &&
        shell?.getAttribute("data-trainable-import-status") === "loaded" &&
        shell?.getAttribute("data-trainable-import-model") === "trainable-local-artifact" &&
        shell?.getAttribute("data-trainable-import-file") === "model-artifact.json" &&
        shell?.getAttribute("data-trainable-artifact-load-route") === "local-file" &&
        shell?.getAttribute("data-trainable-artifact-path") === "local://model-artifact.json" &&
        shell?.getAttribute("data-trainable-training-status") === "loss_down" &&
        shell?.getAttribute("data-trainable-training-image-loss-decreased") === "true" &&
        Number(shell?.getAttribute("data-trainable-artifact-loaded-count") ?? 0) >= 2 &&
        training?.getAttribute("data-training-renderer") === "cpu-image-point-splat-differentiable-v1"
      );
    }, undefined, { timeout: 15000 });
    const selection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const target = world?.objectSelections?.find((entry) => entry.modelId === "trainable-local-artifact");
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
        modelCount: world?.modelCount ?? 0,
      };
    });
    if (!selection.ok || selection.modelId !== "trainable-local-artifact" || selection.modelCount !== 8) {
      throw new Error(`expected local artifact object selection: ${JSON.stringify(selection)}`);
    }
    const gaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: window.__OBJGAUSS_WORLD__?.assignmentSource ?? null,
        assignmentProbeStatus: window.__OBJGAUSS_WORLD__?.assignmentProbeStatus ?? null,
        assignmentProbeMargin: window.__OBJGAUSS_WORLD__?.assignmentProbeMargin ?? null,
      };
    }, selection.selectionId);
    if (
      !gaussian.ok ||
      gaussian.assignmentSource !== "trainable_kernel_model_artifact" ||
      gaussian.assignmentProbeStatus !== "confident" ||
      !(Number(gaussian.assignmentProbeMargin) > 0.55)
    ) {
      throw new Error(`expected local artifact Gaussian probe: ${JSON.stringify(gaussian)}`);
    }
    await clickSelector(page, "[data-trainable-frame-button='1']");
    await page.waitForFunction(() => {
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const types = new Set(events.map((event) => event.type));
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const frameSelector = document.querySelector("[data-trainable-frame-selector='true']");
      const snapshot = window.__OBJGAUSS_DEBUG_SNAPSHOT__;
      return (
        shell?.getAttribute("data-selected-model") === "trainable-local-artifact" &&
        shell?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        shell?.getAttribute("data-assignment-probe-status") === "confident" &&
        Number(shell?.getAttribute("data-assignment-probe-margin") ?? 0) > 0.45 &&
        shell?.getAttribute("data-trainable-artifact-frame-index") === "1" &&
        frameSelector?.getAttribute("data-selected-frame") === "1" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2 &&
        snapshot?.model?.id === "trainable-local-artifact" &&
        snapshot?.delivery?.loadRoute === "local-file" &&
        types.has("import-artifact") &&
        types.has("gaussian-probe") &&
        types.has("frame-select")
      );
    }, undefined, { timeout: 15000 });
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-local-artifact.png", fullPage: false });
    return { status: "local-file" };
  } finally {
    await page.close();
  }
}

async function auditLocalOgcArtifactImport(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.locator("[data-ogc-artifact-file-input='true']").setInputFiles([
      "public/models/ogc-url-fixture/scene.index.json",
      "public/models/ogc-url-fixture/scene.ogc",
    ]);
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const pill = document.querySelector(".modelPill[data-model-row-id='ogc-local-artifact']");
      const button = document.querySelector("[data-ogc-artifact-import-button='true']");
      const importedFile = shell?.getAttribute("data-ogc-import-file") ?? "";
      return (
        pill?.getAttribute("data-model-load-state") === "loaded" &&
        button?.getAttribute("data-import-status") === "loaded" &&
        shell?.getAttribute("data-model-count") === "8" &&
        shell?.getAttribute("data-catalog-model-count") === "7" &&
        shell?.getAttribute("data-selected-model") === "ogc-local-artifact" &&
        shell?.getAttribute("data-ogc-import-status") === "loaded" &&
        shell?.getAttribute("data-ogc-import-model") === "ogc-local-artifact" &&
        importedFile.includes("scene.index.json") &&
        importedFile.includes("scene.ogc") &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "local-file" &&
        shell?.getAttribute("data-ogc-artifact-index-path") === "local://scene.index.json" &&
        shell?.getAttribute("data-ogc-artifact-payload-path") === "local://scene.ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "0" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "41" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "40" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "2" &&
        Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0) >= 2
      );
    }, undefined, { timeout: 15000 });
    const selection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const targets = world?.objectSelections?.filter((entry) => entry.modelId === "ogc-local-artifact") ?? [];
      const target = targets[0];
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
        objectCount: targets.length,
      };
    });
    if (
      !selection.ok ||
      selection.modelId !== "ogc-local-artifact" ||
      selection.objectCount !== 2
    ) {
      throw new Error(`expected local OGC object selection: ${JSON.stringify(selection)}`);
    }
    await page.waitForFunction((selectionId) => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const world = window.__OBJGAUSS_WORLD__;
      return (
        world?.selectedId === selectionId &&
        shell?.getAttribute("data-selected-model") === "ogc-local-artifact" &&
        shell?.getAttribute("data-assignment-source") === "derived_from_object_id" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2
      );
    }, selection.selectionId, { timeout: 15000 });
    const gaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, selection.selectionId);
    if (!gaussian.ok || gaussian.assignmentSource !== "derived_from_object_id") {
      throw new Error(`expected local OGC Gaussian probe: ${JSON.stringify(gaussian)}`);
    }
    await clickSelector(page, "[data-ogc-lod-button='1']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const selector = document.querySelector("[data-ogc-lod-selector='true']");
      return (
        shell?.getAttribute("data-selected-model") === "ogc-local-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "local-file" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "1" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "41" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "2" &&
        selector?.getAttribute("data-selected-lod") === "1" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2
      );
    }, undefined, { timeout: 15000 });
    await clickSelector(page, "[data-ogc-chunk-button='0']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const chunkSelector = document.querySelector("[data-ogc-chunk-selector='true']");
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const types = new Set(events.map((event) => event.type));
      return (
        shell?.getAttribute("data-selected-model") === "ogc-local-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "local-file" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "1" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "41" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "10" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "1" &&
        shell?.getAttribute("data-ogc-artifact-chunk-scope") === "0" &&
        panel?.getAttribute("data-ogc-chunk-scope") === "0" &&
        chunkSelector?.getAttribute("data-selected-chunks") === "0" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 1 &&
        types.has("import-ogc") &&
        types.has("gaussian-probe") &&
        types.has("ogc-lod") &&
        types.has("ogc-chunks")
      );
    }, undefined, { timeout: 15000 });
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-local-ogc.png", fullPage: false });
    return { status: "local-file-lod-chunk-ui" };
  } finally {
    await page.close();
  }
}

async function auditLocalOgcManifestPackageImport(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  try {
    const manifestPath = writeLocalOgcManifestFixture();
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.locator("[data-ogc-artifact-file-input='true']").setInputFiles([
      manifestPath,
      "public/models/ogc-url-fixture/scene.index.json",
      "public/models/ogc-url-fixture/scene.ogc",
    ]);
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const pill = document.querySelector(".modelPill[data-model-row-id='ogc-local-artifact']");
      const button = document.querySelector("[data-ogc-artifact-import-button='true']");
      const importedFile = shell?.getAttribute("data-ogc-import-file") ?? "";
      return (
        pill?.getAttribute("data-model-load-state") === "loaded" &&
        button?.getAttribute("data-import-status") === "loaded" &&
        shell?.getAttribute("data-model-count") === "8" &&
        shell?.getAttribute("data-catalog-model-count") === "7" &&
        shell?.getAttribute("data-selected-model") === "ogc-local-artifact" &&
        shell?.getAttribute("data-ogc-import-status") === "loaded" &&
        shell?.getAttribute("data-ogc-import-model") === "ogc-local-artifact" &&
        importedFile.includes("objgauss-local-ogc-model-artifact.json") &&
        importedFile.includes("scene.index.json") &&
        importedFile.includes("scene.ogc") &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "local-file" &&
        shell?.getAttribute("data-ogc-artifact-index-path") === "local://scene.index.json" &&
        shell?.getAttribute("data-ogc-artifact-payload-path") === "local://scene.ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "0" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "41" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "40" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "2" &&
        Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0) >= 2
      );
    }, undefined, { timeout: 15000 });
    const selection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const targets = world?.objectSelections?.filter((entry) => entry.modelId === "ogc-local-artifact") ?? [];
      const target = targets[0];
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
        objectCount: targets.length,
      };
    });
    if (
      !selection.ok ||
      selection.modelId !== "ogc-local-artifact" ||
      selection.objectCount !== 2
    ) {
      throw new Error(`expected local OGC manifest package object selection: ${JSON.stringify(selection)}`);
    }
    const gaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, selection.selectionId);
    if (!gaussian.ok || gaussian.assignmentSource !== "derived_from_object_id") {
      throw new Error(`expected local OGC manifest package Gaussian probe: ${JSON.stringify(gaussian)}`);
    }
    await clickSelector(page, "[data-ogc-lod-button='1']");
    await clickSelector(page, "[data-ogc-chunk-button='0']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const types = new Set(events.map((event) => event.type));
      return (
        shell?.getAttribute("data-selected-model") === "ogc-local-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "local-file" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "1" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "41" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "10" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "1" &&
        shell?.getAttribute("data-ogc-artifact-chunk-scope") === "0" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 1 &&
        types.has("import-ogc") &&
        types.has("gaussian-probe") &&
        types.has("ogc-lod") &&
        types.has("ogc-chunks")
      );
    }, undefined, { timeout: 15000 });
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-local-ogc-manifest.png", fullPage: false });
    return { status: "local-manifest-file-lod-chunk-ui" };
  } finally {
    await page.close();
  }
}

async function auditUrlOgcArtifact(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  try {
    const artifactUrl = new URL(url);
    artifactUrl.searchParams.set("ogcIndex", "/models/ogc-url-fixture/scene.index.json");
    artifactUrl.searchParams.set("ogcPayload", "/models/ogc-url-fixture/scene.ogc");
    artifactUrl.searchParams.set("ogcLod", "1");
    await page.goto(String(artifactUrl), { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const pill = document.querySelector(".modelPill[data-model-row-id='ogc-url-artifact']");
      return (
        pill?.getAttribute("data-model-load-state") === "loaded" &&
        shell?.getAttribute("data-model-count") === "8" &&
        shell?.getAttribute("data-selected-model") === "ogc-url-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "range-ogc" &&
        shell?.getAttribute("data-ogc-artifact-index-path") === "/models/ogc-url-fixture/scene.index.json" &&
        shell?.getAttribute("data-ogc-artifact-payload-path") === "/models/ogc-url-fixture/scene.ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "1" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "2" &&
        document.querySelector("[data-ogc-lod-selector='true']")?.getAttribute("data-selected-lod") === "1" &&
        Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0) >= 2
      );
    }, undefined, { timeout: 15000 });
    const selection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const targets = world?.objectSelections?.filter((entry) => entry.modelId === "ogc-url-artifact") ?? [];
      const target = targets[0];
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
        objectCount: targets.length,
      };
    });
    if (!selection.ok || selection.modelId !== "ogc-url-artifact" || selection.objectCount !== 2) {
      throw new Error(`expected URL OGC object selection: ${JSON.stringify(selection)}`);
    }
    await page.waitForFunction((selectionId) => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const world = window.__OBJGAUSS_WORLD__;
      return (
        world?.selectedId === selectionId &&
        shell?.getAttribute("data-selected-model") === "ogc-url-artifact" &&
        shell?.getAttribute("data-assignment-source") === "derived_from_object_id" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2
      );
    }, selection.selectionId, { timeout: 15000 });
    const gaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, selection.selectionId);
    if (!gaussian.ok || gaussian.assignmentSource !== "derived_from_object_id") {
      throw new Error(`expected URL OGC Gaussian probe: ${JSON.stringify(gaussian)}`);
    }
    await clickSelector(page, "[data-ogc-lod-button='0']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const selector = document.querySelector("[data-ogc-lod-selector='true']");
      const chunkSelector = document.querySelector("[data-ogc-chunk-selector='true']");
      return (
        shell?.getAttribute("data-selected-model") === "ogc-url-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "range-ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "0" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "40" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "40" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "2" &&
        shell?.getAttribute("data-ogc-artifact-chunk-scope") === "all" &&
        selector?.getAttribute("data-selected-lod") === "0" &&
        chunkSelector?.getAttribute("data-selected-chunks") === "all" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2
      );
    }, undefined, { timeout: 15000 });
    await clickSelector(page, "[data-ogc-chunk-button='0']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-object-debug-panel='true']");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const chunkSelector = document.querySelector("[data-ogc-chunk-selector='true']");
      return (
        shell?.getAttribute("data-selected-model") === "ogc-url-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "range-ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "0" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "1" &&
        shell?.getAttribute("data-ogc-artifact-chunk-scope") === "0" &&
        panel?.getAttribute("data-ogc-chunk-scope") === "0" &&
        chunkSelector?.getAttribute("data-selected-chunks") === "0" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 1
      );
    }, undefined, { timeout: 15000 });
    const chunkSelection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const targets = world?.objectSelections?.filter((entry) => entry.modelId === "ogc-url-artifact") ?? [];
      const target = targets[0];
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        objectCount: targets.length,
      };
    });
    if (!chunkSelection.ok || chunkSelection.objectCount !== 1) {
      throw new Error(`expected single URL OGC chunk selection: ${JSON.stringify(chunkSelection)}`);
    }
    const chunkGaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, chunkSelection.selectionId);
    if (!chunkGaussian.ok || chunkGaussian.assignmentSource !== "derived_from_object_id") {
      throw new Error(`expected URL OGC chunk Gaussian probe: ${JSON.stringify(chunkGaussian)}`);
    }
    await clickSelector(page, "[data-ogc-chunk-button='all']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const chunkSelector = document.querySelector("[data-ogc-chunk-selector='true']");
      return (
        shell?.getAttribute("data-selected-model") === "ogc-url-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "range-ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "0" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "40" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "40" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "2" &&
        shell?.getAttribute("data-ogc-artifact-chunk-scope") === "all" &&
        chunkSelector?.getAttribute("data-selected-chunks") === "all" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2
      );
    }, undefined, { timeout: 15000 });
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-url-ogc.png", fullPage: false });
    return { status: "range-ogc-lod-chunk-ui" };
  } finally {
    await page.close();
  }
}

async function auditUrlOgcManifestArtifact(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  try {
    const artifactUrl = new URL(url);
    artifactUrl.searchParams.set("ogcManifest", "/models/ogc-url-fixture/model-artifact.json");
    artifactUrl.searchParams.set("ogcLod", "1");
    await page.goto(String(artifactUrl), { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const pill = document.querySelector(".modelPill[data-model-row-id='ogc-manifest-artifact']");
      return (
        pill?.getAttribute("data-model-load-state") === "loaded" &&
        shell?.getAttribute("data-model-count") === "8" &&
        shell?.getAttribute("data-selected-model") === "ogc-manifest-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "range-ogc" &&
        shell?.getAttribute("data-ogc-artifact-index-path") === "/models/ogc-url-fixture/scene.index.json" &&
        shell?.getAttribute("data-ogc-artifact-payload-path") === "/models/ogc-url-fixture/scene.ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "1" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "2" &&
        document.querySelector("[data-ogc-lod-selector='true']")?.getAttribute("data-selected-lod") === "1" &&
        Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0) >= 2
      );
    }, undefined, { timeout: 15000 });
    const selection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const targets = world?.objectSelections?.filter((entry) => entry.modelId === "ogc-manifest-artifact") ?? [];
      const target = targets[0];
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
        objectCount: targets.length,
      };
    });
    if (!selection.ok || selection.modelId !== "ogc-manifest-artifact" || selection.objectCount !== 2) {
      throw new Error(`expected URL OGC manifest object selection: ${JSON.stringify(selection)}`);
    }
    const gaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, selection.selectionId);
    if (!gaussian.ok || gaussian.assignmentSource !== "derived_from_object_id") {
      throw new Error(`expected URL OGC manifest Gaussian probe: ${JSON.stringify(gaussian)}`);
    }
    await clickSelector(page, "[data-ogc-lod-button='0']");
    await clickSelector(page, "[data-ogc-chunk-button='0']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const types = new Set(events.map((event) => event.type));
      return (
        shell?.getAttribute("data-selected-model") === "ogc-manifest-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "range-ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "0" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "1" &&
        shell?.getAttribute("data-ogc-artifact-chunk-scope") === "0" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 1 &&
        types.has("gaussian-probe") &&
        types.has("ogc-lod") &&
        types.has("ogc-chunks")
      );
    }, undefined, { timeout: 15000 });
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-url-ogc-manifest.png", fullPage: false });
    return { status: "url-manifest-range-lod-chunk-ui" };
  } finally {
    await page.close();
  }
}

async function auditAlgorithmManifestBundle(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  try {
    const artifactUrl = new URL(url);
    artifactUrl.searchParams.set("modelArtifactManifest", "/models/algorithm-bundle-fixture/model-artifact.json");
    artifactUrl.searchParams.set("ogcLod", "1");
    await page.goto(String(artifactUrl), { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const quality = document.querySelector("[data-quality-report='true']");
      const benchmark = document.querySelector("[data-object-state-benchmark='true']");
      const snapshot = document.querySelector("[data-debug-snapshot-panel='true']");
      const gateRows = document.querySelector("[data-quality-gates='true']");
      const entropyGate = document.querySelector("[data-quality-gate-name='assignment_entropy']");
      const slotGate = document.querySelector("[data-quality-gate-name='slot_utilization']");
      const manifestPill = document.querySelector(".modelPill[data-model-row-id='model-artifact-manifest']");
      const trainablePill = document.querySelector(".modelPill[data-model-row-id='model-manifest-trainable-artifact']");
      const ogcPill = document.querySelector(".modelPill[data-model-row-id='model-manifest-ogc-artifact']");
      return (
        manifestPill?.getAttribute("data-model-load-state") === "loaded" &&
        trainablePill?.getAttribute("data-model-load-state") === "loaded" &&
        ogcPill?.getAttribute("data-model-load-state") === "loaded" &&
        shell?.getAttribute("data-model-count") === "10" &&
        shell?.getAttribute("data-selected-model") === "model-manifest-trainable-artifact" &&
        shell?.getAttribute("data-trainable-artifact-load-route") === "model-manifest-json" &&
        shell?.getAttribute("data-trainable-artifact-path") === "/models/trainable-mvp-debug/model-artifact.json" &&
        shell?.getAttribute("data-trainable-training-status") === "loss_down" &&
        shell?.getAttribute("data-trainable-training-image-loss-decreased") === "true" &&
        shell?.getAttribute("data-quality-report-status") === "warn" &&
        shell?.getAttribute("data-quality-report-schema") === "objgauss-object-state-quality-report-v1" &&
        shell?.getAttribute("data-quality-report-object-purity") === "1" &&
        shell?.getAttribute("data-object-state-benchmark-status") === "pass" &&
        shell?.getAttribute("data-object-state-benchmark-schema") === "objgauss-object-state-stability-benchmark-v1" &&
        shell?.getAttribute("data-object-state-benchmark-case-count") === "8" &&
        shell?.getAttribute("data-object-state-benchmark-warn-count") === "0" &&
        shell?.getAttribute("data-object-state-benchmark-observed-warn-count") === "6" &&
        shell?.getAttribute("data-object-state-benchmark-failure-mode-count") === "12" &&
        shell?.getAttribute("data-object-state-benchmark-active-case") === "uniform_mixed" &&
        shell?.getAttribute("data-object-state-benchmark-active-failure-modes") === "uniform_assignment,mixed_slots,low_object_purity" &&
        quality?.getAttribute("data-quality-report-status") === "warn" &&
        quality?.getAttribute("data-quality-report-gate-count") === "3" &&
        quality?.getAttribute("data-quality-report-failing-gate-names") === "assignment_entropy" &&
        benchmark?.getAttribute("data-object-state-benchmark-status") === "pass" &&
        benchmark?.getAttribute("data-object-state-benchmark-case-count") === "8" &&
        benchmark?.getAttribute("data-object-state-benchmark-first-case") === "clean_sparse" &&
        benchmark?.getAttribute("data-object-state-benchmark-active-case") === "uniform_mixed" &&
        gateRows?.getAttribute("data-quality-gate-count") === "3" &&
        entropyGate?.getAttribute("data-quality-gate-status") === "warn" &&
        entropyGate?.getAttribute("data-quality-gate-threshold") === "0.5" &&
        slotGate?.getAttribute("data-quality-gate-status") === "pass" &&
        snapshot?.getAttribute("data-debug-snapshot-quality-status") === "warn" &&
        window.__OBJGAUSS_DEBUG_SNAPSHOT__?.benchmark?.status === "pass" &&
        Number(shell?.getAttribute("data-trainable-artifact-loaded-count") ?? 0) >= 2 &&
        Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0) >= 2
      );
    }, undefined, { timeout: 15000 });
    const temporalJitterCase = page.locator("[data-object-state-benchmark-case-name='temporal_jitter']");
    await openContainingDetails(temporalJitterCase);
    await temporalJitterCase.click();
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const benchmark = document.querySelector("[data-object-state-benchmark='true']");
      const selectedRow = document.querySelector(
        "[data-object-state-benchmark-case-name='temporal_jitter']",
      );
      return (
        shell?.getAttribute("data-object-state-benchmark-active-case") === "temporal_jitter" &&
        shell?.getAttribute("data-object-state-benchmark-active-observed-status") === "warn" &&
        shell?.getAttribute("data-object-state-benchmark-active-failure-modes") === "temporal_jitter" &&
        shell?.getAttribute("data-object-state-benchmark-active-diagnostics") === "high_temporal_drift" &&
        shell?.getAttribute("data-object-state-benchmark-active-temporal-drift") === "0.08" &&
        shell?.getAttribute("data-object-state-benchmark-active-dynamic-proposals") === "0" &&
        benchmark?.getAttribute("data-object-state-benchmark-active-case") === "temporal_jitter" &&
        benchmark?.getAttribute("data-object-state-benchmark-active-failure-modes") === "temporal_jitter" &&
        selectedRow?.getAttribute("data-object-state-benchmark-case-selected") === "true"
      );
    }, undefined, { timeout: 15000 });
    const trainableSelection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const target = world?.objectSelections?.find(
        (entry) => entry.modelId === "model-manifest-trainable-artifact",
      );
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
      };
    });
    if (!trainableSelection.ok || trainableSelection.modelId !== "model-manifest-trainable-artifact") {
      throw new Error(`expected algorithm manifest trainable object selection: ${JSON.stringify(trainableSelection)}`);
    }
    const trainableGaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
        assignmentProbeStatus: window.__OBJGAUSS_WORLD__?.assignmentProbeStatus ?? null,
        assignmentProbeMargin: window.__OBJGAUSS_WORLD__?.assignmentProbeMargin ?? null,
      };
    }, trainableSelection.selectionId);
    if (
      !trainableGaussian.ok ||
      trainableGaussian.assignmentSource !== "trainable_kernel_model_artifact" ||
      trainableGaussian.assignmentProbeStatus !== "confident" ||
      !(Number(trainableGaussian.assignmentProbeMargin) > 0.55)
    ) {
      throw new Error(`expected algorithm manifest trainable Gaussian probe: ${JSON.stringify(trainableGaussian)}`);
    }

    await clickModelPill(page, "model-manifest-ogc-artifact");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      return (
        shell?.getAttribute("data-selected-model") === "model-manifest-ogc-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "range-ogc" &&
        shell?.getAttribute("data-quality-report-status") === "warn" &&
        shell?.getAttribute("data-ogc-artifact-index-path") === "/models/ogc-url-fixture/scene.index.json" &&
        shell?.getAttribute("data-ogc-artifact-payload-path") === "/models/ogc-url-fixture/scene.ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "1" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "20" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "2" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2
      );
    }, undefined, { timeout: 15000 });
    const ogcSelection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const targets = world?.objectSelections?.filter(
        (entry) => entry.modelId === "model-manifest-ogc-artifact",
      ) ?? [];
      const target = targets[0];
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
        objectCount: targets.length,
      };
    });
    if (!ogcSelection.ok || ogcSelection.modelId !== "model-manifest-ogc-artifact" || ogcSelection.objectCount !== 2) {
      throw new Error(`expected algorithm manifest OGC object selection: ${JSON.stringify(ogcSelection)}`);
    }
    const ogcGaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
        assignmentProbeStatus: window.__OBJGAUSS_WORLD__?.assignmentProbeStatus ?? null,
        assignmentProbeMargin: window.__OBJGAUSS_WORLD__?.assignmentProbeMargin ?? null,
      };
    }, ogcSelection.selectionId);
    if (
      !ogcGaussian.ok ||
      ogcGaussian.assignmentSource !== "derived_from_object_id" ||
      ogcGaussian.assignmentProbeStatus !== "confident" ||
      !(Number(ogcGaussian.assignmentProbeMargin) > 0.85)
    ) {
      throw new Error(`expected algorithm manifest OGC Gaussian probe: ${JSON.stringify(ogcGaussian)}`);
    }
    await clickSelector(page, "[data-ogc-chunk-button='0']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const types = new Set(events.map((event) => event.type));
      return (
        shell?.getAttribute("data-selected-model") === "model-manifest-ogc-artifact" &&
        shell?.getAttribute("data-ogc-artifact-chunk-scope") === "0" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "1" &&
        types.has("manifest-load") &&
        types.has("gaussian-probe") &&
        types.has("ogc-chunks")
      );
    }, undefined, { timeout: 15000 });
    const snapshotExportButton = page.locator("[data-debug-snapshot-export-button='true']");
    await openContainingDetails(snapshotExportButton);
    await snapshotExportButton.click();
    const snapshotExportHandle = await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-debug-snapshot-panel='true']");
      const button = document.querySelector("[data-debug-snapshot-export-button='true']");
      const exported = window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SNAPSHOT__;
      const text = window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SNAPSHOT_TEXT__;
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      let parsed = null;
      if (typeof text === "string") {
        try {
          parsed = JSON.parse(text);
        } catch {
          parsed = null;
        }
      }
      const fileName = shell?.getAttribute("data-debug-snapshot-export-file") ?? "";
      const hasExportEvent = events.some((event) => event.type === "export-snapshot");
      if (
        shell?.getAttribute("data-debug-snapshot-export-status") !== "exported" ||
        panel?.getAttribute("data-debug-snapshot-export-status") !== "exported" ||
        button?.getAttribute("data-export-status") !== "exported" ||
        shell?.getAttribute("data-debug-snapshot-export-schema") !== "objgauss-object-state-debug-snapshot-v1" ||
        exported?.schema !== "objgauss-object-state-debug-snapshot-v1" ||
        parsed?.schema !== "objgauss-object-state-debug-snapshot-v1" ||
        parsed?.protocol !== "object-state-debug-os-v1" ||
        parsed?.export?.schema !== "objgauss-debug-snapshot-export-v1" ||
        parsed?.model?.id !== "model-manifest-ogc-artifact" ||
        parsed?.debug?.overlayMode !== "full" ||
        parsed?.assignment?.probe?.status !== "confident" ||
        !(Number(parsed?.assignment?.probe?.margin) > 0.85) ||
        parsed?.continuity?.schema !== "objgauss-object-continuity-summary-v1" ||
        !parsed?.continuity?.status ||
        parsed?.temporal?.schema !== "objgauss-object-temporal-summary-v1" ||
        !parsed?.temporal?.status ||
        parsed?.explainability?.schema !== "objgauss-object-explainability-summary-v1" ||
        !parsed?.explainability?.status ||
        parsed?.quality?.status !== "warn" ||
        parsed?.quality?.gates?.find?.((gate) => gate.name === "assignment_entropy")?.status !== "warn" ||
        parsed?.benchmark?.status !== "pass" ||
        parsed?.benchmark?.caseCount !== 8 ||
        parsed?.benchmark?.activeCase?.name !== "temporal_jitter" ||
        parsed?.benchmark?.activeCase?.failureModeNames !== "temporal_jitter" ||
        parsed?.delivery?.chunkIds?.[0] !== 0 ||
        !fileName.endsWith(".json") ||
        !parsed?.export?.fileName ||
        !hasExportEvent
      ) {
        return null;
      }
      return {
        status: shell.getAttribute("data-debug-snapshot-export-status"),
        schema: parsed.schema,
        fileName,
        eventCount: events.length,
      };
    }, undefined, { timeout: 15000 });
    const snapshotExport = await snapshotExportHandle.jsonValue();
    const sessionExportButton = page.locator("[data-debug-session-export-button='true']");
    await openContainingDetails(sessionExportButton);
    await sessionExportButton.click();
    const sessionExportHandle = await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-debug-snapshot-panel='true']");
      const button = document.querySelector("[data-debug-session-export-button='true']");
      const exported = window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SESSION__;
      const text = window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SESSION_TEXT__;
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      let parsed = null;
      if (typeof text === "string") {
        try {
          parsed = JSON.parse(text);
        } catch {
          parsed = null;
        }
      }
      const fileName = shell?.getAttribute("data-debug-session-export-file") ?? "";
      const eventTypes = new Set((parsed?.events ?? []).map((event) => event.type));
      const modelIds = new Set((parsed?.models ?? []).map((model) => model.id));
      const hasExportSessionEvent = events.some((event) => event.type === "export-session");
      if (
        shell?.getAttribute("data-debug-session-export-status") !== "exported" ||
        panel?.getAttribute("data-debug-session-export-status") !== "exported" ||
        button?.getAttribute("data-export-status") !== "exported" ||
        shell?.getAttribute("data-debug-session-export-schema") !== "objgauss-object-state-debug-session-v1" ||
        exported?.schema !== "objgauss-object-state-debug-session-v1" ||
        parsed?.schema !== "objgauss-object-state-debug-session-v1" ||
        parsed?.protocol !== "object-state-debug-os-v1" ||
        parsed?.export?.schema !== "objgauss-debug-session-export-v1" ||
        parsed?.snapshot?.schema !== "objgauss-object-state-debug-snapshot-v1" ||
        parsed?.snapshot?.model?.id !== "model-manifest-ogc-artifact" ||
        parsed?.snapshot?.debug?.overlayMode !== "full" ||
        parsed?.snapshot?.assignment?.probe?.status !== "confident" ||
        !(Number(parsed?.snapshot?.assignment?.probe?.margin) > 0.85) ||
        parsed?.snapshot?.continuity?.schema !== "objgauss-object-continuity-summary-v1" ||
        !parsed?.snapshot?.continuity?.status ||
        parsed?.snapshot?.temporal?.schema !== "objgauss-object-temporal-summary-v1" ||
        !parsed?.snapshot?.temporal?.status ||
        parsed?.snapshot?.explainability?.schema !== "objgauss-object-explainability-summary-v1" ||
        !parsed?.snapshot?.explainability?.status ||
        parsed?.snapshot?.quality?.gates?.find?.((gate) => gate.name === "assignment_entropy")?.status !== "warn" ||
        parsed?.snapshot?.benchmark?.status !== "pass" ||
        parsed?.snapshot?.benchmark?.caseCount !== 8 ||
        parsed?.snapshot?.benchmark?.activeCase?.name !== "temporal_jitter" ||
        parsed?.summary?.modelCount !== 10 ||
        parsed?.summary?.trainableArtifactCount < 2 ||
        parsed?.summary?.ogcArtifactCount < 2 ||
        !modelIds.has("model-manifest-trainable-artifact") ||
        !modelIds.has("model-manifest-ogc-artifact") ||
        !eventTypes.has("export-snapshot") ||
        !eventTypes.has("ogc-chunks") ||
        parsed?.exportPolicy?.scope !== "browser-local-download-only" ||
        parsed?.exportPolicy?.trainingOutputs !== "not_committed" ||
        !fileName.endsWith(".json") ||
        !parsed?.export?.fileName ||
        !hasExportSessionEvent
      ) {
        return null;
      }
      return {
        status: shell.getAttribute("data-debug-session-export-status"),
        schema: parsed.schema,
        fileName,
        eventCount: events.length,
      };
    }, undefined, { timeout: 15000 });
    const sessionExport = await sessionExportHandle.jsonValue();
    const sessionText = await page.evaluate(() => window.__OBJGAUSS_LAST_EXPORTED_DEBUG_SESSION_TEXT__ ?? "");
    await page.locator("[data-debug-session-file-input='true']").setInputFiles({
      name: "objgauss-debug-session-import.json",
      mimeType: "application/json",
      buffer: Buffer.from(sessionText, "utf8"),
    });
    const sessionImportHandle = await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-debug-session-archive='true']");
      const protocol = document.querySelector("[data-debug-snapshot-panel='true']");
      const button = document.querySelector("[data-debug-session-import-button='true']");
      const archive = window.__OBJGAUSS_IMPORTED_DEBUG_SESSION__;
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const eventTypes = new Set(events.map((event) => event.type));
      if (
        shell?.getAttribute("data-debug-session-import-status") !== "loaded" ||
        shell?.getAttribute("data-debug-session-import-schema") !== "objgauss-object-state-debug-session-v1" ||
        shell?.getAttribute("data-debug-session-import-file") !== "objgauss-debug-session-import.json" ||
        shell?.getAttribute("data-debug-session-archive-schema") !== "objgauss-object-state-debug-session-v1" ||
        shell?.getAttribute("data-debug-session-archive-model") !== "model-manifest-ogc-artifact" ||
        shell?.getAttribute("data-debug-session-archive-quality") !== "warn" ||
        Number(shell?.getAttribute("data-debug-session-archive-event-count") ?? 0) < 2 ||
        Number(shell?.getAttribute("data-debug-session-archive-model-count") ?? 0) !== 10 ||
        shell?.getAttribute("data-debug-session-diff-status") !== "match" ||
        shell?.getAttribute("data-debug-session-diff-model-match") !== "true" ||
        shell?.getAttribute("data-debug-session-diff-source-match") !== "true" ||
        shell?.getAttribute("data-debug-session-diff-quality-match") !== "true" ||
        shell?.getAttribute("data-debug-session-diff-training-match") !== "true" ||
        Number(shell?.getAttribute("data-debug-session-diff-slot-delta") ?? NaN) !== 0 ||
        Number(shell?.getAttribute("data-debug-session-diff-entropy-delta") ?? NaN) !== 0 ||
        Number(shell?.getAttribute("data-debug-session-diff-field-count") ?? NaN) !== 0 ||
        shell?.getAttribute("data-debug-session-diff-fields") !== "" ||
        panel?.getAttribute("data-debug-session-archive-status") !== "loaded" ||
        panel?.getAttribute("data-debug-session-archive-file") !== "objgauss-debug-session-import.json" ||
        panel?.getAttribute("data-debug-session-archive-schema") !== "objgauss-object-state-debug-session-v1" ||
        panel?.getAttribute("data-debug-session-archive-model") !== "model-manifest-ogc-artifact" ||
        panel?.getAttribute("data-debug-session-archive-quality") !== "warn" ||
        panel?.getAttribute("data-debug-session-diff-status") !== "match" ||
        panel?.getAttribute("data-debug-session-diff-model-match") !== "true" ||
        panel?.getAttribute("data-debug-session-diff-source-match") !== "true" ||
        panel?.getAttribute("data-debug-session-diff-quality-match") !== "true" ||
        panel?.getAttribute("data-debug-session-diff-training-match") !== "true" ||
        Number(panel?.getAttribute("data-debug-session-diff-field-count") ?? NaN) !== 0 ||
        panel?.getAttribute("data-debug-session-diff-fields") !== "" ||
        protocol?.getAttribute("data-debug-session-import-status") !== "loaded" ||
        button?.getAttribute("data-import-status") !== "loaded" ||
        archive?.schema !== "objgauss-object-state-debug-session-v1" ||
        archive?.snapshot?.model?.id !== "model-manifest-ogc-artifact" ||
        archive?.snapshot?.debug?.overlayMode !== "full" ||
        archive?.snapshot?.assignment?.probe?.status !== "confident" ||
        !(Number(archive?.snapshot?.assignment?.probe?.margin) > 0.85) ||
        archive?.snapshot?.continuity?.schema !== "objgauss-object-continuity-summary-v1" ||
        !archive?.snapshot?.continuity?.status ||
        archive?.snapshot?.temporal?.schema !== "objgauss-object-temporal-summary-v1" ||
        !archive?.snapshot?.temporal?.status ||
        archive?.snapshot?.explainability?.schema !== "objgauss-object-explainability-summary-v1" ||
        !archive?.snapshot?.explainability?.status ||
        archive?.snapshot?.quality?.gates?.find?.((gate) => gate.name === "assignment_entropy")?.status !== "warn" ||
        archive?.snapshot?.benchmark?.status !== "pass" ||
        archive?.snapshot?.benchmark?.caseCount !== 8 ||
        archive?.snapshot?.benchmark?.activeCase?.name !== "temporal_jitter" ||
        archive?.summary?.modelCount !== 10 ||
        !eventTypes.has("import-session")
      ) {
        return null;
      }
      return {
        status: shell.getAttribute("data-debug-session-import-status"),
        schema: archive.schema,
        model: archive.snapshot.model.id,
        diffStatus: shell.getAttribute("data-debug-session-diff-status"),
      };
    }, undefined, { timeout: 15000 });
    const sessionImport = await sessionImportHandle.jsonValue();
    await clickModelPill(page, "model-manifest-trainable-artifact");
    const sessionDriftHandle = await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const panel = document.querySelector("[data-debug-session-archive='true']");
      const fields = shell?.getAttribute("data-debug-session-diff-fields") ?? "";
      const fieldSet = new Set(fields.split(",").filter(Boolean));
      if (
        shell?.getAttribute("data-selected-model") !== "model-manifest-trainable-artifact" ||
        shell?.getAttribute("data-debug-session-archive-model") !== "model-manifest-ogc-artifact" ||
        shell?.getAttribute("data-debug-session-diff-status") !== "changed" ||
        shell?.getAttribute("data-debug-session-diff-model-match") !== "false" ||
        shell?.getAttribute("data-debug-session-diff-source-match") !== "false" ||
        shell?.getAttribute("data-debug-session-diff-training-match") !== "false" ||
        Number(shell?.getAttribute("data-debug-session-diff-field-count") ?? 0) < 3 ||
        !fieldSet.has("model") ||
        !fieldSet.has("source") ||
        !fieldSet.has("training") ||
        !fieldSet.has("delivery") ||
        panel?.getAttribute("data-debug-session-diff-status") !== "changed" ||
        panel?.getAttribute("data-debug-session-diff-fields") !== fields
      ) {
        return null;
      }
      return {
        status: shell.getAttribute("data-debug-session-diff-status"),
        fields,
      };
    }, undefined, { timeout: 15000 });
    const sessionDrift = await sessionDriftHandle.jsonValue();
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-algorithm-manifest.png", fullPage: false });
    return {
      status: "manifest-trainable-ogc-debug-os",
      qualityReportStatus: "warn",
      snapshotExportStatus: snapshotExport.status,
      sessionExportStatus: sessionExport.status,
      sessionImportStatus: sessionImport.status,
      sessionDiffStatus: sessionImport.diffStatus,
      sessionDriftStatus: sessionDrift.status,
      objectStateBenchmarkStatus: "pass",
    };
  } finally {
    await page.close();
  }
}

async function auditLocalModelManifestBundleImport(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  try {
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.locator("[data-model-artifact-file-input='true']").setInputFiles([
      "public/models/algorithm-bundle-fixture/model-artifact.json",
      "public/models/algorithm-bundle-fixture/quality-report.json",
      "public/models/algorithm-bundle-fixture/object-state-benchmark.json",
      "public/models/trainable-mvp-debug/model-artifact.json",
      "public/models/ogc-url-fixture/scene.index.json",
      "public/models/ogc-url-fixture/scene.ogc",
    ]);
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const quality = document.querySelector("[data-quality-report='true']");
      const benchmark = document.querySelector("[data-object-state-benchmark='true']");
      const snapshot = document.querySelector("[data-debug-snapshot-panel='true']");
      const entropyGate = document.querySelector("[data-quality-gate-name='assignment_entropy']");
      const button = document.querySelector("[data-model-artifact-import-button='true']");
      const parent = document.querySelector(".modelPill[data-model-row-id='model-local-manifest']");
      const trainable = document.querySelector(".modelPill[data-model-row-id='model-local-manifest-trainable-artifact']");
      const ogc = document.querySelector(".modelPill[data-model-row-id='model-local-manifest-ogc-artifact']");
      const importedFile = shell?.getAttribute("data-model-manifest-import-file") ?? "";
      return (
        button?.getAttribute("data-import-status") === "loaded" &&
        parent?.getAttribute("data-model-load-state") === "loaded" &&
        trainable?.getAttribute("data-model-load-state") === "loaded" &&
        ogc?.getAttribute("data-model-load-state") === "loaded" &&
        shell?.getAttribute("data-model-count") === "10" &&
        shell?.getAttribute("data-catalog-model-count") === "7" &&
        shell?.getAttribute("data-selected-model") === "model-local-manifest-trainable-artifact" &&
        shell?.getAttribute("data-model-manifest-import-status") === "loaded" &&
        shell?.getAttribute("data-model-manifest-import-model") === "model-local-manifest" &&
        importedFile.includes("model-artifact.json") &&
        importedFile.includes("quality-report.json") &&
        importedFile.includes("object-state-benchmark.json") &&
        importedFile.includes("scene.index.json") &&
        importedFile.includes("scene.ogc") &&
        shell?.getAttribute("data-trainable-artifact-load-route") === "local-manifest-file" &&
        shell?.getAttribute("data-trainable-artifact-path") === "local://model-artifact.json" &&
        shell?.getAttribute("data-trainable-training-status") === "loss_down" &&
        shell?.getAttribute("data-trainable-training-image-loss-decreased") === "true" &&
        shell?.getAttribute("data-quality-report-status") === "warn" &&
        shell?.getAttribute("data-object-state-benchmark-status") === "pass" &&
        shell?.getAttribute("data-object-state-benchmark-case-count") === "8" &&
        shell?.getAttribute("data-object-state-benchmark-active-case") === "uniform_mixed" &&
        quality?.getAttribute("data-quality-report-status") === "warn" &&
        quality?.getAttribute("data-quality-report-gate-count") === "3" &&
        benchmark?.getAttribute("data-object-state-benchmark-status") === "pass" &&
        benchmark?.getAttribute("data-object-state-benchmark-first-case") === "clean_sparse" &&
        benchmark?.getAttribute("data-object-state-benchmark-active-case") === "uniform_mixed" &&
        entropyGate?.getAttribute("data-quality-gate-status") === "warn" &&
        snapshot?.getAttribute("data-debug-snapshot-quality-status") === "warn" &&
        window.__OBJGAUSS_DEBUG_SNAPSHOT__?.benchmark?.status === "pass" &&
        Number(shell?.getAttribute("data-trainable-artifact-loaded-count") ?? 0) >= 2 &&
        Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0) >= 2
      );
    }, undefined, { timeout: 15000 });
    const trainableSelection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const target = world?.objectSelections?.find(
        (entry) => entry.modelId === "model-local-manifest-trainable-artifact",
      );
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
      };
    });
    if (!trainableSelection.ok || trainableSelection.modelId !== "model-local-manifest-trainable-artifact") {
      throw new Error(`expected local model manifest trainable object selection: ${JSON.stringify(trainableSelection)}`);
    }
    const trainableGaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, trainableSelection.selectionId);
    if (!trainableGaussian.ok || trainableGaussian.assignmentSource !== "trainable_kernel_model_artifact") {
      throw new Error(`expected local model manifest trainable Gaussian probe: ${JSON.stringify(trainableGaussian)}`);
    }

    await clickModelPill(page, "model-local-manifest-ogc-artifact");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      return (
        shell?.getAttribute("data-selected-model") === "model-local-manifest-ogc-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "local-file" &&
        shell?.getAttribute("data-quality-report-status") === "warn" &&
        shell?.getAttribute("data-ogc-artifact-index-path") === "local://scene.index.json" &&
        shell?.getAttribute("data-ogc-artifact-payload-path") === "local://scene.ogc" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "0" &&
        shell?.getAttribute("data-ogc-artifact-fetched-bytes") === "41" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "40" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "2" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2
      );
    }, undefined, { timeout: 15000 });
    const ogcSelection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const targets = world?.objectSelections?.filter(
        (entry) => entry.modelId === "model-local-manifest-ogc-artifact",
      ) ?? [];
      const target = targets[0];
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
        objectCount: targets.length,
      };
    });
    if (!ogcSelection.ok || ogcSelection.modelId !== "model-local-manifest-ogc-artifact" || ogcSelection.objectCount !== 2) {
      throw new Error(`expected local model manifest OGC object selection: ${JSON.stringify(ogcSelection)}`);
    }
    const ogcGaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, ogcSelection.selectionId);
    if (!ogcGaussian.ok || ogcGaussian.assignmentSource !== "derived_from_object_id") {
      throw new Error(`expected local model manifest OGC Gaussian probe: ${JSON.stringify(ogcGaussian)}`);
    }
    await clickSelector(page, "[data-ogc-lod-button='1']");
    await clickSelector(page, "[data-ogc-chunk-button='0']");
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const types = new Set(events.map((event) => event.type));
      return (
        shell?.getAttribute("data-selected-model") === "model-local-manifest-ogc-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "local-file" &&
        shell?.getAttribute("data-ogc-artifact-lod-level") === "1" &&
        shell?.getAttribute("data-ogc-artifact-requested-bytes") === "10" &&
        shell?.getAttribute("data-ogc-artifact-decoded-windows") === "1" &&
        shell?.getAttribute("data-ogc-artifact-chunk-scope") === "0" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 1 &&
        types.has("import-model-manifest") &&
        types.has("gaussian-probe") &&
        types.has("ogc-lod") &&
        types.has("ogc-chunks")
      );
    }, undefined, { timeout: 15000 });
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-local-model-manifest.png", fullPage: false });
    return {
      status: "local-manifest-trainable-ogc-debug-os",
      qualityReportStatus: "warn",
      objectStateBenchmarkStatus: "pass",
    };
  } finally {
    await page.close();
  }
}

async function auditLocalTrainableManifestPackageImport(browser, url) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 820 } });
  try {
    const { manifestPath, qualityReportPath } = writeLocalTrainableManifestFixture();
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
    await page.locator(".worldShell").waitFor({ timeout: 15000 });
    await page.locator("[data-model-artifact-file-input='true']").setInputFiles([
      manifestPath,
      "public/models/trainable-mvp-debug/model-artifact.json",
      qualityReportPath,
    ]);
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const training = document.querySelector("[data-training-evidence='true']");
      const quality = document.querySelector("[data-quality-report='true']");
      const entropyGate = document.querySelector("[data-quality-gate-name='assignment_entropy']");
      const driftGate = document.querySelector("[data-quality-gate-name='temporal_drift']");
      const parent = document.querySelector(".modelPill[data-model-row-id='model-local-manifest']");
      const trainable = document.querySelector(".modelPill[data-model-row-id='model-local-manifest-trainable-artifact']");
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      const types = new Set(events.map((event) => event.type));
      return (
        parent?.getAttribute("data-model-load-state") === "loaded" &&
        trainable?.getAttribute("data-model-load-state") === "loaded" &&
        shell?.getAttribute("data-selected-model") === "model-local-manifest-trainable-artifact" &&
        shell?.getAttribute("data-model-manifest-import-status") === "loaded" &&
        shell?.getAttribute("data-trainable-artifact-load-route") === "local-manifest-file" &&
        shell?.getAttribute("data-trainable-artifact-path") === "local://model-artifact.json" &&
        shell?.getAttribute("data-trainable-training-status") === "loss_down" &&
        shell?.getAttribute("data-trainable-training-image-loss-decreased") === "true" &&
        shell?.getAttribute("data-quality-report-status") === "warn" &&
        shell?.getAttribute("data-quality-report-schema") === "objgauss-object-state-quality-report-v1" &&
        training?.getAttribute("data-training-status") === "loss_down" &&
        quality?.getAttribute("data-quality-report-status") === "warn" &&
        quality?.getAttribute("data-quality-report-gate-count") === "3" &&
        quality?.getAttribute("data-quality-report-failing-gate-names") === "assignment_entropy" &&
        entropyGate?.getAttribute("data-quality-gate-status") === "warn" &&
        driftGate?.getAttribute("data-quality-gate-status") === "pass" &&
        Number(shell?.getAttribute("data-trainable-artifact-loaded-count") ?? 0) >= 2 &&
        types.has("import-model-manifest")
      );
    }, undefined, { timeout: 15000 });
    const selection = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const target = world?.objectSelections?.find(
        (entry) => entry.modelId === "model-local-manifest-trainable-artifact",
      );
      return {
        ok: world?.selectObjectForAudit?.(target?.selectionId) ?? false,
        selectionId: target?.selectionId ?? null,
        modelId: target?.modelId ?? null,
      };
    });
    if (!selection.ok || selection.modelId !== "model-local-manifest-trainable-artifact") {
      throw new Error(`expected local trainable manifest object selection: ${JSON.stringify(selection)}`);
    }
    const gaussian = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      return {
        ok: world?.selectGaussianForAudit?.(selectionId, 0) ?? false,
        assignmentSource: world?.assignmentSource ?? null,
      };
    }, selection.selectionId);
    if (!gaussian.ok || gaussian.assignmentSource !== "trainable_kernel_model_artifact") {
      throw new Error(`expected local trainable manifest Gaussian probe: ${JSON.stringify(gaussian)}`);
    }
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const snapshot = window.__OBJGAUSS_DEBUG_SNAPSHOT__;
      return (
        shell?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        shell?.getAttribute("data-debug-snapshot-model") === "model-local-manifest-trainable-artifact" &&
        snapshot?.model?.id === "model-local-manifest-trainable-artifact" &&
        snapshot?.training?.status === "loss_down" &&
        snapshot?.quality?.status === "warn"
      );
    }, undefined, { timeout: 15000 });
    return { status: "local-trainable-manifest-quality-debug-os" };
  } finally {
    await page.close();
  }
}

function writeLocalOgcManifestFixture() {
  const index = JSON.parse(readFileSync("public/models/ogc-url-fixture/scene.index.json", "utf8"));
  const manifestPath = "/tmp/objgauss-local-ogc-model-artifact.json";
  const manifest = {
    schema: "objgauss-model-artifact-manifest-v1",
    manifest_id: "objgauss-local-ogc-package-fixture",
    asset_id: "objgauss-local-ogc-package-fixture",
    name: "OGC manifest package fixture",
    stage: "audit-fixture",
    source: {
      type: "audit_local_manifest_package",
      index: "scene.index.json",
      payload: "scene.ogc",
    },
    license: "fixture",
    counts: {
      gaussians: index.gaussian_count,
      objects: index.object_count,
    },
    artifacts: [
      {
        role: "compressed_chunked",
        path: "scene.ogc",
        format: ".ogc",
        delivery_tier: "browser_edit",
        browser_ready: true,
        gaussian_count: index.gaussian_count,
        object_count: index.object_count,
        byte_size: index.payload.byte_size,
        sha256: index.payload.sha256,
        chunk_index: {
          schema: index.schema,
          path: "scene.index.json",
          chunk_count: index.chunks.length,
          sort_key: index.sort_key,
          chunk_size_target: index.chunk_size_target,
        },
        compression: index.compression,
        lod: index.lod,
        object_id_coverage: index.object_id_coverage,
      },
    ],
    quality_evidence: [
      {
        kind: "audit-local-ogc-package-fixture",
        status: "fixture",
      },
    ],
    limitations: ["Tiny audit fixture for local model artifact manifest package import."],
  };
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  return manifestPath;
}

function writeLocalTrainableManifestFixture() {
  const trainable = JSON.parse(readFileSync("public/models/trainable-mvp-debug/model-artifact.json", "utf8"));
  const manifestPath = "/tmp/objgauss-local-trainable-model-artifact.json";
  const qualityReportPath = "/tmp/objgauss-local-trainable-quality-report.json";
  const sample = trainable.source?.sample ?? {};
  const qualityReport = {
    schema: "objgauss-object-state-quality-report-v1",
    report_id: "objgauss-local-trainable-package-quality",
    status: "warn",
    source: {
      type: "audit_local_trainable_package",
      artifact: "model-artifact.json",
    },
    metrics: {
      assignment_entropy: 0.68875,
      slot_utilization: 1.0,
      object_purity: 0.79625,
      temporal_drift: 0.017205,
      assignment_jitter: 0.0225,
      bbox_stability: 0.960402,
      spatial_compactness: 0.423428,
    },
    gates: [
      { name: "slot_utilization", status: "pass", value: 1.0, threshold: 0.7 },
      { name: "assignment_entropy", status: "warn", value: 0.68875, threshold: 0.5 },
      { name: "temporal_drift", status: "pass", value: 0.017205, threshold: 0.08 },
    ],
    limitations: ["Tiny audit fixture for trainable quality report package import."],
  };
  const manifest = {
    schema: "objgauss-model-artifact-manifest-v1",
    manifest_id: "objgauss-local-trainable-package-fixture",
    asset_id: "objgauss-local-trainable-package-fixture",
    name: "Local trainable package fixture",
    stage: "audit-fixture",
    source: {
      type: "audit_local_trainable_package",
      input: trainable.source?.input,
      target_source: sample.target_source,
    },
    license: "fixture",
    counts: {
      gaussians: sample.sampled_count,
      objects: trainable.training?.slots,
    },
    artifacts: [
      {
        role: "trainable_kernel",
        path: "model-artifact.json",
        format: ".json",
        delivery_tier: "browser_edit",
        browser_ready: true,
        gaussian_count: sample.sampled_count,
        object_count: trainable.training?.slots,
        label: "trainable-kernel-model-artifact",
        note: "Audit fixture for trainable-only model manifest package import.",
      },
      {
        role: "quality_report",
        path: "objgauss-local-trainable-quality-report.json",
        format: ".json",
        delivery_tier: "browser_edit",
        browser_ready: true,
        label: "ObjectState quality report",
      },
    ],
    quality_evidence: [
      {
        kind: "trainable_kernel_training_summary",
        source: "model-artifact.json",
        summary: trainable.training,
      },
    ],
    limitations: ["Tiny audit fixture for local trainable model artifact manifest package import."],
    created_from: {
      trainable_model_artifact: "model-artifact.json",
      schema: trainable.schema,
      input: trainable.source?.input,
    },
  };
  writeFileSync(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  writeFileSync(qualityReportPath, `${JSON.stringify(qualityReport, null, 2)}\n`, "utf8");
  return { manifestPath, qualityReportPath };
}

function launchOptions() {
  const executablePath = firstExisting([
    process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE,
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
  ]);
  return executablePath ? { executablePath } : {};
}

function startServer(port) {
  const child = spawn(
    "npm",
    ["run", "dev", "--", "--port", String(port), "--strictPort"],
    { detached: true, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stdout.on("data", (chunk) => process.stdout.write(chunk));
  child.stderr.on("data", (chunk) => process.stderr.write(chunk));
  return child;
}

function stopServer(child) {
  if (!child.pid) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    child.kill("SIGTERM");
  }
}

async function clickModelPill(page, modelId) {
  const clicked = await page.evaluate((id) => {
    const selector = `.modelPill[data-model-row-id="${id}"]`;
    const element = document.querySelector(selector);
    element?.click();
    return Boolean(element);
  }, modelId);
  if (!clicked) {
    throw new Error(`model pill not found: ${modelId}`);
  }
}

async function clickSelector(page, selector) {
  const clicked = await page.evaluate((targetSelector) => {
    const element = document.querySelector(targetSelector);
    element?.click();
    return Boolean(element);
  }, selector);
  if (!clicked) {
    throw new Error(`click target not found: ${selector}`);
  }
}

async function waitForApp(url) {
  const deadline = Date.now() + 30000;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch (error) {
      lastError = error;
    }
    await sleep(250);
  }
  throw new Error(`app did not become ready at ${url}: ${lastError?.message ?? "timeout"}`);
}

async function closeBrowserWithTimeout(browser, timeoutMs = 5000) {
  await Promise.race([
    browser.close(),
    sleep(timeoutMs).then(() => undefined),
  ]);
}

async function openContainingDetails(locator) {
  await locator.evaluate((node) => {
    node.closest("details")?.setAttribute("open", "");
  });
  await locator.scrollIntoViewIfNeeded();
}

function firstExisting(paths) {
  return paths.find((entry) => entry && existsSync(entry));
}

function parseArgs(rawArgs) {
  const parsed = {};
  for (let index = 0; index < rawArgs.length; index += 1) {
    const arg = rawArgs[index];
    if (!arg.startsWith("--")) continue;
    const key = arg.slice(2);
    const next = rawArgs[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}
