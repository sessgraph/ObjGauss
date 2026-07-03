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
  const summary = await auditWorld(baseUrl);
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
      `debugEvents=${summary.debugEventCount}`,
      `ogcLoaded=${summary.ogcLoadedCount}`,
      `trainableArtifacts=${summary.trainableArtifactLoadedCount}`,
      `trainableRoute=${summary.trainableArtifactLoadRoute}`,
      `trainableArtifact=${JSON.stringify(summary.trainableArtifactPath)}`,
      `trainableFrame=${summary.trainableFrameIndex}/${summary.trainableFrameCount}`,
      `trainLoss=${summary.trainableTrainingFinalLoss}`,
      `trainLossDelta=${summary.trainableTrainingLossDelta}`,
      `trainImageLoss=${summary.trainableTrainingFinalImageLoss}`,
      `urlArtifact=${summary.urlArtifactStatus}`,
      `urlOgc=${summary.urlOgcStatus}`,
      `urlOgcManifest=${summary.urlOgcManifestStatus}`,
      `algorithmManifest=${summary.algorithmManifestStatus}`,
      `localModelManifest=${summary.localModelManifestStatus}`,
      `localArtifact=${summary.localArtifactStatus}`,
      `localOgc=${summary.localOgcStatus}`,
      `localOgcManifest=${summary.localOgcManifestStatus}`,
      `assignmentSlots=${summary.assignmentSlots}`,
      `assignmentSource=${summary.assignmentSource}`,
      `stability=${summary.stabilityStatus}`,
      `slotUtil=${summary.slotUtilization}`,
      `mixedSlots=${summary.mixedSlots}`,
      `purity=${summary.meanPurity}`,
      `temporalDrift=${summary.meanTemporalDrift}`,
      `compactness=${summary.meanSpatialCompactness}`,
      `assignmentJitter=${summary.meanAssignmentJitter}`,
      `bboxStability=${summary.meanBboxStability}`,
      `hoveredObject=${JSON.stringify(summary.hoveredObjectId)}`,
      `hoveredGaussians=${summary.hoveredGaussianCount}`,
      `selectedGaussian=${JSON.stringify(summary.selectedGaussian)}`,
      `sidebars=${summary.sidebars}`,
      `screenshot=${summary.screenshotPath}`,
      `mobileScreenshot=${summary.mobileScreenshotPath}`,
    ].join(" "),
  );
} finally {
  if (server) stopServer(server);
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
    await page.goto(url, { waitUntil: "networkidle", timeout: 30000 });
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
    await page.locator("[data-assignment-heatmap='true']").waitFor({ timeout: 15000 });
    await page.locator("[data-stability-dashboard='true']").waitFor({ timeout: 15000 });
    const pills = await page.locator(".modelPill").count();
    if (pills < 7) {
      throw new Error(`expected at least 7 model pills, found ${pills}`);
    }
    const ogcPill = page.locator(".modelPill[data-model-row-id='ogc-debug']");
    await ogcPill.waitFor({ timeout: 15000 });
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const pill = document.querySelector(".modelPill[data-model-row-id='ogc-debug']");
      return pill?.getAttribute("data-model-load-state") === "loaded" &&
        Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0) > 0;
    }, undefined, { timeout: 15000 });
    await ogcPill.click();
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
    await trainablePill.click();
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
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      const stability = document.querySelector("[data-stability-dashboard='true']");
      const training = document.querySelector("[data-training-evidence='true']");
      return (
        shell?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        heatmap?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2 &&
        stability?.getAttribute("data-stability-status") === shell?.getAttribute("data-stability-status") &&
        training?.getAttribute("data-training-status") === "loss_down" &&
        training?.getAttribute("data-training-renderer") === "cpu-image-point-splat-differentiable-v1" &&
        training?.getAttribute("data-training-image-loss-decreased") === "true" &&
        Number(training?.getAttribute("data-training-final-total-loss") ?? 0) > 0 &&
        Number(training?.getAttribute("data-training-loss-delta") ?? 0) > 0 &&
        Number(training?.getAttribute("data-training-final-image-loss") ?? 0) > 0 &&
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
    await page.locator("[data-trainable-frame-button='1']").click();
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
      };
    }, frameSwitch.selectionId);
    if (!frameGaussian.ok || frameGaussian.assignmentSource !== "trainable_kernel_model_artifact") {
      throw new Error(`expected frame 1 Gaussian probe to remain trainable artifact sourced: ${JSON.stringify(frameGaussian)}`);
    }
    await page.locator("[data-debug-lens-button='confidence']").click();
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
    await page.locator("[data-debug-lens-button='entropy']").click();
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
    const hoverSelection = await page.evaluate((selectionId) => {
      const world = window.__OBJGAUSS_WORLD__;
      const result = world?.hoverObjectForAudit?.(selectionId) ?? null;
      return {
        result,
        hoveredId: window.__OBJGAUSS_WORLD__?.hoveredId ?? null,
        hoveredGaussianCount: window.__OBJGAUSS_WORLD__?.hoveredGaussianCount ?? 0,
        hoveredAssignmentSource: window.__OBJGAUSS_WORLD__?.hoveredAssignmentSource ?? null,
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
    await page.waitForFunction((selectionId) => {
      const shell = document.querySelector(".worldShell");
      const world = window.__OBJGAUSS_WORLD__;
      return (
        world?.hoveredId === selectionId &&
        shell?.getAttribute("data-hovered-target") === selectionId &&
        shell?.getAttribute("data-hovered-model") === "trainable-mvp-debug" &&
        Number(shell?.getAttribute("data-hovered-gaussians") ?? 0) > 0
      );
    }, trainableSelection.selectionId, { timeout: 15000 });
    const visibilityToggle = await page.evaluate(() => {
      const world = window.__OBJGAUSS_WORLD__;
      const selection = world?.objectSelections?.[0];
      const before = world?.visibleObjectCount ?? null;
      const visibleAfterToggle = world?.toggleObjectVisibilityForAudit?.(selection?.selectionId) ?? null;
      const after = window.__OBJGAUSS_WORLD__?.visibleObjectCount ?? null;
      return { before, after, visibleAfterToggle };
    });
    if (!(visibilityToggle.after < visibilityToggle.before)) {
      throw new Error(`expected object visibility toggle to reduce visible count: ${JSON.stringify(visibilityToggle)}`);
    }
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
    const localArtifact = await auditLocalTrainableArtifactImport(browser, url);
    const localOgc = await auditLocalOgcArtifactImport(browser, url);
    const localOgcManifest = await auditLocalOgcManifestPackageImport(browser, url);
    const world = await page.evaluate(() => {
      const handle = window.__OBJGAUSS_WORLD__;
      const snapshot = window.__OBJGAUSS_DEBUG_SNAPSHOT__;
      const shell = document.querySelector(".worldShell");
      const stability = document.querySelector("[data-stability-dashboard='true']");
      const training = document.querySelector("[data-training-evidence='true']");
      const snapshotPanel = document.querySelector("[data-debug-snapshot-panel='true']");
      const tracePanel = document.querySelector("[data-debug-event-trace='true']");
      const events = window.__OBJGAUSS_DEBUG_EVENTS__ ?? [];
      return {
        modelCount: handle.modelCount,
        objectCount: handle.objectCount,
        draggableObjectCount: handle.draggableObjectCount,
        selectedId: handle.selectedId,
        selectedModelId: handle.selectedModelId,
        selectedObjectId: handle.selectedObjectId,
        debugMode: handle.debugMode,
        debugLens: handle.debugLens,
        debugProtocol: handle.debugProtocol,
        debugSnapshotSchema: snapshot?.schema ?? null,
        debugSnapshotProtocol: snapshot?.protocol ?? null,
        debugSnapshotModel: snapshot?.model?.id ?? null,
        debugSnapshotObject: snapshot?.selection?.objectId ?? null,
        debugSnapshotLens: snapshot?.debug?.lens ?? null,
        debugSnapshotAssignmentSource: snapshot?.assignment?.source ?? null,
        debugSnapshotAssignmentSlots: Number(snapshot?.assignment?.slotCount ?? 0),
        debugSnapshotTrainingStatus: snapshot?.training?.status ?? null,
        debugSnapshotEventCount: Array.isArray(snapshot?.events) ? snapshot.events.length : 0,
        debugSnapshotEventTypes: Array.isArray(snapshot?.events) ? snapshot.events.map((event) => event.type) : [],
        shellDebugSnapshotSchema: shell?.getAttribute("data-debug-snapshot-schema") ?? null,
        shellDebugSnapshotModel: shell?.getAttribute("data-debug-snapshot-model") ?? null,
        shellDebugSnapshotSlots: Number(shell?.getAttribute("data-debug-snapshot-assignment-slots") ?? 0),
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
        assignmentSource: handle.assignmentSource,
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
        shellHoveredTarget: shell?.getAttribute("data-hovered-target") ?? null,
        shellHoveredGaussians: Number(shell?.getAttribute("data-hovered-gaussians") ?? 0),
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
      world.debugLens === "entropy" &&
      world.entropyLensSamples.length >= 2 &&
      world.entropyLensSamples.every((sample) => sample.opacityLens === "entropy") &&
      world.entropyLensSamples.some((sample) => sample.opacity > 0.32 && sample.opacity < 1)
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
      world.debugSnapshotTrainingStatus === "loss_down" &&
      world.shellDebugSnapshotSchema === world.debugSnapshotSchema &&
      world.shellDebugSnapshotModel === world.debugSnapshotModel &&
      world.shellDebugSnapshotSlots === world.debugSnapshotAssignmentSlots &&
      world.shellDebugSnapshotStability === world.stabilityStatus &&
      world.panelDebugSnapshotSchema === world.debugSnapshotSchema &&
      world.panelDebugSnapshotModel === world.debugSnapshotModel &&
      world.panelDebugSnapshotLens === world.debugSnapshotLens &&
      world.panelDebugSnapshotSlots === world.debugSnapshotAssignmentSlots
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
      urlArtifactStatus: urlArtifact.status,
      urlOgcStatus: urlOgc.status,
      urlOgcManifestStatus: urlOgcManifest.status,
      algorithmManifestStatus: algorithmManifest.status,
      localModelManifestStatus: localModelManifest.status,
      localArtifactStatus: localArtifact.status,
      localOgcStatus: localOgc.status,
      localOgcManifestStatus: localOgcManifest.status,
      assignmentSource: world.assignmentSource,
      stabilityStatus: world.stabilityStatus,
      slotUtilization: world.slotUtilization,
      mixedSlots: world.mixedSlots,
      meanPurity: world.meanPurity,
      meanTemporalDrift: world.meanTemporalDrift,
      meanSpatialCompactness: world.meanSpatialCompactness,
      meanAssignmentJitter: world.meanAssignmentJitter,
      meanBboxStability: world.meanBboxStability,
      hoveredObjectId: world.hoveredObjectId,
      hoveredGaussianCount: world.hoveredGaussianCount,
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
    await page.locator(".modelPill[data-model-row-id='trainable-mvp-debug']").click();
    await page.locator("[data-trainable-frame-button='1']").click();
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
    await page.locator("[data-trainable-frame-button='1']").click();
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
      };
    }, selection.selectionId);
    if (!gaussian.ok || gaussian.assignmentSource !== "trainable_kernel_model_artifact") {
      throw new Error(`expected local artifact Gaussian probe: ${JSON.stringify(gaussian)}`);
    }
    await page.locator("[data-trainable-frame-button='1']").click();
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
    await page.locator("[data-ogc-lod-button='1']").click();
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
    await page.locator("[data-ogc-chunk-button='0']").click();
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
    await page.locator("[data-ogc-lod-button='1']").click();
    await page.locator("[data-ogc-chunk-button='0']").click();
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
    await page.locator("[data-ogc-lod-button='0']").click();
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
    await page.locator("[data-ogc-chunk-button='0']").click();
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
    await page.locator("[data-ogc-chunk-button='all']").click();
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
    await page.locator("[data-ogc-lod-button='0']").click();
    await page.locator("[data-ogc-chunk-button='0']").click();
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
        Number(shell?.getAttribute("data-trainable-artifact-loaded-count") ?? 0) >= 2 &&
        Number(shell?.getAttribute("data-ogc-loaded-count") ?? 0) >= 2
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
      };
    }, trainableSelection.selectionId);
    if (!trainableGaussian.ok || trainableGaussian.assignmentSource !== "trainable_kernel_model_artifact") {
      throw new Error(`expected algorithm manifest trainable Gaussian probe: ${JSON.stringify(trainableGaussian)}`);
    }

    await page.locator(".modelPill[data-model-row-id='model-manifest-ogc-artifact']").click();
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      return (
        shell?.getAttribute("data-selected-model") === "model-manifest-ogc-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "range-ogc" &&
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
      };
    }, ogcSelection.selectionId);
    if (!ogcGaussian.ok || ogcGaussian.assignmentSource !== "derived_from_object_id") {
      throw new Error(`expected algorithm manifest OGC Gaussian probe: ${JSON.stringify(ogcGaussian)}`);
    }
    await page.locator("[data-ogc-chunk-button='0']").click();
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
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-algorithm-manifest.png", fullPage: false });
    return { status: "manifest-trainable-ogc-debug-os" };
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
      "public/models/trainable-mvp-debug/model-artifact.json",
      "public/models/ogc-url-fixture/scene.index.json",
      "public/models/ogc-url-fixture/scene.ogc",
    ]);
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
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
        importedFile.includes("scene.index.json") &&
        importedFile.includes("scene.ogc") &&
        shell?.getAttribute("data-trainable-artifact-load-route") === "local-manifest-file" &&
        shell?.getAttribute("data-trainable-artifact-path") === "local://model-artifact.json" &&
        shell?.getAttribute("data-trainable-training-status") === "loss_down" &&
        shell?.getAttribute("data-trainable-training-image-loss-decreased") === "true" &&
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

    await page.locator(".modelPill[data-model-row-id='model-local-manifest-ogc-artifact']").click();
    await page.waitForFunction(() => {
      const shell = document.querySelector(".worldShell");
      const heatmap = document.querySelector("[data-assignment-heatmap='true']");
      return (
        shell?.getAttribute("data-selected-model") === "model-local-manifest-ogc-artifact" &&
        shell?.getAttribute("data-ogc-artifact-load-route") === "local-file" &&
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
    await page.locator("[data-ogc-lod-button='1']").click();
    await page.locator("[data-ogc-chunk-button='0']").click();
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
    return { status: "local-manifest-trainable-ogc-debug-os" };
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
