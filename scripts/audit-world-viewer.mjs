import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
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
      `ogcLoaded=${summary.ogcLoadedCount}`,
      `trainableArtifacts=${summary.trainableArtifactLoadedCount}`,
      `trainableRoute=${summary.trainableArtifactLoadRoute}`,
      `trainableArtifact=${JSON.stringify(summary.trainableArtifactPath)}`,
      `trainableFrame=${summary.trainableFrameIndex}/${summary.trainableFrameCount}`,
      `urlArtifact=${summary.urlArtifactStatus}`,
      `urlOgc=${summary.urlOgcStatus}`,
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
      return (
        shell?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        heatmap?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2 &&
        stability?.getAttribute("data-stability-status") === shell?.getAttribute("data-stability-status") &&
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
    const world = await page.evaluate(() => {
      const handle = window.__OBJGAUSS_WORLD__;
      const shell = document.querySelector(".worldShell");
      const stability = document.querySelector("[data-stability-dashboard='true']");
      return {
        modelCount: handle.modelCount,
        objectCount: handle.objectCount,
        draggableObjectCount: handle.draggableObjectCount,
        selectedId: handle.selectedId,
        selectedModelId: handle.selectedModelId,
        selectedObjectId: handle.selectedObjectId,
        debugMode: handle.debugMode,
        debugProtocol: handle.debugProtocol,
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
      ogcLoadedCount: world.ogcLoadedCount,
      trainableArtifactLoadedCount: world.trainableArtifactLoadedCount,
      trainableArtifactLoadRoute: world.trainableArtifactLoadRoute,
      trainableArtifactPath: world.trainableArtifactPath,
      trainableFrameIndex: world.trainableFrameIndex,
      trainableFrameCount: world.trainableFrameCount,
      urlArtifactStatus: urlArtifact.status,
      urlOgcStatus: urlOgc.status,
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
      return (
        shell?.getAttribute("data-selected-model") === "trainable-url-artifact" &&
        shell?.getAttribute("data-assignment-source") === "trainable_kernel_model_artifact" &&
        shell?.getAttribute("data-trainable-artifact-frame-index") === "1" &&
        frameSelector?.getAttribute("data-selected-frame") === "1" &&
        Number(heatmap?.getAttribute("data-assignment-slots") ?? 0) === 2
      );
    }, undefined, { timeout: 15000 });
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-url-artifact.png", fullPage: false });
    return { status: "fetch-json" };
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
    await page.screenshot({ path: "/tmp/objgauss-world-viewer-url-ogc.png", fullPage: false });
    return { status: "range-ogc" };
  } finally {
    await page.close();
  }
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
