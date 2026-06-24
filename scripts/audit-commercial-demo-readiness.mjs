import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { ASSET_LIBRARY } from "../src/assetLibrary.js";

const MODE = "commercial-demo-readiness-v1";
const DEFAULT_ROUTE_SUMMARIES = [
  "/tmp/objgauss-spark-commercial-route/summary.json",
  "/tmp/objgauss-spark-commercial-route-availability/summary.json",
  "/tmp/objgauss-spark-commercial-route-report/summary.json",
];
const DEFAULT_QUALITY_SUMMARIES = [
  "/tmp/objgauss-hard-mask-quality/summary.json",
  "/tmp/objgauss-hard-mask-quality-default/summary.json",
];

const args = parseArgs(process.argv.slice(2));
const outputDir = String(
  args.outputDir ?? args["output-dir"] ?? "/tmp/objgauss-commercial-demo-readiness",
);
const routeSummaryPaths = pathList(args.routeSummary ?? args["route-summary"], DEFAULT_ROUTE_SUMMARIES);
const qualitySummaryPaths = pathList(
  args.qualitySummary ?? args["quality-summary"],
  DEFAULT_QUALITY_SUMMARIES,
);
const requireRouteEvidence = flagEnabled(args.requireRouteEvidence ?? args["require-route-evidence"]);
const requirePublicCommercial = flagEnabled(
  args.requirePublicCommercial ?? args["require-public-commercial"],
);

const routeSummary = firstExistingJson(routeSummaryPaths);
const qualitySummary = firstExistingJson(qualitySummaryPaths);
if (!routeSummary) {
  throw new Error(
    `missing Spark route summary; run npm run acceptance:spark-commercial-route first or pass --route-summary`,
  );
}
if (!qualitySummary) {
  throw new Error(
    `missing hard-mask quality summary; run npm run audit:hard-mask-quality first or pass --quality-summary`,
  );
}

const routeByAsset = buildRouteIndex(routeSummary.payload);
const qualityByAsset = buildQualityIndex(qualitySummary.payload);
const renderableAssets = ASSET_LIBRARY.filter(
  (asset) => asset.sourceType === "gaussian" && asset.localPath && asset.splatPath,
);
const routeOnlyIds = new Set([
  ...routeByAsset.keys(),
  ...qualityByAsset.keys(),
]);
const rows = [
  ...renderableAssets.map((asset) => readinessRow(asset, routeByAsset, qualityByAsset)),
  ...[...routeOnlyIds]
    .filter((assetId) => !renderableAssets.some((asset) => asset.id === assetId))
    .map((assetId) =>
      readinessRow(
        { id: assetId, name: assetId, sourceType: "gaussian", license: "", localPath: "", splatPath: "" },
        routeByAsset,
        qualityByAsset,
      ),
    ),
].sort(compareRows);

const summary = {
  mode: MODE,
  generatedAt: new Date().toISOString(),
  outputDir,
  inputs: {
    routeSummary: routeSummary.path,
    qualitySummary: qualitySummary.path,
  },
  requirements: {
    requireRouteEvidence,
    requirePublicCommercial,
  },
  passed: true,
  gates: {},
  rows,
};
summary.gates = buildGates(rows);
summary.passed =
  (!requireRouteEvidence || summary.gates.routeReadyRows > 0) &&
  (!requirePublicCommercial || summary.gates.publicCommercialCandidateRows > 0);

writeReport(outputDir, summary);
for (const row of rows) {
  console.log(
    `commercial_demo_asset=${row.productDemoTier === "research-diagnostic" ? "diagnostic" : "qa"} ` +
      `asset=${JSON.stringify(row.assetId)} tier=${JSON.stringify(row.productDemoTier)} ` +
      `route=${JSON.stringify(row.routeStatus)}:${JSON.stringify(row.routeKind)} ` +
      `quality=${JSON.stringify(row.qualityInterpretation)} ` +
      `license=${JSON.stringify(row.licenseScope)} ` +
      `public=${JSON.stringify(row.publicCommercialEligibility)} ` +
      `screenshot=${JSON.stringify(row.screenshotStatus)}:${JSON.stringify(row.screenshotPath || "")}`,
  );
}
console.log(
  `commercial_demo_readiness=${summary.passed ? "passed" : "failed"} ` +
    `mode=${JSON.stringify(MODE)} rows=${rows.length} ` +
    `routeReady=${summary.gates.routeReadyRows} ` +
    `showcaseCaveated=${summary.gates.showcaseRouteCaveatedRows} ` +
    `publicCommercial=${summary.gates.publicCommercialCandidateRows} ` +
    `report=${JSON.stringify(path.join(outputDir, "summary.md"))}`,
);

if (!summary.passed) process.exitCode = 1;

function readinessRow(asset, routeByAssetMap, qualityByAssetMap) {
  const route = routeByAssetMap.get(asset.id) ?? null;
  const quality = qualityByAssetMap.get(asset.id) ?? null;
  const licenseScope = classifyLicense(asset);
  const productDemo = classifyProductDemo({ route, quality });
  const publicCommercialEligibility =
    licenseScope === "commercial-license-clean" &&
    ["showcase-default", "showcase-route-caveated"].includes(productDemo.tier)
      ? "public-commercial-candidate"
      : "not-cleared-for-public-commercial-demo";
  const screenshotPath = route?.screenshot || quality?.screenshotPath || "";

  return {
    assetId: asset.id,
    name: asset.name ?? asset.id,
    sourceName: asset.sourceName ?? "",
    license: asset.license ?? "",
    licenseScope,
    productDemoTier: productDemo.tier,
    productDemoLabel: productDemo.label,
    requiredCopy: productDemo.requiredCopy,
    publicCommercialEligibility,
    routeStatus: route ? "present" : "missing",
    routeKind: route?.kind ?? "",
    routeSource: route?.source ?? "",
    routeBoundary: route?.boundary ?? "",
    visibleGaussians: route?.visibleGaussians ?? null,
    baseGaussians: route?.baseGaussians ?? null,
    qualityStatus: quality ? quality.residualStatus : "missing",
    qualityInterpretation: quality?.interpretation ?? "hard-mask-quality-unmeasured",
    hardMaskGapScore: quality?.hardMaskGapScore ?? null,
    residualCoverageRatio: quality?.residualCoverageRatio ?? null,
    screenshotPath,
    screenshotStatus: screenshotPath ? (existsSync(screenshotPath) ? "present" : "missing-file") : "missing",
    notes: productDemo.notes,
  };
}

function classifyProductDemo({ route, quality }) {
  if (!route) {
    return {
      tier: "qa-pending",
      label: "待 route QA",
      requiredCopy: "需要先跑 Spark route gate",
      notes: ["No Spark commercial route evidence was found for this asset."],
    };
  }
  const interpretation = quality?.interpretation ?? "hard-mask-quality-unmeasured";
  const residualStatus = quality?.residualStatus ?? "missing";
  if (interpretation === "low-hard-mask-risk" && residualStatus === "passed") {
    return {
      tier: "showcase-default",
      label: "商业展示默认路线",
      requiredCopy: "可展示 Spark source/original object edit route",
      notes: ["Route evidence and hard-mask quality are both clean enough for default product route review."],
    };
  }
  if (interpretation === "boundary-mixing-dominant" && residualStatus === "passed") {
    return {
      tier: "showcase-route-caveated",
      label: "商业展示路线可演示",
      requiredCopy: "必须同时显示“对象 mask，无补洞 / 边界混合主导”",
      notes: [
        "The Spark route is available, but hard object boundaries are the expected source of delete-preview grain.",
      ],
    };
  }
  if (interpretation === "coverage-hole-risk") {
    return {
      tier: "showcase-route-caveated",
      label: "商业展示路线需谨慎",
      requiredCopy: "必须标注 coverage hole risk，避免展示为最终删除效果",
      notes: ["The deleted object owns enough unique projected coverage that hard masking can create holes."],
    };
  }
  if (interpretation === "browser-residual-dominant" || residualStatus === "failed") {
    return {
      tier: "research-diagnostic",
      label: "研究 / 诊断样例",
      requiredCopy: "不得标成商业展示默认效果",
      notes: [
        "Spark reconstruction or source mismatch dominates the visual residual; use for renderer diagnosis or research evidence.",
      ],
    };
  }
  return {
    tier: "qa-pending",
    label: "质量证据待补",
    requiredCopy: "需要补 hard-mask quality chain",
    notes: ["Route exists, but hard-mask quality evidence is missing or incomplete."],
  };
}

function classifyLicense(asset) {
  const license = String(asset.license ?? "").toLowerCase();
  if (license.includes("cc0")) return "commercial-license-clean";
  if (license.includes("仅用于本地测试")) return "local-test-only";
  if (license.includes("研究") || license.includes("nerf")) return "research-only";
  return "license-review-required";
}

function buildRouteIndex(routeSummary) {
  const map = new Map();
  for (const route of routeSummary?.routes?.native ?? []) {
    map.set(route.asset, {
      kind: "native-no-sh",
      source: route.source,
      boundary: route.boundary,
      visibleGaussians: Number(route.visibleGaussians),
      baseGaussians: Number(route.baseGaussians),
      screenshot: route.screenshot ?? "",
    });
  }
  for (const route of routeSummary?.routes?.trained ?? []) {
    map.set(route.asset, {
      kind: "trained-sh-heavy",
      source: route.spark?.join("/") ?? "",
      boundary: route.delete?.join("/") ?? "",
      visibleGaussians: Number(route.visibleGaussians),
      baseGaussians: Number(route.baseGaussians),
      screenshot: route.screenshot ?? "",
    });
  }
  return map;
}

function buildQualityIndex(qualitySummary) {
  const map = new Map();
  for (const row of qualitySummary?.rows ?? []) {
    map.set(row.assetId, {
      interpretation: row.interpretation,
      residualStatus: row.residualStatus,
      routeKind: row.routeKind,
      hardMaskGapScore: row.boundary?.hardMaskGapScore ?? null,
      residualCoverageRatio: row.residual?.coverageRatio ?? null,
      screenshotPath: row.residual?.screenshotPath ?? "",
    });
  }
  return map;
}

function buildGates(rows) {
  const routeReadyRows = rows.filter((row) => row.routeStatus === "present").length;
  const showcaseDefaultRows = rows.filter((row) => row.productDemoTier === "showcase-default").length;
  const showcaseRouteCaveatedRows = rows.filter((row) => row.productDemoTier === "showcase-route-caveated").length;
  const researchDiagnosticRows = rows.filter((row) => row.productDemoTier === "research-diagnostic").length;
  const publicCommercialCandidateRows = rows.filter(
    (row) => row.publicCommercialEligibility === "public-commercial-candidate",
  ).length;
  return {
    routeReadyRows,
    showcaseDefaultRows,
    showcaseRouteCaveatedRows,
    researchDiagnosticRows,
    publicCommercialCandidateRows,
    pendingRows: rows.length - routeReadyRows,
  };
}

function compareRows(a, b) {
  const rank = {
    "showcase-default": 0,
    "showcase-route-caveated": 1,
    "research-diagnostic": 2,
    "qa-pending": 3,
  };
  return (
    (rank[a.productDemoTier] ?? 9) - (rank[b.productDemoTier] ?? 9) ||
    a.assetId.localeCompare(b.assetId)
  );
}

function writeReport(outputDirPath, summary) {
  mkdirSync(outputDirPath, { recursive: true });
  writeFileSync(path.join(outputDirPath, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  writeFileSync(path.join(outputDirPath, "summary.md"), renderMarkdown(summary));
}

function renderMarkdown(summary) {
  const lines = [
    "# Commercial Demo Readiness",
    "",
    `- Status: ${summary.passed ? "passed" : "failed"}`,
    `- Mode: ${summary.mode}`,
    `- Generated: ${summary.generatedAt}`,
    `- Spark route summary: ${summary.inputs.routeSummary}`,
    `- Hard-mask quality summary: ${summary.inputs.qualitySummary}`,
    "",
    "This report separates product-route readiness from public-commercial licensing. A sample can pass the Spark commercial route gate and still be unsuitable as a public commercial demo asset if the source license is local-test-only or research-only.",
    "",
    "## Gates",
    "",
    `- Route-ready rows: ${summary.gates.routeReadyRows}`,
    `- Showcase default rows: ${summary.gates.showcaseDefaultRows}`,
    `- Showcase caveated rows: ${summary.gates.showcaseRouteCaveatedRows}`,
    `- Research / diagnostic rows: ${summary.gates.researchDiagnosticRows}`,
    `- Public commercial candidates: ${summary.gates.publicCommercialCandidateRows}`,
    `- Pending rows: ${summary.gates.pendingRows}`,
    "",
    "## Sample Readiness",
    "",
    "| Asset | Route tier | Quality | Route | License scope | Public commercial | Required copy | Screenshot |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |",
  ];
  for (const row of summary.rows) {
    lines.push(
      `| ${escapeMarkdown(row.assetId)} | ${escapeMarkdown(row.productDemoLabel)} | ` +
        `${escapeMarkdown(row.qualityInterpretation)} | ${escapeMarkdown(row.routeStatus + "/" + (row.routeKind || "none"))} | ` +
        `${escapeMarkdown(row.licenseScope)} | ${escapeMarkdown(row.publicCommercialEligibility)} | ` +
        `${escapeMarkdown(row.requiredCopy)} | ${escapeMarkdown(row.screenshotPath || row.screenshotStatus)} |`,
    );
  }
  lines.push(
    "",
    "## Decisions",
    "",
    "- `商业展示默认路线` is reserved for rows with clean route evidence and low hard-mask risk.",
    "- `商业展示路线可演示` means the product renderer route is usable, but the UI must show the hard-mask boundary explanation.",
    "- `研究 / 诊断样例` must not be presented as the commercial default visual result.",
    "- `public-commercial-candidate` also requires a clean source license; current research/local samples do not become license-clean just because the renderer route passes.",
    "",
  );
  return lines.join("\n");
}

function firstExistingJson(paths) {
  for (const summaryPath of paths) {
    if (!existsSync(summaryPath)) continue;
    return { path: summaryPath, payload: readJson(summaryPath) };
  }
  return null;
}

function readJson(summaryPath) {
  return JSON.parse(readFileSync(summaryPath, "utf8"));
}

function pathList(value, defaults) {
  if (value === undefined || value === null || value === false) return defaults;
  if (value === true) throw new Error("summary path flag requires a comma-separated path list");
  return String(value)
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const entry = argv[index];
    if (!entry.startsWith("--")) continue;
    const key = entry.slice(2);
    const next = argv[index + 1];
    if (!next || next.startsWith("--")) {
      parsed[key] = true;
    } else {
      parsed[key] = next;
      index += 1;
    }
  }
  return parsed;
}

function flagEnabled(value) {
  if (value === true) return true;
  if (value === undefined || value === null || value === false) return false;
  return ["1", "true", "yes", "on"].includes(String(value).toLowerCase());
}

function escapeMarkdown(value) {
  return String(value ?? "").replace(/\|/g, "\\|");
}
