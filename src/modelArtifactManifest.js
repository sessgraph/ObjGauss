export const MODEL_ARTIFACT_MANIFEST_SCHEMA = "objgauss-model-artifact-manifest-v1";

const BROWSER_READY_TIERS = new Set(["browser_quick", "browser_edit"]);
const LARGE_SIZE_PATTERN = /\b(?:\d+(?:\.\d+)?)\s*GB\b/i;

export function withModelArtifactManifest(asset) {
  if (!canBuildModelArtifactManifest(asset)) {
    return {
      ...asset,
      modelArtifactManifest: null,
      modelArtifactRoutes: emptyModelArtifactRoutes(),
    };
  }
  const manifest = asset.modelArtifactManifest ?? manifestFromAssetLibraryEntry(asset);
  return {
    ...asset,
    modelArtifactManifest: manifest,
    modelArtifactRoutes: resolveModelArtifactRoutes(manifest),
  };
}

export function manifestFromAssetLibraryEntry(asset) {
  const artifacts = [];
  if (asset.splatPath) {
    artifacts.push(
      compactObject({
        role: "quick_splat",
        path: asset.splatPath,
        format: ".splat",
        delivery_tier: "browser_quick",
        browser_ready: true,
        gaussian_count: positiveInteger(asset.gaussianCount),
        byte_size: sizeLabelToBytes(asset.splatSizeLabel),
        label: asset.splatFileName,
      }),
    );
  }
  if (asset.localPath) {
    const route = objectArtifactRoute(asset);
    artifacts.push(
      compactObject({
        role: route.role,
        path: asset.localPath,
        format: extensionFromPath(asset.localPath, ".ply"),
        delivery_tier: route.deliveryTier,
        browser_ready: route.browserReady,
        gaussian_count: positiveInteger(asset.objectPlyGaussianCount ?? asset.gaussianCount),
        object_count: positiveInteger(asset.objectCount),
        byte_size: sizeLabelToBytes(asset.objectPlySizeLabel),
        label: asset.fileName,
        note: route.role === "diagnostic_full"
          ? "Deferred or large object PLY from asset library; not a default browser artifact."
          : undefined,
      }),
    );
  }
  return {
    schema: MODEL_ARTIFACT_MANIFEST_SCHEMA,
    manifest_id: `${asset.id}-asset-model-artifacts`,
    asset_id: asset.id,
    name: asset.name,
    stage: "development",
    source: compactObject({
      type: "asset_library_entry",
      source_type: asset.sourceType,
      source_name: asset.sourceName,
      source_url: asset.sourceUrl,
      pipeline_stage: asset.pipelineStage,
    }),
    license: asset.license,
    counts: {
      gaussians: positiveInteger(asset.objectPlyGaussianCount ?? asset.gaussianCount),
      objects: positiveInteger(asset.objectCount),
    },
    artifacts,
    quality_evidence: [],
    limitations: [
      ...(asset.deferObjectPly ? ["Object PLY is deferred and must not be requested by quick-view routes."] : []),
      ...(asset.objectPlySizeLabel ? [`Object PLY size label: ${asset.objectPlySizeLabel}`] : []),
    ],
    created_from: {
      asset_library_entry: asset.id,
      category: asset.category,
      status: asset.status,
      use_cases: asset.useCases ?? [],
    },
  };
}

export function resolveModelArtifactRoutes(manifest) {
  if (!manifest?.artifacts) return emptyModelArtifactRoutes();
  const artifacts = Array.isArray(manifest.artifacts) ? manifest.artifacts : [];
  const quickSplat = artifacts.find(
    (artifact) =>
      artifact.role === "quick_splat" &&
      artifact.delivery_tier === "browser_quick" &&
      artifact.browser_ready === true,
  ) ?? null;
  const objectEdit = artifacts.find(
    (artifact) =>
      artifact.role === "object_edit" &&
      artifact.delivery_tier === "browser_edit" &&
      artifact.browser_ready === true,
  ) ?? null;
  const diagnosticFull = artifacts.find(
    (artifact) =>
      artifact.role === "diagnostic_full" &&
      artifact.browser_ready !== true,
  ) ?? null;
  const compressedChunked = artifacts.find(
    (artifact) =>
      artifact.role === "compressed_chunked" &&
      artifact.delivery_tier === "browser_edit" &&
      artifact.browser_ready === true,
  ) ?? null;
  const trainableKernel = artifacts.find(
    (artifact) =>
      artifact.role === "trainable_kernel" &&
      artifact.delivery_tier === "browser_edit" &&
      artifact.browser_ready === true,
  ) ?? null;
  const qualityReport = artifacts.find(
    (artifact) =>
      artifact.role === "quality_report" &&
      artifact.delivery_tier === "browser_edit" &&
      artifact.browser_ready === true,
  ) ?? null;
  return {
    manifestId: manifest.manifest_id ?? "",
    schema: manifest.schema ?? "",
    browserReadyCount: artifacts.filter((artifact) => artifact.browser_ready === true).length,
    quickSplat,
    objectEdit,
    trainableKernel,
    qualityReport,
    compressedChunked,
    diagnosticFull,
  };
}

export function routeArtifactTelemetry(artifact) {
  if (!artifact) {
    return {
      role: "",
      deliveryTier: "",
      browserReady: "",
      path: "",
    };
  }
  return {
    role: artifact.role ?? "",
    deliveryTier: artifact.delivery_tier ?? "",
    browserReady: String(artifact.browser_ready === true),
    path: artifact.path ?? "",
  };
}

export function browserReadyArtifact(asset, role) {
  const routes = asset?.modelArtifactRoutes ?? resolveModelArtifactRoutes(asset?.modelArtifactManifest);
  if (role === "quick_splat") return routes.quickSplat;
  if (role === "object_edit") return routes.objectEdit;
  if (role === "trainable_kernel") return routes.trainableKernel;
  if (role === "quality_report") return routes.qualityReport;
  if (role === "compressed_chunked") return routes.compressedChunked;
  return null;
}

function canBuildModelArtifactManifest(asset) {
  return Boolean(asset?.id && asset?.name && asset?.license && (asset.splatPath || asset.localPath));
}

function emptyModelArtifactRoutes() {
  return {
    manifestId: "",
    schema: "",
    browserReadyCount: 0,
    quickSplat: null,
    objectEdit: null,
    trainableKernel: null,
    qualityReport: null,
    compressedChunked: null,
    diagnosticFull: null,
  };
}

function objectArtifactRoute(asset) {
  if (asset.deferObjectPly || LARGE_SIZE_PATTERN.test(String(asset.objectPlySizeLabel ?? ""))) {
    return {
      role: "diagnostic_full",
      deliveryTier: "diagnostic",
      browserReady: false,
    };
  }
  return {
    role: "object_edit",
    deliveryTier: "browser_edit",
    browserReady: true,
  };
}

function positiveInteger(value) {
  if (value === undefined || value === null || value === "") return undefined;
  const number = Number(value);
  return Number.isInteger(number) && number >= 0 ? number : undefined;
}

function sizeLabelToBytes(label) {
  if (!label) return undefined;
  const match = String(label).trim().match(/^(\d+(?:\.\d+)?)\s*(KB|MB|GB)$/i);
  if (!match) return undefined;
  const value = Number(match[1]);
  const suffix = match[2].toUpperCase();
  const multiplier = suffix === "GB" ? 1024 ** 3 : suffix === "MB" ? 1024 ** 2 : 1024;
  return Math.floor(value * multiplier);
}

function extensionFromPath(path, fallback) {
  const match = String(path).match(/(\.[a-z0-9]+)(?:[?#].*)?$/i);
  return match?.[1] ?? fallback;
}

function compactObject(object) {
  return Object.fromEntries(Object.entries(object).filter(([, value]) => value !== undefined));
}
