import { createEpisodeValidator } from "./contract-validator.mjs";
import { isSafeResourceUri } from "./resource-uri.mjs";

async function fetchBytes(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`PR-00 resource fetch failed: ${response.status} ${url}`);
  }
  return response.arrayBuffer();
}

async function fetchJson(url) {
  const bytes = await fetchBytes(url);
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  return { bytes, value: JSON.parse(text) };
}

async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) {
    throw new Error("Web Crypto is required for PR-00 checksum validation");
  }
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected ${expected}, got ${actual}`);
  }
}

export async function loadPr00Contract() {
  const baseUrl = new URL("./", import.meta.url);
  const [manifestResult, schemaResult, episodeResult, reportResult] = await Promise.all([
    fetchJson(new URL("manifest.json", baseUrl)),
    fetchJson(new URL("schema.json", baseUrl)),
    fetchJson(new URL("episode.json", baseUrl)),
    fetchJson(new URL("report.json", baseUrl)),
  ]);
  const { value: manifest } = manifestResult;
  const { value: schema } = schemaResult;
  const { value: episode } = episodeResult;
  const { value: report } = reportResult;
  assertEqual(await sha256Hex(schemaResult.bytes), manifest.schema.sha256, "schema SHA-256");
  assertEqual(await sha256Hex(episodeResult.bytes), manifest.episode.sha256, "episode SHA-256");
  const validation = createEpisodeValidator(schema)(episode);
  if (!validation.valid) {
    const errors = [...validation.schema_errors, ...validation.semantic_errors]
      .map((error) => `${error.path} [${error.category}] ${error.message}`)
      .join("; ");
    throw new Error(`PR-00 browser contract validation failed: ${errors}`);
  }

  const expectedResources = new Map();
  for (const expected of manifest.resources) {
    if (expectedResources.has(expected.uri)) {
      throw new Error(`duplicate PR-00 manifest resource URI: ${expected.uri}`);
    }
    expectedResources.set(expected.uri, `${expected.sha256}:${expected.bytes}`);
  }
  assertEqual(
    validation.resource_descriptors.length,
    expectedResources.size,
    "episode/manifest resource count",
  );
  for (const descriptor of validation.resource_descriptors) {
    assertEqual(
      descriptor.signature,
      expectedResources.get(descriptor.uri),
      `${descriptor.uri} episode/manifest descriptor`,
    );
  }

  const resources = new Map();
  for (const expected of manifest.resources) {
    if (!isSafeResourceUri(expected.uri)) {
      throw new Error(`unsafe PR-00 resource URI: ${expected.uri}`);
    }
    const bytes = await fetchBytes(new URL(expected.uri, baseUrl));
    assertEqual(bytes.byteLength, expected.bytes, `${expected.uri} byte length`);
    assertEqual(await sha256Hex(bytes), expected.sha256, `${expected.uri} SHA-256`);
    resources.set(expected.uri, bytes);
  }
  if (report.verdict !== "supported") {
    throw new Error(`PR-00 machine verdict is ${report.verdict}: ${report.reason}`);
  }
  assertEqual(
    report.inputs.fixture_manifest_sha256,
    await sha256Hex(manifestResult.bytes),
    "report fixture manifest SHA-256",
  );
  assertEqual(report.inputs.schema_sha256, manifest.schema.sha256, "report schema SHA-256");
  assertEqual(report.inputs.episode_sha256, manifest.episode.sha256, "report episode SHA-256");
  assertEqual(report.checks.schema_and_semantics, "pass", "report schema and semantic status");
  assertEqual(report.checks.resource_checksums, "valid", "report resource checksum status");
  assertEqual(report.checks.primary_endpoint.endpoint, "max_camera_reprojection_error_px", "report endpoint");
  if (!(report.checks.primary_endpoint.max_error_px < report.checks.primary_endpoint.threshold_exclusive_px)) {
    throw new Error("PR-00 report does not satisfy its frozen exclusive threshold");
  }
  return { manifest, schema, episode, report, resources, validation };
}
