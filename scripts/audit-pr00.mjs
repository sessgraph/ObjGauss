import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalStringify } from "../src/pr00/canonical-json.mjs";
import { createEpisodeValidator, validateEpisodeOrThrow } from "../src/pr00/contract-validator.mjs";
import { projectEpisodePoint } from "../src/pr00/projector.mjs";
import { evaluateReprojection } from "../src/pr00/reprojection-evaluator.mjs";
import { auditResources } from "../src/pr00/resource-audit.mjs";
import { sha256Hex } from "../src/pr00/node-hash.mjs";
import { createVerdictReport } from "../src/pr00/verdict.mjs";
import { assertSafeResourceUri } from "../src/pr00/resource-uri.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUTPUT = resolve(ROOT, "generated/pr00");

const [schemaBytes, manifestBytes, episodeBytes, reportBytes] = await Promise.all([
  readFile(resolve(OUTPUT, "schema.json")),
  readFile(resolve(OUTPUT, "manifest.json")),
  readFile(resolve(OUTPUT, "episode.json")),
  readFile(resolve(OUTPUT, "report.json")),
]);
const schema = JSON.parse(schemaBytes.toString("utf8"));
const manifest = JSON.parse(manifestBytes.toString("utf8"));
const episode = JSON.parse(episodeBytes.toString("utf8"));
const report = JSON.parse(reportBytes.toString("utf8"));
if (sha256Hex(schemaBytes) !== manifest.schema.sha256) {
  throw new Error("generated schema checksum does not match frozen manifest");
}
if (sha256Hex(episodeBytes) !== manifest.episode.sha256) {
  throw new Error("generated episode checksum does not match frozen manifest");
}
const resources = await Promise.all(manifest.resources.map(async (resource) => ({
  uri: resource.uri,
  bytes: await readFile(resolve(OUTPUT, assertSafeResourceUri(resource.uri))),
})));
const validation = validateEpisodeOrThrow(createEpisodeValidator(schema), episode);
const resourceAudit = auditResources({ manifest, resources });
const reprojection = evaluateReprojection({ episode, project: projectEpisodePoint });
const expectedReport = createVerdictReport({
  schemaSha256: manifest.schema.sha256,
  fixtureManifestSha256: sha256Hex(manifestBytes),
  episodeSha256: manifest.episode.sha256,
  validation,
  resourceAudit,
  reprojection,
});
if (canonicalStringify(report) !== canonicalStringify(expectedReport)) {
  throw new Error("generated report does not match independently recomputed report");
}
if (report.verdict !== "supported") {
  throw new Error(`PR-00 verdict is ${report.verdict}`);
}
console.log(JSON.stringify({
  status: "valid",
  fixture_id: episode.fixture_id,
  resource_count: resources.length,
  verdict: report.verdict,
  max_camera_reprojection_error_px: reprojection.max_error_px,
}, null, 2));
