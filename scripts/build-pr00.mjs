import { readFile, mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";
import { canonicalStringify } from "../src/pr00/canonical-json.mjs";
import { createEpisodeValidator, validateEpisodeOrThrow } from "../src/pr00/contract-validator.mjs";
import { projectEpisodePoint } from "../src/pr00/projector.mjs";
import { evaluateReprojection } from "../src/pr00/reprojection-evaluator.mjs";
import { auditResources } from "../src/pr00/resource-audit.mjs";
import { createSyntheticAudit } from "../src/pr00/synthetic-audit.mjs";
import { sha256Hex } from "../src/pr00/node-hash.mjs";
import { createVerdictReport } from "../src/pr00/verdict.mjs";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SCHEMA_PATH = resolve(ROOT, "contracts/objgauss/0.1.0/episode.schema.json");
const MANIFEST_PATH = resolve(ROOT, "contracts/fixtures/synthetic-audit-v0.manifest.json");
const OUTPUT = resolve(ROOT, "generated/pr00");

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label} mismatch: expected ${expected}, got ${actual}`);
  }
}

const [schemaBytes, manifestBytes] = await Promise.all([
  readFile(SCHEMA_PATH),
  readFile(MANIFEST_PATH),
]);
const schema = JSON.parse(schemaBytes.toString("utf8"));
const manifest = JSON.parse(manifestBytes.toString("utf8"));
const { episode, resources } = createSyntheticAudit();
const episodeBytes = Buffer.from(canonicalStringify(episode));

assertEqual(sha256Hex(schemaBytes), manifest.schema.sha256, "schema SHA-256");
assertEqual(episode.producer.config_sha256, manifest.producer.config_sha256, "producer config SHA-256");
assertEqual(episode.producer.seed, manifest.producer.seed, "producer seed");
assertEqual(sha256Hex(episodeBytes), manifest.episode.sha256, "episode SHA-256");
const validation = validateEpisodeOrThrow(createEpisodeValidator(schema), episode);
const resourceAudit = auditResources({ manifest, resources });
if (resourceAudit.status !== "valid") {
  throw new Error(`resource audit failed: ${JSON.stringify(resourceAudit.failures)}`);
}
const reprojection = evaluateReprojection({ episode, project: projectEpisodePoint });
if (reprojection.status !== "supported") {
  throw new Error(`primary endpoint ${reprojection.status}: ${reprojection.reason}`);
}
const report = createVerdictReport({
  schemaSha256: manifest.schema.sha256,
  fixtureManifestSha256: sha256Hex(manifestBytes),
  episodeSha256: manifest.episode.sha256,
  validation,
  resourceAudit,
  reprojection,
});

await mkdir(resolve(OUTPUT, "assets"), { recursive: true });
await Promise.all(resources.map(async ({ uri, bytes }) => {
  const path = resolve(OUTPUT, uri);
  if (!path.startsWith(`${OUTPUT}/`)) {
    throw new Error(`unsafe generated path: ${uri}`);
  }
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, bytes);
}));
await Promise.all([
  writeFile(resolve(OUTPUT, "schema.json"), schemaBytes),
  writeFile(resolve(OUTPUT, "manifest.json"), manifestBytes),
  writeFile(resolve(OUTPUT, "episode.json"), episodeBytes),
  writeFile(resolve(OUTPUT, "report.json"), canonicalStringify(report)),
]);
await build({
  entryPoints: [resolve(ROOT, "src/pr00/browser-consumer.mjs")],
  outfile: resolve(OUTPUT, "contract-consumer.mjs"),
  bundle: true,
  format: "esm",
  platform: "browser",
  target: ["es2022"],
  sourcemap: false,
  legalComments: "none",
});

console.log(JSON.stringify({
  fixture_id: episode.fixture_id,
  episode_sha256: manifest.episode.sha256,
  resources: resourceAudit.verified_resource_count,
  primary_endpoint: reprojection,
  verdict: report.verdict,
}, null, 2));
