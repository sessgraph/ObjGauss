import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { canonicalStringify } from "../src/pr00/canonical-json.mjs";
import { createEpisodeValidator } from "../src/pr00/contract-validator.mjs";
import { sha256Hex } from "../src/pr00/node-hash.mjs";
import { auditResources } from "../src/pr00/resource-audit.mjs";
import { isSafeResourceUri } from "../src/pr00/resource-uri.mjs";
import { createSyntheticAudit, SYNTHETIC_AUDIT_SEED } from "../src/pr00/synthetic-audit.mjs";

const schemaBytes = readFileSync("contracts/objgauss/0.1.0/episode.schema.json");
const schema = JSON.parse(schemaBytes);
const manifest = JSON.parse(readFileSync("contracts/fixtures/synthetic-audit-v0.manifest.json"));

test("JSON Schema 2020-12 is the only frozen 0.1.0 contract source", () => {
  assert.equal(schema.$schema, "https://json-schema.org/draft/2020-12/schema");
  assert.equal(schema.$id, "https://github.com/sessgraph/ObjGauss/contracts/objgauss/0.1.0/episode.schema.json");
  assert.equal(schema.properties.schema_version.const, "0.1.0");
  assert.equal(schema.unevaluatedProperties, false);
  assert.equal(sha256Hex(schemaBytes), manifest.schema.sha256);
});

test("synthetic-audit-v0 is deterministic, schema-valid, and frozen by checksum", () => {
  const first = createSyntheticAudit();
  const second = createSyntheticAudit();
  assert.equal(SYNTHETIC_AUDIT_SEED, 0x50_52_30_30);
  assert.equal(canonicalStringify(first.episode), canonicalStringify(second.episode));
  assert.equal(sha256Hex(canonicalStringify(first.episode)), manifest.episode.sha256);
  assert.equal(first.episode.producer.config_sha256, manifest.producer.config_sha256);
  const validation = createEpisodeValidator(schema)(first.episode);
  assert.deepEqual(validation.schema_errors, []);
  assert.deepEqual(validation.semantic_errors, []);
  assert.equal(validation.valid, true);
  assert.equal(first.episode.audit.primary_points.length, 36);
});

test("all derived resources match the committed checksum manifest", () => {
  const { resources } = createSyntheticAudit();
  const audit = auditResources({ manifest, resources });
  assert.equal(audit.status, "valid");
  assert.equal(audit.expected_resource_count, 12);
  assert.equal(audit.verified_resource_count, 12);
  assert.deepEqual(audit.failures, []);
});

test("resource audit rejects unsafe and duplicate URIs before they can alias", () => {
  const { resources } = createSyntheticAudit();
  const unsafeManifest = structuredClone(manifest);
  unsafeManifest.resources[0].uri = "assets/%2e%2e/episode.json";
  assert.equal(isSafeResourceUri(unsafeManifest.resources[0].uri), false);
  assert.equal(auditResources({ manifest: unsafeManifest, resources }).status, "invalid");

  const duplicateManifest = structuredClone(manifest);
  duplicateManifest.resources.push(structuredClone(duplicateManifest.resources[0]));
  const audit = auditResources({ manifest: duplicateManifest, resources });
  assert.equal(audit.status, "invalid");
  assert.ok(audit.failures.some((failure) => failure.reason === "duplicate-manifest-resource"));
});

test("fixture keeps commanded, executed, hold, and missing action semantics separate", () => {
  const { episode } = createSyntheticAudit();
  const [hold, push] = episode.interventions;
  assert.equal(hold.commanded_action.value.kind, "hold");
  assert.equal(hold.executed_action.value.kind, "hold");
  assert.deepEqual(hold.commanded_action.value.vector_W_N, [0, 0, 0]);
  assert.equal(push.commanded_action.value.kind, "push");
  assert.ok(Math.hypot(...push.commanded_action.value.vector_W_N) > 0);
  assert.deepEqual(push.executed_action, { availability: "missing", reason: "not_measured" });
});

test("canonical frames and symmetry are producer-authored rather than inferred", () => {
  const { episode } = createSyntheticAudit();
  for (const observation of episode.observations) {
    for (const object of observation.objects) {
      assert.deepEqual(object.canonical_frame, {
        origin: "center_of_mass",
        axes_source: "producer_authored",
        handedness: "right",
      });
      assert.equal(object.symmetry.availability, "present");
    }
  }
  assert.equal(episode.observations[0].objects[0].symmetry.value.kind, "none");
  assert.deepEqual(episode.observations[0].objects[1].symmetry.value, {
    kind: "continuous_axis",
    axis_O: [0, 0, 1],
  });
});
