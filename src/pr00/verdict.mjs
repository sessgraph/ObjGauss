export const CLAIM_POLICY = Object.freeze({
  allowed: [
    "synthetic-audit-v0 schema and semantic contract validation",
    "Robotics/OpenCV frame-chain correctness on frozen synthetic primary points",
    "independent max camera reprojection endpoint below the frozen threshold",
  ],
  denied: [
    "real-data validity",
    "Gaussian reconstruction quality",
    "world-model capability",
    "dynamics prediction",
    "planning value",
  ],
});

export function createVerdictReport({
  schemaSha256,
  fixtureManifestSha256,
  episodeSha256,
  validation,
  resourceAudit,
  reprojection,
}) {
  let verdict = "invalid";
  let reason = "validation-or-audit-invalid";
  if (validation.valid && resourceAudit.status === "valid") {
    verdict = reprojection.status;
    reason = reprojection.reason;
  }
  return {
    report_version: "0.1.0",
    fixture_id: "synthetic-audit-v0",
    schema_version: "0.1.0",
    inputs: {
      schema_sha256: schemaSha256,
      fixture_manifest_sha256: fixtureManifestSha256,
      episode_sha256: episodeSha256,
    },
    checks: {
      schema_and_semantics: validation.valid ? "pass" : "invalid",
      resource_checksums: resourceAudit.status,
      primary_endpoint: reprojection,
    },
    verdict,
    reason,
    claims: CLAIM_POLICY,
  };
}
