import { sha256Hex } from "./node-hash.mjs";
import { isSafeResourceUri } from "./resource-uri.mjs";

export function auditResources({ manifest, resources }) {
  const failures = [];
  const expectedByUri = new Map();
  for (const resource of manifest.resources) {
    if (!isSafeResourceUri(resource.uri)) {
      failures.push({ uri: resource.uri, reason: "unsafe-manifest-resource-uri" });
      continue;
    }
    if (expectedByUri.has(resource.uri)) {
      failures.push({ uri: resource.uri, reason: "duplicate-manifest-resource" });
      continue;
    }
    expectedByUri.set(resource.uri, resource);
  }
  const actualByUri = new Map();
  for (const resource of resources) {
    if (!isSafeResourceUri(resource.uri)) {
      failures.push({ uri: resource.uri, reason: "unsafe-actual-resource-uri" });
      continue;
    }
    if (actualByUri.has(resource.uri)) {
      failures.push({ uri: resource.uri, reason: "duplicate-actual-resource" });
      continue;
    }
    actualByUri.set(resource.uri, resource.bytes);
  }
  let verifiedResourceCount = 0;
  for (const [uri, expected] of expectedByUri) {
    const bytes = actualByUri.get(uri);
    if (bytes === undefined) {
      failures.push({ uri, reason: "missing-resource" });
      continue;
    }
    if (bytes.byteLength !== expected.bytes) {
      failures.push({ uri, reason: "byte-length-mismatch", expected: expected.bytes, actual: bytes.byteLength });
      continue;
    }
    const actualSha256 = sha256Hex(bytes);
    if (actualSha256 !== expected.sha256) {
      failures.push({ uri, reason: "sha256-mismatch", expected: expected.sha256, actual: actualSha256 });
      continue;
    }
    verifiedResourceCount += 1;
  }
  for (const uri of actualByUri.keys()) {
    if (!expectedByUri.has(uri)) {
      failures.push({ uri, reason: "unexpected-resource" });
    }
  }
  return {
    status: failures.length === 0 ? "valid" : "invalid",
    expected_resource_count: expectedByUri.size,
    verified_resource_count: verifiedResourceCount,
    failures,
  };
}
