export function isSafeResourceUri(uri) {
  if (typeof uri !== "string" || uri.length === 0 || uri.startsWith("/")) {
    return false;
  }
  if (!/^[A-Za-z0-9._/-]+$/.test(uri)) {
    return false;
  }
  const segments = uri.split("/");
  return segments.every((segment) => segment !== "" && segment !== "." && segment !== "..");
}

export function assertSafeResourceUri(uri) {
  if (!isSafeResourceUri(uri)) {
    throw new Error(`unsafe PR-00 resource URI: ${String(uri)}`);
  }
  return uri;
}
