#!/usr/bin/env bash
set -euo pipefail

readonly COMMIT="1267e2135660e1f4197f94c045453fe40c209b0e"
readonly URL="https://raw.githubusercontent.com/GitHubDragonFly/GitHubDragonFly.github.io/${COMMIT}/viewers/examples/legobrick.splat"
readonly DESTINATION="data/local-preview/legobrick-1267e213/legobrick.splat"
readonly EXPECTED_BYTES="3297920"
readonly EXPECTED_SHA256="d5131a664a12a8764da70552c85f567d276313110f63f1efd48424845917899e"

verify_asset() {
  local path="$1"
  local actual_bytes
  local actual_sha256
  actual_bytes="$(stat -c '%s' "$path")"
  actual_sha256="$(sha256sum "$path" | cut -d ' ' -f 1)"
  if [[ "$actual_bytes" != "$EXPECTED_BYTES" ]]; then
    echo "size mismatch: expected ${EXPECTED_BYTES}, got ${actual_bytes}" >&2
    return 1
  fi
  if [[ "$actual_sha256" != "$EXPECTED_SHA256" ]]; then
    echo "SHA-256 mismatch: expected ${EXPECTED_SHA256}, got ${actual_sha256}" >&2
    return 1
  fi
}

if [[ -f "$DESTINATION" ]] && verify_asset "$DESTINATION"; then
  echo "verified existing ignored preview: ${DESTINATION}"
  exit 0
fi

mkdir -p "$(dirname "$DESTINATION")"
temporary_file="${DESTINATION}.download"
trap 'rm -f "$temporary_file"' EXIT

curl --fail --location --retry 3 --output "$temporary_file" "$URL"
verify_asset "$temporary_file"
mv "$temporary_file" "$DESTINATION"
trap - EXIT

echo "downloaded and verified: ${DESTINATION}"
echo "asset provenance remains unverified; use only for the local rendering preview"
