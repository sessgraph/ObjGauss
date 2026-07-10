#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_DIR="${REPO_ROOT}/outputs/assets/raw/rbo-articulated-objects"
MODE="list"
TIER="p0"

RBO_RECORD_ID="1036660"
RBO_BASE_URL="https://zenodo.org/api/records/${RBO_RECORD_ID}/files"
RBO_INDEX_URL="${RBO_BASE_URL}/interactions_index.csv/content"
RBO_INDEX_SIZE=23134
RBO_INDEX_MD5="dbfe883bf2e1ee9ade51dbd736ac5713"

# P0 maximizes the index-level proxies for a strict clear -> occluded -> clear
# event while preserving camera motion and measured F/T data. These properties
# are acquisition filters only; they do not prove the pixel-level visibility gate.
P0_FILES=(
  "treasurebox25_o.tar.gz"
  "laptop26_o.tar.gz"
  "globe25_o.tar.gz"
  "treasurebox.tar.gz"
  "laptop.tar.gz"
  "globe.tar.gz"
  "ftSensor.tar.gz"
)
P0_SIZES=(
  386645684
  492820903
  201433620
  15294186
  1285300
  2101575
  66502
)
P0_MD5S=(
  "2797319123f3293790bdb277cf2a3ed1"
  "c246fed9294383925cc21ad674798794"
  "37c65bcec9f673a95075db5aed7f8029"
  "a7c7f1943755878778f4bfe14e00d40d"
  "89230a0d2bfbe0eadf656a05678ae2be"
  "de73afa6bf391407f00bdf46e9567b52"
  "9fa0ab792afc2ae9b756d73a18681dff"
)

# P1 is downloaded only with --tier all. The treasurebox model is already in
# P0. tripod.tar is also listed so this script remains self-checking; the
# initial RBO evidence subset normally provides that already-local model.
P1_FILES=(
  "treasurebox24_o.tar.gz"
  "tripod24_o.tar.gz"
  "pliers24_o.tar.gz"
  "ikeasmall23_o.tar.gz"
  "tripod.tar"
  "pliers.tar.gz"
  "ikeasmall.tar.gz"
)
P1_SIZES=(
  224874604
  250515083
  280555056
  191911421
  2669316
  1088816
  4325597
)
P1_MD5S=(
  "3931e779657c3cd333b23a1f82035951"
  "83b58fec05f8b8178d725df8676a8e5b"
  "47569b7b1eb73a4870e5ed284bb41209"
  "5fc02c6a599e71e3f5177b64108c1c14"
  "0b27993a32296e95e959e79244e37c44"
  "423b8690d637777a839573bd2114a39c"
  "841536856d3f434324fc29c887bf3901"
)

usage() {
  cat <<'EOF'
Download RBO follow-up candidates for the strict occlusion-return probe.

Usage:
  scripts/download-rbo-occlusion-followup.sh [options]

Options:
  --list              Print the frozen follow-up selection (default).
  --download          Download with retry/resume, then verify.
  --verify-only       Verify files that are already present.
  --tier p0|all       Download P0 only or P0 plus P1 (default: p0).
  --output-dir PATH   Destination (default: outputs/assets/raw/rbo-articulated-objects).
  -h, --help          Show this help.

Examples:
  scripts/download-rbo-occlusion-followup.sh --list
  scripts/download-rbo-occlusion-followup.sh --download --tier p0
  scripts/download-rbo-occlusion-followup.sh --verify-only --tier all

The archives remain under ignored outputs/. Selection by metadata is not an
occlusion pass: each downloaded sequence still requires independent RGB-D/mesh
visibility recomputation. The script does not extract, adapt, train, or promote
the data to gate evidence.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

file_size() {
  wc -c < "$1" | tr -d '[:space:]'
}

verify_file() {
  local path="$1"
  local expected_size="$2"
  local expected_md5="$3"
  local kind="$4"
  local actual_size actual_md5

  [[ -f "${path}" ]] || die "missing file: ${path}"
  actual_size="$(file_size "${path}")"
  [[ "${actual_size}" == "${expected_size}" ]] ||
    die "size mismatch for ${path}: expected ${expected_size}, got ${actual_size}"
  actual_md5="$(md5sum "${path}" | awk '{print $1}')"
  [[ "${actual_md5}" == "${expected_md5}" ]] ||
    die "MD5 mismatch for ${path}: expected ${expected_md5}, got ${actual_md5}"

  case "${kind}" in
    plain)
      ;;
    tar.gz)
      gzip -t "${path}"
      tar -tzf "${path}" >/dev/null
      ;;
    *)
      die "unsupported file kind: ${kind}"
      ;;
  esac
  echo "verified: ${path} (${actual_size} bytes)"
}

download_file() {
  local name="$1"
  local expected_size="$2"
  local expected_md5="$3"
  local kind="$4"
  local url="${RBO_BASE_URL}/${name}/content"
  local destination="${OUTPUT_DIR}/${name}"
  local partial="${destination}.part"

  mkdir -p "${OUTPUT_DIR}"
  if [[ -f "${destination}" ]]; then
    verify_file "${destination}" "${expected_size}" "${expected_md5}" "${kind}"
    return 0
  fi
  if [[ -f "${partial}" ]] && (( $(file_size "${partial}") > expected_size )); then
    die "partial file is larger than expected; remove it before retrying: ${partial}"
  fi

  echo "downloading: ${url}"
  echo "destination: ${destination}"
  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-delay 2 \
    --connect-timeout 30 \
    --continue-at - \
    --output "${partial}" \
    "${url}"
  verify_file "${partial}" "${expected_size}" "${expected_md5}" "${kind}"
  mv -- "${partial}" "${destination}"
  echo "saved: ${destination}"
}

validate_selection() {
  local index_path="$1"
  local tier="$2"
  python3 - "${index_path}" "${tier}" <<'PY'
import csv
import sys

path, tier = sys.argv[1:]
expected = {
    "treasurebox25": {
        "Object": "treasurebox",
        "Camera Motion": "1",
        "Only Internal Interaction": "1",
        "Small Interaction": "0",
        "cluttered": "1",
        "force/torque sensor used": "1",
        "comment": "",
    },
    "laptop26": {
        "Object": "laptop",
        "Camera Motion": "1",
        "Only Internal Interaction": "1",
        "Small Interaction": "1",
        "cluttered": "0",
        "force/torque sensor used": "1",
        "comment": "laptop close to camera",
    },
    "globe25": {
        "Object": "globe",
        "Camera Motion": "1",
        "Only Internal Interaction": "1",
        "Small Interaction": "1",
        "cluttered": "1",
        "force/torque sensor used": "1",
        "comment": "",
    },
}
if tier == "all":
    expected.update(
        {
            "treasurebox24": {
                "Object": "treasurebox",
                "Camera Motion": "1",
                "Only Internal Interaction": "1",
                "Small Interaction": "1",
                "cluttered": "1",
                "force/torque sensor used": "1",
                "comment": "funny (ee got caught in box)",
            },
            "pliers24": {
                "Object": "pliers",
                "Camera Motion": "1",
                "Only Internal Interaction": "0",
                "Small Interaction": "0",
                "cluttered": "0",
                "force/torque sensor used": "1",
                "comment": "",
            },
            "tripod24": {
                "Object": "tripod",
                "Camera Motion": "1",
                "Only Internal Interaction": "1",
                "Small Interaction": "0",
                "cluttered": "0",
                "force/torque sensor used": "1",
                "comment": "",
            },
            "ikeasmall23": {
                "Object": "ikeasmall",
                "Camera Motion": "1",
                "Only Internal Interaction": "1",
                "Small Interaction": "0",
                "cluttered": "0",
                "force/torque sensor used": "1",
                "comment": "",
            },
        }
    )

with open(path, encoding="utf-8-sig", newline="") as handle:
    rows = {row["Name"]: row for row in csv.DictReader(handle)}

for name, contract in expected.items():
    if name not in rows:
        raise SystemExit(f"RBO index is missing follow-up sequence {name}")
    mismatches = {
        key: (value, rows[name].get(key))
        for key, value in contract.items()
        if rows[name].get(key) != value
    }
    if mismatches:
        raise SystemExit(f"RBO selection contract changed for {name}: {mismatches}")

print(f"validated RBO occlusion follow-up against: {path} ({len(expected)} interactions)")
PY
}

print_plan() {
  echo "RBO strict-occlusion follow-up candidates"
  echo "record=https://doi.org/10.5281/zenodo.${RBO_RECORD_ID}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "tier=${TIER}"
  echo
  echo "P0 interactions: treasurebox25, laptop26, globe25"
  echo "P0 payload with deduplicated companion models: 1099647770 bytes (about 1.024 GiB)"
  for i in "${!P0_FILES[@]}"; do
    printf '  %-30s %12s bytes  md5=%s\n' \
      "${P0_FILES[$i]}" "${P0_SIZES[$i]}" "${P0_MD5S[$i]}"
  done
  if [[ "${TIER}" == "all" ]]; then
    echo
    echo "P1 interactions: treasurebox24, tripod24, pliers24, ikeasmall23"
    echo "P1 selected payload: 955939893 bytes (about 0.890 GiB)"
    echo "P1 new-download payload with existing tripod model: 953270577 bytes (about 0.888 GiB)"
    for i in "${!P1_FILES[@]}"; do
      printf '  %-30s %12s bytes  md5=%s\n' \
        "${P1_FILES[$i]}" "${P1_SIZES[$i]}" "${P1_MD5S[$i]}"
    done
    echo "P0+P1 selected payload: 2055587663 bytes (about 1.914 GiB)"
  fi
  echo
  echo "Metadata proxies are not a visibility result; strict RGB-D V-O-V must be recomputed."
}

download_index() {
  local destination="${OUTPUT_DIR}/interactions_index.csv"
  local partial="${destination}.part"
  mkdir -p "${OUTPUT_DIR}"
  if [[ -f "${destination}" ]]; then
    verify_file "${destination}" "${RBO_INDEX_SIZE}" "${RBO_INDEX_MD5}" plain
    return 0
  fi
  curl \
    --fail \
    --location \
    --retry 5 \
    --retry-delay 2 \
    --connect-timeout 30 \
    --continue-at - \
    --output "${partial}" \
    "${RBO_INDEX_URL}"
  verify_file "${partial}" "${RBO_INDEX_SIZE}" "${RBO_INDEX_MD5}" plain
  mv -- "${partial}" "${destination}"
}

operate_files() {
  local operation="$1"
  local i
  for i in "${!P0_FILES[@]}"; do
    if [[ "${operation}" == "download" ]]; then
      download_file "${P0_FILES[$i]}" "${P0_SIZES[$i]}" "${P0_MD5S[$i]}" tar.gz
    else
      verify_file "${OUTPUT_DIR}/${P0_FILES[$i]}" "${P0_SIZES[$i]}" "${P0_MD5S[$i]}" tar.gz
    fi
  done
  if [[ "${TIER}" == "all" ]]; then
    for i in "${!P1_FILES[@]}"; do
      if [[ "${operation}" == "download" ]]; then
        download_file "${P1_FILES[$i]}" "${P1_SIZES[$i]}" "${P1_MD5S[$i]}" tar.gz
      else
        verify_file "${OUTPUT_DIR}/${P1_FILES[$i]}" "${P1_SIZES[$i]}" "${P1_MD5S[$i]}" tar.gz
      fi
    done
  fi
}

while (($#)); do
  case "$1" in
    --list)
      MODE="list"
      ;;
    --download)
      MODE="download"
      ;;
    --verify-only)
      MODE="verify"
      ;;
    --tier)
      shift
      (($#)) || die "--tier requires p0 or all"
      TIER="$1"
      ;;
    --output-dir)
      shift
      (($#)) || die "--output-dir requires a path"
      OUTPUT_DIR="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
  shift
done

case "${TIER}" in
  p0|all)
    ;;
  *)
    die "--tier must be p0 or all"
    ;;
esac

require_command md5sum
require_command gzip
require_command tar
require_command python3

case "${MODE}" in
  list)
    print_plan
    ;;
  download)
    require_command curl
    download_index
    validate_selection "${OUTPUT_DIR}/interactions_index.csv" "${TIER}"
    operate_files download
    ;;
  verify)
    verify_file \
      "${OUTPUT_DIR}/interactions_index.csv" \
      "${RBO_INDEX_SIZE}" \
      "${RBO_INDEX_MD5}" \
      plain
    validate_selection "${OUTPUT_DIR}/interactions_index.csv" "${TIER}"
    operate_files verify
    ;;
  *)
    die "unsupported mode: ${MODE}"
    ;;
esac

echo "completed mode=${MODE} tier=${TIER}"
