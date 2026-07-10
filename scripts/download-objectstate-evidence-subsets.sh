#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

OUTPUT_ROOT="${REPO_ROOT}/outputs/assets/raw"
DATASET="all"
MODE="list"

RBO_RECORD_ID="1036660"
RBO_BASE_URL="https://zenodo.org/api/records/${RBO_RECORD_ID}/files"
RBO_INDEX_URL="${RBO_BASE_URL}/interactions_index.csv/content"
RBO_INDEX_SIZE=23134
RBO_INDEX_MD5="dbfe883bf2e1ee9ade51dbd736ac5713"

RBO_FILES=(
  "cardboardbox22_o.tar.gz"
  "tripod25_o.tar.gz"
  "cabinet20_o.tar.gz"
  "cardboardbox.tar.gz"
  "tripod.tar"
  "cabinet.tar.gz"
)
RBO_SIZES=(
  273817075
  370591226
  218716369
  1863310
  2669316
  402747
)
RBO_MD5S=(
  "81085e2ca7d470528d374939c29f5eb0"
  "ba592e92ba4727376b8fa9ba9ae1ee06"
  "107f954d9bee53d8e04dc307f37fa69a"
  "eb73ac4d68389b9f970fd7bc2486a8a2"
  "0b27993a32296e95e959e79244e37c44"
  "976710615705eeba043c82a26e5f6431"
)
RBO_KINDS=(
  "tar.gz"
  "tar.gz"
  "tar.gz"
  "tar.gz"
  "tar"
  "tar.gz"
)

RRC_BASE_URL="https://download.is.tue.mpg.de/rrc2020"
RRC_INDEX_URL="${RRC_BASE_URL}/rrc2020_dataset_index.db"
RRC_QUERY_URL="${RRC_BASE_URL}/rrc_dataset_query.py"
RRC_INDEX_SIZE=1695744
RRC_QUERY_SIZE=7467
RRC_INDEX_SHA256="deb7e9f9f2e26b3c2e3c6478cd5122e5cb9287b287c0e4783465b5780aa837af"
RRC_QUERY_SHA256="7978bef3ad8ec46d648c27c7eecce8eb33b48ff3732518321d702fca0b011e64"

RRC_FILES=(
  "7969.zip"
  "8076.zip"
  "9505.zip"
)
RRC_SIZES=(
  300897648
  307489148
  301996343
)

usage() {
  cat <<'EOF'
Download the frozen ObjGauss RBO/RRC real-evidence acquisition candidates.

Usage:
  scripts/download-objectstate-evidence-subsets.sh [options]

Options:
  --list                 Print the frozen selection without downloading (default).
  --download             Download with retry/resume, then verify.
  --verify-only          Verify files that are already present.
  --dataset all|rbo|rrc  Limit the operation to one dataset (default: all).
  --output-root PATH     Raw asset root (default: outputs/assets/raw).
  -h, --help             Show this help.

Examples:
  scripts/download-objectstate-evidence-subsets.sh --list
  scripts/download-objectstate-evidence-subsets.sh --download --dataset all
  scripts/download-objectstate-evidence-subsets.sh --verify-only --dataset rbo

The payloads remain local under ignored outputs/assets/raw/. This script does not
extract, normalize, adapt, train on, or promote any sequence to gate evidence.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

file_size() {
  wc -c < "$1" | tr -d '[:space:]'
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

verify_checksum() {
  local path="$1"
  local algorithm="$2"
  local expected="$3"
  local actual

  case "${algorithm}" in
    md5)
      actual="$(md5sum "${path}" | awk '{print $1}')"
      ;;
    sha256)
      actual="$(sha256sum "${path}" | awk '{print $1}')"
      ;;
    none)
      return 0
      ;;
    *)
      die "unsupported checksum algorithm: ${algorithm}"
      ;;
  esac

  [[ "${actual}" == "${expected}" ]] ||
    die "checksum mismatch for ${path}: expected ${expected}, got ${actual}"
}

verify_archive() {
  local path="$1"
  local kind="$2"

  case "${kind}" in
    plain)
      ;;
    tar.gz)
      gzip -t "${path}"
      tar -tzf "${path}" >/dev/null
      ;;
    tar)
      tar -tf "${path}" >/dev/null
      ;;
    zip)
      unzip -tq "${path}" >/dev/null
      ;;
    *)
      die "unsupported archive kind: ${kind}"
      ;;
  esac
}

verify_file() {
  local path="$1"
  local expected_size="$2"
  local algorithm="$3"
  local expected_checksum="$4"
  local kind="$5"
  local actual_size

  [[ -f "${path}" ]] || die "missing file: ${path}"
  actual_size="$(file_size "${path}")"
  [[ "${actual_size}" == "${expected_size}" ]] ||
    die "size mismatch for ${path}: expected ${expected_size}, got ${actual_size}"
  verify_checksum "${path}" "${algorithm}" "${expected_checksum}"
  verify_archive "${path}" "${kind}"
  echo "verified: ${path} (${actual_size} bytes)"
}

download_file() {
  local url="$1"
  local destination="$2"
  local expected_size="$3"
  local algorithm="$4"
  local expected_checksum="$5"
  local kind="$6"
  local partial="${destination}.part"

  mkdir -p "$(dirname -- "${destination}")"

  if [[ -f "${destination}" ]]; then
    verify_file "${destination}" "${expected_size}" "${algorithm}" "${expected_checksum}" "${kind}"
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

  verify_file "${partial}" "${expected_size}" "${algorithm}" "${expected_checksum}" "${kind}"
  mv -- "${partial}" "${destination}"
  echo "saved: ${destination}"
}

selected() {
  [[ "${DATASET}" == "all" || "${DATASET}" == "$1" ]]
}

print_plan() {
  echo "ObjGauss real-evidence acquisition candidates"
  echo "output_root=${OUTPUT_ROOT}"
  echo

  if selected rbo; then
    echo "RBO Articulated Objects (CC BY 4.0)"
    echo "  record=https://doi.org/10.5281/zenodo.${RBO_RECORD_ID}"
    echo "  selected interactions: cardboardbox22, tripod25, cabinet20"
    echo "  selection contract: camera_motion=1, small_interaction=0, force_torque=1, blank warning comment"
    echo "  payload plus companion models: 868060043 bytes"
    for i in "${!RBO_FILES[@]}"; do
      printf '  %-30s %12s bytes  md5=%s\n' \
        "${RBO_FILES[$i]}" "${RBO_SIZES[$i]}" "${RBO_MD5S[$i]}"
    done
    echo
  fi

  if selected rrc; then
    echo "RRC 2020 (CC BY-NC-SA 4.0)"
    echo "  source=https://people.tuebingen.mpg.de/mpi-is-software/data/rrc2020/"
    echo "  selected jobs: 7969, 8076, 9505 (phase 2, level 4, robot roboch1)"
    echo "  index filter: sustained height >=0.08m, displacement >=0.12m, goal distance <=0.01m"
    echo "  payload: 910383139 bytes"
    for i in "${!RRC_FILES[@]}"; do
      printf '  %-30s %12s bytes\n' "${RRC_FILES[$i]}" "${RRC_SIZES[$i]}"
    done
    echo
  fi

  if [[ "${DATASET}" == "all" ]]; then
    echo "combined payload: 1778443182 bytes (about 1.66 GiB)"
    echo
  fi

  echo "These are acquisition candidates, not gate-pass evidence."
}

validate_rbo_selection() {
  local index_path="$1"
  local sample

  for sample in cardboardbox22 tripod25 cabinet20; do
    awk -F, -v sample="${sample}" '
      $1 == sample && $5 == "1" && $7 == "0" && $9 == "1" && $11 == "" {
        found = 1
      }
      END { exit(found ? 0 : 1) }
    ' "${index_path}" || die "RBO selection no longer matches frozen criteria: ${sample}"
  done
  echo "validated RBO selection against: ${index_path}"
}

validate_rrc_selection() {
  local index_path="$1"

  python3 - "${index_path}" <<'PY'
import sqlite3
import sys

path = sys.argv[1]
selected = (7969, 8076, 9505)
connection = sqlite3.connect(path)
rows = connection.execute(
    """
    SELECT job_id, challenge_phase, robot_name, difficulty_level,
           cumulative_reward, baseline_reward, min_distance_to_goal_30,
           max_height_30, furthest_from_start_30
      FROM jobs
     WHERE job_id IN (?, ?, ?)
    """,
    selected,
).fetchall()
connection.close()

if {row[0] for row in rows} != set(selected):
    raise SystemExit("RRC index does not contain the frozen three-job selection")

for row in rows:
    (
        job_id,
        phase,
        robot_name,
        difficulty,
        reward,
        baseline,
        min_distance_30,
        max_height_30,
        displacement_30,
    ) = row
    valid = (
        phase == 2
        and robot_name == "roboch1"
        and difficulty == 4
        and reward > baseline
        and min_distance_30 <= 0.01
        and max_height_30 >= 0.08
        and displacement_30 >= 0.12
    )
    if not valid:
        raise SystemExit(f"RRC job {job_id} no longer matches frozen criteria")

print(f"validated RRC selection against: {path}")
PY
}

download_rbo() {
  local output_dir="${OUTPUT_ROOT}/rbo-articulated-objects"
  local index_path="${output_dir}/interactions_index.csv"

  download_file \
    "${RBO_INDEX_URL}" \
    "${index_path}" \
    "${RBO_INDEX_SIZE}" \
    md5 \
    "${RBO_INDEX_MD5}" \
    plain
  validate_rbo_selection "${index_path}"

  for i in "${!RBO_FILES[@]}"; do
    download_file \
      "${RBO_BASE_URL}/${RBO_FILES[$i]}/content" \
      "${output_dir}/${RBO_FILES[$i]}" \
      "${RBO_SIZES[$i]}" \
      md5 \
      "${RBO_MD5S[$i]}" \
      "${RBO_KINDS[$i]}"
  done
}

download_rrc() {
  local output_dir="${OUTPUT_ROOT}/rrc2020"
  local index_path="${output_dir}/rrc2020_dataset_index.db"

  download_file \
    "${RRC_INDEX_URL}" \
    "${index_path}" \
    "${RRC_INDEX_SIZE}" \
    sha256 \
    "${RRC_INDEX_SHA256}" \
    plain
  download_file \
    "${RRC_QUERY_URL}" \
    "${output_dir}/rrc_dataset_query.py" \
    "${RRC_QUERY_SIZE}" \
    sha256 \
    "${RRC_QUERY_SHA256}" \
    plain
  validate_rrc_selection "${index_path}"

  for i in "${!RRC_FILES[@]}"; do
    download_file \
      "${RRC_BASE_URL}/zarr/${RRC_FILES[$i]}" \
      "${output_dir}/${RRC_FILES[$i]}" \
      "${RRC_SIZES[$i]}" \
      none \
      "" \
      zip
  done
}

verify_rbo() {
  local output_dir="${OUTPUT_ROOT}/rbo-articulated-objects"
  local index_path="${output_dir}/interactions_index.csv"

  verify_file "${index_path}" "${RBO_INDEX_SIZE}" md5 "${RBO_INDEX_MD5}" plain
  validate_rbo_selection "${index_path}"
  for i in "${!RBO_FILES[@]}"; do
    verify_file \
      "${output_dir}/${RBO_FILES[$i]}" \
      "${RBO_SIZES[$i]}" \
      md5 \
      "${RBO_MD5S[$i]}" \
      "${RBO_KINDS[$i]}"
  done
}

verify_rrc() {
  local output_dir="${OUTPUT_ROOT}/rrc2020"
  local index_path="${output_dir}/rrc2020_dataset_index.db"

  verify_file "${index_path}" "${RRC_INDEX_SIZE}" sha256 "${RRC_INDEX_SHA256}" plain
  verify_file \
    "${output_dir}/rrc_dataset_query.py" \
    "${RRC_QUERY_SIZE}" \
    sha256 \
    "${RRC_QUERY_SHA256}" \
    plain
  validate_rrc_selection "${index_path}"
  for i in "${!RRC_FILES[@]}"; do
    verify_file \
      "${output_dir}/${RRC_FILES[$i]}" \
      "${RRC_SIZES[$i]}" \
      none \
      "" \
      zip
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      MODE="list"
      shift
      ;;
    --download)
      MODE="download"
      shift
      ;;
    --verify-only)
      MODE="verify"
      shift
      ;;
    --dataset)
      [[ $# -ge 2 ]] || die "--dataset requires a value"
      DATASET="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || die "--output-root requires a value"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

case "${DATASET}" in
  all|rbo|rrc)
    ;;
  *)
    die "--dataset must be one of: all, rbo, rrc"
    ;;
esac

print_plan

if [[ "${MODE}" == "list" ]]; then
  exit 0
fi

require_command awk
require_command curl
require_command gzip
require_command md5sum
require_command python3
require_command sha256sum
require_command tar
require_command unzip

if [[ "${MODE}" == "download" ]]; then
  selected rbo && download_rbo
  selected rrc && download_rrc
elif [[ "${MODE}" == "verify" ]]; then
  selected rbo && verify_rbo
  selected rrc && verify_rrc
else
  die "unsupported mode: ${MODE}"
fi

echo "completed mode=${MODE} dataset=${DATASET}"
