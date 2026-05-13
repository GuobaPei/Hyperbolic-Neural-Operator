#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_DIR="${AIRFRANS_DATA_DIR:-${1:-}}"
if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 <AIRFRANS_DATA_DIR>   (or set AIRFRANS_DATA_DIR)" >&2
  exit 1
fi

TASK="${AIRFRANS_TASK:-full}"

python large_scale/airfrans/main.py \
  --data_dir "${DATA_DIR}" \
  --task "${TASK}"
