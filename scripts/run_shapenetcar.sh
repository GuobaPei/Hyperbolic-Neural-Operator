#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_DIR="${SHAPENETCAR_DATA_DIR:-${1:-}}"
SAVE_DIR="${SHAPENETCAR_SAVE_DIR:-${2:-}}"
if [[ -z "${DATA_DIR}" || -z "${SAVE_DIR}" ]]; then
  echo "Usage: $0 <SHAPENETCAR_DATA_DIR> <SHAPENETCAR_SAVE_DIR>   (or set SHAPENETCAR_DATA_DIR/SHAPENETCAR_SAVE_DIR)" >&2
  exit 1
fi

python large_scale/shapenetcar/main.py \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}"
