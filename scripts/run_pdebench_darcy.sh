#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${PDEBENCH_DATA_ROOT:-${1:-}}"
if [[ -z "${DATA_ROOT}" ]]; then
  echo "Usage: $0 <PDEBENCH_DATA_ROOT>   (or set PDEBENCH_DATA_ROOT)" >&2
  exit 1
fi

python -m pdebench.scripts.train_darcy \
  --data_path "${DATA_ROOT}/darcy" \
  --out_dir "${ROOT_DIR}/outputs/pdebench/darcy"

