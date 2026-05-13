#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOT="${PDEBENCH_DATA_ROOT:-${1:-}}"
if [[ -z "${DATA_ROOT}" ]]; then
  echo "Usage: $0 <PDEBENCH_DATA_ROOT>   (or set PDEBENCH_DATA_ROOT)" >&2
  exit 1
fi

"${ROOT_DIR}/scripts/run_pdebench_darcy.sh" "${DATA_ROOT}"
"${ROOT_DIR}/scripts/run_pdebench_navier_stokes.sh" "${DATA_ROOT}"
"${ROOT_DIR}/scripts/run_pdebench_airfoil.sh" "${DATA_ROOT}"
"${ROOT_DIR}/scripts/run_pdebench_pipe.sh" "${DATA_ROOT}"
"${ROOT_DIR}/scripts/run_pdebench_plasticity.sh" "${DATA_ROOT}"
"${ROOT_DIR}/scripts/run_pdebench_elasticity.sh" "${DATA_ROOT}"

