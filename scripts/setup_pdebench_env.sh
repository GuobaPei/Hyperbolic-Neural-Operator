#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv_pdebench}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] PYTHON_BIN='${PYTHON_BIN}' not found. Set PYTHON_BIN to your Python (e.g., python)." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements_pdebench.txt

echo "[ok] PDEBench env ready."
echo "Activate with: source ${VENV_DIR}/bin/activate"

