#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv_large_scale}"

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

echo "[info] Installing base requirements for large-scale tasks..."
python -m pip install -r requirements_large_scale.txt || true

echo
echo "[note] Large-scale tasks require extra system/compiled deps (vtk/pyvista/torch_geometric)."
echo "If installation failed, install them following your platform's recommended instructions,"
echo "then re-run this script."
echo
python - <<'PY'
import importlib


def check(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        print(f"[ok] import {pkg}")
        return True
    except Exception as e:
        print(f"[warn] import {pkg} failed: {e}")
        return False


check("torch")
check("einops")
check("timm")
check("pyvista")
check("vtk")
check("torch_geometric")
check("torch_scatter")
check("torch_sparse")
check("torch_cluster")
check("torch_spline_conv")
check("pyg_lib")
PY
echo
echo "[ok] Large-scale env setup attempted."
echo "Activate with: source ${VENV_DIR}/bin/activate"
