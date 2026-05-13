#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -B - <<'PY'
import sys
from pathlib import Path

root = Path(".").resolve()
failed = []
for p in root.rglob("*.py"):
    try:
        source = p.read_text(encoding="utf-8")
        compile(source, str(p), "exec")
    except Exception as e:
        failed.append((p, e))

if failed:
    print("PY_SYNTAX_CHECK_FAILED")
    for p, e in failed:
        print(p)
        print(e)
    sys.exit(1)

import pdebench
import pdebench.hno
print("smoke test ok")
PY
