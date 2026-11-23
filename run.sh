#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ "\${1:-}" == "install" ]]; then
  shift
  exec ./install.sh "$@"
fi
PY=python3
[[ -x .venv/bin/python ]] && PY=.venv/bin/python
exec "$PY" fm.py "$@"
