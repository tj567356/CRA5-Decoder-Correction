#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/fourcastnet_us_mpca"
export PYTHONPATH="$ROOT/CRA5_src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
exec "$ROOT/.venv_cra5/bin/python" -u revision_2026/scripts/decode_2022_cra5.py "$@"
