#!/usr/bin/env bash
# Decode the CRA5 archive into 20-channel CONUS arrays.
#
# The point of this wrapper is that the environment is not optional and is not
# guessable. Three settings have to be right or the run fails in ways that do
# not point at their cause:
#
#   .venv_cra5      python3.10. The main .venv cannot run this -- CRA5 pulls in
#                   compressai and an older torch stack that conflicts with the
#                   pipeline's.
#   PYTHONPATH      the PyPI `cra5` package is an empty shell. The real code is
#                   in CRA5_src and is not installed into site-packages.
#   expandable_segments
#                   VAEformer fragments the allocator badly enough to OOM on a
#                   10 GB card even with the SDPA patch.
#
# Previously these lived only in STATUS.md as prose and in whatever shell the
# command was typed into. Now they live here.
#
# Usage:
#   ./run_decode.sh --validate-only              one frame, prints tcwv sanity checks
#   ./run_decode.sh --years 2018                 one year
#   ./run_decode.sh                              all five years (~hours, GPU)
#
# Afterwards, the metric:
#   ../../.venv/bin/python cra5_rmse.py          -> results/cra5_rmse.json
# (that one only needs numpy+zarr, so it runs under the MAIN venv)
set -euo pipefail

ROOT="$HOME/fourcastnet_us_mpca"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$ROOT/CRA5_src"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$ROOT/.venv_cra5/bin/python"
[ -x "$PY" ] || { echo "missing $PY -- rebuild it with ./install_cra5.sh" >&2; exit 1; }
[ -d "$ROOT/CRA5_src/cra5" ] || { echo "missing $ROOT/CRA5_src/cra5" >&2; exit 1; }

# decode_conus.py imports sdpa_patch from its own directory, so run from there
cd "$HERE"

free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$free_mib" -lt 6000 ]; then
    echo "WARNING: only ${free_mib} MiB free on the GPU. VAEformer needs headroom" >&2
    echo "         even with the SDPA patch; check nothing else is training." >&2
fi

echo "python      $PY"
echo "PYTHONPATH  $PYTHONPATH"
echo "gpu free    ${free_mib} MiB"
echo

exec "$PY" -u decode_conus.py "$@"
