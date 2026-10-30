#!/bin/bash
V=$HOME/fourcastnet_us_mpca/.venv_cra5
S=$HOME/fourcastnet_us_mpca/CRA5_src
PIP=$V/bin/pip
PY=$V/bin/python

echo "=== batch install likely-missing deps ==="
$PIP install -q cdsapi opencv-python-headless plotly timm 2>&1 | tail -5

echo
echo "=== iterate: import until it stops complaining ==="
for i in 1 2 3 4 5 6 7 8; do
  ERR=$(PYTHONPATH=$S $PY -c "from cra5.api import cra5_api" 2>&1)
  if [ -z "$ERR" ]; then echo "IMPORT OK after $((i-1)) extra installs"; break; fi
  MOD=$(echo "$ERR" | grep -oP "No module named '\K[^']+" | head -1 | cut -d. -f1)
  if [ -z "$MOD" ]; then echo "NON-IMPORT ERROR:"; echo "$ERR" | tail -15; break; fi
  echo ">>> missing: $MOD -> installing"
  $PIP install -q "$MOD" 2>&1 | tail -2 || echo "    (pip install $MOD FAILED)"
done

echo
echo "=== final import check ==="
PYTHONPATH=$S $PY -c "from cra5.api import cra5_api; print('API OK:', [m for m in dir(cra5_api) if not m.startswith('_')])"

echo
echo "=== compressai C extension present? (needed for entropy decoding) ==="
PYTHONPATH=$S $PY -c "
import glob, os
d=os.path.expanduser('~/fourcastnet_us_mpca/CRA5_src/cra5/models/compressai')
print('so files:', glob.glob(d+'/**/*.so', recursive=True))
try:
    from cra5.models.compressai import ans; print('ans module:', ans.__file__)
except Exception as e:
    print('ans import failed:', type(e).__name__, e)
"
