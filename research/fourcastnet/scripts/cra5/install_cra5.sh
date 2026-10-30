#!/bin/bash
ROOT=$HOME/fourcastnet_us_mpca
VENV=$ROOT/.venv_cra5

rm -rf $VENV

echo "=== create venv: python3.10 -m venv --without-pip (bypasses missing ensurepip) ==="
/usr/bin/python3.10 -m venv --without-pip $VENV || { echo "VENV_CREATE_FAILED"; exit 1; }
ls $VENV/bin

echo
echo "=== bootstrap pip via get-pip.py ==="
curl -sS -o /tmp/get-pip.py https://bootstrap.pypa.io/pip/get-pip.py
$VENV/bin/python /tmp/get-pip.py -q || { echo "GETPIP_FAILED"; exit 1; }

echo
echo "=== sanity: is site-packages on sys.path? ==="
$VENV/bin/python -c "import sys,sysconfig; print('purelib:',sysconfig.get_paths()['purelib']); print('on path:',[p for p in sys.path if 'site-packages' in p])"
$VENV/bin/pip -V

echo
echo "=== smoke test ==="
$VENV/bin/pip install -q six && $VENV/bin/python -c "import six; print('six OK ->', six.__file__)"

echo
echo "=== installing cra5 (--pre: only release is 0.0.3.dev1) ==="
$VENV/bin/pip install --pre cra5
echo "INSTALL_EXIT=$?"

echo
echo "=== import check ==="
$VENV/bin/python -c "import cra5; print('cra5 at', cra5.__file__)"

echo "=== cra5_api surface ==="
$VENV/bin/python -c "from cra5.api import cra5_api; print([m for m in dir(cra5_api) if not m.startswith('_')])"

echo "=== torch / cuda ==="
$VENV/bin/python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
