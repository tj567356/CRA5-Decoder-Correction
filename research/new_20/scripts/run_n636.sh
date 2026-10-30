#!/usr/bin/env bash
# The three thresholds at n=636 (735x), run back to back.
# 99% uses the ORIGINAL project's cores and basis -- they already exist and are
# byte-identical to what produced the published e97 point, so re-encoding would
# only risk drift.
set -eu
S=/home/tong/new_20/scripts
OLD=/home/tong/fourcastnet_us_mpca
bash $S/run_point.sh s99 636 "$OLD" "$OLD/mpca_train_basis_bundle/models/mpca_spatial_99energy"
bash $S/run_point.sh s98 636 /home/tong/new_20/cores_s98 /home/tong/new_20/mpca_train_basis_bundle/models/mpca_spatial_s98
bash $S/run_point.sh s97 636 /home/tong/new_20/cores_s97 /home/tong/new_20/mpca_train_basis_bundle/models/mpca_spatial_s97
echo "=== n636 sweep all done $(date) ==="
