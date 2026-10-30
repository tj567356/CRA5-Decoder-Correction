import os, sys, json
import matplotlib; matplotlib.use("Agg")
import torch

# cra5_api.__init__ loads the net onto the GPU; we only want metadata, so read
# the config directly instead.
from cra5.utils.config import Config
import cra5.api as _api

d = os.path.dirname(os.path.abspath(_api.__file__))
cfg = Config.fromfile(f"{d}/cra5_268v_config.py")

print("=== vnames ===")
for k, v in cfg.vnames.items():
    print(f"{k}: {v}")

print()
print("=== pressure_level ===")
print(cfg.pressure_level)
print("n levels:", len(cfg.pressure_level))

print()
print("=== other cfg keys ===")
for k in cfg.keys():
    if k not in ("vnames", "pressure_level", "total_levels"):
        val = cfg[k]
        s = str(val)
        print(f"{k}: {s[:200]}")
