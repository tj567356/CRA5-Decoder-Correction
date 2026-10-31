# Four real validation frames

`cra5_2017_smoke.npz` contains the first four input/normalized-residual-target
frames from the original 2017 validation caches, shape 4x20x97x241. The JSON
records source attribution, channel order, hashes and expected per-channel
errors. These are derived ERA5/CRA5 scientific fields, not synthetic inputs.

Use `python scripts/smoke_corrector.py --device cpu --out outputs/smoke_cpu`.
This only verifies frozen inference on four validation frames. It is neither
training, model selection nor a full-year 2018/2022 accuracy reproduction.
The ground-truth residual target is used for scoring, never fed to the network.
