# Reproduction status

Collected: eight correction checkpoints, 2014-2016 normalization statistics,
channel order, both paper-facing MPCA spatial bases, PCA metadata, preprocessing
scripts and training records. All prior checkpoint hashes and the normalization
hash match; the n4384 checkpoint has a new inventory hash but no historical hash
in the old Figure 1 result JSON. Original UNet2 class AST is unchanged.

Four PCA basis binaries are separate private Release assets. Their sizes, SHA256
and install paths are in artifacts/RELEASE_ASSETS.json. Yearly raw ERA5/CRA5,
decoded fields and training caches remain external. Official large CRA5 weights
and the decoder package are upstream dependencies, not redistributed here.

Still required before running on a new machine: configure data and model paths,
prepare the historical 2014-2018 truth/caches or rerun their preprocessing,
download Release PCA assets, and install compatible main/CRA5 decoder runtimes.
Many original scripts preserve original Ubuntu paths and some execute at import.
The frozen result records are available; a clean-environment end-to-end rerun has
not been performed for this packaging task. Do not claim one-command reproduction.
