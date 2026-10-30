# CRA5 baseline

Decodes the CRA5 compressed archive into the same 20-channel CONUS grid as our
own pipeline, so the two can be compared channel by channel.

    run_decode.sh            <- start here; sets the env this needs and runs decode_conus.py
    decode_conus.py          .bin -> cra5_conus/{year}.npy  (float32, T x 20 x 97 x 241)
    sdpa_patch.py            imported by decode_conus.py; without it VAEformer OOMs
    cra5_rmse.py             cra5_conus/ -> results/cra5_rmse.json
    check_align.py           confirms the CONUS crop lines up with our ERA5 arrays
    verify_channels.py       confirms the CRA5 channel indices map to the right variables
    install_cra5.sh, deps.sh rebuild .venv_cra5 from CRA5_src

Three upstream traps, all handled in the code but worth knowing:

1. The PyPI `cra5` package is an empty shell. Install from CRA5_src and set
   PYTHONPATH; run_decode.sh does both.
2. `decode_from_bin`'s default is `return_format='de_normlized'` -- misspelled
   upstream. The body only tests for 'normalized' and 'de_normalized', so the
   default falls through every branch and returns **None** with no error.
   Always pass 'de_normalized' explicitly.
3. VAEformer's attention materialises the full (B, heads, N, N) matrix, 6.4 GiB
   for one frame. sdpa_patch swaps in torch's SDPA, which is mathematically
   identical here (no mask, no relative position bias) and never builds it.

One caveat about the comparison itself: **tcwv is not in CRA5**. decode_conus.py
derives it by integrating specific humidity over 37 pressure levels,
(1/g)*int q dp with pressure clamped at the surface. That channel therefore
carries a discretisation error unrelated to compression, which inflates CRA5's
tcwv error. The bias runs against CRA5, so the 19:1 headline is unaffected --
but say so in the paper.
