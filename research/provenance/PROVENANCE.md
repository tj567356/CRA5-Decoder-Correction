# Provenance

## Snapshot

The files in this directory were copied read-only from the Ubuntu host alias
`clh703` on 2026-09-06.

Source roots:

```text
/home/tong/new_20
/home/tong/fourcastnet_us_mpca
```

The current paper source in `paper/` comes from the Windows working copy and is
the authoritative manuscript snapshot. The Ubuntu `new_20/paper/` directory was
not copied because its `main.tex` differs from the current paper.

## Result mapping

`research/new_20/results/` contains the rate-matched and crossover records,
including `rate_matched_val.json`, `crossover.json`, `rate_spread.json`, and
the stage-1/stage-2 comparison records.

`research/fourcastnet/results/` contains the CRA5 correction, temporal,
generality, spatial-scale, and terrain-control records. The main paper-facing
records include:

```text
cra5_rmse.json
cra5_temporal.json
generality.json
error_vs_elevation.json
error_vs_elevation.csv
error_scale.json
error_scale.csv
error_static_fraction.json
error_static_fraction.csv
elevation_controls.json
sp_terrain_control.json
```

The two `STATUS.md` files preserve the experiment notes from each source tree.

## Intentionally excluded

The following remain on Ubuntu and are not GitHub artifacts:

- raw data and Zarr stores
- decoded arrays and MPCA cores
- virtual environments
- the full checkpoint sweeps
- external CRA5 source trees
- old backup logs and machine-specific infrastructure files

The source data roots were approximately 227 GB for `new_20` and 214 GB for
`fourcastnet_us_mpca` at the time of inspection.

## Reproduction caveats

The scripts retain their original Ubuntu paths and are an audit archive, not yet
a one-command reproduction pipeline. The paper's formal spatial diagnostic is
reported as 5x5; the older Windows exploratory scripts `why_sp.py` and
`why_r500.py` still contain 3x3 settings and must not be treated as the source
of the formal 5x5 numbers without reconciliation.
