"""Numerical checks for scientific failure modes; synthetic values are never paper results."""
import importlib.util
from pathlib import Path
import numpy as np

ROOT=Path('/home/tong/fourcastnet_us_mpca')
spec=importlib.util.spec_from_file_location('terrain',ROOT/'revision_2026/scripts/terrain_roughness.py')
terrain=importlib.util.module_from_spec(spec);spec.loader.exec_module(terrain)

# Ocean elevation must be irrelevant; boundary windows must not reflect or repeat land.
h=np.arange(49,dtype=float).reshape(7,7)*10
land=np.ones((7,7),dtype=bool);land[:,0]=False
r,count=terrain.roughness(h,land,min_neighbors=1)
changed=h.copy();changed[~land]=1e9
r2,_=terrain.roughness(changed,land,min_neighbors=1)
assert np.allclose(r,r2,equal_nan=True)
for i,j in [(0,1),(3,3),(6,6)]:
    block=h[max(0,i-2):i+3,max(0,j-2):j+3]
    mask=land[max(0,i-2):i+3,max(0,j-2):j+3]
    assert np.isclose(r[i,j],block[mask].std(ddof=0))
assert count[0,1]==9 and np.isnan(r[0,0])

# Regional deciles form a disjoint partition and are independent of errors/predictions.
mask=np.isfinite(r)
large=np.arange(400,dtype=float).reshape(20,20)
sets=terrain.bins(large,np.ones_like(large,dtype=bool))
assert len(sets)==10
assert np.all(np.stack([s[3] for s in sets]).sum(0)==1)

# Real output statistics must exactly reconstitute reported metrics.
d=np.load(ROOT/'results/revision_2026/eval2018/sufficient_statistics.npz')
assert np.allclose(d['before_map'].sum((1,2)),d['before_time'].sum(0),rtol=1e-12)
assert np.allclose(d['after_map'].sum((1,2)),d['after_time'].sum(0),rtol=1e-12)
assert np.all(d['valid_count']==97*241)
print('PASS: masked coast/boundaries, population SD, disjoint deciles, spatial/temporal SSE agreement')
