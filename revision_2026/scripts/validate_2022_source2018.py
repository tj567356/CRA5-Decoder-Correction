from pathlib import Path
import numpy as np,json,zarr
root=Path('/home/tong/fourcastnet_us_mpca')
x=np.load(root/'revision_2026/data/era5_validation2018/2018-01-01T00:00:00.npy')
rhpath=root/'revision_2026/data/native_rh_validation2018/20180101_rh.npz'
if rhpath.exists():x[17:19]=np.load(rhpath)['fields'][0]
y=zarr.open(str(root/'data/processed/fourcastnet20_conus/2018.zarr'),mode='r')['fields'][0]
names=json.loads((root/'revision_2026/data/era5_validation2018/2018_truth_manifest.json').read_text())['channels']
report=[]
for j,n in enumerate(names):
 if not np.isfinite(x[j]).all():continue
 d=x[j].astype(float)-y[j];entry={'channel':n,'arco_mean':float(x[j].mean()),'original_mean':float(y[j].mean()),'rmse':float(np.sqrt((d*d).mean())),'max_absolute_difference':float(abs(d).max()),'bitwise_equal':bool(np.array_equal(x[j],y[j]))};report.append(entry);print(entry,flush=True)
(root/'revision_2026/data/2022_source2018_validation.json').write_text(json.dumps(report,indent=2))
