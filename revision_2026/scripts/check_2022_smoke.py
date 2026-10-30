"""Data/codec smoke verification for four timestamps; does not evaluate U-Net."""
import datetime as dt,hashlib,json
from pathlib import Path
import numpy as np
root=Path.home()/'fourcastnet_us_mpca/revision_2026/data'
stamps=[(dt.datetime(2022,1,1)+dt.timedelta(hours=6*i)).isoformat() for i in range(4)]
report={'timestamps':stamps,'frames':4,'full_year_evaluation':False,'checks':[]}
for ts in stamps:
 decoded=root/'cra5_decoded_2022'/f'{ts}.npy';truth=root/'era5_2022'/f'{ts}.npy';receipt=json.loads(decoded.with_suffix('.json').read_text())
 assert hashlib.sha256(decoded.read_bytes()).hexdigest()==receipt['sha256']
 assert receipt['timestamp']==ts and receipt['codec_weight_sha256']=='812a6d09ff039e4ab468915942e7fd00f11e7c58b839cd1b60fe1d0586a40dad'
 y=np.load(truth);x=np.load(decoded);rh=np.load(root/'native_rh_2022/20220101_rh.npz')
 assert str(rh['timestamps'][int(ts[11:13])//6])==ts;y[17:19]=rh['fields'][int(ts[11:13])//6]
 assert x.shape==y.shape==(20,97,241) and x.dtype==y.dtype==np.float32 and np.isfinite(x).all() and np.isfinite(y).all()
 assert 150<y[2].min()<350 and 20000<y[3].min()<110000 and y[16].mean()>100000 and y[19].min()>=0
 report['checks'].append({'timestamp':ts,'shape':[20,97,241],'decoded_sha256':receipt['sha256'],'all_twenty_channels_finite':True,'native_rh_timestamps_aligned':True,'truth_z_is_geopotential':True,'native_era5_tcwv':True,'truth_channel_means':y.mean(axis=(1,2)).tolist(),'decoded_channel_means':x.mean(axis=(1,2)).tolist()})
out=Path.home()/'fourcastnet_us_mpca/revision_2026/2022_four_frame_smoke.json';out.write_text(json.dumps(report,indent=2));print('PASS four-frame data/codec smoke: exact timestamps,20channels,shape,dtype,hash,weightidentity,finite,units/nativeRH/TCWV. Full-year evaluation NOT claimed.')
