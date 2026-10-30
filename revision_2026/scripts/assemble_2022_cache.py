"""Strict full-year alignment and original float16 normalization; no fitting."""
import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np

CHANNELS=['u10','v10','t2m','sp','msl','t850','u1000','v1000','z1000','u850','v850','z850','u500','v500','z500','t500','z50','r500','r850','tcwv']
CODEC_SHA='812a6d09ff039e4ab468915942e7fd00f11e7c58b839cd1b60fe1d0586a40dad'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/tong/fourcastnet_us_mpca'))
    p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--decoded',type=Path,required=True)
    p.add_argument('--rh',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    assert not a.out.exists(),'Use a fresh output directory to preserve old artifacts'
    stamps=[(dt.datetime(2022,1,1)+dt.timedelta(hours=6*i)).isoformat() for i in range(1460)]
    decoded_manifest=json.loads((a.decoded/'2022_decoded_manifest.json').read_text())
    assert decoded_manifest['status']=='complete' and decoded_manifest['completed_frames']==1460
    assert decoded_manifest['timestamps']==stamps
    assert decoded_manifest['codec_weight_sha256']==CODEC_SHA
    bitstream_manifest=json.loads((a.root/'revision_2026/data/cra5/2022_data_manifest.json').read_text())
    assert bitstream_manifest['status']=='complete' and bitstream_manifest['verified_count']==1460
    bit_hash={p['path'].split('/')[-1][:-4]:p['lfs']['oid'] for p in bitstream_manifest['source_files']}
    assert set(bit_hash)==set(stamps)
    truth_manifest=json.loads((a.truth/'2022_truth_manifest.json').read_text())
    assert truth_manifest['completed_frames']==1460 and truth_manifest['channels']==CHANNELS
    assert np.array_equal(truth_manifest['lat'],np.arange(49,24.99,-.25))
    assert np.array_equal(truth_manifest['lon'],np.arange(-130,-69.99,.25))
    stats=a.root/'mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz'
    assert sha(stats)=='272cc45d231ec349025c4fa1e9f69366ae874739c98299ce666921821f13dba2'
    st=np.load(stats)
    assert st['channel_names'].astype(str).tolist()==CHANNELS
    mean=st['mean'].astype(np.float32)[:,None,None];std=st['std'].astype(np.float32)[:,None,None]
    assert (std>0).all()
    # Fail before creating any output if a timestamp/receipt is absent.
    for stamp in stamps:
        for directory in [a.truth,a.decoded]:
            assert (directory/f'{stamp}.npy').is_file() and (directory/f'{stamp}.json').is_file(),stamp
        assert (a.rh/f'{stamp[:10].replace("-", "")}_rh.npz').is_file(),stamp
    a.out.mkdir(parents=True)
    shape=(1460,20,97,241)
    x=np.lib.format.open_memmap(a.out/'input_2022.npy',mode='w+',dtype=np.float16,shape=shape)
    y=np.lib.format.open_memmap(a.out/'target_2022.npy',mode='w+',dtype=np.float16,shape=shape)
    rows=[];day_cache={}
    for i,stamp in enumerate(stamps):
        truth_file=a.truth/f'{stamp}.npy';decoded_file=a.decoded/f'{stamp}.npy'
        receipts=[]
        for path in [truth_file,decoded_file]:
            receipt=json.loads(path.with_suffix('.json').read_text())
            assert receipt['timestamp']==stamp
            assert sha(path)==receipt['sha256']
            receipts.append(receipt)
        decoded_receipt=receipts[1]
        assert [s.replace('_','') for s in decoded_receipt['channels']]==CHANNELS
        assert decoded_receipt['codec_weight_sha256']==CODEC_SHA
        assert decoded_receipt['bin_sha256']==bit_hash[stamp]
        truth=np.load(truth_file).astype(np.float32);recon=np.load(decoded_file).astype(np.float32)
        assert truth.shape==recon.shape==(20,97,241)
        date=stamp[:10].replace('-','');rhfile=a.rh/f'{date}_rh.npz'
        if date not in day_cache:
            rreceipt=json.loads(rhfile.with_suffix('.json').read_text())
            assert sha(rhfile)==rreceipt['sha256']
            assert rreceipt['date']==date and rreceipt['units']=='%'
            assert rreceipt['time_units']=='hours since 1900-01-01 00:00:00'
            assert rreceipt['calendar']=='gregorian' and rreceipt['pressure_levels_hpa']==[500,850]
            assert np.array_equal(rreceipt['latitude'],np.arange(49,24.99,-.25))
            assert np.array_equal(rreceipt['longitude'],np.arange(-130,-69.99,.25))
            day_cache={date:dict(np.load(rhfile))}
            assert day_cache[date]['timestamps'].astype(str).tolist()==rreceipt['timestamps']
        rh=day_cache[date];hour=int(stamp[11:13]);j=hour//6
        assert str(rh['timestamps'][j])==stamp
        assert rh['channels'].astype(str).tolist()==['r500','r850']
        assert rh['fields'].shape==(4,2,97,241)
        assert np.isfinite(rh['fields']).all() and np.max(np.abs(rh['fields']))<1000
        truth[17:19]=rh['fields'][j]
        assert np.isfinite(truth).all() and np.isfinite(recon).all()
        x[i]=((recon-mean)/std).astype(np.float16)
        y[i]=((truth-recon)/std).astype(np.float16)
        assert np.isfinite(x[i]).all() and np.isfinite(y[i]).all()
        rows.append({'index':i,'timestamp_utc':stamp,'truth_sha256':receipts[0]['sha256'],
                     'native_rh_sha256':sha(rhfile),'decoded_sha256':receipts[1]['sha256']})
        if i%100==0:print(i+1,stamp,flush=True)
    x.flush();y.flush();del x,y
    with (a.out/'2022_full_timestamps.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    manifest={'year':2022,'frames':1460,'alignment_verified':True,'channels':CHANNELS,
              'stats_sha256':sha(stats),'normalization':'frozen 2014–2016, float32 arithmetic then float16 cache',
              'truth':str(a.truth),'decoded':str(a.decoded),'native_rh':str(a.rh),
              'input_sha256':sha(a.out/'input_2022.npy'),'target_sha256':sha(a.out/'target_2022.npy'),
              'timestamps_sha256':sha(a.out/'2022_full_timestamps.csv'),
              'tcwv':'native ERA5 truth; original pressure-clipped humidity integral for CRA5',
              'complete':True}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('FULL 1460-frame cache ready',flush=True)

if __name__=='__main__':main()
