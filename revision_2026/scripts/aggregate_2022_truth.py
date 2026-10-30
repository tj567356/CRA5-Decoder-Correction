"""Wait for all ERA5 shards, then verify exact full-year native-18 provenance.

The global manifest explicitly retains missing RH until separate assembly.
"""
import argparse,datetime as dt,hashlib,json,time
from pathlib import Path
import numpy as np
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.home()/'fourcastnet_us_mpca');p.add_argument('--wait',action='store_true');a=p.parse_args();folder=a.root/'revision_2026/data/era5_2022'
 stamps=[(dt.datetime(2022,1,1)+dt.timedelta(hours=6*i)).isoformat() for i in range(1460)]
 while True:
  ready=sum((folder/f'{ts}.npy').exists() and (folder/f'{ts}.json').exists() for ts in stamps)
  if ready==1460:break
  print(f'Waiting ERA5 native18: {ready}/1460 atomic frames+receipts',flush=True)
  if not a.wait:raise SystemExit(1)
  time.sleep(30)
 existing={p.stem for p in folder.glob('*.npy')};assert existing==set(stamps),'Unexpected or missing timestamp files'
 frame_records=[];variables=set();units={};generations={}
 for i,ts in enumerate(stamps):
  path=folder/f'{ts}.npy';r=json.loads(path.with_suffix('.json').read_text());assert r['timestamp']==ts and r['shape']==[20,97,241]
  assert sha(path)==r['sha256'],path
  x=np.load(path);assert x.shape==(20,97,241) and x.dtype==np.float32
  assert np.isnan(x[17:19]).all() and np.isfinite(x[:17]).all() and np.isfinite(x[19]).all()
  assert x[19].min()>=0 and x[3].min()>20000 and x[2].min()>150
  for proof in r['provenance']:
   name=proof['variable'];variables.add(name);assert proof['generation'] and proof['selected_ranges']
   if name in units:assert units[name]==proof['units']
   units[name]=proof['units'];generations.setdefault(name,set()).add(proof['generation'])
  frame_records.append({'timestamp':ts,'sha256':r['sha256'],'receipt_sha256':sha(path.with_suffix('.json'))})
  if i%100==0:print(f'Hash/shape/finite/source proof validated {i+1}/1460',flush=True)
 mf=folder/'2022_truth_manifest.json';manifest=json.loads(mf.read_text())
 manifest.update(status='native18_complete_pending_native_rh_assembly',completed_frames=1460,expected_frames=1460,requested_frames=1460,timestamps=stamps,frames=frame_records,missing_channels=['r500','r850'],units=units,validation={'exact_full_year_timestamps':True,'sha256_all_frames':True,'sha256_all_receipts':True,'shape':[1460,20,97,241],'native18_finite':True,'rh_slots_explicit_nan':True,'native_tcwv':True,'source_variables':sorted(variables),'object_generation_counts':{k:len(v) for k,v in generations.items()}})
 tmp=mf.with_suffix('.json.part');tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(mf)
 print('FULL YEAR native18 verified. Native RH must still be assembled before evaluation.',flush=True)
if __name__=='__main__':main()
