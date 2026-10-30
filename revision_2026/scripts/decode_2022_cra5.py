"""Resume exact original CRA5 decoding into atomic, hash-verified 2022 frames.

Invoke ONLY via run_2022_decode.sh using the original .venv_cra5.
Never trains or changes original codec/checkpoints/2018 outputs.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np,torch

EXPECTED_WEIGHT='812a6d09ff039e4ab468915942e7fd00f11e7c58b839cd1b60fe1d0586a40dad'
def atomic_json(path,data):
 tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.home()/'fourcastnet_us_mpca');p.add_argument('--wait-download',action='store_true');a=p.parse_args()
 root=a.root;source=root/'revision_2026/data/cra5';out=root/'revision_2026/data/cra5_decoded_2022';out.mkdir(parents=True,exist_ok=True)
 weights=Path.home()/'.cache/torch/hub/checkpoints/cra5_268v_300k.pth';assert sha(weights)==EXPECTED_WEIGHT
 while True:
  mf=source/'2022_data_manifest.json'
  try:manifest=json.loads(mf.read_text()) if mf.exists() else {}
  except json.JSONDecodeError:manifest={}
  if manifest.get('status')=='complete' and manifest.get('verified_count')==1460:break
  if not a.wait_download:raise RuntimeError('Complete checksum-verified 1460-file download required')
  print('Waiting for complete 1460-file checksum-verified CRA5 manifest',flush=True);time.sleep(30)
 assert len(set(manifest['expected_timestamps']))==1460
 sys.path.insert(0,str(root/'scripts/cra5'))
 import decode_conus as original
 print('original decoder helper SHA256',sha(root/'scripts/cra5/decode_conus.py'),flush=True)
 api=original.cra5_api(local_root=str(root/'cra5'));print(original.sdpa_patch.apply(),flush=True)
 name_to_idx,direct,q_idx=original.build_index(api);levels=[float(x) for x in api.cfg.pressure_level];order=np.argsort(levels)
 pressures=torch.tensor([levels[i]*100 for i in order],dtype=torch.float32,device=api.device);q_order=[q_idx[i] for i in order]
 take=[direct[n] for n in original.FCN20 if n!='tcwv'];start=time.time();completed=[]
 status={'status':'decoding','completed_frames':0,'expected_frames':1460,'codec_weight_sha256':EXPECTED_WEIGHT}
 for i,stamp in enumerate(manifest['expected_timestamps']):
  binfile=source/'2022'/f'{stamp}.bin';path=out/f'{stamp}.npy';receipt=path.with_suffix('.json')
  expected=manifest['source_files'][i]['lfs']['oid']
  try:r=json.loads(receipt.read_text()) if receipt.exists() else {}
  except json.JSONDecodeError:r={}
  if path.exists() and r.get('timestamp')==stamp and r.get('codec_weight_sha256')==EXPECTED_WEIGHT and r.get('bin_sha256')==expected and r.get('channels')==original.FCN20 and sha(path)==r.get('sha256'):
   completed.append(stamp);continue
  assert sha(binfile)==expected
  with torch.inference_mode():
   result=api.decode_from_bin(custom_path=str(binfile),return_format='de_normalized');x=result['x_hat']
   frame=np.empty((20,97,241),np.float32);frame[:19]=x[take][:,164:261,920:1161].cpu().numpy()
   frame[19]=original.make_tcwv(x,q_idx,name_to_idx['sp'],pressures,q_order)[164:261,920:1161].cpu().numpy()
   assert np.isfinite(frame).all()
  tmp=path.with_suffix('.part')
  with tmp.open('wb') as f:np.save(f,frame)
  tmp.replace(path);atomic_json(receipt,{'timestamp':stamp,'shape':[20,97,241],'sha256':sha(path),'bin_sha256':expected,'codec_weight_sha256':EXPECTED_WEIGHT,'tcwv':'original pressure-clipped q integral','channels':original.FCN20})
  completed.append(stamp)
  status={'status':'decoding','completed_frames':len(completed),'expected_frames':1460,'last_timestamp':stamp,'elapsed_seconds':time.time()-start,'codec_weight_sha256':EXPECTED_WEIGHT}
  atomic_json(out/'2022_decoded_manifest.json',status);print(f'{i+1}/1460 {stamp} seconds={time.time()-start:.1f}',flush=True)
  del result,x
 assert len(completed)==1460
 status.update(status='complete',completed_frames=1460,timestamps=completed)
 atomic_json(out/'2022_decoded_manifest.json',status)
if __name__=='__main__':main()
