"""Wait for independently downloaded full-year inputs, then normalize/evaluate exactly once."""
import datetime as dt
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path('/home/tong/fourcastnet_us_mpca')
REV=ROOT/'revision_2026'
SCRIPTS=REV/'scripts'
RESULTS=ROOT/'results/revision_2026'
STAMPS=[(dt.datetime(2022,1,1)+dt.timedelta(hours=6*i)).isoformat() for i in range(1460)]

def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}

def run(label,args):
    print('START',label,flush=True)
    with (RESULTS/f'{label}.log').open('x') as log:
        subprocess.run([sys.executable,*map(str,args)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('DONE',label,flush=True)

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    truth=REV/'data/era5_2022';decoded=REV/'data/cra5_decoded_2022';rh=REV/'data/native_rh_2022'
    last=None
    while True:
        tm=read(truth/'2022_truth_manifest.json');dm=read(decoded/'2022_decoded_manifest.json')
        nrh=sum((rh/f'{(dt.datetime(2022,1,1)+dt.timedelta(days=d)).strftime("%Y%m%d")}_rh.npz').exists() and
                 (rh/f'{(dt.datetime(2022,1,1)+dt.timedelta(days=d)).strftime("%Y%m%d")}_rh.json').exists() for d in range(365))
        counts=(sum((truth/f'{t}.npy').exists() for t in STAMPS),dm.get('completed_frames',0),nrh)
        if counts!=last:
            print(dt.datetime.now(dt.timezone.utc).isoformat(),'truth18_frames,decoded20_frames,nativeRH_days',counts,flush=True);last=counts
        if tm.get('completed_frames')==1460 and dm.get('status')=='complete' and nrh==365:break
        time.sleep(30)
    cache=REV/'data/memmap_cra5_2022'
    run('2022_normalization',[SCRIPTS/'assemble_2022_cache.py','--truth',truth,'--decoded',decoded,'--rh',rh,'--out',cache])
    run('2022_eval',[SCRIPTS/'evaluate_frozen.py','--year',2022,'--mem',cache,'--out',RESULTS/'eval2022'])
    run('2022_bootstrap',[SCRIPTS/'bootstrap_gain.py',RESULTS/'eval2022/sufficient_statistics.npz','--out',RESULTS/'bootstrap2022'])
    run('2022_export',[SCRIPTS/'export_2022_results.py','--eval2018',RESULTS/'eval2018','--eval2022',RESULTS/'eval2022','--cache',cache,'--out',RESULTS/'2022_fullyear'])
    (RESULTS/'2022_pipeline_complete.json').write_text(json.dumps({'status':'complete','frames':1460,'training_performed':False,
                                                                 'finished_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2))
    print('FULL-YEAR 2022 EVALUATION COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();sys.exit(1)
