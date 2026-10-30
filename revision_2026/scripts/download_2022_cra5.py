"""Fetch exactly the full-year six-hourly official CRA5 files, verified by LFS SHA256.

Only writes under the explicit output root. Existing files are hash-validated.
"""
import argparse, concurrent.futures, datetime as dt, hashlib, json, time
from pathlib import Path
import httpx

REPO='taohan10200/CRA5-Dataset'
def atomic_json(path,data):
 tmp=path.with_suffix('.json.part');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=True)
 with httpx.Client(follow_redirects=True,timeout=120) as c:
  r=c.get(f'https://huggingface.co/api/datasets/{REPO}');r.raise_for_status();rev=r.json()['sha']
  url=f'https://huggingface.co/api/datasets/{REPO}/tree/{rev}/2022?limit=1000';entries=[]
  while url:
   r=c.get(url);r.raise_for_status();entries.extend(r.json());url=r.links.get('next',{}).get('url')
 wanted=[(dt.datetime(2022,1,1)+dt.timedelta(hours=6*i)).isoformat() for i in range(1460)]
 lookup={e['path']:e for e in entries if e['type']=='file'}
 selected=[lookup[f'2022/{t}.bin'] for t in wanted]
 manifest={'repo':REPO,'revision':rev,'expected_count':1460,'expected_timestamps':wanted,'source_files':selected,'status':'downloading'}
 dest=a.out/'2022_data_manifest.json';atomic_json(dest,manifest)
 def fetch(e):
  path=a.out/e['path'];path.parent.mkdir(parents=True,exist_ok=True);expected=e['lfs']['oid']
  if path.exists() and path.stat().st_size==e['size'] and sha(path)==expected:return e['path'],'verified_existing',expected
  for attempt in range(6):
   try:
    tmp=path.with_suffix('.bin.part')
    with httpx.stream('GET',f'https://huggingface.co/datasets/{REPO}/resolve/{rev}/{e["path"]}',follow_redirects=True,timeout=120) as r:
     r.raise_for_status()
     with tmp.open('wb') as f:
      for b in r.iter_bytes():f.write(b)
    actual=sha(tmp)
    if tmp.stat().st_size!=e['size'] or actual!=expected:raise ValueError('size or SHA256 mismatch')
    tmp.replace(path);return e['path'],'downloaded_verified',actual
   except Exception:
    if attempt==5:raise
    time.sleep(2**attempt)
 errors=[];success=[];start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
  futures={ex.submit(fetch,e):e for e in selected}
  for f in concurrent.futures.as_completed(futures):
   try:
    item=f.result();success.append(item)
    with (a.out/'2022_cra5_verified.jsonl').open('a') as log:log.write(json.dumps({'path':item[0],'status':item[1],'sha256':item[2]})+'\n')
   except Exception as e:errors.append({'path':futures[f]['path'],'error':repr(e)})
   n=len(success)+len(errors)
   if n%20==0 or n==1460:print(f'{n}/1460; verified {len(success)}; errors {len(errors)}; seconds {time.time()-start:.0f}',flush=True)
 manifest.update(status='complete' if len(success)==1460 else 'incomplete',verified_count=len(success),errors=errors,elapsed_seconds=time.time()-start)
 atomic_json(dest,manifest)
 if errors:raise SystemExit(1)
if __name__=='__main__':main()
