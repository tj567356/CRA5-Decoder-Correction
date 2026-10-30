"""Exact CONUS extraction from official ARCO ERA5 using Blosc HTTP block ranges.

The native total_column_water_vapour is retained as truth. No regridding or
unit conversion is performed: z is geopotential (m2/s2), r is percent.
All selected compressed blocks are copied unchanged before blosc_getitem.
Run --verify-full once to compare the block extraction to full decompression.
"""
import argparse, concurrent.futures, ctypes, datetime as dt, hashlib, json, math, struct, time
from pathlib import Path
import httpx, numpy as np, numcodecs, numcodecs.blosc

BASE='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/'
CHANNELS=['u10','v10','t2m','sp','msl','t850','u1000','v1000','z1000','u850','v850','z850','u500','v500','z500','t500','z50','r500','r850','tcwv']
FIELDS={
 '10m_u_component_of_wind':[(0,None)],'10m_v_component_of_wind':[(1,None)],
 '2m_temperature':[(2,None)],'surface_pressure':[(3,None)],'mean_sea_level_pressure':[(4,None)],
 'temperature':[(5,850),(15,500)],'u_component_of_wind':[(6,1000),(9,850),(12,500)],
 'v_component_of_wind':[(7,1000),(10,850),(13,500)],
 'geopotential':[(8,1000),(11,850),(14,500),(16,50)],
 'relative_humidity':[(17,500),(18,850)],'total_column_water_vapour':[(19,None)]}
LIB=ctypes.CDLL(numcodecs.blosc.__file__)
GETITEM=LIB.blosc_getitem;GETITEM.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];GETITEM.restype=ctypes.c_int

def digest(data):return hashlib.sha256(data).hexdigest()
def filehash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

class Extractor:
 def __init__(self,verify=False):
  self.client=httpx.Client(timeout=120,follow_redirects=True);self.verify=verify
  r=self.client.get(BASE+'.zmetadata');r.raise_for_status();self.metadata_bytes=r.content;self.meta=r.json()['metadata']
  self.level=self.coord('level');self.lat=self.coord('latitude');self.lon=self.coord('longitude')
  assert np.array_equal(self.lat[164:261],np.arange(49,24.9,-.25))
  assert np.array_equal(self.lon[920:1161]-360,np.arange(-130,-69.9,.25))
  self.epoch=dt.datetime.fromisoformat(self.meta['time/.zattrs']['units'].split('since ')[1])
  for name in FIELDS:
   z=self.meta[name+'/.zarray'];assert z['dtype']=='<f4' and z['filters'] is None and z['order']=='C'
   assert z['chunks'] in [[1,721,1440],[1,37,721,1440]],(name,z)
 def coord(self,name):
  m=self.meta[name+'/.zarray'];r=self.client.get(BASE+name+'/0');r.raise_for_status()
  return np.frombuffer(numcodecs.get_codec(m['compressor']).decode(r.content),dtype=m['dtype'])[:m['shape'][0]]
 def range(self,url,start,stop,generation=None):
  params={'generation':generation} if generation else None
  for attempt in range(6):
   try:
    r=self.client.get(url,params=params,headers={'Range':f'bytes={start}-{stop-1}'})
    r.raise_for_status();assert r.status_code==206 and len(r.content)==stop-start,(r.status_code,len(r.content),start,stop)
    return r
   except Exception:
    if attempt==5:raise
    time.sleep(2**attempt)
 def extract(self,name,stamp):
  idx=int((stamp-self.epoch).total_seconds()/3600);spec=FIELDS[name];z=self.meta[name+'/.zarray']
  url=BASE+name+'/'+str(idx)+('.0.0.0' if len(z['shape'])==4 else '.0.0')
  head=self.range(url,0,16);generation=head.headers.get('x-goog-generation');header=head.content
  version,compversion,flags,typesize,nbytes,blocksize,cbytes=struct.unpack('<4B3I',header)
  assert version==2 and typesize==4 and not(flags&2),'Unsupported Blosc format or memcpy chunk'
  nblocks=math.ceil(nbytes/blocksize);tab=self.range(url,16,16+4*nblocks,generation).content
  offsets=struct.unpack('<'+'I'*nblocks,tab);ends=dict(zip(sorted(offsets),sorted(offsets)[1:]+[cbytes]))
  jobs=[];blocks=set()
  for channel,level in spec:
   lev=0 if level is None else int(np.flatnonzero(self.level==level)[0])
   start=(lev*721+164)*1440;count=97*1440
   blocks.update(range(start*4//blocksize,math.ceil((start+count)*4/blocksize)))
   jobs.append((channel,start,count))
  intervals=sorted((offsets[i],ends[offsets[i]]) for i in blocks);merged=[]
  for start,stop in intervals:
   if merged and merged[-1][1]==start:merged[-1]=(merged[-1][0],stop)
   else:merged.append((start,stop))
  compressed=bytearray(cbytes);compressed[:16]=header;compressed[16:16+len(tab)]=tab;proof=[]
  for start,stop in merged:
   r=self.range(url,start,stop,generation);compressed[start:stop]=r.content
   proof.append({'start':start,'stop':stop,'sha256':digest(r.content)})
  src=(ctypes.c_char*len(compressed)).from_buffer(compressed);values={}
  for channel,start,count in jobs:
   out=np.empty(count,dtype=np.float32);n=GETITEM(src,start,count,out.ctypes.data);assert n==count*4,(name,n)
   selected=out.reshape(97,1440)[:,920:1161].copy();assert np.isfinite(selected).all(),name;values[channel]=selected
  if self.verify:
   r=self.client.get(url,params={'generation':generation});r.raise_for_status()
   full=np.frombuffer(numcodecs.get_codec(z['compressor']).decode(r.content),dtype=np.float32).reshape(z['chunks'])
   for channel,level in spec:
    lev=0 if level is None else int(np.flatnonzero(self.level==level)[0]);expected=full[0,lev,164:261,920:1161] if level is not None else full[0,164:261,920:1161]
    assert np.array_equal(values[channel],expected),'Partial/full mismatch'
  return values,{'variable':name,'url':url,'generation':generation,'etag':head.headers.get('etag'),'header_sha256':digest(header+tab),'compressed_bytes':cbytes,'selected_ranges':proof,'units':self.meta[name+'/.zattrs'].get('units'),'partial_equals_full':self.verify}

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--year',type=int,default=2022);p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=3);p.add_argument('--start-index',type=int,default=0);p.add_argument('--stride',type=int,default=1);p.add_argument('--verify-full',action='store_true');p.add_argument('--skip-rh',action='store_true',help='Explicit incomplete preparation: leave unavailable native r500/r850 as NaN');a=p.parse_args()
 if a.skip_rh:FIELDS.pop('relative_humidity',None)
 a.out.mkdir(parents=True,exist_ok=True);ext=Extractor(a.verify_full)
 stamps=[dt.datetime(a.year,1,1)+dt.timedelta(hours=6*i) for i in range(1460)]
 if a.limit:stamps=stamps[:a.limit]
 stamps=stamps[a.start_index::a.stride]
 manifest_path=a.out/(f'{a.year}_truth_manifest.json' if a.stride==1 else f'{a.year}_truth_shard_{a.start_index}_of_{a.stride}.json')
 (a.out/'arco_metadata.json').write_bytes(ext.metadata_bytes)
 manifest={'source':BASE,'metadata_sha256':digest(ext.metadata_bytes),'year':a.year,'expected_frames':1460,'requested_frames':len(stamps),'channels':CHANNELS,'missing_channels':['r500','r850'] if a.skip_rh else [],'units':{k:ext.meta[k+'/.zattrs'].get('units') for k in FIELDS},'lat':ext.lat[164:261].tolist(),'lon':(ext.lon[920:1161]-360).tolist(),'tcwv':'native ERA5 total_column_water_vapour','status':'downloading'}
 manifest.update(start_index=a.start_index,stride=a.stride)
 manifest_path.write_text(json.dumps(manifest,indent=2))
 complete=[];start=time.time()
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for i,stamp in enumerate(stamps):
   path=a.out/f'{stamp.isoformat()}.npy';receipt=path.with_suffix('.json')
   if path.exists() and receipt.exists() and filehash(path)==json.loads(receipt.read_text())['sha256']:
    complete.append(stamp.isoformat());continue
   frame=np.full((20,97,241),np.nan,np.float32);proof=[]
   for values,record in pool.map(lambda n:ext.extract(n,stamp),FIELDS):
    for ch,val in values.items():frame[ch]=val
    proof.append(record)
   assert frame[19].min()>=0 and frame[3].min()>20000 and frame[2].min()>150
   tmp=path.with_suffix('.part')
   with tmp.open('wb') as f:np.save(f,frame)
   tmp.replace(path)
   receipt.write_text(json.dumps({'timestamp':stamp.isoformat(),'sha256':filehash(path),'shape':list(frame.shape),'provenance':proof},indent=2))
   complete.append(stamp.isoformat());manifest.update(completed_frames=len(complete),last_timestamp=complete[-1],elapsed_seconds=time.time()-start)
   manifest_path.write_text(json.dumps(manifest,indent=2))
   print(f'{i+1}/{len(stamps)} {stamp.isoformat()} seconds={time.time()-start:.1f}',flush=True)
 manifest.update(status='incomplete_native_rh_missing' if a.skip_rh else ('complete' if len(complete)==1460 else 'partial_validation'),completed_frames=len(complete),timestamps=complete)
 manifest_path.write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
