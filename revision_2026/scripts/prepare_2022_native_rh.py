"""Extract native ERA5 RH via verified HTTPS HDF5 range reads from NCAR GDEX."""
import argparse, datetime as dt, hashlib, io, json, sys, time
from collections import OrderedDict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deps2022'))
import h5py,httpx,numpy as np

class RangeFile(io.RawIOBase):
 def __init__(self,url,block=262144):
  self.url=url;self.pos=0;self.block=block;self.cache=OrderedDict();self.proof=[];self.client=httpx.Client(timeout=120,follow_redirects=True)
  head=self.fetch(0,64);assert head[:8]==b'\x89HDF\r\n\x1a\n' and head[8]==0 and head[13]==8,head
  self.length=int.from_bytes(head[40:48],'little')
 def readable(self):return True
 def seekable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.length+offset
  return self.pos
 def fetch(self,start,stop):
  for attempt in range(6):
   try:
    r=self.client.get(self.url,headers={'Range':f'bytes={start}-{stop-1}'});r.raise_for_status()
    assert r.status_code==206 and len(r.content)==stop-start,(r.status_code,len(r.content),start,stop)
    self.proof.append({'start':start,'stop':stop,'sha256':hashlib.sha256(r.content).hexdigest()});return r.content
   except Exception:
    if attempt==5:raise
    time.sleep(2**attempt)
 def read(self,size=-1):
  if size<0:size=self.length-self.pos
  if size>self.block*2:
   stop=min(self.length,self.pos+size);data=self.fetch(self.pos,stop);self.pos=stop;return data
  stop=min(self.length,self.pos+size);result=[]
  while self.pos<stop:
   b=self.pos//self.block
   if b not in self.cache:
    self.cache[b]=self.fetch(b*self.block,min(self.length,(b+1)*self.block))
    if len(self.cache)>128:self.cache.popitem(last=False)
   data=self.cache[b];off=self.pos%self.block;n=min(stop-self.pos,len(data)-off);result.append(data[off:off+n]);self.pos+=n
  return b''.join(result)
 def readinto(self,buffer):
  data=self.read(len(buffer));buffer[:len(data)]=data;return len(data)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--year',type=int,default=2022);p.add_argument('--days',type=int,default=365);p.add_argument('--start-day',type=int,default=0);p.add_argument('--step',type=int,default=1);p.add_argument('--inspect-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 for d in range(a.start_day,a.days,a.step):
  day=dt.datetime(a.year,1,1)+dt.timedelta(days=d);date=day.strftime('%Y%m%d');month=day.strftime('%Y%m')
  url=f'https://data.gdex.ucar.edu/d633000/e5.oper.an.pl/{month}/e5.oper.an.pl.128_157_r.ll025sc.{date}00_{date}23.nc'
  out=a.out/f'{date}_rh.npz';receipt=out.with_suffix('.json')
  try:record=json.loads(receipt.read_text()) if receipt.exists() else {}
  except json.JSONDecodeError:record={}
  if out.exists() and record.get('date')==date and record.get('url')==url and record.get('units')=='%' and hashlib.sha256(out.read_bytes()).hexdigest()==record.get('sha256'):continue
  remote=RangeFile(url)
  with h5py.File(remote,'r',rdcc_nbytes=200*1024**2) as f:
   if a.inspect_only:
    print([(k,v.shape,dict(v.attrs)) for k,v in f.items()],flush=True);return
   rh=f['R'];print(date,'shape',rh.shape,'chunks',rh.chunks,'units',rh.attrs['units'],flush=True)
   levels=f['level'][:];lats=f['latitude'][:];lons=f['longitude'][:]
   assert np.array_equal(lats[164:261],np.arange(49,24.9,-.25)) and np.array_equal(lons[920:1161]-360,np.arange(-130,-69.9,.25))
   assert f['time'].attrs['units'] in [b'hours since 1900-01-01 00:00:00','hours since 1900-01-01 00:00:00']
   assert f['time'].attrs['calendar'] in [b'gregorian','gregorian']
   times=f['time'][:];epoch=dt.datetime(1900,1,1)
   assert all(epoch+dt.timedelta(hours=int(times[h]))==day+dt.timedelta(hours=h) for h in [0,6,12,18])
   indices=[int(np.flatnonzero(levels==p)[0]) for p in [500,850]]
   frames=[]
   for hour in [0,6,12,18]:
    values=[]
    for level in indices:
     x=rh[hour,level,164:261,920:1161].astype(np.float32)
     assert np.isfinite(x).all() and (abs(x)<1000).all();values.append(x)
    frames.append(values)
   data=np.array(frames,dtype=np.float32);assert data.shape==(4,2,97,241)
   assert rh.attrs['units'] in [b'%', '%'],dict(rh.attrs)
  tmp=out.with_suffix('.npz.part')
  with tmp.open('wb') as f:np.savez(f,fields=data,timestamps=np.array([(day+dt.timedelta(hours=h)).isoformat() for h in [0,6,12,18]]),channels=np.array(['r500','r850']))
  tmp.replace(out);receipt_tmp=receipt.with_suffix('.json.part')
  receipt_tmp.write_text(json.dumps({'date':date,'url':url,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'units':'%','source':'native ECMWF ERA5 relative humidity distributed by NCAR GDEX','ranges':remote.proof},indent=2));receipt_tmp.replace(receipt);remote.client.close()
  print(f'{d+1}/{a.days} {date} complete; range_bytes={sum(x["stop"]-x["start"] for x in remote.proof)}',flush=True)
if __name__=='__main__':main()
