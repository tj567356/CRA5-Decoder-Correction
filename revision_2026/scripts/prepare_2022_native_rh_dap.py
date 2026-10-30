"""Exact NCAR native ERA5 RH through official OPeNDAP server-side subsets.

Validates the returned time/level/latitude/longitude arrays for every day.
Uses the same atomic output schema as the HDF5 range-reader fallback.
"""
import argparse,datetime as dt,hashlib,json,os,re,time,urllib.parse
from pathlib import Path
import httpx,numpy as np
DIRECT='https://data.gdex.ucar.edu/d633000/'
DAP='https://tds.gdex.ucar.edu/thredds/dodsC/files/g/d633000/'
QUERY='R.R[0:6:18][21:9:30][164:1:260][920:1:1160],time[0:6:18],level[21:9:30],latitude[164:1:260],longitude[920:1:1160]'

def monthly_metadata(client,cache,month,url):
 """Verify unit/calendar schema once per calendar month, retaining actual origin."""
 cache.mkdir(parents=True,exist_ok=True);path=cache/f'{month}.das';receipt=cache/f'{month}.json';lock=cache/f'{month}.lock'
 while True:
  if path.exists() and receipt.exists():
   record=json.loads(receipt.read_text());payload=path.read_bytes()
   if hashlib.sha256(payload).hexdigest()==record['sha256']:return payload,record
  try:
   descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
   os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
  except FileExistsError:
   try:
    holder=int(lock.read_text());os.kill(holder,0)
   except ProcessLookupError:
    lock.unlink(missing_ok=True);continue
   except (ValueError,FileNotFoundError):pass
   time.sleep(1);continue
  try:
   response=client.get(url);response.raise_for_status();text=response.text
   assert re.search(r'R\s*\{[^}]*String units "%";',text,re.S),text[:2000]
   assert re.search(r'time\s*\{[^}]*hours since 1900-01-01 00:00:00',text,re.S)
   assert re.search(r'time\s*\{[^}]*String calendar "gregorian"',text,re.S)
   record={'url':url,'sha256':hashlib.sha256(response.content).hexdigest(),'scope':'native ECMWF ERA5 parameter 157 file family, same calendar month','month':month,'verified_units':'%','verified_time_units':'hours since 1900-01-01 00:00:00','verified_calendar':'gregorian'}
   temp=path.with_suffix('.das.part');temp.write_bytes(response.content);temp.replace(path)
   temp=receipt.with_suffix('.json.part');temp.write_text(json.dumps(record,indent=2));temp.replace(receipt)
   return response.content,record
  finally:lock.unlink(missing_ok=True)

def parse(payload,expected_names=None):
 header,binary=payload.split(b'\nData:\n',1);header=header.decode();result={};offset=0
 for kind,name,dims in re.findall(r'(Float32|Float64|Int32)\s+(\w+)((?:\[[^\]]+\])+);',header):
  shape=tuple(map(int,re.findall(r'=\s*(\d+)',dims)));count=int(np.prod(shape));length=np.frombuffer(binary[offset:offset+8],dtype='>u4');offset+=8
  assert list(length)==[count,count],(name,length,shape)
  dtype={'Float32':'>f4','Float64':'>f8','Int32':'>i4'}[kind];size=count*np.dtype(dtype).itemsize
  result[name]=np.frombuffer(binary[offset:offset+size],dtype=dtype).reshape(shape).copy();offset+=size
 assert offset==len(binary),(offset,len(binary));assert set(result)==(expected_names or {'R','time','level','latitude','longitude'}),header
 return result,header

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--year',type=int,default=2022);p.add_argument('--days',type=int,default=365);p.add_argument('--start-day',type=int,default=0);p.add_argument('--step',type=int,default=1);p.add_argument('--compare',type=Path);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 with httpx.Client(timeout=240,follow_redirects=True) as client:
  for d in range(a.start_day,a.days,a.step):
   day=dt.datetime(a.year,1,1)+dt.timedelta(days=d);date=day.strftime('%Y%m%d');month=day.strftime('%Y%m');relative=f'e5.oper.an.pl/{month}/e5.oper.an.pl.128_157_r.ll025sc.{date}00_{date}23.nc'
   direct=DIRECT+relative;url=DAP+relative+'.dods?'+urllib.parse.quote(QUERY,safe=':,.');out=a.out/f'{date}_rh.npz';receipt=out.with_suffix('.json')
   try:record=json.loads(receipt.read_text()) if receipt.exists() else {}
   except json.JSONDecodeError:record={}
   existing_valid=out.exists() and record.get('date')==date and record.get('url')==direct and record.get('units')=='%' and hashlib.sha256(out.read_bytes()).hexdigest()==record.get('sha256')
   if existing_valid and all(k in record for k in ['timestamps','pressure_levels_hpa','latitude','longitude','time_units','calendar']):continue
   metadata_only=bool(existing_valid)
   if metadata_only:url=DAP+relative+'.dods?'+urllib.parse.quote(QUERY.split(',',1)[1],safe=':,.')
   started=time.time()
   for attempt in range(6):
    try:
     metadata_payload,metadata_record=monthly_metadata(client,a.out.parent/'native_rh_metadata_2022',month,DAP+relative+'.das')
     response=client.get(url);response.raise_for_status();values,header=parse(response.content,{'time','level','latitude','longitude'} if metadata_only else None);break
    except Exception as error:
     print(f'{date} attempt {attempt+1}: {error!r}',flush=True)
     if attempt==5:raise
     time.sleep(2**attempt)
   assert np.array_equal(values['latitude'],np.arange(49,24.9,-.25))
   assert np.array_equal(values['longitude']-360,np.arange(-130,-69.9,.25))
   assert np.array_equal(values['level'],[500,850])
   timestamps=[(dt.datetime(1900,1,1)+dt.timedelta(hours=int(h))).isoformat() for h in values['time']]
   expected=[(day+dt.timedelta(hours=h)).isoformat() for h in [0,6,12,18]];assert timestamps==expected
   data=np.load(out)['fields'] if metadata_only else values['R'].astype(np.float32);assert data.shape==(4,2,97,241) and np.isfinite(data).all() and (abs(data)<1000).all()
   if a.compare:
    old=np.load(a.compare)['fields'];assert np.array_equal(data,old),'OPeNDAP/full-HDF native-RH mismatch';print('BITWISE equality to full-HDF extraction confirmed for all8fields',flush=True)
   if not metadata_only:
    tmp=out.with_suffix('.npz.part')
    with tmp.open('wb') as f:np.savez(f,fields=data,timestamps=np.array(timestamps),channels=np.array(['r500','r850']))
    tmp.replace(out)
   previous=dict(record)
   record={'date':date,'url':direct,'access_url':url,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'units':'%','source':'native ECMWF ERA5 relative humidity distributed by NCAR GDEX; HDF5 fields + exact OPeNDAP coordinate verification' if metadata_only else 'native ECMWF ERA5 relative humidity distributed by NCAR GDEX; exact OPeNDAP subset','source_response_sha256':hashlib.sha256(response.content).hexdigest(),'source_response_bytes':len(response.content),'source_metadata_sha256':hashlib.sha256(metadata_payload).hexdigest(),'source_metadata_url':metadata_record['url'],'source_metadata_scope':metadata_record['scope'],'source_metadata_month':month,'time_units':'hours since 1900-01-01 00:00:00','calendar':'gregorian','timestamps':timestamps,'pressure_levels_hpa':[500,850],'latitude':values['latitude'].tolist(),'longitude':(values['longitude']-360).tolist(),'dap_header':header,'verified_against_full_hdf':bool(a.compare)}
   if metadata_only:record['previous_hdf_provenance']=previous
   tmp=receipt.with_suffix('.json.part');tmp.write_text(json.dumps(record,indent=2));tmp.replace(receipt)
   print(f'{d+1}/{a.days} {date} exact nativeRH complete; seconds={time.time()-started:.2f}; bytes={len(response.content)}',flush=True)
if __name__=='__main__':main()
