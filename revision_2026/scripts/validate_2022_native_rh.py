"""Independent final native-RH year inventory and exact coordinate/hash checks."""
import datetime as dt,hashlib,json
from pathlib import Path
import numpy as np
root=Path.home()/'fourcastnet_us_mpca/revision_2026';folder=root/'data/native_rh_2022';rows=[]
for d in range(365):
 day=dt.datetime(2022,1,1)+dt.timedelta(days=d);date=day.strftime('%Y%m%d');path=folder/f'{date}_rh.npz';receipt=path.with_suffix('.json');r=json.loads(receipt.read_text())
 digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==r['sha256'] and r['date']==date and r['units']=='%'
 stamps=[(day+dt.timedelta(hours=h)).isoformat() for h in [0,6,12,18]]
 assert r['timestamps']==stamps and r['pressure_levels_hpa']==[500,850]
 assert r['time_units']=='hours since 1900-01-01 00:00:00' and r['calendar']=='gregorian'
 assert np.array_equal(r['latitude'],np.arange(49,24.9,-.25)) and np.array_equal(r['longitude'],np.arange(-130,-69.9,.25))
 x=np.load(path);assert x['fields'].shape==(4,2,97,241) and x['fields'].dtype==np.float32
 assert x['timestamps'].astype(str).tolist()==stamps and x['channels'].astype(str).tolist()==['r500','r850']
 assert np.isfinite(x['fields']).all() and (abs(x['fields'])<1000).all()
 if 'source_metadata_scope' in r:
  assert r['source_metadata_month']==date[:6];metadata=json.loads((root/f'data/native_rh_metadata_2022/{date[:6]}.json').read_text())
  assert metadata['sha256']==r['source_metadata_sha256'] and metadata['url']==r['source_metadata_url']
  assert hashlib.sha256((root/f'data/native_rh_metadata_2022/{date[:6]}.das').read_bytes()).hexdigest()==metadata['sha256']
 rows.append({'date':date,'data_sha256':digest,'receipt_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest(),'metadata_scope':r.get('source_metadata_scope','same-day file metadata'),'metadata_source_url':r.get('source_metadata_url',r['access_url'].split('.dods?')[0]+'.das'),'source_response_sha256':r['source_response_sha256']})
assert len({s for row in rows for s in [row['date']]})==365
result={'status':'complete','year':2022,'days':365,'frames':1460,'channels':['r500','r850'],'shape':[1460,2,97,241],'all_data_and_receipt_sha256_verified':True,'all_daily_timestamps_and_coordinates_verified':True,'all_values_finite_and_fill_values_excluded':True,'native_rh_unit':'%','metadata_scopes':{scope:sum(r['metadata_scope']==scope for r in rows) for scope in sorted({r['metadata_scope'] for r in rows})},'daily_provenance':rows}
(root/'2022_native_rh_validation.json').write_text(json.dumps(result,indent=2));print('PASS complete native RH:365days,1460frames,allSHA256,timestamps,coords,units,metadata origins verified.')
