"""Fetch actual ERA5 z/lsm from Google ARCO; preserve metadata and raw chunk hashes."""
import hashlib
import json
import urllib.request
from pathlib import Path
import numpy as np
import numcodecs

BASE='https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
OUT=Path('/home/tong/fourcastnet_us_mpca/revision_2026/static')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metadata=json.load(urllib.request.urlopen(BASE+'/.zmetadata',timeout=60))
    (OUT/'arco_metadata.json').write_text(json.dumps(metadata))
    m=metadata['metadata']
    # Select 2018-01-01 from encoded hours since epoch (read, do not assume).
    print(m['time/.zattrs'],flush=True)
    units=m['time/.zattrs']['units']
    assert units.startswith('hours since '),units
    origin=np.datetime64(units.split('since ')[1].split('.')[0])
    time_index=int((np.datetime64('2018-01-01T00:00:00')-origin)/np.timedelta64(1,'h'))
    provenance={'base_url':BASE,'time_index':time_index,'timestamp':'2018-01-01T00:00:00','chunks':{}}
    def get(var,key):
        raw=urllib.request.urlopen(f'{BASE}/{var}/{key}',timeout=120).read()
        (OUT/f'{var}_{key}.chunk').write_bytes(raw)
        z=m[var+'/.zarray']
        data=numcodecs.get_codec(z['compressor']).decode(raw)
        arr=np.frombuffer(data,dtype=z['dtype']).reshape(z['chunks'])
        provenance['chunks'][var]={'url':f'{BASE}/{var}/{key}','sha256':hashlib.sha256(raw).hexdigest(),
                                   'metadata':z,'attrs':m[var+'/.zattrs']}
        return arr
    lat=get('latitude','0'); lon=get('longitude','0')
    assert np.allclose(lat[164:261],np.arange(49,24.99,-.25))
    assert np.allclose(lon[920:1161]-360,np.arange(-130,-69.99,.25))
    z=get('geopotential_at_surface',f'{time_index}.0.0')[0,164:261,920:1161]
    lsm=get('land_sea_mask',f'{time_index}.0.0')[0,164:261,920:1161]
    assert np.isfinite(z).all() and np.isfinite(lsm).all()
    assert lsm.min()>=0 and lsm.max()<=1
    np.savez_compressed(OUT/'era5_static_conus.npz',z=z,lsm=lsm,lat=lat[164:261],lon=lon[920:1161]-360)
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('static ready',z.shape,'land',int((lsm>.5).sum()),flush=True)

if __name__=='__main__': main()
