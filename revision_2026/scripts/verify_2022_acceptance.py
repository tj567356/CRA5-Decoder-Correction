"""Independent full-year acceptance from executed results and archived receipts."""
import csv
import datetime as dt
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REV=ROOT/'revision_2026'
RES=ROOT/'research/fourcastnet/results/revision_2026'
CK='44c57a0f2a18e92c9c0745cfe75337079627ab8729e93c6ec098a5ddeb007435'
ST='272cc45d231ec349025c4fa1e9f69366ae874739c98299ce666921821f13dba2'
CODEC='812a6d09ff039e4ab468915942e7fd00f11e7c58b839cd1b60fe1d0586a40dad'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert np.allclose(a,b,rtol=1e-12,atol=1e-14),(a,b)

def main():
    stamps=[(dt.datetime(2022,1,1)+dt.timedelta(hours=6*i)).isoformat() for i in range(1460)]
    cache=load(REV/'2022_cache_manifest.json')
    names=cache['channels'];assert len(names)==20 and len(set(names))==20
    assert cache['complete'] and cache['alignment_verified'] and cache['frames']==1460 and cache['year']==2022
    assert cache['stats_sha256']==ST
    folder=RES/'2022_fullyear'
    assert load(folder/'2022_data_manifest.json')==cache
    stampfile=folder/'2022_full_timestamps.csv'
    assert sha(stampfile)==cache['timestamps_sha256']
    rows=list(csv.DictReader(stampfile.open(encoding='utf-8',newline='')))
    assert len(rows)==1460 and [r['timestamp_utc'] for r in rows]==stamps
    assert [int(r['index']) for r in rows]==list(range(1460))
    dl=load(REV/'2022_cra5_download_manifest.json')
    dec=load(REV/'2022_decoded_preparation_manifest.json')
    truth=load(REV/'2022_truth_preparation_manifest.json')
    rh=load(REV/'2022_native_rh_validation.json')
    assert dl['status']=='complete' and dl['verified_count']==1460 and dl['expected_timestamps']==stamps and not dl['errors']
    assert dec['status']=='complete' and dec['completed_frames']==1460 and dec['timestamps']==stamps and dec['codec_weight_sha256']==CODEC
    assert truth['completed_frames']==1460 and truth['timestamps']==stamps and truth['channels']==names
    close(truth['lat'],np.arange(49,24.99,-.25));close(truth['lon'],np.arange(-130,-69.99,.25))
    assert rh['days']==365 and rh['frames']==1460 and rh['channels']==['r500','r850']
    assert rh['all_data_and_receipt_sha256_verified'] and rh['all_daily_timestamps_and_coordinates_verified'] and rh['all_values_finite_and_fill_values_excluded']
    bybit={r['path'].split('/')[-1][:-4]:r['lfs']['oid'] for r in dl['source_files']}
    bytruth={r['timestamp']:r for r in truth['frames']}
    byr={r['date']:r for r in rh['daily_provenance']}
    assert set(bybit)==set(bytruth)==set(stamps) and len(byr)==365
    with tarfile.open(REV/'provenance_2022.tar.gz') as archive:
        archived={m.name:archive.extractfile(m).read() for m in archive if m.isfile()}
        def raw(p):return archived['revision_2026/data/'+p]
        def rec(p):return json.loads(raw(p))
        assert rec('memmap_cra5_2022/manifest.json')==cache
        assert rec('cra5/2022_data_manifest.json')==dl
        checked_days=set()
        for stamp,row in zip(stamps,rows):
            tr=rec(f'era5_2022/{stamp}.json');de=rec(f'cra5_decoded_2022/{stamp}.json')
            assert tr['timestamp']==de['timestamp']==stamp
            assert tr['sha256']==row['truth_sha256']==bytruth[stamp]['sha256']
            assert hashlib.sha256(raw(f'era5_2022/{stamp}.json')).hexdigest()==bytruth[stamp]['receipt_sha256']
            assert de['sha256']==row['decoded_sha256'] and de['codec_weight_sha256']==CODEC and de['bin_sha256']==bybit[stamp]
            assert [x.replace('_','') for x in de['channels']]==names and de['shape']==[20,97,241]
            date=stamp[:10].replace('-','');rr=rec(f'native_rh_2022/{date}_rh.json')
            assert rr['sha256']==row['native_rh_sha256']==byr[date]['data_sha256'] and rr['units']=='%' and rr['date']==date
            assert hashlib.sha256(raw(f'native_rh_2022/{date}_rh.json')).hexdigest()==byr[date]['receipt_sha256']
            if date not in checked_days:
                checked_days.add(date)
                day=dt.datetime.fromisoformat(stamp[:10]);expected=[(day+dt.timedelta(hours=h)).isoformat() for h in [0,6,12,18]]
                assert rr['timestamps']==expected and rr['pressure_levels_hpa']==[500,850]
                close(rr['latitude'],truth['lat']);close(rr['longitude'],truth['lon'])
                assert rr['time_units']=='hours since 1900-01-01 00:00:00' and rr['calendar']=='gregorian'
    summary=load(RES/'eval2022/summary.json');assert summary==load(folder/'2022_full_summary.json')
    log=(RES/'2022_eval.log').read_text(encoding='utf-8')
    assert json.loads(log[log.index('{'):])==summary['groups']
    assert 'FULL 1460-frame cache ready' in (RES/'2022_normalization.log').read_text(encoding='utf-8')
    original=load(RES/'eval2018/summary.json')
    assert summary['checkpoint_sha256']==original['checkpoint_sha256']==CK
    assert summary['stats_sha256']==original['stats_sha256']==ST
    assert summary['frames']==1460 and summary['parameters']==1646548
    d=np.load(RES/'eval2022/sufficient_statistics.npz')
    assert int(d['frames'])==1460 and int(d['year'])==2022 and d['channel_names'].astype(str).tolist()==names
    assert d['before_time'].shape==d['after_time'].shape==d['valid_count'].shape==(1460,20)
    assert np.all(d['valid_count']==97*241)
    for key in ['before_time','after_time','before_map','after_map']:assert np.isfinite(d[key]).all() and (d[key]>=0).all()
    close(d['before_time'].sum(0),d['before_map'].sum((1,2)));close(d['after_time'].sum(0),d['after_map'].sum((1,2)))
    b=np.sqrt(d['before_time'].sum(0)/(1460*97*241));a=np.sqrt(d['after_time'].sum(0)/(1460*97*241))
    per=list(csv.DictReader((folder/'2022_full_per_channel.csv').open(encoding='utf-8')))
    assert [r['channel'] for r in per]==names
    near=['u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850']
    upper=['u500','v500','z500','t500','r500','z50']
    groups={'all20':list(range(20)),'direct19':[i for i,n in enumerate(names) if n!='tcwv'],
            'near':[names.index(n) for n in near],'upper':[names.index(n) for n in upper],'tcwv':[names.index('tcwv')]}
    for i,n in enumerate(names):
        close(b[i],summary['channels'][n]['before_nrmse']);close(a[i],summary['channels'][n]['after_nrmse'])
        close(float(per[i]['before_nrmse']),b[i]);close(float(per[i]['after_nrmse']),a[i]);close(float(per[i]['gain_pct']),100*(1-a[i]/b[i]))
    for g,ii in groups.items():
        entry=summary['groups'][g];close(entry['before_nrmse'],b[ii].mean());close(entry['after_nrmse'],a[ii].mean())
        close(entry['G_pct'],100*(1-a[ii].sum()/b[ii].sum()));close(entry['Gbar_pct'],100*np.mean(1-a[ii]/b[ii]))
    close(summary['groups']['direct19']['before_nrmse'],(13*summary['groups']['near']['before_nrmse']+6*summary['groups']['upper']['before_nrmse'])/19)
    close(summary['groups']['direct19']['after_nrmse'],(13*summary['groups']['near']['after_nrmse']+6*summary['groups']['upper']['after_nrmse'])/19)
    v500=float(per[names.index('v500')]['gain_pct']);assert v500<0
    ci=list(csv.DictReader((RES/'bootstrap2022/confidence_intervals.csv').open(encoding='utf-8')))
    for r in ci:
        estimate=summary['groups'][r['group']]['G_pct'] if r['group']!='near_minus_upper' else summary['groups']['near']['G_pct']-summary['groups']['upper']['G_pct']
        close(float(r['estimate_pct']),estimate)
        assert float(r['lower_95_pct'])<estimate<float(r['upper_95_pct'])
    report={'status':'passed_independent_executed_full_year_acceptance','year':2022,'frames':1460,'first_timestamp':stamps[0],'last_timestamp':stamps[-1],
            'missing_timestamps':0,'duplicate_timestamps':0,'truth_receipts_checked':1460,'decoded_receipts_checked':1460,'native_rh_receipts_checked':365,
            'receipt_archive_sha256':sha(REV/'provenance_2022.tar.gz'),'cache_manifest_sha256':sha(REV/'2022_cache_manifest.json'),
            'timestamp_file_sha256':sha(stampfile),'checkpoint_sha256':CK,'stats_sha256':ST,'codec_weight_sha256':CODEC,
            'metric_recomputation':'all20 per-channel nRMSE and5groups recomputed from actual temporal SSE; map/time SSE consistency; CSV/JSON equality',
            'execution_logs':'evaluation stdout groups exactly match summary; normalization log confirms full1460cache completion',
            'v500_gain_pct_preserved':v500,'native_rh_metadata_scopes':rh['metadata_scopes'],
            'verification_scope':'Independent local receipt-chain and sufficient-statistics audit; original large cache/data hashes are attested by completed assembly/evaluation/producer manifests, not redownloaded and rehashed locally.',
            'groups':summary['groups']}
    (REV/'2022_acceptance_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    manifest_path=REV/'data_manifest.json';manifest=load(manifest_path)
    oldrecords=[r for r in manifest['records'] if r['year']!=2022]
    assert [r['year'] for r in oldrecords]==[2014,2015,2016,2017,2018]
    oldbytes=json.dumps(oldrecords,sort_keys=True)
    entry={'year':2022,'truth':cache['truth'],'decoded':cache['decoded'],'native_rh':cache['native_rh'],
           'truth_shape':[1460,20,97,241],'frames':1460,'channels':names,'alignment_verified':True,
           'timestamp_first':stamps[0],'timestamp_last':stamps[-1],'missing_timestamps':0,'duplicate_timestamps':0,
           'validation':'executed full-year frozen evaluation; independent full receipt chain and SSE/CSV/JSON metric reconstruction',
           'cache_manifest':'revision_2026/2022_cache_manifest.json','acceptance_report':'revision_2026/2022_acceptance_verification.json',
           'source_manifests':['revision_2026/2022_truth_preparation_manifest.json','revision_2026/2022_cra5_download_manifest.json','revision_2026/2022_decoded_preparation_manifest.json','revision_2026/2022_native_rh_validation.json'],
           'stats_sha256':ST,'checkpoint_sha256':CK,'codec_weight_sha256':CODEC,
           'input_sha256':cache['input_sha256'],'target_sha256':cache['target_sha256'],'timestamps_sha256':cache['timestamps_sha256'],
           'native_rh_metadata_scopes':rh['metadata_scopes'],'source_precision_note':'2018 validation:18 ARCOchannels small archivepacking differences;2nativeRH channels bitwise equal; all2022fields native ERA5, frozen float32 arithmetic then float16cache',
           'status':'newly_executed_and_independently_verified'}
    manifest['records']=oldrecords+[entry]
    assert json.dumps(manifest['records'][:5],sort_keys=True)==oldbytes
    tmp=manifest_path.with_suffix('.json.tmp');tmp.write_text(json.dumps(manifest,indent=2),encoding='utf-8');tmp.replace(manifest_path)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
