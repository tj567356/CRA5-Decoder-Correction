"""Independent completed-artifact acceptance for the fixed CRA5 RF ablation.

NumPy/stdlib only; no Torch, GPU, training cache arrays or model execution.
Recomputes metrics and paired bootstrap from saved sufficient statistics, and
checks selection/structural/gradient-ERF/export consistency. Never assumes that
Dilated or affine models improve. Writes acceptance.json only after all checks.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

NAMES = ['u10','v10','t2m','sp','msl','t850','u1000','v1000','z1000',
         'u850','v850','z850','u500','v500','z500','t500','z50','r500','r850','tcwv']
NEAR = ['u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850']
UPPER = ['u500','v500','z500','t500','r500','z50']
ORIGINAL_HASH = '44c57a0f2a18e92c9c0745cfe75337079627ab8729e93c6ec098a5ddeb007435'
STATS_HASH = '272cc45d231ec349025c4fa1e9f69366ae874739c98299ce666921821f13dba2'
AFFINE_HASH = '31743a1a03669a6b20f7a39c79c7ae3738fa059468ea789a56e6ab7588c64f5d'
GROUPS = {'direct19':[i for i,n in enumerate(NAMES) if n!='tcwv'],
          'all20':list(range(20)), 'near':[NAMES.index(n) for n in NEAR],
          'upper':[NAMES.index(n) for n in UPPER]}
FRAMES = np.linspace(0,1459,8,dtype=int).tolist()
POSITIONS = [(48,120),(24,60),(72,180)]
ERF_KEYS = ['energy_r50_chebyshev_pixels','energy_r90_chebyshev_pixels',
            'energy_r95_chebyshev_pixels','mass_outside_original_structural_support']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, atol=1e-9):
    require(np.isfinite(float(actual)) and np.isfinite(float(expected)), f'{label}: nonfinite')
    require(np.isclose(float(actual), float(expected), rtol=1e-10, atol=atol),
            f'{label}: {actual!r} != recomputed {expected!r}')


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return json.load(f)


def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def required_files(run, original, affine):
    rel=['structural_validation.json','provenance.json','training_data.json','history.json',
         'selection.json','best.pt','performance_complete.json','inference_cost.json',
         'analysis_complete.json','comparison.csv','comparison.json','per_channel_comparison.csv',
         'paired_model_differences.csv','paired_model_differences.json',
         'erf/summary.json','erf/gradient_probes.csv','erf/mean_normalized_gradient_energy_maps.npz',
         'erf/gradient_energy_maps.pdf','erf/gradient_energy_maps.png']
    for year in [2018,2022]:
        rel += [f'eval{year}/{name}' for name in ['summary.json','sufficient_statistics.npz','per_channel.csv']]
        rel += [f'gain_comparison_{year}.pdf',f'gain_comparison_{year}.png']
    rel += ['eval2022/verified_data_manifest.json']
    rel += [f'affine_eval2022/{name}' for name in ['summary.json','sufficient_statistics.npz','per_channel.csv','verified_data_manifest.json']]
    files=[run/p for p in rel]
    files += [original/f'eval{year}/{p}' for year in [2018,2022]
              for p in ['summary.json','sufficient_statistics.npz']]
    files += [affine/p for p in ['summary.json','2018_per_channel.csv','2018_time_statistics.npz','best.pt']]
    missing=[str(p) for p in files if not p.is_file()]
    require(not missing, 'Missing required artifacts:\n'+'\n'.join(missing))
    require(all(p.stat().st_size>0 for p in files), 'Empty required artifact')
    return files


def load_statistics(path):
    with np.load(path,allow_pickle=False) as d:
        require(d['channel_names'].astype(str).tolist()==NAMES, f'{path}: channel order')
        result={k:np.array(d[k],dtype=np.float64) for k in ['before_time','after_time','valid_count']}
    for key,arr in result.items():
        require(arr.shape==(1460,20),f'{path}: {key} shape {arr.shape}')
        require(np.isfinite(arr).all(),f'{path}: {key} nonfinite')
        require(np.all(arr>=0),f'{path}: {key} negative')
    require(np.all(result['valid_count']==97*241),f'{path}: incomplete/masked grid counts')
    return result


def metric_values(stats, index=None):
    if index is None:index=np.arange(1460)
    count=stats['valid_count'][index].sum(axis=0)
    before=np.sqrt(stats['before_time'][index].sum(axis=0)/count)
    after=np.sqrt(stats['after_time'][index].sum(axis=0)/count)
    require(np.all(before>0),'Zero baseline RMSE makes gains undefined')
    groups={}
    for name,ids in GROUPS.items():
        groups[name]={'before_nrmse':float(before[ids].mean()),
                      'after_nrmse':float(after[ids].mean()),
                      'G_pct':float(100*(1-after[ids].sum()/before[ids].sum())),
                      'Gbar_pct':float(np.mean(100*(1-after[ids]/before[ids])))}
    upper=groups['upper']['G_pct']
    groups['near_upper_gain_ratio']=groups['near']['G_pct']/upper if upper!=0 else None
    groups['near_minus_upper_percentage_points']=groups['near']['G_pct']-upper
    channels={n:{'before_nrmse':float(before[i]),'after_nrmse':float(after[i]),
                 'G_pct':float(100*(1-after[i]/before[i]))} for i,n in enumerate(NAMES)}
    return {'groups':groups,'channels':channels}


def verify_summary(summary, computed, label, checkpoint=None):
    require(int(summary.get('frames',0))==1460,f'{label}: frames')
    if checkpoint is not None:
        # Historical affine stores its hash in selection/provenance, not this summary.
        require(summary.get('checkpoint_sha256')==checkpoint,f'{label}: checkpoint hash')
    for name in GROUPS:
        for key,value in computed['groups'][name].items():
            close(summary['groups'][name][key],value,f'{label}:{name}:{key}')
    if 'channels' in summary:
        require(set(summary['channels'])==set(NAMES),f'{label}: missing/extra channels')
        for name,row in computed['channels'].items():
            for key,value in row.items():
                if key in summary['channels'][name]:
                    close(summary['channels'][name][key],value,f'{label}:{name}:{key}')
    for key in ['near_upper_gain_ratio','near_minus_upper_percentage_points']:
        if key in summary['groups']:
            actual=summary['groups'][key];expected=computed['groups'][key]
            require(actual is None,f'{label}:{key} must be null') if expected is None else close(actual,expected,f'{label}:{key}')


def verify_channel_csv(path,computed):
    rows=read_csv(path)
    require(len(rows)==20 and {r['channel'] for r in rows}==set(NAMES),f'{path}: 20 unique channels required')
    for row in rows:
        for key,value in computed['channels'][row['channel']].items():
            close(row[key],value,f'{path}:{row["channel"]}:{key}')


def verify_structure(run):
    s=read_json(run/'structural_validation.json')
    for key in ['passed','original_UNet2_AST_unchanged','forward_inherited_unchanged','identical_initial_weights_and_state_keys']:
        require(s.get(key) is True,f'structural guard {key}')
    require(s['parameters_original']==s['parameters_dilated']==1646548,'parameter counts')
    require(s['tested_shapes']==[[1,20,97,241],[8,20,97,241]],'B=1/B=8 shape preflight missing')
    changes=s['only_conv_metadata_changes']
    require([r['module'] for r in changes]==['bottleneck.0','bottleneck.2'],'unexpected modified modules')
    for row,dilation in zip(changes,[2,3]):
        old,new=row['original'],row['dilated']
        require(set(old)==set(new),'convolution attributes differ')
        require({k for k in old if old[k]!=new[k]}=={'padding','dilation'},'non-RF convolution change')
        require(old['padding']==old['dilation']==[1,1],'original padding/dilation')
        require(new['padding']==new['dilation']==[dilation,dilation],'new padding/dilation')
        require(old['kernel_size']==[3,3] and old['in_channels']==old['out_channels']==128,'bottleneck dimensions')


def verify_selection(run):
    selection=read_json(run/'selection.json');history=read_json(run/'history.json')
    n=len(history)
    require(1<=n<=200 and [int(r['epoch']) for r in history]==list(range(1,n+1)),'history epochs')
    losses=np.array([r['val_loss'] for r in history],dtype=float)
    require(np.isfinite(losses).all() and np.all(losses>=0),'invalid validation losses')
    require(all(np.isfinite(float(r['train_loss'])) and float(r['train_loss'])>=0 for r in history),'invalid training losses')
    best_epoch=int(np.argmin(losses))+1
    require(selection['best_epoch']==best_epoch and selection['stopped_epoch']==n,'selected/stopped epoch mismatch')
    close(selection['best_val_loss'],losses[best_epoch-1],'selection minimum loss',atol=1e-12)
    best=float('inf');stale=0
    for i,row in enumerate(history):
        improved=float(row['val_loss'])<best
        require(row['improved'] is improved,f'epoch{i+1}: improvement marker')
        if improved:best=float(row['val_loss']);stale=0
        else:stale+=1
        if i<n-1:require(stale<12,'training continued after patience12')
    require(n==200 or stale==12,'run stopped before patience/max budget')
    require(selection['test_data_used_for_selection'] is False,'test selection flag')
    require(selection['training_steps']==n*548 and selection['validation_steps']==n*183,'step counts')
    require(float(selection['training_seconds'])>0,'training time')
    recipe=selection['recipe']
    expected={'seed':0,'batch_size':8,'learning_rate':3e-4,'weight_decay':1e-4,'gradient_clip':1.,
              'maximum_epochs':200,'patience':12,'amp':True,'workers':4,'optimizer':'AdamW'}
    for key,value in expected.items():require(recipe[key]==value,f'recipe {key}')
    require('2017' in selection['criterion'] and 'batch' in selection['criterion'],'2017 batch-MSE criterion')
    digest=sha(run/'best.pt')
    require(digest==selection['checkpoint_sha256'],'selected checkpoint bytes/hash mismatch')
    p=read_json(run/'provenance.json')
    require(p['status']=='evaluation_complete','provenance incomplete')
    require(p['original_checkpoint_sha256']==ORIGINAL_HASH and p['stats_sha256']==STATS_HASH,'original/stats guards')
    require(p['selection']==selection,'provenance selection copy mismatch')
    data=read_json(run/'training_data.json')
    require(data['test_arrays_not_opened_before_selection'] is True,'test gating record')
    require([(r['year'],r['frames']) for r in data['records']]==[(2014,1460),(2015,1460),(2016,1464),(2017,1460)],'training/validation data years')
    require('cra5' in data['cache_manifest']['decoded_dir'].lower(),'not CRA5 residual cache')
    complete=read_json(run/'performance_complete.json')
    require(complete['status']=='complete' and complete['years']==[2018,2022] and complete['frames_each']==1460,'performance marker')
    require(complete['checkpoint_sha256']==digest,'performance selected checkpoint')
    require(read_json(run/'analysis_complete.json')['status']=='complete','analysis incomplete')
    return digest,selection


def verify_manifest(path):
    m=read_json(path)
    require(m['complete'] is True and m['alignment_verified'] is True,'2022 manifest incomplete/unaligned')
    require(m['year']==2022 and m['frames']==1460 and m['channels']==NAMES,'2022 manifest dimensions/channels')
    require(m['stats_sha256']==STATS_HASH,'2022 statistics hash')
    for key in ['input_sha256','target_sha256','timestamps_sha256']:
        require(len(m[key])==64 and all(c in '0123456789abcdef' for c in m[key]),f'invalid {key}')


def axis_bounds(n,dilations=(1,1)):
    """Independent interval propagation for the contiguous structural supports."""
    def merge(values):return (min(v[0] for v in values),max(v[1] for v in values))
    def conv(a,d=1):return [merge([a[j] for j in (i-d,i,i+d) if 0<=j<len(a)]) for i in range(len(a))]
    def down(a):return conv([merge(a[2*i:2*i+2]) for i in range(len(a)//2)])
    def resize(a,size):
        b=[]
        for i in range(size):
            lo,rem=divmod((2*i+1)*len(a)-size,2*size)
            ids=[max(0,min(len(a)-1,lo))]
            if rem:ids.append(max(0,min(len(a)-1,lo+1)))
            b.append(merge([a[j] for j in ids]))
        return b
    e0=conv(conv([(i,i) for i in range(n)]));e1=conv(conv(down(e0)))
    a=conv(conv(down(e1),dilations[0]),dilations[1])
    for skip in [e1,e0]:
        a=resize(conv(a),len(skip));a=[merge([v,s]) for v,s in zip(a,skip)];a=conv(conv(a))
    return conv(a)


def verify_erf(run,checkpoint):
    summary=read_json(run/'erf/summary.json');rows=read_csv(run/'erf/gradient_probes.csv')
    require(summary['frames']==FRAMES and [tuple(p) for p in summary['positions']]==POSITIONS,'ERF fixed probes')
    require(summary['probes_per_model']==480 and len(rows)==960,'ERF probe count')
    require(summary['checkpoint_original_sha256']==ORIGINAL_HASH and summary['checkpoint_dilated_sha256']==checkpoint,'ERF checkpoint hashes')
    require('2017' in summary['inputs'] and 'correction-output' in summary['method'],'ERF input/output scope')
    expected={(model,frame,pos,name) for model in ['original','dilated'] for frame in FRAMES for pos in range(3) for name in NAMES}
    seen=set();parsed=[]
    for row in rows:
        key=(row['model'],int(row['frame']),int(row['position']),row['channel'])
        require(key not in seen and key in expected,'duplicate/unexpected ERF probe');seen.add(key)
        row=dict(row);row['frame']=int(row['frame']);row['position']=int(row['position'])
        cy,cx=POSITIONS[row['position']]
        require((int(row['y']),int(row['x']))==(cy,cx),'ERF position coordinates')
        require(row['zero_gradient'] in ['True','False'],'ERF zero flag')
        zero=row['zero_gradient']=='True';total=float(row['gradient_squared_sum'])
        require(math.isfinite(total) and total>=0 and zero==(total==0),'ERF energy/zero mismatch')
        row['zero_gradient']=zero
        if not zero:
            for k in ERF_KEYS+['mass_outside_own_structural_support']:
                row[k]=float(row[k]);require(math.isfinite(row[k]),f'ERF nonfinite {k}')
            radii=[row[k] for k in ERF_KEYS[:3]]
            require(all(v==int(v) and v>=0 for v in radii) and radii==sorted(radii),'ERF energy radii ordering')
            require(radii[-1]<=max(cy,96-cy,cx,240-cx),'ERF radius outside domain')
            require(0<=row[ERF_KEYS[3]]<=1+1e-10,'ERF outside-original mass')
            require(0<=row['mass_outside_own_structural_support']<1e-6,'ERF own-support leakage')
            if row['model']=='original':require(row[ERF_KEYS[3]]<1e-6,'Original leakage outside original support')
        parsed.append(row)
    require(seen==expected,'missing ERF probes')
    def belongs(row,group):return group=='all20' or row['channel'] in (NEAR if group=='near' else UPPER if group=='upper' else ['tcwv'])
    maps=dict(np.load(run/'erf/mean_normalized_gradient_energy_maps.npz',allow_pickle=False))
    expected_maps=set()
    model_summaries={}
    for model,dils in [('original',(1,1)),('dilated',(2,3))]:
        axes=[axis_bounds(97,dils),axis_bounds(241,dils)];oldaxes=[axis_bounds(97),axis_bounds(241)]
        for group in ['all20','near','upper','tcwv']:
            subset=[r for r in parsed if r['model']==model and belongs(r,group)]
            valid=[r for r in subset if not r['zero_gradient']]
            recorded=summary['summaries'][f'{model}_{group}']
            require(recorded['probes']==len(subset) and recorded['nonzero_probes']==len(valid) and recorded['zero_probes']==len(subset)-len(valid),'ERF group counts')
            for k in ERF_KEYS:
                values=np.array([r[k] for r in valid])
                if not len(values):require(recorded[k] is None,'empty ERF group summary')
                else:
                    for stat,value in [('mean',values.mean()),('median',np.median(values)),('minimum',values.min()),('maximum',values.max())]:
                        close(recorded[k][stat],value,f'ERF {model}/{group}/{k}/{stat}')
            for pos,(cy,cx) in enumerate(POSITIONS):
                key=f'{model}_{group}_position{pos}';vv=[r for r in valid if r['position']==pos]
                recorded_pos=summary['per_position_summaries'][key]
                require(recorded_pos['nonzero_probes']==len(vv),'ERF position count')
                for k in ERF_KEYS:
                    values=np.array([r[k] for r in vv])
                    if not len(values):require(recorded_pos[k] is None,'empty ERF position summary')
                    else:
                        close(recorded_pos[k]['mean'],values.mean(),f'{key}/{k}/mean')
                        close(recorded_pos[k]['median'],np.median(values),f'{key}/{k}/median')
                if not vv:continue
                expected_maps.add(key);require(key in maps,f'missing ERF map {key}')
                arr=maps[key]
                require(arr.shape==(97,241) and np.isfinite(arr).all() and np.all(arr>=0),f'{key}: invalid map')
                close(arr.sum(),1,f'{key}: unit gradient mass',atol=1e-10)
                mask=np.zeros_like(arr,dtype=bool);ymn,ymx=axes[0][cy];xmn,xmx=axes[1][cx];mask[ymn:ymx+1,xmn:xmx+1]=True
                oldmask=np.zeros_like(mask);ymn,ymx=oldaxes[0][cy];xmn,xmx=oldaxes[1][cx];oldmask[ymn:ymx+1,xmn:xmx+1]=True
                require(arr[~mask].sum()<1e-6,f'{key}: map structural leakage')
                close(arr[~oldmask].sum(),np.mean([r[ERF_KEYS[3]] for r in vv]),f'{key}: outside-original map/CSV consistency',atol=1e-10)
        for pos in range(3):
            allkey=f'{model}_all20_position{pos}'
            if allkey not in maps:continue
            valid_counts={g:sum(1 for r in parsed if r['model']==model and r['position']==pos and belongs(r,g) and not r['zero_gradient']) for g in ['near','upper','tcwv']}
            total=sum(valid_counts.values());combined=sum((maps[f'{model}_{g}_position{pos}']*n for g,n in valid_counts.items() if n),np.zeros((97,241)))/total
            require(np.allclose(maps[allkey],combined,rtol=1e-9,atol=1e-14),'ERF all/group map consistency')
        theory=summary['theory']['models'][model]
        center=[axes[0][48][1]-axes[0][48][0]+1,axes[1][120][1]-axes[1][120][0]+1]
        require(theory['center_bounding_box_hw']==center,'theory center mismatch')
        require(theory['max_actual_axis_spans_hw']==[max(b-a+1 for a,b in axis) for axis in axes],'theory clipped max mismatch')
        require(theory['unclipped_max_axis_span']==(76 if model=='original' else 100),'theory unclipped max mismatch')
        model_summaries[model]={'probes':len([r for r in parsed if r['model']==model]),'zero_probes':sum(r['zero_gradient'] for r in parsed if r['model']==model)}
    require(set(maps)==expected_maps,'unexpected ERF maps')
    return model_summaries


def verify_costs(run):
    costs=read_json(run/'inference_cost.json')
    require(costs['dtype']=='float32','cost dtype')
    require(costs['cudnn_deterministic'] is True and costs['cudnn_benchmark'] is False,'cost cuDNN flags')
    for model in ['original','dilated']:
        c=costs[model]
        require(c['parameters']==1646548 and c['weight_bytes']==6586192 and c['batch_size']==8,'cost model dimensions')
        for key in ['mean_batch_ms','amortized_ms_per_frame','frames_per_second','peak_allocated_bytes','profiled_flops_per_frame']:
            require(math.isfinite(float(c[key])) and float(c[key])>0,f'cost {model}/{key}')
        close(c['amortized_ms_per_frame'],c['mean_batch_ms']/8,f'{model}: amortized latency')
        close(c['frames_per_second'],8000/c['mean_batch_ms'],f'{model}: throughput')
    close(costs['original']['profiled_flops_per_frame'],costs['dilated']['profiled_flops_per_frame'],'parameter-matched profiler convolution FLOPs')
    return costs


def comparison_expected(model,year,computed):
    g=computed['groups']
    return dict(model=model,year=year,parameters=3620 if model=='affine_low_capacity' else 1646548,frames=1460,
                before_direct19=g['direct19']['before_nrmse'],after_direct19=g['direct19']['after_nrmse'],
                G_direct19_pct=g['direct19']['G_pct'],G_near_pct=g['near']['G_pct'],G_upper_pct=g['upper']['G_pct'],
                near_upper_gain_ratio=g['near_upper_gain_ratio'],near_minus_upper_percentage_points=g['near_minus_upper_percentage_points'])


def check_record(actual,expected,label):
    for key,value in expected.items():
        require(key in actual,f'{label}: missing {key}')
        if isinstance(value,str):require(actual[key]==value,f'{label}: {key}')
        elif value is None:require(actual[key] in [None,''],f'{label}: {key} expected null')
        else:close(actual[key],value,f'{label}:{key}')


def verify_exports(run,computed):
    expected={(model,year):comparison_expected(model,year,c) for (model,year),c in computed.items()}
    for path in [run/'comparison.csv',run/'comparison.json']:
        rows=read_csv(path) if path.suffix=='.csv' else read_json(path)
        require(len(rows)==6,f'{path}: six rows required')
        keys=[(r['model'],int(r['year'])) for r in rows]
        require(set(keys)==set(expected) and len(set(keys))==6,f'{path}: model/year coverage')
        for r,key in zip(rows,keys):check_record(r,expected[key],str(path))
    rows=read_csv(run/'per_channel_comparison.csv')
    require(len(rows)==120,'per-channel comparison must have120 rows')
    keys=set()
    for r in rows:
        key=(r['model'],int(r['year']));name=r['channel'];uid=(*key,name)
        require(key in computed and name in NAMES and uid not in keys,'per-channel comparison identity')
        keys.add(uid);check_record(r,computed[key]['channels'][name],f'per-channel {uid}')
    return {'comparison_rows':6,'per_channel_rows':120,
            'negative_per_channel_gains':sum(v['G_pct']<0 for c in computed.values() for v in c['channels'].values())}


def paired_differences(old,new,index):
    # Recompute both models independently from shared sampled indices/counts.
    a=metric_values(old,index)['groups'];b=metric_values(new,index)['groups']
    result={g:b[g]['G_pct']-a[g]['G_pct'] for g in ['direct19','near','upper']}
    result['near_minus_upper']=b['near_minus_upper_percentage_points']-a['near_minus_upper_percentage_points']
    return result


def verify_paired(run,statistics):
    rows=read_csv(run/'paired_model_differences.csv');stored=read_json(run/'paired_model_differences.json')
    require(len(rows)==len(stored['results'])==8,'eight paired-CI rows required')
    require('1000' in stored['method'] and '20261026' in stored['method'],'paired CI method metadata')
    expected={}
    for year in [2018,2022]:
        old,new=statistics[('original',year)],statistics[('dilated',year)]
        require(np.array_equal(old['before_time'],new['before_time']),'Original/Dilated baseline SSE differs')
        require(np.array_equal(old['valid_count'],new['valid_count']),'Original/Dilated valid counts differ')
        point=paired_differences(old,new,np.arange(1460));draws={k:[] for k in point};rng=np.random.default_rng(20261026)
        for _ in range(1000):
            starts=rng.integers(0,1460,size=math.ceil(1460/28));sample=((starts[:,None]+np.arange(28))%1460).reshape(-1)[:1460]
            for key,value in paired_differences(old,new,sample).items():draws[key].append(value)
        for metric,value in point.items():
            lo,hi=np.percentile(draws[metric],[2.5,97.5])
            expected[(year,metric)]=dict(year=year,metric=metric,dilated_minus_original_pp=value,lower95_pp=lo,upper95_pp=hi)
    for collection in [rows,stored['results']]:
        keys=[(int(r['year']),r['metric']) for r in collection]
        require(set(keys)==set(expected) and len(set(keys))==8,'paired metric/year coverage')
        for r,key in zip(collection,keys):
            check_record(r,expected[key],f'paired {key}')
            require(float(r['lower95_pp'])<=float(r['upper95_pp']),'CI endpoints reversed')
    return {'rows':8,'replicates_per_year':1000,'block_frames':28,'seed':20261026,'intervals_recomputed':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--original-results',type=Path,help='Root containing original eval2018/eval2022; default run.parent')
    args=p.parse_args();run=args.run.resolve();original=(args.original_results or run.parent).resolve()
    candidates=[original/'linear_corrector',Path(__file__).resolve().parents[1]/'linear_corrector']
    affine=next((p for p in candidates if (p/'2018_time_statistics.npz').is_file()),candidates[0])
    require(not (run/'acceptance.json').exists(),'acceptance.json already exists; verifier does not overwrite prior acceptance')
    files=required_files(run,original,affine)
    verify_structure(run);checkpoint,selection=verify_selection(run);computed={};statistics={}
    for year in [2018,2022]:
        for model,directory in [('original',original/f'eval{year}'),('dilated',run/f'eval{year}')]:
            d=load_statistics(directory/'sufficient_statistics.npz');statistics[(model,year)]=d
            c=metric_values(d);computed[(model,year)]=c
            verify_summary(read_json(directory/'summary.json'),c,f'{model}{year}',ORIGINAL_HASH if model=='original' else checkpoint)
            if model=='dilated':verify_channel_csv(directory/'per_channel.csv',c)
    require(sha(affine/'best.pt')==AFFINE_HASH,'historical affine checkpoint hash')
    for year,directory,statfile,summaryfile,csvfile in [
        (2018,affine,'2018_time_statistics.npz','summary.json','2018_per_channel.csv'),
        (2022,run/'affine_eval2022','sufficient_statistics.npz','summary.json','per_channel.csv')]:
        d=load_statistics(directory/statfile);statistics[('affine_low_capacity',year)]=d
        require(np.array_equal(d['before_time'],statistics[('original',year)]['before_time']),'affine baseline SSE mismatch')
        c=metric_values(d);computed[('affine_low_capacity',year)]=c
        verify_summary(read_json(directory/summaryfile),c,f'affine{year}',AFFINE_HASH if year==2022 else None)
        verify_channel_csv(directory/csvfile,c)
    verify_manifest(run/'eval2022/verified_data_manifest.json');verify_manifest(run/'affine_eval2022/verified_data_manifest.json')
    erf=verify_erf(run,checkpoint);cost=verify_costs(run);exports=verify_exports(run,computed);paired=verify_paired(run,statistics)
    result={'status':'accepted','accepted_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'verifier_sha256':sha(Path(__file__)),'run':str(run),'original_results':str(original),
            'checkpoint_sha256':checkpoint,'best_epoch':selection['best_epoch'],'stopped_epoch':selection['stopped_epoch'],
            'annual_frames':{'2018':1460,'2022':1460},'parameters_each':1646548,
            'checks':['structural guards','history/selection/patience','checkpoint hashes','annual SSE/summary/channel CSV',
                      'baseline SSE equality','2022 alignment manifests','ERF coverage/maps/support/summaries',
                      'same-run inference cost consistency','six-row/120-channel exports','paired CIs recomputed'],
            'exports':exports,'erf':erf,'paired_uncertainty':paired,
            'limitations':['No GPU inference or training rerun by this verifier.',
                           'Manifest hashes are checked as recorded metadata; large remote raw arrays are not rehashed locally.',
                           'Saved ERF probe radii are checked for internal consistency; individual raw gradient maps are not retained.',
                           'No requirement that new model improve; negative gains/differences are preserved.'],
            'artifact_sha256':{str(path):sha(path) for path in files}}
    with (run/'acceptance.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'status':'accepted','acceptance':str(run/'acceptance.json'),'exports':exports,'erf':erf},indent=2))


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,OSError,TypeError,IndexError) as error:
        print(json.dumps({'status':'failed','error':str(error)},indent=2),file=sys.stderr)
        sys.exit(1)
