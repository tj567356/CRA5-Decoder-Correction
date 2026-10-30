"""Gradient ERF diagnostic on fixed 2017 validation inputs, never test selection.

Differentiate each normalized correction output f(x), not x+f(x). Aggregate
squared input gradients over 20 input channels; normalize EACH nonzero probe.
Probes: 8 evenly spaced validation frames x all20 outputs x 3 fixed positions.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
from rf_theory import support_axis, theoretical_report

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(p):
    s=importlib.util.spec_from_file_location('rf_models',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/tong/fourcastnet_us_mpca'))
    p.add_argument('--model',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    root=a.root;run=a.run;out=run/'erf'
    assert json.loads((run/'performance_complete.json').read_text())['status']=='complete'
    selected=json.loads((run/'selection.json').read_text());assert sha(run/'best.pt')==selected['checkpoint_sha256']
    out.mkdir(exist_ok=False)
    original=root/'checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt'
    assert sha(original)=='44c57a0f2a18e92c9c0745cfe75337079627ab8729e93c6ec098a5ddeb007435'
    m=module(a.model);cache=root/'data/memmap_cra5';data=np.load(cache/'input_2017.npy',mmap_mode='r')
    frames=np.linspace(0,1459,8,dtype=int).tolist();positions=[(48,120),(24,60),(72,180)]
    names=np.load(root/'mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz')['channel_names'].astype(str).tolist()
    assert len(names)==20 and data.shape==(1460,20,97,241)
    assert sha(root/'mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz')=='272cc45d231ec349025c4fa1e9f69366ae874739c98299ce666921821f13dba2'
    near={'u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850'}
    upper={'u500','v500','z500','t500','r500','z50'}
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    rows=[];maps={};aggregate={}
    original_support=[support_axis(97),support_axis(241)]
    yy,xx=np.indices((97,241))
    for label,constructor,ck,dils in [('original',m.UNet2,original,(1,1)),('dilated',m.DilatedUNet2,run/'best.pt',(2,3))]:
        net=constructor(20,64,3,2,128).cuda().eval();net.load_state_dict(torch.load(ck,map_location='cuda',weights_only=True))
        supports=[support_axis(97,dils),support_axis(241,dils)]
        sums={};counts={}
        for frame in frames:
            x=torch.tensor(np.array(data[frame:frame+1],dtype=np.float32),device='cuda',requires_grad=True)
            prediction=net(x)
            probe=0
            for pos_id,(cy,cx) in enumerate(positions):
                allowed=np.zeros((97,241),bool);allowed[np.ix_(sorted(supports[0][cy]),sorted(supports[1][cx]))]=True
                old=np.zeros_like(allowed);old[np.ix_(sorted(original_support[0][cy]),sorted(original_support[1][cx]))]=True
                distance=np.maximum(abs(yy-cy),abs(xx-cx))
                for channel,name in enumerate(names):
                    probe+=1
                    g=torch.autograd.grad(prediction[0,channel,cy,cx],x,retain_graph=probe<60)[0]
                    energy=g[0].detach().double().square().sum(0).cpu().numpy();total=float(energy.sum())
                    assert np.isfinite(total) and np.isfinite(energy).all()
                    row=dict(model=label,frame=frame,position=pos_id,y=cy,x=cx,channel=name,gradient_squared_sum=total,zero_gradient=total==0)
                    group='near' if name in near else ('upper' if name in upper else 'tcwv')
                    if total:
                        energy/=total
                        assert energy[~allowed].sum()<1e-6, 'gradient outside theoretical dependency support'
                        order=np.argsort(distance.ravel(),kind='stable');cumulative=np.cumsum(energy.ravel()[order]);radii=distance.ravel()[order]
                        for q in [50,90,95]:row[f'energy_r{q}_chebyshev_pixels']=int(radii[min(np.searchsorted(cumulative,q/100),len(radii)-1)])
                        row['mass_outside_original_structural_support']=float(energy[~old].sum())
                        row['mass_outside_own_structural_support']=float(energy[~allowed].sum())
                        for key in [('all20',pos_id),(group,pos_id)]:
                            sums[key]=sums.get(key,np.zeros((97,241)))+energy;counts[key]=counts.get(key,0)+1
                    rows.append(row)
            print(label,'validation frame',frame,'ERF probes',probe,flush=True)
        for key,value in sums.items():maps[f'{label}_{key[0]}_position{key[1]}']=value/counts[key]
        for group in ['all20','near','upper','tcwv']:
            subset=[r for r in rows if r['model']==label and (group=='all20' or (r['channel'] in near if group=='near' else r['channel'] in upper if group=='upper' else r['channel']=='tcwv'))]
            valid=[r for r in subset if not r['zero_gradient']]
            entry=dict(probes=len(subset),nonzero_probes=len(valid),zero_probes=len(subset)-len(valid))
            for key in ['energy_r50_chebyshev_pixels','energy_r90_chebyshev_pixels','energy_r95_chebyshev_pixels','mass_outside_original_structural_support']:
                values=[r[key] for r in valid];entry[key]=dict(mean=float(np.mean(values)),median=float(np.median(values)),minimum=float(np.min(values)),maximum=float(np.max(values))) if values else None
            aggregate[f'{label}_{group}']=entry
        del net
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with open(out/'gradient_probes.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'mean_normalized_gradient_energy_maps.npz',**maps)
    per_position={}
    for label in ['original','dilated']:
        for position in range(3):
            for group in ['all20','near','upper','tcwv']:
                subset=[r for r in rows if r['model']==label and r['position']==position and not r['zero_gradient']
                        and (group=='all20' or (r['channel'] in near if group=='near' else r['channel'] in upper if group=='upper' else r['channel']=='tcwv'))]
                entry={'nonzero_probes':len(subset)}
                for key in ['energy_r50_chebyshev_pixels','energy_r90_chebyshev_pixels','energy_r95_chebyshev_pixels','mass_outside_original_structural_support']:
                    values=[r[key] for r in subset]
                    entry[key]={'mean':float(np.mean(values)),'median':float(np.median(values))} if values else None
                per_position[f'{label}_{group}_position{position}']=entry
    report=dict(method='squared gradient ERF of individual correction-output channels; per-probe unit-mass normalization',
                normalization='sum squared gradients over all20 standardized input channels, then normalize each probe',
                inputs='2017 validation only; fixed in advance, no model selection',frames=frames,positions=positions,
                probes_per_model=480,units='input-grid pixels; Chebyshev radius centered at output position',
                checkpoint_original_sha256=sha(original),checkpoint_dilated_sha256=sha(run/'best.pt'),
                source_sha256=sha(__file__),torch=str(torch.__version__),dtype='float32',cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                summaries=aggregate,per_position_summaries=per_position,
                limitations='Local input/weight-dependent sensitivities, not proof of use of every theoretically reachable location; eight frames, fixed probes, one seed. Zero gradients reported, not replaced.',
                theory=theoretical_report()[0])
    (out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()
