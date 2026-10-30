"""Run only after frozen performance/ERF finish; export auditable comparison.

Never selects a model. Historical affine weights may be evaluated on 2022 once
as an explicitly low-capacity, non-parameter-matched reference.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rf_ablation import evaluate, sha, summarize

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/tong/fourcastnet_us_mpca'))
    p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.root;run=a.run
    assert json.loads((run/'performance_complete.json').read_text())['status']=='complete'
    assert (run/'erf/summary.json').exists()
    names=np.load(root/'mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz')['channel_names'].astype(str).tolist()
    linear=root/'results/revision_2026/linear_corrector'
    if not linear.exists():linear=root/'revision_2026/linear_corrector'
    assert sha(linear/'best.pt')=='31743a1a03669a6b20f7a39c79c7ae3738fa059468ea789a56e6ab7588c64f5d'
    net=nn.Conv2d(20,20,3,padding=1,bias=True).cuda().eval()
    net.load_state_dict(torch.load(linear/'best.pt',map_location='cuda',weights_only=True))
    evaluate(net,root/'revision_2026/data/memmap_cra5_2022',2022,run/'affine_eval2022',names,sha(linear/'best.pt'))
    rows=[];channels=[]
    for label,params in [('original',1646548),('dilated',1646548),('affine_low_capacity',3620)]:
        for year in [2018,2022]:
            if label=='original':s=json.loads((root/f'results/revision_2026/eval{year}/summary.json').read_text())
            elif label=='dilated':s=json.loads((run/f'eval{year}/summary.json').read_text())
            elif year==2022:s=json.loads((run/'affine_eval2022/summary.json').read_text())
            else:
                s=json.loads((linear/'summary.json').read_text())
                with open(linear/'2018_per_channel.csv',newline='') as f:
                    s['channels']={r['channel']:{k:float(r[k]) for k in ['before_nrmse','after_nrmse','G_pct']} for r in csv.DictReader(f)}
            # Historical affine uses same group schema; fail loudly if provenance differs.
            g=s['groups'];near=g['near']['G_pct'];upper=g['upper']['G_pct']
            rows.append(dict(model=label,year=year,parameters=params,frames=1460,
                before_direct19=g['direct19']['before_nrmse'],after_direct19=g['direct19']['after_nrmse'],
                G_direct19_pct=g['direct19']['G_pct'],G_near_pct=near,G_upper_pct=upper,
                near_upper_gain_ratio=near/upper if upper!=0 else None,
                near_minus_upper_percentage_points=near-upper))
            for name,v in s['channels'].items():
                channels.append(dict(model=label,year=year,channel=name,before_nrmse=v['before_nrmse'],
                    after_nrmse=v['after_nrmse'],G_pct=100*(1-v['after_nrmse']/v['before_nrmse'])))
    for filename,data in [('comparison.csv',rows),('per_channel_comparison.csv',channels)]:
        with open(run/filename,'x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (run/'comparison.json').write_text(json.dumps(rows,indent=2))
    log=root/'revision_2026/cra5_train.log'
    if log.exists():
        times=[float(x) for x in re.findall(r'Epoch\s+\d+/\d+.*?\s([\d.]+)s',log.read_text())]
        (run/'original_historical_training_cost.json').write_text(json.dumps(dict(
            epochs=len(times),sum_logged_epoch_seconds=sum(times),mean_logged_epoch_seconds=float(np.mean(times)),
            scope='historical rounded per-epoch train+validation times, not a contemporaneous original retraining benchmark',
            source=str(log),source_sha256=sha(log)),indent=2))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for year in [2018,2022]:
        fig,ax=plt.subplots(figsize=(6.5,3.5),layout='constrained')
        selected=[r for r in rows if r['year']==year];x=np.arange(len(selected));width=.24
        for offset,key,title in [(-width,'G_direct19_pct','19 direct'),(0,'G_near_pct','Surface/lower'),(width,'G_upper_pct','Higher levels')]:
            ax.bar(x+offset,[r[key] for r in selected],width,label=title)
        ax.set_xticks(x,['Original U-Net','Dilated U-Net','Affine (low capacity)']);ax.set_ylabel('nRMSE reduction (%)');ax.set_title(f'Frozen-checkpoint {year} evaluation');ax.legend();ax.axhline(0,color='black',lw=.7)
        fig.savefig(run/f'gain_comparison_{year}.pdf');fig.savefig(run/f'gain_comparison_{year}.png',dpi=180);plt.close(fig)
    erf=json.loads((run/'erf/summary.json').read_text());maps=np.load(run/'erf/mean_normalized_gradient_energy_maps.npz')
    fig,axs=plt.subplots(2,2,figsize=(8,5),layout='constrained')
    for i,label in enumerate(['original','dilated']):
        for j,group in enumerate(['near','upper']):
            arr=maps[f'{label}_{group}_position0'];im=axs[i,j].imshow(np.log10(np.maximum(arr,1e-12)),vmin=-10,vmax=-1,cmap='magma',origin='upper')
            axs[i,j].set_title(f'{label}: {group}, center output');axs[i,j].set_xlabel('Longitude grid index');axs[i,j].set_ylabel('Latitude grid index')
    fig.colorbar(im,ax=axs,label='log10 normalized squared-gradient energy',shrink=.8)
    fig.savefig(run/'erf/gradient_energy_maps.pdf');fig.savefig(run/'erf/gradient_energy_maps.png',dpi=180);plt.close(fig)
    (run/'analysis_complete.json').write_text(json.dumps(dict(status='complete',scope='full frozen evaluation, gradient ERF, compute, exports; bootstrap separate'),indent=2))

if __name__=='__main__':main()
