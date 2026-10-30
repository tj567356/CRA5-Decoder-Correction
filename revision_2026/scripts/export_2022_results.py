"""Export only a successfully completed full-year frozen evaluation; no missing-value backfill."""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--eval2018',type=Path,required=True)
    p.add_argument('--eval2022',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    old=json.loads((a.eval2018/'summary.json').read_text())
    new=json.loads((a.eval2022/'summary.json').read_text())
    cache=json.loads((a.cache/'manifest.json').read_text())
    assert old['frames']==new['frames']==1460 and new['year']==2022 and cache['complete']
    assert new['checkpoint_sha256']==old['checkpoint_sha256']
    assert new['stats_sha256']==old['stats_sha256']==cache['stats_sha256']
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for ch,values in new['channels'].items():
        rows.append({'channel':ch,'group':'derived' if ch=='tcwv' else ('near' if ch in
                     ['u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850'] else 'upper'),
                     **values,'gain_pct':100*(1-values['after_nrmse']/values['before_nrmse'])})
    with (a.out/'2022_full_per_channel.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    comparison=[]
    for data in [old,new]:
        for g,values in data['groups'].items():comparison.append({'year':data['year'],'group':g,**values})
    with (a.out/'2018_vs_2022_fullyear.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=comparison[0]);w.writeheader();w.writerows(comparison)
    (a.out/'2022_full_summary.json').write_text(json.dumps(new,indent=2))
    (a.out/'2022_data_manifest.json').write_text(json.dumps(cache,indent=2))
    shutil.copy2(a.cache/'2022_full_timestamps.csv',a.out/'2022_full_timestamps.csv')
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(4.5,2.7),layout='constrained')
    for off,key,label,color in [(-.17,'before_nrmse','CRA5 before','#2166ac'),(.17,'after_nrmse','CRA5 + U-Net','#d6604d')]:
        ax.bar(np.arange(2)+off,[d['groups']['direct19'][key] for d in [old,new]],width=.32,label=label,color=color)
    ax.set_xticks([0,1],['2018','2022']);ax.set(ylabel='Mean nRMSE (19 direct channels)',xlabel='Full test year (1,460 frames)')
    ax.legend(fontsize=9);fig.savefig(a.out/'2018_vs_2022_fullyear.pdf');fig.savefig(a.out/'2018_vs_2022_fullyear.png',dpi=250);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3),layout='constrained')
    ix=np.arange(len(rows));ax.plot(ix,[r['before_nrmse'] for r in rows],'o-',label='CRA5 before',color='#2166ac')
    ax.plot(ix,[r['after_nrmse'] for r in rows],'s-',label='CRA5 + U-Net',color='#d6604d')
    ax.set_xticks(ix,[r['channel']+('*' if r['channel']=='tcwv' else '') for r in rows],rotation=65)
    ax.set(ylabel='2022 nRMSE',xlabel='* Derived tcwv, excluded from direct-channel means')
    ax.legend(fontsize=9);fig.savefig(a.out/'2022_per_channel.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(4.5,2.7),layout='constrained')
    for off,key,label,color in [(-.17,'before_nrmse','CRA5 before','#2166ac'),(.17,'after_nrmse','CRA5 + U-Net','#d6604d')]:
        ax.bar(np.arange(2)+off,[new['groups'][g][key] for g in ['near','upper']],width=.32,label=label,color=color)
    ax.set_xticks([0,1],['Near-surface','Upper-air']);ax.set(ylabel='2022 mean nRMSE')
    ax.legend(fontsize=9);fig.savefig(a.out/'2022_near_upper.pdf');plt.close(fig)
    print(json.dumps(new['groups'],indent=2))

if __name__=='__main__':main()
