"""Conditional paired temporal-bootstrap differences: dilated minus original.

Both models share each resampled timestamp sequence. No model selection.
Units for gain differences are percentage points, ratios are dimensionless.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/tong/fourcastnet_us_mpca'))
    p.add_argument('--run',type=Path,required=True);a=p.parse_args();rows=[]
    for year in [2018,2022]:
        old=np.load(a.root/f'results/revision_2026/eval{year}/sufficient_statistics.npz')
        new=np.load(a.run/f'eval{year}/sufficient_statistics.npz')
        assert np.array_equal(old['before_time'],new['before_time']) and np.array_equal(old['valid_count'],new['valid_count'])
        names=old['channel_names'].astype(str).tolist();assert names==new['channel_names'].astype(str).tolist()
        near=['u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850']
        groups={'direct19':[i for i,n in enumerate(names) if n!='tcwv'],'near':[names.index(n) for n in near],
                'upper':[i for i,n in enumerate(names) if n not in near+['tcwv']]}
        def statistic(ix):
            before=np.sqrt(old['before_time'][ix].sum(0)/old['valid_count'][ix].sum(0))
            gains=[]
            for d in [old,new]:
                after=np.sqrt(d['after_time'][ix].sum(0)/d['valid_count'][ix].sum(0))
                g={name:100*(1-after[j].sum()/before[j].sum()) for name,j in groups.items()}
                g['near_minus_upper']=g['near']-g['upper'];gains.append(g)
            return {k:gains[1][k]-gains[0][k] for k in gains[0]}
        point=statistic(np.arange(1460));draws={k:[] for k in point};rng=np.random.default_rng(20261026)
        for _ in range(1000):
            starts=rng.integers(1460,size=53);ix=((starts[:,None]+np.arange(28))%1460).ravel()[:1460]
            for k,v in statistic(ix).items():draws[k].append(v)
        for k,v in point.items():
            lo,hi=np.percentile(draws[k],[2.5,97.5]);rows.append(dict(year=year,metric=k,dilated_minus_original_pp=v,lower95_pp=lo,upper95_pp=hi))
    with open(a.run/'paired_model_differences.csv','x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (a.run/'paired_model_differences.json').write_text(json.dumps(dict(method='paired circular7-day block percentile bootstrap,1000 draws,seed20261026',
         scope='conditional temporal uncertainty, excludes retraining seeds/spatial sampling',results=rows),indent=2))

if __name__=='__main__':main()
