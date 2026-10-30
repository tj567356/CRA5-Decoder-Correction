"""Paired circular moving-block bootstrap over full-year frame SSE, not grid pixels."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

NEAR=['u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850']
UPPER=['u500','v500','z500','t500','r500','z50']

def main():
    p=argparse.ArgumentParser()
    p.add_argument('statistics',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--block',type=int,default=28)
    p.add_argument('--repetitions',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20261026)
    args=p.parse_args()
    d=np.load(args.statistics)
    b,a,c=d['before_time'],d['after_time'],d['valid_count']
    names=d['channel_names'].astype(str).tolist(); n=len(b)
    assert b.shape==a.shape==c.shape and n==1460
    idxs={'direct19':[i for i,x in enumerate(names) if x!='tcwv'],
          'near':[names.index(x) for x in NEAR],'upper':[names.index(x) for x in UPPER]}
    def stat(ix):
        br=np.sqrt(b[ix].sum(0)/c[ix].sum(0)); ar=np.sqrt(a[ix].sum(0)/c[ix].sum(0))
        out={g:100*(1-ar[ixg].sum()/br[ixg].sum()) for g,ixg in idxs.items()}
        out['near_minus_upper']=out['near']-out['upper']
        return out
    estimate=stat(np.arange(n)); draws={g:[] for g in estimate}
    rng=np.random.default_rng(args.seed)
    for _ in range(args.repetitions):
        starts=rng.integers(n,size=int(np.ceil(n/args.block)))
        ix=((starts[:,None]+np.arange(args.block))%n).ravel()[:n]
        for g,v in stat(ix).items(): draws[g].append(v)
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for g in estimate:
        lo,hi=np.percentile(draws[g],[2.5,97.5])
        rows.append(dict(group=g,estimate_pct=estimate[g],lower_95_pct=lo,upper_95_pct=hi))
    with open(args.out/'confidence_intervals.csv','x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (args.out/'bootstrap_manifest.json').write_text(json.dumps({
        'source':str(args.statistics),'method':'paired circular moving block percentile bootstrap',
        'block_frames':args.block,'repetitions':args.repetitions,'seed':args.seed,
        'scope':'temporal uncertainty conditional on this fitted checkpoint, year and CONUS grid; not spatial or training-seed uncertainty'},indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
