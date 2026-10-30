"""Assemble Table 1 only from full precision frozen evaluation JSONs."""
import argparse
import csv
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory',type=Path,required=True)
    args = p.parse_args()
    d = args.directory
    rows=[]
    for tag,label,matched in [('cra5','CRA5','reference'),('mpca_only','MPCA-only','2017-selected mean rate'),
                              ('e995','MPCA+PCA e995','non-matched'),('e99','MPCA+PCA e99','non-matched'),
                              ('e97','MPCA+PCA e97','non-matched')]:
        if tag == 'mpca_only':
            r=json.loads((d/tag/'summary.json').read_text())
            r0=r['rows'][0]
            channels=r0['channels']
            groups={'near':r0['group_gains']['near_surface'],'upper':r0['group_gains']['upper_air']}
            for group in ('all20','direct19'):
                names=[n for n in channels if group=='all20' or n!='tcwv']
                b=sum(channels[n]['before_nrmse'] for n in names)/len(names)
                a=sum(channels[n]['after_nrmse'] for n in names)/len(names)
                groups[group]={'before_nrmse':b,'after_nrmse':a,'G_pct':100*(1-a/b)}
            cr=r0['test_ratio']
        else:
            path=d/'pca_quantized'/f'{tag}.json' if tag.startswith('e') else d/f'{tag}.json'
            r=json.loads(path.read_text())
            groups=r['groups']
            cr=r['ratio_measured'] if tag.startswith('e') else 471.59613161721586
        row={'method':label,'year':2018,'frames':1460,'compression_ratio':cr,'rate_status':matched,
             'checkpoint':r['checkpoint'],'checkpoint_sha256':r['checkpoint_sha256'],'precision':r['precision']}
        for g in ('all20','direct19','near','upper'):
            for metric in ('before_nrmse','after_nrmse','G_pct'):
                row[f'{g}_{metric}']=groups[g][metric]
        row['near_upper_gain_ratio']=groups['near']['G_pct']/groups['upper']['G_pct']
        rows.append(row)
    notes=[
        'All metrics computed from raw evaluation arrays, never reconstructed from rounded tables.',
        'CRA5 and unquantized caches use float32 inference; quantized MPCA paths retain original autocast.',
        'New Table 1 e995/e99/e97 errors use actual quantized reconstructions at stored steps, paired with measured entropy-coded rates.',
        'Separate e995/e99/e97.json retain unquantized cache results for historical comparability.',
        'All methods reuse previously trained distinct checkpoints; no training/checkpoint search/rate tuning in revision.',
        'CRA5 global 268-channel codec and regional CONUS 20-channel MPCA differ in domain; only mean rate is approximately matched.',
        'CR includes amortized coder sigma storage as in original codec accounting, not correction network weights or full basis/checkpoint deployment costs.',
    ]
    (d/'revised_table1.json').write_text(json.dumps({'rows':rows,'notes':notes},indent=2))
    with (d/'revised_table1.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
