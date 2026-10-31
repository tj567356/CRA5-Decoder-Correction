"""Re-export paper tables/figures from recorded measurements, without inference.

This is result reconstruction, not a claim to rerun compression/training/data prep.
All outputs go to a new directory; original results and paper files are read-only.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RF = 'rf_ablation_d23_seed0_20261029'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def csvrows(p):
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def write(p, rows):
    with p.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def plot_module():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=['fig1','fig2','table1','table2','table3','all'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fcn-results', type=Path, default=ROOT/'research/fourcastnet/results')
    p.add_argument('--mpca-results', type=Path, default=ROOT/'research/new_20/results')
    p.add_argument('--revision-root', type=Path, default=ROOT/'revision_2026')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    results = a.fcn_results; revision = results/'revision_2026'
    targets = ['fig1','table1','table2','fig2','table3'] if a.target=='all' else [a.target]
    sources = []
    def load(path): sources.append(str(path)); return read(path)
    if 'fig1' in targets:
        rm = load(a.mpca_results/'rate_matched_val.json')
        s = load(revision/'eval2018/summary.json')['groups']['all20']
        rows = [{'method':'CRA5','before':s['before_nrmse'],'after':s['after_nrmse']}]
        for label, row in zip(['MPCA-only','MPCA + global PCA'], rm['rows']):
            rows.append({'method':label,'before':row['pre_cnn'],'after':row['post_cnn']})
        assert [round(x['after'],5) for x in rows]==[.06321,.08688,.12599]
        write(a.out/'fig1.csv',rows)
        plt=plot_module(); fig,ax=plt.subplots(figsize=(5.2,2.65),layout='constrained')
        for i,row in enumerate(rows):
            y=2-i; b=row['before']; c=row['after']
            ax.plot([c,b],[y,y],color='#777777')
            ax.scatter(b,y,facecolor='white',edgecolor='#2166ac',zorder=3)
            ax.scatter(c,y,color='#d6604d',zorder=3)
            ax.text(b,y+.17,f'{b:.5f}',ha='center',fontsize=9)
            ax.text(c,y-.28,f'{c:.5f}',ha='center',fontsize=9)
        ax.set_yticks([2,1,0],[x['method'] for x in rows])
        ax.set(xlim=(.049,.144),ylim=(-.7,2.55),xlabel='2018 mean nRMSE (20 channels)')
        ax.grid(axis='x',alpha=.25)
        fig.savefig(a.out/'fig1.pdf'); plt.close(fig)
    if 'table1' in targets:
        rows=[]
        for label in ['U-Net','Dilated','Affine']:
            for year in [2018,2022]:
                if label=='U-Net': path=revision/f'eval{year}/summary.json'
                elif label=='Dilated': path=revision/RF/f'eval{year}/summary.json'
                elif year==2018: path=a.revision_root/'linear_corrector/summary.json'
                else: path=revision/RF/'affine_eval2022/summary.json'
                g=load(path)['groups']
                rows.append({'corrector':label,'year':year,'After20':g['all20']['after_nrmse'],
                    'G20_pct':g['all20']['G_pct'],'G19_pct':g['direct19']['G_pct'],
                    'G_near_pct':g['near']['G_pct'],'G_upper_pct':g['upper']['G_pct']})
        assert [round(x['After20'],5) for x in rows]==[.06321,.06236,.06354,.06270,.08471,.08425]
        write(a.out/'table1.csv',rows)
    if 'table2' in targets:
        path=next(p for p in [revision/'baselines/revised_table1.csv',revision/'revised_table1.csv'] if p.is_file())
        sources.append(str(path))
        rows=[]
        for r in csvrows(path):
            for key in ['near','upper']:
                rows.append({'system':r['method'],'CR':r['compression_ratio'],
                    'status':r['rate_status'],'group':key,'before':r[key+'_before_nrmse'],
                    'after':r[key+'_after_nrmse'],'G_pct':r[key+'_G_pct']})
        assert len(rows)==10 and round(float(rows[0]['G_pct']),2)==32.18
        write(a.out/'table2.csv',rows)
    if 'fig2' in targets:
        plt=plot_module(); fig,axes=plt.subplots(1,2,figsize=(7.5,3.2),layout='constrained')
        exported=[]
        for ax,region in zip(axes,['all_land','west_land']):
            path=next(p for p in [revision/'terrain'/f'terrain_roughness_{region}.csv',
                                  revision/f'terrain_roughness_{region}.csv'] if p.is_file())
            sources.append(str(path))
            rows=[r for r in csvrows(path) if r['group']=='direct19']
            assert len(rows)==10
            x=[float(r['mean_roughness_m']) for r in rows]
            for key,label,marker in [('before_nrmse','Before','o'),('after_nrmse','After U-Net','s')]:
                ax.plot(x,[float(r[key]) for r in rows],marker=marker,label=label)
            ax.set(title=region,xlabel='Terrain roughness (m)',ylabel='nRMSE')
            ax.legend(); exported.extend(rows)
        write(a.out/'fig2.csv',exported)
        fig.savefig(a.out/'fig2.pdf'); plt.close(fig)
    if 'table3' in targets:
        path=results/'elevation_controls.csv'; sources.append(str(path))
        selected=[r for r in csvrows(path) if r['stratum'].startswith('陸+西 ')]
        rows=[]
        for band in ['0-200m','200-600m','600-1200m','1200-2000m','2000-+m']:
            near=next(r for r in selected if r['stratum']=='陸+西 '+band and r['group']=='near')
            upper=next(r for r in selected if r['stratum']=='陸+西 '+band and r['group']=='upper')
            assert near['n']==upper['n']
            rows.append({'elevation_band':band,'cells':int(near['n']),
                'near_gain_pct':float(near['gain_pct']),'upper_gain_pct':float(upper['gain_pct']),
                'ratio':float(near['gain_pct'])/float(upper['gain_pct']),
                'metric':'legacy pooled-error reduction, not channel-mean G'})
        assert [r['cells'] for r in rows]==[415,850,2311,3113,984]
        assert [round(r['ratio'],1) for r in rows]==[7.9,7.6,7.1,8.6,9.7]
        write(a.out/'table3.csv',rows)
    report={'mode':'reconstruction_from_stored_measurements_no_model_inference',
        'targets':targets,'sources':sources,'passed':True,
        'scope':'Numeric tables and re-plotted curves; not pixel-identical publication layout or end-to-end rerun.'}
    (a.out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
