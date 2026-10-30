"""Pre-specified 5x5 land-only elevation SD and paired reconstruction-error curves."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import convolve
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NEAR=['u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850']
UPPER=['u500','v500','z500','t500','r500','z50']

def roughness(elevation,land,min_neighbors=13):
    # Outside CONUS and ocean cells are excluded, never inserted as zero elevation.
    kernel=np.ones((5,5))
    count=convolve(land.astype(float),kernel,mode='constant',cval=0)
    total=convolve(np.where(land,elevation,0),kernel,mode='constant',cval=0)
    total2=convolve(np.where(land,elevation**2,0),kernel,mode='constant',cval=0)
    mean=np.divide(total,count,out=np.zeros_like(total),where=count>0)
    var=np.divide(total2,count,out=np.zeros_like(total2),where=count>0)-mean**2
    r=np.sqrt(np.maximum(var,0))
    r[~land | (count<min_neighbors)]=np.nan
    return r,count

def bins(r,mask):
    values=r[mask]
    if len(values)<100: return []
    edges=np.quantile(values,np.linspace(0,1,11))
    # Do not silently drop bins or jitter tied roughness values.
    if np.any(np.diff(edges)<=0): raise ValueError('Nonunique decile boundaries')
    return [(i,edges[i],edges[i+1],mask & (r>=edges[i]) &
             ((r<=edges[i+1]) if i==9 else (r<edges[i+1]))) for i in range(10)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--static',type=Path,required=True)
    p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--reference-grid',type=Path,required=True,help='2018 original Zarr grid')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    d=np.load(args.static); stat=np.load(args.statistics)
    import zarr
    ref=zarr.open_group(str(args.reference_grid),mode='r')
    lat,lon=d['lat'],d['lon']
    assert np.allclose(ref['lat'][:],lat,atol=1e-8)
    assert np.allclose(((ref['lon'][:]+180)%360)-180,lon,atol=1e-8)
    assert lat.shape==(97,) and lon.shape==(241,)
    h=d['z'].astype(float)/9.80665; land=d['lsm']>.5
    r,count=roughness(h,land)
    assert int(stat['frames'])==1460 and int(stat['year'])==2018
    before=stat['before_map']/1460; after=stat['after_map']/1460
    names=stat['channel_names'].astype(str).tolist()
    groups={'direct19':[i for i,x in enumerate(names) if x!='tcwv'],
            'near':[names.index(x) for x in NEAR],'upper':[names.index(x) for x in UPPER],
            'tcwv':[names.index('tcwv')]}
    valid=np.isfinite(r); masks={'all_land':valid,'west_land':valid & (lon[None,:]<-100)}
    rows=[]
    def compute(region,mask):
        for k,lo,hi,m in bins(r,mask):
            if not m.any(): raise ValueError('Empty bin')
            b=np.sqrt(before[:,m].mean(1)); a=np.sqrt(after[:,m].mean(1))
            for g,idx in groups.items():
                rows.append(dict(region=region,group=g,bin=k+1,roughness_lower_m=lo,
                    roughness_upper_m=hi,mean_roughness_m=float(r[m].mean()),
                    median_roughness_m=float(np.median(r[m])),grid_count=int(m.sum()),
                    mean_elevation_m=float(h[m].mean()),before_nrmse=float(b[idx].mean()),
                    after_nrmse=float(a[idx].mean()),G_pct=float(100*(1-a[idx].sum()/b[idx].sum())),
                    Gbar_pct=float(100*np.mean(1-a[idx]/b[idx]))))
    for region,m in masks.items(): compute(region,m)
    controls=[]
    for lo,hi in [(0,200),(200,600),(600,1200),(1200,2000),(2000,10000)]:
        m=masks['west_land'] & (h>=lo) & (h<hi)
        controls.append(dict(lower_m=lo,upper_m=hi,grid_count=int(m.sum()),
                             status='computed' if m.sum()>=100 else 'insufficient_cells'))
        if m.sum()>=100: compute(f'west_elevation_{lo}_{hi}',m)
    def write(name,records):
        with open(args.out/name,'x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    for region in masks:
        write(f'terrain_roughness_{region}.csv',[x for x in rows if x['region']==region and x['group']=='direct19'])
    write('terrain_roughness_near_upper.csv',[x for x in rows if x['group'] in ('near','upper')])
    write('terrain_roughness_all_groups_controls.csv',rows)
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'ps.fonttype':42})
    def curve(ax,region,g):
        data=[x for x in rows if x['region']==region and x['group']==g]
        xx=[x['mean_roughness_m'] for x in data]
        ax.plot(xx,[x['before_nrmse'] for x in data],'o-',color='#2166ac',label='CRA5 before correction',ms=4)
        ax.plot(xx,[x['after_nrmse'] for x in data],'s-',color='#d6604d',label='CRA5 + U-Net',ms=4)
        ax.set(xlabel='Terrain roughness (m)',ylabel='Reconstruction error (nRMSE)')
        ax.grid(alpha=.2)
    fig,axes=plt.subplots(1,2,figsize=(7,3.0),layout='constrained')
    for ax,region,title in zip(axes,masks,['(a) All CONUS land','(b) Land west of 100°W']):
        curve(ax,region,'direct19');ax.set_title(title,fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2,fontsize=10)
    fig.savefig(args.out/'terrain_roughness_main_curve.pdf');fig.savefig(args.out/'terrain_roughness_main_curve.png',dpi=250);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for ax,g in zip(axes,['near','upper']): curve(ax,'all_land',g);ax.set_title(g+' (all land)')
    axes[0].legend(fontsize=8);fig.savefig(args.out/'terrain_roughness_near_upper.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3),layout='constrained')
    for region in masks:
        data=[x for x in rows if x['region']==region and x['group']=='direct19']
        ax.plot([x['mean_roughness_m'] for x in data],[x['G_pct'] for x in data],'o-',label=region.replace('_',' '))
    ax.set(xlabel='Terrain roughness (m)',ylabel='Correction gain G (%)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(args.out/'terrain_roughness_gain.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,3),layout='constrained')
    im=ax.pcolormesh(lon,lat,r,shading='nearest',cmap='viridis');fig.colorbar(im,ax=ax,label='5×5 land elevation SD (m)')
    ax.set(xlabel='Longitude (°E)',ylabel='Latitude (°N)');fig.savefig(args.out/'terrain_roughness_map.pdf');plt.close(fig)
    corr={}
    for region,m in masks.items():
        corr[region]={'roughness_elevation_spearman':float(spearmanr(r[m],h[m]).statistic),'cells':int(m.sum())}
    result={'year':2018,'frames':1460,'roughness':'population SD, 5x5, masked land neighborhoods',
            'min_land_neighbors':13,'boundary':'exclude cells outside CONUS; no reflected or padded land',
            'land_rule':'ERA5 lsm > 0.5','normalization':'2014–2016 training channel std',
            'bins':'10 terrain-only deciles within each region/band; identical before/after membership',
            'main_channels':19,'excluded_land_cells':int((land & ~valid).sum()),
            'correlations':corr,'elevation_controls':controls,
            'inference':'Descriptive spatial associations only; cells are not independent replicates.',
            'source_statistics':str(args.statistics),'source_static':str(args.static)}
    (args.out/'terrain_roughness_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
