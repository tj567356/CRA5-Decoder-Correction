"""Frozen inference and sufficient statistics; never trains or selects weights."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

NEAR = ['u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850']
UPPER = ['u500','v500','z500','t500','r500','z50']

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def groups(names):
    return {'direct19':[i for i,n in enumerate(names) if n != 'tcwv'],
            'all20':list(range(20)), 'near':[names.index(n) for n in NEAR],
            'upper':[names.index(n) for n in UPPER], 'tcwv':[names.index('tcwv')]}

def metrics(b, a, names):
    return {g: {'before_nrmse':float(b[idx].mean()),'after_nrmse':float(a[idx].mean()),
                'G_pct':float(100*(1-a[idx].sum()/b[idx].sum())),
                'Gbar_pct':float(100*np.mean(1-a[idx]/b[idx]))}
            for g,idx in groups(names).items()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/home/tong/fourcastnet_us_mpca'))
    p.add_argument('--year',type=int,default=2018)
    p.add_argument('--mem',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--batch',type=int,default=8)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    root=args.root
    sys.path.insert(0,str(root/'scripts/cnn'))
    from model import UNet2
    stats=root/'mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz'
    assert sha256(stats)=='272cc45d231ec349025c4fa1e9f69366ae874739c98299ce666921821f13dba2'
    names=np.load(stats)['channel_names'].astype(str).tolist()
    ck=args.checkpoint or root/'checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt'
    mem=args.mem or root/'data/memmap_cra5'
    x=np.load(mem/f'input_{args.year}.npy',mmap_mode='r')
    y=np.load(mem/f'target_{args.year}.npy',mmap_mode='r')
    assert x.shape==y.shape==(1460,20,97,241), (x.shape,y.shape)
    if args.year != 2018:
        manifest=json.loads((mem/'manifest.json').read_text())
        assert manifest['complete'] and manifest['alignment_verified'] and manifest['year']==args.year
        assert manifest['frames']==1460 and manifest['channels']==names
        assert manifest['stats_sha256']==sha256(stats)
        assert manifest['input_sha256']==sha256(mem/f'input_{args.year}.npy')
        assert manifest['target_sha256']==sha256(mem/f'target_{args.year}.npy')
        assert manifest['timestamps_sha256']==sha256(mem/f'{args.year}_full_timestamps.csv')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    net=UNet2(20,64,3,2,128).to(device).eval()
    ck_hash=sha256(ck)
    if args.year==2022:
        assert ck_hash=='44c57a0f2a18e92c9c0745cfe75337079627ab8729e93c6ec098a5ddeb007435'
        assert manifest['normalization']=='frozen 2014–2016, float32 arithmetic then float16 cache'
    net.load_state_dict(torch.load(ck,map_location=device,weights_only=True))
    n=len(x)
    sb=np.zeros((20,97,241),dtype=np.float64)
    sa=np.zeros_like(sb)
    tb=np.zeros((n,20),dtype=np.float64)
    ta=np.zeros_like(tb)
    evaluation_start=time.perf_counter()
    with torch.inference_mode():
        for s in range(0,n,args.batch):
            e=min(s+args.batch,n)
            xb=np.array(x[s:e],dtype=np.float32)
            target=np.array(y[s:e],dtype=np.float64)
            assert np.isfinite(xb).all() and np.isfinite(target).all()
            pred=net(torch.from_numpy(xb).to(device)).cpu().numpy().astype(np.float64)
            assert np.isfinite(pred).all()
            b=target**2
            a=(pred-target)**2
            sb+=b.sum(axis=0); sa+=a.sum(axis=0)
            tb[s:e]=b.sum(axis=(2,3)); ta[s:e]=a.sum(axis=(2,3))
            if s%160==0: print(f'{e}/{n}',flush=True)
    assert sha256(ck)==ck_hash
    evaluation_seconds=time.perf_counter()-evaluation_start
    np.savez_compressed(args.out/'sufficient_statistics.npz',before_map=sb,after_map=sa,
                        before_time=tb,after_time=ta,valid_count=np.full_like(tb,97*241),
                        channel_names=names,frames=n,year=args.year)
    b=np.sqrt(tb.sum(axis=0)/(n*97*241)); a=np.sqrt(ta.sum(axis=0)/(n*97*241))
    result={'year':args.year,'frames':n,'checkpoint':str(ck),'checkpoint_sha256':ck_hash,
            'stats':str(stats),'stats_sha256':sha256(stats),'input':str(mem),
            'device':device,'precision':'float32 inference; float64 aggregation; existing float16 cache',
            'implementation_sha256':{'evaluation_script':sha256(Path(__file__)),
                                     'model_source':sha256(root/'scripts/cnn/model.py')},
            'torch_version':str(torch.__version__),'numpy_version':str(np.__version__),
            'evaluation_seconds':evaluation_seconds,
            'timing_scope':'full evaluation loop including cache reads, transfers, forward passes and SSE; not isolated inference latency',
            'parameters':sum(v.numel() for v in net.parameters()),'groups':metrics(b,a,names),
            'channels':{name:{'before_nrmse':float(b[i]),'after_nrmse':float(a[i])} for i,name in enumerate(names)}}
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['groups'],indent=2),flush=True)

if __name__=='__main__': main()
