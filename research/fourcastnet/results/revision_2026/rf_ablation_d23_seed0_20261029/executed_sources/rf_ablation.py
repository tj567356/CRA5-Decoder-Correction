"""Fixed, parameter-matched CRA5 bottleneck RF ablation. No test selection.

Outputs are new, exclusive directories. Original model/checkpoint are read-only.
Training: 2014-16; checkpoint selection: 2017 batch-mean normalized MSE.
Only after selection: full 2018 and full aligned 2022, RF diagnostics and costs.
"""
import argparse
import ast
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, ConcatDataset, DataLoader

ORIGINAL_HASH='44c57a0f2a18e92c9c0745cfe75337079627ab8729e93c6ec098a5ddeb007435'
STATS_HASH='272cc45d231ec349025c4fa1e9f69366ae874739c98299ce666921821f13dba2'
NEAR=['u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850']
UPPER=['u500','v500','z500','t500','r500','z50']
RECIPE=dict(seed=0,batch_size=8,learning_rate=3e-4,weight_decay=1e-4,
            gradient_clip=1.0,maximum_epochs=200,patience=12,amp=True,workers=4,
            objective='normalized residual MSE',optimizer='AdamW',
            selection='2017 mean batch validation MSE, including final partial batch')

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False))
def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def load_module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def cls_ast(p,name):
    return ast.dump(next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name),include_attributes=False)

def validate_models(module,original_path,out):
    assert cls_ast(Path(module.__file__),'UNet2')==cls_ast(original_path,'UNet2')
    torch.manual_seed(0);a=module.UNet2(20,64,3,2,128)
    torch.manual_seed(0);b=module.DilatedUNet2(20,64,3,2,128)
    assert type(b).forward is type(a).forward
    assert len(a.state_dict())==len(b.state_dict())
    assert len(list(a.modules()))==len(list(b.modules()))
    changes=[]
    for (ka,va),(kb,vb) in zip(a.state_dict().items(),b.state_dict().items()):
        assert ka==kb and va.shape==vb.shape and torch.equal(va,vb)
    for (name,ma),(nb,mb) in zip(a.named_modules(),b.named_modules()):
        assert name==nb
        if isinstance(ma,nn.Conv2d):
            da={k:getattr(ma,k) for k in ['in_channels','out_channels','kernel_size','stride','padding','dilation','groups','padding_mode']}
            db={k:getattr(mb,k) for k in da}
            if da!=db:changes.append(dict(module=name,original=da,dilated=db))
    assert [x['module'] for x in changes]==['bottleneck.0','bottleneck.2']
    for net in [a,b]:
        assert sum(p.numel() for p in net.parameters())==1646548
        net=net.cuda().eval()
        for batch in [1,8]:
            with torch.inference_mode():assert tuple(net(torch.zeros(batch,20,97,241,device='cuda')).shape)==(batch,20,97,241)
        net.cpu()
    result=dict(parameters_original=1646548,parameters_dilated=1646548,
                original_UNet2_AST_unchanged=True,forward_inherited_unchanged=True,
                identical_initial_weights_and_state_keys=True,only_conv_metadata_changes=changes,
                tested_shapes=[[1,20,97,241],[8,20,97,241]],passed=True)
    save(out/'structural_validation.json',result)

class CacheYear(Dataset):
    def __init__(self,cache,year,records):
        self.x=np.load(cache/f'input_{year}.npy',mmap_mode='r');self.y=np.load(cache/f'target_{year}.npy',mmap_mode='r')
        n=1464 if year==2016 else 1460
        assert self.x.shape==self.y.shape==(n,20,97,241) and self.x.dtype==self.y.dtype==np.float16
        records.append(dict(year=year,frames=n,input=str(cache/f'input_{year}.npy'),target=str(cache/f'target_{year}.npy')))
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return torch.from_numpy(np.array(self.x[i],dtype=np.float32)),torch.from_numpy(np.array(self.y[i],dtype=np.float32))

def summarize(before,after,names):
    groups={'direct19':[i for i,n in enumerate(names) if n!='tcwv'],'all20':list(range(20)),
            'near':[names.index(n) for n in NEAR],'upper':[names.index(n) for n in UPPER]}
    d={k:dict(before_nrmse=float(before[ix].mean()),after_nrmse=float(after[ix].mean()),
              G_pct=float(100*(1-after[ix].sum()/before[ix].sum())),
              Gbar_pct=float(100*np.mean(1-after[ix]/before[ix]))) for k,ix in groups.items()}
    d['near_upper_gain_ratio']=d['near']['G_pct']/d['upper']['G_pct'] if d['upper']['G_pct']!=0 else None
    d['near_minus_upper_percentage_points']=d['near']['G_pct']-d['upper']['G_pct']
    return d

def evaluate(net,cache,year,out,names,checkpoint_hash):
    out.mkdir(exist_ok=False)
    x=np.load(cache/f'input_{year}.npy',mmap_mode='r');y=np.load(cache/f'target_{year}.npy',mmap_mode='r')
    assert x.shape==y.shape==(1460,20,97,241)
    if year==2022:
        m=json.loads((cache/'manifest.json').read_text())
        assert m['complete'] and m['alignment_verified'] and m['frames']==1460 and m['year']==2022
        assert m['channels']==names and m['stats_sha256']==STATS_HASH
        for key,path in [('input_sha256',cache/'input_2022.npy'),('target_sha256',cache/'target_2022.npy'),('timestamps_sha256',cache/'2022_full_timestamps.csv')]:assert m[key]==sha(path)
        save(out/'verified_data_manifest.json',m)
    b=np.zeros((1460,20),dtype=np.float64);a=np.zeros_like(b);start=time.perf_counter();net.eval()
    with torch.inference_mode():
        for s in range(0,1460,8):
            xx=np.array(x[s:s+8],dtype=np.float32);yy=np.array(y[s:s+8],dtype=np.float64)
            pp=net(torch.from_numpy(xx).cuda()).cpu().numpy().astype(np.float64)
            assert np.isfinite(pp).all() and np.isfinite(yy).all()
            b[s:s+8]=(yy**2).sum((2,3));a[s:s+8]=((pp-yy)**2).sum((2,3))
    br=np.sqrt(b.sum(0)/(1460*97*241));ar=np.sqrt(a.sum(0)/(1460*97*241))
    result=dict(year=year,frames=1460,checkpoint_sha256=checkpoint_hash,
                groups=summarize(br,ar,names),evaluation_seconds=time.perf_counter()-start,
                precision='float32 inference; float64 SSE; original normalized float16 cache',
                channels={n:dict(before_nrmse=float(br[i]),after_nrmse=float(ar[i]),G_pct=float(100*(1-ar[i]/br[i]))) for i,n in enumerate(names)})
    np.savez_compressed(out/'sufficient_statistics.npz',before_time=b,after_time=a,valid_count=np.full_like(b,97*241),channel_names=names)
    save(out/'summary.json',result)
    with open(out/'per_channel.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['channel','before_nrmse','after_nrmse','G_pct']);w.writeheader()
        for n,row in result['channels'].items():w.writerow(dict(channel=n,**row))
    print('EVALUATED',year,json.dumps(result['groups']),flush=True)
    return result

def costs(net,x):
    net.eval()
    with torch.inference_mode():
        for _ in range(10):net(x)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(100):net(x)
        end.record();torch.cuda.synchronize();ms=begin.elapsed_time(end)/100
        peak=torch.cuda.max_memory_allocated()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True,record_shapes=True) as p:net(x)
        torch.cuda.synchronize()
    return dict(parameters=sum(p.numel() for p in net.parameters()),weight_bytes=sum(p.numel()*p.element_size() for p in net.parameters()),
                batch_size=8,mean_batch_ms=ms,amortized_ms_per_frame=ms/8,frames_per_second=8000/ms,
                peak_allocated_bytes=peak,profiled_flops_per_frame=sum(e.flops for e in p.key_averages())/8,
                scope='GPU-resident forward only, fixed 2017 inputs, 10 warmup/100 timed batches; excludes decoding/I-O/transfers')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/home/tong/fourcastnet_us_mpca'))
    p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();root=args.root;out=args.out;out.mkdir(parents=True,exist_ok=False)
    original=root/'checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt';stats=root/'mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz'
    assert sha(original)==ORIGINAL_HASH and sha(stats)==STATS_HASH
    mod=load_module(args.model,'rf_ablation_model');validate_models(mod,root/'scripts/cnn/model.py',out)
    if args.validate_only:return
    assert torch.cuda.is_available()
    provenance=dict(started_utc=utc(),status='training',intervention='only bottleneck dilation/padding (1,1)->(2,3)',
                    purpose='parameter-matched receptive-field ablation, not architecture search',recipe=RECIPE,
                    original_checkpoint_sha256=ORIGINAL_HASH,stats_sha256=STATS_HASH,model_source_sha256=sha(args.model),
                    training_source_sha256=sha(Path(__file__)),original_model_source_sha256=sha(root/'scripts/cnn/model.py'),
                    torch=str(torch.__version__),numpy=str(np.__version__),gpu=torch.cuda.get_device_name(),
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,original_training_log_sha256=sha(root/'checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/train.log') if (root/'checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/train.log').exists() else None)
    save(out/'provenance.json',provenance)
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    net=mod.DilatedUNet2(20,64,3,2,128).cuda()
    cache=root/'data/memmap_cra5';records=[];generator=torch.Generator().manual_seed(0)
    def loader(years,shuffle):return DataLoader(ConcatDataset([CacheYear(cache,y,records) for y in years]),batch_size=8,
              shuffle=shuffle,num_workers=4,pin_memory=True,drop_last=False,generator=generator)
    train=loader([2014,2015,2016],True);val=loader([2017],False)
    manifest=json.loads((cache/'manifest.json').read_text());assert 'cra5' in manifest['decoded_dir'].lower()
    save(out/'training_data.json',dict(records=records,cache_manifest=manifest,cache_manifest_sha256=sha(cache/'manifest.json'),
                                     test_arrays_not_opened_before_selection=True))
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=True)
    criterion=nn.MSELoss();best=float('inf');stale=0;history=[];started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,201):
        t=time.perf_counter();net.train();losses=[]
        for x,y in train:
            x=x.cuda(non_blocking=False);y=y.cuda(non_blocking=False)
            with torch.autocast(device_type='cuda',dtype=torch.float16):loss=criterion(net(x),y)
            assert torch.isfinite(loss)
            opt.zero_grad();scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(net.parameters(),1.0)
            scaler.step(opt);scaler.update();losses.append(loss.item())
        net.eval();vl=[]
        with torch.no_grad():
            for x,y in val:
                with torch.autocast(device_type='cuda',dtype=torch.float16):loss=criterion(net(x.cuda()),y.cuda())
                assert torch.isfinite(loss);vl.append(loss.item())
        v=float(np.mean(vl));improved=v<best
        if improved:best=v;stale=0;torch.save(net.state_dict(),out/'best.pt');best_epoch=epoch
        else:stale+=1
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_loss=v,seconds=time.perf_counter()-t,improved=improved)
        history.append(row);save(out/'history.json',history)
        print(json.dumps(row),flush=True)
        if stale>=12:break
    torch.cuda.synchronize();elapsed=time.perf_counter()-started;peak=torch.cuda.max_memory_allocated()
    ck_hash=sha(out/'best.pt')
    selection=dict(selected_utc=utc(),best_epoch=best_epoch,stopped_epoch=epoch,best_val_loss=best,
                   criterion=RECIPE['selection'],checkpoint_sha256=ck_hash,training_seconds=elapsed,
                   training_peak_allocated_bytes=peak,training_steps=epoch*len(train),validation_steps=epoch*len(val),
                   test_data_used_for_selection=False,recipe=RECIPE)
    save(out/'selection.json',selection)  # Persist selection before opening either test year.
    net.load_state_dict(torch.load(out/'best.pt',map_location='cuda',weights_only=True));net.eval()
    names=np.load(stats)['channel_names'].astype(str).tolist()
    annual={}
    for year,mem in [(2018,cache),(2022,root/'revision_2026/data/memmap_cra5_2022')]:
        annual[str(year)]=evaluate(net,mem,year,out/f'eval{year}',names,ck_hash)
    original_net=mod.UNet2(20,64,3,2,128).cuda().eval()
    original_net.load_state_dict(torch.load(original,map_location='cuda',weights_only=True))
    x=torch.from_numpy(np.array(np.load(cache/'input_2017.npy',mmap_mode='r')[:8],dtype=np.float32)).cuda()
    save(out/'inference_cost.json',dict(original=costs(original_net,x),dilated=costs(net,x),
             gpu=torch.cuda.get_device_name(),torch=str(torch.__version__),cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
             dtype='float32',cudnn_deterministic=torch.backends.cudnn.deterministic,
             cudnn_benchmark=torch.backends.cudnn.benchmark,
             profiler_scope='recognized operations, primarily convolutions; not complete hardware operations'))
    assert sha(original)==ORIGINAL_HASH and sha(stats)==STATS_HASH and sha(out/'best.pt')==ck_hash
    provenance.update(status='evaluation_complete',finished_utc=utc(),selection=selection)
    save(out/'provenance.json',provenance)
    save(out/'performance_complete.json',dict(status='complete',years=[2018,2022],frames_each=1460,checkpoint_sha256=ck_hash))

if __name__=='__main__':main()
