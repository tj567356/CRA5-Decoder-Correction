"""Optional fixed 3x3 affine corrector, gated behind complete P0 2022 evaluation.

No training starts until results/revision_2026/2022_pipeline_complete.json says
status=complete, frames=1460, training_performed=false. 2022 values are never read.
Use --check for a standard-library-only protocol/source check (no torch import).
Run once in the original Ubuntu .venv; all outputs stay in revision_2026.
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

RECIPE = dict(seed=0, batch_size=8, learning_rate=3e-4, weight_decay=1e-4,
              grad_clip=1.0, maximum_epochs=200, patience=12, amp=True,
              optimizer='AdamW', objective='normalized residual MSE',
              selection='2017 validation MSE only', workers=4)
NEAR = ['u10','v10','t2m','sp','msl','u1000','v1000','z1000','u850','v850','z850','t850','r850']
UPPER = ['u500','v500','z500','t500','r500','z50']
STATS_HASH = '272cc45d231ec349025c4fa1e9f69366ae874739c98299ce666921821f13dba2'
UNET_HASH = '44c57a0f2a18e92c9c0745cfe75337079627ab8729e93c6ec098a5ddeb007435'


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def gate_ready(path):
    try:
        marker = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (marker.get('status') == 'complete' and marker.get('frames') == 1460
            and marker.get('training_performed') is False)


def protocol():
    return dict(architecture='Conv2d(20,20,kernel_size=3,padding=1,bias=True)',
                nonlinear_activation=False, parameters=20*20*3*3+20,
                receptive_field='3x3 grid cells; affine map including bias',
                train_years=[2014,2015,2016], validation_years=[2017],
                test_years=[2018], normalization='frozen 2014-2016 cache/statistics',
                recipe=RECIPE, test_evaluations=1,
                caveat='U-Net comparison changes capacity and receptive field as well as nonlinearity; no isolated nonlinearity attribution.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/tong/fourcastnet_us_mpca'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--check', action='store_true', help='No GPU/training or dataset access; print immutable protocol.')
    args = parser.parse_args()
    if args.check:
        assert protocol()['parameters'] == 3620
        assert RECIPE['maximum_epochs'] == 200 and RECIPE['patience'] == 12
        print(json.dumps(dict(status='implemented_not_executed', protocol=protocol(), source_sha256=digest(__file__)), indent=2))
        return
    root = args.root.resolve()
    out = (args.out or root/'revision_2026/linear_corrector').resolve()
    if not out.is_relative_to(root/'revision_2026'):
        raise ValueError('Output must be below this project revision_2026 directory')
    if out.exists():
        raise FileExistsError('One-shot output exists; refusing retraining or a second test evaluation: '+str(out))
    marker = root/'results/revision_2026/2022_pipeline_complete.json'
    print(utc(), 'WAITING_FOR_P0', marker, flush=True)
    while not gate_ready(marker):
        time.sleep(30)
    # All scientific imports/GPU initialization occur after the P0 marker.
    import numpy as np
    import torch
    from torch import nn
    from torch.utils.data import Dataset, ConcatDataset, DataLoader
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    cache = root/'data/memmap_cra5'
    stats = root/'mpca_train_basis_bundle/stats/train_2014_2016_channel_stats.npz'
    original = root/'checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt'
    if digest(stats) != STATS_HASH or digest(original) != UNET_HASH:
        raise ValueError('Original training statistics or protected U-Net checkpoint hash changed')
    names = np.load(stats)['channel_names'].astype(str).tolist()
    assert len(names) == 20 and set(NEAR+UPPER+['tcwv']) == set(names)
    provenance = dict(status='preparing', started_utc=utc(), protocol=protocol(),
                      source_sha256=digest(__file__), python=sys.version, torch=torch.__version__,
                      numpy=np.__version__, original_unet_sha256=UNET_HASH,
                      stats=str(stats), stats_sha256=STATS_HASH,
                      gate=str(marker), gate_sha256=digest(marker), cache=str(cache),
                      recipe_sources={str(p.relative_to(root)):digest(p) for p in
                                      [root/'scripts/cnn/train_cnn.py', root/'scripts/cnn/dataset.py',
                                       root/'scripts/cnn/run_cra5_unet.sh']})
    cache_manifest = None
    if (cache/'manifest.json').exists():
        provenance['cache_manifest_sha256'] = digest(cache/'manifest.json')
        cache_manifest = json.loads((cache/'manifest.json').read_text())
        if Path(cache_manifest['stats_path']).resolve() != stats.resolve():
            raise ValueError('Cache provenance uses different normalization statistics')
    else:
        provenance['cache_manifest_status'] = 'absent; full input/target hashes recorded independently'
    save_json(out/'provenance.json', provenance)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    provenance['device'] = str(device)
    file_records = []

    class CachedYear(Dataset):
        def __init__(self, year):
            self.year = year
            self.x = np.load(cache/f'input_{year}.npy', mmap_mode='r')
            self.y = np.load(cache/f'target_{year}.npy', mmap_mode='r')
            expected = 1464 if year == 2016 else 1460
            assert self.x.shape == self.y.shape == (expected,20,97,241)
            assert self.x.dtype == self.y.dtype == np.float16
            for kind in ['input','target']:
                p = cache/f'{kind}_{year}.npy'
                file_records.append(dict(path=str(p), bytes=p.stat().st_size, sha256=digest(p)))
                if cache_manifest:
                    array = self.x if kind == 'input' else self.y
                    recorded = cache_manifest['years'][str(year)][kind]
                    frame_hash = hashlib.sha1(np.ascontiguousarray(array[0]).tobytes()).hexdigest()
                    if recorded['shape'] != list(array.shape) or recorded['sha1_frame0'] != frame_hash:
                        raise ValueError('Cache file no longer matches original manifest: '+str(p))
        def __len__(self):
            return len(self.x)
        def __getitem__(self, i):
            return (torch.from_numpy(np.array(self.x[i], dtype=np.float32)),
                    torch.from_numpy(np.array(self.y[i], dtype=np.float32)))

    generator = torch.Generator().manual_seed(0)
    def loader(years, shuffle):
        return DataLoader(ConcatDataset([CachedYear(y) for y in years]),
                          batch_size=8, shuffle=shuffle, num_workers=4,
                          pin_memory=(device.type == 'cuda'), drop_last=False,
                          generator=generator)
    # Test cache is deliberately not opened or hashed before checkpoint selection.
    train = loader([2014,2015,2016], True)
    val = loader([2017], False)
    model = nn.Conv2d(20,20,3,padding=1,bias=True).to(device)
    assert sum(p.numel() for p in model.parameters()) == 3620
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = nn.MSELoss()
    def validation_loss():
        model.eval()
        total, batches = 0., 0
        with torch.inference_mode():
            for x,y in val:
                x,y=x.to(device),y.to(device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    loss=criterion(model(x),y)
                total += float(loss)
                batches += 1
        return total/batches
    preparation_seconds = time.perf_counter()-started
    best, best_epoch, stale = math.inf, None, 0
    history = []
    train_started = time.perf_counter()
    with (out/'train_validation.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle, fieldnames=['epoch','train_loss','validation_loss','seconds','selected'])
        writer.writeheader()
        for epoch in range(1,201):
            tick=time.perf_counter()
            model.train()
            total,batches=0.,0
            for x,y in train:
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                    loss=criterion(model(x),y)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer)
                scaler.update()
                total += float(loss.detach())
                batches += 1
            value=validation_loss()
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite validation loss')
            selected=value<best
            if selected:
                best,best_epoch,stale=value,epoch,0
                torch.save(model.state_dict(),out/'best.pt.tmp')
                (out/'best.pt.tmp').replace(out/'best.pt')
            else:
                stale += 1
            row=dict(epoch=epoch,train_loss=total/batches,validation_loss=value,
                     seconds=time.perf_counter()-tick,selected=selected)
            writer.writerow(row);handle.flush();history.append(row)
            print(utc(),json.dumps(row),flush=True)
            if stale>=12:
                break
    training_seconds=time.perf_counter()-train_started
    selected_hash=digest(out/'best.pt')
    save_json(out/'selection.json',dict(selected_epoch=best_epoch,validation_MSE=best,
              checkpoint_sha256=selected_hash,criterion='2017 normalized validation MSE',
              epochs_executed=len(history),test_access_before_selection=False,selected_utc=utc()))
    # One and only one full test traversal after selection; float32 frozen inference.
    test=loader([2018],False)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
    model.eval()
    test_started=time.perf_counter()
    before_time,after_time=[],[]
    with torch.inference_mode():
        for x,y in test:
            target=y.numpy().astype(np.float64)
            pred=model(x.to(device)).cpu().numpy().astype(np.float64)
            assert np.isfinite(target).all() and np.isfinite(pred).all()
            before_time.append((target**2).sum(axis=(2,3)))
            after_time.append(((pred-target)**2).sum(axis=(2,3)))
    before_time=np.concatenate(before_time);after_time=np.concatenate(after_time)
    assert before_time.shape==after_time.shape==(1460,20)
    b=np.sqrt(before_time.sum(0)/(1460*97*241))
    a=np.sqrt(after_time.sum(0)/(1460*97*241))
    groups={'direct19':[i for i,n in enumerate(names) if n!='tcwv'],'all20':list(range(20)),
            'near':[names.index(n) for n in NEAR],'upper':[names.index(n) for n in UPPER],
            'tcwv':[names.index('tcwv')]}
    metrics={g:dict(before_nrmse=float(b[ix].mean()),after_nrmse=float(a[ix].mean()),
                   G_pct=float(100*(1-a[ix].sum()/b[ix].sum())),
                   Gbar_pct=float(100*np.mean(1-a[ix]/b[ix]))) for g,ix in groups.items()}
    channels=[dict(channel=n,group=('near' if n in NEAR else 'upper' if n in UPPER else 'derived'),
                   before_nrmse=float(b[i]),after_nrmse=float(a[i]),G_pct=float(100*(1-a[i]/b[i])))
              for i,n in enumerate(names)]
    with (out/'2018_per_channel.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(channels[0]));writer.writeheader();writer.writerows(channels)
    np.savez_compressed(out/'2018_time_statistics.npz',before_time=before_time,after_time=after_time,
                        valid_count=np.full_like(before_time,97*241),channel_names=names,frames=1460,year=2018)
    summary=dict(status='complete',test_year=2018,frames=1460,test_evaluations=1,
                 protocol=protocol(),groups=metrics,checkpoint_sha256=selected_hash,
                 selected_epoch=best_epoch,best_validation_MSE=best,epochs_executed=len(history),
                 timings_seconds=dict(preparation=preparation_seconds,training=training_seconds,
                                      test=time.perf_counter()-test_started,total=time.perf_counter()-started),
                 precision='AMP training/validation, float32 test inference, float64 normalized error aggregation',
                 test_data_not_used_for_selection=True,finished_utc=utc())
    if digest(original)!=UNET_HASH or digest(stats)!=STATS_HASH or digest(out/'best.pt')!=selected_hash:
        raise ValueError('Protected/source or selected-checkpoint hash changed during execution')
    provenance.update(status='complete',cache_files=file_records,finished_utc=utc())
    save_json(out/'provenance.json',provenance)
    save_json(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
