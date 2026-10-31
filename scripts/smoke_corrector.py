"""Frozen inference on four real 2017 validation-cache frames; not a yearly test."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=ROOT)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); root=a.repo
    manifest=json.loads((root/'examples/cra5_2017_smoke.json').read_text())
    sample=root/'examples/cra5_2017_smoke.npz'
    ck=root/'artifacts/correctors/cra5/best.pt'
    stats=root/'artifacts/statistics/train_2014_2016_channel_stats.npz'
    assert sha(sample)==manifest['sample_sha256']
    assert sha(ck)==manifest['checkpoint_sha256'] and sha(stats)==manifest['stats_sha256']
    model=root/'research/fourcastnet/scripts/cnn/model.py'
    spec=importlib.util.spec_from_file_location('smoke_model',model)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    net=module.UNet2(20,64,3,2,128).to(a.device).eval()
    assert sum(p.numel() for p in net.parameters())==1646548
    net.load_state_dict(torch.load(ck,map_location=a.device,weights_only=True))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=np.load(sample,allow_pickle=False)
    x=data['input'].astype(np.float32); y=data['target'].astype(np.float32)
    assert x.shape==y.shape==(4,20,97,241)
    with torch.inference_mode():pred=net(torch.from_numpy(x).to(a.device)).cpu().numpy()
    assert pred.shape==x.shape and np.isfinite(pred).all()
    before=np.sqrt(np.mean(y.astype(np.float64)**2,axis=(0,2,3)))
    after=np.sqrt(np.mean((pred.astype(np.float64)-y.astype(np.float64))**2,axis=(0,2,3)))
    expected=np.asarray(manifest['expected_after_per_channel_nrmse'])
    np.testing.assert_allclose(after,expected,rtol=1e-3,atol=1e-5)
    result={'passed':True,'frames':4,'year':2017,'split':'validation','device':a.device,
        'input_shape':list(x.shape),'output_shape':list(pred.shape),'parameters':1646548,
        'before_all20_nrmse':float(before.mean()),'after_all20_nrmse':float(after.mean()),
        'before_per_channel_nrmse':before.tolist(),'after_per_channel_nrmse':after.tolist(),
        'scope':'Four-frame frozen inference smoke test; not full-year paper accuracy.'}
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
