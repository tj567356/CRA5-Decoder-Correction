"""Measure the fixed corrector's deployment cost after full CRA5 decoding releases GPU."""
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/home/tong/fourcastnet_us_mpca')
OUT=ROOT/'results/revision_2026/corrector_cost.json'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if OUT.exists():raise FileExistsError(OUT)
    done=ROOT/'revision_2026/data/cra5_decoded_2022/2022_decoded_manifest.json'
    while True:
        try:status=json.loads(done.read_text()).get('status')
        except (FileNotFoundError,json.JSONDecodeError):status=None
        if status=='complete':break
        time.sleep(30)
    sys.path.insert(0,str(ROOT/'scripts/cnn'))
    from model import UNet2
    ck=ROOT/'checkpoints/cra5/unet_d2_c64_lr3e-4_clip1/best.pt'
    original_hash=sha(ck)
    assert original_hash=='44c57a0f2a18e92c9c0745cfe75337079627ab8729e93c6ec098a5ddeb007435'
    model=UNet2(20,64,3,2,128).cuda().eval()
    model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=True))
    source=ROOT/'data/memmap_cra5/input_2017.npy'
    x=torch.from_numpy(np.array(np.load(source,mmap_mode='r')[:8],dtype=np.float32)).cuda()
    torch.backends.cudnn.benchmark=False
    with torch.inference_mode():
        for _ in range(10):model(x)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(100):model(x)
        end.record();torch.cuda.synchronize()
        batch_ms=begin.elapsed_time(end)/100
        peak=torch.cuda.max_memory_allocated()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=True,with_flops=True) as profile:
            model(x)
        torch.cuda.synchronize()
    flops=sum(e.flops for e in profile.key_averages())/8
    result={'checkpoint_sha256':original_hash,'parameters':sum(p.numel() for p in model.parameters()),
            'weight_float32_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),
            'serialized_checkpoint_bytes':ck.stat().st_size,'gpu':torch.cuda.get_device_name(),
            'torch':str(torch.__version__),'dtype':'float32','batch_size':8,
            'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'warmup_batches':10,'timed_batches':100,'mean_batch_ms':batch_ms,
            'amortized_ms_per_frame':batch_ms/8,'frames_per_second':8000/batch_ms,
            'peak_allocated_bytes':peak,'profiled_flops_per_frame':flops,
            'flops_scope':'PyTorch-profiler recognized operations, primarily convolutions; not a complete hardware-operation count',
            'timing_scope':'GPU-resident corrector forward passes only; excludes CRA5 decoding, data I/O, host-device transfer and evaluation statistics',
            'input':'first 8 fixed 2017 validation decoded cache frames; no model/parameter selection',
            'source_sha256':sha(Path(__file__))}
    assert sha(ck)==original_hash
    OUT.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
