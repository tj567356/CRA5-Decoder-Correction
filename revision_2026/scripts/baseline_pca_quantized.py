"""Frozen deployment-rate MPCA+PCA evaluation; no checkpoint or step selection."""
from pathlib import Path
import hashlib
import json

ROOT = Path('/home/tong/fourcastnet_us_mpca')
OUT = ROOT / 'results/revision_2026/baselines/pca_quantized'
OUT.mkdir(parents=True, exist_ok=False)
source_path = ROOT / 'scripts/analysis/codec_all_cnn.py'
source = source_path.read_text()
original_hash = hashlib.sha256(source.encode()).hexdigest()
start = source.index('    R0 = decode(S)')
end = source.index('\n\nrows = []', start)
source = source[:start] + '''    path = BASE / f"checkpoints/{energy}/unet_d2_c64_lr3e-4_clip1/best.pt"
    net = UNet2(20, 64, 3, 2, 128).to(dev).eval()
    net.load_state_dict(torch.load(path, map_location=dev, weights_only=True))
    record = next(r for r in json.load(open(BASE / "results/pca_codec_all.json")) if r["energy"] == energy)
    D = record["quant_step"]
    Rq = decode(np.rint(S / D) * D)
    Pq = refine(net, Rq)
    def channel_nr(x):
        return np.sqrt(((x.astype(np.float64)-T)**2).mean(axis=(0,2,3))) / std_ref
    b, a = channel_nr(Rq), channel_nr(Pq)
    near = ["u10","v10","t2m","sp","msl","u1000","v1000","z1000","u850","v850","z850","t850","r850"]
    upper = ["u500","v500","z500","t500","r500","z50"]
    groups = {"all20":list(range(20)), "direct19":[i for i,v in enumerate(names) if v != "tcwv"],
              "near":[names.index(v) for v in near], "upper":[names.index(v) for v in upper]}
    import hashlib
    result = {"energy":energy, "year":2018, "frames":1460,
              "quant_step":D, "ratio_measured":record["ratio_measured"],
              "checkpoint":str(path), "checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
              "precision":"float32 physical reconstruction; float16 autocast correction; float64 errors",
              "no_training_or_selection":True,
              "groups":{g:{"before_nrmse":float(b[ii].mean()), "after_nrmse":float(a[ii].mean()),
                           "G_pct":float(100*(1-a[ii].sum()/b[ii].sum())),
                           "Gbar_pct":float(100*np.mean(1-a[ii]/b[ii]))} for g,ii in groups.items()},
              "channels":{v:{"before_nrmse":float(b[i]), "after_nrmse":float(a[i])} for i,v in enumerate(names)}}
    (OUTPUT / f"{energy}.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result["groups"],indent=2),flush=True)
    return result

rows = [run(e) for e in ("e995", "e99", "e97")]
(OUTPUT / "summary.json").write_text(json.dumps(rows,indent=2))
'''
effective_path = OUT / 'effective_codec_all_cnn.py'
effective_path.write_text(source)
(OUT / 'source_provenance.json').write_text(json.dumps({
    'source':str(source_path), 'source_sha256':original_hash,
    'effective_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'changes':['remove test-based checkpoint reproduction/selection',
               'fixed pre-existing checkpoint path per compressor',
               'exact existing quant_step from pca_codec_all.json',
               'add full precision group/channel errors', 'revision output only'],
},indent=2))
exec(compile(source,str(effective_path),'exec'), {'__name__':'__main__','OUTPUT':OUT})
