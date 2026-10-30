"""Replay the existing 2017-selected MPCA-only rate without any rate search.

Runs on the original Ubuntu host in the original .venv. The effective source is
saved beside the results for an exact audit trail. No original result is written.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path('/home/tong/fourcastnet_us_mpca')
NEW = Path('/home/tong/new_20')
OUT = ROOT / 'results/revision_2026/baselines/mpca_only'
OUT.mkdir(parents=True, exist_ok=False)
source_path = NEW / 'scripts/rate_matched_val.py'
source = source_path.read_text()
original_sha256 = hashlib.sha256(source.encode()).hexdigest()
source = source.replace('st = np.load(STATS)', 'CONFIGS = CONFIGS[:1]\n\nst = np.load(STATS)')
source = source.replace('for y in (VAL, TEST)}\n\nprint(', 'for y in (TEST,)}\n\nprint(')
start = source.index('    lo, hi = cfg["bracket"]')
end = source.index('    b_val, _ = coded(D, VAL)', start)
source = source[:start] + '''    # Exact frozen choice recorded in the existing 2017 validation result.
    D = 0.7659035172375452
''' + source[end:]
source = source.replace('"n_channels": len(idx),', '"n_channels": len(idx),\n            "before_nrmse": float(b.mean()),\n            "after_nrmse": float(a.mean()),')
source = source.replace('"group_gains": group_gains})', '"group_gains": group_gains,\n                "channels": {v:{"before_nrmse":float(before_ch[i]), "after_nrmse":float(after_ch[i])} for i,v in enumerate(names)}})')
start = source.index('rows = [run(c) for c in CONFIGS]')
source = source[:start] + '''rows = [run(c) for c in CONFIGS]
import hashlib
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
result = {
    "status": "newly_executed_frozen_forward_evaluation",
    "frames": 1460,
    "year": TEST,
    "frozen_step": 0.7659035172375452,
    "step_selected_on": 2017,
    "no_bisection_or_training": True,
    "checkpoint": str(CONFIGS[0]["ckpt"]),
    "checkpoint_sha256": digest(CONFIGS[0]["ckpt"]),
    "stats": str(STATS), "stats_sha256": digest(STATS),
    "precision": "original float32 physical reconstruction; float16 autocast correction; float64 error aggregation",
    "rows": rows,
}
(OUTPUT / "summary.json").write_text(json.dumps(result, indent=2))
'''
effective_path = OUT / 'effective_rate_matched_frozen.py'
effective_path.write_text(source)
(OUT / 'source_provenance.json').write_text(json.dumps({
    'original_source':str(source_path), 'original_source_sha256':original_sha256,
    'effective_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'modifications':['first configuration only','exact frozen quantizer step; search removed',
                     'ground truth test year only','add unrounded group and channel metrics',
                     'write only independent revision directory'],
}, indent=2))
exec(compile(source, str(effective_path), 'exec'), {'__name__':'__main__', 'OUTPUT':OUT})
