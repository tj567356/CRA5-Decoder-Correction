"""Package the current manuscript's actual TeX dependencies and per-file SHA256."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--paper',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();paper=a.paper.resolve()
    paths={paper/'main.tex',paper/'refs.bib',paper/'main.bbl',paper/'llncs.cls',paper/'splncs04.bst'}
    paths.update(paper.glob('sec*.tex'))
    for tex in list(paths):
        if tex.suffix!='.tex':continue
        for figure in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',tex.read_text(encoding='utf-8')):
            source=paper/figure
            options=[source] if source.suffix else [source.with_suffix(x) for x in ['.pdf','.png','.jpg','.eps']]
            paths.add(next(x for x in options if x.is_file()))
    assert all(p.is_file() and paper in p.parents for p in paths)
    manifest={str(p.relative_to(paper)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    readme='''Springer LNCS A4 manuscript source

Entry point: main.tex. All figures actually referenced by the manuscript are included.
llncs.cls and splncs04.bst are included unchanged from the original project.

Current validation engine: pdfLaTeX, MiKTeX 25.12. Run:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

Alternative engine used for the earlier version: Tectonic 0.17.0 (XeTeX):
  tectonic --keep-logs --keep-intermediates main.tex

The source contains no engine-specific custom layout settings. Publisher installations
may use a different engine; pagination must be checked after any rebuild.
SOURCE_MANIFEST.json gives SHA256 for each manuscript dependency.
Author signature, registration/payment, and final OpenReview submission are separate.
'''
    with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted(paths):z.write(path,str(path.relative_to(paper)))
        z.writestr('BUILD_README.txt',readme)
        z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2))
    a.out.with_suffix('.manifest.json').write_text(json.dumps({'source_files':manifest,
                         'zip_sha256':hashlib.sha256(a.out.read_bytes()).hexdigest()},indent=2))
    print(a.out, len(paths),'source dependencies')

if __name__=='__main__':main()
