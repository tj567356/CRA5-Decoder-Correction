"""PDF structural QA and a content digest for independent source-ZIP rebuild checks."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from pypdf import PdfReader

def main():
    p=argparse.ArgumentParser();p.add_argument('pdf',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=PdfReader(a.pdf)
    fonts={};page_sizes=[];text=[]
    seen=set()
    def check_resources(resources):
        resources=resources.get_object()
        for name,ref in resources.get('/Font',{}).items():
            f=ref.get_object();descendants=f.get('/DescendantFonts')
            actual=descendants[0].get_object() if descendants else f
            desc=actual.get('/FontDescriptor');desc=desc.get_object() if desc else {}
            fonts[str(f.get('/BaseFont',name))]=any(k in desc for k in ['/FontFile','/FontFile2','/FontFile3'])
        for ref in resources.get('/XObject',{}).values():
            obj=ref.get_object()
            if id(obj) in seen:continue
            seen.add(id(obj))
            if '/Resources' in obj:check_resources(obj['/Resources'])
    for page in r.pages:
        page_sizes.append([float(page.mediabox.width),float(page.mediabox.height)])
        text.append(page.extract_text())
        check_resources(page['/Resources'])
    normalized='\n'.join(re.sub(r'\s+',' ',t).strip() for t in text)
    result={'pdf':str(a.pdf.resolve()),'pages':len(r.pages),'within_6_11':6<=len(r.pages)<=11,
            'page_sizes_pt':page_sizes,'a4_all':all(abs(w-595.28)<1 and abs(h-841.89)<1 for w,h in page_sizes),
            'fonts':fonts,'all_fonts_embedded':all(fonts.values()),
            'text_sha256':hashlib.sha256(normalized.encode()).hexdigest(),
            'pdf_sha256':hashlib.sha256(a.pdf.read_bytes()).hexdigest(),
            'visual_inspection':'Separate rendered-page inspection required; this script does not certify layout.'}
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
