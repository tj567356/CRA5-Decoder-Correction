"""Larger labels at intended LNCS text width, using original measured 2018 results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/home/tong/fourcastnet_us_mpca')
out=root/'results/revision_2026/figures';out.mkdir(parents=True,exist_ok=True)
rm=json.loads(Path('/home/tong/new_20/results/rate_matched_val.json').read_text())
one,two=rm['rows']
rows=[('CRA5',.08752,.06321),('MPCA-only',one['pre_cnn'],one['post_cnn']),
      ('MPCA + global PCA',two['pre_cnn'],two['post_cnn'])]
plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(5.2,2.65),layout='constrained')
for i,(name,b,a) in enumerate(rows):
    j=2-i
    ax.plot([a,b],[j,j],color='#777777',lw=1.3)
    ax.scatter(b,j,s=60,facecolor='white',edgecolor='#2166ac',lw=1.5,zorder=3)
    ax.scatter(a,j,s=60,color='#d6604d',zorder=3)
    ax.text(b,j+.17,f'{b:.5f}',ha='center',fontsize=9)
    ax.text(a,j-.28,f'{a:.5f}',ha='center',fontsize=9)
ax.set_yticks([2,1,0],[x[0] for x in rows],fontsize=10)
ax.set(xlim=(.049,.144),ylim=(-.7,2.55),xlabel='2018 mean nRMSE (20 channels)')
ax.grid(axis='x',alpha=.25);ax.tick_params(axis='y',length=0)
for s in ['top','right','left']:ax.spines[s].set_visible(False)
handles=[plt.Line2D([],[],marker='o',ls='',mfc='white',mec='#2166ac',label='Before'),
         plt.Line2D([],[],marker='o',ls='',color='#d6604d',label='After U-Net')]
ax.legend(handles=handles,ncol=2,loc='lower center',fontsize=9,frameon=False,bbox_to_anchor=(.5,-.02))
fig.savefig(out/'main_result.pdf');fig.savefig(out/'main_result.png',dpi=250)
