import numpy as np, hashlib, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, Counter

D=Path('data/acne04/Classification'); imgs=D/'JPEGImages'
ham=np.load('/tmp/ham.npy'); cos=np.load('/tmp/clip_emb.npy'); cos=cos@cos.T
n=ham.shape[0]

class UF:
    def __init__(s,m): s.p=list(range(m))
    def find(s,x):
        while s.p[x]!=x: s.p[x]=s.p[s.p[x]]; x=s.p[x]
        return x
    def union(s,a,b):
        ra,rb=s.find(a),s.find(b)
        if ra!=rb: s.p[ra]=rb
def largest(mask):
    uf=UF(n); ii,jj=np.where(np.triu(mask,k=1))
    for a,b in zip(ii.tolist(),jj.tolist()): uf.union(a,b)
    return max(Counter(uf.find(i) for i in range(n)).values())/n

ph_t=list(range(0,13,2)); ph_y=[largest(ham<=t) for t in ph_t]
cl_t=[0.99,0.98,0.97,0.96,0.95,0.93,0.90]; cl_y=[largest(cos>=t) for t in cl_t]

fig,ax=plt.subplots(1,3,figsize=(15,4.4))
a=ax[0]
a.plot(ph_t,[100*v for v in ph_y],'o-',color='#c0392b',label='pHash (Hamming ≤ t)')
a2=a.twiny(); a2.plot(cl_t,[100*v for v in cl_y],'s-',color='#2471a3',label='CLIP (cosine ≥ t)')
a2.invert_xaxis()
a.axhline(5,ls='--',c='gray',lw=1); a.text(0.3,6,'5% guard',fontsize=8,color='gray')
a.axvline(8,ls=':',c='#c0392b',lw=1); a.text(8.2,55,'shipped\ndefault',fontsize=8,color='#c0392b')
a.set_xlabel('pHash Hamming threshold',color='#c0392b'); a2.set_xlabel('CLIP cosine threshold',color='#2471a3')
a.set_ylabel('largest cluster (% of corpus)')
a.set_title('A. Generic embedders percolate on ACNE04',fontsize=10,loc='left')
a.set_ylim(-2,100)
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
a.legend(h1+h2,l1+l2,fontsize=8,loc='upper left')

b=ax[1]
folds=[0,1,2,3,4]; free=[7,15,10,11,9]; wrong=[5,0,2,0,3]
b.bar(folds,[100*f/292 for f in free],color='#27ae60',label='memorisable → free correct')
b.bar(folds,[100*w/292 for w in wrong],bottom=[100*f/292 for f in free],color='#c0392b',label='conflicting label → forced error')
b.axhline(4.25,ls='--',c='k',lw=1); b.text(-0.45,4.35,'mean 4.25%',fontsize=8)
b.set_ylim(0,6.4)
b.set_xlabel("ACNE04 published fold"); b.set_ylabel('% of the 292-image test set')
b.set_title('B. Every published fold leaks byte-identical images',fontsize=10,loc='left')
b.set_xticks(folds); b.legend(fontsize=8,loc='upper center',ncol=1,framealpha=0.95)

c=ax[2]
diffs=[0]*12+[1]*8+[2]*5+[3]*4+[4]*3+[7,8,12,15,15,16]
c.hist(diffs,bins=np.arange(-0.5,17.5,1),color='#7f8c8d',edgecolor='white')
c.set_xlabel('|Δ lesion count| between two annotations of the SAME image')
c.set_ylabel('duplicate pairs (n=38)')
c.set_title('C. The same image, counted twice',fontsize=10,loc='left')
c.text(6.0,9.0,'grade agreement 78.9%\n(95% CI 63.7–88.9%)\n\npublished ACNE04\naccuracy: 83.7–87.3%',
       fontsize=9,va='top',bbox=dict(fc='#fdf2e9',ec='#e67e22'))
plt.tight_layout()
out='docs/examples/acne04_audit.png'
Path('docs/examples').mkdir(parents=True,exist_ok=True)
plt.savefig(out,dpi=150)
print('wrote',out)
