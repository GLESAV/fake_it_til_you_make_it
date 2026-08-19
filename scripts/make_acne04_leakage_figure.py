#!/usr/bin/env python
"""Figure: identity structure, subject leakage, and cross-team label disagreement."""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from fitymi.controls.embedders import cached_embedder, face_embedder
from fitymi.data.acne04 import load_acne04
from fitymi.data.dedup import UnionFind

recs=list(load_acne04('data/acne04')); paths=[r.path for r in recs]
names=[Path(p).name for p in paths]; where={p:i for i,p in enumerate(paths)}
E=np.asarray(cached_embedder(face_embedder(),'data/splits/face_embeddings.npz')(paths),np.float32)
det=np.abs(E).sum(1)>0; keep=np.where(det)[0]; cos=E[keep]@E[keep].T
kp={int(k):i for i,k in enumerate(keep)}
iu=np.triu_indices(len(keep),k=1)
THR=[0.85,0.80,0.75,0.70,0.65,0.60,0.55,0.50,0.45]

D=next(p.parent for p in Path('data/acne04').rglob('NNEW_trainval_0.txt'))
def fold(n): return {l.split()[0] for l in (D/n).read_text().splitlines() if len(l.split())>=2}

fig,ax=plt.subplots(1,3,figsize=(15.5,4.6))

a=ax[0]
subs=[];larg=[]
for t in THR:
    uf=UnionFind(len(keep))
    ii,jj=np.where(np.triu(cos>=t,k=1))
    for x,y in zip(ii.tolist(),jj.tolist()): uf.union(x,y)
    s=Counter(uf.find(i) for i in range(len(keep))); subs.append(len(s)); larg.append(max(s.values()))
a.plot(THR,subs,'o-',color='#2471a3',label='distinct subjects')
a.axvspan(0.50,0.75,color='#2471a3',alpha=0.08)
a.text(0.625,1150,'stable band',ha='center',fontsize=8,color='#2471a3')
a.set_xlabel('ArcFace cosine threshold'); a.set_ylabel('distinct subjects'); a.invert_xaxis()
a2=a.twinx(); a2.plot(THR,larg,'s--',color='#c0392b',label='largest cluster'); a2.set_ylabel('largest cluster (images)',color='#c0392b')
a.axhline(1457,ls=':',c='gray',lw=1); a.text(0.86,1480,'1,457 images',fontsize=8,color='gray')
a.set_title('A. 1,457 photographs are ~600 people',fontsize=10,loc='left')
h1,l1=a.get_legend_handles_labels(); h2,l2=a2.get_legend_handles_labels()
a.legend(h1+h2,l1+l2,fontsize=8,loc='lower left')

b=ax[1]
chance=[float((cos[iu]>=t).mean())*100 for t in THR]
pub=[]
for t in THR:
    r=[]
    for k in range(5):
        tr,te=fold(f'NNEW_trainval_{k}.txt'),fold(f'NNEW_test_{k}.txt')
        ti=[kp[where[p]] for p,n in zip(paths,names) if n in tr and det[where[p]]]
        ei=[kp[where[p]] for p,n in zip(paths,names) if n in te and det[where[p]]]
        r.append(float((cos[np.ix_(ei,ti)].max(1)>=t).mean())*100)
    pub.append(np.mean(r))
def ours(sd):
    def ld(n):
        return [kp[where[json.loads(l)['path']]] for l in (Path(sd)/f'{n}.jsonl').read_text().splitlines()
                if l.strip() and json.loads(l)['path'] in where and det[where[json.loads(l)['path']]]]
    m=cos[np.ix_(ld('test'),ld('train'))].max(1)
    return [float((m>=t).mean())*100 for t in THR]
b.plot(THR,pub,'o-',color='#c0392b',label="ACNE04 published folds")
b.plot(THR,ours('data/splits'),'s-',color='#e67e22',label='ours, image-level dedup')
b.plot(THR,ours('data/splits_subject'),'^-',color='#27ae60',label='ours, subject-disjoint')
b.plot(THR,chance,'--',color='gray',lw=1,label='chance (all pairs)')
b.set_yscale('symlog',linthresh=0.05); b.invert_xaxis()
b.set_xlabel('ArcFace cosine threshold'); b.set_ylabel('% of test images sharing a face with train')
b.set_title('B. Subject leakage, direct links only',fontsize=10,loc='left')
b.legend(fontsize=8,loc='lower right'); b.grid(alpha=0.25)

c=ax[2]
v1={}
for f in sorted(D.glob('NNEW_*.txt')):
    for l in f.read_text().splitlines():
        q=l.split()
        if len(q)>=3: v1[q[0]]=int(q[2])
S='/private/tmp/claude-502/-Users-gs/7c55be80-87ce-4679-96d2-48e13cd6a2e9/scratchpad'
j=json.load(open(f'{S}/acne04v2/Acne04-v2_annotations.json'))
id2={im['id']:im['file_name'] for im in j['images']}
c2=Counter(id2[a['image_id']] for a in j['annotations'])
sh=sorted(set(v1)&set(c2)); x=np.array([v1[n] for n in sh]); y=np.array([c2[n] for n in sh])
c.scatter(x,y,s=6,alpha=0.28,color='#34495e',edgecolors='none')
lim=[0.8,500]; c.plot(lim,lim,'k--',lw=1,label='equal counts')
c.plot(lim,[3.2*v for v in lim],'-',color='#c0392b',lw=1,label='3.2x (mean ratio)')
for v in (5.5,20.5,50.5):
    c.axvline(v,color='#e67e22',lw=0.8,alpha=0.7); c.axhline(v,color='#e67e22',lw=0.8,alpha=0.7)
c.set_xscale('log'); c.set_yscale('log'); c.set_xlim(0.8,120); c.set_ylim(0.8,500)
c.set_xlabel('v1 lesion count (Wu et al. 2019)'); c.set_ylabel('v2 lesion count (AcneAI 2024)')
c.set_title('C. Same 1,204 images, two expert teams',fontsize=10,loc='left')
c.legend(fontsize=8,loc='upper left')
c.text(1.05,1.6,'orange lines = Hayashi grade boundaries\nSpearman rho 0.471   grade agreement 30.1%\n90.1% of images above the dashed line',
       fontsize=7.5,va='bottom',bbox=dict(fc='#fdf2e9',ec='#e67e22'))
plt.tight_layout(); out='docs/examples/acne04_leakage.png'
plt.savefig(out,dpi=150); print('wrote',out)
