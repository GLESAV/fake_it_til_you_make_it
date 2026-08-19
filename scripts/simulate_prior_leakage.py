#!/usr/bin/env python
"""How much subject contamination does ACNE04's identity structure force on any split?

Motivated by the one existing fully-synthetic acne result (Zein et al., PLOS ONE 2024),
which trained StyleGAN2 per severity level on 1,073 ACNE04 images and evaluated a
synthetic-trained classifier on "unseen real acneic face images" without stating where
those came from. If they are the ACNE04 remainder -- the most economical reading -- the
identity structure alone predicts the contamination this script measures.

Requires the cached face embeddings; run scripts/audit_acne04_subjects.py first.
See docs/01_related_work.md section 4.1.1.
"""
import numpy as np
from pathlib import Path
import argparse
from fitymi.controls.embedders import cached_embedder, face_embedder
from fitymi.data.acne04 import load_acne04

ap = argparse.ArgumentParser()
ap.add_argument('--root', default='data/acne04')
ap.add_argument('--cache', default='data/splits/face_embeddings.npz')
ap.add_argument('--trials', type=int, default=200)
args = ap.parse_args()

paths = np.array([r.path for r in load_acne04(args.root)])
E = np.asarray(cached_embedder(face_embedder(), args.cache)(list(paths)), np.float32)
ok = np.abs(E).sum(1) > 0
idx=np.where(ok)[0]; cos=(E[idx]@E[idx].T); n=len(idx)
rng=np.random.default_rng(0)

print(f'{n} ACNE04 images with a face embedding\n')
print('Zein et al. trained StyleGAN2 on 1,073 ACNE04 images (+400 web images) and')
print('tested a synthetic-trained classifier on "unseen real acneic face images".')
print('Simulating that split over the identity structure:\n')
print(f"{'cosine':>7} {'chance':>8} {'held-out sharing a subject with GAN training':>46}")
for THR in (0.85,0.80,0.75,0.70,0.60):
    chance=float((cos[np.triu_indices(n,k=1)]>=THR).mean())
    rates=[]
    for _ in range(args.trials):
        perm=rng.permutation(n); tr=perm[:1073]; ho=perm[1073:]
        rates.append(float((cos[np.ix_(ho,tr)].max(1)>=THR).mean()))
    lo,hi=np.percentile(rates,[2.5,97.5])
    print(f"{THR:>7.2f} {100*chance:>7.3f}%   {100*np.mean(rates):>10.1f}%   (95% range {100*lo:.1f}-{100*hi:.1f})")

print('\nSame simulation restricted to a per-severity split, which is what they did')
print('(StyleGAN2 was trained per severity level):')
D=Path('data/acne04/Classification')
lab={}
for f in sorted(D.glob('NNEW_*.txt')):
    for line in f.read_text().splitlines():
        q=line.split()
        if len(q)>=3: lab[q[0]]=int(q[1])
names=[Path(p).name for p in paths[idx]]
y=np.array([lab.get(nm,-1) for nm in names])
for THR in (0.85,0.75):
    out=[]
    for _ in range(args.trials):
        tr_all=[];ho_all=[]
        for c in (0,1,2,3):
            m=np.where(y==c)[0]
            if len(m)<4: continue
            perm=rng.permutation(len(m)); k=int(round(0.736*len(m)))  # 1073/1457
            tr_all+=list(m[perm[:k]]); ho_all+=list(m[perm[k:]])
        out.append(float((cos[np.ix_(ho_all,tr_all)].max(1)>=THR).mean()))
    print(f'  cosine {THR}: {100*np.mean(out):.1f}% of held-out images share a subject with the generator training set')
