"""Does a plausible camera difference move ITA as far as a real skin-tone difference does?

The stratified analysis shows ITA carries a device signal. This asks the mechanistic
question directly: apply exposure and white-balance shifts of the size that ordinary
cameras differ by, and measure how many ITA units -- and how many tone bins -- an
otherwise identical image moves.
"""
import json, numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from fitymi.controls.skintone import ITA_BINS, estimate_ita
import tempfile, os

names=[b[0] for b in ITA_BINS]
def binof(v):
    for n,lo,hi in ITA_BINS:
        if lo<=v<hi: return n
    return names[0] if v>=55 else names[-1]

real=[]
for s in ('train',):
    for l in (Path('data/splits_subject')/f'{s}.jsonl').read_text().splitlines():
        if l.strip(): real.append(json.loads(l)['path'])
rng=np.random.default_rng(0)
sample=[real[i] for i in rng.permutation(len(real))[:60]]

# Exposure +/-0.5 EV and white balance +/-500K are well inside the spread between two
# consumer cameras photographing the same scene on auto settings.
TRANSFORMS = {
    "baseline":            lambda a: a,
    "exposure -0.5 EV":    lambda a: a * (2 ** -0.5),
    "exposure +0.5 EV":    lambda a: a * (2 ** 0.5),
    "warm WB (+500K-ish)": lambda a: a * np.array([1.06, 1.00, 0.94]),
    "cool WB (-500K-ish)": lambda a: a * np.array([0.94, 1.00, 1.06]),
}
rows={k:[] for k in TRANSFORMS}
tmp=Path(tempfile.mkdtemp())
for path in sample:
    with Image.open(path) as im:
        arr=np.asarray(im.convert('RGB').resize((256,256), Image.LANCZOS), dtype=np.float32)
    for name,fn in TRANSFORMS.items():
        out=np.clip(fn(arr),0,255).astype(np.uint8)
        f=tmp/'x.png'; Image.fromarray(out).save(f)
        rows[name].append(estimate_ita(str(f)).ita)

base=np.array(rows["baseline"])
print(f"{len(sample)} ACNE04 images, each re-measured under camera-scale transforms\n")
print(f"{'transform':>22} {'mean dITA':>10} {'max |dITA|':>11} {'% changing bin':>15}")
for name in TRANSFORMS:
    if name=="baseline": continue
    v=np.array(rows[name]); d=v-base
    moved=np.mean([binof(a)!=binof(b) for a,b in zip(v,base)])
    print(f"{name:>22} {d.mean():>+10.1f} {np.abs(d).max():>11.1f} {100*moved:>14.0f}%")
widths=[hi-lo for _,lo,hi in ITA_BINS]
print(f"\nITA bin widths: {widths}  (narrowest {min(widths):.0f} units)")
print(f"Adjacent Fitzpatrick types in the generated pool differ by a median of "
      f"~7 ITA units (46.8 -> 40.0 -> 35.2 -> 30.9).")
