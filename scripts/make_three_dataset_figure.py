#!/usr/bin/env python
"""Figure: the same leak in three datasets, and where it concentrates."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = Path("/private/tmp/claude-502/-Users-gs/7c55be80-87ce-4679-96d2-48e13cd6a2e9/scratchpad")
DEFECT, CLEAN, INK, MUTED = "#B0433B", "#2E6E5B", "#141C1A", "#5B6B66"

fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))

# ---- A. leakage tracks repeats per subject -------------------------------------------
a = ax[0]
pts = [("ACNE04", 2.40, 77.5, "face embedding"),
       ("SCIN", 2.07, 75.6, "published schema"),
       ("HAM10000", 1.34, 38.5, "published schema")]
for name, ipl, leak, how in pts:
    colour = DEFECT if how == "face embedding" else CLEAN
    a.scatter([ipl], [leak], s=140, color=colour, zorder=3)
    a.annotate(f"{name}\n{leak:.1f}%", (ipl, leak), textcoords="offset points",
               xytext=(0, -34), ha="center", fontsize=9, color=INK)
x = np.linspace(1.0, 2.6, 100)
# A random image-level split leaks a test image whenever its subject also lands in train.
# For a subject with k images and a train fraction f, that is 1 - (1-f)^(k-1) in expectation.
a.plot(x, 100 * (1 - 0.2 ** (x - 1)), "--", color=MUTED, lw=1.2,
       label="expected, if subjects were split at random")
a.set_xlabel("images per subject / lesion"); a.set_ylabel("% of test images leaked")
a.set_title("A. The leak is a function of redundancy", fontsize=10, loc="left")
a.set_xlim(1.0, 2.7); a.set_ylim(0, 100); a.grid(alpha=0.25)
a.legend(fontsize=8, loc="lower right")
a.scatter([], [], s=90, color=DEFECT, label="needed a model")
a.scatter([], [], s=90, color=CLEAN, label="metadata only")
handles, labels = a.get_legend_handles_labels()
a.legend(handles, labels, fontsize=8, loc="lower right")

# ---- B. HAM10000 by diagnosis ---------------------------------------------------------
b = ax[1]
rows = [{k: v.strip('"') for k, v in r.items()}
        for r in csv.DictReader(open(S / "ham10000_metadata.tsv"), delimiter="\t")]
lesions = [r["lesion_id"] for r in rows]; dxs = [r["dx"] for r in rows]
NAMES = {"mel": "melanoma", "nv": "nevus", "bcc": "basal cell", "akiec": "actinic ker.",
         "bkl": "benign ker.", "df": "dermatofibroma", "vasc": "vascular"}
rng = np.random.default_rng(0)
acc = {d: [] for d in set(dxs)}
for _ in range(200):
    o = rng.permutation(len(lesions)); cut = int(0.8 * len(lesions))
    train = {lesions[i] for i in o[:cut]}; test = o[cut:]
    leaked = np.array([lesions[i] in train for i in test])
    td = np.array([dxs[i] for i in test])
    for d in acc:
        m = td == d
        if m.sum(): acc[d].append(leaked[m].mean())
order = sorted(acc, key=lambda d: np.mean(acc[d]))
vals = [100 * np.mean(acc[d]) for d in order]
cols = [DEFECT if d == "mel" else "#9AA8A3" for d in order]
b.barh([NAMES.get(d, d) for d in order], vals, color=cols)
b.axvline(38.5, ls="--", c=INK, lw=1)
b.text(39.5, -0.4, "overall 38.5%", fontsize=8, color=INK)
for i, v in enumerate(vals):
    b.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8, color=INK)
b.set_xlabel("% of test images sharing a lesion with training")
b.set_title("B. HAM10000: the leak lands on melanoma", fontsize=10, loc="left")
b.set_xlim(0, 80)

# ---- C. what it cost on ACNE04 --------------------------------------------------------
c = ax[2]
grades = ["mild", "moderate", "severe", "very severe"]
people = [116, 126, 57, 47]
delta = [-1.5, 0.8, 9.9, 9.0]
cols = [CLEAN if d < 2 else DEFECT for d in delta]
c.bar(grades, delta, color=cols)
c.axhline(0, color=INK, lw=1)
for i, (d, n) in enumerate(zip(delta, people)):
    c.text(i, d + (0.6 if d >= 0 else -1.4), f"{d:+.1f}", ha="center", fontsize=9, color=INK)
    c.text(i, -3.2, f"{n} people\nin training", ha="center", fontsize=8, color=MUTED)
c.set_ylim(-4.5, 15.5)
c.set_ylabel("recall gained from image-level splitting (points)")
c.set_title("C. ACNE04: what the leak was worth", fontsize=10, loc="left")
c.text(0.5, 0.98, "aggregate +4.51 balanced accuracy   95% CI [+2.33, +7.04]",
       transform=c.transAxes, va="top", ha="center", fontsize=8.5,
       bbox=dict(fc="#FDF2E9", ec="#E67E22"))

plt.tight_layout()
out = "docs/examples/three_datasets.png"
plt.savefig(out, dpi=150)
print("wrote", out)
