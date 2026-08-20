"""Is the compression severity failure, or framing?

ACNE04 is a tight half-face crop: the cheek fills the frame. Gemini returns a studio
portrait where the face occupies perhaps a third of it, so far fewer lesion pixels reach a
224px classifier input. That would compress predicted severity without the generator having
got severity wrong -- and the fix would be a prompt change costing nothing.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, 'scripts')


def main():
    from fitymi.data.records import Corpus, Record, Source
    from fitymi.train.loop import evaluate_corpus
    from grade_fidelity import load_scorer, spearman

    model, cfg = load_scorer('models/grade_scorer.pt')
    cfg.num_workers = 0
    src = sorted(Path('data/synthetic/gemini_calib').glob('*.png'))
    labels = [int(p.name[1]) for p in src]
    print(f'{len(src)} generated images, rescored at several crop fractions:')

    for frac, tag in ((1.0, 'full'), (0.7, '70%'), (0.5, '50%'), (0.35, '35%')):
        out = Path(f'/tmp/crop_{tag}'); out.mkdir(exist_ok=True)
        paths = []
        for p in src:
            q = out / p.name
            if not q.exists():
                im = Image.open(p).convert('RGB'); w, h = im.size
                cw, ch = int(w * frac), int(h * frac)
                left, top = (w - cw) // 2, int((h - ch) * 0.35)
                im.crop((left, top, left + cw, top + ch)).save(q)
            paths.append(str(q))
        corpus = Corpus(Record(path=p, label=l, source=Source.SYNTH_OPEN)
                        for p, l in zip(paths, labels))
        res, pr = evaluate_corpus(model, corpus, cfg, return_predictions=True)
        yt, yp = np.array(pr['y_true']), np.array(pr['y_pred'])
        means = {g: round(float(yp[yt == g].mean()), 2) for g in sorted(set(yt.tolist()))}
        print(f'  crop {tag:>4}: spearman {spearman(yt.astype(float), yp.astype(float)):+.3f}  '
              f'exact {float((yt == yp).mean()):.3f}  means {means}  '
              f'hist {dict(sorted(Counter(yp.tolist()).items()))}')


if __name__ == "__main__":
    main()
