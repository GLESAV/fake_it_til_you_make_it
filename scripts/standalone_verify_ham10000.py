# Standalone verification. No repo, no model, no images downloaded.
# Only needs: pandas, and a public URL.
import io, urllib.request, pandas as pd, numpy as np

URL = "https://dataverse.harvard.edu/api/access/datafile/3172582"   # HAM10000_metadata
req = urllib.request.Request(URL, headers={"User-Agent": "curl/8"})  # Dataverse blocks the default UA
df = pd.read_csv(io.BytesIO(urllib.request.urlopen(req).read()), sep="\t")

print(f"{len(df)} images, {df.lesion_id.nunique()} lesions, "
      f"{len(df)/df.lesion_id.nunique():.2f} images per lesion")

rng, rates = np.random.default_rng(0), {}
for _ in range(100):
    perm = rng.permutation(len(df)); cut = int(0.8 * len(df))
    train = set(df.lesion_id.values[perm[:cut]]); test = df.iloc[perm[cut:]]
    rates.setdefault("ALL", []).append(test.lesion_id.isin(train).mean())
    for dx, g in test.groupby("dx"):
        rates.setdefault(dx, []).append(g.lesion_id.isin(train).mean())

for k in sorted(rates, key=lambda k: -np.mean(rates[k])):
    print(f"  {k:>6}: {100*np.mean(rates[k]):5.1f}% of test images share a lesion with train")
