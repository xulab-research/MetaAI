import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

df = pd.read_csv("20260119.csv")
muts = df["mut_name"].astype(str).tolist()

pos = sorted({int(s.strip()[1:-1]) for m in muts for s in str(m).split(",")})
idx = {p: i for i, p in enumerate(pos)}
y = np.zeros((len(muts), len(pos)), int)
for i, m in enumerate(muts):
    for s in str(m).split(","):
        y[i, idx[int(s.strip()[1:-1])]] = 1

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=0)
folds = np.empty(len(df), int)
for f, (_, v) in enumerate(mskf.split(np.zeros((len(df), 1)), y)):
    folds[v] = f

df["fold"] = folds
df.to_csv("train.csv", index=False)
