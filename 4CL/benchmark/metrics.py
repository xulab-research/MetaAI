import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau

df = pd.read_csv("summary.csv")
y = df.iloc[:, 3]

esm_replicate_cols = list(df.columns[4:9])

models = {}
models["esm1v_t33_650M_UR90S"] = df[esm_replicate_cols].mean(axis=1, skipna=True)

for c in df.columns[9:]:
    models[str(c)] = df[c]


def corr_metrics(y_true, y_pred):
    m = pd.concat([y_true, y_pred], axis=1).dropna()
    if len(m) < 2:
        return {"Spearman": np.nan, "Pearson": np.nan, "Kendall": np.nan, "N": len(m)}

    yt = m.iloc[:, 0].to_numpy()
    yp = m.iloc[:, 1].to_numpy()

    sp = spearmanr(yt, yp).correlation
    pe = pearsonr(yt, yp)[0]
    ke = kendalltau(yt, yp).correlation
    return {"Spearman": sp, "Pearson": pe, "Kendall": ke, "N": len(m)}


rows = []
for name, pred in models.items():
    rows.append({"Model": name, **corr_metrics(y, pred)})

res = pd.DataFrame(rows).sort_values(by="Spearman", ascending=False).reset_index(drop=True)
res.to_csv("correlation_metrics.csv", index=False)
