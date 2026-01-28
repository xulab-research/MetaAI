import torch
import numpy as np
import pandas as pd

from model import AA_TO_IDX, load_tensor_maybe_dict, build_mut_tensors, predict_fitness

FEATURE_NAME = "esmc_300m"

INFER_COLS = ["CombinedMutation", "MutationA", "MutationB"]

WT_FASTA = "../../features/wt/result.fasta"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open(WT_FASTA, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith(">")]
wt_seq = "".join(lines).upper()

wt_idx = torch.tensor([AA_TO_IDX[c] for c in wt_seq], dtype=torch.long, device=device)

esm_path = "../../features/wt/esm2_650m_embedding.pt" if FEATURE_NAME == "esm2_650m" else "../../features/wt/esmc_600m_embedding.pt" if FEATURE_NAME == "esmc_600m" else "../../features/wt/esmc_300m_embedding.pt"
esm = load_tensor_maybe_dict(esm_path).to(torch.float32).unsqueeze(0).to(device)
spired = load_tensor_maybe_dict("../../features/wt/spired_fitness_embedding.pt").to(torch.float32).unsqueeze(0).to(device)

model = torch.load("../../01_model_training/results/train_best.pt", map_location="cpu", weights_only=False).to(device).eval()

with torch.no_grad():
    single_mut, high_delta = model(esm, spired, wt_idx)

df = pd.read_csv("epistasis.csv")

pred_name = {"CombinedMutation": "CombinedMutationPred", "MutationA": "MutationAPred", "MutationB": "MutationBPred"}

for col in INFER_COLS:
    mut_list = df[col].fillna("").astype(str).tolist()

    pred = np.zeros(len(mut_list), dtype=np.float32)
    idx = [i for i, m in enumerate(mut_list) if m]

    if idx:
        mut_non = [mut_list[i] for i in idx]
        pos, aa, mask = build_mut_tensors(mut_non, device)
        with torch.no_grad():
            pred_non = predict_fitness(single_mut, high_delta, pos, aa, mask).detach().float().cpu().numpy()
        pred[idx] = pred_non

    df[pred_name[col]] = pred

df["epistasisPred"] = df["CombinedMutationPred"] - df["MutationAPred"] - df["MutationBPred"]

df.to_csv("epistasis_pred.csv", index=False)
