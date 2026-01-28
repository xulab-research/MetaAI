import os
import torch
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from model import AA_TO_IDX, FusionModel, load_tensor_maybe_dict, build_mut_tensors, spearman_loss, spearman_corr, predict_fitness

DEVICE = 0
FEATURE_NAME = "esmc_300m"
R = 16
BATCH_SIZE = 32
TOTAL_EPOCHS = 240
INITIAL_LR = 3e-7
MAX_LR = 3e-4
MIN_LR = 3e-8
WARMUP_FRAC = 0.1
GRAD_CLIP_MAX_NORM = 10.0
PLATEAU_FACTOR = 0.1
PLATEAU_PATIENCE = 10


def to_gpu(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    if isinstance(obj, list):
        return [to_gpu(i, device=device) for i in obj]
    if isinstance(obj, tuple):
        return tuple(to_gpu(i, device=device) for i in obj)
    if isinstance(obj, dict):
        return {k: to_gpu(v, device=device) for k, v in obj.items()}
    return obj


def stratified_sampling_for_mutation_data(mut_info_list):
    positions = set()
    for multiple_mut_info in mut_info_list:
        for single_mut_info in str(multiple_mut_info).split(","):
            s = single_mut_info.strip()
            if s:
                positions.add(int(s[1:-1]))
    sorted_pos = sorted(positions)
    index_map = {p: i for i, p in enumerate(sorted_pos)}

    vectors = {}
    for multiple_mut_info in mut_info_list:
        vec = [0] * len(index_map)
        for single_mut_info in str(multiple_mut_info).split(","):
            s = single_mut_info.strip()
            if s:
                vec[index_map[int(s[1:-1])]] = 1
        vectors[str(multiple_mut_info)] = vec

    return sorted_pos, index_map, vectors


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, indices, y_all):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y_all = np.asarray(y_all, dtype=np.float32)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        return torch.tensor(idx, dtype=torch.long), torch.tensor(self.y_all[idx], dtype=torch.float32)


device = torch.device(f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu")

all_csv = pd.read_csv("../data/20260119.csv")
mut_info_list = all_csv["mut_name"].astype(str).tolist()
_, _, vectors = stratified_sampling_for_mutation_data(mut_info_list)
y_mut_pos = np.array([vectors[str(m)] for m in mut_info_list], dtype=np.int64)

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2)
train_index, test_index = next(msss.split(y_mut_pos, y_mut_pos))

file_dir = "results"
os.makedirs(file_dir, exist_ok=True)

pos_all, aa_all, mask_all = build_mut_tensors(mut_info_list, device=device)

esm_path = "../features/wt/esm2_650m_embedding.pt" if FEATURE_NAME == "esm2_650m" else "../features/wt/esmc_600m_embedding.pt" if FEATURE_NAME == "esmc_600m" else "../features/wt/esmc_300m_embedding.pt"

esm_raw = load_tensor_maybe_dict(esm_path)
spired_raw = load_tensor_maybe_dict("../features/wt/spired_fitness_embedding.pt")

wt_seq = "".join(l.strip() for l in open("../features/wt/result.fasta", "r", encoding="utf-8") if l.strip() and not l.startswith(">")).upper()

wt_idx = to_gpu(torch.tensor([AA_TO_IDX[c] for c in wt_seq], dtype=torch.long), device)

esm_emb = to_gpu(esm_raw.to(torch.float32).unsqueeze(0), device)
spired_emb = to_gpu(spired_raw.to(torch.float32).unsqueeze(0), device)

y_np = all_csv["label"].values
train_loader = torch.utils.data.DataLoader(IndexDataset(train_index, y_np), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(IndexDataset(test_index, y_np), batch_size=1, shuffle=False)

model = FusionModel(int(esm_raw.shape[1]), int(R), spired_dim=32, esm_out=96).to(device)
for name, param in model.named_parameters():
    if "SPIRED" in name or "Fitness" in name or "finetune_coef" in name:
        param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=float(INITIAL_LR), weight_decay=1e-3)
warmup_epochs = int(WARMUP_FRAC * TOTAL_EPOCHS)
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: min(1.0, e / max(1, warmup_epochs)) * (float(MAX_LR) / max(1e-12, float(INITIAL_LR))))
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(PLATEAU_FACTOR), patience=int(PLATEAU_PATIENCE), min_lr=float(MIN_LR))

best_corr, loss_df = float("-inf"), pd.DataFrame()
for epoch in range(int(TOTAL_EPOCHS)):
    model.train()
    epoch_loss = 0.0
    for batch_idx, label in train_loader:
        batch_idx, label = to_gpu(batch_idx, device), to_gpu(label, device)
        optimizer.zero_grad()

        single_mut, high_delta = model(esm_emb, spired_emb, wt_idx)
        y_hat = predict_fitness(single_mut, high_delta, pos_all[batch_idx], aa_all[batch_idx], mask_all[batch_idx])

        loss = spearman_loss(y_hat.unsqueeze(0), label.unsqueeze(0), 1e-2, "kl")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=float(GRAD_CLIP_MAX_NORM), error_if_nonfinite=True)
        optimizer.step()
        epoch_loss += float(loss.item())

    train_loss_value = epoch_loss / max(1, len(train_loader))

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for one_idx, one_label in test_loader:
            one_idx, one_label = to_gpu(one_idx, device), to_gpu(one_label, device)
            single_mut, high_delta = model(esm_emb, spired_emb, wt_idx)
            y_hat = predict_fitness(single_mut, high_delta, pos_all[one_idx], aa_all[one_idx], mask_all[one_idx])
            preds += y_hat.detach().cpu().tolist()
            trues += one_label.detach().cpu().tolist()

    test_corr = spearman_corr(torch.tensor(preds), torch.tensor(trues))
    loss_df.loc[f"{epoch}", "train_loss"] = train_loss_value
    loss_df.loc[f"{epoch}", "test_corr"] = test_corr
    loss_df.loc[f"{epoch}", "learning_rate"] = optimizer.param_groups[0]["lr"]
    loss_df.to_csv(f"{file_dir}/loss.csv")

    (warmup_scheduler.step() if epoch < warmup_epochs else plateau_scheduler.step(-test_corr))

    if test_corr > best_corr:
        best_corr = test_corr
        torch.save(model, f"{file_dir}/train_best.pt")
    elif optimizer.param_groups[0]["lr"] <= float(MIN_LR) * (1 + 1e-6) and epoch > warmup_epochs:
        print(f"Stopping at epoch {epoch} due to no improvement in test loss.")
        break
