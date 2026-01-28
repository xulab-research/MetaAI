import math
import torch
import torch.nn as nn
from soft_rank_pytorch import soft_rank

AA_TO_IDX = {a: i for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}


def load_tensor_maybe_dict(path):
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in ("embedding", "emb", "repr", "representations", "x"):
            v = obj.get(k)
            if torch.is_tensor(v):
                return v
        for v in obj.values():
            if torch.is_tensor(v):
                return v
    raise ValueError(f"Unsupported tensor format in: {path}")


def build_mut_tensors(mut_names, device):
    mut_lists = []
    for mut_name in mut_names:
        muts = []
        for tok in str(mut_name).split(","):
            tok = tok.strip()
            if tok:
                muts.append((int(tok[1:-1]), AA_TO_IDX[tok[-1]]))
        mut_lists.append(muts)

    b = len(mut_lists)
    m = max((len(x) for x in mut_lists), default=0)
    pos = torch.zeros((b, m), dtype=torch.long, device=device)
    aa = torch.zeros((b, m), dtype=torch.long, device=device)
    mask = torch.zeros((b, m), dtype=torch.float32, device=device)
    for i, muts in enumerate(mut_lists):
        for j, (p, a) in enumerate(muts):
            pos[i, j], aa[i, j], mask[i, j] = p, a, 1.0
    return pos, aa, mask


def predict_fitness(single_mut, high_delta, pos, aa, mask):
    if single_mut.dim() == 3:
        single_mut = single_mut[0]
    if high_delta.dim() == 4:
        high_delta = high_delta[0]
    single_sum = (single_mut[pos, aa] * mask).sum(1)
    high_vals = high_delta[pos, aa] * mask.unsqueeze(-1)
    ge2 = (1.0 + high_vals).prod(1) - 1.0 - high_vals.sum(1)
    return single_sum + ge2.sum(-1) / math.sqrt(high_delta.shape[-1])


def spearman_corr(pred_1d, true_1d):
    def _rank_avg_ties_1d(x):
        v, idx = torch.sort(x, descending=True)
        n = x.numel()
        r = torch.arange(1, n + 1, dtype=torch.float32, device=x.device)
        out = torch.empty(n, dtype=torch.float32, device=x.device)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[j + 1] == v[i]:
                j += 1
            out[idx[i : j + 1]] = r[i : j + 1].mean()
            i = j + 1
        return out

    p = _rank_avg_ties_1d(pred_1d.detach().float().cpu())
    t = _rank_avg_ties_1d(true_1d.detach().float().cpu())
    p, t = p - p.mean(), t - t.mean()
    denom = (p.norm() * t.norm()).item()
    return 0.0 if denom < 1e-12 else float((p @ t).item() / denom)


def spearman_loss(pred, true, regularization_strength=1e-2, regularization="kl"):
    def _rank_avg_ties_1d(x):
        v, idx = torch.sort(x, descending=True)
        n = x.numel()
        r = torch.arange(1, n + 1, dtype=torch.float32, device=x.device)
        out = torch.empty(n, dtype=torch.float32, device=x.device)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[j + 1] == v[i]:
                j += 1
            out[idx[i : j + 1]] = r[i : j + 1].mean()
            i = j + 1
        return out

    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
    if true.ndim == 1:
        true = true.unsqueeze(0)
    if pred.shape != true.shape or pred.ndim != 2 or pred.shape[0] != 1:
        raise ValueError(f"pred/true must be shape (1, N), got pred={tuple(pred.shape)}, true={tuple(true.shape)}")

    device = pred.device
    soft_pred = soft_rank(pred, direction="DESCENDING", regularization_strength=float(regularization_strength), regularization=regularization).squeeze(0)

    hard_true = _rank_avg_ties_1d(true.squeeze(0).detach().float().cpu()).to(device)
    soft_pred, hard_true = soft_pred - soft_pred.mean(), hard_true - hard_true.mean()
    return -(soft_pred @ hard_true) / (soft_pred.norm() * hard_true.norm() + 1e-12)


class FusionModel(nn.Module):
    def __init__(self, esm_dim, r, spired_dim=32, esm_out=96):
        super().__init__()
        esm_dim, esm_out, spired_dim, r = int(esm_dim), int(esm_out), int(spired_dim), int(r)
        self.esm_proj = nn.Sequential(nn.LayerNorm(esm_dim), nn.Linear(esm_dim, esm_out), nn.ReLU())
        self.base = ESMEpiModel(d_model=esm_out + spired_dim, r=r)

    def forward(self, esm_emb, spired_emb, wt_idx):
        if esm_emb.dim() == 2:
            esm_emb = esm_emb.unsqueeze(0)
        if spired_emb.dim() == 2:
            spired_emb = spired_emb.unsqueeze(0)
        return self.base(torch.cat([self.esm_proj(esm_emb), spired_emb], dim=-1), wt_idx)


class ESMEpiModel(nn.Module):
    def __init__(self, d_model=128, r=64):
        super().__init__()
        self.d_model, self.r = int(d_model), int(r)

        self.single_head = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 256), nn.GELU(), nn.Dropout(0.35), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.35), nn.Linear(128, 20))

        self.high_head = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, 256), nn.GELU(), nn.Dropout(0.35), nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.35), nn.Linear(128, self.r))
        self.aa_embed = nn.Parameter(torch.randn(20, self.r))
        self.u_mix = nn.Linear(self.r, self.r, bias=False)

    def forward(self, embedding, wt_idx):
        if embedding.dim() == 2:
            embedding = embedding.unsqueeze(0)
        if wt_idx.dim() == 2:
            wt_idx = wt_idx.squeeze(0)
        wt_idx = wt_idx.to(dtype=torch.long, device=embedding.device)

        single_mut = self.single_head(embedding).squeeze(0)
        single_mut = single_mut - single_mut.gather(1, wt_idx[:, None])

        h = self.high_head(embedding).squeeze(0)
        U = self.u_mix(h[:, None, :] * self.aa_embed[None, :, :])
        high_delta = U - U.gather(1, wt_idx[:, None, None].expand(-1, 1, self.r))
        return single_mut, high_delta
