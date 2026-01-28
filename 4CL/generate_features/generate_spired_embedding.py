import os
import sys
import torch
from Bio import SeqIO
from tqdm import tqdm

sys.path.append("..")
from scripts.model import SPIRED_Fitness_Union

saved_folder = "../features"

device = 0

model = SPIRED_Fitness_Union(device_list=[f"cuda:{device}"])
model.load_state_dict(torch.load("../model_pth/SPIRED-Fitness.pth", weights_only=True))
model.eval().to(device)

esm2_650m_model, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
esm2_650m_model.eval().to(device)

esm2_3b_model, esm2_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
esm2_3b_model.eval().to(device)
esm2_batch_converter = esm2_alphabet.get_batch_converter()

with torch.no_grad():
    for mut_info in tqdm(os.listdir(saved_folder)):
        seq = str(list(SeqIO.parse(f"{saved_folder}/{mut_info}/result.fasta", "fasta"))[0].seq)
        _, _, target_tokens = esm2_batch_converter([("", seq)])
        target_tokens = target_tokens.cuda()

        esm2_650m_embedding = esm2_650m_model(target_tokens, repr_layers=[33], return_contacts=False)["representations"][33][:, 1:-1, :]

        tmp = esm2_3b_model(target_tokens, repr_layers=range(37), need_head_weights=False, return_contacts=False)
        esm2_3b_embedding = torch.stack([v[:, 1:-1, :] for _, v in sorted(tmp["representations"].items())], dim=2)

        pretrained_embedding = model(target_tokens[:, 1:-1], esm2_650m_embedding, esm2_3b_embedding)
        torch.save(pretrained_embedding[0].detach().cpu().clone(), f"{saved_folder}/{mut_info}/spired_fitness_embedding.pt")
