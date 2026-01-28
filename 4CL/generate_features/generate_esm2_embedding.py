import os
import torch
from Bio import SeqIO
from tqdm import tqdm

saved_folder = "../features"

with torch.no_grad():
    esm2_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    esm2_model = esm2_model.eval().cuda()

    for mut_info in tqdm(os.listdir(saved_folder)):
        seq = str(list(SeqIO.parse(f"{saved_folder}/{mut_info}/result.fasta", "fasta"))[0].seq)
        _, _, batch_tokens = batch_converter([("", seq)])
        result = esm2_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)
        torch.save(result["representations"][33][0, 1:-1, :].detach().cpu().clone(), f"{saved_folder}/{mut_info}/esm2_650m_embedding.pt")

        torch.save(batch_tokens[0, 1:-1].detach().cpu().clone(), f"{saved_folder}/{mut_info}/tokens.pt")
