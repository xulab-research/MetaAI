import os
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModelForMaskedLM

saved_folder = "../features"

with torch.no_grad():
    esmc_model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESMplusplus_small", torch_dtype=torch.float32, trust_remote_code=True).eval().cuda()
    tokenizer = esmc_model.tokenizer

    for mut_info in tqdm(os.listdir(saved_folder)):
        seq = str(list(SeqIO.parse(f"{saved_folder}/{mut_info}/result.fasta", "fasta"))[0].seq)
        tokenized = tokenizer([seq], padding=True, return_tensors="pt")
        tokenized = {key: value.cuda() for key, value in tokenized.items()}
        result = esmc_model(**tokenized).last_hidden_state
        torch.save(result[0, 1:-1, :].detach().cpu().clone(), f"{saved_folder}/{mut_info}/esmc_300m_embedding.pt")

    esmc_model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESMplusplus_large", torch_dtype=torch.float32, trust_remote_code=True).eval().cuda()
    tokenizer = esmc_model.tokenizer

    for mut_info in tqdm(os.listdir(saved_folder)):
        seq = str(list(SeqIO.parse(f"{saved_folder}/{mut_info}/result.fasta", "fasta"))[0].seq)
        tokenized = tokenizer([seq], padding=True, return_tensors="pt")
        tokenized = {key: value.cuda() for key, value in tokenized.items()}
        result = esmc_model(**tokenized).last_hidden_state
        torch.save(result[0, 1:-1, :].detach().cpu().clone(), f"{saved_folder}/{mut_info}/esmc_600m_embedding.pt")
