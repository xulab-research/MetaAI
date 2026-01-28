import os
import pandas as pd
from Bio import SeqIO
from scipy.stats import zscore

wt_seq = str(list(SeqIO.parse("../features/wt/result.fasta", "fasta"))[0].seq)

file_name = "MutationB"
data = pd.read_excel(f"{file_name}.xlsx", index_col=0)

dup_list = []
result = pd.DataFrame()
for multi_mut_info in data.index:
    tmp_str = ""
    for single_mut_info in multi_mut_info.split(","):
        mut_pos = int(single_mut_info[1:-1]) - 1
        assert wt_seq[mut_pos] == single_mut_info[0]
        assert single_mut_info[0] != single_mut_info[-1]
        tmp_str += single_mut_info[0] + str(mut_pos) + single_mut_info[-1] + ","
    tmp_str = tmp_str[:-1]
    if len(data[data.index == multi_mut_info]) >= 2 and multi_mut_info not in dup_list:
        dup_list.append(multi_mut_info)
        print(data[data.index == multi_mut_info])
        result.loc[tmp_str, "label"] = data.loc[multi_mut_info, "label"].mean()
        print("mean: ", result.loc[tmp_str, "label"])
    elif multi_mut_info not in dup_list:
        result.loc[tmp_str, "label"] = data.loc[multi_mut_info, "label"]

    mut_seq = list(wt_seq)
    for single_mut_info in multi_mut_info.split(","):
        mut_pos = int(single_mut_info[1:-1]) - 1
        mut_seq[mut_pos] = single_mut_info[-1]
    result.loc[tmp_str, "mut_seq"] = "".join(mut_seq)

os.makedirs(f"../features/wt", exist_ok=True)
with open(f"../features/wt/result.fasta", "w") as f:
    f.write(">{}\n{}\n".format(tmp_str, result.loc[tmp_str, "mut_seq"]))

result.index.name = "mut_name"
result["wt_seq"] = wt_seq
result = result[["wt_seq", "mut_seq", "label"]]

result["label"] = zscore(result["label"])
result.to_csv(f"{file_name}.csv")
