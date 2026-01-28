#!/usr/bin/env bash

set -Eeuo pipefail

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup

conda activate codonPTDE

CUDA_VISIBLE_DEVICES=1 nohup python inference_muts.py --mut_counts 2 > log/mut_counts_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python inference_muts.py --mut_counts 3 > log/mut_counts_3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python inference_muts.py --mut_counts 4 > log/mut_counts_4.log 2>&1 &
