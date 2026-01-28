#!/usr/bin/env bash

set -Eeuo pipefail

eval "$(conda shell.bash hook)"
conda activate spired_de

CUDA_VISIBLE_DEVICES=0 python generate_esm2_embedding.py

CUDA_VISIBLE_DEVICES=0 python generate_esmc_embedding.py

CUDA_VISIBLE_DEVICES=0 python generate_spired_embedding.py
