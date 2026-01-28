#!/usr/bin/env bash

set -Eeuo pipefail

mkdir -p log

nohup python generate_unpredicted_muts_csv.py --mut_counts 2 > log/mut_counts_2.log 2>&1 &
nohup python generate_unpredicted_muts_csv.py --mut_counts 3 > log/mut_counts_3.log 2>&1 &
nohup python generate_unpredicted_muts_csv.py --mut_counts 4 > log/mut_counts_4.log 2>&1 &
