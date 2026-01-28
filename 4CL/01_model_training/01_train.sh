#!/usr/bin/env bash

set -Eeuo pipefail

nohup python train.py > train.log 2>&1 &

