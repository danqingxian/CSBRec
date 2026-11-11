#!/bin/bash
set -euo pipefail
# CSBRec on SASRec, GRU4Rec and BERT4Rec x Yelp, Fashion, Beauty x seeds 42, 43, 44 (default settings).
# Usage: bash scripts/run_main.sh [gpu_id]
gpu=${1:-0}
for bb in sasrec gru4rec bert4rec; do
  for ds in yelp fashion beauty; do
    for seed in 42 43 44; do
      python -m csbrec.train --backbone $bb --dataset $ds --seed $seed --gpu_id $gpu \
          --out runs/main/$bb/$ds/s$seed
    done
  done
done
