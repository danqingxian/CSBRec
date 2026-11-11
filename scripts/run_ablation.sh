#!/bin/bash
set -euo pipefail
# Module ablation on SASRec, seeds 42-44. "base" = LLM-ESR with the flipped-classroom framework and no CSBRec module.
# Usage: bash scripts/run_ablation.sh [gpu_id]
gpu=${1:-0}
declare -A FLAGS=(
  [base]="--ngg none --lambda_cv 0 --lambda_sp 0"
  [ngg_only]="--lambda_cv 0 --lambda_sp 0"
  [ccs_only]="--ngg none --lambda_sp 0"
  [iusp_only]="--ngg none --lambda_cv 0"
  [wo_ngg]="--ngg none"
  [wo_ccs]="--lambda_cv 0"
  [wo_iusp]="--lambda_sp 0"
  [full]=""
)
for ds in yelp fashion beauty; do
  for seed in 42 43 44; do
    for v in base ngg_only ccs_only iusp_only wo_ngg wo_ccs wo_iusp full; do
      python -m csbrec.train --backbone sasrec --dataset $ds --seed $seed --gpu_id $gpu ${FLAGS[$v]} \
          --out runs/ablation/$ds/s$seed/$v
    done
  done
done
