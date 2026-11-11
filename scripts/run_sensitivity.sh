#!/bin/bash
set -euo pipefail
# Hyper-parameter sensitivity on SASRec, seed 42: NGG kappa, CCS weight lambda_cv, IUSP weight lambda_sp
# (one varied at a time around the defaults kappa -> inf, lambda_cv 0.25, lambda_sp 1).
# Usage: bash scripts/run_sensitivity.sh [gpu_id]
gpu=${1:-0}
for ds in yelp fashion beauty; do
  run() { python -m csbrec.train --backbone sasrec --dataset $ds --seed 42 --gpu_id $gpu "$@"; }
  for k in 1 4; do run --ngg ratio --kappa $k --out runs/sensitivity/$ds/kappa_$k; done
  for w in 0.5 1; do run --lambda_cv $w --out runs/sensitivity/$ds/lambda_cv_$w; done
  for w in 0.3 3; do run --lambda_sp $w --out runs/sensitivity/$ds/lambda_sp_$w; done
done
