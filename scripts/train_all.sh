#!/usr/bin/env bash
# Trains every model on both splits, skipping runs that already have results, then summarises.
#   ./scripts/train_all.sh                    # all models
#   MODELS="cnn_lstm_rr" ./scripts/train_all.sh
set -euo pipefail
export TF_CPP_MIN_LOG_LEVEL=2

MODELS=${MODELS:-"cnn_lstm_rr cnn_lstm lstm"}
mkdir -p models
for split in inter intra; do
  for model in $MODELS; do
    if [ -f "results/$split/$model.json" ]; then
      echo "skip $split $model (results/$split/$model.json exists)"
      continue
    fi
    uv run python -m ecg.train --model "$model" --split "$split" 2>&1 | tee -a models/train.log
  done
done
uv run python -m ecg.report          # results/summary.md and results/figures/
