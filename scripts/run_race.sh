#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for dataset in cub pets flowers cars aircraft; do
  for seed in 1 2 3; do
    python -m race.train --dataset "$dataset" --seed "$seed" \
      --output "runs/race_${dataset}_s${seed}"
    python -m race.evaluate --run "runs/race_${dataset}_s${seed}"
  done
done
