#!/usr/bin/env bash
# Multi-seed training: Ours Full × seeds 0-4
# Run from project root: bash configs/multiseed/run_multiseed.sh
set -e

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

echo "Starting multi-seed training..."
echo ""

for seed in 0 1 2 3 4; do
  echo "[Seed $seed] Ours Full (4 GPUs)..."
  rm -rf "checkpoints/cholec_vals/multiseed/ours_full_s${seed}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_scpnet.py \
    -c "configs/multiseed/ours_full_seed${seed}.yaml"
  echo "Done: seed ${seed}"
  echo ""
done

echo "=========================================="
echo "All 5 training runs completed!"
echo "Checkpoints: checkpoints/cholec_vals/multiseed/"
echo "=========================================="
