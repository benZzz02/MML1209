#!/usr/bin/env bash
# Ours Full multi-seed training × seeds 1-4 (seed0 already has trained checkpoint)
set -e
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
for seed in 1 2 3 4; do
  echo "[Seed $seed] Ours Full (4 GPUs)..."
  rm -rf "checkpoints/cholec_vals/multiseed/ours_full_s${seed}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_scpnet.py \
    -c "configs/multiseed/ours_full_seed${seed}.yaml"
  echo ""
done
echo "All done."
