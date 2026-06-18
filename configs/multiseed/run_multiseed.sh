#!/usr/bin/env bash
# Multi-seed training: Ours Full vs MML-Base, seeds 1,2
# Run from project root: bash configs/multiseed/run_multiseed.sh
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Starting multi-seed training..."
echo "Logs will be saved to checkpoints/cholec_vals/multiseed/"
echo ""

# ── ours_full seed1 ──
echo "[1/4] Ours Full, seed=1 (GPU 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scpnet.py -c configs/multiseed/ours_full_seed1.yaml
echo "Done: ours_full seed1"
echo ""

# ── ours_full seed2 ──
echo "[2/4] Ours Full, seed=2 (GPU 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scpnet.py -c configs/multiseed/ours_full_seed2.yaml
echo "Done: ours_full seed2"
echo ""

# ── mml_base seed1 ──
echo "[3/4] MML-Base, seed=1 (GPU 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scpnet.py -c configs/multiseed/mml_base_seed1.yaml
echo "Done: mml_base seed1"
echo ""

# ── mml_base seed2 ──
echo "[4/4] MML-Base, seed=2 (GPU 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scpnet.py -c configs/multiseed/mml_base_seed2.yaml
echo "Done: mml_base seed2"
echo ""

echo "=========================================="
echo "All 4 training runs completed!"
echo "Checkpoints: checkpoints/cholec_vals/multiseed/"
echo "=========================================="
