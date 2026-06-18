#!/usr/bin/env bash
# PP protocol ablation: Full vs Base vs w/o Ext (single seed, unified external matrix)
# Run from project root: bash configs/pp/run_pp.sh
set -e

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "Starting PP ablation training..."
echo ""

# ── PP Full ──
echo "[1/3] PP Full (GPU 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scpnet.py -c configs/pp/pp_full.yaml
echo "Done: PP Full"
echo ""

# ── PP Base ──
echo "[2/3] PP Base (GPU 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scpnet.py -c configs/pp/pp_base.yaml
echo "Done: PP Base"
echo ""

# ── PP w/o Ext ──
echo "[3/3] PP w/o Ext (GPU 0,1)..."
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_scpnet.py -c configs/pp/pp_noext.yaml
echo "Done: PP w/o Ext"
echo ""

echo "=========================================="
echo "All 3 PP training runs completed!"
echo "Checkpoints: checkpoints/cholec_vals/pp_*/"
echo "=========================================="
