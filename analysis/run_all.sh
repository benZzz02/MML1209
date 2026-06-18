#!/usr/bin/env bash
# =============================================================================
# run_all.sh — 转投补实验全流程
#
# 前置条件:
#   1. GPU (建议 ≥24GB显存)
#   2. 项目根目录 /data/MML1209 下的 data/checkpoints 完整
#   3. 已安装 requirements.txt 环境
#
# 用法:
#   bash analysis/run_all.sh [--skip-inference]
#
# 参数:
#   --skip-inference  跳过 Step 1 推理（已有 Step 1 输出时使用）
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ANALYSIS="$ROOT/analysis"
OUTPUT="$ANALYSIS/output"
mkdir -p "$OUTPUT"

SKIP_INFERENCE=false
if [[ "${1:-}" == "--skip-inference" ]]; then
    SKIP_INFERENCE=true
    echo "[run_all] Skipping Step 1 inference"
fi

# ── 显卡检测 ──
GPU_AVAILABLE=false
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    GPU_AVAILABLE=true
    echo "[run_all] GPU detected"
fi

# =============================================================================
# Step 1: Unified Inference
# =============================================================================
if [ "$SKIP_INFERENCE" = false ]; then

# 1a. Full model (round3: 最佳模型位置, LC + Ignore + External GPT)
FULL_CKPT="$ROOT/checkpoints/cholec_vals/scpnet_hill_Ignore_sp/round3__(最佳模型位置)/trial_scpnet_hill_Ignore_sp/epoch_3_mAP_6.03_loss_64.5186_ema.ckpt"
FULL_OUT="$OUTPUT/full_epoch3"
echo "[Step 1a] Full model (epoch 3)..."
python "$ANALYSIS/step1_unified_inference.py" \
    -c "$ROOT/configs/scpnet_hill_Ignore_sp.yaml" \
    --ckpt "$FULL_CKPT" \
    --output-dir "$FULL_OUT" \
    --external-matrix "$ROOT/surgical_matrix_clip_boosted_gpt-5.2.npy"

# 1b. 裸模型 (round48: 消融实验-LC-SNI, no LC, no Ignore)
NOBOTH_CKPT="$ROOT/checkpoints/cholec_vals/scpnet_hill_Ignore_sp/round48__(消融实验-LC-SNI)/trial_scpnet_hill_Ignore_sp/epoch_3_mAP_5.46_loss_90.7614_ema.ckpt"
NOBOTH_OUT="$OUTPUT/noboth"
echo "[Step 1b] 裸模型 (no LC, no Ignore)..."
python "$ANALYSIS/step1_unified_inference.py" \
    -c "$ROOT/configs/scpnet_hill_Ignore_sp.yaml" \
    --ckpt "$NOBOTH_CKPT" \
    --output-dir "$NOBOTH_OUT" \
    --disable-lc --disable-ignore \
    --external-matrix "$ROOT/surgical_matrix_clip_boosted_gpt-5.2.npy"

# 1c. wo External (round58: 消融-替换矩阵, no external matrix)
NOEXT_CKPT="$ROOT/checkpoints/cholec_vals/scpnet_hill_Ignore_sp/round58__（消融-替换矩阵）/trial_scpnet_hill_Ignore_sp/epoch_3_mAP_5.90_loss_65.3619_ema.ckpt"
NOEXT_OUT="$OUTPUT/noext"
echo "[Step 1c] w/o External..."
python "$ANALYSIS/step1_unified_inference.py" \
    -c "$ROOT/configs/scpnet_hill_Ignore_sp.yaml" \
    --ckpt "$NOEXT_CKPT" \
    --output-dir "$NOEXT_OUT" \
    --no-external

# 1d. PP model (Ignore_pp, epoch 3)
PP_CKPT="$ROOT/checkpoints/cholec_vals/scpnet_hill_Ignore_pp/round2/trial_scpnet_hill_Ignore_pp/epoch_3_mAP_5.93_loss_66.8087_ema.ckpt"
PP_OUT="$OUTPUT/pp"
echo "[Step 1d] PP model..."
python "$ANALYSIS/step1_unified_inference.py" \
    -c "$ROOT/configs/scpnet_hill_Ignore_pp.yaml" \
    --ckpt "$PP_CKPT" \
    --output-dir "$PP_OUT" \
    --no-external

echo "[Step 1] All inference done."

fi  # SKIP_INFERENCE

# =============================================================================
# Step 2: Hidden-positive recovery + Prior diagnostic
# =============================================================================
echo ""
echo "============================================"
echo "Step 2: Hidden-positive recovery + Prior"
echo "============================================"
python "$ANALYSIS/step2_hidden_pos.py" \
    "$FULL_OUT" "$NOBOTH_OUT" "$NOEXT_OUT" "$PP_OUT" \
    --labels "Full" "裸模型" "w/o Ext" "PP" \
    --save "$OUTPUT/step2_results.json"

# =============================================================================
# Step 4: LC lambda sweep (uses 裸模型 data, applies LC offline)
# =============================================================================
echo ""
echo "============================================"
echo "Step 4: LC lambda sweep"
echo "============================================"
python "$ANALYSIS/step4_lc_sweep.py" \
    "$NOBOTH_OUT" \
    --save "$OUTPUT/step4_lcsweep.json"

# =============================================================================
# Step 5: Bootstrap (Full vs 裸模型)
# =============================================================================
echo ""
echo "============================================"
echo "Step 5: Video-level bootstrap"
echo "============================================"
python "$ANALYSIS/step5_bootstrap.py" \
    --ref "$FULL_OUT" --baseline "$NOBOTH_OUT" \
    --n-bootstrap 1000 \
    --label-ref "Full" --label-base "裸模型" \
    --save "$OUTPUT/step5_bootstrap.json"

# =============================================================================
# Step 6: Frequency-stratified analysis
# =============================================================================
echo ""
echo "============================================"
echo "Step 6: Frequency analysis"
echo "============================================"
python "$ANALYSIS/step6_freq_analysis.py" \
    --ref "$FULL_OUT" --baseline "$NOBOTH_OUT" \
    --label-ref "Full" --label-base "裸模型" \
    --save "$OUTPUT/step6_freq.json"

echo ""
echo "============================================"
echo "All done! Results in: $OUTPUT/"
echo "============================================"
echo "  step2_results.json  — Hidden pos recovery + prior"
echo "  step4_lcsweep.json  — LC lambda sweep"
echo "  step5_bootstrap.json — Bootstrap CI"
echo "  step6_freq.json     — Frequency-stratified"
echo "============================================"
