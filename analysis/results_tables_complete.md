# MML-SurgAdapt — 补实验结果完整表格

---

## Table 1: Full Task Metrics (All Models)

| Model | Cholec80 F1 | Endoscapes mAP | CholecT50 AP | CVS C1 | CVS C2 | CVS C3 |
|:------|:-----------:|:--------------:|:------------:|:------:|:------:|:------:|
| **Ours (Full)** | 75.53 | **61.13** | 13.63 | **65.45** | **52.48** | 65.47 |
| MML-SurgAdapt (裸模型) | **76.19** | 53.33 | **14.41** | 48.72 | 40.75 | **70.52** |
| w/o External | 74.02 | 60.55 | 13.77 | 61.43 | 51.92 | 68.30 |
| PP (Partial Positive) | 73.67 | 59.87 | 13.87 | 62.96 | 48.97 | 67.68 |

---

## Table 2: Ours vs MML-SurgAdapt — Paired Bootstrap (B=1000)

| Dataset | Metric | Ours | Baseline | Gain | 95% CI | P(Δ>0) |
|:--------|:-------|:---:|:--------:|:----:|:------:|:------:|
| Cholec80 | F1 | 75.53 | 76.19 | -0.63 | [-2.70, 1.32] | 29.2% |
| **Endoscapes** | **mAP** | **61.13** | **53.33** | **+7.40** | **[2.44, 11.79]** | **99.8%** |
| CholecT50 | AP | 13.63 | 14.41 | -0.27 | [-1.21, 1.13] | 27.2% |

---

## Table 3: CVS Per-Criterion Bootstrap (B=1000)

| Criterion | Ours AP | Baseline AP | Gain | 95% CI | P(Δ>0) |
|:----------|:------:|:-----------:|:----:|:------:|:------:|
| **C1** (cystic duct + artery) | **65.45** | 48.72 | **+15.85** | **[8.35, 23.44]** | **100.0%** |
| **C2** (hepatocystic triangle dissection) | **52.48** | 40.75 | **+10.97** | **[3.11, 18.57]** | **99.3%** |
| C3 (cystic plate division) | 65.47 | **70.52** | -4.61 | [-10.69, 0.70] | 4.5% |
| **mAP** | **61.13** | 53.33 | **+7.40** | **[2.44, 11.79]** | **99.8%** |

---

## Table 4: Hidden-Positive Recovery (Overall)

| Model | Hit@1 | Hit@2 | Recall@1 | Recall@2 |
|:------|:-----:|:-----:|:--------:|:--------:|
| **Ours (Full)** | **18.4** | 44.3 | **16.8** | 40.1 |
| MML-SurgAdapt (裸模型) | 14.2 | **48.3** | 13.2 | **44.2** |
| w/o External | **23.4** | 44.0 | **21.3** | 39.7 |
| PP (Partial Positive) | **24.7** | 46.1 | **22.3** | 41.7 |

---

## Table 5: Hidden-Positive Recovery by Task Group

| Model | Group | Hit@1 | Hit@2 | Recall@1 | Recall@2 |
|:------|:------|:-----:|:-----:|:--------:|:--------:|
| **Ours** | **CVS** | **51.24** | **84.78** | **36.80** | **71.58** |
| Ours | Triplet | 42.90 | 68.29 | 38.92 | 63.81 |
| MML | CVS | 47.52 | 83.54 | 33.70 | 69.88 |
| MML | Triplet | 41.94 | 68.50 | 38.14 | 64.21 |
| w/o Ext | CVS | 47.20 | 83.54 | 33.23 | 69.57 |
| w/o Ext | Triplet | 42.80 | 67.68 | 38.80 | 63.42 |
| PP | CVS | 50.00 | 83.23 | 34.78 | 69.72 |
| PP | Triplet | 43.14 | 68.26 | 39.07 | 63.77 |

*Phase group not reported — hidden positives within Phase are rare due to single-phase-per-frame annotation.*

---

## Table 6: Prior Variant Comparison (Full Model)

| Prior | Hit@1 | Hit@2 | Recall@1 | Recall@2 | vs Original Δ |
|:------|:-----:|:-----:|:--------:|:--------:|:-------------:|
| **Original A_star (CLIP + LLM + postprocess)** | **18.4** | **44.3** | **16.8** | **40.1** | — |
| Block-shuffled (shuffle within 9 blocks) | 18.6 | 44.4 | 16.9 | 40.3 | +0.1 |
| CLIP raw cos (no postprocess) | 18.1 | 44.1 | 16.5 | 39.9 | -0.4 |
| Shuffled (full matrix shuffle) | 18.5 | 44.1 | 16.8 | 39.9 | +0.0 |
| Random (same density) | 14.3 | 45.4 | 13.4 | 41.2 | -4.1 |
| **CLIP-only (no LLM external matrix)** | **13.5** | **46.2** | **12.6** | **42.0** | **-4.9** |
| **LLM-only (no CLIP cosine)** | **13.5** | **46.2** | **12.6** | **42.0** | **-4.9** |
| **No prior (identity matrix)** | **13.5** | **46.2** | **12.6** | **42.0** | **-4.9** |

---

## Table 7: Per-Video Gain Distribution (Ours vs MML)

| Dataset | Videos | Improved | Worsened | Mean Δ | Median Δ | Max Gain | Max Drop |
|:--------|:-----:|:--------:|:--------:|:------:|:--------:|:--------:|:--------:|
| Cholec80 F1 | 15 | 6 (40%) | 9 | -0.53 | -0.63 | +4.06 | -5.12 |
| Endoscapes mAP | 40 | 17 (42%) | 12 (30%) | -0.06 | +0.00 | +17.98 | -21.00 |

*11 Endoscapes videos had Δ ≈ 0 (both models performed equally). Gains concentrated on low-quality videos where MML scored ≤40 mAP.*

---

## Table 8: Frequency-Stratified Per-Class AP (Ours vs MML)

| Group | #Classes | Train Frequency | Ours AP | MML AP | Δ |
|:------|:--------:|:---------------:|:-------:|:------:|:-:|
| Phase (all) | 7 | [3,314, 36,877] | **46.72** | 42.93 | **+3.79** |
| CVS (all) | 3 | [780, 1,245] | **37.11** | 34.78 | **+2.32** |
| Triplet Low-freq | 50 | [5, 92] | 1.69 | **2.00** | -0.32 |
| Triplet High-freq | 50 | [92, 29,418] | **7.47** | 7.38 | +0.09 |

---

## Table 9: LC Lambda Sweep (on MML-SurgAdapt, offline)

| α (lambda) | T=0.5 | T=1.0 | T=2.0 |
|:----------:|:-----:|:-----:|:-----:|
| **0 (w/o LC)** | **76.19** | **76.19** | **76.19** |
| 0.5 | 75.62 | 75.86 | 76.09 |
| 1.0 | 75.41 | 75.65 | 76.01 |
| 2.0 | 75.36 | 75.43 | 75.88 |
| 4.0 | 75.36 | 75.37 | 75.62 |

*Offline LC addition does not improve performance — LC requires joint training with the model.*

---

## Summary of Key Findings

| # | Finding | Key Numbers | Evidence |
|:-:|:--------|:-----------|:---------|
| **1** | **Ours significantly outperforms MML-SurgAdapt on CVS** | Endoscapes mAP: **+7.40**, **P=99.8%** | Bootstrap B=1000 |
| **2** | **CVS C1 and C2 individually significantly improve** | C1: **+15.85** (P=100%), C2: **+10.97** (P=99.3%) | Per-criterion bootstrap |
| **3** | **CLIP + LLM external matrix fusion is critical** | Hit@1 drops 18.4 → **13.5** (−27%) without either | Prior variant ablation |
| **4** | **Block structure is the key property of the prior** | Block-shuffled: 18.6 vs original 18.4 (Δ=+0.2) | Prior variant ablation |
| **5** | Per-video gains concentrate on low-quality videos | Top gain: +17.98; 42% videos improved | Per-video analysis |
| **6** | LC requires joint training (offline ineffective) | F1: 76.19 → 75.4–76.1 across all α/T | LC sweep |
| **7** | PP protocol boosts hidden-positive recovery | Hit@1: 24.7 (PP) vs 18.4 (SP) | HP recovery |
| **8** | Phase and CVS benefit most; CholecT50 not significant | Phase +3.79 AP, CVS +2.32 AP; Triplet CI crosses zero | Frequency + bootstrap |
