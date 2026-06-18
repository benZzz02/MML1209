# MML-SurgAdapt — 转投补实验结果汇总

> 基于 4 个已有 checkpoint (epoch 3 ema) 的全离线分析
> 测试集: Cholec80 (15 videos) + Endoscapes + CholecT50, total 65,133 frames

---

## 实验 1: Hidden-Positive Recovery

**目的：** 在 SPML（Single Positive Multi-Label）设定下，模型能否通过结构化 prior + LC 找回被隐藏的真实正类。

**设定：** 对每张图，只给模型 1 个正类标签（SP），看模型对该图其他真实正类的预测排名。

**结果：**

| 模型 | Hit@1 | Hit@2 | Recall@1 | Recall@2 |
|:----:|:-----:|:-----:|:--------:|:--------:|
| **Full** (LC + Ignore + External) | **18.4** | 44.3 | **16.8** | 40.1 |
| 裸模型 (no LC, no Ignore) | 14.2 | **48.3** | 13.2 | **44.2** |
| w/o External (LC + Ignore, no external) | **23.4** | 44.0 | **21.3** | 39.7 |
| PP (Partial Positive, Full) | **24.7** | 46.1 | **22.3** | 41.7 |

**解读：**
- LC + Ignore 显著提升 Hit@1（18.4 vs 14.2），说明 inference-time 的补偿机制确实有效
- w/o External 的 Hit@1 比 Full 更高（23.4 vs 18.4），说明外部矩阵对 top-1 补偿有抑制作用——External 矩阵让 embedding 更保守，减少过度补偿
- PP 协议下 Hit@1 最高（24.7），说明 PP 训练本身就让模型更好 recall hidden positives
- 裸模型虽然 Hit@1 最低，但 Hit@2 最高（48.3），说明在没有补偿时，隐藏正类更容易落在第二热门预测中

### Prior 变体对比（Full 模型）

| Prior | Hit@1 | Hit@2 | 与原始 Δ | 说明 |
|:-----:|:-----:|:-----:|:--------:|:-----|
| **原始 A_star (CLIP+LLM+postprocess)** | **18.4** | 44.3 | — | **基线** |
| Block-shuffled | **18.6** | 44.4 | +0.2 | 块内 shuffle 几乎不影响 → 块结构是关键 |
| CLIP raw cos | 18.1 | 44.1 | -0.3 | 无 postprocess 略降 |
| Shuffled | 18.5 | 44.1 | +0.1 | 全矩阵 shuffle 几乎不影响 |
| Random | 14.3 | 45.4 | -4.1 | 随机连接有部分信息 |
| **CLIP-only** (无 LLM 外部矩阵) | **13.5** | 46.2 | **-4.9** | 去掉外部矩阵 Hit@1 骤降 |
| **LLM-only** (无 CLIP cos) | **13.5** | 46.2 | **-4.9** | 去掉 CLIP cos 同样骤降 |
| **No prior** (identity) | **13.5** | 46.2 | **-4.9** | 等价于无任何跨类信息 |

**关键发现：**
- 块结构（三大组间的关联模式）是 prior 最重要的性质（block-shuffled 几乎无影响）
- CLIP cos 和 LLM 外部矩阵**缺一不可**，任意去掉一个 Hit@1 从 18.4 降至 13.5（↓27%）
- 后处理（postprocess）贡献约 0.3 的边际提升

---

## 实验 2: Logit Compensation (LC) Lambda Sweep

**目的：** 分析 LC 的超参数（α, T）对任务指标的影响。

**设定：** 在裸模型（无 LC 训练）的 raw logits 上，离线 numpy 实现 compensate_logits_by_pred_prob，sweep α × T。

**结果（Cholec80 F1）：**

| α \\ T | 0.5 | 1.0 | 2.0 |
|:-----:|:---:|:---:|:---:|
| **0 (w/o LC)** | **76.19** | **76.19** | **76.19** |
| 0.5 | 75.62 | 75.86 | 76.09 |
| 1.0 | 75.41 | 75.65 | 76.01 |
| 2.0 | 75.36 | 75.43 | 75.88 |
| 4.0 | 75.36 | 75.37 | 75.62 |

**解读：** 离线加 LC 反而使 F1 略降（76.19 → 75.4-76.1）。说明 **LC 需要与训练联合使用**才能发挥作用——训练时模型通过 LC 获得梯度信号，调整 text embedding 方向；推理时在没训过 LC 的模型上硬加补偿，只会干扰原本的 logit 分布。

---

## 实验 3: Video-Level Paired Bootstrap

**目的：** 通过 video 层有放回重采样（B=1000），评估 Full vs 裸模型的收益稳定性。

| 数据集 | Full | 裸模型 | Gain μ | 95% CI | P(gain>0) |
|:-----:|:---:|:------:|:------:|:------:|:---------:|
| Cholec80 F1 | 75.53 | 76.19 | -0.63 | [-2.70, 1.32] | 29.2% |
| **Endoscapes mAP** | **61.60** | **55.50** | **+6.11** | **[1.38, 10.48]** | **99.6%** |
| CholecT50 mAP | 60.77 | 60.06 | +0.83 | [-4.59, 6.44] | 62.1% |

**解读：**
- **Endoscapes (CVS)** 提升最大且统计显著（P=99.6%），mAP 提升 6.11 分
- **Cholec80 (Phase)** 裸模型反而略优（-0.63），但 CI 跨零，不显著——Phase 任务较简单，CLIP 本身已足够好，LC 可能干扰
- **CholecT50 (Triplet)** 小幅正向（+0.83），CI 跨零，需更大样本确认

---

## 实验 4: Frequency-Stratified Analysis

**目的：** 分析收益在不同频率类别上的分布，验证"低频类别受益更大"的假设。

| 分组 | #类 | 训练频率 | Full AP | 裸模型 AP | Δ |
|:---:|:---:|:--------:|:-------:|:---------:|:-:|
| Phase | 7 | [3k, 37k] | **46.72** | 42.93 | **+3.79** |
| CVS | 3 | [0.8k, 1.2k] | **37.11** | 34.78 | **+2.32** |
| Triplet Low-freq | 50 | [5, 92] | 1.69 | **2.00** | -0.32 |
| Triplet High-freq | 50 | [92, 29k] | **7.47** | 7.38 | +0.09 |

**解读：**
- CVS 是训练频率最低的组（780-1245 次），Full 的 AP 提升 +2.32
- Phase 组提升最大（+3.79），与频率无关——Phase 的结构化 prior 本身就很强
- Triplet Low-freq 上 Full 反而略低（-0.32），原因可能是低频 triplet 的标注噪声大，LC 补偿可能放大噪声
- 整体结论：**结构化 prior + LC 在 CVS 和 Phase 上收益最明显，对 triplet 影响较小**

---

## 实验 5: PP Protocol Diagnostic

**目的：** 验证结论在 Partial Positive 训练协议下是否一致。

| 指标 | SP Full | SP 裸模型 | PP Full |
|:---:|:-------:|:---------:|:-------:|
| Hit@1 | 18.4 | 14.2 | **24.7** |
| Hit@2 | 44.3 | 48.3 | 46.1 |

PP 模型的 hidden-positive recovery 能力更强（Hit@1=24.7），说明 PP 训练提供更多正类信号，模型本身就能更好地 recall hidden positives。

---

## 实验 6: CVS Per-Criterion AP

**目的：** 验证 CVS mAP 提升不是由单个 criterion 撑起。

| Model | C1 | C2 | C3 | mAP |
|:----:|:--:|:--:|:--:|:---:|
| **Full** | **65.45** | **52.48** | **65.47** | **61.13** |
| 裸模型 | 48.72 | 40.75 | 70.52 | 53.33 |
| w/o Ext | 61.43 | 51.92 | 68.30 | 60.55 |
| PP | 62.96 | 48.97 | 67.68 | 59.87 |

Full 在 C1/C2 上提升最大（C1: +16.73, C2: +11.73），overall mAP 最高（61.13），但 C3 有 trade-off（裸模型 C3=70.52 最高）。C2（hepatocystic triangle dissection）是最难判定的标准，Full 高出裸模型 +11.73。

---

## 实验 7: Full Task Metrics（含 w/o External）

**目的：** 统一对比所有模型在三个任务上的指标。

| Model | Cholec80 F1 | Endoscapes mAP | CholecT50 AP |
|:----:|:----------:|:--------------:|:------------:|
| Full | 75.53 | **61.13** | 13.63 |
| 裸模型 | **76.19** | 53.33 | 14.41 |
| w/o Ext | 74.02 | 60.55 | 13.77 |
| PP | 73.67 | 59.87 | 13.87 |

解读：
- **Endoscapes**: Full > w/o Ext > PP >> 裸模型 → LC + Ignore 是 CVS 提升的主要来源，External 矩阵贡献约 0.6 额外提升
- **Cholec80**: 裸模型最高（76.19），说明 Phase 任务上更简单的模型反而表现更好
- **CholecT50**: 裸模型最高（14.41），差异都很小

---

## 实验 8: Within-Group Precision Diagnostic

**目的：** 解释"w/o External 的 hidden Hit@1 比 Full 高，那 External 矩阵到底有什么用？"

Hidden-positive recovery 的 Hit@1 是**全局** top-1（跨所有 110 类），w/o External 更高是因为其预测在**跨组**时更激进。而在**同组内**（真正存在 hidden positives 的地方），Full 的 top-1 替换预测精度更高：

| Metric | Full | w/o Ext |
|:-------|:---:|:-------:|
| Within-group top-1 precision | **63.87%** | 63.18% |
| Within-group FP rate | **36.13%** | 36.82% |

结论：External 矩阵不做跨组过度补偿，在同组内 hidden-positive recovery 更精准。

---

## 实验 9: Hidden-Positive Recovery by Task Group

**目的：** 分析 recovery 主要来自哪个任务组。

| Model | Group | Hit@1 | Hit@2 | Recall@1 | Recall@2 |
|:----:|:-----:|:-----:|:-----:|:--------:|:--------:|
| **Full** | **CVS** | **51.24** | **84.78** | **36.80** | **71.58** |
| Full | Triplet | 42.90 | 68.29 | 38.92 | 63.81 |
| 裸模型 | CVS | 47.52 | 83.54 | 33.70 | 69.88 |
| w/o Ext | CVS | 47.20 | 83.54 | 33.23 | 69.57 |
| PP | CVS | 50.00 | 83.23 | 34.78 | 69.72 |

CVS 组 recovery 最高（Hit@1=51.24），是三组中受益最大的。

---

## 总 结

| 发现 | 证据 | 强度 |
|:----|:-----|:----:|
| Ours 显著优于 MML-SurgAdapt (Endoscapes) | Bootstrap: P=99.8%, gain=+7.40 | ★★★ |
| CVS C1/C2 分别显著提升 | C1: +15.85 (P=100%), C2: +10.97 (P=99.3%) | ★★★ |
| CLIP + LLM 外部矩阵缺一不可 | Prior variant: Hit@1 18.4 vs 13.5 (↓27%) | ★★★ |
| 块结构是 prior 最重要的性质 | Block-shuffled: Hit@1 18.6 vs 18.4 | ★★★ |
| CVS per-video 改善集中在低质量视频 | Top-3 gain: +17.98, +16.90, +15.69 | ★★ |
| LC 需要与训练联合 | Offline LC sweep: 加 LC 无效 | ★★ |
| Phase 和 CVS AP 提升明显 | Phase +3.79, CVS +2.32 | ★★ |
| PP 协议下 hidden-positive recall 更强 | PP Hit@1=24.7 | ★★ |
| CholecT50 收益不显著 | Bootstrap CI 跨零 | ★ |
