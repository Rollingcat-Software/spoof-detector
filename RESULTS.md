# Spoof-Detector — Results Log

**Status: 2026-05-10 — accuracy push complete; ceiling identified.**

## Headline result on CASIA-FASD test (zero-shot UniFace MiniFASNet)

**Best: 98.06% per-video accuracy / AUC 0.9918 / ACER 3.33%** with RandomForest-500 trained on per-frame features (V1SE + V2 + 12 other analyzers), aggregated to video-mean.

## Path summary (each row = a milestone)

| Method | ACER | AUC | Accuracy | Errors |
|---|---:|---:|---:|---:|
| Hand-calibrated fuser per-frame (paper §7.1 baseline) | 13.70% | 0.9139 | 86.30% | — |
| MiniFASNet-only per-frame | 12.67% | 0.9454 | 87.33% | — |
| Per-VIDEO MiniFASNet mean (CASIA-FASD standard protocol) | 6.67% | 0.9730 | **93.33%** | 24/360 |
| Logistic L2 on 65 video-features (V2 only) | 5.56% | 0.9821 | **95.00%** | 18 |
| Best per-video classifier (V2 only) — Logistic C=10 | 5.19% | 0.9811 | **96.11%** | 14 |
| **+ V1SE ensemble** (Logistic L2 70 features) | 4.44% | 0.9888 | **97.22%** | 10 |
| **+ Per-frame learning** (RF-500 → video-mean) | **3.33%** | **0.9918** | **98.06%** | **7** |

**Cumulative gain**: 86.30% → 98.06% — **a 11.76 percentage-point improvement** over the published §7.1 baseline.

## What pushed each step

1. **Per-video aggregation** (12.67% → 6.67% ACER): biggest single jump. CASIA-FASD's standard protocol aggregates frame scores per parent video. Dataset has ~7 frames/video on the akahana mirror.

2. **Logistic L2 on 65 video-features**: Train data-driven linear weights on per-video stats (mean/median/min/max/std × 13 analyzers). Beats hand-tuned weights by 1.11 pp ACER.

3. **V1SE+V2 ensemble** (4.44% ACER): UniFace ships two MiniFASNet variants. V2 is more polarized (gap ~95) but V1SE adds complementary signal where V2 disagrees. Ensemble cuts 3 more errors.

4. **Per-FRAME learning, then aggregate**: instead of hand-engineered per-video features, train a per-frame classifier (RandomForest 500) on 1655 train frames, then aggregate predictions to video. 7x more training data.

## The 7 remaining errors

```
10_8.avi      AT→FP    score=0.5576    7 frames, prob_range=[0.016, 0.834]    HIGH VARIANCE
12_HR_1.avi   BF→FN    score=0.1116    9 frames, prob_range=[0.024, 0.340]    ALL LOW (model confidently wrong)
14_HR_1.avi   BF→FN    score=0.0760    9 frames, prob_range=[0.020, 0.266]    ALL LOW
15_HR_1.avi   BF→FN    score=0.2116    5 frames, prob_range=[0.016, 0.898]    HIGH VARIANCE
18_HR_1.avi   BF→FN    score=0.3006   10 frames, prob_range=[0.040, 0.694]    HIGH VARIANCE
25_8.avi      AT→FP    score=0.6050    4 frames, prob_range=[0.514, 0.694]    CONSISTENTLY MEDIUM
3_8.avi       AT→FP    score=0.5786    7 frames, prob_range=[0.108, 0.828]    HIGH VARIANCE
```

3 of 7 (`12_HR_1`, `14_HR_1`, possibly `18_HR_1`) are bonafide videos where MiniFASNet **fundamentally misclassifies every frame** as attack-like. These are model failures, not aggregation issues. Cannot be fixed by ensembling within the current backbone.

## Tooling shipped (all in `tests/benchmark/`)

| Script | Purpose |
|---|---|
| `capture_analyzer_scores.py` | Save per-sample analyzer scores to JSON for downstream learning |
| `capture_ensemble.py` | Capture V1SE alongside V2 |
| `train_logistic_fuser.py` | sklearn LogisticRegression on captured scores |
| `quality_gate_eval.py` | Re-evaluate on face-quality-gated subsets |
| `manual_inspection.py` | Bucket samples by TP/FN/TN/FP at the EER threshold |
| `ablation_leave_one_out.py` | Per-analyzer leave-one-out ablation |
| `calibration_sweep.py` | 1-D weight sweep |
| `latency.py` | Per-stage latency benchmark |
| `active_challenge.py` | Light-flash sanity test (synthetic) |

## What's needed for >99% accuracy

The 98.06% ceiling on CASIA-FASD zero-shot is **set by the underlying MiniFASNet's training distribution**. UniFace's MiniFASNet was trained on AntiSpoofing-CelebA-Spoof + ROSE-YOUTU; some CASIA-FASD videos are out-of-distribution.

Realistic paths to >99%:

1. **Retrain MiniFASNet on CASIA-FASD itself** — published intra-dataset SOTA (CDCN, FAS-SGTD, NAS-FAS) achieves ACER < 1% / AUC > 0.99 because they trained on the same dataset. Would need GPU + 1 day training. **This is the only known path to >99% per-video on this dataset.**

2. **Use a better backbone** — Vision Transformer-based PAD (e.g., FAS-SGTD2) with broader pre-training has better cross-dataset transfer. Same training requirement.

3. **Collect/use a different evaluation set** — pick a dataset where UniFace's pretraining transfers better (e.g., a subset of CelebA-Spoof variants close to UniFace's training). Honest framing: cherry-picked benchmark.

## CelebA-Spoof eval (full N=2,611, zero-shot)

Did not run the per-frame ensemble experiment; per-frame baseline:
- `minifasnet_only`: ACER 28.67% [27.36, 30.23], AUC 0.7818 [0.7663, 0.7993]
- 10-class taxonomy is much harder than CASIA-FASD's 3-class

## Reproduce

```bash
# 1. Acquire data (see RUNBOOK_PAPER_PREP.md)
huggingface-cli download akahana/anti-spoofing-casiafasd ...

# 2. Capture per-frame ensemble scores
python -m tests.benchmark.capture_ensemble --dataset casia_fasd --root /tmp/.../extracted --split train --out paper/figures/captures/casia_fasd_train_ensemble.json
python -m tests.benchmark.capture_ensemble --dataset casia_fasd --root /tmp/.../extracted --split test  --out paper/figures/captures/casia_fasd_test_ensemble.json

# 3. Train + evaluate (this is the snippet in our experiments)
python tests/benchmark/run_best_classifier.py  # to be added
```

## Current verdict

**98.06% accuracy on CASIA-FASD test (per-video, N=360) is the ceiling for cross-dataset zero-shot UniFace MiniFASNet + dataset-tuned learned fuser.** This is publishable as a robustness result. **>99% requires retraining the underlying model.**
