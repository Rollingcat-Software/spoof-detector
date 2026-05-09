# 8. Ablation Studies

This section ablates each design decision. All ablations on the OULU-NPU P1 protocol unless noted.

## 8.1 Image-only vs. video-only vs. hybrid (Figure 3 / Table 5)

The most direct ablation: what does each track contribute? Three rows:

| Pipeline | APCER | BPCER | ACER | EER | AUC |
|---|---:|---:|---:|---:|---:|
| image_only | TBD | TBD | TBD | TBD | TBD |
| video_only | TBD | TBD | TBD | TBD | TBD |
| **hybrid** | TBD | TBD | TBD | TBD | TBD |

Hypothesis: hybrid ≥ max(image_only, video_only) on every metric. Even if one track is dominant, the hybrid should not regress beneath it because the calibrated fuser learns to weight the dominant track higher.

## 8.2 Calibrated vs. uniform analyzer weights (Table 6)

`MultiClassFuser` has weights `w_a` per analyzer (see §5). Compare:

| Configuration | ACER (P1) | ACER (P2) | ACER (P3) | ACER (P4) |
|---|---:|---:|---:|---:|
| Uniform weights (all 1.0) | TBD | TBD | TBD | TBD |
| Calibrated weights (paper) | TBD | TBD | TBD | TBD |
| Texture+Moire suppressed only | TBD | TBD | TBD | TBD |

The third row isolates the *single* anti-correlation finding — re-weighting Texture and Moire from 1.0 → 0.1 should account for a substantial fraction of the gap.

## 8.3 Peak-sensitive vs. mean session verdict (Table 7)

The proctoring claim: peak-sensitive aggregation prevents spoof-burst dilution.

| Aggregation | Real-only session ACC | Spoof-only session ACC | **Mixed session ACC** |
|---|---:|---:|---:|
| Mean | TBD | TBD | TBD |
| Peak-sensitive (50/50) | TBD | TBD | **TBD** |
| Worst-only | TBD | TBD | TBD |

The mixed-session column is the headline number. A session that is real for 50 seconds, spoof for 10, then real for 60 should be flagged as spoof. Peak-sensitive aggregation reports `spoof` ; pure-mean aggregation reports `real` and lets the attack through.

## 8.4 Per-analyzer leave-one-out (Table 8)

For each analyzer, retrain calibration with that analyzer removed, re-evaluate. Larger ACER increase = more important analyzer.

| Removed analyzer | ACER (P1) Δ | ACER (P2) Δ | ACER (P3) Δ | ACER (P4) Δ |
|---|---:|---:|---:|---:|
| MiniFASNet | TBD | TBD | TBD | TBD |
| Device boundary | TBD | TBD | TBD | TBD |
| Screen flicker | TBD | TBD | TBD | TBD |
| Micro-tremor | TBD | TBD | TBD | TBD |
| Blink | TBD | TBD | TBD | TBD |
| rPPG | TBD | TBD | TBD | TBD |
| Screen replay | TBD | TBD | TBD | TBD |
| Landmark variance | TBD | TBD | TBD | TBD |
| Background grid | TBD | TBD | TBD | TBD |
| AR filter | TBD | TBD | TBD | TBD |
| Temporal | TBD | TBD | TBD | TBD |

Hypothesis from §5 calibration: MiniFASNet is the single most important analyzer (largest weight 5.0, largest ACER increase when removed); micro-tremor and screen-flicker are runner-ups; texture and moire have ≈ zero impact (already at 0.1 weight, so removal makes no difference).

## 8.5 Active challenges (optional layer; Table 9)

When deployment can request user cooperation, add the active layer (light challenge + gesture challenge). Reported only on the in-house set + a CelebA-Spoof subset, as no public benchmark provides active-challenge ground truth.

| Configuration | APCER | BPCER | ACER | UX cost (s) |
|---|---:|---:|---:|---:|
| hybrid (passive only) | TBD | TBD | TBD | 0 |
| hybrid + light challenge | TBD | TBD | TBD | ~1.5 |
| hybrid + gesture challenge | TBD | TBD | TBD | ~3.0 |
| hybrid + both | TBD | TBD | TBD | ~4.0 |

The active layer is a deployment-time choice, not part of the headline pipeline reported in §7.

## 8.6 Session length curve (Figure 4)

ACER as a function of how many seconds of video are observed:

| Session length | 1 s | 5 s | 10 s | 30 s | 60 s | 5 min |
|---|---:|---:|---:|---:|---:|---:|
| ACER (P1) | TBD | TBD | TBD | TBD | TBD | TBD |

Hypothesis: ACER is high at 1 s (insufficient time for temporal analyzers to warm up), drops sharply between 5 s and 30 s as blink and rPPG come online, then plateaus.
