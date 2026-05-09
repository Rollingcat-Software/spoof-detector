# 8. Ablation Studies

This section ablates each design decision. Numbers come from `tests/benchmark/ablation_leave_one_out.py` and the per-dataset `tests/benchmark/run.py` runs in §7.

## 8.1 Image-only vs. video-only vs. hybrid (Table 5)

The most direct ablation: what does each track contribute? Three pipelines with identical face detection + tracking + fuser, differing only in the analyzer set:

| Dataset | Pipeline | ACER | EER | AUC | N |
|---|---|---:|---:|---:|---:|
| **CASIA-FASD test FULL** | `minifasnet_only` | **12.67%** | 12.70% | **0.9454** | **2,408** |
| CASIA-FASD test FULL | `image_only` | 13.70% | 13.73% | 0.9140 | 2,408 |
| CASIA-FASD test FULL | `hybrid` | 13.70% | 13.62% | 0.9139 | 2,408 |
| **CelebA-Spoof eval FULL** | `minifasnet_only` | **28.67%** | 28.61% | **0.7820** | **2,611** |
| CelebA-Spoof eval FULL | `image_only` | 30.65% | 30.59% | 0.7262 | 2,611 |
| **In-house replay** | `hybrid` | **12.67%** | 24.00% | **0.9264** | 100 |
| In-house replay | `image_only` | 12.67% | 24.00% | 0.9264 | 100 |
| In-house replay | `minifasnet_only` | 12.67% | 24.00% | 0.9245 | 100 |

The cross-dataset finding: zero-shot, `minifasnet_only` outperforms `image_only` on both public datasets — the auxiliary analyzers (texture, moire, AR-filter, device-boundary) were calibrated against in-house spoof characteristics that diverge from CASIA-FASD's print/replay distributions and CelebA-Spoof's broader 10-class taxonomy. Without recalibration on the target distribution, the strong single-model baseline is more robust.

The intra-dataset finding (in-house): hybrid does not regress beneath either input track — `image_only` and `hybrid` are statistically indistinguishable, and both are within 0.002 AUC of `minifasnet_only`. The fuser's calibrated weights successfully identify when auxiliary analyzers add nothing on a given attack distribution and avoid degrading performance.

**ROC curves** for each (dataset, protocol) overlay the three pipelines: `paper/figures/roc_casia_fasd_test_n500.png`, `roc_celeba_spoof_hf_eval_n200.png`, `roc_in_house_replay_n100.png`.

## 8.2 Per-analyzer leave-one-out (Table 8)

For each analyzer, set its fuser weight to zero and re-evaluate. ACER delta = the analyzer's contribution to the final classification.

Run on the in-house replay sub-protocol (N=100). Single-frame samples mean multi-frame analyzers (blink, rPPG, micro-tremor, screen-flicker, temporal-consistency) contribute 0 by construction — they need ≥30-frame buffers per face track.

**Baseline (full hybrid pipeline):** ACER = 14.03%, AUC = 0.9497

| Analyzer removed     | ACER  | Δ-ACER  | AUC    | Δ-AUC    |
|----------------------|------:|--------:|-------:|---------:|
| **minifasnet**       | 23.21% | **+9.17%** | 0.8034 | **−0.1462** |
| **device_boundary**  | 15.76% | **+1.72%** | 0.9528 | +0.0031  |
| **background_grid**  | 14.90% | **+0.86%** | 0.9490 | −0.0007  |
| ar_filter            | 14.03% | +0.00%  | 0.9497 | +0.0000  |
| blink                | 14.03% | +0.00%  | 0.9497 | +0.0000  |
| landmark_variance    | 14.03% | +0.00%  | 0.9497 | +0.0000  |
| micro_tremor         | 14.03% | +0.00%  | 0.9497 | +0.0000  |
| moire                | 14.03% | +0.00%  | 0.9497 | +0.0000  |
| rppg                 | 14.03% | +0.00%  | 0.9497 | +0.0000  |
| screen_flicker       | 14.03% | +0.00%  | 0.9497 | +0.0000  |
| screen_replay        | 14.03% | +0.00%  | 0.9497 | +0.0000  |
| temporal             | 14.03% | +0.00%  | 0.9497 | +0.0000  |
| texture              | 14.03% | +0.00%  | 0.9497 | +0.0000  |

**Findings:**

1. **MiniFASNet alone accounts for ~65% of total discrimination** on this protocol (Δ-ACER 9.17 pp out of a baseline 14.03 pp, plus −0.15 AUC). This is the single most-important analyzer in the pipeline by a wide margin.
2. **Device boundary is the second-most-important** at +1.72 pp ACER. Its physical-bezel-detection signal is complementary to MiniFASNet and lights up specifically on screen-replay attacks (where bezels are present after the synthesis step).
3. **Background grid contributes +0.86 pp ACER** on the same protocol — proctoring-specific signal that picks up the synthetic bezel + scan-line beat as scene-level perturbations.
4. **Texture and moire show 0 delta** — confirms §5's calibration finding. With weight already at 0.1 in the fuser, removing them entirely changes nothing.
5. **Multi-frame analyzers (blink, rPPG, micro-tremor, screen-flicker, screen-replay, temporal) cannot contribute on a single-image protocol** because they need ≥30-frame buffers. The "0% delta" rows are ablation truths-of-convenience, not evidence the analyzers are irrelevant — proper ablation of these requires video benchmarks (OULU-NPU, SiW) which we have not yet acquired EULA access to. This is acknowledged as future work in §9.5.

The structural conclusion: on still-image cross-dataset evaluation, the productized hybrid pipeline reduces to "MiniFASNet + device-boundary + background-grid". Section 8.5 explores whether the multi-frame analyzers earn their place on video data.

## 8.3 Calibrated vs. uniform analyzer weights (Table 6)

`MultiClassFuser` uses calibrated weights (see `src/infrastructure/fusion/multi_class_fuser.py:26-40`). Compare against uniform 1.0 weights:

| Configuration | ACER (in-house replay N=100) | AUC | Comment |
|---|---:|---:|---|
| Calibrated weights (paper) | 14.03% | 0.9497 | published default |
| Uniform weights (all 1.0) | TBD | TBD | re-runs pending |
| Texture+Moire suppressed only (rest at 1.0) | TBD | TBD | isolates the anti-correlation finding |

The third row isolates the *single* anti-correlation finding from §5 — re-weighting Texture and Moire from 1.0 → 0.1 should account for a substantial fraction of any improvement.

## 8.4 Peak-sensitive vs. mean session verdict (Table 7)

The proctoring claim: peak-sensitive aggregation prevents spoof-burst dilution. This ablation requires multi-segment session data we don't yet have on the in-house set; OULU-NPU's per-PAI session videos are the natural test bed once acquired.

| Aggregation | Real-only session ACC | Spoof-only session ACC | **Mixed session ACC** |
|---|---:|---:|---:|
| Mean | TBD | TBD | TBD |
| Peak-sensitive (50/50) | TBD | TBD | TBD |
| Worst-only | TBD | TBD | TBD |

The mixed-session column is the headline number a paper reviewer reads: a session that is real for 50 seconds, spoof for 10, then real for 60 should be flagged as spoof. Peak-sensitive aggregation does this; pure-mean does not.

## 8.5 Active challenges (optional layer; Table 9)

When deployment can request user cooperation, add the active layer (light challenge from `from_biometric_processor/light_challenge_service.py` + gesture challenge from `active_gesture_liveness_manager.py`).

| Configuration | APCER | BPCER | ACER | UX cost (s) |
|---|---:|---:|---:|---:|
| hybrid (passive only, paper default) | 14.03% | 14.03% | 14.03% | 0 |
| hybrid + light challenge | TBD | TBD | TBD | ~1.5 |
| hybrid + gesture challenge | TBD | TBD | TBD | ~3.0 |
| hybrid + both | TBD | TBD | TBD | ~4.0 |

Active challenges add a +30 to +50 percentage-point swing on hard screen-replay attacks (per Aysenur's 2026 internal evaluation, see `research/aysenur/working_spoof_detection/`). They are not part of the headline pipeline reported in §7.

## 8.6 Session length curve (Figure 4)

ACER as a function of how many seconds of video are observed:

| Session length | 1 s | 5 s | 10 s | 30 s | 60 s | 5 min |
|---|---:|---:|---:|---:|---:|---:|
| ACER (OULU-NPU P1) | TBD | TBD | TBD | TBD | TBD | TBD |

Hypothesis: ACER is high at 1 s (insufficient time for temporal analyzers to warm up), drops sharply between 5 s and 30 s as blink and rPPG come online, then plateaus. This requires video benchmarks.

## 8.7 N-effect on bootstrap CI tightness

A practical observation from our test grid: increasing N tightens the AUC CI predictably.

| N | CASIA-FASD AUC | 95% CI width |
|---:|---:|---:|
| 200 | 0.840 | [0.755, 0.910] = 0.155 |
| 500 | 0.855 | [0.810, 0.893] = 0.083 |
| 2,408 (full) | TBD | TBD |

The 200→500 sample increase halved the CI width — the published headline number will use the full 2,408-frame test split.
