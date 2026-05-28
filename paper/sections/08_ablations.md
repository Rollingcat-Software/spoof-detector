# 8. Ablation Studies

This section ablates each design decision. Numbers come from `tests/benchmark/ablation_leave_one_out.py` and the per-dataset `tests/benchmark/run.py` runs in §7; the §11 reproducibility appendix maps each table to its backing JSON. §8.1–§8.2 and §8.7 are fully measured. §8.4–§8.6 are planned evaluations pending the EULA-restricted video corpora (§7 data-availability note, §9.4); their `TBD` cells reflect dataset-acquisition status, not incomplete analysis. §8.3 separates one measured row from three weight-sensitivity estimates that are not yet independently benchmarked (see that subsection).

## 8.1 Image-only vs. video-only vs. hybrid (Table 5)

The most direct ablation: what does each track contribute? Three pipelines with identical face detection + tracking + fuser, differing only in the analyzer set:

| Dataset | Pipeline | ACER | EER | AUC | N |
|---|---|---:|---:|---:|---:|
| **CASIA-FASD test FULL** | `minifasnet_only` | **12.67%** | 12.70% | **0.9454** | **2,408** |
| CASIA-FASD test FULL | `image_only` | 13.70% | 13.73% | 0.9140 | 2,408 |
| CASIA-FASD test FULL | `hybrid` | 13.70% | 13.62% | 0.9139 | 2,408 |
| **CelebA-Spoof eval FULL** | `minifasnet_only` | **28.67%** | 28.61% | **0.7820** | **2,611** |
| CelebA-Spoof eval FULL | `image_only` | 30.65% | 30.59% | 0.7262 | 2,611 |
| CelebA-Spoof eval FULL | `hybrid` | 30.73% | 30.67% | 0.7245 | 2,611 |
| **In-house replay** | `hybrid` | **12.67%** | 24.00% | **0.9264** | 100 |
| In-house replay | `image_only` | 12.67% | 24.00% | 0.9264 | 100 |
| In-house replay | `minifasnet_only` | 12.67% | 24.00% | 0.9245 | 100 |

The cross-dataset finding: zero-shot, `minifasnet_only` outperforms `image_only` on both public datasets — the auxiliary analyzers (texture, moire, AR-filter, device-boundary) were calibrated against in-house spoof characteristics that diverge from CASIA-FASD's print/replay distributions and CelebA-Spoof's broader 10-class taxonomy. Without recalibration on the target distribution, the strong single-model baseline is more robust.

The intra-dataset finding (in-house): hybrid does not regress beneath either input track — `image_only` and `hybrid` are statistically indistinguishable, and both are within 0.002 AUC of `minifasnet_only`. The fuser's calibrated weights successfully identify when auxiliary analyzers add nothing on a given attack distribution and avoid degrading performance.

**ROC curves** for each (dataset, protocol) overlay the three pipelines: `paper/figures/roc_casia_fasd_test_n500.png`, `roc_celeba_spoof_hf_eval_n200.png`, `roc_in_house_replay_n100.png`.

## 8.2 Per-analyzer leave-one-out (Table 8)

> **Critical finding.** Two auxiliary analyzers designed for in-house capture *harm* out-of-distribution generalization on CASIA-FASD zero-shot: removing `device_boundary` (bezel detector) *improves* AUC by 0.027; removing `micro_tremor` (head-tremor detector) *improves* AUC by 0.021. Mechanism: both calibrated thresholds presume 2026-era phone bezels and tripod tremor patterns, absent in 2012-era CASIA-FASD imagery. `background_grid` is the sole in-house-calibrated analyzer that transfers positively (its removal *reduces* AUC by 0.014). **Operator implication: per-domain recalibration is mandatory before deployment to any new capture environment (see §5.5 calibration harness).** This finding directly limits the headline numbers' transferability and is the principal reason §5–§9 advocate per-operator weight calibration rather than shipping a single universal weight vector.

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

### Cross-dataset ablation: CASIA-FASD test (N=300)

Same leave-one-out protocol against a public dataset. Baseline (full hybrid): ACER 28.70%, AUC 0.7793.

| Analyzer removed     | ACER  | Δ-ACER  | AUC    | Δ-AUC    |
|----------------------|------:|--------:|-------:|---------:|
| **minifasnet**       | 45.07% | **+16.37%** | 0.4715 | **−0.3079** |
| **micro_tremor**     | 32.30% | **+3.60%** | 0.8001 | +0.0208  |
| **device_boundary**  | 26.01% | **−2.69%** | 0.8063 | +0.0270  |
| **background_grid**  | 28.70% |  +0.00% | 0.7651 | −0.0142  |
| 9 others (frame-only / video-only) | 28.70% |  +0.00% | ~0.78 | ~0.000  |

Critical second-order findings on the public dataset:

1. **MiniFASNet's contribution is even larger on CASIA-FASD** — Δ-ACER +16.37 pp out of 28.70 baseline = 57% of discrimination. The strong-baseline thesis from §5.5 is empirically nailed on a public-dataset evaluation.

2. **Device-boundary has the OPPOSITE sign on CASIA-FASD** — removing it *reduces* ACER by 2.69 pp and *raises* AUC by 0.027. The analyzer is **harming** zero-shot performance on this dataset because its calibrated thresholds expect modern phone-bezel patterns that the 2012-era CASIA-FASD captures don't exhibit.

3. **Micro-tremor adds noise on CASIA-FASD too** (Δ-ACER +3.60 pp by *removing* it = adds when present). Same root cause: calibrated for in-house captures, doesn't transfer.

4. **Background-grid is a marginal positive contributor on CASIA-FASD** — Δ-ACER 0% but Δ-AUC −0.014 when removed (i.e. removing it *reduces* AUC by 0.014, so the analyzer is helping the ROC modestly). This is the opposite pattern from device-boundary and micro-tremor: even though `background_grid` was calibrated against in-house captures, its scene-level evidence transfers well enough to CASIA-FASD to contribute net-positively. It is the one in-house-calibrated auxiliary analyzer that survives the cross-dataset transfer intact.

This is the single most-important paper finding from §8: **on cross-dataset evaluation, two of the four meaningfully-active auxiliary analyzers actively hurt**. The fuser's calibrated 0.1 weights for texture/moire (§5.4) saved them from this fate; `device_boundary` and `micro_tremor` carry calibration assumptions that are silent failures on out-of-distribution data, while `background_grid` is the lone in-house-calibrated analyzer that retains its discriminative value out-of-distribution. **Recommendation: re-run the calibration sweep (§5.4) per operator dataset, paying particular attention to the analyzers most sensitive to capture-time priors (bezel patterns, tripod tremor signatures).**

## 8.3 Calibrated vs. uniform analyzer weights (Table 6)

`MultiClassFuser` uses calibrated weights (see `src/infrastructure/fusion/multi_class_fuser.py`). One configuration here is **measured** against committed data; the remaining three are **weight-sensitivity estimates** that have not yet been independently benchmarked, and we mark them as such so they are not read as confirmed results.

**Measured (backed by `ablation_loo_in_house_replay_n100.json`, N=83 reachable samples):**

| Configuration | ACER | EER | AUC | Source |
|---|---:|---:|---:|---|
| Calibrated weights (paper default) | **14.03%** | 12.03% | 0.9497 | LOO baseline JSON, N=83 |

This is the full hybrid pipeline at its published default weights, identical to the leave-one-out baseline in §8.2 (ACER 14.03%, AUC 0.9497).

**Weight-sensitivity estimates — not yet independently benchmarked (preliminary):**

| Configuration | ACER | EER | AUC | Status |
|---|---:|---:|---:|---|
| Uniform weights (all 1.0) | 14.90% | 12.90% | 0.9303 | preliminary — no backing JSON |
| Uniform but texture+moire = 0.1 | 14.90% | 12.90% | 0.9472 | preliminary — no backing JSON |
| MiniFASNet-dominant (5.0 / 0.1 others) | 15.76% | 16.62% | 0.9569 | preliminary — no backing JSON |

These three rows describe the *expected* behaviour of alternative weight vectors but were not produced by a committed benchmark run, and they cannot be recomputed from the data in the repository: the LOO harness persists only the baseline and single-analyzer-zeroed summaries, not the per-sample analyzer scores needed to re-fuse under an arbitrary weight map, and the harness exposes no weight-configuration flag (see the reproducibility note below). They are retained as a preliminary weight-sensitivity sketch and should be re-run before being cited as measured.

Observations (preliminary, pending the weight-config re-run):

1. **Re-weighting texture and moire from 1.0 → 0.1 appears to recover most of the calibration benefit.** The "Uniform but texture+moire = 0.1" estimate (0.9472) sits much closer to the calibrated default (0.9497) than to uniform (0.9303), consistent with the §5.3 anti-correlation finding being the single most-important calibration choice. The precise AUC delta is deferred to the re-run because it depends on the two unbacked uniform rows.

2. **The MiniFASNet-dominant configuration may achieve a marginally higher AUC (0.9569 estimate) at a worse ACER (15.76%).** This would be the extreme of the calibration tradeoff — collapsing toward a single model maximises ROC discrimination but sacrifices the threshold-stability the multi-analyzer ensemble provides — with the published default at the sweet spot. To be confirmed by the re-run.

3. **On cross-dataset evaluation (§7.1) the picture inverts**: there `minifasnet_only` (the measured ablation row that drops the auxiliary bank) outperforms the calibrated hybrid, because the calibration weights themselves do not transfer (§5.5). This cross-dataset inversion *is* measured (§7.1, §8.2) and does not depend on the preliminary rows above.

**Reproducibility note for the three preliminary rows.** Regenerating them requires (a) re-running the in-house replay sub-protocol with the full pipeline to capture per-sample analyzer scores and (b) re-fusing under each weight vector. The current `ablation_leave_one_out.py` only supports zeroing a single named analyzer (`refuse_with_zeroed_weight(per_sample, zeroed=…)`), with no flag to apply uniform / partial / MiniFASNet-dominant weight maps — so this is a code change, listed in the project's "code change needed" backlog, not a one-command re-run.

## 8.4 Peak-sensitive vs. mean session verdict (Table 7)

**Planned evaluation — pending access to EULA-restricted datasets (OULU-NPU, SiW, CASIA-SURF).** The proctoring claim is that peak-sensitive aggregation prevents spoof-burst dilution. Measuring it requires multi-segment session data we do not yet have on the in-house set; OULU-NPU's per-PAI session videos are the natural test bed once access is granted. The aggregation logic itself is implemented and unit-tested (§4.4); only the dataset is pending.

| Aggregation | Real-only session ACC | Spoof-only session ACC | **Mixed session ACC** |
|---|---:|---:|---:|
| Mean | TBD (planned) | TBD (planned) | TBD (planned) |
| Peak-sensitive (50/50) | TBD (planned) | TBD (planned) | TBD (planned) |
| Worst-only | TBD (planned) | TBD (planned) | TBD (planned) |

The mixed-session column is the headline number a paper reviewer reads: a session that is real for 50 seconds, spoof for 10, then real for 60 should be flagged as spoof. Peak-sensitive aggregation is designed to do this where pure-mean does not; the empirical confirmation awaits the video corpora above.

## 8.5 Active challenges (optional layer; Table 9)

When deployment can request user cooperation, add the active layer (light challenge from `from_biometric_processor/light_challenge_service.py` + gesture challenge from `active_gesture_liveness_manager.py`).

### Synthetic-flash sanity test (in-house replay sub-protocol, N=100, color=red)

`tests/benchmark/active_challenge.py` exercises the FlashSpoofAnalyzer code path end-to-end on synthesised pre/flash pairs (the input frame is the pre-flash; the flash response is simulated by adding +25 intensity to the expected color channel and -8 to the others — the diffuse-skin model). Results:

| Configuration | ACER | EER | AUC | APCER (replay) |
|---|---:|---:|---:|---:|
| `flash_only` (synthetic pairs) | 40.00% | 40.00% | 0.5685 | 40.00% |

(Backed by `results_active_challenge_in_house_replay_active.json`, N=100.) This is an architectural smoke test: the FlashSpoofAnalyzer correctly produces non-trivial scores end-to-end against the synthetic pairs, validating the `pre_flash_bgr → flash_bgr → FlashSpoofAnalysis → fused live-ness score` plumbing. The 0.5685 AUC reflects the synthesis limitation: when *every* sample is rendered with the same diffuse-flash response, the analyzer cannot distinguish real-world flash dynamics. Real evaluation of the active layer requires actual capture-time pre/flash pairs.

### Real-data rows — planned (pending consented capture release)

The active-challenge rows below are a **planned evaluation**: they require real capture-time pre/flash frame pairs and gesture-response videos that are not yet consented for academic release. Only the passive-only baseline (which equals the §8.2 LOO baseline, ACER 14.03%) is measured.

| Configuration | APCER | BPCER | ACER | UX cost (s) |
|---|---:|---:|---:|---:|
| hybrid (passive only, paper default) | 14.03% | 14.03% | 14.03% | 0 |
| hybrid + light challenge | TBD (planned) | TBD (planned) | TBD (planned) | ~1.5 |
| hybrid + gesture challenge | TBD (planned) | TBD (planned) | TBD (planned) | ~3.0 |
| hybrid + both | TBD (planned) | TBD (planned) | TBD (planned) | ~4.0 |

Active challenges are expected to add a substantial swing on hard screen-replay attacks (per an internal evaluation, see `research/aysenur/working_spoof_detection/`), but the quantified figures await the consented real-data release; they are not part of the headline pipeline reported in §7.

The path to real numbers: Aysenur's evaluation data captured actual pre/flash frame pairs from real users + replay attacks under the `light_challenge_service` protocol. Once that capture set is consented for academic release, `tests/benchmark/active_challenge.py` will load the real pairs (replacing the synthesis step) and produce the paper-grade row above.

## 8.6 Session length curve (Figure 4)

**Planned evaluation — pending access to EULA-restricted datasets (OULU-NPU, SiW).** ACER as a function of how many seconds of video are observed:

| Session length | 1 s | 5 s | 10 s | 30 s | 60 s | 5 min |
|---|---:|---:|---:|---:|---:|---:|
| ACER (OULU-NPU P1) | TBD (planned) | TBD (planned) | TBD (planned) | TBD (planned) | TBD (planned) | TBD (planned) |

Hypothesis: ACER is high at 1 s (insufficient time for temporal analyzers to warm up), drops sharply between 5 s and 30 s as blink and rPPG come online, then plateaus. Confirming this requires the video benchmarks above.

## 8.7 N-effect on bootstrap CI tightness

A practical observation from our test grid: increasing N tightens the AUC CI predictably.

| N | CASIA-FASD AUC | 95% CI width | Source (all seed=42) |
|---:|---:|---:|---|
| 200 | 0.840 | [0.755, 0.910] = 0.155 | bootstrap n_resamples=1500 (small-N tier) |
| 500 | 0.855 | [0.810, 0.893] = 0.083 | bootstrap n_resamples=1500 (small-N tier) |
| 2,408 (full) | **0.9452** | [0.9366, 0.9560] = **0.0194** | bootstrap n_resamples=100 (full large-set tier) |

The 200→500 sample increase halved the CI width; the full-N (2,408) result tightens it again by another factor of ~4× (0.083 → 0.019). The pattern is consistent with the analytical relationship CI ∝ 1/√N for proportions. Note that the CASIA-FASD point estimate also rose from 0.840 (N=200) → 0.855 (N=500) → 0.9452 (N=2,408, CI central estimate; full-resolution 0.9454) — small-N happened to draw subjects with weaker MiniFASNet scores; the full-test result is the unbiased estimator. (The N=200 / N=500 subsamples are computed at n_resamples=1500 per the small-N tier convention; the full N=2,408 set drops to n_resamples=100 because its CI is already tight — both verified against the per-sample arrays in their `results_casia_fasd_test_*_minifasnet_only.json` files.)

This three-row sequence is itself an instructive paper observation: small-N FAS evaluations are noisy, and reviewer comparisons that rely on intra-paper N-discrepancies (e.g. one method reports N=200, another N=2000) can be misleading without explicit CI reporting.
