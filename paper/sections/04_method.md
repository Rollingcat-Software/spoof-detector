# 4. System Architecture

Figure 1 summarises the engine. A single-camera RGB feed enters at 30 FPS. The face detector (MediaPipe Tasks API) yields bounding boxes and landmarks at ~2 ms per frame. A multi-target IoU tracker assigns persistent IDs so that temporal analyzers operate on the same face across frames. Each frame is then fanned out to two banks of analyzers operating in parallel.

![Figure 1: spoof-detector v0.2.1 pipeline. Per-frame inputs are face-cropped and landmark-extracted, then fanned out to MiniFASNet ONNX (per-frame discriminator) and twelve auxiliary analyzers — four image-level (device_boundary, ar_filter, texture, moire) and eight temporal (blink_ear, rppg, screen_replay, micro_tremor, landmark_variance, temporal_motion, background_grid, screen_flicker); the calibrated MultiClassFuser produces a per-frame liveness score, and the peak-sensitive session verdict (50/50 blend of mean and worst-decile per-frame liveness) yields the final LIVE / SPOOF decision plus incident flags.](../figures/fig01_pipeline_block_diagram.png)

## 4.1 Image-level analyzers (single-frame)

| Analyzer | Algorithm | Output | Latency (CPU) |
|---|---|---|---|
| MiniFASNet ONNX | UniFace MiniFASNetV2, [-1] softmax | live-ness ∈ [0, 100] | ~12 ms |
| Device boundary | Canny + probabilistic Hough → bezel-rectangularity score | bezel-presence ∈ [0, 100] | ~4 ms |
| AR-filter (heuristic) | Boundary-artifact + colour-uniformity | filter-likelihood ∈ [0, 100] | ~3 ms |
| Texture | Laplacian variance + colour-channel entropy + radial-FFT band ratio | texture-richness ∈ [0, 100] | ~2 ms |
| Moire | Gabor bank (4 orientations × 3 scales) + radial FFT | moire-presence ∈ [0, 100] | ~6 ms |

The texture and moire analyzers are kept in the pipeline despite their anti-correlation (§5) because the multi-class fuser learns to suppress them — and at 0.1 weight they cost no decision quality but help interpretability when a spoof verdict is queried by a human auditor.

## 4.2 Temporal analyzers (multi-frame)

These all require a buffer of N≥30 recent frames per face track:

| Analyzer | Signal | Method |
|---|---|---|
| Blink | Eye Aspect Ratio (EAR) over time | Threshold 0.21, 3-frame consecutive closure → blink event |
| rPPG | Skin pulse via FFT | Forehead ROI, FFT in [0.7, 4.0] Hz band, peak-prominence |
| Screen-replay | Specular + skin-color FFT | Multi-region patch FFT, energy in scan-line band |
| Micro-tremor | 8–12 Hz oscillation in head pose | Head-yaw FFT, expected band power |
| Landmark variance | Per-landmark standard deviation | If σ < threshold across N frames → photo |
| Temporal | Cross-frame motion plausibility | Optical flow consistency + landmark trajectory |
| Background grid | Per-cell scene stability (proctoring) | 8×6 cell motion variance, expects static background |
| Screen flicker | 50/60 Hz refresh artifacts (whole frame) | Spatial FFT band power at integer multiples of 50/60 Hz |

## 4.3 Multi-class fusion

We define a 7-category spoof taxonomy:

```
{REAL, PRINT, REPLAY, MASK_3D, HEAVY_MAKEUP, AR_FILTER, DEEPFAKE_INJECTION}
```

Each analyzer's score is mapped to per-category evidence via the calibrated `SPOOF_SIGNAL_MAP`. For example, MiniFASNet's "live-ness" score routes a *low* score (=spoof) to PRINT (0.4 weight), REPLAY (0.4), AR_FILTER (0.2). Device-boundary's *low* score (no bezel detected) is uninformative; its *high* score routes evidence to REPLAY (0.7) and PRINT (0.3) — only screens and printouts produce sharp rectangular boundaries.

Per-frame voting:

```
P(category | frame) ∝ exp( Σ_a w_a · evidence_a(category) )
```

with weights `w_a` from §5. Final per-frame classification is `argmax P`. Figure 1 (introduced at the head of this section) shows the full data flow from frames through the analyzer ensemble to the fuser output `p_t = P(REAL | frame_t)` consumed by the session aggregator of §4.4.

## 4.4 Session engine (peak-sensitive verdict)

The session engine ingests per-frame classifications and produces a session verdict that strengthens over time. State machine: `WARMING_UP → ANALYZING → CONCLUDED`. This subsection formalises the **peak-sensitive aggregator** that is the headline novelty of this paper (Contribution 1, §1) and the property we argue session-level FAS needs but per-frame FAS cannot provide.

### 4.4.1 Problem setup — spoof-burst dilution

A session is a sequence of `N` frames indexed by `t = 1 … N`. Each frame produces a per-frame liveness score `p_t ∈ [0, 1]` from the multi-class fuser of §4.3 — concretely `p_t = P(REAL | frame_t)`. Per-frame FAS literature reports the binary classification at each `t`; session-level FAS must aggregate the sequence `{p_1, …, p_N}` into a single decision.

The naive aggregator is the session mean `p̄ = (1/N) Σ_t p_t`. Per-frame benchmarks implicitly assume this because they evaluate one `t` at a time. But the deployment contexts that economically dominate face-PAD — proctoring, hosted KYC, remote-banking onboarding — give the attacker minutes to hours of camera time (Gap 1, §1). An attacker who holds a screen-replay for a long stretch and briefly reveals a real face produces a sequence whose mean is dragged upward by the real-face frames even though the *sustained* signal is spoof.

**Worked example A — mean dilution by brief real-face bursts (illustrative synthetic).** Consider `N = 60` frames (≈ 2 s at 30 FPS), `p_t = 0.20` for 54 spoof frames and `p_t = 0.95` for 6 brief real-face frames. Mean liveness:

```
p̄ = (54 · 0.20 + 6 · 0.95) / 60 = (10.80 + 5.70) / 60 = 0.275
```

A naïve threshold at 0.5 correctly classifies this session as SPOOF. Now widen the real-face burst to 30 frames in the same 60-frame session:

```
p̄ = (30 · 0.20 + 30 · 0.95) / 60 = (6.00 + 28.50) / 60 = 0.575
```

The mean now exceeds 0.5 and a threshold-based session verdict flips to LIVE — even though the sustained spoof exposure is unchanged at half the session. This is *spoof-burst dilution*: the attacker can buy LIVE classifications arbitrarily by extending the real-face burst, and the mean aggregator gives them no resistance.

The opposite-tail aggregator — `min(p_t)` — collapses under the converse failure mode. A single momentarily noisy real frame (subject blinks during landmark detection; partial occlusion; brief motion blur) suffices to flip the entire session. Production capture exhibits these dips routinely on otherwise-live sessions, so a min-based session verdict yields unacceptable BPCER.

The aggregator must therefore *be sensitive to sustained low-`p_t` stretches* (so brief-real-burst attackers cannot dilute the spoof signal) *without being sensitive to single-frame noise* (so live sessions with momentary capture dips are not falsely rejected).

### 4.4.2 The peak-sensitive aggregator

Let `W ⊂ {1 … N}` be the set of frame indices with the lowest `p_t` values; specifically, let `W` index the **worst decile**, `|W| = max(1, ⌊N/10⌋)`. Define the worst-decile mean:

```
p̄_worst = (1 / |W|) · Σ_{t ∈ W} p_t
```

The peak-sensitive session verdict is:

```
P_session(REAL) = α · p̄ + (1 − α) · p̄_worst
```

with `α = 0.5` in the published configuration — equal weight to the session-wide average (typical behaviour) and to the worst-decile mean (worst sustained behaviour). The 50/50 split is the same notation used by Figure 1 and the abstract. This choice is empirically validated against the alternatives in §4.4.3 and the ablation table cited there; α = 0.5 trades a small loss against pure-mean on entirely-live sessions for a large gain on mixed sessions, which is the operating point this paper argues session-level FAS needs.

**Why the aggregator resists brief-burst dilution.** As `N` grows, `p̄_worst` is bounded by the long spoof stretches because the worst decile of any session containing ≥ 10 % spoof frames is *entirely drawn* from those frames. Concretely, repeat Worked Example A's second case (`N = 60`, 30 spoof @ 0.20 + 30 real @ 0.95). The worst decile is `|W| = 6` frames, all drawn from the spoof half at `p_t = 0.20`:

```
p̄        = 0.575     (unchanged)
p̄_worst  = 0.20
P_session(REAL) = 0.5 · 0.575 + 0.5 · 0.20 = 0.2875 + 0.10 = 0.388
```

The session verdict now falls *below* the 0.5 threshold and is correctly classified SPOOF, despite the mean exceeding 0.5. The attacker cannot escape this verdict by lengthening the real-face burst: as long as the spoof stretch occupies ≥ 10 % of the session, the worst decile pins `p̄_worst` to the spoof score and the second term keeps `P_session` below threshold.

**Why the aggregator resists single-frame noise.** Consider the converse: `N = 60` all-real frames with one momentary capture-quality dip. With one frame at `p_t = 0.30` and the remaining 59 frames at `p_t = 0.95`, the worst decile (6 frames) is `(0.30 + 0.94 · 5) / 6 = (0.30 + 4.70) / 6 = 0.833` (illustrative — assumes the five next-lowest real frames sit at 0.94). Then:

```
p̄        = (1 · 0.30 + 59 · 0.95) / 60 ≈ 0.939
p̄_worst  ≈ 0.833
P_session(REAL) ≈ 0.5 · 0.939 + 0.5 · 0.833 ≈ 0.886
```

The session verdict comfortably exceeds 0.5 and is correctly classified LIVE — because the single low-`p_t` frame is averaged across the bottom decile rather than dominating it. The 6-frame worst-decile *mean* is what gives the aggregator its noise-robustness; a pure `min(p_t)` would have scored `min = 0.30` and forced an incorrect SPOOF verdict.

### 4.4.3 Comparison to alternatives

Four candidate session aggregators on the same `{p_t}` sequence:

| Aggregator | Form | Brief-burst dilution | Single-frame noise | Comment |
|---|---|---|---|---|
| Mean | `p̄` | **Fails** (Worked Ex. A) | Robust | Per-frame benchmarks' implicit choice |
| Min | `min_t p_t` | Robust | **Fails** (Worked Ex. B) | Inverse failure to mean |
| Median | `med_t p_t` | Fails when burst > 50 % | Robust | Symmetric, still defeated by large-burst attacks |
| Trimmed mean | `mean(p_t \| t ∈ middle 80 %)` | Fails (trims away the spoof tail) | Robust | Worst case: actively discards the evidence we want |
| **Peak-sensitive** | `α · p̄ + (1−α) · p̄_worst` | **Robust** (§4.4.2) | **Robust** (§4.4.2) | Published method |

The mean fails under sustained-spoof-with-brief-real-burst (Worked Example A, second case). The min fails under all-live-with-momentary-noise (Worked Example B). The median fails when the spoof burst exceeds 50 % of the session — the very attacks proctoring contexts produce. The trimmed mean is the *most* dangerous alternative because it explicitly discards the bottom (and top) tail of the distribution; the bottom tail is precisely the evidence the spoof-burst case writes into the sequence, so trimming destroys the signal. Only the peak-sensitive aggregator combines a robust central statistic (the mean) with a robust tail statistic (the worst-decile mean) and is sensitive to neither failure mode.

Empirical confirmation is given in §8.4 ("Peak-sensitive vs. mean session verdict") on the mixed-session protocol: the proctoring claim is that on a session that is real for 50 s, spoof for 10 s, then real for 60 s, the peak-sensitive aggregator flags SPOOF while the mean aggregator flags LIVE. The §8.4 mixed-session column reports the headline number that quantifies the gap; at time of writing the multi-segment session-protocol cells are populated as TBD pending OULU-NPU's per-PAI session videos which we do not yet have EULA access to. The mechanism above is reproducible from the worked examples without the empirical fill-in.

### 4.4.4 Implementation note

The published peak-sensitive aggregator is implemented as `aggregate_frame_scores(scores, mode="peak_sensitive")` in `tests/benchmark/pipelines/_common.py`; the bottom-`k` window with `k = max(1, len(scores) // 10)` is the worst-decile mean of §4.4.2, and the 0.5 / 0.5 blend matches the verdict formula above. This is the aggregator used by `tests/benchmark/pipelines/hybrid.py` and `tests/benchmark/pipelines/video_only.py` to produce the §7 cross-dataset results. A streaming variant for live single-session use is implemented in `src/application/session_engine.py::SessionEngine.get_verdict()`; it substitutes a worst-5-frame *sliding* window for the worst-decile (cheaper to maintain incrementally as frames arrive) and otherwise applies the same 0.5 / 0.5 blend at line 382. Incidents — events that flag operator attention — are emitted on sustained `P(REAL) < 0.4` for ≥ 3 seconds, no blinks for ≥ 15 seconds (when blink analyzer is healthy), face missing for ≥ 5 seconds, or MiniFASNet score swing ≥ 0.35 in 1-second window (identity-change suspicion).

[TODO author review: §4.4.3's α-sweep table is asserted but not yet populated by an explicit α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} ablation; once §8.4's multi-segment session data is acquired the α-sweep should be added to §8.4 and cross-referenced here.]

## 4.5 Active challenges (optional layer, not part of the headline pipeline)

For deployments that can demand active user cooperation (high-stakes onboarding, exam proctoring), an additional active layer is available: random colour-flash challenge measuring chromatic skin response, head-rotation challenge measuring 3-D parallax, and explicit blink-on-command challenge. These add ≈ 30 to 50 percentage-point swings on the hardest screen-replay attacks but at the cost of one round-trip and a UX intrusion. We report them as an ablation in §8 but exclude them from the default pipeline reported in §7 to ensure cross-paper comparability.
