# 4. System Architecture

Figure 1 summarises the engine. A single-camera RGB feed enters at 30 FPS. The face detector (MediaPipe Tasks API) yields bounding boxes and landmarks at ~2 ms per frame. A multi-target IoU tracker assigns persistent IDs so that temporal analyzers operate on the same face across frames. Each frame is then fanned out to two banks of analyzers operating in parallel.

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

with weights `w_a` from §5. Final per-frame classification is `argmax P`.

## 4.4 Session engine (peak-sensitive verdict)

The session engine ingests per-frame classifications and produces a session verdict that strengthens over time. State machine: `WARMING_UP → ANALYZING → CONCLUDED`. The published verdict aggregator is:

```
P_session(REAL) = 0.5 · mean(P_frame(REAL) | t > t_warmup)
                 + 0.5 · mean(P_frame(REAL) | t ∈ worst-decile-window)
```

The 50/50 blend is the key proctoring property: an attacker who flashes their real face for one second in a sixty-second session does not get a session verdict averaged near "real". The worst-decile window dominates the second term and pulls the session verdict down to the low live-ness of the spoof minutes.

Incidents — events that flag operator attention — are emitted on:

- Sustained `P(REAL) < 0.4` for ≥ 3 seconds.
- No blinks for ≥ 15 seconds (when blink analyzer is healthy).
- Face missing for ≥ 5 seconds.
- MiniFASNet score swing ≥ 0.35 in 1-second window (identity-change suspicion).

## 4.5 Active challenges (optional layer, not part of the headline pipeline)

For deployments that can demand active user cooperation (high-stakes onboarding, exam proctoring), an additional active layer is available: random colour-flash challenge measuring chromatic skin response, head-rotation challenge measuring 3-D parallax, and explicit blink-on-command challenge. These add ≈ 30 to 50 percentage-point swings on the hardest screen-replay attacks but at the cost of one round-trip and a UX intrusion. We report them as an ablation in §8 but exclude them from the default pipeline reported in §7 to ensure cross-paper comparability.
