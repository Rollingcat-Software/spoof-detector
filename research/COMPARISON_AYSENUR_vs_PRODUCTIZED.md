# Aysenur's research → productized `src/` map

A side-by-side reference for paper writers and reviewers: every Aysenur (`@Aysenur15`) module that fed into the productized `src/` library, the algorithmic differences if any, and which version is the paper-grade reference.

## Why split image-track vs video-track?

The paper's central claim is that face-anti-spoofing benefits from a **hybrid image + video pipeline**. The tracks are:

- **Image track (Ahmet's contribution)** — what one frame can tell you. MiniFASNet's per-frame discrimination is enormous (+94.7 gap on our calibration set). The image track works on still photos and on video by averaging.
- **Video track (Aysenur's contribution)** — what only multiple frames in time can tell you. Real human pulse (rPPG), real eye blink (EAR), screen flicker (50/60 Hz), micro-tremor (8–12 Hz), screen-replay temporal aliasing.

Neither track is sufficient alone. The image track misses *every* AR-filter / deepfake injection that produces still-frame-plausible faces. The video track misses every print attack (no temporal signal because nothing moves).

## Module map

### Per-frame, image-level (Ahmet's track)

| `src/` module                               | Aysenur's research counterpart                                                  | Algorithmic relation                                       |
|---------------------------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------|
| `analyzers/minifasnet_analyzer.py`          | `research/aysenur/working_spoof_detection/.../uniface_liveness_detector.py`     | UniFace MiniFASNet → productized as MiniFASNetAnalyzer.    |
| `analyzers/device_boundary_analyzer.py`     | `research/aysenur/liveness_capture/.../device_boundary_detector.py`             | Same Canny+Hough idea, productized with calibrated thresh. |
| `analyzers/moire_analyzer.py`               | `research/aysenur/.../moire_pattern_analysis.py`                                | Gabor-bank + FFT, identical architecture.                   |
| `analyzers/texture_analyzer.py`             | `research/aysenur/.../texture_liveness_detector.py`                             | Laplacian + colour + FFT, identical architecture.           |
| `analyzers/ar_filter_analyzer.py`           | (no Aysenur counterpart — Ahmet original)                                        | Heuristic boundary-artifact detector; planned ONNX upgrade. |

### Multi-frame, video-level (Aysenur's track)

| `src/` module                               | Aysenur's research counterpart                                                  | Algorithmic relation                                       |
|---------------------------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------|
| `analyzers/blink_analyzer.py`               | `research/aysenur/.../enhanced_liveness_detector.py` (blink portion)            | EAR over time; Aysenur's threshold (0.21) carried forward.  |
| `analyzers/rppg_analyzer.py`                | `research/aysenur/.../rppg_analyzer.py`                                         | Verbatim port; ROADMAP P0 calls for notch-filter retune.    |
| `analyzers/screen_replay_analyzer.py`       | `research/aysenur/.../screen_replay_anti_spoof.py`                              | Verbatim algorithm; productized with per-bbox cropping.     |
| `analyzers/temporal_analyzer.py`            | `research/aysenur/.../temporal_consistency_analyzer.py`                         | Same micro-motion idea, productized as analyzer interface.  |
| `analyzers/landmark_variance_analyzer.py`   | (no Aysenur counterpart — Ahmet original)                                        | Detects "zero motion = photo" via landmark stddev.          |
| `analyzers/micro_tremor_analyzer.py`        | (no Aysenur counterpart — Ahmet original)                                        | 8–12 Hz oscillation FFT — catches video replay.             |
| `analyzers/screen_flicker_analyzer.py`      | (no Aysenur counterpart — Ahmet original)                                        | 50/60 Hz LCD/OLED flicker — catches *any* screen.           |
| `analyzers/background_grid_analyzer.py`     | (no Aysenur counterpart — Ahmet original)                                        | Per-cell scene stability — catches background swaps.        |

### Active challenges (still in research, not productized)

| File | Status |
|---|---|
| `from_biometric_processor/light_challenge_service.py` | **active** challenge (flash sequence + skin response). Wired into bio prod but disabled — flag `ANTISPOOF_CUTOUT_ENABLED=false`. Aysenur's. |
| `from_biometric_processor/flash_spoof_analyzer.py`    | **active** flash analyzer companion to `light_challenge_service`. Aysenur's. |
| `from_biometric_processor/active_liveness_*.py`        | **active** gesture / blink-on-command. Authored by Ahmet for proctoring. |

These four modules are the route to a "Phase B" paper extension — *active* challenges add a +30 to +50 percentage-point swing on screen-replay attacks that survive every passive analyzer.

## Recommended productization (next graduations)

Two Aysenur modules deserve to graduate from `research/` to `src/`:

1. **`enhanced_liveness_detector.py`** (1262 LOC) — combines blink, head pose, expression, and EAR-jitter into a single composite "liveness ProverScore" that the session engine could consume directly. Currently src/ has them as separate analyzers; combining lowers latency.
2. **`hybrid_liveness_detector.py`** (123 LOC) — Aysenur's original fusion logic that pre-dates the multi-class fuser. Could land as `src/fusion/legacy_hybrid_evaluator.py` and be referenced in the paper as the ablation baseline ("vs. the unweighted fusion").

## Calibration data parity

Per `src/infrastructure/fusion/multi_class_fuser.py`, the calibrated analyzer weights are:

| Analyzer        | Weight | Discrimination gap | Source                                   |
|-----------------|-------:|-------------------:|------------------------------------------|
| minifasnet      |    5.0 |             +94.7 | Ahmet's calibration on in-house 43 set    |
| screen_flicker  |    3.0 |     (per-attack)   | Ahmet's calibration                      |
| device_boundary |    2.5 |             +19.2 | Aysenur's bezel detection                |
| micro_tremor    |    2.5 |     (video only)   | Ahmet's contribution                     |
| landmark_variance|    2.0 |     (photo only)  | Ahmet's contribution                     |
| background_grid |    1.5 |     (proctoring)   | Ahmet's contribution                     |
| blink           |    0.5 |     (moderate)     | Aysenur's EAR threshold                  |
| screen_replay   |    0.5 |              +9.6 | Aysenur's anti-spoof, productized        |
| ar_filter       |    0.3 |     (heuristic)    | Ahmet (planned MobileNetV3-Small)        |
| temporal        |    0.3 |     (neutral)      | Aysenur, productized                     |
| texture         |    0.1 |              -6.3 | Aysenur, **anti-correlated → suppressed** |
| moire           |    0.1 |              -5.0 | Aysenur, **anti-correlated → suppressed** |
| rppg            |    0.0 |     (disabled)     | Aysenur, **needs notch-filter retune**    |

The anti-correlation finding is itself a paper contribution: in real-world data, two of the five "obvious" anti-spoof signals (texture & moire) score spoofs *higher* than reals — because high-quality replays have more texture detail than low-light real face frames. Suppressing them via 0.1 weight produced the largest single-step improvement in our calibration sweeps.

## Cross-validation matrix (for paper Table 2)

| Analyzer | Real video | Print | Screen replay | AR filter | Deepfake injection |
|---|---|---|---|---|---|
| minifasnet | high | high | high | medium | high |
| device_boundary | low (no bezel) | low (no bezel) | high | low | low |
| screen_replay | low | low | high | low | low |
| screen_flicker | low | low | high | low | low |
| micro_tremor | varies | low | high | low | medium |
| blink | varies | low (no eye motion) | varies | varies | low |
| rppg | high (after retune) | low | low | low | low |
| landmark_variance | high | low | high | high | high |
| ar_filter | low | low | low | high | low |
| background_grid | high | low (depends) | low | low | low |

This matrix is the paper's "method coverage" figure (§5).
