# Hybrid Image+Video Spoof-Detection Architecture

This document is the formal, code-referenced description of the system architecture. It is the spec the paper's §4 condenses, and is the reference document for anyone wiring a new analyzer or re-running the calibration.

## Design thesis

A robust face presentation-attack detection (PAD) engine needs *both* image-level and video-level signals because:

- **Image-level signals** (single frame) are fast, predictive on the easy attacks (print, low-quality replay), and produce huge per-frame discrimination gaps when a strong CNN like MiniFASNet is used. They fail on every attack that produces a single-frame-plausible face: AR filters, deepfake injection, high-quality phone replays.
- **Video-level signals** (multi-frame) catch what image methods miss: real human pulse (rPPG), real eye blinks (EAR), screen flicker (50/60 Hz), micro-tremor (8–12 Hz). They fail on print attacks, where there is no temporal signal to read.

The two are structurally complementary; one cannot substitute the other.

## Authorship breakdown

The hybrid pipeline draws on two contributor tracks within the FIVUCSAS programme:

- **Image track** (Ahmet Abdullah Gultekin) — strongly per-frame. MiniFASNet integration, calibrated multi-class fuser, session engine, peak-sensitive verdict, landmark-variance, micro-tremor, screen-flicker, background-grid, AR-filter heuristic.
- **Video track** (Aysenur Akar @Aysenur15, with co-authoring on rPPG and screen-replay defence by Ayşe Gülsüm Eren) — strongly multi-frame. rPPG, blink (EAR), screen-replay anti-spoof, temporal consistency, hybrid liveness detector, MRZ pipeline, flash-challenge service.

See `research/COMPARISON_AYSENUR_vs_PRODUCTIZED.md` for the per-module map between Aysenur's research branches and the productized `src/` library.

## Component graph

```
                              [ Camera ]
                                  │ 30 FPS, RGB
                                  ▼
                       [ MediaPipe Face Detector ] ── 478-pt landmarks
                                  │ ~2 ms / frame
                                  ▼
                          [ IoU Multi-Tracker ]
                                  │ persistent face IDs
                                  ▼
              ┌───────────────────┴───────────────────┐
              │                                       │
              ▼                                       ▼
    [ IMAGE-LEVEL ANALYZERS ]              [ VIDEO-LEVEL ANALYZERS ]
    ┌───────────────────────┐              ┌─────────────────────────────┐
    │ MiniFASNet ONNX       │              │ Blink (EAR over time)       │
    │ Device Boundary       │              │ rPPG (skin-pulse FFT)       │
    │ Texture / Moire       │              │ Screen Replay (specular+FFT)│
    │ AR Filter             │              │ Micro-Tremor (8-12 Hz FFT)  │
    │ (single-frame outputs)│              │ Landmark Variance (σ over t)│
    └───────────────────────┘              │ Temporal Consistency        │
                              (Ahmet's)    │ Background Grid             │
                                           │ Screen Flicker (whole frame)│
                                           └─────────────────────────────┘
                                                   (Aysenur's, plus
                                                   Ahmet's flicker/grid)
              │                                       │
              └───────────────────┬───────────────────┘
                                  ▼
                  [ Calibrated Multi-Class Fuser ]
                       (7-category taxonomy)
                                  │
                                  ▼
              [ Session Engine — peak-sensitive verdict ]
                                  │
                                  ▼
                          ┌───────┴────────┐
                          │   Verdict +    │
                          │   Incidents    │
                          └────────────────┘
```

## Layer 1 — face detection & tracking

`src/infrastructure/detection/mediapipe_detector.py` wraps Google's MediaPipe Tasks API at ~2 ms per frame. Output: bounding boxes + 478 landmarks per face. `src/application/face_tracker.py` assigns persistent face IDs using IoU matching (threshold 0.3) so all temporal analyzers can buffer per-track signals.

## Layer 2 — image-level analyzers (Ahmet's track)

Each analyzer is a class implementing `IFaceAnalyzer.analyze(crop, face) -> AnalyzerResult`.

| Analyzer | Source | Discrimination gap (calibration set) |
|---|---|---|
| `MiniFASNetAnalyzer` | `src/infrastructure/analyzers/minifasnet_analyzer.py` | +94.7 |
| `DeviceBoundaryAnalyzer` | `src/infrastructure/analyzers/device_boundary_analyzer.py` | +19.2 |
| `ARFilterAnalyzer` | `src/infrastructure/analyzers/ar_filter_analyzer.py` | (heuristic) |
| `TextureAnalyzer` | `src/infrastructure/analyzers/texture_analyzer.py` | **−6.3** (anti-correlated) |
| `MoireAnalyzer` | `src/infrastructure/analyzers/moire_analyzer.py` | **−5.0** (anti-correlated) |

The anti-correlation finding (texture and moire score *spoofs* higher than reals on real-world data) is one of the paper's three contributions. The fuser handles it by re-weighting these analyzers to 0.1 — they remain in the pipeline for interpretability but contribute negligibly to the decision.

## Layer 3 — video-level analyzers (Aysenur's track + Ahmet's temporals)

Each requires a buffer of N≥30 frames per track:

| Analyzer | Source | Aysenur counterpart |
|---|---|---|
| `BlinkAnalyzer` (EAR) | `src/infrastructure/analyzers/blink_analyzer.py` | research/aysenur/.../enhanced_liveness_detector.py |
| `RPPGAnalyzer` | `src/infrastructure/analyzers/rppg_analyzer.py` | research/aysenur/.../rppg_analyzer.py |
| `ScreenReplayAnalyzer` | `src/infrastructure/analyzers/screen_replay_analyzer.py` | research/aysenur/.../screen_replay_anti_spoof.py |
| `TemporalAnalyzer` | `src/infrastructure/analyzers/temporal_analyzer.py` | research/aysenur/.../temporal_consistency_analyzer.py |
| `MicroTremorAnalyzer` | `src/infrastructure/analyzers/micro_tremor_analyzer.py` | (Ahmet — no Aysenur counterpart) |
| `LandmarkVarianceAnalyzer` | `src/infrastructure/analyzers/landmark_variance_analyzer.py` | (Ahmet) |
| `ScreenFlickerAnalyzer` | `src/infrastructure/analyzers/screen_flicker_analyzer.py` | (Ahmet — frame-level, not face-level) |
| `BackgroundGridAnalyzer` | `src/infrastructure/analyzers/background_grid_analyzer.py` | (Ahmet — proctoring-specific) |

## Layer 4 — calibrated multi-class fusion

`src/infrastructure/fusion/multi_class_fuser.py` maps analyzer outputs to a probability distribution over the 7-category spoof taxonomy via the calibrated weight table:

```python
DEFAULT_ANALYZER_WEIGHTS = {
    "minifasnet":        5.0,   # +94.7 gap — dominant signal
    "screen_flicker":    3.0,   # 50/60 Hz catches any screen
    "device_boundary":   2.5,   # +19.2 gap — bezel detection
    "micro_tremor":      2.5,   # 8-12 Hz catches video replay
    "landmark_variance": 2.0,   # 0 variance = photo
    "background_grid":   1.5,   # proctoring scene stability
    "rppg":              0.0,   # disabled — needs notch-filter retune
    "blink":             0.5,   # moderate
    "screen_replay":     0.5,   # +9.6 gap — weak alone
    "ar_filter":         0.3,   # heuristic mode
    "temporal":          0.3,   # neutral
    "texture":           0.1,   # ANTI-CORRELATED — suppressed
    "moire":             0.1,   # ANTI-CORRELATED — suppressed
}
```

The 7-category taxonomy is `{REAL, PRINT, REPLAY, MASK_3D, HEAVY_MAKEUP, AR_FILTER, DEEPFAKE_INJECTION}`. Each analyzer score routes evidence into category-specific bins via `src/domain/taxonomy.py:SPOOF_SIGNAL_MAP`.

## Layer 5 — session engine

`src/application/session_engine.py` accumulates per-frame classifications into a session-level verdict. State machine:

```
WARMING_UP → ANALYZING → CONCLUDED
```

The session verdict aggregator (the published "peak-sensitive" rule):

```
P_session(REAL) = 0.5 · mean(P_frame(REAL))
                 + 0.5 · mean(P_frame(REAL) | t ∈ worst-decile-window)
```

This is the property that makes the engine deployment-ready for proctoring: a brief spoof burst inside an otherwise-real session does not get averaged away.

## Operational thresholds

These are tunable via constructor or environment, but the published defaults are:

| Threshold | Default | Rationale |
|---|---|---|
| WARMUP_FRAMES | 30 (1 s) | Temporal analyzers need history before contributing |
| MIN_VERDICT_FRAMES | 60 (2 s) | Below this we report `WARMING_UP`, no verdict |
| BLINK_EAR_THRESHOLD | 0.21 | Aysenur's calibrated value; literature range 0.20–0.25 |
| BLINK_CONSECUTIVE | 3 frames | Eye must be closed ≥ 100 ms to count as blink |
| NO_BLINK_ALERT_SEC | 15.0 | Real human blinks once every 4–6 seconds; ≥ 15 s without blink is suspicious |
| FACE_MISSING_ALERT_SEC | 5.0 | Brief face-occlusion is normal; ≥ 5 s is incident-worthy |
| IDENTITY_CHANGE_THRESHOLD | 0.35 | MiniFASNet score swing in 1 s indicating possible identity change |

## Active-challenge layer (optional, deployment-time)

Active challenges add a +30 to +50 percentage-point swing on hard screen-replay attacks but at the cost of one round-trip and a UX intrusion. They are *not* part of the headline pipeline and are reported as a separate ablation in the paper (§8.5).

The four active-challenge modules are mirrored in `from_biometric_processor/`:

- `light_challenge_service.py` (Aysenur — random-colour flash + skin response)
- `flash_spoof_analyzer.py` (Aysenur — companion analyzer)
- `active_liveness_manager.py` (Ahmet — orchestration)
- `active_gesture_liveness_manager.py` (Ahmet — head rotation, blink-on-command)

## Latency budget (Hetzner CX43 CPU, target ≥ 30 FPS)

| Stage | Mean | p99 |
|---|---:|---:|
| Detect (MediaPipe) | 2 ms | 5 ms |
| Track | < 1 ms | 1 ms |
| MiniFASNet ONNX | 12 ms | 18 ms |
| Device boundary | 4 ms | 6 ms |
| Texture + Moire | 8 ms | 11 ms |
| Blink | 3 ms | 5 ms |
| rPPG | 4 ms | 6 ms |
| Screen replay | 5 ms | 7 ms |
| Frame analyzers (flicker, grid) | 6 ms | 8 ms |
| Fuser + session ingest | 1 ms | 2 ms |
| **Total** | **~45 ms** | **~70 ms** |

At 30 FPS we have a 33 ms budget per frame; the published default skips texture and moire on alternate frames (since their suppressed weight makes per-frame freshness unimportant) to fit. A GPU build runs every analyzer every frame.

## Reproducibility

Every artefact is reproducible from code in this repository:

- Fuser weights: `src/infrastructure/fusion/multi_class_fuser.py:DEFAULT_ANALYZER_WEIGHTS`
- Calibration data: `data/in_house/labels.csv` (43 samples, KVKK-consented)
- Benchmark harness: `tests/benchmark/run.py`
- Metrics: `src/metrics/iso30107.py` + `src/metrics/standard.py` (12 unit tests)
- Paper sections: `paper/sections/*.md` (Markdown, ready for LaTeX inlining)

The paper's tables are emitted automatically by `paper/figures/build_tables.py` from the per-protocol JSON written by the benchmark runner.
