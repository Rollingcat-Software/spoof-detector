# Spoof Detector

[![Tests](https://img.shields.io/badge/tests-114%20passing-brightgreen)](#testing)
[![Version](https://img.shields.io/badge/version-0.2.0-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![ISO 30107-3](https://img.shields.io/badge/ISO%2030107--3-Grade%20C-yellow)](#iso-30107-3-evaluation-2026-05-02)

Multi-signal face anti-spoofing engine that produces **session-level verdicts** by accumulating evidence across 5 seconds to 3 hours. Designed as a research engine and as a drop-in companion to a face-recognition pipeline.

Originally extracted from the [FIVUCSAS](https://github.com/Rollingcat-Software/FIVUCSAS) biometric-authentication platform's R&D track and now maintained as a standalone repository so it can be reused, evaluated and cited independently.

> **Status (2026-05-09):** runnable working example. `pytest` is green (114/114). MiniFASNet ONNX, MediaPipe FaceLandmarker, and the calibrated 7-class fuser are wired end-to-end. v0.2.0 adds the **gates** (`src/gates/`), **fusion** (`src/fusion/`), and **pipeline assembler** (`src/pipeline/`) sub-packages so a downstream service can consume just the parts it needs without bringing up the full session engine.

## What's new in v0.2.0 (2026-05-09)

Three new sub-packages, all importable independently of the main session engine:

- **`src.gates`** — pre-liveness face usability gates (no-face / occlusion / illumination quality). Useful as a short-circuit before any heavier ML runs.
  - `FaceUsabilityGate` — frame-level usable/blocked verdict with hysteresis.
  - `CriticalRegionVisibilityGate` — per-region (eyes/nose/mouth/lower-face) visibility scoring from pixel forensics + optional landmarks.
  - `FaceQualityIlluminationGate` — under/over-exposure + shadow-asymmetry verdict.
- **`src.fusion`** — `HybridFusionEvaluator` calibrated against MiniFASNet + heuristic device-replay signals.
- **`src.pipeline`** — `AntispoofPipelineAssembler` — one duck-typed adapter that runs the gates, fusion evaluator, and a caller-supplied device-spoof evaluator. Returns a single `AntispoofPipelineResult` with an advisory `recommended_action` (`allow` / `review` / `block`). The assembler **never enforces** — the caller decides.

These modules are Aysenur's work — see [AUTHORS.md](AUTHORS.md). They were ported from the FIVUCSAS `biometric-processor` R&D branch so they can be evaluated and cited independently.

### Quick install

```bash
pip install "git+https://github.com/Rollingcat-Software/spoof-detector.git@v0.2.0"
```

The lean install (`pip install spoof-detector`) brings in only `numpy + opencv-python`. To run the full session engine (MediaPipe, MiniFASNet, the existing 14-analyzer pipeline) install the `[full]` extra:

```bash
pip install "spoof-detector[full] @ git+https://github.com/Rollingcat-Software/spoof-detector.git@v0.2.0"
```

## What it does

Real-time face presentation-attack detection that classifies every session into one of seven categories:

| # | Category               | Primary signals                                       | Status     |
|---|------------------------|-------------------------------------------------------|------------|
| 0 | Real (live person)     | MiniFASNet + temporal motion + blink/landmark variance | working    |
| 1 | Static image (photo)   | MiniFASNet + device boundary + texture                | working    |
| 2 | Video replay (screen)  | MiniFASNet + screen flicker + moire + screen replay   | working    |
| 3 | 3D mask                | MiniFASNet (partial)                                  | partial    |
| 4 | Heavy makeup           | (planned)                                             | planned    |
| 5 | AR filter              | MobileNetV3-Small head (planned)                      | planned    |
| 6 | Deepfake injection     | Active illumination (planned)                         | planned    |

Per-frame the engine runs **14 analyzers** in three layers (pixel forensics / behavioural signals / environment), routes their scores into a calibrated multi-class fuser, and feeds the per-frame classification into a **session engine** that maintains a "guilty until proven innocent" liveness proof score (blinks + motion + rotation + expression up to 75/100). The session verdict is **peak-sensitive**: a single sustained spoof burst permanently affects the verdict so a real-face tail can't dilute the attack.

## ISO 30107-3 evaluation (2026-05-02)

Measured against four scripted attack scenarios (real face, phone-screen photo, printed photo, video replay):

| Metric  | Value       |
|---------|-------------|
| BPCER   | **0.00 %**  |
| APCER   | **30 %**    |
| ACER    | **15 %**    |
| Grade   | **C**       |

Per-scenario session verdicts:

- **Real face** — LIVE 78 %, liveness 63/100 PROVEN, 5 blinks, 0 incidents
- **Phone-screen photo** — SPOOF 43 %, liveness 23/100, 0 blinks, 7 incidents
- **Printed photo** — SPOOF 58 %, liveness 50/100, 3 incidents
- **Video replay** — LIVE 60 % — remaining open challenge (replay still shows real blinks/motion)

Calibration findings:

- **MiniFASNet** is the only reliable per-frame discriminator (+94.7 score gap real-vs-spoof). Weight 5.0×.
- **rPPG** is anti-correlated for screen attacks (detects display flicker as a false pulse) — **disabled**.
- **Texture / moire** are anti-correlated for screen attacks — **suppressed to 0.1×**.
- Blink (EAR=0.20) works for real faces but fires on video playback — weight kept low (0.5×).

## Quick start

```bash
# 1. Clone and install
git clone https://github.com/Rollingcat-Software/spoof-detector.git
cd spoof-detector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Single-image diagnostic (no webcam needed)
python tools/diagnose.py --image path/to/face.jpg

# 3. Live camera demo (needs a working webcam)
python main.py
#   q / ESC = quit, d = detail panel, s = save frame, h = help

# 4. Run the test suite (no camera, no model download)
pytest tests/ -v
```

The MiniFASNet ONNX model (~1.7 MB) is downloaded automatically on first use by [`uniface`](https://pypi.org/project/uniface/) and cached in your home directory.

See [`DEMO.md`](DEMO.md) for an annotated walk-through with expected output.

## Architecture

```
Session Engine (5 s – 3 hr)
  - "guilty until proven innocent" liveness prover
  - accumulates per-frame evidence + liveness proof score (0–100)
  - detects incidents (spoof bursts, frozen face, MiniFASNet instability)
  - peak-sensitive verdict (worst-window prevents dilution)
  - session report with liveness breakdown on exit

Pipeline (per frame)
  - MediaPipe face detection (~2 ms)
  - MediaPipe FaceLandmarker (478 points, shared across analyzers)
  - IoU multi-face tracking
  - 14 analyzers in 3 layers:

     Layer 1 — pixel forensics
        MiniFASNet ONNX        (5.0x)  +94.7 gap, proven
        Screen flicker         (3.0x)  50/60 Hz temporal aliasing
        Device boundary        (2.5x)  phone bezel detection
        Screen replay          (0.5x)  FFT + skin colour

     Layer 2 — behavioural signals
        Micro-tremor           (2.5x)  8–12 Hz involuntary oscillation
        Landmark variance      (2.0x)  478-point motion tracking
        Blink (EAR)            (0.5x)  V-shape validated blinks
        Temporal               (0.3x)  micro-motion naturalness
        rPPG                   (0.0x)  DISABLED (false pulse on screens)

     Layer 3 — environment
        Background grid        (1.5x)  6×4 cell stability
        AR filter              (0.3x)  heuristic (ONNX model planned)
        Texture                (0.1x)  anti-correlated, suppressed
        Moire                  (0.1x)  anti-correlated, suppressed

  - MultiClassFuser (calibrated weights) -> 7-category probabilities
  - LivenessProver: blinks(25) + motion(20) + rotation(15) + expression(15) = 75 max
```

## Repository layout

```
spoof-detector/
  main.py                           Live-camera entry point (OpenCV GUI)
  config.yaml                       Configuration (analyzer toggles, weights)
  src/
    domain/                         Pure types: BBox, FaceROI, SpoofCategory(7), SessionState
    application/                    SessionEngine, Pipeline, FaceTracker, LivenessProver
    infrastructure/
      detection/                    MediaPipe face + landmarker
      analyzers/                    14 analyzers (one file each)
      fusion/                       Calibrated 7-class probability fuser
      logging/                      JSONL session logger
    presentation/                   OpenCV HUD, threaded camera, app loop
  tests/                            68 unit tests (analyzers, domain, session)
  tools/
    diagnose.py                     Live diagnostic dashboard / single-image mode
    benchmark.py                    Per-analyzer timing + accuracy
    test_protocol.py                Guided attack-scenario walkthrough
    analyze_captures.py             Replay saved captures with ground truth
    evaluate.py                     ISO 30107-3 metric computation
    label_tool.py                   Annotation helper for new captures
    collect_ar_dataset.py           AR-filter dataset builder
    train_ar_detector.py            MobileNetV3-Small training entry (Phase 5)
  paper/
    outline.md                      Academic paper outline (BIOSIG / IJCB target)
```

## Testing

```bash
# Unit tests — 68 tests, ~3 s, no camera, no model download
pytest tests/ -v

# Per-analyzer benchmark
python tools/benchmark.py

# Guided test protocol (walks you through real / photo / screen / video scenarios)
python tools/test_protocol.py

# Single-image diagnostic
python tools/diagnose.py --image path/to/face.jpg
```

## Requirements

- Python 3.11+
- OpenCV (with GUI support — the headless `opencv-python-headless` will block `main.py` but not `pytest` / `tools/diagnose.py --image`)
- MediaPipe 0.10.9+
- `uniface` 3.0+ (MiniFASNet ONNX) and `onnxruntime` (CPU build by default; replace with `onnxruntime-gpu` for CUDA)
- Webcam (only for `main.py` / `tools/diagnose.py` live mode)

## Roadmap

- [x] Phase 1 — foundation (detection, tracking, overlay)
- [x] Phase 2 — analyzer integration (MiniFASNet, texture, moire, screen-replay, temporal)
- [x] Phase 2.5 — device boundary, calibrated fusion, session engine
- [x] Phase 3 — temporal analyzers (blink EAR, rPPG, landmark variance)
- [x] Phase 3.5 — "guilty until proven innocent" liveness architecture
- [x] Phase 3.6 — three-layer detection (screen flicker, micro-tremor, background grid)
- [x] Phase 3.7 — ISO 30107-3 first measurement (Grade C, 2026-05-02)
- [ ] Phase 4 — data collection (AR-filter dataset)
- [ ] Phase 5 — AR-filter detector training (MobileNetV3-Small)
- [ ] Phase 6 — expanded evaluation, target Grade B (APCER < 15 %)
- [ ] Phase 7 — production integration into FIVUCSAS `biometric-processor`

## Academic paper

Target venues: BIOSIG 2026 / IJCB 2026. Working title:

> **AR-Spoofing: Session-Based Multi-Method Face Presentation Attack Detection**

Outline lives in [`paper/outline.md`](paper/outline.md). The paper is **not** ready for submission yet — the current focus is shipping a stable, working detector. See [`ROADMAP.md`](ROADMAP.md) for the full plan.

## Authors

- **Ahmet Abdullah Gültekin** — lead author, architecture, calibration, session engine
- **Ayşe Gülsüm Eren** — gesture / liveness research thread
- Marmara University — Department of Computer Engineering

This work originated as part of the [FIVUCSAS](https://github.com/Rollingcat-Software/FIVUCSAS) capstone project (CSE4297 / CSE4197).

## License

MIT — see [`LICENSE`](LICENSE).
