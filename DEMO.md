# DEMO — Running the spoof detector end-to-end

This document is a **one-page walk-through** that shows how to feed the engine a frame or a live camera and read its verdicts. For background, calibration findings and ISO 30107-3 numbers see [`README.md`](README.md).

---

## 1. Install

```bash
git clone https://github.com/Rollingcat-Software/spoof-detector.git
cd spoof-detector
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The first time MiniFASNet runs it downloads its ONNX weights (~1.7 MB) into `~/.cache/uniface/` automatically.

## 2. Smoke-test without a camera

Run the unit test suite — no webcam, no GUI, no model download:

```bash
pytest tests/ -v
```

Expected:

```
============================== 68 passed in 2.78s ==============================
```

That covers domain types, all analyzers (texture, moire, screen-replay, temporal, MiniFASNet, device-boundary, blink, rPPG, AR-filter, multi-face tracker, fuser), session lifecycle, verdict logic, and incident detection.

## 3. Run all analyzers on a single image

This is the easiest end-to-end demo — no webcam needed, just one face image:

```bash
python tools/diagnose.py --image path/to/face.jpg
```

Expected output (real verbatim run on a synthetic face image, 2026-05-09):

```
============================================================
  Analyzing: synthetic_test.png
============================================================
  Size: 480x480
  Faces detected: 1

  Face #1 (193x193, conf=0.60)
         Analyzer     Score        Time  Details
  ────────────────────────────────────────────────────────────
       minifasnet    96.0  [ LIVE]    10.89ms  is_real=True, confidence=0.920, context=padded_crop
          texture    79.5  [ LIVE]     1.30ms  texture_score=100.000, color_score=66.455, frequency_score=65.229
            moire    40.4  [UNSURE]     5.76ms  moire_risk=0.596, gabor_risk=0.673, fft_risk=0.144, response_fraction=1.000
    screen_replay    27.5  [SPOOF]     7.70ms  fft_score=49.334, laplacian_score=0.001, laplacian_var=6705.787, skin_score=24.900
```

Each analyzer reports a 0–100 score (higher = more likely real), an elapsed time, and a `details` dict that explains *why*. The full pipeline runs in roughly **25–30 ms per frame on CPU**.

## 4. Live camera demo

```bash
python main.py
```

Controls in the OpenCV window: `q` / `ESC` quit, `d` toggle detail panel, `s` save the current frame and analysis, `h` help overlay.

Side-by-side: a per-analyzer dashboard with scores, timings, and 30-frame sparklines:

```bash
python tools/diagnose.py
```

CSV-log per-analyzer scores for off-line analysis (writes to `logs/diag_*.csv`):

```bash
python tools/diagnose.py --log-csv
```

## 5. Guided attack scenarios

The test protocol script walks you through four scripted scenarios (real face, phone-screen photo, printed photo, video replay) and writes a JSON session report to `data/protocol/`:

```bash
python tools/test_protocol.py
```

Each scenario produces a session verdict (LIVE / SPOOF) with confidence, liveness proof score 0–100, blink count, and an incident timeline. Three real protocol reports from 2026-05-02 ship under `data/protocol/`. See [`README.md` § ISO 30107-3 evaluation](README.md#iso-30107-3-evaluation-2026-05-02) for the headline numbers those reports produce.

## 6. Where to look in the source

| Question                                       | File                                                                  |
|------------------------------------------------|-----------------------------------------------------------------------|
| Which analyzers exist and what they output     | `src/infrastructure/analyzers/*.py` (one per analyzer)                |
| How analyzer scores are fused into 7 classes   | `src/infrastructure/fusion/multi_class_fuser.py`                      |
| Session lifecycle, verdict, liveness scoring   | `src/application/session_engine.py` + `src/domain/session.py`         |
| The full per-frame pipeline                    | `src/application/pipeline.py`                                         |
| Calibrated default weights                     | `MultiClassFuser.DEFAULT_WEIGHTS` in the fuser file                   |
| Configuration knobs                            | `config.yaml`                                                         |

## 7. Production integration into FIVUCSAS

This standalone repo is the research / reference implementation. Production wiring into the FIVUCSAS `biometric-processor` (Stage 2/3) will:

- expose a `verify_session(frames) -> SessionVerdict` Python entry-point
- be called by the `/verify` and `/enroll` FastAPI handlers
- replace the current single-frame `LIVENESS_BACKEND=uniface` passive check

That work is **upcoming**, not shipped. The first deliverable was making this engine runnable, tested and citable in isolation — which is what this repository gives you.
