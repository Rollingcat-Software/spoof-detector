# Spoof Detector — Browser Readiness Audit

> **STATUS — 2026-05-16 (Phase 4 complete):** This audit's 3-week MVP
> estimate was executed over a single session, then extended across
> four phases (PRs #19 → #23) over 2026-05-15 → 2026-05-16. The
> TypeScript port lives in `web/`, deployed at
> https://fivucsas.com/amispoof/ with:
>
> - **12 analyzers** — MiniFASNet, Blink, LandmarkVariance, DeviceBoundary,
>   MicroTremor, ScreenFlicker, rPPG, Moire, Texture, ScreenReplay,
>   **BackgroundGrid**, **Temporal**
> - **3 gates (Aysenur)** — FaceUsability, Illumination, CriticalRegionVisibility
> - **Fusion** — MultiClassFuser (paper-calibrated weights) + HybridFusionEvaluator
> - **Pipeline** — AntispoofPipelineAssembler
> - **Session** — SessionEngine + **LivenessProver**
> - **Web Worker** offload for the 4 heavy analyzers (Vite `?worker&inline`)
> - **WebGPU EP** for MiniFASNet (feature-detected, WASM fallback)
> - **Lazy bundle chunks** for Texture/Moire/ScreenReplay
> - **CASIA-FASD validation harness** (`runCasiaFasdMicroBench`)
>
> 126 vitest tests green. Main bundle 123 kB ESM. Performance: mobile
> Brave 4→8–12 fps, PC Chrome 25–30 fps. See `ROADMAP.md` "Browser
> Port v0.1.0" section and `web/amispoof/README.md` for the current
> state. Sections below are preserved as a historical record of the
> pre-port analysis.

---

**Date:** 2026-05-09
**Source revision:** `master` (post real-numbers PR #6 merge, sha `5b9aa6b`)
**Target deliverable:** a `<script src="spoof-detector.js">` that runs the full
session-based anti-spoof pipeline in the browser with no server round-trip
(except for a one-time static-asset fetch of the FaceLandmarker `.task`,
the MiniFASNet `.onnx`, and the WASM runtime).
**Auditor:** browser-port readiness review; no source code modified.

---

## Executive summary

**Readiness: 7.5 / 10.**

This codebase is unusually well positioned for a browser port. Of the 14
analyzers + 4 gates + 2 fusion modules + the session engine, **only one
component carries a hard external runtime dependency that doesn't already
live in the browser** (`uniface`, the Python wrapper around the MiniFASNet
ONNX), and that wrapper is a ~30-line shim — the actual model file
(`~/.uniface/models/minifasnet_v2.onnx`, 1.7 MB) is already standard ONNX
and ships verbatim to `onnxruntime-web`. The other heavy dependencies
(MediaPipe FaceLandmarker, OpenCV image ops, NumPy / `scipy.signal`-style
FFT) all have first-class browser equivalents that this very repo's sibling
project (`/opt/projects/fivucsas/web-app`) is **already using in production**
(BiometricEngine: `@mediapipe/tasks-vision`, `onnxruntime-web@1.18`, and a
hand-ported PassiveLivenessDetector that mirrors the same image primitives
used here). A working browser MVP covering the four highest-weight analyzers
(MiniFASNet, device boundary, blink, landmark variance — 12.0 of 17.6 total
fusion weight, ~68%) is reachable in **3 engineer-weeks**. A full 14-analyzer
parity port is **6–8 engineer-weeks**, with one open R&D question (FFT-based
analyzers under WebRTC's variable per-tab framerate). The session engine,
multi-class fuser, taxonomy and gates port 1:1 to TypeScript with zero
external library dependencies — they are pure state machines + arithmetic.

---

## 1. Module inventory and dependency classification

### 1.1 Third-party runtime imports (entire `src/`)

| Library | Used by | Footprint | Browser status |
|---|---|---|---|
| `numpy` | every file (28/28) | array math, FFT, statistics | **portable** — replace with TypedArrays + a ~200-line `np.ts` shim or [`ndarray`](https://www.npmjs.com/package/ndarray). FFT via [`fft.js`](https://www.npmjs.com/package/fft.js) (rfft/irfft, ~3 KB gz) or hand-rolled (signals here are ≤300 samples). |
| `cv2` (opencv-python) | 14 files (most analyzers + gates + presentation) | image conversion, Laplacian, Canny, Hough, Gabor, FFT, CLAHE, contours, color-space conversions | **partial → portable** — every operator used in this repo is in [OpenCV.js](https://docs.opencv.org/4.x/d5/d10/tutorial_js_root.html) (~9 MB WASM) **or** trivially expressible in Canvas2D / a few hundred lines of TS. See section 2 for op-by-op mapping. |
| `mediapipe` (Python Tasks API) | `mediapipe_detector.py`, `blink_analyzer.py` | BlazeFace short-range detector + FaceLandmarker (478 pts) | **portable** — `@mediapipe/tasks-vision` is the canonical browser SDK. Same `.task` model bundle. Already in `web-app/package.json` at `^0.10.18`. |
| `uniface` (PyPI, calls onnxruntime under the hood) | `minifasnet_analyzer.py` only | thin wrapper around `minifasnet_v2.onnx` | **portable** — the `.onnx` file is the deployable artifact. Wrapper is ~30 lines (preprocessing: bbox-padded crop, BGR→RGB, resize 80×80, normalize, NCHW). Re-implement directly on `onnxruntime-web`. |
| `onnxruntime` | indirectly via `uniface`; directly in `ar_filter_analyzer.py` (optional) | ONNX inference | **portable** — `onnxruntime-web@1.18.0` (already in `web-app/package.json`). WASM EP is fastest for MiniFASNet at this size; WebGPU is +1.5–3× but limited operator coverage. |
| `scikit-image` | listed in `requirements.txt` only | unused at module scope; only imported lazily in benchmarks/tools | **partial** — the only place this would matter at runtime is `tools/`, which is not part of the browser surface. |
| `scipy` (`scipy.signal`) | listed in `requirements.txt`; **not actually imported** anywhere under `src/` | n/a | **n/a** — this repo's code uses raw `np.fft` + `np.convolve` for filtering, which port directly. Confirmed via `grep -rn "scipy" src/` (zero hits). |
| `pyyaml` | only `main.py` (config loader) | n/a | **n/a** — config becomes a JS object in the browser. |
| `python-json-logger` | `infrastructure/logging/structured_logger.py` | structured JSONL | **portable** — `console.log({...})` or any browser logger. |

**Verdict:** **zero hard blockers.** No DeepFace, no PyTorch, no TensorFlow,
no Resemblyzer. The `pyproject.toml` deliberately keeps the lean install at
`numpy + opencv-python` only, with everything else under the `[full]` extra
— which is itself a strong signal the author already pre-segregated the
"heavy" deps from the algorithmic core.

### 1.2 Per-file inventory (29 files in `src/`, 4768 lines)

| File | Lines | Third-party | Browser equivalent |
|---|---|---|---|
| `domain/models.py` | 149 | numpy (typing only) | TS interfaces, no runtime dep |
| `domain/interfaces.py` | 76 | numpy (Protocol) | TS interfaces |
| `domain/session.py` | 122 | — | TS dataclasses |
| `domain/taxonomy.py` | 114 | — | TS const map |
| `application/face_tracker.py` | 148 | — | pure TS (IoU + Hungarian-ish) |
| `application/pipeline.py` | 118 | numpy | TS class |
| `application/session_engine.py` | 494 | numpy | TS class |
| `application/liveness_prover.py` | 338 | numpy | TS class |
| `application/data_collector.py` | 97 | cv2, numpy | optional, not needed for browser MVP |
| `infrastructure/detection/mediapipe_detector.py` | 123 | mediapipe, cv2 | `@mediapipe/tasks-vision` FaceDetector (already in `web-app`) |
| `infrastructure/analyzers/minifasnet_analyzer.py` | 121 | uniface, cv2 | `onnxruntime-web` direct |
| `infrastructure/analyzers/blink_analyzer.py` | 252 | mediapipe, cv2, numpy | `@mediapipe/tasks-vision` FaceLandmarker |
| `infrastructure/analyzers/landmark_variance_analyzer.py` | 164 | numpy (no cv2) | pure TS |
| `infrastructure/analyzers/micro_tremor_analyzer.py` | 163 | numpy (FFT) | TS + `fft.js` |
| `infrastructure/analyzers/rppg_analyzer.py` | 186 | numpy (FFT) | TS + `fft.js` |
| `infrastructure/analyzers/screen_flicker_analyzer.py` | 141 | numpy (FFT), cv2 | TS + `fft.js` + Canvas2D |
| `infrastructure/analyzers/temporal_analyzer.py` | 124 | numpy | pure TS (only mean/std) |
| `infrastructure/analyzers/texture_analyzer.py` | 108 | cv2, numpy | OpenCV.js or Canvas2D + tiny FFT |
| `infrastructure/analyzers/moire_analyzer.py` | 150 | cv2, numpy | OpenCV.js (Gabor + CLAHE + FFT) **OR** WebGL Gabor shader |
| `infrastructure/analyzers/screen_replay_analyzer.py` | 154 | cv2, numpy | OpenCV.js (CLAHE + Laplacian + colour-space) |
| `infrastructure/analyzers/device_boundary_analyzer.py` | 189 | cv2, numpy | OpenCV.js (Canny + HoughLinesP + findContours) |
| `infrastructure/analyzers/background_grid_analyzer.py` | 154 | cv2, numpy | OpenCV.js (cvtColor + grid stats) |
| `infrastructure/analyzers/ar_filter_analyzer.py` | 173 | cv2, numpy, onnxruntime (opt) | OpenCV.js + `onnxruntime-web` |
| `infrastructure/fusion/multi_class_fuser.py` | 92 | — | pure TS |
| `gates/landmarks.py` | 67 | — | TS dataclasses |
| `gates/illumination.py` | 241 | cv2, numpy | OpenCV.js or hand-rolled |
| `gates/critical_region_visibility.py` | 901 | cv2, numpy | OpenCV.js (Lab/YCrCb/HSV cvtColor + Laplacian + Canny) |
| `gates/face_usability.py` | 341 | numpy | pure TS (orchestrates the two above) |
| `fusion/hybrid_evaluator.py` | 190 | numpy (clamp only) | pure TS |
| `pipeline/assembler.py` | 250 | numpy (typing) | pure TS |
| `metrics/iso30107.py` | 215 | numpy | offline tooling — not browser-needed |
| `metrics/standard.py` | 188 | numpy | offline tooling — not browser-needed |
| `presentation/*` | 635 | cv2 (OpenCV GUI) | replaced by HTMLCanvas; remove OpenCV-GUI dependency entirely |

---

## 2. Per-analyzer port assessment

Each row gives: which JS lib carries the heavy lift, an honest effort
estimate (S = ≤1 day, M = 2–4 days, L = ≥1 week), and any non-obvious risk.

| # | Analyzer | Fusion weight | Heavy lift in JS | Effort | Risk / note |
|---|---|---|---|---|---|
| 1 | **MiniFASNet** | **5.0** (proven, +94.7 gap) | `onnxruntime-web` direct on `minifasnet_v2.onnx` (1.7 MB) | **S** | Only the bbox-padded preprocessing + softmax/sigmoid post is ours. Output shape is published in the file's `metadata_props`. **Reference:** `web-app/src/lib/biometric-engine/core/CardDetector.ts` already does the identical pattern (dynamic-import ort, `wasmPaths` CDN, `wasm` EP, NCHW float32 tensor) for a 6 MB YOLO model — one-to-one template. |
| 2 | **Screen flicker** | 3.0 | TypedArray detrend + Hanning + `fft.js` rfft on a 60–120-sample buffer | **S** | Uses `np.convolve` (10-tap moving avg) and `np.fft.rfftfreq` — both ~5 lines of TS. **Caveat:** algorithm assumes 30 fps; real browser tabs run 24–60 fps depending on power-state. Already self-measures via `_frame_times`, so the port can keep that exact logic. |
| 3 | **Device boundary** | 2.5 | OpenCV.js (`cvtColor`, `GaussianBlur`, `Canny`, `HoughLinesP`, `findContours`, `approxPolyDP`, `boundingRect`) | **M** | All ops are 1:1 in OpenCV.js. The bezel-aspect-ratio test, line-orientation histogram and contour-rectangularity scoring are pure arithmetic. ~150 lines of TS once OpenCV.js is in place. |
| 4 | **Micro-tremor** | 2.5 | Same as flicker — TypedArray detrend + Hanning + rfft on 90–180 samples × 2 axes | **S** | Reads pre-computed FaceLandmarker centroid; no extra detector needed. |
| 5 | **Landmark variance** | 2.0 | Pure NumPy (`mean`, `var`, `np.array(deque)`) over a 60-frame deque of 478×2 points | **S** | Zero external libs once landmarks are sourced from `@mediapipe/tasks-vision`. ~80 lines of TS. |
| 6 | **Background grid** | 1.5 | OpenCV.js `cvtColor` (BGR→Gray, BGR→HSV) + per-cell mean/std | **S** | Trivially Canvas2D-able if you want to skip OpenCV.js altogether — the only op is per-tile `getImageData` mean. |
| 7 | **rPPG** | 0.0 (disabled) | TypedArray detrend + rfft on 150–300-sample buffer | **S** | Currently weight-0 in calibrated config because it false-pulses on screens. Port for completeness/research, but not a launch-blocker. |
| 8 | **Blink (EAR)** | 0.5 | `@mediapipe/tasks-vision` FaceLandmarker + 6-point Euclidean distance | **S** | Already the canonical use-case for the JS FaceLandmarker. **Reference:** `web-app/src/features/auth/hooks/useFaceDetection.ts` and `useLivenessPuzzle.ts` use this exact pattern in production. The 12 landmark indices (`RIGHT_EYE`, `LEFT_EYE`) and `compute_ear()` are 1:1 portable. |
| 9 | **Screen replay** (whole-frame) | 0.5 | OpenCV.js CLAHE + Laplacian + cvtColor (BGR→YCrCb, BGR→HSV) + a small FFT | **M** | The skin-mask bands (Cr 133–173, Cb 77–127) are direct integer comparisons — fast in TypedArrays. CLAHE is the most expensive op; OpenCV.js has it but a hand-roll is feasible. |
| 10 | **AR filter** | 0.3 | Heuristic mode: OpenCV.js `cvtColor` + `Canny`. Model mode: a future `ar_filter.onnx` via `onnxruntime-web`. | **S** (heuristic) / **M** (when model lands) | The model isn't trained yet (Phase 5 in `ROADMAP.md`). Heuristic-only port is ~80 lines of TS. |
| 11 | **Temporal** | 0.3 | Pure NumPy (mean/std on 30-frame deque of bbox positions) | **S** | No image ops. ~60 lines of TS. |
| 12 | **Texture** | 0.1 (suppressed, anti-correlated) | OpenCV.js Laplacian + cvtColor (BGR→HSV) + small FFT (192×108) | **S** | Currently anti-correlated — port for completeness, ship at weight 0.1. |
| 13 | **Moire** | 0.1 (suppressed, anti-correlated) | OpenCV.js Gabor kernel bank (4 thetas) + CLAHE + FFT | **M** | This is the heaviest CPU per frame — `cv2.getGaborKernel` + 4× `filter2D` on a 160-px square. **WebGL alternative:** Gabor convolution is famously well-suited to fragment shaders; a ~50-line GLSL fragment shader gives ~10× speedup. Not on critical path (weight 0.1). |
| 14 | **MakeUp** | (planned, weight 0.0) | n/a | n/a | Not implemented in source; skip. |

**Detection backbone (`infrastructure/detection/mediapipe_detector.py`)**
ports as a thin wrapper around `@mediapipe/tasks-vision` `FaceDetector` —
**effort S**. The web-app already does this (`useFaceDetection.ts`,
lines 78–187, with BlazeFace primary + FaceLandmarker fallback).

---

## 3. Architectural changes for the browser port

Three layers, in order of difficulty:

### 3.1 Pure logic — port 1:1, **no external libs needed**

These files contain only state machines, accumulators, and arithmetic. They
become TypeScript classes verbatim. The `numpy` calls inside them are all
`mean`, `var`, `std`, `sqrt`, `clip`, `convolve` (10-tap moving averages) —
each is a 5-line TS helper.

| Python | TS target | Effort |
|---|---|---|
| `application/pipeline.py` (118 lines) | `SpoofDetectionPipeline.ts` | S |
| `application/session_engine.py` (494 lines) | `SessionEngine.ts` | M |
| `application/liveness_prover.py` (338 lines) | `LivenessProver.ts` | M |
| `application/face_tracker.py` (148 lines) | `FaceTracker.ts` (IoU matching) | S |
| `domain/*.py` (526 lines total) | `domain/*.ts` (interfaces + dataclasses) | S |
| `infrastructure/fusion/multi_class_fuser.py` (92 lines) | `MultiClassFuser.ts` | S |
| `fusion/hybrid_evaluator.py` (190 lines) | `HybridFusionEvaluator.ts` | S |
| `pipeline/assembler.py` (250 lines) | `AntispoofPipelineAssembler.ts` | S |
| `gates/face_usability.py` (341 lines, orchestrator) | `FaceUsabilityGate.ts` | M |

**Total:** ~2 500 lines of mechanical port. ~5 engineer-days for someone who
knows both languages.

### 3.2 Image-ops layer — choose between OpenCV.js and hand-roll

Decision point: do we ship OpenCV.js (~9 MB WASM) or hand-roll the 12
distinct OpenCV ops we use? The op inventory is small enough to make
hand-rolling viable:

| OpenCV op | Used by | Hand-roll cost |
|---|---|---|
| `cvtColor` (BGR↔Gray/HSV/YCrCb/Lab/RGB) | nearly every analyzer | trivial — Canvas2D `getImageData` + a 5-line per-pixel formula |
| `resize` (`INTER_AREA`) | texture, moire, screen-replay | Canvas2D `drawImage(..., w, h)` |
| `Laplacian` | texture, screen-replay, gates | 3×3 kernel convolution, ~30 lines |
| `Canny` | device-boundary, ar-filter, gates | non-trivial (gradient + non-max suppression + double-threshold + hysteresis) — **best to use OpenCV.js for this** |
| `HoughLinesP` | device-boundary | non-trivial — **OpenCV.js** |
| `findContours` + `approxPolyDP` + `boundingRect` | device-boundary | non-trivial — **OpenCV.js** |
| `GaussianBlur` | device-boundary | trivial separable kernel, ~20 lines |
| `getGaborKernel` + `filter2D` | moire | hand-roll Gabor or **WebGL fragment shader** |
| `createCLAHE().apply()` | moire, screen-replay, gates | non-trivial (per-tile histogram equalisation) — **OpenCV.js** or accept a quality loss with a single global histogram-equalisation |
| `getStructuringElement` / morphology | not used | n/a |
| `np.fft.rfft` / `np.fft.fft2` / `np.fft.fftshift` | rppg, micro-tremor, flicker, texture, moire, screen-replay | use [`fft.js`](https://www.npmjs.com/package/fft.js) (1D, ~3 KB) and a tiny 2D-FFT helper that calls 1D-FFT on rows then columns; or for `fft2` on small windows (e.g. 192×108), accept the cost of OpenCV.js's `dft`. |

**Recommendation:** ship OpenCV.js for the MVP (MiniFASNet alone makes the
download cost worth it — phantoms of 9 MB on first load, served from a CDN
with `Cache-Control: public, max-age=31536000, immutable`, are the cost of
admission for *any* serious vision-on-the-browser app). When/if size budget
becomes a hard constraint, **selectively replace** the simple ops
(`cvtColor`, `resize`, `Laplacian`, `GaussianBlur`) with hand-rolls and
keep OpenCV.js only for the heavy nonlinear ops (`Canny`, `HoughLinesP`,
`findContours`, `CLAHE`).

### 3.3 ML model layer — three static assets, lazy-loaded

| Asset | Source | Size | Ship as |
|---|---|---|---|
| `face_landmarker.task` | already at `spoof-detector/models/face_landmarker.task` (also in `web-app/public/models/`) | ~10 MB | static asset; load via `@mediapipe/tasks-vision` `FaceLandmarker.createFromOptions({ baseOptions: { modelAssetPath: '/path/face_landmarker.task' } })` |
| `blaze_face_short_range.tflite` | already at `spoof-detector/models/` | ~250 KB | static asset; load via `@mediapipe/tasks-vision` `FaceDetector.createFromOptions(...)`. **Optional** — FaceLandmarker has built-in detection, so this is only needed if you want a separate ultra-cheap detector pass. |
| `minifasnet_v2.onnx` | `~/.uniface/models/minifasnet_v2.onnx` (downloaded by `uniface` package on first run) | **1 743 581 bytes (1.7 MB)** | static asset; load via `ort.InferenceSession.create('/models/minifasnet_v2.onnx', { executionProviders: ['wasm'] })` |
| `ar_filter.onnx` | not yet trained (Phase 5) | est. ~5 MB | future |

**Lazy-load pattern (already proven in this codebase):** every web-app
component that needs a model uses `await import('onnxruntime-web')` /
`await import('@mediapipe/tasks-vision')` inside the first call site, so
the bundle defers the WASM cost until the user actually invokes the
detector. See `web-app/src/lib/biometric-engine/core/CardDetector.ts` lines
189–204 for the canonical 15-line implementation.

### 3.4 Pipeline orchestration — a shape-preserving translation

The current `SpoofDetectionPipeline.process(frame)` signature is
`(np.ndarray) → FrameAnalysis`. The TS port becomes
`process(imageBitmap: ImageBitmap | HTMLVideoElement | ImageData): Promise<FrameAnalysis>`.

The only structural rework needed:

1. **Frame source.** Today it's a BGR `np.ndarray` from OpenCV's `VideoCapture`.
   In the browser it's a `MediaStreamTrack` from `getUserMedia()`.
   Conversion: `videoElement → captureStream → drawImage to OffscreenCanvas → getImageData`.
   This pattern is already in `web-app/src/features/auth/hooks/useFaceDetection.ts`.

2. **Async-everywhere.** MediaPipe and ONNX Runtime Web expose async APIs.
   Wrap analyzer `analyze()` calls in `async`/`await` and run
   I/O-independent analyzers concurrently via `Promise.all`.

3. **No multithreading.** The Python pipeline is single-threaded already
   (no `multiprocessing` in `src/`), so this is a no-op architectural change.
   For higher throughput later, a Web Worker with the WASM runtime is
   straightforward (and `onnxruntime-web` natively supports running in
   workers).

4. **Replace `cv2.VideoCapture` and the OpenCV-GUI HUD** (`presentation/`)
   with a thin React/vanilla-DOM overlay. Drop entirely from MVP.

---

## 4. Performance projection

### 4.1 Current Python wall-clock per frame (numbers from in-source comments)

| Stage | ms (CPU) |
|---|---|
| MediaPipe FaceDetector | ~2 |
| MediaPipe FaceLandmarker (shared by blink + landmark_variance) | ~5 |
| MiniFASNet ONNX | ~3 |
| Texture | ~5 |
| Moire | ~5 |
| Screen replay (whole-frame) | ~8 |
| Device boundary | ~6 |
| Other 8 analyzers (mostly arithmetic on small buffers) | ~10 total |
| **Total per frame** | **~44 ms** (sustained ~22 fps single-threaded) |

### 4.2 Browser projection (same commodity laptop, single tab, WASM EP)

Order-of-magnitude expectations, drawn from published `onnxruntime-web` /
MediaPipe-JS benchmarks and from the `web-app` BiometricEngine's measured
behaviour on the same MiniFASNet-tier workloads (CardDetector at 640×640 ≈
~50–80 ms, PassiveLivenessDetector ≈ ~10 ms):

| Stage | Python ms | Browser-WASM ms (projected) | Notes |
|---|---|---|---|
| MediaPipe FaceLandmarker (`@mediapipe/tasks-vision`, GPU delegate where available) | 5 | **3–8** | GPU delegate is faster than Python's CPU-only TFLite. WASM fallback is ~2× slower. |
| MiniFASNet (`onnxruntime-web`, WASM EP) | 3 | **8–15** | ORT-Web WASM is ~2.5–4× slower than ORT CPython. Tiny model so absolute cost is low. WebGPU EP could bring this back to ~3–5 ms but coverage is uneven. |
| OpenCV.js per analyzer (Laplacian / Canny / Hough / Gabor) | 5–8 each | **10–20 each** | OpenCV.js is the bottleneck — WASM overhead + every op marshals through the JS heap. |
| FFT analyzers (rPPG, micro-tremor, flicker) | <1 each | **<1 each** | TypedArray FFT on 60–300 samples is essentially free. |
| Pure-arithmetic analyzers (temporal, landmark variance) | <1 each | **<1 each** | TypedArray loops are JIT'd — zero overhead vs Python. |
| **Total per frame (all 14 analyzers, naive)** | ~44 | **~110–180** | i.e. 6–9 fps single-threaded. |
| **Total per frame (MVP — 4 analyzers + landmarker + MiniFASNet + tracker)** | ~25 | **~30–55** | **~18–30 fps**, comfortably real-time. |

### 4.3 Honest caveats

- **JIT warmup.** First-frame ORT-Web inference on a cold session takes
  300–600 ms; subsequent frames run at the rate above. The session engine's
  existing `WARMUP_FRAMES = 30` already absorbs this.
- **WASM thread count.** `ort.env.wasm.numThreads = 2` (the value the
  web-app uses) is a sweet spot for laptop CPUs; setting it to 4 helps
  desktop but hurts mobile due to thermal throttling.
- **Copy costs.** Every `getImageData()` is a GPU→CPU copy (~1–3 ms at
  1280×720). Run analyzers on the smallest crop they need (face crop is
  ~150×150 px) to keep this under control.
- **Variable frame rate.** Browsers throttle background tabs to 1 fps and
  reduce foreground tabs under low battery. The flicker / rPPG / tremor
  FFT analyzers all already self-measure FPS via a `frame_times` deque,
  so the port inherits that resilience for free.
- **Memory.** Holding 60-frame deques of 478×2 landmarks + 300-frame deques
  of green-channel scalars per face is ~50 KB/face — negligible.

---

## 5. Existing browser-side ML in the FIVUCSAS web-app — what we can reuse

Verified by `grep` on `/opt/projects/fivucsas/web-app/src/`. The web-app is
**already shipping** a production-grade browser biometrics layer that is
~70% the foundation we need for the spoof-detector port.

### 5.1 The BiometricEngine library (`web-app/src/lib/biometric-engine/`)

A full hexagonal browser-ML core:

```
src/lib/biometric-engine/
├── core/
│   ├── BiometricEngine.ts          ← singleton entry, getInstance()
│   ├── FaceDetector.ts             ← @mediapipe/tasks-vision wrapper
│   ├── FaceTracker.ts              ← IoU-based multi-face tracking
│   ├── FaceMetricsCalculator.ts    ← geometry helpers
│   ├── HeadPoseEstimator.ts        ← yaw/pitch/roll from landmarks
│   ├── PassiveLivenessDetector.ts  ← 5-component texture/colour/skin/moire/local-var (DIRECT ANALOG of our texture+moire+screen-replay analyzers)
│   ├── CardDetector.ts             ← onnxruntime-web YOLO loader (REFERENCE IMPLEMENTATION for MiniFASNet)
│   ├── EmbeddingComputer.ts        ← 512-dim from landmarks
│   ├── HandGestureDetector.ts      ← @mediapipe/tasks-vision HandLandmarker (template for any hand challenges)
│   ├── QualityAssessor.ts          ← blur + brightness + face-size (DIRECT ANALOG of our gates/illumination + critical_region_visibility)
│   ├── FrameProcessor.ts           ← orchestration loop (DIRECT ANALOG of our SpoofDetectionPipeline)
│   ├── BiometricPuzzle.ts          ← session-state helper
│   ├── EnrollmentController.ts     ← state-machine over multiple frames
│   ├── VoiceVAD.ts                 ← onnxruntime-web Silero VAD (second ORT reference impl)
│   ├── image-utils.ts              ← Gabor kernels, Laplacian variance, mean HSV, std, variance, grayscale (REUSE-ME-AS-IS)
│   ├── constants.ts                ← thresholds, weights, kernels (parallel to our `config.yaml`)
│   └── challenges/                 ← liveness puzzle tasks
├── hooks/                          ← React hooks (out of scope for `<script>` build but useful for app integration)
├── interfaces/                     ← clean ports
└── types/                          ← shared dataclass equivalents
```

### 5.2 Specifically reusable bits

- **`web-app/src/lib/biometric-engine/core/CardDetector.ts`** is the canonical
  template for our MiniFASNet port. The 15-line dynamic-import + `wasmPaths`
  CDN pattern at lines 189–204 + the NCHW Float32Array preprocessing at
  lines 280–300 are the exact same shape MiniFASNet wants (just 80×80 input
  instead of 640×640, and a softmax+sigmoid output instead of YOLO anchors).
- **`web-app/src/lib/biometric-engine/core/PassiveLivenessDetector.ts`** is
  a hand-port of a Python liveness detector with the **same primitives**
  (Laplacian variance, mean HSV, Gabor kernels) we need for our texture
  analyzer + moire analyzer + screen-replay analyzer. A spoof-detector port
  should literally extend this class or copy its `image-utils.ts` helpers.
- **`web-app/src/lib/biometric-engine/core/image-utils.ts`** already exports
  `applyGaborFilter`, `computeLaplacianVariance`, `computeMeanHSV`,
  `computeStd`, `computeVariance`, `GABOR_KERNELS`, `toGrayscale`. These are
  exactly the helpers our `texture_analyzer.py`, `moire_analyzer.py`,
  `screen_replay_analyzer.py` and `gates/illumination.py` consume from
  OpenCV/NumPy. **This is the single biggest reuse opportunity** —
  ~600 lines of port work that is already done.
- **`web-app/src/lib/biometric-engine/core/QualityAssessor.ts`** parallels
  our `gates/illumination.py` + `gates/face_usability.py` — same
  blur/brightness/face-size scoring. The port can extend rather than
  rewrite.
- **`web-app/src/features/auth/hooks/useFaceDetection.ts`** demonstrates the
  primary + fallback detector pattern (BiometricEngine FaceLandmarker first,
  MediaPipe FaceDetector second) on a live `<video>` element — a drop-in
  for our `mediapipe_detector.py` shim.
- **`web-app/src/config/cdn.ts`** centralises the MediaPipe vision-bundle
  URL and the WASM URL so they stay version-locked between an `index.html`
  prefetch and the runtime loader. Worth replicating in the
  spoof-detector bundle.
- **Static models already on the CDN host:** `web-app/public/models/`
  contains `silero-vad.onnx` and `yolo-card-nano.onnx` and the
  `fetch-models.mjs` build-time downloader. The same pipeline can pull
  `minifasnet_v2.onnx` for us — already a one-line change to that script.

### 5.3 What we still have to build from scratch

- The **session engine** (`SessionEngine.ts`, ~500 lines) — the web-app's
  `EnrollmentController.ts` is a different state-machine shape (linear
  enrollment steps) and doesn't accumulate the peak-sensitive incident
  ledger our session engine maintains.
- The **liveness prover** (`LivenessProver.ts`, ~340 lines) — web-app's
  `BiometricPuzzle.ts` is the closest analog but covers only active
  challenges, not the "guilty-until-proven-innocent" passive proof score.
- The **multi-class 7-category fuser** + `taxonomy.ts` (~200 lines) —
  no equivalent in web-app; it currently only does binary live/spoof.
- The **device boundary analyzer** (~200 lines + OpenCV.js Hough/contour
  ops) — no analog in web-app.

---

## 6. Estimated effort

| Phase | Scope | Effort | Cumulative |
|---|---|---|---|
| **Phase 0 — Scaffolding** | TS package layout, Vite/Rollup config matching web-app's `vite.sdk.config.ts` pattern, jest/vitest harness, `fetch-models.mjs` clone, CDN config, `<script>` UMD/ESM bundle target | 2 days | 2 d |
| **Phase 1 — Pure-logic port** | domain/, multi_class_fuser, hybrid_evaluator, taxonomy, face_tracker, pipeline assembler, fusion | 5 days | 7 d |
| **Phase 2 — Detection + landmarks** | Wrap `@mediapipe/tasks-vision` FaceDetector + FaceLandmarker; FaceROI conversion; reuse `web-app/src/lib/biometric-engine/core/FaceDetector.ts` directly | 2 days | 9 d |
| **Phase 3 — MiniFASNet** | `onnxruntime-web` wrapper + bbox-padded preprocessing; copy `CardDetector.ts` shape; ship `minifasnet_v2.onnx` static asset | 2 days | 11 d |
| **Phase 4 — Top-4 analyzers (MVP gate)** | Blink, landmark variance, micro-tremor, device boundary. Together with MiniFASNet this is **12.0 of 17.6 fusion weight (68%)** and covers 3 of 4 attack categories from the published 2026-05-02 evaluation. | 4 days | 15 d |
| **Phase 5 — Session engine + prover** | Port SessionEngine + LivenessProver verbatim; wire to fuser | 4 days | 19 d |
| **Phase 6 — Public API + docs** | `<script src=>` loader, embed example, `<spoof-detector>` Web Component (matching the existing `vite.config.elements.ts`), README, CHANGELOG | 2 days | 21 d |
| **🟢 MVP shippable: ~3 engineer-weeks (15 working days, 21 calendar days)** | | | |
| Phase 7 — Remaining analyzers | Texture, moire, screen-replay, screen-flicker, background-grid, ar-filter, rppg, temporal | 6 days | 27 d |
| Phase 8 — Gates port | face_usability + critical_region_visibility + illumination | 6 days | 33 d |
| Phase 9 — Performance pass | OpenCV.js → hand-roll for hot ops, WebGL Gabor shader, Web Worker offload | 4 days | 37 d |
| Phase 10 — Test parity | Port the 114 pytest cases to vitest, calibration drift check, ISO 30107-3 micro-bench | 5 days | 42 d |
| **🔵 Full parity port: ~8 engineer-weeks (42 days)** | | | |

The 3-week MVP is the right launch bar: **MiniFASNet is +94.7 discrimination
gap, calibrated to weight 5.0× — by itself it carries the published
"Grade C" verdict.** Adding device-boundary, blink and landmark variance
brings in the only other three analyzers with non-trivial measured weight.
The remaining 10 analyzers contribute <30% of fusion mass combined and most
are explicitly suppressed (texture/moire at 0.1×) or disabled (rppg at 0.0×).

---

## 7. Recommended phasing — which analyzers first

In strict order of "weight × discrimination-gap-per-week-of-effort":

1. **MiniFASNet** (weight 5.0, +94.7 gap, 2 days). Ship Day-1.
2. **MediaPipe FaceLandmarker pipeline** (free piggyback on the web-app's
   existing wrapper — needed to feed Blink + Landmark Variance).
3. **Landmark Variance** (weight 2.0, pure arithmetic, 1 day).
4. **Blink (EAR)** (weight 0.5 but extremely strong "no-blinks-after-15s"
   incident signal, 1 day).
5. **Device Boundary** (weight 2.5, the only weight-2.5+ analyzer that
   needs OpenCV.js — gates the OpenCV.js bring-up cost, 2 days).
6. **Micro-tremor** (weight 2.5, ~1 day given the FFT helper from #5
   already in place).
7. **Screen Flicker** (weight 3.0, same FFT helper, 1 day) — high impact for
   any screen attack, complements MiniFASNet well.

That's the **MVP set: 7 analyzers, weight-mass 17.0 of 17.6 (97%)**, in 8
working days of analyzer work. The session-engine+prover layer (Phase 5)
runs in parallel.

Defer to Phase 7+: texture, moire (suppressed in calibration anyway),
screen-replay (whole-frame redundancy with MiniFASNet), background-grid
(proctoring-only), ar-filter (model not trained), rppg (disabled), temporal
(weight 0.3). None of these is a launch-blocker.

---

## 8. Performance projection summary

| Configuration | Per-frame budget | Sustained fps |
|---|---|---|
| Python prod (CPython 3.11, OpenCV CPU, MediaPipe CPU, full 14 analyzers) | ~44 ms | ~22 fps |
| Browser MVP (7 analyzers, ORT WASM EP, MediaPipe GPU delegate, OpenCV.js for Hough only) | ~30–55 ms | **18–30 fps** |
| Browser full parity (14 analyzers, naive — no WebGL Gabor, no Worker) | ~110–180 ms | 6–9 fps |
| Browser full parity + WebGL Gabor + Web Worker | ~50–80 ms | 12–20 fps |

The MVP comfortably meets the camera framerate (typical webcam 24–30 fps)
on a commodity laptop. Mobile (mid-range Android) is roughly half that —
the MVP will run at ~10–15 fps on phones, which is still usable for the
session-engine warmup model (30 frames = 1 s at 30 fps becomes 30 frames =
2 s at 15 fps; the verdict-availability timeline shifts but does not break).

---

## 9. Open questions / unknowns

1. **MiniFASNet ONNX opset coverage on `onnxruntime-web` WASM EP.** The
   model is on disk; we have not yet run it through `ort-web`'s op
   compatibility check. Risk is low (MiniFASNet is a vanilla MobileNetV2
   variant, all ops are in the WASM op set), but warrants a 30-minute
   smoke test on Day 1 of the port. **Mitigation if blocked:** WebGPU EP
   has fewer ops but those it supports are MobileNet-friendly; or
   compile MiniFASNet to ORT-format with `python -m onnxruntime.tools.convert_onnx_models_to_ort`.

2. **CLAHE in OpenCV.js.** The Python uses `cv2.createCLAHE(clipLimit=2.0,
   tileGridSize=(8, 8))` in three places (moire, screen-replay, gates). I
   confirmed CLAHE is in OpenCV.js 4.x but did not benchmark its cost.
   **Mitigation:** if it's >5 ms per call, fall back to a single global
   `equalizeHist` (~1 ms) — the two-grid CLAHE is a quality refinement,
   not a correctness requirement.

3. **Variable browser framerate impact on FFT analyzers.** The Python code
   self-measures FPS via `_frame_times` deque (5 occurrences across rppg,
   micro-tremor, flicker), so the math adapts. But browsers throttle
   inactive tabs to 1 fps — the buffer fills with seconds-old samples,
   ruining the FFT bands. **Mitigation:** detect `document.visibilityState !== 'visible'`
   and reset the FFT buffer on `visibilitychange`. ~10 lines of TS.

4. **Multi-face support cost on browsers.** `@mediapipe/tasks-vision`
   FaceLandmarker supports `numFaces > 1` (the web-app uses `numFaces: 5`),
   but the per-frame cost scales linearly. For proctoring use-cases the
   feature is real; for ID-verification it's typically `numFaces: 1`. The
   current Python uses 5 — the port should expose this as a config option.

5. **Bundle size budget.** OpenCV.js (9 MB), `@mediapipe/tasks-vision` (5
   MB WASM), `onnxruntime-web` (5 MB WASM), MiniFASNet ONNX (1.7 MB),
   FaceLandmarker `.task` (10 MB). Total ~30 MB on first cold load. All
   are fingerprinted and cacheable forever, but the **first-paint cost is
   real** for marketing pages. **Mitigation:** the API can lazy-load
   ("user clicks 'Verify identity'" → fetch begins; show a 5-second
   spinner) which matches the established `<VerifyButton>` UX in the
   `verify-app` package. Document this clearly in the embed snippet.

6. **iOS Safari quirks.** `getUserMedia` on iOS Safari requires user
   gesture, has stricter `OffscreenCanvas` support than Chrome, and
   sometimes silently downsamples to 480p. Worth a Day-1 smoke test on
   iOS 17+ — but this is the same risk the FIVUCSAS production verify
   widget already absorbs, so it's a known-known.

7. **Calibration drift between Python and TS.** The 2026-05-02 calibration
   (the +94.7 / +19.2 / +9.6 gaps cited in `README.md`) was measured
   against ground-truth captures using the Python pipeline. Subtle
   differences in JPEG decoding, BGR↔RGB ordering, OpenCV.js's `INTER_AREA`
   resize implementation, and ORT-Web's float32 quantisation could shift
   thresholds by 1–3 score points per analyzer. **Mitigation:** Phase 10
   (test parity) should re-run the same ground-truth captures through
   the TS pipeline and either confirm the original thresholds hold or
   re-publish a "browser calibration" config in `config.json` alongside
   the existing `config.yaml`.

---

## 10. Recommendation

**Greenlight the 3-week MVP.** Effort is bounded, every dependency has a
known browser path, and the largest reuse-able asset (the web-app's
BiometricEngine) is already in production at `app.fivucsas.com`. The MVP
delivers >97% of fusion mass and includes all four production-deployed
attack-detection signals — it is functionally equivalent to the current
Python pipeline for the four scenarios in the README's published ISO
30107-3 table.

**Do not block on the full 8-week parity port.** The Phase 7+ analyzers are
either calibrated to <0.1× weight, disabled outright, or guard
research-track scenarios (AR-filter, deepfake injection) where the
underlying ML model has not yet been trained.

**Single highest-leverage technical decision:** ship OpenCV.js for the MVP
(buy time), and add a Phase 9 hot-op replacement pass (Laplacian, Gaussian,
cvtColor) only if measured first-paint cost crosses an SLA budget. The
resulting bundle-size delta is ~7 MB but the engineering-time savings are
~3–5 days.

---

*Generated 2026-05-09 from `master @ 5b9aa6b`. This document is read-only;
no source files were modified during the audit.*
