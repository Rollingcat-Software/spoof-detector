# @rollingcat/spoof-detector

Browser/TypeScript port of the [spoof-detector](https://github.com/rollingcat/spoof-detector)
session-based anti-spoof pipeline. **Phases 1 and 2** ship 6 of the
fuser's analyzers, covering ~95% of its weight mass:

| Analyzer | Weight | Phase | Source line range |
|---|---|---|---|
| `minifasnet` | 5.0 | 1 | `src/infrastructure/analyzers/minifasnet_analyzer.py` |
| `screen_flicker` | 3.0 | 2 | `src/infrastructure/analyzers/screen_flicker_analyzer.py:1-141` |
| `device_boundary` | 2.5 | 2 | `src/infrastructure/analyzers/device_boundary_analyzer.py:1-189` |
| `micro_tremor` | 2.5 | 2 | `src/infrastructure/analyzers/micro_tremor_analyzer.py:1-163` |
| `landmark_variance` | 2.0 | 2 | `src/infrastructure/analyzers/landmark_variance_analyzer.py:1-165` |
| `blink` | 0.5 | 2 | `src/infrastructure/analyzers/blink_analyzer.py:1-253` |

The multi-class fuser and session-state engine are also ported. Phase 3+
will port the lower-weight analyzers (texture, moire, screen_replay, rppg,
ar_filter, background_grid) plus the LivenessProver.

> Status: source pass — files compile, but the demo requires
> `npm install` to fetch `onnxruntime-web` and `@mediapipe/tasks-vision`,
> and the MiniFASNet ONNX model must be placed at `public/models/minifasnet_v2.onnx`
> (see "Model assets" below).

## Why use this

- Runs entirely in the browser. No server round-trip.
- 1.7 MB ONNX model + thin TypeScript wrapper. No PyTorch / TensorFlow.
- Same multi-class taxonomy and fusion weights as the Python pipeline,
  so JSON snapshots are interoperable.
- Session engine (`WARMING_UP → ANALYZING → CONCLUDED`) accumulates
  evidence across frames — single-frame false positives don't decide
  the verdict.

## Installation

```bash
npm install @rollingcat/spoof-detector \
            onnxruntime-web@^1.18.0 \
            @mediapipe/tasks-vision@^0.10.18
```

## Quick start

```ts
import { createSpoofDetector } from "@rollingcat/spoof-detector";

const video = document.querySelector<HTMLVideoElement>("video")!;
const stream = await navigator.mediaDevices.getUserMedia({ video: true });
video.srcObject = stream;
await video.play();

const detector = await createSpoofDetector({
  miniFasNetModelUrl: "/models/minifasnet_v2.onnx",
  faceLandmarkerTaskUrl: "/models/face_landmarker.task",
});

const canvas = document.createElement("canvas");
canvas.width = 640;
canvas.height = 480;
const ctx = canvas.getContext("2d")!;

function loop() {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  detector.analyzeFrame(canvas).then(() => {
    const verdict = detector.getVerdict();
    document.getElementById("verdict")!.textContent = verdict.summary;
    requestAnimationFrame(loop);
  });
}
loop();
```

## Public API

```ts
function createSpoofDetector(options?: SpoofDetectorOptions): Promise<SpoofDetector>;

class SpoofDetector {
  analyzeFrame(input: HTMLCanvasElement | ImageData): Promise<FrameAnalysis>;
  getVerdict(): SessionVerdict;
  conclude(): SessionVerdict;
  reset(): void;
}
```

See `src/index.ts` for the full type signatures and `src/domain/models.ts`
for the data shapes.

## Model assets

| File | Source | Size | Where to put it |
|------|--------|------|-----------------|
| `minifasnet_v2.onnx` | `~/.uniface/models/` (auto-downloaded by the `uniface` Python pkg on first run) | 1.7 MB | `public/models/minifasnet_v2.onnx` |
| `face_landmarker.task` | repo root: `spoof-detector/models/` | ~10 MB | `public/models/face_landmarker.task` |

The build does **not** bundle these — they're served as static assets
by the host application so they can be cached with
`Cache-Control: public, max-age=31536000, immutable`.

A `fetch-models.mjs` helper script will be added in Phase 2 (mirrors
the pattern used by the FIVUCSAS web-app's BiometricEngine).

## Phasing

- **Phase 1:** MiniFASNet + multi-class fuser + session engine + MediaPipe FaceLandmarker wrapper.
- **Phase 2 (this PR):** Landmark Variance + Blink (EAR) + Device Boundary (Sobel + hand-rolled Hough) + Micro-tremor (hand-rolled DFT) + Screen Flicker (hand-rolled DFT). 95% cumulative weight coverage.
- **Phase 3:** Texture / Moire / Screen Replay / rPPG / AR Filter / Background Grid (suppressed-weight analyzers — port for completeness).
- **Phase 4:** Liveness Prover + full session-engine parity tests.

### Phase-2 implementation notes

- All Phase-2 analyzers are pure TypeScript with no extra deps. **No `npm install` step is required** — `onnxruntime-web` and `@mediapipe/tasks-vision` from Phase 1 still cover the runtime.
- `DeviceBoundaryAnalyzer` ships a documented ~10% accuracy approximation vs the OpenCV reference (Canvas2D + Sobel + axis-aligned line scan instead of `cv2.HoughLinesP` + `cv2.findContours`). The opt-in to lazy-load OpenCV.js is on the Phase 3 backlog.
- `MicroTremorAnalyzer` and `ScreenFlickerAnalyzer` use a hand-rolled DFT instead of `numpy.fft.rfft`. At the default `historyLen=30` the DFT is < 1 ms; callers who need a finer FFT can lazy-load `fft.js` and replace `bandPowerRatio()`.
- All analyzers honour the same `score_history`-style buffer naming as the Python source, so JSON snapshots remain interoperable.

See `../SPOOF_DETECTOR_BROWSER_READINESS.md` (in the repo root) for the
full readiness audit, op-by-op port mapping and effort estimates.

## Build & test

```bash
npm install
npm run typecheck
npm run build       # → dist/spoof-detector.{js,umd.cjs,d.ts}
npm test
```

## Repository

- **Source repo:** https://github.com/rollingcat/spoof-detector
- **Python pipeline:** [`../src/`](../src/) — this package mirrors the
  `domain/` + `infrastructure/fusion/` + `application/session_engine.py`
  modules.
- **Readiness audit:** [`../SPOOF_DETECTOR_BROWSER_READINESS.md`](../SPOOF_DETECTOR_BROWSER_READINESS.md)

## License

Apache-2.0, same as the Python pipeline.
