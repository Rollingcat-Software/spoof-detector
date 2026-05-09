# @rollingcat/spoof-detector

Browser/TypeScript port of the [spoof-detector](https://github.com/rollingcat/spoof-detector)
session-based anti-spoof pipeline. **Phase 1** ships the highest-weight
discriminator (MiniFASNet, fusion weight 5.0 of 17.6) plus the multi-class
fuser and the session-state engine. Subsequent phases will add Landmark
Variance, Blink, Device Boundary, Micro-tremor and Screen Flicker.

> Status: scaffolding pass — files compile, but the demo requires
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

- **Phase 1 (this PR):** MiniFASNet + multi-class fuser + session engine + MediaPipe FaceLandmarker wrapper.
- **Phase 2:** Landmark Variance + Blink (EAR) — both pure arithmetic on the FaceLandmarker output.
- **Phase 3:** Device Boundary (OpenCV.js Hough/Canny) + Micro-tremor + Screen Flicker (FFT-based).
- **Phase 4:** Texture / Moire / Screen Replay / rPPG / AR Filter / Background Grid (suppressed-weight analyzers — port for completeness).
- **Phase 5:** Liveness Prover + full session-engine parity tests.

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
