// Port of src/infrastructure/analyzers/landmark_variance_analyzer.py:1-165
//
// Landmark Variance — fusion weight 2.0.
//
// Tracks all 478 MediaPipe FaceLandmarker points over time. Real faces
// show natural variance from micro-movements, blinks, and breathing.
// Photos show near-zero variance after global-translation removal.
//
// Implementation notes:
//   * No external deps. Pure arithmetic over the (T, 478, 2) ring buffer.
//   * The Python source ingests landmarks via a side channel
//     (`set_landmarks(np.ndarray)`); here we read them directly off the
//     FaceROI.landmarks Float32Array, which MediaPipeFaceDetector already
//     populates. A `setLandmarks()` shim is preserved for API parity.
//   * Region indices (REGION_*_EYE, REGION_MOUTH, REGION_FOREHEAD) are
//     copied verbatim from the Python source.
//   * History buffer length: 30 frames (matches the Phase 1 task spec).
//     The Python original uses 60. This is a conservative trade-off
//     for memory at higher face counts.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";

// MediaPipe face mesh indices — verbatim from landmark_variance_analyzer.py:32-35.
const REGION_LEFT_EYE: number[] = [
  ...range(33, 42),
  133, 144, 153, 154, 155, 157, 158, 159, 160, 161,
];
const REGION_RIGHT_EYE: number[] = [
  ...range(362, 371),
  263, 373, 380, 381, 382, 384, 385, 386, 387, 388,
];
const REGION_MOUTH: number[] = [
  ...range(61, 68),
  ...range(291, 298),
  0, 13, 14, 17, 37, 39, 40, 78, 80, 81, 82, 87, 88, 95, 178, 181, 267, 269,
  270, 308, 310, 311, 312, 317, 318, 324, 402, 405,
];
const REGION_FOREHEAD: number[] = [
  10, 67, 69, 104, 108, 109, 151, 338, 337, 297, 299, 333,
];

const NUM_LANDMARKS = 478;

interface LandmarkHistory {
  /** Each entry: Float32Array length 2*N (xy interleaved). */
  frames: Float32Array[];
  frame_count: number;
}

export interface LandmarkVarianceOptions {
  /** Frames-per-track ring buffer length. Default 30 (matches WARMUP_FRAMES). */
  historyLen?: number;
  /** Frames before we score (else neutral 50.0). Default 15. */
  warmupFrames?: number;
}

export class LandmarkVarianceAnalyzer implements IFaceAnalyzer {
  readonly name = "landmark_variance";

  // Verbatim from Python (LandmarkVarianceAnalyzer.WARMUP_FRAMES = 15).
  static readonly WARMUP_FRAMES = 15;
  static readonly ZERO_VAR_THRESHOLD = 0.5;

  private readonly historyLen: number;
  private readonly warmupFrames: number;
  private histories: Map<number, LandmarkHistory> = new Map();
  private currentLandmarks: Float32Array | null = null;

  constructor(options: LandmarkVarianceOptions = {}) {
    this.historyLen = options.historyLen ?? 30;
    this.warmupFrames = options.warmupFrames ?? LandmarkVarianceAnalyzer.WARMUP_FRAMES;
  }

  /** Optional API-parity hook with the Python source. The orchestrator can
   *  also let the analyzer pull landmarks off `face.landmarks` directly. */
  setLandmarks(landmarks: Float32Array | null): void {
    this.currentLandmarks = landmarks;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    const fid = face.face_id;

    let hist = this.histories.get(fid);
    if (!hist) {
      hist = { frames: [], frame_count: 0 };
      this.histories.set(fid, hist);
    }
    hist.frame_count += 1;

    // Pull landmarks from face.landmarks (preferred) or from setLandmarks().
    const landmarks = face.landmarks ?? this.currentLandmarks;
    if (!landmarks || landmarks.length < 468 * 2) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "no_landmarks" },
        performance.now() - start,
      );
    }

    // Store xy positions (Float32Array of length 2*N).
    // We slice to a safe upper bound to keep frame shapes consistent.
    const want = Math.min(NUM_LANDMARKS, Math.floor(landmarks.length / 2));
    const xy = landmarks.slice(0, want * 2);
    hist.frames.push(xy);
    if (hist.frames.length > this.historyLen) hist.frames.shift();

    if (hist.frame_count < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { warmup: true, frames: hist.frame_count },
        performance.now() - start,
      );
    }

    // === Compute variance across stored frames (after centroid removal). ===
    const T = hist.frames.length;
    const N = want;

    // Per-frame centroid (translation removal).
    const centroidX = new Float32Array(T);
    const centroidY = new Float32Array(T);
    for (let t = 0; t < T; t++) {
      const f = hist.frames[t];
      let sx = 0;
      let sy = 0;
      for (let i = 0; i < N; i++) {
        sx += f[2 * i];
        sy += f[2 * i + 1];
      }
      centroidX[t] = sx / N;
      centroidY[t] = sy / N;
    }

    // Per-landmark mean (after centroid removal).
    // We could materialize the (T, N, 2) normalized array but it's wasteful.
    // Two-pass mean+variance keeps memory at O(N) per call.
    const meanX = new Float32Array(N);
    const meanY = new Float32Array(N);
    for (let t = 0; t < T; t++) {
      const f = hist.frames[t];
      const cx = centroidX[t];
      const cy = centroidY[t];
      for (let i = 0; i < N; i++) {
        meanX[i] += f[2 * i] - cx;
        meanY[i] += f[2 * i + 1] - cy;
      }
    }
    for (let i = 0; i < N; i++) {
      meanX[i] /= T;
      meanY[i] /= T;
    }

    // Per-landmark variance (sum across X+Y).
    const perLandmarkTotalVar = new Float32Array(N);
    for (let t = 0; t < T; t++) {
      const f = hist.frames[t];
      const cx = centroidX[t];
      const cy = centroidY[t];
      for (let i = 0; i < N; i++) {
        const dx = (f[2 * i] - cx) - meanX[i];
        const dy = (f[2 * i + 1] - cy) - meanY[i];
        perLandmarkTotalVar[i] += dx * dx + dy * dy;
      }
    }
    for (let i = 0; i < N; i++) perLandmarkTotalVar[i] /= T;

    let overallVar = 0;
    let maxVar = 0;
    for (let i = 0; i < N; i++) {
      overallVar += perLandmarkTotalVar[i];
      if (perLandmarkTotalVar[i] > maxVar) maxVar = perLandmarkTotalVar[i];
    }
    overallVar /= N;

    const eyeVar = meanIndices(perLandmarkTotalVar, [
      ...REGION_LEFT_EYE,
      ...REGION_RIGHT_EYE,
    ]);
    const mouthVar = meanIndices(perLandmarkTotalVar, REGION_MOUTH);
    const foreheadVar = meanIndices(perLandmarkTotalVar, REGION_FOREHEAD);

    // Expression ratio: real faces have higher eye/mouth variance than forehead.
    let expressionRatio: number;
    if (foreheadVar > 0.01) {
      expressionRatio = (eyeVar + mouthVar) / (2.0 * foreheadVar);
    } else {
      expressionRatio = eyeVar < 0.01 ? 0.0 : 10.0;
    }

    // === Score (verbatim port of Python piecewise mapping). ===
    let score: number;
    if (overallVar < LandmarkVarianceAnalyzer.ZERO_VAR_THRESHOLD) {
      score = Math.max(
        0.0,
        10.0 * (overallVar / LandmarkVarianceAnalyzer.ZERO_VAR_THRESHOLD),
      );
    } else if (overallVar < 2.0) {
      score =
        10.0 +
        30.0 *
          ((overallVar - LandmarkVarianceAnalyzer.ZERO_VAR_THRESHOLD) / 1.5);
    } else if (overallVar < 5.0) {
      score = 40.0 + 20.0 * ((overallVar - 2.0) / 3.0);
    } else {
      score = 60.0 + Math.min(40.0, overallVar * 2.0);
    }

    if (expressionRatio > 2.0) {
      score = Math.min(100.0, score + 10.0);
    }

    score = Math.max(0.0, Math.min(100.0, score));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        overall_var: round(overallVar, 4),
        max_var: round(maxVar, 4),
        eye_var: round(eyeVar, 4),
        mouth_var: round(mouthVar, 4),
        forehead_var: round(foreheadVar, 4),
        expression_ratio: round(expressionRatio, 2),
        n_frames: T,
      },
      performance.now() - start,
    );
  }

  /** Reset all per-face buffers (mirrors `_histories.clear()`). */
  reset(): void {
    this.histories.clear();
    this.currentLandmarks = null;
  }
}

function range(startInclusive: number, endExclusive: number): number[] {
  const out: number[] = [];
  for (let i = startInclusive; i < endExclusive; i++) out.push(i);
  return out;
}

function meanIndices(arr: Float32Array, indices: number[]): number {
  if (indices.length === 0) return 0;
  let s = 0;
  let n = 0;
  for (const idx of indices) {
    if (idx >= 0 && idx < arr.length) {
      s += arr[idx];
      n += 1;
    }
  }
  return n > 0 ? s / n : 0;
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
