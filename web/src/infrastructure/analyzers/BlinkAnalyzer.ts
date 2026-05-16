// Port of src/infrastructure/analyzers/blink_analyzer.py:1-253
//
// Blink — fusion weight 0.5, but high-impact for incident detection.
//
// Eye Aspect Ratio (Soukupova & Cech, 2016):
//   EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
//   Open eye: 0.25-0.35    Closed eye: < 0.21
//
// Differences from the Python source:
//   * The Python implementation runs its OWN MediaPipe FaceLandmarker —
//     here we reuse the landmarks that MediaPipeFaceDetector already
//     emits onto FaceROI.landmarks. The Python `set_frame()` /
//     `_ensure_init()` plumbing is therefore omitted.
//   * The deque buffer is a fixed-length Float32Array (length 90 = 3s @ 30fps).

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { RingBuffer } from "../../domain/session";

// Verbatim from Python — MediaPipe FaceMesh landmark indices for eyes.
const RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144] as const;
const LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380] as const;

interface BlinkState {
  ear_history: RingBuffer<number>;
  blink_count: number;
  eyes_closed_frames: number;
  last_blink_frame: number;
  frame_count: number;
}

export interface BlinkAnalyzerOptions {
  /** EAR threshold below which the eye is considered closed. Default 0.20. */
  earThreshold?: number;
  /** Reopen threshold to validate a blink. Default 0.22. */
  reopenThreshold?: number;
  /** Frames the eye must remain closed. Default 2 (matches CONSECUTIVE_FRAMES). */
  consecutiveFrames?: number;
  /** Min frames between blinks (anti-noise). Default 6. */
  minOpenBetween?: number;
  /** Frames before scoring (else neutral 50.0). Default 30 (= 1s @ 30fps). */
  warmupFrames?: number;
  /** Expected blinks/min for a real person. Default 17. */
  normalBlinkRate?: number;
  /** Per-track ear_history ring buffer length. Default 90 (3s @ 30fps). */
  historyLen?: number;
}

export class BlinkAnalyzer implements IFaceAnalyzer {
  readonly name = "blink";

  private readonly earThreshold: number;
  private readonly reopenThreshold: number;
  private readonly consecutiveFrames: number;
  private readonly minOpenBetween: number;
  private readonly warmupFrames: number;
  private readonly normalBlinkRate: number;
  private readonly historyLen: number;

  private states: Map<number, BlinkState> = new Map();

  constructor(options: BlinkAnalyzerOptions = {}) {
    this.earThreshold = options.earThreshold ?? 0.20;
    this.reopenThreshold = options.reopenThreshold ?? 0.22;
    // Default 1 (was 2 to match the Python source at 30 fps). A typical
    // human blink lasts 100-400ms — at 30 fps that's 3-12 frames, at 4-5 fps
    // (mobile Android Chrome) it collapses to 1 frame, and a 2-frame
    // requirement silently dropped most real blinks on phones. The reopen
    // + min_open_between guards are still in place to filter landmark
    // jitter, so requiring 1 closed frame is safe.
    this.consecutiveFrames = options.consecutiveFrames ?? 1;
    this.minOpenBetween = options.minOpenBetween ?? 6;
    // Python uses 45; the task spec asks for 30 (matches WARMUP_FRAMES).
    this.warmupFrames = options.warmupFrames ?? 30;
    this.normalBlinkRate = options.normalBlinkRate ?? 17.0;
    this.historyLen = options.historyLen ?? 90;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    const fid = face.face_id;
    let state = this.states.get(fid);
    if (!state) {
      state = {
        ear_history: new RingBuffer<number>(this.historyLen),
        blink_count: 0,
        eyes_closed_frames: 0,
        last_blink_frame: 0,
        frame_count: 0,
      };
      this.states.set(fid, state);
    }
    state.frame_count += 1;

    const lm = face.landmarks;
    if (!lm || lm.length < 388 * 2) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "no_landmarks" },
        performance.now() - start,
      );
    }

    const leftEar = computeEar(lm, LEFT_EYE_INDICES);
    const rightEar = computeEar(lm, RIGHT_EYE_INDICES);
    const avgEar = (leftEar + rightEar) / 2.0;
    state.ear_history.append(avgEar);

    // V-shape blink validation: closed for ≥CONSECUTIVE_FRAMES, then a
    // proper reopen above REOPEN_THRESHOLD, anti-debounce via MIN_OPEN_BETWEEN.
    if (avgEar < this.earThreshold) {
      state.eyes_closed_frames += 1;
    } else {
      if (
        state.eyes_closed_frames >= this.consecutiveFrames &&
        avgEar >= this.reopenThreshold &&
        state.frame_count - state.last_blink_frame >= this.minOpenBetween
      ) {
        state.blink_count += 1;
        state.last_blink_frame = state.frame_count;
      }
      state.eyes_closed_frames = 0;
    }

    if (state.frame_count < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        {
          warmup: true,
          frames: state.frame_count,
          ear: round(avgEar, 3),
          blinks: state.blink_count,
        },
        performance.now() - start,
      );
    }

    // Score: blink presence relative to expected rate (assume 30 fps).
    const durationSec = state.frame_count / 30.0;
    const expectedBlinks = this.normalBlinkRate * (durationSec / 60.0);

    let score: number;
    if (state.blink_count === 0) {
      if (durationSec > 5.0) score = 10.0;
      else if (durationSec > 3.0) score = 25.0;
      else score = 40.0;
    } else if (state.blink_count >= expectedBlinks * 0.3) {
      score = 90.0;
    } else {
      const ratio = state.blink_count / Math.max(expectedBlinks, 0.1);
      score = 40.0 + ratio * 50.0;
    }
    score = Math.max(0.0, Math.min(100.0, score));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        ear: round(avgEar, 3),
        blinks: state.blink_count,
        blink_rate_per_min: round(
          state.blink_count / Math.max(durationSec / 60, 0.01),
          1,
        ),
        duration_sec: round(durationSec, 1),
        eyes_open: avgEar >= this.earThreshold,
      },
      performance.now() - start,
    );
  }

  /** Latest blink count for a tracked face. */
  getBlinkCount(face_id: number): number {
    return this.states.get(face_id)?.blink_count ?? 0;
  }

  reset(): void {
    this.states.clear();
  }
}

function computeEar(
  landmarks: Float32Array,
  eyeIndices: ReadonlyArray<number>,
): number {
  // Pull 6 (x,y) points.
  const pts: Array<[number, number]> = [];
  for (const idx of eyeIndices) {
    pts.push([landmarks[2 * idx], landmarks[2 * idx + 1]]);
  }
  const v1 = dist(pts[1], pts[5]);
  const v2 = dist(pts[2], pts[4]);
  const h = dist(pts[0], pts[3]);
  if (h < 1e-6) return 0.3; // neutral
  return (v1 + v2) / (2.0 * h);
}

function dist(a: [number, number], b: [number, number]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
