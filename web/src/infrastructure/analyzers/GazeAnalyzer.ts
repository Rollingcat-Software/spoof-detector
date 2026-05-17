// GazeAnalyzer — TS-only addition (no Python source).
//
// MediaPipe FaceLandmarker exposes eight per-eye gaze blendshapes:
//
//   eyeLookInLeft / eyeLookInRight     (nasal)
//   eyeLookOutLeft / eyeLookOutRight   (temporal)
//   eyeLookUpLeft / eyeLookUpRight     (superior)
//   eyeLookDownLeft / eyeLookDownRight (inferior)
//
// We collapse them into a single 2D gaze vector per frame:
//
//   gaze_x = (eyeLookInLeft - eyeLookOutLeft + eyeLookOutRight - eyeLookInRight) / 2
//   gaze_y = (eyeLookUpLeft + eyeLookUpRight - eyeLookDownLeft - eyeLookDownRight) / 2
//
// Each component lives in [-1, 1]. The score has two contributions:
//
//   1. Spatial coverage — stddev of the gaze vector across the rolling
//      window. A static photo holds gaze fixed → near zero. A live user
//      shifts gaze even during steady eye contact (~3 saccades/sec is the
//      human baseline, Rayner 1998).
//   2. Saccade count — frame-to-frame jumps > saccadeThreshold. A live
//      session accumulates saccades; an animated avatar typically renders
//      smooth-pursuit only and is detectable by a low saccade rate.
//
// Score = clamp01(0.7 * std_score + 0.3 * saccade_score) * 100

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { RingBuffer } from "../../domain/session";

export interface GazeAnalyzerOptions {
  /** Rolling window length (90 = 3 s @ 30 fps). */
  historyLen?: number;
  /** Frames before scoring (else neutral 50). */
  warmupFrames?: number;
  /** stddev-to-score gain. Natural sessions cluster around 0.08–0.15. */
  stdGain?: number;
  /** Frame-to-frame delta above this counts as a saccade. */
  saccadeThreshold?: number;
}

interface GazeState {
  history: RingBuffer<[number, number]>;
  lastVec: [number, number];
  saccadeCount: number;
  framesSeen: number;
}

const KEYS = [
  "eyeLookInLeft",
  "eyeLookInRight",
  "eyeLookOutLeft",
  "eyeLookOutRight",
  "eyeLookUpLeft",
  "eyeLookUpRight",
  "eyeLookDownLeft",
  "eyeLookDownRight",
] as const;

export class GazeAnalyzer implements IFaceAnalyzer {
  readonly name = "gaze";

  private readonly historyLen: number;
  private readonly warmupFrames: number;
  private readonly stdGain: number;
  private readonly saccadeThreshold: number;
  private states: Map<number, GazeState> = new Map();

  constructor(options: GazeAnalyzerOptions = {}) {
    this.historyLen = options.historyLen ?? 90;
    this.warmupFrames = options.warmupFrames ?? 30;
    this.stdGain = options.stdGain ?? 8;
    this.saccadeThreshold = options.saccadeThreshold ?? 0.05;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    let state = this.states.get(face.face_id);
    if (!state) {
      state = {
        history: new RingBuffer<[number, number]>(this.historyLen),
        lastVec: [0, 0],
        saccadeCount: 0,
        framesSeen: 0,
      };
      this.states.set(face.face_id, state);
    }

    if (!face.blendshapes) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_blendshapes" },
        performance.now() - start,
      );
    }

    // Pull all 8 axes (missing → 0). Reject the frame entirely if every
    // key is missing — MediaPipe didn't produce gaze data.
    let presentCount = 0;
    const v: Record<string, number> = {};
    for (const k of KEYS) {
      const x = face.blendshapes.get(k);
      if (typeof x === "number") {
        v[k] = x;
        presentCount += 1;
      } else {
        v[k] = 0;
      }
    }
    if (presentCount === 0) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_gaze_blendshapes" },
        performance.now() - start,
      );
    }

    const gx =
      (v.eyeLookInLeft -
        v.eyeLookOutLeft +
        v.eyeLookOutRight -
        v.eyeLookInRight) /
      2;
    const gy =
      (v.eyeLookUpLeft +
        v.eyeLookUpRight -
        v.eyeLookDownLeft -
        v.eyeLookDownRight) /
      2;

    state.framesSeen += 1;
    if (state.framesSeen > 1) {
      const dx = gx - state.lastVec[0];
      const dy = gy - state.lastVec[1];
      const delta = Math.hypot(dx, dy);
      if (delta > this.saccadeThreshold) state.saccadeCount += 1;
    }
    state.lastVec = [gx, gy];
    state.history.append([gx, gy]);

    if (state.history.length < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50,
        {
          warming: true,
          frames: state.history.length,
          gaze_x: round(gx, 3),
          gaze_y: round(gy, 3),
        },
        performance.now() - start,
      );
    }

    const arr = state.history.toArray();
    let mx = 0;
    let my = 0;
    for (const [x, y] of arr) {
      mx += x;
      my += y;
    }
    mx /= arr.length;
    my /= arr.length;
    let ssx = 0;
    let ssy = 0;
    for (const [x, y] of arr) {
      const dx = x - mx;
      const dy = y - my;
      ssx += dx * dx;
      ssy += dy * dy;
    }
    const stdX = Math.sqrt(ssx / arr.length);
    const stdY = Math.sqrt(ssy / arr.length);
    const std = Math.hypot(stdX, stdY);

    const stdScore = Math.max(0, Math.min(1, std * this.stdGain));
    // Saccades per second across the window, normalized to the
    // ~3 saccades/sec human baseline (= 1.0 saturation).
    const saccadeRate = state.saccadeCount / Math.max(1, arr.length / 30);
    const saccadeScore = Math.max(0, Math.min(1, saccadeRate / 3.0));
    const score = (0.7 * stdScore + 0.3 * saccadeScore) * 100;

    return makeAnalyzerResult(
      this.name,
      score,
      {
        gaze_x: round(gx, 3),
        gaze_y: round(gy, 3),
        std_x: round(stdX, 4),
        std_y: round(stdY, 4),
        saccade_count: state.saccadeCount,
        saccade_rate_per_sec: round(saccadeRate, 2),
        frames: arr.length,
      },
      performance.now() - start,
    );
  }

  reset(): void {
    this.states.clear();
  }
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
