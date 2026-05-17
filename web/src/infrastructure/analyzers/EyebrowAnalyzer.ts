// EyebrowAnalyzer — TS-only addition (no Python source).
//
// Scores eyebrow motion from MediaPipe FaceLandmarker blendshape outputs.
// Five ARKit categories cover the brow region:
//
//   browInnerUp        — both inner brows raise (surprise, curiosity)
//   browDownLeft       — left brow lowers (concentration, anger)
//   browDownRight      — right brow lowers
//   browOuterUpLeft    — left outer brow raises
//   browOuterUpRight   — right outer brow raises
//
// Per frame we collapse them to a single mean-activation scalar and store
// it in a 90-frame ring buffer. The 0–100 score is the stddev across that
// rolling window, scaled and clamped. A static printed photo holds brow
// blendshapes near constant → score → 0. A live face naturally cycles
// brow activations during talking / thinking / surprise → score → high.
//
// Falls back to score 50 (neutral) until the buffer has at least
// `warmupFrames` samples, mirroring the BlinkAnalyzer warm-up contract.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { RingBuffer } from "../../domain/session";

const BROW_BLENDSHAPES = [
  "browInnerUp",
  "browDownLeft",
  "browDownRight",
  "browOuterUpLeft",
  "browOuterUpRight",
] as const;

export interface EyebrowAnalyzerOptions {
  /** Ring buffer length (90 = 3 s @ 30 fps; ~13 s @ 7 fps). */
  historyLen?: number;
  /** Frames before scoring (else neutral 50). */
  warmupFrames?: number;
  /**
   * stddev-to-score gain. A natural session shows brow stddev ~0.04–0.08
   * across the buffer; multiplying by 1500 maps the upper end of natural
   * variation to ~100. Calibrated against the 2026-05-17 Brave Mobile
   * trace where landmark_variance.forehead_var was 19.
   */
  gain?: number;
}

interface EyebrowState {
  history: RingBuffer<number>;
  lastActivation: number;
}

export class EyebrowAnalyzer implements IFaceAnalyzer {
  readonly name = "eyebrow_motion";

  private readonly historyLen: number;
  private readonly warmupFrames: number;
  private readonly gain: number;
  private states: Map<number, EyebrowState> = new Map();

  constructor(options: EyebrowAnalyzerOptions = {}) {
    this.historyLen = options.historyLen ?? 90;
    this.warmupFrames = options.warmupFrames ?? 30;
    this.gain = options.gain ?? 1500;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    let state = this.states.get(face.face_id);
    if (!state) {
      state = {
        history: new RingBuffer<number>(this.historyLen),
        lastActivation: 0,
      };
      this.states.set(face.face_id, state);
    }

    if (!face.blendshapes) {
      // Blendshapes weren't requested or model didn't emit them — neutral.
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_blendshapes" },
        performance.now() - start,
      );
    }

    // Sum then average the 5 brow blendshapes. Each is in [0, 1].
    let sum = 0;
    let n = 0;
    const per: Record<string, number> = {};
    for (const key of BROW_BLENDSHAPES) {
      const v = face.blendshapes.get(key);
      if (typeof v === "number") {
        sum += v;
        n += 1;
        per[key] = round(v, 3);
      }
    }
    if (n === 0) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_brow_blendshapes" },
        performance.now() - start,
      );
    }
    const activation = sum / n;
    state.history.append(activation);
    state.lastActivation = activation;

    if (state.history.length < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50,
        {
          warming: true,
          frames: state.history.length,
          activation: round(activation, 3),
          per_blendshape: per,
        },
        performance.now() - start,
      );
    }

    const arr = state.history.toArray();
    let mean = 0;
    for (const v of arr) mean += v;
    mean /= arr.length;
    let sse = 0;
    for (const v of arr) {
      const d = v - mean;
      sse += d * d;
    }
    const std = Math.sqrt(sse / arr.length);
    const score = Math.max(0, Math.min(100, std * this.gain));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        activation: round(activation, 3),
        mean: round(mean, 3),
        std: round(std, 4),
        frames: arr.length,
        per_blendshape: per,
      },
      performance.now() - start,
    );
  }

  /** Drop all per-face state. */
  reset(): void {
    this.states.clear();
  }
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
