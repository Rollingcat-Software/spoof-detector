// ExpressionDynamicsAnalyzer — TS-only addition (no Python source).
//
// Acts as a passive emotion proxy without shipping a full emotion
// classifier. Reads ~12 mouth / cheek / nose blendshapes that change
// during natural expression (smile, frown, dimple, cheek puff/squint,
// nose sneer, lip stretch/press). Per frame, takes the sum across all
// tracked categories and stores in a rolling buffer. Score is the
// per-buffer stddev × gain — high when expression is animating,
// low when face is held in a single fixed pose.
//
// The details surface the top-3 most-active blendshape names so the
// downstream JSON (and amispoof Copy / Download surfaces) can show
// "currently smiling/squinting/etc." without forking a dedicated
// emotion-recognition model.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { RingBuffer } from "../../domain/session";

const TRACKED = [
  "mouthSmileLeft",
  "mouthSmileRight",
  "mouthFrownLeft",
  "mouthFrownRight",
  "mouthDimpleLeft",
  "mouthDimpleRight",
  "mouthPressLeft",
  "mouthPressRight",
  "mouthStretchLeft",
  "mouthStretchRight",
  "cheekSquintLeft",
  "cheekSquintRight",
  "cheekPuff",
  "noseSneerLeft",
  "noseSneerRight",
] as const;

export interface ExpressionDynamicsAnalyzerOptions {
  /** Rolling window length (90 = 3 s @ 30 fps). */
  historyLen?: number;
  /** Frames before scoring (else neutral 50). */
  warmupFrames?: number;
  /** stddev-to-score gain. Calibrated for natural micro-expression. */
  gain?: number;
}

interface ExprState {
  history: RingBuffer<number>;
  lastValues: Record<string, number>;
}

export class ExpressionDynamicsAnalyzer implements IFaceAnalyzer {
  readonly name = "expression_dynamics";

  private readonly historyLen: number;
  private readonly warmupFrames: number;
  private readonly gain: number;
  private states: Map<number, ExprState> = new Map();

  constructor(options: ExpressionDynamicsAnalyzerOptions = {}) {
    this.historyLen = options.historyLen ?? 90;
    this.warmupFrames = options.warmupFrames ?? 30;
    this.gain = options.gain ?? 400;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    let state = this.states.get(face.face_id);
    if (!state) {
      state = {
        history: new RingBuffer<number>(this.historyLen),
        lastValues: {},
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

    let total = 0;
    let present = 0;
    const values: Record<string, number> = {};
    for (const k of TRACKED) {
      const v = face.blendshapes.get(k);
      if (typeof v === "number") {
        values[k] = v;
        total += v;
        present += 1;
      }
    }
    if (present === 0) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_expression_blendshapes" },
        performance.now() - start,
      );
    }
    state.lastValues = values;
    state.history.append(total);

    // Top-3 active expressions this frame (by blendshape value).
    const top = Object.entries(values)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([k, v]) => ({ blendshape: k, value: round(v, 3) }));

    if (state.history.length < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50,
        {
          warming: true,
          frames: state.history.length,
          total: round(total, 3),
          top_active: top,
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
        total: round(total, 3),
        mean: round(mean, 3),
        std: round(std, 4),
        frames: arr.length,
        top_active: top,
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
