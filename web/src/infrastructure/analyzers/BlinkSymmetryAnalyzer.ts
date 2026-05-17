// BlinkSymmetryAnalyzer — TS-only addition (no Python source).
//
// Measures the per-frame Pearson correlation between the `eyeBlinkLeft`
// and `eyeBlinkRight` ARKit blendshapes across a rolling 90-frame window.
// Real humans blink synchronously — both eyes track the same EAR curve
// over time, so the two blendshape time series correlate at r ≥ 0.7
// even on noisy mobile-camera footage. Several attack classes desync:
//
//   * Deepfake-injection avatars (DeepFaceLive, FaceSwap) often render
//     the two eyes from a single source and end up with delays / phase
//     skips between left and right.
//   * AR-filter overlays sometimes blink one eye at a time as a "stylized"
//     effect or skip per-eye blendshapes entirely.
//   * Static photos hold both eyes at constant — the analyzer returns
//     neutral 50 (zero variance ⇒ undefined correlation) rather than
//     punishing them; the EyebrowAnalyzer / other motion axes catch that
//     attack class.
//
// Output: score = max(0, corr) × 100 with a min-frames floor at neutral 50.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { RingBuffer } from "../../domain/session";

export interface BlinkSymmetryAnalyzerOptions {
  /** Rolling window length (90 = 3 s @ 30 fps). */
  historyLen?: number;
  /** Frames before scoring (else neutral 50). */
  warmupFrames?: number;
  /**
   * Stddev floor below which we treat both signals as flat and return
   * neutral 50 — correlation of two constant signals is undefined and
   * NaN-out would mislead consumers. Real blinks produce stddev > 0.05
   * even at low fps; this floor protects against the photo case.
   */
  flatStdFloor?: number;
}

interface SymmetryState {
  left: RingBuffer<number>;
  right: RingBuffer<number>;
}

export class BlinkSymmetryAnalyzer implements IFaceAnalyzer {
  readonly name = "blink_symmetry";

  private readonly historyLen: number;
  private readonly warmupFrames: number;
  private readonly flatStdFloor: number;
  private states: Map<number, SymmetryState> = new Map();

  constructor(options: BlinkSymmetryAnalyzerOptions = {}) {
    this.historyLen = options.historyLen ?? 90;
    this.warmupFrames = options.warmupFrames ?? 30;
    this.flatStdFloor = options.flatStdFloor ?? 0.02;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    let state = this.states.get(face.face_id);
    if (!state) {
      state = {
        left: new RingBuffer<number>(this.historyLen),
        right: new RingBuffer<number>(this.historyLen),
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

    const left = face.blendshapes.get("eyeBlinkLeft");
    const right = face.blendshapes.get("eyeBlinkRight");
    if (typeof left !== "number" || typeof right !== "number") {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_blink_blendshapes" },
        performance.now() - start,
      );
    }
    state.left.append(left);
    state.right.append(right);

    if (state.left.length < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50,
        {
          warming: true,
          frames: state.left.length,
          eye_blink_left: round(left, 3),
          eye_blink_right: round(right, 3),
        },
        performance.now() - start,
      );
    }

    const xs = state.left.toArray();
    const ys = state.right.toArray();
    const { corr, stdX, stdY } = pearson(xs, ys);

    // Both signals flat ⇒ correlation undefined. Treat as neutral.
    if (stdX < this.flatStdFloor && stdY < this.flatStdFloor) {
      return makeAnalyzerResult(
        this.name,
        50,
        {
          flat: true,
          corr: null,
          std_left: round(stdX, 4),
          std_right: round(stdY, 4),
          frames: xs.length,
        },
        performance.now() - start,
      );
    }

    // Negative correlation (one eye opens as the other closes) is a
    // strong spoof signal but score is bounded ≥ 0; the proof axis will
    // see this as a zero credit and the no-blink incident path covers
    // the actual attack detection.
    const score = Math.max(0, Math.min(100, corr * 100));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        corr: round(corr, 3),
        std_left: round(stdX, 4),
        std_right: round(stdY, 4),
        eye_blink_left: round(left, 3),
        eye_blink_right: round(right, 3),
        frames: xs.length,
      },
      performance.now() - start,
    );
  }

  reset(): void {
    this.states.clear();
  }
}

function pearson(
  xs: number[],
  ys: number[],
): { corr: number; stdX: number; stdY: number } {
  const n = Math.min(xs.length, ys.length);
  if (n === 0) return { corr: 0, stdX: 0, stdY: 0 };
  let sx = 0;
  let sy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i];
    sy += ys[i];
  }
  const mx = sx / n;
  const my = sy / n;
  let cov = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    cov += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  const stdX = Math.sqrt(sxx / n);
  const stdY = Math.sqrt(syy / n);
  const denom = Math.sqrt(sxx * syy);
  const corr = denom > 1e-12 ? cov / denom : 0;
  return { corr, stdX, stdY };
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
