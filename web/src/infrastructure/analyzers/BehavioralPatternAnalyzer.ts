// BehavioralPatternAnalyzer — TS-only addition (Phase B).
//
// Reuses signals the pipeline already produces (blink count, gaze vector,
// jaw blendshape) but analyzes their *temporal distribution* rather than
// their per-frame values. Three sub-signals are combined:
//
//   1. Blink-interval coefficient of variation (CV)
//      Real humans blink ≈ 15-20/min with substantial irregularity
//      (CV of inter-blink intervals ≈ 0.5-1.0). Looped videos or
//      animated avatars show abnormally regular intervals (CV → 0)
//      or no blinks at all.
//
//   2. Saccade rate per second
//      Human gaze produces ~3 saccades/sec under steady viewing
//      (Rayner 1998). Smooth-pursuit-only avatars score zero.
//      Sourced by re-deriving the gaze vector from eyeLook* blendshapes
//      so this analyzer is self-contained and doesn't depend on
//      GazeAnalyzer being registered or running first.
//
//   3. Temporal entropy of a composite signal
//      Shannon entropy of a quantized rolling buffer of jawOpen +
//      browInnerUp + (blink toggle). Live sessions = high-entropy
//      multi-axis noise. Looped videos = low entropy because the
//      signals repeat.
//
// Final score = 100 × (0.4 × blinkCV + 0.3 × saccadeRate + 0.3 × entropy).

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { RingBuffer } from "../../domain/session";

export interface BehavioralPatternAnalyzerOptions {
  /** Window length for all internal buffers (300 = 10 s @ 30 fps). */
  historyLen?: number;
  /** Frames before scoring (else neutral 50). */
  warmupFrames?: number;
  /** Per-frame delta above which a gaze movement counts as a saccade. */
  saccadeThreshold?: number;
  /** Quantization bins for the entropy computation (8 → 3-bit code). */
  entropyBins?: number;
}

interface BehaviorState {
  /** Most recent absolute blink count (we derive events from deltas). */
  lastBlinkCount: number;
  /** Frame index when the latest blink event was observed (for IBI). */
  lastBlinkFrameIdx: number;
  /** Inter-blink intervals in *frames*. Converted to seconds via fps. */
  ibi: RingBuffer<number>;
  /** Recent gaze vectors for saccade counting. */
  gazeHistory: RingBuffer<[number, number]>;
  saccadeCount: number;
  /** Composite signal samples (jawOpen + browInnerUp + blink toggle). */
  composite: RingBuffer<number>;
  /** Frame counter for IBI conversion + warmup gating. */
  framesSeen: number;
  /** First-frame wallclock-ish anchor for rough fps estimation. */
  startTimeMs: number;
}

export class BehavioralPatternAnalyzer implements IFaceAnalyzer {
  readonly name = "behavioral_pattern";

  private readonly historyLen: number;
  private readonly warmupFrames: number;
  private readonly saccadeThreshold: number;
  private readonly entropyBins: number;
  private states: Map<number, BehaviorState> = new Map();

  constructor(options: BehavioralPatternAnalyzerOptions = {}) {
    this.historyLen = options.historyLen ?? 300;
    this.warmupFrames = options.warmupFrames ?? 60;
    this.saccadeThreshold = options.saccadeThreshold ?? 0.05;
    this.entropyBins = options.entropyBins ?? 8;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    let state = this.states.get(face.face_id);
    if (!state) {
      state = {
        lastBlinkCount: 0,
        lastBlinkFrameIdx: 0,
        ibi: new RingBuffer<number>(40),
        gazeHistory: new RingBuffer<[number, number]>(this.historyLen),
        saccadeCount: 0,
        composite: new RingBuffer<number>(this.historyLen),
        framesSeen: 0,
        startTimeMs: performance.now(),
      };
      this.states.set(face.face_id, state);
    }

    state.framesSeen += 1;

    // We expect Phase-A FaceROI to carry blendshapes. Without them this
    // analyzer can't compute anything meaningful — neutral 50.
    if (!face.blendshapes) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_blendshapes" },
        performance.now() - start,
      );
    }

    // --- 1. Inter-blink interval tracking. The pipeline-wide blink count
    //   comes from BlinkAnalyzer. If it's not in this face's results yet
    //   (first-frame ordering) we read 0 and skip the event.
    //   We'd ideally also receive the blink count from outside, but the
    //   analyzer contract gives us only the FaceROI; we settle for reading
    //   the per-eye blink blendshapes and inferring closure events.
    const leftBlink = face.blendshapes.get("eyeBlinkLeft") ?? 0;
    const rightBlink = face.blendshapes.get("eyeBlinkRight") ?? 0;
    const meanBlink = (leftBlink + rightBlink) / 2;
    // Detect a blink event when meanBlink crosses 0.5 from below.
    const blinkEvent = meanBlink > 0.5 && state.lastBlinkCount === 0;
    if (blinkEvent) {
      const ibiFrames = state.framesSeen - state.lastBlinkFrameIdx;
      if (state.lastBlinkFrameIdx > 0) state.ibi.append(ibiFrames);
      state.lastBlinkFrameIdx = state.framesSeen;
    }
    // Hysteresis: clear blink toggle once the eye reopens (mean < 0.3).
    if (meanBlink < 0.3) state.lastBlinkCount = 0;
    else if (meanBlink > 0.5) state.lastBlinkCount = 1;

    // --- 2. Saccade tracking — re-derive gaze vector from blendshapes.
    const gx =
      ((face.blendshapes.get("eyeLookInLeft") ?? 0) -
        (face.blendshapes.get("eyeLookOutLeft") ?? 0) +
        (face.blendshapes.get("eyeLookOutRight") ?? 0) -
        (face.blendshapes.get("eyeLookInRight") ?? 0)) /
      2;
    const gy =
      ((face.blendshapes.get("eyeLookUpLeft") ?? 0) +
        (face.blendshapes.get("eyeLookUpRight") ?? 0) -
        (face.blendshapes.get("eyeLookDownLeft") ?? 0) -
        (face.blendshapes.get("eyeLookDownRight") ?? 0)) /
      2;
    if (state.gazeHistory.length > 0) {
      const last = state.gazeHistory.last()!;
      const delta = Math.hypot(gx - last[0], gy - last[1]);
      if (delta > this.saccadeThreshold) state.saccadeCount += 1;
    }
    state.gazeHistory.append([gx, gy]);

    // --- 3. Composite signal for entropy. Combine jaw + brow + a blink
    //   toggle into a single scalar; quantize and Shannon-entropy the
    //   buffer to detect repetitive (low-entropy) signal patterns.
    const jaw = face.blendshapes.get("jawOpen") ?? 0;
    const brow = face.blendshapes.get("browInnerUp") ?? 0;
    const composite = 0.4 * jaw + 0.4 * brow + 0.2 * meanBlink;
    state.composite.append(composite);

    if (state.framesSeen < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50,
        {
          warming: true,
          frames: state.framesSeen,
          ibi_samples: state.ibi.length,
          saccade_count: state.saccadeCount,
        },
        performance.now() - start,
      );
    }

    // Effective fps for IBI → seconds conversion.
    const elapsedSec = (performance.now() - state.startTimeMs) / 1000;
    const fps = elapsedSec > 0.1 ? state.framesSeen / elapsedSec : 30;

    // 1. Blink-interval CV → score
    let blinkCV = 0;
    let cvScore = 0;
    if (state.ibi.length >= 2) {
      const xs = state.ibi.toArray();
      let mean = 0;
      for (const v of xs) mean += v;
      mean /= xs.length;
      let sse = 0;
      for (const v of xs) {
        const d = v - mean;
        sse += d * d;
      }
      const std = Math.sqrt(sse / xs.length);
      blinkCV = mean > 0 ? std / mean : 0;
      // CV ≥ 0.5 = full credit (real human range).
      cvScore = Math.max(0, Math.min(1, blinkCV / 0.5));
    }

    // 2. Saccade rate per second
    const saccadeRate = state.saccadeCount / Math.max(1, elapsedSec);
    const saccadeScore = Math.max(0, Math.min(1, saccadeRate / 3.0));

    // 3. Shannon entropy of the composite buffer
    const arr = state.composite.toArray();
    let minV = Infinity;
    let maxV = -Infinity;
    for (const v of arr) {
      if (v < minV) minV = v;
      if (v > maxV) maxV = v;
    }
    let entropy = 0;
    if (maxV > minV + 1e-6) {
      const bins = new Array(this.entropyBins).fill(0);
      for (const v of arr) {
        const norm = (v - minV) / (maxV - minV);
        const idx = Math.min(
          this.entropyBins - 1,
          Math.floor(norm * this.entropyBins),
        );
        bins[idx] += 1;
      }
      const total = arr.length;
      for (const b of bins) {
        if (b === 0) continue;
        const p = b / total;
        entropy -= p * Math.log2(p);
      }
    }
    const entropyScore = entropy / Math.log2(this.entropyBins); // normalized 0-1

    const score =
      100 * (0.4 * cvScore + 0.3 * saccadeScore + 0.3 * entropyScore);

    return makeAnalyzerResult(
      this.name,
      Math.max(0, Math.min(100, score)),
      {
        blink_ibi_samples: state.ibi.length,
        blink_cv: round(blinkCV, 3),
        blink_cv_score: round(cvScore, 3),
        saccade_count: state.saccadeCount,
        saccade_rate_per_sec: round(saccadeRate, 2),
        saccade_score: round(saccadeScore, 3),
        entropy_nats: round(entropy, 3),
        entropy_score: round(entropyScore, 3),
        composite_min: round(minV, 3),
        composite_max: round(maxV, 3),
        fps: round(fps, 1),
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
