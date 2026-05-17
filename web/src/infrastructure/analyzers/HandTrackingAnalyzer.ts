// HandTrackingAnalyzer — Phase D2 (opt-in).
//
// Pairs with MediaPipeHandDetector. Per-frame signals:
//
//   * Hand presence (0, 1, 2) — informational; not directly scored.
//   * Per-hand wrist position (landmark idx 0) tracked across the rolling
//     window. Stddev of wrist motion → natural-gesture credit. A user
//     resting hands out-of-view scores neutral 50 (no signal). A user
//     gesturing naturally scores high. A static photo where the operator
//     accidentally has a hand visible scores 0.
//   * Anomaly flag — > 2 hands in a single frame (a deepfake artifact
//     class) is flagged in details (`anomaly_third_hand: true`) and
//     pushes the score down.
//
// Detection is rate-limited (default once per 4 frames) so the model
// doesn't dominate the per-frame CPU budget on mobile.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { SourceImage } from "../../utils/imageOps";
import {
  DetectedHand,
  MediaPipeHandDetector,
} from "../detection/MediaPipeHandDetector";

export interface HandTrackingAnalyzerOptions {
  detector?: MediaPipeHandDetector;
  /** Per-hand wrist-history window length (60 = 10 s at 1 sample / 4 fr @ 30 fps). */
  historyLen?: number;
  /** Run detection once every N frames. Default 4. */
  detectEveryN?: number;
  /** stddev-to-score gain for hand motion. Calibrated for natural gesture. */
  gain?: number;
  /** Frames before scoring (else neutral 50). */
  warmupSamples?: number;
}

interface HandState {
  /** Per-detect-call wrist positions (mean across all hands present). */
  wristHistory: Array<[number, number]>;
  frameCount: number;
  lastScore: number;
  lastHandCount: number;
  lastAnomaly: boolean;
}

export class HandTrackingAnalyzer implements IFaceAnalyzer {
  readonly name = "hand_tracking";

  private readonly detector: MediaPipeHandDetector;
  private readonly historyLen: number;
  private readonly detectEveryN: number;
  private readonly gain: number;
  private readonly warmupSamples: number;
  private currentFrame: SourceImage | null = null;
  private state: HandState = {
    wristHistory: [],
    frameCount: 0,
    lastScore: 50,
    lastHandCount: 0,
    lastAnomaly: false,
  };

  constructor(options: HandTrackingAnalyzerOptions = {}) {
    this.detector = options.detector ?? new MediaPipeHandDetector();
    this.historyLen = options.historyLen ?? 60;
    this.detectEveryN = Math.max(1, options.detectEveryN ?? 4);
    this.gain = options.gain ?? 200;
    this.warmupSamples = options.warmupSamples ?? 8;
  }

  setFrame(frame: SourceImage): void {
    this.currentFrame = frame;
  }

  async analyzeAsync(
    _faceCrop: ImageData | null,
    _face: FaceROI,
  ): Promise<AnalyzerResult> {
    const start = performance.now();
    this.state.frameCount += 1;

    if (
      this.state.frameCount % this.detectEveryN !== 0 ||
      !this.currentFrame
    ) {
      return makeAnalyzerResult(
        this.name,
        this.state.lastScore,
        {
          cached: true,
          frame_count: this.state.frameCount,
          last_hand_count: this.state.lastHandCount,
        },
        performance.now() - start,
      );
    }

    let hands: DetectedHand[] = [];
    try {
      hands = await this.detector.detect(
        this.currentFrame as never,
        performance.now(),
      );
    } catch (err) {
      return makeAnalyzerResult(
        this.name,
        50,
        {
          error: "detect_failed",
          message: (err as Error)?.message ?? String(err),
        },
        performance.now() - start,
      );
    }
    const handCount = hands.length;
    this.state.lastHandCount = handCount;
    // Anomaly: > 2 hands in one frame — deepfake artefact class. The
    // detector is configured for numHands=2 by default so this is rare
    // unless overridden, but flag it for downstream consumers.
    const anomaly = handCount > 2;
    this.state.lastAnomaly = anomaly;

    if (handCount === 0) {
      // No hands visible — neutral. No update to wrist history.
      return makeAnalyzerResult(
        this.name,
        50,
        {
          hand_count: 0,
          samples: this.state.wristHistory.length,
        },
        performance.now() - start,
      );
    }

    // Mean wrist position across visible hands (landmark idx 0).
    let mx = 0;
    let my = 0;
    let n = 0;
    for (const h of hands) {
      const w = h.landmarks[0];
      if (w) {
        mx += w.x;
        my += w.y;
        n += 1;
      }
    }
    if (n === 0) {
      return makeAnalyzerResult(
        this.name,
        50,
        { hand_count: handCount, error: "no_wrist_landmarks" },
        performance.now() - start,
      );
    }
    this.state.wristHistory.push([mx / n, my / n]);
    if (this.state.wristHistory.length > this.historyLen) {
      this.state.wristHistory.shift();
    }

    if (this.state.wristHistory.length < this.warmupSamples) {
      const r = makeAnalyzerResult(
        this.name,
        50,
        {
          warming: true,
          samples: this.state.wristHistory.length,
          hand_count: handCount,
        },
        performance.now() - start,
      );
      this.state.lastScore = 50;
      return r;
    }

    let cmx = 0;
    let cmy = 0;
    for (const [x, y] of this.state.wristHistory) {
      cmx += x;
      cmy += y;
    }
    const sz = this.state.wristHistory.length;
    cmx /= sz;
    cmy /= sz;
    let ss = 0;
    for (const [x, y] of this.state.wristHistory) {
      ss += (x - cmx) ** 2 + (y - cmy) ** 2;
    }
    const std = Math.sqrt(ss / sz);
    let score = Math.max(0, Math.min(100, std * this.gain));
    if (anomaly) score = Math.min(score, 20); // hard cap on anomaly frames
    this.state.lastScore = score;

    const handednessArr = hands.map((h) => h.handedness ?? "unknown");
    return makeAnalyzerResult(
      this.name,
      score,
      {
        hand_count: handCount,
        handedness: handednessArr,
        wrist_x: round(this.state.wristHistory.at(-1)?.[0] ?? 0, 3),
        wrist_y: round(this.state.wristHistory.at(-1)?.[1] ?? 0, 3),
        wrist_std: round(std, 4),
        samples: sz,
        anomaly_third_hand: anomaly,
      },
      performance.now() - start,
    );
  }

  analyze(
    faceCrop: ImageData | null,
    face: FaceROI,
  ): Promise<AnalyzerResult> {
    return this.analyzeAsync(faceCrop, face);
  }

  reset(): void {
    this.state = {
      wristHistory: [],
      frameCount: 0,
      lastScore: 50,
      lastHandCount: 0,
      lastAnomaly: false,
    };
  }

  async close(): Promise<void> {
    await this.detector.close();
  }
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
