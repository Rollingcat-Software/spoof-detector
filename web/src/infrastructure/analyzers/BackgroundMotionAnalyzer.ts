// BackgroundMotionAnalyzer — Phase D1 (opt-in).
//
// Pairs with MediaPipeSelfieSegmenter to measure how much the *scene
// behind the face* shifts over time. A real environment has subtle
// background drift (lighting changes, distant motion, the user
// reflecting on a wall behind them); a printed photo or static
// phone-screen replay holds the background near-constant.
//
// Strategy: every N frames the analyzer samples the mean RGB of all
// pixels classified as "background" (confidence < 0.5 by SelfieSegmenter)
// from the current frame. Those means accumulate in a rolling window;
// the stddev across the window × gain → 0–100 score.
//
// To keep the segmenter call rate low on mobile, the analyzer runs
// segmentation once every `segmentEveryN` frames (default 5). Between
// runs it carries the most recent score through unchanged.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { SourceImage } from "../../utils/imageOps";
import { MediaPipeSelfieSegmenter } from "../detection/MediaPipeSelfieSegmenter";

export interface BackgroundMotionAnalyzerOptions {
  /** Pre-constructed segmenter (testability + DI). Optional. */
  segmenter?: MediaPipeSelfieSegmenter;
  /** Rolling window length (60 = 10 s at one sample / 5 frames @ 30 fps). */
  historyLen?: number;
  /** Run segmenter once every N frames. Default 5. */
  segmentEveryN?: number;
  /** Stddev-to-score gain. Calibrated for indoor lighting drift. */
  gain?: number;
  /** Frames before scoring (else neutral 50). */
  warmupSamples?: number;
}

interface BgState {
  /** Per-sample mean R/G/B of the background region. */
  history: Array<[number, number, number]>;
  frameCount: number;
  lastScore: number;
  lastSample: [number, number, number] | null;
  lastBgPixelRatio: number;
}

export class BackgroundMotionAnalyzer implements IFaceAnalyzer {
  readonly name = "background_motion";

  private readonly segmenter: MediaPipeSelfieSegmenter;
  private readonly historyLen: number;
  private readonly segmentEveryN: number;
  private readonly gain: number;
  private readonly warmupSamples: number;
  private currentFrame: SourceImage | null = null;
  private state: BgState = {
    history: [],
    frameCount: 0,
    lastScore: 50,
    lastSample: null,
    lastBgPixelRatio: 0,
  };

  constructor(options: BackgroundMotionAnalyzerOptions = {}) {
    this.segmenter = options.segmenter ?? new MediaPipeSelfieSegmenter();
    this.historyLen = options.historyLen ?? 60;
    this.segmentEveryN = Math.max(1, options.segmentEveryN ?? 5);
    this.gain = options.gain ?? 12;
    this.warmupSamples = options.warmupSamples ?? 10;
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

    // Rate-limit segmentation; on skipped frames the cached score is
    // re-emitted so consumers see a stable per-frame readout.
    if (
      this.state.frameCount % this.segmentEveryN !== 0 ||
      !this.currentFrame
    ) {
      return makeAnalyzerResult(
        this.name,
        this.state.lastScore,
        {
          cached: true,
          frame_count: this.state.frameCount,
          samples: this.state.history.length,
        },
        performance.now() - start,
      );
    }

    let segOut: Awaited<ReturnType<MediaPipeSelfieSegmenter["segment"]>>;
    try {
      segOut = await this.segmenter.segment(
        this.currentFrame as never,
        performance.now(),
      );
    } catch (err) {
      // Segmenter failed (model fetch, GPU init, etc) — neutral and
      // log via details so the consumer knows.
      return makeAnalyzerResult(
        this.name,
        50,
        {
          error: "segment_failed",
          message: (err as Error)?.message ?? String(err),
        },
        performance.now() - start,
      );
    }
    if (!segOut) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_mask" },
        performance.now() - start,
      );
    }

    // Sample background pixels from the frame using the mask.
    const sample = this.sampleBackgroundMean(segOut);
    if (!sample) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "empty_background" },
        performance.now() - start,
      );
    }
    this.state.lastSample = sample.rgb;
    this.state.lastBgPixelRatio = sample.bgRatio;

    this.state.history.push(sample.rgb);
    if (this.state.history.length > this.historyLen) {
      this.state.history.shift();
    }

    if (this.state.history.length < this.warmupSamples) {
      const r = makeAnalyzerResult(
        this.name,
        50,
        {
          warming: true,
          samples: this.state.history.length,
          bg_pixel_ratio: round(sample.bgRatio, 3),
        },
        performance.now() - start,
      );
      this.state.lastScore = 50;
      return r;
    }

    let mR = 0;
    let mG = 0;
    let mB = 0;
    for (const [r, g, b] of this.state.history) {
      mR += r;
      mG += g;
      mB += b;
    }
    const n = this.state.history.length;
    mR /= n;
    mG /= n;
    mB /= n;
    let ssR = 0;
    let ssG = 0;
    let ssB = 0;
    for (const [r, g, b] of this.state.history) {
      ssR += (r - mR) ** 2;
      ssG += (g - mG) ** 2;
      ssB += (b - mB) ** 2;
    }
    const stdR = Math.sqrt(ssR / n);
    const stdG = Math.sqrt(ssG / n);
    const stdB = Math.sqrt(ssB / n);
    // Combined per-channel stddev (normalised to 0..255 range).
    const drift = (stdR + stdG + stdB) / 3;
    const score = Math.max(0, Math.min(100, drift * this.gain));
    this.state.lastScore = score;

    return makeAnalyzerResult(
      this.name,
      score,
      {
        samples: n,
        bg_pixel_ratio: round(sample.bgRatio, 3),
        std_r: round(stdR, 3),
        std_g: round(stdG, 3),
        std_b: round(stdB, 3),
        drift: round(drift, 3),
      },
      performance.now() - start,
    );
  }

  /** Synchronous wrapper — Promise-returning for the analyzer interface. */
  analyze(
    faceCrop: ImageData | null,
    face: FaceROI,
  ): Promise<AnalyzerResult> {
    return this.analyzeAsync(faceCrop, face);
  }

  reset(): void {
    this.state = {
      history: [],
      frameCount: 0,
      lastScore: 50,
      lastSample: null,
      lastBgPixelRatio: 0,
    };
  }

  async close(): Promise<void> {
    await this.segmenter.close();
  }

  /**
   * Project the mask onto the source frame, sample mean R/G/B across
   * background pixels (mask < 0.5), and return both the mean and the
   * fraction of background pixels seen (so we can detect frames where
   * the user fills the entire view).
   */
  private sampleBackgroundMean(
    seg: { mask: Float32Array; width: number; height: number },
  ): { rgb: [number, number, number]; bgRatio: number } | null {
    if (!this.currentFrame) return null;
    const frame = this.toImageData(this.currentFrame);
    if (!frame) return null;
    const fw = frame.width;
    const fh = frame.height;
    const data = frame.data;
    const mw = seg.width;
    const mh = seg.height;
    const m = seg.mask;
    const sx = fw / mw;
    const sy = fh / mh;
    let bgCount = 0;
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    // Stride through the mask; sample 1 frame pixel per mask cell.
    for (let my = 0; my < mh; my++) {
      for (let mx = 0; mx < mw; mx++) {
        const conf = m[my * mw + mx];
        if (conf >= 0.5) continue; // person, skip
        const fx = Math.min(fw - 1, Math.floor(mx * sx));
        const fy = Math.min(fh - 1, Math.floor(my * sy));
        const fp = (fy * fw + fx) * 4;
        sumR += data[fp];
        sumG += data[fp + 1];
        sumB += data[fp + 2];
        bgCount += 1;
      }
    }
    if (bgCount === 0) return null;
    return {
      rgb: [sumR / bgCount, sumG / bgCount, sumB / bgCount],
      bgRatio: bgCount / (mw * mh),
    };
  }

  private toImageData(frame: SourceImage): ImageData | null {
    // Duck-typed: anything with .data + .width + .height is ImageData-like.
    // (Also catches the plain-object mocks vitest tests construct.)
    const f = frame as unknown as {
      data?: Uint8ClampedArray;
      width?: number;
      height?: number;
    };
    if (
      f &&
      typeof f.width === "number" &&
      typeof f.height === "number" &&
      f.data &&
      typeof (f.data as Uint8ClampedArray).length === "number"
    ) {
      return frame as ImageData;
    }
    // Canvas / OffscreenCanvas / HTMLVideoElement — draw onto a temp
    // canvas. We use a small downsample (192×108) to keep this cheap.
    const TW = 192;
    const TH = 108;
    const cnv =
      typeof OffscreenCanvas !== "undefined"
        ? new OffscreenCanvas(TW, TH)
        : typeof document !== "undefined"
          ? (() => {
              const c = document.createElement("canvas");
              c.width = TW;
              c.height = TH;
              return c;
            })()
          : null;
    if (!cnv) return null;
    const ctx = (cnv as HTMLCanvasElement).getContext("2d") as
      | CanvasRenderingContext2D
      | OffscreenCanvasRenderingContext2D
      | null;
    if (!ctx) return null;
    ctx.drawImage(frame as never, 0, 0, TW, TH);
    return ctx.getImageData(0, 0, TW, TH);
  }
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
