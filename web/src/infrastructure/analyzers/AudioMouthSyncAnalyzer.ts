// AudioMouthSyncAnalyzer — Phase D3 (opt-in).
//
// Cross-correlates the audio RMS time series (from AudioCapture) with
// the mouth-open blendshape time series (jawOpen from FaceROI). When
// the subject is speaking live, the two correlate strongly: every "ah"
// is a jaw drop AND an audio energy spike. A video replay either has
// no audio (silent attack) or has audio that's desynced from the
// mouth movements (frame-dropped replay, dubbed video).
//
// Per frame:
//   * Append current jawOpen to a rolling window (60 samples @ 30 fps = 2 s).
//   * Sample the matching last-2-seconds of audio RMS, downsampled to
//     the same length via simple nearest-neighbour bucketing.
//   * Compute Pearson correlation. score = max(0, corr) × 100.
//
// Returns neutral 50 when either source is silent (no speech) — sync
// is only meaningful when there IS audio to compare.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { RingBuffer } from "../../domain/session";
import { AudioCaptureLike } from "../audio/AudioCapture";

export interface AudioMouthSyncAnalyzerOptions {
  audio: AudioCaptureLike;
  /** Rolling window length in frames. Default 60 (= 2 s @ 30 fps). */
  historyLen?: number;
  /** Frames before scoring (else neutral 50). */
  warmupFrames?: number;
  /**
   * RMS threshold below which we treat the audio as silent and return
   * neutral 50 instead of a low correlation. Default 0.005.
   */
  silenceThreshold?: number;
  /** jawOpen variation floor — below this, return neutral. Default 0.02. */
  mouthSilenceThreshold?: number;
}

export class AudioMouthSyncAnalyzer implements IFaceAnalyzer {
  readonly name = "audio_mouth_sync";

  private readonly audio: AudioCaptureLike;
  private readonly historyLen: number;
  private readonly warmupFrames: number;
  private readonly silenceThreshold: number;
  private readonly mouthSilenceThreshold: number;
  private mouthHistory: Map<number, RingBuffer<number>> = new Map();

  constructor(options: AudioMouthSyncAnalyzerOptions) {
    this.audio = options.audio;
    this.historyLen = options.historyLen ?? 60;
    this.warmupFrames = options.warmupFrames ?? 30;
    this.silenceThreshold = options.silenceThreshold ?? 0.005;
    this.mouthSilenceThreshold = options.mouthSilenceThreshold ?? 0.02;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    if (!this.audio.isActive) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "audio_inactive" },
        performance.now() - start,
      );
    }
    if (!face.blendshapes) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_blendshapes" },
        performance.now() - start,
      );
    }
    const jaw = face.blendshapes.get("jawOpen") ?? 0;

    let buf = this.mouthHistory.get(face.face_id);
    if (!buf) {
      buf = new RingBuffer<number>(this.historyLen);
      this.mouthHistory.set(face.face_id, buf);
    }
    buf.append(jaw);

    if (buf.length < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50,
        {
          warming: true,
          frames: buf.length,
          jaw_open: round(jaw, 3),
        },
        performance.now() - start,
      );
    }

    const mouthSeries = buf.toArray();
    const windowSec = mouthSeries.length / 30; // approx, matching default fps
    const audioSeries = this.audio.getRecentRms(windowSec);
    if (audioSeries.length < 2) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_audio_samples" },
        performance.now() - start,
      );
    }

    // Resample audio to the same length as the mouth series using
    // nearest-neighbour bucketing (audio is typically much higher rate).
    const audioMatched = resampleNearest(audioSeries, mouthSeries.length);

    const audioStats = stddev(audioMatched);
    const mouthStats = stddev(mouthSeries);

    if (
      audioStats.std < this.silenceThreshold ||
      mouthStats.std < this.mouthSilenceThreshold
    ) {
      // Either side silent — correlation undefined. Neutral score lets
      // the proof axis not punish quiet moments.
      return makeAnalyzerResult(
        this.name,
        50,
        {
          silence: true,
          audio_std: round(audioStats.std, 4),
          mouth_std: round(mouthStats.std, 4),
          frames: mouthSeries.length,
        },
        performance.now() - start,
      );
    }

    const corr = pearsonAligned(
      audioMatched,
      mouthSeries,
      audioStats.mean,
      mouthStats.mean,
    );
    const score = Math.max(0, Math.min(100, corr * 100));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        corr: round(corr, 3),
        audio_std: round(audioStats.std, 4),
        mouth_std: round(mouthStats.std, 4),
        jaw_open: round(jaw, 3),
        frames: mouthSeries.length,
      },
      performance.now() - start,
    );
  }

  reset(): void {
    this.mouthHistory.clear();
  }
}

function stddev(xs: ArrayLike<number>): { mean: number; std: number } {
  const n = xs.length;
  if (n === 0) return { mean: 0, std: 0 };
  let m = 0;
  for (let i = 0; i < n; i++) m += xs[i];
  m /= n;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = xs[i] - m;
    ss += d * d;
  }
  return { mean: m, std: Math.sqrt(ss / n) };
}

function pearsonAligned(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  mx: number,
  my: number,
): number {
  const n = Math.min(xs.length, ys.length);
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
  const denom = Math.sqrt(sxx * syy);
  return denom > 1e-12 ? cov / denom : 0;
}

function resampleNearest(src: Float32Array, targetLen: number): Float32Array {
  if (src.length === targetLen) return src;
  const out = new Float32Array(targetLen);
  const ratio = src.length / targetLen;
  for (let i = 0; i < targetLen; i++) {
    out[i] = src[Math.min(src.length - 1, Math.floor(i * ratio))];
  }
  return out;
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
