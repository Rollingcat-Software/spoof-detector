// Port of src/infrastructure/analyzers/screen_flicker_analyzer.py:1-141
//
// Screen Flicker — fusion weight 3.0 (highest of the Phase 2 batch).
//
// Detects 50/60 Hz screen-camera beat frequencies by FFTing the per-frame
// mean-intensity time series of the face crop. Real faces have constant
// lighting → no significant power in the flicker bands. Screens leave
// strong peaks near 10/20/30 Hz (camera × screen aliasing).
//
// Approximation:
//   * Hand-rolled DFT replaces `np.fft.rfft` (same approach as
//     MicroTremorAnalyzer; see header there). At the default
//     historyLen=30, halfN=16 — ~480 mults, well below 1ms.
//   * `np.convolve(..., mode='same')` for detrending → centered moving
//     average. Same shape, same purpose.
//   * No spatial FFT down each column (the task description hinted at
//     it, but the Python source does TEMPORAL FFT on the per-frame mean
//     intensity — that's what we implement). Documented here so future
//     readers don't conflate the two.
//
// History buffer: 30 frames per face track (matches WARMUP_FRAMES).
// The Python uses 120 (4s @ 30fps); 30 is a memory-conscious default —
// callers who need cleaner peaks can pass `historyLen: 120`.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { SourceImage, toImageData } from "../../utils/imageOps";

// Verbatim from Python — bands chosen for 50/60 Hz monitors with 30 fps cameras.
const FLICKER_BANDS: ReadonlyArray<readonly [number, number]> = [
  [8.0, 15.0],
  [18.0, 25.0],
  [28.0, 35.0],
];
const MIN_FRAMES = 30;
const GOOD_FRAMES = 60;

export interface ScreenFlickerOptions {
  /** Fallback fps if not yet measured. Default 30. */
  fps?: number;
  /** Per-track ring buffer length. Default 30. */
  historyLen?: number;
}

export class ScreenFlickerAnalyzer implements IFaceAnalyzer {
  readonly name = "screen_flicker";

  private fps: number;
  private readonly historyLen: number;

  // Per-face: array of mean intensities, last `historyLen` entries.
  private states: Map<number, number[]> = new Map();
  private frameTimes: number[] = [];
  private currentFrame: SourceImage | null = null;

  constructor(options: ScreenFlickerOptions = {}) {
    this.fps = options.fps ?? 30.0;
    this.historyLen = options.historyLen ?? 30;
  }

  /** Set the current full frame so the analyzer can crop the face ROI itself
   *  if `faceCrop` isn't provided by the orchestrator. */
  setFrame(frame: SourceImage): void {
    this.currentFrame = frame;
  }

  analyze(faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();

    // === Measure fps. ===
    this.frameTimes.push(start);
    if (this.frameTimes.length > 60) this.frameTimes.shift();
    if (this.frameTimes.length > 10) {
      const dt =
        (this.frameTimes[this.frameTimes.length - 1] - this.frameTimes[0]) /
        1000.0;
      if (dt > 0) {
        // Clamp implausible measurements (synthetic loops, throttled tabs).
        const measured = (this.frameTimes.length - 1) / dt;
        if (measured >= 1 && measured <= 120) this.fps = measured;
      }
    }

    const fid = face.face_id;
    let buf = this.states.get(fid);
    if (!buf) {
      buf = [];
      this.states.set(fid, buf);
    }

    // === Compute mean RGB intensity over the face crop. ===
    const intensity = this.meanIntensity(faceCrop, face);
    if (intensity === null) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "no_pixels" },
        performance.now() - start,
      );
    }
    buf.push(intensity);
    if (buf.length > this.historyLen) buf.shift();

    if (buf.length < Math.min(MIN_FRAMES, this.historyLen)) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { warmup: true, frames: buf.length },
        performance.now() - start,
      );
    }

    const n = buf.length;
    const signal = new Float64Array(n);
    for (let i = 0; i < n; i++) signal[i] = buf[i];

    // === Detrend (Python: subtract a length-10 centered moving average). ===
    const ma = movingAverageSame(signal, Math.min(10, Math.max(1, n)));
    for (let i = 0; i < n; i++) signal[i] -= ma[i];

    // === Hanning window. ===
    for (let i = 0; i < n; i++) {
      const w =
        n > 1 ? 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1))) : 1.0;
      signal[i] *= w;
    }

    // === Hand-rolled DFT (positive freqs only). ===
    const halfN = Math.floor(n / 2) + 1;
    const magnitude = new Float64Array(halfN);
    for (let k = 0; k < halfN; k++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < n; t++) {
        const angle = (-2 * Math.PI * k * t) / n;
        re += signal[t] * Math.cos(angle);
        im += signal[t] * Math.sin(angle);
      }
      magnitude[k] = Math.sqrt(re * re + im * im);
    }
    const dF = this.fps / n;

    // Noise floor (Python: mean(magnitude[1:])).
    let noiseSum = 0;
    let noiseCount = 0;
    for (let k = 1; k < halfN; k++) {
      noiseSum += magnitude[k];
      noiseCount += 1;
    }
    const noiseFloor = noiseCount > 0 ? noiseSum / noiseCount : 1.0;

    // Find the strongest peak across the flicker bands.
    let maxFlickerPower = 0;
    let dominantFreq = 0;
    for (const [low, high] of FLICKER_BANDS) {
      let bandMaxPower = 0;
      let bandPeakFreq = 0;
      let any = false;
      for (let k = 0; k < halfN; k++) {
        const f = k * dF;
        if (f >= low && f <= high) {
          any = true;
          if (magnitude[k] > bandMaxPower) {
            bandMaxPower = magnitude[k];
            bandPeakFreq = f;
          }
        }
      }
      if (any && bandMaxPower > maxFlickerPower) {
        maxFlickerPower = bandMaxPower;
        dominantFreq = bandPeakFreq;
      }
    }

    const flickerSnr = maxFlickerPower / Math.max(noiseFloor, 1e-6);
    const dataQuality = Math.min(1.0, n / GOOD_FRAMES);

    let score: number;
    if (flickerSnr > 4.0) {
      score = 10.0;
    } else if (flickerSnr > 2.5) {
      score = 25.0 + (4.0 - flickerSnr) * 10.0;
    } else if (flickerSnr > 1.5) {
      score = 50.0;
    } else {
      score = 70.0 + dataQuality * 30.0;
    }
    score = Math.max(0.0, Math.min(100.0, score));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        flicker_snr: round(flickerSnr, 2),
        dominant_freq_hz: round(dominantFreq, 1),
        max_flicker_power: round(maxFlickerPower, 2),
        noise_floor: round(noiseFloor, 2),
        measured_fps: round(this.fps, 1),
        frames: n,
      },
      performance.now() - start,
    );
  }

  /**
   * Mean RGB intensity over the face crop or — if faceCrop is null — the
   * face bbox region of the most recently set frame. Returns null when
   * neither is available.
   */
  private meanIntensity(
    faceCrop: ImageData | null,
    face: FaceROI,
  ): number | null {
    if (faceCrop && faceCrop.width > 0 && faceCrop.height > 0) {
      return averageRgb(faceCrop.data);
    }
    if (this.currentFrame) {
      const data = toImageData(this.currentFrame);
      return averageRgbBbox(
        data,
        Math.max(0, face.bbox.x1),
        Math.max(0, face.bbox.y1),
        Math.min(data.width, face.bbox.x2),
        Math.min(data.height, face.bbox.y2),
      );
    }
    return null;
  }

  reset(): void {
    this.states.clear();
    this.frameTimes = [];
  }
}

function averageRgb(data: Uint8ClampedArray): number {
  let s = 0;
  const px = data.length / 4;
  for (let p = 0; p < data.length; p += 4) {
    s += data[p] + data[p + 1] + data[p + 2];
  }
  return s / (px * 3);
}

function averageRgbBbox(
  img: ImageData,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): number | null {
  const w = Math.floor(x2 - x1);
  const h = Math.floor(y2 - y1);
  if (w <= 0 || h <= 0) return null;
  const stride = img.width * 4;
  let s = 0;
  let count = 0;
  for (let y = Math.floor(y1); y < Math.floor(y2); y++) {
    let off = y * stride + Math.floor(x1) * 4;
    for (let x = 0; x < w; x++, off += 4) {
      s += img.data[off] + img.data[off + 1] + img.data[off + 2];
      count += 1;
    }
  }
  return count > 0 ? s / (count * 3) : null;
}

function movingAverageSame(
  signal: ArrayLike<number>,
  k: number,
): Float64Array {
  const n = signal.length;
  const out = new Float64Array(n);
  const half = Math.floor(k / 2);
  for (let i = 0; i < n; i++) {
    let s = 0;
    let c = 0;
    for (let j = i - half; j <= i + half; j++) {
      if (j >= 0 && j < n) {
        s += signal[j];
        c += 1;
      }
    }
    out[i] = c > 0 ? s / c : 0;
  }
  return out;
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
