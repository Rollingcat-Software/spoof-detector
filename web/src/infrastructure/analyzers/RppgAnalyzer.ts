// Port of src/infrastructure/analyzers/rppg_analyzer.py:1-186
//
// Remote Photoplethysmography (rPPG) — pulse detection from subtle
// green-channel variations on the forehead region. Real faces show a
// 45-240 BPM pulse; screens / photos / masks show none. Gold standard
// for passive liveness — zero false positives on static images.
//
// Approximation:
//   * Hand-rolled DFT replaces `np.fft.rfft` (same approach as
//     MicroTremorAnalyzer + ScreenFlickerAnalyzer). At the default
//     historyLen=150, halfN=76 — ~11k mults per axis, still sub-ms.
//   * `np.convolve(..., mode='same')` for slow-drift detrending →
//     centered moving average (length-15 kernel, matching Python).
//   * `np.hanning` is hand-rolled.
//   * Forehead ROI is taken as the upper 40% of the face crop's vertical
//     extent and the central 60% of its horizontal extent (matches
//     `face_crop[0:int(h*0.4), int(w*0.2):int(w*0.8)]`). If `faceCrop`
//     is null we fall back to the bbox of the most-recently-set frame.
//
// History buffer: 150 frames per face track (matches Python GOOD_FRAMES
// = 5s @ 30fps). The Python `deque(maxlen=300)` is a 10s ceiling we
// don't need — the score uses GOOD_FRAMES as the data-quality denom.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { SourceImage, toImageData } from "../../utils/imageOps";

// Verbatim from Python.
const PULSE_LOW_HZ = 0.75; // 45 BPM
const PULSE_HIGH_HZ = 4.0; // 240 BPM
const MIN_FRAMES = 60; // ~2s @ 30fps minimum for any signal
const GOOD_FRAMES = 150; // ~5s for reliable pulse detection

export interface RppgOptions {
  /** Fallback fps if not yet measured. Default 30. */
  fps?: number;
  /** Per-track ring buffer length. Default 150 (matches Python GOOD_FRAMES). */
  historyLen?: number;
}

interface PulseState {
  greenValues: number[];
  frameCount: number;
  estimatedBpm: number | null;
  signalStrength: number;
}

export class RppgAnalyzer implements IFaceAnalyzer {
  readonly name = "rppg";

  private fps: number;
  private readonly historyLen: number;

  private states: Map<number, PulseState> = new Map();
  private frameTimes: number[] = [];
  private currentFrame: SourceImage | null = null;

  constructor(options: RppgOptions = {}) {
    this.fps = options.fps ?? 30.0;
    this.historyLen = options.historyLen ?? GOOD_FRAMES;
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
        // Clamp implausible measurements (synthetic loops, throttled tabs)
        // — see MicroTremor/ScreenFlicker for the same guard.
        const measured = (this.frameTimes.length - 1) / dt;
        if (measured >= 1 && measured <= 120) this.fps = measured;
      }
    }

    const fid = face.face_id;
    let state = this.states.get(fid);
    if (!state) {
      state = {
        greenValues: [],
        frameCount: 0,
        estimatedBpm: null,
        signalStrength: 0.0,
      };
      this.states.set(fid, state);
    }
    state.frameCount += 1;

    // === Extract green-channel mean from the forehead region. ===
    const greenMean = this.foreheadGreenMean(faceCrop, face);
    state.greenValues.push(greenMean);
    if (state.greenValues.length > this.historyLen) state.greenValues.shift();

    // Not enough data yet.
    if (state.frameCount < MIN_FRAMES) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { warmup: true, frames: state.frameCount, need: MIN_FRAMES },
        performance.now() - start,
      );
    }

    const n = state.greenValues.length;
    const signal = new Float64Array(n);
    for (let i = 0; i < n; i++) signal[i] = state.greenValues[i];

    // === Detrend (Python: subtract a length-15 centered moving average). ===
    const ma = movingAverageSame(signal, Math.min(15, Math.max(1, n)));
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
    // Frequency axis: freqs[k] = k * fps / n.
    const dF = this.fps / n;

    // === Find dominant peak in pulse band, and noise mean outside it. ===
    let peakMagnitude = 0;
    let peakFreq = 0;
    let anyPulseBin = false;
    let noiseSum = 0;
    let noiseCount = 0;
    for (let k = 0; k < halfN; k++) {
      const f = k * dF;
      const inPulse = f >= PULSE_LOW_HZ && f <= PULSE_HIGH_HZ;
      if (inPulse) {
        anyPulseBin = true;
        if (magnitude[k] > peakMagnitude) {
          peakMagnitude = magnitude[k];
          peakFreq = f;
        }
      } else if (f > 0.1) {
        // Python: noise_mask = ~pulse_mask & (freqs > 0.1). DC excluded.
        noiseSum += magnitude[k];
        noiseCount += 1;
      }
    }
    if (!anyPulseBin) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "no_pulse_band", measured_fps: round(this.fps, 1) },
        performance.now() - start,
      );
    }
    const noiseMean = noiseCount > 0 ? noiseSum / noiseCount : 1e-6;

    const snr = peakMagnitude / Math.max(noiseMean, 1e-6);
    const bpm = peakFreq * 60.0;

    state.estimatedBpm = snr > 2.0 ? bpm : null;
    state.signalStrength = Math.min(1.0, snr / 5.0);

    // === Score based on signal quality. ===
    const dataQuality = Math.min(1.0, state.frameCount / GOOD_FRAMES);

    let score: number;
    if (snr > 4.0 && bpm >= 45 && bpm <= 200) {
      score = 70.0 + dataQuality * 30.0; // Strong pulse = live
    } else if (snr > 2.5 && bpm >= 45 && bpm <= 200) {
      score = 50.0 + dataQuality * 20.0; // Weak pulse = probably live
    } else if (snr > 1.5) {
      score = 30.0 + dataQuality * 20.0; // Ambiguous
    } else if (state.frameCount > GOOD_FRAMES) {
      score = 10.0; // Enough data, no pulse = spoof
    } else {
      score = 30.0; // Not enough data yet
    }
    score = Math.max(0.0, Math.min(100.0, score));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        bpm: snr > 2.0 ? round(bpm, 1) : null,
        snr: round(snr, 2),
        signal_strength: round(state.signalStrength, 3),
        peak_freq_hz: round(peakFreq, 3),
        measured_fps: round(this.fps, 1),
        frames: state.frameCount,
        data_quality: round(dataQuality, 2),
      },
      performance.now() - start,
    );
  }

  /** Public BPM accessor (parity with Python `get_bpm`). */
  getBpm(faceId: number): number | null {
    const s = this.states.get(faceId);
    return s ? s.estimatedBpm : null;
  }

  /**
   * Mean green-channel intensity over the forehead region: upper 40% rows,
   * central 60% columns of either the face crop or — if faceCrop is null —
   * the bbox region of the most recently set frame.
   */
  private foreheadGreenMean(
    faceCrop: ImageData | null,
    face: FaceROI,
  ): number {
    if (faceCrop && faceCrop.width > 0 && faceCrop.height > 0) {
      const w = faceCrop.width;
      const h = faceCrop.height;
      const x1 = Math.floor(w * 0.2);
      const x2 = Math.floor(w * 0.8);
      const y1 = 0;
      const y2 = Math.floor(h * 0.4);
      return meanGreenRect(faceCrop, x1, y1, x2, y2);
    }
    if (this.currentFrame) {
      const data = toImageData(this.currentFrame);
      const bx1 = Math.max(0, face.bbox.x1);
      const by1 = Math.max(0, face.bbox.y1);
      const bx2 = Math.min(data.width, face.bbox.x2);
      const by2 = Math.min(data.height, face.bbox.y2);
      const bw = bx2 - bx1;
      const bh = by2 - by1;
      if (bw <= 0 || bh <= 0) return 0.0;
      const fx1 = Math.floor(bx1 + bw * 0.2);
      const fx2 = Math.floor(bx1 + bw * 0.8);
      const fy1 = Math.floor(by1);
      const fy2 = Math.floor(by1 + bh * 0.4);
      return meanGreenRect(data, fx1, fy1, fx2, fy2);
    }
    return 0.0;
  }

  reset(): void {
    this.states.clear();
    this.frameTimes = [];
    this.currentFrame = null;
  }
}

/** Mean of the green channel inside an (x1,y1)-(x2,y2) rect. */
function meanGreenRect(
  img: ImageData,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): number {
  const w = Math.max(0, x2 - x1);
  const h = Math.max(0, y2 - y1);
  if (w <= 0 || h <= 0) return 0.0;
  const stride = img.width * 4;
  let s = 0;
  let count = 0;
  for (let y = y1; y < y2; y++) {
    let off = y * stride + x1 * 4;
    for (let x = 0; x < w; x++, off += 4) {
      s += img.data[off + 1]; // green channel
      count += 1;
    }
  }
  return count > 0 ? s / count : 0.0;
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
