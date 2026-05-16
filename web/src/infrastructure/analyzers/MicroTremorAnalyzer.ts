// Port of src/infrastructure/analyzers/micro_tremor_analyzer.py:1-163
//
// Micro-Tremor — fusion weight 2.5.
//
// Detects 8-12 Hz involuntary physiological tremor that all live humans
// exhibit. Screens cannot transmit this band (pixel grid filters it),
// hand tremor lives at 3-5 Hz so a held photo can't fake it either.
//
// Approximation:
//   * The Python source uses `np.fft.rfft` over ~90-frame windows. We
//     replace it with a hand-rolled DFT — at N≈90 the cost is well
//     under 1 ms (90 × 45 mults). Documented inline.
//   * `np.convolve(..., mode='same')` is replaced with a centered moving
//     average (handle endpoints by clamping). Same shape, same purpose.
//   * `np.hanning` is also hand-rolled.
//
// History buffer: 30 frames per face track (matches WARMUP_FRAMES). The
// Python original allocates 180 (= 6s @ 30fps); the task spec asks for
// 30 — at 30 fps that gives a 1 Hz frequency resolution which is just
// barely enough to resolve the 8-12 Hz band. Documented as a known
// trade-off — callers who need higher precision should pass
// `historyLen: 90` for parity with the Python.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";

const TREMOR_LOW_HZ = 7.0;
const TREMOR_HIGH_HZ = 13.0;
const MIN_FRAMES = 30; // ~1s @ 30fps. Python uses 45.
const GOOD_FRAMES = 90; // ~3s @ 30fps.

export interface MicroTremorOptions {
  /** Fallback fps if not yet measured. Default 30. */
  fps?: number;
  /** Per-track ring buffer length. Default 30 (matches WARMUP_FRAMES). */
  historyLen?: number;
}

export class MicroTremorAnalyzer implements IFaceAnalyzer {
  readonly name = "micro_tremor";

  private fps: number;
  private readonly historyLen: number;

  // Per-face: array of (cx, cy) positions, last `historyLen` entries.
  private states: Map<number, Array<[number, number]>> = new Map();
  // Frame-time ring (last 60). Used to estimate effective fps.
  private frameTimes: number[] = [];
  private currentLandmarks: Float32Array | null = null;

  constructor(options: MicroTremorOptions = {}) {
    this.fps = options.fps ?? 30.0;
    this.historyLen = options.historyLen ?? 30;
  }

  /** Optional API-parity hook: provide the latest landmarks set so we can
   *  use the centroid instead of the bbox center. */
  setLandmarks(landmarks: Float32Array | null): void {
    this.currentLandmarks = landmarks;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();

    // === Measure fps. ===
    this.frameTimes.push(start);
    if (this.frameTimes.length > 60) this.frameTimes.shift();
    if (this.frameTimes.length > 10) {
      const dt =
        (this.frameTimes[this.frameTimes.length - 1] - this.frameTimes[0]) /
        1000.0;
      if (dt > 0) {
        // Real browser frame rates land in 1-120 fps; anything outside
        // means synthetic test loop (too fast) or background-tab throttle
        // (too slow) — both garbage signals for our FFT band math.
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

    // Centroid: prefer landmarks (pulled off FaceROI or via setLandmarks).
    const lm = face.landmarks ?? this.currentLandmarks;
    let cx: number;
    let cy: number;
    if (lm && lm.length >= 20) {
      let sx = 0;
      let sy = 0;
      const n = Math.floor(lm.length / 2);
      for (let i = 0; i < n; i++) {
        sx += lm[2 * i];
        sy += lm[2 * i + 1];
      }
      cx = sx / n;
      cy = sy / n;
    } else {
      cx = face.bbox.center[0];
      cy = face.bbox.center[1];
    }

    buf.push([cx, cy]);
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

    // Analyze X and Y axes independently.
    const ratios: number[] = [];
    for (let axis = 0; axis < 2; axis++) {
      const signal = new Float64Array(n);
      for (let i = 0; i < n; i++) signal[i] = buf[i][axis];
      ratios.push(this.bandPowerRatio(signal));
    }

    const avgRatio = (ratios[0] + ratios[1]) / 2.0;
    const dataQuality = Math.min(1.0, n / GOOD_FRAMES);

    let score: number;
    if (avgRatio > 1.5) {
      score = 70.0 + dataQuality * 30.0;
    } else if (avgRatio > 1.0) {
      score = 50.0 + avgRatio * 15.0;
    } else if (avgRatio > 0.5) {
      score = 30.0 + avgRatio * 20.0;
    } else {
      score = n > GOOD_FRAMES ? 10.0 : 30.0;
    }
    score = Math.max(0.0, Math.min(100.0, score));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        tremor_ratio: round(avgRatio, 4),
        tremor_x: round(ratios[0], 4),
        tremor_y: round(ratios[1], 4),
        measured_fps: round(this.fps, 1),
        frames: n,
        data_quality: round(dataQuality, 2),
      },
      performance.now() - start,
    );
  }

  /**
   * Detrend → window → DFT → tremor-band power / total-power.
   * Returns a unitless ratio; >1 means tremor band has above-average power.
   */
  private bandPowerRatio(signal: Float64Array): number {
    const n = signal.length;
    // === Detrend (Python: signal - moving-average kernel, mode='same'). ===
    const k = Math.min(15, Math.max(1, Math.floor(n / 3)));
    if (k > 1) {
      const ma = movingAverageSame(signal, k);
      for (let i = 0; i < n; i++) signal[i] -= ma[i];
    }

    // === Hanning window (in-place). ===
    for (let i = 0; i < n; i++) {
      // Python np.hanning: 0.5 * (1 - cos(2π i / (N-1))). N==1 → 1.0.
      const w =
        n > 1 ? 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1))) : 1.0;
      signal[i] *= w;
    }

    // === Hand-rolled DFT (real input, only positive freqs needed). ===
    // ~30 lines, O(n²) but n≤90 so fast enough.
    const halfN = Math.floor(n / 2) + 1;
    const magnitude = new Float64Array(halfN);
    for (let kIdx = 0; kIdx < halfN; kIdx++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < n; t++) {
        const angle = (-2 * Math.PI * kIdx * t) / n;
        re += signal[t] * Math.cos(angle);
        im += signal[t] * Math.sin(angle);
      }
      magnitude[kIdx] = Math.sqrt(re * re + im * im);
    }
    // Frequency axis: freqs[k] = k * fps / n.
    const dF = this.fps / n;

    let tremorSum = 0;
    let tremorCount = 0;
    let totalSum = 0;
    let totalCount = 0;
    for (let kIdx = 0; kIdx < halfN; kIdx++) {
      const f = kIdx * dF;
      if (f >= TREMOR_LOW_HZ && f <= TREMOR_HIGH_HZ) {
        tremorSum += magnitude[kIdx];
        tremorCount += 1;
      }
      if (f > 0.5) {
        totalSum += magnitude[kIdx];
        totalCount += 1;
      }
    }
    if (tremorCount === 0) return 0;
    const tremorPower = tremorSum / tremorCount;
    const totalPower = totalCount > 0 ? totalSum / totalCount : 1e-6;
    return tremorPower / Math.max(totalPower, 1e-6);
  }

  reset(): void {
    this.states.clear();
    this.frameTimes = [];
    this.currentLandmarks = null;
  }
}

/**
 * Centered moving average with mode='same' semantics:
 * out[i] = mean(signal[i-k/2 .. i+k/2]), clamping window at endpoints.
 */
function movingAverageSame(
  signal: ArrayLike<number>,
  k: number,
): Float64Array {
  const n = signal.length;
  const out = new Float64Array(n);
  const half = Math.floor(k / 2);
  // Naive O(nk). k≤15, n≤90 — < 1k mults.
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
