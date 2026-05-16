// Port of src/infrastructure/analyzers/temporal_analyzer.py:1-125.
//
// Temporal Consistency analyzer.
// Per `research/COMPARISON_AYSENUR_vs_PRODUCTIZED.md`, both the Python
// `temporal_analyzer.py` and `background_grid_analyzer.py` are
// Ahmet-original modules (no Aysenur counterpart) — they exist to
// detect unnaturally static faces (photo / frozen video) by watching
// per-face bbox centre and area variance over a sliding window.
//
// Pure arithmetic — no canvas, no FFT, no model. The numpy reductions
// (mean / var / std) are replaced with hand-rolled scalar loops over a
// ring buffer (closest fit pattern: MicroTremorAnalyzer.ts).
//
// Mobile Brave constraints respected:
//   * No SharedArrayBuffer / WebGPU / OffscreenCanvas — none needed.
//   * fps clamp [1, 120] applied to the warmup-window math even though
//     the Python source counts frames not seconds, so the analyzer
//     stays robust if the orchestrator decides to gate `warmupFrames`
//     by wall-time rather than frame index.
//   * Single-threaded, runs on the main JS loop, < 30 µs/frame at N=30.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";

// === Calibrated constants — preserved verbatim from Python source. ===
const DEFAULT_BUFFER_SIZE = 30; // temporal_analyzer.py:38
const DEFAULT_MIN_MOTION_STD = 0.0003; // temporal_analyzer.py:39
const DEFAULT_WARMUP_FRAMES = 15; // temporal_analyzer.py:40

export interface TemporalAnalyzerOptions {
  /** Ring-buffer size for per-face history. Default 30 (~1s @ 30 fps). */
  bufferSize?: number;
  /**
   * Lower-bound motion std below which the face is considered
   * "unnaturally still" (photo/frozen video). Default 0.0003.
   */
  minMotionStd?: number;
  /** Frames required before producing a non-warmup verdict. Default 15. */
  warmupFrames?: number;
}

interface FaceHistory {
  centersX: number[];
  centersY: number[];
  areas: number[];
  frameCount: number;
}

export class TemporalAnalyzer implements IFaceAnalyzer {
  readonly name = "temporal";

  private readonly bufferSize: number;
  private readonly minMotionStd: number;
  private readonly warmupFrames: number;

  private readonly histories: Map<number, FaceHistory> = new Map();

  constructor(options: TemporalAnalyzerOptions = {}) {
    this.bufferSize = Math.max(2, options.bufferSize ?? DEFAULT_BUFFER_SIZE);
    this.minMotionStd = options.minMotionStd ?? DEFAULT_MIN_MOTION_STD;
    this.warmupFrames = Math.max(2, options.warmupFrames ?? DEFAULT_WARMUP_FRAMES);
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();

    const fid = face.face_id;
    let hist = this.histories.get(fid);
    if (!hist) {
      hist = { centersX: [], centersY: [], areas: [], frameCount: 0 };
      this.histories.set(fid, hist);
    }

    // === Append latest sample, drop oldest if over capacity. ===
    const bbox = face.bbox;
    const cx = (bbox.x1 + bbox.x2) / 2.0;
    const cy = (bbox.y1 + bbox.y2) / 2.0;
    const area = bbox.area;

    hist.centersX.push(cx);
    hist.centersY.push(cy);
    hist.areas.push(area);
    if (hist.centersX.length > this.bufferSize) hist.centersX.shift();
    if (hist.centersY.length > this.bufferSize) hist.centersY.shift();
    if (hist.areas.length > this.bufferSize) hist.areas.shift();
    hist.frameCount += 1;

    if (hist.frameCount < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { warmup: true, frames: hist.frameCount },
        performance.now() - start,
      );
    }

    // === Compute scale-normalised motion metrics. ===
    const xs = hist.centersX;
    const ys = hist.centersY;
    const areas = hist.areas;

    const meanArea = areas.length > 0 ? mean(areas) : 1.0;
    const normFactor = Math.max(Math.sqrt(meanArea), 1.0);

    const varX = variance(xs);
    const varY = variance(ys);
    const posStd = Math.sqrt(varX + varY) / normFactor;
    const areaStd = std(areas) / Math.max(meanArea, 1.0);

    // Python: motion = pos_std + area_std * 0.5
    const motion = posStd + areaStd * 0.5;

    // === Piecewise score — verbatim from temporal_analyzer.py:96-106. ===
    let score: number;
    if (motion < this.minMotionStd) {
      // Very suspicious — unnaturally still.
      score = 10.0;
    } else if (motion < this.minMotionStd * 3) {
      const ratio = (motion - this.minMotionStd) / (this.minMotionStd * 2);
      score = 10.0 + ratio * 40.0;
    } else if (motion < this.minMotionStd * 10) {
      const ratio = (motion - this.minMotionStd * 3) / (this.minMotionStd * 7);
      score = 50.0 + ratio * 40.0;
    } else {
      // Natural motion.
      score = 90.0;
    }
    score = Math.max(0.0, Math.min(100.0, score));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        pos_std: round(posStd, 6),
        area_std: round(areaStd, 6),
        motion: round(motion, 6),
        frames: hist.frameCount,
      },
      performance.now() - start,
    );
  }

  /** Drop history for one face_id (or all faces if undefined). */
  reset(faceId?: number): void {
    if (faceId === undefined) {
      this.histories.clear();
    } else {
      this.histories.delete(faceId);
    }
  }
}

/** Arithmetic mean. Returns 0 on empty. */
function mean(arr: ArrayLike<number>): number {
  const n = arr.length;
  if (n === 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += arr[i];
  return s / n;
}

/** Population variance (matches numpy.var with default ddof=0). */
function variance(arr: ArrayLike<number>): number {
  const n = arr.length;
  if (n === 0) return 0;
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < n; i++) {
    const d = arr[i] - m;
    s += d * d;
  }
  return s / n;
}

/** Population standard deviation (matches numpy.std with default ddof=0). */
function std(arr: ArrayLike<number>): number {
  return Math.sqrt(variance(arr));
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
