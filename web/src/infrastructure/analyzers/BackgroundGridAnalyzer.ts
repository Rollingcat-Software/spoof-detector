// Port of src/infrastructure/analyzers/background_grid_analyzer.py:1-155.
//
// Background Grid analyzer.
// Per `research/COMPARISON_AYSENUR_vs_PRODUCTIZED.md`, both this and
// `temporal_analyzer.py` are Ahmet-original modules (no Aysenur
// counterpart). The paper's leave-one-out study shows background_grid
// is the only marginal positive auxiliary contributor at +0.014 AUC
// zero-shot — kept in the bundle for that ablation contribution.
//
// Algorithm: divide the full frame into a 6×4 grid (24 cells), drop
// cells whose centre overlaps the face bbox, then per remaining cell:
//   * track grayscale mean over a 60-frame history; flag as "stable"
//     if the std over the last 15 samples is below STABILITY_THRESHOLD;
//   * flag as "specular" if >5% of pixels read bright + low-saturation
//     in HSV (screen glare signature);
//   * flag as "cool" if >15% of pixels lie in the blue HSV hue band
//     (screen / LED colour temperature signature).
// Score = stable*60 + 30 − specular*20 − cool*10, clamped [0, 100].
//
// cv2 ops the Python source uses + the hand-rolled equivalents shipped
// here (closest fit pattern: DeviceBoundaryAnalyzer.ts):
//   * cv2.cvtColor BGR→GRAY     → luminance loop (BT.601 coeffs).
//   * cv2.cvtColor BGR→HSV      → rgbToHsv8 (OpenCV 8-bit convention:
//                                  H ∈ [0..179], S ∈ [0..255], V ∈ [0..255]).
//   * np.mean / np.std          → scalar loops over Uint8Array slices.
//   * collections.deque(60)     → number[] with shift() once length > 60.
//
// Mobile Brave constraints respected:
//   * No SharedArrayBuffer / WebGPU / OffscreenCanvas access.
//   * Single-threaded, runs on the main JS loop. A 640×480 frame at
//     a 6×4 grid produces ~24 cells of ~12800 pixels each; the per-frame
//     work is ≈300k pixel reads — comfortably under 5 ms on mobile.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { SourceImage, toImageData } from "../../utils/imageOps";

// === Calibrated constants — preserved verbatim from Python source. ===
const GRID_COLS = 6; // background_grid_analyzer.py:26
const GRID_ROWS = 4; // background_grid_analyzer.py:27
const MIN_FRAMES = 15; // background_grid_analyzer.py:28 (~0.5s baseline)
const STABILITY_THRESHOLD = 12.0; // background_grid_analyzer.py:29
const HISTORY_LEN = 60; // background_grid_analyzer.py:100 (deque(maxlen=60))
const RECENT_WINDOW = 15; // background_grid_analyzer.py:106 (history[-15:])

// Per-cell HSV thresholds — verbatim from Python source.
const SPECULAR_VAL_MIN = 230; // val > 230
const SPECULAR_SAT_MAX = 40; // sat < 40
const SPECULAR_RATIO_FLAG = 0.05; // > 0.05 → cell flagged
const COOL_HUE_LOW = 100; // OpenCV 8-bit H range (0..179)
const COOL_HUE_HIGH = 130;
const COOL_RATIO_FLAG = 0.15; // > 0.15 → cell flagged

export interface BackgroundGridAnalyzerOptions {
  /**
   * If true, return a neutral 50.0 score with `error: "no_frame"` until
   * `setFrame()` has been called at least once. Default true — matches
   * the Python source's behaviour.
   */
  requireFrame?: boolean;
}

export class BackgroundGridAnalyzer implements IFaceAnalyzer {
  readonly name = "background_grid";

  private readonly requireFrame: boolean;
  private currentFrame: SourceImage | null = null;
  private frameCount = 0;

  // Per-cell rolling history of grayscale means (key = row * COLS + col).
  private readonly cellHistory: Map<number, number[]> = new Map();

  constructor(options: BackgroundGridAnalyzerOptions = {}) {
    this.requireFrame = options.requireFrame !== false;
  }

  /**
   * Provide the latest full frame. The analyzer needs the full frame
   * (not just the face crop) because it inspects background cells
   * outside the face bbox.
   */
  setFrame(frame: SourceImage): void {
    this.currentFrame = frame;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    this.frameCount += 1;

    if (!this.currentFrame) {
      if (this.requireFrame) {
        return makeAnalyzerResult(
          this.name,
          50.0,
          { error: "no_frame" },
          performance.now() - start,
        );
      }
    }

    const frame = toImageData(this.currentFrame as SourceImage);
    const W = frame.width;
    const H = frame.height;

    if (W <= 0 || H <= 0) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "empty_frame" },
        performance.now() - start,
      );
    }

    const cellH = Math.floor(H / GRID_ROWS);
    const cellW = Math.floor(W / GRID_COLS);
    if (cellH <= 0 || cellW <= 0) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "frame_too_small" },
        performance.now() - start,
      );
    }

    const fb = face.bbox;

    let stableCells = 0;
    let totalBgCells = 0;
    let specularCells = 0;
    let coolCells = 0;

    for (let row = 0; row < GRID_ROWS; row++) {
      for (let col = 0; col < GRID_COLS; col++) {
        const y1 = row * cellH;
        const x1 = col * cellW;
        const y2 = Math.min(H, y1 + cellH);
        const x2 = Math.min(W, x1 + cellW);

        // === Skip cells whose centre overlaps the face bbox. ===
        const cellCx = Math.floor((x1 + x2) / 2);
        const cellCy = Math.floor((y1 + y2) / 2);
        if (
          fb.x1 <= cellCx &&
          cellCx <= fb.x2 &&
          fb.y1 <= cellCy &&
          cellCy <= fb.y2
        ) {
          continue;
        }
        totalBgCells += 1;

        // === Per-cell pixel sweep: gray mean + HSV ratios in one pass. ===
        const cellMetrics = sweepCell(frame.data, W, x1, y1, x2, y2);
        const cellMean = cellMetrics.meanGray;

        const key = row * GRID_COLS + col;
        let hist = this.cellHistory.get(key);
        if (!hist) {
          hist = [];
          this.cellHistory.set(key, hist);
        }
        hist.push(cellMean);
        if (hist.length > HISTORY_LEN) hist.shift();

        // === Stability check (std over last 15 samples). ===
        if (hist.length >= MIN_FRAMES) {
          const recent = hist.slice(-RECENT_WINDOW);
          const recentStd = stdNum(recent);
          if (recentStd < STABILITY_THRESHOLD) {
            stableCells += 1;
          }
        }

        // === Specular check (screen-glare signature). ===
        if (cellMetrics.specularRatio > SPECULAR_RATIO_FLAG) {
          specularCells += 1;
        }

        // === Cool-hue check (screen / LED colour-temp signature). ===
        if (cellMetrics.coolRatio > COOL_RATIO_FLAG) {
          coolCells += 1;
        }
      }
    }

    const elapsed = performance.now() - start;

    if (totalBgCells === 0 || this.frameCount < MIN_FRAMES) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { warmup: true },
        elapsed,
      );
    }

    const stabilityRatio = stableCells / totalBgCells;
    const specularRatio = specularCells / totalBgCells;
    const coolRatio = coolCells / totalBgCells;

    // === Final score — verbatim from background_grid_analyzer.py:135-139. ===
    const stabilityScore = stabilityRatio * 60.0;
    const specularPenalty = specularRatio * 20.0;
    const coolPenalty = coolRatio * 10.0;
    const score = Math.max(
      0.0,
      Math.min(100.0, stabilityScore + 30.0 - specularPenalty - coolPenalty),
    );

    return makeAnalyzerResult(
      this.name,
      score,
      {
        stability_ratio: round(stabilityRatio, 3),
        stable_cells: stableCells,
        total_bg_cells: totalBgCells,
        specular_cells: specularCells,
        cool_cells: coolCells,
        specular_ratio: round(specularRatio, 3),
        cool_ratio: round(coolRatio, 3),
      },
      elapsed,
    );
  }

  reset(): void {
    this.cellHistory.clear();
    this.frameCount = 0;
    this.currentFrame = null;
  }
}

/**
 * Single pass over a cell rectangle in RGBA frame data:
 *   * accumulates grayscale (BT.601) mean,
 *   * counts HSV-specular pixels (bright + low saturation),
 *   * counts HSV-cool pixels (hue ∈ [100, 130] in OpenCV 8-bit).
 *
 * Returns ratios in [0, 1] (specular / cool) and a float gray mean.
 *
 * cv2 equivalents collapsed into this loop:
 *   * cv2.cvtColor(frame, COLOR_BGR2GRAY) → luminance via 0.299R+0.587G+0.114B.
 *   * cv2.cvtColor(frame, COLOR_BGR2HSV)  → rgbToHsv8 inlined per pixel.
 *   * np.mean / boolean-mask reductions   → scalar counters.
 */
function sweepCell(
  data: Uint8ClampedArray,
  frameW: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): { meanGray: number; specularRatio: number; coolRatio: number } {
  const w = x2 - x1;
  const h = y2 - y1;
  const n = w * h;
  if (n <= 0) return { meanGray: 0, specularRatio: 0, coolRatio: 0 };

  let graySum = 0;
  let specularCount = 0;
  let coolCount = 0;
  const stride = frameW * 4;

  for (let yy = y1; yy < y2; yy++) {
    let off = yy * stride + x1 * 4;
    for (let xx = x1; xx < x2; xx++) {
      const r = data[off];
      const g = data[off + 1];
      const b = data[off + 2];
      off += 4;

      // === BT.601 luma (matches cv2 COLOR_BGR2GRAY). ===
      graySum += 0.299 * r + 0.587 * g + 0.114 * b;

      // === Inlined RGB→HSV (OpenCV 8-bit convention). ===
      const mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
      const mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
      const dlt = mx - mn;
      const V = mx; // 0..255
      const S = mx === 0 ? 0 : Math.round((dlt * 255) / mx); // 0..255

      // Hue in [0..360) → OpenCV [0..179] via H/2.
      let H8 = 0;
      if (dlt > 0) {
        let hDeg: number;
        if (mx === r) hDeg = ((g - b) / dlt) % 6;
        else if (mx === g) hDeg = (b - r) / dlt + 2;
        else hDeg = (r - g) / dlt + 4;
        hDeg *= 60;
        if (hDeg < 0) hDeg += 360;
        H8 = Math.round(hDeg / 2);
      }

      if (V > SPECULAR_VAL_MIN && S < SPECULAR_SAT_MAX) specularCount += 1;
      if (H8 >= COOL_HUE_LOW && H8 <= COOL_HUE_HIGH) coolCount += 1;
    }
  }

  return {
    meanGray: graySum / n,
    specularRatio: specularCount / n,
    coolRatio: coolCount / n,
  };
}

/** Population standard deviation of a number[] (numpy.std default ddof=0). */
function stdNum(arr: ArrayLike<number>): number {
  const n = arr.length;
  if (n === 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += arr[i];
  const m = s / n;
  let v = 0;
  for (let i = 0; i < n; i++) {
    const d = arr[i] - m;
    v += d * d;
  }
  return Math.sqrt(v / n);
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
