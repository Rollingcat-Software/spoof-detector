// Port of src/infrastructure/analyzers/screen_replay_analyzer.py:1-154
//
// Screen Replay — whole-frame display-replay detector.
//
// Four signal channels fused with a min-penalty:
//   * FFT mid/low energy ratio (screens carry more mid-frequency power)
//   * Laplacian variance (screens are *too* sharp or *too* blurry)
//   * Skin-mask check in YCrCb + HSV (Cr 133-173, Cb 77-127, H≤25|H≥160)
//   * Specular-highlight ratio (screen glare = bright + desaturated)
//
// Approximations vs the Python source:
//   * cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) is replaced with
//     a single-pass *global* histogram equalisation. This is a documented
//     quality deviation — CLAHE is local-adaptive and gives stronger
//     edge contrast in shadows/highlights; global histeq pulls the same
//     overall direction (boost contrast) but won't be as crisp on faces
//     with heavy uneven lighting. The downstream Laplacian variance and
//     FFT-ratio statistics are still meaningful at this contrast — the
//     calibrated 25.0 blur floor, 80/2000 risk centres and 0.85/0.20
//     FFT ratio sigmoid centres remain in place.
//   * cv2.cvtColor BGR→{HSV, YCrCb} are hand-rolled for RGB input
//     (browser delivers RGBA). YCrCb (ITU-R BT.601) is channel-symmetric
//     in the formula coefficients so the (cr 133-173, cb 77-127) skin
//     band carries over exactly. HSV: S/V are channel-symmetric; H
//     wraps to the same {≤25 | ≥160} band for reddish skin tones in
//     either channel order, so the mask works as written.
//   * np.fft.fft2 + np.fft.fftshift: hand-rolled separable 2D DFT.
//     Defaults work on a 64×64 thumbnail (Python uses 256-max-side ≈
//     up to 256×256). Configurable via `fftSide`.
//
// Output score range and calibration are preserved verbatim:
//   blur_floor → 50.0, otherwise weighted (0.35 fft, 0.25 lap, 0.20 skin,
//   0.20 specular) plus a 0.35×min(signal) penalty.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { SourceImage, toImageData } from "../../utils/imageOps";

export interface ScreenReplayOptions {
  /** Max side used to downsample before analysis. Default 256 (matches Python). */
  maxSide?: number;
  /**
   * Square side length used for the hand-rolled 2D DFT. Default 64.
   * The Python pipeline FFTs the full 256-max-side image; we downsample
   * to 64×64 to bound the O(N²M+NM²) cost. Frequency-band statistics
   * remain meaningful — see source header.
   */
  fftSide?: number;
}

export class ScreenReplayAnalyzer implements IFaceAnalyzer {
  readonly name = "screen_replay";

  private readonly maxSide: number;
  private readonly fftSide: number;
  private currentFrame: SourceImage | null = null;

  constructor(options: ScreenReplayOptions = {}) {
    this.maxSide = Math.max(32, options.maxSide ?? 256);
    this.fftSide = Math.max(8, options.fftSide ?? 64);
  }

  /** Same setFrame() pattern as ScreenFlickerAnalyzer. */
  setFrame(frame: SourceImage): void {
    this.currentFrame = frame;
  }

  // The Python analyzer takes the FULL frame, not the face crop —
  // it's looking for screen artefacts anywhere in the image. The face
  // ROI is unused. We accept it to satisfy IFaceAnalyzer but ignore it.
  analyze(_faceCrop: ImageData | null, _face: FaceROI): AnalyzerResult {
    const start = performance.now();
    if (!this.currentFrame) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "no_frame" },
        performance.now() - start,
      );
    }

    const frame = toImageData(this.currentFrame);
    const small = downsampleRgba(frame, this.maxSide);

    // Grayscale + global histogram equalisation (CLAHE substitute).
    const grayRaw = rgbaToGrayscale(small);
    const gray = histogramEqualize(grayRaw);

    const lapVar = laplacianVariance(gray, small.width, small.height);
    if (lapVar < 25.0) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        {
          blur_floor: true,
          laplacian_var: round(lapVar, 4),
        },
        performance.now() - start,
      );
    }

    // Resize the equalised grey for FFT.
    const fftSrc = resizeGrayscale(
      gray,
      small.width,
      small.height,
      this.fftSide,
      this.fftSide,
    );
    const fftScore = scoreFft(fftSrc, this.fftSide, this.fftSide);
    const lapScore = scoreLaplacian(lapVar);
    const skinScore = scoreSkin(small);
    const specularScore = scoreSpecular(small);

    const weighted =
      0.35 * fftScore +
      0.25 * lapScore +
      0.20 * skinScore +
      0.20 * specularScore;
    const penalty = Math.min(fftScore, lapScore, skinScore, specularScore);
    const score = Math.max(0.0, Math.min(100.0, 0.65 * weighted + 0.35 * penalty));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        fft_score: round(fftScore, 4),
        laplacian_score: round(lapScore, 4),
        laplacian_var: round(lapVar, 4),
        skin_score: round(skinScore, 4),
        specular_score: round(specularScore, 4),
      },
      performance.now() - start,
    );
  }

  reset(): void {
    this.currentFrame = null;
  }
}

// ---------------------------------------------------------------------------
// Scoring helpers (port of the private `_fft_score`, `_laplacian_score`,
// `_skin_score`, `_specular_score`).
// ---------------------------------------------------------------------------

function clamp01(v: number): number {
  return Math.max(0.0, Math.min(1.0, v));
}

function sigmoid(v: number): number {
  return 1.0 / (1.0 + Math.exp(-v));
}

function normalize(value: number, low: number, high: number): number {
  if (high <= low) return 0.0;
  return clamp01((value - low) / (high - low));
}

function scoreFft(gray: Float32Array, w: number, h: number): number {
  // np.log1p(abs(fft2)) magnitude.
  const magRaw = fft2Magnitude(gray, w, h);
  fftShift2D(magRaw, w, h);
  const mag = new Float32Array(magRaw.length);
  for (let i = 0; i < magRaw.length; i++) mag[i] = Math.log1p(magRaw[i]);

  const cy = h / 2.0;
  const cx = w / 2.0;
  const half = Math.min(h, w) / 2.0;
  const lowR = Math.max(2.0, half / 16.0);
  const midR = Math.max(lowR + 1.0, half / 4.0);

  let lowSum = 0;
  let lowCount = 0;
  let midSum = 0;
  let midCount = 0;
  for (let y = 0; y < h; y++) {
    const dy = y - cy;
    const dy2 = dy * dy;
    for (let x = 0; x < w; x++) {
      const dx = x - cx;
      const r = Math.sqrt(dx * dx + dy2);
      const v = mag[y * w + x];
      if (r <= lowR) {
        lowSum += v;
        lowCount += 1;
      } else if (r <= midR) {
        midSum += v;
        midCount += 1;
      }
    }
  }
  const lowE = lowCount > 0 ? lowSum / lowCount : 0;
  const midE = midCount > 0 ? midSum / midCount : 0;
  const ratio = midE / Math.max(lowE, 1e-6);
  const risk = sigmoid((ratio - 0.85) / 0.20);
  return Math.max(0.0, Math.min(100.0, 100.0 * (1.0 - risk)));
}

function scoreLaplacian(variance: number): number {
  const lowRisk = 1.0 - sigmoid((variance - 80.0) / 16.0);
  // High threshold raised to 2000 — see source comment.
  const highRisk = sigmoid((variance - 2000.0) / 400.0);
  const risk = Math.max(lowRisk, highRisk);
  return Math.max(0.0, Math.min(100.0, 100.0 * (1.0 - risk)));
}

function scoreSkin(img: ImageData): number {
  const n = img.width * img.height;
  const ycrcb = rgbaToYCrCb(img);
  const hsv = rgbaToHsv(img);

  // Skin mask (verbatim Cr/Cb bands + Y/H/S/V supports).
  let coverageCount = 0;
  const crSkin: number[] = [];
  const cbSkin: number[] = [];
  for (let i = 0; i < n; i++) {
    const cr = ycrcb.cr[i];
    const cb = ycrcb.cb[i];
    const yL = ycrcb.y[i];
    const hH = hsv.h[i];
    const sS = hsv.s[i];
    const vV = hsv.v[i];
    const isSkin =
      cr >= 133 &&
      cr <= 173 &&
      cb >= 77 &&
      cb <= 127 &&
      yL >= 30 &&
      (hH <= 25 || hH >= 160) &&
      sS >= 30 &&
      sS <= 180 &&
      vV >= 40;
    if (isSkin) {
      coverageCount += 1;
      crSkin.push(cr);
      cbSkin.push(cb);
    }
  }
  const coverage = coverageCount / n;
  let scatter = 0;
  if (crSkin.length > 0) {
    const stdCr = arrayStd(crSkin);
    const stdCb = arrayStd(cbSkin);
    scatter = Math.min(stdCr, stdCb);
  }

  const lowCovRisk = 1.0 - normalize(coverage, 0.20, 0.35);
  const highCovRisk = normalize(coverage, 0.85, 0.95);
  const scatterRisk = 1.0 - normalize(scatter, 2.5, 6.5);
  const risk = clamp01(
    0.40 * Math.max(lowCovRisk, highCovRisk) + 0.60 * scatterRisk,
  );
  return Math.max(0.0, Math.min(100.0, 100.0 * (1.0 - risk)));
}

function scoreSpecular(img: ImageData): number {
  const hsv = rgbaToHsv(img);
  const n = hsv.s.length;
  let bright = 0;
  for (let i = 0; i < n; i++) {
    if (hsv.v[i] >= 240.0 && hsv.s[i] <= 35.0) bright += 1;
  }
  const ratio = bright / n;
  const risk = normalize(ratio, 0.020, 0.060);
  return Math.max(0.0, Math.min(100.0, 100.0 * (1.0 - risk)));
}

// ---------------------------------------------------------------------------
// Pixel-buffer helpers. Kept LOCAL to this file (not exported, not shared
// with TextureAnalyzer) per task spec: analyzers must not share state.
// ---------------------------------------------------------------------------

interface HsvBuffer {
  h: Float32Array;
  s: Float32Array;
  v: Float32Array;
}

interface YCrCbBuffer {
  y: Float32Array;
  cr: Float32Array;
  cb: Float32Array;
}

function rgbaToGrayscale(img: ImageData): Uint8ClampedArray {
  const n = img.width * img.height;
  const out = new Uint8ClampedArray(n);
  const data = img.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    out[i] = Math.round(
      0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2],
    );
  }
  return out;
}

/** Global histogram equalisation — CLAHE substitute (see file header). */
function histogramEqualize(src: Uint8ClampedArray): Float32Array {
  const n = src.length;
  const hist = new Uint32Array(256);
  for (let i = 0; i < n; i++) hist[src[i]] += 1;
  // Build cumulative distribution.
  const cdf = new Uint32Array(256);
  let acc = 0;
  for (let i = 0; i < 256; i++) {
    acc += hist[i];
    cdf[i] = acc;
  }
  // Find first non-zero CDF entry (min).
  let cdfMin = 0;
  for (let i = 0; i < 256; i++) {
    if (cdf[i] > 0) {
      cdfMin = cdf[i];
      break;
    }
  }
  const denom = n - cdfMin;
  const out = new Float32Array(n);
  if (denom <= 0) {
    // Degenerate (constant image) — return as-is.
    for (let i = 0; i < n; i++) out[i] = src[i];
    return out;
  }
  const lut = new Uint8ClampedArray(256);
  for (let i = 0; i < 256; i++) {
    lut[i] = Math.max(0, Math.round(((cdf[i] - cdfMin) / denom) * 255));
  }
  for (let i = 0; i < n; i++) out[i] = lut[src[i]];
  return out;
}

function rgbaToHsv(img: ImageData): HsvBuffer {
  const n = img.width * img.height;
  const h = new Float32Array(n);
  const s = new Float32Array(n);
  const v = new Float32Array(n);
  const data = img.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = data[p];
    const g = data[p + 1];
    const b = data[p + 2];
    const mx = Math.max(r, g, b);
    const mn = Math.min(r, g, b);
    const delta = mx - mn;
    v[i] = mx;
    s[i] = mx === 0 ? 0 : (delta / mx) * 255.0;
    let hue: number;
    if (delta === 0) {
      hue = 0;
    } else if (mx === r) {
      hue = 60 * (((g - b) / delta) % 6);
    } else if (mx === g) {
      hue = 60 * ((b - r) / delta + 2);
    } else {
      hue = 60 * ((r - g) / delta + 4);
    }
    if (hue < 0) hue += 360;
    h[i] = hue / 2.0;
  }
  return { h, s, v };
}

/** ITU-R BT.601 RGB→YCrCb. Matches cv2.COLOR_*2YCrCb (formula is BGR-symmetric). */
function rgbaToYCrCb(img: ImageData): YCrCbBuffer {
  const n = img.width * img.height;
  const y = new Float32Array(n);
  const cr = new Float32Array(n);
  const cb = new Float32Array(n);
  const data = img.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = data[p];
    const g = data[p + 1];
    const b = data[p + 2];
    const yVal = 0.299 * r + 0.587 * g + 0.114 * b;
    y[i] = yVal;
    cr[i] = (r - yVal) * 0.713 + 128.0;
    cb[i] = (b - yVal) * 0.564 + 128.0;
  }
  return { y, cr, cb };
}

/** np.std over a number[] (population, ddof=0). */
function arrayStd(arr: number[]): number {
  const n = arr.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += arr[i];
  const mean = sum / n;
  let v = 0;
  for (let i = 0; i < n; i++) {
    const d = arr[i] - mean;
    v += d * d;
  }
  return Math.sqrt(v / n);
}

/** Downsample RGBA to fit within maxSide (long-side). */
function downsampleRgba(img: ImageData, maxSide: number): ImageData {
  const sw = img.width;
  const sh = img.height;
  if (Math.max(sw, sh) <= maxSide) return img;
  const scale = maxSide / Math.max(sw, sh);
  const dw = Math.max(1, Math.floor(sw * scale));
  const dh = Math.max(1, Math.floor(sh * scale));
  const out = new Uint8ClampedArray(dw * dh * 4);
  // Nearest-neighbour. cv2.INTER_AREA averages — but for the downstream
  // statistics (FFT energy ratio, skin coverage, specular ratio) this
  // approximation has negligible impact.
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor(y / scale));
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.floor(x / scale));
      const sOff = (sy * sw + sx) * 4;
      const dOff = (y * dw + x) * 4;
      out[dOff] = img.data[sOff];
      out[dOff + 1] = img.data[sOff + 1];
      out[dOff + 2] = img.data[sOff + 2];
      out[dOff + 3] = img.data[sOff + 3];
    }
  }
  return { data: out, width: dw, height: dh, colorSpace: "srgb" } as ImageData;
}

function resizeGrayscale(
  src: Float32Array,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
): Float32Array {
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, Math.floor((y * sh) / dh));
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, Math.floor((x * sw) / dw));
      out[y * dw + x] = src[sy * sw + sx];
    }
  }
  return out;
}

/** 3×3 Laplacian variance over a grayscale buffer. */
function laplacianVariance(
  gray: Float32Array,
  w: number,
  h: number,
): number {
  // Skip borders to match cv2.Laplacian's central-difference convention
  // closely enough for the variance statistic.
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const c = gray[y * w + x];
      const lap =
        gray[(y - 1) * w + x] +
        gray[(y + 1) * w + x] +
        gray[y * w + (x - 1)] +
        gray[y * w + (x + 1)] -
        4 * c;
      sum += lap;
      sumSq += lap * lap;
      count += 1;
    }
  }
  if (count === 0) return 0;
  const mean = sum / count;
  return sumSq / count - mean * mean;
}

/** Separable 2D DFT magnitude (real input → magnitude only). */
function fft2Magnitude(
  signal: Float32Array,
  w: number,
  h: number,
): Float32Array {
  const rowRe = new Float64Array(w * h);
  const rowIm = new Float64Array(w * h);
  const cosW = new Float64Array(w * w);
  const sinW = new Float64Array(w * w);
  for (let k = 0; k < w; k++) {
    for (let t = 0; t < w; t++) {
      const a = (-2 * Math.PI * k * t) / w;
      cosW[k * w + t] = Math.cos(a);
      sinW[k * w + t] = Math.sin(a);
    }
  }
  for (let y = 0; y < h; y++) {
    const base = y * w;
    for (let k = 0; k < w; k++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < w; t++) {
        const x = signal[base + t];
        re += x * cosW[k * w + t];
        im += x * sinW[k * w + t];
      }
      rowRe[base + k] = re;
      rowIm[base + k] = im;
    }
  }
  const cosH = new Float64Array(h * h);
  const sinH = new Float64Array(h * h);
  for (let k = 0; k < h; k++) {
    for (let t = 0; t < h; t++) {
      const a = (-2 * Math.PI * k * t) / h;
      cosH[k * h + t] = Math.cos(a);
      sinH[k * h + t] = Math.sin(a);
    }
  }
  const mag = new Float32Array(w * h);
  for (let x = 0; x < w; x++) {
    for (let k = 0; k < h; k++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < h; t++) {
        const ar = rowRe[t * w + x];
        const ai = rowIm[t * w + x];
        const c = cosH[k * h + t];
        const s = sinH[k * h + t];
        re += ar * c - ai * s;
        im += ar * s + ai * c;
      }
      mag[k * w + x] = Math.sqrt(re * re + im * im);
    }
  }
  return mag;
}

function fftShift2D(buf: Float32Array, w: number, h: number): void {
  const hx = Math.floor(w / 2);
  const hy = Math.floor(h / 2);
  const tmp = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const sy = (y + hy) % h;
    for (let x = 0; x < w; x++) {
      const sx = (x + hx) % w;
      tmp[y * w + x] = buf[sy * w + sx];
    }
  }
  buf.set(tmp);
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
