// Port of src/infrastructure/analyzers/texture_analyzer.py:1-108
//
// Texture Analyzer — passive liveness via three channels:
//   * Laplacian variance (sharp vs blurry texture)
//   * HSV mean-saturation / std-of-value naturalness check
//   * FFT high/low frequency ratio on a 192×108 thumbnail
//
// Approximations vs Python source:
//   * cv2.cvtColor BGR→HSV is replaced with hand-rolled RGB→HSV. The
//     browser delivers RGBA so this is the *natural* form; in the Python
//     pipeline cv2 reads BGR. RGB↔BGR re-orders only which channel is
//     "max" for the Hue computation; Saturation and Value (used by the
//     color score) are channel-symmetric (depend on min/max/max of all
//     three) so the threshold calibration is preserved.
//   * cv2.Laplacian(CV_64F) is a 3×3 [[0,1,0],[1,-4,1],[0,1,0]] kernel.
//     Variance over the result is the same statistic Python measures.
//   * np.fft.fft2 + np.fft.fftshift on the 192×108 thumbnail is replaced
//     with a hand-rolled 2D DFT (row 1D-DFT → column 1D-DFT). At 192×108
//     this is O(N·M·(N+M)) ≈ 6.2M mults; we drop the thumbnail to 48×27
//     by default to keep this under ~120k mults (~5ms in node, well under
//     the Python budget). Configurable via `fftDownsample`.
//   * The 2D fftshift convention (move DC to centre) is preserved so the
//     low/high region masks line up exactly with the Python source.
//
// All calibrated thresholds (texture 100.0, color 0.3, frequency 0.5,
// 50% / 80% colour-score break points) are carried over verbatim.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { SourceImage, toImageData } from "../../utils/imageOps";

export interface TextureOptions {
  /** Variance break point for the texture sub-score. Default 100.0. */
  textureThreshold?: number;
  /** Deviation break point for the color sub-score. Default 0.3. */
  colorThreshold?: number;
  /** High/low FFT energy ratio break point. Default 0.5. */
  frequencyThreshold?: number;
  /**
   * Thumbnail size [w, h] for the FFT pass. The Python default is
   * (192, 108); the hand-rolled DFT here defaults to (48, 27) for a
   * ~16× speed-up. The mid/low region split is computed from `rows//8`
   * and `cols//8` so the ratio remains comparable.
   */
  fftDownsample?: readonly [number, number];
}

const DEFAULT_FFT_DOWNSAMPLE: readonly [number, number] = [48, 27];

export class TextureAnalyzer implements IFaceAnalyzer {
  readonly name = "texture";

  private readonly textureThreshold: number;
  private readonly colorThreshold: number;
  private readonly frequencyThreshold: number;
  private readonly fftW: number;
  private readonly fftH: number;
  private currentFrame: SourceImage | null = null;

  // Verbatim from Python._weights.
  private readonly weights = {
    texture: 0.40,
    color: 0.30,
    frequency: 0.30,
  } as const;

  constructor(options: TextureOptions = {}) {
    this.textureThreshold = options.textureThreshold ?? 100.0;
    this.colorThreshold = options.colorThreshold ?? 0.3;
    this.frequencyThreshold = options.frequencyThreshold ?? 0.5;
    const ds = options.fftDownsample ?? DEFAULT_FFT_DOWNSAMPLE;
    this.fftW = Math.max(8, Math.floor(ds[0]));
    this.fftH = Math.max(8, Math.floor(ds[1]));
  }

  /** Same setFrame() shape as ScreenFlickerAnalyzer / DeviceBoundary. */
  setFrame(frame: SourceImage): void {
    this.currentFrame = frame;
  }

  analyze(faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();

    // === Resolve the working pixel buffer. ===
    const crop = this.resolveCrop(faceCrop, face);
    if (!crop) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "no_pixels" },
        performance.now() - start,
      );
    }

    const w = crop.width;
    const h = crop.height;
    if (w < 4 || h < 4) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: "too_small", width: w, height: h },
        performance.now() - start,
      );
    }

    const gray = rgbaToGrayscale(crop);
    const hsv = rgbaToHsv(crop);

    const textureScore = this.textureScore(gray, w, h);
    const colorScore = this.colorScore(hsv);

    const small = resizeGrayscale(gray, w, h, this.fftW, this.fftH);
    const frequencyScore = this.frequencyScore(small, this.fftW, this.fftH);

    const combined =
      textureScore * this.weights.texture +
      colorScore * this.weights.color +
      frequencyScore * this.weights.frequency;
    const score = Math.max(0.0, Math.min(100.0, combined));

    return makeAnalyzerResult(
      this.name,
      score,
      {
        texture_score: round(textureScore, 4),
        color_score: round(colorScore, 4),
        frequency_score: round(frequencyScore, 4),
      },
      performance.now() - start,
    );
  }

  reset(): void {
    this.currentFrame = null;
  }

  private resolveCrop(
    faceCrop: ImageData | null,
    face: FaceROI,
  ): ImageData | null {
    if (faceCrop && faceCrop.width > 0 && faceCrop.height > 0) {
      return faceCrop;
    }
    if (!this.currentFrame) return null;
    const frame = toImageData(this.currentFrame);
    const x1 = Math.max(0, Math.floor(face.bbox.x1));
    const y1 = Math.max(0, Math.floor(face.bbox.y1));
    const x2 = Math.min(frame.width, Math.floor(face.bbox.x2));
    const y2 = Math.min(frame.height, Math.floor(face.bbox.y2));
    const cw = x2 - x1;
    const ch = y2 - y1;
    if (cw <= 0 || ch <= 0) return null;
    const out = new Uint8ClampedArray(cw * ch * 4);
    const stride = frame.width * 4;
    for (let y = 0; y < ch; y++) {
      const srcOff = (y1 + y) * stride + x1 * 4;
      const dstOff = y * cw * 4;
      out.set(frame.data.subarray(srcOff, srcOff + cw * 4), dstOff);
    }
    return { data: out, width: cw, height: ch, colorSpace: "srgb" } as ImageData;
  }

  /** Port of `_texture_score`. */
  private textureScore(gray: Float32Array, w: number, h: number): number {
    const lap = laplacian3x3(gray, w, h);
    const variance = sampleVariance(lap);
    if (variance >= this.textureThreshold) {
      return Math.min(
        100.0,
        50.0 + (variance - this.textureThreshold) * 0.2,
      );
    }
    return Math.max(0.0, (variance / this.textureThreshold) * 50.0);
  }

  /** Port of `_color_score`. Uses S (mean) and V (std) channels of HSV. */
  private colorScore(hsv: HsvBuffer): number {
    const n = hsv.s.length;
    let sSum = 0;
    let vSum = 0;
    for (let i = 0; i < n; i++) {
      sSum += hsv.s[i];
      vSum += hsv.v[i];
    }
    const sMean = sSum / n;
    const vMean = vSum / n;
    let vVar = 0;
    for (let i = 0; i < n; i++) {
      const d = hsv.v[i] - vMean;
      vVar += d * d;
    }
    const vStd = Math.sqrt(vVar / n);

    const satDeviation = Math.abs(sMean - 80) / 128.0;
    const valDeviation = Math.abs(vStd - 50) / 64.0;
    const combined = (satDeviation + valDeviation) / 2.0;
    if (combined <= this.colorThreshold) {
      return 100.0 - (combined / this.colorThreshold) * 30.0;
    }
    return Math.max(0.0, 70.0 - (combined - this.colorThreshold) * 100.0);
  }

  /** Port of `_frequency_score`. */
  private frequencyScore(
    graySmall: Float32Array,
    w: number,
    h: number,
  ): number {
    // 2D DFT magnitude, fft-shifted (DC at centre).
    const magnitude = fft2Magnitude(graySmall, w, h);
    fftShift2D(magnitude, w, h);

    const cr = Math.floor(h / 2);
    const cc = Math.floor(w / 2);
    const dRow8 = Math.floor(h / 8);
    const dCol8 = Math.floor(w / 8);
    const dRow4 = Math.floor(h / 4);
    const dCol4 = Math.floor(w / 4);

    // Low region: central ±(rows//8, cols//8) box.
    let lowSum = 0;
    let lowCount = 0;
    for (let y = cr - dRow8; y < cr + dRow8; y++) {
      if (y < 0 || y >= h) continue;
      for (let x = cc - dCol8; x < cc + dCol8; x++) {
        if (x < 0 || x >= w) continue;
        lowSum += magnitude[y * w + x];
        lowCount += 1;
      }
    }

    // High mask = everything OUTSIDE the central ±(rows//4, cols//4) box.
    let highSum = 0;
    let highCount = 0;
    for (let y = 0; y < h; y++) {
      const insideY = y >= cr - dRow4 && y < cr + dRow4;
      for (let x = 0; x < w; x++) {
        const insideX = x >= cc - dCol4 && x < cc + dCol4;
        if (insideY && insideX) continue;
        highSum += magnitude[y * w + x];
        highCount += 1;
      }
    }

    const lowMean = (lowCount > 0 ? lowSum / lowCount : 0) + 1e-6;
    const highMean = (highCount > 0 ? highSum / highCount : 0) + 1e-6;
    const ratio = highMean / lowMean;

    if (ratio < this.frequencyThreshold) {
      return 100.0 - (1.0 - ratio / this.frequencyThreshold) * 40.0;
    }
    if (ratio > this.frequencyThreshold * 2) {
      return Math.max(
        0.0,
        60.0 - (ratio - this.frequencyThreshold * 2) * 50.0,
      );
    }
    return 80.0;
  }
}

// ---------------------------------------------------------------------------
// Pixel-buffer helpers (shared shape with ScreenReplayAnalyzer, kept local
// per task spec — analyzers must not share state).
// ---------------------------------------------------------------------------

interface HsvBuffer {
  h: Float32Array; // 0..179 (cv2 scale)
  s: Float32Array; // 0..255
  v: Float32Array; // 0..255
}

/** RGBA → 8-bit-style grayscale stored as Float32 (matches cv2 ITU-R 601). */
function rgbaToGrayscale(img: ImageData): Float32Array {
  const n = img.width * img.height;
  const out = new Float32Array(n);
  const data = img.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    out[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  }
  return out;
}

/** Standard RGB→HSV; cv2 scale (H ∈ [0,179], S/V ∈ [0,255]). */
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
    // cv2 stores H in [0,179] (= degrees / 2).
    h[i] = hue / 2.0;
  }
  return { h, s, v };
}

/** 3×3 Laplacian: [[0,1,0],[1,-4,1],[0,1,0]]. Border replicated. */
function laplacian3x3(
  gray: Float32Array,
  w: number,
  h: number,
): Float32Array {
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const yu = y > 0 ? y - 1 : 0;
    const yd = y < h - 1 ? y + 1 : h - 1;
    for (let x = 0; x < w; x++) {
      const xl = x > 0 ? x - 1 : 0;
      const xr = x < w - 1 ? x + 1 : w - 1;
      const c = gray[y * w + x];
      out[y * w + x] =
        gray[yu * w + x] +
        gray[yd * w + x] +
        gray[y * w + xl] +
        gray[y * w + xr] -
        4 * c;
    }
  }
  return out;
}

/** Population variance (np.var default ddof=0). */
function sampleVariance(buf: Float32Array): number {
  const n = buf.length;
  if (n === 0) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += buf[i];
  const mean = s / n;
  let v = 0;
  for (let i = 0; i < n; i++) {
    const d = buf[i] - mean;
    v += d * d;
  }
  return v / n;
}

/**
 * Nearest-neighbour resize used for the FFT thumbnail. cv2.INTER_AREA
 * would give better aliasing properties; for the energy-ratio statistic
 * the difference is negligible at the small output sizes we use.
 */
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

/**
 * 2D DFT magnitude via separable row/column 1D DFTs.
 * Real input → complex output; we only need |F|.
 */
function fft2Magnitude(
  signal: Float32Array,
  w: number,
  h: number,
): Float32Array {
  // Stage 1: 1D DFT on each row. We'll store complex output as parallel
  // Float64Arrays sized w*h.
  const rowRe = new Float64Array(w * h);
  const rowIm = new Float64Array(w * h);
  const cosK = new Float64Array(w * w);
  const sinK = new Float64Array(w * w);
  for (let k = 0; k < w; k++) {
    for (let t = 0; t < w; t++) {
      const a = (-2 * Math.PI * k * t) / w;
      cosK[k * w + t] = Math.cos(a);
      sinK[k * w + t] = Math.sin(a);
    }
  }
  for (let y = 0; y < h; y++) {
    const base = y * w;
    for (let k = 0; k < w; k++) {
      let re = 0;
      let im = 0;
      for (let t = 0; t < w; t++) {
        const x = signal[base + t];
        re += x * cosK[k * w + t];
        im += x * sinK[k * w + t];
      }
      rowRe[base + k] = re;
      rowIm[base + k] = im;
    }
  }

  // Stage 2: 1D DFT on each column. Complex input → complex output.
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
        // (ar + i*ai) * (c + i*s) = (ar*c - ai*s) + i*(ar*s + ai*c)
        re += ar * c - ai * s;
        im += ar * s + ai * c;
      }
      mag[k * w + x] = Math.sqrt(re * re + im * im);
    }
  }
  return mag;
}

/** np.fft.fftshift: swap quadrants so DC ends up at the centre. */
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
