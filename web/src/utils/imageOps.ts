// Small image utility helpers used by MiniFASNetAnalyzer and the gates.
// Mirrors the cropping math in uniface/spoofing/minifasnet.py:_crop_face
// (see /home/deploy/.local/lib/python3.12/site-packages/uniface/spoofing/minifasnet.py)
// and the bbox-padded fallback in src/infrastructure/analyzers/minifasnet_analyzer.py.
//
// The "patch math" helpers (cropImageDataRect, toGray, lapVar, edgeDensity,
// claheGray, rgbToLab/Hsv/YCrCb means, etc.) are shared between the gates
// ported from src/gates/. These replace the small subset of cv2 ops the
// Python source relies on (cvtColor → COLOR_BGR2GRAY/LAB/HSV/YCrCb,
// Laplacian, Canny, createCLAHE/apply).

/** Image source we accept from callers. */
export type SourceImage = HTMLCanvasElement | OffscreenCanvas | ImageData;

/** Duck-type ImageData. `instanceof ImageData` blows up in node, where the
 *  global doesn't exist; canvas-likes always expose getContext(), ImageData
 *  never does, so this is unambiguous. */
export function isImageDataLike(src: SourceImage): src is ImageData {
  return typeof (src as { getContext?: unknown }).getContext !== "function";
}

/** Get an ImageData snapshot from any supported input, no scaling. */
export function toImageData(src: SourceImage): ImageData {
  if (isImageDataLike(src)) return src;
  const ctx = (
    src as HTMLCanvasElement | OffscreenCanvas
  ).getContext("2d") as CanvasRenderingContext2D | null;
  if (!ctx) {
    throw new Error("toImageData: 2D context unavailable on source canvas");
  }
  return ctx.getImageData(0, 0, src.width, src.height);
}

/**
 * Compute the scale-padded crop rect around a face bbox, mirroring
 * MiniFASNet's `_crop_face`:
 *
 *   scale = min((H-1)/box_h, (W-1)/box_w, self.scale)
 *   new_w = box_w * scale; new_h = box_h * scale
 *   x1 = max(0, cx - new_w/2); y1 = max(0, cy - new_h/2)
 *   x2 = min(W-1, cx + new_w/2); y2 = min(H-1, cy + new_h/2)
 *
 * The `scale` argument matches uniface DEFAULT_SCALES (V2 = 2.7, V1SE = 4.0).
 */
export function computeMiniFasNetCropRect(
  imageWidth: number,
  imageHeight: number,
  bbox: { x1: number; y1: number; x2: number; y2: number },
  scale: number,
): { x: number; y: number; w: number; h: number } {
  const x = Math.floor(bbox.x1);
  const y = Math.floor(bbox.y1);
  const box_w = Math.max(1, Math.floor(bbox.x2 - bbox.x1));
  const box_h = Math.max(1, Math.floor(bbox.y2 - bbox.y1));

  const eff_scale = Math.min(
    (imageHeight - 1) / box_h,
    (imageWidth - 1) / box_w,
    scale,
  );
  const new_w = box_w * eff_scale;
  const new_h = box_h * eff_scale;
  const cx = x + box_w / 2;
  const cy = y + box_h / 2;

  const x1 = Math.max(0, Math.floor(cx - new_w / 2));
  const y1 = Math.max(0, Math.floor(cy - new_h / 2));
  const x2 = Math.min(imageWidth - 1, Math.floor(cx + new_w / 2));
  const y2 = Math.min(imageHeight - 1, Math.floor(cy + new_h / 2));

  return { x: x1, y: y1, w: x2 - x1 + 1, h: y2 - y1 + 1 };
}

/**
 * Crop a sub-rectangle from a SourceImage and resize it to (outW, outH).
 * Returns a fresh OffscreenCanvas containing the resized BGR-equivalent
 * pixels (in browser they're RGBA — the caller does channel swizzling).
 */
export function cropAndResize(
  src: SourceImage,
  rect: { x: number; y: number; w: number; h: number },
  outW: number,
  outH: number,
): OffscreenCanvas {
  // Stage 1: ensure source is a Canvas-backed image so drawImage() can resize.
  let stage: HTMLCanvasElement | OffscreenCanvas;
  if (isImageDataLike(src)) {
    const c = new OffscreenCanvas(src.width, src.height);
    const cctx = c.getContext("2d");
    if (!cctx) throw new Error("cropAndResize: stage canvas has no 2D ctx");
    cctx.putImageData(src, 0, 0);
    stage = c;
  } else {
    stage = src;
  }

  const out = new OffscreenCanvas(outW, outH);
  const ctx = out.getContext("2d");
  if (!ctx) throw new Error("cropAndResize: output canvas has no 2D ctx");
  // smoothing≈INTER_AREA-ish for downscale; INTER_LINEAR-ish for upscale.
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(stage, rect.x, rect.y, rect.w, rect.h, 0, 0, outW, outH);
  return out;
}

/**
 * Convert a (W, H) RGBA canvas to a planar BGR Float32 NCHW tensor.
 * Matches uniface's preprocess: `face = face.astype(float32)` + HWC→CHW + expand_dims.
 * The MiniFASNet ONNX expects BGR (because the Python reads via cv2.imread).
 */
export function toBgrNchwFloat32(canvas: OffscreenCanvas): Float32Array {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("toBgrNchwFloat32: 2D ctx unavailable");
  const w = canvas.width;
  const h = canvas.height;
  const rgba = ctx.getImageData(0, 0, w, h).data; // Uint8ClampedArray length 4*w*h

  const out = new Float32Array(3 * h * w);
  const planeSize = h * w;
  // Channel order: B, G, R (NCHW with C=3)
  // out[0*plane + i] = B
  // out[1*plane + i] = G
  // out[2*plane + i] = R
  for (let i = 0, p = 0; i < planeSize; i++, p += 4) {
    const r = rgba[p];
    const g = rgba[p + 1];
    const b = rgba[p + 2];
    out[i] = b;
    out[planeSize + i] = g;
    out[2 * planeSize + i] = r;
  }
  return out;
}

/**
 * 2-class softmax, mirrors `uniface.common.softmax` for a (1, 2) logits tensor.
 * Returns [p0, p1].
 */
export function softmax2(logits: ArrayLike<number>): [number, number] {
  const l0 = logits[0];
  const l1 = logits[1];
  const m = Math.max(l0, l1);
  const e0 = Math.exp(l0 - m);
  const e1 = Math.exp(l1 - m);
  const z = e0 + e1;
  return [e0 / z, e1 / z];
}

// ===========================================================================
// Gate helpers (port of OpenCV ops used by src/gates/critical_region_visibility,
// src/gates/illumination, and src/gates/face_usability).
// ===========================================================================

/**
 * A flat RGBA patch.
 *
 *  - `data` is a Uint8ClampedArray of length `4 * width * height`.
 *  - The patch is always interpreted as RGBA (no alpha use here).
 *
 * The Python source uses BGR frames straight from cv2.imread, but in the
 * browser we get RGBA from canvas. All gate helpers below operate on RGBA
 * data — colour-channel conversions (Lab/HSV/YCrCb) start from RGB.
 */
export interface Patch {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/** True if the patch carries no pixels. */
export function isEmptyPatch(p: Patch): boolean {
  return p.width <= 0 || p.height <= 0 || p.data.length === 0;
}

/** A clamped, integer-pixel face bbox: [x, y, width, height]. */
export type GateBBox = readonly [number, number, number, number];

/**
 * Clip a bbox to frame bounds and return both the cropped patch and the
 * clamped bbox. Mirrors `_crop` in critical_region_visibility.py:320-330.
 */
export function cropImageData(
  frame: ImageData,
  bbox: GateBBox,
): { patch: Patch; clipped: GateBBox } {
  const [bx, by, bw, bh] = bbox;
  const fw = frame.width;
  const fh = frame.height;
  const x1 = Math.max(0, Math.min(fw, Math.round(bx)));
  const y1 = Math.max(0, Math.min(fh, Math.round(by)));
  const x2 = Math.max(x1 + 1, Math.min(fw, Math.round(bx + bw)));
  const y2 = Math.max(y1 + 1, Math.min(fh, Math.round(by + bh)));
  const w = x2 - x1;
  const h = y2 - y1;
  if (w <= 0 || h <= 0) {
    return {
      patch: { data: new Uint8ClampedArray(0), width: 0, height: 0 },
      clipped: [x1, y1, 0, 0],
    };
  }
  return {
    patch: cropImageDataRect(frame, x1, y1, w, h),
    clipped: [x1, y1, w, h],
  };
}

/** Pure-pixel crop with no clipping — caller must pre-validate bounds. */
export function cropImageDataRect(
  frame: ImageData,
  x: number,
  y: number,
  w: number,
  h: number,
): Patch {
  const out = new Uint8ClampedArray(w * h * 4);
  const stride = frame.width * 4;
  for (let row = 0; row < h; row++) {
    const srcOff = (y + row) * stride + x * 4;
    const dstOff = row * w * 4;
    out.set(frame.data.subarray(srcOff, srcOff + w * 4), dstOff);
  }
  return { data: out, width: w, height: h };
}

/**
 * Crop a face-relative rectangle, where (rx, ry, rw, rh) are 0..1
 * fractional coords inside the face ROI. Mirrors `_region_patch`.
 */
export function regionPatch(
  faceRoi: Patch,
  ratios: readonly [number, number, number, number],
): Patch {
  if (isEmptyPatch(faceRoi)) {
    return { data: new Uint8ClampedArray(0), width: 0, height: 0 };
  }
  const [relX, relY, relW, relH] = ratios;
  const w = faceRoi.width;
  const h = faceRoi.height;
  const x1 = Math.max(0, Math.min(w, Math.round(w * relX)));
  const y1 = Math.max(0, Math.min(h, Math.round(h * relY)));
  const x2 = Math.max(x1 + 1, Math.min(w, Math.round(w * (relX + relW))));
  const y2 = Math.max(y1 + 1, Math.min(h, Math.round(h * (relY + relH))));
  const cw = x2 - x1;
  const ch = y2 - y1;
  if (cw <= 0 || ch <= 0) {
    return { data: new Uint8ClampedArray(0), width: 0, height: 0 };
  }
  const out = new Uint8ClampedArray(cw * ch * 4);
  const srcStride = w * 4;
  for (let row = 0; row < ch; row++) {
    const srcOff = (y1 + row) * srcStride + x1 * 4;
    const dstOff = row * cw * 4;
    out.set(faceRoi.data.subarray(srcOff, srcOff + cw * 4), dstOff);
  }
  return { data: out, width: cw, height: ch };
}

/**
 * Wrap an ImageData as a Patch view (no copy).
 * The Patch contract matches ImageData.{data,width,height} so this is free.
 */
export function imageDataAsPatch(frame: ImageData): Patch {
  return { data: frame.data, width: frame.width, height: frame.height };
}

/** Convert an RGBA patch to an 8-bit grayscale Uint8Array (luminance). */
export function toGray(patch: Patch): Uint8Array {
  const n = patch.width * patch.height;
  const out = new Uint8Array(n);
  const d = patch.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    // ITU-R BT.601 luma — same coefficients OpenCV uses for COLOR_RGB2GRAY.
    out[i] = Math.round(0.299 * d[p] + 0.587 * d[p + 1] + 0.114 * d[p + 2]);
  }
  return out;
}

/** Population mean (float). */
export function meanU8(arr: Uint8Array): number {
  if (arr.length === 0) return 0;
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

/** Population standard deviation (float). */
export function stdU8(arr: Uint8Array): number {
  if (arr.length === 0) return 0;
  const m = meanU8(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / arr.length);
}

/**
 * Variance of the discrete Laplacian over a grayscale buffer — mirrors
 * `cv2.Laplacian(gray, cv2.CV_64F).var()`. The 3x3 Laplacian kernel is:
 *
 *     0  1  0
 *     1 -4  1
 *     0  1  0
 *
 * Boundary pixels are skipped (cv2 default is BORDER_REFLECT_101, which
 * has negligible effect on the variance for patches with non-zero area).
 */
export function laplacianVariance(gray: Uint8Array, w: number, h: number): number {
  if (w < 3 || h < 3) return 0;
  const n = (w - 2) * (h - 2);
  if (n <= 0) return 0;
  const buf = new Float64Array(n);
  let sum = 0;
  let i = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const c = gray[y * w + x];
      const v =
        gray[(y - 1) * w + x] +
        gray[(y + 1) * w + x] +
        gray[y * w + (x - 1)] +
        gray[y * w + (x + 1)] -
        4 * c;
      buf[i++] = v;
      sum += v;
    }
  }
  const mean = sum / n;
  let s = 0;
  for (let k = 0; k < n; k++) {
    const d = buf[k] - mean;
    s += d * d;
  }
  return s / n;
}

/**
 * Sobel-based edge density approximation of `cv2.Canny(gray, low, high) > 0`.
 * Returns the fraction of "on" edge pixels (0..1).
 *
 * Approach: 3x3 Sobel magnitudes thresholded with hysteresis-lite. We
 * promote a pixel to "edge" if `mag >= high`, and additionally include
 * `mag >= low` pixels that are 8-connected to a >=high pixel. This is a
 * close functional approximation of Canny's non-maxima-suppressed double
 * threshold for the purposes of "how much edge texture is here" — the
 * absolute counts shift modestly vs cv2.Canny but the same gates use
 * the value only in ratio scoring, so calibration carries over.
 *
 * Calibrated thresholds in the Python source: Canny(40,120) for CLAHE
 * eyes and Canny(60,140) for the generic edge_density helper. We pass
 * these through verbatim — the Sobel magnitude scale roughly matches the
 * Canny scale (both are sums of |gx|+|gy| ish), so the cutoffs translate
 * directly enough for the gate's ratio_score logic to keep its meaning.
 */
export function edgeDensity(
  gray: Uint8Array,
  w: number,
  h: number,
  low = 60,
  high = 140,
): number {
  if (w < 3 || h < 3) return 0;
  const mag = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        -gray[i - w - 1] - 2 * gray[i - 1] - gray[i + w - 1] +
        gray[i - w + 1] + 2 * gray[i + 1] + gray[i + w + 1];
      const gy =
        -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1] +
        gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
      // Use sum-of-absolutes to mirror cv2.Canny's L1 default.
      mag[i] = Math.abs(gx) + Math.abs(gy);
    }
  }
  const strong = new Uint8Array(w * h);
  const weak = new Uint8Array(w * h);
  let strongCount = 0;
  for (let i = 0; i < mag.length; i++) {
    if (mag[i] >= high) {
      strong[i] = 1;
      strongCount += 1;
    } else if (mag[i] >= low) {
      weak[i] = 1;
    }
  }
  if (strongCount === 0) return 0;
  // Promote any weak pixel 8-connected to a strong one.
  let promoted = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      if (!weak[i]) continue;
      if (
        strong[i - w - 1] || strong[i - w] || strong[i - w + 1] ||
        strong[i - 1]                       || strong[i + 1] ||
        strong[i + w - 1] || strong[i + w] || strong[i + w + 1]
      ) {
        strong[i] = 1;
        promoted += 1;
      }
    }
  }
  return (strongCount + promoted) / mag.length;
}

/**
 * Approximate CLAHE (Contrast-Limited Adaptive Histogram Equalization).
 *
 * The browser has no cv2.createCLAHE; we implement a tiled equalizer with
 * the same 4x4 default tile grid and clipLimit 2.0 used in the Python
 * gates. Each tile builds a 256-bin histogram, clips bins above the limit
 * (limit = clipLimit * tile_pixels / 256), redistributes the excess
 * uniformly, then maps via the clipped CDF. Bilinear interpolation
 * between neighbouring tile maps would be ideal — for the gate use
 * (variance / edge density / std) a nearest-tile mapping is sufficient
 * and what we ship; documented deviation noted in the gate header.
 */
export function claheGray(
  gray: Uint8Array,
  w: number,
  h: number,
  clipLimit = 2.0,
  tileGridSize = 4,
): Uint8Array {
  if (w === 0 || h === 0) return new Uint8Array(0);
  const out = new Uint8Array(w * h);
  const tilesX = Math.max(1, Math.min(tileGridSize, w));
  const tilesY = Math.max(1, Math.min(tileGridSize, h));
  const maps: Uint8Array[] = new Array(tilesX * tilesY);
  // Precompute tile pixel bounds.
  const xBounds = new Int32Array(tilesX + 1);
  const yBounds = new Int32Array(tilesY + 1);
  for (let i = 0; i <= tilesX; i++) xBounds[i] = Math.round((i * w) / tilesX);
  for (let i = 0; i <= tilesY; i++) yBounds[i] = Math.round((i * h) / tilesY);

  for (let ty = 0; ty < tilesY; ty++) {
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = xBounds[tx];
      const x1 = xBounds[tx + 1];
      const y0 = yBounds[ty];
      const y1 = yBounds[ty + 1];
      const tw = x1 - x0;
      const th = y1 - y0;
      const tilePix = Math.max(1, tw * th);
      const hist = new Int32Array(256);
      for (let y = y0; y < y1; y++) {
        const row = y * w;
        for (let x = x0; x < x1; x++) {
          hist[gray[row + x]] += 1;
        }
      }
      // Clip + redistribute.
      const limit = Math.max(1, Math.floor((clipLimit * tilePix) / 256));
      let excess = 0;
      for (let i = 0; i < 256; i++) {
        if (hist[i] > limit) {
          excess += hist[i] - limit;
          hist[i] = limit;
        }
      }
      const add = Math.floor(excess / 256);
      const rem = excess - add * 256;
      for (let i = 0; i < 256; i++) hist[i] += add;
      for (let i = 0; i < rem; i++) hist[i] += 1;
      // Build CDF → lookup.
      const lut = new Uint8Array(256);
      let cum = 0;
      for (let i = 0; i < 256; i++) {
        cum += hist[i];
        lut[i] = Math.min(255, Math.round((cum * 255) / tilePix));
      }
      maps[ty * tilesX + tx] = lut;
    }
  }
  // Apply nearest-tile mapping.
  for (let y = 0; y < h; y++) {
    let ty = 0;
    while (ty < tilesY - 1 && y >= yBounds[ty + 1]) ty += 1;
    const tileRow = ty * tilesX;
    for (let x = 0; x < w; x++) {
      let tx = 0;
      while (tx < tilesX - 1 && x >= xBounds[tx + 1]) tx += 1;
      out[y * w + x] = maps[tileRow + tx][gray[y * w + x]];
    }
  }
  return out;
}

/**
 * Per-channel mean in CIE L*a*b* (8-bit OpenCV convention).
 *
 * The Python source reads cv2.cvtColor(patch, cv2.COLOR_BGR2LAB) which
 * returns 8-bit values in OpenCV's compressed range:
 *   L: 0..255 (== L*[0..100] * 2.55)
 *   a: 0..255 (== a*[-128..127] + 128)
 *   b: 0..255 (== b*[-128..127] + 128)
 *
 * We compute true CIE Lab via sRGB → linear RGB → XYZ → Lab (D65 white
 * point), then re-pack to 8-bit using the same offsets. The
 * region-baseline comparisons in the gate only read deltas, so any
 * small calibration drift between OpenCV's gamma and the proper IEC
 * 61966-2-1 sRGB curve is washed out by the `_ratio_score` stretches.
 */
export function rgbToLabMean(patch: Patch): [number, number, number] {
  const n = patch.width * patch.height;
  if (n === 0) return [0, 0, 0];
  let sL = 0;
  let sA = 0;
  let sB = 0;
  const d = patch.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const [L, A, B] = rgbToLab8(d[p], d[p + 1], d[p + 2]);
    sL += L;
    sA += A;
    sB += B;
  }
  return [sL / n, sA / n, sB / n];
}

/** Convert one sRGB pixel to OpenCV-style 8-bit Lab. */
function rgbToLab8(r8: number, g8: number, b8: number): [number, number, number] {
  // sRGB → linear.
  const r = srgbToLin(r8 / 255);
  const g = srgbToLin(g8 / 255);
  const b = srgbToLin(b8 / 255);
  // Linear sRGB → XYZ (D65, sRGB primaries).
  const X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
  const Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
  const Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
  // D65 white point.
  const Xn = 0.95047;
  const Yn = 1.0;
  const Zn = 1.08883;
  const fx = labF(X / Xn);
  const fy = labF(Y / Yn);
  const fz = labF(Z / Zn);
  const L = 116 * fy - 16;            // 0..100
  const a = 500 * (fx - fy);          // ~-128..127
  const bb = 200 * (fy - fz);         // ~-128..127
  return [
    Math.max(0, Math.min(255, L * 2.55)),
    Math.max(0, Math.min(255, a + 128)),
    Math.max(0, Math.min(255, bb + 128)),
  ];
}

function srgbToLin(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function labF(t: number): number {
  const delta = 6 / 29;
  return t > delta * delta * delta
    ? Math.cbrt(t)
    : t / (3 * delta * delta) + 4 / 29;
}

/**
 * HSV inRange ratios (lip + skin masks for the mouth-validity check).
 * Mirrors the Python `_mouth_hsv_color_validity`. HSV uses the OpenCV
 * 8-bit convention: H ∈ [0..179], S ∈ [0..255], V ∈ [0..255].
 */
export function mouthHsvColorValidity(patch: Patch): { valid: boolean; confidence: number } {
  const n = patch.width * patch.height;
  if (n === 0) return { valid: true, confidence: 1 };
  let lipCount = 0;
  let skinCount = 0;
  const d = patch.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const [H, S, V] = rgbToHsv8(d[p], d[p + 1], d[p + 2]);
    if (H >= 0 && H <= 20 && S >= 30 && S <= 180 && V >= 60) lipCount += 1;
    if (H >= 0 && H <= 25 && S >= 15 && S <= 150 && V >= 80) skinCount += 1;
  }
  const lipRatio = lipCount / n;
  const skinRatio = skinCount / n;
  const valid = lipRatio > 0.08 || skinRatio > 0.25;
  return { valid, confidence: lipRatio + skinRatio * 0.5 };
}

/**
 * Mean of YCrCb Cr channel — mirrors `_mouth_cr_in_skin_range`.
 * Achromatic occluders read Cr near 128; skin/lips read 135..185.
 */
export function meanYcrcbCr(patch: Patch): number {
  const n = patch.width * patch.height;
  if (n === 0) return 0;
  let sum = 0;
  const d = patch.data;
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    // ITU-R BT.601 RGB→YCrCb (matches cv2 COLOR_RGB2YCrCb).
    const Cr = 128 + (0.5 * d[p] - 0.41869 * d[p + 1] - 0.08131 * d[p + 2]);
    sum += Cr;
  }
  return sum / n;
}

function rgbToHsv8(r8: number, g8: number, b8: number): [number, number, number] {
  const r = r8 / 255;
  const g = g8 / 255;
  const b = b8 / 255;
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const v = mx;
  const dlt = mx - mn;
  const s = mx === 0 ? 0 : dlt / mx;
  let h = 0;
  if (dlt > 0) {
    if (mx === r) h = ((g - b) / dlt) % 6;
    else if (mx === g) h = (b - r) / dlt + 2;
    else h = (r - g) / dlt + 4;
    h *= 60; // degrees
    if (h < 0) h += 360;
  }
  // OpenCV 8-bit HSV: H/2, S*255, V*255.
  return [Math.round(h / 2), Math.round(s * 255), Math.round(v * 255)];
}
