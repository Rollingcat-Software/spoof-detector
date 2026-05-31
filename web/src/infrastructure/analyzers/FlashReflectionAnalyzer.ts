// FlashReflectionAnalyzer — active-illumination liveness (OPT-IN).
//
// Ported from the FIVUCSAS research tree
//   research/aysenur/.../light_challenge_service.py
//   research/aysenur/.../flash_spoof_analyzer.py
//
// This is the strongest anti-SCREEN signal and it is camera-focus-independent.
// It is INTENTIONALLY NOT part of the passive per-frame fusion: the proctoring
// profile must stay fully passive (no challenges that interrupt the user), so
// this runs only when a caller explicitly invokes a flash challenge (e.g. an
// amispoof "verify with light" button).
//
// Protocol (driven by the caller / UI):
//   1. capture a BASELINE face-crop frame (screen at rest)
//   2. flash the screen a known color (red / green / blue / white) ~150 ms
//   3. capture a FLASH face-crop frame during the flash
//   4. call scoreResponse(baseline, flash, color)
//
// Physics:
//   * A real 3D face REFLECTS the flash diffusely — the emitted colour's
//     channel rises across the skin, the rise is strongest in the matching
//     channel (chroma/dominance gain), and its magnitude VARIES by region
//     because the face has 3D relief (forehead vs cheeks vs nose).
//   * A screen EMITS its own light — it barely responds to our flash, or
//     responds uniformly/specularly with no 3D region spread.
//
// Scoring mirrors the reference:
//   color_shift = 0.45*absolute_gain + 0.40*chroma_gain + 0.15*dominance_gain
// blended with a region-spread term that rewards 3D diffuse variation.
//
// NOTE: the absolute thresholds below are the reference defaults and WILL need
// live calibration per camera / ambient light (same as the planarity veto was
// tuned live on 2026-05-24). The directional behaviour is unit-tested.

export type FlashColor = "red" | "green" | "blue" | "white";

export interface FlashReflectionResult {
  /** 0-100, higher = more live (genuine diffuse reflection). */
  score: number;
  /** score >= liveThreshold AND not inconclusive. */
  isLive: boolean;
  /**
   * True when the flash produced no measurable face-brightness change — i.e.
   * the screen's light never reached the face (a desktop monitor in normal
   * ambient). We CANNOT judge live/spoof in this case, so the caller must NOT
   * treat it as SPOOF. Only `isLive=false && !inconclusive` is a real spoof.
   */
  inconclusive: boolean;
  color: FlashColor;
  /** Mean target-channel gain across regions, normalised 0-1. */
  targetGain: number;
  /** Mean non-target-channel gain across regions, normalised 0-1. */
  otherGain: number;
  /** Blended reference metric (absolute + chroma + dominance). */
  colorShift: number;
  /** Std of per-region target gain — 3D faces vary, flat screens are uniform. */
  regionSpread: number;
}

export interface FlashReflectionOptions {
  /** color_shift at/above which a single channel response is "real". Default 0.05. */
  colorShiftThreshold?: number;
  /** color_shift mapped to score 100 (saturation). Default 0.15. */
  colorShiftSaturation?: number;
  /** Region-spread mapped to its full bonus. Default 0.03 (normalised gain std). */
  regionSpreadSaturation?: number;
  /** Final score at/above which the response is judged live. Default 50. */
  liveThreshold?: number;
  /**
   * Minimum total photometric response (target+other gain, 0-1) below which
   * the flash is judged to have NOT reached the face → inconclusive. Default
   * 0.012 (~3/255 brightness). Prevents a false SPOOF when the screen's light
   * can't illuminate the face (desktop / bright room).
   */
  inconclusiveFloor?: number;
}

type Region = { x0: number; y0: number; x1: number; y1: number };

// Face-crop sub-regions as fractions of the crop (x: cols, y: rows).
const REGIONS: Readonly<Record<string, Region>> = {
  forehead: { x0: 0.25, y0: 0.05, x1: 0.75, y1: 0.30 },
  leftCheek: { x0: 0.12, y0: 0.45, x1: 0.38, y1: 0.78 },
  rightCheek: { x0: 0.62, y0: 0.45, x1: 0.88, y1: 0.78 },
  nose: { x0: 0.40, y0: 0.35, x1: 0.60, y1: 0.65 },
};

export class FlashReflectionAnalyzer {
  readonly name = "flash_reflection";

  private readonly colorShiftThreshold: number;
  private readonly colorShiftSaturation: number;
  private readonly regionSpreadSaturation: number;
  private readonly liveThreshold: number;
  private readonly inconclusiveFloor: number;

  constructor(options: FlashReflectionOptions = {}) {
    this.colorShiftThreshold = options.colorShiftThreshold ?? 0.05;
    this.colorShiftSaturation = options.colorShiftSaturation ?? 0.15;
    this.regionSpreadSaturation = options.regionSpreadSaturation ?? 0.03;
    this.liveThreshold = options.liveThreshold ?? 50;
    this.inconclusiveFloor = options.inconclusiveFloor ?? 0.012;
  }

  /**
   * Score the face's photometric response to a screen flash. `baseline` and
   * `flash` are face-crop ImageData of the same dimensions; `color` is the
   * flashed colour.
   */
  scoreResponse(
    baseline: ImageData,
    flash: ImageData,
    color: FlashColor,
  ): FlashReflectionResult {
    const targetIdx = color === "red" ? 0 : color === "green" ? 1 : color === "blue" ? 2 : -1; // -1 = white

    const perRegionTargetGain: number[] = [];
    let sumAbsGain = 0;
    let sumChromaGain = 0;
    let sumDominanceGain = 0;
    let sumTargetGain = 0;
    let sumOtherGain = 0;
    let n = 0;

    for (const key of Object.keys(REGIONS)) {
      const r = REGIONS[key];
      const base = regionMeanRgb(baseline, r);
      const flsh = regionMeanRgb(flash, r);
      if (!base || !flsh) continue;

      const dR = (flsh[0] - base[0]) / 255;
      const dG = (flsh[1] - base[1]) / 255;
      const dB = (flsh[2] - base[2]) / 255;
      const deltas = [dR, dG, dB];

      let targetDelta: number;
      let otherDelta: number;
      if (targetIdx >= 0) {
        targetDelta = deltas[targetIdx];
        otherDelta = (deltas[0] + deltas[1] + deltas[2] - targetDelta) / 2;
      } else {
        // white flash → brightness response across all channels
        targetDelta = (dR + dG + dB) / 3;
        otherDelta = 0;
      }

      const absoluteGain = Math.max(0, targetDelta - otherDelta);

      // Chroma gain: rise in target-channel share of total brightness.
      const baseSum = base[0] + base[1] + base[2] + 1e-6;
      const flshSum = flsh[0] + flsh[1] + flsh[2] + 1e-6;
      const baseShare =
        targetIdx >= 0 ? base[targetIdx] / baseSum : 1 / 3;
      const flshShare =
        targetIdx >= 0 ? flsh[targetIdx] / flshSum : 1 / 3;
      const chromaGain = clamp01((flshShare - baseShare) / 0.08);

      const dominanceGain =
        targetIdx >= 0 ? clamp01((flshShare - 0.45) / 0.35) : clamp01(targetDelta / 0.2);

      sumAbsGain += absoluteGain;
      sumChromaGain += chromaGain;
      sumDominanceGain += dominanceGain;
      sumTargetGain += Math.max(0, targetDelta);
      sumOtherGain += Math.max(0, otherDelta);
      perRegionTargetGain.push(Math.max(0, targetDelta));
      n += 1;
    }

    if (n === 0) {
      return {
        score: 0,
        isLive: false,
        inconclusive: true,
        color,
        targetGain: 0,
        otherGain: 0,
        colorShift: 0,
        regionSpread: 0,
      };
    }

    const colorShift =
      0.45 * (sumAbsGain / n) +
      0.40 * (sumChromaGain / n) +
      0.15 * (sumDominanceGain / n);
    const targetGain = sumTargetGain / n;
    const otherGain = sumOtherGain / n;
    const regionSpread = stddev(perRegionTargetGain);

    // Map to 0-100: the channel response carries most of the score; the 3D
    // region-spread term adds a diffuse-reflection bonus that a flat, uniform
    // screen response can't earn.
    const responsePart = clamp01(
      (colorShift - this.colorShiftThreshold) /
        Math.max(1e-6, this.colorShiftSaturation - this.colorShiftThreshold),
    );
    const spreadPart = clamp01(regionSpread / this.regionSpreadSaturation);
    const score = Math.round((0.75 * responsePart + 0.25 * spreadPart) * 100);

    // Did the flash actually register on the face? If the total photometric
    // response is below the floor, the screen's light didn't reach the face
    // (desktop / bright ambient) — we can't judge, so report inconclusive
    // rather than a false SPOOF.
    const flashMagnitude = targetGain + otherGain;
    const inconclusive = flashMagnitude < this.inconclusiveFloor;

    return {
      score,
      isLive: !inconclusive && score >= this.liveThreshold,
      inconclusive,
      color,
      targetGain: round(targetGain, 4),
      otherGain: round(otherGain, 4),
      colorShift: round(colorShift, 4),
      regionSpread: round(regionSpread, 4),
    };
  }
}

/** Mean [r,g,b] over a fractional sub-region of an ImageData, or null if empty. */
function regionMeanRgb(
  img: ImageData,
  r: Region,
): number[] | null {
  const x0 = Math.max(0, Math.floor(r.x0 * img.width));
  const y0 = Math.max(0, Math.floor(r.y0 * img.height));
  const x1 = Math.min(img.width, Math.ceil(r.x1 * img.width));
  const y1 = Math.min(img.height, Math.ceil(r.y1 * img.height));
  const w = x1 - x0;
  const h = y1 - y0;
  if (w <= 0 || h <= 0) return null;
  const stride = img.width * 4;
  let sr = 0, sg = 0, sb = 0, count = 0;
  for (let y = y0; y < y1; y++) {
    let off = y * stride + x0 * 4;
    for (let x = 0; x < w; x++, off += 4) {
      sr += img.data[off];
      sg += img.data[off + 1];
      sb += img.data[off + 2];
      count += 1;
    }
  }
  if (count === 0) return null;
  return [sr / count, sg / count, sb / count];
}

function stddev(xs: number[]): number {
  const n = xs.length;
  if (n === 0) return 0;
  let mean = 0;
  for (const x of xs) mean += x;
  mean /= n;
  let sse = 0;
  for (const x of xs) {
    const d = x - mean;
    sse += d * d;
  }
  return Math.sqrt(sse / n);
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
