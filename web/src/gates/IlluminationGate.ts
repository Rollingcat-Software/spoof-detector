// Port of src/gates/illumination.py — algorithms by @Aysenur15.
//
// Preview-only face illumination quality gate.
//
// Classifies poor illumination (under/over-exposure, shadow asymmetry,
// uneven per-region brightness) separately from physical occlusion. The
// `FaceUsabilityGate` orchestrator uses the result to decide whether to
// drop the frame on quality grounds without flagging the user for spoof.
//
// Calibrated thresholds are preserved verbatim from the Python source so
// the paper-cited 0.58 quality threshold / 0.40-0.55 shadow asymmetry
// band remain meaningful in the browser bundle.
//
// Deviations from the Python (documented):
//   * cv2.createCLAHE (clipLimit=2.0, tileGridSize=(4,4)) → claheGray()
//     tiled-equalizer approximation in imageOps.ts (nearest-tile, no
//     bilinear interpolation between tile maps). Sub-pixel-level
//     calibration drift; gate uses CLAHE only for ratio scores so it is
//     absorbed.
//   * cv2.Laplacian + .var() → laplacianVariance() over a Float64 buffer.
//   * cv2.Canny(40, 120) → edgeDensity(low=40, high=120) Sobel-based
//     approximation with hysteresis-lite.
//   * cv2.cvtColor(BGR2GRAY) → toGray() with ITU-R BT.601 weights (same
//     coefficients cv2 uses for RGB2GRAY; we hand RGBA, not BGR).

import {
  GateBBox,
  Patch,
  claheGray,
  cropImageData,
  edgeDensity,
  imageDataAsPatch,
  isEmptyPatch,
  laplacianVariance,
  meanU8,
  regionPatch,
  stdU8,
  toGray,
} from "../utils/imageOps";

const QUALITY_REGIONS: Record<string, readonly [number, number, number, number]> = {
  left_eye:   [0.14, 0.24, 0.24, 0.18],
  right_eye:  [0.62, 0.24, 0.24, 0.18],
  nose:       [0.39, 0.35, 0.22, 0.24],
  mouth:      [0.28, 0.63, 0.44, 0.16],
  lower_face: [0.18, 0.56, 0.64, 0.34],
};

const UNDEREXPOSED_BRIGHTNESS = 52.0;
const OVEREXPOSED_BRIGHTNESS = 218.0;
const GLOBAL_BRIGHTNESS_LOW = 62.0;
const GLOBAL_BRIGHTNESS_HIGH = 210.0;
const QUALITY_SCORE_THRESHOLD = 0.58;

export type QualityStatus = "OK" | "LOW_QUALITY" | "-";

/** Result returned by `FaceQualityIlluminationGate.evaluate`. */
export interface IlluminationResult {
  qualityOk: boolean;
  qualityStatus: QualityStatus;
  qualityReason: string;
  perRegionBrightness: Record<string, number>;
  brightnessUniformity: number;
  illuminationScore: number;
  globalFaceBrightness: number;
  shadowAsymmetry: number;
  underexposedRegions: readonly string[];
  overexposedRegions: readonly string[];
}

const EMPTY_RESULT: IlluminationResult = Object.freeze({
  qualityOk: false,
  qualityStatus: "LOW_QUALITY",
  qualityReason: "poor_face_illumination",
  perRegionBrightness: {},
  brightnessUniformity: 0,
  illuminationScore: 0,
  globalFaceBrightness: 0,
  shadowAsymmetry: 0,
  underexposedRegions: [],
  overexposedRegions: [],
});

export class IlluminationGate {
  /**
   * Evaluate a single frame.
   *
   * @param frame full-frame RGBA ImageData (camera frame).
   * @param faceBbox [x, y, width, height] face rectangle in frame coords.
   *   `null` short-circuits to a LOW_QUALITY empty result, mirroring the
   *   Python "no frame / no bbox" branch.
   */
  evaluate(
    frame: ImageData | null,
    faceBbox: GateBBox | null,
  ): IlluminationResult {
    if (!frame || frame.data.length === 0 || !faceBbox) {
      return { ...EMPTY_RESULT };
    }

    const { patch: faceRoi } = cropImageData(frame, faceBbox);
    if (isEmptyPatch(faceRoi)) {
      return { ...EMPTY_RESULT };
    }

    const faceGray = toGray(faceRoi);
    const globalFaceBrightness = meanU8(faceGray);
    const globalContrast = stdU8(faceGray);

    // Left/right halves for shadow asymmetry.
    const half = Math.max(1, Math.floor(faceRoi.width / 2));
    const left = subPatchCols(faceRoi, 0, half);
    const right = subPatchCols(faceRoi, half, faceRoi.width);
    const leftBrightness = meanU8(toGray(left));
    const rightBrightness = meanU8(toGray(right));
    const shadowAsymmetry =
      Math.abs(leftBrightness - rightBrightness) / Math.max(globalFaceBrightness, 1.0);

    const perRegionBrightness: Record<string, number> = {};
    const normalizedDetailScores: Record<string, number> = {};
    const underexposedRegions: string[] = [];
    const overexposedRegions: string[] = [];

    for (const [regionName, ratios] of Object.entries(QUALITY_REGIONS)) {
      const patch = regionPatch(faceRoi, ratios);
      if (isEmptyPatch(patch)) continue;
      const patchGray = toGray(patch);
      const brightness = meanU8(patchGray);
      perRegionBrightness[regionName] = brightness;
      normalizedDetailScores[regionName] = normalizedDetailScore(patchGray, patch.width, patch.height);
      if (brightness < UNDEREXPOSED_BRIGHTNESS) {
        underexposedRegions.push(regionName);
      } else if (brightness > OVEREXPOSED_BRIGHTNESS) {
        overexposedRegions.push(regionName);
      }
    }

    const brightnessValues = Object.values(perRegionBrightness);
    const referenceValues = brightnessValues.length > 0 ? brightnessValues : [globalFaceBrightness];
    const brightnessStd = stdOfNumbers(referenceValues);
    const brightnessUniformity = clamp01(1.0 - brightnessStd / 70.0);
    const regionCount = Math.max(Object.keys(perRegionBrightness).length, 1);
    const underexposedRatio = underexposedRegions.length / regionCount;
    const overexposedRatio = overexposedRegions.length / regionCount;
    const detailValues = Object.values(normalizedDetailScores);
    const normalizedDetailMean = detailValues.length > 0 ? mean(detailValues) : 0;

    const brightnessScore = bandScore(
      globalFaceBrightness,
      42.0,
      225.0,
      72.0,
      188.0,
    );
    const contrastScore = clamp01(globalContrast / 40.0);
    const asymmetryScore = clamp01(1.0 - shadowAsymmetry / 0.4);
    const exposureScore = clamp01(
      1.0 - 0.7 * underexposedRatio - 0.45 * overexposedRatio,
    );
    const illuminationScore = clamp01(
      0.34 * brightnessScore +
        0.2 * brightnessUniformity +
        0.18 * exposureScore +
        0.12 * asymmetryScore +
        0.1 * contrastScore +
        0.06 * normalizedDetailMean,
    );

    let qualityReason = "face_quality_ok";
    let qualityOk = true;
    // Shadow asymmetry tiering (verbatim from Python lines 154-157):
    //   > 0.55                            → hard block
    //   0.40–0.55 + illum_score < 0.75    → soft block
    //   0.40–0.55 + illum_score >= 0.75   → warning-only
    //   < 0.40                            → always pass
    const shadowBlocks =
      shadowAsymmetry > 0.55 ||
      (shadowAsymmetry >= 0.4 && shadowAsymmetry <= 0.55 && illuminationScore < 0.75);

    if (
      globalFaceBrightness < GLOBAL_BRIGHTNESS_LOW ||
      underexposedRatio >= 0.4 ||
      (illuminationScore < QUALITY_SCORE_THRESHOLD &&
        brightnessScore < 0.55 &&
        normalizedDetailMean >= 0.28)
    ) {
      qualityOk = false;
      qualityReason = "poor_face_illumination";
    } else if (
      globalFaceBrightness > GLOBAL_BRIGHTNESS_HIGH ||
      overexposedRatio >= 0.35 ||
      shadowBlocks ||
      brightnessUniformity < 0.44
    ) {
      qualityOk = false;
      qualityReason = "uneven_face_lighting";
    } else if (illuminationScore < QUALITY_SCORE_THRESHOLD) {
      qualityOk = false;
      qualityReason = "poor_face_illumination";
    }

    return {
      qualityOk,
      qualityStatus: qualityOk ? "OK" : "LOW_QUALITY",
      qualityReason,
      perRegionBrightness,
      brightnessUniformity,
      illuminationScore,
      globalFaceBrightness,
      shadowAsymmetry,
      underexposedRegions,
      overexposedRegions,
    };
  }
}

// === helpers ===

function subPatchCols(p: Patch, x0: number, x1: number): Patch {
  const w = Math.max(0, x1 - x0);
  const h = p.height;
  if (w <= 0 || h <= 0) {
    return { data: new Uint8ClampedArray(0), width: 0, height: 0 };
  }
  const out = new Uint8ClampedArray(w * h * 4);
  const srcStride = p.width * 4;
  for (let y = 0; y < h; y++) {
    const srcOff = y * srcStride + x0 * 4;
    const dstOff = y * w * 4;
    out.set(p.data.subarray(srcOff, srcOff + w * 4), dstOff);
  }
  return { data: out, width: w, height: h };
}

function normalizedDetailScore(gray: Uint8Array, w: number, h: number): number {
  const normalized = claheGray(gray, w, h, 2.0, 4);
  const texture = laplacianVariance(normalized, w, h);
  const edges = edgeDensity(normalized, w, h, 40, 120);
  const contrast = stdU8(normalized);
  return clamp01(
    0.45 * Math.min(texture / 140.0, 1.0) +
      0.3 * Math.min(edges / 0.16, 1.0) +
      0.25 * Math.min(contrast / 38.0, 1.0),
  );
}

function bandScore(
  value: number,
  low: number,
  high: number,
  innerLow: number,
  innerHigh: number,
): number {
  if (!(value >= low && value <= high)) return 0;
  if (value >= innerLow && value <= innerHigh) return 1;
  if (value < innerLow) {
    return clamp01((value - low) / Math.max(innerLow - low, 1e-6));
  }
  return clamp01((high - value) / Math.max(high - innerHigh, 1e-6));
}

function clamp01(v: number): number {
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

function mean(arr: readonly number[]): number {
  if (arr.length === 0) return 0;
  let s = 0;
  for (const v of arr) s += v;
  return s / arr.length;
}

function stdOfNumbers(arr: readonly number[]): number {
  if (arr.length === 0) return 0;
  const m = mean(arr);
  let s = 0;
  for (const v of arr) {
    const d = v - m;
    s += d * d;
  }
  return Math.sqrt(s / arr.length);
}

// Re-exports for the orchestrator: `FaceUsabilityGate` needs `imageDataAsPatch`
// to construct a Patch view of an external ImageData without re-copying.
export { imageDataAsPatch };
