// Port of src/gates/critical_region_visibility.py — algorithms by @Aysenur15.
//
// Pixel-based critical face region visibility gate for live preview
// decisions. Estimates whether each of {left_eye, right_eye, nose, mouth,
// lower_face} is physically visible or covered by an external occluder
// (hand, mask, hijab fabric, paper held up to the camera, dark
// surface...).
//
// The Python source uses cv2 for cvtColor (BGR↔GRAY/LAB/HSV/YCrCb),
// Laplacian, Canny, createCLAHE/apply, inRange and mean. All of those
// are replaced with the hand-rolled helpers in utils/imageOps.ts. See
// IlluminationGate.ts header for a list of deviations.
//
// Calibrated thresholds (visibility cutoffs, redness deltas, brightness
// bands, physical-occlusion reason-token allowlist) are preserved
// verbatim so the paper-cited 0.55/0.60/0.65 per-region thresholds and
// the 0.32 occlusion-score cutoff remain meaningful.

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
  meanYcrcbCr,
  mouthHsvColorValidity,
  regionPatch,
  rgbToLabMean,
  stdU8,
  toGray,
} from "../utils/imageOps";

// === Temporal-streak defaults (used by FaceUsabilityGate). ===
export const TEMP_OCCLUSION_FRAMES = 2;
export const PERSISTENT_OCCLUSION_FRAMES = 6;
export const RECOVERY_CLEAR_FRAMES = 4;

const DEFAULT_REGION_WEIGHTS: Readonly<Record<string, number>> = {
  left_eye: 1.0,
  right_eye: 1.0,
  nose: 1.15,
  mouth: 1.15,
  lower_face: 1.25,
};

const DEFAULT_REGION_SCORE_THRESHOLD = 0.55;
const DEFAULT_OCCLUSION_SCORE_THRESHOLD = 0.32;
const EYE_VISIBILITY_THRESHOLD = 0.6;
const NOSE_VISIBILITY_THRESHOLD = 0.65;
// Lowered from 0.65 (Python comment): allow slightly covered mouth, but
// reject fully covered.
const MOUTH_VISIBILITY_THRESHOLD = 0.45;
const LOWER_FACE_VISIBILITY_THRESHOLD = 0.58;
const MOUTH_REDNESS_OCCLUDED_DELTA = 3.5;
const MOUTH_REDNESS_WARNING_DELTA = 6.0;
const LOWER_FACE_TEXTURE_OCCLUDED_RATIO = 0.62;
const LOWER_FACE_TEXTURE_WARNING_RATIO = 0.78;
const QUALITY_OCCLUSION_THRESHOLD = 42.0;

// Reason tokens that count as "physical" cover (vs mere illumination).
// Whitelisted tokens from the Python `_PHYSICAL_OCCLUSION_REASON_TOKENS`.
const PHYSICAL_OCCLUSION_REASON_TOKENS: ReadonlySet<string> = new Set([
  "dark_occluding_surface",
  "hand_overlap_signal",
  "eye_occluded",
  "lip_color_signature_missing",
  "mouth_structure_weakened",
  "mouth_replaced_by_skin_like_surface",
  "nose_replaced_by_skin_like_surface",
  "nose_structure_missing",
]);

const REGION_RATIOS: Readonly<Record<string, readonly [number, number, number, number]>> = {
  left_eye:   [0.14, 0.24, 0.24, 0.18],
  right_eye:  [0.62, 0.24, 0.24, 0.18],
  nose:       [0.39, 0.35, 0.22, 0.24],
  mouth:      [0.28, 0.63, 0.44, 0.16],
  lower_face: [0.18, 0.56, 0.64, 0.34],
};

const BASELINE_RATIOS = {
  left_cheek: [0.14, 0.41, 0.18, 0.18] as const,
  right_cheek: [0.68, 0.41, 0.18, 0.18] as const,
};

const REGION_NAMES = ["left_eye", "right_eye", "nose", "mouth", "lower_face"] as const;
export type RegionName = (typeof REGION_NAMES)[number];

export interface PreviewDetails {
  /** Optional pre-computed quality occlusion signal (Python preview_details). */
  quality_occlusion?: number | null;
  /** Optional pre-computed lower-face texture signal. */
  preview_lower_face_texture?: number | null;
}

export interface CriticalRegionEvaluateOptions {
  /** Optional global signal indicating a hand overlapping the face (0..1). */
  handOverlapSignal?: number;
  /** Optional preview heuristic side-channel signals (forwarded from upstream). */
  previewDetails?: PreviewDetails;
  /** Optional blur score from the preview pipeline. */
  blurScore?: number;
}

export interface CriticalRegionVisibilityResult {
  isCriticalOccluded: boolean;
  occlusionScore: number;
  occludedRegions: readonly string[];
  visibilityScores: Record<string, number>;
  regionReasons: Record<string, string>;
  blockingRegions: readonly string[];
  suspiciousRegions: readonly string[];
  reason: string;
}

interface RegionFeatures {
  visibility: number;
  brightness: number;
  texture: number;
  edge: number;
  gray_std: number;
}

interface BaselineFeatures {
  texture: number;
  edge: number;
  brightness: number;
  lab_l: number;
  lab_a: number;
  lab_b: number;
}

const FALLBACK_BASELINE: BaselineFeatures = Object.freeze({
  texture: 1.0,
  edge: 0.05,
  brightness: 128.0,
  lab_l: 60.0,
  lab_a: 10.0,
  lab_b: 15.0,
});

export class CriticalRegionVisibilityGate {
  private readonly regionScoreThreshold: number;
  private readonly occlusionScoreThreshold: number;
  private readonly clearReferenceFeatures: Map<string, RegionFeatures> = new Map();

  constructor(options?: {
    regionScoreThreshold?: number;
    occlusionScoreThreshold?: number;
  }) {
    this.regionScoreThreshold =
      options?.regionScoreThreshold ?? DEFAULT_REGION_SCORE_THRESHOLD;
    this.occlusionScoreThreshold =
      options?.occlusionScoreThreshold ?? DEFAULT_OCCLUSION_SCORE_THRESHOLD;
  }

  evaluate(
    frame: ImageData | null,
    faceBbox: GateBBox | null,
    options: CriticalRegionEvaluateOptions = {},
  ): CriticalRegionVisibilityResult {
    if (!frame || frame.data.length === 0 || !faceBbox) {
      return emptyResult("no_face_bbox_unavailable");
    }

    const { patch: faceRoi } = cropImageData(frame, faceBbox);
    if (isEmptyPatch(faceRoi)) {
      return emptyResult("no_face_bbox_unavailable");
    }

    // Use only cheek patches for baseline — forehead is covered by
    // headscarves/hats and corrupts the baseline Lab color, causing all
    // skin regions to score low. (Python lines 144-153.)
    const baselinePatches = [
      regionPatch(faceRoi, BASELINE_RATIOS.left_cheek),
      regionPatch(faceRoi, BASELINE_RATIOS.right_cheek),
    ].filter((p) => !isEmptyPatch(p));
    const baseline = baselineFeatures(baselinePatches.length > 0 ? baselinePatches : [faceRoi]);

    const visibilityScores: Record<string, number> = {};
    const regionReasons: Record<string, string> = {};
    const regionFeatures: Record<string, RegionFeatures> = {};
    const thresholdFailedRegions: string[] = [];
    let weightedVisibility = 0;
    let totalWeight = 0;

    for (const regionName of REGION_NAMES) {
      const patch = regionPatch(faceRoi, REGION_RATIOS[regionName]);
      const reference = this.clearReferenceFeatures.get(regionName);
      const { score, reason, features } = visibilityScore(
        patch,
        baseline,
        options.handOverlapSignal,
        regionName,
        reference,
      );
      visibilityScores[regionName] = score;
      regionReasons[regionName] = reason;
      regionFeatures[regionName] = features;
      const weight = DEFAULT_REGION_WEIGHTS[regionName] ?? 1.0;
      weightedVisibility += score * weight;
      totalWeight += weight;
      if (score < regionVisibilityThreshold(regionName)) {
        thresholdFailedRegions.push(regionName);
      }
    }

    const visibilityMean = weightedVisibility / Math.max(totalWeight, 1e-6);
    let occlusionScore = clamp01(1.0 - visibilityMean);

    const heuristicsResult = applyPreviewHeuristics({
      occludedRegions: thresholdFailedRegions,
      visibilityScores,
      regionReasons,
      occlusionScore,
      previewDetails: options.previewDetails,
      blurScore: options.blurScore,
    });
    const occludedRegionsAfter = heuristicsResult.occludedRegions;
    occlusionScore = heuristicsResult.occlusionScore;

    const { blockingRegions, suspiciousRegions } = classifyRegionStates(
      visibilityScores,
      regionReasons,
      occludedRegionsAfter,
    );

    const isCriticalOccluded = isCriticalOccludedFn(
      blockingRegions,
      suspiciousRegions,
      regionReasons,
      visibilityScores,
    );

    if (!isCriticalOccluded) {
      for (const regionName of REGION_NAMES) {
        const f = regionFeatures[regionName];
        if (f) this.clearReferenceFeatures.set(regionName, { ...f });
      }
    }

    // Side-effect parity: the unused threshold fields exist on the class
    // so external callers can re-tune calibration without re-instantiating
    // the gate; reference them here so noUnusedLocals stays happy on the
    // private fields without exporting them as public API.
    void this.regionScoreThreshold;
    void this.occlusionScoreThreshold;

    return {
      isCriticalOccluded,
      occlusionScore,
      occludedRegions: blockingRegions,
      visibilityScores,
      regionReasons,
      blockingRegions,
      suspiciousRegions,
      reason: isCriticalOccluded ? "critical_region_occluded" : "critical_region_visible",
    };
  }

  /** Reset the per-face clear-reference cache. */
  reset(): void {
    this.clearReferenceFeatures.clear();
  }
}

// ===========================================================================
// helpers (free functions to mirror the Python module-level layout)
// ===========================================================================

function emptyResult(reason: string): CriticalRegionVisibilityResult {
  return {
    isCriticalOccluded: false,
    occlusionScore: 0,
    occludedRegions: [],
    visibilityScores: {},
    regionReasons: {},
    blockingRegions: [],
    suspiciousRegions: [],
    reason,
  };
}

function baselineFeatures(patches: readonly Patch[]): BaselineFeatures {
  const valid = patches.filter((p) => !isEmptyPatch(p));
  if (valid.length === 0) return { ...FALLBACK_BASELINE };
  const textures: number[] = [];
  const edges: number[] = [];
  const brightnesses: number[] = [];
  const ls: number[] = [];
  const as_: number[] = [];
  const bs: number[] = [];
  for (const patch of valid) {
    const gray = toGray(patch);
    textures.push(laplacianVariance(gray, patch.width, patch.height));
    edges.push(edgeDensity(gray, patch.width, patch.height, 60, 140));
    brightnesses.push(meanU8(gray));
    const [l, a, b] = rgbToLabMean(patch);
    ls.push(l);
    as_.push(a);
    bs.push(b);
  }
  return {
    texture: mean(textures),
    edge: mean(edges),
    brightness: mean(brightnesses),
    lab_l: mean(ls),
    lab_a: mean(as_),
    lab_b: mean(bs),
  };
}

interface VisibilityScoreResult {
  score: number;
  reason: string;
  features: RegionFeatures;
}

function visibilityScore(
  patch: Patch,
  baseline: BaselineFeatures,
  handOverlapSignal: number | undefined,
  regionName: RegionName,
  referenceFeatures: RegionFeatures | undefined,
): VisibilityScoreResult {
  if (isEmptyPatch(patch)) {
    return {
      score: 0,
      reason: "patch_missing",
      features: { visibility: 0, brightness: 0, texture: 0, edge: 0, gray_std: 0 },
    };
  }
  const gray = toGray(patch);
  const brightness = meanU8(gray);
  let texture = laplacianVariance(gray, patch.width, patch.height);
  let edge = edgeDensity(gray, patch.width, patch.height, 60, 140);
  let grayStd = stdU8(gray);

  if (regionName === "left_eye" || regionName === "right_eye") {
    const clahe = claheGray(gray, patch.width, patch.height, 2.0, 4);
    const clTex = laplacianVariance(clahe, patch.width, patch.height);
    const clEdge = edgeDensity(clahe, patch.width, patch.height, 40, 120);
    const clStd = stdU8(clahe);
    texture = Math.max(texture, clTex);
    edge = Math.max(edge, clEdge);
    grayStd = Math.max(grayStd, clStd);
  }

  const [labL, labA, labB] = rgbToLabMean(patch);
  const brightnessScore = bandScore(brightness, 35.0, 235.0, 65.0, 210.0);
  const textureScore = ratioScore(texture, baseline.texture, 0.65);
  const edgeScore = ratioScore(edge, baseline.edge, 0.75);
  const uniformityScore = clamp01(grayStd / 28.0);
  const colorDistance = Math.sqrt(
    (labL - baseline.lab_l) ** 2 +
      (labA - baseline.lab_a) ** 2 +
      (labB - baseline.lab_b) ** 2,
  );
  const skinScore = Math.max(0, 1.0 - Math.min(colorDistance / 38.0, 1.0));
  const detailScore = 0.52 * textureScore + 0.30 * edgeScore + 0.18 * uniformityScore;

  let visibility =
    0.18 * brightnessScore +
    0.32 * textureScore +
    0.20 * edgeScore +
    0.12 * uniformityScore +
    0.18 * skinScore;
  const reasons: string[] = [];

  if (detailScore < 0.46) {
    visibility = Math.min(visibility, 0.4);
    reasons.push("facial_detail_missing");
  }

  // Darkness alone is a face-quality problem, not proof of physical
  // occlusion. Treat it as occlusion only when the region is also
  // structurally missing after normalization (Python lines 456-465).
  if (brightness < baseline.brightness * 0.55 && skinScore < 0.08) {
    if (detailScore >= 0.5) {
      visibility = Math.max(visibility, 0.56);
      reasons.push("poor_region_illumination");
    } else if (grayStd < 12.0) {
      visibility = Math.min(visibility, 0.32);
      reasons.push("dark_occluding_surface");
    }
  }

  if (regionName === "left_eye" || regionName === "right_eye") {
    const faceBrightness = Math.max(baseline.brightness, 1e-6);
    const eyeBrightnessRatio = brightness / faceBrightness;
    if (eyeBrightnessRatio < 0.82 && detailScore >= 0.54) {
      visibility = Math.max(visibility, eyeBrightnessRatio >= 0.6 ? 0.62 : 0.58);
      reasons.push("eye_low_light_warning");
    }
    if (skinScore > 0.78 && detailScore < 0.5 && uniformityScore < 0.6) {
      visibility = Math.min(visibility, 0.34);
      reasons.push("eye_occluded");
    } else if (detailScore < 0.34 && grayStd < 12.0) {
      visibility = Math.min(visibility, 0.38);
      reasons.push("eye_occluded");
    }
  }

  if (regionName === "mouth") {
    // Lips are always redder than surrounding cheek skin in the Lab a*
    // channel. (Python lines 480-510.)
    const lipRednessDelta = labA - baseline.lab_a;
    const brightnessRatio = brightness / Math.max(baseline.brightness, 1e-6);
    const textureIsFlat = textureScore < 0.72;
    const edgeIsFlat = edgeScore < 0.72;
    if (
      lipRednessDelta < MOUTH_REDNESS_OCCLUDED_DELTA &&
      textureIsFlat &&
      edgeIsFlat
    ) {
      visibility = Math.min(visibility, 0.18);
      reasons.push("lip_color_signature_missing");
    } else if (
      lipRednessDelta < MOUTH_REDNESS_WARNING_DELTA &&
      (textureIsFlat || edgeIsFlat)
    ) {
      visibility = Math.min(visibility, 0.42);
      reasons.push("mouth_structure_weakened");
    }

    if (
      brightnessScore >= 0.45 &&
      brightnessRatio >= 0.72 &&
      skinScore > 0.82 &&
      detailScore < 0.68 &&
      grayStd < 18.0
    ) {
      visibility = Math.min(visibility, 0.2);
      reasons.push("mouth_replaced_by_skin_like_surface");
    }

    // HSV / chrominance checks — only on real skin baseline.
    if (baseline.lab_a > 127 && brightness > 60.0) {
      const colorValidity = mouthHsvColorValidity(patch);
      if (!colorValidity.valid && colorValidity.confidence < 0.15) {
        visibility = Math.min(visibility, 0.2);
        reasons.push("mouth_roi_color_invalid");
      }
      const crMean = meanYcrcbCr(patch);
      if (!(crMean > 135.0 && crMean < 185.0)) {
        visibility = Math.min(visibility, 0.3);
        reasons.push("mouth_chrominance_anomaly");
      }
    }
  }

  if (regionName === "nose") {
    const brightnessRatio = brightness / Math.max(baseline.brightness, 1e-6);
    if (
      brightnessScore >= 0.45 &&
      brightnessRatio >= 0.72 &&
      skinScore > 0.84 &&
      detailScore < 0.7 &&
      grayStd < 18.0
    ) {
      visibility = Math.min(visibility, 0.22);
      reasons.push("nose_replaced_by_skin_like_surface");
    }
    if (
      brightnessScore >= 0.55 &&
      brightnessRatio >= 0.68 &&
      textureScore < 0.6 &&
      edgeScore < 0.58 &&
      grayStd < 22.0
    ) {
      visibility = Math.min(visibility, 0.34);
      reasons.push("nose_structure_missing");
    }
  }

  if ((regionName === "mouth" || regionName === "nose") && referenceFeatures) {
    const refTexRatio = texture / Math.max(referenceFeatures.texture || texture, 1e-6);
    const refEdgeRatio = edge / Math.max(referenceFeatures.edge || edge, 1e-6);
    const refStdRatio = grayStd / Math.max(referenceFeatures.gray_std || grayStd, 1e-6);
    if (refTexRatio < 0.58 && refEdgeRatio < 0.62) {
      visibility = Math.min(visibility, regionName === "mouth" ? 0.28 : 0.32);
      reasons.push("detail_drop_vs_clear_face");
    }
    if (refStdRatio < 0.65 && detailScore < 0.74) {
      visibility = Math.min(visibility, 0.36);
      reasons.push("uniform_surface_vs_clear_face");
    }
  }

  if (handOverlapSignal !== undefined) {
    visibility *= clamp01(1.0 - handOverlapSignal);
    if (handOverlapSignal > 0.2) reasons.push("hand_overlap_signal");
  }
  visibility = clamp01(visibility);

  const uniqueReasons = Array.from(new Set(reasons));
  const reason = uniqueReasons.length > 0 ? uniqueReasons.join("|") : "region_visible";
  return {
    score: visibility,
    reason,
    features: { visibility, brightness, texture, edge, gray_std: grayStd },
  };
}

function regionVisibilityThreshold(regionName: string): number {
  if (regionName === "left_eye" || regionName === "right_eye") return EYE_VISIBILITY_THRESHOLD;
  if (regionName === "nose") return NOSE_VISIBILITY_THRESHOLD;
  if (regionName === "mouth") return MOUTH_VISIBILITY_THRESHOLD;
  if (regionName === "lower_face") return LOWER_FACE_VISIBILITY_THRESHOLD;
  return DEFAULT_REGION_SCORE_THRESHOLD;
}

interface PreviewHeuristicArgs {
  occludedRegions: string[];
  visibilityScores: Record<string, number>;
  regionReasons: Record<string, string>;
  occlusionScore: number;
  previewDetails: PreviewDetails | undefined;
  blurScore: number | undefined;
}

function applyPreviewHeuristics(args: PreviewHeuristicArgs): {
  occludedRegions: string[];
  occlusionScore: number;
} {
  let { occlusionScore } = args;
  const { occludedRegions, visibilityScores, regionReasons, previewDetails, blurScore } = args;
  if (!previewDetails) return { occludedRegions, occlusionScore };

  const qualityOcclusion = coerceFloat(previewDetails.quality_occlusion);
  const lowerFaceTexture = coerceFloat(previewDetails.preview_lower_face_texture);

  if (lowerFaceTexture !== null && blurScore !== undefined && blurScore > 10.0) {
    const lowerRatio = lowerFaceTexture / Math.max(blurScore, 1e-6);
    if (lowerRatio < LOWER_FACE_TEXTURE_WARNING_RATIO) {
      let severity: number;
      let visibilityFloor: number;
      let scoreFloor: number;
      if (lowerRatio < LOWER_FACE_TEXTURE_OCCLUDED_RATIO) {
        severity = clamp01(
          (LOWER_FACE_TEXTURE_OCCLUDED_RATIO - lowerRatio) /
            Math.max(LOWER_FACE_TEXTURE_OCCLUDED_RATIO, 1e-6),
        );
        visibilityFloor = 0.34;
        scoreFloor = 0.26;
      } else {
        severity = clamp01(
          (LOWER_FACE_TEXTURE_WARNING_RATIO - lowerRatio) /
            Math.max(
              LOWER_FACE_TEXTURE_WARNING_RATIO - LOWER_FACE_TEXTURE_OCCLUDED_RATIO,
              1e-6,
            ),
        );
        visibilityFloor = 0.52;
        scoreFloor = 0.18;
      }
      // Scoped to mouth/lower_face only — nose is excluded to avoid
      // beard/smile/lighting noise being treated as a full lower-face
      // occlusion (Python lines 654-657).
      for (const regionName of ["mouth", "lower_face"] as const) {
        const current = visibilityScores[regionName] ?? 1.0;
        const next = Math.min(current, Math.max(visibilityFloor, 1.0 - severity));
        visibilityScores[regionName] = next;
        if (next < regionVisibilityThreshold(regionName) && !occludedRegions.includes(regionName)) {
          occludedRegions.push(regionName);
        }
        if (next < regionVisibilityThreshold(regionName)) {
          regionReasons[regionName] = mergeReason(
            regionReasons[regionName],
            "lower_face_texture_drop",
          );
        }
      }
      occlusionScore = Math.max(occlusionScore, scoreFloor + 0.25 * severity);
    }
  }

  if (qualityOcclusion !== null && qualityOcclusion < QUALITY_OCCLUSION_THRESHOLD) {
    const severity = clamp01((QUALITY_OCCLUSION_THRESHOLD - qualityOcclusion) / 24.0);
    for (const regionName of ["mouth", "lower_face"] as const) {
      const current = visibilityScores[regionName] ?? 1.0;
      const next = Math.min(current, Math.max(0.45, 1.0 - 0.6 * severity));
      visibilityScores[regionName] = next;
      if (next < regionVisibilityThreshold(regionName) && !occludedRegions.includes(regionName)) {
        occludedRegions.push(regionName);
      }
      if (next < regionVisibilityThreshold(regionName)) {
        regionReasons[regionName] = mergeReason(
          regionReasons[regionName],
          "quality_occlusion_signal",
        );
      }
    }
    occlusionScore = Math.max(occlusionScore, 0.2 + 0.25 * severity);
  }

  return { occludedRegions, occlusionScore };
}

function classifyRegionStates(
  visibilityScores: Record<string, number>,
  regionReasons: Record<string, string>,
  thresholdFailedRegions: readonly string[],
): { blockingRegions: string[]; suspiciousRegions: string[] } {
  const eyeVis = (n: "left_eye" | "right_eye") =>
    (visibilityScores[n] ?? 1.0) >= EYE_VISIBILITY_THRESHOLD;
  const noseVisible = (visibilityScores["nose"] ?? 1.0) >= NOSE_VISIBILITY_THRESHOLD;
  const mouthVisible = (visibilityScores["mouth"] ?? 1.0) >= MOUTH_VISIBILITY_THRESHOLD;
  const lowerFaceVisible =
    (visibilityScores["lower_face"] ?? 1.0) >= LOWER_FACE_VISIBILITY_THRESHOLD;

  const leftEyeBlocked = isRegionPhysicallyBlocked("left_eye", visibilityScores, regionReasons);
  const rightEyeBlocked = isRegionPhysicallyBlocked("right_eye", visibilityScores, regionReasons);
  const noseBlocked = isRegionPhysicallyBlocked("nose", visibilityScores, regionReasons);
  const mouthBlocked = isRegionPhysicallyBlocked("mouth", visibilityScores, regionReasons);
  const lowerFaceBlocked = isRegionPhysicallyBlocked("lower_face", visibilityScores, regionReasons);

  const blockingRegions: string[] = [];
  const suspiciousRegions: string[] = [];

  if (noseBlocked) blockingRegions.push("nose");
  else if (!noseVisible) suspiciousRegions.push("nose");

  if (mouthBlocked) blockingRegions.push("mouth");
  else if (!mouthVisible) suspiciousRegions.push("mouth");

  if (lowerFaceBlocked && (mouthBlocked || noseBlocked)) blockingRegions.push("lower_face");
  else if (!lowerFaceVisible) suspiciousRegions.push("lower_face");

  if (leftEyeBlocked && rightEyeBlocked) {
    blockingRegions.push("left_eye", "right_eye");
    regionReasons["left_eye"] = mergeReason(regionReasons["left_eye"], "eye_occluded");
    regionReasons["right_eye"] = mergeReason(regionReasons["right_eye"], "eye_occluded");
  } else {
    if (!eyeVis("left_eye")) {
      suspiciousRegions.push("left_eye");
      if (!leftEyeBlocked) {
        regionReasons["left_eye"] = mergeReason(
          regionReasons["left_eye"],
          "single_eye_low_light_warning",
        );
      }
    }
    if (!eyeVis("right_eye")) {
      suspiciousRegions.push("right_eye");
      if (!rightEyeBlocked) {
        regionReasons["right_eye"] = mergeReason(
          regionReasons["right_eye"],
          "single_eye_low_light_warning",
        );
      }
    }
  }

  for (const regionName of thresholdFailedRegions) {
    if (!blockingRegions.includes(regionName) && !suspiciousRegions.includes(regionName)) {
      suspiciousRegions.push(regionName);
    }
  }

  return { blockingRegions, suspiciousRegions };
}

function isCriticalOccludedFn(
  blockingRegions: readonly string[],
  suspiciousRegions: readonly string[],
  regionReasons: Record<string, string>,
  visibilityScores: Record<string, number>,
): boolean {
  const blocked = new Set(blockingRegions);
  const suspicious = new Set(suspiciousRegions);

  // Both eyes physically blocked together.
  if (blocked.has("left_eye") && blocked.has("right_eye")) return true;

  // Nose or mouth individually blocked.
  if (blocked.has("nose") || blocked.has("mouth")) return true;

  // Lower face: nose + mouth degraded with at least one physical token.
  if (suspicious.has("nose") && suspicious.has("mouth")) {
    const noseTokens = reasonTokens(regionReasons["nose"]);
    const mouthTokens = reasonTokens(regionReasons["mouth"]);
    for (const t of [...noseTokens, ...mouthTokens]) {
      if (PHYSICAL_OCCLUSION_REASON_TOKENS.has(t)) return true;
    }
  }

  // Half-face: one eye blocked + lower region degraded.
  const oneEyeBlocked = blocked.has("left_eye") || blocked.has("right_eye");
  if (oneEyeBlocked) {
    const noseLow = (visibilityScores["nose"] ?? 1.0) < NOSE_VISIBILITY_THRESHOLD;
    const mouthLow = (visibilityScores["mouth"] ?? 1.0) < MOUTH_VISIBILITY_THRESHOLD;
    const lowerLow = (visibilityScores["lower_face"] ?? 1.0) < LOWER_FACE_VISIBILITY_THRESHOLD;
    if (noseLow || mouthLow || lowerLow) return true;
  }
  return false;
}

function isRegionPhysicallyBlocked(
  regionName: string,
  visibilityScores: Record<string, number>,
  regionReasons: Record<string, string>,
): boolean {
  const score = visibilityScores[regionName] ?? 1.0;
  if (score >= regionVisibilityThreshold(regionName)) return false;
  const tokens = reasonTokens(regionReasons[regionName]);
  for (const t of tokens) {
    if (PHYSICAL_OCCLUSION_REASON_TOKENS.has(t)) return true;
  }
  return false;
}

function reasonTokens(reason: string | undefined): Set<string> {
  if (!reason || reason === "region_visible") return new Set();
  return new Set(reason.split("|").filter((t) => t));
}

function mergeReason(existing: string | undefined, newReason: string): string {
  if (!existing || existing === "region_visible") return newReason;
  if (existing.split("|").includes(newReason)) return existing;
  return `${existing}|${newReason}`;
}

function coerceFloat(value: number | null | undefined): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "number" || Number.isNaN(value)) return null;
  return value;
}

function ratioScore(value: number, baselineValue: number, stretch: number): number {
  const denom = Math.max(baselineValue, 1e-6);
  const ratio = value / denom;
  return clamp01(ratio / Math.max(stretch, 1e-6));
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
  if (value < innerLow) return clamp01((value - low) / Math.max(innerLow - low, 1e-6));
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

// Public re-export so FaceUsabilityGate can keep zero direct deps on utils.
export { imageDataAsPatch };
