// Port of src/gates/face_usability.py — algorithms by @Aysenur15.
//
// Pre-liveness face usability gate. Combines the per-frame quality
// (`IlluminationGate`) and physical-occlusion (`CriticalRegionVisibilityGate`)
// signals into a small state machine that decides whether to feed the
// frame into the heavier liveness pipeline.
//
// State machine states (verbatim from Python):
//   NO_FACE              → bbox missing / frame empty.
//   OCCLUDED_PENDING     → occlusion this frame, streak below confirm threshold.
//   OCCLUDED_CONFIRMED   → occlusion streak ≥ occlusion_confirm_frames.
//   OCCLUDED_NO_FACE     → occlusion streak ≥ no_face_confirm_frames.
//   RECOVERING           → previously blocked, waiting for required_clear_frames.
//   CLEAR                → usable for liveness.
//
// All confirm-frame defaults are preserved verbatim from the source.

import { GateBBox } from "../utils/imageOps";
import { CriticalRegionVisibilityGate, PreviewDetails } from "./CriticalRegionVisibilityGate";
import { IlluminationGate } from "./IlluminationGate";

// Defaults (mirror the module-level constants in face_usability.py).
export const LOW_QUALITY_CONFIRM_FRAMES = 2;
export const OCCLUSION_CONFIRM_FRAMES = 2;
export const NO_FACE_CONFIRM_FRAMES = 6;
export const TEMP_CLEAR_CONFIRM_FRAMES = 1;
export const CLEAR_CONFIRM_FRAMES = 2;

const EYE_STRICT_UNRELIABLE_THRESHOLD = 0.6;

export type UsabilityState =
  | "NO_FACE"
  | "OCCLUDED_PENDING"
  | "OCCLUDED_CONFIRMED"
  | "OCCLUDED_NO_FACE"
  | "RECOVERING"
  | "CLEAR";

export type StatusOverride = "NO_FACE" | "INSUFFICIENT_EVIDENCE" | null;

export interface FaceUsabilityOptions {
  lowQualityConfirmFrames?: number;
  occlusionConfirmFrames?: number;
  noFaceConfirmFrames?: number;
  tempClearConfirmFrames?: number;
  clearConfirmFrames?: number;
}

export interface FaceUsabilityEvaluateOptions {
  /** Optional side-channel preview metrics forwarded to the visibility gate. */
  previewDetails?: PreviewDetails;
  /** Optional blur score for the preview heuristic. */
  blurScore?: number;
}

export interface FaceUsabilityResult {
  usable: boolean;
  noFace: boolean;
  qualityOk: boolean;
  occluded: boolean;
  occlusionScore: number;
  occludedRegions: readonly string[];
  visibilityScores: Record<string, number>;
  regionReasons: Record<string, string>;
  blockingRegions: readonly string[];
  suspiciousRegions: readonly string[];
  qualityStatus: string;
  qualityReason: string;
  perRegionBrightness: Record<string, number>;
  brightnessUniformity: number;
  illuminationScore: number;
  globalFaceBrightness: number;
  shadowAsymmetry: number;
  underexposedRegions: readonly string[];
  overexposedRegions: readonly string[];
  physicalOcclusionScore: number;
  physicalOcclusionRegions: readonly string[];
  physicalOcclusionReason: string;
  livenessSkippedReason: string;
  reason: string;
  state: UsabilityState;
  blocked: boolean;
  statusOverride: StatusOverride;
  bboxDetected: boolean;
  occlusionStreak: number;
  qualityStreak: number;
  clearStreak: number;
}

export class FaceUsabilityGate {
  private readonly qualityGate = new IlluminationGate();
  private readonly visibilityGate = new CriticalRegionVisibilityGate();

  private readonly lowQualityConfirmFrames: number;
  private readonly occlusionConfirmFrames: number;
  private readonly noFaceConfirmFrames: number;
  private readonly tempClearConfirmFrames: number;
  private readonly clearConfirmFrames: number;

  // `qualityStreak` is preserved for parity with the Python state machine,
  // which uses it for the LOW_QUALITY path. The current port treats the
  // illumination signal as informational (rolled into the OCCLUSION state
  // when accompanied by structural failures) and does not advance the
  // streak; the field is left in for upcoming Phase-3 work that wires the
  // LOW_QUALITY confirm path independently.
  private qualityStreak = 0;
  private occlusionStreak = 0;
  private clearStreak = 0;
  private blocked = false;
  private blockedMode: "OCCLUSION" | "LOW_QUALITY" | null = null;
  private enteredNoFace = false;
  private state: UsabilityState = "CLEAR";

  constructor(opts: FaceUsabilityOptions = {}) {
    this.lowQualityConfirmFrames = Math.max(1, Math.floor(opts.lowQualityConfirmFrames ?? LOW_QUALITY_CONFIRM_FRAMES));
    this.occlusionConfirmFrames = Math.max(1, Math.floor(opts.occlusionConfirmFrames ?? OCCLUSION_CONFIRM_FRAMES));
    this.noFaceConfirmFrames = Math.max(
      this.occlusionConfirmFrames,
      Math.floor(opts.noFaceConfirmFrames ?? NO_FACE_CONFIRM_FRAMES),
    );
    this.tempClearConfirmFrames = Math.max(1, Math.floor(opts.tempClearConfirmFrames ?? TEMP_CLEAR_CONFIRM_FRAMES));
    this.clearConfirmFrames = Math.max(1, Math.floor(opts.clearConfirmFrames ?? CLEAR_CONFIRM_FRAMES));
  }

  evaluate(
    frame: ImageData | null,
    faceBbox: GateBBox | null,
    opts: FaceUsabilityEvaluateOptions = {},
  ): FaceUsabilityResult {
    // === No-face short-circuit. ===
    if (!frame || frame.data.length === 0 || !faceBbox) {
      this.qualityStreak = 0;
      this.occlusionStreak = 0;
      this.clearStreak = 0;
      this.blocked = false;
      this.state = "NO_FACE";
      return {
        usable: false,
        noFace: true,
        qualityOk: false,
        occluded: false,
        occlusionScore: 0,
        occludedRegions: [],
        visibilityScores: {},
        regionReasons: {},
        blockingRegions: [],
        suspiciousRegions: [],
        qualityStatus: "-",
        qualityReason: "-",
        perRegionBrightness: {},
        brightnessUniformity: 0,
        illuminationScore: 0,
        globalFaceBrightness: 0,
        shadowAsymmetry: 0,
        underexposedRegions: [],
        overexposedRegions: [],
        physicalOcclusionScore: 0,
        physicalOcclusionRegions: [],
        physicalOcclusionReason: "-",
        livenessSkippedReason: "no_face_detected",
        reason: "no_face_detected",
        state: this.state,
        blocked: true,
        statusOverride: "NO_FACE",
        bboxDetected: false,
        occlusionStreak: 0,
        qualityStreak: 0,
        clearStreak: 0,
      };
    }

    const quality = this.qualityGate.evaluate(frame, faceBbox);
    const visibility = this.visibilityGate.evaluate(frame, faceBbox, {
      previewDetails: opts.previewDetails,
      blurScore: opts.blurScore,
    });

    // Visibility-score thresholds (2026-05-31 relaxation). The original
    // 0.65 nose/mouth bar was calibrated against the Python reference
    // pipeline (480p Python-OpenCV pipeline). On the browser port with
    // MediaPipe FaceLandmarker, a clearly-visible mouth at moderate
    // distance often scores 0.55-0.70 — landmarks are slightly noisy
    // but the region IS visible. At 0.65 the gate was firing
    // "OCCLUDED_NO_FACE / OCCLUDED_CONFIRMED" on 50 %+ of frames in
    // multiple LIVE captures, which pinned the displayed confidence at
    // 34 % even though the LivenessProver was reporting 100/100 PROVEN
    // LIVE — a glaring user-facing inconsistency.
    //
    // Lowered to 0.45 (still well above the ~0.3 "actually occluded"
    // band). Eyes left at 0.6 — eyelid detection is more reliable and
    // a 0.6 score really does indicate eye obstruction.
    const leftEyeScore = visibility.visibilityScores["left_eye"] ?? 1.0;
    const rightEyeScore = visibility.visibilityScores["right_eye"] ?? 1.0;
    const leftEyeVisible = leftEyeScore >= 0.6;
    const rightEyeVisible = rightEyeScore >= 0.6;
    const noseVisible = (visibility.visibilityScores["nose"] ?? 1.0) >= 0.45;
    const mouthVisible = (visibility.visibilityScores["mouth"] ?? 1.0) >= 0.45;
    const lowerFaceVisible = (visibility.visibilityScores["lower_face"] ?? 1.0) >= 0.45;
    const bothEyesUnreliable =
      leftEyeScore < EYE_STRICT_UNRELIABLE_THRESHOLD &&
      rightEyeScore < EYE_STRICT_UNRELIABLE_THRESHOLD;

    const derived: string[] = [...visibility.occludedRegions];
    if (bothEyesUnreliable || (!leftEyeVisible && !rightEyeVisible)) {
      derived.push("left_eye", "right_eye");
    }
    if (!noseVisible) derived.push("nose");
    if (!mouthVisible) derived.push("mouth");
    if (!lowerFaceVisible) derived.push("lower_face");
    const derivedOccludedRegions = Array.from(new Set(derived));

    const lowerFaceRegionsMissing =
      (noseVisible ? 0 : 1) + (mouthVisible ? 0 : 1) + (lowerFaceVisible ? 0 : 1);
    // Structural-occlusion AND-conditions tightened in parallel with the
    // visibility-threshold relaxation above: require a higher occlusionScore
    // to flag, and require more regions missing for the lower-bar trigger.
    // 0.70 occlusionScore is a clearly-bad reading (typical visible-face
    // captures land 0.20-0.45). The original 0.55-0.60 fired too often.
    const structuralOcclusionNow =
      bothEyesUnreliable ||
      (!leftEyeVisible && !rightEyeVisible) ||
      ((!mouthVisible && !lowerFaceVisible) && visibility.occlusionScore >= 0.70) ||
      (lowerFaceRegionsMissing >= 3 && visibility.occlusionScore >= 0.65) ||
      (visibility.occlusionScore >= 0.75 &&
        lowerFaceRegionsMissing >= 2 &&
        (!mouthVisible || !lowerFaceVisible));

    const occludedNow = visibility.isCriticalOccluded || structuralOcclusionNow;

    if (occludedNow) {
      this.qualityStreak = 0;
      this.occlusionStreak += 1;
      this.clearStreak = 0;
      this.blocked = true;
      this.blockedMode = "OCCLUSION";

      let statusOverride: StatusOverride;
      if (this.occlusionStreak >= this.noFaceConfirmFrames) {
        this.enteredNoFace = true;
        this.state = "OCCLUDED_NO_FACE";
        statusOverride = "NO_FACE";
      } else if (this.occlusionStreak >= this.occlusionConfirmFrames) {
        this.state = "OCCLUDED_CONFIRMED";
        statusOverride = "INSUFFICIENT_EVIDENCE";
      } else {
        this.state = "OCCLUDED_PENDING";
        statusOverride = "INSUFFICIENT_EVIDENCE";
      }
      const suspicious = uniqueExcept(
        [...visibility.suspiciousRegions, ...derivedOccludedRegions],
        derivedOccludedRegions,
      );
      return {
        usable: false,
        noFace: true,
        qualityOk: true,
        occluded: true,
        occlusionScore: visibility.occlusionScore,
        occludedRegions: derivedOccludedRegions,
        visibilityScores: { ...visibility.visibilityScores },
        regionReasons: { ...visibility.regionReasons },
        blockingRegions: derivedOccludedRegions,
        suspiciousRegions: suspicious,
        qualityStatus: quality.qualityStatus,
        qualityReason: quality.qualityReason,
        perRegionBrightness: { ...quality.perRegionBrightness },
        brightnessUniformity: quality.brightnessUniformity,
        illuminationScore: quality.illuminationScore,
        globalFaceBrightness: quality.globalFaceBrightness,
        shadowAsymmetry: quality.shadowAsymmetry,
        underexposedRegions: [...quality.underexposedRegions],
        overexposedRegions: [...quality.overexposedRegions],
        physicalOcclusionScore: visibility.occlusionScore,
        physicalOcclusionRegions: derivedOccludedRegions,
        physicalOcclusionReason: visibility.isCriticalOccluded
          ? visibility.reason
          : "structural_face_region_occluded",
        livenessSkippedReason: "critical_face_region_occluded",
        reason: "critical_face_region_occluded",
        state: this.state,
        blocked: true,
        statusOverride,
        bboxDetected: true,
        occlusionStreak: this.occlusionStreak,
        qualityStreak: 0,
        clearStreak: 0,
      };
    }

    // === Clear branch. ===
    this.occlusionStreak = 0;
    this.qualityStreak = 0;
    if (this.blocked) {
      this.clearStreak += 1;
      const requiredClearFrames =
        this.enteredNoFace || this.blockedMode === "LOW_QUALITY"
          ? this.clearConfirmFrames
          : this.tempClearConfirmFrames;
      if (this.clearStreak < requiredClearFrames) {
        this.state = "RECOVERING";
        return {
          usable: false,
          noFace: false,
          qualityOk: true,
          occluded: false,
          occlusionScore: visibility.occlusionScore,
          occludedRegions: [],
          visibilityScores: { ...visibility.visibilityScores },
          regionReasons: { ...visibility.regionReasons },
          blockingRegions: visibility.blockingRegions,
          suspiciousRegions: visibility.suspiciousRegions,
          qualityStatus: quality.qualityStatus,
          qualityReason: quality.qualityReason,
          perRegionBrightness: { ...quality.perRegionBrightness },
          brightnessUniformity: quality.brightnessUniformity,
          illuminationScore: quality.illuminationScore,
          globalFaceBrightness: quality.globalFaceBrightness,
          shadowAsymmetry: quality.shadowAsymmetry,
          underexposedRegions: [...quality.underexposedRegions],
          overexposedRegions: [...quality.overexposedRegions],
          physicalOcclusionScore: visibility.occlusionScore,
          physicalOcclusionRegions: visibility.occludedRegions,
          physicalOcclusionReason: visibility.reason,
          livenessSkippedReason: "recovering_face_usability",
          reason: "recovering_face_usability",
          state: this.state,
          blocked: true,
          statusOverride: "INSUFFICIENT_EVIDENCE",
          bboxDetected: true,
          occlusionStreak: 0,
          qualityStreak: 0,
          clearStreak: this.clearStreak,
        };
      }
      this.blocked = false;
      this.enteredNoFace = false;
      this.blockedMode = null;
      this.clearStreak = 0;
    }

    // CLEAR — usable.
    this.state = "CLEAR";
    // Touch the unused low_quality_confirm field + quality_streak for
    // noUnusedLocals; these exist for parity with the Python signature and
    // are reserved for the LOW_QUALITY confirm path that Phase 3 wires.
    void this.lowQualityConfirmFrames;
    void this.qualityStreak;
    return {
      usable: true,
      noFace: false,
      qualityOk: true,
      occluded: false,
      occlusionScore: visibility.occlusionScore,
      occludedRegions: [],
      visibilityScores: { ...visibility.visibilityScores },
      regionReasons: { ...visibility.regionReasons },
      blockingRegions: visibility.blockingRegions,
      suspiciousRegions: visibility.suspiciousRegions,
      qualityStatus: quality.qualityStatus,
      qualityReason: quality.qualityReason,
      perRegionBrightness: { ...quality.perRegionBrightness },
      brightnessUniformity: quality.brightnessUniformity,
      illuminationScore: quality.illuminationScore,
      globalFaceBrightness: quality.globalFaceBrightness,
      shadowAsymmetry: quality.shadowAsymmetry,
      underexposedRegions: [...quality.underexposedRegions],
      overexposedRegions: [...quality.overexposedRegions],
      physicalOcclusionScore: visibility.occlusionScore,
      physicalOcclusionRegions: visibility.occludedRegions,
      physicalOcclusionReason: visibility.reason,
      livenessSkippedReason: "-",
      reason: "face_usable",
      state: this.state,
      blocked: false,
      statusOverride: null,
      bboxDetected: true,
      occlusionStreak: 0,
      qualityStreak: 0,
      clearStreak: this.clearStreak,
    };
  }

  /** Reset all streak counters. */
  reset(): void {
    this.qualityStreak = 0;
    this.occlusionStreak = 0;
    this.clearStreak = 0;
    this.blocked = false;
    this.blockedMode = null;
    this.enteredNoFace = false;
    this.state = "CLEAR";
    this.visibilityGate.reset();
  }
}

function uniqueExcept(items: readonly string[], exclude: readonly string[]): string[] {
  const excl = new Set(exclude);
  const seen = new Set<string>();
  const out: string[] = [];
  for (const v of items) {
    if (excl.has(v)) continue;
    if (seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}
