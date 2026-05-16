// Port of src/domain/models.py
// Data classes + enums used across the spoof-detector pipeline.
// Names mirror the Python source so JSON snapshots are interoperable
// (e.g. SpoofCategory.STATIC_IMAGE === "static_image").

/**
 * Multi-class spoof taxonomy.
 *
 * Covers all known face presentation attack types:
 * - Static: printed photo or digital still on screen
 * - Replay: pre-recorded video shown on a display
 * - Mask: realistic 3D silicone/latex mask
 * - Makeup: heavy contouring or prosthetics
 * - AR Filter: live AR overlay (Snapchat, Instagram, OBS)
 * - Deepfake: virtual webcam injection (DeepFaceLive etc.)
 * - Real: genuine live person
 */
export enum SpoofCategory {
  REAL = "real",
  STATIC_IMAGE = "static_image",
  VIDEO_REPLAY = "video_replay",
  MASK_3D = "mask_3d",
  HEAVY_MAKEUP = "heavy_makeup",
  AR_FILTER = "ar_filter",
  DEEPFAKE_INJECT = "deepfake_inject",
}

export const ALL_SPOOF_CATEGORIES: readonly SpoofCategory[] = Object.freeze([
  SpoofCategory.REAL,
  SpoofCategory.STATIC_IMAGE,
  SpoofCategory.VIDEO_REPLAY,
  SpoofCategory.MASK_3D,
  SpoofCategory.HEAVY_MAKEUP,
  SpoofCategory.AR_FILTER,
  SpoofCategory.DEEPFAKE_INJECT,
]);

/** Display labels for overlay (mirrors CATEGORY_LABELS). */
export const CATEGORY_LABELS: Readonly<Record<SpoofCategory, string>> = {
  [SpoofCategory.REAL]: "Real",
  [SpoofCategory.STATIC_IMAGE]: "Static Image",
  [SpoofCategory.VIDEO_REPLAY]: "Video Replay",
  [SpoofCategory.MASK_3D]: "Mask",
  [SpoofCategory.HEAVY_MAKEUP]: "Makeup",
  [SpoofCategory.AR_FILTER]: "AR Filter",
  [SpoofCategory.DEEPFAKE_INJECT]: "Deepfake",
};

/** Face bounding box in pixel coordinates. */
export class BBox {
  constructor(
    public readonly x1: number,
    public readonly y1: number,
    public readonly x2: number,
    public readonly y2: number,
  ) {}

  get width(): number {
    return this.x2 - this.x1;
  }

  get height(): number {
    return this.y2 - this.y1;
  }

  get center(): [number, number] {
    return [
      Math.floor((this.x1 + this.x2) / 2),
      Math.floor((this.y1 + this.y2) / 2),
    ];
  }

  get area(): number {
    return this.width * this.height;
  }

  /** Intersection over Union with another bbox. */
  iou(other: BBox): number {
    const ix1 = Math.max(this.x1, other.x1);
    const iy1 = Math.max(this.y1, other.y1);
    const ix2 = Math.min(this.x2, other.x2);
    const iy2 = Math.min(this.y2, other.y2);
    const inter =
      Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
    const union = this.area + other.area - inter;
    return inter / Math.max(union, 1);
  }
}

/** Detected face region of interest. Landmarks are Nx2 flat [x0,y0,x1,y1,...]. */
export interface FaceROI {
  /** Tracker-assigned identity. */
  face_id: number;
  bbox: BBox;
  /** Detector confidence, 0..1. */
  confidence: number;
  /** Optional Nx2 landmarks (flat Float32Array of length 2N). */
  landmarks?: Float32Array;
  /** Optional cropped face region as ImageData. */
  crop?: ImageData;
}

/** Result from a single analyzer. */
export interface AnalyzerResult {
  name: string;
  /** 0-100, higher = more live-like. */
  score: number;
  details: Record<string, unknown>;
  elapsed_ms: number;
}

/** Helper: build an AnalyzerResult with sane defaults. */
export function makeAnalyzerResult(
  name: string,
  score: number,
  details: Record<string, unknown> = {},
  elapsed_ms = 0,
): AnalyzerResult {
  return { name, score, details, elapsed_ms };
}

/**
 * Common interface for all per-face analyzers (port of the duck-typed
 * `analyze(face_crop, face_roi)` contract in the Python pipeline).
 *
 * Analyzers that need the full original frame (e.g. MiniFASNet, Device
 * Boundary, Screen Flicker) implement `setFrame()`.
 *
 * Analyzers that need landmark data (e.g. Landmark Variance, Blink,
 * Micro-Tremor) implement `setLandmarks()`. The orchestrator pulls
 * landmarks off `FaceROI.landmarks` and forwards them once per frame
 * — analyzers don't run MediaPipe themselves.
 */
export interface IFaceAnalyzer {
  /** Stable name (e.g. "minifasnet", "blink"). Drives fusion weights. */
  readonly name: string;
  /** Analyze a face. `faceCrop` may be null for analyzers that need the full frame. */
  analyze(
    faceCrop: ImageData | null,
    face: FaceROI,
  ): Promise<AnalyzerResult> | AnalyzerResult;
}

/** Final multi-class spoof classification for a face. */
export interface SpoofClassification {
  face_id: number;
  /** Probability per category, sums to ~1.0. */
  probabilities: Record<SpoofCategory, number>;
  dominant_category: SpoofCategory;
  /** Confidence of dominant category (0-1). */
  confidence: number;
  analyzer_results: Record<string, AnalyzerResult>;
}

/**
 * Build a SpoofClassification from a raw probability map.
 * Mirrors `SpoofClassification.from_probabilities` in models.py.
 */
export function classificationFromProbabilities(
  face_id: number,
  probs: Record<SpoofCategory, number>,
  analyzer_results: Record<string, AnalyzerResult> = {},
): SpoofClassification {
  let total = 0;
  for (const cat of ALL_SPOOF_CATEGORIES) {
    total += probs[cat] ?? 0;
  }

  const normalized: Record<SpoofCategory, number> = {} as Record<
    SpoofCategory,
    number
  >;
  if (total > 0) {
    for (const cat of ALL_SPOOF_CATEGORIES) {
      normalized[cat] = (probs[cat] ?? 0) / total;
    }
  } else {
    const uniform = 1.0 / ALL_SPOOF_CATEGORIES.length;
    for (const cat of ALL_SPOOF_CATEGORIES) {
      normalized[cat] = uniform;
    }
  }

  let dominant = SpoofCategory.REAL;
  let dominantP = -Infinity;
  for (const cat of ALL_SPOOF_CATEGORIES) {
    const p = normalized[cat];
    if (p > dominantP) {
      dominant = cat;
      dominantP = p;
    }
  }

  return {
    face_id,
    probabilities: normalized,
    dominant_category: dominant,
    confidence: dominantP,
    analyzer_results,
  };
}

/** Complete analysis of a single frame. */
export interface FrameAnalysis {
  frame_id: number;
  faces: FaceROI[];
  /** face_id → classification. */
  classifications: Record<number, SpoofClassification>;
  frame_signals: Record<string, number>;
  total_ms: number;
  /**
   * Optional usability gate result (Aysenur's FaceUsabilityGate). Present
   * when the SpoofDetector is constructed with enableFaceUsabilityGate
   * (default true). Inspect `gate_result.usable` for the boolean verdict
   * and `gate_result.reason` for the blocking reason. The fusion pipeline
   * runs regardless — gates are advisory.
   */
  gate_result?: {
    usable: boolean;
    blocked: boolean;
    reason: string;
    state: string;
    occluded: boolean;
    qualityOk: boolean;
    occlusionScore: number;
    illuminationScore: number;
    occludedRegions: readonly string[];
    underexposedRegions: readonly string[];
    overexposedRegions: readonly string[];
  };
}
