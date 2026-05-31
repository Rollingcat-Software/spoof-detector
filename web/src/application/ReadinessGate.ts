// ReadinessGate — pre-flight capture-quality precondition (PROCTORING).
//
// A pure, per-frame evaluator that decides whether a session is allowed to
// START. It does NOT issue a liveness/spoof verdict; it answers "is the capture
// good enough to judge at all?". This implements the abstain-first stance: most
// bad captures (too dark, over-lit, occluded, no/multiple faces, dead camera)
// become a "fix this" instruction BEFORE any verdict, instead of a false SPOOF.
//
// Motivation (live findings, 2026-05-25):
//   * pitch-dark room → 0 faces, yet the engine asserted SPOOF(static_image) and
//     flooded "face missing" incidents — it should have refused to start.
//   * bright room → face near sensor saturation; the active flash probe abstains.
//   * the usable capture window is dim-to-MODERATE; nothing enforced it.
//
// Signals are read from what the SDK already produces per frame
// (`analysis.faces` for count + bbox area, `analysis.gate_result` for
// brightness + occlusion from the FaceUsabilityGate), so this is a thin,
// stateless function over existing data. Stability (requiring N consecutive
// green frames before enabling "Begin") is the caller's concern.

export interface ReadinessSignals {
  /** Number of faces detected this frame. */
  faceCount: number;
  /** Detected face bbox area as a fraction of the frame (0-1). Ignored unless faceCount === 1. */
  faceAreaFraction: number;
  /** Detected face bbox area in absolute pixels (width*height). Preferred over
   *  fraction when supplied: at 480p, 5 % fraction = 14 000 pixels (analyzable);
   *  at 1080p, 5 % = 100 000 pixels — same fraction, vastly different analysis
   *  fidelity. The pixel floor (default 10 000 ≈ 100×100) reflects what the
   *  texture / moire / MiniFASNet analyzers actually need to produce a stable
   *  reading. When undefined, falls back to faceAreaFraction. */
  faceAreaPixels?: number;
  /** Mean face-region brightness, 0-255 (FaceUsabilityGate.globalFaceBrightness). */
  faceBrightness: number;
  /** Whether a critical face region is occluded (FaceUsabilityGate.occluded). */
  occluded: boolean;
  /** Continuous occlusion score 0-1 from FaceUsabilityGate, when available.
   *  Preferred over the boolean — the readiness gate uses a HIGH threshold
   *  (0.85) so only severe occlusion blocks session start. The boolean
   *  fires at ~0.5 and produces too many false alarms on visible faces at
   *  awkward distance / pose. */
  occlusionScore?: number;
  /** Occluded region names, for the fix-it message. */
  occludedRegions?: readonly string[];
  /** Camera stream live AND frames advancing (caller-supplied). */
  cameraResponsive: boolean;
}

export type ReadinessCheckId = "camera" | "face" | "size" | "lighting" | "occlusion";

export interface ReadinessCheck {
  id: ReadinessCheckId;
  label: string;
  pass: boolean;
  /** Fix-it text when failing; a short OK note when passing. */
  message: string;
}

export interface ReadinessResult {
  /** True iff every check passes — the only state in which a session may start. */
  ready: boolean;
  checks: ReadinessCheck[];
}

export interface ReadinessOptions {
  /** Min face bbox area / frame area. Below ⇒ "move closer". Default 0.05. */
  minFaceAreaFraction?: number;
  /** Min face bbox area in absolute pixels (width*height). Default 10 000
   *  (≈ 100×100 — the floor below which texture / MiniFASNet readings become
   *  noisy regardless of frame size). When the caller supplies `faceAreaPixels`
   *  in the signals, BOTH this and `minFaceAreaFraction` must pass. */
  minFaceAreaPixels?: number;
  /** Min mean face brightness (0-255). Below ⇒ "too dark". Default 60. */
  minFaceBrightness?: number;
  /** Max mean face brightness (0-255). Above ⇒ "too bright" (matches the flash
   *  over-lit ceiling so the two stay consistent). Default 185. */
  maxFaceBrightness?: number;
}

export class ReadinessGate {
  private readonly minFaceAreaFraction: number;
  private readonly minFaceAreaPixels: number;
  private readonly minFaceBrightness: number;
  private readonly maxFaceBrightness: number;

  constructor(options: ReadinessOptions = {}) {
    this.minFaceAreaFraction = options.minFaceAreaFraction ?? 0.05;
    this.minFaceAreaPixels = options.minFaceAreaPixels ?? 10000;
    this.minFaceBrightness = options.minFaceBrightness ?? 60;
    this.maxFaceBrightness = options.maxFaceBrightness ?? 185;
  }

  evaluate(s: ReadinessSignals): ReadinessResult {
    const checks: ReadinessCheck[] = [];

    // 1. Camera responsive.
    checks.push({
      id: "camera",
      label: "Camera",
      pass: s.cameraResponsive,
      message: s.cameraResponsive ? "Camera live" : "Camera not responding",
    });

    // 2. Exactly one face.
    const faceOk = s.faceCount === 1;
    checks.push({
      id: "face",
      label: "Face",
      pass: faceOk,
      message:
        s.faceCount === 1
          ? "One face detected"
          : s.faceCount === 0
            ? "No face — center your face (check your lighting)"
            : "Multiple faces — only one person in frame",
    });

    // The remaining checks need a single, detected face to be meaningful.
    if (!faceOk) {
      for (const [id, label] of [
        ["size", "Distance"],
        ["lighting", "Lighting"],
        ["occlusion", "Visibility"],
      ] as [ReadinessCheckId, string][]) {
        checks.push({ id, label, pass: false, message: "Waiting for a single face" });
      }
      return { ready: false, checks };
    }

    // 3. Face large enough (not too far). Combined fraction + pixel test:
    // fraction guards the user's framing intuition ("face takes a reasonable
    // share of the frame"), pixels guard analyzer fidelity ("texture analyzer
    // actually has enough pixels to score from"). The pixel floor is the
    // honest constraint for our analyzers; the fraction is the human-readable
    // one. BOTH must pass when pixels are supplied.
    const fractionOk = s.faceAreaFraction >= this.minFaceAreaFraction;
    const pixelsOk =
      s.faceAreaPixels === undefined || s.faceAreaPixels >= this.minFaceAreaPixels;
    const sizeOk = fractionOk && pixelsOk;
    checks.push({
      id: "size",
      label: "Distance",
      pass: sizeOk,
      message: sizeOk
        ? "Good distance"
        : !fractionOk
          ? "Move closer to the camera"
          : `Move closer — face is too small in pixels (${s.faceAreaPixels} < ${this.minFaceAreaPixels})`,
    });

    // 4. Lighting in the dim-to-moderate band.
    const tooDark = s.faceBrightness < this.minFaceBrightness;
    const tooBright = s.faceBrightness > this.maxFaceBrightness;
    const lightingOk = !tooDark && !tooBright;
    checks.push({
      id: "lighting",
      label: "Lighting",
      pass: lightingOk,
      message: tooDark
        ? "Too dark — add light so your face is clearly visible"
        : tooBright
          ? "Too bright — reduce or redirect light off your face"
          : "Lighting OK",
    });

    // 5. Face not occluded.
    //
    // Decision history:
    //   V1 — use FaceUsabilityGate's BOOLEAN `occluded`. False-positive on a
    //        clearly visible face when the user is too far (small landmarks
    //        → coarse region pixels → mis-read as occlusion). User saw
    //        "Uncover your face (mouth)" while their face was plainly visible.
    //   V2 — defer the check when faceAreaFraction < 0.10. Helped at distance
    //        but a 30 cm capture (face IS big, score = 0.51) still fired
    //        because the boolean is calibrated for the Python pipeline, not
    //        browser landmarks at all distances.
    //   V3 (this) — prefer the continuous `occlusionScore`. Only fire the
    //        block when it's CONFIDENTLY high (>= 0.85). The boolean trips
    //        around 0.5; that's too aggressive to gate session start. The
    //        in-session capture-quality floor still handles mid-range
    //        occlusion (it routes to quality_uncertain), so a session that
    //        squeaks past readiness with score 0.4-0.85 isn't unsafe — it
    //        just gets UNCERTAIN if quality stays poor, never a false LIVE.
    const OCCLUSION_BLOCK_THRESHOLD = 0.85;
    const occludedConfident =
      typeof s.occlusionScore === "number"
        ? s.occlusionScore >= OCCLUSION_BLOCK_THRESHOLD
        : s.occluded; // fallback when score isn't supplied (legacy callers)
    const regions = (s.occludedRegions ?? []).join(", ").replace(/_/g, " ");
    checks.push({
      id: "occlusion",
      label: "Visibility",
      pass: !occludedConfident,
      message: occludedConfident
        ? regions
          ? `Uncover your face (${regions})`
          : "Uncover your face"
        : typeof s.occlusionScore === "number" && s.occlusionScore >= 0.5
          ? "Face mostly visible (continue)"
          : "Face fully visible",
    });

    return { ready: checks.every((c) => c.pass), checks };
  }
}
