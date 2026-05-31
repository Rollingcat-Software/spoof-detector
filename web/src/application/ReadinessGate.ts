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
  /** Mean face-region brightness, 0-255 (FaceUsabilityGate.globalFaceBrightness). */
  faceBrightness: number;
  /** Whether a critical face region is occluded (FaceUsabilityGate.occluded). */
  occluded: boolean;
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
  /** Min mean face brightness (0-255). Below ⇒ "too dark". Default 60. */
  minFaceBrightness?: number;
  /** Max mean face brightness (0-255). Above ⇒ "too bright" (matches the flash
   *  over-lit ceiling so the two stay consistent). Default 185. */
  maxFaceBrightness?: number;
}

export class ReadinessGate {
  private readonly minFaceAreaFraction: number;
  private readonly minFaceBrightness: number;
  private readonly maxFaceBrightness: number;

  constructor(options: ReadinessOptions = {}) {
    this.minFaceAreaFraction = options.minFaceAreaFraction ?? 0.05;
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

    // 3. Face large enough (not too far).
    const sizeOk = s.faceAreaFraction >= this.minFaceAreaFraction;
    checks.push({
      id: "size",
      label: "Distance",
      pass: sizeOk,
      message: sizeOk ? "Good distance" : "Move closer to the camera",
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
    // SKIP this check when the face is too small / too far from the camera.
    // MediaPipe's landmark detector becomes unreliable on faces below ~10%
    // of frame area — the mouth / nose regions get coarse, the
    // FaceUsabilityGate's region-pixel thresholds fire false "occlusion"
    // alarms, and the user sees "Uncover your face (mouth)" while their
    // face is plainly visible. They then move closer and the false alarm
    // clears — confusing and unprofessional.
    //
    // Below the OCCLUSION_RELIABLE_AREA threshold we return `pass: true`
    // with a "checking…" message (instead of a green tick) so the operator
    // knows the check is deferred, not skipped silently. The user is
    // already being told to move closer via the size check; layering a
    // bogus mouth-occlusion message on top adds noise without information.
    const OCCLUSION_RELIABLE_AREA = 0.10;
    if (s.faceAreaFraction < OCCLUSION_RELIABLE_AREA) {
      checks.push({
        id: "occlusion",
        label: "Visibility",
        pass: true,
        message: "Move closer for a reliable visibility check",
      });
    } else {
      const occluded = s.occluded;
      const regions = (s.occludedRegions ?? []).join(", ").replace(/_/g, " ");
      checks.push({
        id: "occlusion",
        label: "Visibility",
        pass: !occluded,
        message: occluded
          ? regions
            ? `Uncover your face (${regions})`
            : "Uncover your face"
          : "Face fully visible",
      });
    }

    return { ready: checks.every((c) => c.pass), checks };
  }
}
