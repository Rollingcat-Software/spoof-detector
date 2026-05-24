// LandmarkPlanarityAnalyzer — TS-only addition (no Python source).
//
// Catches the attack the rest of the pipeline misses: a hand-held printed
// photo (or any flat surface) presented to a camera that focuses on it
// sharply. MiniFASNet is camera-quality dependent — a crisp, frame-filling
// print scores ~100 ("real") on a PC webcam even though it is a flat object.
// See the 2026-05-24 amispoof benchmark: print → LIVE 90%, MiniFASNet 100.
//
// The discriminator here is GEOMETRIC and camera-focus-independent: a flat
// surface and a real 3D face move differently under head/hand rotation.
//
//   * A flat plane rotating in front of the camera produces landmark motion
//     that a single 2D affine map explains almost perfectly (weak-perspective
//     planar motion) → tiny reprojection residual.
//   * A real face has out-of-plane structure (the nose protrudes, the eyes
//     recede). Under rotation that depth produces motion PARALLAX which no
//     single affine map can capture → large reprojection residual.
//
// So we fit the best affine transform between a rotated reference frame's
// landmarks and the current frame's, and read the residual:
//
//   residual_norm small  → planar → SPOOF-like → low score
//   residual_norm large  → 3D     → live-like  → high score
//
// We ONLY score when the head/print has actually rotated (parallax is only
// observable under rotation). With no rotation we return a neutral 50 and
// flag `measured: false`, so a frontal, still real face is never penalised —
// the still-photo case is already covered by the no-blink / static-motion
// incident detectors. This keeps the false-reject rate untouched while
// closing the moving-print false-accept hole.
//
// Weak-perspective (affine) is the model here rather than a full projective
// homography: it is closed-form, well-conditioned, and accurate for a face
// at typical webcam distance. A homography upgrade is a future option if
// wide-FOV / close-range capture shows affine perspective leakage.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";

export interface LandmarkPlanarityAnalyzerOptions {
  /** Per-face frame history depth (60 = 2 s @ 30 fps). */
  historyLen?: number;
  /** Minimum head-rotation (deg) between reference and current to score. */
  rotationGateDeg?: number;
  /**
   * Fallback motion gate (mean landmark displacement / face scale) used when
   * no facial-transform matrix is available to measure rotation directly.
   */
  motionGateNorm?: number;
  /** Normalised residual at which the score saturates to 100 (fully 3D). */
  residualSaturation?: number;
  /**
   * Rotation-INVARIANT calibration knob: the depth measure (residual divided
   * by sin(rotation)) at which the score saturates to 100. Parallax residual
   * scales with rotation angle, so a raw-residual threshold false-rejects a
   * real face making only small head turns (2026-05-24 regression). Dividing
   * by sin(rotation) yields a rotation-invariant depth proxy.
   */
  depthSaturation?: number;
  /** Landmarks beyond this index are ignored (drops the 10 iris points). */
  maxLandmarks?: number;
}

interface FrameSample {
  /** Flat [x0,y0,x1,y1,…] landmark pixel coordinates. */
  lm: Float32Array;
  /** Upper-left 3×3 rotation, or null when no transform matrix was present. */
  R: number[][] | null;
  /** Face scale (sqrt of bbox area) in pixels. */
  scale: number;
}

const NEUTRAL = 50;
// Floor for sin(rotation) when rotation-normalising — guards the divide from
// blowing up at tiny angles (~sin 8°). Rotation below the gate doesn't score.
const MIN_SIN = 0.139;

export class LandmarkPlanarityAnalyzer implements IFaceAnalyzer {
  readonly name = "planarity";

  private readonly historyLen: number;
  private readonly rotationGateDeg: number;
  private readonly motionGateNorm: number;
  private readonly residualSaturation: number;
  private readonly depthSaturation: number;
  private readonly maxLandmarks: number;
  private readonly histories: Map<number, FrameSample[]> = new Map();

  constructor(options: LandmarkPlanarityAnalyzerOptions = {}) {
    this.historyLen = options.historyLen ?? 60;
    this.rotationGateDeg = options.rotationGateDeg ?? 15.0;
    this.motionGateNorm = options.motionGateNorm ?? 0.04;
    this.residualSaturation = options.residualSaturation ?? 0.02;
    this.depthSaturation = options.depthSaturation ?? 0.045;
    this.maxLandmarks = options.maxLandmarks ?? 468;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();

    const lm = face.landmarks;
    const scale = Math.sqrt(Math.max(1, face.bbox.area));
    if (!lm || lm.length < 8) {
      return makeAnalyzerResult(
        this.name,
        NEUTRAL,
        { measured: false, reason: "no_landmarks" },
        performance.now() - start,
      );
    }

    const R = rotationFromColumnMajor(face.transformMatrix);
    const cur: FrameSample = { lm, R, scale };

    let hist = this.histories.get(face.face_id);
    if (!hist) {
      hist = [];
      this.histories.set(face.face_id, hist);
    }

    // Pick the historical reference with the largest qualifying motion vs the
    // current frame: rotation delta when a transform matrix is available,
    // otherwise normalised landmark displacement. Searching for the MAX
    // (not the first) maximises observable parallax for the residual test.
    let ref: FrameSample | null = null;
    let bestRotDeg = 0;
    let bestMotion = 0;
    let refLag = 0;
    for (let i = 0; i < hist.length; i++) {
      const cand = hist[i];
      if (cand.lm.length !== lm.length) continue;
      if (R && cand.R) {
        const deg = rotationAngleDeg(cand.R, R);
        if (deg > bestRotDeg) {
          bestRotDeg = deg;
          ref = cand;
          refLag = hist.length - i;
        }
      } else {
        const m = meanDisplacementNorm(cand.lm, lm, scale, this.maxLandmarks);
        if (m > bestMotion) {
          bestMotion = m;
          ref = cand;
          refLag = hist.length - i;
        }
      }
    }

    // Push current AFTER the reference search so we never compare a frame to
    // itself, then cap the ring.
    hist.push(cur);
    if (hist.length > this.historyLen) hist.shift();

    const haveRotation = R !== null && ref !== null && ref.R !== null;
    const gatePassed = haveRotation
      ? bestRotDeg >= this.rotationGateDeg
      : ref !== null && bestMotion >= this.motionGateNorm;

    if (!ref || !gatePassed) {
      return makeAnalyzerResult(
        this.name,
        NEUTRAL,
        {
          measured: false,
          reason: haveRotation ? "insufficient_rotation" : "insufficient_motion",
          rot_deg: round(bestRotDeg, 2),
          motion_norm: round(bestMotion, 4),
          frames: hist.length,
        },
        performance.now() - start,
      );
    }

    const residualNorm = affineResidualNorm(
      ref.lm,
      lm,
      scale,
      this.maxLandmarks,
    );
    if (residualNorm === null) {
      return makeAnalyzerResult(
        this.name,
        NEUTRAL,
        { measured: false, reason: "degenerate_fit", frames: hist.length },
        performance.now() - start,
      );
    }

    // Rotation-INVARIANT depth measure. Parallax residual scales with the
    // rotation angle, so a real 3D face at a small head-turn yields a small
    // residual a raw threshold misreads as "flat". Dividing by sin(rotation)
    // recovers a depth proxy that stays high for a genuine face at ANY turn
    // and low for a flat surface. When rotation is available (transform
    // matrix) we use it; the motion-gate fallback keeps the raw residual.
    let depthMeasure: number;
    let saturation: number;
    if (haveRotation && bestRotDeg > 0) {
      const sinRot = Math.sin((bestRotDeg * Math.PI) / 180);
      depthMeasure = residualNorm / Math.max(sinRot, MIN_SIN);
      saturation = this.depthSaturation;
    } else {
      depthMeasure = residualNorm;
      saturation = this.residualSaturation;
    }
    // depthMeasure small → flat (spoof) → 0; >= saturation → 3D (live) → 100.
    const score = clamp01(depthMeasure / saturation) * 100;

    return makeAnalyzerResult(
      this.name,
      score,
      {
        measured: true,
        residual_norm: round(residualNorm, 4),
        depth_measure: round(depthMeasure, 4),
        rotation_normalized: haveRotation,
        rot_deg: round(bestRotDeg, 2),
        motion_norm: round(bestMotion, 4),
        ref_lag_frames: refLag,
        frames: hist.length,
      },
      performance.now() - start,
    );
  }

  reset(): void {
    this.histories.clear();
  }
}

/**
 * Best-fit affine reprojection residual between two landmark sets, RMS over
 * points and normalised by face scale. Returns null on a degenerate fit
 * (collinear / coincident source points).
 */
function affineResidualNorm(
  src: Float32Array,
  dst: Float32Array,
  scale: number,
  maxLandmarks: number,
): number | null {
  const n = Math.min(src.length, dst.length) >> 1;
  const count = Math.min(n, maxLandmarks);
  if (count < 3) return null;

  // Normal-equation accumulators for the shared 3×3 design matrix.
  let Sxx = 0, Sxy = 0, Sx = 0, Syy = 0, Sy = 0;
  // RHS for x' (a,b,c) and y' (d,e,f).
  let bxX = 0, bxY = 0, bxC = 0;
  let byX = 0, byY = 0, byC = 0;
  for (let i = 0; i < count; i++) {
    const x = src[2 * i];
    const y = src[2 * i + 1];
    const xp = dst[2 * i];
    const yp = dst[2 * i + 1];
    Sxx += x * x;
    Sxy += x * y;
    Sx += x;
    Syy += y * y;
    Sy += y;
    bxX += x * xp;
    bxY += y * xp;
    bxC += xp;
    byX += x * yp;
    byY += y * yp;
    byC += yp;
  }

  const m: Matrix3 = [
    [Sxx, Sxy, Sx],
    [Sxy, Syy, Sy],
    [Sx, Sy, count],
  ];
  const ax = solve3x3(m, [bxX, bxY, bxC]);
  const ay = solve3x3(m, [byX, byY, byC]);
  if (!ax || !ay) return null;

  let sse = 0;
  for (let i = 0; i < count; i++) {
    const x = src[2 * i];
    const y = src[2 * i + 1];
    const predX = ax[0] * x + ax[1] * y + ax[2];
    const predY = ay[0] * x + ay[1] * y + ay[2];
    const dx = predX - dst[2 * i];
    const dy = predY - dst[2 * i + 1];
    sse += dx * dx + dy * dy;
  }
  const rms = Math.sqrt(sse / count);
  return rms / Math.max(scale, 1);
}

/** Mean per-landmark displacement between two sets, normalised by face scale. */
function meanDisplacementNorm(
  a: Float32Array,
  b: Float32Array,
  scale: number,
  maxLandmarks: number,
): number {
  const n = Math.min(a.length, b.length) >> 1;
  const count = Math.min(n, maxLandmarks);
  if (count === 0) return 0;
  let s = 0;
  for (let i = 0; i < count; i++) {
    const dx = a[2 * i] - b[2 * i];
    const dy = a[2 * i + 1] - b[2 * i + 1];
    s += Math.hypot(dx, dy);
  }
  return s / count / Math.max(scale, 1);
}

type Matrix3 = [
  [number, number, number],
  [number, number, number],
  [number, number, number],
];

/** Solve a 3×3 linear system by Cramer's rule. Returns null if near-singular. */
function solve3x3(m: Matrix3, b: [number, number, number]): [number, number, number] | null {
  const det = det3(m);
  if (Math.abs(det) < 1e-9) return null;
  const mx: Matrix3 = [
    [b[0], m[0][1], m[0][2]],
    [b[1], m[1][1], m[1][2]],
    [b[2], m[2][1], m[2][2]],
  ];
  const my: Matrix3 = [
    [m[0][0], b[0], m[0][2]],
    [m[1][0], b[1], m[1][2]],
    [m[2][0], b[2], m[2][2]],
  ];
  const mz: Matrix3 = [
    [m[0][0], m[0][1], b[0]],
    [m[1][0], m[1][1], b[1]],
    [m[2][0], m[2][1], b[2]],
  ];
  return [det3(mx) / det, det3(my) / det, det3(mz) / det];
}

function det3(m: Matrix3): number {
  return (
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
  );
}

/**
 * Extract the upper-left 3×3 rotation from MediaPipe's COLUMN-MAJOR 4×4
 * facial transformation matrix (same layout note as Pose3DConsistencyAnalyzer:
 * r[i][j] = m[j*4 + i]). Returns null when no matrix is present.
 */
function rotationFromColumnMajor(m: Float32Array | undefined): number[][] | null {
  if (!m || m.length !== 16) return null;
  return [
    [m[0], m[4], m[8]],
    [m[1], m[5], m[9]],
    [m[2], m[6], m[10]],
  ];
}

/** Geodesic angle (deg) between two rotation matrices: angle of R2·R1ᵀ. */
function rotationAngleDeg(r1: number[][], r2: number[][]): number {
  // trace(R2 · R1ᵀ) = Σ_i Σ_k r2[i][k] * r1[i][k]
  let tr = 0;
  for (let i = 0; i < 3; i++) {
    for (let k = 0; k < 3; k++) tr += r2[i][k] * r1[i][k];
  }
  const c = Math.max(-1, Math.min(1, (tr - 1) / 2));
  return (Math.acos(c) * 180) / Math.PI;
}

function clamp01(x: number): number {
  return Math.max(0, Math.min(1, x));
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
