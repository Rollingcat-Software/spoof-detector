// Pose3DConsistencyAnalyzer — TS-only addition (no Python source).
//
// MediaPipe FaceLandmarker emits a 4×4 facial transformation matrix
// (row-major, 16 floats) that describes the face's 3D rotation +
// translation in camera/world space. Real human faces project to a
// well-conditioned SE(3) transform; flat printed photos, AR-overlay
// stickers, and 2D screen replays often produce subtly degenerate
// fits because the underlying landmark geometry doesn't match a true
// 3D face. We probe that with two signals:
//
//   1. Rotation orthonormality residual. The upper-left 3×3 of a
//      valid SE(3) transform satisfies Rᵀ R ≈ I and det(R) ≈ 1. A
//      well-fit real face holds the Frobenius norm of (Rᵀ R − I)
//      under ~0.05. Flat objects or partial-face fits drift higher.
//
//   2. Z-translation variance over a rolling window. A real user
//      naturally drifts toward/away from the camera (mm-scale per
//      frame); a locked printed photo holds Z near constant.
//
// Both signals are normalized and blended:
//   score = clamp01(0.6 * (1 - ortho_residual/k1) + 0.4 * (z_var/k2)) * 100
//
// Per the proctoring profile, we explicitly do not require the user
// to perform — natural sitting motion produces enough Z variance to
// credit a real session above a static-photo baseline.

import {
  AnalyzerResult,
  FaceROI,
  IFaceAnalyzer,
  makeAnalyzerResult,
} from "../../domain/models";
import { RingBuffer } from "../../domain/session";

export interface Pose3DConsistencyAnalyzerOptions {
  /** Rolling window for Z-translation variance (90 = 3 s @ 30 fps). */
  historyLen?: number;
  /** Frames before scoring (else neutral 50). */
  warmupFrames?: number;
  /** Orthonormality residual at which the residual-derived score hits 0. */
  orthoResidualSaturation?: number;
  /** Z-translation stddev at which the z-motion score saturates. */
  zMotionSaturation?: number;
}

interface PoseState {
  zHistory: RingBuffer<number>;
  lastOrthoResidual: number;
}

export class Pose3DConsistencyAnalyzer implements IFaceAnalyzer {
  readonly name = "pose_3d_consistency";

  private readonly historyLen: number;
  private readonly warmupFrames: number;
  private readonly orthoResidualSaturation: number;
  private readonly zMotionSaturation: number;
  private states: Map<number, PoseState> = new Map();

  constructor(options: Pose3DConsistencyAnalyzerOptions = {}) {
    this.historyLen = options.historyLen ?? 90;
    this.warmupFrames = options.warmupFrames ?? 30;
    this.orthoResidualSaturation = options.orthoResidualSaturation ?? 0.15;
    this.zMotionSaturation = options.zMotionSaturation ?? 8.0;
  }

  analyze(_faceCrop: ImageData | null, face: FaceROI): AnalyzerResult {
    const start = performance.now();
    let state = this.states.get(face.face_id);
    if (!state) {
      state = {
        zHistory: new RingBuffer<number>(this.historyLen),
        lastOrthoResidual: 0,
      };
      this.states.set(face.face_id, state);
    }

    const m = face.transformMatrix;
    if (!m || m.length !== 16) {
      return makeAnalyzerResult(
        this.name,
        50,
        { error: "no_transform_matrix" },
        performance.now() - start,
      );
    }

    // Row-major 4×4. Upper-left 3×3 is the rotation; column 3 (indices
    // 3, 7, 11) is the translation; (15) is the homogeneous scale.
    //
    //   [ m00 m01 m02 | tx ]
    //   [ m10 m11 m12 | ty ]
    //   [ m20 m21 m22 | tz ]
    //   [  0   0   0  |  1 ]
    const r = [
      [m[0], m[1], m[2]],
      [m[4], m[5], m[6]],
      [m[8], m[9], m[10]],
    ];
    const tz = m[11];

    // Compute Rᵀ R and its Frobenius distance to the identity matrix.
    let orthoResidual = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += r[k][i] * r[k][j];
        const target = i === j ? 1 : 0;
        const d = s - target;
        orthoResidual += d * d;
      }
    }
    orthoResidual = Math.sqrt(orthoResidual);
    state.lastOrthoResidual = orthoResidual;
    state.zHistory.append(tz);

    if (state.zHistory.length < this.warmupFrames) {
      return makeAnalyzerResult(
        this.name,
        50,
        {
          warming: true,
          frames: state.zHistory.length,
          ortho_residual: round(orthoResidual, 4),
          tz: round(tz, 3),
        },
        performance.now() - start,
      );
    }

    const zs = state.zHistory.toArray();
    let mean = 0;
    for (const v of zs) mean += v;
    mean /= zs.length;
    let sse = 0;
    for (const v of zs) {
      const d = v - mean;
      sse += d * d;
    }
    const zStd = Math.sqrt(sse / zs.length);

    const orthoScore = Math.max(
      0,
      1 - orthoResidual / this.orthoResidualSaturation,
    );
    const zScore = Math.max(0, Math.min(1, zStd / this.zMotionSaturation));
    const score = (0.6 * orthoScore + 0.4 * zScore) * 100;

    return makeAnalyzerResult(
      this.name,
      score,
      {
        ortho_residual: round(orthoResidual, 4),
        ortho_score: round(orthoScore, 3),
        tz: round(tz, 3),
        tz_std: round(zStd, 3),
        z_score: round(zScore, 3),
        frames: zs.length,
      },
      performance.now() - start,
    );
  }

  reset(): void {
    this.states.clear();
  }
}

function round(x: number, digits: number): number {
  const k = Math.pow(10, digits);
  return Math.round(x * k) / k;
}
