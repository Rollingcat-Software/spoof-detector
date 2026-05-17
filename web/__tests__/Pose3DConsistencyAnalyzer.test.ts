import { describe, expect, it } from "vitest";
import { Pose3DConsistencyAnalyzer } from "../src/infrastructure/analyzers/Pose3DConsistencyAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

// MediaPipe uses COLUMN-MAJOR 4×4 matrices. Identity is identity in both
// row-major and column-major so I4 looks the same either way, but the
// translation column lives at indices 12, 13, 14.
const I4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function identityWithTz(tz: number): Float32Array {
  // Column-major: translation column is the LAST 4 floats.
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, tz, 1]);
}

// A degenerate non-orthogonal rotation block (simulates a flat 2D fit).
// In column-major the rotation block is m[0..2], m[4..6], m[8..10].
function degenerateMatrix(tz: number): Float32Array {
  return new Float32Array([
    1.4, 0.3, 0, 0,    // col 0
    0.3, 1.4, 0, 0,    // col 1
    0, 0, 0.2, 0,      // col 2
    0, 0, tz, 1,       // col 3 (translation)
  ]);
}

function makeFace(m: Float32Array | null, face_id = 0): FaceROI {
  return {
    face_id,
    bbox: new BBox(0, 0, 100, 100),
    confidence: 0.99,
    transformMatrix: m ?? undefined,
  };
}

describe("Pose3DConsistencyAnalyzer", () => {
  it("returns neutral 50 with no transform matrix", () => {
    const a = new Pose3DConsistencyAnalyzer();
    const r = a.analyze(null, makeFace(null));
    expect(r.score).toBe(50);
    expect(r.details.error).toBe("no_transform_matrix");
  });

  it("returns neutral 50 during warmup", () => {
    const a = new Pose3DConsistencyAnalyzer({ warmupFrames: 30 });
    for (let i = 0; i < 10; i++) {
      const r = a.analyze(null, makeFace(identityWithTz(0.5)));
      expect(r.score).toBe(50);
    }
  });

  it("scores high on a well-formed orthonormal pose with natural Z drift", () => {
    const a = new Pose3DConsistencyAnalyzer({
      warmupFrames: 30,
      historyLen: 90,
    });
    // Identity rotation + Z that drifts ±2 over the window.
    for (let i = 0; i < 90; i++) {
      const tz = 5 + 2 * Math.sin((i / 90) * Math.PI * 2);
      a.analyze(null, makeFace(identityWithTz(tz)));
    }
    const r = a.analyze(null, makeFace(identityWithTz(5)));
    expect(r.score).toBeGreaterThan(60);
    expect(r.details.ortho_score).toBeGreaterThan(0.9);
  });

  it("scores low on degenerate (non-orthogonal) rotation block (flat photo)", () => {
    const a = new Pose3DConsistencyAnalyzer({
      warmupFrames: 30,
      historyLen: 90,
    });
    for (let i = 0; i < 90; i++) {
      a.analyze(null, makeFace(degenerateMatrix(5)));
    }
    const r = a.analyze(null, makeFace(degenerateMatrix(5)));
    // Both signals fail: ortho residual large, Z static.
    expect(r.score).toBeLessThan(20);
  });

  it("scores lower on a well-formed pose with NO Z motion (locked photo)", () => {
    const a = new Pose3DConsistencyAnalyzer({
      warmupFrames: 30,
      historyLen: 90,
    });
    for (let i = 0; i < 90; i++) {
      a.analyze(null, makeFace(I4));
    }
    const r = a.analyze(null, makeFace(I4));
    // Orthonormal (good) but no Z motion (bad) → mid-low score.
    expect(r.score).toBeLessThan(70);
    expect(r.details.tz_std).toBe(0);
  });

  it("regression: reads tz from column-major index 14, not row-major 11", () => {
    // SE(3) identity with tx=10, ty=20, tz=500 (mm). In column-major
    // the translation column is the LAST 4 floats. A row-major reader
    // would see tz=0 here because row-major-index 11 falls in the
    // bottom row (always 0 for SE(3)).
    const mat = new Float32Array([
      1, 0, 0, 0, // col 0 (rotation)
      0, 1, 0, 0, // col 1
      0, 0, 1, 0, // col 2
      10, 20, 500, 1, // col 3 (tx, ty, tz, 1)
    ]);
    const a = new Pose3DConsistencyAnalyzer({
      warmupFrames: 1,
      historyLen: 2,
    });
    a.analyze(null, makeFace(mat));
    const r = a.analyze(null, makeFace(mat));
    expect(r.details.tz).toBe(500);
  });

  it("regression: rotation block read from column-major preserves orthonormality detection", () => {
    // 90° rotation around Y in column-major:
    //   col 0: [cos θ, 0, -sin θ, 0]  → [0, 0, -1, 0]
    //   col 1: [0,     1,  0,    0]
    //   col 2: [sin θ, 0,  cos θ, 0] → [1, 0,  0, 0]
    //   col 3: [0, 0, 0, 1]
    const c = Math.cos(Math.PI / 2);
    const s = Math.sin(Math.PI / 2);
    const mat = new Float32Array([
      c, 0, -s, 0,
      0, 1, 0, 0,
      s, 0, c, 0,
      0, 0, 0, 1,
    ]);
    const a = new Pose3DConsistencyAnalyzer({
      warmupFrames: 1,
      historyLen: 2,
    });
    a.analyze(null, makeFace(mat));
    const r = a.analyze(null, makeFace(mat));
    expect(r.details.ortho_residual as number).toBeLessThan(0.01);
  });
});
