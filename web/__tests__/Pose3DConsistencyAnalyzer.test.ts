import { describe, expect, it } from "vitest";
import { Pose3DConsistencyAnalyzer } from "../src/infrastructure/analyzers/Pose3DConsistencyAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

// Row-major 4×4 identity.
const I4 = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);

function identityWithTz(tz: number): Float32Array {
  return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, tz, 0, 0, 0, 1]);
}

// A degenerate non-orthogonal rotation block (simulates a flat 2D fit).
function degenerateMatrix(tz: number): Float32Array {
  return new Float32Array([
    1.4, 0.3, 0, 0,
    0.3, 1.4, 0, 0,
    0, 0, 0.2, tz,
    0, 0, 0, 1,
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
});
