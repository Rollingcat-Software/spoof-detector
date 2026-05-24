// LandmarkPlanarityAnalyzer tests.
//
// We validate the core geometric claim under an ORTHOGRAPHIC projector — the
// weak-perspective regime the analyzer is built for. In that regime:
//   * a flat plane (Z=0) rotating about the Y axis projects to landmark
//     motion that a single 2D affine map explains EXACTLY → residual 0 →
//     SPOOF-like → low score.
//   * a surface with out-of-plane depth (a nose-like bump) projects to motion
//     an affine map cannot capture → residual > 0 → live-like → high score.
// We also confirm that with no rotation the analyzer abstains (measured:false,
// neutral 50) so a frontal, still real face is never penalised.

import { describe, expect, it } from "vitest";
import { LandmarkPlanarityAnalyzer } from "../src/infrastructure/analyzers/LandmarkPlanarityAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

const CX = 320;
const CY = 240;

interface Point3D {
  x: number;
  y: number;
  z: number;
}

/** 7×7 grid of object points spanning ±90 in X/Y. `depthFn` sets Z. */
function grid(depthFn: (x: number, y: number) => number): Point3D[] {
  const pts: Point3D[] = [];
  for (let gy = -3; gy <= 3; gy++) {
    for (let gx = -3; gx <= 3; gx++) {
      const x = gx * 30;
      const y = gy * 30;
      pts.push({ x, y, z: depthFn(x, y) });
    }
  }
  return pts;
}

const FLAT = grid(() => 0);
// A central protrusion (nose-like) — strongly non-affine in (X,Y).
const BUMPY = grid((x, y) => 70 * Math.exp(-(x * x + y * y) / (2 * 45 * 45)));

/** Column-major MediaPipe-style 4×4 for a rotation of `deg` about the Y axis. */
function transformY(deg: number): Float32Array {
  const t = (deg * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  const m = new Float32Array(16);
  // col 0 = [R00,R10,R20,0]
  m[0] = c; m[1] = 0; m[2] = -s; m[3] = 0;
  // col 1 = [R01,R11,R21,0]
  m[4] = 0; m[5] = 1; m[6] = 0; m[7] = 0;
  // col 2 = [R02,R12,R22,0]
  m[8] = s; m[9] = 0; m[10] = c; m[11] = 0;
  // col 3 = translation
  m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;
  return m;
}

/** Orthographically project object points after a Y-axis rotation of `deg`. */
function projectFace(points: Point3D[], deg: number, face_id = 0): FaceROI {
  const t = (deg * Math.PI) / 180;
  const c = Math.cos(t);
  const s = Math.sin(t);
  const lm = new Float32Array(points.length * 2);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const u = p.x * c + p.z * s + CX; // X' = X cosθ + Z sinθ
    const v = p.y + CY;               // Y unchanged by Y-axis rotation
    lm[2 * i] = u;
    lm[2 * i + 1] = v;
    if (u < minX) minX = u;
    if (v < minY) minY = v;
    if (u > maxX) maxX = u;
    if (v > maxY) maxY = v;
  }
  return {
    face_id,
    bbox: new BBox(minX, minY, maxX, maxY),
    confidence: 0.95,
    landmarks: lm,
    transformMatrix: transformY(deg),
  };
}

/** Sweep rotation 0→maxDeg in 2° steps, return the final analyzer result. */
function sweep(
  a: LandmarkPlanarityAnalyzer,
  points: Point3D[],
  maxDeg: number,
) {
  let last = a.analyze(null, projectFace(points, 0));
  for (let deg = 2; deg <= maxDeg; deg += 2) {
    last = a.analyze(null, projectFace(points, deg));
  }
  return last;
}

describe("LandmarkPlanarityAnalyzer", () => {
  it("abstains (measured:false, neutral 50) before any rotation", () => {
    const a = new LandmarkPlanarityAnalyzer();
    const r = a.analyze(null, projectFace(FLAT, 0));
    expect(r.score).toBe(50);
    expect(r.details.measured).toBe(false);
  });

  it("abstains when rotation stays under the gate", () => {
    const a = new LandmarkPlanarityAnalyzer({ rotationGateDeg: 4 });
    let last = a.analyze(null, projectFace(FLAT, 0));
    last = a.analyze(null, projectFace(FLAT, 1));
    last = a.analyze(null, projectFace(FLAT, 2)); // max delta 2° < 4°
    expect(last.details.measured).toBe(false);
    expect(last.details.reason).toBe("insufficient_rotation");
  });

  it("flat plane under rotation → near-zero residual → low (spoof) score", () => {
    const a = new LandmarkPlanarityAnalyzer();
    const r = sweep(a, FLAT, 24);
    expect(r.details.measured).toBe(true);
    expect(r.details.residual_norm as number).toBeLessThan(0.002);
    expect(r.score).toBeLessThan(25);
  });

  it("3D depth under rotation → large residual → high (live) score", () => {
    const a = new LandmarkPlanarityAnalyzer();
    const r = sweep(a, BUMPY, 24);
    expect(r.details.measured).toBe(true);
    expect(r.details.residual_norm as number).toBeGreaterThan(0.02);
    expect(r.score).toBeGreaterThan(50);
  });

  it("separates flat from 3D by a wide margin under identical motion", () => {
    const flat = sweep(new LandmarkPlanarityAnalyzer(), FLAT, 24);
    const solid = sweep(new LandmarkPlanarityAnalyzer(), BUMPY, 24);
    expect((solid.details.residual_norm as number)).toBeGreaterThan(
      (flat.details.residual_norm as number) * 5,
    );
    expect(solid.score - flat.score).toBeGreaterThan(40);
  });

  it("abstains below the 15° rotation gate — small head turns are never judged (no false-reject)", () => {
    // The 2026-05-24 regression: a real face making ~8-10° turns scored as
    // "flat" because parallax residual is tiny at small angles. The gate now
    // abstains under 15° so genuine small movements can't trip the veto.
    const a = new LandmarkPlanarityAnalyzer();
    const r = sweep(a, BUMPY, 10); // only 10° of rotation
    expect(r.details.measured).toBe(false);
    expect(r.details.reason).toBe("insufficient_rotation");
  });

  it("3D depth reads live at small AND large turns once past the gate (rotation-invariant)", () => {
    const near = sweep(new LandmarkPlanarityAnalyzer(), BUMPY, 16);
    const far = sweep(new LandmarkPlanarityAnalyzer(), BUMPY, 30);
    expect(near.details.measured).toBe(true);
    expect(far.details.measured).toBe(true);
    expect(near.score).toBeGreaterThan(50);
    expect(far.score).toBeGreaterThan(50);
  });

  it("reset() clears per-face history", () => {
    const a = new LandmarkPlanarityAnalyzer();
    sweep(a, BUMPY, 24);
    a.reset();
    const r = a.analyze(null, projectFace(BUMPY, 0));
    expect(r.details.measured).toBe(false);
  });
});
