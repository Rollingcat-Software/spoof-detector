// LandmarkVarianceAnalyzer behavioural tests.
// Mirrors the spirit of tests/test_analyzers.py — verifies the
// piecewise score mapping at the two extremes:
//   * frozen landmarks → near-zero variance → very low score
//   * jittered landmarks → high variance → high score

import { describe, expect, it } from "vitest";
import { LandmarkVarianceAnalyzer } from "../src/infrastructure/analyzers/LandmarkVarianceAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

const NUM_LANDMARKS = 478;

function makeFace(landmarks: Float32Array, face_id = 0): FaceROI {
  return {
    face_id,
    bbox: new BBox(0, 0, 100, 100),
    confidence: 0.9,
    landmarks,
  };
}

function frozenLandmarks(seed = 1): Float32Array {
  // Deterministic grid — every frame is identical, ergo zero variance.
  const lm = new Float32Array(NUM_LANDMARKS * 2);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    lm[2 * i] = 50 + ((i * seed) % 7);
    lm[2 * i + 1] = 50 + ((i * seed) % 11);
  }
  return lm;
}

function jitteredLandmarks(rngSeed: number): Float32Array {
  // Pseudorandom jitter (~ ±5 px) per frame ⇒ high per-landmark variance.
  let s = rngSeed;
  const lm = new Float32Array(NUM_LANDMARKS * 2);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const jx = ((s >>> 16) & 0xffff) / 0xffff - 0.5;
    s = (s * 1664525 + 1013904223) >>> 0;
    const jy = ((s >>> 16) & 0xffff) / 0xffff - 0.5;
    lm[2 * i] = 50 + jx * 10;
    lm[2 * i + 1] = 50 + jy * 10;
  }
  return lm;
}

describe("LandmarkVarianceAnalyzer", () => {
  it("returns warmup score (50.0) before WARMUP_FRAMES", () => {
    const a = new LandmarkVarianceAnalyzer();
    const r = a.analyze(null, makeFace(frozenLandmarks()));
    expect(r.score).toBe(50.0);
    expect(r.details.warmup).toBe(true);
  });

  it("frozen landmarks → near-zero variance → score below 30", () => {
    const a = new LandmarkVarianceAnalyzer();
    let last = a.analyze(null, makeFace(frozenLandmarks()));
    for (let i = 0; i < 30; i++) {
      last = a.analyze(null, makeFace(frozenLandmarks()));
    }
    expect(last.details.warmup).toBeUndefined();
    expect(last.score).toBeLessThan(30);
  });

  it("jittered landmarks → high variance → score above 60", () => {
    const a = new LandmarkVarianceAnalyzer();
    let last = a.analyze(null, makeFace(jitteredLandmarks(1)));
    for (let i = 0; i < 30; i++) {
      last = a.analyze(null, makeFace(jitteredLandmarks(i + 2)));
    }
    expect(last.details.warmup).toBeUndefined();
    expect(last.score).toBeGreaterThan(60);
  });

  it("returns no_landmarks error when none provided", () => {
    const a = new LandmarkVarianceAnalyzer();
    const face: FaceROI = {
      face_id: 0,
      bbox: new BBox(0, 0, 100, 100),
      confidence: 0.9,
    };
    const r = a.analyze(null, face);
    expect(r.score).toBe(50.0);
    expect(r.details.error).toBe("no_landmarks");
  });
});
