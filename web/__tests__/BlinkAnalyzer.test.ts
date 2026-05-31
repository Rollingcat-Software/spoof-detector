// BlinkAnalyzer tests.
// Verifies:
//   * EAR computation drops from open (~0.30) to closed (~0.10) over a
//     scripted 100-frame sequence with 3 simulated blinks, and the
//     analyzer counts ≥1 blink.
//   * No-blink sequences score low after warmup.

import { describe, expect, it } from "vitest";
import { BlinkAnalyzer } from "../src/infrastructure/analyzers/BlinkAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

const NUM_LANDMARKS = 478;

// MediaPipe FaceMesh indices used by BlinkAnalyzer:
//   RIGHT_EYE = [33, 160, 158, 133, 153, 144]
//   LEFT_EYE  = [362, 385, 387, 263, 373, 380]

/**
 * Build a landmarks Float32Array where the eye 6-pt cluster has the
 * given vertical opening. `vOffset` controls the distance between
 * upper and lower lid landmarks. open ≈ 8 px, closed ≈ 1 px.
 */
function makeLandmarks(vOffset: number): Float32Array {
  const lm = new Float32Array(NUM_LANDMARKS * 2);
  // Default fill — irrelevant for EAR.
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    lm[2 * i] = 50;
    lm[2 * i + 1] = 50;
  }
  // RIGHT_EYE indices: 33, 160, 158, 133, 153, 144
  // p1=33 (left corner), p4=133 (right corner)  → horizontal 20 px.
  // p2=160, p6=144 verticals.
  // p3=158, p5=153 verticals.
  setXY(lm, 33, 100, 200);
  setXY(lm, 133, 120, 200);
  setXY(lm, 160, 105, 200 - vOffset);
  setXY(lm, 144, 105, 200 + vOffset);
  setXY(lm, 158, 115, 200 - vOffset);
  setXY(lm, 153, 115, 200 + vOffset);

  // LEFT_EYE indices: 362, 385, 387, 263, 373, 380
  setXY(lm, 362, 200, 200);
  setXY(lm, 263, 220, 200);
  setXY(lm, 385, 205, 200 - vOffset);
  setXY(lm, 380, 205, 200 + vOffset);
  setXY(lm, 387, 215, 200 - vOffset);
  setXY(lm, 373, 215, 200 + vOffset);
  return lm;
}

function setXY(lm: Float32Array, idx: number, x: number, y: number): void {
  lm[2 * idx] = x;
  lm[2 * idx + 1] = y;
}

function makeFace(lm: Float32Array): FaceROI {
  return {
    face_id: 0,
    bbox: new BBox(0, 0, 300, 300),
    confidence: 0.9,
    landmarks: lm,
  };
}

describe("BlinkAnalyzer", () => {
  it("detects an open eye → high EAR, no blink", () => {
    const a = new BlinkAnalyzer();
    const r = a.analyze(null, makeFace(makeLandmarks(3))); // open
    expect(r.details.ear).toBeGreaterThan(0.21);
  });

  it("counts blinks across a scripted open/close/open sequence", () => {
    const a = new BlinkAnalyzer();
    // 30 frames open, 4 frames closed, 10 frames open ... ×3 simulated blinks.
    // We use vOffset=4 for open (EAR ≈ 0.40) and 0.5 for closed (EAR ≈ 0.05).
    const open = makeLandmarks(4);
    const closed = makeLandmarks(0.5);
    for (let i = 0; i < 30; i++) a.analyze(null, makeFace(open));
    for (let i = 0; i < 4; i++) a.analyze(null, makeFace(closed));
    for (let i = 0; i < 10; i++) a.analyze(null, makeFace(open));
    for (let i = 0; i < 4; i++) a.analyze(null, makeFace(closed));
    for (let i = 0; i < 10; i++) a.analyze(null, makeFace(open));
    for (let i = 0; i < 4; i++) a.analyze(null, makeFace(closed));
    const last = a.analyze(null, makeFace(open));
    expect(last.details.blinks as number).toBeGreaterThanOrEqual(1);
  });

  it("warmup returns 50 below WARMUP_FRAMES", () => {
    const a = new BlinkAnalyzer();
    const r = a.analyze(null, makeFace(makeLandmarks(4)));
    expect(r.score).toBe(50.0);
    expect(r.details.warmup).toBe(true);
  });

  it("rejects a shallow EAR dip (perspective fake-blink) but counts a true closure", () => {
    const a = new BlinkAnalyzer();
    const open = makeLandmarks(4); // EAR ≈ 0.40
    // EAR = vOffset/10, so vOffset 1.8 → EAR 0.18: below the 0.20 closing
    // threshold (so it enters a closure) but ABOVE the 0.16 true-closed gate —
    // exactly the foreshortening a tilted flat photo produces. Must NOT count.
    const shallow = makeLandmarks(1.8);
    for (let i = 0; i < 30; i++) a.analyze(null, makeFace(open));
    for (let i = 0; i < 4; i++) a.analyze(null, makeFace(shallow));
    let r = a.analyze(null, makeFace(open));
    expect(r.details.blinks).toBe(0);

    // A genuine closure (EAR ≈ 0.05) clears the depth gate and counts.
    const deep = makeLandmarks(0.5);
    for (let i = 0; i < 6; i++) a.analyze(null, makeFace(open));
    for (let i = 0; i < 4; i++) a.analyze(null, makeFace(deep));
    r = a.analyze(null, makeFace(open));
    expect(r.details.blinks as number).toBeGreaterThanOrEqual(1);
  });

  it("no blinks for >30s → low score (re-calibrated for low-fps browser)", () => {
    // Pin fps so the synthetic-loop frames produce a real duration_sec.
    // Score-ramp re-calibrated 2026-05-31 for low-fps browser capture:
    // 5 s was too aggressive — at 6-9 fps a real passive user can easily
    // have blinks missed between sample frames at the 5 s mark, and
    // scoring them 10 (looks like a photo) is a false-positive against
    // a real face. New thresholds:
    //   < 15 s with 0 blinks  → 50 (no evidence yet)
    //   15-30 s with 0 blinks → 25
    //   > 30 s with 0 blinks  → 10
    const a = new BlinkAnalyzer({ warmupFrames: 30, fps: 30 });
    const open = makeLandmarks(4);
    let last = a.analyze(null, makeFace(open));
    // 31 + 1000 frames @ 30 fps ≈ 34 s — past the new 30 s threshold.
    for (let i = 0; i < 1000; i++) last = a.analyze(null, makeFace(open));
    expect(last.details.blinks).toBe(0);
    expect(last.score).toBeLessThanOrEqual(20);
  });

  it("no blinks for < 15s → abstain (score 50)", () => {
    const a = new BlinkAnalyzer({ warmupFrames: 30, fps: 30 });
    const open = makeLandmarks(4);
    let last = a.analyze(null, makeFace(open));
    // 31 + 150 frames @ 30 fps = 6 s — below the new 15 s "no evidence" cutoff.
    for (let i = 0; i < 180; i++) last = a.analyze(null, makeFace(open));
    expect(last.details.blinks).toBe(0);
    expect(last.score).toBe(50);
  });
});
