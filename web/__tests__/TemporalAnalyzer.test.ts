// TemporalAnalyzer tests.
// Covers three regimes — warmup, frozen face (suspicious), and natural
// motion (live-like) — plus the per-face_id state isolation contract.

import { describe, expect, it } from "vitest";
import { TemporalAnalyzer } from "../src/infrastructure/analyzers/TemporalAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

function makeFace(cx: number, cy: number, half = 50, face_id = 0): FaceROI {
  return {
    face_id,
    bbox: new BBox(cx - half, cy - half, cx + half, cy + half),
    confidence: 0.9,
  };
}

describe("TemporalAnalyzer", () => {
  it("returns warmup details below warmupFrames", () => {
    const a = new TemporalAnalyzer();
    const r = a.analyze(null, makeFace(100, 100));
    expect(r.score).toBe(50.0);
    expect(r.details.warmup).toBe(true);
    expect(r.details.frames).toBe(1);
  });

  it("frozen face after warmup → low score (suspicious / spoof-like)", () => {
    const a = new TemporalAnalyzer();
    let last = a.analyze(null, makeFace(100, 100));
    // Repeat the identical bbox enough times to exit warmup.
    for (let i = 0; i < 30; i++) {
      last = a.analyze(null, makeFace(100, 100));
    }
    expect(last.details.warmup).toBeUndefined();
    // Zero variance → motion === 0 → score === 10.0.
    expect(last.details.motion as number).toBeLessThan(0.0003);
    expect(last.score).toBe(10.0);
  });

  it("natural motion → high score (live-like)", () => {
    const a = new TemporalAnalyzer();
    let last = a.analyze(null, makeFace(100, 100));
    // ~5-px wander around (100, 100) plus mild area breathing.
    for (let i = 0; i < 30; i++) {
      const cx = 100 + 8 * Math.sin(i * 0.4);
      const cy = 100 + 8 * Math.cos(i * 0.35);
      const half = 50 + 3 * Math.sin(i * 0.5);
      last = a.analyze(null, makeFace(cx, cy, half));
    }
    expect(last.details.warmup).toBeUndefined();
    // 5–8 px wander on a 100-px face is well above the 0.0003 floor.
    expect(last.details.motion as number).toBeGreaterThan(0.0003 * 10);
    expect(last.score).toBe(90.0);
  });

  it("micro-motion in the linear-interp band → mid score", () => {
    // Face ~100 px wide (half=50) → norm_factor = sqrt(meanArea)=100.
    // sin(i) at integer i has std ≈ 0.67 → for jitter amplitude A,
    // each-axis std ≈ 0.67*A, pos_std = sqrt(2)*0.67*A / 100 ≈ A*0.0095.
    // To land between minMotionStd=0.0003 and 10*minMotionStd=0.003,
    // we need A ≈ 0.05–0.30 px. Pick A=0.10 → pos_std ≈ 0.00095,
    // which lies in the third regime (3*minMotionStd .. 10*minMotionStd).
    const a = new TemporalAnalyzer();
    let last = a.analyze(null, makeFace(100, 100));
    for (let i = 0; i < 30; i++) {
      const cx = 100 + 0.10 * Math.sin(i);
      const cy = 100 + 0.10 * Math.cos(i);
      last = a.analyze(null, makeFace(cx, cy));
    }
    expect(last.details.warmup).toBeUndefined();
    // Should land strictly above the floor (score > 10) and well below 90.
    expect(last.score).toBeGreaterThan(10);
    expect(last.score).toBeLessThan(90);
  });

  it("isolates state per face_id", () => {
    const a = new TemporalAnalyzer();
    // Face A reaches warmup; face B is brand-new on the next call.
    for (let i = 0; i < 20; i++) a.analyze(null, makeFace(50, 50, 50, 1));
    const rB = a.analyze(null, makeFace(200, 200, 50, 2));
    expect(rB.details.warmup).toBe(true);
    expect(rB.details.frames).toBe(1);
  });

  it("reset(face_id) clears only that face's buffer", () => {
    const a = new TemporalAnalyzer();
    for (let i = 0; i < 20; i++) a.analyze(null, makeFace(50, 50, 50, 1));
    a.reset(1);
    const r = a.analyze(null, makeFace(50, 50, 50, 1));
    expect(r.details.warmup).toBe(true);
    expect(r.details.frames).toBe(1);
  });
});
