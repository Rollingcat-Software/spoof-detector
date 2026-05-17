import { describe, expect, it } from "vitest";
import { BlinkSymmetryAnalyzer } from "../src/infrastructure/analyzers/BlinkSymmetryAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

function makeFace(left: number, right: number, face_id = 0): FaceROI {
  return {
    face_id,
    bbox: new BBox(0, 0, 100, 100),
    confidence: 0.99,
    blendshapes: new Map<string, number>([
      ["eyeBlinkLeft", left],
      ["eyeBlinkRight", right],
    ]),
  };
}

function syntheticBlink(t: number): number {
  // A short blink dip — closed for ~3 frames in every 30.
  const phase = t % 30;
  return phase < 3 ? 0.85 : 0.05;
}

describe("BlinkSymmetryAnalyzer", () => {
  it("returns neutral 50 when no blendshapes present", () => {
    const a = new BlinkSymmetryAnalyzer();
    const r = a.analyze(null, {
      face_id: 0,
      bbox: new BBox(0, 0, 100, 100),
      confidence: 0.99,
    });
    expect(r.score).toBe(50);
    expect(r.details.error).toBe("no_blendshapes");
  });

  it("returns neutral 50 during warmup window", () => {
    const a = new BlinkSymmetryAnalyzer({ warmupFrames: 30 });
    for (let i = 0; i < 20; i++) {
      const r = a.analyze(null, makeFace(syntheticBlink(i), syntheticBlink(i)));
      expect(r.score).toBe(50);
    }
  });

  it("scores high (>=70) when both eyes blink synchronously", () => {
    const a = new BlinkSymmetryAnalyzer({ warmupFrames: 30, historyLen: 90 });
    for (let i = 0; i < 90; i++) {
      const v = syntheticBlink(i);
      a.analyze(null, makeFace(v, v));
    }
    const r = a.analyze(null, makeFace(0.05, 0.05));
    expect(r.score).toBeGreaterThanOrEqual(70);
    expect(r.details.corr).toBeGreaterThan(0.7);
  });

  it("scores low (~0) when eyes blink anti-correlated (animated avatar)", () => {
    const a = new BlinkSymmetryAnalyzer({ warmupFrames: 30, historyLen: 90 });
    for (let i = 0; i < 90; i++) {
      a.analyze(null, makeFace(syntheticBlink(i), syntheticBlink(i + 15)));
    }
    const r = a.analyze(null, makeFace(0.85, 0.05));
    // Inverted phase ⇒ negative corr ⇒ score 0 (clamped).
    expect(r.score).toBeLessThan(20);
  });

  it("returns neutral 50 when both signals are flat (static photo)", () => {
    const a = new BlinkSymmetryAnalyzer({ warmupFrames: 30, historyLen: 90 });
    for (let i = 0; i < 90; i++) {
      a.analyze(null, makeFace(0.05, 0.05));
    }
    const r = a.analyze(null, makeFace(0.05, 0.05));
    expect(r.score).toBe(50);
    expect(r.details.flat).toBe(true);
  });

  it("reset() drops per-face state", () => {
    const a = new BlinkSymmetryAnalyzer({ warmupFrames: 5 });
    for (let i = 0; i < 6; i++) {
      a.analyze(null, makeFace(syntheticBlink(i), syntheticBlink(i)));
    }
    a.reset();
    const r = a.analyze(null, makeFace(0.05, 0.05));
    expect(r.score).toBe(50);
    expect(r.details.warming).toBe(true);
  });
});
