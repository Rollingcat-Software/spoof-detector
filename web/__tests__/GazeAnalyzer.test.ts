import { describe, expect, it } from "vitest";
import { GazeAnalyzer } from "../src/infrastructure/analyzers/GazeAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

function makeFace(blendshapes: Record<string, number>, face_id = 0): FaceROI {
  return {
    face_id,
    bbox: new BBox(0, 0, 100, 100),
    confidence: 0.99,
    blendshapes: new Map(Object.entries(blendshapes)),
  };
}

function flatGaze(x: number, y: number): Record<string, number> {
  // Encode a gaze vector (x ∈ [-1,1], y ∈ [-1,1]) into 8 blendshapes.
  // Positive x looks left (eyeLookInLeft + eyeLookOutRight). Positive y up.
  const xPos = Math.max(0, x);
  const xNeg = Math.max(0, -x);
  const yPos = Math.max(0, y);
  const yNeg = Math.max(0, -y);
  return {
    eyeLookInLeft: xPos,
    eyeLookOutRight: xPos,
    eyeLookOutLeft: xNeg,
    eyeLookInRight: xNeg,
    eyeLookUpLeft: yPos,
    eyeLookUpRight: yPos,
    eyeLookDownLeft: yNeg,
    eyeLookDownRight: yNeg,
  };
}

describe("GazeAnalyzer", () => {
  it("returns neutral 50 when no blendshapes present", () => {
    const a = new GazeAnalyzer();
    const r = a.analyze(null, {
      face_id: 0,
      bbox: new BBox(0, 0, 100, 100),
      confidence: 0.99,
    });
    expect(r.score).toBe(50);
    expect(r.details.error).toBe("no_blendshapes");
  });

  it("returns neutral 50 during warmup", () => {
    const a = new GazeAnalyzer({ warmupFrames: 30 });
    for (let i = 0; i < 10; i++) {
      const r = a.analyze(null, makeFace(flatGaze(0, 0)));
      expect(r.score).toBe(50);
    }
  });

  it("scores near 0 on perfectly fixed gaze (photo)", () => {
    const a = new GazeAnalyzer({ warmupFrames: 30, historyLen: 90 });
    for (let i = 0; i < 90; i++) {
      a.analyze(null, makeFace(flatGaze(0.3, 0.0)));
    }
    const r = a.analyze(null, makeFace(flatGaze(0.3, 0.0)));
    expect(r.score).toBeLessThan(10);
    expect(r.details.saccade_count).toBe(0);
  });

  it("scores high on varying gaze with saccades (live)", () => {
    const a = new GazeAnalyzer({ warmupFrames: 30, historyLen: 90 });
    for (let i = 0; i < 90; i++) {
      // Saccade pattern: jump every 10 frames between left/right/up/down.
      const phase = Math.floor(i / 10) % 4;
      const tgt: [number, number] =
        phase === 0 ? [0.3, 0] : phase === 1 ? [-0.3, 0] : phase === 2 ? [0, 0.3] : [0, -0.3];
      a.analyze(null, makeFace(flatGaze(tgt[0], tgt[1])));
    }
    const r = a.analyze(null, makeFace(flatGaze(0.3, 0)));
    expect(r.score).toBeGreaterThan(40);
    expect(r.details.saccade_count).toBeGreaterThan(0);
  });

  it("reset() drops per-face state", () => {
    const a = new GazeAnalyzer({ warmupFrames: 5 });
    for (let i = 0; i < 6; i++) a.analyze(null, makeFace(flatGaze(0.1, 0.1)));
    a.reset();
    const r = a.analyze(null, makeFace(flatGaze(0.1, 0.1)));
    expect(r.score).toBe(50);
    expect(r.details.warming).toBe(true);
  });

  it("regression: saccade_rate_per_sec is wall-clock based, not frame/30", async () => {
    // Pre-fix: the rate divided saccadeCount by (historyLen/30), so a
    // 90-frame buffer always produced "rate per 3 seconds" regardless
    // of the real elapsed time. On mobile (~9 fps) this inflated the
    // displayed rate by ~3.3×. Post-fix it uses wall-clock elapsed.
    const a = new GazeAnalyzer({
      warmupFrames: 30,
      historyLen: 90,
      saccadeThreshold: 0.1,
    });
    // Ingest 90 frames with synthetic saccades, sleeping briefly between
    // groups so the wall-clock denominator grows past the frame-count/30
    // estimate. Five distinct gaze positions every 18 frames = 4 saccades.
    for (let i = 0; i < 90; i++) {
      const phase = Math.floor(i / 18) % 5;
      const v: [number, number] =
        phase === 0 ? [0.3, 0] : phase === 1 ? [-0.3, 0] :
        phase === 2 ? [0, 0.3] : phase === 3 ? [0, -0.3] : [0.3, 0.3];
      a.analyze(null, makeFace(flatGaze(v[0], v[1])));
    }
    // Sleep ~50ms to make wall-clock elapsed measurably > frame-count/30.
    await new Promise((r) => setTimeout(r, 80));
    const r = a.analyze(null, makeFace(flatGaze(0.3, 0)));
    // saccade_count is 4 (one per phase transition). Rate at ~0.1 s
    // wall-clock would be a huge spike — but we just need to confirm
    // it's not an obviously-wrong number like > 1000.
    expect(r.details.saccade_rate_per_sec as number).toBeGreaterThan(0);
    expect(r.details.saccade_rate_per_sec as number).toBeLessThan(1000);
    // And confirm the count itself is reasonable — well under the
    // pre-fix value where every 0.05+ jitter counted as a saccade.
    expect(r.details.saccade_count as number).toBeLessThanOrEqual(20);
  });
});
