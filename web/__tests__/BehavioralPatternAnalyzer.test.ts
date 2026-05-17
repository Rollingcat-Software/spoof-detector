import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { BehavioralPatternAnalyzer } from "../src/infrastructure/analyzers/BehavioralPatternAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

function makeFace(blendshapes: Record<string, number>, face_id = 0): FaceROI {
  return {
    face_id,
    bbox: new BBox(0, 0, 100, 100),
    confidence: 0.99,
    blendshapes: new Map(Object.entries(blendshapes)),
  };
}

const flatLive = {
  eyeBlinkLeft: 0.05,
  eyeBlinkRight: 0.05,
  jawOpen: 0.1,
  browInnerUp: 0.05,
};
const blink = { ...flatLive, eyeBlinkLeft: 0.85, eyeBlinkRight: 0.85 };

describe("BehavioralPatternAnalyzer", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-05-17T00:00:00Z"));
  });
  afterEach(() => vi.useRealTimers());

  it("returns neutral 50 when no blendshapes present", () => {
    const a = new BehavioralPatternAnalyzer();
    const r = a.analyze(null, {
      face_id: 0,
      bbox: new BBox(0, 0, 100, 100),
      confidence: 0.99,
    });
    expect(r.score).toBe(50);
    expect(r.details.error).toBe("no_blendshapes");
  });

  it("returns neutral 50 during warmup", () => {
    const a = new BehavioralPatternAnalyzer({ warmupFrames: 60 });
    for (let i = 0; i < 20; i++) {
      const r = a.analyze(null, makeFace(flatLive));
      expect(r.score).toBe(50);
    }
  });

  it("scores near 0 on a frozen photo (no blinks, no saccades, low entropy)", () => {
    const a = new BehavioralPatternAnalyzer({
      warmupFrames: 30,
      historyLen: 60,
    });
    for (let i = 0; i < 60; i++) a.analyze(null, makeFace(flatLive));
    const r = a.analyze(null, makeFace(flatLive));
    expect(r.score).toBeLessThan(15);
    expect(r.details.blink_ibi_samples).toBe(0);
  });

  it("scores higher on a session with blinks + gaze saccades + multi-axis motion", () => {
    const a = new BehavioralPatternAnalyzer({
      warmupFrames: 30,
      historyLen: 300,
    });
    // 200 frames: blink every 60 frames (with jitter), saccade pattern,
    // varying jaw/brow values for entropy.
    for (let i = 0; i < 200; i++) {
      const isBlink =
        (i + ((i * 13) % 7)) % 60 < 3; // blink for 3 frames with jitter
      const phase = Math.floor(i / 12) % 4;
      const gazeIn =
        phase === 0 ? 0.4 : phase === 2 ? 0.0 : phase === 1 ? 0 : 0.4;
      const gazeUp =
        phase === 1 ? 0.4 : phase === 3 ? 0.4 : 0;
      const jaw = 0.05 + 0.4 * Math.abs(Math.sin(i / 30));
      const brow = 0.05 + 0.3 * Math.abs(Math.cos(i / 25));
      a.analyze(
        null,
        makeFace({
          eyeBlinkLeft: isBlink ? 0.85 : 0.05,
          eyeBlinkRight: isBlink ? 0.85 : 0.05,
          eyeLookInLeft: gazeIn,
          eyeLookOutRight: gazeIn,
          eyeLookUpLeft: gazeUp,
          eyeLookUpRight: gazeUp,
          jawOpen: jaw,
          browInnerUp: brow,
        }),
      );
    }
    const r = a.analyze(null, makeFace(flatLive));
    expect(r.score).toBeGreaterThan(25);
    expect(r.details.saccade_count).toBeGreaterThan(0);
    expect(r.details.entropy_score as number).toBeGreaterThan(0.4);
  });

  it("reset() drops per-face state", () => {
    const a = new BehavioralPatternAnalyzer({ warmupFrames: 5 });
    for (let i = 0; i < 6; i++) a.analyze(null, makeFace(flatLive));
    a.reset();
    const r = a.analyze(null, makeFace(flatLive));
    expect(r.score).toBe(50);
    expect(r.details.warming).toBe(true);
  });
});
