import { describe, expect, it } from "vitest";
import { EyebrowAnalyzer } from "../src/infrastructure/analyzers/EyebrowAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

function makeFace(
  blendshapes: Record<string, number> | null,
  face_id = 0,
): FaceROI {
  return {
    face_id,
    bbox: new BBox(0, 0, 100, 100),
    confidence: 0.99,
    landmarks: undefined,
    blendshapes: blendshapes === null ? undefined : new Map(Object.entries(blendshapes)),
  };
}

describe("EyebrowAnalyzer", () => {
  it("returns neutral 50 when no blendshapes are present", () => {
    const a = new EyebrowAnalyzer();
    const r = a.analyze(null, makeFace(null));
    expect(r.score).toBe(50);
    expect(r.details.error).toBe("no_blendshapes");
  });

  it("returns neutral 50 during warmup window", () => {
    const a = new EyebrowAnalyzer({ warmupFrames: 30 });
    for (let i = 0; i < 10; i++) {
      const r = a.analyze(
        null,
        makeFace({
          browInnerUp: 0.5,
          browDownLeft: 0.3,
          browDownRight: 0.3,
          browOuterUpLeft: 0.4,
          browOuterUpRight: 0.4,
        }),
      );
      expect(r.score).toBe(50);
      expect(r.details.warming).toBe(true);
    }
  });

  it("scores near 0 on perfectly static brows (photo)", () => {
    const a = new EyebrowAnalyzer({ warmupFrames: 30, historyLen: 90 });
    // Feed 90 frames of identical brow blendshapes — a printed photo.
    for (let i = 0; i < 90; i++) {
      a.analyze(
        null,
        makeFace({
          browInnerUp: 0.1,
          browDownLeft: 0.1,
          browDownRight: 0.1,
          browOuterUpLeft: 0.1,
          browOuterUpRight: 0.1,
        }),
      );
    }
    const r = a.analyze(
      null,
      makeFace({
        browInnerUp: 0.1,
        browDownLeft: 0.1,
        browDownRight: 0.1,
        browOuterUpLeft: 0.1,
        browOuterUpRight: 0.1,
      }),
    );
    expect(r.score).toBeLessThan(5);
  });

  it("scores high on naturally varying brows (live face)", () => {
    const a = new EyebrowAnalyzer({ warmupFrames: 30, historyLen: 90 });
    // Feed 90 frames where activation oscillates between 0.05 and 0.5.
    for (let i = 0; i < 90; i++) {
      const phase = (i % 30) / 30;
      const act = 0.05 + 0.45 * Math.abs(Math.sin(phase * Math.PI));
      a.analyze(
        null,
        makeFace({
          browInnerUp: act,
          browDownLeft: act * 0.8,
          browDownRight: act * 0.8,
          browOuterUpLeft: act,
          browOuterUpRight: act,
        }),
      );
    }
    const r = a.analyze(
      null,
      makeFace({
        browInnerUp: 0.5,
        browDownLeft: 0.4,
        browDownRight: 0.4,
        browOuterUpLeft: 0.5,
        browOuterUpRight: 0.5,
      }),
    );
    expect(r.score).toBeGreaterThan(30);
  });

  it("reset() drops per-face state", () => {
    const a = new EyebrowAnalyzer({ warmupFrames: 5 });
    for (let i = 0; i < 6; i++) {
      a.analyze(null, makeFace({ browInnerUp: 0.5 }));
    }
    // Past warmup now.
    a.reset();
    const r = a.analyze(null, makeFace({ browInnerUp: 0.5 }));
    expect(r.score).toBe(50);
    expect(r.details.warming).toBe(true);
  });
});
