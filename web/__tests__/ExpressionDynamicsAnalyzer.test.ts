import { describe, expect, it } from "vitest";
import { ExpressionDynamicsAnalyzer } from "../src/infrastructure/analyzers/ExpressionDynamicsAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

function makeFace(blendshapes: Record<string, number>, face_id = 0): FaceROI {
  return {
    face_id,
    bbox: new BBox(0, 0, 100, 100),
    confidence: 0.99,
    blendshapes: new Map(Object.entries(blendshapes)),
  };
}

describe("ExpressionDynamicsAnalyzer", () => {
  it("returns neutral 50 with no blendshapes", () => {
    const a = new ExpressionDynamicsAnalyzer();
    const r = a.analyze(null, {
      face_id: 0,
      bbox: new BBox(0, 0, 100, 100),
      confidence: 0.99,
    });
    expect(r.score).toBe(50);
    expect(r.details.error).toBe("no_blendshapes");
  });

  it("returns neutral 50 during warmup", () => {
    const a = new ExpressionDynamicsAnalyzer({ warmupFrames: 30 });
    for (let i = 0; i < 10; i++) {
      const r = a.analyze(
        null,
        makeFace({ mouthSmileLeft: 0.3, mouthSmileRight: 0.3 }),
      );
      expect(r.score).toBe(50);
    }
  });

  it("scores near 0 on flat fixed expression (photo)", () => {
    const a = new ExpressionDynamicsAnalyzer({
      warmupFrames: 30,
      historyLen: 90,
    });
    for (let i = 0; i < 90; i++) {
      a.analyze(
        null,
        makeFace({ mouthSmileLeft: 0.2, mouthSmileRight: 0.2, cheekPuff: 0.05 }),
      );
    }
    const r = a.analyze(
      null,
      makeFace({ mouthSmileLeft: 0.2, mouthSmileRight: 0.2, cheekPuff: 0.05 }),
    );
    expect(r.score).toBeLessThan(5);
  });

  it("scores high on animated expression (live talking)", () => {
    const a = new ExpressionDynamicsAnalyzer({
      warmupFrames: 30,
      historyLen: 90,
    });
    for (let i = 0; i < 90; i++) {
      const t = i / 90;
      const smile = 0.05 + 0.6 * Math.abs(Math.sin(t * Math.PI * 4));
      const frown = 0.05 + 0.4 * Math.abs(Math.cos(t * Math.PI * 3));
      a.analyze(
        null,
        makeFace({
          mouthSmileLeft: smile,
          mouthSmileRight: smile,
          mouthFrownLeft: frown,
          mouthFrownRight: frown,
          cheekSquintLeft: smile * 0.6,
          cheekSquintRight: smile * 0.6,
        }),
      );
    }
    const r = a.analyze(
      null,
      makeFace({
        mouthSmileLeft: 0.6,
        mouthSmileRight: 0.6,
        cheekSquintLeft: 0.3,
        cheekSquintRight: 0.3,
      }),
    );
    expect(r.score).toBeGreaterThan(30);
    expect(Array.isArray(r.details.top_active)).toBe(true);
    expect((r.details.top_active as unknown[]).length).toBe(3);
  });
});
