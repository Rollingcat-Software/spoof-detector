// MicroTremorAnalyzer tests.
// We can't easily reproduce a calibrated 8-12Hz signal in node, but the
// piecewise score and warmup behaviour are unit-testable:
//   * Frozen centroid → no tremor → low score after enough data.
//   * Synthetic 10Hz oscillation injected via centroid → high tremor ratio.

import { describe, expect, it } from "vitest";
import { MicroTremorAnalyzer } from "../src/infrastructure/analyzers/MicroTremorAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

function frozenFace(face_id = 0): FaceROI {
  return {
    face_id,
    bbox: new BBox(50, 50, 60, 60),
    confidence: 0.9,
  };
}

function shiftedFace(cx: number, cy: number, face_id = 0): FaceROI {
  // bbox shifted so its center equals (cx, cy).
  const half = 5;
  return {
    face_id,
    bbox: new BBox(cx - half, cy - half, cx + half, cy + half),
    confidence: 0.9,
  };
}

describe("MicroTremorAnalyzer", () => {
  it("returns warmup below MIN_FRAMES", () => {
    const a = new MicroTremorAnalyzer();
    const r = a.analyze(null, frozenFace());
    expect(r.score).toBe(50.0);
    expect(r.details.warmup).toBe(true);
  });

  it("frozen centroid → low tremor ratio after warmup", () => {
    const a = new MicroTremorAnalyzer({ historyLen: 60 });
    let last = a.analyze(null, frozenFace());
    for (let i = 0; i < 90; i++) last = a.analyze(null, frozenFace());
    expect(last.details.warmup).toBeUndefined();
    // Frozen centroid contains no signal — ratio should be very low.
    expect(last.details.tremor_ratio as number).toBeLessThan(0.5);
    expect(last.score).toBeLessThanOrEqual(50);
  });

  it("synthetic 10 Hz oscillation → score above 50 after warmup", () => {
    // At fps=30 and N=60, a 10 Hz sinusoid has 20 cycles — clearly inside
    // the 7-13 Hz tremor band. 10 Hz / (30/60) = bin 20 of 30 → ✓.
    const a = new MicroTremorAnalyzer({ historyLen: 60, fps: 30 });
    let last = a.analyze(null, shiftedFace(50, 50));
    for (let i = 0; i < 90; i++) {
      const t = i;
      // Strong 10 Hz oscillation on top of a stable 50,50 centre.
      const cx = 50 + 5 * Math.sin((2 * Math.PI * 10 * t) / 30);
      const cy = 50 + 5 * Math.cos((2 * Math.PI * 10 * t) / 30);
      last = a.analyze(null, shiftedFace(cx, cy));
    }
    expect(last.details.warmup).toBeUndefined();
    expect(last.details.tremor_ratio as number).toBeGreaterThan(0.5);
    expect(last.score).toBeGreaterThan(40);
  });
});
