// BackgroundGridAnalyzer tests.
// Procedural 320×240 frames cover:
//   * no_frame error path (before setFrame()),
//   * uniform background + face → high stability score after warmup,
//   * noisy background → low stability score,
//   * a screen-cool flood → cool_ratio penalty triggers.

import { describe, expect, it } from "vitest";
import { BackgroundGridAnalyzer } from "../src/infrastructure/analyzers/BackgroundGridAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

const W = 320;
const H = 240;

function uniform(r: number, g: number, b: number): ImageData {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = 255;
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

/**
 * Background that flips between dark and bright across frames so per-cell
 * grayscale **means** swing by ~100 luma — well above STABILITY_THRESHOLD (12).
 * Per-pixel noise alone won't move the cell mean (law of large numbers on
 * ~12k pixels per cell), so we test instability by varying the global level.
 */
function flickering(frameIdx: number): ImageData {
  // Alternate between ~80 and ~200 each frame.
  const v = frameIdx % 2 === 0 ? 80 : 200;
  return uniform(v, v, v);
}

const face: FaceROI = {
  // Centred on the frame so the face cell-mask drops the middle 4–6 cells.
  face_id: 0,
  bbox: new BBox(120, 80, 200, 160),
  confidence: 0.9,
};

describe("BackgroundGridAnalyzer", () => {
  it("returns no_frame error before setFrame()", () => {
    const a = new BackgroundGridAnalyzer();
    const r = a.analyze(null, face);
    expect(r.score).toBe(50.0);
    expect(r.details.error).toBe("no_frame");
  });

  it("returns warmup until MIN_FRAMES seen", () => {
    const a = new BackgroundGridAnalyzer();
    a.setFrame(uniform(180, 180, 180));
    // First call counts as frame 1 — under the 15-frame threshold.
    const r = a.analyze(null, face);
    expect(r.score).toBe(50.0);
    expect(r.details.warmup).toBe(true);
  });

  it("uniform static background → high stability score after warmup", () => {
    const a = new BackgroundGridAnalyzer();
    const frame = uniform(180, 180, 180);
    a.setFrame(frame);
    let last = a.analyze(null, face);
    // Run for >MIN_FRAMES (15) frames so the cell-history std evaluator
    // fires and registers every background cell as stable.
    for (let i = 0; i < 20; i++) {
      a.setFrame(frame);
      last = a.analyze(null, face);
    }
    expect(last.details.warmup).toBeUndefined();
    // Uniform grey → stable_ratio = 1.0, specular/cool ratios = 0.
    expect(last.details.stability_ratio as number).toBeCloseTo(1.0, 2);
    expect(last.details.specular_ratio as number).toBe(0);
    expect(last.details.cool_ratio as number).toBe(0);
    // 1.0*60 + 30 = 90.0.
    expect(last.score).toBeCloseTo(90.0, 1);
  });

  it("flickering background → lower stability score", () => {
    const a = new BackgroundGridAnalyzer();
    let last = a.analyze(null, face);
    for (let i = 0; i < 25; i++) {
      a.setFrame(flickering(i));
      last = a.analyze(null, face);
    }
    expect(last.details.warmup).toBeUndefined();
    // Cell means oscillate by ~120 luma frame-to-frame → recent_std ~60 ≫ 12,
    // so no cell counts as stable.
    expect(last.details.stability_ratio as number).toBeLessThan(0.3);
    // Score = stability*60 + 30 − penalties ≤ 30 + small positives.
    expect(last.score).toBeLessThan(60);
  });

  it("cool blue flood triggers cool_ratio penalty", () => {
    // Pure blue (0,0,255) lands hue 240° → OpenCV H8 = 120 → inside [100,130].
    const a = new BackgroundGridAnalyzer();
    const frame = uniform(0, 0, 255);
    a.setFrame(frame);
    let last = a.analyze(null, face);
    for (let i = 0; i < 20; i++) {
      a.setFrame(frame);
      last = a.analyze(null, face);
    }
    expect(last.details.warmup).toBeUndefined();
    expect(last.details.cool_ratio as number).toBeGreaterThan(0.9);
  });
});
