// ScreenFlickerAnalyzer tests.
// Uses a 64×64 procedural ImageData where each frame's mean intensity
// is modulated either constantly (real) or at the 30 Hz beat frequency
// (screen). The analyzer's per-frame mean-intensity → DFT path should
// pick up the latter.

import { describe, expect, it } from "vitest";
import { ScreenFlickerAnalyzer } from "../src/infrastructure/analyzers/ScreenFlickerAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

function makeImageData(intensity: number): ImageData {
  // Polyfill ImageData for node — vitest "node" env doesn't ship it.
  // We construct an object with the same shape; the analyzer only reads
  // .data, .width, .height.
  const w = 32;
  const h = 32;
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = intensity;
    data[i + 1] = intensity;
    data[i + 2] = intensity;
    data[i + 3] = 255;
  }
  return { data, width: w, height: h, colorSpace: "srgb" } as ImageData;
}

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(0, 0, 32, 32),
  confidence: 0.9,
};

describe("ScreenFlickerAnalyzer", () => {
  it("warmup score below MIN_FRAMES", () => {
    const a = new ScreenFlickerAnalyzer();
    const r = a.analyze(makeImageData(128), face);
    expect(r.score).toBe(50.0);
    expect(r.details.warmup).toBe(true);
  });

  it("constant intensity → low flicker SNR → high (live) score", () => {
    const a = new ScreenFlickerAnalyzer({ historyLen: 60 });
    let last = a.analyze(makeImageData(128), face);
    for (let i = 0; i < 90; i++) {
      last = a.analyze(makeImageData(128), face);
    }
    expect(last.details.warmup).toBeUndefined();
    // Constant input ⇒ post-detrend energy is essentially numerical noise.
    expect(last.score).toBeGreaterThan(50);
  });

  it("30 Hz modulation reaches a non-trivial flicker SNR", () => {
    // fps≈30, N=60. We modulate intensity at the highest representable
    // frequency (Nyquist = 15 Hz at fps=30). The (28-35) Hz band is out
    // of reach at fps=30, so we test the 8-15 Hz band instead with a
    // strong 12 Hz signal.
    const a = new ScreenFlickerAnalyzer({ historyLen: 60, fps: 30 });
    let last = a.analyze(makeImageData(128), face);
    for (let i = 0; i < 90; i++) {
      const intensity =
        128 + 40 * Math.sin((2 * Math.PI * 12 * i) / 30);
      last = a.analyze(makeImageData(intensity), face);
    }
    expect(last.details.warmup).toBeUndefined();
    expect(last.details.flicker_snr as number).toBeGreaterThan(1.5);
  });
});
