// TextureAnalyzer tests.
// Constructs procedural ImageData and verifies:
//   * no_pixels error when neither crop nor frame is set
//   * uniform block (flat colour) → very low texture variance → low texture
//     sub-score → fused score below 50.
//   * random noise → high Laplacian variance → texture sub-score reaches
//     its top half → fused score above 30.
//   * synthetic FFT thumbnail driven by setFrame fallback works.

import { describe, expect, it } from "vitest";
import { TextureAnalyzer } from "../src/infrastructure/analyzers/TextureAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

const W = 64;
const H = 64;

function uniformImage(rgb: number): ImageData {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = rgb;
    data[i + 1] = rgb;
    data[i + 2] = rgb;
    data[i + 3] = 255;
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

function noisyImage(seed: number): ImageData {
  // Deterministic LCG so tests are reproducible.
  let s = seed >>> 0;
  function rand(): number {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  }
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    // Add a face-like reddish tint so HSV sees non-zero saturation.
    const base = 90 + Math.floor(rand() * 100);
    data[i] = Math.min(255, base + 40);
    data[i + 1] = base;
    data[i + 2] = Math.max(0, base - 20);
    data[i + 3] = 255;
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(0, 0, W, H),
  confidence: 0.9,
};

describe("TextureAnalyzer", () => {
  it("returns no_pixels error before any frame or crop", () => {
    const a = new TextureAnalyzer();
    const r = a.analyze(null, face);
    expect(r.score).toBe(50.0);
    expect(r.details.error).toBe("no_pixels");
  });

  it("uniform-colour block → very low texture variance → low texture sub-score", () => {
    const a = new TextureAnalyzer();
    const r = a.analyze(uniformImage(180), face);
    // Uniform input: Laplacian is identically zero → texture_score = 0.
    // The texture sub-score weight is 0.40 → fused score capped by the
    // colour/freq channels, but well below 60.
    expect(r.details.texture_score as number).toBe(0);
    expect(r.score).toBeLessThan(60);
  });

  it("noisy block → texture sub-score crosses the 50.0 threshold", () => {
    const a = new TextureAnalyzer();
    const r = a.analyze(noisyImage(42), face);
    // Random noise produces a Laplacian variance well over the 100.0
    // calibrated threshold, so the texture sub-score lands at ≥50.
    expect(r.details.texture_score as number).toBeGreaterThanOrEqual(50);
    // And the overall fused score should be > 30 (not "obviously spoof").
    expect(r.score).toBeGreaterThan(30);
  });

  it("setFrame fallback path produces same shape as direct crop", () => {
    const a = new TextureAnalyzer();
    a.setFrame(noisyImage(123));
    const r = a.analyze(null, {
      face_id: 1,
      bbox: new BBox(0, 0, W, H),
      confidence: 0.95,
    });
    expect(r.details.error).toBeUndefined();
    expect(typeof r.details.texture_score).toBe("number");
    expect(typeof r.details.color_score).toBe("number");
    expect(typeof r.details.frequency_score).toBe("number");
  });
});
