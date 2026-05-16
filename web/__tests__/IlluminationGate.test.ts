// IlluminationGate tests.
// Procedural-ImageData approach mirrors DeviceBoundaryAnalyzer.test.ts.
//
// Covers:
//   * no-frame / null bbox → LOW_QUALITY empty result.
//   * uniform-grey degenerate input → LOW_QUALITY (low illumination_score).
//   * positive bright/even input → quality_ok = true.

import { describe, expect, it } from "vitest";
import { IlluminationGate } from "../src/gates/IlluminationGate";

const W = 128;
const H = 128;
const FACE_BBOX = [16, 16, 96, 96] as const;

function uniform(value: number): ImageData {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = value;
    data[i + 1] = value;
    data[i + 2] = value;
    data[i + 3] = 255;
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

function gradientWithTexture(): ImageData {
  // Bright, evenly lit, with some per-region texture so the
  // normalized_detail_score is non-zero and the gate passes.
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const off = (y * W + x) * 4;
      // Skin-tone-ish base (~140 luma) with subtle xy modulation.
      const baseR = 175 + ((x + y) % 11) - 5;
      const baseG = 145 + ((x * 2 + y) % 9) - 4;
      const baseB = 125 + ((x + y * 3) % 13) - 6;
      // Add a 4-px chequer to drive some Laplacian texture/edge density.
      const c = ((x >> 2) + (y >> 2)) & 1;
      const delta = c ? 12 : -12;
      data[off] = clamp8(baseR + delta);
      data[off + 1] = clamp8(baseG + delta);
      data[off + 2] = clamp8(baseB + delta);
      data[off + 3] = 255;
    }
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

function clamp8(v: number): number {
  if (v < 0) return 0;
  if (v > 255) return 255;
  return v;
}

describe("IlluminationGate", () => {
  it("no frame → LOW_QUALITY with poor_face_illumination", () => {
    const g = new IlluminationGate();
    const r = g.evaluate(null, [...FACE_BBOX]);
    expect(r.qualityOk).toBe(false);
    expect(r.qualityStatus).toBe("LOW_QUALITY");
    expect(r.qualityReason).toBe("poor_face_illumination");
    expect(r.illuminationScore).toBe(0);
  });

  it("null bbox → LOW_QUALITY empty result", () => {
    const g = new IlluminationGate();
    const r = g.evaluate(uniform(180), null);
    expect(r.qualityOk).toBe(false);
    expect(r.qualityStatus).toBe("LOW_QUALITY");
    expect(r.globalFaceBrightness).toBe(0);
  });

  it("uniform mid-grey face → returns structurally valid metrics", () => {
    const g = new IlluminationGate();
    // Mid-grey 128 is dead-center in the brightness band (inner_low=72,
    // inner_high=188) so brightness_score=1.0; uniformity=1.0 too. The
    // calibrated band-score logic gives a passing aggregate illumination
    // score even with zero contrast — same behaviour as the Python source.
    // What we assert is the structural integrity of the metrics.
    const r = g.evaluate(uniform(128), [...FACE_BBOX]);
    expect(r.brightnessUniformity).toBeGreaterThan(0.9);
    expect(r.shadowAsymmetry).toBeLessThan(0.05);
    // Every quality region must have a brightness reading.
    expect(Object.keys(r.perRegionBrightness).length).toBeGreaterThanOrEqual(5);
    expect(r.illuminationScore).toBeGreaterThan(0);
    expect(r.illuminationScore).toBeLessThanOrEqual(1);
  });

  it("very dark uniform face → poor_face_illumination", () => {
    const g = new IlluminationGate();
    const r = g.evaluate(uniform(20), [...FACE_BBOX]);
    expect(r.qualityOk).toBe(false);
    expect(r.qualityReason).toBe("poor_face_illumination");
    expect(r.globalFaceBrightness).toBeLessThan(62);
  });

  it("bright textured face → quality_ok = true", () => {
    const g = new IlluminationGate();
    const r = g.evaluate(gradientWithTexture(), [...FACE_BBOX]);
    expect(r.globalFaceBrightness).toBeGreaterThan(70);
    expect(r.globalFaceBrightness).toBeLessThan(210);
    expect(r.shadowAsymmetry).toBeLessThan(0.4);
    expect(r.qualityOk).toBe(true);
    expect(r.qualityStatus).toBe("OK");
    expect(r.qualityReason).toBe("face_quality_ok");
    expect(Object.keys(r.perRegionBrightness).length).toBeGreaterThan(0);
  });
});
