// ScreenReplayAnalyzer tests.
// Constructs procedural full-frame ImageData and verifies:
//   * no_frame error before setFrame()
//   * uniform / very flat frame → blur_floor branch (laplacian_var < 25)
//     short-circuits to 50.0.
//   * Noisy skin-toned frame → full pipeline runs (no blur_floor), all
//     four sub-scores present, score within [0,100].
//   * Bright low-saturation frame (mimics screen glare) → specular sub-
//     score is depressed vs a balanced one.

import { describe, expect, it } from "vitest";
import { ScreenReplayAnalyzer } from "../src/infrastructure/analyzers/ScreenReplayAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

const W = 96;
const H = 96;

function uniformImage(rgb: [number, number, number]): ImageData {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = rgb[0];
    data[i + 1] = rgb[1];
    data[i + 2] = rgb[2];
    data[i + 3] = 255;
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

function noisySkinImage(seed: number): ImageData {
  let s = seed >>> 0;
  function rand(): number {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  }
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    // Skin-tone range with noise so Laplacian variance >> 25.
    const r = 180 + Math.floor(rand() * 40);
    const g = 130 + Math.floor(rand() * 30);
    const b = 110 + Math.floor(rand() * 30);
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = 255;
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

function brightDesaturatedNoisyImage(seed: number): ImageData {
  // Mimics screen glare: high V, low S, with structure so blur_floor
  // doesn't short-circuit.
  let s = seed >>> 0;
  function rand(): number {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  }
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    // ~250 with very tight RGB spread → very low saturation.
    const base = 245 + Math.floor(rand() * 10);
    data[i] = base;
    data[i + 1] = base;
    data[i + 2] = base;
    data[i + 3] = 255;
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(20, 20, 80, 80),
  confidence: 0.9,
};

describe("ScreenReplayAnalyzer", () => {
  it("returns no_frame error before setFrame()", () => {
    const a = new ScreenReplayAnalyzer();
    const r = a.analyze(null, face);
    expect(r.score).toBe(50.0);
    expect(r.details.error).toBe("no_frame");
  });

  it("uniform flat frame → blur_floor branch returns 50.0", () => {
    const a = new ScreenReplayAnalyzer();
    a.setFrame(uniformImage([180, 140, 120]));
    const r = a.analyze(null, face);
    // Constant input → Laplacian variance = 0 → < 25 floor.
    expect(r.details.blur_floor).toBe(true);
    expect(r.score).toBe(50.0);
  });

  it("noisy skin-toned frame → full pipeline, all four sub-scores present", () => {
    const a = new ScreenReplayAnalyzer();
    a.setFrame(noisySkinImage(7));
    const r = a.analyze(null, face);
    expect(r.details.blur_floor).toBeUndefined();
    expect(typeof r.details.fft_score).toBe("number");
    expect(typeof r.details.laplacian_score).toBe("number");
    expect(typeof r.details.skin_score).toBe("number");
    expect(typeof r.details.specular_score).toBe("number");
    expect(r.score).toBeGreaterThanOrEqual(0);
    expect(r.score).toBeLessThanOrEqual(100);
  });

  it("bright desaturated noisy frame depresses the specular sub-score vs balanced", () => {
    const balanced = new ScreenReplayAnalyzer();
    balanced.setFrame(noisySkinImage(11));
    const rBal = balanced.analyze(null, face);

    const glare = new ScreenReplayAnalyzer();
    glare.setFrame(brightDesaturatedNoisyImage(11));
    const rGlare = glare.analyze(null, face);

    // The glare image has ~100% of pixels matching the
    // (V≥240, S≤35) specular mask — ratio well above the 0.06 high
    // calibration point — so specular_score should be ≤ balanced one
    // (typically near 0).
    expect(rGlare.details.specular_score as number).toBeLessThanOrEqual(
      rBal.details.specular_score as number,
    );
  });
});
