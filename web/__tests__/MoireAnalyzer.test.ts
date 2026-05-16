// MoireAnalyzer tests.
//
// Constructs procedural 96×96 grayscale ImageData face crops that
// exercise three regimes:
//   * No frame at all (warmup / no_frame branch).
//   * Uniform-input (no moire, no Gabor response) ⇒ high real-leaning score.
//   * Synthetic vertical stripe pattern at a 4-px period — exactly the
//     kind of periodic pixel-grid artifact moire detection targets ⇒
//     non-trivial moire_risk on at least one of the gabor / fft channels.

import { describe, expect, it } from "vitest";
import { MoireAnalyzer } from "../src/infrastructure/analyzers/MoireAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

const W = 96;
const H = 96;

function makeImage(fill: (x: number, y: number) => number): ImageData {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const v = Math.max(0, Math.min(255, Math.round(fill(x, y))));
      const off = (y * W + x) * 4;
      data[off] = v;
      data[off + 1] = v;
      data[off + 2] = v;
      data[off + 3] = 255;
    }
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(0, 0, W, H),
  confidence: 0.9,
};

describe("MoireAnalyzer", () => {
  it("returns no_frame error when faceCrop is null", () => {
    const a = new MoireAnalyzer();
    const r = a.analyze(null, face);
    expect(r.score).toBe(50.0);
    expect(r.details.error).toBe("no_frame");
  });

  it("uniform input → low moire_risk → high (live) score", () => {
    const a = new MoireAnalyzer();
    const img = makeImage(() => 128);
    const r = a.analyze(img, face);
    // Uniform input: Gabor response std collapses to ~0, FFT power
    // concentrates at DC, so moire_risk should sit near 0.
    expect(r.details.moire_risk as number).toBeLessThan(0.25);
    expect(r.details.gabor_risk as number).toBeLessThan(0.25);
    expect(r.score).toBeGreaterThan(75);
  });

  it("alternating-stripe pattern (4-px period) → non-trivial moire signal", () => {
    // Vertical black/white stripes at a 4-pixel period — this is exactly
    // the kind of periodic grid that a camera sees when photographing
    // a screen. The vertical-oriented Gabor (θ=π/2) should fire and the
    // FFT mid-band should pick up the periodicity.
    const a = new MoireAnalyzer();
    const img = makeImage((x) => ((Math.floor(x / 2) % 2) === 0 ? 0 : 255));
    const r = a.analyze(img, face);
    expect(r.details.moire_risk as number).toBeGreaterThan(0.1);
    // At least one of the two channels must register the periodicity.
    const gabor = r.details.gabor_risk as number;
    const fft = r.details.fft_risk as number;
    expect(Math.max(gabor, fft)).toBeGreaterThan(0.1);
    // And the score should reflect a riskier (lower) verdict than the
    // uniform-input baseline.
    expect(r.score).toBeLessThan(95);
  });
});
