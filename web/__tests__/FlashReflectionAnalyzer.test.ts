// FlashReflectionAnalyzer tests.
//
// Validate the directional claim under synthetic face-crop frames:
//   * a diffuse, region-varied target-channel rise (real 3D skin reflecting
//     the flash) → high score, isLive true.
//   * no response, or a uniform flat rise with no region spread (a screen
//     emitting its own light) → low score, isLive false.
// Absolute thresholds need live calibration; these pin the ordering.

import { describe, expect, it } from "vitest";
import {
  FlashReflectionAnalyzer,
  FlashColor,
} from "../src/infrastructure/analyzers/FlashReflectionAnalyzer";

const W = 64;
const H = 64;

/** Build a fake face-crop ImageData; `rgbAt(x,y)` returns [r,g,b] per pixel. */
function img(rgbAt: (x: number, y: number) => [number, number, number]): ImageData {
  const data = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const off = (y * W + x) * 4;
      const [r, g, b] = rgbAt(x, y);
      data[off] = r;
      data[off + 1] = g;
      data[off + 2] = b;
      data[off + 3] = 255;
    }
  }
  // Plain object shaped like ImageData — the analyzer only reads data/width/height.
  return { data, width: W, height: H } as unknown as ImageData;
}

const baseline = img(() => [100, 100, 100]);

/** Real face under a red flash: R rises, magnitude VARIES by vertical band
 *  (3D relief), other channels barely move. */
function diffuseRedFlash(): ImageData {
  return img((_x, y) => {
    const band = y / H;
    // forehead (top) brightest, nose (mid) medium, cheeks vary — region spread.
    const rGain = band < 0.3 ? 70 : band < 0.65 ? 45 : 30;
    return [100 + rGain, 105, 105];
  });
}

/** Screen / desktop in bright ambient: the flash barely registers (~+1/255
 *  sensor noise), below the inconclusive floor. */
function noResponseFlash(): ImageData {
  return img(() => [101, 101, 101]);
}

/** Flat uniform rise (a bright but planar/uniform screen response): target
 *  channel up everywhere by the SAME amount → no 3D region spread. */
function uniformRedFlash(): ImageData {
  return img(() => [150, 105, 105]);
}

describe("FlashReflectionAnalyzer", () => {
  it("diffuse region-varied red reflection → live", () => {
    const a = new FlashReflectionAnalyzer();
    const r = a.scoreResponse(baseline, diffuseRedFlash(), "red");
    expect(r.colorShift).toBeGreaterThan(0.05);
    expect(r.regionSpread).toBeGreaterThan(0);
    expect(r.isLive).toBe(true);
    expect(r.score).toBeGreaterThan(50);
  });

  it("no light reaching the face → inconclusive (NOT a false SPOOF)", () => {
    const a = new FlashReflectionAnalyzer();
    const r = a.scoreResponse(baseline, noResponseFlash(), "red");
    // A negligible photometric change means the flash never reached the face
    // (desktop / bright room) — report inconclusive, never LIVE or SPOOF.
    expect(r.inconclusive).toBe(true);
    expect(r.isLive).toBe(false);
  });

  it("diffuse 3D response outscores a flat uniform response of similar gain", () => {
    const a = new FlashReflectionAnalyzer();
    const diffuse = a.scoreResponse(baseline, diffuseRedFlash(), "red");
    const uniform = a.scoreResponse(baseline, uniformRedFlash(), "red");
    // Same channel dominance, but the uniform screen response has ~zero region
    // spread, so it must score strictly lower than the 3D diffuse face.
    expect(uniform.regionSpread).toBeLessThan(diffuse.regionSpread);
    expect(diffuse.score).toBeGreaterThan(uniform.score);
  });

  it("white flash scores a broadband brightness rise as live", () => {
    const a = new FlashReflectionAnalyzer();
    const whiteFlash = img((_x, y) => {
      const band = y / H;
      const gain = band < 0.3 ? 70 : band < 0.65 ? 45 : 30;
      return [100 + gain, 100 + gain, 100 + gain];
    });
    const r = a.scoreResponse(baseline, whiteFlash, "white" as FlashColor);
    expect(r.isLive).toBe(true);
  });
});
