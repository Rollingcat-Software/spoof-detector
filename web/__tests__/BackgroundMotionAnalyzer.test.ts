import { describe, expect, it } from "vitest";
import { BackgroundMotionAnalyzer } from "../src/infrastructure/analyzers/BackgroundMotionAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";
import type { MediaPipeSelfieSegmenter } from "../src/infrastructure/detection/MediaPipeSelfieSegmenter";

function makeFrame(brightness: number): ImageData {
  const W = 192;
  const H = 108;
  const data = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < data.length; i += 4) {
    data[i] = brightness;
    data[i + 1] = brightness;
    data[i + 2] = brightness;
    data[i + 3] = 255;
  }
  return { data, width: W, height: H, colorSpace: "srgb" } as ImageData;
}

function makeMask(W = 32, H = 18, personRatio = 0.3): Float32Array {
  const m = new Float32Array(W * H);
  // Make the centre `personRatio` fraction the person (high confidence).
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const dx = (x - W / 2) / (W / 2);
      const dy = (y - H / 2) / (H / 2);
      const inside = Math.hypot(dx, dy) < personRatio;
      m[y * W + x] = inside ? 1.0 : 0.0;
    }
  }
  return m;
}

class StubSegmenter {
  constructor(public personRatio = 0.3) {}
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async warmup(): Promise<void> {}
  async segment(): Promise<{ mask: Float32Array; width: number; height: number }> {
    return { mask: makeMask(32, 18, this.personRatio), width: 32, height: 18 };
  }
  async close(): Promise<void> {}
}

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(0, 0, 192, 108),
  confidence: 0.9,
};

describe("BackgroundMotionAnalyzer", () => {
  it("returns neutral 50 + cached when no frame is set", async () => {
    const a = new BackgroundMotionAnalyzer({
      segmenter: new StubSegmenter() as unknown as MediaPipeSelfieSegmenter,
      segmentEveryN: 1,
    });
    const r = await a.analyze(null, face);
    expect(r.score).toBe(50);
  });

  it("returns neutral 50 during warmup", async () => {
    const a = new BackgroundMotionAnalyzer({
      segmenter: new StubSegmenter() as unknown as MediaPipeSelfieSegmenter,
      segmentEveryN: 1,
      warmupSamples: 10,
    });
    for (let i = 0; i < 8; i++) {
      a.setFrame(makeFrame(120));
      const r = await a.analyze(null, face);
      expect(r.score).toBe(50);
    }
  });

  it("scores near 0 on a perfectly static background (photo)", async () => {
    const a = new BackgroundMotionAnalyzer({
      segmenter: new StubSegmenter() as unknown as MediaPipeSelfieSegmenter,
      segmentEveryN: 1,
      warmupSamples: 5,
      historyLen: 30,
    });
    for (let i = 0; i < 30; i++) {
      a.setFrame(makeFrame(120));
      await a.analyze(null, face);
    }
    a.setFrame(makeFrame(120));
    const r = await a.analyze(null, face);
    expect(r.score).toBeLessThan(10);
  });

  it("scores high when background brightness drifts (real lighting)", async () => {
    const a = new BackgroundMotionAnalyzer({
      segmenter: new StubSegmenter() as unknown as MediaPipeSelfieSegmenter,
      segmentEveryN: 1,
      warmupSamples: 5,
      historyLen: 30,
      gain: 4,
    });
    for (let i = 0; i < 30; i++) {
      a.setFrame(makeFrame(80 + Math.floor((i / 30) * 80)));
      await a.analyze(null, face);
    }
    a.setFrame(makeFrame(160));
    const r = await a.analyze(null, face);
    expect(r.score).toBeGreaterThan(20);
  });

  it("rate-limits via segmentEveryN", async () => {
    const a = new BackgroundMotionAnalyzer({
      segmenter: new StubSegmenter() as unknown as MediaPipeSelfieSegmenter,
      segmentEveryN: 5,
      warmupSamples: 2,
    });
    a.setFrame(makeFrame(140));
    const r1 = await a.analyze(null, face);
    // Frame 1 is NOT a multiple of 5 → cached path returns 50 (no sample yet).
    expect(r1.details.cached).toBe(true);
  });

  it("reset() drops history", async () => {
    const a = new BackgroundMotionAnalyzer({
      segmenter: new StubSegmenter() as unknown as MediaPipeSelfieSegmenter,
      segmentEveryN: 1,
      warmupSamples: 5,
      historyLen: 20,
    });
    for (let i = 0; i < 20; i++) {
      a.setFrame(makeFrame(100 + i));
      await a.analyze(null, face);
    }
    a.reset();
    a.setFrame(makeFrame(150));
    const r = await a.analyze(null, face);
    expect(r.score).toBe(50);
    expect(r.details.warming).toBe(true);
  });
});
