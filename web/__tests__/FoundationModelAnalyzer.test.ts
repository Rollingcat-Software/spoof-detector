// FoundationModelAnalyzer tests.
//
// What is unit-tested here (no GPU / no canvas needed):
//   * rgbaToRgbNchwFloat32Normalized — the RGB + ImageNet-normalize input
//     contract shared with tools/train_fas_adapter.py.
//   * foundationScoreFromLogits — the [spoof, real] -> 0-100 score mapping.
//   * warmup() + the null-source guard, with onnxruntime-web mocked.
//
// The full crop -> resize -> ORT-run path uses OffscreenCanvas + a real WebGPU
// session, neither of which exists under vitest/node — so (exactly like
// MiniFASNetAnalyzer, which ships untested for the same reason) that path is
// proven by the Python ONNX round-trip (tools/train_fas_adapter.py --smoke) and
// a browser smoke run, not here.

import { describe, expect, it, vi } from "vitest";
import {
  rgbaToRgbNchwFloat32Normalized,
  IMAGENET_MEAN,
  IMAGENET_STD,
} from "../src/utils/imageOps";
import {
  FoundationModelAnalyzer,
  foundationScoreFromLogits,
} from "../src/infrastructure/analyzers/FoundationModelAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";

// Minimal onnxruntime-web mock so warmup() can build a "session" without a real
// WASM/WebGPU runtime. run() is never reached by the tests below (the null-source
// guard short-circuits first), but a sane stub keeps the contract honest.
vi.mock("onnxruntime-web", () => ({
  InferenceSession: {
    create: vi.fn(async () => ({
      inputNames: ["pixel_values"],
      outputNames: ["logits"],
      run: vi.fn(async () => ({ logits: { data: new Float32Array([0, 1]) } })),
    })),
  },
  Tensor: class {
    constructor(
      public type: string,
      public data: unknown,
      public dims: number[],
    ) {}
  },
  env: { wasm: {} },
}));

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(40, 40, 120, 120),
  confidence: 0.9,
};

describe("rgbaToRgbNchwFloat32Normalized", () => {
  it("produces planar RGB normalized with ImageNet stats", () => {
    // 2x1 image: px0 = (255,0,128), px1 = (0,255,255).
    const rgba = new Uint8ClampedArray([255, 0, 128, 255, 0, 255, 255, 255]);
    const out = rgbaToRgbNchwFloat32Normalized(rgba, 2, 1);
    expect(out.length).toBe(6); // 3 planes * 2 px
    const [mR, mG, mB] = IMAGENET_MEAN;
    const [sR, sG, sB] = IMAGENET_STD;
    // R plane (indices 0,1)
    expect(out[0]).toBeCloseTo((255 / 255 - mR) / sR, 5);
    expect(out[1]).toBeCloseTo((0 / 255 - mR) / sR, 5);
    // G plane (indices 2,3)
    expect(out[2]).toBeCloseTo((0 / 255 - mG) / sG, 5);
    expect(out[3]).toBeCloseTo((255 / 255 - mG) / sG, 5);
    // B plane (indices 4,5)
    expect(out[4]).toBeCloseTo((128 / 255 - mB) / sB, 5);
    expect(out[5]).toBeCloseTo((255 / 255 - mB) / sB, 5);
  });

  it("honors custom mean/std", () => {
    const rgba = new Uint8ClampedArray([255, 255, 255, 255]);
    const out = rgbaToRgbNchwFloat32Normalized(rgba, 1, 1, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    // (1 - 0.5) / 0.5 == 1 on every channel.
    expect(out[0]).toBeCloseTo(1, 6);
    expect(out[1]).toBeCloseTo(1, 6);
    expect(out[2]).toBeCloseTo(1, 6);
  });
});

describe("foundationScoreFromLogits", () => {
  it("maps real-dominant logits to a high score (> 50)", () => {
    const r = foundationScoreFromLogits([-2, 2]); // index1=real wins
    expect(r.isReal).toBe(true);
    expect(r.pReal).toBeGreaterThan(r.pSpoof);
    expect(r.score).toBeGreaterThan(50);
    expect(r.score).toBeLessThanOrEqual(100);
  });

  it("maps spoof-dominant logits to a low score (< 50)", () => {
    const r = foundationScoreFromLogits([3, -1]); // index0=spoof wins
    expect(r.isReal).toBe(false);
    expect(r.pSpoof).toBeGreaterThan(r.pReal);
    expect(r.score).toBeLessThan(50);
    expect(r.score).toBeGreaterThanOrEqual(0);
  });

  it("maps equal logits to ~50 (max uncertainty)", () => {
    const r = foundationScoreFromLogits([1, 1]);
    expect(r.pSpoof).toBeCloseTo(0.5, 6);
    expect(r.score).toBeCloseTo(50, 5);
  });
});

describe("FoundationModelAnalyzer", () => {
  it("warms up via the mocked ORT session", async () => {
    const a = new FoundationModelAnalyzer({ modelUrl: "/models/fas_head.onnx" });
    await expect(a.warmup()).resolves.toBeUndefined();
  });

  it("returns no_frame_no_crop when no frame set and no crop given", async () => {
    const a = new FoundationModelAnalyzer({ modelUrl: "/models/fas_head.onnx" });
    const r = await a.analyze(null, face);
    expect(r.score).toBe(50.0);
    expect(r.details.error).toBe("no_frame_no_crop");
  });
});
