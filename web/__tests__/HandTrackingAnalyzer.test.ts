import { describe, expect, it } from "vitest";
import { HandTrackingAnalyzer } from "../src/infrastructure/analyzers/HandTrackingAnalyzer";
import { BBox, FaceROI } from "../src/domain/models";
import type {
  DetectedHand,
  MediaPipeHandDetector,
} from "../src/infrastructure/detection/MediaPipeHandDetector";

function makeHand(x: number, y: number, handedness = "Right"): DetectedHand {
  // 21 landmarks; only landmark 0 (wrist) is used by the analyzer score.
  const lms = [] as Array<{ x: number; y: number; z: number }>;
  for (let i = 0; i < 21; i++) lms.push({ x, y, z: 0 });
  return { landmarks: lms, handedness, handednessScore: 0.99 };
}

class StubDetector {
  constructor(
    public framesToReturn: DetectedHand[][] | (() => DetectedHand[]) = [[]],
    public idx = 0,
  ) {}
  async warmup(): Promise<void> {}
  async detect(): Promise<DetectedHand[]> {
    if (typeof this.framesToReturn === "function") return this.framesToReturn();
    const r = this.framesToReturn[Math.min(this.idx, this.framesToReturn.length - 1)];
    this.idx += 1;
    return r;
  }
  async close(): Promise<void> {}
}

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(0, 0, 192, 108),
  confidence: 0.9,
};

const FAKE_FRAME = { width: 192, height: 108 } as unknown as ImageData;

describe("HandTrackingAnalyzer", () => {
  it("returns cached neutral 50 on skipped frames", async () => {
    const a = new HandTrackingAnalyzer({
      detector: new StubDetector([]) as unknown as MediaPipeHandDetector,
      detectEveryN: 4,
    });
    a.setFrame(FAKE_FRAME as never);
    const r = await a.analyze(null, face);
    expect(r.details.cached).toBe(true);
    expect(r.score).toBe(50);
  });

  it("returns neutral 50 when no hands are detected", async () => {
    const a = new HandTrackingAnalyzer({
      detector: new StubDetector([[]]) as unknown as MediaPipeHandDetector,
      detectEveryN: 1,
    });
    a.setFrame(FAKE_FRAME as never);
    const r = await a.analyze(null, face);
    expect(r.score).toBe(50);
    expect(r.details.hand_count).toBe(0);
  });

  it("scores low on a static hand (operator holding photo with hand visible)", async () => {
    const stub = new StubDetector(
      Array(30).fill([makeHand(0.4, 0.6)]),
    );
    const a = new HandTrackingAnalyzer({
      detector: stub as unknown as MediaPipeHandDetector,
      detectEveryN: 1,
      warmupSamples: 5,
      historyLen: 30,
      gain: 200,
    });
    for (let i = 0; i < 30; i++) {
      a.setFrame(FAKE_FRAME as never);
      await a.analyze(null, face);
    }
    a.setFrame(FAKE_FRAME as never);
    const r = await a.analyze(null, face);
    expect(r.score).toBeLessThan(10);
  });

  it("scores higher on a moving hand (natural gesture)", async () => {
    // Simulate the wrist moving in a small arc each frame.
    const motion = Array.from({ length: 30 }, (_, i) => [
      makeHand(0.4 + 0.1 * Math.sin(i / 5), 0.6 + 0.1 * Math.cos(i / 5)),
    ]);
    const stub = new StubDetector(motion);
    const a = new HandTrackingAnalyzer({
      detector: stub as unknown as MediaPipeHandDetector,
      detectEveryN: 1,
      warmupSamples: 5,
      historyLen: 30,
      gain: 400,
    });
    for (let i = 0; i < 30; i++) {
      a.setFrame(FAKE_FRAME as never);
      await a.analyze(null, face);
    }
    a.setFrame(FAKE_FRAME as never);
    const r = await a.analyze(null, face);
    expect(r.score).toBeGreaterThan(10);
  });

  it("flags >2 hands as a deepfake-style anomaly and caps the score", async () => {
    const threeHands = [makeHand(0.3, 0.5), makeHand(0.5, 0.5), makeHand(0.7, 0.5)];
    const stub = new StubDetector(Array(30).fill(threeHands));
    const a = new HandTrackingAnalyzer({
      detector: stub as unknown as MediaPipeHandDetector,
      detectEveryN: 1,
      warmupSamples: 5,
      historyLen: 30,
    });
    for (let i = 0; i < 30; i++) {
      a.setFrame(FAKE_FRAME as never);
      await a.analyze(null, face);
    }
    a.setFrame(FAKE_FRAME as never);
    const r = await a.analyze(null, face);
    expect(r.details.anomaly_third_hand).toBe(true);
    expect(r.score).toBeLessThanOrEqual(20);
  });

  it("reset() drops state", async () => {
    const stub = new StubDetector([[makeHand(0.5, 0.5)]]);
    const a = new HandTrackingAnalyzer({
      detector: stub as unknown as MediaPipeHandDetector,
      detectEveryN: 1,
    });
    for (let i = 0; i < 5; i++) {
      a.setFrame(FAKE_FRAME as never);
      await a.analyze(null, face);
    }
    a.reset();
    a.setFrame(FAKE_FRAME as never);
    const r = await a.analyze(null, face);
    expect(r.score).toBe(50);
  });
});
