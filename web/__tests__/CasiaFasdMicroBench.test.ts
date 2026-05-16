// Phase 5E-3: scoring-math test for CasiaFasdMicroBench.
//
// The harness drives DOM (Image, document.createElement('canvas')) which
// vitest's default Node environment lacks, so we stub the minimum surface:
//   * global.Image — onload fires synchronously to a fake image
//   * global.document.createElement('canvas') — returns a canvas with a
//     no-op 2D context.
//
// Once those are in place we drive the harness with a FakeSpoofDetector
// that returns a deterministic verdict per sample, then assert the
// accuracy + correct/total math is wired correctly.

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  runCasiaFasdMicroBench,
  CasiaFasdBenchSample,
} from "../src/validation/CasiaFasdMicroBench";
import type { SpoofDetector } from "../src/index";
import type { SessionVerdict } from "../src/domain/session";
import { SpoofCategory } from "../src/domain/models";

// ---------------------------------------------------------------------------
// Stubs for the DOM surface the harness touches.
// ---------------------------------------------------------------------------

class FakeImage {
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  width = 100;
  height = 100;
  private _src = "";
  get src(): string {
    return this._src;
  }
  set src(v: string) {
    this._src = v;
    // Fire onload on the next microtask so callers that attach handlers
    // *after* setting src still receive the event.
    queueMicrotask(() => this.onload?.());
  }
}

function fakeCanvas(): HTMLCanvasElement {
  // Vitest's node env has no canvas — provide just enough surface for the
  // harness's drawImage / fillRect / getContext calls.
  const ctx = {
    drawImage: () => undefined,
    fillRect: () => undefined,
    set fillStyle(_v: string) {
      /* noop */
    },
  };
  return {
    width: 0,
    height: 0,
    getContext: () => ctx,
  } as unknown as HTMLCanvasElement;
}

// ---------------------------------------------------------------------------
// A SpoofDetector double that returns a scripted is_live per call to
// getVerdict(). analyzeFrame() / reset() are no-ops.
// ---------------------------------------------------------------------------
function makeFakeDetector(verdicts: Array<{ is_live: boolean; confidence: number }>): SpoofDetector {
  // Track which sample we're on via getVerdict() call counter — the harness
  // calls getVerdict() exactly once per sample (after the warmup loop).
  // reset() is called per-sample too but BEFORE the analyze loop, so we
  // can't key off it (the harness's first reset would put us at sample 1
  // instead of 0). Using getVerdict() as the counter avoids that race.
  let verdictCallIdx = 0;
  const det = {
    async analyzeFrame() {
      return {
        frame_id: 0,
        faces: [],
        classifications: {},
        frame_signals: {},
        total_ms: 0,
      };
    },
    getVerdict(): SessionVerdict {
      const v =
        verdicts[verdictCallIdx] ?? verdicts[verdicts.length - 1];
      verdictCallIdx += 1;
      return {
        is_live: v.is_live,
        confidence: v.confidence,
        dominant_threat: v.is_live ? null : SpoofCategory.STATIC_IMAGE,
        category_scores: {} as SessionVerdict["category_scores"],
        incidents: [],
        session_duration_sec: 1,
        frames_analyzed: 30,
        face_detected_ratio: 1,
        blink_count: 0,
        estimated_bpm: null,
        identity_changes: 0,
        summary: v.is_live ? "LIVE" : "SPOOF",
      };
    },
    reset() {
      /* no-op for the fake — the verdict counter is keyed off getVerdict(). */
    },
  };
  return det as unknown as SpoofDetector;
}

// ---------------------------------------------------------------------------

describe("CasiaFasdMicroBench", () => {
  beforeEach(() => {
    (globalThis as unknown as { Image: typeof FakeImage }).Image = FakeImage;
    (globalThis as unknown as { document: { createElement: (t: string) => HTMLCanvasElement } }).document =
      { createElement: (_t: string) => fakeCanvas() };
  });
  afterEach(() => {
    delete (globalThis as unknown as { Image?: unknown }).Image;
    delete (globalThis as unknown as { document?: unknown }).document;
  });

  it("scores 10/10 perfect run as accuracy=1.0", async () => {
    // 5 live samples (first half) returning is_live=true,
    // 5 spoof samples (second half) returning is_live=false.
    const detector = makeFakeDetector([
      ...Array.from({ length: 5 }, () => ({ is_live: true, confidence: 0.9 })),
      ...Array.from({ length: 5 }, () => ({ is_live: false, confidence: 0.85 })),
    ]);
    const urls = Array.from({ length: 10 }, (_, i) => `./samples/${i}.jpg`);
    const result = await runCasiaFasdMicroBench(detector, urls, {
      warmupFrames: 1,
    });
    expect(result.total).toBe(10);
    expect(result.correct).toBe(10);
    expect(result.accuracy).toBe(1);
    expect(result.perSample).toHaveLength(10);
    expect(result.perSample[0].expected).toBe("live");
    expect(result.perSample[5].expected).toBe("spoof");
  });

  it("scores 0/10 inverted run as accuracy=0.0", async () => {
    // All wrong: live samples return spoof, spoof samples return live.
    const detector = makeFakeDetector([
      ...Array.from({ length: 5 }, () => ({ is_live: false, confidence: 0.7 })),
      ...Array.from({ length: 5 }, () => ({ is_live: true, confidence: 0.6 })),
    ]);
    const urls = Array.from({ length: 10 }, (_, i) => `./samples/${i}.jpg`);
    const result = await runCasiaFasdMicroBench(detector, urls, {
      warmupFrames: 1,
    });
    expect(result.correct).toBe(0);
    expect(result.accuracy).toBe(0);
  });

  it("scores 8/10 = 0.8 partial run with per-sample breakdown", async () => {
    // 4 live correct, 1 live wrong, 4 spoof correct, 1 spoof wrong = 8/10.
    const detector = makeFakeDetector([
      { is_live: true, confidence: 0.9 },
      { is_live: true, confidence: 0.9 },
      { is_live: true, confidence: 0.9 },
      { is_live: true, confidence: 0.9 },
      { is_live: false, confidence: 0.4 }, // live sample misclassified
      { is_live: false, confidence: 0.8 },
      { is_live: false, confidence: 0.8 },
      { is_live: false, confidence: 0.8 },
      { is_live: false, confidence: 0.8 },
      { is_live: true, confidence: 0.3 }, // spoof sample misclassified
    ]);
    const urls = Array.from({ length: 10 }, (_, i) => `./samples/${i}.jpg`);
    const result = await runCasiaFasdMicroBench(detector, urls, {
      warmupFrames: 1,
    });
    expect(result.correct).toBe(8);
    expect(result.total).toBe(10);
    expect(result.accuracy).toBeCloseTo(0.8, 5);
    // Spot-check that the misclassified rows are flagged correctly.
    expect(result.perSample[4]).toMatchObject({
      expected: "live",
      got: "spoof",
    });
    expect(result.perSample[9]).toMatchObject({
      expected: "spoof",
      got: "live",
    });
  });

  it("accepts explicit labeled samples (overload form)", async () => {
    const detector = makeFakeDetector([
      { is_live: true, confidence: 0.95 },
      { is_live: false, confidence: 0.95 },
    ]);
    const samples: CasiaFasdBenchSample[] = [
      { url: "./samples/a.jpg", expected: "live" },
      { url: "./samples/b.jpg", expected: "spoof" },
    ];
    const result = await runCasiaFasdMicroBench(detector, samples, {
      warmupFrames: 1,
    });
    expect(result.correct).toBe(2);
    expect(result.accuracy).toBe(1);
    expect(result.perSample[0].confidence).toBeCloseTo(0.95);
  });

  it("returns accuracy 0 for an empty sample list (no division-by-zero)", async () => {
    const detector = makeFakeDetector([{ is_live: true, confidence: 1 }]);
    const result = await runCasiaFasdMicroBench(detector, [] as string[], {
      warmupFrames: 1,
    });
    expect(result.total).toBe(0);
    expect(result.correct).toBe(0);
    expect(result.accuracy).toBe(0);
    expect(result.perSample).toEqual([]);
  });
});
