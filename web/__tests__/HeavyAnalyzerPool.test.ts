// HeavyAnalyzerPool tests — exercises the in-line fallback path (vitest
// runs in node so `Worker` is undefined → the pool short-circuits to the
// synchronous path) and the SpoofDetector frame-skip behaviour.
//
// We can't realistically boot the worker bundle from inside vitest (no
// Vite plugin, no DOM Worker). What we CAN do is:
//   * Drive HeavyAnalyzerPool.analyze() directly and assert it returns
//     AnalyzerResults for all 4 analyzer names via the inline path.
//   * Force the inline path even with a Worker global present, to prove
//     the `forceInline` option works.
//   * Drive the SpoofDetector frame-skip scheduler through a stubbed
//     detector + minifasnet so we can count how often the 4 heavy
//     analyzers fire across N frames.

import { describe, expect, it } from "vitest";
import { HeavyAnalyzerPool } from "../src/infrastructure/workers/HeavyAnalyzerPool";
import {
  HEAVY_ANALYZER_NAMES,
  runHeavyAnalyzers,
  createHeavyAnalyzerContext,
} from "../src/infrastructure/workers/HeavyAnalyzerWorker";
import {
  BBox,
  FaceROI,
  type AnalyzerResult,
} from "../src/domain/models";
import { SpoofDetector } from "../src/index";

// ---------------------------------------------------------------------------
// Helpers: build a noisy ImageData so each analyzer has real pixels to chew
// on (uniform inputs hit the no_pixels / blur_floor early-exit branches).
// ---------------------------------------------------------------------------

const W = 96;
const H = 96;

function noisyImage(seed: number, w = W, h = H): ImageData {
  let s = seed >>> 0;
  const rand = (): number => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < data.length; i += 4) {
    const base = 90 + Math.floor(rand() * 100);
    data[i] = Math.min(255, base + 40);
    data[i + 1] = base;
    data[i + 2] = Math.max(0, base - 20);
    data[i + 3] = 255;
  }
  return { data, width: w, height: h, colorSpace: "srgb" } as ImageData;
}

const face: FaceROI = {
  face_id: 0,
  bbox: new BBox(0, 0, W, H),
  confidence: 0.9,
};

// ---------------------------------------------------------------------------

describe("HeavyAnalyzerPool", () => {
  it("returns AnalyzerResult for all 4 heavy analyzer names (inline path)", async () => {
    // vitest node env has no Worker global, so this exercises the
    // synchronous fallback automatically.
    expect(typeof (globalThis as { Worker?: unknown }).Worker).toBe(
      "undefined",
    );

    const pool = new HeavyAnalyzerPool();
    const crop = noisyImage(7);
    const fullFrame = noisyImage(13);
    const results = await pool.analyze(crop, face, fullFrame);

    for (const name of HEAVY_ANALYZER_NAMES) {
      expect(results[name], `missing result for ${name}`).toBeDefined();
      expect(typeof results[name].score).toBe("number");
      expect(results[name].score).toBeGreaterThanOrEqual(0);
      expect(results[name].score).toBeLessThanOrEqual(100);
      expect(results[name].name).toBe(name);
    }
    pool.dispose();
  });

  it("forceInline: true bypasses the Worker constructor even when available", async () => {
    // Inject a fake Worker that throws if its constructor runs — proves
    // the pool never tries to boot it.
    let workerConstructed = false;
    class ThrowingWorker {
      constructor() {
        workerConstructed = true;
        throw new Error("worker boot must not happen when forceInline=true");
      }
    }
    const prev = (globalThis as { Worker?: unknown }).Worker;
    (globalThis as { Worker?: unknown }).Worker =
      ThrowingWorker as unknown as typeof Worker;
    try {
      const pool = new HeavyAnalyzerPool({ forceInline: true });
      const results = await pool.analyze(noisyImage(1), face, noisyImage(2));
      expect(workerConstructed).toBe(false);
      expect(Object.keys(results).sort()).toEqual(
        [...HEAVY_ANALYZER_NAMES].sort(),
      );
      pool.dispose();
    } finally {
      if (prev === undefined) {
        delete (globalThis as { Worker?: unknown }).Worker;
      } else {
        (globalThis as { Worker?: unknown }).Worker = prev;
      }
    }
  });

  it("runHeavyAnalyzers (shared with worker scope) produces stable results", async () => {
    // Run twice with the same context — second pass should reuse cached
    // analyzer instances (no exception) and still return 4 named results.
    const ctx = createHeavyAnalyzerContext();
    const out1 = await runHeavyAnalyzers(
      ctx,
      noisyImage(11),
      noisyImage(11),
      face,
    );
    const out2 = await runHeavyAnalyzers(
      ctx,
      noisyImage(11),
      noisyImage(11),
      face,
    );
    expect(Object.keys(out1).sort()).toEqual([...HEAVY_ANALYZER_NAMES].sort());
    expect(Object.keys(out2).sort()).toEqual([...HEAVY_ANALYZER_NAMES].sort());
    // The screen-replay / texture / device-boundary analyzers are not
    // stateful across calls (no temporal accumulator), so identical
    // inputs give identical scores.
    expect(out1.texture.score).toBeCloseTo(out2.texture.score, 5);
    expect(out1.moire.score).toBeCloseTo(out2.moire.score, 5);
  });
});

// ---------------------------------------------------------------------------
// SpoofDetector frame-skip integration test.
//
// We can't drive a real SpoofDetector inside vitest (MediaPipe + ONNX both
// need a browser). What we CAN do is construct one with bogus URLs, then
// intercept its private fields with controlled stubs that count
// invocations. The worker-pool path is replaced with a counting stub so
// we can assert it's called exactly every Nth frame.
// ---------------------------------------------------------------------------

describe("SpoofDetector frame-skip scheduler", () => {
  it("invokes the heavy pool every N=3 frames over a 10-frame window", async () => {
    const det = new SpoofDetector({
      miniFasNetModelUrl: "noop://minifasnet",
      faceLandmarkerTaskUrl: "noop://facelandmarker",
      heavyAnalyzerFrameSkip: 3,
      gateFrameSkip: 5,
      enableHeavyWorker: true,
      enableFaceUsabilityGate: false, // out of scope here
    });

    // Stub the face detector to return a stable single face per frame.
    const stableFace: FaceROI = {
      face_id: 42,
      bbox: new BBox(10, 10, 74, 74),
      confidence: 0.99,
    };
    (det as unknown as { detector: { detect: () => Promise<FaceROI[]> } }).detector =
      { detect: async () => [stableFace] };

    // Stub MiniFASNet to record a no-op result instantly.
    (det as unknown as {
      minifasnet: {
        setFrame: () => void;
        analyze: () => Promise<AnalyzerResult>;
      };
    }).minifasnet = {
      setFrame: () => undefined,
      analyze: async () =>
        ({
          name: "minifasnet",
          score: 80,
          details: {},
          elapsed_ms: 0,
        }) as AnalyzerResult,
    };

    // Replace the HeavyAnalyzerPool field with a counting stub. Returns
    // the 4 expected analyzer names so the merge path is exercised.
    let heavyCalls = 0;
    const heavyResult: Record<string, AnalyzerResult> = {
      texture: { name: "texture", score: 80, details: {}, elapsed_ms: 0 },
      moire: { name: "moire", score: 70, details: {}, elapsed_ms: 0 },
      screen_replay: {
        name: "screen_replay",
        score: 90,
        details: {},
        elapsed_ms: 0,
      },
      device_boundary: {
        name: "device_boundary",
        score: 75,
        details: {},
        elapsed_ms: 0,
      },
    };
    const countingPool = {
      analyze: async () => {
        heavyCalls += 1;
        return heavyResult;
      },
      dispose: () => undefined,
    };
    (det as unknown as { heavyPool: typeof countingPool }).heavyPool =
      countingPool;

    // Disable every analyzer toggle except the heavies so the inline
    // fast-path doesn't kick (avoids needing landmarks / temporal state).
    (det as unknown as {
      toggles: Record<string, boolean>;
    }).toggles = {
      landmarkVariance: false,
      blink: false,
      deviceBoundary: true,
      microTremor: false,
      screenFlicker: false,
      rppg: false,
      moire: true,
      texture: true,
      screenReplay: true,
      faceUsabilityGate: false,
    };

    // Build a tiny canvas-shaped object the detector accepts. The detector
    // stub ignores it and the heavy pool stub never touches it either, so
    // a duck-typed canvas with width/height + a getContext stub is enough
    // to satisfy `instanceof HTMLCanvasElement` … which it WON'T be in
    // node. So we feed ImageData instead — the detector branches on
    // `instanceof HTMLCanvasElement` and falls through to OffscreenCanvas
    // path. Node has no OffscreenCanvas either, which is why we stubbed
    // the face detector above (skipping the canvas path entirely).
    //
    // Trick: provide a global OffscreenCanvas stub so the early-stage
    // OffscreenCanvas() call doesn't throw.
    class FakeOffscreen {
      width: number;
      height: number;
      constructor(w: number, h: number) {
        this.width = w;
        this.height = h;
      }
      getContext() {
        return {
          putImageData: () => undefined,
          drawImage: () => undefined,
          getImageData: () => noisyImage(0, this.width, this.height),
        };
      }
    }
    (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas =
      FakeOffscreen as unknown as typeof OffscreenCanvas;
    class FakeHTMLCanvasElement {}
    (globalThis as { HTMLCanvasElement?: unknown }).HTMLCanvasElement =
      FakeHTMLCanvasElement as unknown as typeof HTMLCanvasElement;

    const frame = noisyImage(99, 64, 64);
    try {
      for (let i = 0; i < 10; i += 1) {
        // eslint-disable-next-line no-await-in-loop
        await det.analyzeFrame(frame);
      }
    } finally {
      delete (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas;
      delete (globalThis as { HTMLCanvasElement?: unknown }).HTMLCanvasElement;
    }

    // Heavy schedule with N=3 runs on frames 1, 4, 7, 10 → 4 calls
    // across a 10-frame window.
    expect(heavyCalls).toBe(4);
  });

  it("honours enableHeavyWorker: false by never instantiating the pool", async () => {
    const det = new SpoofDetector({
      miniFasNetModelUrl: "noop://minifasnet",
      faceLandmarkerTaskUrl: "noop://facelandmarker",
      enableHeavyWorker: false,
      heavyAnalyzerFrameSkip: 2,
      gateFrameSkip: 100,
      enableFaceUsabilityGate: false,
      // Turn off the four heavies so the inline path is also a no-op —
      // we just want to assert the pool field stays null.
      enableMoire: false,
      enableTexture: false,
      enableScreenReplay: false,
      enableDeviceBoundary: false,
    });

    // Stub detector + minifasnet (see above).
    (det as unknown as { detector: { detect: () => Promise<FaceROI[]> } }).detector =
      { detect: async () => [] };
    (det as unknown as {
      minifasnet: {
        setFrame: () => void;
        analyze: () => Promise<AnalyzerResult>;
      };
    }).minifasnet = {
      setFrame: () => undefined,
      analyze: async () =>
        ({
          name: "minifasnet",
          score: 80,
          details: {},
          elapsed_ms: 0,
        }) as AnalyzerResult,
    };

    class FakeOffscreen {
      width: number;
      height: number;
      constructor(w: number, h: number) {
        this.width = w;
        this.height = h;
      }
      getContext() {
        return {
          putImageData: () => undefined,
          drawImage: () => undefined,
          getImageData: () => noisyImage(0, this.width, this.height),
        };
      }
    }
    (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas =
      FakeOffscreen as unknown as typeof OffscreenCanvas;
    // The detector branches on `input instanceof HTMLCanvasElement`; that
    // global doesn't exist in node. Provide a placeholder class so the
    // `instanceof` check evaluates (and returns false, since we feed
    // ImageData).
    class FakeHTMLCanvasElement {}
    (globalThis as { HTMLCanvasElement?: unknown }).HTMLCanvasElement =
      FakeHTMLCanvasElement as unknown as typeof HTMLCanvasElement;
    try {
      await det.analyzeFrame(noisyImage(1, 64, 64));
      await det.analyzeFrame(noisyImage(2, 64, 64));
    } finally {
      delete (globalThis as { OffscreenCanvas?: unknown }).OffscreenCanvas;
      delete (globalThis as { HTMLCanvasElement?: unknown }).HTMLCanvasElement;
    }

    expect(
      (det as unknown as { heavyPool: HeavyAnalyzerPool | null }).heavyPool,
    ).toBeNull();
  });
});
