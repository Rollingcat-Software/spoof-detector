// Phase 5E-3: Tiny in-page accuracy harness for the /amispoof/ tester.
//
// Loads a hand-curated, copyright-safe micro-mirror of CASIA-FASD-style
// frames (5 live + 5 spoof) from a same-origin path, runs each through the
// full SpoofDetector pipeline for a short warmup window, captures the
// session verdict, and returns an accuracy + per-sample breakdown.
//
// Why same-origin only:
//   Brave (and Safari ITP) treat cross-origin image fetches as opaque under
//   `strict-origin-when-cross-origin`. An `<img crossorigin="anonymous">`
//   draw onto a canvas would taint it and `getImageData()` would throw
//   `SecurityError`. Sample URLs are assumed to be served from the same
//   host as the amispoof page (typically `./samples/*.jpg`).
//
// Why 30 warmup frames per sample:
//   The SessionEngine ignores verdicts until `frames_analyzed >= 30` (it's
//   in "warming_up" state before that). Feeding the same still 30× gives
//   the temporal analyzers (BlinkAnalyzer EAR, MicroTremor FFT, etc.) a
//   stable history to operate on without artificial extreme verdicts.
//
// Output contract:
//   accuracy is a 0-1 ratio (live verdicts on live samples + spoof verdicts
//   on spoof samples) / total. perSample[i].got is whatever the engine
//   reported in lastVerdict.is_live.

import type { SpoofDetector } from "../index";

/** Per-sample input for the bench. */
export interface CasiaFasdBenchSample {
  /** Same-origin URL to the image. */
  url: string;
  /** Ground-truth label. */
  expected: "live" | "spoof";
}

/** One row of the bench output. */
export interface CasiaFasdBenchRow {
  url: string;
  expected: "live" | "spoof";
  got: "live" | "spoof";
  /** SpoofDetector's `verdict.confidence` (0-1). */
  confidence: number;
}

/** Aggregate result. */
export interface CasiaFasdBenchResult {
  /** Fraction of correct verdicts (0-1). */
  accuracy: number;
  /** correct / total integer counts, useful for "8/10 correct" headlines. */
  correct: number;
  total: number;
  perSample: CasiaFasdBenchRow[];
}

/**
 * Default frame budget per sample. Matches the SessionEngine warmup window
 * (the verdict before this many frames is always "warming_up").
 */
const DEFAULT_WARMUP_FRAMES = 30;

/** Default canvas dims — matches the amispoof live-camera path. */
const CANVAS_W = 640;
const CANVAS_H = 480;

/**
 * Convenience overload that accepts either `string[]` (10 URLs, the first
 * 5 are assumed live and the next 5 spoof) OR the explicit labeled form.
 * The string[] form mirrors the integration brief — callers pass 5 live +
 * 5 spoof URLs in that order.
 */
export async function runCasiaFasdMicroBench(
  detector: SpoofDetector,
  sampleUrls: string[],
  options?: { warmupFrames?: number; canvasW?: number; canvasH?: number },
): Promise<CasiaFasdBenchResult>;
export async function runCasiaFasdMicroBench(
  detector: SpoofDetector,
  samples: CasiaFasdBenchSample[],
  options?: { warmupFrames?: number; canvasW?: number; canvasH?: number },
): Promise<CasiaFasdBenchResult>;
export async function runCasiaFasdMicroBench(
  detector: SpoofDetector,
  samplesOrUrls: string[] | CasiaFasdBenchSample[],
  options: { warmupFrames?: number; canvasW?: number; canvasH?: number } = {},
): Promise<CasiaFasdBenchResult> {
  const samples: CasiaFasdBenchSample[] = normalizeSamples(samplesOrUrls);
  const warmupFrames = Math.max(
    1,
    Math.floor(options.warmupFrames ?? DEFAULT_WARMUP_FRAMES),
  );
  const canvasW = Math.max(1, Math.floor(options.canvasW ?? CANVAS_W));
  const canvasH = Math.max(1, Math.floor(options.canvasH ?? CANVAS_H));

  const perSample: CasiaFasdBenchRow[] = [];

  for (const sample of samples) {
    const canvas = await loadImageToCanvas(sample.url, canvasW, canvasH);
    detector.reset();
    // Feed the same still WARMUP_FRAMES times so SessionEngine clears its
    // warming_up gate and the temporal analyzers have history to work on.
    for (let i = 0; i < warmupFrames; i += 1) {
      // eslint-disable-next-line no-await-in-loop
      await detector.analyzeFrame(canvas);
    }
    const verdict = detector.getVerdict();
    perSample.push({
      url: sample.url,
      expected: sample.expected,
      got: verdict.is_live ? "live" : "spoof",
      confidence: verdict.confidence,
    });
  }

  const correct = perSample.reduce(
    (acc, row) => acc + (row.got === row.expected ? 1 : 0),
    0,
  );
  return {
    accuracy: samples.length > 0 ? correct / samples.length : 0,
    correct,
    total: samples.length,
    perSample,
  };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function normalizeSamples(
  samplesOrUrls: string[] | CasiaFasdBenchSample[],
): CasiaFasdBenchSample[] {
  if (samplesOrUrls.length === 0) return [];
  if (typeof samplesOrUrls[0] === "string") {
    const urls = samplesOrUrls as string[];
    // Convention: first half live, second half spoof. For 10 URLs that's
    // the 5 live + 5 spoof shape the integration brief specifies.
    const half = Math.floor(urls.length / 2);
    return urls.map((url, i) => ({
      url,
      expected: i < half ? ("live" as const) : ("spoof" as const),
    }));
  }
  return samplesOrUrls as CasiaFasdBenchSample[];
}

/**
 * Load `url` into an `HTMLImageElement`, draw onto a `width × height`
 * canvas, return the canvas. The detector's `analyzeFrame()` accepts
 * `HTMLCanvasElement` directly.
 *
 * Throws if the image fails to load or the canvas context is unavailable.
 */
async function loadImageToCanvas(
  url: string,
  width: number,
  height: number,
): Promise<HTMLCanvasElement> {
  const img = await loadImage(url);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("CasiaFasdMicroBench: 2D canvas context unavailable");
  }
  // Letterbox-fit so non-4:3 samples aren't grossly stretched (the analyzer
  // pipeline is bbox-relative so a small black border doesn't change the
  // verdict — but a horizontal squish does).
  const scale = Math.min(width / img.width, height / img.height);
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  const dx = Math.floor((width - w) / 2);
  const dy = Math.floor((height - h) / 2);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(img, dx, dy, w, h);
  return canvas;
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    // Same-origin only — see file-top note. We do NOT set crossOrigin so
    // a third-party URL would fail to draw rather than silently produce
    // an opaque-tainted canvas the analyzer can't read pixels from.
    img.onload = () => resolve(img);
    img.onerror = () =>
      reject(new Error(`CasiaFasdMicroBench: failed to load ${url}`));
    img.src = url;
  });
}
