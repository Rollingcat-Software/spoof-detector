// HeavyAnalyzerWorker — runs the 4 heavy synchronous analyzers off the
// main thread so the per-frame analyze() budget on mobile Brave stays
// under ~30 ms instead of 50 ms+.
//
// Offloaded analyzers (each 5–15 ms per frame):
//   * TextureAnalyzer        — Laplacian + HSV + 2D DFT
//   * MoireAnalyzer          — Gabor bank + 2D DFT
//   * ScreenReplayAnalyzer   — Laplacian + 2D DFT + skin mask
//   * DeviceBoundaryAnalyzer — Sobel + hand-rolled Hough scan
//
// Hosted as a Vite-bundled inline worker (see HeavyAnalyzerPool.ts). The
// build pipeline emits the worker as a base64 data URL embedded in the
// library bundle so consumers don't have to serve an extra file.
//
// Brave / mobile constraints (see task spec):
//   * No SharedArrayBuffer (Brave disables it for fingerprint resistance,
//     and we don't ship COOP/COEP headers on Hostinger).
//   * No Atomics.wait or cross-thread sync primitives.
//   * postMessage() with Transferable ImageData / ArrayBuffer for zero-copy.
//   * Code path inside this worker does NOT use OffscreenCanvas — every
//     offloaded analyzer operates on raw ImageData, so worker scope is
//     enough.
//
// Wire protocol:
//   in:  { kind: "analyze", frameId, frameImageData?, faceCropImageData?, face }
//   out: { kind: "result",  frameId, results }            // success
//        { kind: "error",   frameId, message }            // analyzer threw
//
// All buffers are postMessage()'d with the underlying ArrayBuffer in the
// transfer list, which detaches them from the sender — the orchestrator
// must treat the sent ImageData as consumed.

import {
  BBox,
  type AnalyzerResult,
  type FaceROI,
} from "../../domain/models";
// The 4 heavy analyzer modules are imported eagerly INSIDE the worker
// bundle on purpose: a Web Worker is its own JS-execution context with
// its own module graph, so this import does NOT bloat the main-thread
// bundle. The lazy code-splitting on the main-thread path (Phase 5E-1)
// is independent — when the SpoofDetector inline fallback runs them,
// the dynamic `import()` inside SpoofDetector.ensureTexture etc. is
// what wins.
//
// Vite's bundler reports a "static + dynamic" mixed-import warning at
// build time because of this duplication; it is benign: each context
// gets its own chunk (the worker bundle vs the main-thread async
// chunk) and tree-shaking removes neither.
import { DeviceBoundaryAnalyzer } from "../analyzers/DeviceBoundaryAnalyzer";
import { MoireAnalyzer } from "../analyzers/MoireAnalyzer";
import { ScreenReplayAnalyzer } from "../analyzers/ScreenReplayAnalyzer";
import { TextureAnalyzer } from "../analyzers/TextureAnalyzer";

/** Heavy analyzer names that are routed through this worker. */
export const HEAVY_ANALYZER_NAMES = [
  "texture",
  "moire",
  "screen_replay",
  "device_boundary",
] as const;

export type HeavyAnalyzerName = (typeof HEAVY_ANALYZER_NAMES)[number];

/** Plain-object shape used to ship FaceROI across the postMessage boundary. */
export interface FaceROIWire {
  face_id: number;
  bbox: { x1: number; y1: number; x2: number; y2: number };
  confidence: number;
  landmarks?: Float32Array;
}

export interface HeavyAnalyzeRequest {
  kind: "analyze";
  frameId: number;
  /** Full original frame ImageData (needed by ScreenReplay + DeviceBoundary). */
  frameImageData: ImageData | null;
  /** Pre-cropped face region (used by Texture + Moire). */
  faceCropImageData: ImageData | null;
  face: FaceROIWire;
}

export interface HeavyAnalyzeSuccess {
  kind: "result";
  frameId: number;
  results: Record<string, AnalyzerResult>;
}

export interface HeavyAnalyzeError {
  kind: "error";
  frameId: number;
  message: string;
}

export type HeavyAnalyzeResponse = HeavyAnalyzeSuccess | HeavyAnalyzeError;

/** Rebuild a real FaceROI (with a BBox class instance) from the wire payload. */
export function reviveFaceROI(wire: FaceROIWire): FaceROI {
  const out: FaceROI = {
    face_id: wire.face_id,
    bbox: new BBox(wire.bbox.x1, wire.bbox.y1, wire.bbox.x2, wire.bbox.y2),
    confidence: wire.confidence,
  };
  if (wire.landmarks) out.landmarks = wire.landmarks;
  return out;
}

/** Strip a FaceROI down to a postMessage-safe wire object. */
export function toWireFaceROI(face: FaceROI): FaceROIWire {
  const w: FaceROIWire = {
    face_id: face.face_id,
    bbox: {
      x1: face.bbox.x1,
      y1: face.bbox.y1,
      x2: face.bbox.x2,
      y2: face.bbox.y2,
    },
    confidence: face.confidence,
  };
  if (face.landmarks) w.landmarks = face.landmarks;
  return w;
}

/**
 * Run all 4 heavy analyzers on a single face. Exported so the
 * HeavyAnalyzerPool fallback path (when `typeof Worker === "undefined"`)
 * can call the same code on the main thread without booting a worker.
 *
 * Analyzers are instantiated lazily and cached on a per-context object
 * so the per-frame call doesn't pay reconstruction cost.
 */
export interface HeavyAnalyzerContext {
  texture: TextureAnalyzer | null;
  moire: MoireAnalyzer | null;
  screenReplay: ScreenReplayAnalyzer | null;
  deviceBoundary: DeviceBoundaryAnalyzer | null;
}

export function createHeavyAnalyzerContext(): HeavyAnalyzerContext {
  return { texture: null, moire: null, screenReplay: null, deviceBoundary: null };
}

export async function runHeavyAnalyzers(
  ctx: HeavyAnalyzerContext,
  frameImageData: ImageData | null,
  faceCrop: ImageData | null,
  face: FaceROI,
): Promise<Record<string, AnalyzerResult>> {
  if (!ctx.texture) ctx.texture = new TextureAnalyzer();
  if (!ctx.moire) ctx.moire = new MoireAnalyzer();
  if (!ctx.screenReplay) ctx.screenReplay = new ScreenReplayAnalyzer();
  if (!ctx.deviceBoundary) ctx.deviceBoundary = new DeviceBoundaryAnalyzer();

  // ScreenReplay + DeviceBoundary read the FULL frame (not the face crop),
  // so they need the original ImageData fed in via setFrame() each tick.
  if (frameImageData) {
    ctx.screenReplay.setFrame(frameImageData);
    ctx.deviceBoundary.setFrame(frameImageData);
    // Texture falls back to setFrame() only when crop is null — wire it
    // anyway so the analyzer can self-recover from a missing crop.
    ctx.texture.setFrame(frameImageData);
  }

  const out: Record<string, AnalyzerResult> = {};
  const t = await Promise.resolve(ctx.texture.analyze(faceCrop, face));
  out[t.name] = t;
  const m = await Promise.resolve(ctx.moire.analyze(faceCrop, face));
  out[m.name] = m;
  const sr = await Promise.resolve(ctx.screenReplay.analyze(faceCrop, face));
  out[sr.name] = sr;
  const db = await Promise.resolve(ctx.deviceBoundary.analyze(faceCrop, face));
  out[db.name] = db;
  return out;
}

// ---------------------------------------------------------------------------
// Worker bootstrap — only runs when this module is loaded as a Web Worker.
// Detection is purposefully defensive so the same file can be imported
// from main-thread fallback code (HeavyAnalyzerPool synchronous path).
// ---------------------------------------------------------------------------

// Minimal structural type for the worker global. We deliberately avoid
// `lib.webworker.d.ts` (which would clash with `lib.dom.d.ts` on Window
// types) and instead duck-type the two methods we need. `globalThis` is
// always present in modern targets.
interface WorkerLikeGlobal {
  addEventListener(type: "message", listener: (ev: MessageEvent) => void): void;
  postMessage(message: unknown): void;
  importScripts?: unknown;
  document?: unknown;
}

const workerGlobal = globalThis as unknown as WorkerLikeGlobal;
const isWorkerScope =
  typeof workerGlobal.importScripts === "function" &&
  typeof workerGlobal.document === "undefined";

if (isWorkerScope) {
  const ctx = createHeavyAnalyzerContext();
  workerGlobal.addEventListener("message", async (ev: MessageEvent) => {
    const msg = ev.data as HeavyAnalyzeRequest;
    if (!msg || msg.kind !== "analyze") return;
    try {
      const face = reviveFaceROI(msg.face);
      const results = await runHeavyAnalyzers(
        ctx,
        msg.frameImageData,
        msg.faceCropImageData,
        face,
      );
      const response: HeavyAnalyzeSuccess = {
        kind: "result",
        frameId: msg.frameId,
        results,
      };
      workerGlobal.postMessage(response);
    } catch (err) {
      const response: HeavyAnalyzeError = {
        kind: "error",
        frameId: msg.frameId,
        message: err instanceof Error ? err.message : String(err),
      };
      workerGlobal.postMessage(response);
    }
  });
}
