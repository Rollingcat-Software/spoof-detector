// HeavyAnalyzerPool — main-thread wrapper that owns the heavy-analyzer
// Web Worker and exposes a tiny `analyze(crop, face, frame)` promise API.
//
// Boot pattern (per task spec):
//   * Worker is built from an inline blob URL. The worker code is produced
//     at build time by Vite's `?worker&inline` import, which base64-inlines
//     the entire worker bundle (including the 4 heavy analyzers) into the
//     library bundle. The consumer therefore does NOT need to host a
//     separate worker.js file alongside spoof-detector.js.
//   * The `?worker&inline` module is loaded via a dynamic `import()` so
//     test environments (vitest in node, no Worker global) never
//     attempt to resolve the Vite-specific virtual module — the
//     `shouldRunInline()` guard short-circuits the path long before the
//     import is reached.
//   * Worker is lazy-created on first call so importing the SpoofDetector
//     does not spin up a thread for callers that only want fast analyzers.
//
// SAB-free design: every cross-thread payload travels through structured
// clone with ArrayBuffer transferables. Frame ImageData is moved (not
// copied) by listing its `.data.buffer` in the transfer list. We do NOT
// touch SharedArrayBuffer or Atomics.wait — both are unavailable in
// mobile Brave without COOP/COEP headers (which Hostinger does not set).
//
// Brave fallback notes:
//   * No OffscreenCanvas is required inside the worker because every
//     offloaded analyzer operates on raw ImageData buffers.
//   * If the host environment fails to instantiate the worker for any
//     reason (e.g. Brave Strict shield, Worker CSP violation), the pool
//     transparently switches to the synchronous in-line path on the next
//     call. Each subsequent call is a constant-cost branch — no per-frame
//     Worker construction.

import type { AnalyzerResult, FaceROI } from "../../domain/models";
import {
  createHeavyAnalyzerContext,
  runHeavyAnalyzers,
  toWireFaceROI,
  type HeavyAnalyzeRequest,
  type HeavyAnalyzeResponse,
  type HeavyAnalyzerContext,
} from "./HeavyAnalyzerWorker";

export interface HeavyAnalyzerPoolOptions {
  /**
   * Force the synchronous in-line fallback even when a Worker constructor
   * is available. Useful for benchmarking and for environments that have
   * `Worker` but no actual worker-spawn capability (Safari Lockdown Mode,
   * Brave Strict shield, some Capacitor wrappers).
   */
  forceInline?: boolean;
}

interface PendingCall {
  resolve: (results: Record<string, AnalyzerResult>) => void;
  reject: (err: Error) => void;
  /** Wall-clock deadline; rejects with a timeout error if the worker stalls. */
  timer: ReturnType<typeof setTimeout>;
}

const WORKER_CALL_TIMEOUT_MS = 5_000;

export class HeavyAnalyzerPool {
  private readonly forceInline: boolean;
  private worker: Worker | null = null;
  /** Cached promise so concurrent first-callers all await the same boot. */
  private workerBootPromise: Promise<Worker | null> | null = null;
  private workerBootFailed = false;
  private nextCallId = 1;
  private readonly pending = new Map<number, PendingCall>();
  /** Main-thread analyzer cache for the inline fallback path. */
  private readonly inlineCtx: HeavyAnalyzerContext = createHeavyAnalyzerContext();

  constructor(options: HeavyAnalyzerPoolOptions = {}) {
    this.forceInline = options.forceInline === true;
  }

  /**
   * Run the 4 heavy analyzers for one face. Returns a results map keyed
   * by analyzer.name (matches the in-line analyzer wiring shape).
   *
   * @param faceCrop  pre-cropped ImageData for the face (used by Texture + Moire)
   * @param face      FaceROI (bbox + landmarks)
   * @param fullFrame original full-frame ImageData (used by ScreenReplay +
   *                  DeviceBoundary). MAY be null only if the caller
   *                  guarantees ScreenReplay + DeviceBoundary will not
   *                  need the surroundings — in practice always pass it.
   */
  async analyze(
    faceCrop: ImageData | null,
    face: FaceROI,
    fullFrame: ImageData | null,
  ): Promise<Record<string, AnalyzerResult>> {
    if (this.shouldRunInline()) {
      return runHeavyAnalyzers(this.inlineCtx, fullFrame, faceCrop, face);
    }
    const worker = await this.ensureWorker();
    if (!worker) {
      // Boot failed — degrade to the inline path forever (this run).
      return runHeavyAnalyzers(this.inlineCtx, fullFrame, faceCrop, face);
    }

    const callId = this.nextCallId++;
    const request: HeavyAnalyzeRequest = {
      kind: "analyze",
      frameId: callId,
      frameImageData: fullFrame,
      faceCropImageData: faceCrop,
      face: toWireFaceROI(face),
    };
    const transfers: ArrayBuffer[] = [];
    if (fullFrame) transfers.push(fullFrame.data.buffer);
    if (faceCrop && faceCrop.data.buffer !== fullFrame?.data.buffer) {
      transfers.push(faceCrop.data.buffer);
    }
    // Landmarks travel by value (small, ~1 KB) and we keep main-thread
    // ownership so subsequent fast analyzers can read them. Don't add
    // to the transfer list.

    return new Promise<Record<string, AnalyzerResult>>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(callId);
        reject(new Error(`HeavyAnalyzerPool: timeout after ${WORKER_CALL_TIMEOUT_MS}ms`));
      }, WORKER_CALL_TIMEOUT_MS);
      this.pending.set(callId, { resolve, reject, timer });
      try {
        worker.postMessage(request, transfers);
      } catch (err) {
        clearTimeout(timer);
        this.pending.delete(callId);
        // postMessage can throw if a transferable is detached / not
        // transferable — fall back to inline so the caller still gets
        // a result.
        runHeavyAnalyzers(this.inlineCtx, fullFrame, faceCrop, face).then(
          resolve,
          () => reject(err instanceof Error ? err : new Error(String(err))),
        );
      }
    });
  }

  /** Release the worker and any pending pool state. Safe to call repeatedly. */
  dispose(): void {
    if (this.worker) {
      try {
        this.worker.terminate();
      } catch {
        // Ignore — terminate() should never throw in browsers but
        // polyfills/jsdom can be flaky.
      }
      this.worker = null;
    }
    this.workerBootPromise = null;
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(new Error("HeavyAnalyzerPool disposed"));
    }
    this.pending.clear();
  }

  /** True when the inline fallback path must be used. */
  private shouldRunInline(): boolean {
    if (this.forceInline) return true;
    if (this.workerBootFailed) return true;
    if (typeof Worker === "undefined") return true;
    return false;
  }

  /**
   * Boot the worker on demand. Returns null when boot fails. Subsequent
   * callers share the same boot promise — no thundering-herd boot.
   */
  private async ensureWorker(): Promise<Worker | null> {
    if (this.worker) return this.worker;
    if (this.workerBootFailed) return null;
    if (this.workerBootPromise) return this.workerBootPromise;
    this.workerBootPromise = this.bootWorker();
    return this.workerBootPromise;
  }

  private async bootWorker(): Promise<Worker | null> {
    try {
      // Dynamic import keeps the Vite-specific `?worker&inline` virtual
      // module out of the static dependency graph — vitest / node never
      // resolves it because `shouldRunInline()` short-circuits first.
      // eslint-disable-next-line @typescript-eslint/ban-ts-comment
      // @ts-ignore — Vite-specific virtual module not visible to tsc.
      const mod = (await import("./HeavyAnalyzerWorker?worker&inline")) as {
        default: new () => Worker;
      };
      const WorkerCtor = mod.default;
      const worker = new WorkerCtor();
      worker.addEventListener("message", this.handleMessage);
      worker.addEventListener("error", this.handleError);
      this.worker = worker;
      return worker;
    } catch (err) {
      // Boot failure → permanent inline fallback for this pool. Logged so
      // operators can spot the degradation in production telemetry.
      // eslint-disable-next-line no-console
      console.warn("[spoof-detector] HeavyAnalyzerPool worker boot failed:", err);
      this.workerBootFailed = true;
      return null;
    }
  }

  private readonly handleMessage = (ev: MessageEvent<HeavyAnalyzeResponse>) => {
    const msg = ev.data;
    if (!msg) return;
    const pending = this.pending.get(msg.frameId);
    if (!pending) return;
    clearTimeout(pending.timer);
    this.pending.delete(msg.frameId);
    if (msg.kind === "result") {
      pending.resolve(msg.results);
    } else {
      pending.reject(new Error(msg.message));
    }
  };

  private readonly handleError = (ev: ErrorEvent) => {
    // Reject every in-flight call. Next analyze() will fall back to
    // inline because workerBootFailed flips true.
    const err = new Error(ev.message || "HeavyAnalyzerPool worker error");
    for (const [, p] of this.pending) {
      clearTimeout(p.timer);
      p.reject(err);
    }
    this.pending.clear();
    if (this.worker) {
      try {
        this.worker.terminate();
      } catch {
        // ignore
      }
      this.worker = null;
    }
    this.workerBootFailed = true;
  };
}
