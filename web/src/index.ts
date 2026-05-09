// Public API for @rollingcat/spoof-detector (Phase 1).
//
// Exposes:
//   - createSpoofDetector(options): factory that wires
//       MediaPipeFaceDetector + MiniFASNetAnalyzer + MultiClassFuser + SessionEngine
//   - SpoofDetector class with analyzeFrame() / getVerdict() / conclude() / reset()
//   - All domain types (re-exported from ./domain/* for snapshot interop)
//
// Phase 2+ will expand the analyzer roster (Landmark Variance, Blink, etc.).
// The signature here is forward-compatible — callers won't need to change.

import { SessionEngine } from "./application/SessionEngine";
import {
  AnalyzerResult,
  FaceROI,
  FrameAnalysis,
  SpoofClassification,
} from "./domain/models";
import { SessionVerdict } from "./domain/session";
import { MiniFASNetAnalyzer } from "./infrastructure/analyzers/MiniFASNetAnalyzer";
import { MediaPipeFaceDetector } from "./infrastructure/detection/MediaPipeFaceDetector";
import { MultiClassFuser } from "./infrastructure/fusion/MultiClassFuser";
import { toImageData } from "./utils/imageOps";

export * from "./domain/models";
export * from "./domain/session";
export * from "./domain/taxonomy";
export { MultiClassFuser, DEFAULT_ANALYZER_WEIGHTS } from "./infrastructure/fusion/MultiClassFuser";
export { MiniFASNetAnalyzer } from "./infrastructure/analyzers/MiniFASNetAnalyzer";
export { MediaPipeFaceDetector } from "./infrastructure/detection/MediaPipeFaceDetector";
export { SessionEngine } from "./application/SessionEngine";

export interface SpoofDetectorOptions {
  /** URL to minifasnet_v2.onnx (1.7 MB). */
  miniFasNetModelUrl: string;
  /** URL to face_landmarker.task (~10 MB). */
  faceLandmarkerTaskUrl: string;
  /** Optional MediaPipe vision-WASM CDN base URL. */
  mediaPipeWasmBaseUrl?: string;
  /** ONNX runtime EPs for MiniFASNet. Default: ["wasm"]. */
  ortExecutionProviders?: ReadonlyArray<"wasm" | "webgpu" | "webgl" | "cpu">;
  /** Optional analyzer weight overrides. */
  analyzerWeights?: Record<string, number>;
  /** Concurrent face count (1 for ID verification, 5 for proctoring). */
  numFaces?: number;
  /** GPU delegate for FaceLandmarker. Default true. */
  useGpu?: boolean;
  /** Optional explicit session id. */
  sessionId?: string;
}

/**
 * SpoofDetector — Phase 1 facade combining detection, MiniFASNet anti-spoof,
 * fusion and session aggregation.
 */
export class SpoofDetector {
  private readonly detector: MediaPipeFaceDetector;
  private readonly minifasnet: MiniFASNetAnalyzer;
  private readonly fuser: MultiClassFuser;
  private readonly engine: SessionEngine;
  private frameId = 0;
  private started = false;

  constructor(opts: SpoofDetectorOptions) {
    this.detector = new MediaPipeFaceDetector({
      modelAssetPath: opts.faceLandmarkerTaskUrl,
      wasmBaseUrl: opts.mediaPipeWasmBaseUrl,
      runningMode: "VIDEO",
      numFaces: opts.numFaces ?? 1,
      useGpu: opts.useGpu !== false,
    });
    this.minifasnet = new MiniFASNetAnalyzer({
      modelUrl: opts.miniFasNetModelUrl,
      executionProviders: opts.ortExecutionProviders,
    });
    this.fuser = new MultiClassFuser(opts.analyzerWeights);
    this.engine = new SessionEngine({ sessionId: opts.sessionId });
  }

  /** Lazy-load both models. Optional — they auto-warmup on first analyzeFrame. */
  async warmup(): Promise<void> {
    await Promise.all([this.detector.warmup(), this.minifasnet.warmup()]);
  }

  /**
   * Analyze a single frame. Accepts:
   *   - HTMLCanvasElement: a canvas already containing the current video frame
   *   - ImageData: raw pixels
   *
   * For best perf, callers should reuse a single canvas across frames.
   */
  async analyzeFrame(
    input: HTMLCanvasElement | ImageData,
  ): Promise<FrameAnalysis> {
    if (!this.started) {
      this.engine.start();
      this.started = true;
    }
    const start = performance.now();
    this.frameId += 1;

    const width = input.width;
    const height = input.height;
    const tsMs = performance.now();

    // 1) Detect faces (and landmarks).
    let faces: FaceROI[] = [];
    if (input instanceof HTMLCanvasElement) {
      faces = await this.detector.detect(input, tsMs, width, height);
    } else {
      // ImageData: stage to OffscreenCanvas because MediaPipe can't ingest ImageData directly.
      const stage = new OffscreenCanvas(width, height);
      stage.getContext("2d")!.putImageData(input, 0, 0);
      faces = await this.detector.detect(stage, tsMs, width, height);
    }

    // 2) Set the original frame on MiniFASNet so it gets context, not a crop.
    this.minifasnet.setFrame(input);

    // 3) Per-face analysis. Only MiniFASNet in Phase 1.
    const classifications: Record<number, SpoofClassification> = {};
    for (const face of faces) {
      const results: Record<string, AnalyzerResult> = {};
      const r = await this.minifasnet.analyze(toFaceCropOrNull(input, face), face);
      results[r.name] = r;

      const cls = this.fuser.fuse(face.face_id, results);
      classifications[face.face_id] = cls;
    }

    const total_ms = performance.now() - start;
    const analysis: FrameAnalysis = {
      frame_id: this.frameId,
      faces,
      classifications,
      frame_signals: {},
      total_ms,
    };

    this.engine.ingest(analysis);
    return analysis;
  }

  /** Get current (interim) verdict. Call between frames or at session end. */
  getVerdict(): SessionVerdict {
    return this.engine.getVerdict();
  }

  /** Conclude the session and return the final verdict. */
  conclude(): SessionVerdict {
    return this.engine.conclude();
  }

  /** Reset session state — reuses the loaded models. */
  reset(): void {
    this.engine.reset();
    this.frameId = 0;
    this.started = false;
  }

  /** Release native resources. */
  async close(): Promise<void> {
    await this.detector.close();
  }
}

/**
 * Factory that builds a SpoofDetector and warms up the models.
 */
export async function createSpoofDetector(
  options: SpoofDetectorOptions,
): Promise<SpoofDetector> {
  const det = new SpoofDetector(options);
  await det.warmup();
  return det;
}

/** Best-effort face crop (currently unused — MiniFASNet uses the full frame). */
function toFaceCropOrNull(
  input: HTMLCanvasElement | ImageData,
  face: FaceROI,
): ImageData | null {
  const w = face.bbox.width;
  const h = face.bbox.height;
  if (w <= 0 || h <= 0) return null;
  try {
    const data = toImageData(input);
    // Allocate a stage canvas and slice via drawImage.
    const stage = new OffscreenCanvas(data.width, data.height);
    stage.getContext("2d")!.putImageData(data, 0, 0);
    const out = new OffscreenCanvas(w, h);
    out
      .getContext("2d")!
      .drawImage(stage, face.bbox.x1, face.bbox.y1, w, h, 0, 0, w, h);
    return out.getContext("2d")!.getImageData(0, 0, w, h);
  } catch {
    return null;
  }
}
