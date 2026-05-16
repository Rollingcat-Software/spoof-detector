// Public API for @rollingcat/spoof-detector.
//
// Phase 1 shipped:
//   * MediaPipeFaceDetector + MiniFASNetAnalyzer + MultiClassFuser + SessionEngine
//
// Phase 2 added 5 video-track analyzers:
//   * LandmarkVariance / DeviceBoundary / Blink / MicroTremor / ScreenFlicker
//
// Phase 3 (this commit) lands Aysenur's full algorithmic surface:
//   * 4 new analyzers — Rppg, Moire, Texture, ScreenReplay
//   * 1 orchestrating gate  — FaceUsabilityGate (wraps Illumination +
//     CriticalRegionVisibility), result attached to every FrameAnalysis
//     as an advisory signal (we never block the analyzer pipeline on it).
// HybridEvaluator + Assembler are also ported under src/fusion / src/pipeline
// for callers who want the alternate fusion + advisory verdict shape — they
// are NOT wired into this facade (which sticks with MultiClassFuser).

import { SessionEngine } from "./application/SessionEngine";
import {
  AnalyzerResult,
  FaceROI,
  FrameAnalysis,
  SpoofClassification,
} from "./domain/models";
import type { IFaceAnalyzer } from "./domain/models";
import { SessionVerdict } from "./domain/session";
import { FaceUsabilityGate } from "./gates/FaceUsabilityGate";
import { BlinkAnalyzer } from "./infrastructure/analyzers/BlinkAnalyzer";
import { DeviceBoundaryAnalyzer } from "./infrastructure/analyzers/DeviceBoundaryAnalyzer";
import { LandmarkVarianceAnalyzer } from "./infrastructure/analyzers/LandmarkVarianceAnalyzer";
import { MicroTremorAnalyzer } from "./infrastructure/analyzers/MicroTremorAnalyzer";
import { MiniFASNetAnalyzer } from "./infrastructure/analyzers/MiniFASNetAnalyzer";
import { MoireAnalyzer } from "./infrastructure/analyzers/MoireAnalyzer";
import { RppgAnalyzer } from "./infrastructure/analyzers/RppgAnalyzer";
import { ScreenFlickerAnalyzer } from "./infrastructure/analyzers/ScreenFlickerAnalyzer";
import { ScreenReplayAnalyzer } from "./infrastructure/analyzers/ScreenReplayAnalyzer";
import { TextureAnalyzer } from "./infrastructure/analyzers/TextureAnalyzer";
import { MediaPipeFaceDetector } from "./infrastructure/detection/MediaPipeFaceDetector";
import { MultiClassFuser } from "./infrastructure/fusion/MultiClassFuser";
import { GateBBox, toImageData } from "./utils/imageOps";

export * from "./domain/models";
export * from "./domain/session";
export * from "./domain/taxonomy";
export { MultiClassFuser, DEFAULT_ANALYZER_WEIGHTS } from "./infrastructure/fusion/MultiClassFuser";
export { MiniFASNetAnalyzer } from "./infrastructure/analyzers/MiniFASNetAnalyzer";
export { LandmarkVarianceAnalyzer } from "./infrastructure/analyzers/LandmarkVarianceAnalyzer";
export { BlinkAnalyzer } from "./infrastructure/analyzers/BlinkAnalyzer";
export { DeviceBoundaryAnalyzer } from "./infrastructure/analyzers/DeviceBoundaryAnalyzer";
export { MicroTremorAnalyzer } from "./infrastructure/analyzers/MicroTremorAnalyzer";
export { ScreenFlickerAnalyzer } from "./infrastructure/analyzers/ScreenFlickerAnalyzer";
export { RppgAnalyzer } from "./infrastructure/analyzers/RppgAnalyzer";
export { MoireAnalyzer } from "./infrastructure/analyzers/MoireAnalyzer";
export { TextureAnalyzer } from "./infrastructure/analyzers/TextureAnalyzer";
export { ScreenReplayAnalyzer } from "./infrastructure/analyzers/ScreenReplayAnalyzer";
export { MediaPipeFaceDetector } from "./infrastructure/detection/MediaPipeFaceDetector";
export { SessionEngine } from "./application/SessionEngine";
export { FaceUsabilityGate } from "./gates/FaceUsabilityGate";
export { IlluminationGate } from "./gates/IlluminationGate";
export { CriticalRegionVisibilityGate } from "./gates/CriticalRegionVisibilityGate";
export { HybridFusionEvaluator } from "./fusion/HybridEvaluator";
export { AntispoofPipelineAssembler } from "./pipeline/Assembler";

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
  /**
   * Analyzer toggles. All default to ENABLED. Disabling an analyzer skips
   * its `analyze()` call AND removes its weight from the fusion total —
   * safer than setting weight to 0 because the score never enters the
   * evidence calculation.
   */
  enableLandmarkVariance?: boolean;
  enableBlink?: boolean;
  enableDeviceBoundary?: boolean;
  enableMicroTremor?: boolean;
  enableScreenFlicker?: boolean;
  // Phase 3 (Aysenur's additional analyzers).
  enableRppg?: boolean;
  enableMoire?: boolean;
  enableTexture?: boolean;
  enableScreenReplay?: boolean;
  /** Run Aysenur's face-usability gate. Result attached to each FrameAnalysis. Default true. */
  enableFaceUsabilityGate?: boolean;
}

/**
 * SpoofDetector — facade combining detection, all 10 ported analyzers,
 * the face-usability gate, fusion and session aggregation.
 */
export class SpoofDetector {
  private readonly detector: MediaPipeFaceDetector;
  private readonly minifasnet: MiniFASNetAnalyzer;

  // Lazy-init analyzers (no models to load).
  private landmarkVariance: LandmarkVarianceAnalyzer | null = null;
  private blink: BlinkAnalyzer | null = null;
  private deviceBoundary: DeviceBoundaryAnalyzer | null = null;
  private microTremor: MicroTremorAnalyzer | null = null;
  private screenFlicker: ScreenFlickerAnalyzer | null = null;
  private rppg: RppgAnalyzer | null = null;
  private moire: MoireAnalyzer | null = null;
  private texture: TextureAnalyzer | null = null;
  private screenReplay: ScreenReplayAnalyzer | null = null;
  private faceUsabilityGate: FaceUsabilityGate | null = null;

  private readonly fuser: MultiClassFuser;
  private readonly engine: SessionEngine;
  private readonly toggles: Required<{
    landmarkVariance: boolean;
    blink: boolean;
    deviceBoundary: boolean;
    microTremor: boolean;
    screenFlicker: boolean;
    rppg: boolean;
    moire: boolean;
    texture: boolean;
    screenReplay: boolean;
    faceUsabilityGate: boolean;
  }>;
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
    this.toggles = {
      landmarkVariance: opts.enableLandmarkVariance !== false,
      blink: opts.enableBlink !== false,
      deviceBoundary: opts.enableDeviceBoundary !== false,
      microTremor: opts.enableMicroTremor !== false,
      screenFlicker: opts.enableScreenFlicker !== false,
      rppg: opts.enableRppg !== false,
      moire: opts.enableMoire !== false,
      texture: opts.enableTexture !== false,
      screenReplay: opts.enableScreenReplay !== false,
      faceUsabilityGate: opts.enableFaceUsabilityGate !== false,
    };
  }

  /** Lazy-load both heavy models. Phase 2/3 analyzers don't need a warmup. */
  async warmup(): Promise<void> {
    await Promise.all([this.detector.warmup(), this.minifasnet.warmup()]);
  }

  /**
   * Analyze a single frame. Accepts:
   *   - HTMLCanvasElement: a canvas already containing the current video frame
   *   - ImageData: raw pixels
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
      const stage = new OffscreenCanvas(width, height);
      stage.getContext("2d")!.putImageData(input, 0, 0);
      faces = await this.detector.detect(stage, tsMs, width, height);
    }

    // 2) Set the original frame on analyzers that need surroundings.
    this.minifasnet.setFrame(input);
    if (this.toggles.deviceBoundary) this.ensureDeviceBoundary().setFrame(input);
    if (this.toggles.screenFlicker) this.ensureScreenFlicker().setFrame(input);
    if (this.toggles.screenReplay) this.ensureScreenReplay().setFrame(input);
    if (this.toggles.texture) this.ensureTexture().setFrame(input);
    if (this.toggles.rppg) this.ensureRppg().setFrame(input);

    // 3) Run Aysenur's face-usability gate (advisory — never blocks).
    let gateResult: FrameAnalysis["gate_result"] = undefined;
    if (this.toggles.faceUsabilityGate) {
      try {
        const primary = faces[0];
        const fullImageData = toImageData(input);
        const gateBbox: GateBBox | null = primary
          ? [primary.bbox.x1, primary.bbox.y1, primary.bbox.x2, primary.bbox.y2]
          : null;
        const g = this.ensureFaceUsabilityGate().evaluate(
          fullImageData,
          gateBbox,
        );
        gateResult = {
          usable: g.usable,
          blocked: g.blocked,
          reason: g.reason,
          state: g.state,
          occluded: g.occluded,
          qualityOk: g.qualityOk,
          occlusionScore: g.occlusionScore,
          illuminationScore: g.illuminationScore,
          occludedRegions: g.occludedRegions,
          underexposedRegions: g.underexposedRegions,
          overexposedRegions: g.overexposedRegions,
        };
      } catch {
        // Gate failures are silent — the analyzer pipeline must keep working.
      }
    }

    // 4) Per-face analysis. Run every enabled analyzer.
    const classifications: Record<number, SpoofClassification> = {};
    for (const face of faces) {
      const results: Record<string, AnalyzerResult> = {};
      const crop = toFaceCropOrNull(input, face);

      // MiniFASNet first (highest weight).
      const mfn = await this.minifasnet.analyze(crop, face);
      results[mfn.name] = mfn;

      if (this.toggles.screenFlicker) {
        const a = this.ensureScreenFlicker();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.deviceBoundary) {
        const a = this.ensureDeviceBoundary();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.microTremor) {
        const a = this.ensureMicroTremor();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.landmarkVariance) {
        const a = this.ensureLandmarkVariance();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.blink) {
        const a = this.ensureBlink();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      // Phase 3 (Aysenur).
      if (this.toggles.rppg) {
        const a = this.ensureRppg();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.screenReplay) {
        const a = this.ensureScreenReplay();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.texture) {
        const a = this.ensureTexture();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.moire) {
        const a = this.ensureMoire();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }

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
      gate_result: gateResult,
    };

    this.engine.ingest(analysis);
    return analysis;
  }

  /** Get current (interim) verdict. */
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
    this.landmarkVariance?.reset();
    this.blink?.reset();
    this.deviceBoundary?.reset();
    this.microTremor?.reset();
    this.screenFlicker?.reset();
    this.rppg?.reset();
    this.screenReplay?.reset();
    this.texture?.reset();
    // MoireAnalyzer has no reset() — it's stateless per-frame.
    // Gates are stateful (streak counters) — replace with a fresh instance.
    this.faceUsabilityGate = null;
  }

  /** Release native resources. */
  async close(): Promise<void> {
    await this.detector.close();
  }

  // === Lazy-init helpers ===
  private ensureLandmarkVariance(): LandmarkVarianceAnalyzer {
    if (!this.landmarkVariance) this.landmarkVariance = new LandmarkVarianceAnalyzer();
    return this.landmarkVariance;
  }
  private ensureBlink(): BlinkAnalyzer {
    if (!this.blink) this.blink = new BlinkAnalyzer();
    return this.blink;
  }
  private ensureDeviceBoundary(): DeviceBoundaryAnalyzer {
    if (!this.deviceBoundary) this.deviceBoundary = new DeviceBoundaryAnalyzer();
    return this.deviceBoundary;
  }
  private ensureMicroTremor(): MicroTremorAnalyzer {
    if (!this.microTremor) this.microTremor = new MicroTremorAnalyzer();
    return this.microTremor;
  }
  private ensureScreenFlicker(): ScreenFlickerAnalyzer {
    if (!this.screenFlicker) this.screenFlicker = new ScreenFlickerAnalyzer();
    return this.screenFlicker;
  }
  private ensureRppg(): RppgAnalyzer {
    if (!this.rppg) this.rppg = new RppgAnalyzer();
    return this.rppg;
  }
  private ensureMoire(): MoireAnalyzer {
    if (!this.moire) this.moire = new MoireAnalyzer();
    return this.moire;
  }
  private ensureTexture(): TextureAnalyzer {
    if (!this.texture) this.texture = new TextureAnalyzer();
    return this.texture;
  }
  private ensureScreenReplay(): ScreenReplayAnalyzer {
    if (!this.screenReplay) this.screenReplay = new ScreenReplayAnalyzer();
    return this.screenReplay;
  }
  private ensureFaceUsabilityGate(): FaceUsabilityGate {
    if (!this.faceUsabilityGate) this.faceUsabilityGate = new FaceUsabilityGate();
    return this.faceUsabilityGate;
  }
}

/**
 * Factory that builds a SpoofDetector and warms up the heavy models.
 */
export async function createSpoofDetector(
  options: SpoofDetectorOptions,
): Promise<SpoofDetector> {
  const det = new SpoofDetector(options);
  await det.warmup();
  return det;
}

/** Best-effort face crop. Used by analyzers that prefer the crop over the full frame. */
function toFaceCropOrNull(
  input: HTMLCanvasElement | ImageData,
  face: FaceROI,
): ImageData | null {
  const w = face.bbox.width;
  const h = face.bbox.height;
  if (w <= 0 || h <= 0) return null;
  try {
    const data = toImageData(input);
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

/** Promote a sync IFaceAnalyzer return value to a Promise for uniform handling. */
function asyncify<T>(v: T | Promise<T>): Promise<T> {
  return v instanceof Promise ? v : Promise.resolve(v);
}

// Re-export the interface so consumer code can author its own analyzers.
export type { IFaceAnalyzer };
