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

import { LivenessProver } from "./application/LivenessProver";
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
import { BackgroundGridAnalyzer } from "./infrastructure/analyzers/BackgroundGridAnalyzer";
import { BackgroundMotionAnalyzer } from "./infrastructure/analyzers/BackgroundMotionAnalyzer";
import { MediaPipeSelfieSegmenter } from "./infrastructure/detection/MediaPipeSelfieSegmenter";
import { HandTrackingAnalyzer } from "./infrastructure/analyzers/HandTrackingAnalyzer";
import { MediaPipeHandDetector } from "./infrastructure/detection/MediaPipeHandDetector";
import { AudioCapture } from "./infrastructure/audio/AudioCapture";
import { VoiceActivityAnalyzer } from "./infrastructure/analyzers/VoiceActivityAnalyzer";
import { AudioMouthSyncAnalyzer } from "./infrastructure/analyzers/AudioMouthSyncAnalyzer";
import { BehavioralPatternAnalyzer } from "./infrastructure/analyzers/BehavioralPatternAnalyzer";
import { BlinkAnalyzer } from "./infrastructure/analyzers/BlinkAnalyzer";
import { BlinkSymmetryAnalyzer } from "./infrastructure/analyzers/BlinkSymmetryAnalyzer";
import { DeviceBoundaryAnalyzer } from "./infrastructure/analyzers/DeviceBoundaryAnalyzer";
import { ExpressionDynamicsAnalyzer } from "./infrastructure/analyzers/ExpressionDynamicsAnalyzer";
import { EyebrowAnalyzer } from "./infrastructure/analyzers/EyebrowAnalyzer";
import { GazeAnalyzer } from "./infrastructure/analyzers/GazeAnalyzer";
import { LandmarkVarianceAnalyzer } from "./infrastructure/analyzers/LandmarkVarianceAnalyzer";
import { MicroTremorAnalyzer } from "./infrastructure/analyzers/MicroTremorAnalyzer";
import { MiniFASNetAnalyzer } from "./infrastructure/analyzers/MiniFASNetAnalyzer";
import { Pose3DConsistencyAnalyzer } from "./infrastructure/analyzers/Pose3DConsistencyAnalyzer";
import { LandmarkPlanarityAnalyzer } from "./infrastructure/analyzers/LandmarkPlanarityAnalyzer";
import { RppgAnalyzer } from "./infrastructure/analyzers/RppgAnalyzer";
import { ScreenFlickerAnalyzer } from "./infrastructure/analyzers/ScreenFlickerAnalyzer";
import { TemporalAnalyzer } from "./infrastructure/analyzers/TemporalAnalyzer";
import { MediaPipeFaceDetector } from "./infrastructure/detection/MediaPipeFaceDetector";
import { MultiClassFuser } from "./infrastructure/fusion/MultiClassFuser";
import { HeavyAnalyzerPool } from "./infrastructure/workers/HeavyAnalyzerPool";
import { GateBBox, toImageData } from "./utils/imageOps";

// Type-only imports for the lazy-loaded heavy analyzers (Phase 5E-1). The
// runtime modules are loaded via dynamic `import()` inside the ensure*()
// helpers below, which lets the bundler emit each as its own chunk that
// the browser can defer until the first frame the corresponding analyzer
// actually fires on. Mobile Brave / Safari benefit most: JS-parse cost
// dominates time-to-interactive on the /amispoof/ page, and these three
// (Texture Laplacian+DFT, Moire Gabor+2D-FFT, ScreenReplay CLAHE+skin-mask)
// carry the heaviest per-frame code.
import type { MoireAnalyzer } from "./infrastructure/analyzers/MoireAnalyzer";
import type { ScreenReplayAnalyzer } from "./infrastructure/analyzers/ScreenReplayAnalyzer";
import type { TextureAnalyzer } from "./infrastructure/analyzers/TextureAnalyzer";

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
export { LandmarkPlanarityAnalyzer } from "./infrastructure/analyzers/LandmarkPlanarityAnalyzer";
export { FlashReflectionAnalyzer } from "./infrastructure/analyzers/FlashReflectionAnalyzer";
export type {
  FlashColor,
  FlashReflectionResult,
  FlashReflectionOptions,
} from "./infrastructure/analyzers/FlashReflectionAnalyzer";
export { FlashTemporalAnalyzer } from "./infrastructure/analyzers/FlashTemporalAnalyzer";
export type {
  FlashTemporalResult,
  FlashTemporalOptions,
} from "./infrastructure/analyzers/FlashTemporalAnalyzer";
export { BackgroundGridAnalyzer } from "./infrastructure/analyzers/BackgroundGridAnalyzer";
export { TemporalAnalyzer } from "./infrastructure/analyzers/TemporalAnalyzer";
export { LivenessProver } from "./application/LivenessProver";
export { ReadinessGate } from "./application/ReadinessGate";
export type {
  ReadinessSignals,
  ReadinessCheck,
  ReadinessCheckId,
  ReadinessResult,
  ReadinessOptions,
} from "./application/ReadinessGate";
export {
  IdentityMatcher,
  l2normalize,
  cosine,
  meanVector,
} from "./identity/IdentityMatcher";
export type {
  FaceEmbedder,
  IdentityState,
  IdentityMatchResult,
  IdentityMatcherOptions,
} from "./identity/IdentityMatcher";
// MoireAnalyzer / TextureAnalyzer / ScreenReplayAnalyzer are intentionally
// NOT re-exported eagerly here (Phase 5E-1) so Vite/Rollup emits each as
// its own lazy chunk under dist/. Callers that need the constructors can
// either: (a) deep-import the module path, or (b) call `loadHeavyAnalyzers()`
// to resolve all three concurrently.
export type { MoireAnalyzer } from "./infrastructure/analyzers/MoireAnalyzer";
export type { TextureAnalyzer } from "./infrastructure/analyzers/TextureAnalyzer";
export type { ScreenReplayAnalyzer } from "./infrastructure/analyzers/ScreenReplayAnalyzer";
/**
 * Lazily resolve the three heavy-analyzer constructors as a single batch.
 * Useful for "preload heavy chunks while the camera permission dialog is
 * up" callers, or for unit-tests that need synchronous instantiation.
 */
export async function loadHeavyAnalyzers(): Promise<{
  MoireAnalyzer: typeof import("./infrastructure/analyzers/MoireAnalyzer").MoireAnalyzer;
  TextureAnalyzer: typeof import("./infrastructure/analyzers/TextureAnalyzer").TextureAnalyzer;
  ScreenReplayAnalyzer: typeof import("./infrastructure/analyzers/ScreenReplayAnalyzer").ScreenReplayAnalyzer;
}> {
  const [moire, texture, screenReplay] = await Promise.all([
    import("./infrastructure/analyzers/MoireAnalyzer"),
    import("./infrastructure/analyzers/TextureAnalyzer"),
    import("./infrastructure/analyzers/ScreenReplayAnalyzer"),
  ]);
  return {
    MoireAnalyzer: moire.MoireAnalyzer,
    TextureAnalyzer: texture.TextureAnalyzer,
    ScreenReplayAnalyzer: screenReplay.ScreenReplayAnalyzer,
  };
}
export { MediaPipeFaceDetector } from "./infrastructure/detection/MediaPipeFaceDetector";
export { SessionEngine } from "./application/SessionEngine";
export { FaceUsabilityGate } from "./gates/FaceUsabilityGate";
export { IlluminationGate } from "./gates/IlluminationGate";
export { CriticalRegionVisibilityGate } from "./gates/CriticalRegionVisibilityGate";
export { HybridFusionEvaluator } from "./fusion/HybridEvaluator";
export { AntispoofPipelineAssembler } from "./pipeline/Assembler";
export { HeavyAnalyzerPool } from "./infrastructure/workers/HeavyAnalyzerPool";
// Phase 5E-3 — in-page accuracy harness against a tiny CASIA-FASD micro
// mirror. Same-origin only (no third-party CDN — strict-origin-when-cross-origin
// would break the canvas readback on mobile Brave).
export {
  runCasiaFasdMicroBench,
} from "./validation/CasiaFasdMicroBench";
export type {
  CasiaFasdBenchSample,
  CasiaFasdBenchResult,
  CasiaFasdBenchRow,
} from "./validation/CasiaFasdMicroBench";

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
  // Phase 4 (parity with Python `src/`).
  enableBackgroundGrid?: boolean;
  enableTemporal?: boolean;
  // Phase A (blendshape + 3D matrix derived; require MediaPipe
  // outputFaceBlendshapes + outputFacialTransformationMatrixes — both
  // enabled by default in MediaPipeFaceDetector).
  enableEyebrow?: boolean;
  enableBlinkSymmetry?: boolean;
  enableGaze?: boolean;
  enableExpressionDynamics?: boolean;
  enablePose3DConsistency?: boolean;
  /**
   * Camera-independent flat-surface (printed-photo / screen) detector. Fits
   * an affine landmark-reprojection residual under head rotation: a flat
   * print moves as one plane (low residual → spoof), a real 3D face has
   * parallax that breaks the fit (high residual → live). Default true.
   * Backed by a SessionEngine planar-print veto. See LandmarkPlanarityAnalyzer.
   */
  enablePlanarity?: boolean;
  // Phase B — temporal-pattern analyzer (no new MediaPipe data).
  enableBehavioralPattern?: boolean;
  /**
   * Phase D1 (opt-in) — MediaPipe SelfieSegmenter-based background
   * motion analyzer. Default `false`: requires an additional ~250 KB
   * model fetch on first use. Adds `background_motion_points` (cap 8)
   * to the LivenessProver passive ceiling when enabled.
   */
  enableBackgroundSegmentation?: boolean;
  /** Optional override for the SelfieSegmenter .tflite URL. */
  selfieSegmenterModelUrl?: string;
  /**
   * Phase D2 (opt-in) — MediaPipe HandLandmarker for hand presence and
   * gesture tracking. Default `false`: requires an additional ~6 MB
   * model fetch on first use. Adds `hand_naturalness_points` (cap 8)
   * to the LivenessProver passive ceiling when enabled.
   */
  enableHandTracking?: boolean;
  /** Optional override for the HandLandmarker .task URL. */
  handLandmarkerModelUrl?: string;
  /**
   * Phase D3 (opt-in) — microphone capture + voice-activity detection
   * + audio-mouth-sync analyzer. Default `false`: prompts the user for
   * mic permission on `detector.startAudio()`. Until that's called the
   * VoiceActivity and AudioMouthSync analyzers return neutral 50.
   * Adds `voice_activity_points` (cap 6) and `audio_mouth_sync_points`
   * (cap 12) to the LivenessProver passive ceiling.
   */
  enableAudio?: boolean;
  /** Run Aysenur's face-usability gate. Result attached to each FrameAnalysis. Default true. */
  enableFaceUsabilityGate?: boolean;
  /** Run the LivenessProver passive proof scorer alongside the SessionEngine. Default true. */
  enableLivenessProver?: boolean;
  /**
   * Enable LivenessProver active challenges (turn head, nod, blink-on-cue).
   * Default true (Python parity). Proctoring use cases should pass `false`
   * — challenges mid-exam are disruptive. With challenges off the prover
   * still scores passively from observed movement (blinks, head rotation,
   * eye/mouth/face motion, landmark variance, expression), and the new
   * passive axes are sized so a natural live face reaches the 60-point
   * proven-live threshold without any prompts.
   */
  enableLivenessChallenges?: boolean;
  /**
   * Override LivenessProver passive gates. Defaults are Python-parity
   * values (1.2 / 3.0° / 1.0). Proctoring typically passes more permissive
   * values (e.g. 0.4 / 2.0° / 0.5) so natural-observation evidence
   * accumulates without forcing the user to perform.
   */
  livenessProverThresholds?: {
    expressionRatioGate?: number;
    rotationThreshold?: number;
    landmarkVarThreshold?: number;
  };
  /**
   * Offload the 4 heavy synchronous analyzers (Texture, Moire,
   * ScreenReplay, DeviceBoundary) to a Web Worker so they stop
   * blocking the main thread. Default true.
   *
   * Worker is built from an inline blob URL — no extra file to serve.
   * If `typeof Worker === "undefined"` or worker boot fails, the pool
   * transparently falls back to running them on the main thread.
   *
   * SAB-free: payloads cross the postMessage boundary as Transferable
   * ImageData (zero-copy). No COOP/COEP needed.
   */
  enableHeavyWorker?: boolean;
  /**
   * Frame-skip schedule for the 4 heavy analyzers (whether on the
   * worker pool or the inline fallback). Default 3 — i.e. heavy
   * analyzers run on frames 1, 4, 7, … Skipped frames REUSE the
   * previous-frame result per face from `heavyCache`.
   *
   * Tradeoff: lower N = sharper temporal resolution, more CPU. The
   * fast analyzers (MiniFASNet, MediaPipe, Blink, etc.) still run
   * every frame — only the slow ones are skipped.
   */
  heavyAnalyzerFrameSkip?: number;
  /**
   * Frame-skip schedule for the FaceUsabilityGate. Default 5 — the
   * gate is advisory and its inputs change slowly (lighting, occlusion).
   * Skipped frames reuse the previous gate result from `gateCache`.
   */
  gateFrameSkip?: number;
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
  private backgroundGrid: BackgroundGridAnalyzer | null = null;
  private temporal: TemporalAnalyzer | null = null;
  private faceUsabilityGate: FaceUsabilityGate | null = null;
  private livenessProver: LivenessProver | null = null;
  // Phase A — blendshape / 3D matrix derived analyzers (lazy-init).
  private eyebrow: EyebrowAnalyzer | null = null;
  private blinkSymmetry: BlinkSymmetryAnalyzer | null = null;
  private gaze: GazeAnalyzer | null = null;
  private expressionDynamics: ExpressionDynamicsAnalyzer | null = null;
  private pose3DConsistency: Pose3DConsistencyAnalyzer | null = null;
  private planarity: LandmarkPlanarityAnalyzer | null = null;
  private behavioralPattern: BehavioralPatternAnalyzer | null = null;
  private backgroundMotion: BackgroundMotionAnalyzer | null = null;
  private readonly selfieSegmenterModelUrl: string | undefined;
  private handTracking: HandTrackingAnalyzer | null = null;
  private readonly handLandmarkerModelUrl: string | undefined;
  private audioCapture: AudioCapture | null = null;
  private voiceActivity: VoiceActivityAnalyzer | null = null;
  private audioMouthSync: AudioMouthSyncAnalyzer | null = null;

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
    backgroundGrid: boolean;
    temporal: boolean;
    faceUsabilityGate: boolean;
    livenessProver: boolean;
    eyebrow: boolean;
    blinkSymmetry: boolean;
    gaze: boolean;
    expressionDynamics: boolean;
    pose3DConsistency: boolean;
    planarity: boolean;
    behavioralPattern: boolean;
    backgroundSegmentation: boolean;
    handTracking: boolean;
    audio: boolean;
  }>;
  private frameId = 0;
  private started = false;

  // === Worker offload + frame-skip scheduler state. ===
  private readonly enableHeavyWorker: boolean;
  private readonly heavyAnalyzerFrameSkip: number;
  private readonly gateFrameSkip: number;
  private heavyPool: HeavyAnalyzerPool | null = null;
  /** Cached heavy-analyzer results per face_id (reused on skipped frames). */
  private readonly heavyCache = new Map<number, Record<string, AnalyzerResult>>();
  /** Cached FaceUsabilityGate result (reused on skipped frames). */
  private gateCache: FrameAnalysis["gate_result"] = undefined;

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
      backgroundGrid: opts.enableBackgroundGrid !== false,
      temporal: opts.enableTemporal !== false,
      faceUsabilityGate: opts.enableFaceUsabilityGate !== false,
      livenessProver: opts.enableLivenessProver !== false,
      eyebrow: opts.enableEyebrow !== false,
      blinkSymmetry: opts.enableBlinkSymmetry !== false,
      gaze: opts.enableGaze !== false,
      expressionDynamics: opts.enableExpressionDynamics !== false,
      pose3DConsistency: opts.enablePose3DConsistency !== false,
      planarity: opts.enablePlanarity !== false,
      behavioralPattern: opts.enableBehavioralPattern !== false,
      // Phase D1 default OFF — separate model fetch.
      backgroundSegmentation: opts.enableBackgroundSegmentation === true,
      // Phase D2 default OFF — separate ~6 MB model fetch.
      handTracking: opts.enableHandTracking === true,
      // Phase D3 default OFF — mic permission UX.
      audio: opts.enableAudio === true,
    };
    // LivenessProver is owned by the SessionEngine so its `isProvenLive` gate
    // joins the verdict AND-condition (matches Python design). When the
    // toggle is off we pass null and the engine falls back to fusion-only.
    if (this.toggles.livenessProver) {
      const t = opts.livenessProverThresholds ?? {};
      this.livenessProver = new LivenessProver({
        enableChallenges: opts.enableLivenessChallenges !== false,
        expressionRatioGate: t.expressionRatioGate,
        rotationThreshold: t.rotationThreshold,
        landmarkVarThreshold: t.landmarkVarThreshold,
      });
    }
    this.engine = new SessionEngine({
      sessionId: opts.sessionId,
      prover: this.livenessProver,
    });
    this.enableHeavyWorker = opts.enableHeavyWorker !== false;
    // Clamp to >= 1: a value of 0 or 1 means "run every frame" (no skip).
    this.heavyAnalyzerFrameSkip = Math.max(
      1,
      Math.floor(opts.heavyAnalyzerFrameSkip ?? 3),
    );
    this.gateFrameSkip = Math.max(1, Math.floor(opts.gateFrameSkip ?? 5));
    this.selfieSegmenterModelUrl = opts.selfieSegmenterModelUrl;
    this.handLandmarkerModelUrl = opts.handLandmarkerModelUrl;
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

    // Frame-skip schedule. Pattern: run on frames 1, 1+N, 1+2N, …
    // i.e. the very first frame and every Nth frame thereafter. This
    // guarantees a fresh result on the first call (so the cache is
    // never empty when consumed) AND a steady-state period of N.
    const runHeavy =
      ((this.frameId - 1) % this.heavyAnalyzerFrameSkip) === 0;
    const runGate =
      ((this.frameId - 1) % this.gateFrameSkip) === 0;

    // 2) Set the original frame on analyzers that need surroundings.
    //    The 4 heavy analyzers (Texture/Moire/ScreenReplay/DeviceBoundary)
    //    are only fed here when the worker is DISABLED — when the worker
    //    is on, the pool owns its own analyzer instances and ships the
    //    full frame across the postMessage boundary. We also skip the
    //    setFrame calls on frames where runHeavy is false: no point
    //    paying the chunk-download cost for an analyzer we won't call.
    this.minifasnet.setFrame(input);
    if (this.toggles.screenFlicker) this.ensureScreenFlicker().setFrame(input);
    if (this.toggles.rppg) this.ensureRppg().setFrame(input);
    if (this.toggles.backgroundGrid) this.ensureBackgroundGrid().setFrame(input);
    if (this.toggles.backgroundSegmentation)
      this.ensureBackgroundMotion().setFrame(input as never);
    if (this.toggles.handTracking)
      this.ensureHandTracking().setFrame(input as never);
    if (runHeavy && !this.enableHeavyWorker) {
      if (this.toggles.deviceBoundary) this.ensureDeviceBoundary().setFrame(input);
      if (this.toggles.screenReplay) (await this.ensureScreenReplay()).setFrame(input);
      if (this.toggles.texture) (await this.ensureTexture()).setFrame(input);
    }

    // 3) Run Aysenur's face-usability gate (advisory — never blocks).
    //    Frame-skipped: skipped frames reuse `this.gateCache`.
    let gateResult: FrameAnalysis["gate_result"] = this.gateCache;
    if (this.toggles.faceUsabilityGate && runGate) {
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
        this.gateCache = gateResult;
      } catch {
        // Gate failures are silent — the analyzer pipeline must keep working.
      }
    }

    // 4) Per-face analysis. Heavy analyzers may be:
    //    (a) running on the worker pool (Promise.all-concurrent with the
    //        main-thread fast analyzers), OR
    //    (b) running inline on the main thread, OR
    //    (c) skipped this tick — cached results are reused.
    //
    //    The fast analyzers (MiniFASNet, Blink, LandmarkVar, MicroTremor,
    //    ScreenFlicker, Rppg) always run every frame; each is <2 ms and
    //    several depend on per-frame landmarks that don't compress
    //    across skips.
    const classifications: Record<number, SpoofClassification> = {};

    // Pre-compute the full-frame ImageData once if we'll need it for
    // the worker hand-off — avoids re-converting per face.
    const needsFullFrameImageData =
      runHeavy && this.enableHeavyWorker && faces.length > 0;
    const fullFrameImageData = needsFullFrameImageData
      ? toImageData(input)
      : null;

    // Stage A: kick off the heavy-analyzer worker promises (parallel
    // across faces). The worker runs concurrently with the synchronous
    // main-thread analyzers below.
    type HeavyTask = Promise<{
      face_id: number;
      results: Record<string, AnalyzerResult>;
    }>;
    const heavyTasks: HeavyTask[] = [];
    const heavyCropMap = new Map<number, ImageData | null>();
    if (runHeavy) {
      for (const face of faces) {
        const crop = toFaceCropOrNull(input, face);
        heavyCropMap.set(face.face_id, crop);
        if (this.enableHeavyWorker) {
          const pool = this.ensureHeavyPool();
          heavyTasks.push(
            pool
              .analyze(crop, face, fullFrameImageData)
              .then((results) => ({ face_id: face.face_id, results })),
          );
        }
      }
    }

    // Stage B: main-thread per-face work — fast analyzers always run,
    // plus the inline heavy path when the worker is disabled.
    for (const face of faces) {
      const results: Record<string, AnalyzerResult> = {};
      // Heavy-skipped frames still need a crop (the fast analyzers below
      // consume it). We can't reuse heavyCropMap unconditionally because
      // it's only populated when runHeavy is true.
      const crop = heavyCropMap.has(face.face_id)
        ? (heavyCropMap.get(face.face_id) ?? null)
        : toFaceCropOrNull(input, face);

      // MiniFASNet first (highest weight).
      const mfn = await this.minifasnet.analyze(crop, face);
      results[mfn.name] = mfn;

      if (this.toggles.screenFlicker) {
        const a = this.ensureScreenFlicker();
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
      if (this.toggles.rppg) {
        const a = this.ensureRppg();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.backgroundGrid) {
        const a = this.ensureBackgroundGrid();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.temporal) {
        const a = this.ensureTemporal();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      // Phase A — blendshape / 3D matrix driven analyzers. All read
      // directly from face.blendshapes / face.transformMatrix; no heavy
      // CV cost, sub-2-ms each on the main thread.
      if (this.toggles.eyebrow) {
        const a = this.ensureEyebrow();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.blinkSymmetry) {
        const a = this.ensureBlinkSymmetry();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.gaze) {
        const a = this.ensureGaze();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.expressionDynamics) {
        const a = this.ensureExpressionDynamics();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.pose3DConsistency) {
        const a = this.ensurePose3DConsistency();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.planarity) {
        const a = this.ensurePlanarity();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.behavioralPattern) {
        const a = this.ensureBehavioralPattern();
        results[a.name] = await asyncify(a.analyze(crop, face));
      }
      if (this.toggles.backgroundSegmentation) {
        const a = this.ensureBackgroundMotion();
        results[a.name] = await a.analyze(crop, face);
      }
      if (this.toggles.handTracking) {
        const a = this.ensureHandTracking();
        results[a.name] = await a.analyze(crop, face);
      }
      if (this.toggles.audio) {
        const va = this.ensureVoiceActivity();
        results[va.name] = await asyncify(va.analyze(crop, face));
        const ms = this.ensureAudioMouthSync();
        results[ms.name] = await asyncify(ms.analyze(crop, face));
      }

      // Heavy analyzers — inline path. (Worker path is awaited in Stage C.)
      if (runHeavy && !this.enableHeavyWorker) {
        if (this.toggles.deviceBoundary) {
          const a = this.ensureDeviceBoundary();
          results[a.name] = await asyncify(a.analyze(crop, face));
        }
        if (this.toggles.screenReplay) {
          const a = await this.ensureScreenReplay();
          results[a.name] = await asyncify(a.analyze(crop, face));
        }
        if (this.toggles.texture) {
          const a = await this.ensureTexture();
          results[a.name] = await asyncify(a.analyze(crop, face));
        }
        if (this.toggles.moire) {
          const a = await this.ensureMoire();
          results[a.name] = await asyncify(a.analyze(crop, face));
        }
      }

      classifications[face.face_id] = this.fuser.fuse(face.face_id, results);
    }

    // Stage C: await the worker pool (if it was kicked off in Stage A),
    // then fold the heavy results into each face's classification.
    if (heavyTasks.length > 0) {
      const heavyOutputs = await Promise.all(heavyTasks);
      for (const { face_id, results: heavyResults } of heavyOutputs) {
        const filtered = this.filterHeavyByToggle(heavyResults);
        this.heavyCache.set(face_id, filtered);
        const cls = classifications[face_id];
        if (cls) {
          // Re-fuse with the heavy results merged in.
          const merged = { ...cls.analyzer_results, ...filtered };
          classifications[face_id] = this.fuser.fuse(face_id, merged);
        }
      }
    } else if (runHeavy && !this.enableHeavyWorker) {
      // Inline path already merged the heavies into each face's results
      // above. Cache the heavy subset now so skipped frames can reuse it.
      for (const face of faces) {
        const cls = classifications[face.face_id];
        if (!cls) continue;
        const heavy = this.filterHeavyByToggle(cls.analyzer_results);
        if (Object.keys(heavy).length > 0) {
          this.heavyCache.set(face.face_id, heavy);
        }
      }
    } else {
      // Frame skipped — replay the previous heavy results from cache.
      for (const face of faces) {
        const cached = this.heavyCache.get(face.face_id);
        if (!cached) continue;
        const cls = classifications[face.face_id];
        if (cls) {
          const merged = { ...cls.analyzer_results, ...cached };
          classifications[face.face_id] = this.fuser.fuse(face.face_id, merged);
        }
      }
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

    // LivenessProver is now driven inside SessionEngine.ingest() so the
    // two stay in lockstep — single source of truth for prover lifecycle.
    this.engine.ingest(analysis);
    return analysis;
  }

  /** Get the current LivenessProver passive-proof score (Phase 4C). */
  getProof(): ReturnType<LivenessProver["getProof"]> | null {
    if (!this.toggles.livenessProver || !this.livenessProver) return null;
    return this.livenessProver.getProof();
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
    this.backgroundGrid?.reset();
    this.temporal?.reset();
    this.eyebrow?.reset();
    this.blinkSymmetry?.reset();
    this.gaze?.reset();
    this.expressionDynamics?.reset();
    this.pose3DConsistency?.reset();
    this.planarity?.reset();
    this.behavioralPattern?.reset();
    this.backgroundMotion?.reset();
    this.handTracking?.reset();
    this.audioMouthSync?.reset();
    // VoiceActivity is stateless (just reads AudioCapture).
    // LivenessProver is reset inside SessionEngine.reset() (single source of truth).
    // MoireAnalyzer has no reset() — it's stateless per-frame.
    // Gates are stateful (streak counters) — replace with a fresh instance.
    this.faceUsabilityGate = null;
    // Frame-skip + worker offload caches must be flushed too, otherwise
    // a fresh session would inherit the previous session's heavy scores.
    this.heavyCache.clear();
    this.gateCache = undefined;
  }

  /** Release native resources. */
  async close(): Promise<void> {
    if (this.heavyPool) {
      this.heavyPool.dispose();
      this.heavyPool = null;
    }
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
  private ensureBackgroundGrid(): BackgroundGridAnalyzer {
    if (!this.backgroundGrid) this.backgroundGrid = new BackgroundGridAnalyzer();
    return this.backgroundGrid;
  }
  private ensureTemporal(): TemporalAnalyzer {
    if (!this.temporal) this.temporal = new TemporalAnalyzer();
    return this.temporal;
  }
  private ensureEyebrow(): EyebrowAnalyzer {
    if (!this.eyebrow) this.eyebrow = new EyebrowAnalyzer();
    return this.eyebrow;
  }
  private ensureBlinkSymmetry(): BlinkSymmetryAnalyzer {
    if (!this.blinkSymmetry)
      this.blinkSymmetry = new BlinkSymmetryAnalyzer();
    return this.blinkSymmetry;
  }
  private ensureGaze(): GazeAnalyzer {
    if (!this.gaze) this.gaze = new GazeAnalyzer();
    return this.gaze;
  }
  private ensureExpressionDynamics(): ExpressionDynamicsAnalyzer {
    if (!this.expressionDynamics)
      this.expressionDynamics = new ExpressionDynamicsAnalyzer();
    return this.expressionDynamics;
  }
  private ensurePose3DConsistency(): Pose3DConsistencyAnalyzer {
    if (!this.pose3DConsistency)
      this.pose3DConsistency = new Pose3DConsistencyAnalyzer();
    return this.pose3DConsistency;
  }
  private ensurePlanarity(): LandmarkPlanarityAnalyzer {
    if (!this.planarity) this.planarity = new LandmarkPlanarityAnalyzer();
    return this.planarity;
  }
  private ensureBehavioralPattern(): BehavioralPatternAnalyzer {
    if (!this.behavioralPattern)
      this.behavioralPattern = new BehavioralPatternAnalyzer();
    return this.behavioralPattern;
  }
  private ensureAudioCapture(): AudioCapture {
    if (!this.audioCapture) this.audioCapture = new AudioCapture();
    return this.audioCapture;
  }
  private ensureVoiceActivity(): VoiceActivityAnalyzer {
    if (!this.voiceActivity)
      this.voiceActivity = new VoiceActivityAnalyzer({
        audio: this.ensureAudioCapture(),
      });
    return this.voiceActivity;
  }
  private ensureAudioMouthSync(): AudioMouthSyncAnalyzer {
    if (!this.audioMouthSync)
      this.audioMouthSync = new AudioMouthSyncAnalyzer({
        audio: this.ensureAudioCapture(),
      });
    return this.audioMouthSync;
  }
  /**
   * Phase D3 public API. Consumer prompts for mic permission by calling
   * this; nothing happens automatically because mic permission is
   * intentionally a deliberate user action. No-op if `enableAudio` is
   * false or audio is already active.
   */
  async startAudio(): Promise<void> {
    if (!this.toggles.audio) return;
    await this.ensureAudioCapture().start();
  }
  async stopAudio(): Promise<void> {
    if (this.audioCapture) await this.audioCapture.stop();
  }
  /** True if the mic capture is currently running. */
  get audioActive(): boolean {
    return this.audioCapture?.isActive === true;
  }

  private ensureHandTracking(): HandTrackingAnalyzer {
    if (!this.handTracking) {
      this.handTracking = new HandTrackingAnalyzer(
        this.handLandmarkerModelUrl
          ? {
              detector: new MediaPipeHandDetector({
                modelAssetPath: this.handLandmarkerModelUrl,
              }),
            }
          : {},
      );
    }
    return this.handTracking;
  }
  private ensureBackgroundMotion(): BackgroundMotionAnalyzer {
    if (!this.backgroundMotion) {
      // First call also lazy-constructs the segmenter — model fetch only
      // happens when the consumer has flipped the toggle on AND the page
      // has reached the first frame.
      this.backgroundMotion = new BackgroundMotionAnalyzer(
        this.selfieSegmenterModelUrl
          ? {
              segmenter: new MediaPipeSelfieSegmenter({
                modelAssetPath: this.selfieSegmenterModelUrl,
              }),
            }
          : {},
      );
    }
    return this.backgroundMotion;
  }
  // LivenessProver is constructed eagerly in the constructor when the toggle
  // is on and injected into the SessionEngine — no lazy helper needed.

  // === Lazy-import helpers for heavy analyzers (Phase 5E-1) ===
  // Returning Promise<T> instead of T forces every call site to `await`,
  // which is what makes Vite emit each of these modules as its own
  // dist/spoof-detector-*.js chunk (verifiable via `npm run build`).
  private async ensureMoire(): Promise<MoireAnalyzer> {
    if (!this.moire) {
      const mod = await import("./infrastructure/analyzers/MoireAnalyzer");
      this.moire = new mod.MoireAnalyzer();
    }
    return this.moire;
  }
  private async ensureTexture(): Promise<TextureAnalyzer> {
    if (!this.texture) {
      const mod = await import("./infrastructure/analyzers/TextureAnalyzer");
      this.texture = new mod.TextureAnalyzer();
    }
    return this.texture;
  }
  private async ensureScreenReplay(): Promise<ScreenReplayAnalyzer> {
    if (!this.screenReplay) {
      const mod = await import(
        "./infrastructure/analyzers/ScreenReplayAnalyzer"
      );
      this.screenReplay = new mod.ScreenReplayAnalyzer();
    }
    return this.screenReplay;
  }
  private ensureFaceUsabilityGate(): FaceUsabilityGate {
    if (!this.faceUsabilityGate) this.faceUsabilityGate = new FaceUsabilityGate();
    return this.faceUsabilityGate;
  }

  /**
   * Lazy-create the heavy-analyzer worker pool. The pool itself defers
   * Worker construction until its first call, so importing SpoofDetector
   * with `enableHeavyWorker: false` (or in a no-Worker environment) does
   * NOT spin up a thread.
   */
  private ensureHeavyPool(): HeavyAnalyzerPool {
    if (!this.heavyPool) this.heavyPool = new HeavyAnalyzerPool();
    return this.heavyPool;
  }

  /**
   * Strip the worker-pool result down to ONLY the heavy analyzers
   * currently enabled via toggles. The worker always runs all four for
   * simplicity; the gate at this layer is cheap and keeps the fusion
   * results identical to the inline path.
   */
  private filterHeavyByToggle(
    results: Record<string, AnalyzerResult>,
  ): Record<string, AnalyzerResult> {
    const out: Record<string, AnalyzerResult> = {};
    if (this.toggles.deviceBoundary && results.device_boundary) {
      out.device_boundary = results.device_boundary;
    }
    if (this.toggles.screenReplay && results.screen_replay) {
      out.screen_replay = results.screen_replay;
    }
    if (this.toggles.texture && results.texture) {
      out.texture = results.texture;
    }
    if (this.toggles.moire && results.moire) {
      out.moire = results.moire;
    }
    return out;
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
