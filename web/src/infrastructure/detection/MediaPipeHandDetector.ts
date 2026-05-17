// MediaPipeHandDetector — Phase D2 (opt-in).
//
// Wraps @mediapipe/tasks-vision HandLandmarker with the `hand_landmarker`
// model from Google's CDN (~6 MB). Returns per-hand 21-point landmarks
// in normalised [0, 1] image-space coordinates. Caller uses these for
// gesture / presence-based liveness signals.
//
// Lazy-loaded: warmup() defers WASM + model fetch until first detect().
// Off by default at the SpoofDetector layer (`enableHandTracking: false`).

import type {
  HandLandmarker,
  HandLandmarkerResult,
  NormalizedLandmark,
} from "@mediapipe/tasks-vision";

const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const DEFAULT_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

export interface MediaPipeHandDetectorOptions {
  /** Override the hand landmarker .task URL. Default = MediaPipe CDN. */
  modelAssetPath?: string;
  /** Override the MediaPipe WASM base URL. */
  wasmBaseUrl?: string;
  /** Max hands to track per frame. Default 2. */
  numHands?: number;
  /** "IMAGE" or "VIDEO" running mode. Default "VIDEO". */
  runningMode?: "IMAGE" | "VIDEO";
  /** Min detection confidence. Default 0.5. */
  minHandDetectionConfidence?: number;
  /** Min tracking confidence between frames. Default 0.5. */
  minTrackingConfidence?: number;
}

export type HandFrameSource =
  | HTMLVideoElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap;

/** Concise per-hand summary the analyzer actually uses. */
export interface DetectedHand {
  /** 21 NormalizedLandmark points (x/y/z ∈ [0, 1]). */
  landmarks: ReadonlyArray<NormalizedLandmark>;
  /** "Left" | "Right" handedness (MediaPipe convention). */
  handedness?: string;
  /** Detector confidence for the handedness classification. */
  handednessScore?: number;
}

export class MediaPipeHandDetector {
  private readonly options: MediaPipeHandDetectorOptions;
  private landmarker: HandLandmarker | null = null;
  private warmupPromise: Promise<void> | null = null;

  constructor(options: MediaPipeHandDetectorOptions = {}) {
    this.options = options;
  }

  async warmup(): Promise<void> {
    if (this.landmarker) return;
    if (this.warmupPromise) return this.warmupPromise;
    this.warmupPromise = this.doWarmup();
    return this.warmupPromise;
  }

  private async doWarmup(): Promise<void> {
    const tasksVision = await import("@mediapipe/tasks-vision");
    const { FilesetResolver, HandLandmarker } = tasksVision;
    const wasmBase = this.options.wasmBaseUrl ?? DEFAULT_WASM_BASE;
    const resolver = await FilesetResolver.forVisionTasks(wasmBase);
    this.landmarker = await HandLandmarker.createFromOptions(resolver, {
      baseOptions: {
        modelAssetPath: this.options.modelAssetPath ?? DEFAULT_MODEL_URL,
      },
      runningMode: this.options.runningMode ?? "VIDEO",
      numHands: this.options.numHands ?? 2,
      minHandDetectionConfidence: this.options.minHandDetectionConfidence ?? 0.5,
      minTrackingConfidence: this.options.minTrackingConfidence ?? 0.5,
    });
  }

  async detect(
    image: HandFrameSource,
    timestampMs: number,
  ): Promise<DetectedHand[]> {
    await this.warmup();
    if (!this.landmarker) return [];
    let result: HandLandmarkerResult | null = null;
    if (this.options.runningMode === "IMAGE") {
      result = this.landmarker.detect(image as never);
    } else {
      result = this.landmarker.detectForVideo(image as never, timestampMs);
    }
    if (!result?.landmarks || result.landmarks.length === 0) return [];
    const out: DetectedHand[] = [];
    for (let i = 0; i < result.landmarks.length; i++) {
      const handednessArr = result.handedness?.[i];
      const top = handednessArr?.[0];
      out.push({
        landmarks: result.landmarks[i],
        handedness: top?.categoryName,
        handednessScore: top?.score,
      });
    }
    return out;
  }

  async close(): Promise<void> {
    if (this.landmarker) {
      this.landmarker.close();
      this.landmarker = null;
    }
  }
}
