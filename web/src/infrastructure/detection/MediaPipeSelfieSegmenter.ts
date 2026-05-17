// MediaPipeSelfieSegmenter — Phase D1 (opt-in).
//
// Thin wrapper around @mediapipe/tasks-vision ImageSegmenter loaded with
// Google's hosted `selfie_segmenter` model (~250 KB float16). Returns a
// per-pixel confidence mask: values close to 1.0 are person, close to 0.0
// are background. Consumed by BackgroundMotionAnalyzer to track scene
// motion over time — a real environment has subtle background drift,
// a static phone-screen replay has none.
//
// Lazy-loaded: nothing is fetched or initialised until warmup() is called
// (which happens on the first segment() call). Off by default at the
// SpoofDetector layer (`enableBackgroundSegmentation: false`).

import type {
  ImageSegmenter,
  ImageSegmenterResult,
} from "@mediapipe/tasks-vision";

const DEFAULT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";
const DEFAULT_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

export interface MediaPipeSelfieSegmenterOptions {
  /** Override the segmenter model URL. Defaults to MediaPipe's CDN. */
  modelAssetPath?: string;
  /** Override the MediaPipe WASM base URL. */
  wasmBaseUrl?: string;
  /** "IMAGE" or "VIDEO" running mode. Default "VIDEO". */
  runningMode?: "IMAGE" | "VIDEO";
}

export type SegmenterFrameSource =
  | HTMLVideoElement
  | HTMLCanvasElement
  | OffscreenCanvas
  | ImageBitmap
  | ImageData;

export class MediaPipeSelfieSegmenter {
  private readonly options: MediaPipeSelfieSegmenterOptions;
  private segmenter: ImageSegmenter | null = null;
  private warmupPromise: Promise<void> | null = null;

  constructor(options: MediaPipeSelfieSegmenterOptions = {}) {
    this.options = options;
  }

  async warmup(): Promise<void> {
    if (this.segmenter) return;
    if (this.warmupPromise) return this.warmupPromise;
    this.warmupPromise = this.doWarmup();
    return this.warmupPromise;
  }

  private async doWarmup(): Promise<void> {
    const tasksVision = await import("@mediapipe/tasks-vision");
    const { FilesetResolver, ImageSegmenter } = tasksVision;
    const wasmBase = this.options.wasmBaseUrl ?? DEFAULT_WASM_BASE;
    const resolver = await FilesetResolver.forVisionTasks(wasmBase);
    this.segmenter = await ImageSegmenter.createFromOptions(resolver, {
      baseOptions: {
        modelAssetPath: this.options.modelAssetPath ?? DEFAULT_MODEL_URL,
      },
      runningMode: this.options.runningMode ?? "VIDEO",
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    });
  }

  /**
   * Segment the given frame and return the confidence mask flattened to
   * a Float32Array. Caller is responsible for the mask shape (mask.width
   * × mask.height) — typically smaller than the source frame for speed.
   * Returns null when the segmenter hasn't warmed up or produced a mask
   * for this frame (e.g. dropped because of frame timing).
   */
  async segment(
    image: SegmenterFrameSource,
    timestampMs: number,
  ): Promise<{
    mask: Float32Array;
    width: number;
    height: number;
  } | null> {
    await this.warmup();
    if (!this.segmenter) return null;

    let result: ImageSegmenterResult | null = null;
    if (this.options.runningMode === "IMAGE") {
      // Sync overload.
      result = this.segmenter.segment(image as never);
    } else {
      result = this.segmenter.segmentForVideo(image as never, timestampMs);
    }
    const cm = result?.confidenceMasks;
    if (!cm || cm.length === 0) return null;
    const first = cm[0];
    try {
      const data = first.getAsFloat32Array();
      return { mask: data, width: first.width, height: first.height };
    } finally {
      // MPMask holds GPU resources — release them after we've copied out
      // the float view we needed for stats.
      for (const m of cm) m.close?.();
    }
  }

  async close(): Promise<void> {
    if (this.segmenter) {
      this.segmenter.close();
      this.segmenter = null;
    }
  }
}
