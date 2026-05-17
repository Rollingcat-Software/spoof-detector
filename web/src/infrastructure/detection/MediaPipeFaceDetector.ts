// Port of src/infrastructure/detection/mediapipe_detector.py
//
// Wraps @mediapipe/tasks-vision's FaceLandmarker to produce the same
// FaceROI shape the rest of the pipeline expects. We use FaceLandmarker
// (not FaceDetector) so downstream analyzers — Blink, Landmark Variance,
// Micro-tremor — get the 478-point mesh for free without a second pass.
//
// The web-app's `useFaceDetection.ts` uses the same library. This wrapper
// is intentionally minimal — refer to that file for the production
// primary+fallback pattern when wiring this into a real app.

import { BBox, FaceROI } from "../../domain/models";
import type {
  FaceLandmarker,
  FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";

export interface MediaPipeFaceDetectorOptions {
  /** URL to face_landmarker.task (e.g. "/models/face_landmarker.task"). */
  modelAssetPath: string;
  /**
   * URL prefix where MediaPipe vision-WASM bundle is hosted.
   * Default: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm".
   */
  wasmBaseUrl?: string;
  /** "VIDEO" (default) for live streams, "IMAGE" for one-shot frames. */
  runningMode?: "VIDEO" | "IMAGE";
  /** Default 1; web-app uses 5 for proctoring. */
  numFaces?: number;
  /** Whether to delegate to GPU when available. Default true. */
  useGpu?: boolean;
}

const DEFAULT_WASM_BASE =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm";

/**
 * Thin wrapper around @mediapipe/tasks-vision FaceLandmarker.
 * Produces FaceROI[] with the same field names the analyzers expect.
 */
export class MediaPipeFaceDetector {
  private landmarker: FaceLandmarker | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(private readonly options: MediaPipeFaceDetectorOptions) {}

  async warmup(): Promise<void> {
    if (this.landmarker) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.initInternal();
    return this.initPromise;
  }

  private async initInternal(): Promise<void> {
    const tasksVision = await import("@mediapipe/tasks-vision");
    const { FilesetResolver, FaceLandmarker } = tasksVision;

    const wasmBase = this.options.wasmBaseUrl ?? DEFAULT_WASM_BASE;
    const resolver = await FilesetResolver.forVisionTasks(wasmBase);

    this.landmarker = await FaceLandmarker.createFromOptions(resolver, {
      baseOptions: {
        modelAssetPath: this.options.modelAssetPath,
        delegate: this.options.useGpu === false ? "CPU" : "GPU",
      },
      runningMode: this.options.runningMode ?? "VIDEO",
      numFaces: this.options.numFaces ?? 1,
      // 52 ARKit-style blendshape coefficients per face (eyebrow / per-eye
      // blink / gaze direction / mouth / cheek / nose / jaw / tongue) — see
      // EyebrowAnalyzer / BlinkSymmetryAnalyzer / GazeAnalyzer /
      // ExpressionDynamicsAnalyzer for consumers. Zero extra model cost.
      outputFaceBlendshapes: true,
      // 4×4 facial transformation matrix (world-space 3D pose) — consumed
      // by Pose3DConsistencyAnalyzer for landmark reprojection checks.
      outputFacialTransformationMatrixes: true,
    });
  }

  /**
   * Detect faces in a frame.
   *
   * The Python detector returns BGR bbox coords in pixel space.
   * MediaPipe's TS API gives us normalized landmarks (0-1). We compute
   * the bbox by min/max over the landmarks * width/height — the same
   * trick `useFaceDetection.ts` uses.
   */
  async detect(
    image: HTMLVideoElement | HTMLCanvasElement | OffscreenCanvas | ImageBitmap,
    timestampMs: number,
    width: number,
    height: number,
  ): Promise<FaceROI[]> {
    await this.warmup();
    if (!this.landmarker) return [];

    const result: FaceLandmarkerResult =
      this.options.runningMode === "IMAGE"
        ? this.landmarker.detect(image as never)
        : this.landmarker.detectForVideo(image as never, timestampMs);

    const out: FaceROI[] = [];
    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      return out;
    }

    for (let faceIdx = 0; faceIdx < result.faceLandmarks.length; faceIdx++) {
      const lmSet = result.faceLandmarks[faceIdx];
      let minX = Infinity;
      let minY = Infinity;
      let maxX = -Infinity;
      let maxY = -Infinity;

      const flat = new Float32Array(lmSet.length * 2);
      for (let i = 0; i < lmSet.length; i++) {
        const lm = lmSet[i];
        const px = lm.x * width;
        const py = lm.y * height;
        flat[2 * i] = px;
        flat[2 * i + 1] = py;
        if (px < minX) minX = px;
        if (py < minY) minY = py;
        if (px > maxX) maxX = px;
        if (py > maxY) maxY = py;
      }

      // Clamp to image bounds, integer pixel coords.
      const x1 = Math.max(0, Math.floor(minX));
      const y1 = Math.max(0, Math.floor(minY));
      const x2 = Math.min(width - 1, Math.ceil(maxX));
      const y2 = Math.min(height - 1, Math.ceil(maxY));

      // Blendshapes — 52 ARKit categories per face. The result shape is
      // result.faceBlendshapes[faceIdx].categories[]: { categoryName, score, … }.
      // We flatten to a Map<categoryName, score> so consumers can do O(1)
      // lookups by name without scanning the array each frame.
      let blendshapes: ReadonlyMap<string, number> | undefined;
      const fbs = result.faceBlendshapes?.[faceIdx];
      if (fbs && Array.isArray(fbs.categories)) {
        const m = new Map<string, number>();
        for (const c of fbs.categories) {
          if (typeof c.categoryName === "string" && typeof c.score === "number") {
            m.set(c.categoryName, c.score);
          }
        }
        blendshapes = m;
      }

      // Facial transformation matrix — 4×4 row-major, 16 floats.
      let transformMatrix: Float32Array | undefined;
      const ftm = result.facialTransformationMatrixes?.[faceIdx];
      if (ftm && Array.isArray(ftm.data) && ftm.data.length === 16) {
        transformMatrix = Float32Array.from(ftm.data);
      }

      // Phase-1 face-id assignment: index-within-frame. This is correct
      // for the common numFaces=1 case (always face_id=0) but does NOT
      // do real cross-frame tracking — that's the FaceTracker port in
      // Phase 2. See SPOOF_DETECTOR_BROWSER_READINESS.md §3.1.
      out.push({
        face_id: out.length,
        bbox: new BBox(x1, y1, x2, y2),
        confidence: 0.95, // MediaPipe FaceLandmarker doesn't expose a per-face score
        landmarks: flat,
        blendshapes,
        transformMatrix,
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
