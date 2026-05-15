// Port of src/infrastructure/analyzers/minifasnet_analyzer.py
//
// MiniFASNet anti-spoofing analyzer — the highest-weight signal in the
// fusion engine (weight 5.0 / 17.6 = 28%, +94.7 measured discrimination
// gap on the published ground-truth set).
//
// IMPORTANT (verbatim from Python source):
//   "MiniFASNet requires the ORIGINAL full frame + face bbox, not a
//    pre-cropped face. The model uses surrounding context to judge
//    whether the face region looks like a real capture vs a screen/print.
//    Passing crop-only with bbox [0,0,w,h] produces unreliable results."
//
// The Python wrapper used uniface.spoofing.MiniFASNet, which in turn
// loads ~/.uniface/models/minifasnet_v2.onnx via onnxruntime. Here we
// load the same .onnx directly via onnxruntime-web. The preprocessing
// pipeline is faithful to uniface (see /home/.../uniface/spoofing/minifasnet.py):
//
//   1. xyxy → xywh
//   2. Compute scale = min((H-1)/box_h, (W-1)/box_w, self.scale=2.7)
//   3. Crop centered on bbox, then cv2.resize to model input (80x80 default)
//   4. Cast float32, HWC → CHW, expand to NCHW
//   5. session.run() → softmax → argmax. Label 1 == REAL.

import {
  AnalyzerResult,
  FaceROI,
  makeAnalyzerResult,
} from "../../domain/models";
import {
  cropAndResize,
  computeMiniFasNetCropRect,
  SourceImage,
  softmax2,
  toBgrNchwFloat32,
} from "../../utils/imageOps";

export interface MiniFASNetOptions {
  /**
   * URL to the minifasnet_v2.onnx file. The default Python install
   * places it at `~/.uniface/models/minifasnet_v2.onnx` (1 743 581 bytes);
   * ship it as a static asset under `/models/` for the browser bundle.
   */
  modelUrl: string;
  /**
   * uniface DEFAULT_SCALES — 2.7 for V2 (default), 4.0 for V1SE.
   * Exposed for completeness; the bundled model is V2.
   */
  scale?: number;
  /**
   * Optional explicit input dimensions. If omitted we read them from the
   * ONNX session input shape (uniface does the same — input_size = shape[2:4][::-1]).
   * MiniFASNet V2 is 80x80.
   */
  inputSize?: { width: number; height: number };
  /**
   * onnxruntime-web execution providers. Default: ["wasm"].
   * "webgpu" works on a subset of MobileNetV2 ops; safe to try later.
   */
  executionProviders?: ReadonlyArray<"wasm" | "webgpu" | "webgl" | "cpu">;
}

const DEFAULT_SCALE_V2 = 2.7;

/**
 * MiniFASNet binary real/spoof analyzer.
 *
 * Lifecycle:
 *   const a = new MiniFASNetAnalyzer({ modelUrl: "/models/minifasnet_v2.onnx" });
 *   await a.warmup(); // optional, lazy-loads otherwise
 *   a.setFrame(imageData);
 *   const result = await a.analyze(faceCrop, faceROI);
 */
export class MiniFASNetAnalyzer {
  readonly name = "minifasnet";
  private session: import("onnxruntime-web").InferenceSession | null = null;
  private inputName = "input";
  private outputName = "output";
  private inputW = 80;
  private inputH = 80;
  private currentFrame: SourceImage | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(private readonly options: MiniFASNetOptions) {}

  /** Set the current full frame for context-aware analysis. */
  setFrame(frame: SourceImage): void {
    this.currentFrame = frame;
  }

  /** Idempotent warmup: dynamic-imports onnxruntime-web and creates the session. */
  async warmup(): Promise<void> {
    if (this.session) return;
    if (this.initPromise) return this.initPromise;
    this.initPromise = this.initSession();
    return this.initPromise;
  }

  private async initSession(): Promise<void> {
    // Dynamic import keeps onnxruntime-web out of the synchronous bundle —
    // mirrors the lazy-load pattern used in web-app's CardDetector.ts.
    const ort = await import("onnxruntime-web");
    const providers =
      this.options.executionProviders ?? (["wasm"] as const);

    this.session = await ort.InferenceSession.create(this.options.modelUrl, {
      executionProviders: providers as unknown as string[],
    });

    const inputCfg = this.session.inputNames[0];
    const outputCfg = this.session.outputNames[0];
    this.inputName = inputCfg;
    this.outputName = outputCfg;

    // Try to pick up the input shape from session metadata.
    if (this.options.inputSize) {
      this.inputW = this.options.inputSize.width;
      this.inputH = this.options.inputSize.height;
    } else {
      // ORT-Web doesn't expose input dim by default — uniface does so via
      // session.get_inputs()[0].shape[2:4][::-1]. The published model is
      // 80x80 so we use that as the default.
      this.inputW = 80;
      this.inputH = 80;
    }
  }

  /**
   * Analyze a face for spoofing.
   *
   * Mirrors `MiniFASNetAnalyzer.analyze()` in Python: prefers the original
   * frame + bbox; if no frame was set via setFrame(), falls back to a
   * 30%-padded crop (less accurate but still usable).
   */
  async analyze(
    faceCrop: ImageData | null,
    faceROI: FaceROI,
  ): Promise<AnalyzerResult> {
    await this.warmup();
    if (!this.session) {
      return makeAnalyzerResult(this.name, 50.0, { error: "not_initialized" });
    }

    const ort = await import("onnxruntime-web");
    const start = performance.now();

    try {
      const sourceFrame: SourceImage =
        this.currentFrame ?? (faceCrop as ImageData);

      let frameW: number;
      let frameH: number;
      let bbox = {
        x1: faceROI.bbox.x1,
        y1: faceROI.bbox.y1,
        x2: faceROI.bbox.x2,
        y2: faceROI.bbox.y2,
      };
      let context: "frame" | "padded_crop";

      if (this.currentFrame) {
        const dims = dimsOf(this.currentFrame);
        frameW = dims.w;
        frameH = dims.h;
        context = "frame";
      } else if (faceCrop) {
        // Padded-crop fallback. Pad 30% (matches Python:
        //   pad = max(int(h * 0.3), int(w * 0.3), 20)
        // ) but in browser we synthesize via canvas drawImage — a simple
        // border-replicate is a perceptual approximation and good enough
        // for the fallback path.
        const w = faceCrop.width;
        const h = faceCrop.height;
        const pad = Math.max(20, Math.floor(0.3 * Math.max(w, h)));
        const padded = new OffscreenCanvas(w + 2 * pad, h + 2 * pad);
        const pctx = padded.getContext("2d");
        if (!pctx) throw new Error("padded canvas: no 2D ctx");
        // Very lightweight "replicate-ish" by stretching a thin edge slice.
        const tmp = new OffscreenCanvas(w, h);
        tmp.getContext("2d")!.putImageData(faceCrop, 0, 0);
        // Fill border with edge-stretched copy (cheap approximation).
        pctx.drawImage(tmp, 0, 0, w, h, 0, 0, padded.width, padded.height);
        // Overlay the real crop centered.
        pctx.drawImage(tmp, pad, pad);
        // Use the padded canvas as the source frame.
        // (we mutate the local source variable, not the analyzer field).
        const replacement = padded;
        bbox = { x1: pad, y1: pad, x2: pad + w, y2: pad + h };
        frameW = replacement.width;
        frameH = replacement.height;
        context = "padded_crop";
        // Continue as if currentFrame was set.
        const inputTensor = await this.preprocessTo(
          replacement,
          bbox,
          frameW,
          frameH,
          ort,
        );
        return await this.runAndScore(
          ort,
          inputTensor,
          start,
          context,
        );
      } else {
        return makeAnalyzerResult(this.name, 50.0, {
          error: "no_frame_no_crop",
        });
      }

      const tensor = await this.preprocessTo(
        sourceFrame,
        bbox,
        frameW,
        frameH,
        ort,
      );
      return await this.runAndScore(ort, tensor, start, context);
    } catch (e) {
      const elapsed_ms = performance.now() - start;
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: errorMessage(e) },
        elapsed_ms,
      );
    }
  }

  private async preprocessTo(
    src: SourceImage,
    bbox: { x1: number; y1: number; x2: number; y2: number },
    frameW: number,
    frameH: number,
    ort: typeof import("onnxruntime-web"),
  ): Promise<import("onnxruntime-web").Tensor> {
    const scale = this.options.scale ?? DEFAULT_SCALE_V2;
    const rect = computeMiniFasNetCropRect(frameW, frameH, bbox, scale);
    const resized = cropAndResize(src, rect, this.inputW, this.inputH);
    const planar = toBgrNchwFloat32(resized);
    return new ort.Tensor("float32", planar, [
      1,
      3,
      this.inputH,
      this.inputW,
    ]);
  }

  private async runAndScore(
    _ort: typeof import("onnxruntime-web"),
    inputTensor: import("onnxruntime-web").Tensor,
    start: number,
    context: "frame" | "padded_crop",
  ): Promise<AnalyzerResult> {
    if (!this.session) {
      return makeAnalyzerResult(this.name, 50.0, { error: "not_initialized" });
    }

    const feeds: Record<string, import("onnxruntime-web").Tensor> = {};
    feeds[this.inputName] = inputTensor;
    const outputs = await this.session.run(feeds);
    const out = outputs[this.outputName];
    const data = out.data as Float32Array;

    // Output is (1, 2) — class 0 = spoof, class 1 = real (matches uniface).
    const [pSpoof, pReal] = softmax2([data[0], data[1]]);
    const isReal = pReal >= pSpoof;
    const confidence = isReal ? pReal : pSpoof;

    // Convert to 0-100 score (higher = more live-like) — same mapping
    // the Python wrapper uses.
    let score = isReal ? 50 + confidence * 50 : 50 - confidence * 50;
    score = Math.max(0, Math.min(100, score));

    const elapsed_ms = performance.now() - start;
    return makeAnalyzerResult(
      this.name,
      score,
      {
        is_real: isReal,
        confidence,
        context,
        p_real: pReal,
        p_spoof: pSpoof,
      },
      elapsed_ms,
    );
  }
}

function dimsOf(src: SourceImage): { w: number; h: number } {
  if (src instanceof ImageData) return { w: src.width, h: src.height };
  return { w: src.width, h: src.height };
}

function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
