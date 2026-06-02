// FoundationModelAnalyzer — a vision-foundation-model face-anti-spoofing head
// that runs on the WebGPU execution provider.
//
// WHY THIS EXISTS
//   The hand-tuned analyzer bank does not transfer across subjects/cameras
//   (paper §8.2: MiniFASNet alone beats the calibrated hybrid zero-shot). The
//   2025-26 FAS literature's answer to domain generalization is foundation-model
//   features (DINOv2 / CLIP) + a small trained head. This analyzer is the
//   browser half of that: it loads the ONNX produced by
//   `tools/train_fas_adapter.py` and runs it on the client GPU via WebGPU.
//
// WHY WebGPU HERE (and not for MiniFASNet)
//   MiniFASNet is 1.7 MB / 80x80 — WebGPU dispatch overhead exceeds its compute,
//   so it correctly stays on WASM. A DINOv2/CLIP backbone (tens of MB, 224x224)
//   is exactly the "compute-intensive" case where WebGPU wins (~2x vs WASM on a
//   discrete GPU). This is the analyzer that finally justifies the client GPU.
//
// I/O CONTRACT (must match tools/train_fas_adapter.py):
//   input  "pixel_values" : float32 [1, 3, 224, 224], RGB, ImageNet-normalized
//   output "logits"       : float32 [1, 2]   index 0 = SPOOF, index 1 = REAL
//
// STATUS: default OFF and NOT wired into the fusion weights — shipping the
// runtime ahead of a trained model so the integration is testable and the
// detector's calibrated behavior is untouched until a real model + multi-subject
// validation exist.

import {
  AnalyzerResult,
  FaceROI,
  makeAnalyzerResult,
} from "../../domain/models";
import {
  cropAndResize,
  IMAGENET_MEAN,
  IMAGENET_STD,
  SourceImage,
  softmax2,
  toImageData,
  rgbaToRgbNchwFloat32Normalized,
} from "../../utils/imageOps";

export interface FoundationModelOptions {
  /** URL to the exported FAS-head ONNX (see tools/train_fas_adapter.py). */
  modelUrl: string;
  /** Square model input side. DINOv2 /14 wants a multiple of 14; default 224. */
  inputSize?: number;
  /**
   * Fractional padding added around the face bbox before the square crop, so
   * the backbone sees a little context (hair/ears) like training crops do.
   */
  cropPad?: number;
  /** Per-channel normalization (defaults to ImageNet, matching the trainer). */
  mean?: readonly [number, number, number];
  std?: readonly [number, number, number];
  /**
   * onnxruntime-web execution providers. Default: WebGPU-first with WASM
   * fallback — the opposite of MiniFASNet, because this model is heavy enough
   * for WebGPU to pay off.
   */
  executionProviders?: ReadonlyArray<"wasm" | "webgpu" | "webgl" | "cpu">;
}

/**
 * Map raw [spoof, real] logits to the engine's 0-100 score (higher = more
 * live-like) using the same convention as MiniFASNetAnalyzer. Pure + exported
 * so the score mapping is unit-testable without ORT/WebGPU.
 */
export function foundationScoreFromLogits(logits: ArrayLike<number>): {
  score: number;
  pSpoof: number;
  pReal: number;
  isReal: boolean;
  confidence: number;
} {
  const [pSpoof, pReal] = softmax2([logits[0], logits[1]]);
  const isReal = pReal >= pSpoof;
  const confidence = isReal ? pReal : pSpoof;
  // Monotonic in p_real: tie -> 50 (neutral), strong real -> ~100, strong spoof
  // -> ~0. Cleaner than MiniFASNet's `50 ± conf*50` fold (which never emits
  // 25-75 and maps a tie to 75); 50 = neutral matches the fuser's convention.
  const score = Math.max(0, Math.min(100, pReal * 100));
  return { score, pSpoof, pReal, isReal, confidence };
}

export class FoundationModelAnalyzer {
  readonly name = "foundation_fas";
  private session: import("onnxruntime-web").InferenceSession | null = null;
  private inputName = "pixel_values";
  private outputName = "logits";
  private readonly side: number;
  private readonly cropPad: number;
  private readonly mean: readonly [number, number, number];
  private readonly std: readonly [number, number, number];
  private currentFrame: SourceImage | null = null;
  private initPromise: Promise<void> | null = null;

  constructor(private readonly options: FoundationModelOptions) {
    this.side = options.inputSize ?? 224;
    this.cropPad = options.cropPad ?? 0.3;
    this.mean = options.mean ?? IMAGENET_MEAN;
    this.std = options.std ?? IMAGENET_STD;
  }

  /** Set the current full frame so the crop can include facial context. */
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
    const ort = await import("onnxruntime-web");
    const providers =
      this.options.executionProviders ?? (await tryWebGpuProviders());
    try {
      this.session = await ort.InferenceSession.create(this.options.modelUrl, {
        executionProviders: providers as unknown as string[],
      });
      // eslint-disable-next-line no-console
      console.info(
        `[FoundationFAS] ORT session created with EP order: [${providers.join(", ")}]`,
      );
    } catch (primaryErr) {
      if (providers.length > 0 && providers[0] !== "wasm") {
        // eslint-disable-next-line no-console
        console.info(
          `[FoundationFAS] WebGPU EP failed (${
            (primaryErr as Error).message
          }), retrying WASM-only`,
        );
        this.session = await ort.InferenceSession.create(this.options.modelUrl, {
          executionProviders: ["wasm"] as unknown as string[],
        });
      } else {
        throw primaryErr;
      }
    }
    // Honor the actual graph names if they differ from the trainer defaults.
    if (this.session.inputNames[0]) this.inputName = this.session.inputNames[0];
    if (this.session.outputNames[0]) {
      this.outputName = this.session.outputNames[0];
    }
  }

  /** Analyze a face for spoofing. Returns score 0-100 (higher = more live). */
  async analyze(
    faceCrop: ImageData | null,
    faceROI: FaceROI,
  ): Promise<AnalyzerResult> {
    await this.warmup();
    if (!this.session) {
      return makeAnalyzerResult(this.name, 50.0, { error: "not_initialized" });
    }
    const source: SourceImage | null = this.currentFrame ?? faceCrop;
    if (!source) {
      return makeAnalyzerResult(this.name, 50.0, { error: "no_frame_no_crop" });
    }

    const ort = await import("onnxruntime-web");
    const start = performance.now();
    try {
      const rect = this.paddedSquareRect(source.width, source.height, faceROI);
      const resized = cropAndResize(source, rect, this.side, this.side);
      const { data } = toImageData(resized);
      const planar = rgbaToRgbNchwFloat32Normalized(
        data,
        this.side,
        this.side,
        this.mean,
        this.std,
      );
      const input = new ort.Tensor("float32", planar, [1, 3, this.side, this.side]);
      const outputs = await this.session.run({ [this.inputName]: input });
      const logits = outputs[this.outputName].data as Float32Array;
      const { score, pSpoof, pReal, isReal, confidence } =
        foundationScoreFromLogits(logits);
      return makeAnalyzerResult(
        this.name,
        score,
        { is_real: isReal, confidence, p_real: pReal, p_spoof: pSpoof },
        performance.now() - start,
      );
    } catch (e) {
      return makeAnalyzerResult(
        this.name,
        50.0,
        { error: e instanceof Error ? e.message : String(e) },
        performance.now() - start,
      );
    }
  }

  /**
   * Square crop rect around the face bbox, padded by `cropPad` on each side and
   * clamped to the frame. Square (not aspect-preserving) because the model is
   * trained on square resized crops.
   */
  private paddedSquareRect(
    frameW: number,
    frameH: number,
    faceROI: FaceROI,
  ): { x: number; y: number; w: number; h: number } {
    const { x1, y1, x2, y2 } = faceROI.bbox;
    const bw = Math.max(1, x2 - x1);
    const bh = Math.max(1, y2 - y1);
    const cx = x1 + bw / 2;
    const cy = y1 + bh / 2;
    const side = Math.max(bw, bh) * (1 + 2 * this.cropPad);
    let x = Math.floor(cx - side / 2);
    let y = Math.floor(cy - side / 2);
    let s = Math.floor(side);
    // Clamp to frame, keeping it square.
    s = Math.min(s, frameW, frameH);
    x = Math.max(0, Math.min(x, frameW - s));
    y = Math.max(0, Math.min(y, frameH - s));
    return { x, y, w: s, h: s };
  }
}

/**
 * WebGPU-first EP negotiation. Returns `['webgpu','wasm']` when `navigator.gpu`
 * is present so ORT-Web negotiates the fallback chain itself; WASM-only
 * otherwise (Node/SSR, Firefox without the flag, fingerprint-shielded Brave).
 */
async function tryWebGpuProviders(): Promise<
  ReadonlyArray<"wasm" | "webgpu" | "webgl" | "cpu">
> {
  try {
    if (
      typeof navigator !== "undefined" &&
      "gpu" in navigator &&
      (navigator as Navigator & { gpu?: unknown }).gpu != null
    ) {
      return ["webgpu", "wasm"];
    }
  } catch {
    // navigator access can throw under unusual sandboxing — fall through.
  }
  return ["wasm"];
}
