// Port of src/pipeline/assembler.py
//
// Algorithm origin: Aysenur (anti-spoof pipeline assembler, cherry-pick 5/5).
//
// Single duck-typed adapter that runs three independent layers and assembles
// one structured advisory verdict:
//
//   1. FaceUsabilityGate          — pre-liveness frame quality + occlusion
//   2. DeviceSpoofRiskEvaluator   — device replay / moire / flash / cutout
//   3. HybridFusionEvaluator      — fuses pretrained MiniFASNet + device signals
//
// Each layer is optional; missing evaluators are simply skipped. Every
// evaluator call is wrapped in a try/catch — failures are swallowed
// (fail-soft) so an anti-spoof bug can never block verification. The
// recommended_action is informational only ("allow" | "review" | "block").
//
// The TS port replaces the Python `np.ndarray` frame with a `FrameInput`
// union that covers the browser-side surface (`ImageData`, `HTMLCanvasElement`,
// `OffscreenCanvas`, `HTMLVideoElement`). The "non-empty" check is the
// runtime width/height > 0 — equivalent to Python's `frame_bgr.size == 0`.
//
// Duck typing: the three evaluators are declared as TS interfaces describing
// only the methods the assembler invokes. Any object satisfying the shape
// can be injected — matching the Python `Any | None` annotation.

import {
  HybridFusionEvaluator,
  FusionResult,
  CustomSignals,
} from "../fusion/HybridEvaluator";

/** Frame surface accepted by the assembler. Browser-equivalent of np.ndarray. */
export type FrameInput =
  | ImageData
  | HTMLCanvasElement
  | OffscreenCanvas
  | HTMLVideoElement;

/** Result returned by a face-usability gate. Only the fields used here. */
export interface FaceUsabilityGateResult {
  usable?: boolean;
  quality_reason?: string | null;
  physical_occlusion_reason?: string | null;
}

/** Duck-typed face-usability gate. */
export interface IFaceUsabilityGate {
  evaluate(args: {
    frame_bgr: FrameInput;
    landmark_result: unknown;
  }): FaceUsabilityGateResult | Promise<FaceUsabilityGateResult>;
}

/** Result of a device-spoof-risk evaluator. */
export interface DeviceSpoofAssessment {
  to_dict(): Record<string, unknown>;
}

/** Duck-typed device-spoof risk evaluator. */
export interface IDeviceSpoofRiskEvaluator {
  evaluate(args: {
    frame_bgr: FrameInput;
    face_bounding_box?: [number, number, number, number] | null;
  }): DeviceSpoofAssessment | Promise<DeviceSpoofAssessment>;
}

/** Duck-typed hybrid fusion evaluator (the concrete HybridFusionEvaluator fits). */
export interface IHybridFusionEvaluator {
  evaluate(
    pretrained_spoof_score: number,
    custom_signals: CustomSignals,
  ): FusionResult | Promise<FusionResult>;
}

/** Caller-supplied lazy provider for the pretrained spoof score. */
export type PretrainedSpoofScoreProvider = (args: {
  frame_bgr: FrameInput;
}) => number | Promise<number>;

/** Combined assembled output. */
export interface AntispoofPipelineResult {
  /** True iff face-usability gate produced a blocking verdict. */
  face_usability_block: boolean | null;
  /** Human-readable reason for the block (e.g. 'occluded', 'no_face'). */
  face_usability_reason: string | null;
  /** [0, 1] device-replay risk produced by DeviceSpoofRiskEvaluator. */
  device_replay_risk: number | null;
  /** Full device-spoof breakdown (moire/reflection/flicker/flash/cutout/...). */
  device_signals: Record<string, unknown> | null;
  /** Verdict from HybridFusionEvaluator combining pretrained + device signals. */
  hybrid_fusion_is_spoof: boolean | null;
  /** [0, 1] spoof score from HybridFusionEvaluator. */
  hybrid_fusion_score: number | null;
  hybrid_fusion_reasoning: string | null;
  /** Soft recommendation: 'allow' | 'review' | 'block'. */
  recommended_action: "allow" | "review" | "block";
  /** Layers that actually executed (informational, ordered). */
  layers_evaluated: readonly string[];
}

export interface AntispoofPipelineAssemblerOptions {
  face_usability_gate?: IFaceUsabilityGate | null;
  device_spoof_risk_evaluator?: IDeviceSpoofRiskEvaluator | null;
  hybrid_fusion_evaluator?: IHybridFusionEvaluator | null;
  pretrained_spoof_score_provider?: PretrainedSpoofScoreProvider | null;
}

export interface AntispoofEvaluateOptions {
  frame_bgr: FrameInput;
  landmark_result?: unknown;
  face_bounding_box?: [number, number, number, number] | null;
  pretrained_spoof_score?: number | null;
  /**
   * When true, surfaces `cutout_anomaly_forced` in `layers_evaluated`. The
   * underlying maths is unaffected — DeviceSpoofRisk already runs the
   * cutout detector internally — this just mirrors the Python observability
   * hook for the ANTISPOOF_CUTOUT_ENABLED flag.
   */
  cutout_enabled?: boolean;
}

function frameIsEmpty(frame: FrameInput | null | undefined): boolean {
  if (frame === null || frame === undefined) return true;
  // ImageData / HTMLCanvasElement / OffscreenCanvas / HTMLVideoElement
  // all expose `width` and `height`. HTMLVideoElement uses videoWidth/Height
  // when actively playing; we fall back to that.
  const w =
    "width" in frame && typeof frame.width === "number" ? frame.width : 0;
  const h =
    "height" in frame && typeof frame.height === "number" ? frame.height : 0;
  if (w > 0 && h > 0) return false;
  if (
    typeof HTMLVideoElement !== "undefined" &&
    frame instanceof HTMLVideoElement
  ) {
    return !(frame.videoWidth > 0 && frame.videoHeight > 0);
  }
  return true;
}

function coerceFiniteNumber(v: unknown, fallback: number): number {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return fallback;
}

/**
 * Run the configured layers and assemble a single result. Mirrors the
 * Python class of the same name. Re-exported on the public API surface via
 * a future SpoofDetector integration step (not done here).
 */
export class AntispoofPipelineAssembler {
  private readonly faceUsabilityGate: IFaceUsabilityGate | null;
  private readonly deviceSpoofRiskEvaluator: IDeviceSpoofRiskEvaluator | null;
  private readonly hybridFusionEvaluator: IHybridFusionEvaluator | null;
  private readonly pretrainedSpoofScoreProvider:
    | PretrainedSpoofScoreProvider
    | null;

  constructor(opts: AntispoofPipelineAssemblerOptions = {}) {
    this.faceUsabilityGate = opts.face_usability_gate ?? null;
    this.deviceSpoofRiskEvaluator = opts.device_spoof_risk_evaluator ?? null;
    this.hybridFusionEvaluator = opts.hybrid_fusion_evaluator ?? null;
    this.pretrainedSpoofScoreProvider =
      opts.pretrained_spoof_score_provider ?? null;
  }

  /** Run every layer the assembler was constructed with. */
  async evaluate(opts: AntispoofEvaluateOptions): Promise<AntispoofPipelineResult> {
    const {
      frame_bgr,
      landmark_result = null,
      face_bounding_box = null,
      pretrained_spoof_score = null,
      cutout_enabled = false,
    } = opts;

    if (frameIsEmpty(frame_bgr)) {
      return {
        face_usability_block: null,
        face_usability_reason: null,
        device_replay_risk: null,
        device_signals: null,
        hybrid_fusion_is_spoof: null,
        hybrid_fusion_score: null,
        hybrid_fusion_reasoning: null,
        recommended_action: "allow",
        layers_evaluated: [],
      };
    }

    const layers: string[] = [];
    let face_block: boolean | null = null;
    let face_reason: string | null = null;
    let device_risk: number | null = null;
    let device_signals: Record<string, unknown> | null = null;
    let fusion_is_spoof: boolean | null = null;
    let fusion_score: number | null = null;
    let fusion_reasoning: string | null = null;

    // -- Layer 1: face usability gate ---------------------------------
    if (this.faceUsabilityGate !== null && landmark_result !== null) {
      try {
        const gate_result = await Promise.resolve(
          this.faceUsabilityGate.evaluate({
            frame_bgr,
            landmark_result,
          }),
        );
        // Python: `bool(getattr(gate_result, "usable", True) is False)`
        // Default True ⇒ if .usable missing/true, block=false.
        const usable = gate_result.usable !== undefined ? gate_result.usable : true;
        face_block = usable === false;
        if (face_block) {
          face_reason =
            gate_result.quality_reason ||
            gate_result.physical_occlusion_reason ||
            "unusable_face";
        }
        layers.push("face_usability");
      } catch {
        // fail-soft
      }
    }

    // -- Layer 2: device-spoof risk evaluator -------------------------
    if (this.deviceSpoofRiskEvaluator !== null) {
      try {
        const assessment = await Promise.resolve(
          this.deviceSpoofRiskEvaluator.evaluate({
            frame_bgr,
            face_bounding_box,
          }),
        );
        device_signals = assessment.to_dict();
        device_risk = coerceFiniteNumber(
          device_signals["device_replay_risk"],
          0.0,
        );
        layers.push("device_spoof_risk");
        if (cutout_enabled) {
          layers.push("cutout_anomaly_forced");
        }
      } catch {
        // fail-soft
      }
    }

    // -- Layer 3: hybrid fusion evaluator -----------------------------
    if (this.hybridFusionEvaluator !== null) {
      let score: number | null = pretrained_spoof_score;
      if (score === null && this.pretrainedSpoofScoreProvider !== null) {
        try {
          const v = await Promise.resolve(
            this.pretrainedSpoofScoreProvider({ frame_bgr }),
          );
          const n = Number(v);
          score = Number.isFinite(n) ? n : null;
        } catch {
          score = null;
        }
      }

      if (score !== null && device_signals !== null) {
        try {
          const custom_signals: CustomSignals = {
            flicker_score: coerceFiniteNumber(device_signals["flicker_risk"], 0.0),
            flash_response_score: coerceFiniteNumber(
              device_signals["flash_response_score"],
              0.0,
            ),
            // Python: assembler sees one frame, hard-codes 1 sample.
            flash_response_samples: 1,
            moire_score: coerceFiniteNumber(device_signals["moire_risk"], 0.0),
            device_replay_score: coerceFiniteNumber(device_risk ?? 0.0, 0.0),
          };
          const fusion = await Promise.resolve(
            this.hybridFusionEvaluator.evaluate(Number(score), custom_signals),
          );
          fusion_is_spoof = Boolean(fusion.is_spoof);
          fusion_score = Number(fusion.spoof_score);
          fusion_reasoning = String(fusion.reasoning);
          layers.push("hybrid_fusion");
        } catch {
          // fail-soft
        }
      }
    }

    const action = AntispoofPipelineAssembler.recommend({
      face_block,
      device_risk,
      fusion_is_spoof,
    });

    return {
      face_usability_block: face_block,
      face_usability_reason: face_reason,
      device_replay_risk: device_risk,
      device_signals,
      hybrid_fusion_is_spoof: fusion_is_spoof,
      hybrid_fusion_score: fusion_score,
      hybrid_fusion_reasoning: fusion_reasoning,
      recommended_action: action,
      layers_evaluated: Object.freeze([...layers]),
    };
  }

  /** Soft recommendation. Caller decides whether to enforce. */
  private static recommend(args: {
    face_block: boolean | null;
    device_risk: number | null;
    fusion_is_spoof: boolean | null;
  }): "allow" | "review" | "block" {
    if (args.face_block === true) return "block";
    if (args.fusion_is_spoof === true) return "block";
    if (args.device_risk !== null && args.device_risk >= 0.65) return "review";
    return "allow";
  }
}

// Re-export concrete helper for tests + downstream wiring.
export { HybridFusionEvaluator };
