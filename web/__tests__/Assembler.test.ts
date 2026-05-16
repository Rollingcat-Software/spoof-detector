// Behavioural tests for AntispoofPipelineAssembler.
//
// Strategy: exercise the assembler against hand-rolled duck-typed fakes
// (the gate, the device-risk evaluator, the fusion evaluator, the
// pretrained-score provider). The Python source uses Any-typed kwargs;
// the TS port exposes those as interfaces, but tests only need objects
// that implement the right `evaluate(...)` shape.

import { describe, expect, it } from "vitest";
import {
  AntispoofPipelineAssembler,
  type DeviceSpoofAssessment,
  type FaceUsabilityGateResult,
  type FrameInput,
  type IDeviceSpoofRiskEvaluator,
  type IFaceUsabilityGate,
  type IHybridFusionEvaluator,
  type PretrainedSpoofScoreProvider,
} from "../src/pipeline/Assembler";
import {
  HybridFusionEvaluator,
  type FusionResult,
} from "../src/fusion/HybridEvaluator";

// -- Helpers ------------------------------------------------------------

/** Build a minimal ImageData stand-in (16x16, zero-filled). */
function makeFrame(width = 16, height = 16): FrameInput {
  // ImageData isn't trivially constructible in node without a polyfill,
  // but the assembler only inspects width/height (browser-safe access).
  // We synthesize a duck-typed object matching the surface used.
  return {
    width,
    height,
    data: new Uint8ClampedArray(width * height * 4),
    colorSpace: "srgb",
  } as unknown as FrameInput;
}

function makeGate(result: FaceUsabilityGateResult): IFaceUsabilityGate {
  return { evaluate: () => result };
}

function makeFailingGate(): IFaceUsabilityGate {
  return {
    evaluate: () => {
      throw new Error("gate boom");
    },
  };
}

function makeDeviceEvaluator(
  signals: Record<string, unknown>,
): IDeviceSpoofRiskEvaluator {
  const assessment: DeviceSpoofAssessment = {
    to_dict: () => signals,
  };
  return { evaluate: () => assessment };
}

function makeFailingDeviceEvaluator(): IDeviceSpoofRiskEvaluator {
  return {
    evaluate: () => {
      throw new Error("device boom");
    },
  };
}

function makeFusionEvaluator(
  out: FusionResult,
): IHybridFusionEvaluator {
  return { evaluate: () => out };
}

function makeFailingFusionEvaluator(): IHybridFusionEvaluator {
  return {
    evaluate: () => {
      throw new Error("fusion boom");
    },
  };
}

// -- Tests --------------------------------------------------------------

describe("AntispoofPipelineAssembler", () => {
  describe("empty frame", () => {
    it("returns allow when frame is null-ish", async () => {
      const assembler = new AntispoofPipelineAssembler({
        face_usability_gate: makeGate({ usable: false, quality_reason: "x" }),
      });
      const out = await assembler.evaluate({
        frame_bgr: { width: 0, height: 0 } as unknown as FrameInput,
      });
      expect(out.recommended_action).toBe("allow");
      expect(out.layers_evaluated).toEqual([]);
      expect(out.face_usability_block).toBeNull();
    });
  });

  describe("no evaluators wired", () => {
    it("returns allow with empty layers", async () => {
      const assembler = new AntispoofPipelineAssembler();
      const out = await assembler.evaluate({ frame_bgr: makeFrame() });
      expect(out.recommended_action).toBe("allow");
      expect(out.layers_evaluated).toEqual([]);
      expect(out.face_usability_block).toBeNull();
      expect(out.device_replay_risk).toBeNull();
      expect(out.hybrid_fusion_is_spoof).toBeNull();
    });
  });

  describe("face usability gate", () => {
    it("usable=true → no block, layer recorded", async () => {
      const assembler = new AntispoofPipelineAssembler({
        face_usability_gate: makeGate({ usable: true }),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        landmark_result: {},
      });
      expect(out.face_usability_block).toBe(false);
      expect(out.face_usability_reason).toBeNull();
      expect(out.layers_evaluated).toContain("face_usability");
      expect(out.recommended_action).toBe("allow");
    });

    it("usable=false → block + reason → recommended_action=block", async () => {
      const assembler = new AntispoofPipelineAssembler({
        face_usability_gate: makeGate({
          usable: false,
          quality_reason: "occluded",
        }),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        landmark_result: {},
      });
      expect(out.face_usability_block).toBe(true);
      expect(out.face_usability_reason).toBe("occluded");
      expect(out.recommended_action).toBe("block");
    });

    it("usable=false with no reason → 'unusable_face' fallback", async () => {
      const assembler = new AntispoofPipelineAssembler({
        face_usability_gate: makeGate({ usable: false }),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        landmark_result: {},
      });
      expect(out.face_usability_reason).toBe("unusable_face");
    });

    it("missing landmark_result → gate skipped silently", async () => {
      const assembler = new AntispoofPipelineAssembler({
        face_usability_gate: makeGate({ usable: false }),
      });
      const out = await assembler.evaluate({ frame_bgr: makeFrame() });
      expect(out.face_usability_block).toBeNull();
      expect(out.layers_evaluated).not.toContain("face_usability");
    });

    it("gate throws → swallowed, allow", async () => {
      const assembler = new AntispoofPipelineAssembler({
        face_usability_gate: makeFailingGate(),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        landmark_result: {},
      });
      expect(out.face_usability_block).toBeNull();
      expect(out.recommended_action).toBe("allow");
    });
  });

  describe("device-spoof risk", () => {
    it("device_replay_risk ≥ 0.65 → review", async () => {
      const assembler = new AntispoofPipelineAssembler({
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.7,
        }),
      });
      const out = await assembler.evaluate({ frame_bgr: makeFrame() });
      expect(out.device_replay_risk).toBeCloseTo(0.7, 5);
      expect(out.recommended_action).toBe("review");
      expect(out.layers_evaluated).toContain("device_spoof_risk");
    });

    it("device_replay_risk < 0.65 → allow", async () => {
      const assembler = new AntispoofPipelineAssembler({
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.4,
        }),
      });
      const out = await assembler.evaluate({ frame_bgr: makeFrame() });
      expect(out.recommended_action).toBe("allow");
    });

    it("cutout_enabled adds observability layer", async () => {
      const assembler = new AntispoofPipelineAssembler({
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.0,
        }),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        cutout_enabled: true,
      });
      expect(out.layers_evaluated).toContain("cutout_anomaly_forced");
    });

    it("device evaluator throws → swallowed, allow", async () => {
      const assembler = new AntispoofPipelineAssembler({
        device_spoof_risk_evaluator: makeFailingDeviceEvaluator(),
      });
      const out = await assembler.evaluate({ frame_bgr: makeFrame() });
      expect(out.device_replay_risk).toBeNull();
      expect(out.recommended_action).toBe("allow");
    });
  });

  describe("hybrid fusion layer", () => {
    it("is_spoof=true → block (overrides device review)", async () => {
      const fusionOut: FusionResult = {
        is_spoof: true,
        confidence: 0.9,
        spoof_score: 0.9,
        breakdown: {},
        reasoning: "SPOOF synthetic",
      };
      const assembler = new AntispoofPipelineAssembler({
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.7,
        }),
        hybrid_fusion_evaluator: makeFusionEvaluator(fusionOut),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        pretrained_spoof_score: 0.8,
      });
      expect(out.hybrid_fusion_is_spoof).toBe(true);
      expect(out.hybrid_fusion_score).toBe(0.9);
      expect(out.hybrid_fusion_reasoning).toBe("SPOOF synthetic");
      expect(out.recommended_action).toBe("block");
      expect(out.layers_evaluated).toContain("hybrid_fusion");
    });

    it("pretrained provider lazily supplies score", async () => {
      const provider: PretrainedSpoofScoreProvider = () => 0.42;
      const fusionOut: FusionResult = {
        is_spoof: false,
        confidence: 0.1,
        spoof_score: 0.1,
        breakdown: {},
        reasoning: "LIVE",
      };
      const assembler = new AntispoofPipelineAssembler({
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.1,
          flicker_risk: 0.0,
          flash_response_score: 0.0,
          moire_risk: 0.0,
        }),
        hybrid_fusion_evaluator: makeFusionEvaluator(fusionOut),
        pretrained_spoof_score_provider: provider,
      });
      const out = await assembler.evaluate({ frame_bgr: makeFrame() });
      expect(out.hybrid_fusion_is_spoof).toBe(false);
      expect(out.layers_evaluated).toContain("hybrid_fusion");
    });

    it("no pretrained score AND no provider → fusion skipped", async () => {
      const assembler = new AntispoofPipelineAssembler({
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.0,
        }),
        hybrid_fusion_evaluator: makeFusionEvaluator({
          is_spoof: true,
          confidence: 1,
          spoof_score: 1,
          breakdown: {},
          reasoning: "x",
        }),
      });
      const out = await assembler.evaluate({ frame_bgr: makeFrame() });
      expect(out.hybrid_fusion_is_spoof).toBeNull();
      expect(out.layers_evaluated).not.toContain("hybrid_fusion");
    });

    it("fusion evaluator throws → swallowed, allow", async () => {
      const assembler = new AntispoofPipelineAssembler({
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.1,
        }),
        hybrid_fusion_evaluator: makeFailingFusionEvaluator(),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        pretrained_spoof_score: 0.5,
      });
      expect(out.hybrid_fusion_is_spoof).toBeNull();
      expect(out.recommended_action).toBe("allow");
    });

    it("end-to-end with real HybridFusionEvaluator → expected verdict shape", async () => {
      const assembler = new AntispoofPipelineAssembler({
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.1,
          flicker_risk: 0.0,
          flash_response_score: 0.05, // strong flash → flash spoof score 0.95
          moire_risk: 0.05,
        }),
        hybrid_fusion_evaluator: new HybridFusionEvaluator(),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        pretrained_spoof_score: 0.05,
      });
      expect(out.layers_evaluated).toEqual([
        "device_spoof_risk",
        "hybrid_fusion",
      ]);
      expect(out.hybrid_fusion_is_spoof).toBe(false);
      expect(typeof out.hybrid_fusion_score).toBe("number");
      expect(typeof out.hybrid_fusion_reasoning).toBe("string");
    });
  });

  describe("all-pass gates → allow", () => {
    it("usable=true + low device + non-spoof fusion → allow + all 3 layers", async () => {
      const assembler = new AntispoofPipelineAssembler({
        face_usability_gate: makeGate({ usable: true }),
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.1,
        }),
        hybrid_fusion_evaluator: makeFusionEvaluator({
          is_spoof: false,
          confidence: 0.2,
          spoof_score: 0.2,
          breakdown: {},
          reasoning: "LIVE",
        }),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        landmark_result: {},
        pretrained_spoof_score: 0.2,
      });
      expect(out.recommended_action).toBe("allow");
      expect(out.layers_evaluated).toEqual([
        "face_usability",
        "device_spoof_risk",
        "hybrid_fusion",
      ]);
      expect(out.face_usability_block).toBe(false);
      expect(out.hybrid_fusion_is_spoof).toBe(false);
    });

    it("one failing gate → block (usability blocks even with allow device)", async () => {
      const assembler = new AntispoofPipelineAssembler({
        face_usability_gate: makeGate({
          usable: false,
          quality_reason: "blur",
        }),
        device_spoof_risk_evaluator: makeDeviceEvaluator({
          device_replay_risk: 0.1,
        }),
      });
      const out = await assembler.evaluate({
        frame_bgr: makeFrame(),
        landmark_result: {},
      });
      expect(out.recommended_action).toBe("block");
      expect(out.face_usability_reason).toBe("blur");
    });
  });
});
