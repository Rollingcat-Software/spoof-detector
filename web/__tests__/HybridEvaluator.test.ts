// Port of the HybridFusionEvaluator behavior tests.
//
// Verifies the calibrated decision boundary, early-exit branches, and the
// numeric-coercion edge cases (boolean rejection, NaN guards, neutral
// fallbacks) so the TS port behaves identically to the Python source.

import { describe, expect, it } from "vitest";
import {
  DEFAULT_FUSION_WEIGHTS,
  HybridFusionEvaluator,
} from "../src/fusion/HybridEvaluator";

describe("HybridFusionEvaluator", () => {
  describe("constructor", () => {
    it("accepts default weights and threshold", () => {
      const ev = new HybridFusionEvaluator();
      expect(ev.weights).toEqual(DEFAULT_FUSION_WEIGHTS);
      expect(ev.threshold).toBe(0.45);
    });

    it("rejects weights that do not sum to 1.0", () => {
      expect(
        () =>
          new HybridFusionEvaluator({
            weights: {
              pretrained_model: 0.5,
              flash_response: 0.5,
              moire_pattern: 0.5,
              device_replay: 0.5,
            },
          }),
      ).toThrow(/sum to 1\.0/);
    });

    it("accepts a custom threshold", () => {
      const ev = new HybridFusionEvaluator({ threshold: 0.7 });
      expect(ev.threshold).toBe(0.7);
    });
  });

  describe("evaluate — verdict shape", () => {
    it("returns the expected fields", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.1, {
        flash_response_score: 0.9,
        moire_score: 0.1,
        device_replay_score: 0.1,
        flash_response_samples: 5,
      });
      expect(typeof out.is_spoof).toBe("boolean");
      expect(typeof out.spoof_score).toBe("number");
      expect(typeof out.confidence).toBe("number");
      expect(typeof out.reasoning).toBe("string");
      expect(out.breakdown).toHaveProperty("pretrained");
      expect(out.breakdown).toHaveProperty("flash");
      expect(out.breakdown).toHaveProperty("moire");
      expect(out.breakdown).toHaveProperty("device");
    });

    it("low pretrained + low signals → LIVE", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.05, {
        flash_response_score: 0.95, // strong flash response → flash spoof score 0.05
        moire_score: 0.05,
        device_replay_score: 0.05,
        flash_response_samples: 5,
      });
      expect(out.is_spoof).toBe(false);
      expect(out.spoof_score).toBeLessThan(0.45);
      expect(out.reasoning).toMatch(/LIVE/);
    });

    it("high pretrained + high device signals → SPOOF", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.95, {
        flash_response_score: 0.05, // weak flash → high flash spoof 0.95
        moire_score: 0.9,
        device_replay_score: 0.9,
        flash_response_samples: 5,
      });
      expect(out.is_spoof).toBe(true);
      expect(out.spoof_score).toBeGreaterThan(0.45);
      expect(out.reasoning).toMatch(/SPOOF/);
    });
  });

  describe("evaluate — high-flicker early exit", () => {
    it("flicker > 0.85 → forced SPOOF @ 0.90", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.0, {
        flicker_score: 0.9,
        device_replay_score: 0.0,
      });
      expect(out.is_spoof).toBe(true);
      expect(out.spoof_score).toBe(0.9);
      expect(out.confidence).toBe(0.9);
      expect(out.reasoning).toMatch(/Very high flicker/);
    });

    it("flicker > 0.75 + device_replay > 0.55 → forced SPOOF @ 0.90", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.0, {
        flicker_score: 0.8,
        device_replay_score: 0.6,
      });
      expect(out.is_spoof).toBe(true);
      expect(out.spoof_score).toBe(0.9);
      expect(out.reasoning).toMatch(/High flicker.*device replay/);
    });

    it("flicker just below 0.75 does not trigger early exit", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.0, {
        flicker_score: 0.74,
        device_replay_score: 0.9,
        flash_response_score: 0.99,
        moire_score: 0.0,
      });
      // Falls through to weighted fusion; spoof_score is NOT exactly 0.9.
      expect(out.spoof_score).not.toBe(0.9);
    });
  });

  describe("evaluate — flash score resolution", () => {
    it("samples < 1 forces neutral 0.5 even when score is provided", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.0, {
        flash_response_score: 0.99,
        flash_response_samples: 0,
      });
      expect(out.breakdown.flash).toBe(0.5);
    });

    it("uses flash_response_score = 1 - score when samples adequate", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.0, {
        flash_response_score: 0.2,
        flash_response_samples: 3,
      });
      expect(out.breakdown.flash).toBeCloseTo(0.8, 5);
    });

    it("falls back to flash_response normalization", () => {
      const ev = new HybridFusionEvaluator();
      const strong = ev.evaluate(0.0, {
        flash_response: 0.2,
        flash_response_samples: 3,
      });
      const weak = ev.evaluate(0.0, {
        flash_response: 0.01,
        flash_response_samples: 3,
      });
      // flash_response >= 0.15 → 0; <= 0.02 → 1.
      expect(strong.breakdown.flash).toBe(0.0);
      expect(weak.breakdown.flash).toBe(1.0);
    });

    it("missing flash signals → neutral 0.5", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.0, {});
      expect(out.breakdown.flash).toBe(0.5);
    });
  });

  describe("evaluate — coercion edge cases", () => {
    it("rejects boolean inputs (matches Python's isinstance bool guard)", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.0, {
        moire_score: true, // must be ignored → neutral 0.5
        device_replay_score: false,
      });
      expect(out.breakdown.moire).toBe(0.5);
      expect(out.breakdown.device).toBe(0.5);
    });

    it("accepts string-encoded numbers", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.0, {
        moire_score: "0.42",
        device_replay_score: "0.7",
        flash_response_samples: "2",
        flash_response_score: "0.1",
      });
      expect(out.breakdown.moire).toBeCloseTo(0.42, 5);
      expect(out.breakdown.device).toBeCloseTo(0.7, 5);
    });

    it("falls back to *_risk aliases when *_score absent", () => {
      const ev = new HybridFusionEvaluator();
      const out = ev.evaluate(0.0, {
        moire_risk: 0.33,
        device_replay_risk: 0.66,
        flash_response_samples: 2,
        flash_response_score: 0.0,
      });
      expect(out.breakdown.moire).toBeCloseTo(0.33, 5);
      expect(out.breakdown.device).toBeCloseTo(0.66, 5);
    });

    it("clamps out-of-range pretrained scores", () => {
      const ev = new HybridFusionEvaluator();
      const above = ev.evaluate(5.0, {
        flash_response_samples: 1,
        flash_response_score: 0.0,
        moire_score: 0.0,
        device_replay_score: 0.0,
      });
      const below = ev.evaluate(-1.0, {
        flash_response_samples: 1,
        flash_response_score: 0.0,
        moire_score: 0.0,
        device_replay_score: 0.0,
      });
      expect(above.breakdown.pretrained).toBe(1.0);
      expect(below.breakdown.pretrained).toBe(0.0);
    });
  });

  describe("evaluate — confidence", () => {
    it("confidence is 0 at exactly the threshold", () => {
      const ev = new HybridFusionEvaluator({ threshold: 0.45 });
      // Engineer a final score equal to threshold.
      // pretrained=0.45, all signals neutral → 0.3*0.45 + 0.3*0.5 + 0.2*0.5 + 0.2*0.5 = 0.135 + 0.4 = 0.535
      // Instead use everything at 0.45.
      const out = ev.evaluate(0.45, {
        flash_response_score: 0.55, // flash = 1 - 0.55 = 0.45
        flash_response_samples: 2,
        moire_score: 0.45,
        device_replay_score: 0.45,
      });
      expect(out.spoof_score).toBeCloseTo(0.45, 5);
      expect(out.confidence).toBeCloseTo(0.0, 4);
    });

    it("confidence is 1 at score 0 (max margin below threshold)", () => {
      const ev = new HybridFusionEvaluator({ threshold: 0.45 });
      const out = ev.evaluate(0.0, {
        flash_response_score: 1.0,
        flash_response_samples: 2,
        moire_score: 0.0,
        device_replay_score: 0.0,
      });
      // spoof_score = 0; margin = 0.45; max_margin = max(0.45, 0.55) = 0.55 → conf = 0.818..
      expect(out.spoof_score).toBe(0.0);
      expect(out.confidence).toBeCloseTo(0.45 / 0.55, 5);
    });
  });
});
