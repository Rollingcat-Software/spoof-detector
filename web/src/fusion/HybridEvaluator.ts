// Port of src/fusion/hybrid_evaluator.py
//
// Algorithm origin: Aysenur (calibrated hybrid fusion of pretrained MiniFASNet
// output + heuristic device-replay signals — moire, flash, device-replay,
// rPPG sanity). Authored before MultiClassFuser; coexists with it. See
// COMPARISON_AYSENUR_vs_PRODUCTIZED.md.
//
// Calibration constants (weights summing to 1.0, threshold 0.45, flash
// normalization 0.02..0.15, high-flicker gates 0.85 / (0.75 & 0.55)) are
// preserved verbatim from the Python source.
//
// Single-frame: when callers supply at most one flash response sample, the
// flash signal is held at the neutral 0.5 — same shortcut as Python.
//
// All numeric inputs are coerced through _coerceFloat which rejects booleans
// and non-finite values, matching Python's behavior (bool is a subtype of
// int there; we still want to reject it explicitly).

/** Weights for combining model and heuristic spoof signals. Must sum to 1.0. */
export interface FusionWeights {
  pretrained_model: number;
  flash_response: number;
  moire_pattern: number;
  device_replay: number;
}

/** Default weights — mirror Python `FusionWeights()` defaults exactly. */
export const DEFAULT_FUSION_WEIGHTS: Readonly<FusionWeights> = Object.freeze({
  pretrained_model: 0.30,
  flash_response: 0.30,
  moire_pattern: 0.20,
  device_replay: 0.20,
});

/** Hybrid fusion decision output. */
export interface FusionResult {
  is_spoof: boolean;
  /** [0, 1] decision confidence (margin from threshold, normalized). */
  confidence: number;
  /** [0, 1] fused spoof probability. */
  spoof_score: number;
  /** Per-signal contributions plus passthrough of caller-supplied signals. */
  breakdown: Record<string, number>;
  /** Human-readable reasoning string. */
  reasoning: string;
}

/** Custom signal payload — duck-typed to mirror the Python dict[str, Any]. */
export type CustomSignals = Record<string, unknown>;

/** Validate fusion weights sum to ~1.0 (np.isclose default rtol=1e-5, atol=1e-8). */
function validateWeights(w: FusionWeights): void {
  const total =
    w.pretrained_model + w.flash_response + w.moire_pattern + w.device_replay;
  // np.isclose default: atol=1e-8, rtol=1e-5; |a-b| <= atol + rtol*|b|
  const tolerance = 1e-8 + 1e-5 * 1.0;
  if (Math.abs(total - 1.0) > tolerance) {
    throw new Error(`Weights must sum to 1.0, got ${total}`);
  }
}

/** Fuse pretrained liveness output with replay- and physiology-based signals. */
export class HybridFusionEvaluator {
  readonly weights: FusionWeights;
  readonly threshold: number;

  constructor(opts?: { weights?: FusionWeights; threshold?: number }) {
    const w = opts?.weights ?? DEFAULT_FUSION_WEIGHTS;
    validateWeights(w);
    // Freeze a copy so external mutation doesn't desync the validation.
    this.weights = Object.freeze({ ...w });
    this.threshold = opts?.threshold !== undefined ? Number(opts.threshold) : 0.45;
  }

  /** Combine all signals into a final spoof decision. */
  evaluate(
    pretrained_spoof_score: number,
    custom_signals: CustomSignals,
  ): FusionResult {
    const pretrained_score = HybridFusionEvaluator.clamp01(pretrained_spoof_score);
    const flicker_score = this.resolveNumericSignal(
      custom_signals["flicker_score"],
      0.0,
    );
    const device_replay_score = this.resolveNumericSignal(
      custom_signals["device_replay_score"],
      0.0,
    );

    // High-flicker early-exit (matches Python branch verbatim, including the
    // reasoning-string asymmetry — the >0.85 branch reports flicker only).
    if (
      flicker_score > 0.85 ||
      (flicker_score > 0.75 && device_replay_score > 0.55)
    ) {
      const reasoning =
        flicker_score <= 0.85
          ? `High flicker (${flicker_score.toFixed(2)}) + device replay (${device_replay_score.toFixed(2)})`
          : `Very high flicker detected (${flicker_score.toFixed(2)})`;
      const breakdown: Record<string, number> = {
        pretrained: pretrained_score,
        flicker: flicker_score,
        device_replay: device_replay_score,
      };
      // Passthrough any caller-supplied NUMERIC signals (Python does
      // **custom_signals; we mirror by including only finite numbers so
      // the breakdown stays Record<string, number>).
      for (const k of Object.keys(custom_signals)) {
        const v = HybridFusionEvaluator.coerceFloat(custom_signals[k]);
        if (v !== null && !(k in breakdown)) {
          breakdown[k] = v;
        }
      }
      return {
        is_spoof: true,
        confidence: 0.90,
        spoof_score: 0.90,
        breakdown,
        reasoning,
      };
    }

    const signal_scores = this.computeSignalScores(custom_signals);
    const final_spoof_score = HybridFusionEvaluator.clamp01(
      this.weights.pretrained_model * pretrained_score +
        this.weights.flash_response * signal_scores.flash +
        this.weights.moire_pattern * signal_scores.moire +
        this.weights.device_replay * signal_scores.device,
    );
    const is_spoof = final_spoof_score > this.threshold;
    const confidence = this.decisionConfidence(final_spoof_score);
    const breakdown: Record<string, number> = {
      pretrained: pretrained_score,
      flash: signal_scores.flash,
      moire: signal_scores.moire,
      device: signal_scores.device,
    };
    const reasoning = HybridFusionEvaluator.generateReasoning(
      is_spoof,
      final_spoof_score,
      breakdown,
    );
    return {
      is_spoof,
      confidence,
      spoof_score: final_spoof_score,
      breakdown,
      reasoning,
    };
  }

  // -- internals --------------------------------------------------------

  private computeSignalScores(signals: CustomSignals): {
    flash: number;
    moire: number;
    device: number;
  } {
    const flash_score = this.resolveFlashScore(signals);
    const moire_raw =
      signals["moire_score"] !== undefined
        ? signals["moire_score"]
        : signals["moire_risk"];
    const moire_score = this.resolveNumericSignal(moire_raw, 0.5);
    const device_raw =
      signals["device_replay_score"] !== undefined
        ? signals["device_replay_score"]
        : signals["device_replay_risk"];
    const device_score = this.resolveNumericSignal(device_raw, 0.5);
    return { flash: flash_score, moire: moire_score, device: device_score };
  }

  private resolveFlashScore(signals: CustomSignals): number {
    const flash_samples = HybridFusionEvaluator.coerceFloat(
      signals["flash_response_samples"] !== undefined
        ? signals["flash_response_samples"]
        : signals["flash_response_sample_count"],
    );
    if (flash_samples !== null && flash_samples < 1.0) {
      return 0.5;
    }

    const flash_response_score = HybridFusionEvaluator.coerceFloat(
      signals["flash_response_score"],
    );
    if (flash_response_score !== null) {
      return HybridFusionEvaluator.clamp01(1.0 - flash_response_score);
    }

    const flash_response = HybridFusionEvaluator.coerceFloat(
      signals["flash_response"],
    );
    if (flash_response === null) {
      return 0.5;
    }
    return this.normalizeFlashScore(flash_response);
  }

  private resolveNumericSignal(value: unknown, neutral: number): number {
    const numeric = HybridFusionEvaluator.coerceFloat(value);
    if (numeric === null) return neutral;
    return HybridFusionEvaluator.clamp01(numeric);
  }

  private normalizeFlashScore(flash_response: number): number {
    if (flash_response >= 0.15) return 0.0;
    if (flash_response <= 0.02) return 1.0;
    return HybridFusionEvaluator.clamp01(
      1.0 - (flash_response - 0.02) / (0.15 - 0.02),
    );
  }

  private decisionConfidence(spoof_score: number): number {
    const decision_margin = Math.abs(spoof_score - this.threshold);
    const max_margin = Math.max(this.threshold, 1.0 - this.threshold, 1e-6);
    return HybridFusionEvaluator.clamp01(decision_margin / max_margin);
  }

  private static generateReasoning(
    is_spoof: boolean,
    score: number,
    breakdown: Record<string, number>,
  ): string {
    let top_key = "";
    let top_val = -Infinity;
    for (const k of Object.keys(breakdown)) {
      const v = breakdown[k];
      if (v > top_val) {
        top_val = v;
        top_key = k;
      }
    }
    if (is_spoof) {
      return `SPOOF detected (score=${score.toFixed(2)}). Primary indicator: ${top_key} (${top_val.toFixed(2)})`;
    }
    return `LIVE verified (score=${score.toFixed(2)}). Strongest remaining spoof cue: ${top_key} (${top_val.toFixed(2)})`;
  }

  /**
   * Coerce a duck-typed value to float. Rejects:
   *  - null / undefined
   *  - booleans (Python's `isinstance(value, bool)` guard)
   *  - values that fail `Number(...)` conversion or yield NaN/±Infinity
   */
  private static coerceFloat(value: unknown): number | null {
    if (value === null || value === undefined) return null;
    if (typeof value === "boolean") return null;
    if (typeof value === "number") {
      if (!Number.isFinite(value)) return null;
      return value;
    }
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed.length === 0) return null;
      const n = Number(trimmed);
      if (!Number.isFinite(n)) return null;
      return n;
    }
    return null;
  }

  private static clamp01(value: number): number {
    return Math.max(0.0, Math.min(1.0, Number(value)));
  }
}
