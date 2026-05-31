// Port of src/infrastructure/fusion/multi_class_fuser.py
//
// Combines per-analyzer scores into a probability distribution
// over the 7-category spoof taxonomy.
//
// Evidence-based weighting: analyzers with proven discrimination
// power get higher weights. Analyzers that are anti-correlated
// (score spoofs higher than real) get near-zero weight.
//
// Calibration data (from analyze_captures.py ground truth test):
//   MiniFASNet:       real=99.9  spoof=5.1   gap=+94.7  GOOD
//   screen_replay:    real=46.7  spoof=37.1  gap=+9.6   WEAK
//   device_boundary:  (new, untested — expected HIGH)
//   moire:            real=39.1  spoof=44.1  gap=-5.0   ANTI-CORRELATED
//   texture:          real=72.1  spoof=78.4  gap=-6.3   ANTI-CORRELATED
//   temporal:         real=90.0  (single-frame only)     NEUTRAL

import {
  ALL_SPOOF_CATEGORIES,
  AnalyzerResult,
  classificationFromProbabilities,
  SpoofCategory,
  SpoofClassification,
} from "../../domain/models";
import { SPOOF_SIGNAL_MAP } from "../../domain/taxonomy";

/**
 * Calibrated weights based on measured discrimination power.
 * Higher weight = analyzer score has more influence on final classification.
 *
 * Mirrors `DEFAULT_ANALYZER_WEIGHTS` in multi_class_fuser.py — and now
 * incorporates the paper's leave-one-out findings (`paper/sections/00_abstract.md`):
 *   (i) Laplacian-texture & Gabor-moire are *anti-correlated* on modern
 *       high-resolution capture — re-weighting them from 1.0 to 0.1
 *       recovered 0.017 of the 0.019 AUC gap; here we ship them at 0.0
 *       (skipped by the fuser's `weight <= 0` short-circuit).
 *   (ii) device_boundary & micro_tremor were calibrated on in-house data
 *        and HURT zero-shot CASIA-FASD AUC (−0.027, −0.021). Reduced
 *        from 2.5 → 0.5 so they still contribute but no longer dominate.
 *   (iii) background_grid is the sole positive transferable contributor
 *         (+0.014 LOO AUC). Held at 1.5.
 * Consumers can re-enable texture/moire at their original 1.0 via
 * constructor `analyzerWeights` override for use on lower-resolution
 * capture or for ablation studies.
 */
export const DEFAULT_ANALYZER_WEIGHTS: Readonly<Record<string, number>> = {
  minifasnet: 5.0,        // PROVEN: +94.7 gap
  planarity: 2.0,         // 2026-05-24: camera-independent flat-surface (print/screen) detector — see LandmarkPlanarityAnalyzer. Backed by a session-level veto.
  screen_flicker: 3.0,    // 50/60Hz temporal detection — catches ANY screen
  landmark_variance: 2.0, // STRONG: zero variance = photo
  background_grid: 1.5,   // PAPER LOO: +0.014 AUC, sole positive transfer
  device_boundary: 0.5,   // PAPER LOO: -0.027 AUC zero-shot, reduced 2.5→0.5
  micro_tremor: 0.5,      // PAPER LOO: -0.021 AUC zero-shot, reduced 2.5→0.5
  rppg: 0.5,              // green-channel pulse
  blink: 0.5,             // EAR over time
  screen_replay: 0.5,     // WEAK: +9.6 gap
  ar_filter: 0.3,         // heuristic mode
  temporal: 0.3,          // micro-motion
  // 2026-05-31 in-house frame-log dataset (3 LIVE / 3 SPOOF including a
  // live video-call) measured texture.score cross-session AUC = 0.919,
  // d-prime 2.67 (LIVE 71.0 vs SPOOF 55.9). Cross-dataset CASIA-FASD
  // judged it anti-correlated (which is why this was 0.0); in-house it
  // is one of the strongest passive replay signals. The bulk of the work
  // for replay detection is done by SessionEngine.checkTextureCollapseReplay,
  // which vetoes on the texture_score SUB-feature (cliff 100→16) and
  // raises VIDEO_REPLAY incidents. The 1.5 weight here is the modest
  // top-line nudge to keep evidence accumulating even when the veto
  // hasn't reached its 3-incident override threshold yet.
  texture: 1.5,           // IN-HOUSE: AUC 0.919; CASIA cross-dataset anti-correlated — pair with checkTextureCollapseReplay veto
  moire: 0.0,             // PAPER ANTI-CORRELATED — disabled by default (in-house sub-feature std_mean AUC 0.91 but top-line score not separating; keep at 0 until a sub-feature is promoted)
};

/**
 * Nyquist floor (2026-05-24): the minimum measured frame rate at which each
 * frequency-domain analyzer can actually resolve its target band. Below this
 * the analyzer is BLIND (its band sits above fps/2) and its reading is
 * meaningless — yet it still emits a confident score. On the ~8-13 fps this
 * runs at in practice, `screen_flicker` (8-35 Hz bands) reports a confident
 * "no flicker -> 85 (live)" that PROPS UP screen-replay attacks, and
 * `micro_tremor` (8-12 Hz) / `rppg` (0.75-4 Hz) alias to a confident
 * "spoof-like" LOW that drags genuine faces down. The fuser skips any of
 * these whose `measured_fps` is below its floor, so a blind/aliased reading
 * never enters the evidence — leaving the fps-independent signals (MiniFASNet,
 * planarity, landmark variance, blink) to carry the verdict.
 */
export const NYQUIST_MIN_FPS: Readonly<Record<string, number>> = {
  screen_flicker: 18, // lowest band 8 Hz needs fps/2 >= 8 → >=16; 18 for margin
  micro_tremor: 20,   // 8-12 Hz tremor band
  rppg: 10,           // 0.75-4 Hz pulse band (top alias guard)
};

/**
 * Fuses analyzer scores into per-category probabilities.
 *
 * For each analyzer:
 *  - A HIGH score (close to 100) means "live-like" → increases P(REAL)
 *  - A LOW score (close to 0) means "spoof-like" → distributes evidence
 *    across spoof categories based on SPOOF_SIGNAL_MAP weights
 *
 * Analyzer weights are calibrated from ground-truth testing.
 */
export class MultiClassFuser {
  private readonly weights: Record<string, number>;

  constructor(analyzerWeights?: Record<string, number>) {
    this.weights = analyzerWeights ?? { ...DEFAULT_ANALYZER_WEIGHTS };
  }

  fuse(
    face_id: number,
    results: Record<string, AnalyzerResult>,
  ): SpoofClassification {
    const evidence: Record<SpoofCategory, number> = {} as Record<
      SpoofCategory,
      number
    >;
    for (const cat of ALL_SPOOF_CATEGORIES) evidence[cat] = 0.0;

    let total_weight = 0.0;

    for (const analyzer_name of Object.keys(results)) {
      const result = results[analyzer_name];
      const weight = this.weights[analyzer_name] ?? 0.5;
      if (weight <= 0) continue;

      // Nyquist gate: drop a frequency-domain analyzer whose measured frame
      // rate is too low to resolve its band — its score is blind/aliased and
      // would otherwise inject false evidence (see NYQUIST_MIN_FPS).
      const minFps = NYQUIST_MIN_FPS[analyzer_name];
      if (minFps !== undefined) {
        const fps = result.details["measured_fps"];
        if (typeof fps === "number" && fps > 0 && fps < minFps) continue;
      }

      total_weight += weight;

      const score = result.score; // 0-100, higher = more live
      const spoof_strength = (100.0 - score) / 100.0; // 0-1

      // High score → evidence for REAL
      evidence[SpoofCategory.REAL] += weight * (score / 100.0);

      // Low score → distribute across spoof categories
      const category_map = SPOOF_SIGNAL_MAP[analyzer_name];
      if (category_map) {
        for (const catKey of Object.keys(category_map) as SpoofCategory[]) {
          const cat_weight = category_map[catKey] ?? 0;
          evidence[catKey] += weight * spoof_strength * cat_weight;
        }
      }
    }

    // Normalize by total_weight (NOT by sum of evidence — matches Python).
    if (total_weight > 0) {
      for (const cat of ALL_SPOOF_CATEGORIES) {
        evidence[cat] = evidence[cat] / total_weight;
      }
    }

    // classificationFromProbabilities re-normalizes so probs sum to 1.0,
    // matching SpoofClassification.from_probabilities in Python.
    return classificationFromProbabilities(face_id, evidence, results);
  }
}
