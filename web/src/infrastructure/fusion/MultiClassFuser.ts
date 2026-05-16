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
 * Mirrors `DEFAULT_ANALYZER_WEIGHTS` in multi_class_fuser.py exactly.
 */
export const DEFAULT_ANALYZER_WEIGHTS: Readonly<Record<string, number>> = {
  minifasnet: 5.0,        // PROVEN: +94.7 gap
  screen_flicker: 3.0,    // NEW: 50/60Hz temporal detection — catches ANY screen
  device_boundary: 2.5,   // GOOD: physical bezel detection
  micro_tremor: 2.5,      // NEW: 8-12Hz oscillation — catches video replay
  landmark_variance: 2.0, // STRONG: zero variance = photo
  background_grid: 1.5,   // NEW: background stability for proctoring
  rppg: 0.5,              // ACTIVE in browser bundle (2026-05-16): pulse from green channel
  blink: 0.5,             // MODERATE: blink count
  screen_replay: 0.5,     // WEAK: +9.6 gap
  ar_filter: 0.3,         // MODERATE: heuristic mode
  temporal: 0.3,          // NEUTRAL: micro-motion
  texture: 0.1,           // ANTI-CORRELATED: suppressed
  moire: 0.1,             // ANTI-CORRELATED: suppressed
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
