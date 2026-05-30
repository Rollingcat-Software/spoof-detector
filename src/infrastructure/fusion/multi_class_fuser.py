"""Multi-class spoof fusion engine.

Combines per-analyzer scores into a probability distribution
over the 7-category spoof taxonomy.

HEURISTIC weighting (NOT a swept/optimised calibration): the weights below are
hand-set from the *sign* of each analyzer's discrimination gap (real-mean minus
spoof-mean). Analyzers with a strong positive gap get higher weights; analyzers
measured to be anti-correlated (score spoofs higher than real) get near-zero
weight. The original 1-D sweep that motivated the texture/moire 0.1 value was run
on an in-house *synthetic same-source* set with the threshold chosen on the test
set; that result is unreproducible and has been withdrawn (see paper §5.4 and
paper/figures/WITHDRAWN_in_house_synthetic_results.md). Do not describe these as
"optimised" or "swept". TODO: re-derive per-operator with a proper Dev/Test split.

Discrimination-gap signs that motivated the weights (indicative magnitudes only;
from the in-house synthetic set, so trust the SIGN, not the absolute value):
  MiniFASNet:       real=99.9  spoof=5.1   gap=+94.7  STRONG POSITIVE
  screen_replay:    real=46.7  spoof=37.1  gap=+9.6   WEAK POSITIVE
  device_boundary:  positive on in-house; HARMS zero-shot transfer (paper §8.2)
  moire:            real=39.1  spoof=44.1  gap=-5.0   ANTI-CORRELATED
  texture:          real=72.1  spoof=78.4  gap=-6.3   ANTI-CORRELATED
  temporal:         real=90.0  (single-frame only)     NEUTRAL
"""

from __future__ import annotations

from src.domain.models import SpoofCategory, SpoofClassification, AnalyzerResult
from src.domain.taxonomy import SPOOF_SIGNAL_MAP

# Heuristic, hand-set weights grounded in per-analyzer discrimination-gap SIGNS
# (see module docstring). NOT a swept/optimised calibration — do not relabel as such.
# Higher weight = analyzer score has more influence on final classification.
# NOTE: these exact values are mirrored by the TypeScript browser port; changing a
# value here changes runtime detector behavior. This integrity cleanup leaves the
# values unchanged and only corrects the provenance description.
DEFAULT_ANALYZER_WEIGHTS: dict[str, float] = {
    "minifasnet": 5.0,          # PROVEN: +94.7 gap
    "screen_flicker": 3.0,     # NEW: 50/60Hz temporal detection — catches ANY screen
    "device_boundary": 2.5,    # GOOD: physical bezel detection
    "micro_tremor": 2.5,       # NEW: 8-12Hz oscillation — catches video replay
    "landmark_variance": 2.0,  # STRONG: zero variance = photo
    "background_grid": 1.5,    # NEW: background stability for proctoring
    "rppg": 0.0,               # DISABLED: needs notch filter fix first
    "blink": 0.5,              # MODERATE: blink count
    "screen_replay": 0.5,      # WEAK: +9.6 gap
    "ar_filter": 0.3,          # MODERATE: heuristic mode
    "temporal": 0.3,           # NEUTRAL: micro-motion
    "texture": 0.1,            # ANTI-CORRELATED: suppressed
    "moire": 0.1,              # ANTI-CORRELATED: suppressed
}


class MultiClassFuser:
    """Fuses analyzer scores into per-category probabilities.

    For each analyzer:
    - A HIGH score (close to 100) means "live-like" → increases P(REAL)
    - A LOW score (close to 0) means "spoof-like" → distributes evidence
      across spoof categories based on SPOOF_SIGNAL_MAP weights

    Analyzer weights are heuristic (hand-set from discrimination-gap signs),
    not a swept/optimised calibration — see module docstring.
    """

    def __init__(self, analyzer_weights: dict[str, float] | None = None):
        self._weights = analyzer_weights or DEFAULT_ANALYZER_WEIGHTS

    def fuse(
        self,
        face_id: int,
        results: dict[str, AnalyzerResult],
    ) -> SpoofClassification:
        evidence: dict[SpoofCategory, float] = {cat: 0.0 for cat in SpoofCategory}
        total_weight = 0.0

        for analyzer_name, result in results.items():
            weight = self._weights.get(analyzer_name, 0.5)
            if weight <= 0:
                continue
            total_weight += weight

            score = result.score  # 0-100, higher = more live
            spoof_strength = (100.0 - score) / 100.0  # 0-1, higher = more spoof

            # High score → evidence for REAL
            evidence[SpoofCategory.REAL] += weight * (score / 100.0)

            # Low score → distribute across spoof categories
            if analyzer_name in SPOOF_SIGNAL_MAP:
                category_map = SPOOF_SIGNAL_MAP[analyzer_name]
                for category, cat_weight in category_map.items():
                    evidence[category] += weight * spoof_strength * cat_weight

        # Normalize to probabilities
        if total_weight > 0:
            for cat in evidence:
                evidence[cat] /= total_weight

        return SpoofClassification.from_probabilities(
            face_id=face_id,
            probs=evidence,
            analyzer_results=results,
        )
