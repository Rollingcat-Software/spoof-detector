"""Hybrid fusion of pretrained-model + heuristic spoof signals.

Aysenur's R&D fuser. Calibrated on FIVUCSAS scenarios 2026-05; ported here
2026-05-09.
"""

from src.fusion.hybrid_evaluator import (
    FusionWeights,
    HybridFusionEvaluator,
)

__all__ = ["FusionWeights", "HybridFusionEvaluator"]
