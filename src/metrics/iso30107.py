"""ISO/IEC 30107-3 presentation attack detection metrics.

Reference: ISO/IEC 30107-3:2017 — Information technology — Biometric
presentation attack detection — Part 3: Testing and reporting.

These are the canonical metrics every FAS paper reports:

  APCER (Attack Presentation Classification Error Rate)
    Proportion of presentation attacks classified as bona fide.
    Per attack type, then max across types.

  BPCER (Bona-fide Presentation Classification Error Rate)
    Proportion of bona-fide presentations classified as attacks.

  ACER (Average Classification Error Rate)
    (APCER_max + BPCER) / 2  — single-number summary.

  EER  (Equal Error Rate)
    The threshold at which APCER == BPCER.

The numerical convention here:
- score is a "live-ness" score in [0, 1] — high = bona-fide, low = attack.
- predicted bona-fide iff score >= threshold.

This matches every published OULU-NPU / SiW / CASIA-SURF leaderboard.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np


def apcer(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    attack_types: Sequence[str] | None,
    threshold: float,
) -> tuple[float, dict[str, float]]:
    """Attack Presentation Classification Error Rate.

    Returns:
        (apcer_max, per_type_apcer)
        - apcer_max: max across attack types — the value reported as APCER.
        - per_type_apcer: {attack_type: apcer} broken down per PA species.

    A bona-fide sample contributes nothing to APCER (look at BPCER for those).
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_bonafide = np.asarray(is_bonafide, dtype=bool)

    if attack_types is None:
        attack_types = np.asarray(["unknown"] * len(scores))
    else:
        attack_types = np.asarray(attack_types)

    per_type: dict[str, float] = {}
    for at in np.unique(attack_types[~is_bonafide]):
        mask = (~is_bonafide) & (attack_types == at)
        n = mask.sum()
        if n == 0:
            continue
        # An attack misclassified as bona-fide iff score >= threshold.
        misclassified = (scores[mask] >= threshold).sum()
        per_type[str(at)] = float(misclassified) / float(n)

    apcer_max = max(per_type.values()) if per_type else 0.0
    return apcer_max, per_type


def bpcer(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    threshold: float,
) -> float:
    """Bona-fide Presentation Classification Error Rate.

    Proportion of bona-fide samples classified as attacks (score < threshold).
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_bonafide = np.asarray(is_bonafide, dtype=bool)

    n = is_bonafide.sum()
    if n == 0:
        return 0.0
    misclassified = (scores[is_bonafide] < threshold).sum()
    return float(misclassified) / float(n)


def acer(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    attack_types: Sequence[str] | None,
    threshold: float,
) -> tuple[float, float, float]:
    """Average Classification Error Rate.

    ACER = (APCER + BPCER) / 2

    Returns:
        (acer_value, apcer_max, bpcer_value)
    """
    apcer_value, _ = apcer(scores, is_bonafide, attack_types, threshold)
    bpcer_value = bpcer(scores, is_bonafide, threshold)
    return (apcer_value + bpcer_value) / 2.0, apcer_value, bpcer_value


def eer(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    attack_types: Sequence[str] | None = None,
) -> tuple[float, float]:
    """Equal Error Rate — the threshold where APCER == BPCER.

    Returns:
        (eer_value, threshold_at_eer)

    Implementation: sweep candidate thresholds (unique scores), find the
    crossover. Uses linear interpolation for sub-sample-resolution EER.
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_bonafide = np.asarray(is_bonafide, dtype=bool)

    candidates = np.unique(scores)
    # Add edges so we capture full sweep
    candidates = np.concatenate([[candidates[0] - 1e-6], candidates, [candidates[-1] + 1e-6]])

    apcer_curve = []
    bpcer_curve = []
    for th in candidates:
        a, _ = apcer(scores, is_bonafide, attack_types, th)
        b = bpcer(scores, is_bonafide, th)
        apcer_curve.append(a)
        bpcer_curve.append(b)
    apcer_curve = np.asarray(apcer_curve)
    bpcer_curve = np.asarray(bpcer_curve)

    # Find crossover: sign change in (apcer - bpcer)
    diff = apcer_curve - bpcer_curve
    sign_changes = np.where(np.diff(np.signbit(diff)))[0]
    if len(sign_changes) == 0:
        # No crossover — return min average error
        idx = int(np.argmin((apcer_curve + bpcer_curve) / 2))
        return float((apcer_curve[idx] + bpcer_curve[idx]) / 2), float(candidates[idx])

    i = int(sign_changes[0])
    # Linear interpolate between candidates[i] and candidates[i+1]
    if (apcer_curve[i + 1] - apcer_curve[i]) - (bpcer_curve[i + 1] - bpcer_curve[i]) == 0:
        eer_th = float(candidates[i])
    else:
        # Solve apcer(t) = bpcer(t) by linear interp
        slope_a = apcer_curve[i + 1] - apcer_curve[i]
        slope_b = bpcer_curve[i + 1] - bpcer_curve[i]
        denom = (slope_a - slope_b)
        t = (bpcer_curve[i] - apcer_curve[i]) / denom if denom != 0 else 0.5
        eer_th = float(candidates[i] + t * (candidates[i + 1] - candidates[i]))

    eer_value = float((apcer_curve[i] + bpcer_curve[i]) / 2)
    return eer_value, eer_th


def far_at_frr(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    attack_types: Sequence[str] | None,
    target_frr: float,
) -> tuple[float, float]:
    """APCER at a fixed BPCER (operating-point report).

    Useful for "what's our attack pass-through rate when we accept 99% of
    real users?" (target_frr = 0.01).

    Returns:
        (apcer_at_target, threshold_used)
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_bonafide = np.asarray(is_bonafide, dtype=bool)

    bonafide_scores = np.sort(scores[is_bonafide])
    if len(bonafide_scores) == 0:
        return 0.0, 0.0

    # Threshold = the (target_frr) quantile of bona-fide scores.
    # Below this threshold we reject — exactly target_frr fraction of bona-fide rejected.
    idx = int(np.clip(target_frr * len(bonafide_scores), 0, len(bonafide_scores) - 1))
    threshold = float(bonafide_scores[idx])
    a, _ = apcer(scores, is_bonafide, attack_types, threshold)
    return a, threshold


def frr_at_far(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    target_far: float,
) -> tuple[float, float]:
    """BPCER at a fixed APCER (operating-point report).

    Useful for "what's our user-rejection rate when we let through 1% of attacks?"
    (target_far = 0.01).

    Returns:
        (bpcer_at_target, threshold_used)
    """
    scores = np.asarray(scores, dtype=np.float64)
    is_bonafide = np.asarray(is_bonafide, dtype=bool)

    attack_scores = np.sort(scores[~is_bonafide])
    if len(attack_scores) == 0:
        return 0.0, 0.0

    # We want target_far fraction of attacks to slip through.
    # Threshold = (1 - target_far) quantile of attack scores.
    idx = int(np.clip((1 - target_far) * len(attack_scores), 0, len(attack_scores) - 1))
    threshold = float(attack_scores[idx])
    b = bpcer(scores, is_bonafide, threshold)
    return b, threshold
