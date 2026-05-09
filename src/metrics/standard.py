"""Standard binary-classification metrics for FAS reporting.

These complement the ISO 30107-3 metrics in `iso30107.py`:

  HTER (Half Total Error Rate)
    (FAR + FRR) / 2 — same shape as ACER but parameterized in
    biometric-verification language. Many older FAS papers use HTER
    where newer ISO papers use ACER. Provided for cross-paper comparability.

  ROC curve
    (FAR, FRR) sweep across thresholds; AUC summarizes.

  Confusion matrix
    Standard 2x2 (bona-fide vs attack) and N+1 multiclass
    (bona-fide + per-attack-type).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from src.metrics.iso30107 import apcer, bpcer


@dataclass
class ROCPoint:
    threshold: float
    far: float  # APCER at this threshold
    frr: float  # BPCER at this threshold
    far_per_type: dict[str, float]


@dataclass
class ROCResult:
    points: list[ROCPoint]
    auc: float
    eer: float
    eer_threshold: float


def hter(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    attack_types: Sequence[str] | None,
    threshold: float,
) -> tuple[float, float, float]:
    """Half Total Error Rate (legacy name; numerically equivalent to ACER for
    binary bona-fide vs attack-max).

    Returns (hter, far, frr).
    """
    far, _ = apcer(scores, is_bonafide, attack_types, threshold)
    frr = bpcer(scores, is_bonafide, threshold)
    return (far + frr) / 2.0, far, frr


def roc_curve(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    attack_types: Sequence[str] | None = None,
    n_points: int = 200,
) -> ROCResult:
    """Compute ROC curve as a sweep over thresholds.

    Threshold sweep is `n_points` linearly-spaced quantiles of the score
    distribution (so we never spend points where there's no data).
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    is_bonafide_arr = np.asarray(is_bonafide, dtype=bool)

    sorted_scores = np.sort(np.unique(scores_arr))
    if len(sorted_scores) <= n_points:
        thresholds = sorted_scores
    else:
        idx = np.linspace(0, len(sorted_scores) - 1, n_points).astype(int)
        thresholds = sorted_scores[idx]
    # Also include extreme thresholds so curve goes (1,0) → (0,1)
    thresholds = np.concatenate([[sorted_scores[0] - 1e-6], thresholds, [sorted_scores[-1] + 1e-6]])
    thresholds = np.unique(thresholds)

    points: list[ROCPoint] = []
    for th in thresholds:
        far, per_type = apcer(scores_arr, is_bonafide_arr, attack_types, th)
        frr = bpcer(scores_arr, is_bonafide_arr, th)
        points.append(ROCPoint(threshold=float(th), far=float(far), frr=float(frr), far_per_type=per_type))

    far_vals = np.array([p.far for p in points])
    frr_vals = np.array([p.frr for p in points])
    # AUC via trapezoidal rule on (far, 1-frr) — i.e. (FPR, TPR)
    # Sort by FPR for monotonic integration.
    order = np.argsort(far_vals)
    # np.trapezoid (numpy >= 2.0) replaces np.trapz; fall back for older numpy
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]
    auc = float(_trapz(1.0 - frr_vals[order], far_vals[order]))

    # EER from intersection of FAR and FRR curves
    diff = far_vals - frr_vals
    sign_changes = np.where(np.diff(np.signbit(diff)))[0]
    if len(sign_changes) > 0:
        i = int(sign_changes[0])
        eer_value = float((far_vals[i] + frr_vals[i]) / 2)
        eer_threshold = float(thresholds[i])
    else:
        idx = int(np.argmin((far_vals + frr_vals) / 2))
        eer_value = float((far_vals[idx] + frr_vals[idx]) / 2)
        eer_threshold = float(thresholds[idx])

    return ROCResult(points=points, auc=auc, eer=eer_value, eer_threshold=eer_threshold)


def confusion_matrix(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    threshold: float,
    attack_types: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Multi-class confusion matrix.

    If attack_types provided, returns a (bonafide + per-attack-type) row × (bonafide, attack) col matrix.
    Otherwise binary 2x2.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    is_bonafide_arr = np.asarray(is_bonafide, dtype=bool)
    pred_bonafide = scores_arr >= threshold

    cm: dict[str, dict[str, int]] = {}

    # Bona-fide row
    cm["bonafide"] = {
        "pred_bonafide": int((is_bonafide_arr & pred_bonafide).sum()),
        "pred_attack": int((is_bonafide_arr & ~pred_bonafide).sum()),
    }

    if attack_types is None:
        cm["attack"] = {
            "pred_bonafide": int((~is_bonafide_arr & pred_bonafide).sum()),
            "pred_attack": int((~is_bonafide_arr & ~pred_bonafide).sum()),
        }
    else:
        attack_types_arr = np.asarray(attack_types)
        for at in np.unique(attack_types_arr[~is_bonafide_arr]):
            mask = (~is_bonafide_arr) & (attack_types_arr == at)
            cm[f"attack:{at}"] = {
                "pred_bonafide": int((mask & pred_bonafide).sum()),
                "pred_attack": int((mask & ~pred_bonafide).sum()),
            }

    return cm


def classification_report(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    attack_types: Sequence[str] | None = None,
    threshold: float | None = None,
) -> dict[str, float | dict]:
    """One-call summary: APCER (max + per-type), BPCER, ACER, HTER, EER, AUC.

    If threshold is None, uses the EER threshold (typical paper-reporting choice).
    """
    from src.metrics.iso30107 import apcer, bpcer, acer, eer  # local re-import for clarity

    if threshold is None:
        eer_value, eer_th = eer(scores, is_bonafide, attack_types)
        threshold = eer_th
    else:
        eer_value, eer_th = eer(scores, is_bonafide, attack_types)

    apcer_max, per_type = apcer(scores, is_bonafide, attack_types, threshold)
    bpcer_value = bpcer(scores, is_bonafide, threshold)
    acer_value = (apcer_max + bpcer_value) / 2.0

    roc = roc_curve(scores, is_bonafide, attack_types)

    return {
        "threshold": threshold,
        "apcer_max": apcer_max,
        "apcer_per_type": per_type,
        "bpcer": bpcer_value,
        "acer": acer_value,
        "hter": acer_value,  # numerically identical for binary FAS
        "eer": eer_value,
        "eer_threshold": eer_th,
        "auc": roc.auc,
        "n_bonafide": int(np.sum(is_bonafide)),
        "n_attack": int(np.sum(~np.asarray(is_bonafide, dtype=bool))),
    }
