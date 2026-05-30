"""Bootstrap confidence intervals for FAS metrics.

Why this matters for a paper: a single point estimate (ACER = 12.67%)
gives reviewers no sense of whether the result is robust. With N=100,
the standard error is non-trivial. The bootstrap CI tells reviewers
"the ACER is somewhere between X and Y with 95% confidence on resampled
versions of this exact test set."

Usage:
    from src.metrics.bootstrap import bootstrap_ci

    # NOTE: ACER needs a decision threshold. The honest way is to pass a
    # Dev-derived threshold (see prefer auc_ci / eer_ci, which are threshold-free).
    # The example below opts into EER-on-test purely for illustration.
    ci = bootstrap_ci(
        scores, is_bonafide, attack_types,
        metric=lambda *a: classification_report(*a, allow_test_set_threshold=True)["acer"],
        n_resamples=2000, alpha=0.05, seed=42,
    )
    print(f"ACER = {ci.estimate:.3f} (95% CI [{ci.low:.3f}, {ci.high:.3f}])")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence
import numpy as np


@dataclass
class CIResult:
    estimate: float          # point estimate from the original sample
    low: float               # lower bound of 1-alpha CI
    high: float              # upper bound of 1-alpha CI
    median: float            # median of the bootstrap distribution
    std: float               # std of the bootstrap distribution
    n_resamples: int         # number of bootstrap replicates


def bootstrap_ci(
    scores: Sequence[float],
    is_bonafide: Sequence[bool],
    attack_types: Sequence[str] | None,
    *,
    metric: Callable[[Sequence[float], Sequence[bool], Sequence[str] | None], float],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 42,
    stratified: bool = True,
) -> CIResult:
    """Stratified bootstrap CI over (bonafide, per-attack-type) cells.

    Stratification keeps the bonafide:attack ratio + per-attack-type counts
    constant across resamples, which is the convention for FAS papers
    (so the CI captures only the per-cell sampling noise, not the
    composition-of-the-test-set noise).
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    is_bf_arr = np.asarray(is_bonafide, dtype=bool)
    if attack_types is not None:
        types_arr = np.asarray(attack_types)
    else:
        types_arr = None

    rng = np.random.default_rng(seed)
    n = len(scores_arr)

    # Build stratification cells
    if stratified:
        bf_idx = np.where(is_bf_arr)[0]
        attack_idx_by_type: dict = {}
        if types_arr is not None:
            for at in np.unique(types_arr[~is_bf_arr]):
                attack_idx_by_type[at] = np.where((~is_bf_arr) & (types_arr == at))[0]
        else:
            attack_idx_by_type[""] = np.where(~is_bf_arr)[0]
    else:
        bf_idx = None
        attack_idx_by_type = None

    # Original-sample point estimate
    estimate = float(metric(scores_arr, is_bf_arr, types_arr))

    # Bootstrap replicates
    replicates: list[float] = []
    for _ in range(n_resamples):
        if stratified and bf_idx is not None and attack_idx_by_type is not None:
            sampled = []
            sampled.append(rng.choice(bf_idx, size=len(bf_idx), replace=True))
            for idx in attack_idx_by_type.values():
                sampled.append(rng.choice(idx, size=len(idx), replace=True))
            sample_idx = np.concatenate(sampled)
        else:
            sample_idx = rng.integers(0, n, size=n)
        try:
            v = float(metric(
                scores_arr[sample_idx],
                is_bf_arr[sample_idx],
                None if types_arr is None else types_arr[sample_idx],
            ))
            replicates.append(v)
        except Exception:
            # Some draws may yield degenerate samples (e.g. all bonafide); skip.
            continue

    arr = np.asarray(replicates)
    low = float(np.quantile(arr, alpha / 2))
    high = float(np.quantile(arr, 1 - alpha / 2))
    return CIResult(
        estimate=estimate,
        low=low,
        high=high,
        median=float(np.median(arr)),
        std=float(np.std(arr)),
        n_resamples=len(arr),
    )


def acer_ci(scores, is_bonafide, attack_types, **kw) -> CIResult:
    """Convenience: ACER bootstrap CI at the EER threshold."""
    from src.metrics.iso30107 import eer, acer

    def _acer(s, ib, at):
        _, eer_th = eer(s, ib, at)
        v, _, _ = acer(s, ib, at, eer_th)
        return v
    return bootstrap_ci(scores, is_bonafide, attack_types, metric=_acer, **kw)


def auc_ci(scores, is_bonafide, attack_types=None, **kw) -> CIResult:
    """Convenience: AUC bootstrap CI."""
    from src.metrics.standard import roc_curve

    def _auc(s, ib, at):
        return roc_curve(s, ib, at, n_points=100).auc
    return bootstrap_ci(scores, is_bonafide, attack_types, metric=_auc, **kw)


def eer_ci(scores, is_bonafide, attack_types=None, **kw) -> CIResult:
    """Convenience: EER bootstrap CI."""
    from src.metrics.iso30107 import eer

    def _eer(s, ib, at):
        v, _ = eer(s, ib, at)
        return v
    return bootstrap_ci(scores, is_bonafide, attack_types, metric=_eer, **kw)
