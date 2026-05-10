"""Train a logistic-regression fuser from captured per-analyzer scores.

Replaces the hand-calibrated MultiClassFuser weights with a learned binary
classifier (REAL vs ATTACK). This is the §5.5 paper recommendation made
operational: re-calibrate per dataset.

Usage:
    # 1. Capture train + test sets:
    python -m tests.benchmark.capture_analyzer_scores \\
        --dataset casia_fasd --root /tmp/fas_datasets/akahana_casiafasd/extracted \\
        --split train --out paper/figures/captures/casia_fasd_train.json

    python -m tests.benchmark.capture_analyzer_scores \\
        --dataset casia_fasd --root /tmp/fas_datasets/akahana_casiafasd/extracted \\
        --split test --out paper/figures/captures/casia_fasd_test.json

    # 2. Train logistic fuser on train, evaluate on test:
    python -m tests.benchmark.train_logistic_fuser \\
        --train paper/figures/captures/casia_fasd_train.json \\
        --test  paper/figures/captures/casia_fasd_test.json \\
        --out   paper/figures/learned_fuser_casia_fasd.json

Compares 4 fusers on the test set:
  - hand_calibrated_fixed (current MultiClassFuser DEFAULT_ANALYZER_WEIGHTS)
  - logistic_l2 (sklearn LogisticRegression with L2)
  - logistic_l1 (LogisticRegression with L1 — feature selection)
  - average_pool (uniform mean of all analyzer scores)

Reports AUC + ACER + EER for each on the test split.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_capture(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    """Load a capture JSON. Returns (X [n, k], y [n], feature_names, raw_records)."""
    data = json.loads(Path(path).read_text())
    records = data["per_sample"]

    # Determine analyzer keys from the first record that has scores
    analyzer_keys: list[str] = []
    for r in records:
        if r.get("analyzer_scores"):
            analyzer_keys = sorted(r["analyzer_scores"].keys())
            break
    if not analyzer_keys:
        raise SystemExit(f"no analyzer scores in {path}")

    X = np.zeros((len(records), len(analyzer_keys)), dtype=np.float64)
    y = np.zeros(len(records), dtype=np.int64)
    for i, r in enumerate(records):
        scores = r.get("analyzer_scores", {})
        for j, k in enumerate(analyzer_keys):
            X[i, j] = scores.get(k, 50.0)  # neutral fallback
        y[i] = 1 if r["is_bonafide"] else 0
    return X / 100.0, y, analyzer_keys, records


def hand_calibrated_score(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Use the current MultiClassFuser weights to compute per-sample P(REAL)."""
    from src.infrastructure.fusion.multi_class_fuser import DEFAULT_ANALYZER_WEIGHTS
    from src.domain.taxonomy import SPOOF_SIGNAL_MAP
    from src.domain.models import SpoofCategory

    out = np.zeros(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        cat_evidence = {cat: 0.0 for cat in SpoofCategory}
        for j, name in enumerate(feature_names):
            w = DEFAULT_ANALYZER_WEIGHTS.get(name, 0.0)
            if w <= 0: continue
            score_norm = X[i, j]
            cat_evidence[SpoofCategory.REAL] += w * score_norm
            spoof_e = w * (1 - score_norm)
            for cat, sw in SPOOF_SIGNAL_MAP.get(name, {}).items():
                cat_evidence[cat] += spoof_e * sw
        max_e = max(cat_evidence.values())
        exps = {c: np.exp(e - max_e) for c, e in cat_evidence.items()}
        z = sum(exps.values()) or 1e-9
        out[i] = exps[SpoofCategory.REAL] / z
    return out


def fit_logistic(X: np.ndarray, y: np.ndarray, *, penalty: str = "l2", C: float = 1.0):
    """Fit sklearn LogisticRegression. Returns model + score function."""
    from sklearn.linear_model import LogisticRegression

    solver = "liblinear" if penalty == "l1" else "lbfgs"
    clf = LogisticRegression(
        penalty=penalty, C=C, solver=solver,
        max_iter=2000, class_weight="balanced",
    )
    clf.fit(X, y)
    return clf


def evaluate_pipeline_scores(
    scores: np.ndarray,
    y: np.ndarray,
    types: list[str],
    label: str,
):
    """Run the standard FAS metrics + bootstrap CI on the score array."""
    from src.metrics import classification_report, acer_ci, auc_ci, eer_ci

    is_bf = [bool(v) for v in y]
    types_list = [t or "unknown" for t in types]
    report = classification_report(scores.tolist(), is_bf, types_list)

    ci_acer = acer_ci(scores.tolist(), is_bf, types_list, n_resamples=200, alpha=0.05, seed=42)
    ci_auc  = auc_ci(scores.tolist(), is_bf, types_list, n_resamples=200, alpha=0.05, seed=42)
    ci_eer  = eer_ci(scores.tolist(), is_bf, types_list, n_resamples=200, alpha=0.05, seed=42)

    print(f"\n  === {label} ===")
    print(f"    ACER = {ci_acer.estimate*100:5.2f}%  (95% CI [{ci_acer.low*100:5.2f}%, {ci_acer.high*100:5.2f}%])")
    print(f"    EER  = {ci_eer.estimate*100:5.2f}%  (95% CI [{ci_eer.low*100:5.2f}%, {ci_eer.high*100:5.2f}%])")
    print(f"    AUC  = {ci_auc.estimate:.4f}  (95% CI [{ci_auc.low:.4f}, {ci_auc.high:.4f}])")
    return {
        "label": label,
        "metrics": report,
        "acer_ci": [ci_acer.low, ci_acer.high],
        "auc_ci": [ci_auc.low, ci_auc.high],
        "eer_ci": [ci_eer.low, ci_eer.high],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--train", required=True, help="Path to training capture JSON.")
    p.add_argument("--test", required=True, help="Path to test capture JSON.")
    p.add_argument("--out", required=True, help="Output JSON for results.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose)

    print(f"Loading captures...")
    X_train, y_train, feature_names, _ = load_capture(Path(args.train))
    X_test, y_test, feature_names_test, test_records = load_capture(Path(args.test))
    if feature_names != feature_names_test:
        print(f"warning: feature names differ — train: {feature_names}, test: {feature_names_test}")
    types = [r.get("attack_type") for r in test_records]
    print(f"  train: {X_train.shape[0]} samples, {X_train.shape[1]} features, {y_train.sum()} bonafide")
    print(f"  test:  {X_test.shape[0]} samples, {y_test.sum()} bonafide")

    results = {}

    # Baseline: hand-calibrated fixed weights
    print(f"\n[baseline] hand_calibrated_fixed (zero-shot, no training)")
    sc_baseline = hand_calibrated_score(X_test, feature_names)
    results["hand_calibrated_fixed"] = evaluate_pipeline_scores(sc_baseline, y_test, types, "hand_calibrated_fixed")

    # Average pool — naive baseline
    print(f"\n[baseline] average_pool (uniform mean of analyzer scores)")
    sc_avg = X_test.mean(axis=1)
    results["average_pool"] = evaluate_pipeline_scores(sc_avg, y_test, types, "average_pool")

    # Logistic L2
    print(f"\n[learned] logistic_l2")
    clf_l2 = fit_logistic(X_train, y_train, penalty="l2", C=1.0)
    sc_l2 = clf_l2.predict_proba(X_test)[:, 1]
    results["logistic_l2"] = evaluate_pipeline_scores(sc_l2, y_test, types, "logistic_l2")
    print(f"    learned weights: {dict(zip(feature_names, clf_l2.coef_[0].round(3)))}")
    print(f"    intercept: {float(clf_l2.intercept_[0]):.3f}")

    # Logistic L1 (sparser)
    print(f"\n[learned] logistic_l1")
    clf_l1 = fit_logistic(X_train, y_train, penalty="l1", C=1.0)
    sc_l1 = clf_l1.predict_proba(X_test)[:, 1]
    results["logistic_l1"] = evaluate_pipeline_scores(sc_l1, y_test, types, "logistic_l1")
    print(f"    learned weights: {dict(zip(feature_names, clf_l1.coef_[0].round(3)))}")

    # Save results
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "train_capture": str(args.train),
        "test_capture": str(args.test),
        "feature_names": feature_names,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "results": results,
        "logistic_l2_weights": dict(zip(feature_names, clf_l2.coef_[0].tolist())),
        "logistic_l1_weights": dict(zip(feature_names, clf_l1.coef_[0].tolist())),
    }, indent=2, default=str))
    print(f"\nwrote {out_path}")

    # Summary table
    print(f"\n=== Summary (lower ACER / higher AUC = better) ===")
    print(f"{'method':<28s} {'ACER':>10s} {'AUC':>10s}")
    print("-" * 52)
    for method, r in results.items():
        m = r["metrics"]
        print(f"{method:<28s} {m['acer']*100:>9.2f}% {m['auc']:>10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
