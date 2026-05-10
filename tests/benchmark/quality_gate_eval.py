"""Filter samples by face-detection quality, re-evaluate.

Hypothesis: pipeline accuracy is bounded by face-detection reliability.
A sample where MediaPipe detects no face (or a tiny face, or multiple
faces) is essentially noise — every analyzer falls back to neutral
scores. Excluding those samples should tighten accuracy.

This script reads a capture JSON (from `capture_analyzer_scores.py`) and
re-computes ACER + AUC on three subsets:

  - all                     — every captured sample
  - face_detected           — drops samples where face_detected=False
  - large_face              — face_detected AND face_area_px >= threshold
  - single_large_face       — single-face AND area >= threshold

Usage:
    python -m tests.benchmark.quality_gate_eval \\
        --capture paper/figures/captures/casia_fasd_test.json \\
        --area-threshold 5000

Reports the four metric tables side-by-side.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def hand_calibrated_score_one(analyzer_scores: dict[str, float]) -> float:
    """Compute P(REAL) using DEFAULT_ANALYZER_WEIGHTS (matches train_logistic_fuser)."""
    from src.infrastructure.fusion.multi_class_fuser import DEFAULT_ANALYZER_WEIGHTS
    from src.domain.taxonomy import SPOOF_SIGNAL_MAP
    from src.domain.models import SpoofCategory

    cat_evidence = {cat: 0.0 for cat in SpoofCategory}
    for name, score in analyzer_scores.items():
        w = DEFAULT_ANALYZER_WEIGHTS.get(name, 0.0)
        if w <= 0: continue
        score_norm = score / 100.0
        cat_evidence[SpoofCategory.REAL] += w * score_norm
        spoof_e = w * (1 - score_norm)
        for cat, sw in SPOOF_SIGNAL_MAP.get(name, {}).items():
            cat_evidence[cat] += spoof_e * sw
    max_e = max(cat_evidence.values())
    exps = {c: np.exp(e - max_e) for c, e in cat_evidence.items()}
    z = sum(exps.values()) or 1e-9
    return float(exps[SpoofCategory.REAL] / z)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--capture", required=True, help="Path to capture JSON.")
    p.add_argument("--area-threshold", type=int, default=5000,
                   help="Minimum face_area_px for the 'large_face' subset (default 5000 = 70x70 px).")
    args = p.parse_args(argv)

    from src.metrics import classification_report

    data = json.loads(Path(args.capture).read_text())
    records = data["per_sample"]

    # Compute baseline score per record
    for r in records:
        r["baseline_score"] = (
            hand_calibrated_score_one(r["analyzer_scores"])
            if r.get("analyzer_scores") else 0.5
        )

    def evaluate(subset, label):
        if not subset:
            print(f"\n  === {label} ===")
            print(f"    (empty subset)")
            return None
        scores = [r["baseline_score"] for r in subset]
        is_bf = [r["is_bonafide"] for r in subset]
        types = [r.get("attack_type") or "unknown" for r in subset]
        bf_n = sum(is_bf)
        report = classification_report(scores, is_bf, types)
        print(f"\n  === {label} ===  N={len(subset)} bf={bf_n} attack={len(subset)-bf_n}")
        print(f"    ACER = {report['acer']*100:5.2f}%   EER = {report['eer']*100:5.2f}%   AUC = {report['auc']:.4f}")
        return {"label": label, "N": len(subset), "bonafide": bf_n, "metrics": report}

    print(f"\n=== Quality-gate evaluation: {args.capture} ===")
    print(f"  total captured: {len(records)}  (face_area threshold = {args.area_threshold} px)")

    s_all = records
    s_detected = [r for r in records if r["face_detected"]]
    s_large = [r for r in s_detected if r.get("face_area_px", 0) >= args.area_threshold]
    s_single_large = [r for r in s_large if r.get("n_faces", 0) == 1]

    out = {"capture": str(args.capture), "area_threshold": args.area_threshold, "subsets": {}}
    out["subsets"]["all"] = evaluate(s_all, "all (no gating)")
    out["subsets"]["face_detected"] = evaluate(s_detected, "face_detected (drops detection failures)")
    out["subsets"]["large_face"] = evaluate(s_large, f"large_face (area >= {args.area_threshold} px)")
    out["subsets"]["single_large"] = evaluate(s_single_large, "single_large_face (single face, large)")

    out_path = Path(args.capture).with_name(Path(args.capture).stem + "_quality_gated.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
