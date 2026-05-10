"""Inspect failure-mode samples to identify systematic accuracy gaps.

Walks a capture JSON and groups samples by classification correctness:
  - true_positive   (bonafide, score >= threshold)
  - false_negative  (bonafide, score <  threshold)  -- bonafide rejected
  - true_negative   (attack,   score <  threshold)
  - false_positive  (attack,   score >= threshold)  -- attack passed through

For each false sample, dumps:
  - sample_id, score, threshold, is_bonafide, attack_type
  - per-analyzer score breakdown
  - face metadata (n_faces, face_area_px, face_detected)
  - source image path (if present in `payload`)

Output: paper/figures/inspection_<dataset>_<split>.json

Use this to:
  1. Identify if there's a tail of "always-fails" attack types
  2. Find clusters of low-quality face crops
  3. Confirm whether MiniFASNet alone or the fuser is the failure point
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def hand_score(scores: dict[str, float]) -> float:
    """Replicate MultiClassFuser P(REAL) using DEFAULT_ANALYZER_WEIGHTS."""
    from src.infrastructure.fusion.multi_class_fuser import DEFAULT_ANALYZER_WEIGHTS
    from src.domain.taxonomy import SPOOF_SIGNAL_MAP
    from src.domain.models import SpoofCategory

    cat_evidence = {cat: 0.0 for cat in SpoofCategory}
    for name, sc in scores.items():
        w = DEFAULT_ANALYZER_WEIGHTS.get(name, 0.0)
        if w <= 0: continue
        score_norm = sc / 100.0
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
    p.add_argument("--capture", required=True)
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold. Default: EER threshold from this capture.")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    data = json.loads(Path(args.capture).read_text())
    records = data["per_sample"]

    # Compute scores
    for r in records:
        r["baseline_score"] = (
            hand_score(r["analyzer_scores"]) if r.get("analyzer_scores") else 0.5
        )

    # Determine threshold
    if args.threshold is None:
        from src.metrics.iso30107 import eer
        scores = [r["baseline_score"] for r in records]
        is_bf = [r["is_bonafide"] for r in records]
        types = [r.get("attack_type") or "unknown" for r in records]
        eer_value, eer_th = eer(scores, is_bf, types)
        threshold = eer_th
        print(f"using EER threshold = {threshold:.4f} (EER = {eer_value*100:.2f}%)")
    else:
        threshold = args.threshold

    # Bucket
    buckets = {"true_positive": [], "false_negative": [], "true_negative": [], "false_positive": []}
    for r in records:
        score = r["baseline_score"]
        is_bf = r["is_bonafide"]
        accept = score >= threshold
        if is_bf and accept:
            buckets["true_positive"].append(r)
        elif is_bf and not accept:
            buckets["false_negative"].append(r)
        elif not is_bf and accept:
            buckets["false_positive"].append(r)
        else:
            buckets["true_negative"].append(r)

    print(f"\n=== Inspection {args.capture} (threshold={threshold:.4f}) ===")
    print(f"  TP (bonafide accepted):  {len(buckets['true_positive']):4d}")
    print(f"  FN (bonafide rejected):  {len(buckets['false_negative']):4d}  <-- false negatives")
    print(f"  TN (attack rejected):    {len(buckets['true_negative']):4d}")
    print(f"  FP (attack accepted):    {len(buckets['false_positive']):4d}  <-- false positives")
    if buckets["true_positive"] or buckets["false_negative"]:
        bpcer = len(buckets["false_negative"]) / (len(buckets["true_positive"]) + len(buckets["false_negative"]))
        print(f"  BPCER = {bpcer*100:.2f}% (FN / bonafide_total)")
    if buckets["true_negative"] or buckets["false_positive"]:
        apcer = len(buckets["false_positive"]) / (len(buckets["true_negative"]) + len(buckets["false_positive"]))
        print(f"  APCER = {apcer*100:.2f}% (FP / attack_total)")

    # Failure-mode analysis: cluster failures by face metadata
    def summarize_failures(group, label):
        if not group:
            print(f"\n  no {label}s")
            return
        print(f"\n  --- {label}s (N={len(group)}) ---")
        # Histograms
        n_faces = [r.get("n_faces", 0) for r in group]
        face_areas = [r.get("face_area_px", 0) for r in group]
        no_face = sum(1 for r in group if not r.get("face_detected", True))
        print(f"    no face detected:   {no_face} ({100*no_face/len(group):.1f}%)")
        print(f"    n_faces=1:          {sum(1 for n in n_faces if n == 1)}")
        print(f"    n_faces>=2:         {sum(1 for n in n_faces if n >= 2)}")
        print(f"    face_area median:   {int(np.median(face_areas))} px (min {min(face_areas)}, max {max(face_areas)})")
        # Per-attack-type breakdown if present
        if any(r.get("attack_type") for r in group):
            from collections import Counter
            counts = Counter(r.get("attack_type") or "unknown" for r in group)
            print(f"    per-attack-type:    {dict(counts)}")
        # Score distribution
        scores = [r["baseline_score"] for r in group]
        print(f"    score median:       {np.median(scores):.4f} (min {min(scores):.4f}, max {max(scores):.4f})")

    summarize_failures(buckets["false_negative"], "false_negative")
    summarize_failures(buckets["false_positive"], "false_positive")

    # Save details
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "capture": str(args.capture),
        "threshold": threshold,
        "buckets": {
            k: [{"sample_id": r["sample_id"],
                 "is_bonafide": r["is_bonafide"],
                 "attack_type": r.get("attack_type"),
                 "score": r["baseline_score"],
                 "analyzer_scores": r.get("analyzer_scores", {}),
                 "n_faces": r.get("n_faces"),
                 "face_area_px": r.get("face_area_px"),
                 "face_detected": r.get("face_detected"),
                 } for r in v]
            for k, v in buckets.items()
        },
    }, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
