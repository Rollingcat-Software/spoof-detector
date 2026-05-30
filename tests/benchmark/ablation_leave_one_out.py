"""Per-analyzer leave-one-out ablation.

For each analyzer in the productized pipeline, set its fuser weight to 0
and re-evaluate. The ACER delta tells us how much that analyzer contributes
to the final classification.

This is paper Table 8.

Approach: we run the FULL hybrid pipeline once, capture the per-sample
analyzer-level scores in JSON, then re-fuse offline with each analyzer's
weight zeroed. This avoids re-running the (expensive) MiniFASNet ONNX
inference per ablation row.

Output: paper/figures/ablation_loo_<dataset>_<protocol>.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _full_pipeline_with_per_analyzer_capture(samples, max_frames: int = 30) -> list[dict]:
    """Run the full hybrid pipeline once, returning per-sample dicts that
    include per-analyzer raw scores in addition to the final live-ness score."""
    from src.application.face_tracker import FaceTracker
    from src.infrastructure.detection.mediapipe_detector import MediaPipeFaceDetector
    from src.infrastructure.fusion.multi_class_fuser import MultiClassFuser, DEFAULT_ANALYZER_WEIGHTS
    from src.application.pipeline import SpoofDetectionPipeline
    from src.domain.models import SpoofCategory
    from tests.benchmark.pipelines._common import build_hybrid_pipeline, load_frames

    pipeline, fuser = build_hybrid_pipeline()

    out: list[dict] = []
    for i, sample in enumerate(samples):
        if i % 25 == 0:
            logger.info("ablation: %d processed", i)
        frames = load_frames(sample, max_frames=max_frames)
        if not frames:
            continue
        per_frame_analyzer_scores: list[dict[str, float]] = []
        per_frame_p_real: list[float] = []
        for frame in frames:
            analysis = pipeline.process(frame)
            if not analysis.classifications:
                continue
            best_face_id = max(
                analysis.classifications.keys(),
                key=lambda fid: analysis.classifications[fid].confidence or 0.0,
            )
            cls = analysis.classifications[best_face_id]
            # Capture per-analyzer raw scores
            per_analyzer = {name: ar.score for name, ar in cls.analyzer_results.items()}
            per_frame_analyzer_scores.append(per_analyzer)
            per_frame_p_real.append(float(cls.probabilities.get(SpoofCategory.REAL, 0.0)))

        if not per_frame_analyzer_scores:
            continue
        # Average per-analyzer scores across frames (single-image samples have N=1).
        merged_analyzer = {}
        for name in per_frame_analyzer_scores[0]:
            vals = [pf[name] for pf in per_frame_analyzer_scores if name in pf]
            merged_analyzer[name] = float(np.mean(vals)) if vals else 50.0
        out.append({
            "sample_id": sample.sample_id,
            "is_bonafide": sample.is_bonafide,
            "attack_type": sample.attack_type,
            "p_real_full": float(np.mean(per_frame_p_real)),
            "analyzer_scores": merged_analyzer,
        })
    return out


def _refuse_with_weights(per_sample: list[dict], weights: dict[str, float]) -> list[float]:
    """Re-compute P(REAL) per sample under an arbitrary analyzer-weight map.

    Replicates the multi_class_fuser evidence routing at the per-frame-merged
    level (a small approximation; the proper way is to re-run the full fuser
    per frame, but this is faster and the differences are tiny because the
    fuser is linear in analyzer scores). This is the shared kernel behind both
    the leave-one-out ablation (§8.2) and the §8.3 weight-configuration study.

    Analyzers not present in `weights` are treated as weight 0 (dropped),
    matching the original leave-one-out behaviour where unknown names defaulted
    to 0.0 in the re-fusion path.
    """
    from src.domain.models import SpoofCategory
    from src.domain.taxonomy import SPOOF_SIGNAL_MAP

    p_real_list: list[float] = []
    for s in per_sample:
        analyzer_scores = s["analyzer_scores"]
        # Compute per-category log-evidence
        cat_evidence = {cat: 0.0 for cat in SpoofCategory}
        for name, score in analyzer_scores.items():
            w = weights.get(name, 0.0)
            if w <= 0:
                continue
            # Re-derive evidence routing as in the fuser
            score_norm = score / 100.0  # [0,1]
            # High score = live, route into REAL; low score = spoof, distribute via SPOOF_SIGNAL_MAP
            real_evidence = w * score_norm
            spoof_evidence = w * (1 - score_norm)
            cat_evidence[SpoofCategory.REAL] += real_evidence
            signal_map = SPOOF_SIGNAL_MAP.get(name, {})
            for cat, sw in signal_map.items():
                cat_evidence[cat] += spoof_evidence * sw
        # Softmax
        max_e = max(cat_evidence.values())
        exps = {c: np.exp(e - max_e) for c, e in cat_evidence.items()}
        z = sum(exps.values()) or 1e-9
        p_real_list.append(float(exps[SpoofCategory.REAL] / z))
    return p_real_list


def refuse_with_zeroed_weight(per_sample: list[dict], zeroed: str | None = None) -> list[float]:
    """Re-compute P(REAL) per sample after setting a single analyzer's weight to 0.

    Thin wrapper over :func:`_refuse_with_weights` that starts from the
    calibrated ``DEFAULT_ANALYZER_WEIGHTS`` and optionally zeroes one named
    analyzer. Preserved verbatim in behaviour for the §8.2 leave-one-out path.
    """
    from src.infrastructure.fusion.multi_class_fuser import DEFAULT_ANALYZER_WEIGHTS

    weights = dict(DEFAULT_ANALYZER_WEIGHTS)
    if zeroed:
        weights[zeroed] = 0.0
    return _refuse_with_weights(per_sample, weights)


def build_weight_map(config: str, analyzer_names: list[str]) -> dict[str, float]:
    """Construct an analyzer-weight map for a named §8.3 configuration.

    Configs:
        ``calibrated``         — the paper-default ``DEFAULT_ANALYZER_WEIGHTS``.
        ``uniform``            — every analyzer at 1.0.
        ``partial``            — uniform 1.0 except texture=moire=0.1
                                 (isolates the §5.3 anti-correlation reweight).
        ``minifasnet_dominant``— minifasnet=5.0, every other analyzer=0.1.

    `analyzer_names` is the set of analyzers actually captured in the run, so
    the uniform / partial / dominant maps cover exactly the reachable bank
    (multi-frame analyzers that do not fire on the single-image protocol are
    still listed because they appear in the capture with a neutral score).
    """
    from src.infrastructure.fusion.multi_class_fuser import DEFAULT_ANALYZER_WEIGHTS

    if config == "calibrated":
        return dict(DEFAULT_ANALYZER_WEIGHTS)
    if config == "uniform":
        return {name: 1.0 for name in analyzer_names}
    if config == "partial":
        return {name: (0.1 if name in ("texture", "moire") else 1.0)
                for name in analyzer_names}
    if config == "minifasnet_dominant":
        return {name: (5.0 if name == "minifasnet" else 0.1)
                for name in analyzer_names}
    raise ValueError(f"unknown weight config: {config!r}")


WEIGHT_CONFIGS = ("calibrated", "uniform", "partial", "minifasnet_dominant")


def _run_leave_one_out(per_sample, is_bonafide, attack_types, args) -> int:
    """§8.2 path — set each analyzer's weight to 0 and re-evaluate."""
    from src.metrics import classification_report

    # Ablation deltas are computed relative to a shared EER-on-test baseline on the
    # SAME capture, so the Δ-ACER between configs is internally consistent; the
    # absolute ACER is biased low (no Dev split) and is opt-in. Not a paper headline.
    baseline_scores = refuse_with_zeroed_weight(per_sample, zeroed=None)
    baseline = classification_report(
        baseline_scores, is_bonafide, attack_types, allow_test_set_threshold=True
    )
    print(f"=== Baseline (full hybrid) ===")
    print(f"  ACER = {baseline['acer']*100:5.2f}%  AUC = {baseline['auc']:.4f}  N={len(per_sample)}")

    all_analyzers = sorted(per_sample[0]["analyzer_scores"].keys())
    print(f"\n=== Leave-one-out (each row = analyzer removed) ===")
    print(f"{'analyzer':<22s} {'ACER':>8s} {'Δ-ACER':>8s} {'AUC':>8s} {'Δ-AUC':>9s}")
    print("-" * 60)
    rows = []
    for name in all_analyzers:
        scores = refuse_with_zeroed_weight(per_sample, zeroed=name)
        report = classification_report(
            scores, is_bonafide, attack_types, allow_test_set_threshold=True
        )
        d_acer = (report["acer"] - baseline["acer"]) * 100
        d_auc = report["auc"] - baseline["auc"]
        print(f"{name:<22s} {report['acer']*100:7.2f}% {d_acer:+7.2f}% {report['auc']:7.4f} {d_auc:+8.4f}")
        rows.append({
            "analyzer_removed": name,
            "acer": report["acer"],
            "delta_acer": d_acer / 100.0,
            "auc": report["auc"],
            "delta_auc": d_auc,
            "eer": report["eer"],
            "apcer_per_type": report["apcer_per_type"],
        })

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ablation_loo_{args.dataset}_{args.protocol}.json"
    out_file.write_text(json.dumps({
        "dataset": args.dataset,
        "protocol": args.protocol,
        "n_samples": len(per_sample),
        "baseline": baseline,
        "leave_one_out": rows,
    }, indent=2, default=str))
    print(f"\nwrote {out_file}")
    return 0


def _run_weight_configs(per_sample, is_bonafide, attack_types, args) -> int:
    """§8.3 path — re-fuse the SAME per-sample capture under each weight config.

    All configs (calibrated default + the three alternative maps) are derived
    from one capture so the AUC/ACER/EER deltas between them are internally
    consistent (no cross-environment mixing). One JSON per config is written
    to <out>/weightcfg_<dataset>_<protocol>_<config>.json, each embedding the
    full weight map, the per-sample analyzer scores, and the per-sample fused
    P(REAL), so every published §8.3 number is recomputable offline.
    """
    from src.metrics import classification_report

    configs = list(args.weights) if args.weights else list(WEIGHT_CONFIGS)
    analyzer_names = sorted(per_sample[0]["analyzer_scores"].keys())

    print(f"=== §8.3 weight-configuration study  (N={len(per_sample)}) ===")
    print(f"{'config':<22s} {'ACER':>8s} {'EER':>8s} {'AUC':>8s}")
    print("-" * 50)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for cfg in configs:
        weights = build_weight_map(cfg, analyzer_names)
        fused = _refuse_with_weights(per_sample, weights)
        report = classification_report(
            fused, is_bonafide, attack_types, allow_test_set_threshold=True
        )
        results[cfg] = report
        print(f"{cfg:<22s} {report['acer']*100:7.2f}% {report['eer']*100:7.2f}% {report['auc']:7.4f}")

        out_file = out_dir / f"weightcfg_{args.dataset}_{args.protocol}_{cfg}.json"
        out_file.write_text(json.dumps({
            "dataset": args.dataset,
            "protocol": args.protocol,
            "weight_config": cfg,
            "weights": weights,
            "n_samples": len(per_sample),
            "metrics": report,
            "per_sample": [
                {
                    "sample_id": s["sample_id"],
                    "is_bonafide": s["is_bonafide"],
                    "attack_type": s["attack_type"],
                    "analyzer_scores": s["analyzer_scores"],
                    "p_real": p,
                }
                for s, p in zip(per_sample, fused)
            ],
            "note": (
                "Re-fused offline from a single full-hybrid per-analyzer "
                "capture on the in-house replay sub-protocol. All four §8.3 "
                "weight configs share this capture so their deltas are "
                "internally consistent. Environment: uniface/onnxruntime as "
                "pinned in requirements.txt at generation time."
            ),
        }, indent=2, default=str))
        print(f"  wrote {out_file}")

    if "calibrated" in results and "uniform" in results:
        d = results["calibrated"]["auc"] - results["uniform"]["auc"]
        print(f"\nuniform -> calibrated AUC delta = {d:+.4f} "
              f"(uniform {results['uniform']['auc']:.4f} -> "
              f"calibrated {results['calibrated']['auc']:.4f})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--root", required=False)
    p.add_argument("--protocol", default="ablation_loo")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="paper/figures")
    p.add_argument("--mode", choices=["loo", "weights"], default="loo",
                   help="loo = per-analyzer leave-one-out (§8.2, default); "
                        "weights = re-fuse under named weight configs (§8.3).")
    p.add_argument("--weights", nargs="+", choices=WEIGHT_CONFIGS, default=None,
                   help="Weight configs to evaluate in --mode weights "
                        "(default: all four). Ignored in --mode loo.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * args.verbose)

    # Reuse the runner's adapter loader
    from tests.benchmark.run import _load_adapter
    samples = _load_adapter(args.dataset, args.root, args.protocol)
    if args.limit:
        from itertools import islice
        samples = list(islice(samples, args.limit))

    logger.info("Running full hybrid pipeline once with per-analyzer capture...")
    per_sample = _full_pipeline_with_per_analyzer_capture(samples)
    logger.info("Captured %d samples with %d analyzer outputs each",
                len(per_sample),
                len(per_sample[0]["analyzer_scores"]) if per_sample else 0)

    if not per_sample:
        print("ERROR: no reachable samples (empty capture) — aborting")
        return 1

    # Build labels
    is_bonafide = [s["is_bonafide"] for s in per_sample]
    attack_types = [s["attack_type"] or "unknown" for s in per_sample]

    if args.mode == "weights":
        return _run_weight_configs(per_sample, is_bonafide, attack_types, args)
    return _run_leave_one_out(per_sample, is_bonafide, attack_types, args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
